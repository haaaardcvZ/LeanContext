"""Reusable single-view loss evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import DataLoader

from lean_proof_training.common.data import TokenCollator
from lean_proof_training.common.io import iter_jsonl
from lean_proof_training.common.jsonl import IndexedJsonlDataset, load_id_set
from lean_proof_training.common.metrics import evaluate_loader
from lean_proof_training.common.modeling import load_model, load_tokenizer, resolved_loss_config
from lean_proof_training.common.runtime import build_accelerator


def run_evaluation(
    config: dict[str, Any],
    *,
    architecture: str,
    loss_function: Callable[..., Any],
) -> dict[str, Any]:
    accelerator = build_accelerator(config)
    tokenizer = load_tokenizer(config, tokenized=True)
    loss_config = resolved_loss_config(config, architecture=architecture, tokenizer=tokenizer)
    allow_ids = load_id_set(config["evaluation"]["id_file"]) if config["evaluation"].get("id_file") else None
    dataset = IndexedJsonlDataset(
        config["evaluation"]["data_file"],
        stage=config["evaluation"].get("stage"),
        allow_ids=allow_ids,
    )
    collator = TokenCollator(tokenizer.pad_token_id, tokenizer.eos_token_id)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["per_device_eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=collator,
    )
    model = load_model(config, architecture=architecture, tokenizer=tokenizer, trainable=False)
    model, loader = accelerator.prepare(model, loader)
    state_path = config["evaluation"].get("accelerate_state_path")
    if state_path:
        accelerator.load_state(str(state_path))
    local_rows: list[dict[str, Any]] = []
    result = evaluate_loader(
        model, loader, accelerator, loss_function, loss_config, local_rows=local_rows
    )
    output = Path(config["evaluation"]["output_file"])
    per_sample_output = Path(
        config["evaluation"].get(
            "per_sample_output_file", output.with_name(output.stem + "_per_sample.jsonl")
        )
    )
    rank_output = per_sample_output.with_name(
        per_sample_output.name + f".rank{accelerator.process_index}.tmp"
    )
    rank_output.parent.mkdir(parents=True, exist_ok=True)
    with rank_output.open("w", encoding="utf-8") as handle:
        for row in local_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        merged: dict[str, dict[str, Any]] = {}
        for rank in range(accelerator.num_processes):
            part = per_sample_output.with_name(per_sample_output.name + f".rank{rank}.tmp")
            for row in iter_jsonl(part):
                merged[str(row["sample_id"])] = row
            part.unlink()
        with per_sample_output.open("w", encoding="utf-8") as handle:
            for sample_id in sorted(merged):
                handle.write(json.dumps(merged[sample_id], ensure_ascii=False) + "\n")
        result["per_sample_output_file"] = str(per_sample_output)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accelerator.wait_for_everyone()
    return result
