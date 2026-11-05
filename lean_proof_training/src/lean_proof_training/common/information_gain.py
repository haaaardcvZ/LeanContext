"""Paired information-gain evaluation using one frozen checkpoint."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from lean_proof_training.common.batching import move_batch
from lean_proof_training.common.data import AlignedPairDataset, PairCollator, TokenCollator
from lean_proof_training.common.io import iter_jsonl
from lean_proof_training.common.jsonl import load_id_set
from lean_proof_training.common.modeling import load_model, load_tokenizer, resolved_loss_config
from lean_proof_training.common.runtime import build_accelerator


def budget_sort_key(name: str) -> tuple[int, str]:
    if name == "full":
        return (10**12, name)
    if name.startswith("b") and name[1:].isdigit():
        return (int(name[1:]), name)
    raise ValueError(f"invalid budget name: {name}")


@torch.no_grad()
def evaluate_pair(
    *,
    model: Any,
    accelerator: Any,
    tokenizer: Any,
    left_file: Path,
    right_file: Path,
    allow_ids: set[str],
    stage: str | None,
    repeats: int,
    batch_size: int,
    workers: int,
    loss_function: Callable[..., Any],
    loss_config: dict[str, Any],
    output_file: Path,
    label: str,
) -> None:
    dataset = AlignedPairDataset(
        left_file, right_file, stage=stage, allow_ids=allow_ids
    )
    collator = PairCollator(TokenCollator(tokenizer.pad_token_id, tokenizer.eos_token_id))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
    )
    loader = accelerator.prepare(loader)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for paired in loader:
            left = move_batch(paired["left"], accelerator.device)
            right = move_batch(paired["right"], accelerator.device)
            left_repeats: list[torch.Tensor] = []
            right_repeats: list[torch.Tensor] = []
            for repeat in range(repeats):
                left_output = loss_function(
                    model,
                    left,
                    config=loss_config,
                    evaluating=True,
                    repeat_index=repeat,
                )
                right_output = loss_function(
                    model,
                    right,
                    config=loss_config,
                    evaluating=True,
                    repeat_index=repeat,
                )
                left_repeats.append(left_output.per_sample_loss.detach().float().cpu())
                right_repeats.append(right_output.per_sample_loss.detach().float().cpu())
            left_loss = torch.stack(left_repeats).median(dim=0).values
            right_loss = torch.stack(right_repeats).median(dim=0).values
            for index, sample_id in enumerate(left["sample_ids"]):
                row = {
                    "sample_id": sample_id,
                    "proof_family": left["proof_families"][index],
                    "comparison": label,
                    "left_loss": float(left_loss[index]),
                    "right_loss": float(right_loss[index]),
                    "gain": float(left_loss[index] - right_loss[index]),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    accelerator.wait_for_everyone()


def merge_rank_outputs(root: Path, world_size: int) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for rank in range(world_size):
        path = root / f"rank_{rank}.jsonl"
        for row in iter_jsonl(path):
            merged[(str(row["sample_id"]), str(row["comparison"]))] = row
    return [merged[key] for key in sorted(merged)]


def choose_budget(summary: dict[str, dict[str, float]], threshold: float) -> str:
    absolute = {
        key.removeprefix("absolute/"): value["mean_gain"]
        for key, value in summary.items()
        if key.startswith("absolute/")
    }
    if not absolute:
        raise ValueError("no absolute Context-budget comparisons")
    maximum = max(absolute.values())
    if maximum <= 0:
        return "none"
    eligible = [budget for budget, gain in absolute.items() if gain >= threshold * maximum]
    for budget in sorted(eligible, key=budget_sort_key):
        hard = summary.get(f"hard/{budget}")
        if hard is None or hard["mean_gain"] > 0:
            return budget
    return max(absolute, key=absolute.get)


def run_information_gain(
    config: dict[str, Any],
    *,
    architecture: str,
    loss_function: Callable[..., Any],
) -> dict[str, Any]:
    accelerator = build_accelerator(config)
    tokenizer = load_tokenizer(config, tokenized=True)
    loss_config = resolved_loss_config(config, architecture=architecture, tokenizer=tokenizer)
    model = load_model(config, architecture=architecture, tokenizer=tokenizer, trainable=False)
    model = accelerator.prepare(model)
    model.eval()
    gain_config = config["information_gain"]
    allow_ids = load_id_set(gain_config["id_file"])
    tokenized_root = Path(config["data"]["tokenized_root"])
    output_root = Path(gain_config["output_dir"])
    if accelerator.is_main_process and output_root.exists():
        shutil.rmtree(output_root)
    accelerator.wait_for_everyone()
    repeats = int(gain_config.get("mask_repeats", 8 if architecture == "dllm" else 1))
    batch_size = int(config["training"]["per_device_eval_batch_size"])
    workers = int(config["training"].get("num_workers", 4))
    stage = gain_config.get("stage", "main")
    budgets = [str(value) for value in gain_config["budgets"]]
    comparisons: list[tuple[str, Path, Path]] = []
    without = tokenized_root / "statement_only" / "b0" / "train.jsonl"
    for budget in budgets:
        matched = tokenized_root / "matched" / budget / "train.jsonl"
        comparisons.append((f"absolute/{budget}", without, matched))
        for control, label in (("random_control", "random"), ("hard_control", "hard")):
            control_file = tokenized_root / control / budget / "train.jsonl"
            if control_file.is_file():
                comparisons.append((f"{label}/{budget}", control_file, matched))
    for label, left_file, right_file in comparisons:
        safe = label.replace("/", "__")
        evaluate_pair(
            model=model,
            accelerator=accelerator,
            tokenizer=tokenizer,
            left_file=left_file,
            right_file=right_file,
            allow_ids=allow_ids,
            stage=stage,
            repeats=repeats,
            batch_size=batch_size,
            workers=workers,
            loss_function=loss_function,
            loss_config=loss_config,
            output_file=output_root / safe / f"rank_{accelerator.process_index}.jsonl",
            label=label,
        )

    result: dict[str, Any] = {}
    if accelerator.is_main_process:
        all_scores: list[dict[str, Any]] = []
        summary: dict[str, dict[str, float]] = {}
        by_family: dict[str, dict[str, dict[str, float]]] = {}
        for label, _, _ in comparisons:
            safe = label.replace("/", "__")
            rows = merge_rank_outputs(output_root / safe, accelerator.num_processes)
            all_scores.extend(rows)
            gains = [float(row["gain"]) for row in rows if math.isfinite(float(row["gain"]))]
            summary[label] = {
                "samples": len(gains),
                "mean_gain": sum(gains) / max(len(gains), 1),
                "positive_fraction": sum(gain > 0 for gain in gains) / max(len(gains), 1),
            }
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                gain = float(row["gain"])
                if math.isfinite(gain):
                    grouped[str(row["proof_family"])].append(gain)
            by_family[label] = {
                family: {
                    "samples": len(values),
                    "mean_gain": sum(values) / len(values),
                    "positive_fraction": sum(value > 0 for value in values) / len(values),
                }
                for family, values in sorted(grouped.items())
            }
        scores_path = output_root / "scores.jsonl"
        with scores_path.open("w", encoding="utf-8") as handle:
            for row in sorted(all_scores, key=lambda item: (item["sample_id"], item["comparison"])):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        selected = choose_budget(summary, float(gain_config.get("selection_fraction", 0.95)))
        result = {
            "selected_budget": selected,
            "selection_fraction": float(gain_config.get("selection_fraction", 0.95)),
            "same_checkpoint": str(config.get("tuning", {}).get("initial_adapter_path") or "base"),
            "summary": summary,
            "by_proof_family": by_family,
        }
        (output_root / "policy.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    return result
