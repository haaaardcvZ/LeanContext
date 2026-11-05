#!/usr/bin/env python3
"""Generate 32 sampled candidates plus one deterministic temperature-0 proof."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from .current_inputs import input_for_generation, invalid_input_candidates, selector_input_for_generation
from .generators import ARGenerator, DLLMGenerator
from .io_utils import (
    append_jsonl,
    load_config,
    read_jsonl,
    require_path,
    select_rows,
    stable_int,
    write_json,
    write_jsonl,
)
from .model_loader import load_model_and_tokenizer
from .replacement_inputs import replacement_input_for_generation, validate_replacement_artifacts, unique_rows
from .reconstruction_inputs import reconstruction_input_for_generation, validate_reconstruction_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def load_allowed_ids(path_value: Any) -> set[str] | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    ids: set[str] = set()
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if sample_id:
            ids.add(sample_id)
    return ids


def generation_rows(
    *,
    generator: Any,
    row: dict[str, Any],
    prompt: str,
    prompt_ids: list[int],
    context_items: int,
    generation: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    sampled_config = generation["sampled"]
    greedy_config = generation["greedy"]
    sample_count = int(sampled_config.get("num_candidates", 32))
    if sample_count != 32:
        raise ValueError("formal evaluation requires sampled.num_candidates=32")
    temperature = float(sampled_config.get("temperature", 0.6))
    if temperature <= 0:
        raise ValueError("sampled.temperature must be > 0; temperature=0 is the greedy run")
    if int(greedy_config.get("num_candidates", 1)) != 1:
        raise ValueError("greedy.num_candidates must be 1")
    if float(greedy_config.get("temperature", 0.0)) != 0.0:
        raise ValueError("greedy.temperature must be exactly 0")

    input_limit = int(generation.get("input_limit", 1024))
    max_new_tokens = int(generation.get("max_new_tokens", 1024))
    if max_new_tokens != 1024:
        raise ValueError("controlled protocol requires independent max_new_tokens=1024")
    if len(prompt_ids) > input_limit:
        raise ValueError(
            f"{row['sample_id']}: prompt has {len(prompt_ids)} tokens, above input_limit={input_limit}"
        )
    batch_size = max(1, int(generation.get("candidate_batch_size", 4)))
    common = {
        "sample_id": str(row["sample_id"]),
        "proof_family": str(row["proof_family"]),
        "context_view": str(generation["context_view"]),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_tokens": len(prompt_ids),
        "context_items": context_items,
        "max_new_tokens": max_new_tokens,
        "input_valid": True,
        "forbid_reasoning_markers": list(generation.get("forbid_reasoning_markers") or []),
    }
    output: list[dict[str, Any]] = []
    for start in range(0, sample_count, batch_size):
        count = min(batch_size, sample_count - start)
        chunk_seed = stable_int(seed, row["sample_id"], "sampled", start)
        proofs = generator.generate(
            prompt_ids,
            count=count,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=float(sampled_config.get("top_p", 0.95)),
            top_k=int(sampled_config.get("top_k", 0)),
            seed=chunk_seed,
        )
        if len(proofs) != count:
            raise RuntimeError("generation backend returned the wrong number of proofs")
        for offset, proof in enumerate(proofs):
            output.append(
                {
                    **common,
                    "mode": "sampled",
                    "candidate_id": start + offset,
                    "proof": proof,
                    "temperature": temperature,
                    "top_p": float(sampled_config.get("top_p", 0.95)),
                    "top_k": int(sampled_config.get("top_k", 0)),
                    "seed": chunk_seed,
                }
            )

    greedy_seed = stable_int(seed, row["sample_id"], "greedy_temperature_0")
    greedy_proofs = generator.generate(
        prompt_ids,
        count=1,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=greedy_seed,
    )
    output.append(
        {
            **common,
            "mode": "greedy_temperature_0",
            "candidate_id": 0,
            "proof": greedy_proofs[0],
            # AR generation uses do_sample=False. Temperature is therefore not
            # passed to transformers.generate; 0 here records its semantics.
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": greedy_seed,
        }
    )
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    generation = config["generation"]
    input_format = str(generation.get("input_format", "training_view"))
    replacement_raw_rows = None
    if input_format == "replacement_pair":
        # Check frozen membership and input comparability before loading weights.
        validate_replacement_artifacts(config)
        replacement_raw_rows = unique_rows(require_path(config, "data", "replacement_raw_pairs_file"))
    elif input_format == "reconstruction_pair":
        validate_reconstruction_artifacts(config)
    rank, world_size, local_rank, initialized_here = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("model generation requires CUDA")
    device = torch.device("cuda", local_rank)
    architecture = str(config["architecture"]).lower()
    model, tokenizer, model_metadata = load_model_and_tokenizer(
        config, architecture=architecture, device=device
    )
    if architecture == "ar":
        backend: Any = ARGenerator(model, tokenizer, device)
    else:
        backend = DLLMGenerator(
            model,
            tokenizer,
            device,
            mask_token_id=generation.get("mask_token_id"),
            num_steps=int(generation.get("num_diffusion_steps", 64)),
        )

    dataset_file = require_path(config, "data", "test_file")
    tokenized_file = require_path(config, "data", "tokenized_input_file")
    tokenized_rows = {str(row["sample_id"]): row for row in read_jsonl(tokenized_file)}
    allowed_ids = load_allowed_ids(config.get("data", {}).get("allowed_ids_file"))
    rows = [
        row
        for row in read_jsonl(dataset_file)
        if allowed_ids is None or str(row.get("sample_id")) in allowed_ids
    ]
    max_samples = (
        int(args.max_samples)
        if args.max_samples is not None
        else int(config.get("evaluation", {}).get("max_samples", 0))
    )
    seed = int(config.get("seed", 20260818))
    rows = select_rows(rows, max_samples=max_samples, seed=seed)
    if bool(config.get("evaluation", {}).get("require_all_targets", True)):
        if max_samples > 0:
            raise ValueError("declared-population evaluation forbids max-sample filtering")
        if allowed_ids is not None and {str(row["sample_id"]) for row in rows} != allowed_ids:
            raise ValueError("dataset does not contain every ID in the declared evaluation population")
        expected_population = config.get("evaluation", {}).get("expected_population_size")
        if expected_population is not None and len(rows) != int(expected_population):
            raise ValueError(f"expected evaluation population{expected_population}, found{len(rows)}")
    local_rows = [row for index, row in enumerate(rows) if index % world_size == rank]
    output_dir = require_path(config, "output", "directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_jsonl(output_dir / "evaluation_targets.jsonl", [{"sample_id": str(row["sample_id"])} for row in rows])
    shard = output_dir / f"predictions.rank_{rank:05d}.jsonl"
    shard.write_text("", encoding="utf-8")
    started = time.time()
    for index, row in enumerate(local_rows, start=1):
        if input_format == "selector_pair":
            pair = tokenized_rows.get(str(row["sample_id"]))
            if pair is None or str(pair["statement"]) != str(row["statement"]):
                raise ValueError("selector pair is missing or changes the target statement")
            prompt, prompt_ids, context_items, invalid = selector_input_for_generation(
                pair, policy=str(generation["selector_policy"]), tokenizer=tokenizer,
                backbone="llama31" if architecture == "ar" else "llada",
                input_limit=int(generation.get("input_limit", 1024)))
        elif input_format == "replacement_pair":
            sid = str(row["sample_id"])
            pair = tokenized_rows.get(sid)
            if pair is None or replacement_raw_rows is None or sid not in replacement_raw_rows:
                raise ValueError("replacement is missing a frozen target; no fallback or target dropping is allowed")
            prompt, prompt_ids, context_items, invalid = replacement_input_for_generation(
                row, pair, replacement_raw_rows[sid], condition=str(generation["replacement_condition"]),
                tokenizer=tokenizer, backbone="llama31" if architecture == "ar" else "llada")
        elif input_format == "reconstruction_pair":
            pair = tokenized_rows.get(str(row["sample_id"]))
            if pair is None:
                raise ValueError("missing frozen reconstruction target; no fallback or dropping is allowed")
            prompt, prompt_ids, context_items, invalid = reconstruction_input_for_generation(
                row, pair, variant=str(generation["reconstruction_variant"]),
                view=str(generation["reconstruction_view"]), tokenizer=tokenizer,
                backbone="llama31" if architecture == "ar" else "llada")
        elif input_format == "training_view":
            prompt, prompt_ids, context_items, invalid = input_for_generation(
                row, tokenized_rows.get(str(row["sample_id"])),
                view=str(generation["context_view"]), tokenizer=tokenizer,
                backbone=str(generation.get("tokenizer_mode", "llama31" if architecture == "ar" else "llada")),
                input_limit=int(generation.get("input_limit", 1024)))
        else:
            raise ValueError("unsupported generation input_format")
        if invalid is not None:
            invalid_length = None if input_format == "selector_pair" else len(prompt_ids)
            for candidate in invalid_input_candidates(row, str(generation["context_view"]), prompt, invalid_length, invalid):
                append_jsonl(shard, candidate)
            continue
        for candidate in generation_rows(
            generator=backend,
            row=row,
            prompt=prompt,
            prompt_ids=prompt_ids,
            context_items=context_items,
            generation=generation,
            seed=seed,
        ):
            append_jsonl(shard, candidate)
        if index == len(local_rows) or index % 10 == 0:
            print(
                f"[generate rank={rank}] {index}/{len(local_rows)} samples",
                flush=True,
            )

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        combined: list[dict[str, Any]] = []
        for shard_rank in range(world_size):
            combined.extend(read_jsonl(output_dir / f"predictions.rank_{shard_rank:05d}.jsonl"))
        combined.sort(
            key=lambda row: (
                str(row["sample_id"]),
                0 if row["mode"] == "sampled" else 1,
                int(row["candidate_id"]),
            )
        )
        predictions_file = output_dir / "predictions.jsonl"
        write_jsonl(predictions_file, combined)
        write_json(
            output_dir / "generation_manifest.json",
            {
                "config_file": config["_config_path"],
                "dataset_file": str(dataset_file),
                "tokenized_input_file": str(tokenized_file),
                "generation_target_filter_uses_reference_proof": False,
                "selected_samples": len(rows),
                "world_size": world_size,
                "sampled_candidates_per_sample": 32,
                "deterministic_candidates_per_sample": 1,
                "deterministic_mode": "greedy argmax (temperature=0 semantics)",
                "model": model_metadata,
                "generation": generation,
                "intervention": config.get("intervention"),
                "elapsed_seconds_rank_0": time.time() - started,
                "predictions_file": str(predictions_file),
            },
        )
        print(f"wrote {predictions_file}", flush=True)
    if world_size > 1:
        torch.distributed.barrier()
    del backend, model
    gc.collect()
    torch.cuda.empty_cache()
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
