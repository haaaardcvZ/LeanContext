#!/usr/bin/env python3
"""Compile stratified gold proofs before spending GPU time on generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from .io_utils import load_config, read_jsonl, require_path, write_json
from .proofs import Candidate
from .verify import run_candidate, verify_lean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def stratified_rows(index: dict[str, dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered: set[tuple[str, str]] = set()
    for sample_id in sorted(index):
        row = index[sample_id]
        labels = {
            ("kind", str(row["kind"])),
            ("proof_family", str(row["proof_family"])),
        }
        if labels - covered:
            selected.append(row)
            selected_ids.add(sample_id)
            covered.update(labels)
    for sample_id in sorted(index):
        if len(selected) >= count:
            break
        if sample_id not in selected_ids:
            selected.append(index[sample_id])
    return selected[:count]


def main() -> None:
    config = load_config(parse_args().config)
    output_dir = require_path(config, "output", "directory")
    index_file = output_dir / "source_index.jsonl"
    if not index_file.is_file():
        raise FileNotFoundError(f"run source_index first: {index_file}")
    index = {str(row["sample_id"]): row for row in read_jsonl(index_file)}
    verification = config["verification"]
    count = int(verification.get("gold_preflight_samples", 8))
    if count <= 0:
        print("gold preflight disabled", flush=True)
        return
    mathlib_root = require_path(config, "source", "mathlib_root")
    command = shlex.split(str(verification.get("lean_command", "lake env lean")))
    timeout = float(verification.get("timeout_seconds", 120.0))
    verify_lean(
        command,
        mathlib_root,
        timeout,
        str(verification.get("preflight_import", "Mathlib")),
    )
    selected = stratified_rows(index, min(count, len(index)))
    results: list[dict[str, Any]] = []
    source_cache: dict[str, bytes] = {}
    for candidate_id, index_row in enumerate(selected):
        relpath = str(index_row["relpath"])
        if relpath not in source_cache:
            source = (mathlib_root / relpath).read_bytes()
            if hashlib.sha256(source).hexdigest() != str(index_row["source_sha256"]):
                raise RuntimeError(f"source hash mismatch: {relpath}")
            source_cache[relpath] = source
        candidate = Candidate(
            sample_id=str(index_row["sample_id"]),
            mode="gold_preflight",
            candidate_id=candidate_id,
            proof=str(index_row["gold_target"]),
            proof_hash=hashlib.sha256(
                str(index_row["gold_target"]).encode("utf-8")
            ).hexdigest(),
            raw={},
        )
        result = run_candidate(
            index_row=index_row,
            candidate=candidate,
            mathlib_root=mathlib_root,
            task_root=output_dir / "gold_preflight_tasks",
            source_bytes=source_cache[relpath],
            command=command,
            timeout=timeout,
            keep_all=False,
            keep_failed=True,
        )
        results.append(result)
        print(
            f"[gold-preflight] {candidate_id + 1}/{len(selected)} "
            f"sample={candidate.sample_id} passed={result['passed']}",
            flush=True,
        )
    write_json(
        output_dir / "gold_preflight.json",
        {
            "samples": len(results),
            "passed": sum(bool(row["passed"]) for row in results),
            "results": results,
        },
    )
    failures = [row for row in results if not row["passed"]]
    if failures:
        raise RuntimeError(
            f"gold preflight failed for {len(failures)}/{len(results)} samples; "
            "do not run model generation"
        )
    print("gold preflight passed", flush=True)


if __name__ == "__main__":
    main()
