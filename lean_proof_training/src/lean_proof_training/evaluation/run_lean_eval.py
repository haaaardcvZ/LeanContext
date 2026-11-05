#!/usr/bin/env python3
"""Compile generated proof candidates in their original Lean source context."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from lean_proof_training.common.io import iter_jsonl
from lean_proof_training.evaluation.candidates import (
    Candidate,
    load_candidates,
    precheck_proof,
    validate_prediction_coverage,
)
from lean_proof_training.evaluation.lean_tasks import build_task_file
from lean_proof_training.evaluation.summarize_passk import parse_k, write_report
from lean_proof_training.evaluation.passk import summarize


SORRY_WARNING = "declaration uses 'sorry'"
INFRASTRUCTURE_PATTERNS = (
    "error during download",
    "failed to connect",
    "couldn't connect",
    "no such file or directory",
    "toolchain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--predictions-file", type=Path, required=True)
    parser.add_argument("--mathlib-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--placement", choices=("strict_before", "insert_after"), default="strict_before")
    parser.add_argument("--expected-candidates", type=int, default=64)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--k-values", default="1,4,8,16,32,64")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--lean-command", default="lake env lean")
    parser.add_argument("--keep-lean-files", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--keep-failed-lean-files", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def failure_result(
    index_row: dict[str, Any], candidate: Candidate, reason: str, *, placement: str
) -> dict[str, Any]:
    return {
        "sample_id": candidate.sample_id,
        "candidate_id": candidate.candidate_id,
        "proof_family": str(index_row["proof_family"]),
        "proof_hash": candidate.proof_hash,
        "placement": placement,
        "passed": False,
        "failure_reason": reason,
        "exit_code": None,
        "elapsed_seconds": 0.0,
        "lean_file": None,
        "stdout": "",
        "stderr": "",
    }


def run_task(
    *,
    index_row: dict[str, Any],
    candidate: Candidate,
    args: argparse.Namespace,
    source_bytes: bytes,
) -> dict[str, Any]:
    precheck = precheck_proof(candidate.proof)
    if precheck:
        return failure_result(index_row, candidate, precheck, placement=args.placement)
    started = time.time()
    try:
        lean_file = build_task_file(
            index_row=index_row,
            candidate=candidate,
            mathlib_root=args.mathlib_root,
            task_root=args.output_dir / "lean_tasks",
            placement=args.placement,
            source_bytes=source_bytes,
        )
    except Exception as exc:  # setup failures are explicit candidate failures
        row = failure_result(index_row, candidate, "task_construction_error", placement=args.placement)
        row["stderr"] = f"{type(exc).__name__}: {exc}"
        return row
    command = shlex.split(args.lean_command) + [str(lean_file)]
    try:
        process = subprocess.run(
            command,
            cwd=args.mathlib_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(args.timeout),
            check=False,
        )
        combined = process.stdout + "\n" + process.stderr
        sorry_warning = SORRY_WARNING in combined
        passed = process.returncode == 0 and not sorry_warning
        reason = None if passed else ("sorry_warning" if sorry_warning else "lean_error")
        row = {
            "sample_id": candidate.sample_id,
            "candidate_id": candidate.candidate_id,
            "proof_family": str(index_row["proof_family"]),
            "proof_hash": candidate.proof_hash,
            "placement": args.placement,
            "passed": passed,
            "failure_reason": reason,
            "exit_code": process.returncode,
            "elapsed_seconds": time.time() - started,
            "lean_file": str(lean_file),
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-8000:],
        }
        if not args.keep_lean_files and (passed or not args.keep_failed_lean_files):
            lean_file.unlink(missing_ok=True)
            row["lean_file"] = None
        return row
    except subprocess.TimeoutExpired as exc:
        row = failure_result(index_row, candidate, "timeout", placement=args.placement)
        row["elapsed_seconds"] = time.time() - started
        row["lean_file"] = str(lean_file)
        row["stdout"] = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        row["stderr"] = (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else ""
        if not args.keep_lean_files and not args.keep_failed_lean_files:
            lean_file.unlink(missing_ok=True)
            row["lean_file"] = None
        return row


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def verify_lean_environment(args: argparse.Namespace) -> None:
    command = shlex.split(args.lean_command) + ["--version"]
    try:
        process = subprocess.run(
            command,
            cwd=args.mathlib_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(float(args.timeout), 120.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Lean preflight failed: {exc}") from exc
    if process.returncode != 0:
        raise RuntimeError(
            "Lean preflight failed; no pass@k was computed.\n"
            + (process.stdout + "\n" + process.stderr)[-8000:]
        )


def main() -> None:
    args = parse_args()
    if args.expected_candidates <= 0:
        raise ValueError("expected-candidates must be positive")
    verify_lean_environment(args)
    index = {str(row["sample_id"]): row for row in iter_jsonl(args.index_file)}
    if args.max_samples > 0:
        selected = sorted(index)[: args.max_samples]
        index = {sample_id: index[sample_id] for sample_id in selected}
    all_grouped = load_candidates(args.predictions_file)
    if args.max_samples <= 0 and not args.allow_incomplete:
        extra = sorted(set(all_grouped) - set(index))
        if extra:
            raise ValueError(f"predictions contain unknown sample IDs: {extra[:5]}")
    grouped = {sample_id: rows for sample_id, rows in all_grouped.items() if sample_id in index}
    if not args.allow_incomplete:
        validate_prediction_coverage(
            set(index), grouped, expected_candidates=int(args.expected_candidates)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_cache: dict[str, bytes] = {}
    for row in index.values():
        relpath = str(row["relpath"])
        if relpath in source_cache:
            continue
        source = (args.mathlib_root / relpath).read_bytes()
        if hashlib.sha256(source).hexdigest() != str(row["source_sha256"]):
            raise RuntimeError(f"source hash mismatch before evaluation: {relpath}")
        source_cache[relpath] = source
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "index_file": str(args.index_file),
                "predictions_file": str(args.predictions_file),
                "mathlib_root": str(args.mathlib_root),
                "placement": args.placement,
                "placement_warning": (
                    None
                    if args.placement == "strict_before"
                    else "insert_after exposes the original theorem to automation; do not use as the sole main result"
                ),
                "expected_candidates": int(args.expected_candidates),
                "allow_incomplete": bool(args.allow_incomplete),
                "k_values": args.k_values,
                "lean_command": args.lean_command,
                "timeout": float(args.timeout),
                "jobs": int(args.jobs),
                "keep_lean_files": bool(args.keep_lean_files),
                "keep_failed_lean_files": bool(args.keep_failed_lean_files),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    results_file = args.output_dir / "lean_check_results.jsonl"
    results_file.write_text("", encoding="utf-8")
    tasks = [
        (index[sample_id], candidate)
        for sample_id in sorted(index)
        for candidate in grouped.get(sample_id, [])
    ]
    results: list[dict[str, Any]] = []
    completed = 0

    def record(row: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        results.append(row)
        append_result(results_file, row)
        if completed == len(tasks) or completed % 100 == 0:
            passed = sum(bool(item["passed"]) for item in results)
            print(f"[lean-eval] {completed}/{len(tasks)} passed={passed}", flush=True)

    if int(args.jobs) <= 1:
        for index_row, candidate in tasks:
            record(
                run_task(
                    index_row=index_row,
                    candidate=candidate,
                    args=args,
                    source_bytes=source_cache[str(index_row["relpath"])],
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.jobs)) as pool:
            futures = [
                pool.submit(
                    run_task,
                    index_row=index_row,
                    candidate=candidate,
                    args=args,
                    source_bytes=source_cache[str(index_row["relpath"])],
                )
                for index_row, candidate in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                record(future.result())

    results.sort(key=lambda row: (str(row["sample_id"]), int(row["candidate_id"])))
    with results_file.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    infrastructure_failures = [
        row
        for row in results
        if row.get("failure_reason") == "task_construction_error"
        or any(
            pattern in (str(row.get("stdout") or "") + str(row.get("stderr") or "")).lower()
            for pattern in INFRASTRUCTURE_PATTERNS
        )
    ]
    if infrastructure_failures:
        raise RuntimeError(
            f"{len(infrastructure_failures)} infrastructure failures detected; "
            "results were saved but pass@k was not computed"
        )
    k_values = parse_k(args.k_values)
    payload = summarize(results, index, k_values)
    write_report(payload, args.output_dir, k_values)
    summary = {key: value for key, value in payload.items() if key != "per_sample"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
