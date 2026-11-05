#!/usr/bin/env python3
"""Replace original proofs, compile with Lean, classify errors, and compute metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .error_types import classify
from .io_utils import append_jsonl, load_config, read_jsonl, require_path, write_json, write_jsonl
from .lean_replacement import replace_original_declaration
from .metrics import summarize, validate_coverage
from .proofs import Candidate, load_candidates, precheck_proof
from .reporting import write_reports


SORRY_WARNING = "declaration uses 'sorry'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_result(index_row: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    return {
        "sample_id": candidate.sample_id,
        "mode": candidate.mode,
        "candidate_id": candidate.candidate_id,
        "proof_family": str(index_row["proof_family"]),
        "proof_hash": candidate.proof_hash,
        "passed": False,
        "error_type": None,
        "error_stage": None,
        "diagnostic": "",
        "exit_code": None,
        "elapsed_seconds": 0.0,
        "lean_file": None,
        "stdout": "",
        "stderr": "",
    }


def run_candidate(
    *,
    index_row: dict[str, Any],
    candidate: Candidate,
    mathlib_root: Path,
    task_root: Path,
    source_bytes: bytes,
    command: list[str],
    timeout: float,
    keep_all: bool,
    keep_failed: bool,
) -> dict[str, Any]:
    row = base_result(index_row, candidate)
    precheck = candidate.protocol_error or precheck_proof(candidate.proof)
    if precheck:
        row.update(classify(passed=False, precheck=precheck))
        return row
    started = time.time()
    try:
        lean_file = replace_original_declaration(
            index_row=index_row,
            candidate=candidate,
            mathlib_root=mathlib_root,
            task_root=task_root,
            source_bytes=source_bytes,
            source_verified=True,
        )
        row["lean_file"] = str(lean_file)
    except Exception as exc:  # exact source/index failures are infrastructure failures
        row["stderr"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = time.time() - started
        row.update(
            classify(
                passed=False,
                stderr=row["stderr"],
                task_construction_error=True,
            )
        )
        return row
    try:
        process = subprocess.run(
            [*command, str(lean_file)],
            cwd=mathlib_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stdout = process.stdout[-8000:]
        stderr = process.stderr[-16000:]
        sorry = SORRY_WARNING in (stdout + "\n" + stderr)
        passed = process.returncode == 0 and not sorry
        row.update(
            {
                "passed": passed,
                "exit_code": process.returncode,
                "elapsed_seconds": time.time() - started,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        row.update(classify(passed=passed, stdout=stdout, stderr=stderr))
    except subprocess.TimeoutExpired as exc:
        row.update(
            {
                "elapsed_seconds": time.time() - started,
                "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-16000:] if isinstance(exc.stderr, str) else "",
            }
        )
        row.update(
            classify(
                passed=False,
                stdout=row["stdout"],
                stderr=row["stderr"],
                timed_out=True,
            )
        )
    except OSError as exc:
        row["stderr"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = time.time() - started
        row.update(
            {
                "error_type": "infrastructure_error",
                "error_stage": "lean_execution",
                "diagnostic": row["stderr"],
            }
        )
    if not keep_all and (row["passed"] or not keep_failed):
        lean_file.unlink(missing_ok=True)
        row["lean_file"] = None
    return row


def verify_lean(
    command: list[str], mathlib_root: Path, timeout: float, preflight_import: str
) -> str:
    try:
        process = subprocess.run(
            [*command, "--version"],
            cwd=mathlib_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(timeout, 120.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Lean preflight failed: {exc}") from exc
    if process.returncode != 0:
        raise RuntimeError(
            "Lean preflight failed:\n" + (process.stdout + "\n" + process.stderr)[-8000:]
        )
    version = (process.stdout + "\n" + process.stderr).strip()
    # `lean --version` can succeed even when the pinned Mathlib checkout has no
    # compiled oleans. Compile a real import so this cannot become 600k fake
    # model failures later in the run.
    with tempfile.TemporaryDirectory(prefix="lean_eval_preflight_") as temporary:
        probe = Path(temporary) / "Preflight.lean"
        probe.write_text(f"import {preflight_import}\n#check True\n", encoding="utf-8")
        try:
            import_process = subprocess.run(
                [*command, str(probe)],
                cwd=mathlib_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Mathlib import preflight failed: {exc}") from exc
        if import_process.returncode != 0:
            raise RuntimeError(
                f"Mathlib import preflight failed for `import {preflight_import}`:\n"
                + (import_process.stdout + "\n" + import_process.stderr)[-8000:]
            )
    return version


def validate_prediction_protocol(
    candidates: list[Candidate], index: dict[str, dict[str, Any]]
) -> None:
    skeleton = [
        {
            "sample_id": candidate.sample_id,
            "mode": candidate.mode,
            "candidate_id": candidate.candidate_id,
            "passed": False,
        }
        for candidate in candidates
    ]
    validate_coverage(skeleton, index)


def main() -> None:
    config = load_config(parse_args().config)
    output_dir = require_path(config, "output", "directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "source_index.jsonl"
    predictions_file = output_dir / "predictions.jsonl"
    if not index_file.is_file():
        raise FileNotFoundError(f"run source_index first: {index_file}")
    if not predictions_file.is_file():
        raise FileNotFoundError(f"run generate first: {predictions_file}")
    full_index = {str(row["sample_id"]): row for row in read_jsonl(index_file)}
    candidates = load_candidates(list(read_jsonl(predictions_file)))
    prediction_ids = {candidate.sample_id for candidate in candidates}
    target_file = output_dir / "evaluation_targets.jsonl"
    if not target_file.is_file():
        raise FileNotFoundError("missing evaluation_targets.jsonl; run the evaluation generator first")
    expected_ids = {str(row["sample_id"]) for row in read_jsonl(target_file)}
    if prediction_ids != expected_ids:
        raise ValueError("prediction population differs from declared evaluation denominator")
    unknown = sorted(prediction_ids - set(full_index))
    if unknown:
        raise ValueError(f"predictions contain unknown sample IDs: {unknown[:10]}")
    index = {sample_id: full_index[sample_id] for sample_id in sorted(expected_ids)}
    validate_prediction_protocol(candidates, index)

    verification = config["verification"]
    mathlib_root = require_path(config, "source", "mathlib_root")
    command = shlex.split(str(verification.get("lean_command", "lake env lean")))
    timeout = float(verification.get("timeout_seconds", 120.0))
    preflight_import = str(verification.get("preflight_import", "Mathlib"))
    lean_version = verify_lean(command, mathlib_root, timeout, preflight_import)
    source_cache: dict[str, bytes] = {}
    for row in index.values():
        relpath = str(row["relpath"])
        if relpath in source_cache:
            continue
        source = (mathlib_root / relpath).read_bytes()
        if hashlib.sha256(source).hexdigest() != str(row["source_sha256"]):
            raise RuntimeError(f"source hash mismatch before evaluation: {relpath}")
        source_cache[relpath] = source

    manifest = {
        "config_file": config["_config_path"],
        "index_file": str(index_file),
        "index_sha256": file_sha256(index_file),
        "predictions_file": str(predictions_file),
        "predictions_sha256": file_sha256(predictions_file),
        "mathlib_root": str(mathlib_root),
        "lean_command": command,
        "lean_version": lean_version,
        "preflight_import": preflight_import,
        "replacement_policy": "exact replacement of original declaration byte span",
        "timeout_seconds": timeout,
        "heartbeat_policy": "source_inherited",
        "jobs": int(verification.get("jobs", 4)),
    }
    manifest_file = output_dir / "verification_manifest.json"
    results_file = output_dir / "lean_results.jsonl"
    resume = bool(verification.get("resume", True))
    existing: list[dict[str, Any]] = []
    if resume and results_file.is_file() and manifest_file.is_file():
        old_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for key in ("index_sha256", "predictions_sha256", "mathlib_root", "lean_command"):
            if old_manifest.get(key) != manifest.get(key):
                raise RuntimeError(f"cannot resume: verification manifest changed at {key}")
        existing = list(read_jsonl(results_file))
    else:
        results_file.write_text("", encoding="utf-8")
    write_json(manifest_file, manifest)

    completed_keys = {
        (str(row["sample_id"]), str(row["mode"]), int(row["candidate_id"]))
        for row in existing
    }
    tasks = [
        candidate
        for candidate in candidates
        if (candidate.sample_id, candidate.mode, candidate.candidate_id) not in completed_keys
    ]
    results = list(existing)
    jobs = max(1, int(verification.get("jobs", 4)))
    keep_all = bool(verification.get("keep_lean_files", False))
    keep_failed = bool(verification.get("keep_failed_lean_files", False))
    task_root = output_dir / "lean_tasks"
    started = time.time()

    def submit_args(candidate: Candidate) -> dict[str, Any]:
        index_row = index[candidate.sample_id]
        return {
            "index_row": index_row,
            "candidate": candidate,
            "mathlib_root": mathlib_root,
            "task_root": task_root,
            "source_bytes": source_cache[str(index_row["relpath"])],
            "command": command,
            "timeout": timeout,
            "keep_all": keep_all,
            "keep_failed": keep_failed,
        }

    done = 0
    if jobs == 1:
        iterator = (run_candidate(**submit_args(candidate)) for candidate in tasks)
        for row in iterator:
            results.append(row)
            append_jsonl(results_file, row)
            done += 1
            if done == len(tasks) or done % 100 == 0:
                print(f"[lean] {done}/{len(tasks)}", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_candidate, **submit_args(candidate)) for candidate in tasks]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                results.append(row)
                append_jsonl(results_file, row)
                done += 1
                if done == len(tasks) or done % 100 == 0:
                    print(f"[lean] {done}/{len(tasks)}", flush=True)

    results.sort(
        key=lambda row: (
            str(row["sample_id"]),
            0 if row["mode"] == "sampled" else 1,
            int(row["candidate_id"]),
        )
    )
    write_jsonl(results_file, results)
    infrastructure = [
        row
        for row in results
        if row["error_type"] in {"infrastructure_error", "task_construction_error"}
    ]
    if infrastructure:
        write_json(
            output_dir / "infrastructure_failures.json",
            {"count": len(infrastructure), "examples": infrastructure[:20]},
        )
        raise RuntimeError(
            f"{len(infrastructure)} infrastructure failures detected; Pass@k was not computed"
        )
    payload = summarize(results, index)
    write_reports(output_dir, payload, results)
    manifest["elapsed_seconds"] = time.time() - started
    manifest["evaluated_candidates"] = len(results)
    write_json(manifest_file, manifest)
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
