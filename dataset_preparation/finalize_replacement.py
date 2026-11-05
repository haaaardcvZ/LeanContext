"""Finalize a validated independent replacement extension without editing input data.

The source dataset remains read-only. Source hashes identify the dataset
preparation package used to build and validate the replacement extension.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .matched_replacement import sha256
from .tokenize import iter_jsonl, write_json


def finalize(root: Path, extension: Path) -> dict:
    root, extension = root.resolve(), extension.resolve()
    if extension == root or root in extension.parents:
        raise ValueError("--output-root must be outside --dataset-root")
    out = extension / "matched_replacement"
    report_path = out / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("valid") is not True:
        raise ValueError("Independent replacement validation must pass before finalization")
    validator = Path(__file__).with_name("validate_replacement.py")
    if report.get("validator_sha256") != sha256(validator):
        raise ValueError("Validator source changed; rerun validation")
    frozen = json.loads((out / "frozen_test_manifest.json").read_text(encoding="utf-8"))
    frozen_ids = frozen["target_ids"]
    if len(set(frozen_ids)) != len(frozen_ids) or frozen["actual_final_targets"] != len(frozen_ids):
        raise ValueError("Actual target count is inconsistent")
    if report["actual_final_targets"] != len(frozen_ids):
        raise ValueError("Validated target count changed")
    for model in ("llada", "llama31"):
        for condition in ("Env", "Selected", "Control"):
            path = extension / f"tokenized/{model}/matched_replacement/input_1024/views/{condition}/test.jsonl"
            records = list(iter_jsonl(path))
            if [r["sample_id"] for r in records] != frozen_ids:
                raise ValueError("Generation views differ from frozen membership")
            if any(not r["valid_input"] or r["input_length"] > 1024 for r in records):
                raise ValueError("Generation input budget invalid")
    release = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if release["input_source_test_sha256"] != sha256(root / "benchmark/test.jsonl"):
        raise ValueError("Original benchmark input changed")
    # Check creator bytes before metadata updates; the input dataset is read-only.
    for relative, expected in release["artifacts"].items():
        path = extension / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Release artifact changed: {relative}")
    for name, expected in release["processing_script_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Builder source changed; rebuild extension: {name}")
    release.update(
        independent_validation_status="passed",
        independent_validation_report="matched_replacement/validation_report.json",
        independent_validation_report_sha256=sha256(report_path),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        submitted_dataset_modified=False,
        replacement_changes_generator_input_only_not_verification_source=True,
        scope_match_is_syntactic_only=True,
        serialized_environment_Lean_elaborated=False,
        actual_count_is_not_a_proof_that_no_other_valid_matching_exists=True,
        results_generalization_population="matchable_subset_not_all_benchmark_targets",
        processing_code_package="dataset_preparation",
        processing_package_sha256={
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            if not p.name.startswith("test_")
        },
    )
    release["artifacts"]["matched_replacement/validation_report.json"] = {
        "sha256": sha256(report_path), "bytes": report_path.stat().st_size}
    write_json(out / "manifest.json", release)
    status = json.loads((out / "build_status.json").read_text(encoding="utf-8"))
    status.update(independent_validation_status="passed", release_manifest_sha256=sha256(out / "manifest.json"))
    write_json(out / "build_status.json", status)
    summary = {
        "schema_version": "independent-matched-replacement-extension-release-v1",
        "manifest": "matched_replacement/manifest.json", "sha256": sha256(out / "manifest.json"),
        "sampled_targets": frozen["sampled_targets"], "actual_targets": len(frozen_ids),
        "requested_targets": frozen["requested_targets"],
        "requested_size_reached": frozen["requested_size_reached"],
        "validation_status": "passed", "model_generation_performed": False,
        "serialized_environment_Lean_elaborated": False,
        "submitted_dataset_modified": False,
        "input_source_test_sha256": release["input_source_test_sha256"],
    }
    write_json(extension / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.dataset_root, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
