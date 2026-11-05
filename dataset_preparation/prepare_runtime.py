"""Prepare an independently audited training runtime from the paper-format release.

The default command prints a read-only plan. Execution requires --execute and
a new output directory. The saved four views are copied from the canonical
JSON array, not selected again. Both tokenizers operate offline; no model
weights, generation, training, or Lean execution are needed for preparation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator, TextIO

from .tokenize import SPLITS, VIEWS, load_actual_tokenizer, run_selector, write_json, write_row
from .tokenize_budgets import run as tokenize_training
from .summarize_lengths import run as measure_lengths
from .validate_tokenized import TokenizedValidator


CANONICAL = "dataset.complete.with_four_views.json"
EXPECTED_COUNTS = {"train": 90879, "validation": 19473, "test": 19474}
MODEL_BACKBONES = {"llada": "dllm", "llama31": "llama31"}
SUPPORTING_DIRECTORIES = (
    "command_mapping", "selector", "source_snapshot", "tokenizers",
    "evaluation_populations", "analysis_subsets", "private_posthoc_audit", "audit", "processing",
)
VALIDATION_SCOPE = "exact_token_ids_spans_one_EOS_whole_command_packing"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def iter_array(handle: TextIO, chunk_size: int = 65536) -> Iterator[dict[str, Any]]:
    """Read a standard JSON array without loading its multi-gigabyte contents."""
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False

    def read_more() -> bool:
        nonlocal buffer, eof
        value = handle.read(chunk_size)
        if not value:
            eof = True
            return False
        buffer += value
        return True

    def space() -> None:
        nonlocal buffer
        while True:
            buffer = buffer.lstrip()
            if buffer or eof or not read_more():
                return

    space()
    if not buffer.startswith("["):
        raise ValueError("The canonical dataset must be a standard JSON array")
    buffer = buffer[1:]
    first = True
    while True:
        space()
        if buffer.startswith("]"):
            buffer = buffer[1:]
            space()
            if buffer:
                raise ValueError("Unexpected content after the canonical JSON array")
            return
        if not first:
            if not buffer.startswith(","):
                raise ValueError("Missing comma or closing bracket in the canonical JSON array")
            buffer = buffer[1:]
            space()
            if buffer.startswith("]"):
                raise ValueError("A trailing comma is not valid JSON")
        while True:
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if eof or not read_more():
                    raise ValueError("Invalid or incomplete canonical JSON record") from exc
                continue
            break
        if not isinstance(value, dict):
            raise ValueError("Every canonical record must be a JSON object")
        buffer = buffer[end:]
        yield value
        first = False


def validate_record(record: dict[str, Any]) -> None:
    sid, split = record["sample_id"], record["split"]
    if not isinstance(sid, str) or split not in SPLITS or set(record["views"]) != set(VIEWS):
        raise ValueError("Invalid target ID, split, or four-view set")
    statement, proof = record["statement"], record["target"]
    if not isinstance(statement, str) or not statement.strip() or not isinstance(proof, str) or not proof.strip():
        raise ValueError(f"{sid}: empty statement or reference proof")
    if record.get("verified_by_lean") is not True:
        raise ValueError(f"{sid}: the release reference-proof verification flag is not true")
    full = record["views"]["CFull"]["context_items"]
    if full != sorted(full, key=lambda item: (item["source_order"], item["command_uid"])):
        raise ValueError(f"{sid}: saved Full context is not in source order")
    expected = {"C0": [], "CMath": [item for item in full if item["role"] == "MC"],
                "CEnv": [item for item in full if item["role"] == "EC"], "CFull": full}
    if len({item["command_uid"] for item in full}) != len(full) or any(item["role"] not in {"MC", "EC"} for item in full):
        raise ValueError(f"{sid}: duplicate or invalid context roles")
    for view in VIEWS:
        row = record["views"][view]
        if (row["sample_id"], row["target_id"], row["split"], row["view"], row["statement"], row["target"]) != (sid, sid, split, view, statement, proof):
            raise ValueError(f"{sid}/{view}: target, proof, or view metadata changed")
        if row["context_items"] != expected[view]:
            raise ValueError(f"{sid}/{view}: saved views are not the exact Full-context projections")
        prompt = "\n\n".join([*(item["text"] for item in expected[view]), statement]).rstrip("\r\n") + "\n"
        if row["prompt"] != prompt or row["context_ids"] != [item["command_id"] for item in expected[view]] or row["context_uids"] != [item["command_uid"] for item in expected[view]]:
            raise ValueError(f"{sid}/{view}: saved serializer or command identities changed")


def export_saved_views(source: Path, destination: Path,
                       expected_counts: dict[str, int] = EXPECTED_COUNTS) -> dict[str, int]:
    """Export immutable saved records and views with unchanged field values."""
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with ExitStack() as stack:
        outputs = {}
        benchmarks = {}
        for split in SPLITS:
            path = destination / "benchmark" / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            benchmarks[split] = stack.enter_context(path.open("w", encoding="utf-8"))
            for view in VIEWS:
                path = destination / "training" / "views" / view / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                outputs[split, view] = stack.enter_context(path.open("w", encoding="utf-8"))
        handle = stack.enter_context(source.open(encoding="utf-8"))
        for record in iter_array(handle):
            validate_record(record)
            sid, split = record["sample_id"], record["split"]
            if sid in seen:
                raise ValueError(f"Duplicate target across splits: {sid}")
            seen.add(sid)
            counts[split] += 1
            write_row(benchmarks[split], {key: value for key, value in record.items() if key != "views"})
            for view in VIEWS:
                write_row(outputs[split, view], record["views"][view])
    actual = {split: counts[split] for split in SPLITS}
    if actual != expected_counts:
        raise ValueError(f"Canonical split counts differ: {actual}; expected {expected_counts}")
    return actual


def plan(source: Path, output: Path, models: list[str], materialization: str) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if not models or not set(models) <= set(MODEL_BACKBONES):
        raise ValueError("Select llada, llama31, or both models")
    if output.exists():
        raise ValueError("Use a new output directory; existing runtime artifacts are never overwritten")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("The runtime and source dataset directories must not overlap")
    required = [source / CANONICAL, source / "selector/pairs.jsonl",
                source / "selector/frozen_test_manifest.json", source / "source_snapshot"]
    required.extend(source / "tokenizers" / model / filename for model in models
                    for filename in ("tokenizer.json", "tokenizer_config.json"))
    missing = [str(path) for path in required if not path.exists()]
    if not (source / "command_mapping/source_commands").is_dir() and not (source / "command_mapping/source_commands.jsonl").is_file():
        missing.append(str(source / "command_mapping/source_commands"))
    if missing:
        raise FileNotFoundError("Missing release artifacts: " + "; ".join(missing))
    return {"mode": "plan", "source_dataset": str(source), "runtime_dataset": str(output),
            "models": models, "materialization": materialization,
            "expected_raw_split_counts": EXPECTED_COUNTS, "input_limit": 1024,
            "complete_proof_plus_one_EOS_limit": 1024, "maximum_sequence_length": 2048,
            "uses_saved_four_views_without_reselection": True,
            "independent_token_id_and_packing_validation": True,
            "training_configs_DATASET_ROOT": str(output), "writes_performed": False}


def mark_validated(path: Path, scope: str = VALIDATION_SCOPE) -> None:
    temporary = path.with_suffix(path.suffix + ".validated")
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
        for line in source:
            if line.strip():
                row = json.loads(line)
                row.update(tokenization_validated=True, tokenization_validation_scope=scope)
                write_row(output, row)
    temporary.replace(path)


def refresh_cka_sources(output: Path, models: list[str], artifacts: dict[str, Any]) -> None:
    """Rebind runtime file hashes without changing the frozen CKA sample."""
    path = output / "analysis_subsets/cka_4000_manifest.json"
    if not path.is_file() or set(models) != set(MODEL_BACKBONES):
        return
    from analysis.representation_analysis import make_sample_manifest, validate_sample_manifest
    original = json.loads(path.read_text(encoding="utf-8"))
    validate_sample_manifest(original)
    lengths, sources = {}, {}
    for model in MODEL_BACKBONES:
        with (output / "tokenized" / model / "training/membership/test.jsonl").open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        lengths[model] = {row["sample_id"]: row["answer_length"] for row in records if row["retained"]}
        for view in VIEWS:
            relative = f"tokenized/{model}/training/input_1024/views/{view}/test.jsonl"
            sources[f"{model}/{view}"] = {"sha256": artifacts[relative]["sha256"], "records": len(lengths[model])}
    updated = make_sample_manifest(lengths, sources=sources,
        seed=original["sampling_seed"], size=original["sample_count"], mask_seed=original["mask_seed"])
    validate_sample_manifest(updated)
    if updated["target_ids"] != original["target_ids"]:
        raise ValueError("Runtime membership changed the frozen CKA sample; do not silently resample")
    # Preserve all original policy and provenance fields, replacing file
    # identities only. The entire sample has passed current eligibility checks.
    original["sources"] = sources
    original["source_hash_binding"] = "independently_validated_local_runtime_tokenized_files"
    write_json(path, original)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source, output = args.dataset_root.resolve(), args.output_root.resolve()
    requested = list(dict.fromkeys(args.models))
    result = plan(source, output, requested, args.materialization)
    if not args.execute:
        return result
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preparation_status.json", {"status": "in_progress", "source": str(source)})
    for name in SUPPORTING_DIRECTORIES:
        original = source / name
        if not original.exists():
            continue
        target = output / name
        if args.materialization == "symlink" and name != "analysis_subsets":
            target.symlink_to(original, target_is_directory=True)
        else:
            shutil.copytree(original, target)
    for original in sorted(source.glob("*.json")):
        if original.name != CANONICAL:
            shutil.copy2(original, output / original.name)
    counts = export_saved_views(source / CANONICAL, output)
    reports = {}
    for name in requested:
        backbone = MODEL_BACKBONES[name]
        config = argparse.Namespace(dataset_root=output, tokenizer=str(output / "tokenizers" / name),
            tokenizer_revision=None, backbone=backbone, trust_remote_code=False,
            output_dir=output / "tokenized" / name / "training", input_budgets=[1024], output_limit=1024)
        tokenizer, metadata = load_actual_tokenizer(config)
        training = tokenize_training(config, tokenizer, metadata)
        config.output_dir = output / "tokenized" / name / "selector/input_1024"
        config.input_limit, config.selector_input_file = 1024, None
        selector = run_selector(config, tokenizer, metadata)
        config.output_dir = output / "tokenized" / name / "raw_length_statistics"
        measure_lengths(config)
        auditor = TokenizedValidator(output, backbone, output / "tokenizers" / name)
        auditor.run()
        report_path = output / "tokenized" / name / "validation_report.json"
        write_json(report_path, auditor.report)
        for view in VIEWS:
            for split in SPLITS:
                mark_validated(output / "tokenized" / name / "training/input_1024/views" / view / f"{split}.jsonl")
        mark_validated(output / "tokenized" / name / "selector/input_1024/pairs.jsonl",
                       "exact_token_ids_whole_command_shared_EC_and_paired_MC_packing")
        reports[name] = {"training_retained": {split: item["retained"] for split, item in training["splits"].items()},
                         "selector_targets": selector["seen"], "independent_validation": str(report_path.relative_to(output))}
    artifacts = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted((output / "tokenized").rglob("*")) if path.is_file()}
    refresh_cka_sources(output, requested, artifacts)
    write_json(output / "tokenization_manifest.json", {"schema_version": "leancontext-paper-runtime-v1",
        "source_canonical_sha256": sha256(source / CANONICAL), "raw_split_counts": counts,
        "input_limit": 1024, "proof_plus_one_EOS_limit": 1024,
        "maximum_sequence_length": 2048, "models": reports, "artifacts": artifacts})
    final = {**result, "mode": "complete", "writes_performed": True, "raw_split_counts": counts, "models": reports}
    write_json(output / "preparation_status.json", {"status": "complete", **final})
    return final


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted source dataset")
    parser.add_argument("--output-root", type=Path, required=True, help="New writable training-runtime directory")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_BACKBONES), default=list(MODEL_BACKBONES))
    parser.add_argument("--materialization", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--execute", action="store_true", help="Prepare and independently audit the runtime")
    print(json.dumps(prepare(parser.parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
