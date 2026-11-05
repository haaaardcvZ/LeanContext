#!/usr/bin/env python3
"""Join public samples to exact original Lean command byte spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .io_utils import load_config, read_jsonl, require_path, sha256_bytes, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def private_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = [path / "train.jsonl"] if (path / "train.jsonl").is_file() else []
    validation = path / "val.jsonl"
    if not validation.is_file():
        validation = path / "validation.jsonl"
    if validation.is_file():
        candidates.append(validation)
    if (path / "test.jsonl").is_file():
        candidates.append(path / "test.jsonl")
    if not candidates:
        raise FileNotFoundError(f"no prepared JSONL shards found in {path}")
    return candidates


def load_private_rows(path: Path, needed_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for shard in private_files(path):
        for row in read_jsonl(shard):
            sample_id = str(row.get("sample_id") or "")
            if sample_id not in needed_ids:
                continue
            if sample_id in result:
                raise ValueError(f"duplicate private sample_id: {sample_id}")
            result[sample_id] = row
    missing = sorted(needed_ids - set(result))
    if missing:
        raise ValueError(f"private prepared data misses public IDs: {missing[:10]}")
    return result


def command_path(command_root: Path, relpath: str) -> Path:
    return command_root / Path(relpath).with_suffix(".jsonl")


def load_commands(
    command_root: Path, requests: dict[str, set[str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    commands: dict[tuple[str, str], dict[str, Any]] = {}
    for relpath, command_ids in sorted(requests.items()):
        path = command_path(command_root, relpath)
        if not path.is_file():
            raise FileNotFoundError(f"command span file not found: {path}")
        for row in read_jsonl(path):
            command_id = str(row.get("command_id") or "")
            if command_id in command_ids:
                commands[(relpath, command_id)] = row
        missing = sorted(
            command_ids
            - {command_id for path_name, command_id in commands if path_name == relpath}
        )
        if missing:
            raise KeyError(f"{relpath}: command IDs not found: {missing[:10]}")
    return commands


def build_index(config: dict[str, Any]) -> dict[str, Any]:
    dataset_file = require_path(config, "data", "test_file")
    if bool(config.get("source", {}).get("public_spans", False)):
        return build_public_index(config, dataset_file)
    private_path = require_path(config, "source", "private_prepared")
    command_root = require_path(config, "source", "command_spans_root")
    mathlib_root = require_path(config, "source", "mathlib_root")
    output_dir = require_path(config, "output", "directory")
    index_file = output_dir / "source_index.jsonl"

    public_rows = list(read_jsonl(dataset_file))
    public: dict[str, dict[str, Any]] = {}
    for row in public_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in public:
            raise ValueError(f"missing or duplicate public sample_id: {sample_id!r}")
        public[sample_id] = row
    private = load_private_rows(private_path, set(public))

    requests: dict[str, set[str]] = {}
    joins: dict[str, tuple[str, str]] = {}
    for sample_id, row in private.items():
        metadata = dict(row.get("metadata") or {})
        relpath = str(row.get("relpath") or "")
        command_id = str(metadata.get("target_command_id") or "")
        if not relpath or not command_id:
            raise ValueError(f"{sample_id}: missing relpath or target_command_id")
        joins[sample_id] = (relpath, command_id)
        requests.setdefault(relpath, set()).add(command_id)
    commands = load_commands(command_root, requests)

    source_cache: dict[str, tuple[bytes, str]] = {}
    index_rows: list[dict[str, Any]] = []
    for sample_id in sorted(public):
        clean = public[sample_id]
        private_row = private[sample_id]
        metadata = dict(private_row.get("metadata") or {})
        for field in ("statement", "target"):
            if str(clean[field]) != str(private_row.get(field)):
                raise ValueError(f"{sample_id}: public/private {field} mismatch")
        private_family = str(metadata.get("proof_family") or "")
        if private_family and str(clean["proof_family"]) != private_family:
            raise ValueError(f"{sample_id}: public/private proof_family mismatch")
        relpath, command_id = joins[sample_id]
        command = commands[(relpath, command_id)]
        if relpath not in source_cache:
            source = (mathlib_root / relpath).read_bytes()
            source_cache[relpath] = (source, sha256_bytes(source))
        source, source_hash = source_cache[relpath]
        start = int(command["start_byte"])
        end = int(command["end_byte"])
        if not 0 <= start < end <= len(source):
            raise ValueError(f"{sample_id}: invalid command byte span [{start}, {end})")
        command_bytes = source[start:end]
        raw_text = str(command.get("raw_text") or "").encode("utf-8")
        if command_bytes != raw_text:
            raise ValueError(
                f"{sample_id}: command span differs from the pinned Mathlib checkout"
            )
        kind = str(command.get("kind") or private_row.get("kind") or "")
        if kind not in {"theorem", "lemma", "example"}:
            raise ValueError(f"{sample_id}: unsupported declaration kind {kind!r}")
        index_rows.append(
            {
                "sample_id": sample_id,
                "split": str(clean["split"]),
                "proof_family": str(clean["proof_family"]),
                "statement": str(clean["statement"]),
                "gold_target": str(clean["target"]),
                "relpath": relpath,
                "target_command_id": command_id,
                "kind": kind,
                "original_name": str(command.get("name") or ""),
                "start_byte": start,
                "end_byte": end,
                "start_line": int(command.get("start_line") or 0),
                "end_line": int(command.get("end_line") or 0),
                "source_sha256": source_hash,
                "original_command_sha256": sha256_bytes(command_bytes),
                "mathlib_version": str(metadata.get("mathlib4_version") or ""),
            }
        )
    write_jsonl(index_file, index_rows)
    manifest = {
        "dataset_file": str(dataset_file),
        "private_prepared": str(private_path),
        "command_spans_root": str(command_root),
        "mathlib_root": str(mathlib_root),
        "samples": len(index_rows),
        "source_files": len(source_cache),
        "replacement_policy": "replace exact original declaration byte span",
        "index_file": str(index_file),
    }
    write_json(output_dir / "source_index_manifest.json", manifest)
    return manifest


def build_public_index(config: dict[str, Any], dataset_file: Path) -> dict[str, Any]:
    """Build a source index from public command-span provenance.

    The supplied original source file must match its recorded hash. The target
    span is compared against the exported parser command text before indexing.
    This is a provenance check, not a Lean proof verification claim.
    """
    root = require_path(config, "source", "mathlib_root")
    command_root = require_path(config, "source", "command_spans_root")
    output = require_path(config, "output", "directory")
    rows = list(read_jsonl(dataset_file))
    requests: dict[str, set[str]] = {}
    for row in rows:
        requests.setdefault(str(row["source_file"]), set()).add(str(row["target_command_id"]))
    commands = load_commands(command_root, requests)
    cache: dict[str, bytes] = {}
    result = []
    seen = set()
    for row in rows:
        sid, relative = str(row["sample_id"]), str(row["source_file"])
        if sid in seen:
            raise ValueError(f"duplicate sample ID: {sid}")
        seen.add(sid)
        if relative not in cache:
            cache[relative] = (root / relative).read_bytes()
        source = cache[relative]
        expected = str(row["source_revision"]["source_file_sha256"])
        if sha256_bytes(source) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
        span = row["target_source_span"]
        start, end = int(span["start_byte"]), int(span["end_byte"])
        if not 0 <= start < end <= len(source):
            raise ValueError(f"invalid source span: {sid}")
        command = commands[(relative, str(row["target_command_id"]))]
        if source[start:end] != str(command["raw_text"]).encode():
            raise ValueError(f"parser command/source bytes mismatch: {sid}")
        result.append({"sample_id": sid, "split": row["split"], "proof_family": row["proof_family"],
                       "statement": row["statement"], "gold_target": row["target"], "relpath": relative,
                       "target_command_id": row["target_command_id"], "kind": row["target_kind"],
                       "original_name": row["original_name"], "start_byte": start, "end_byte": end,
                       "start_line": span["start_line"], "end_line": span["end_line"],
                       "source_sha256": expected, "original_command_sha256": sha256_bytes(source[start:end]),
                       "mathlib_version": "v4.14.0-rc1"})
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "source_index.jsonl", result)
    manifest = {"samples": len(result), "source_files": len(cache), "index_basis": "public_source_spans_and_parser_commands",
                "dataset_file": str(dataset_file), "mathlib_root": str(root), "Lean_verification_performed": False}
    write_json(output / "source_index_manifest.json", manifest)
    return manifest


def main() -> None:
    config = load_config(parse_args().config)
    print(json.dumps(build_index(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
