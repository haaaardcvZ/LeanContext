"""Build a private sample-to-original-Lean-source index.

The publishable dataset deliberately omits source paths. Evaluation joins it to
the private prepared split by sample_id and then verifies exact command byte
offsets against the pinned Mathlib checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from lean_proof_training.common.io import iter_jsonl


def command_file(command_root: Path, relpath: str) -> Path:
    return command_root / Path(relpath).with_suffix(".jsonl")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_command(command_root: Path, relpath: str, command_id: str) -> dict[str, Any]:
    path = command_file(command_root, relpath)
    for row in iter_jsonl(path):
        if str(row.get("command_id") or "") == command_id:
            return row
    raise KeyError(f"{relpath}: command not found: {command_id}")


def build_source_index(
    *,
    dataset_file: Path,
    private_prepared_file: Path,
    command_root: Path,
    mathlib_root: Path,
    output_file: Path,
) -> dict[str, Any]:
    public = {str(row["sample_id"]): row for row in iter_jsonl(dataset_file)}
    private = {str(row["sample_id"]): row for row in iter_jsonl(private_prepared_file)}
    if set(public) != set(private):
        missing_private = sorted(set(public) - set(private))
        extra_private = sorted(set(private) - set(public))
        raise ValueError(
            f"public/private sample-ID mismatch: missing={missing_private[:5]} "
            f"extra={extra_private[:5]}"
        )

    file_cache: dict[str, tuple[bytes, str]] = {}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    family_counts: dict[str, int] = {}
    with output_file.open("w", encoding="utf-8") as output:
        for sample_id in sorted(public):
            clean = public[sample_id]
            source = private[sample_id]
            metadata = dict(source.get("metadata") or {})
            for field in ("split", "statement", "target"):
                if str(clean.get(field)) != str(source.get(field)):
                    raise ValueError(f"{sample_id}: public/private {field} mismatch")
            family = str(clean["proof_family"])
            if family != str(metadata.get("proof_family") or ""):
                raise ValueError(f"{sample_id}: proof_family mismatch")
            relpath = str(source.get("relpath") or "")
            command_id = str(metadata.get("target_command_id") or "")
            if not relpath or not command_id:
                raise ValueError(f"{sample_id}: missing relpath or target_command_id")
            command = load_command(command_root, relpath, command_id)
            source_path = mathlib_root / relpath
            if relpath not in file_cache:
                source_bytes = source_path.read_bytes()
                file_cache[relpath] = (source_bytes, sha256_bytes(source_bytes))
            source_bytes, source_sha256 = file_cache[relpath]
            start = int(command["start_byte"])
            end = int(command["end_byte"])
            effective_end = int(command.get("effective_end_byte") or end)
            # effective_end_byte belongs to the comment-stripped parser view
            # and may be smaller than end_byte when a declaration has docs.
            # Only original-source start_byte/end_byte are valid splice points.
            if not 0 <= start < end <= len(source_bytes):
                raise ValueError(f"{sample_id}: invalid source byte range")
            actual = source_bytes[start:end].decode("utf-8")
            if actual != str(command.get("raw_text") or ""):
                raise ValueError(
                    f"{sample_id}: command span does not match Mathlib checkout; "
                    "use the exact extraction Mathlib version"
                )
            source_range = metadata.get("source_range") or {}
            expected_start_line = int((source_range.get("start") or {}).get("line") or 0)
            expected_end_line = int((source_range.get("finish") or {}).get("line") or 0)
            if expected_start_line and expected_start_line != int(command["start_line"]):
                raise ValueError(f"{sample_id}: start-line mismatch")
            if expected_end_line and expected_end_line != int(command["end_line"]):
                raise ValueError(f"{sample_id}: end-line mismatch")
            row = {
                "sample_id": sample_id,
                "split": str(clean["split"]),
                "proof_family": family,
                "statement": str(clean["statement"]),
                "relpath": relpath,
                "original_name": str(source.get("full_name") or command.get("name") or ""),
                "target_command_id": command_id,
                "start_byte": start,
                "end_byte": end,
                "effective_end_byte": effective_end,
                "insert_after_byte": end,
                "start_line": int(command["start_line"]),
                "end_line": int(command["end_line"]),
                "source_sha256": source_sha256,
                "mathlib_version": str(metadata.get("mathlib4_version") or ""),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "samples": len(public),
        "source_files": len(file_cache),
        "proof_families": dict(sorted(family_counts.items())),
        "output_file": str(output_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--private-prepared-file", type=Path, required=True)
    parser.add_argument("--command-root", type=Path, required=True)
    parser.add_argument("--mathlib-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    result = build_source_index(
        dataset_file=args.dataset_file,
        private_prepared_file=args.private_prepared_file,
        command_root=args.command_root,
        mathlib_root=args.mathlib_root,
        output_file=args.output_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
