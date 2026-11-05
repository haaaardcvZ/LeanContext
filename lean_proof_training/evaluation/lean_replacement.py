"""Replace, never mutate, an original Lean declaration in an isolated task file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .proofs import Candidate, replacement_declaration


def replace_original_declaration(
    *,
    index_row: dict[str, Any],
    candidate: Candidate,
    mathlib_root: Path,
    task_root: Path,
    source_bytes: bytes | None = None,
    source_verified: bool = False,
) -> Path:
    source_path = mathlib_root / str(index_row["relpath"])
    source = source_path.read_bytes() if source_bytes is None else source_bytes
    if not source_verified and hashlib.sha256(source).hexdigest() != str(index_row["source_sha256"]):
        raise ValueError(f"source hash mismatch: {index_row['relpath']}")
    start = int(index_row["start_byte"])
    end = int(index_row["end_byte"])
    if not 0 <= start < end <= len(source):
        raise ValueError("original declaration byte span is outside source file")
    original = source[start:end]
    if hashlib.sha256(original).hexdigest() != str(index_row["original_command_sha256"]):
        raise ValueError("original declaration hash mismatch")
    replacement = replacement_declaration(index_row, candidate.proof).encode("utf-8")
    generated = source[:start] + replacement + source[end:]
    destination = (
        task_root
        / candidate.mode
        / candidate.sample_id
        / f"candidate_{candidate.candidate_id}.lean"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(generated)
    return destination
