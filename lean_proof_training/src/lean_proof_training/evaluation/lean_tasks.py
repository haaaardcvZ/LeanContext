"""Construct source-faithful Lean task files from proof candidates."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from lean_proof_training.evaluation.candidates import Candidate


DECL_RE = re.compile(
    r"(?m)(\b(?:private\s+)?(?:protected\s+)?(?:theorem|lemma)\s+)"
    r"([A-Za-z_][A-Za-z0-9_'.]*)"
)
EXAMPLE_RE = re.compile(r"(?m)\bexample\b")
LEADING_ATTRIBUTE_RE = re.compile(r"(?ms)\A\s*(?:@\[[^\]]*\]\s*)+")


def rename_statement(statement: str, new_name: str) -> str:
    # Attributes such as to_additive can create auxiliary declarations and are
    # irrelevant to checking the proof term itself.
    clean = LEADING_ATTRIBUTE_RE.sub("", statement.strip())
    match = DECL_RE.search(clean)
    if match is None:
        example = EXAMPLE_RE.search(clean)
        if example is not None:
            return clean[: example.start()] + f"theorem {new_name}" + clean[example.end() :]
        raise ValueError("cannot locate theorem/lemma/example in statement")
    return clean[: match.start(2)] + new_name + clean[match.end(2) :]


def candidate_declaration(statement: str, candidate: Candidate, unique_name: str) -> str:
    header = rename_statement(statement, unique_name).rstrip()
    proof = candidate.proof.strip()
    if header.endswith(":="):
        # Parenthesizing forces the model output to elaborate as one term and
        # prevents ordinary trailing text from becoming a second command.
        return f"{header}\n(\n{proof}\n)"
    return f"{header} :=\n(\n{proof}\n)"


def build_task_file(
    *,
    index_row: dict[str, Any],
    candidate: Candidate,
    mathlib_root: Path,
    task_root: Path,
    placement: str,
    source_bytes: bytes | None = None,
) -> Path:
    if placement not in {"strict_before", "insert_after"}:
        raise ValueError("placement must be strict_before or insert_after")
    source_path = mathlib_root / str(index_row["relpath"])
    source = source_path.read_bytes() if source_bytes is None else source_bytes
    if source_bytes is None:
        actual_hash = hashlib.sha256(source).hexdigest()
        if actual_hash != str(index_row["source_sha256"]):
            raise ValueError(
                f"source hash mismatch for {index_row['relpath']}; evaluation Mathlib is not pinned"
            )
    sample_id = str(index_row["sample_id"])
    unique_name = f"benchmark_eval_{sample_id[:12]}_{candidate.candidate_id}"
    declaration = candidate_declaration(str(index_row["statement"]), candidate, unique_name)
    insertion = (
        "\n\n-- BEGIN generated benchmark candidate\n"
        + declaration
        + "\n-- END generated benchmark candidate\n\n"
    ).encode("utf-8")
    offset = (
        int(index_row["start_byte"])
        if placement == "strict_before"
        else int(index_row["insert_after_byte"])
    )
    if not 0 <= offset <= len(source):
        raise ValueError("candidate insertion offset is outside source file")
    generated = source[:offset] + insertion + source[offset:]
    destination = task_root / sample_id / f"candidate_{candidate.candidate_id}.lean"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(generated)
    return destination
