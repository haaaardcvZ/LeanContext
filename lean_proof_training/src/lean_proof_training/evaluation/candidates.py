"""Prediction loading and conservative Lean proof-suffix normalization."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lean_proof_training.common.io import iter_jsonl


FENCE_RE = re.compile(r"```(?:lean|lean4)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
FORBIDDEN_PLACEHOLDER_RE = re.compile(
    r"(?<![A-Za-z0-9_'.])(?:sorry|admit)(?![A-Za-z0-9_'.])"
)
COMMAND_INJECTION_RE = re.compile(
    r"(?m)^\s*(?:import|namespace|section|end|theorem|lemma|axiom|opaque|def|"
    r"instance|class|structure|inductive|#eval|#check|#reduce|set_option)\b"
)


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    candidate_id: int
    proof: str
    proof_hash: str
    raw: dict[str, Any]


def clean_proof(value: Any) -> str:
    text = str(value or "").strip()
    if "\n" not in text and "\\n" in text:
        text = text.replace("\\n", "\n").replace("\\t", "\t")
    fenced = FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    for marker in ("<END_OF_PROOF>", "END_OF_PROOF", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    for prefix in ("PROOF_SUFFIX:", "Proof suffix:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    if text.startswith(":="):
        text = text[2:].strip()
    return text


def precheck_proof(proof: str) -> str | None:
    if not proof:
        return "empty_proof"
    if FORBIDDEN_PLACEHOLDER_RE.search(proof):
        return "sorry_or_admit"
    if COMMAND_INJECTION_RE.search(proof):
        return "top_level_command_injection"
    return None


def load_candidates(
    path: Path,
    *,
    sample_field: str = "sample_id",
    candidate_field: str = "candidate_id",
    proof_field: str = "proof",
) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    implicit: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    for row in iter_jsonl(path):
        sample_id = str(row.get(sample_field) or "")
        if not sample_id or proof_field not in row:
            raise ValueError("every prediction must contain sample_id and proof")
        if candidate_field in row:
            candidate_id = int(row[candidate_field])
        else:
            candidate_id = implicit.get(sample_id, 0)
            implicit[sample_id] = candidate_id + 1
        key = (sample_id, candidate_id)
        if key in seen:
            raise ValueError(f"duplicate prediction key: {key}")
        seen.add(key)
        proof = clean_proof(row[proof_field])
        grouped.setdefault(sample_id, []).append(
            Candidate(
                sample_id=sample_id,
                candidate_id=candidate_id,
                proof=proof,
                proof_hash=hashlib.sha256(proof.encode("utf-8")).hexdigest(),
                raw=row,
            )
        )
    for candidates in grouped.values():
        candidates.sort(key=lambda item: item.candidate_id)
    return grouped


def validate_prediction_coverage(
    expected_ids: set[str],
    grouped: dict[str, list[Candidate]],
    *,
    expected_candidates: int,
) -> None:
    if set(grouped) != expected_ids:
        missing = sorted(expected_ids - set(grouped))
        extra = sorted(set(grouped) - expected_ids)
        raise ValueError(f"prediction coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
    expected_ranks = list(range(expected_candidates))
    for sample_id in sorted(expected_ids):
        actual = [item.candidate_id for item in grouped[sample_id]]
        if actual != expected_ranks:
            raise ValueError(
                f"{sample_id}: expected candidate IDs 0..{expected_candidates - 1}, found {actual}"
            )
