"""Conservative generated-proof cleanup and declaration reconstruction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


FENCE_RE = re.compile(r"```(?:lean|lean4)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
PLACEHOLDER_RE = re.compile(r"(?<![A-Za-z0-9_'.])(?:sorry|admit)(?![A-Za-z0-9_'.])")
COMMAND_INJECTION_RE = re.compile(
    r"(?m)^\s*(?:import|namespace|section|end|theorem|lemma|axiom|opaque|def|"
    r"instance|class|structure|inductive|#eval|#check|#reduce|set_option)\b"
)


def decode_proof_tokens(tokenizer: Any, ids: Any) -> str:
    """Remove transport BOS/PAD/EOS only; retain reasoning/control markers.

    ``skip_special_tokens=True`` could hide a generated <think> marker before
    the non-CoT precheck. Keeping all other special text makes this auditable.
    """
    values = [int(value) for value in ids]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and eos in values:
        values = values[:values.index(eos)]
    ignored = {value for value in (getattr(tokenizer, "bos_token_id", None),
                                  getattr(tokenizer, "pad_token_id", None)) if value is not None}
    values = [value for value in values if value not in ignored]
    return tokenizer.decode(values, skip_special_tokens=False).strip()
DECLARATION_NAME_RE = re.compile(
    r"(?m)(\b(?:private\s+)?(?:protected\s+)?(?:theorem|lemma)\s+)"
    r"([A-Za-z_][A-Za-z0-9_'.]*)"
)


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    mode: str
    candidate_id: int
    proof: str
    proof_hash: str
    raw: dict[str, Any]
    protocol_error: str | None = None


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
        return "empty_generation"
    if PLACEHOLDER_RE.search(proof):
        return "placeholder_sorry_or_admit"
    if COMMAND_INJECTION_RE.search(proof):
        return "top_level_command_injection"
    return None


def load_candidates(rows: list[dict[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        mode = str(row.get("mode") or "")
        candidate_id = int(row.get("candidate_id", -1))
        if not sample_id or mode not in {"sampled", "greedy_temperature_0"}:
            raise ValueError("prediction has missing sample_id or invalid mode")
        key = (sample_id, mode, candidate_id)
        if candidate_id < 0 or key in seen:
            raise ValueError(f"invalid or duplicate prediction key: {key}")
        seen.add(key)
        raw_text = str(row.get("proof") or "")
        forbidden = row.get("forbid_reasoning_markers", [])
        if not isinstance(forbidden, list) or any(not isinstance(marker, str) or not marker for marker in forbidden):
            raise ValueError("forbid_reasoning_markers must be a list of nonempty strings")
        # Inspect the original output BEFORE a fenced block or stop marker can
        # hide reasoning. A non-CoT protocol violation must not become a pass.
        protocol_error = "forbidden_reasoning_marker" if any(marker in raw_text for marker in forbidden) else None
        proof = clean_proof(raw_text)
        candidates.append(
            Candidate(
                sample_id=sample_id,
                mode=mode,
                candidate_id=candidate_id,
                proof=proof,
                proof_hash=hashlib.sha256(proof.encode("utf-8")).hexdigest(),
                raw=row,
                protocol_error=protocol_error,
            )
        )
    return candidates


def restore_original_name(statement: str, *, kind: str, original_name: str) -> str:
    header = statement.strip()
    if kind == "example":
        return header
    if not original_name:
        raise ValueError("named theorem/lemma has no original_name")
    match = DECLARATION_NAME_RE.search(header)
    if match is None:
        raise ValueError("cannot locate anonymized theorem/lemma name in statement")
    return header[: match.start(2)] + original_name + header[match.end(2) :]


def replacement_declaration(index_row: dict[str, Any], proof: str) -> str:
    header = restore_original_name(
        str(index_row["statement"]),
        kind=str(index_row["kind"]),
        original_name=str(index_row.get("original_name") or ""),
    ).rstrip()
    # The dataset also contains equation-compiler declarations (`| ... =>`)
    # and structure-style proofs whose header ends in `where`. In every case
    # the extracted target is the exact suffix following this single newline.
    return header + "\n" + proof.strip()
