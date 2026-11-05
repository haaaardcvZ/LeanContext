"""Render the final Context dataset exactly as an inference prompt."""

from __future__ import annotations

from typing import Any


VIEWS = {
    "statement_only",
    "full_context",
    "mathematical_context",
    "environment_context",
}


def validate_row(row: dict[str, Any]) -> None:
    required = {
        "sample_id",
        "split",
        "proof_family",
        "statement",
        "target",
        "mathematical_context",
        "environment_context",
        "full_context",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{row.get('sample_id')}: missing dataset fields {missing}")
    if not str(row["statement"]).strip() or not str(row["target"]).strip():
        raise ValueError(f"{row.get('sample_id')}: empty statement or target")
    for field in ("mathematical_context", "environment_context", "full_context"):
        items = row[field]
        if not isinstance(items, list):
            raise ValueError(f"{row.get('sample_id')}: {field} must be a list")
        for item in items:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                raise ValueError(f"{row.get('sample_id')}: invalid item in {field}")


def _context_tokens(tokenizer: Any, items: list[dict[str, Any]]) -> int:
    text = "\n\n".join(str(item["text"]).strip() for item in items)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def select_context_prefix(
    tokenizer: Any,
    items: list[dict[str, Any]],
    context_token_budget: int | None,
) -> list[dict[str, Any]]:
    """Keep a nested, source-ordered prefix of complete Lean commands."""
    ordered = sorted(items, key=lambda item: int(item.get("position", 0)))
    if context_token_budget is None:
        return ordered
    if context_token_budget < 0:
        raise ValueError("context_token_budget must be non-negative or null")
    selected: list[dict[str, Any]] = []
    for item in ordered:
        proposal = [*selected, item]
        if _context_tokens(tokenizer, proposal) > context_token_budget:
            break
        selected.append(item)
    return selected


def render_prompt(
    row: dict[str, Any],
    *,
    view: str,
    tokenizer: Any,
    context_token_budget: int | None,
    add_bos_token: bool,
) -> tuple[str, list[int], int]:
    validate_row(row)
    if view not in VIEWS:
        raise ValueError(f"unknown Context view {view!r}; expected {sorted(VIEWS)}")
    items = [] if view == "statement_only" else list(row[view])
    selected = select_context_prefix(tokenizer, items, context_token_budget)
    pieces = [str(item["text"]).strip() for item in selected]
    pieces.append(str(row["statement"]).strip())
    # Training owns exactly one newline between prompt and proof target.
    prompt = "\n\n".join(pieces).rstrip(" \t\r\n") + "\n"
    token_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if add_bos_token:
        if tokenizer.bos_token_id is None:
            raise ValueError("add_bos_token=true but tokenizer has no BOS token")
        if not token_ids or token_ids[0] != int(tokenizer.bos_token_id):
            token_ids.insert(0, int(tokenizer.bos_token_id))
    return prompt, token_ids, len(selected)
