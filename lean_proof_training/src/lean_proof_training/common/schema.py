"""Unified raw-dataset schema and Context-budget rendering."""

from __future__ import annotations

from typing import Any


VIEWS = {"statement_only", "matched", "random_control", "hard_control"}


def validate_raw_row(row: dict[str, Any]) -> None:
    required = {
        "sample_id",
        "split",
        "training_stage",
        "proof_family",
        "statement",
        "target",
        "contexts",
    }
    if set(row) != required:
        raise ValueError(f"invalid unified row fields: {sorted(row)}")
    contexts = row["contexts"]
    if not isinstance(contexts, dict) or set(contexts) != {
        "matched",
        "random_control",
        "hard_control",
    }:
        raise ValueError("invalid contexts mapping")
    if not str(row["statement"]).strip() or not str(row["target"]).strip():
        raise ValueError("statement and target must be non-empty")
    if str(row["target"]).startswith(("\n", "\r")):
        raise ValueError("target must not own the prompt separator")
    for key, items in contexts.items():
        if items is None:
            continue
        if not isinstance(items, list):
            raise ValueError(f"{key} must be a list or null")
        for item in items:
            if set(item) != {"kind", "text"} or not str(item["text"]).strip():
                raise ValueError(f"invalid {key} Context item")


def context_items(row: dict[str, Any], view: str) -> list[dict[str, str]] | None:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(VIEWS)}")
    if view == "statement_only":
        return []
    return row["contexts"][view]


def _context_text(items: list[dict[str, str]]) -> str:
    return "\n\n".join(str(item["text"]).strip() for item in items)


def select_budgeted_items(
    tokenizer: Any,
    items: list[dict[str, str]],
    budget: int | None,
    *,
    policy: str = "prefix",
) -> list[dict[str, str]]:
    if budget is None:
        return list(items)
    if budget < 0:
        raise ValueError("Context budget must be non-negative or null")
    if budget == 0 or not items:
        return []
    if policy not in {"prefix", "suffix"}:
        raise ValueError("budget policy must be prefix or suffix")
    ordered = list(items) if policy == "prefix" else list(reversed(items))
    chosen: list[dict[str, str]] = []
    for item in ordered:
        proposal = [*chosen, item]
        source_order = proposal if policy == "prefix" else list(reversed(proposal))
        token_count = len(
            tokenizer(_context_text(source_order), add_special_tokens=False)["input_ids"]
        )
        if token_count <= budget:
            chosen.append(item)
        else:
            # Keep budgeted views strictly nested; never skip an item and then
            # admit a later declaration.
            break
    return chosen if policy == "prefix" else list(reversed(chosen))


def render_prompt(
    row: dict[str, Any],
    view: str,
    *,
    tokenizer: Any | None = None,
    context_budget: int | None = None,
    budget_policy: str = "prefix",
) -> tuple[str, list[dict[str, str]]] | None:
    validate_raw_row(row)
    items = context_items(row, view)
    if items is None:
        return None
    selected = list(items)
    if context_budget is not None:
        if tokenizer is None:
            raise ValueError("budgeted rendering requires a tokenizer")
        selected = select_budgeted_items(
            tokenizer, selected, context_budget, policy=budget_policy
        )
    parts = [str(item["text"]).strip() for item in selected]
    parts.append(str(row["statement"]).strip())
    prompt = "\n\n".join(parts).rstrip(" \t\r\n") + "\n"
    if prompt.endswith("\n\n"):
        raise AssertionError("rendered prompt owns more than one separator newline")
    return prompt, selected
