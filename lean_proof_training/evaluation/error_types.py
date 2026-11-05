"""Mutually exclusive error taxonomy for generated Lean proofs."""

from __future__ import annotations

import re
from typing import Any


PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "infrastructure_error",
        (
            "error during download",
            "failed to connect",
            "couldn't connect",
            "no such file or directory",
            "unknown package",
            "unknown module prefix",
            "missing manifest",
            "toolchain",
        ),
    ),
    (
        "resource_limit",
        (
            "maximum heartbeats exceeded",
            "maximum recursion depth",
            "deterministic timeout",
            "out of memory",
        ),
    ),
    (
        "syntax_or_parser_error",
        (
            "unexpected token",
            "unexpected identifier",
            "unexpected end of input",
            "expected token",
            "invalid 'end'",
            "parser error",
            "unexpected syntax",
        ),
    ),
    (
        "unknown_identifier_or_tactic",
        (
            "unknown identifier",
            "unknown constant",
            "unknown tactic",
            "invalid field notation",
            "invalid field",
        ),
    ),
    (
        "typeclass_synthesis_failure",
        (
            "failed to synthesize",
            "type class instance problem",
            "synthesizing instance",
        ),
    ),
    (
        "unsolved_goals",
        (
            "unsolved goals",
            "unsolved goal",
        ),
    ),
    (
        "tactic_failure",
        (
            "tactic 'assumption' failed",
            "tactic 'rfl' failed",
            "no goals to be solved",
            "rewrite tactic failed",
            "simp made no progress",
            "linarith failed",
            "omega could not prove",
            "aesop: failed",
            "tactic failed",
        ),
    ),
    (
        "type_mismatch",
        (
            "application type mismatch",
            "type mismatch",
            "but is expected to have type",
            "function expected at",
            "invalid argument name",
        ),
    ),
    (
        "declaration_conflict",
        (
            "already declared",
            "declaration has already been declared",
        ),
    ),
    (
        "termination_or_compiler_error",
        (
            "fail to show termination",
            "declaration uses 'sorry'",
            "kernel declaration has metavariables",
            "declaration has metavariables",
        ),
    ),
)

# Paper precedence: extraction > timeout > name/scope > instance synthesis >
# elaboration > unsolved goals > other. Infrastructure errors invalidate an
# evaluation instead of becoming a model failure. Preserve detailed labels.
_PRIORITY = {"infrastructure_error": -2, "syntax_or_parser_error": -1, "resource_limit": 0,
             "unknown_identifier_or_tactic": 1, "typeclass_synthesis_failure": 2,
             "type_mismatch": 3,
             "declaration_conflict": 3, "termination_or_compiler_error": 3,
             "unsolved_goals": 4, "tactic_failure": 5}
PATTERNS = tuple(sorted(PATTERNS, key=lambda pair: _PRIORITY[pair[0]]))
PAPER_CATEGORY = {"resource_limit": "timeout", "unknown_identifier_or_tactic": "name_or_scope",
                  "typeclass_synthesis_failure": "instance_synthesis",
                  "syntax_or_parser_error": "extraction", "type_mismatch": "elaboration",
                  "declaration_conflict": "elaboration", "termination_or_compiler_error": "elaboration",
                  "unsolved_goals": "unsolved_goals", "tactic_failure": "other",
                  "infrastructure_error": "infrastructure"}


def first_diagnostic(text: str, limit: int = 1000) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    for index, line in enumerate(lines):
        if "error:" in line.lower() or line.lower().startswith("error"):
            return "\n".join(lines[index : index + 4])[:limit]
    return lines[0][:limit]


def classify(
    *,
    passed: bool,
    stdout: str = "",
    stderr: str = "",
    precheck: str | None = None,
    timed_out: bool = False,
    task_construction_error: bool = False,
) -> dict[str, Any]:
    if passed:
        return {"error_type": "success", "paper_error_type": "success", "error_stage": None, "diagnostic": ""}
    if precheck is not None:
        return {
            "error_type": precheck,
            "paper_error_type": "extraction",
            "error_stage": "generation_precheck",
            "diagnostic": precheck,
        }
    if task_construction_error:
        return {
            "error_type": "task_construction_error",
            "paper_error_type": "infrastructure",
            "error_stage": "source_replacement",
            "diagnostic": first_diagnostic(stderr or stdout),
        }
    combined = (stdout + "\n" + stderr).lower()
    # Infrastructure invalidates the evaluation. Otherwise parsing/extraction
    # has precedence even when a later process timeout is also recorded.
    for category, patterns in PATTERNS:
        if category not in {"infrastructure_error", "syntax_or_parser_error"}:
            continue
        if any(pattern in combined for pattern in patterns):
            return {"error_type": category, "paper_error_type": PAPER_CATEGORY[category],
                    "error_stage": "lean_parsing" if category == "syntax_or_parser_error" else "lean_execution",
                    "diagnostic": first_diagnostic(stdout + "\n" + stderr)}
    if timed_out:
        return {
            "error_type": "lean_timeout",
            "paper_error_type": "timeout",
            "error_stage": "lean_execution",
            "diagnostic": first_diagnostic(stderr or stdout),
        }
    for category, patterns in PATTERNS:
        if any(pattern in combined for pattern in patterns):
            return {
                "error_type": category,
                "paper_error_type": PAPER_CATEGORY[category],
                "error_stage": "lean_elaboration",
                "diagnostic": first_diagnostic(stdout + "\n" + stderr),
            }
    return {
        "error_type": "other_lean_error",
        "paper_error_type": "other",
        "error_stage": "lean_elaboration",
        "diagnostic": first_diagnostic(stdout + "\n" + stderr),
    }
