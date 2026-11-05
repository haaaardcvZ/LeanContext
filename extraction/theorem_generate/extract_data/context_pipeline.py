from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .llm_client import ChatClient, ChatConfig


TARGET_KINDS = {"theorem", "lemma", "example"}

SCOPE_KINDS = {"namespace", "section"}

# Commands that affect the active Lean environment at the target position.
# These are candidates, not automatically kept in the final context.
STRUCTURAL_KINDS = {
    "open",
    "open_scoped",
    "universe",
    "variable",
    "include",
    "omit",
    "set_option",
    "attribute",
    "local_attribute",
    "scoped_attribute",
    "notation",
    "local_notation",
    "scoped_notation",
    "scoped",
    "Mathlib.Tactic.scopedNS",
    "infix",
    "infixl",
    "infixr",
    "prefix",
    "postfix",
    "local_instance",
    "local",
    "syntax",
    "macro",
    "macro_rules",
    "Mathlib.Notation3.notation3",
    "Lean.Parser.Command.syntax",
    "Lean.Parser.Command.macro",
    "Lean.Parser.Command.macro_rules",
    "Lean.Parser.Command.notation",
    "Lean.Parser.Command.mixfix",
    "Lean.Parser.Command.syntaxCat",
    "Lean.Parser.Command.syntaxAbbrev",
    "Lean.Parser.Command.elab",
    "Lean.Parser.Command.elab_rules",
}

# Named declarations that may be referenced by a later target.  A lemma/theorem
# can be both a target in its own row and a dependency candidate for later rows.
DEPENDENCY_KINDS = {
    "def",
    "abbrev",
    "class",
    "structure",
    "inductive",
    "opaque",
    "instance",
    "lemma",
    "theorem",
    "axiom",
    "constant",
    "alias",
    "export",
}

UNNAMED_DEPENDENCY_KINDS = {"instance"}

MAX_UNNAMED_DEPENDENCY_CANDIDATES = 12


@dataclass
class Scope:
    opener: dict[str, Any] | None
    commands: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TargetCase:
    source_path: Path
    target: dict[str, Any]
    active_scopes: list[Scope]
    prior_declarations: list[dict[str, Any]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{line_no}: invalid JSONL: {e}") from e
    return records


def iter_jsonl_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*.jsonl") if p.is_file()]
    files.sort(key=lambda p: p.as_posix())
    return files


def iter_target_cases(root: Path) -> Iterable[TargetCase]:
    for path in iter_jsonl_files(root):
        records = read_jsonl(path)
        scopes: list[Scope] = [Scope(opener=None)]
        prior_declarations: list[dict[str, Any]] = []

        for rec in records:
            kind = str(rec.get("kind") or "")

            if kind in TARGET_KINDS:
                active_copy = [Scope(s.opener, list(s.commands)) for s in scopes]
                yield TargetCase(
                    source_path=path,
                    target=rec,
                    active_scopes=active_copy,
                    prior_declarations=list(prior_declarations),
                )

            if kind in SCOPE_KINDS:
                scopes.append(Scope(opener=rec))
            elif kind == "end":
                if len(scopes) > 1:
                    scopes.pop()
            elif kind in STRUCTURAL_KINDS:
                scopes[-1].commands.append(rec)

            if kind in DEPENDENCY_KINDS and (
                rec.get("name") or kind in UNNAMED_DEPENDENCY_KINDS
            ):
                prior_declarations.append(rec)


def command_key(rec: dict[str, Any]) -> str:
    return str(rec.get("command_id") or "")


def command_text(rec: dict[str, Any]) -> str:
    text = rec.get("effective_raw_text")
    if not isinstance(text, str):
        return ""
    text = text.strip(" \t\r\n")
    if not text:
        return ""
    return text + "\n"


def name_variants(name: str) -> set[str]:
    variants = {name}
    if "." in name:
        variants.add(name.rsplit(".", 1)[-1])
    return {v for v in variants if v and len(v) >= 2}


def appears_in_text(name: str, text: str) -> bool:
    for variant in name_variants(name):
        pattern = r"(?<![A-Za-z0-9_'.])" + re.escape(variant) + r"(?![A-Za-z0-9_'.])"
        if re.search(pattern, text):
            return True
    return False


def build_structural_candidates(case: TargetCase) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for scope in case.active_scopes:
        if scope.opener is not None:
            candidates.append(scope.opener)
        candidates.extend(scope.commands)
    return dedupe_commands(candidates)


def build_dependency_candidates(
    case: TargetCase,
    *,
    recursive_depth: int,
    max_dependency_candidates: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    unnamed_recent: list[dict[str, Any]] = []
    search_texts = [command_text(case.target)]

    for rec in reversed(case.prior_declarations):
        if len(unnamed_recent) >= MAX_UNNAMED_DEPENDENCY_CANDIDATES:
            break
        if rec.get("kind") in UNNAMED_DEPENDENCY_KINDS and not rec.get("name"):
            unnamed_recent.append(rec)

    for _depth in range(recursive_depth + 1):
        added: list[dict[str, Any]] = []
        combined_text = "\n".join(search_texts)
        for rec in reversed(case.prior_declarations):
            cid = command_key(rec)
            name = rec.get("name")
            if not cid or cid in selected_ids or not isinstance(name, str):
                continue
            if appears_in_text(name, combined_text):
                selected_ids.add(cid)
                selected.append(rec)
                added.append(rec)
        if not added:
            break
        search_texts = [command_text(rec) for rec in added]

    selected.reverse()
    selected = dedupe_commands([*reversed(unnamed_recent), *selected])
    selected.sort(key=lambda rec: int(rec.get("start_byte", 0)))
    if len(selected) > max_dependency_candidates:
        selected = selected[-max_dependency_candidates:]
    return selected


def dedupe_commands(commands: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rec in commands:
        cid = command_key(rec)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(rec)
    return out


def build_candidates(
    case: TargetCase,
    *,
    recursive_depth: int,
    max_dependency_candidates: int,
) -> list[dict[str, Any]]:
    structural = build_structural_candidates(case)
    dependencies = build_dependency_candidates(
        case,
        recursive_depth=recursive_depth,
        max_dependency_candidates=max_dependency_candidates,
    )
    combined = structural + dependencies
    combined = dedupe_commands(combined)
    combined.sort(key=lambda rec: int(rec.get("start_byte", 0)))
    return combined


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n/- TRUNCATED FOR LLM INPUT -/"


def is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in {"_", "'", "."}


def token_at(text: str, pos: int, token: str) -> bool:
    end = pos + len(token)
    if end > len(text) or text[pos:end] != token:
        return False
    before_ok = pos == 0 or not is_ident_char(text[pos - 1])
    after_ok = end == len(text) or not is_ident_char(text[end])
    return before_ok and after_ok


def find_token_outside_strings_comments(text: str, tokens: set[str], *, search_start: int = 0) -> tuple[str, int, int] | None:
    in_string = False
    escaped = False
    block_comment_depth = 0
    i = search_start
    while i < len(text):
        ch = text[i]
        next_ch = text[i + 1] if i + 1 < len(text) else ""

        if block_comment_depth > 0:
            if ch == "/" and next_ch == "-":
                block_comment_depth += 1
                i += 2
                continue
            if ch == "-" and next_ch == "/":
                block_comment_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "-" and next_ch == "-":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and next_ch == "-":
            block_comment_depth = 1
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue

        for token in tokens:
            if token_at(text, i, token):
                return token, i, i + len(token)
        i += 1
    return None


def find_declaration_keyword_end(text: str) -> int:
    found = find_token_outside_strings_comments(text, {"theorem", "lemma", "example"})
    return found[2] if found else 0


def find_where_split(text: str, *, search_start: int = 0) -> int | None:
    found = find_token_outside_strings_comments(text, {"where"}, search_start=search_start)
    return found[1] if found else None


def find_assignment_split(text: str, *, search_start: int = 0) -> int | None:
    in_string = False
    escaped = False
    block_comment_depth = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    angle_depth = 0
    i = search_start
    while i + 1 < len(text):
        ch = text[i]

        if block_comment_depth > 0:
            if ch == "/" and text[i + 1] == "-":
                block_comment_depth += 1
                i += 2
                continue
            if ch == "-" and text[i + 1] == "/":
                block_comment_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "-" and text[i + 1] == "-":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "/" and text[i + 1] == "-":
            block_comment_depth = 1
            i += 2
            continue

        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}" and brace_depth > 0:
            brace_depth -= 1
        elif ch == "⟨":
            angle_depth += 1
        elif ch == "⟩" and angle_depth > 0:
            angle_depth -= 1

        at_command_depth = (
            paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
            and angle_depth == 0
        )
        if ch == ":" and text[i + 1] == "=" and at_command_depth:
            return i
        i += 1
    return None


def find_equation_alternative_split(text: str) -> int | None:
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("|"):
            return offset
        offset += len(line)
    return None


def split_target_text(text: str) -> dict[str, Any]:
    declaration_keyword_end = find_declaration_keyword_end(text)
    assignment_idx = find_assignment_split(text, search_start=declaration_keyword_end)
    where_idx = find_where_split(text, search_start=declaration_keyword_end)

    if where_idx is not None and (assignment_idx is None or where_idx < assignment_idx):
        body_start = where_idx + len("where")
        header = text[:body_start].rstrip()
        body = text[body_start:]
        if body.startswith(" "):
            body = body[1:]
        return {
            "target_header_text": header,
            "target_body_text": body,
            "target_split_found": True,
            "target_split_method": "where",
        }

    if assignment_idx is not None:
        body_start = assignment_idx + 2
        header = text[:body_start].rstrip()
        body = text[body_start:]
        if body.startswith(" "):
            body = body[1:]
        return {
            "target_header_text": header,
            "target_body_text": body,
            "target_split_found": True,
            "target_split_method": "assignment",
        }

    equation_idx = find_equation_alternative_split(text)
    if equation_idx is not None:
        return {
            "target_header_text": text[:equation_idx].rstrip("\n"),
            "target_body_text": text[equation_idx:],
            "target_split_found": True,
            "target_split_method": "equation_alternatives",
        }

    return {
        "target_header_text": text,
        "target_body_text": "",
        "target_split_found": False,
        "target_split_method": "none",
    }


def anonymize_target_header(header_text: str, target_kind: Any, target_name: Any) -> str:
    if target_kind == "example" or not isinstance(target_name, str) or not target_name:
        return header_text

    keyword_match = re.search(r"\b(theorem|lemma)\b", header_text)
    if not keyword_match:
        return header_text

    search_start = keyword_match.end()
    pattern = re.compile(r"(?<![A-Za-z0-9_'.])" + re.escape(target_name) + r"(?![A-Za-z0-9_'.])")
    match = pattern.search(header_text, search_start)
    if match is None and "." in target_name:
        short_name = target_name.rsplit(".", 1)[-1]
        pattern = re.compile(r"(?<![A-Za-z0-9_'.])" + re.escape(short_name) + r"(?![A-Za-z0-9_'.])")
        match = pattern.search(header_text, search_start)
    if match is None:
        return header_text
    return header_text[: match.start()] + "anonymous_target" + header_text[match.end() :]


def detect_mathlib4_version(source_root: Any, source_jsonl: Path) -> str | None:
    if isinstance(source_root, str):
        toolchain = Path(source_root) / "lean-toolchain"
        if toolchain.exists():
            text = toolchain.read_text(encoding="utf-8").strip()
            if ":" in text:
                return text.rsplit(":", 1)[-1]
            return text or None

    for parent in [source_jsonl.parent, *source_jsonl.parents]:
        if parent.name.startswith("mathlib4-"):
            return parent.name.removeprefix("mathlib4-")
    repo_root = Path(__file__).resolve().parents[1]
    lean_parser_root = repo_root / "lean_parser"
    if lean_parser_root.exists():
        versions = sorted(
            p.name.removeprefix("mathlib4-")
            for p in lean_parser_root.iterdir()
            if p.is_dir() and p.name.startswith("mathlib4-")
        )
        if versions:
            return versions[0]
    return None


def make_llm_messages(
    case: TargetCase,
    candidates: list[dict[str, Any]],
    *,
    max_chars_per_command: int,
) -> list[dict[str, str]]:
    target = case.target
    candidate_payload = []
    for rec in candidates:
        candidate_payload.append(
            {
                "command_id": command_key(rec),
                "kind": rec.get("kind"),
                "name": rec.get("name"),
                "start_line": rec.get("start_line"),
                "end_line": rec.get("end_line"),
                "text": truncate_text(command_text(rec), max_chars_per_command),
            }
        )

    user_payload = {
        "task": "Select the minimal Lean context commands needed for the target theorem/lemma/example.",
        "source_file": target.get("source_file"),
        "source_module": target.get("source_module"),
        "target": {
            "command_id": command_key(target),
            "kind": target.get("kind"),
            "name": target.get("name"),
            "start_line": target.get("start_line"),
            "end_line": target.get("end_line"),
            "text": truncate_text(command_text(target), max_chars_per_command * 2),
        },
        "candidate_commands": candidate_payload,
    }

    system = (
        "You are selecting minimal Lean 4 context from parser-extracted commands. "
        "Return only a JSON object. Choose command_ids only from candidate_commands. "
        "Keep a command only if it is needed or strongly useful for parsing, name resolution, "
        "local variables/typeclasses, notation, attributes, or direct/indirect local dependencies "
        "of the target theorem/lemma/example. Do not keep commands merely because they appear earlier. "
        "Do not include the target command_id. Output schema: "
        '{"keep_command_ids":["cmd_000001"],"drop_command_ids":["cmd_000002"],"reason":"short"}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def normalize_keep_ids(llm_obj: dict[str, Any], candidates: list[dict[str, Any]], target_id: str) -> list[str]:
    allowed = {command_key(rec) for rec in candidates}
    raw = llm_obj.get("keep_command_ids", [])
    if not isinstance(raw, list):
        raw = []
    keep: list[str] = []
    seen: set[str] = set()
    for item in raw:
        cid = str(item)
        if cid in allowed and cid != target_id and cid not in seen:
            keep.append(cid)
            seen.add(cid)
    return keep


def close_scope_text(open_text: str, name: Any) -> str:
    if isinstance(name, str) and name.strip():
        return f"end {name.strip()}\n"
    first = open_text.strip().split(maxsplit=1)
    if len(first) == 2 and first[0] in SCOPE_KINDS:
        return f"end {first[1].strip()}\n"
    return "end\n"


def merge_commands(
    candidates: list[dict[str, Any]],
    target: dict[str, Any],
    keep_ids: list[str],
    *,
    include_import_mathlib: bool,
    include_closing_ends: bool,
) -> tuple[str, str, str]:
    by_id = {command_key(rec): rec for rec in candidates}
    kept = [by_id[cid] for cid in keep_ids if cid in by_id]
    kept.sort(key=lambda rec: int(rec.get("start_byte", 0)))

    context_parts: list[str] = []
    open_scopes: list[dict[str, Any]] = []
    if include_import_mathlib:
        context_parts.append("import Mathlib\n")

    for rec in kept:
        context_parts.append(command_text(rec))
        if rec.get("kind") in SCOPE_KINDS:
            open_scopes.append(rec)

    closing_parts = []
    if include_closing_ends:
        closing_parts = [
            close_scope_text(command_text(rec), rec.get("name")) for rec in reversed(open_scopes)
        ]
    context_text = join_lean_parts(context_parts)
    closing_text = join_lean_parts(closing_parts)
    merged_text = join_lean_parts([context_text, command_text(target), closing_text])
    return context_text, closing_text, merged_text


def join_lean_parts(parts: Iterable[str]) -> str:
    out = ""
    for part in parts:
        if not part:
            continue
        if out and not out.endswith("\n\n"):
            out = out.rstrip("\n") + "\n\n"
        out += part if part.endswith("\n") else part + "\n"
    return out


def command_meta(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_id": command_key(rec),
        "kind": rec.get("kind"),
        "name": rec.get("name"),
        "start_line": rec.get("start_line"),
        "end_line": rec.get("end_line"),
    }


def build_output_record(
    case: TargetCase,
    candidates: list[dict[str, Any]],
    keep_ids: list[str],
    *,
    provider: str,
    model: str,
    llm_reason: str | None,
    include_import_mathlib: bool,
    include_closing_ends: bool,
    proof_end_marker: str,
) -> dict[str, Any]:
    target = case.target
    context_text, closing_text, merged_text = merge_commands(
        candidates,
        target,
        keep_ids,
        include_import_mathlib=include_import_mathlib,
        include_closing_ends=include_closing_ends,
    )
    target_split = split_target_text(command_text(target))
    anonymized_header = anonymize_target_header(
        target_split["target_header_text"],
        target.get("kind"),
        target.get("name"),
    )
    sft_input_text = join_lean_parts([context_text, anonymized_header])
    sft_input_text_without_context = join_lean_parts([anonymized_header])
    sft_target_text = target_split["target_body_text"] + proof_end_marker
    kept_set = set(keep_ids)
    kept_commands = [rec for rec in candidates if command_key(rec) in kept_set]
    return {
        "theorem_name": target.get("name"),
        "lean_file_name": target.get("source_file"),
        "mathlib4_version": detect_mathlib4_version(target.get("source_root"), case.source_path),
        "target_command_id": command_key(target),
        "target_kind": target.get("kind"),
        "target_start_line": target.get("start_line"),
        "target_end_line": target.get("end_line"),
        "source_module": target.get("source_module"),
        "source_jsonl": case.source_path.as_posix(),
        "target_text": command_text(target),
        "target_header_text": target_split["target_header_text"],
        "target_header_text_anonymized": anonymized_header,
        "target_body_text": target_split["target_body_text"],
        "target_split_found": target_split["target_split_found"],
        "target_split_method": target_split["target_split_method"],
        "context_text": context_text,
        "sft_input_text": sft_input_text,
        "sft_target_text": sft_target_text,
        "sft_input_text_without_context": sft_input_text_without_context,
        "proof_end_marker": proof_end_marker,
        "candidate_count": len(candidates),
        "candidate_command_ids": [command_key(rec) for rec in candidates],
        "kept_command_ids": keep_ids,
        "kept_commands": [command_meta(rec) for rec in kept_commands],
        "closing_text": closing_text,
        "merged_text": merged_text,
        "selection_provider": provider,
        "selection_model": model,
        "selection_method": "parser_candidates_plus_llm_minimized",
        "selection_reason": llm_reason,
        "verified_by_lean": False,
    }


def load_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_file = rec.get("source_file") or rec.get("lean_file_name")
            target_command_id = rec.get("target_command_id")
            if source_file and target_command_id:
                done.add(f"{source_file}::{target_command_id}")
    return done


def load_source_file_filters(args: argparse.Namespace) -> set[str]:
    filters: set[str] = set(args.source_file or [])
    if args.source_file_list:
        path = Path(args.source_file_list)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    filters.add(line)
    return filters


def filter_cases_by_source_file(cases: list[TargetCase], source_files: set[str]) -> list[TargetCase]:
    if not source_files:
        return cases
    return [case for case in cases if str(case.target.get("source_file")) in source_files]


def case_key(case: TargetCase) -> str:
    t = case.target
    return f"{t.get('source_file')}::{command_key(t)}"


def select_keep_ids(
    *,
    client: ChatClient | None,
    case: TargetCase,
    candidates: list[dict[str, Any]],
    max_chars_per_command: int,
    mode: str,
) -> tuple[list[str], str | None]:
    if mode == "candidates":
        return [], None
    if mode == "heuristic":
        keep_ids = [command_key(rec) for rec in candidates]
        return keep_ids, "heuristic keeps all generated candidates"
    if client is None:
        raise ValueError("llm mode requires a ChatClient")

    messages = make_llm_messages(
        case,
        candidates,
        max_chars_per_command=max_chars_per_command,
    )
    obj = client.chat_json(messages)
    keep_ids = normalize_keep_ids(obj, candidates, command_key(case.target))
    reason = obj.get("reason")
    return keep_ids, reason if isinstance(reason, str) else None


def run_pipeline(
    *,
    provider: str,
    model: str,
    client: ChatClient | None,
    args: argparse.Namespace,
) -> None:
    input_root = Path(args.input_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = list(iter_target_cases(input_root))
    source_file_filters = load_source_file_filters(args)
    cases = filter_cases_by_source_file(cases, source_file_filters)

    if args.preview_sample:
        rng = random.Random(args.seed)
        if args.preview_sample < len(cases):
            cases = rng.sample(cases, args.preview_sample)
        cases.sort(key=lambda c: (str(c.target.get("source_file")), int(c.target.get("start_byte", 0))))

    if args.limit:
        cases = cases[: args.limit]

    done = load_done_keys(output_path) if args.resume and args.mode != "candidates" else set()
    write_mode = "a" if args.resume and output_path.exists() and args.mode != "candidates" else "w"

    processed = 0
    skipped = 0
    with output_path.open(write_mode, encoding="utf-8") as out:
        for idx, case in enumerate(cases, start=1):
            key = case_key(case)
            if key in done:
                skipped += 1
                continue

            candidates = build_candidates(
                case,
                recursive_depth=args.recursive_depth,
                max_dependency_candidates=args.max_dependency_candidates,
            )

            if args.mode == "candidates":
                rec = {
                    "source_file": case.target.get("source_file"),
                    "source_module": case.target.get("source_module"),
                    "source_jsonl": case.source_path.as_posix(),
                    "target_command_id": command_key(case.target),
                    "target_kind": case.target.get("kind"),
                    "target_name": case.target.get("name"),
                    "target_start_line": case.target.get("start_line"),
                    "target_end_line": case.target.get("end_line"),
                    "target_text": command_text(case.target),
                    "candidate_count": len(candidates),
                    "candidates": [command_meta(rec) | {"text": command_text(rec)} for rec in candidates],
                }
            else:
                keep_ids, reason = select_keep_ids(
                    client=client,
                    case=case,
                    candidates=candidates,
                    max_chars_per_command=args.max_chars_per_command,
                    mode=args.mode,
                )
                rec = build_output_record(
                    case,
                    candidates,
                    keep_ids,
                    provider=provider,
                    model=model,
                    llm_reason=reason,
                    include_import_mathlib=args.import_mathlib,
                    include_closing_ends=args.include_closing_ends,
                    proof_end_marker=args.proof_end_marker,
                )

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += 1
            if args.progress_every and processed % args.progress_every == 0:
                print(f"[progress] written={processed} idx={idx}/{len(cases)} skipped={skipped}")

    print(
        json.dumps(
            {
                "input_root": input_root.as_posix(),
                "output": output_path.as_posix(),
                "mode": args.mode,
                "provider": provider,
                "model": model,
                "cases_seen": len(cases),
                "source_file_filters": sorted(source_file_filters),
                "written": processed,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    default_input = Path(__file__).resolve().parents[1] / "lean_parser" / "command_spans_jsonl" / "Mathlib"
    parser.add_argument("--input-root", default=default_input.as_posix())
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["candidates", "heuristic", "llm"],
        default="llm",
        help="candidates writes candidate previews; heuristic keeps all candidates; llm calls the provider.",
    )
    parser.add_argument("--import-mathlib", action="store_true", help="Prefix merged_text with import Mathlib.")
    parser.add_argument(
        "--include-closing-ends",
        action="store_true",
        help="Append matching end commands for kept namespace/section openers.",
    )
    parser.add_argument(
        "--proof-end-marker",
        default="<END_OF_PROOF>",
        help="Marker appended to sft_target_text after the proof body.",
    )
    parser.add_argument("--recursive-depth", type=int, default=2)
    parser.add_argument("--max-dependency-candidates", type=int, default=60)
    parser.add_argument("--max-chars-per-command", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview-sample", type=int, default=None)
    parser.add_argument(
        "--source-file",
        action="append",
        default=[],
        help="Only process this source_file value, e.g. Mathlib/Tactic/Module.lean. Can be repeated.",
    )
    parser.add_argument(
        "--source-file-list",
        default=None,
        help="Text file containing source_file values, one per line.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)


def make_client_from_env(
    *,
    provider: str,
    api_key_envs: list[str],
    base_url_env: str,
    default_base_url: str,
    model_env: str,
    default_model: str,
    args: argparse.Namespace,
) -> tuple[ChatClient | None, str, str]:
    api_key = ""
    for env_name in api_key_envs:
        api_key = os.environ.get(env_name, "")
        if api_key:
            break

    base_url = args.base_url or os.environ.get(base_url_env, default_base_url)
    model = args.model or os.environ.get(model_env, default_model)

    if args.mode != "llm":
        return None, base_url, model
    if not api_key:
        raise RuntimeError(f"missing API key; set one of: {', '.join(api_key_envs)}")

    client = ChatClient(
        ChatConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    )
    return client, base_url, model
