from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


VALID_DECL_KINDS = {"theorem", "lemma", "example"}
DEFAULT_PROOF_END_MARKERS = ("<END_OF_PROOF>", "<END OF PROOF>")

SURFACE_GROUPS = {
    "E00_invalid_or_excluded",
    "NT01_term_atomic_no_by",
    "NT02_term_application_no_by",
    "NT03_term_lambda_no_by",
    "NT04_term_constructor_record_no_by",
    "NT05_term_match_recursion_no_by",
    "NT06_term_structural_other_no_by",
    "C01_calc_top_no_by",
    "C02_calc_top_with_by_leaves",
    "C03_by_calc_shell",
    "H01_outer_term_with_inner_by",
    "H02_by_exact_refine_term_payload",
    "T01_by_single_command",
    "T02_by_multi_command_linear",
    "T03_by_multi_command_branching",
    "O01_other_mixed",
}

TOKEN_BOUNDARY = r"A-Za-z0-9_'."


def token_pattern(token: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![{TOKEN_BOUNDARY}]){re.escape(token)}(?![{TOKEN_BOUNDARY}])")


def contains_token(text: str, token: str) -> bool:
    return bool(token_pattern(token).search(text))


def count_token(text: str, token: str) -> int:
    return len(token_pattern(token).findall(text))


def starts_with_token(text: str, token: str) -> bool:
    stripped = text.lstrip()
    return bool(re.match(rf"{re.escape(token)}(?![{TOKEN_BOUNDARY}])", stripped))


def starts_with_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    return any(starts_with_token(text, token) for token in tokens)


def strip_lean_comments(text: str) -> str:
    out: list[str] = []
    n = len(text)
    i = 0
    in_string = False

    while i < n:
        ch = text[i]

        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if i + 1 < n and text[i] == "-" and text[i + 1] == "-":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue

        if i + 1 < n and text[i] == "/" and text[i + 1] == "-":
            depth = 1
            i += 2
            while i < n and depth > 0:
                if i + 1 < n and text[i] == "/" and text[i + 1] == "-":
                    depth += 1
                    i += 2
                    continue
                if i + 1 < n and text[i] == "-" and text[i + 1] == "/":
                    depth -= 1
                    i += 2
                    continue
                if text[i] in "\r\n":
                    out.append(text[i])
                i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def remove_proof_end_marker(text: str, marker: str | None) -> str:
    markers = []
    if marker:
        markers.append(marker)
    markers.extend(m for m in DEFAULT_PROOF_END_MARKERS if m not in markers)

    out = text
    for item in markers:
        out = out.replace(item, "")
    return out


def clean_target(text: str, marker: str | None) -> str:
    text = remove_proof_end_marker(text, marker)
    text = strip_lean_comments(text)
    return text.strip()


def rough_tokens(text: str) -> list[str]:
    return re.findall(r"[^\s()\[\]{}.,;:]+|[()\[\]{}.,;:]", text)


def proof_num_lines(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.strip().splitlines())


def strip_top_level_by(text: str) -> str:
    stripped = text.lstrip()
    match = re.match(rf"by(?![{TOKEN_BOUNDARY}])", stripped)
    if not match:
        return stripped
    return stripped[match.end() :].lstrip()


def first_nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def minimal_indent(lines: list[str]) -> int:
    indents = [len(line) - len(line.lstrip(" \t")) for line in lines if line.strip()]
    return min(indents) if indents else 0


def split_semicolon_commands(line: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    count = 1
    for ch in line:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "([{⟨":
            depth += 1
            continue
        if ch in ")]}⟩" and depth > 0:
            depth -= 1
            continue
        if ch == ";" and depth == 0:
            count += 1
    return count


def count_top_level_tactic_commands(body: str) -> int:
    lines = first_nonempty_lines(body)
    if not lines:
        return 0
    base_indent = minimal_indent(lines)
    count = 0
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" \t"))
        if not stripped:
            continue
        if stripped.startswith(("·", "|")):
            continue
        if starts_with_token(stripped, "case") or starts_with_token(stripped, "next"):
            continue
        if indent == base_indent:
            count += split_semicolon_commands(stripped)
    return count


def body_after_show_from(body: str) -> str | None:
    stripped = body.strip()
    if not starts_with_token(stripped, "show"):
        return None
    match = token_pattern("from").search(stripped)
    if match is None:
        return None
    return stripped[match.end() :].lstrip()


def is_by_calc_shell(body: str) -> bool:
    stripped = body.strip()
    if starts_with_token(stripped, "calc"):
        return True

    for keyword in ("exact", "refine"):
        if starts_with_token(stripped, keyword):
            payload = stripped[len(keyword) :].lstrip()
            if starts_with_token(payload, "calc"):
                return True

    show_payload = body_after_show_from(stripped)
    return bool(show_payload and starts_with_token(show_payload, "calc"))


def has_branching_structure(body: str) -> bool:
    stripped = body.strip()
    if not stripped:
        return False
    if "·" in stripped or "<;>" in stripped:
        return True
    if re.search(r"(?m)^\s*\|", stripped):
        return True
    if re.search(r"(?m)^\s*(case|next)\b", stripped):
        return True
    if "=>" in stripped and re.search(r"(?m)^\s*(\||case|next)\b", stripped):
        return True
    for token in ("all_goals", "any_goals", "focus", "by_cases", "induction", "cases", "rcases", "split_ifs"):
        if contains_token(stripped, token):
            return True
    if contains_token(stripped, "split"):
        return True
    if contains_token(stripped, "constructor"):
        if "<;>" in stripped or "·" in stripped or count_top_level_tactic_commands(stripped) >= 2:
            return True
    for token in ("left", "right"):
        if contains_token(stripped, token) and count_top_level_tactic_commands(stripped) >= 2:
            return True
    return False


COMPLEX_PAYLOAD_MARKERS = (
    "fun",
    "λ",
    "⟨",
    "{",
    "match",
    "nomatch",
    "Eq.",
    "Iff.",
    "And.",
    "Or.",
    "Exists.",
    "Subtype.",
    "SetLike.",
    "DFunLike.",
)


def is_complex_term_payload(payload: str) -> bool:
    stripped = payload.strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return True
    if stripped.startswith(("⟨", "{", "λ")):
        return True
    for marker in COMPLEX_PAYLOAD_MARKERS:
        if marker in stripped:
            return True
    if starts_with_any_token(stripped, ("fun", "match", "nomatch")):
        return True
    return False


def is_by_exact_refine_term_payload(body: str) -> bool:
    stripped = body.strip()
    if not stripped or has_branching_structure(stripped):
        return False
    if is_by_calc_shell(stripped):
        return False

    for keyword in ("exact", "refine"):
        if starts_with_token(stripped, keyword):
            payload = stripped[len(keyword) :].lstrip()
            return is_complex_term_payload(payload)

    show_payload = body_after_show_from(stripped)
    if show_payload is not None:
        return is_complex_term_payload(show_payload)

    return False


def strip_outer_parens(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed and stripped.startswith("(") and stripped.endswith(")"):
        changed = False
        depth = 0
        in_string = False
        escaped = False
        for idx, ch in enumerate(stripped):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(stripped) - 1:
                    return stripped
        if depth == 0:
            stripped = stripped[1:-1].strip()
            changed = True
    return stripped


def is_atomic_term(text: str) -> bool:
    stripped = strip_outer_parens(text)
    return stripped in {"rfl", "trivial", "infer_instance", "inferInstance"}


def is_lambda_term(text: str) -> bool:
    stripped = text.lstrip()
    return starts_with_token(stripped, "fun") or stripped.startswith("λ")


def is_constructor_or_record_term(text: str) -> bool:
    stripped = text.lstrip()
    if stripped.startswith(("⟨", "{")):
        return True
    for marker in ("And.intro", "Iff.intro", "Exists.intro", "Subtype.mk", "Prod.mk"):
        if marker in stripped:
            return True
    return False


def is_match_or_recursion_term(text: str) -> bool:
    stripped = text.lstrip()
    if stripped.startswith("|"):
        return True
    if starts_with_any_token(stripped, ("match", "nomatch")):
        return True
    if re.search(r"(\.rec\b|\.recOn\b|\bcasesOn\b|\brecOn\b)", stripped):
        return True
    for marker in ("Nat.rec", "List.rec", "Option.rec", "Bool.rec", "Quot.inductionOn"):
        if marker in stripped:
            return True
    return False


def is_structural_term(text: str) -> bool:
    stripped = text.lstrip()
    if stripped.startswith("("):
        return True
    if starts_with_any_token(stripped, ("show", "let", "have", "if", "dite", "cast")):
        return True
    return False


def is_application_term(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if is_structural_term(stripped):
        return False
    if starts_with_any_token(
        stripped,
        ("by", "calc", "fun", "match", "nomatch", "show", "let", "have", "if", "then", "else"),
    ):
        return False
    first = stripped[0]
    if first in "([{⟨{":
        return False
    return bool(re.match(r"[@.A-Za-z_α-ωΑ-ΩÀ-ɏ]", first))


def classify_surface(target_clean: str, decl_kind: str) -> tuple[str, str | None]:
    if decl_kind not in VALID_DECL_KINDS:
        return "E00_invalid_or_excluded", "invalid_decl_kind"
    if not target_clean:
        return "E00_invalid_or_excluded", "empty_target"
    if contains_token(target_clean, "sorry"):
        return "E00_invalid_or_excluded", "contains_sorry"
    if contains_token(target_clean, "admit"):
        return "E00_invalid_or_excluded", "contains_admit"

    if starts_with_token(target_clean, "calc"):
        if contains_token(target_clean, "by"):
            return "C02_calc_top_with_by_leaves", None
        return "C01_calc_top_no_by", None

    if starts_with_token(target_clean, "by"):
        body = strip_top_level_by(target_clean)
        if is_by_calc_shell(body):
            return "C03_by_calc_shell", None
        if is_by_exact_refine_term_payload(body):
            return "H02_by_exact_refine_term_payload", None
        if has_branching_structure(body):
            return "T03_by_multi_command_branching", None
        if count_top_level_tactic_commands(body) >= 2:
            return "T02_by_multi_command_linear", None
        return "T01_by_single_command", None

    if contains_token(target_clean, "by"):
        return "H01_outer_term_with_inner_by", None

    if is_atomic_term(target_clean):
        return "NT01_term_atomic_no_by", None
    if is_lambda_term(target_clean):
        return "NT03_term_lambda_no_by", None
    if is_constructor_or_record_term(target_clean):
        return "NT04_term_constructor_record_no_by", None
    if is_match_or_recursion_term(target_clean):
        return "NT05_term_match_recursion_no_by", None
    if is_application_term(target_clean):
        return "NT02_term_application_no_by", None
    if is_structural_term(target_clean):
        return "NT06_term_structural_other_no_by", None
    return "O01_other_mixed", None


def top_level_form(target_clean: str) -> str:
    if starts_with_token(target_clean, "by"):
        return "by"
    if starts_with_token(target_clean, "calc"):
        return "calc"
    if target_clean:
        return "term"
    return "invalid"


def sample_id_for(rec: dict[str, Any]) -> str:
    existing = rec.get("sample_id")
    if isinstance(existing, str) and existing:
        return existing
    key = "::".join(
        str(rec.get(k) or "")
        for k in ("lean_file_name", "target_command_id", "theorem_name", "target_start_line")
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def make_output_record(rec: dict[str, Any]) -> dict[str, Any]:
    decl_kind = str(rec.get("target_kind") or rec.get("kind") or "")
    marker = rec.get("proof_end_marker")
    marker = marker if isinstance(marker, str) else None
    target_raw = str(rec.get("sft_target_text") or "")
    target_no_marker = remove_proof_end_marker(target_raw, marker).strip()
    target_clean = clean_target(target_raw, marker)
    group, exclude_reason = classify_surface(target_clean, decl_kind)

    top_form = top_level_form(target_clean)
    has_top_by = top_form == "by"
    has_top_calc = top_form == "calc"
    by_count = count_token(target_clean, "by")
    contains_inner_by = by_count > (1 if has_top_by else 0)
    if has_top_calc and by_count > 0:
        contains_inner_by = True

    tactic_count = 0
    branching = False
    if has_top_by:
        body = strip_top_level_by(target_clean)
        tactic_count = count_top_level_tactic_commands(body)
        branching = has_branching_structure(body)

    out = {
        "sample_id": sample_id_for(rec),
        "kind": decl_kind,
        "sft_input_text": rec.get("sft_input_text") or "",
        "sft_input_text_without_context": rec.get("sft_input_text_without_context") or "",
        "sft_target_text": target_no_marker,
        "mathlib4_version": rec.get("mathlib4_version"),
        "surface_group": group,
        "top_level_form": top_form,
        "has_top_level_by": has_top_by,
        "has_top_level_calc": has_top_calc,
        "contains_inner_by": contains_inner_by,
        "is_strict_non_tactic": group.startswith("NT"),
        "is_calc_form": group.startswith("C"),
        "is_hybrid_form": group in {
            "H01_outer_term_with_inner_by",
            "H02_by_exact_refine_term_payload",
            "C02_calc_top_with_by_leaves",
            "C03_by_calc_shell",
        },
        "is_top_level_tactic": group.startswith("T"),
        "proof_num_lines": proof_num_lines(target_clean),
        "proof_num_tokens_rough": len(rough_tokens(target_clean)),
        "tactic_command_count_rough": tactic_count,
        "has_branching_structure": branching,
    }
    if exclude_reason:
        out["exclude_reason"] = exclude_reason
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Lean proof suffixes by source-level surface form.")
    parser.add_argument(
        "--input",
        default="extract_data/theorem_contexts_deepseek_hybrid.statement_context.jsonl",
        help="Input SFT JSONL.",
    )
    parser.add_argument(
        "--output",
        default="extract_data/theorem_contexts_deepseek_hybrid.statement_context.surface.jsonl",
        help="Output JSONL with surface-form classification fields.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    failed = 0
    counts: dict[str, int] = {group: 0 for group in sorted(SURFACE_GROUPS)}

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
                out = make_output_record(rec)
                group = str(out["surface_group"])
                counts[group] = counts.get(group, 0) + 1
            except Exception as e:
                failed += 1
                out = {
                    "sample_id": hashlib.sha1(f"{input_path}:{line_no}".encode("utf-8")).hexdigest()[:24],
                    "kind": "",
                    "sft_input_text": "",
                    "sft_input_text_without_context": "",
                    "sft_target_text": "",
                    "mathlib4_version": None,
                    "surface_group": "E00_invalid_or_excluded",
                    "top_level_form": "invalid",
                    "has_top_level_by": False,
                    "has_top_level_calc": False,
                    "contains_inner_by": False,
                    "is_strict_non_tactic": False,
                    "is_calc_form": False,
                    "is_hybrid_form": False,
                    "is_top_level_tactic": False,
                    "proof_num_lines": 0,
                    "proof_num_tokens_rough": 0,
                    "tactic_command_count_rough": 0,
                    "has_branching_structure": False,
                    "exclude_reason": "classifier_exception",
                    "classifier_error": repr(e),
                    "input_line_no": line_no,
                }
                counts["E00_invalid_or_excluded"] = counts.get("E00_invalid_or_excluded", 0) + 1

            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            if args.progress_every and total % args.progress_every == 0:
                print(f"[progress] total={total} failed={failed}")

    summary = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "total": total,
        "failed": failed,
        "surface_group_counts": {k: v for k, v in sorted(counts.items()) if v},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
