"""Auditable, conservative *textual* references to selected MC declarations.

No value returned by this module is an elaborated Lean dependency. In particular,
the absence of a name in a proof does not imply that the declaration is unused:
typeclass search, simplification, notation and definitional reduction are implicit.
The public analyze_record(record) API accepts the original public split schema.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

VERSION = "mc-named-lexical-v1.3"
DECL_KINDS = {
    "theorem", "lemma", "axiom", "def", "abbrev", "opaque", "constant",
    "structure", "class", "inductive", "instance", "local_instance", "alias",
}
DECL_RE = re.compile(r"(?<![\w.])(?:theorem|lemma|axiom|def|abbrev|opaque|constant|structure|class|inductive|instance|alias)\b")
TACTIC_COLLISIONS = {
    "ext", "simp", "simp_all", "simp_rw", "dsimp", "rw", "rwa", "rfl",
    "exact", "apply", "refine", "intro", "intros", "constructor", "cases",
    "induction", "assumption", "contradiction", "trivial", "decide", "aesop",
    "tauto", "omega", "linarith", "nlinarith", "ring", "norm_num", "left",
    "right", "use", "unfold", "change", "rename_i", "by_contra", "by_cases",
}
VISIBILITY_EC_KINDS = {"namespace", "section", "end", "open", "open_scoped", "export"}


def mask_comments_strings(text: str) -> str:
    """Replace nested comments and string/character contents by spaces, preserving offsets."""
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("/-", i):
            start, depth = i, 1
            i += 2
            while i < n and depth:
                if text.startswith("/-", i):
                    depth += 1; i += 2
                elif text.startswith("-/", i):
                    depth -= 1; i += 2
                else:
                    i += 1
            for k in range(start, i):
                if out[k] != "\n": out[k] = " "
        elif text.startswith("--", i):
            end = text.find("\n", i)
            if end < 0: end = n
            out[i:end] = " " * (end - i)
            i = end
        elif text[i] == '"':
            start = i; i += 1
            while i < n:
                if text[i] == "\\": i += 2
                elif text[i] == '"': i += 1; break
                else: i += 1
            i = min(i, n)
            for k in range(start, i):
                if out[k] != "\n": out[k] = " "
        elif text[i] == "'" and (i == 0 or not _continue(text[i-1])):
            # Lean character literals; do not confuse a primed identifier f'.
            j = i + 1
            if j < n and text[j] == "\\": j += 2
            else: j += 1
            if j < n and text[j] == "'":
                out[i:j+1] = " " * (j + 1 - i); i = j + 1
            else: i += 1
        else:
            i += 1
    return "".join(out)


def _start(ch: str) -> bool:
    return ch == "_" or unicodedata.category(ch).startswith("L")


def _continue(ch: str) -> bool:
    return ch in "_'!?" or ch.isalnum() or unicodedata.category(ch).startswith("M")


def names(text: str):
    """Yield surface identifier tokens as (token,start,end), including dotted names."""
    i = 0
    while i < len(text):
        if text[i] == "«":
            end = text.find("»", i + 1)
            if end < 0: i += 1; continue
            j = end + 1
        elif _start(text[i]):
            j = i + 1
            while j < len(text) and _continue(text[j]): j += 1
        else:
            i += 1; continue
        while j + 1 < len(text) and text[j] == ".":
            if text[j+1] == "«":
                end = text.find("»", j + 2)
                if end < 0: break
                j = end + 1
            elif _start(text[j+1]):
                j += 2
                while j < len(text) and _continue(text[j]): j += 1
            else: break
        yield text[i:j], i, j
        i = j


def _normal(name: str) -> str:
    return name.removeprefix("_root_.")


def declaration_names(item: dict) -> dict:
    """Parse explicit declaration/alias names; generated/anonymous names stay unknown."""
    raw = str(item.get("text", item.get("effective_raw_text", item.get("raw_text", ""))))
    text = mask_comments_strings(raw)
    # @[class] is an attribute, not the declaration keyword of a subsequent
    # structure. Hide balanced attribute blocks before finding the command.
    chars = list(text)
    attr_start = 0
    while True:
        attr_start = text.find("@[", attr_start)
        if attr_start < 0: break
        attr_end, depth = attr_start + 2, 1
        while attr_end < len(text) and depth:
            if text[attr_end] == "[": depth += 1
            elif text[attr_end] == "]": depth -= 1
            attr_end += 1
        for index in range(attr_start, attr_end):
            if chars[index] != "\n": chars[index] = " "
        attr_start = attr_end
    text = "".join(chars)
    kind = item.get("kind", "unknown")
    found = DECL_RE.search(text)
    base = {"position": item.get("position"), "kind": kind,
            "names": [], "parse_status": "unsupported", "alias_source_names": []}
    if not found:
        return base
    actual = found.group()
    tail = text[found.end():].lstrip()
    if actual == "alias":
        if ":=" not in tail:
            base["parse_status"] = "unsupported_alias_syntax"
            return base
        lhs, rhs = tail.split(":=", 1)
        lhs_names = [t for t, _, _ in names(lhs) if t != "_"]
        if not lhs_names:
            return base
        base.update(names=lhs_names, parse_status="explicit_alias",
                    alias_source_names=[t for t, _, _ in names(rhs)])
        return base
    if actual == "instance":
        # The optional priority precedes either a declared instance name or its binders.
        tail = re.sub(r"^\(\s*priority\s*:=\s*[^()]*\)\s*", "", tail)
    token = next(names(tail), None)
    if token and token[1] == 0 and token[0] != "_":
        base.update(names=[token[0]], parse_status="explicit_named")
    elif actual == "instance":
        base["parse_status"] = "anonymous_instance"
    else:
        base["parse_status"] = "unsupported_name_syntax"
    return base


def _binder_fragment_names(fragment: str) -> set[str]:
    """Collect binder/pattern names without treating their annotated types as binders."""
    result = set()
    outside = []
    pairs = {"(": ")", "{": "}", "[": "]"}
    i = 0
    while i < len(fragment):
        ch = fragment[i]
        if ch == ":":
            # An unparenthesized binder annotation occupies the remainder.
            break
        if ch in pairs:
            stack = [pairs[ch]]
            j = i + 1
            while j < len(fragment) and stack:
                if fragment[j] in pairs: stack.append(pairs[fragment[j]])
                elif fragment[j] == stack[-1]: stack.pop()
                j += 1
            inner = fragment[i+1:j-1] if not stack else fragment[i+1:]
            # [C α] is an anonymous instance type; [inst : C α] names inst.
            if ch != "[" or ":" in inner:
                result.update(_binder_fragment_names(inner))
            outside.append(" ")
            i = j
        else:
            outside.append(ch)
            i += 1
    result.update(t for t, _, _ in names("".join(outside)) if t != "_" and "." not in t)
    return result


def possible_local_names(text: str) -> set[str]:
    """Over-approximate explicit binders, for ambiguity flags, not name resolution.

    Scope is deliberately over-approximated to the whole statement/proof. This
    loses some usable hits but prevents presenting a shadowed name as decisive.
    """
    clean = mask_comments_strings(text)
    result: set[str] = set()
    for m in re.finditer(r"[({\[]\s*([^(){}\[\]:\n]+?)\s*:(?!=)", clean):
        lhs = m.group(1)
        toks = list(names(lhs))
        residue = lhs
        for _, a, b in reversed(toks): residue = residue[:a] + " " * (b-a) + residue[b:]
        if not residue.strip() and all("." not in t for t, _, _ in toks):
            result.update(t for t, _, _ in toks if t != "_")
    patterns = [
        r"\b(?:fun|forall)\s+([^\n]*?)(?:=>|↦|,)",
        r"(?:∀|∃)\s+([^\n]*?)(?::|,)",
        r"\b(?:have|let|suffices|obtain)\s+([^\n]*?)(?::=|:|←)",
        r"\b(?:intro|intros|rintro|rename_i)\s+([^\n;]+)",
        r"\b(?:rcases|cases|induction)\s+[^\n]*?\bwith\s+([^\n]+)",
        r"\|\s*([^|\n]*?)(?:=>|↦)",
        r"\bcase\s+([^\n]*?)(?:=>|↦)",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, clean):
            result.update(_binder_fragment_names(m.group(1)))
    return result


def _scan(text: str, decls: list[dict], local_names: set[str], proof: bool):
    exact: dict[str, list[tuple[int, str]]] = {}
    short: dict[str, list[tuple[int, str]]] = {}
    for d in decls:
        for name in d["names"]:
            normal = _normal(name)
            pair = (d["position"], name)
            exact.setdefault(normal, []).append(pair)
            short.setdefault(normal.rsplit(".", 1)[-1], []).append(pair)
    clean = mask_comments_strings(text)
    evidence = []
    for token, start, end in names(clean):
        normal = _normal(token)
        candidates = exact.get(normal, [])
        match_type = "exact_surface_name" if candidates else None
        if not candidates:
            candidates = short.get(normal.rsplit(".", 1)[-1], [])
            if candidates:
                match_type = "unqualified_suffix" if "." not in normal else "qualified_suffix_unresolved_namespace"
        if not candidates:
            # S.field may come from a selected structure S, but this is not a
            # direct textual occurrence of a declared name: keep it separate.
            parts = normal.split(".")
            for cut in range(len(parts)-1, 0, -1):
                candidates = exact.get(".".join(parts[:cut]), [])
                if candidates:
                    match_type = "derived_member_prefix"; break
        if not candidates:
            continue
        issues = []
        if len({(p, n) for p, n in candidates}) > 1:
            issues.append("multiple_selected_declarations")
        if normal.split(".", 1)[0] in local_names:
            issues.append("possible_local_shadowing")
        if proof and "." not in normal and normal in TACTIC_COLLISIONS:
            issues.append("tactic_or_term_name_collision")
        before = clean[:start]
        label_position = bool(re.search(r"[({,]\s*$", before)) or (proof and not before.rsplit("\n", 1)[-1].strip())
        if label_position and re.match(r"\s*:=", clean[end:]):
            # A field assignment or named argument labels a binding; it does not
            # by itself use a selected declaration with the same spelling.
            issues.append("assignment_label_or_binder")
        if start > 0 and clean[start - 1] == ".":
            issues.append("leading_dot_resolution_unverified")
        if start > 0 and clean[start - 1] == "?":
            issues.append("metavariable_name")
        if match_type == "qualified_suffix_unresolved_namespace":
            issues.append("namespace_resolution_unverified")
        direct = match_type in {"exact_surface_name", "unqualified_suffix"} and not issues
        evidence.append({
            "token": token, "start_char": start, "end_char": end,
            "line": text.count("\n", 0, start) + 1,
            "match_type": match_type, "ambiguities": issues,
            "counted_named_lexical_match": direct,
            "selected_mc_candidates": [
                {"position": p, "name": name} for p, name in sorted(set(candidates), key=str)
            ],
        })
    return evidence


def analyze_record(record: dict) -> dict:
    decls = [declaration_names(i) for i in record.get("mathematical_context", [])]
    statement = str(record.get("statement", ""))
    proof = str(record.get("target", ""))
    env_variables = "\n".join(i.get("text", "") for i in record.get("environment_context", []) if i.get("kind") in {"variable", "variables"})
    # Names introduced by EC variable commands and statement binders are in scope
    # at proof entry; proof-local binders are additional possible shadows.
    shared_local = possible_local_names(env_variables + "\n" + statement)
    statement_evidence = _scan(statement, decls, shared_local, False)
    proof_evidence = _scan(proof, decls, shared_local | possible_local_names(proof), True)
    # Reverse structural direction: an EC command names an earlier selected MC
    # declaration. Position filtering prevents edges to later declarations.
    ec_evidence = []
    env_local = possible_local_names(env_variables)
    for ec in record.get("environment_context", []):
        ec_pos = ec.get("position")
        earlier_mc = [d for d in decls if isinstance(d["position"], int) and isinstance(ec_pos, int) and d["position"] < ec_pos]
        hits = _scan(ec.get("text", ""), earlier_mc, env_local | possible_local_names(ec.get("text", "")), False)
        for hit in hits:
            hit["ec_position"] = ec_pos
            hit["ec_kind"] = ec.get("kind", "unknown")
            hit["ec_role"] = "scope_or_visibility" if hit["ec_kind"] in VISIBILITY_EC_KINDS else "configuration_or_expression"
        ec_evidence.extend(hits)
    ec_direct = [e for e in ec_evidence if e["counted_named_lexical_match"] and e["ec_role"] != "scope_or_visibility"]
    ec_derived = [e for e in ec_evidence if e["match_type"] == "derived_member_prefix" and not e["ambiguities"] and e["ec_role"] != "scope_or_visibility"]
    observed_s = any(e["counted_named_lexical_match"] for e in statement_evidence)
    observed_p = any(e["counted_named_lexical_match"] for e in proof_evidence)
    profile = {(True, True): "statement_and_proof", (True, False): "statement_only",
               (False, True): "proof_only", (False, False): "no_observed_named_reference"}[observed_s, observed_p]
    statuses = Counter(d["parse_status"] for d in decls)
    return {
        "mc_label_version": VERSION,
        "mc_reference_profile": profile,
        "mc_named_decl_count": sum(bool(d["names"]) for d in decls),
        "mc_explicit_name_count": sum(len(d["names"]) for d in decls),
        "mc_anonymous_instance_count": statuses["anonymous_instance"],
        "mc_name_parse_unsupported_count": sum(n for s, n in statuses.items() if s.startswith("unsupported")),
        "mc_statement_named_reference_observed": observed_s,
        "mc_proof_named_reference_observed": observed_p,
        "mc_statement_named_reference_token_count": sum(e["counted_named_lexical_match"] for e in statement_evidence),
        "mc_proof_named_reference_token_count": sum(e["counted_named_lexical_match"] for e in proof_evidence),
        "mc_statement_ambiguous_match_observed": any(e["ambiguities"] for e in statement_evidence),
        "mc_proof_ambiguous_match_observed": any(e["ambiguities"] for e in proof_evidence),
        "mc_statement_derived_member_observed": any(e["match_type"] == "derived_member_prefix" for e in statement_evidence),
        "mc_proof_derived_member_observed": any(e["match_type"] == "derived_member_prefix" for e in proof_evidence),
        "mc_dependency_semantic_status": "not_evaluated",
        "mc_name_in_ec_observed": bool(ec_direct),
        "mc_derived_member_in_ec_observed": bool(ec_derived),
        "mc_name_in_ec_visibility_observed": any(e["counted_named_lexical_match"] and e["ec_role"] == "scope_or_visibility" for e in ec_evidence),
        "mc_name_in_ec_ambiguous_observed": any(e["ambiguities"] for e in ec_evidence),
        "mc_name_in_ec_token_count": len(ec_direct),
        "mc_derived_member_in_ec_token_count": len(ec_derived),
        "mc_name_in_ec_kind_counts": dict(sorted(Counter(e["ec_kind"] for e in ec_direct).items())),
        "mc_derived_member_in_ec_kind_counts": dict(sorted(Counter(e["ec_kind"] for e in ec_derived).items())),
        "mc_declarations": decls,
        "mc_statement_reference_evidence": statement_evidence,
        "mc_proof_reference_evidence": proof_evidence,
        "mc_name_in_ec_evidence": ec_evidence,
    }


def analyze_full_prefix(record: dict, preceding_commands: list[dict]) -> dict:
    """Optional lexical audit of the source prefix, still NOT semantic dependence.

    Callers must supply only source commands strictly preceding the target and
    retain source provenance. This helper does not certify command scope/liveness.
    Full-prefix candidates must not be described as dependencies on selected MC.
    """
    mc = []
    for i, item in enumerate(preceding_commands):
        if item.get("kind") in DECL_KINDS:
            mc.append({"position": i, "kind": item["kind"], "text": item.get("effective_raw_text", item.get("raw_text", ""))})
    labels = analyze_record({**record, "mathematical_context": mc})
    return {"source_prefix_" + k: v for k, v in labels.items()}
