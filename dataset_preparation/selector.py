"""Raw proof-blind selection from strictly preceding source commands.

This module deliberately reads only target statements and source provenance.
Preceding declaration bodies may be retained as candidate text, but
their bodies never enter lexical scoring.  Type evidence is an independently
supplied, scrubbed object, never inferred from selected context or gold proofs.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from typing import Any

MC_KINDS = frozenset({
    "theorem", "lemma", "def", "abbrev", "opaque", "axiom", "constant",
    "structure", "class", "inductive", "instance", "local_instance", "alias",
})
WRAPPER_KINDS = frozenset({"open", "open_scoped", "set_option", "variable",
                           "namespace", "section", "local", "scoped", "include", "omit"})
EC_KINDS = frozenset({
    "namespace", "section", "end", "variable", "universe", "include", "omit",
    "open", "open_scoped", "notation", "local_notation", "scoped_notation",
    "infix", "infixl", "infixr", "prefix", "postfix", "attribute", "set_option",
    "export", "local", "scoped", "syntax", "macro", "macro_rules",
})
KEYWORDS = frozenset("""
theorem lemma def abbrev opaque axiom constant structure class inductive instance
local scoped alias private protected noncomputable unsafe partial nonrec where
by do have let show from fun forall exists if then else match with return true false
Type Sort Prop namespace section end variable universe include omit open in
set_option attribute export notation infix infixl infixr prefix postfix syntax
macro macro_rules deriving extends example anonymous_target
""".lower().split())
RULES = {
    "revision": "raw-proof-blind-selector-paper-EC-order-v2",
    "provenance": "proof_blind_source_signature_ranking",
    "candidate_pool": "same_file_strictly_preceding_syntactically_named_MC",
    "direct_theorem_lemma_filter": "source_spelling_lexically_in_target_statement",
    "ranking": "Jaccard(statement_terms,name_and_signature_terms)_desc_distance_asc_id_asc",
    "dependency_priority": "stable_baseline_partition_by_independent_complete_type_command_ids",
    "fallback": "baseline_if_empty_incomplete_or_unresolved_type_mapping",
    "EC": "conservative_source_scope_prefix_target_and_entire_pool_syntactic_only",
    "EC_priority": "target_active_bindings_first_then_source_distance_then_command_ID",
    "tokenization": "actual_tokenizer_whole_command_scope_safe_packing_in_next_stage",
}
RULE_HASH = hashlib.sha256(json.dumps(RULES, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
IDENT = re.compile(r"(?:«[^»]+»|[^\W\d][\w']*)(?:\.(?:«[^»]+»|[^\W\d][\w']*))*", re.UNICODE)
DECL = re.compile(
    r"(?:^|\bin\s+|\n\s*)(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial|nonrec|local|scoped)\s+)*"
    r"(?P<kind>theorem|lemma|def|abbrev|opaque|axiom|constant|structure|class|inductive|instance|alias)\b",
    re.MULTILINE,
)
TARGET_KEYS = ("sample_id", "target_id", "split", "proof_family", "statement",
               "source_file", "source_module", "target_command_id", "target_source_span",
               "source_span", "source_order", "source_revision", "verification_location",
               "command_id", "target_command_uid", "target_kind", "original_name")


def _clean_target(target: dict[str, Any]) -> dict[str, Any]:
    # Do not iterate target keys: even access to a proof-bearing field is forbidden.
    result = {key: target.get(key) for key in TARGET_KEYS if target.get(key) is not None}
    sid = target.get("sample_id") or target.get("target_id")
    if not sid:
        raise ValueError("target requires sample_id or target_id")
    result["sample_id"] = str(sid)
    result["target_id"] = str(sid)
    return result


def choose_manifest(targets: list[dict[str, Any]], n: int = 2000,
                    seed: int = 20261007, *, split: str = "test") -> dict[str, Any]:
    """Fixed manifest, stratified only by existing proof_family.

    Only original both-nonempty targets of the requested split are eligible;
    the default is TEST.  Validation selection preserves its real split.
    Hash order within
    families and exact largest-remainder allocation are independent of dependency
    evidence, target length, model results, and proof bodies.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if split not in {"test", "validation"}:
        raise ValueError("manifest split must be test or validation")
    eligible: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for target in targets:
        if str(target.get("split", "")).lower() != split:
            continue
        both = target.get("context_presence") in {"both", "both_nonempty"}
        both = both or target.get("both_nonempty") is True
        both = both or bool(target.get("mathematical_context") and target.get("environment_context"))
        if not both:
            continue
        clean = _clean_target(target)
        sid = clean["sample_id"]
        if sid in seen:
            raise ValueError(f"duplicate eligible target ID: {sid}")
        seen.add(sid)
        family = str(target.get("proof_family") or "unknown")
        eligible.setdefault(family, []).append(clean)
    total = len(seen)
    if n > total:
        raise ValueError(f"requested {n} targets but only {total} both-nonempty {split} targets exist")
    sizes = {family: len(rows) for family, rows in sorted(eligible.items())}
    quotas = {family: (n * size // total if total else 0) for family, size in sizes.items()}
    remainder_order = sorted(sizes, key=lambda family: (
        -Fraction(n * sizes[family] % total, total) if total else Fraction(0), family))
    for family in remainder_order[:n - sum(quotas.values())]:
        quotas[family] += 1
    chosen = []
    for family, rows in sorted(eligible.items()):
        rows.sort(key=lambda row: (hashlib.sha256(
            f"{seed}\0{row['sample_id']}".encode()).hexdigest(), row["sample_id"]))
        chosen.extend(rows[:quotas[family]])
    chosen.sort(key=lambda row: (hashlib.sha256(
        f"{seed}\0manifest-order\0{row['sample_id']}".encode()).hexdigest(), row["sample_id"]))
    return {
        "schema_version": "proof-blind-selector-target-manifest-v1",
        "manifest_origin": "deterministic_stratified_fixed_subset",
        "selection_seed": seed, "n": n, "eligible_population": total,
        "population": f"original_both_nonempty_{split}_only", "split": split,
        "stratification": "existing_proof_family_exact_largest_remainder",
        "selection_uses_dependencies_or_results": False,
        "family_population_counts": sizes, "family_quotas": quotas,
        "target_ids": [row["sample_id"] for row in chosen], "targets": chosen,
        "rule_revision": RULES["revision"], "rule_revision_sha256": RULE_HASH,
    }


def _mask(text: str) -> str:
    """Replace nested comments and strings by spaces without moving offsets."""
    chars = list(text)
    i, comment_depth = 0, 0
    while i < len(text):
        if comment_depth:
            if text.startswith("/-", i):
                chars[i:i + 2] = "  "; comment_depth += 1; i += 2
            elif text.startswith("-/", i):
                chars[i:i + 2] = "  "; comment_depth -= 1; i += 2
            else:
                if text[i] != "\n": chars[i] = " "
                i += 1
        elif text.startswith("/-", i):
            chars[i:i + 2] = "  "; comment_depth = 1; i += 2
        elif text.startswith("--", i):
            while i < len(text) and text[i] != "\n":
                chars[i] = " "; i += 1
        elif text[i] == '"':
            chars[i] = " "; i += 1
            while i < len(text):
                ch = text[i]; chars[i] = " " if ch != "\n" else "\n"; i += 1
                if ch == "\\" and i < len(text): chars[i] = " "; i += 1
                elif ch == '"': break
        else:
            i += 1
    return "".join(chars)


def identifier_terms(text: str) -> set[str]:
    result: set[str] = set()
    for identifier in IDENT.findall(_mask(text)):
        if identifier.lower() == "anonymous_target":
            continue
        for part in re.split(r"[._']+", identifier.replace("«", "").replace("»", "")):
            part = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", part)
            part = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
            for token in part.lower().split():
                if token and token not in KEYWORDS:
                    result.add(token)
    return result


def _kind(command: dict[str, Any], text: str) -> tuple[str, str]:
    parser = str(command.get("parser_kind") or command.get("syntax_kind") or command.get("kind") or "")
    if parser.startswith("Lean.Parser.Command."):
        parser = parser.rsplit(".", 1)[-1]
    if parser == "instance" and re.match(r"\s*local\s+instance\b", _mask(text)):
        effective = "local_instance"
    else:
        effective = str(command.get("effective_kind") or command.get("kind") or parser)
        if effective.startswith("Lean.Parser.Command."):
            effective = effective.rsplit(".", 1)[-1]
        if effective in WRAPPER_KINDS:
            match = DECL.search(_mask(text))
            if match:
                effective = match.group("kind")
                prefix = _mask(text)[:match.end()]
                if effective == "instance" and re.search(r"\blocal\s+instance\s*$", prefix):
                    effective = "local_instance"
    return parser, effective


def _span(record: dict[str, Any], *, target: bool = False) -> dict[str, Any]:
    nested = record.get("target_source_span") if target else None
    nested = nested or record.get("source_span") or {}
    if not isinstance(nested, dict): nested = {}
    result = {}
    for name in ("start_byte", "end_byte", "start_line", "end_line", "start_col", "end_col"):
        value = nested.get(name)
        if value is None: value = record.get(("target_" if target else "") + name)
        if value is None and target: value = record.get(name)
        if value is not None: result[name] = value
    return result


def _position(span: dict[str, Any], end: bool = False) -> tuple[int, ...] | None:
    stem = "end" if end else "start"
    if span.get(stem + "_byte") is not None:
        return (int(span[stem + "_byte"]),)
    if span.get(stem + "_line") is not None:
        return (int(span[stem + "_line"]), int(span.get(stem + "_col", 0)))
    return None


def _signature(text: str) -> str:
    masked = _mask(text)
    declaration = DECL.search(masked)
    if declaration:
        # A one-command open/option wrapper and attributes are not the
        # declaration's signature and must not affect lexical ranking.
        offset = declaration.start("kind")
        text, masked = text[offset:], masked[offset:]
    depth = 0
    for i, ch in enumerate(masked):
        if ch in "([{": depth += 1
        elif ch in ")]}": depth = max(0, depth - 1)
        is_where = (i == 0 or not (masked[i - 1].isalnum() or masked[i - 1] in "_'.")) and re.match(r"where\b", masked[i:])
        if depth == 0 and (masked.startswith(":=", i) or is_where or ch == "|"):
            return text[:i].strip()
    return text.strip()


def _source_names(text: str, effective_kind: str) -> list[str]:
    if effective_kind not in MC_KINDS: return []
    masked = _mask(text)
    match = DECL.search(masked)
    if not match: return []
    tail = masked[match.end():].lstrip()
    if effective_kind == "alias":
        # Names introduced by an arrow are the alias names, not the old name.
        arrow = re.search(r"(?:←|→|<--|-->)\s*(.*)", tail)
        if arrow:
            return [token for token in IDENT.findall(arrow.group(1).split("\n", 1)[0])
                    if token != "_"]
    name_match = IDENT.match(tail)
    return [name_match.group()] if name_match else []


def _lexical_name_in_statement(name: str, statement: str) -> bool:
    statement_names = set(IDENT.findall(_mask(statement)))
    short = name.rsplit(".", 1)[-1]
    return any(token == name or token == short or token.endswith("." + name)
               or token.rsplit(".", 1)[-1] == short for token in statement_names)


def _ref(command: dict[str, Any], text: str, parser: str, kind: str,
         span: dict[str, Any], order: int) -> dict[str, Any]:
    cid = str(command.get("command_id") or "")
    file = str(command.get("source_file") or "")
    uid = command.get("uid") or command.get("command_uid") or file + "::" + cid
    return {"command_id": cid, "uid": uid, "command_uid": uid,
            "source_file": file, "source_module": command.get("source_module"),
            "source_order": order, "source_span": span, "parser_kind": parser,
            "kind": kind, "effective_kind": kind, "text": text}


def _dependency_evidence(evidence: dict[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {"status": "unresolved_target", "command_ids": [], "constant_evidence": {},
                "unmapped_same_file_constants": [], "nonpreceding_same_file_constants": [],
                "same_target_command_constants": [], "provenance": "no_independent_type_evidence"}
    ids = list(dict.fromkeys(str(cid) for cid in evidence.get("command_ids", [])))
    unmapped = list(evidence.get("unmapped_same_file_constants", []))
    later = list(evidence.get("nonpreceding_same_file_constants", []))
    status = evidence.get("status")
    if status not in {"complete_nonempty", "no_eligible_local_dependencies", "incomplete_mapping", "unresolved_target"}:
        # Never make completeness up from absence of error diagnostics alone.
        status = "incomplete_mapping"
    if unmapped or later: status = "incomplete_mapping"
    if status == "complete_nonempty" and not ids: status = "no_eligible_local_dependencies"
    if status == "no_eligible_local_dependencies" and ids: status = "incomplete_mapping"
    return {"status": status, "command_ids": ids,
            "constant_evidence": evidence.get("constant_evidence", {}),
            "unmapped_same_file_constants": unmapped, "nonpreceding_same_file_constants": later,
            "same_target_command_constants": list(evidence.get("same_target_command_constants", [])),
            "provenance": "independently_supplied_kernel_target_type_evidence"}


def build_pair(target: dict[str, Any], commands: list[dict[str, Any]],
               type_evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Build shared pool, full rankings, and fixed raw EC for two policies."""
    clean_target = _clean_target(target)
    statement = str(target.get("statement") or "")
    file = str(target.get("source_file") or "")
    target_cid = str(target.get("target_command_id") or "")
    target_span = _span(target, target=True)
    target_start = _position(target_span)
    if not statement or not file or not target_cid or target_start is None:
        raise ValueError("target requires statement, source_file, target_command_id and source start")
    query_terms = identifier_terms(statement)
    ordered = sorted(enumerate(commands), key=lambda pair: (
        _position(_span(pair[1])) or (10**18,), str(pair[1].get("command_id") or ""), pair[0]))
    prepared = []
    for index, (_, command) in enumerate(ordered):
        span = _span(command)
        start, end = _position(span), _position(span, end=True)
        source_safe = (str(command.get("source_file") or "") == file and
            str(command.get("command_id") or "") != target_cid and start is not None and end is not None and
            len(start) == len(target_start) and len(end) == len(target_start) and start < target_start and end <= target_start)
        # Current-target and later command text can contain the target proof.
        # Exclude using provenance before even fetching their source text.
        text = str(command.get("text") or command.get("effective_raw_text") or command.get("raw_text") or "").strip() if source_safe else ""
        parser, kind = _kind(command, text)
        order = int(command.get("source_order", index))
        ref = _ref(command, text, parser, kind, span, order)
        ref["source_text_read"] = source_safe
        prepared.append((command, ref))
    target_order = next((ref["source_order"] for _, ref in prepared
                         if ref["source_file"] == file and ref["command_id"] == target_cid), None)
    if target_order is None:
        target_order = int(target.get("source_order", sum(
            1 for _, ref in prepared if ref["source_file"] == file and
            _position(ref["source_span"]) is not None and _position(ref["source_span"]) < target_start)))
    pool, excluded, prior_ec = [], [], []
    seen_ids: set[str] = set()
    scope: list[dict[str, Any]] = []
    scope_valid = True
    namespace = ""
    for command, ref in prepared:
        cid, kind, parser, text = ref["command_id"], ref["kind"], ref["parser_kind"], ref["text"]
        start, end = _position(ref["source_span"]), _position(ref["source_span"], end=True)
        reasons = []
        if ref["source_file"] != file: reasons.append("different_source_file")
        if cid == target_cid: reasons.append("target_command")
        if start is None or end is None: reasons.append("missing_source_span")
        elif len(start) != len(target_start) or len(end) != len(target_start): reasons.append("incomparable_source_span")
        elif not (start < target_start and end <= target_start): reasons.append("not_strictly_preceding")
        if not cid: reasons.append("missing_command_id")
        if ref["source_text_read"] and not text: reasons.append("missing_complete_source_text")
        if ref["source_file"] == file and cid in seen_ids: reasons.append("duplicate_command_id")
        if ref["source_file"] == file: seen_ids.add(cid)
        if reasons:
            excluded.append({key: ref[key] for key in ("command_id", "uid", "source_file", "source_order", "source_span", "kind", "parser_kind")} | {"reasons": reasons})
            continue
        ref["source_distance"] = max(0, target_order - ref["source_order"])
        ref["scope_path"] = [frame["command_id"] for frame in scope]
        ref["active_namespace"] = namespace
        ref["identity_status"] = "syntactic_source_scope_only_not_Lean_elaborated"
        if kind in MC_KINDS:
            lexical_names = _source_names(text, kind)
            explicit_source_name = command.get("name")
            if not lexical_names and isinstance(explicit_source_name, str) and IDENT.fullmatch(explicit_source_name):
                lexical_names = [explicit_source_name]
            introduced = command.get("introduced_declarations") or []
            certified_names = [str(item["name"]) for item in introduced
                               if isinstance(item, dict) and item.get("name") and item.get("basis")]
            # Generated kernel helpers/projections document source ownership,
            # but must not enrich a named declaration's lexical score.
            names = list(dict.fromkeys(lexical_names or certified_names))
            if not names: reasons.append("unresolved_standalone_syntactic_name")
            if parser in {"theorem", "lemma"} and not any(
                    _lexical_name_in_statement(name, statement) for name in lexical_names):
                reasons.append("direct_theorem_lemma_target_statement_name_filter")
            if reasons:
                excluded.append({key: ref[key] for key in ("command_id", "uid", "source_file", "source_order", "source_span", "kind", "parser_kind")} | {"reasons": reasons})
                continue
            signature = _signature(text)
            candidate_terms = identifier_terms(" ".join(names) + "\n" + signature)
            union = query_terms | candidate_terms
            overlap = query_terms & candidate_terms
            ref.update(name=names[0], names=names, signature=signature,
                       score=len(overlap) / len(union) if union else 0.0,
                       score_numerator=len(overlap), score_denominator=len(union),
                       lexical_terms=sorted(candidate_terms), exclusion_reasons=[],
                       lexical_filter_source_names=lexical_names,
                       introduced_declarations=introduced,
                       name_identity_status="syntactic_source_spelling_with_kernel_ownership_evidence" if lexical_names and certified_names else "compiler_certified_declaration_names" if certified_names else "syntactic_source_spelling_only",
                       scope_identity_status="syntactic_valid" if scope_valid else "syntactic_unresolved",
                       signature_extraction_status="syntactic_header_not_Lean_elaborated",
                       syntactic_name_candidates=([name.removeprefix("_root_.") if name.startswith("_root_.")
                           else ".".join(part for part in (namespace, name) if part) for name in lexical_names]
                           if lexical_names else names[:]))
            pool.append(ref)
            continue
        masked = _mask(text)
        single_scope = kind in WRAPPER_KINDS and bool(re.search(r"\bin\s+\S", masked))
        if single_scope:
            excluded.append({key: ref[key] for key in ("command_id", "uid", "source_file", "source_order", "source_span", "kind", "parser_kind")} | {"reasons": ["single_command_wrapper_no_persistent_environment_effect"]})
            continue
        if kind not in EC_KINDS and not any(token in kind.lower() for token in ("notation", "syntax", "attribute")):
            excluded.append({key: ref[key] for key in ("command_id", "uid", "source_file", "source_order", "source_span", "kind", "parser_kind")} | {"reasons": ["not_MC_or_reconstructed_scope_EC"]})
            continue
        ref["scope_role"] = "scope_effect"
        if kind in {"namespace", "section"}:
            scope_match = re.match(r"\s*(?:noncomputable\s+)?(?:namespace|section)\b\s*([^\n]*)", masked)
            parser_scope_name = command.get("name")
            scope_name = (parser_scope_name if isinstance(parser_scope_name, str)
                          else scope_match.group(1).strip() if scope_match else "")
            if scope_match is None and parser_scope_name is None:
                scope_valid = False
                ref["scope_issue"] = "unparsed_scope_opener"
            ref["scope_role"] = "block_opener"
            ref["scope_name"] = scope_name
            ref["scope_pair_command_ids"] = []
            parts = scope_name.split(".") if scope_name else [None]
            for part in parts:
                scope.append({"command_id": cid, "name": part, "kind": kind, "previous_namespace": namespace, "ref": ref})
                if kind == "namespace":
                    if part == "_root_": namespace = ""
                    elif part: namespace = ".".join(x for x in (namespace, part) if x)
                    else: scope_valid = False
        elif kind == "end":
            end_match = re.match(r"\s*end\b\s*([^\n]*)", masked)
            end_name = end_match.group(1).strip() if end_match else ""
            ref.update(scope_role="block_closer", scope_name=end_name, scope_pair_command_ids=[])
            parts = list(reversed(end_name.split("."))) if end_name else [None]
            for part in parts:
                if not scope:
                    scope_valid = False; ref["scope_issue"] = "end_without_scope"; break
                frame = scope.pop()
                if part and part != frame["name"]:
                    scope_valid = False; ref["scope_issue"] = "end_name_mismatch"
                namespace = frame["previous_namespace"]
                ref["scope_pair_command_ids"].append(frame["command_id"])
                frame["ref"]["scope_pair_command_ids"].append(cid)
        prior_ec.append(ref)
    active_ids = set(frame["command_id"] for frame in scope)
    required_paths = set(active_ids)
    for candidate in pool: required_paths.update(candidate["scope_path"])
    for ref in prior_ec:
        if ref["scope_role"] == "block_opener":
            ref["target_scope_active"] = ref["command_id"] in active_ids
        elif ref["scope_role"] == "block_closer":
            ref["target_scope_active"] = False
        else:
            ref["target_scope_active"] = set(ref["scope_path"]).issubset(active_ids)
        ref["scope_stack_status"] = "syntactic_valid" if scope_valid else "syntactic_unresolved"
        # Bindings in active target scopes get first access to the EC budget.
        target_binding = ref["kind"] in {"variable", "universe", "include", "omit"} and ref["target_scope_active"]
        required_structure = ref["command_id"] in required_paths
        # Appendix D.5 gives two tiers, not a separate structure-priority tier.
        # Required scope cohorts still travel atomically during token packing.
        ref["priority_group"] = 0 if target_binding else 1
        ref["priority_reason"] = "target_active_scope_binding" if target_binding else "target_or_pool_scope_structure" if required_structure else "conservative_pretarget_scope_effect"
        ref["required_scope_command_ids"] = list(dict.fromkeys(ref["scope_path"] + ref.get("scope_pair_command_ids", [])))
    ec_priority = sorted(prior_ec, key=lambda ref: (ref["priority_group"], ref["source_distance"], ref["command_id"]))
    for priority, ref in enumerate(ec_priority): ref["selection_priority"] = priority
    baseline = sorted(pool, key=lambda ref: (-Fraction(ref["score_numerator"], ref["score_denominator"] or 1),
                                             ref["source_distance"], ref["command_id"]))
    base_ids = [ref["command_id"] for ref in baseline]
    deps = _dependency_evidence(type_evidence)
    dependency_ids = set(deps["command_ids"])
    pool_ids = set(base_ids)
    fallback = deps["status"] != "complete_nonempty"
    priority_ids = base_ids[:] if fallback else [cid for cid in base_ids if cid in dependency_ids] + [cid for cid in base_ids if cid not in dependency_ids]
    if deps["status"] in {"incomplete_mapping", "unresolved_target"}:
        pool_coverage = "mapping_uncertain"
    elif not dependency_ids:
        pool_coverage = "no_eligible_local_dependencies"
    elif dependency_ids <= pool_ids:
        pool_coverage = "all_dependencies_in_pool"
    elif dependency_ids & pool_ids:
        pool_coverage = "partial_dependencies_in_pool"
    else:
        pool_coverage = "no_dependencies_in_pool"
    return {
        "schema_version": "raw-proof-blind-selector-pair-v1", **clean_target,
        "protocol": RULES.copy(), "rule_revision": RULES["revision"], "rule_revision_sha256": RULE_HASH,
        "current_target_reference_proof_read": False,
        "candidate_pool_origin": "reconstructed_from_full_original_source_commands",
        "candidate_pool": sorted(pool, key=lambda ref: (ref["source_order"], ref["command_id"])),
        "candidate_exclusions": excluded,
        "candidate_exclusion_ids": [ref["command_id"] for ref in excluded],
        "baseline_ranking": base_ids, "dependency_prioritized_ranking": priority_ids,
        "dependency_priority_fallback": fallback,
        "dependency_priority_fallback_reason": deps["status"] if fallback else None,
        "target_type_dependencies": deps,
        "dependency_pool_coverage": {"status": pool_coverage,
            "independent_dependency_count": len(dependency_ids),
            "in_pool_command_ids": sorted(dependency_ids & pool_ids),
            "missing_from_pool_command_ids": sorted(dependency_ids - pool_ids)},
        "shared_environment": sorted(prior_ec, key=lambda ref: (ref["source_order"], ref["command_id"])),
        "shared_environment_command_ids": [ref["command_id"] for ref in sorted(prior_ec, key=lambda ref: (ref["source_order"], ref["command_id"]))],
        "shared_environment_priority": [ref["command_id"] for ref in ec_priority],
        "EC_identity_status": "syntactic_conservative_prefix_not_Lean_binding_certificate",
        "EC_scope_stack_status": "syntactic_valid" if scope_valid else "syntactic_unresolved",
        "EC_same_for_both_policies": True,
        "EC_budget_contract": "reserve_identical_EC_for_both_policies; preserve_or_omit_scope_cohorts_atomically_including_block_closers; never_leak_single_command_wrappers",
        "tokenization_status": "pending", "final_visibility_status": "pending_server_tokenization",
        "final_target_coverage_status": "pending_actual_whole_command_visibility",
    }
