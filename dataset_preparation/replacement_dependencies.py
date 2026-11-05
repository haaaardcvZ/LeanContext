"""Private observed-dependency exclusions for matched replacement controls.

This is deliberately reference-assisted exclusion evidence, not a proof-blind
selector and not a certificate that a replacement is mathematically irrelevant.
Only kernel metadata is read; reference proof text is never returned or ranked.
The recorded same-target compiler auxiliary expansion is retained. Earlier user
declarations are not recursively unfolded to invent a transitive dependency set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


PROTOCOL = "matched-replacement-observed-type-and-reference-value-exclusions-v1"
DEPENDENCY_FIELDS = ("type_dependency", "value_dependency", "proof_extra_dependency",
                     "projection_augmented_proof_extra_dependency")
OWNERSHIP_BASES = {
    "kernel_constructor_inductive", "kernel_recursor_inductives",
    "registered_structure_projection", "registered_aux_recursor_or_no_confusion",
}
CERTIFIED_BASES = OWNERSHIP_BASES | {
    "kernel_selection_range", "kernel_declaration_range", "exact_source_metadata_name",
}
TARGET_IDENTITIES = {
    "kernel_range_and_complete_source_namespace", "kernel_range_and_exact_source_name",
    "kernel_range_and_source_name_suffix", "unique_kernel_theorem_in_source_command",
}


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) and x for x in value)


def dependency_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Whitelist recorded type AND complete value references into JSON-safe data.

    ``mc_semantic_status`` describes value-minus-type only, so completeness also
    checks the separate type/value mappings. Types are never subtracted from the
    exclusion set. Same-command auxiliaries are allowed when source-certified.
    """
    reasons: list[str] = []
    if row.get("reference_proof_exact_raw_body_match") is not True:
        reasons.append("reference_proof_exact_raw_body_match_not_certified")
    if row.get("mc_semantic_status") != "complete":
        reasons.append("reference_value_mapping_not_complete")
    if row.get("target_identity_status") not in TARGET_IDENTITIES or not row.get("target_name"):
        reasons.append("unresolved_target_identity")
    if not all(isinstance(row.get(key), str) and row[key] for key in
               ("sample_id", "source_file", "source_module", "target_command_id")):
        reasons.append("missing_target_source_identity")
    names: set[str] = set()
    for field in ("outer_type_constants", "outer_value_constants", "type_constants",
                  "value_constants", "value_minus_type_constants"):
        values = row.get(field)
        if not _strings(values):
            reasons.append(f"missing_or_malformed_{field}")
        else:
            names.update(values)
    auxiliary_names: set[str] = set()
    auxiliaries: dict[str, list[dict[str, str]]] = {}
    for field in ("expanded_type_auxiliaries", "expanded_value_auxiliaries"):
        values = row.get(field)
        clean = []
        if not isinstance(values, list):
            reasons.append(f"missing_or_malformed_{field}")
        else:
            for item in values:
                if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
                        or not item["name"] or item.get("source_attribution") not in CERTIFIED_BASES):
                    reasons.append(f"uncertified_{field}")
                    continue
                clean.append({"name": item["name"], "source_attribution": item["source_attribution"]})
                auxiliary_names.add(item["name"])
        auxiliaries[field] = clean
    names.update(auxiliary_names)
    mapped: dict[str, list[str]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    constant_evidence: dict[str, Any] = {}
    excluded_commands: set[str] = set()
    for field in DEPENDENCY_FIELDS:
        dependency = row.get(field)
        if not isinstance(dependency, dict):
            reasons.append(f"missing_or_malformed_{field}")
            mapped[field] = []
            continue
        commands = dependency.get("command_ids")
        if not _strings(commands) or len(commands) != len(set(commands)):
            reasons.append(f"malformed_{field}_command_ids")
            commands = []
        mapped[field] = sorted(commands)
        excluded_commands.update(commands)
        evidence = dependency.get("constant_evidence")
        clean_evidence = {}
        if not isinstance(evidence, dict) or set(evidence) != set(commands):
            reasons.append(f"incomplete_{field}_constant_evidence")
            evidence = {}
        for cid, items in evidence.items():
            if not isinstance(items, list) or not items:
                reasons.append(f"malformed_{field}_constant_evidence")
                continue
            clean = []
            for item in items:
                if (not isinstance(item, dict) or not isinstance(item.get("constant"), str)
                        or not item["constant"] or item.get("source_attribution") not in CERTIFIED_BASES):
                    reasons.append(f"uncertified_{field}_constant_evidence")
                    continue
                names.add(item["constant"])
                clean.append({key: item[key] for key in
                              ("constant", "source_attribution", "declaration_kind") if key in item})
            clean_evidence[cid] = clean
        constant_evidence[field] = clean_evidence
        diagnostic = {}
        for label in ("unmapped_same_file_constants", "nonpreceding_same_file_constants",
                      "same_target_command_constants"):
            values = dependency.get(label)
            if not _strings(values):
                reasons.append(f"malformed_{field}_{label}")
                values = []
            diagnostic[label] = list(values)
            names.update(values)
            if values and label != "same_target_command_constants":
                reasons.append(f"{field}_{label}")
            if label == "same_target_command_constants":
                unresolved = set(values) - auxiliary_names - {row.get("target_name")}
                if unresolved:
                    reasons.append(f"{field}_unexpanded_same_target_auxiliaries")
        other = dependency.get("other_local_commands")
        if not isinstance(other, list):
            reasons.append(f"malformed_{field}_other_local_commands")
            other = []
        clean_other = []
        for item in other:
            if (not isinstance(item, dict) or not isinstance(item.get("command_id"), str)
                    or not isinstance(item.get("name"), str)):
                reasons.append(f"malformed_{field}_other_local_commands")
                continue
            excluded_commands.add(item["command_id"])
            names.add(item["name"])
            clean_other.append({key: item[key] for key in ("name", "command_id", "kind") if key in item})
        diagnostic["other_local_commands"] = clean_other
        diagnostics[field] = diagnostic
    if row.get("target_command_id"):
        excluded_commands.add(row["target_command_id"])
    if row.get("target_name"):
        names.add(row["target_name"])
    reasons = sorted(set(reasons))
    return {
        "schema_version": PROTOCOL, "sample_id": row.get("sample_id"),
        "source_file": row.get("source_file"), "source_module": row.get("source_module"),
        "target_command_id": row.get("target_command_id"), "target_name": row.get("target_name"),
        "target_identity_status": row.get("target_identity_status"),
        "reference_proof_exact_raw_body_match": row.get("reference_proof_exact_raw_body_match") is True,
        "availability_complete": not reasons, "rejection_reasons": reasons,
        "exclusion_command_ids": sorted(excluded_commands), "exclusion_constant_names": sorted(names),
        "target_type_command_ids": mapped["type_dependency"],
        "reference_value_command_ids": mapped["value_dependency"],
        "reference_extra_command_ids": mapped["proof_extra_dependency"],
        "projection_augmented_extra_command_ids": mapped["projection_augmented_proof_extra_dependency"],
        "constant_evidence": constant_evidence, "mapping_diagnostics": diagnostics,
        **auxiliaries,
        "reference_assisted_exclusion_only": True, "reference_proof_text_read": False,
        "dependency_scope": "observed_kernel_type_and_reference_value_plus_recorded_same_command_auxiliaries_not_arbitrary_transitive_unfolding",
        "mathematical_irrelevance_guaranteed": False,
    }


def _rows_and_hash(path: Path, digest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for line in handle:
            sha.update(line)
            size += len(line)
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: non-object JSONL record")
                yield value
    digest.update(path=str(path.resolve()), sha256=sha.hexdigest(), bytes=size)


def _kernel_checks(row: dict[str, Any], declarations: dict[str, dict[str, Any]],
                   ownership: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors, projections = [], set()
    target = declarations.get(row.get("target_name"))
    if target is None:
        return ["target_missing_from_actual_kernel_dump"], []
    if target.get("module") != row.get("source_module") or target.get("has_value") is not True:
        errors.append("kernel_target_module_or_value_mismatch")
    for side in ("type", "value"):
        if set(target.get(f"{side}_constants", [])) != set(row.get(f"outer_{side}_constants", [])):
            errors.append(f"outer_{side}_constants_differ_from_actual_kernel_dump")
        expanded = set(row.get(f"outer_{side}_constants", []))
        remaining = {item["name"] for item in row.get(f"expanded_{side}_auxiliaries", [])
                     if isinstance(item, dict) and isinstance(item.get("name"), str)}
        while remaining:
            reachable = remaining & expanded
            if not reachable:
                errors.append(f"unreachable_recorded_{side}_auxiliary")
                break
            for name in reachable:
                auxiliary = declarations.get(name)
                if auxiliary is None or auxiliary.get("module") != row.get("source_module") or auxiliary.get("has_value") is not True:
                    errors.append(f"missing_or_uncertified_kernel_{side}_auxiliary")
                else:
                    expanded.update(auxiliary.get("value_constants", []))
                    for projection in auxiliary.get("value_projections", []):
                        projections.update(x for x in (projection.get("structure"), projection.get("projection_name")) if isinstance(x, str) and x)
                remaining.remove(name)
        if expanded != set(row.get(f"{side}_constants", [])):
            errors.append(f"expanded_{side}_constants_differ_from_actual_kernel_dump")
        for projection in target.get(f"{side}_projections", []):
            projections.update(x for x in (projection.get("structure"), projection.get("projection_name")) if isinstance(x, str) and x)
    if set(row.get("value_minus_type_constants", [])) != set(row.get("value_constants", [])) - set(row.get("type_constants", [])):
        errors.append("recorded_constant_difference_is_not_value_minus_type")
    for field in DEPENDENCY_FIELDS:
        for items in (row.get(field) or {}).get("constant_evidence", {}).values():
            for item in items:
                declaration = declarations.get(item.get("constant"))
                if declaration is None or declaration.get("module") != row.get("source_module"):
                    errors.append("mapped_local_constant_missing_or_wrong_kernel_module")
                basis = item.get("source_attribution")
                if basis in OWNERSHIP_BASES:
                    certificate = ownership.get(item.get("constant"))
                    if certificate is None or certificate.get("basis") != basis or not certificate.get("owners"):
                        errors.append("mapped_constant_ownership_certificate_missing")
    return sorted(set(errors)), sorted(projections)


def load_evidence(audit_root: Path, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load requested metadata and verify it against the actual pinned dumps.

    All wanted IDs are returned, including unavailable IDs. Audit file hashes
    come from the exact bytes scanned, not copied from an old summary. No gold
    text or model-result file is opened. Current command records are checked by
    ``candidate_exclusion_reason`` at matching time.
    """
    audit_root = Path(audit_root)
    wanted = {str(sid) for sid in wanted_ids}
    source_hashes: dict[str, dict[str, Any]] = {}
    semantic, requested_names = {}, set()
    digest: dict[str, Any] = {}
    for number, row in enumerate(_rows_and_hash(audit_root / "semantic_mc_labels.jsonl", digest), 1):
        sid = row.get("sample_id")
        if sid not in wanted:
            continue
        if sid in semantic:
            raise ValueError(f"duplicate requested semantic dependency target: {sid}")
        semantic[sid] = (row, number)
        if row.get("target_name"):
            requested_names.add(row["target_name"])
        for field in ("expanded_type_auxiliaries", "expanded_value_auxiliaries"):
            requested_names.update(item["name"] for item in row.get(field, []) if isinstance(item, dict) and isinstance(item.get("name"), str))
        for field in DEPENDENCY_FIELDS:
            for items in (row.get(field) or {}).get("constant_evidence", {}).values():
                requested_names.update(item["constant"] for item in items if isinstance(item, dict) and isinstance(item.get("constant"), str))
    source_hashes["semantic_mc_labels.jsonl"] = digest
    declarations, digest = {}, {}
    for row in _rows_and_hash(audit_root / "declaration_dependencies.jsonl", digest):
        if row.get("name") in requested_names:
            if row["name"] in declarations:
                raise ValueError(f"duplicate requested kernel declaration: {row['name']}")
            declarations[row["name"]] = row
    source_hashes["declaration_dependencies.jsonl"] = digest
    ownership, digest = {}, {}
    for row in _rows_and_hash(audit_root / "declaration_ownership.jsonl", digest):
        if row.get("name") in requested_names:
            ownership[row["name"]] = row
    source_hashes["declaration_ownership.jsonl"] = digest
    result = {}
    for sid in sorted(wanted):
        raw, number = semantic.get(sid, ({"sample_id": sid}, None))
        evidence = dependency_evidence(raw)
        errors, projections = _kernel_checks(raw, declarations, ownership)
        evidence["rejection_reasons"] = sorted(set(evidence["rejection_reasons"] + errors))
        evidence["availability_complete"] = not evidence["rejection_reasons"]
        evidence["exclusion_projection_reference_names"] = projections
        evidence["exclusion_constant_names"] = sorted(set(evidence["exclusion_constant_names"]) | set(projections))
        evidence["kernel_dump_metadata_verified"] = not errors
        evidence["provenance"] = {"audit_root": str(audit_root.resolve()), "semantic_record_index_1_based": number,
                                  "sources": source_hashes}
        result[sid] = evidence
    return result


def candidate_exclusion_reason(command: dict[str, Any], evidence: dict[str, Any]) -> str | None:
    """Reject unresolved candidate identity; absence is not semantic irrelevance.

    This checks dependency/identity exclusions only. The caller still enforces
    same file, strict precedence, kind/scope, statement-name and length matching.
    """
    if evidence.get("availability_complete") is not True:
        return "target_dependency_evidence_incomplete"
    cid = str(command.get("command_id", ""))
    if cid in set(evidence.get("exclusion_command_ids", [])):
        return "observed_target_type_or_reference_value_dependency_command"
    if command.get("introduced_declarations_status") != "kernel_range_or_certified_ownership_attributed":
        return "candidate_introduced_declarations_unresolved"
    declarations = command.get("introduced_declarations")
    if not isinstance(declarations, list) or not declarations:
        return "candidate_introduced_declarations_unresolved"
    names = set()
    for declaration in declarations:
        if (not isinstance(declaration, dict) or not isinstance(declaration.get("name"), str)
                or not declaration["name"] or declaration.get("basis") not in CERTIFIED_BASES):
            return "candidate_introduced_declarations_uncertified"
        names.add(declaration["name"])
    if names & set(evidence.get("exclusion_constant_names", [])):
        return "candidate_introduces_observed_dependency_constant"
    return None
