#!/usr/bin/env python3
"""Wire real source/kernel/tokenized records into the paper's diagnostic tables.

No model outcomes are generated or invented. D1/D3 use direct target-type
constants; D5 proof-resource audits are posthoc and reference-assisted. Unknown
mapping/visibility is never treated as an empty dependency set or zero coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path

from .evidence_analysis import aggregate_coverage, aggregate_paired, read_attempts
from .representation_analysis import iter_jsonl, sha256, write_json

MC_KINDS = frozenset({"theorem", "lemma", "def", "abbrev", "opaque", "axiom", "constant",
                      "structure", "class", "inductive", "instance", "local_instance", "alias"})


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def indexed_rows(path: Path, wanted: set[str] | None = None) -> dict:
    result = {}
    for row in iter_jsonl(path):
        sid = row["sample_id"]
        if wanted is not None and sid not in wanted:
            continue
        if sid in result:
            raise ValueError(f"Duplicate sample ID in {path.name}: {sid}")
        result[sid] = row
    return result


def load_population(root: Path, population: Path | None, expected: int) -> list[dict]:
    requested = None if population is None else set(indexed_rows(population))
    records = [row for row in iter_jsonl(root / "benchmark/test.jsonl")
               if row["mathematical_context"] and row["environment_context"]
               and (requested is None or row["sample_id"] in requested)]
    ids = {row["sample_id"] for row in records}
    if len(ids) != len(records) or len(records) != expected or (requested is not None and ids != requested):
        raise ValueError(f"Require exactly {expected} original both-nonempty test IDs")
    return records


class SourceCatalogs:
    """Bounded source cache: verify mapped resources independently of selected MC."""
    def __init__(self, root: Path):
        self.root, self.cache = root, OrderedDict()

    def __call__(self, source_file: str) -> dict:
        relative = Path(source_file)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source path")
        if source_file not in self.cache:
            rows = list(iter_jsonl(self.root / "command_mapping/source_commands" / relative.with_suffix(".jsonl")))
            catalog = {row["command_id"]: row for row in rows}
            if len(catalog) != len(rows):
                raise ValueError("Duplicate source-command ID")
            self.cache[source_file] = catalog
            if len(self.cache) > 24:
                self.cache.popitem(last=False)
        self.cache.move_to_end(source_file)
        return self.cache[source_file]


def mapped_dependencies(target: dict, kernel: dict | None, catalog: dict, *, mode="type") -> dict:
    """Use recorded constant-to-command evidence, never subtract command sets.

    type: filter recorded type_dependency evidence by OUTER direct type constants.
    proof: require stored value_minus_type_constants == value_constants-type_constants.
    projection: report the recorded projection-augmented difference mapping as
    a separate variant; its plus-constant sets are not present to recompute here.
    Earlier named declarations are never unfolded by this module.
    """
    result = {"complete": False, "command_uids": [], "command_ids": [], "reasons": [],
              "mode": mode, "reference_proof_used": mode != "type",
              "recursive_unfolding_performed": False, "constant_mapping_recomputed": False}
    if kernel is None:
        result["reasons"].append("kernel_record_missing")
        return result
    if kernel.get("source_file") != target["source_file"] or kernel.get("target_command_id") != target["target_command_id"]:
        result["reasons"].append("kernel_target_source_identity_mismatch")
        return result
    if kernel.get("target_name") is None:
        result["reasons"].append("unresolved_kernel_target_identity")
        return result
    if mode == "type":
        constants = kernel.get("outer_type_constants")
        field = "type_dependency"
        result["basis"] = "direct_elaborated_outer_type_constants_filtered_from_recorded_mapping"
    elif mode == "proof":
        field = "proof_extra_dependency"
        needed = ("value_constants", "type_constants", "value_minus_type_constants")
        if any(not isinstance(kernel.get(key), list) for key in needed):
            result["reasons"].append("constant_difference_evidence_missing")
            return result
        constants = set(kernel["value_constants"]) - set(kernel["type_constants"])
        if constants != set(kernel["value_minus_type_constants"]):
            result["reasons"].append("stored_constant_difference_is_inconsistent")
            return result
        result["basis"] = "recorded_constant_P_minus_T_before_command_mapping_certified_own_auxiliaries_only"
    elif mode == "projection":
        field = "projection_augmented_proof_extra_dependency"
        constants = None
        result["basis"] = "recorded_projection_augmented_constant_difference_mapping_not_recomputed"
        result["projection_plus_constant_sets_available_for_recomputation"] = False
    else:
        raise ValueError("Unknown dependency mode")
    if mode != "projection" and not isinstance(constants, (list, set)):
        result["reasons"].append("direct_constant_evidence_missing")
        return result
    if mode != "type" and kernel.get("reference_proof_exact_raw_body_match") is not True:
        result["reasons"].append("reference_proof_source_identity_unconfirmed")
        return result
    mapping = kernel.get(field)
    if not isinstance(mapping, dict) or not isinstance(mapping.get("constant_evidence"), dict):
        result["reasons"].append("recorded_constant_to_command_mapping_missing")
        return result
    evidence_keys = ("command_ids", "unmapped_same_file_constants", "same_target_command_constants",
                     "nonpreceding_same_file_constants", "other_local_commands")
    if any(not isinstance(mapping.get(key), list) for key in evidence_keys):
        result["reasons"].append("mapping_completeness_fields_missing")
        return result
    result["recorded_mapping_field"] = field
    direct = None if constants is None else set(constants)
    unknown = set(mapping.get("unmapped_same_file_constants", []))
    if direct is not None:
        unknown &= direct
    if unknown:
        result["reasons"].append("unmapped_same_file_constants")
    result["unmapped_same_file_constants"] = sorted(unknown)
    result["excluded_same_target_constants"] = [c for c in mapping.get("same_target_command_constants", [])
                                               if direct is None or c in direct]
    result["excluded_nonpreceding_constants"] = [c for c in mapping.get("nonpreceding_same_file_constants", [])
                                                if direct is None or c in direct]
    result["excluded_other_local_non_MC_commands"] = mapping.get("other_local_commands", [])
    target_start = target.get("target_source_span", {}).get("start_byte")
    if target_start is None:
        result["reasons"].append("target_source_span_missing")
        return result
    selected, evidence = set(), {}
    mapping_ids = mapping.get("command_ids")
    if not isinstance(mapping_ids, list):
        result["reasons"].append("mapped_command_ID_list_missing")
        return result
    for cid in mapping_ids:
        entries = mapping["constant_evidence"].get(cid)
        if not isinstance(entries, list) or not entries:
            result["reasons"].append(f"constant_evidence_missing:{cid}")
            continue
        matching = [entry for entry in entries if direct is None or entry.get("constant") in direct]
        if not matching:
            continue
        command = catalog.get(cid)
        if command is None:
            result["reasons"].append(f"mapped_source_command_missing:{cid}")
            continue
        span = command.get("source_span", {})
        kind = command.get("effective_kind", command.get("kind"))
        if command.get("source_file", target["source_file"]) != target["source_file"] or kind not in MC_KINDS:
            result["reasons"].append(f"mapped_command_not_same_file_MC:{cid}")
            continue
        if span.get("start_byte") is None or span.get("end_byte") is None:
            result["reasons"].append(f"mapped_command_span_missing:{cid}")
            continue
        if cid == target["target_command_id"] or not (span["start_byte"] < target_start and span["end_byte"] <= target_start):
            result["reasons"].append(f"mapped_command_not_strictly_preceding:{cid}")
            continue
        selected.add(cid)
        evidence[cid] = matching
    result["command_ids"] = sorted(selected)
    result["command_uids"] = [f"{target['source_file']}::{cid}" for cid in sorted(selected)]
    result["constant_evidence"] = evidence
    result["complete"] = not result["reasons"]
    result["status"] = "complete_nonempty" if result["complete"] and selected else "complete_empty" if result["complete"] else "incomplete_mapping"
    return result


def final_visibility(row: dict | None, target: dict) -> tuple[bool, list[str], str | None]:
    if row is None:
        return False, [], "final_serialized_input_missing"
    if row.get("sample_id") != target["sample_id"] or row.get("statement", target["statement"]) != target["statement"]:
        return False, [], "final_input_target_mismatch"
    if row.get("view") != "CFull" or not isinstance(row.get("context_items"), list):
        return False, [], "complete_CFull_command_visibility_not_recorded"
    if not isinstance(row.get("input_ids"), list) or not (0 < row.get("prompt_length", 0) <= 1024):
        return False, [], "final_input_token_span_unknown"
    visible = []
    for item in row["context_items"]:
        if item.get("role") == "MC":
            uid = item.get("command_uid")
            if not uid or not uid.startswith(target["source_file"] + "::"):
                return False, [], "final_MC_command_UID_missing_or_wrong_source"
            visible.append(uid)
    return True, sorted(set(visible)), None


def coverage_row(target: dict, dependencies: dict, known: bool, visible: list[str], reason=None) -> dict:
    return {"sample_id": target["sample_id"], "source_file": target["source_file"],
            "dependency_mapping_complete": dependencies["complete"],
            "final_visibility_known": known,
            "eligible_target_dependency_command_uids": dependencies["command_uids"],
            "visible_mc_command_uids": visible, "dependency_evidence": dependencies,
            "visibility_unresolved_reason": reason}


def coverage_summary(rows: list[dict]) -> dict:
    summary = aggregate_coverage(rows)
    summary["evidence_status_counts"] = dict(Counter(
        "mapping_and_visibility_unresolved" if not row["dependency_mapping_complete"] and not row["final_visibility_known"] else
        "mapping_incomplete" if not row["dependency_mapping_complete"] else
        "visibility_unknown" if not row["final_visibility_known"] else
        "empty_eligible_dependencies" if not row["eligible_target_dependency_command_uids"] else "coverage_computable"
        for row in rows))
    return summary


def pool_coverage(dependencies: dict, pool_ids: set[str]) -> str:
    if not dependencies["complete"]:
        return "unresolved_mapping"
    deps = set(dependencies["command_ids"])
    if not deps:
        return "no_eligible_local_dependencies"
    return "all" if deps <= pool_ids else "some" if deps & pool_ids else "none"


def make_length_groups(targets: list[dict], lengths: dict[str, int]) -> dict:
    ids = {row["sample_id"] for row in targets}
    if set(lengths) != ids or len(ids) % 4:
        raise ValueError("Require complete length evidence and a population divisible by four")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in lengths.values()):
        raise ValueError("Every pre-clipping complete input length must be a positive integer")
    ordered = sorted(ids, key=lambda sid: (lengths[sid], sid))
    count = len(ordered) // 4
    return {f"Q{i+1}": ordered[i*count:(i+1)*count] for i in range(4)}


def recorded_mixed_groups(ids: set[str], annotations: dict | None) -> dict:
    if annotations is None:
        return {"status": "not_available", "reason": "complete_recorded_mixed_role_annotations_not_supplied", "groups": None}
    if set(annotations) != ids:
        raise ValueError("Mixed-role annotations must explicitly cover every population target")
    groups = {"none": [], "at_least_one": []}
    for sid in sorted(ids):
        row = annotations[sid]
        value = row.get("has_mixed_role_command")
        if not isinstance(value, bool) or not row.get("annotation_basis"):
            raise ValueError("Recorded mixed-role boolean and annotation_basis are required; no kind heuristic")
        groups["at_least_one" if value else "none"].append(sid)
    return {"status": "complete_recorded_annotations", "groups": groups, "inferred_from_command_kind": False}


def kernel_path(root: Path, supplied) -> Path:
    return Path(supplied) if supplied else root / "private_posthoc_audit/kernel_inputs/semantic_mc_labels.jsonl"


def input_path(root: Path, backbone: str) -> Path:
    return root / f"tokenized/{backbone}/training/input_1024/views/CFull/test.jsonl"


def real_input_lengths(root: Path, targets: list[dict], supplied=None) -> tuple[dict, dict]:
    wanted = {row["sample_id"] for row in targets}
    if supplied:
        rows = indexed_rows(Path(supplied), wanted)
        lengths = {}
        for sid, row in rows.items():
            # Context-only lengths cannot be substituted for complete prompts.
            candidates = [row[key] for key in ("preclipping_full_input_length", "untrimmed_prompt_length", "full_input_length") if key in row]
            if not candidates or len(set(candidates)) != 1:
                raise ValueError("Length rows need a single unambiguous complete pre-clipping CFull input length")
            lengths[sid] = candidates[0]
        if set(lengths) != wanted:
            raise ValueError("Supplied complete-input lengths omit original population targets")
        return lengths, {"basis": "supplied_actual_tokenizer_complete_input_lengths", "sha256": sha256(Path(supplied))}
    path = input_path(root, "llada")
    saved = indexed_rows(path, wanted)
    lengths = {sid: row["untrimmed_prompt_length"] for sid, row in saved.items() if "untrimmed_prompt_length" in row}
    missing = wanted - set(lengths)
    metadata = {"basis": "actual_LLaDA_saved_untrimmed_prompt_lengths", "saved_input_sha256": sha256(path),
                "missing_recomputed_with_actual_tokenizer": len(missing)}
    if missing:
        # Exact tokenizer computation only, no character-length or context-sum estimate.
        from dataset_preparation.tokenize import encode_prompt, render_training_prompt, load_actual_tokenizer
        args = argparse.Namespace(tokenizer=str(root / "tokenizers/llada"), tokenizer_revision=None,
                                  trust_remote_code=False, backbone="dllm")
        tokenizer, tokenizer_metadata = load_actual_tokenizer(args)
        for target in targets:
            if target["sample_id"] in missing:
                prompt = render_training_prompt(target["statement"], target["full_context"])
                lengths[target["sample_id"]] = len(encode_prompt(tokenizer, prompt, "dllm"))
        metadata["tokenizer"] = tokenizer_metadata
    return lengths, metadata


def paired_group_effects(targets: list[dict], groups: dict, args) -> dict | None:
    files = [getattr(args, f"predictions_{view}", None) for view in ("env", "math", "full")]
    if not any(files):
        return None
    if not all(files):
        raise ValueError("Supply all three CEnv, CMath, CFull verified prediction files")
    wanted = {row["sample_id"] for row in targets}
    predictions = {name: read_attempts(Path(path), wanted, 32) for name, path in zip(("env", "math", "full"), files)}
    result = {}
    for name, ids in groups.items():
        subset_ids = set(ids)
        subset = [target for target in targets if target["sample_id"] in subset_ids]
        if not subset:
            result[name] = {"targets": 0, "status": "empty_group"}
            continue
        filtered = {view: {sid: rows for sid, rows in mapping.items() if sid in subset_ids}
                    for view, mapping in predictions.items()}
        result[name] = {
            "MC_given_EC": aggregate_paired(subset, filtered["env"], filtered["full"], draws=args.bootstrap_draws, seed=args.bootstrap_seed),
            "EC_given_MC": aggregate_paired(subset, filtered["math"], filtered["full"], draws=args.bootstrap_draws, seed=args.bootstrap_seed),
        }
    return result


def run_coverage(args) -> None:
    root, output = Path(args.dataset_root), Path(args.output_dir)
    targets = load_population(root, Path(args.population_ids) if args.population_ids else None, args.expected_population)
    wanted = {row["sample_id"] for row in targets}
    kpath = kernel_path(root, args.kernel_labels)
    labels = indexed_rows(kpath, wanted)
    fpath = Path(args.final_inputs) if args.final_inputs else input_path(root, args.backbone)
    final_inputs = indexed_rows(fpath, wanted)
    catalogs = SourceCatalogs(root)
    rows, groups = [], {group: [] for group in ("zero", "partial", "complete", "no_local_dependencies", "unresolved")}
    from .evidence_analysis import target_coverage
    augmented = []
    for target in targets:
        deps = mapped_dependencies(target, labels.get(target["sample_id"]), catalogs(target["source_file"]))
        known, visible, reason = final_visibility(final_inputs.get(target["sample_id"]), target)
        row = coverage_row(target, deps, known, visible, reason)
        rows.append(row)
        group, _ = target_coverage(row)
        groups[group].append(target["sample_id"])
        augmented.append({**target, **({"visible_mc_command_uids": visible} if known else {})})
    write_jsonl(output / "D3_target_coverage.rows.jsonl", rows)
    write_json(output / "D3_target_coverage.summary.json", {**coverage_summary(rows), "backbone": args.backbone,
        "success_denominator": len(targets), "kernel_labels_sha256": sha256(kpath),
        "final_input_file_sha256": sha256(fpath), "groups": groups,
        "group_counts": {group: len(ids) for group, ids in groups.items()},
        "target_type_dependency_set_independent_of_selected_MC_and_pool": True})
    effects = paired_group_effects(augmented, groups, args)
    if effects is not None:
        write_json(output / "D3_paired_context_effects.json", effects)


def run_selector(args) -> None:
    root, output = Path(args.dataset_root), Path(args.output_dir)
    ppath = Path(args.pairs) if args.pairs else root / f"tokenized/{args.backbone}/selector/input_1024/pairs.jsonl"
    pairs = indexed_rows(ppath)
    if len(pairs) != args.expected_population:
        raise ValueError("Selector success denominator must retain every frozen target")
    wanted = set(pairs)
    targets = indexed_rows(root / "benchmark/test.jsonl", wanted)
    if set(targets) != wanted:
        raise ValueError("Frozen selector target is missing from the original test split")
    kpath = kernel_path(root, args.kernel_labels)
    labels = indexed_rows(kpath, wanted)
    catalogs, rows, pools = SourceCatalogs(root), [], Counter()
    policy_rows = {"baseline": [], "dependency_prioritized": []}
    extra_rows = {"proof": {"baseline": [], "dependency_prioritized": []},
                  "projection": {"baseline": [], "dependency_prioritized": []}}
    changes = Counter()
    for sid, pair in pairs.items():
        target = targets[sid]
        catalog = catalogs(target["source_file"])
        kernel = labels.get(sid)
        deps = mapped_dependencies(target, kernel, catalog)
        pool_ids = {row["command_id"] for row in pair["candidate_pool"]}
        pools[pool_coverage(deps, pool_ids)] += 1
        row = {"sample_id": sid, "target_type_dependencies": deps,
               "pool_coverage": pool_coverage(deps, pool_ids), "policies": {}, "additional_proof": {}}
        for policy in policy_rows:
            packed = pair[policy]
            known = packed.get("valid_input") is True and isinstance(packed.get("visible_command_uids"), list)
            visible = sorted(set(packed.get("visible_command_uids", []))) if known else []
            policy_rows[policy].append(coverage_row(target, deps, known, visible,
                                                  None if known else "invalid_or_unknown_final_selector_input"))
            row["policies"][policy] = policy_rows[policy][-1]
        for mode in ("proof", "projection"):
            extra = mapped_dependencies(target, kernel, catalog, mode=mode)
            row["additional_proof"][mode] = extra
            from .evidence_analysis import target_coverage
            values = []
            for policy in policy_rows:
                current = policy_rows[policy][-1]
                extra_row = coverage_row(target, extra, current["final_visibility_known"], current["visible_mc_command_uids"])
                extra_rows[mode][policy].append(extra_row)
                _, value = target_coverage(extra_row)
                values.append(value)
            if all(value is not None for value in values):
                changes[f"{mode}_auditable_paired_targets"] += 1
                changes[f"{mode}_coverage_decreased"] += values[1] < values[0]
        rows.append(row)
    write_jsonl(output / "D5_selector_dependency_audit.rows.jsonl", rows)
    write_json(output / "D5_selector_dependency_audit.summary.json", {
        "backbone": args.backbone, "all_targets_success_denominator": len(pairs),
        "target_coverage": {policy: coverage_summary(values) for policy, values in policy_rows.items()},
        "pool_coverage_counts_independent_of_final_packing": dict(pools),
        "pool_auditable_nonempty_dependency_denominator": sum(pools[key] for key in ("all", "some", "none")),
        "posthoc_additional_reference_proof_coverage": {
            mode: {policy: coverage_summary(values) for policy, values in policies.items()}
            for mode, policies in extra_rows.items()},
        "posthoc_paired_changes": dict(changes),
        "proof_resources_read_for_posthoc_audit_only": True,
        "selector_or_generated_inputs_changed": False,
        "projection_augmented_results_are_separately_labeled_recorded_variant": True,
        "pairs_sha256": sha256(ppath), "kernel_labels_sha256": sha256(kpath),
    })
    if args.predictions_baseline or args.predictions_priority:
        if not args.predictions_baseline or not args.predictions_priority:
            raise ValueError("Supply both verified selector prediction files")
        left = read_attempts(Path(args.predictions_baseline), wanted, 32)
        right = read_attempts(Path(args.predictions_priority), wanted, 32)
        value = aggregate_paired(list(targets.values()), left, right, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
        write_json(output / "D5_selector_verified_success.json", value)


def run_groups(args) -> None:
    root, output = Path(args.dataset_root), Path(args.output_dir)
    targets = load_population(root, Path(args.population_ids) if args.population_ids else None, args.expected_population)
    lengths, provenance = real_input_lengths(root, targets, args.input_lengths)
    quartiles = make_length_groups(targets, lengths)
    annotations = indexed_rows(Path(args.mixed_annotations)) if args.mixed_annotations else None
    mixed = recorded_mixed_groups(set(lengths), annotations)
    write_json(output / "C4_shared_groups.json", {"original_both_population": len(targets),
        "length_basis": "LLaDA_complete_CFull_input_before_clipping_including_statement_special_tokens_and_final_LF",
        "length_provenance": provenance, "quartiles": quartiles,
        "quartile_sizes": {key: len(value) for key, value in quartiles.items()},
        "quartile_tie_breaker": "sample_id", "same_groups_for_both_backbones": True,
        "mixed_role": mixed, "recorded_paper_group_counts_forced": False})
    write_jsonl(output / "C4_preclipping_complete_input_lengths.jsonl", [
        {"sample_id": sid, "preclipping_full_input_length": length, "tokenizer": "llada"}
        for sid, length in sorted(lengths.items())])
    groups = {**quartiles, **({f"mixed_{key}": ids for key, ids in mixed["groups"].items()} if mixed["groups"] else {})}
    effects = paired_group_effects(targets, groups, args)
    if effects is not None:
        write_json(output / "C4_verified_context_effects.json", effects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("coverage", "selector", "groups"):
        child = sub.add_parser(command)
        child.add_argument("--dataset-root", required=True)
        child.add_argument("--output-dir", required=True)
        child.add_argument("--expected-population", type=int, default=2000 if command == "selector" else 11064)
        child.add_argument("--bootstrap-draws", type=int, default=10000)
        child.add_argument("--bootstrap-seed", type=int, default=42)
        if command != "groups":
            child.add_argument("--backbone", choices=("llada", "llama31"), required=True)
            child.add_argument("--kernel-labels")
        if command != "selector":
            child.add_argument("--population-ids")
            for view in ("env", "math", "full"):
                child.add_argument(f"--predictions-{view}")
    sub.choices["coverage"].add_argument("--final-inputs")
    sub.choices["coverage"].set_defaults(func=run_coverage)
    selector = sub.choices["selector"]
    selector.add_argument("--pairs")
    selector.add_argument("--predictions-baseline")
    selector.add_argument("--predictions-priority")
    selector.set_defaults(func=run_selector)
    groups = sub.choices["groups"]
    groups.add_argument("--input-lengths")
    groups.add_argument("--mixed-annotations")
    groups.set_defaults(func=run_groups)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
