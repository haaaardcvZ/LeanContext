"""Build source-grounded matched-MC replacement controls.

Implements the S6.2 constraints with deterministic sampling and matching.
Target proof text is excluded from every generation prompt. Observed proof dependencies
are used offline to exclude candidates, so this experiment is NOT proof-blind.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from .tokenize import encode_prompt, iter_jsonl, source_order, write_json, write_row

MODELS = {"llada": "dllm", "llama31": "llama31"}
CONDITIONS = ("Env", "Selected", "Control")
REVISION = "source-grounded-matched-replacement-v1"
TEMPLATE = "original-EC-and-selected-MC-slots-plain-source-one-newline-v1"
TARGET_FIELDS = (
    "sample_id", "target_id", "split", "proof_family", "statement",
    "source_file", "source_module", "target_command_id", "target_command_uid",
    "command_id", "target_kind", "original_name", "target_source_span",
    "source_revision", "verification_location",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def new_extension_root(dataset_root: Path, output_root: Path) -> Path:
    """Keep the submitted dataset read-only; refuse nonempty existing outputs."""
    source, destination = dataset_root.resolve(), output_root.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("--output-root must be outside --dataset-root")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty extension: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def kind(command: dict[str, Any]) -> str:
    value = str(command.get("effective_kind", command.get("kind", ""))).rsplit(".", 1)[-1]
    if value == "instance" and re.match(r"\s*local\s+instance\b", command.get("text", "")):
        return "local_instance"
    return value


def in_length_band(selected: int, candidate: int) -> bool:
    return selected > 0 and 9 * selected <= 10 * candidate <= 11 * selected


def sampled_ids(population_ids: list[str], n: int, seed: int) -> list[str]:
    if len(set(population_ids)) != len(population_ids):
        raise ValueError("duplicate population IDs")
    return random.Random(seed).sample(sorted(population_ids), n)


def minimum_assignment(edges: list[dict[str, int]]) -> dict[str, Any] | None:
    """Pure-Python rectangular Hungarian assignment; deterministic ID traversal.

    Costs are absolute full-declaration token-length differences. A forbidden
    edge cannot be returned, and alternative commands are used without
    replacement within a target. Tie-breaking is deterministic ascending-column
    traversal, NOT a claim of globally lexicographically smallest assignments.
    """
    n = len(edges)
    if not n or any(not row for row in edges):
        return None
    columns = sorted({key for row in edges for key in row})
    m = len(columns)
    if m < n:
        return None
    infinity = 10**15
    u, v = [0] * (n + 1), [0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0, minv, used = 0, [infinity] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], infinity, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cost = edges[i0 - 1].get(columns[j - 1], infinity)
                cur = cost - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            if delta >= infinity // 2:
                return None
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if not p[j0]:
                break
        while True:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
            if not j0:
                break
    chosen: list[str | None] = [None] * n
    for j in range(1, m + 1):
        if p[j]:
            chosen[p[j] - 1] = columns[j - 1]
    if any(cid is None or cid not in edges[i] for i, cid in enumerate(chosen)):
        return None
    return {"candidate_ids": chosen,
            "total_absolute_token_deviation": sum(edges[i][cid] for i, cid in enumerate(chosen))}


def render_prompt(statement: str, ec: list[dict[str, Any]],
                  mc: list[dict[str, Any]]) -> str:
    """Replacement texts occupy original MC slots; real source_order is untouched."""
    items = [*ec, *mc]
    items.sort(key=lambda c: (c.get("prompt_slot_source_order", c["source_order"]),
                              c.get("prompt_slot_command_uid", c["command_uid"])))
    return "\n\n".join([*(c["text"] for c in items), statement]).rstrip("\r\n") + "\n"


def prompt_record(tokenizer: Any, backbone: str, target: dict[str, Any],
                  ec: list[dict[str, Any]], mc: list[dict[str, Any]],
                  condition: str, limit: int) -> dict[str, Any]:
    text = render_prompt(target["statement"], ec, mc)
    ids = encode_prompt(tokenizer, text, backbone)
    return {
        "condition": condition, "prompt": text, "input_ids": ids,
        "input_length": len(ids), "prompt_length": len(ids),
        "input_limit": limit, "output_limit": 1024,
        "output_limit_includes_terminal_EOS": True,
        "valid_input": len(ids) <= limit,
        "visible_mc_command_ids": [c["command_id"] for c in mc],
        "visible_mc_command_uids": [c["command_uid"] for c in mc],
        "visible_ec_command_ids": [c["command_id"] for c in ec],
        "visible_ec_command_uids": [c["command_uid"] for c in ec],
        "MC_count": len(mc), "EC_count": len(ec), "template_version": TEMPLATE,
        "reference_proof_in_generator_input": False,
        "input_ids_contain_prompt_only": True,
        "serialized_environment_Lean_elaborated": False,
    }


class RealLengths:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if text not in self.cache:
            self.cache[text] = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return self.cache[text]


def match_one(target: dict[str, Any], commands: list[dict[str, Any]],
              profiles: dict[str, dict[str, Any]], evidence: dict[str, Any],
              tokenizer: Any, lengths: RealLengths, backbone: str,
              input_limit: int) -> dict[str, Any]:
    from .replacement_dependencies import candidate_exclusion_reason
    selected = source_order(target["mathematical_context"])
    ec = source_order(target["environment_context"])
    index = {c["command_id"]: c for c in commands}
    exclusion_counts: Counter = Counter()
    base = {"eligible": False, "reason": None, "stage_counts": [],
            "original_MC_count": len(selected), "original_EC_count": len(ec)}
    if not evidence.get("availability_complete", False):
        return {**base, "reason": "dependency_mapping_incomplete_or_unavailable",
                "dependency_rejection_reasons": evidence.get("rejection_reasons", ["no_evidence"])}
    if (evidence.get("sample_id"), evidence.get("source_file"), evidence.get("target_command_id")) != (
            target["sample_id"], target["source_file"], target["target_command_id"]):
        raise ValueError("dependency evidence refers to a different source target")
    selected_prompt = prompt_record(tokenizer, backbone, target, ec, selected, "Selected", input_limit)
    env_prompt = prompt_record(tokenizer, backbone, target, ec, [], "Env", input_limit)
    base["selected_input_length"] = selected_prompt["input_length"]
    base["env_input_length"] = env_prompt["input_length"]
    if not selected_prompt["valid_input"] or not env_prompt["valid_input"]:
        return {**base, "reason": "unchanged_selected_or_Env_prompt_exceeds_input_limit"}
    excluded_ids = {c["command_id"] for c in selected}
    excluded_ids.update(c["command_id"] for c in ec)
    target_start = target["target_source_span"]["start_byte"]
    pool = []
    for c in commands:
        if c["source_span"]["end_byte"] > target_start or c["source_span"]["start_byte"] >= target_start:
            exclusion_counts["not_strictly_preceding_target"] += 1
            continue
        if c["command_id"] in excluded_ids or c["command_id"] == target["target_command_id"]:
            exclusion_counts["original_selected_or_EC_or_target_command"] += 1
            continue
        if kind(c) not in {kind(s) for s in selected} or not c.get("text", "").strip():
            exclusion_counts["not_required_exact_MC_kind_or_empty"] += 1
            continue
        reason = candidate_exclusion_reason(c, evidence)
        if reason:
            exclusion_counts[reason] += 1
            continue
        pool.append(c)
    base["candidate_pool_count"] = len(pool)
    base["candidate_exclusion_counts"] = dict(exclusion_counts)
    edges = []
    token_lengths = []
    for s in selected:
        cid = s["command_id"]
        if cid not in index or index[cid]["text"] != s["text"]:
            raise ValueError(f"{target['sample_id']}: selected source text mismatch {cid}")
        profile = profiles[cid]
        if profile.get("status") != "syntactic_valid":
            return {**base, "reason": "selected_scope_unresolved", "command_id": cid}
        length = lengths(s["text"])
        same_kind = [c for c in pool if kind(c) == kind(s)]
        compatible = [c for c in same_kind
                      if profiles[c["command_id"]].get("status") == "syntactic_valid"
                      and profiles[c["command_id"]]["fingerprint"] == profile["fingerprint"]]
        band = [c for c in compatible if in_length_band(length, lengths(c["text"]))]
        base["stage_counts"].append({"selected_command_id": cid, "exact_kind_candidates": len(same_kind),
                                      "scope_compatible_candidates": len(compatible),
                                      "length_compatible_candidates": len(band), "selected_tokens": length})
        if not same_kind:
            return {**base, "reason": "no_exact_kind_candidate", "command_id": cid}
        if not compatible:
            return {**base, "reason": "no_conservatively_compatible_scope_candidate", "command_id": cid}
        if not band:
            return {**base, "reason": "no_candidate_in_per_declaration_token_length_band", "command_id": cid}
        edges.append({c["command_id"]: abs(lengths(c["text"]) - length) for c in band})
        token_lengths.append(length)
    assignment = minimum_assignment(edges)
    if assignment is None:
        return {**base, "reason": "no_injective_one_to_one_assignment"}
    replacement, pairs = [], []
    for s, candidate_id, selected_length in zip(selected, assignment["candidate_ids"], token_lengths):
        c = index[candidate_id]
        item = {**c, "role": "MC", "prompt_slot_command_id": s["command_id"],
                "prompt_slot_command_uid": s["command_uid"],
                "prompt_slot_source_order": s["source_order"]}
        replacement.append(item)
        pairs.append({
            "selected_command_id": s["command_id"], "selected_command_uid": s["command_uid"],
            "replacement_command_id": c["command_id"], "replacement_command_uid": c["command_uid"],
            "selected_kind": kind(s), "replacement_kind": kind(c),
            "selected_tokens": selected_length, "replacement_tokens": lengths(c["text"]),
            "length_ratio": lengths(c["text"]) / selected_length,
            "scope_fingerprint": profiles[s["command_id"]]["fingerprint"],
            "scope_check": "exact_conservative_syntactic_profile_not_Lean_elaboration",
            "replacement_source_order": c["source_order"], "prompt_slot_source_order": s["source_order"],
        })
    selected_mc_length = lengths("\n\n".join(c["text"] for c in selected))
    control_mc_length = lengths("\n\n".join(c["text"] for c in replacement))
    base.update(selected_MC_tokens=selected_mc_length, control_MC_tokens=control_mc_length,
                assignment_cost=assignment["total_absolute_token_deviation"])
    if not in_length_band(selected_mc_length, control_mc_length):
        return {**base, "reason": "deterministic_assignment_violates_total_MC_token_length_band"}
    control_prompt = prompt_record(tokenizer, backbone, target, ec, replacement, "Control", input_limit)
    base["control_input_length"] = control_prompt["input_length"]
    if not control_prompt["valid_input"]:
        return {**base, "reason": "deterministic_assignment_control_prompt_exceeds_input_limit"}
    return {**base, "eligible": True, "reason": None, "declaration_pairs": pairs,
            "replacement_MC": replacement,
            "conditions": {"Env": env_prompt, "Selected": selected_prompt, "Control": control_prompt}}


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": REVISION,
        "source": "S6.2_matched_replacement_protocol",
        "recorded_rules": {
            "population": "original_MC_and_EC_both_nonempty_test_targets",
            "sample_size": args.sample_size, "seed": args.seed,
            "requested_final_common_targets": args.final_size,
            "candidate_source": "same_original_source_file_strictly_preceding_target_full_MC_commands",
            "exclude_observed_dependencies": "target_elaborated_type_union_reference_proof_dependencies",
            "matching_kind": "exact_effective_declaration_kind",
            "length_tolerance_percent": 10,
            "tokenizers": "local_verified_actual_LLaDA_and_Llama3.1_tokenizers",
            "same_target_EC_and_checkpoint_within_paired_comparison": True,
            "input_limit": args.input_limit, "output_limit": 1024,
            "planned_attempts_per_target_per_condition": 32,
            "planned_checkpoint_training_views": ["CMath", "CEnv", "CFull"],
        },
        "implementation_defaults": {
            "sampling": "random.Random(seed).sample(sorted_population_IDs,sample_size)",
            "complete_dependency_mapping_required": True,
            "original_selected_MC_also_excluded_from_control": True,
            "compiler_identity_of_candidates_required": True,
            "scope": "exact_conservative_active_source_scope_profile_including_one_command_wrappers",
            "scope_is_not_Lean_binding_or_elaboration_certificate": True,
            "matching": "one_to_one_without_replacement_within_target",
            "objective": "minimum_sum_absolute_full_declaration_token_length_deviation",
            "tie_break": "deterministic_ascending_command_ID_column_traversal",
            "length_constraint_unit": "each_declaration_and_whole_MC_joined_with_double_newlines",
            "MC_total_length_band_percent": 10,
            "prompt_placement": "replacement_content_in_original_selected_MC_slots_EC_positions_unchanged",
            "MC_EC_or_statement_truncation": False,
            "reference_proof_length_used_for_inference_eligibility": False,
            "models_may_use_different_replacement_command_IDs": True,
            "selection_of_final_IDs": "first_requested_size_in_frozen_sample_order_among_jointly_eligible",
            "if_insufficient": "save_all_actual_jointly_eligible_without_backfill_or_rule_relaxation",
            "no_rematching_after_minimum_assignment_serialized_length_failure": True,
            "global_budget_feasibility_exhaustively_searched": False,
            "selector_subsets": "separate_manifests_not_forced_disjoint",
        },
        "not_proof_blind": True, "dependencies_used_only_for_offline_control_exclusions": True,
        "mathematical_irrelevance_guaranteed": False,
        "model_generation_performed": False, "pass_rate_results_included": False,
        "candidate_Lean_verification_performed": False,
    }


def main() -> None:
    from .replacement_dependencies import load_evidence
    from .replacement_scope import scope_profiles
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, help="Defaults to DATASET/private_posthoc_audit/kernel_inputs")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New independent extension directory, outside the source dataset")
    parser.add_argument("--sample-size", type=int, default=2500)
    parser.add_argument("--final-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input-limit", type=int, default=1024)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    if (args.sample_size, args.final_size, args.seed, args.input_limit) != (2500, 2000, 2026, 1024):
        raise ValueError("This S6.2 implementation requires sample=2500, final=2000, seed=2026, input=1024")
    root = args.dataset_root.resolve()
    args.audit_root = (args.audit_root or root / "private_posthoc_audit/kernel_inputs").resolve()
    required_inputs = [root / "benchmark/test.jsonl", root / "command_mapping/source_snapshot.json",
                       root / "command_mapping/revision_verification.json",
                       *(args.audit_root / name for name in ("semantic_mc_labels.jsonl", "declaration_dependencies.jsonl", "declaration_ownership.jsonl")),
                       *(root / "tokenizers" / model / "tokenizer.json" for model in MODELS)]
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    extension = new_extension_root(root, args.output_root)
    out = extension / "matched_replacement"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "build_status.json", {"status": "in_progress", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    population = {}
    for row in iter_jsonl(root / "benchmark/test.jsonl"):
        if row["mathematical_context"] and row["environment_context"]:
            population[row["sample_id"]] = row
    drawn = sampled_ids(list(population), args.sample_size, args.seed)
    selection = {"schema_version": "seed-2026-matched-replacement-sampling-v1",
                 "sampling_seed": args.seed,
                 "sample_size": args.sample_size, "population_size": len(population),
                 "population_sha256": stable_hash(sorted(population)),
                 "sampled_target_ids": drawn, "sample_order_is_frozen": True,
                 "proof_results_used_for_sampling": False,
                 "selection": "random.Random(seed).sample(sorted_population_IDs,sample_size)",
                 "source_test_sha256": sha256(root / "benchmark/test.jsonl")}
    write_json(out / "sampled_manifest.json", selection)
    rules = protocol(args)
    write_json(out / "matching_protocol.json", rules)
    with (out / "population_ids.jsonl").open("w", encoding="utf-8") as f:
        for sid in sorted(population):
            row = population[sid]
            write_row(f, {k: row[k] for k in ("sample_id", "source_file", "target_command_id", "proof_family")})
    evidence = load_evidence(args.audit_root, set(drawn))
    audit_path = extension / "private_posthoc_audit/matched_replacement_dependency_exclusions.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        for sid in drawn:
            write_row(f, {"sample_id": sid, **evidence.get(sid, {"availability_complete": False, "rejection_reasons": ["missing_compiler_evidence"]})})
    tokenizers = {model: AutoTokenizer.from_pretrained(str(root / "tokenizers" / model),
                 local_files_only=True, trust_remote_code=False, use_fast=True) for model in MODELS}
    lengths = {model: RealLengths(tokenizer) for model, tokenizer in tokenizers.items()}
    source_cache = {}
    results = {}
    counts = {model: Counter() for model in MODELS}
    with (out / "eligibility.jsonl").open("w", encoding="utf-8") as f:
        for step, sid in enumerate(drawn, 1):
            target = population[sid]
            source_file = target["source_file"]
            if source_file not in source_cache:
                commands = list(iter_jsonl(root / "command_mapping/source_commands" / Path(source_file).with_suffix(".jsonl")))
                source_cache[source_file] = (commands, scope_profiles(commands))
            commands, profiles = source_cache[source_file]
            local = {}
            for model, backbone in MODELS.items():
                result = match_one(target, commands, profiles, evidence.get(sid, {}),
                                   tokenizers[model], lengths[model], backbone, args.input_limit)
                counts[model]["eligible" if result["eligible"] else result["reason"]] += 1
                local[model] = result
            results[sid] = local
            write_row(f, {"sample_id": sid, "sample_rank": step - 1,
                "source_file": source_file, "proof_family": target["proof_family"],
                "both_models_eligible": all(result["eligible"] for result in local.values()),
                "models": {model: {k: v for k, v in value.items() if k not in {"conditions", "replacement_MC", "declaration_pairs"}} for model, value in local.items()}})
            if step % 100 == 0:
                print(json.dumps({"stage": "matching_actual_sources", "processed": step,
                                  "models": {model: dict(counter) for model, counter in counts.items()}}, ensure_ascii=False), flush=True)
    common = [sid for sid in drawn if all(r["eligible"] for r in results[sid].values())]
    final = common[:args.final_size]
    selector_ids = json.loads((root / "selector/frozen_test_manifest.json").read_text())["target_ids"]
    manifest = {"schema_version": REVISION,
        "requested_targets": args.final_size, "sampled_targets": len(drawn),
        "common_eligible_targets": len(common), "actual_final_targets": len(final),
        "requested_size_reached": len(final) == args.final_size,
        "target_ids": final, "sampling_seed": args.seed,
        "model_failure_counts": {model: dict(counter) for model, counter in counts.items()},
        "selector_manifest_separate": True, "selector_overlap_count": len(set(final) & set(selector_ids)),
        "no_generation_outcomes_used": True, "no_backfill_or_matching_rule_relaxation": True,
        "eligibility_not_Lean_elaboration_certificate": True,
        "status": "complete_requested_size" if len(final) == args.final_size else "complete_partial_requested_size_not_reached",
        "model_tokenizer_replacements_may_differ": True,
        "families": dict(Counter(population[sid]["proof_family"] for sid in final)),
    }
    write_json(out / "frozen_test_manifest.json", manifest)
    with (out / "common_eligible_ids.jsonl").open("w", encoding="utf-8") as f:
        for sid in common:
            write_row(f, {"sample_id": sid, "included_in_final": sid in set(final)})
    targets = (out / "targets.jsonl").open("w", encoding="utf-8")
    raw_pairs = (out / "pairs.jsonl").open("w", encoding="utf-8")
    model_handles = {}
    for model in MODELS:
        base = extension / f"tokenized/{model}/matched_replacement/input_{args.input_limit}"
        base.mkdir(parents=True, exist_ok=True)
        model_handles[model] = {"pairs": (base / "pairs.jsonl").open("w", encoding="utf-8")}
        for condition in CONDITIONS:
            folder = base / "views" / condition
            folder.mkdir(parents=True, exist_ok=True)
            model_handles[model][condition] = (folder / "test.jsonl").open("w", encoding="utf-8")
    metrics = {model: {"targets": len(final), "input_limit": args.input_limit, "output_limit": 1024,
        "prompt_length_max": {condition: 0 for condition in CONDITIONS},
        "replacement_differs_from_other_tokenizer_targets": 0} for model in MODELS}
    try:
        for sid in final:
            raw = population[sid]
            target = {k: raw[k] for k in TARGET_FIELDS}
            write_row(targets, target)
            common_pair = {**target, "selected_MC": source_order(raw["mathematical_context"]),
                           "shared_EC": source_order(raw["environment_context"]),
                           "protocol_revision": REVISION, "template_version": TEMPLATE,
                           "current_target_reference_proof_in_input": False,
                           "reference_dependencies_used_for_offline_exclusions": True}
            write_row(raw_pairs, {**common_pair, "models": {model: {k: v for k, v in results[sid][model].items() if k != "conditions"} for model in MODELS}})
            model_controls = [tuple(c["replacement_command_id"] for c in results[sid][model]["declaration_pairs"]) for model in MODELS]
            for model in MODELS:
                result = results[sid][model]
                encoded = {**target, "protocol_revision": REVISION, "template_version": TEMPLATE,
                           "shared_EC_ids": [c["command_id"] for c in common_pair["shared_EC"]],
                           "declaration_pairs": result["declaration_pairs"],
                           "conditions": result["conditions"]}
                write_row(model_handles[model]["pairs"], encoded)
                metrics[model]["replacement_differs_from_other_tokenizer_targets"] += model_controls[0] != model_controls[1]
                for condition in CONDITIONS:
                    view = {**target, **result["conditions"][condition], "backbone": model,
                            "protocol_revision": REVISION}
                    write_row(model_handles[model][condition], view)
                    metrics[model]["prompt_length_max"][condition] = max(metrics[model]["prompt_length_max"][condition], view["input_length"])
    finally:
        targets.close(); raw_pairs.close()
        for handles in model_handles.values():
            for handle in handles.values():
                handle.close()
    for model in MODELS:
        write_json(extension / f"tokenized/{model}/matched_replacement/input_{args.input_limit}/tokenization_metrics.json", metrics[model])
    write_json(out / "experiment_plan.json", {
        "status": "planned_not_run", "targets": len(final),
        "same_frozen_targets_in_all_comparisons": True,
        "models": list(MODELS), "checkpoint_training_views": ["CMath", "CEnv", "CFull"],
        "checkpoint_paths": None, "input_conditions": list(CONDITIONS),
        "candidate_attempts": 32, "input_limit": args.input_limit, "output_limit": 1024,
        "external_retrieval_or_intermediate_Lean_feedback": False,
        "primary_metric": "Pass@32_Selected_minus_Control_percentage_points",
        "secondary_metric": "Pass@32_Selected_minus_Env_and_Control_minus_Env",
        "training_MC_dependence": "compare_Selected_minus_Control_across_Math_Env_Full_trained_checkpoints",
        "verification": "original_source_location_unchanged_target_pinned_snapshot_no_sorry_admit",
        "decoding_settings_and_verification_limits": "must_be_fixed_before_generations_not_filled_by_this_dataset_build",
        "no_model_generation_or_Lean_candidate_verification_performed": True})
    artifacts = {}
    paths = list(out.rglob("*"))
    for model in MODELS:
        paths.extend((extension / f"tokenized/{model}/matched_replacement").rglob("*"))
    paths.append(audit_path)
    for path in sorted(paths):
        if path.is_file() and path.name not in {"manifest.json", "build_status.json"}:
            artifacts[str(path.relative_to(extension))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    release = {**manifest, "artifacts": artifacts,
               "input_dataset_root": str(root), "independent_extension_root": str(extension),
               "submitted_dataset_modified": False,
               "input_source_test_sha256": sha256(root / "benchmark/test.jsonl"),
               "source_snapshot_verification": "command_mapping/revision_verification.json",
               "processing_script_sha256": {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("replacement_scope.py"), Path(__file__).with_name("replacement_dependencies.py")]},
               "independent_validation_status": "pending",
               "model_generation_performed": False, "pass_rate_results_included": False}
    write_json(out / "manifest.json", release)
    write_json(out / "build_status.json", {"status": manifest["status"], "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                                           "requested_targets": args.final_size, "actual_targets": len(final),
                                           "independent_validation_status": "pending"})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
