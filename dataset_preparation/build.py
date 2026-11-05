#!/usr/bin/env python3
"""Create a new, source-grounded raw release without modifying old datasets.

Training uses the existing reference-assisted contexts and original splits.
Selector candidates are independently reconstructed from source prefixes.
Actual tokenizer lengths and model outcomes are never fabricated here.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .provenance import (
    SourceRepository, command_record, enrich_target, iter_jsonl,
    load_private_rows, load_public_rows, load_selections,
)
from .selector import build_pair, choose_manifest


SPLITS = ("train", "validation", "test")
VIEWS = ("C0", "CMath", "CEnv", "CFull")


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def emit(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def progress(stage: str, **details: Any) -> None:
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def type_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Whitelist type-only compiler evidence; no gold/value features escape."""
    dependency = row.get("type_dependency")
    if dependency is None:
        status = "unresolved_target"
        dependency = {}
    elif dependency.get("unmapped_same_file_constants") or dependency.get("nonpreceding_same_file_constants"):
        status = "incomplete_mapping"
    elif row.get("expanded_type_auxiliaries"):
        # This build implements direct target-type references, not an expansion.
        status = "incomplete_mapping"
    elif dependency.get("command_ids"):
        status = "complete_nonempty"
    else:
        status = "no_eligible_local_dependencies"
    result = {
        "sample_id": row["sample_id"],
        "source_file": row["source_file"],
        "source_module": row["source_module"],
        "target_command_id": row["target_command_id"],
        "target_name": row.get("target_name"),
        "target_identity_status": row.get("target_identity_status"),
        "status": status,
        "method": "direct_elaborated_target_type_kernel_constants_mapped_to_preceding_same_file_MC_commands",
        "command_ids": list(dependency.get("command_ids", [])),
        "constant_evidence": dependency.get("constant_evidence", {}),
        "unmapped_same_file_constants": dependency.get("unmapped_same_file_constants", []),
        "nonpreceding_same_file_constants": dependency.get("nonpreceding_same_file_constants", []),
        "same_target_command_constants": dependency.get("same_target_command_constants", []),
        "other_local_commands": dependency.get("other_local_commands", []),
        "direct_type_constants": row.get("outer_type_constants", []),
        "expanded_type_auxiliaries": row.get("expanded_type_auxiliaries", []),
        "reference_proof_used": False,
        "candidate_pool_used_to_define_dependencies": False,
    }
    return result


def kernel_command_index(repo: SourceRepository, source_files: list[str], audit: Path):
    """Attribute compiler identities by real source ranges/certified ownership."""
    spans = {}
    module_to_file = {}
    source_hashes = {}
    for index, source_file in enumerate(source_files, 1):
        commands = repo.commands(source_file)
        module = source_file[:-5].replace("/", ".")
        module_to_file[module] = source_file
        spans[module] = {
            "starts": [(c["start_line"], c["start_col"]) for c in commands],
            "commands": [(c["command_id"], (c["end_line"], c["end_col"])) for c in commands],
        }
        data = repo.source_bytes(source_file)
        source_hashes[source_file] = hashlib.sha256(data).hexdigest()
        cached = audit / "mathlib-semantic" / source_file
        if not cached.is_file() or cached.read_bytes() != data:
            raise ValueError(f"Compiler-audit source snapshot differs: {source_file}")
        if index % 1000 == 0:
            progress("source_bytes_validated", files=index)

    def locate(module, position):
        if module not in spans or not position:
            return None
        where = (position["start_line"], position["start_column"])
        data = spans[module]
        index = bisect.bisect_right(data["starts"], where) - 1
        if index >= 0 and where < data["commands"][index][1]:
            return data["commands"][index][0]
        return None

    identities = {}
    for index, row in enumerate(iter_jsonl(audit / "declaration_dependencies.jsonl"), 1):
        module = row["module"]
        if module not in module_to_file:
            continue
        command_id = locate(module, row.get("selection_range"))
        basis = "kernel_selection_range"
        if command_id is None:
            command_id = locate(module, row.get("range"))
            basis = "kernel_declaration_range" if command_id is not None else "unmapped"
        identities[row["name"]] = [module, command_id, basis, row["kind"]]
        if index % 200000 == 0:
            progress("compiler_identity_index", declarations_read=index)
    ownership = list(iter_jsonl(audit / "declaration_ownership.jsonl"))
    for _ in range(4):
        changed = 0
        for row in ownership:
            metadata = identities.get(row["name"])
            if metadata is None or metadata[1] is not None:
                continue
            owners = [identities.get(name) for name in row["owners"]]
            if owners and all(owner and owner[0] == metadata[0] and owner[1] is not None for owner in owners):
                ids = {owner[1] for owner in owners}
                if len(ids) == 1:
                    metadata[1] = next(iter(ids))
                    metadata[2] = row["basis"]
                    changed += 1
        if not changed:
            break
    by_command = defaultdict(list)
    for name, (module, command_id, basis, kind) in identities.items():
        if command_id is not None:
            by_command[(module_to_file[module], command_id)].append({
                "name": name, "kind": kind, "basis": basis,
            })
    for values in by_command.values():
        values.sort(key=lambda value: value["name"])
    return by_command, source_hashes


def selector_target(row: dict[str, Any]) -> dict[str, Any]:
    """Explicit whitelist: reference proof never enters selector input tables."""
    fields = ("sample_id", "target_id", "split", "proof_family", "statement",
              "command_id", "target_command_id", "target_command_uid", "source_file",
              "source_module", "target_kind", "original_name", "target_source_span")
    return {field: row[field] for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="Original extraction workspace containing the source shards")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-root", type=Path, required=True,
                        help="Kernel-dependency audit input directory")
    parser.add_argument("--selector-size", type=int, default=2000)
    parser.add_argument("--selector-seed", type=int, default=20261007)
    args = parser.parse_args()
    root = args.workspace.resolve()
    out = (args.output or root / "leancontext_dataset").resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source_public = root / "sft/context_benchmark/semantic_environment_dataset_split"
    source_private = root / "sft/current_repaired/prepared/deepseek-hybrid-repaired-with-context"
    raw_selections = root / "theorem_generate/extract_data/theorem_contexts_deepseek_hybrid.statement_context.repaired.jsonl"
    repo = SourceRepository(root, root / "theorem_generate/command_spans_jsonl",
                            root / "theorem_generate/lean_parser/mathlib4-4.14.0-rc1")
    write_json(out / "build_status.json", {"status": "in_progress", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    public = load_public_rows(source_public)
    private = load_private_rows(source_private, set(public))
    selections = load_selections(raw_selections, private)
    source_files = sorted({row["relpath"] for row in private.values()})
    progress("strict_input_join", targets=len(public), source_files=len(source_files))
    kernel_by_command, source_hashes = kernel_command_index(repo, source_files, args.audit_root)
    progress("source_catalog_ready", commands_with_compiler_identities=len(kernel_by_command))

    type_by_id = {}
    audit_by_id = {}
    for record in iter_jsonl(args.audit_root / "semantic_mc_labels.jsonl"):
        sid = record["sample_id"]
        if sid not in public:
            raise ValueError(f"Unexpected dependency target: {sid}")
        if record["source_file"] != private[sid]["relpath"] or record["target_command_id"] != private[sid]["metadata"]["target_command_id"]:
            raise ValueError(f"Dependency/source target mismatch: {sid}")
        type_by_id[sid] = type_evidence(record)
        if public[sid]["split"] in {"validation", "test"}:
            audit_by_id[sid] = {
                "sample_id": sid, "source_file": record["source_file"],
                "target_command_id": record["target_command_id"],
                "reference_proof_extra_dependencies": record.get("proof_extra_dependency"),
                "mapping_status": record.get("mc_semantic_status"),
                "method": "kernel_value_minus_type_constants_after_source_certified_same_command_auxiliary_expansion",
                "stage": "offline_descriptor_for_posthoc_audit_only_not_selector_input",
            }

    handles = {}
    counts = defaultdict(Counter)
    source_split = {}
    normalized_split = {}
    both_targets = defaultdict(list)
    command_total = 0
    with ExitStack() as stack:
        def open_output(relative):
            path = out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            return stack.enter_context(path.open("w", encoding="utf-8"))
        for split in SPLITS:
            handles[("benchmark", split)] = open_output(f"benchmark/{split}.jsonl")
            handles[("targets", split)] = open_output(f"training/targets/{split}.jsonl")
            handles[("ids", split)] = open_output(f"training/split_ids/{split}.jsonl")
            for view in VIEWS:
                handles[(view, split)] = open_output(f"training/views/{view}/{split}.jsonl")
        mapping_handle = open_output("command_mapping/targets.jsonl")
        reverse_handle = open_output("command_mapping/command_to_targets.jsonl")
        history_handle = open_output("command_mapping/original_selection_history.jsonl")
        type_handle = open_output("selector/target_type_dependencies.jsonl")
        for record in sorted(type_by_id.values(), key=lambda r: r["sample_id"]):
            emit(type_handle, record)
        grouped = defaultdict(list)
        for sid in public:
            grouped[private[sid]["relpath"]].append(sid)
        for file_index, source_file in enumerate(source_files, 1):
            commands = repo.commands(source_file)
            enriched_commands = []
            catalog_path = out / "command_mapping/source_commands" / Path(source_file).with_suffix(".jsonl")
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            with catalog_path.open("w", encoding="utf-8") as catalog:
                for order, original in enumerate(commands):
                    record = command_record(original)
                    record.update(source_order=order, kind=record["effective_kind"],
                                  name=original.get("name"), text=original["effective_raw_text"].strip(),
                                  raw_text=original["raw_text"], effective_raw_text=original["effective_raw_text"],
                                  introduced_declarations=kernel_by_command.get((source_file, original["command_id"]), []))
                    record["introduced_declarations_status"] = ("kernel_range_or_certified_ownership_attributed"
                                                               if record["introduced_declarations"] else "not_resolved_not_asserted_empty")
                    enriched_commands.append(record)
                    emit(catalog, record)
                    command_total += 1
            for sid in grouped[source_file]:
                row, mapping, history = enrich_target(public[sid], private[sid], selections[sid], repo)
                row["command_id"] = row["target_command_id"]
                row["source_revision"] = {
                    "mathlib_release": "4.14.0-rc1", "lean_toolchain": "leanprover/lean4:v4.14.0-rc1",
                    "recorded_mathlib_commit": "a7fc949f1b05c2a01e01c027fd9f480496a1253e",
                    "commit_claim_basis": "user_protocol_record; local Git history unavailable",
                    "source_file_sha256": source_hashes[source_file],
                }
                row["verification_location"] = {"source_file": source_file, **row["target_source_span"]}
                row["context_selection_regime"] = "original_reference_assisted"
                split = row["split"]
                previous = source_split.setdefault(source_file, split)
                if previous != split:
                    raise ValueError("Source file crosses splits")
                normalized = " ".join(row["statement"].split())
                previous = normalized_split.setdefault(normalized, split)
                if previous != split:
                    raise ValueError("Normalized statement crosses splits")
                mapping["command_id"] = row["command_id"]
                mapping["introduced_declarations"] = kernel_by_command.get((source_file, row["command_id"]), [])
                mapping["kernel_target_name"] = type_by_id.get(sid, {}).get("target_name")
                mapping["source_revision"] = row["source_revision"]
                emit(mapping_handle, mapping)
                emit(reverse_handle, {"source_file": source_file, "command_id": row["command_id"],
                                      "command_uid": row["target_command_uid"], "target_ids": [sid], "split": split})
                emit(history_handle, history)
                emit(handles[("benchmark", split)], row)
                target_record = {**selector_target(row), "target": row["target"],
                                 "source_revision": row["source_revision"],
                                 "verification_location": row["verification_location"]}
                emit(handles[("targets", split)], target_record)
                emit(handles[("ids", split)], {"sample_id": sid, "target_id": sid,
                                              "source_file": source_file, "command_id": row["command_id"]})
                math_ids = {item["command_id"] for item in row["mathematical_context"]}
                contexts = {
                    "C0": [], "CMath": row["mathematical_context"],
                    "CEnv": row["environment_context"], "CFull": row["full_context"],
                }
                for view, items in contexts.items():
                    context_items = [{**item, "role": "MC" if item["command_id"] in math_ids else "EC"} for item in items]
                    prompt = "\n\n".join([*(item["text"] for item in context_items), row["statement"]]).rstrip("\r\n") + "\n"
                    view_row = {**selector_target(row), "view": view, "prompt": prompt, "target": row["target"],
                                "context_items": context_items, "context_ids": [item["command_id"] for item in items],
                                "context_uids": [item["command_uid"] for item in items],
                                "template_version": "training_source_order_context_then_statement_one_newline_v1",
                                "tokenization_status": "pending_real_backbone_tokenizer"}
                    emit(handles[(view, split)], view_row)
                counts[split]["targets"] += 1
                counts[split]["MC_nonempty"] += bool(row["mathematical_context"])
                counts[split]["EC_nonempty"] += bool(row["environment_context"])
                counts[split]["both_nonempty"] += bool(row["mathematical_context"] and row["environment_context"])
                counts[split]["MC_commands"] += len(row["mathematical_context"])
                counts[split]["EC_commands"] += len(row["environment_context"])
                if row["mathematical_context"] and row["environment_context"]:
                    both_targets[split].append({**selector_target(row), "both_nonempty": True})
            if file_index % 500 == 0:
                progress("raw_training_and_mapping", files=file_index, targets=sum(c["targets"] for c in counts.values()))

    progress("raw_training_complete", counts={key: dict(value) for key, value in counts.items()})
    selector_manifest = choose_manifest(both_targets["test"], n=args.selector_size, seed=args.selector_seed)
    selector_manifest.update(
        manifest_origin="deterministic_stratified_fixed_subset",
        source_population="original_test_both_nonempty",
        input_budget_tokens=1024, output_budget_tokens=1024,
        shared_pool=True, shared_EC=True, checkpoint_policy="reuse_fixed_CFull_trained_checkpoint",
        sampling_attempts_per_target=32, targets_retained_when_input_invalid=True,
    )
    write_json(out / "selector/frozen_test_manifest.json", selector_manifest)
    validation_manifest = choose_manifest(both_targets["validation"], n=min(200, len(both_targets["validation"])), seed=args.selector_seed, split="validation")
    validation_manifest.update(manifest_origin="structural_validation_subset",
                               not_empirical_selector_tuning=True)
    write_json(out / "selector/validation_manifest.json", validation_manifest)
    lookup = {target["sample_id"]: target for values in both_targets.values() for target in values}
    all_selector_ids = selector_manifest["target_ids"] + validation_manifest["target_ids"]
    pairs = {}
    for index, sid in enumerate(all_selector_ids, 1):
        target = lookup[sid]
        originals = list(iter_jsonl(out / "command_mapping/source_commands" / Path(target["source_file"]).with_suffix(".jsonl")))
        evidence = type_by_id.get(sid)
        if evidence is None:
            raise ValueError(f"Missing target-type mapping-status record: {sid}")
        pairs[sid] = build_pair(target, originals, evidence)
        if index % 200 == 0:
            progress("proof_blind_selector_pairs", targets=index)
    selector_counts = defaultdict(Counter)
    for filename, manifest in (("pairs.jsonl", selector_manifest), ("validation_pairs.jsonl", validation_manifest)):
        with (out / "selector" / filename).open("w", encoding="utf-8") as handle:
            for sid in manifest["target_ids"]:
                row = pairs[sid]
                emit(handle, row)
                selector_counts[filename]["targets"] += 1
                selector_counts[filename][type_by_id[sid]["status"]] += 1
                selector_counts[filename]["empty_candidate_pool"] += not bool(row["candidate_pool"])
                selector_counts[filename]["candidate_pool_commands"] += len(row["candidate_pool"])
    with (out / "selector/targets.jsonl").open("w", encoding="utf-8") as handle:
        for sid in selector_manifest["target_ids"]:
            emit(handle, lookup[sid])
    with (out / "selector/test_population_ids.jsonl").open("w", encoding="utf-8") as handle:
        for target in sorted(both_targets["test"], key=lambda row: row["sample_id"]):
            emit(handle, {"sample_id": target["sample_id"], "source_file": target["source_file"], "command_id": target["command_id"]})
    audit_folder = out / "private_posthoc_audit"
    audit_folder.mkdir()
    with (audit_folder / "selector_reference_dependency_descriptors.jsonl").open("w", encoding="utf-8") as handle:
        for sid in selector_manifest["target_ids"]:
            emit(handle, audit_by_id[sid])

    module_folder = out / "processing/dataset_preparation"
    module_folder.mkdir(parents=True)
    for filename in ("__init__.py", "tokenize.py", "tokenize_budgets.py", "summarize_lengths.py"):
        shutil.copyfile(Path(__file__).parent / filename, module_folder / filename)
    protocol = {
        "schema_version": "leancontext-reorganized-raw-v2",
        "training": {"views": list(VIEWS), "same_targets_and_reference_proofs": True,
                     "views_are_separate_checkpoint_conditions": True, "sequence_packing": False,
                     "input_budgets": [256, 512, 1024], "output_limit": 1024,
                     "output_limit_includes_one_terminal_EOS": True,
                     "input_limit_includes_statement_BOS_and_boundary_newline": True,
                     "common_membership_across_views_and_input_budgets": True,
                     "length_limit_basis": "independent_input_output_budgets",
                     "context_trimming": "remove_whole_commands_from_Full_source_order_end_until_all_views_fit_then_derive_subsets",
                     "joint_exclusion": "complete_statement_prompt_exceeds_smallest_input_budget_or_complete_reference_proof_plus_one_EOS_exceeds_output_limit",
                     "EOS_supervision": "proof_tokens_plus_exactly_one_terminal_EOS_no_padding_supervision",
                     "tokenization_status": "not_run_real_tokenizers_required"},
        "selector": {"test_targets": args.selector_size, "input_budgets": [256, 512, 1024], "output_allowance": 1024,
                     "baseline": "proof_blind_signature_lexical_Jaccard", "priority": "direct_target_type_dependency_first_baseline_order_within_groups",
                     "shared_pool_and_EC": True, "candidate_order": "strictly_preceding_same_file_MC",
                     "EC_scope_basis": "conservative_source_syntax_not_semantic_minimality_certificate",
                     "packing": "whole_commands_actual_tokenizer_skip_nonfitting_continue_restore_source_order",
                     "test_selection_uses_gold_or_generation_results": False,
                     "selection_protocol": "proof_blind_source_signature_ranking",
                     "final_visibility_and_coverage": "pending_real_tokenizer"},
    }
    write_json(out / "processing/protocol.json", protocol)
    write_json(out / "command_mapping/source_snapshot.json", {
        "lean_toolchain": (repo.mathlib_root / "lean-toolchain").read_text().strip(),
        "recorded_mathlib_commit": "a7fc949f1b05c2a01e01c027fd9f480496a1253e",
        "commit_claim_basis": "user_protocol_record_not_local_Git_verified",
        "source_byte_comparison": "every_exported_parser_command_compared_exactly_to_original_source_bytes",
        "compiler_audit_source_comparison": "all_exported_source_files_equal_local_compiled_audit_checkout",
        "source_files": source_hashes,
        "lake_manifest_sha256": file_hash(repo.mathlib_root / "lake-manifest.json"),
    })
    artifact_files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "build_status.json")
    artifact_hashes = {str(path.relative_to(out)): {"sha256": file_hash(path), "bytes": path.stat().st_size} for path in artifact_files}
    manifest = {
        "schema_version": "leancontext-reorganized-raw-v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_datasets_unchanged": True, "raw_targets": len(public), "source_files": len(source_files),
        "source_commands": command_total, "split_counts": {key: dict(value) for key, value in counts.items()},
        "source_file_disjoint": True, "normalized_statement_disjoint": True,
        "original_split_manifest_sha256": file_hash(source_public / "manifest.json"),
        "new_selector_manifest": "selector/frozen_test_manifest.json",
        "selector_counts": {key: dict(value) for key, value in selector_counts.items()},
        "selector_has_reference_proof": False, "selector_selection": "proof_blind_source_signature_ranking",
        "tokenized_outputs_created": False, "actual_model_generation_performed": False,
        "generation_or_pass_rate_results_included": False,
        "pending": ["real_backbone_tokenization_and_training_membership", "actual_budgeted_selector_prompts_and_visibility",
                    "validation_empirical_selector_rule_lock_if_tuning_is_used", "model_generations_and_Lean_verification"],
        "inputs": {"public": str(source_public), "private_provenance": str(source_private), "raw_selection": str(raw_selections),
                   "compiler_dependency_dump_sha256": file_hash(args.audit_root / "declaration_dependencies.jsonl"),
                   "type_mapping_source_sha256": file_hash(args.audit_root / "semantic_mc_labels.jsonl")},
        "artifacts": artifact_hashes,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "build_status.json", {"status": "complete_raw_pending_real_tokenization", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    progress("complete_raw_dataset", output=str(out), training_targets=counts["train"]["targets"], selector_targets=args.selector_size)


if __name__ == "__main__":
    main()
