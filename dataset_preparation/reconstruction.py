"""D.4 proof-blind context reconstruction, separate from D.5 frozen-pool selection.

No target reference-proof fields, observed proof dependencies, or model outcomes
are accessed. Preceding declaration bodies may be supplied as complete commands,
but lexical ranking and binding relevance use names/signatures, not their bodies.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .selector import MC_KINDS, TARGET_KEYS, _clean_target, _mask, build_pair, identifier_terms
from .tokenize import (VIEWS, command_id, command_key, encode_prompt, environment_scope_cohorts,
                       iter_jsonl, render_training_prompt, source_order, unique_commands,
                       write_json, write_row)

VARIANTS = ("original", "reconstructed")
MODELS = {"llada": "dllm", "llama31": "llama31"}
PROTOCOL = {
    "schema_version": "D4-proof-blind-reconstruction-v1",
    "population": "same_existing_fixed_2000_selector_test_IDs_no_resampling",
    "alternative_inputs": "target_statement_and_eligible_strictly_preceding_source_commands_only",
    "MC_ranking": "source_named_eligible_pool_Jaccard_of_statement_vs_names_and_signatures_distance_ID_ties",
    "EC": "rebuild_target_and_current_proposal_MC_syntactic_binding_scope_cohorts_after_every_trial",
    "bindings": "whole_variable_universe_include_omit_commands_with_lexically_relevant_bindings_in_anchor_scope",
    "other_scope_effects": "conservative_pre_anchor_effects_visible_in_target_or_selected_MC_scopes",
    "packing": "greedy_MC_trial_each_with_new_EC_complete_full_prompt_and_all_projections_must_fit_1024",
    "original_packing": "tail_remove_complete_commands_from_original_CFull_then_project_without_refill",
    "input_limit": 1024, "output_limit_including_one_EOS": 1024,
    "reference_proof_fields_accessed": False,
    "proof_or_target_type_dependency_evidence_used": False,
    "whole_candidate_pool_EC_frozen_before_MC_selection": False,
    "is_D5_shared_pool_baseline": False,
    "EC_is_Lean_binding_certificate": False,
    "synthetic_context_newly_Lean_elaborated": False,
    "all_2000_targets_retained_in_each_view": True,
    "invalid_context_views": "explicit_failed_inputs_not_dropped; C0_remains_identical_bare_statement",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def view_records(tokenizer: Any, statement: str, mc: list[dict[str, Any]], ec: list[dict[str, Any]],
                 *, backbone: str, invalid_context_reason: str | None = None) -> dict[str, Any]:
    """Plain prompt-only four projections of one common command union."""
    full = source_order(unique_commands([*mc, *ec]))
    math_keys, env_keys = {command_key(c) for c in mc}, {command_key(c) for c in ec}
    if math_keys & env_keys:
        raise ValueError("reconstruction MC/EC overlap")
    projected = {"C0": [], "CMath": [c for c in full if command_key(c) in math_keys],
                 "CEnv": [c for c in full if command_key(c) in env_keys], "CFull": full}
    result = {}
    for view in VIEWS:
        items = projected[view]
        prompt = render_training_prompt(statement, items)
        ids = encode_prompt(tokenizer, prompt, backbone)
        reason = invalid_context_reason if view != "C0" else None
        if len(ids) > 1024:
            reason = reason or "complete_input_exceeds_1024"
        result[view] = {
            "view": view, "prompt": prompt, "input_ids": ids,
            "prompt_length": len(ids), "input_length": len(ids),
            "input_limit": 1024, "output_limit": 1024,
            "output_limit_includes_terminal_EOS": True,
            "input_ids_contain_prompt_only": True, "reference_proof_in_generator_input": False,
            "valid_input": reason is None, "reason": reason,
            "visible_command_ids": [command_id(c) for c in items],
            "visible_command_uids": [command_key(c) for c in items],
            "visible_mc_command_ids": [command_id(c) for c in items if command_key(c) in math_keys],
            "visible_ec_command_ids": [command_id(c) for c in items if command_key(c) in env_keys],
            "context_items": len(items),
        }
    return result


def original_views(tokenizer: Any, statement: str, mc: list[dict[str, Any]], ec: list[dict[str, Any]],
                   *, backbone: str) -> dict[str, Any]:
    full = source_order(unique_commands([*mc, *ec]))
    math_keys = {command_key(c) for c in mc}
    env_keys = {command_key(c) for c in ec}
    removed = []
    while True:
        kept_mc = [c for c in full if command_key(c) in math_keys]
        kept_ec = [c for c in full if command_key(c) in env_keys]
        views = view_records(tokenizer, statement, kept_mc, kept_ec, backbone=backbone)
        if all(v["valid_input"] for v in views.values()) or not full:
            return {"mathematical_context": kept_mc, "environment_context": kept_ec,
                    "views": views, "removed_command_ids": removed,
                    "context_origin": "original_command_ID_mapping_not_reselected"}
        removed.append(command_id(full.pop()))


def binding_terms(command: dict[str, Any]) -> set[str]:
    """Conservative source binder relevance, not semantic free-variable analysis."""
    text = _mask(command["text"])
    kind = command["kind"]
    if kind != "variable":
        return identifier_terms(text)
    pieces = []
    # Named binders use the names before ':'. Instance brackets also retain
    # their type symbols, since an anonymous/named instance is used implicitly.
    for opener, body in re.findall(r"([({\[])([^(){}\[\]]*)[)}\]]", text):
        pieces.append(body if opener == "[" else body.split(":", 1)[0])
    return identifier_terms(" ".join(pieces))


def reconstruct_ec(statement: str, source_pair: dict[str, Any], mc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute EC from target and selected MC anchors, never from whole-pool paths."""
    if source_pair["EC_scope_stack_status"] != "syntactic_valid":
        raise ValueError("source_scope_unresolved")
    environment = source_pair["shared_environment"]
    cohorts = environment_scope_cohorts(environment)
    index = {command_id(c): c for c in environment}
    target_scopes = {command_id(c) for c in environment
                     if c.get("scope_role") == "block_opener" and c.get("target_scope_active") is True}
    anchors = [(target_scopes, 10**18, identifier_terms(statement))]
    anchors.extend((set(c["scope_path"]), c["source_order"], identifier_terms(c["signature"])) for c in mc)
    chosen = set(target_scopes)
    for c in mc:
        chosen.update(c["scope_path"])
    for item in environment:
        if item.get("scope_role") in ("block_opener", "block_closer"):
            continue
        visible_anchors = [terms for scopes, order, terms in anchors
                           if item["source_order"] < order and set(item["scope_path"]) <= scopes]
        if not visible_anchors:
            continue
        if item["kind"] in {"variable", "universe", "include", "omit"}:
            if not any(binding_terms(item) & terms for terms in visible_anchors):
                continue
        # Other source scope effects are conservatively kept when visible in a
        # selected anchor, not assumed unnecessary from lexical mismatch.
        chosen.add(command_id(item))
    if not chosen <= set(index):
        raise ValueError("anchor_scope_has_no_source_EC_record")
    closure = unique_commands(member for cid in sorted(chosen) for member in cohorts[cid])
    return source_order(closure)


def reconstructed_views(tokenizer: Any, target: dict[str, Any], commands: list[dict[str, Any]],
                        *, backbone: str) -> dict[str, Any]:
    clean = _clean_target(target)
    # No original context, type evidence, proof audit or gold field is passed to
    # the independent selector. Its D.5 fixed-EC output is intentionally ignored.
    pair = build_pair(clean, commands, None)
    pool = {command_id(c): c for c in pair["candidate_pool"]}
    statement = clean["statement"]
    chosen = []
    skipped = []
    try:
        ec = reconstruct_ec(statement, pair, [])
        initial = view_records(tokenizer, statement, [], ec, backbone=backbone)
        if not all(v["valid_input"] for v in initial.values()):
            raise ValueError("required_target_EC_or_statement_exceeds_1024")
    except ValueError as exc:
        return {"mathematical_context": [], "environment_context": [],
                "views": view_records(tokenizer, statement, [], [], backbone=backbone,
                                      invalid_context_reason=str(exc)),
                "reconstruction_valid": False, "reason": str(exc),
                "candidate_ranking": pair["baseline_ranking"], "skipped_candidates": [],
                "EC_rebuilt_for_each_MC_trial": True, "D5_fixed_EC_used": False}
    for cid in pair["baseline_ranking"]:
        candidate = pool[cid]
        if candidate.get("scope_identity_status") != "syntactic_valid":
            skipped.append({"command_id": cid, "reason": "candidate_scope_unresolved"})
            continue
        proposal = source_order([*chosen, candidate])
        try:
            proposal_ec = reconstruct_ec(statement, pair, proposal)
            trial = view_records(tokenizer, statement, proposal, proposal_ec, backbone=backbone)
        except ValueError as exc:
            skipped.append({"command_id": cid, "reason": str(exc)})
            continue
        if not all(v["valid_input"] for v in trial.values()):
            skipped.append({"command_id": cid, "reason": "proposal_MC_with_rebuilt_EC_exceeds_1024"})
            continue
        chosen, ec = proposal, proposal_ec
    return {"mathematical_context": chosen, "environment_context": ec,
            "views": view_records(tokenizer, statement, chosen, ec, backbone=backbone),
            "reconstruction_valid": True, "reason": None,
            "candidate_ranking": pair["baseline_ranking"], "skipped_candidates": skipped,
            "EC_rebuilt_for_each_MC_trial": True, "D5_fixed_EC_used": False}


def load_targets(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = root / "selector/frozen_test_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(sid) for sid in manifest["target_ids"]]
    if len(ids) != 2000 or len(set(ids)) != 2000 or manifest.get("n") != 2000 or len(manifest["targets"]) != 2000:
        raise ValueError("expected the existing fixed 2000 selector test IDs")
    targets = {str(row["sample_id"]): _clean_target(row) for row in manifest["targets"]}
    if set(targets) != set(ids) or any(row.get("split") != "test" for row in targets.values()):
        raise ValueError("fixed selector targets missing, duplicated, or not test")
    original = {}
    for row in iter_jsonl(root / "command_mapping/targets.jsonl"):
        sid = str(row["sample_id"])
        if sid in targets:
            if sid in original:
                raise ValueError("duplicate original context-ID target mapping")
            # Deliberately whitelist no proof_sha256, kernel_target_name,
            # introduced declarations, or semantic proof descriptors.
            original[sid] = {key: row[key] for key in (
                "sample_id", "split", "source_file", "target_command_id", "context_commands")}
    if set(original) != set(ids):
        raise ValueError("missing original context-command mapping for frozen targets")
    return manifest, targets, original


def new_output(root: Path, output: Path, *, create: bool) -> Path:
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside the submitted dataset")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty reconstruction output")
    if create:
        output.mkdir(parents=True, exist_ok=True)
    return output


def run(root: Path, output: Path, models: list[str]) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from .validate_replacement import validate_source_catalog
    manifest, targets, original = load_targets(root)
    output = new_output(root, output, create=True)
    write_json(output / "build_status.json", {"status": "in_progress"})
    write_json(output / "protocol.json", PROTOCOL)
    write_json(output / "frozen_test_manifest.json", {
        "target_ids": manifest["target_ids"], "actual_targets": 2000,
        "source_manifest_sha256": sha256(root / "selector/frozen_test_manifest.json"),
        "source_manifest_origin": manifest.get("manifest_origin"),
        "original_manifest_reused_without_resampling": True})
    with (output / "targets.jsonl").open("w", encoding="utf-8") as handle:
        for sid in manifest["target_ids"]:
            write_row(handle, targets[sid])
    snapshots = json.loads((root / "command_mapping/source_snapshot.json").read_text())
    revision = json.loads((root / "command_mapping/revision_verification.json").read_text())
    catalogs = {}
    tokenizer_meta = {}
    statistics = {}
    for model in models:
        acquisition_path = root / "tokenizers" / model / "acquisition_manifest.json"
        acquisition = json.loads(acquisition_path.read_text())
        for name, entry in acquisition["files"].items():
            path = root / "tokenizers" / model / entry.get("local_filename", name)
            if sha256(path) != entry["sha256"]:
                raise ValueError("local actual tokenizer acquisition hash mismatch")
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizers" / model),
            local_files_only=True, trust_remote_code=False, use_fast=True)
        tokenizer_meta[model] = {"acquisition_manifest_sha256": sha256(acquisition_path),
                                 "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id}
        path = output / f"tokenized/{model}/reconstruction/input_1024/pairs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        with path.open("w", encoding="utf-8") as handle:
            for index, sid in enumerate(manifest["target_ids"], 1):
                target, mapping = targets[sid], original[sid]
                source_file = target["source_file"]
                if source_file not in catalogs:
                    relative = Path(source_file)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe source path")
                    commands = list(iter_jsonl(root / "command_mapping/source_commands" / relative.with_suffix(".jsonl")))
                    source = (root / "source_snapshot/mathlib4" / relative).read_bytes()
                    validate_source_catalog(commands, source_file, source, snapshots["source_files"][source_file],
                                            revision["files"][source_file])
                    catalogs[source_file] = commands
                commands = catalogs[source_file]
                by_id = {command_id(c): c for c in commands}
                if (mapping["source_file"], mapping["target_command_id"], mapping["split"]) != (
                        source_file, target["target_command_id"], "test"):
                    raise ValueError("original context mapping identifies a different target")
                original_commands = []
                for item in mapping["context_commands"]:
                    c = by_id[item["command_id"]]
                    if c["context_text_sha256"] != item["context_text_sha256"]:
                        raise ValueError("original context-command source text hash changed")
                    original_commands.append(c)
                mc = [c for c in original_commands if c["effective_kind"] in MC_KINDS]
                ec = [c for c in original_commands if c["effective_kind"] not in MC_KINDS]
                variants = {
                    "original": original_views(tokenizer, target["statement"], mc, ec, backbone=MODELS[model]),
                    "reconstructed": reconstructed_views(tokenizer, target, commands, backbone=MODELS[model]),
                }
                if variants["original"]["views"]["C0"] != variants["reconstructed"]["views"]["C0"]:
                    raise AssertionError("original/reconstructed C0 must be identical")
                write_row(handle, {**target, "backbone": model, "variants": variants,
                    "protocol_version": PROTOCOL["schema_version"], "input_limit": 1024, "output_limit": 1024,
                    "reference_proof_fields_accessed": False, "model_outcomes_used": False})
                for variant in VARIANTS:
                    for view in VIEWS:
                        counts[f"{variant}/{view}/" + ("valid" if variants[variant]["views"][view]["valid_input"] else "invalid")] += 1
                counts["EC_differs"] += [command_id(c) for c in variants["original"]["environment_context"]] != [command_id(c) for c in variants["reconstructed"]["environment_context"]]
                counts["MC_differs"] += [command_id(c) for c in variants["original"]["mathematical_context"]] != [command_id(c) for c in variants["reconstructed"]["mathematical_context"]]
                if index % 100 == 0:
                    print(json.dumps({"model": model, "processed": index, "total": 2000}), flush=True)
        statistics[model] = dict(counts)
    artifacts = {str(p.relative_to(output)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(output.rglob("*")) if p.is_file() and p.name != "build_status.json"}
    result = {"schema_version": PROTOCOL["schema_version"], "status": "complete",
        "actual_targets": 2000, "models": models, "statistics": statistics, "tokenizers": tokenizer_meta,
        "input_source_test_sha256": sha256(root / "benchmark/test.jsonl"),
        "benchmark_file_hashed_for_binding_only_reference_proof_fields_not_accessed": True,
        "input_dataset_modified": False, "model_generation_performed": False,
        "synthetic_context_newly_Lean_elaborated": False, "artifacts": artifacts,
        "processing_package_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
                                      if not p.name.startswith("test_")}}
    write_json(output / "manifest.json", result)
    write_json(output / "build_status.json", {"status": "complete", "actual_targets": 2000,
                                               "manifest_sha256": sha256(output / "manifest.json")})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=("llada", "llama31", "both"), default="both")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Read-only plan (default)")
    mode.add_argument("--execute", action="store_true", help="Build actual tokenizer outputs on CPU")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = new_output(root, args.output_root, create=False)
    models = list(MODELS) if args.backbone == "both" else [args.backbone]
    if args.execute:
        result = run(root, output, models)
    else:
        manifest, _, _ = load_targets(root)
        result = {"execution_performed": False, "protocol": PROTOCOL, "models": models,
                  "targets": len(manifest["target_ids"]), "output_root": str(output)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
