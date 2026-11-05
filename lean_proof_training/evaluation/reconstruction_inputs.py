"""D.4 original/reconstructed four-view inputs, distinct from D.5 selector.

Reference proof bodies are never accessed. Invalid reconstructed inputs retain
their fixed target denominator and become explicit failed candidates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .current_inputs import VIEWS, encode, input_for_generation, ordered, render
from .io_utils import read_jsonl
from .replacement_inputs import IMMUTABLE, digest, no_target_proof_fields, require, unique_rows

VARIANTS = ("original", "reconstructed")


def context_key(command: dict[str, Any]) -> str:
    return str(command["command_uid"])


def validate_pair(target: dict[str, Any], pair: dict[str, Any]) -> None:
    no_target_proof_fields(pair)
    for key in IMMUTABLE:
        require(pair.get(key) == target[key], "reconstruction changed target field: " + key)
    require(pair["input_limit"] == pair["output_limit"] == 1024,
            "reconstruction input/output budgets must independently be 1024")
    require(pair.get("reference_proof_fields_accessed") is False and pair.get("model_outcomes_used") is False,
            "reconstruction must not access proof fields or model outcomes")
    require(set(pair["variants"]) == set(VARIANTS), "missing original/reconstructed inputs")
    original = pair["variants"]["original"]
    original_full = ordered([*original["mathematical_context"], *original["environment_context"]])
    source_full = ordered(target["full_context"])
    require([(context_key(c), c["text"]) for c in original_full] ==
            [(context_key(c), c["text"]) for c in source_full[:len(original_full)]],
            "original input is not a common source-order Full prefix")
    original_math_keys = {context_key(c) for c in target["mathematical_context"]}
    original_env_keys = {context_key(c) for c in target["environment_context"]}
    require(all(context_key(c) in original_math_keys for c in original["mathematical_context"]) and
            all(context_key(c) in original_env_keys for c in original["environment_context"]),
            "original MC/EC classifications changed")
    for variant in VARIANTS:
        record = pair["variants"][variant]
        mc, ec = ordered(record["mathematical_context"]), ordered(record["environment_context"])
        math_keys, env_keys = {context_key(c) for c in mc}, {context_key(c) for c in ec}
        require(not (math_keys & env_keys), "reconstruction MC/EC overlap")
        full = ordered([*mc, *ec])
        require(len(full) == len({context_key(c) for c in full}), "duplicate reconstruction commands")
        require(set(record["views"]) == set(VIEWS), "reconstruction must project four views")
        if variant == "reconstructed":
            require(record.get("EC_rebuilt_for_each_MC_trial") is True and record.get("D5_fixed_EC_used") is False,
                    "D.4 reconstruction must not reuse D.5 fixed whole-pool EC")
        projected = {"C0": [], "CMath": mc, "CEnv": ec, "CFull": full}
        for view, items in projected.items():
            saved = record["views"][view]
            require(saved["view"] == view and saved["prompt"] == render(target["statement"], items),
                    "reconstruction view is not an unrefilled projection of one Full input")
            require(saved.get("input_ids_contain_prompt_only") is True and
                    saved.get("reference_proof_in_generator_input") is False,
                    "reconstruction must contain prompt-only token IDs")
            require(saved["input_limit"] == saved["output_limit"] == 1024 and
                    saved.get("output_limit_includes_terminal_EOS") is True,
                    "reconstruction per-view budgets changed")
            ids = saved["input_ids"]
            require(isinstance(ids, list) and len(ids) == saved["prompt_length"] == saved["input_length"],
                    "reconstruction prompt token span changed or contains appended answers")
            require(saved["visible_command_uids"] == [context_key(c) for c in items] and
                    saved["visible_command_ids"] == [c["command_id"] for c in items],
                    "reconstruction visible command IDs differ")
            require(saved["visible_mc_command_ids"] == [c["command_id"] for c in items if context_key(c) in math_keys] and
                    saved["visible_ec_command_ids"] == [c["command_id"] for c in items if context_key(c) in env_keys],
                    "reconstruction MC/EC view membership differs")
            require(saved["context_items"] == len(items), "reconstruction context count differs")
            if saved["valid_input"]:
                require(len(ids) <= 1024 and saved["reason"] is None, "valid reconstruction view exceeds input budget")
            else:
                require(bool(saved["reason"]), "invalid reconstruction view lacks an explicit reason")
    require(pair["variants"]["original"]["views"]["C0"] == pair["variants"]["reconstructed"]["views"]["C0"],
            "original/reconstructed C0 must be identical")


def reconstruction_input_for_generation(target: dict[str, Any], pair: dict[str, Any], *, variant: str,
                                        view: str, tokenizer: Any, backbone: str):
    require(variant in VARIANTS and view in VIEWS, "unknown reconstruction variant/view")
    validate_pair(target, pair)
    # Independently check the original maximal common prefix with the actual
    # tokenizer, without using reference proof lengths or answer token IDs.
    expected, _, _, _ = input_for_generation(target, None, tokenizer=tokenizer,
                                             backbone=backbone, view=view, input_limit=1024)
    require(pair["variants"]["original"]["views"][view]["prompt"] == expected,
            "original reconstruction comparison differs from standard four-view packing")
    saved = pair["variants"][variant]["views"][view]
    prompt, ids = saved["prompt"], [int(x) for x in saved["input_ids"]]
    require(encode(tokenizer, prompt, backbone) == ids, "reconstruction prompt IDs differ from active tokenizer")
    reason = None if saved["valid_input"] else str(saved["reason"])
    return prompt, ids, saved["context_items"], reason


def extension_info(extension: Path, backbone: str) -> dict[str, Any]:
    require(backbone in ("llada", "llama31"), "unknown reconstruction backbone")
    extension = extension.resolve()
    manifest_path = extension / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((extension / "build_status.json").read_text(encoding="utf-8"))
    require(manifest.get("status") == status.get("status") == "complete" and
            manifest.get("actual_targets") == 2000 and backbone in manifest["models"],
            "reconstruction extension must contain the complete fixed 2000 targets")
    require(status.get("manifest_sha256") == digest(manifest_path), "reconstruction manifest hash changed")
    required = ["frozen_test_manifest.json", "targets.jsonl", "protocol.json",
                f"tokenized/{backbone}/reconstruction/input_1024/pairs.jsonl"]
    for relative in required:
        expected, path = manifest["artifacts"].get(relative), extension / relative
        require(expected is not None and path.is_file() and path.stat().st_size == expected["bytes"] and
                digest(path) == expected["sha256"], "reconstruction artifact changed: " + relative)
    frozen = json.loads((extension / "frozen_test_manifest.json").read_text())
    ids = [str(sid) for sid in frozen["target_ids"]]
    require(len(ids) == len(set(ids)) == frozen["actual_targets"] == 2000,
            "reconstruction population differs from frozen 2000 IDs")
    require(frozen.get("original_manifest_reused_without_resampling") is True,
            "reconstruction must reuse the existing selector manifest")
    protocol = json.loads((extension / "protocol.json").read_text())
    require(protocol.get("is_D5_shared_pool_baseline") is False and
            protocol.get("proof_or_target_type_dependency_evidence_used") is False,
            "D.4 reconstruction cannot be D.5 shared-pool/type-prioritized selection")
    return {"extension_root": str(extension), "manifest_sha256": digest(manifest_path), "population": 2000,
            "allowed_ids_file": str(extension / "targets.jsonl"), "target_ids": ids,
            "pairs_file": str(extension / f"tokenized/{backbone}/reconstruction/input_1024/pairs.jsonl"),
            "input_source_test_sha256": manifest["input_source_test_sha256"],
            "source_selector_manifest_sha256": frozen["source_manifest_sha256"]}


def validate_reconstruction_artifacts(config: dict[str, Any]) -> None:
    info, generation, data = config["intervention"], config["generation"], config["data"]
    backbone = info["backbone"]
    fresh = extension_info(Path(info["extension_root"]), backbone)
    require(fresh["manifest_sha256"] == info["manifest_sha256"], "reconstruction manifest changed after plan")
    require(config["architecture"] == ("dllm" if backbone == "llada" else "ar"), "reconstruction architecture differs")
    require(info["train_view"] == "CFull" and Path(config["model"]["checkpoint"]).parent.name == "full_sft_CFull",
            "reconstruction uses only the fixed CFull-trained checkpoint")
    require(generation["input_format"] == "reconstruction_pair" and generation["reconstruction_variant"] in VARIANTS and
            generation["reconstruction_view"] in VIEWS, "invalid reconstruction generation mode")
    require(generation["input_limit"] == generation["max_new_tokens"] == 1024 and
            generation["max_sequence_length"] == 2048, "reconstruction budgets differ")
    require(generation["sampled"]["num_candidates"] == 32 and float(generation["sampled"]["temperature"]) == 0.6 and
            generation["greedy"]["num_candidates"] == 1 and float(generation["greedy"]["temperature"]) == 0.0,
            "reconstruction decoding differs from fixed 32+greedy protocol")
    require(config["evaluation"]["expected_population_size"] == 2000 and
            config["evaluation"]["require_all_targets"] is True and config["evaluation"]["max_samples"] == 0,
            "reconstruction must retain all 2000 targets, including invalid inputs")
    require(data["allowed_ids_file"] == fresh["allowed_ids_file"] and data["tokenized_input_file"] == fresh["pairs_file"],
            "reconstruction input file paths changed")
    require(digest(Path(data["test_file"])) == fresh["input_source_test_sha256"], "reconstruction original benchmark hash differs")
    dataset_root = Path(data["test_file"]).resolve().parents[1]
    require(digest(dataset_root / "selector/frozen_test_manifest.json") == fresh["source_selector_manifest_sha256"],
            "reconstruction no longer uses the same selector fixed IDs")
    allowed = unique_rows(Path(data["allowed_ids_file"]))
    pairs = unique_rows(Path(data["tokenized_input_file"]))
    require(list(allowed) == list(pairs) == fresh["target_ids"], "reconstruction missing/duplicate/reordered frozen IDs")
    wanted, targets = set(fresh["target_ids"]), {}
    for target in read_jsonl(data["test_file"]):
        sid = str(target["sample_id"])
        if sid in wanted:
            require(sid not in targets, "duplicate reconstruction benchmark target")
            targets[sid] = target
    require(set(targets) == wanted, "benchmark is missing reconstruction fixed IDs")
    catalogs = {}
    catalog_root = Path(config["source"]["command_spans_root"])
    for sid in fresh["target_ids"]:
        target, pair = targets[sid], pairs[sid]
        for key in IMMUTABLE:
            require(allowed[sid].get(key) == target[key], "reconstruction allowed-ID row changed target")
        validate_pair(target, pair)
        source_file = target["source_file"]
        relative = Path(source_file)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe reconstruction source path")
        if source_file not in catalogs:
            catalogs[source_file] = {c["command_id"]: c for c in read_jsonl(catalog_root / relative.with_suffix(".jsonl"))}
        source = catalogs[source_file]
        for variant in VARIANTS:
            record = pair["variants"][variant]
            for c in [*record["mathematical_context"], *record["environment_context"]]:
                canonical = source.get(c["command_id"])
                require(canonical is not None and c["text"] == canonical["text"] and
                        c["command_uid"] == canonical["command_uid"] and
                        c["source_span"] == canonical["source_span"] and
                        canonical["source_span"]["end_byte"] <= target["target_source_span"]["start_byte"],
                        "reconstruction context is not a complete real preceding command")
