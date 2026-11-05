"""Read independent matched-replacement inputs; never read target gold proofs.

The extension is screened offline and is reference-assisted, not proof-blind.
These checks enforce comparable generator inputs, not Lean elaboration of the
synthetically assembled context. Proof checking reuses the original source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .current_inputs import encode, ordered
from .io_utils import read_jsonl

CONDITIONS = ("Env", "Selected", "Control")
TRAIN_VIEWS = ("CMath", "CEnv", "CFull")
IMMUTABLE = ("sample_id", "target_id", "statement", "source_file", "source_module",
             "target_command_id", "target_command_uid", "target_source_span", "split",
             "proof_family", "target_kind", "original_name")
FORBIDDEN = {"target", "reference_proof", "gold_proof", "labels", "answer_ids"}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def no_target_proof_fields(value: Any) -> None:
    if isinstance(value, dict):
        require(not (FORBIDDEN & set(value)), "replacement generator artifact contains target proof/answer fields")
        for item in value.values():
            no_target_proof_fields(item)
    elif isinstance(value, list):
        for item in value:
            no_target_proof_fields(item)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        sid = str(row["sample_id"])
        require(sid not in result, f"duplicate replacement sample_id in {path}: {sid}")
        result[sid] = row
    return result


def render(statement: str, ec: list[dict[str, Any]], mc: list[dict[str, Any]]) -> str:
    items = sorted([*ec, *mc], key=lambda c: (
        c.get("prompt_slot_source_order", c["source_order"]),
        c.get("prompt_slot_command_uid", c["command_uid"])))
    return "\n\n".join([*(c["text"] for c in items), statement]).rstrip("\r\n") + "\n"


def validate_pair_contract(target: dict[str, Any], pair: dict[str, Any], raw: dict[str, Any],
                           *, backbone: str) -> None:
    """Check all three conditions together against the unchanged benchmark target."""
    require(backbone in ("llada", "llama31"), "unknown replacement backbone")
    no_target_proof_fields(pair)
    no_target_proof_fields(raw)
    for key in IMMUTABLE:
        require(key in target and pair.get(key) == raw.get(key) == target[key],
                "replacement changed immutable target field: " + key)
    ec, selected = ordered(target["environment_context"]), ordered(target["mathematical_context"])
    require(ec and selected, "replacement target must have original nonempty MC and EC")
    require(raw["shared_EC"] == ec, "replacement changed original EC")
    require(raw["selected_MC"] == selected, "replacement changed original Selected MC")
    result = raw["models"][backbone]
    require(result.get("eligible") is True, "retained replacement target is not eligible")
    control = result["replacement_MC"]
    require(len(control) == len(selected), "replacement MC command count changed")
    require(pair["declaration_pairs"] == result["declaration_pairs"], "replacement command pairing differs")
    for source, alternate in zip(selected, control):
        require(alternate.get("prompt_slot_command_id") == source["command_id"] and
                alternate.get("prompt_slot_command_uid") == source["command_uid"] and
                alternate.get("prompt_slot_source_order") == source["source_order"],
                "replacement changed original MC prompt slot")
    require(pair["shared_EC_ids"] == [c["command_id"] for c in ec], "replacement EC identity differs")
    require(set(pair["conditions"]) == set(CONDITIONS), "replacement must provide all three conditions")
    for condition, mc in (("Env", []), ("Selected", selected), ("Control", control)):
        saved = pair["conditions"][condition]
        require(saved["condition"] == condition and saved["prompt"] == render(target["statement"], ec, mc),
                "replacement serialized prompt changed target, EC, or MC slots")
        require(saved["input_limit"] == saved["output_limit"] == 1024 and
                saved.get("output_limit_includes_terminal_EOS") is True,
                "replacement requires independent 1024 input/output budgets")
        require(saved.get("input_ids_contain_prompt_only") is True and
                saved.get("reference_proof_in_generator_input") is False,
                "replacement inputs must contain prompt only, without target proof")
        ids = saved["input_ids"]
        require(saved.get("valid_input") is True and isinstance(ids, list) and
                len(ids) == saved["input_length"] == saved["prompt_length"] <= 1024,
                "replacement invalid input or appended answer token IDs")
        require(saved["visible_ec_command_ids"] == [c["command_id"] for c in ec] and
                saved["visible_ec_command_uids"] == [c["command_uid"] for c in ec],
                "replacement EC differs across conditions")
        require(saved["visible_mc_command_ids"] == [c["command_id"] for c in mc] and
                saved["visible_mc_command_uids"] == [c["command_uid"] for c in mc],
                "replacement visible MC differs from paired declarations")
        require(saved["EC_count"] == len(ec) and saved["MC_count"] == len(mc),
                "replacement context counts differ")


def replacement_input_for_generation(target: dict[str, Any], pair: dict[str, Any], raw: dict[str, Any],
                                     *, condition: str, tokenizer: Any,
                                     backbone: str) -> tuple[str, list[int], int, None]:
    require(condition in CONDITIONS, "unknown replacement condition")
    validate_pair_contract(target, pair, raw, backbone=backbone)
    saved = pair["conditions"][condition]
    prompt = saved["prompt"]
    ids = [int(x) for x in saved["input_ids"]]
    require(encode(tokenizer, prompt, backbone) == ids,
            "replacement prompt IDs differ from the active tokenizer")
    return prompt, ids, saved["EC_count"] + saved["MC_count"], None


def extension_info(extension: Path, backbone: str) -> dict[str, Any]:
    require(backbone in ("llada", "llama31"), "unknown replacement backbone")
    extension = extension.resolve()
    out = extension / "matched_replacement"
    frozen_path = out / "frozen_test_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    manifest_path = extension / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_path = out / "manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    require(manifest.get("validation_status") == "passed" and
            release.get("independent_validation_status") == "passed",
            "replacement extension must pass independent validation and finalization")
    require(manifest.get("sha256") == digest(release_path), "replacement extension manifest hash changed")
    required_artifacts = (
        "matched_replacement/frozen_test_manifest.json", "matched_replacement/targets.jsonl",
        "matched_replacement/pairs.jsonl", f"tokenized/{backbone}/matched_replacement/input_1024/pairs.jsonl")
    for relative in required_artifacts:
        expected = release["artifacts"].get(relative)
        path = extension / relative
        require(expected is not None and path.is_file() and path.stat().st_size == expected["bytes"] and
                digest(path) == expected["sha256"], "replacement extension artifact changed: " + relative)
    ids = [str(sid) for sid in frozen["target_ids"]]
    count = len(ids)
    require(count == len(set(ids)) == frozen["actual_final_targets"] == manifest["actual_targets"],
            "replacement frozen population count/duplicates differ")
    require(frozen["requested_targets"] == manifest["requested_targets"] == 2000 and 0 < count <= 2000,
            "replacement inference needs 1–2000 genuinely retained targets")
    reached = count == 2000
    require(frozen["requested_size_reached"] is reached and manifest["requested_size_reached"] is reached,
            "replacement requested_size_reached differs from actual count")
    return {"extension_root": str(extension), "frozen_manifest": str(frozen_path),
            "frozen_manifest_sha256": digest(frozen_path), "target_ids": ids,
            "population": count, "requested_population": 2000, "requested_size_reached": reached,
            "allowed_ids_file": str(out / "targets.jsonl"),
            "raw_pairs_file": str(out / "pairs.jsonl"),
            "pairs_file": str(extension / f"tokenized/{backbone}/matched_replacement/input_1024/pairs.jsonl"),
            "input_source_test_sha256": release["input_source_test_sha256"]}


def validate_replacement_artifacts(config: dict[str, Any]) -> None:
    info = config["intervention"]
    backbone = str(info["backbone"])
    require(config["architecture"] == ("dllm" if backbone == "llada" else "ar"),
            "replacement backbone and generation architecture differ")
    generation = config["generation"]
    require(generation.get("input_format") == "replacement_pair" and
            generation.get("replacement_condition") in CONDITIONS and
            generation["replacement_condition"] == info["input_condition"],
            "replacement condition metadata differs")
    require(generation.get("input_limit") == generation.get("max_new_tokens") == 1024 and
            generation.get("max_sequence_length") == 2048,
            "replacement requires fixed independent 1024 input/output budgets")
    require(generation["sampled"].get("num_candidates") == 32 and
            float(generation["sampled"].get("temperature", -1)) == 0.6 and
            generation["greedy"].get("num_candidates") == 1 and
            float(generation["greedy"].get("temperature", -1)) == 0.0,
            "replacement requires 32 samples at temperature 0.6 and separate greedy temperature 0")
    require(config["evaluation"].get("require_all_targets") is True and
            config["evaluation"].get("max_samples") == 0,
            "replacement forbids filtering the frozen target population")
    require(info["train_view"] in TRAIN_VIEWS and
            Path(str(config["model"]["checkpoint"])).parent.name == "full_sft_" + info["train_view"],
            "replacement fixed checkpoint training view differs from metadata")
    fresh = extension_info(Path(info["extension_root"]), backbone)
    require(fresh["frozen_manifest_sha256"] == info["frozen_manifest_sha256"],
            "replacement frozen manifest changed after planning")
    ids = fresh["target_ids"]
    data = config["data"]
    require(data["allowed_ids_file"] == fresh["allowed_ids_file"] and
            data["tokenized_input_file"] == fresh["pairs_file"] and
            data["replacement_raw_pairs_file"] == fresh["raw_pairs_file"],
            "replacement config no longer reads the frozen extension inputs")
    require(config["evaluation"]["expected_population_size"] == len(ids) == info["population"],
            "replacement denominator is not the actual frozen population")
    test_path = Path(data["test_file"])
    require(digest(test_path) == fresh["input_source_test_sha256"],
            "replacement benchmark test file differs from generation input source")
    allowed = unique_rows(Path(data["allowed_ids_file"]))
    encoded = unique_rows(Path(data["tokenized_input_file"]))
    raw = unique_rows(Path(data["replacement_raw_pairs_file"]))
    require(list(allowed) == list(encoded) == list(raw) == ids,
            "replacement files do not contain every frozen ID in frozen order")
    wanted = set(ids)
    targets = {}
    for target in read_jsonl(test_path):
        sid = str(target["sample_id"])
        if sid in wanted:
            require(sid not in targets, "duplicate replacement target in original benchmark")
            targets[sid] = target
    require(set(targets) == wanted, "original benchmark is missing frozen replacement IDs")
    for sid in ids:
        for key in IMMUTABLE:
            require(allowed[sid].get(key) == targets[sid][key], "allowed-ID row changed target: " + key)
        validate_pair_contract(targets[sid], encoded[sid], raw[sid], backbone=backbone)
