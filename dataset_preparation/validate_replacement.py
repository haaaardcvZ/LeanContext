"""Audit matched replacements without importing the builder's matching helpers.

Prompt rendering, tokenization, sampling, command/provenance checks, dependency
unions and minimum matching costs are reconstructed here. Source scope parsing
is intentionally a shared contract; the report explicitly discloses that limit.
No model generation or Lean candidate elaboration is performed.
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
from typing import Any, Iterable


MODELS = ("llada", "llama31")
CONDITIONS = ("Env", "Selected", "Control")
DEPENDENCY_FIELDS = ("type_dependency", "value_dependency", "proof_extra_dependency",
                     "projection_augmented_proof_extra_dependency")
BASES = {"kernel_selection_range", "kernel_declaration_range", "exact_source_metadata_name",
         "kernel_constructor_inductive", "kernel_recursor_inductives",
         "registered_structure_projection", "registered_aux_recursor_or_no_confusion"}
OWNERSHIP_BASES = BASES - {"kernel_selection_range", "kernel_declaration_range", "exact_source_metadata_name"}
TARGET_IDENTITIES = {"kernel_range_and_complete_source_namespace", "kernel_range_and_exact_source_name",
                     "kernel_range_and_source_name_suffix", "unique_kernel_theorem_in_source_command"}
FORBIDDEN_KEYS = {"target", "reference_proof", "gold_proof", "referenceproof", "answer_ids", "labels"}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{number}: non-object row")
            yield row


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows(path):
        sid = row["sample_id"]
        require(sid not in result, f"{path}: duplicate sample_id {sid}")
        result[sid] = row
    return result


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def ordered(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(commands, key=lambda c: (c["source_order"], c["command_uid"]))


def command_kind(command: dict[str, Any]) -> str:
    value = str(command.get("effective_kind", command.get("kind", ""))).split(".")[-1]
    if value == "instance" and re.match(r"\s*local\s+instance\b", command["text"]):
        return "local_instance"
    return value


def within_band(selected: int, candidate: int) -> bool:
    return selected > 0 and 9 * selected <= candidate * 10 <= 11 * selected


def reconstruct_prompt(statement: str, ec: list[dict[str, Any]],
                       mc: list[dict[str, Any]]) -> str:
    items = sorted([*ec, *mc], key=lambda c: (
        c.get("prompt_slot_source_order", c["source_order"]),
        c.get("prompt_slot_command_uid", c["command_uid"])))
    return "\n\n".join([*(c["text"] for c in items), statement]).rstrip("\r\n") + "\n"


def encode(tokenizer: Any, text: str, model: str) -> list[int]:
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    if model == "llama31" and (not ids or ids[0] != tokenizer.bos_token_id):
        require(tokenizer.bos_token_id is not None, "Llama tokenizer has no BOS")
        ids.insert(0, int(tokenizer.bos_token_id))
    require(model in MODELS, "unknown tokenizer model")
    return ids


def no_proof_fields(value: Any) -> None:
    if isinstance(value, dict):
        require(not (set(value) & FORBIDDEN_KEYS), "generator artifact contains proof/answer fields")
        for item in value.values():
            no_proof_fields(item)
    elif isinstance(value, list):
        for item in value:
            no_proof_fields(item)


def minimum_cost(edges: list[dict[str, int]]) -> int | None:
    """Independent successive-shortest-path min-cost flow, not Hungarian code."""
    if not edges or any(not e for e in edges):
        return None
    columns = sorted({cid for e in edges for cid in e})
    if len(columns) < len(edges):
        return None
    n, m = len(edges), len(columns)
    sink, size = 1 + n + m, 2 + n + m
    graph: list[list[list[int]]] = [[] for _ in range(size)]

    def arc(a: int, b: int, cost: int) -> None:
        graph[a].append([b, len(graph[b]), 1, cost])
        graph[b].append([a, len(graph[a]) - 1, 0, -cost])

    for i, edge in enumerate(edges):
        arc(0, 1 + i, 0)
        for j, cid in enumerate(columns):
            if cid in edge:
                arc(1 + i, 1 + n + j, edge[cid])
    for j in range(m):
        arc(1 + n + j, sink, 0)
    total = 0
    for _ in range(n):
        distances, previous = [10**30] * size, [None] * size
        distances[0] = 0
        for _ in range(size - 1):
            changed = False
            for a, outgoing in enumerate(graph):
                if distances[a] == 10**30:
                    continue
                for k, (b, _, capacity, cost) in enumerate(outgoing):
                    if capacity and distances[a] + cost < distances[b]:
                        distances[b], previous[b] = distances[a] + cost, (a, k)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            return None
        total += distances[sink]
        b = sink
        while b:
            a, k = previous[b]
            edge = graph[a][k]
            edge[2] -= 1
            graph[b][edge[1]][2] += 1
            b = a
    return total


def validate_sampling(population_ids: list[str], sampled: dict[str, Any],
                      eligibility: list[dict[str, Any]], final: dict[str, Any]) -> list[str]:
    require(len(population_ids) == len(set(population_ids)) == 11064, "population is not original 11,064 IDs")
    drawn = random.Random(2026).sample(sorted(population_ids), 2500)
    require(sampled["sampling_seed"] == 2026 and sampled["sample_size"] == 2500,
            "sampling seed/size changed")
    require(sampled["population_size"] == 11064 and sampled["population_sha256"] == object_hash(sorted(population_ids)),
            "population identity differs")
    require(sampled["sampled_target_ids"] == drawn, "seeded sample/order differs")
    require(len(eligibility) == 2500 and [e["sample_id"] for e in eligibility] == drawn,
            "eligibility does not retain all 2,500 sampled targets")
    common = []
    for rank, row in enumerate(eligibility):
        require(row["sample_rank"] == rank and set(row["models"]) == set(MODELS), "eligibility rank/models differ")
        both = all(row["models"][model]["eligible"] is True for model in MODELS)
        require(row["both_models_eligible"] is both, "common eligibility differs")
        if both:
            common.append(row["sample_id"])
    require(final["requested_targets"] == 2000 and final["sampled_targets"] == 2500,
            "requested final/sample size changed")
    require(final["target_ids"] == common[:2000], "final membership/backfill/order differs")
    require(final["common_eligible_targets"] == len(common) and final["actual_final_targets"] == len(common[:2000]),
            "reported membership count differs")
    require(final["requested_size_reached"] is (len(common) >= 2000), "size completion claim differs")
    require(final["no_backfill_or_matching_rule_relaxation"] is True, "matching rule relaxation claimed")
    for model in MODELS:
        counts = Counter("eligible" if e["models"][model]["eligible"] else e["models"][model]["reason"] for e in eligibility)
        require(final["model_failure_counts"][model] == dict(counts), "model rejection/eligibility counts differ")
    return common


def observed_sets(raw: dict[str, Any], declarations: dict[str, dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Rebuild exclusions directly from raw audit and independently read dump."""
    ids = {raw["target_command_id"]} if raw.get("target_command_id") else set()
    names: set[str] = set()
    for field in ("outer_type_constants", "outer_value_constants", "type_constants", "value_constants", "value_minus_type_constants"):
        names.update(raw.get(field) or [])
    if raw.get("target_name"):
        names.add(raw["target_name"])
    for field in ("expanded_type_auxiliaries", "expanded_value_auxiliaries"):
        names.update(x["name"] for x in raw.get(field, []) if isinstance(x, dict) and x.get("name"))
    for field in DEPENDENCY_FIELDS:
        dep = raw.get(field) or {}
        ids.update(dep.get("command_ids") or [])
        for items in (dep.get("constant_evidence") or {}).values():
            names.update(x["constant"] for x in items)
        for label in ("unmapped_same_file_constants", "nonpreceding_same_file_constants", "same_target_command_constants"):
            names.update(dep.get(label) or [])
        for item in dep.get("other_local_commands") or []:
            ids.add(item["command_id"]); names.add(item["name"])
    for name in [raw.get("target_name"), *(x["name"] for field in ("expanded_type_auxiliaries", "expanded_value_auxiliaries") for x in raw.get(field, []))]:
        d = declarations.get(name) or {}
        sides = ("type", "value") if name == raw.get("target_name") else ("value",)
        for side in sides:
            for p in d.get(side + "_projections", []):
                names.update(x for x in (p.get("structure"), p.get("projection_name")) if isinstance(x, str) and x)
    return ids, names


def check_dependency_certificate(raw: dict[str, Any], declarations: dict[str, dict[str, Any]],
                                 ownership: dict[str, dict[str, Any]]) -> None:
    require(raw.get("mc_semantic_status") == "complete" and raw.get("reference_proof_exact_raw_body_match") is True,
            "retained target lacks complete exact-gold dependency certificate")
    require(raw.get("target_identity_status") in TARGET_IDENTITIES, "retained target identity unresolved")
    target = declarations[raw["target_name"]]
    require(target["module"] == raw["source_module"] and target.get("has_value") is True, "kernel target/module/value differs")
    auxiliaries = {x["name"] for side in ("type", "value") for x in raw["expanded_" + side + "_auxiliaries"]}
    for side in ("type", "value"):
        require(set(target[side + "_constants"]) == set(raw["outer_" + side + "_constants"]), "outer kernel constant set differs")
        expanded = set(raw["outer_" + side + "_constants"])
        pending = {x["name"] for x in raw["expanded_" + side + "_auxiliaries"]}
        require(all(x["source_attribution"] in BASES for x in raw["expanded_" + side + "_auxiliaries"]), "auxiliary ownership uncertified")
        while pending:
            reachable = pending & expanded
            require(reachable, "recorded auxiliary is not reachable")
            for name in reachable:
                d = declarations[name]
                require(d["module"] == raw["source_module"] and d.get("has_value") is True, "auxiliary kernel source differs")
                expanded.update(d["value_constants"]); pending.remove(name)
        require(expanded == set(raw[side + "_constants"]), "expanded kernel set differs")
    require(set(raw["value_minus_type_constants"]) == set(raw["value_constants"]) - set(raw["type_constants"]), "value-minus-type is incorrect")
    for field in DEPENDENCY_FIELDS:
        d = raw[field]
        require(not d["unmapped_same_file_constants"] and not d["nonpreceding_same_file_constants"], "retained dependency mapping incomplete")
        require(set(d["same_target_command_constants"]) <= auxiliaries | {raw["target_name"]}, "unexpanded same-target auxiliary")
        require(set(d["command_ids"]) == set(d["constant_evidence"]), "dependency command/evidence cardinality differs")
        for items in d["constant_evidence"].values():
            require(items, "empty dependency evidence")
            for item in items:
                require(item["source_attribution"] in BASES and declarations[item["constant"]]["module"] == raw["source_module"], "dependency attribution uncertified")
                if item["source_attribution"] in OWNERSHIP_BASES:
                    cert = ownership[item["constant"]]
                    require(cert["basis"] == item["source_attribution"] and cert["owners"], "ownership dump differs")


def attributed_command(name: str, basis: str, declarations: dict[str, dict[str, Any]],
                       ownership: dict[str, dict[str, Any]], commands: list[dict[str, Any]],
                       seen: set[str] | None = None) -> str:
    """Locate a kernel identity by real range, or certified same-command owners."""
    seen = set() if seen is None else set(seen)
    require(name not in seen, "cyclic declaration ownership")
    seen.add(name)
    declaration = declarations[name]
    require(declaration["module"] == commands[0]["source_module"], "kernel declaration source module differs")
    if basis in OWNERSHIP_BASES:
        cert = ownership[name]
        require(cert["basis"] == basis and cert["owners"], "ownership certificate absent or changed")
        owners = set()
        for owner in cert["owners"]:
            d = declarations[owner]
            owner_basis = "kernel_selection_range" if d.get("selection_range") else "kernel_declaration_range" if d.get("range") else ownership[owner]["basis"]
            owners.add(attributed_command(owner, owner_basis, declarations, ownership, commands, seen))
        require(len(owners) == 1, "ownership crosses multiple source commands")
        return next(iter(owners))
    if basis == "exact_source_metadata_name":
        matches = [c["command_id"] for c in commands if c.get("name") == name or c.get("original_name") == name]
        require(len(matches) == 1, "exact source metadata name not independently recoverable")
        return matches[0]
    require(basis in {"kernel_selection_range", "kernel_declaration_range"}, "uncertified source attribution")
    location = declaration.get("selection_range" if basis == "kernel_selection_range" else "range")
    require(isinstance(location, dict), "kernel source range missing")
    point = (location["start_line"], location["start_column"])
    matches = [c["command_id"] for c in commands if
               (c["source_span"]["start_line"], c["source_span"]["start_col"]) <= point <
               (c["source_span"]["end_line"], c["source_span"]["end_col"])]
    require(len(matches) == 1, "kernel range does not identify exactly one source command")
    return matches[0]


def validate_model_pair(target: dict[str, Any], raw_pair: dict[str, Any], encoded: dict[str, Any],
                        model: str, tokenizer: Any, commands: list[dict[str, Any]],
                        profiles: dict[str, dict[str, Any]], exclusion_ids: set[str],
                        exclusion_names: set[str]) -> dict[str, int]:
    no_proof_fields(encoded)
    no_proof_fields(raw_pair)
    selected, ec = ordered(target["mathematical_context"]), ordered(target["environment_context"])
    require(raw_pair["selected_MC"] == selected and raw_pair["shared_EC"] == ec, "original MC/EC changed")
    require(encoded["statement"] == raw_pair["statement"] == target["statement"], "target statement changed")
    result = raw_pair["models"][model]
    pairs, control = result["declaration_pairs"], result["replacement_MC"]
    require(result["eligible"] is True and result["reason"] is None, "retained model target not eligible")
    require(len(pairs) == len(control) == len(selected), "one-to-one replacement cardinality differs")
    require(encoded["declaration_pairs"] == pairs and set(encoded["conditions"]) == set(CONDITIONS), "encoded mapping/conditions differ")
    by_id = {c["command_id"]: c for c in commands}
    forbidden = exclusion_ids | {c["command_id"] for c in selected + ec} | {target["target_command_id"]}
    chosen = [c["command_id"] for c in control]
    require(len(set(chosen)) == len(chosen) and not (set(chosen) & forbidden), "noninjective or forbidden replacement")
    target_start = target["target_source_span"]["start_byte"]
    lengths: dict[str, int] = {}

    def length(text: str) -> int:
        if text not in lengths:
            lengths[text] = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return lengths[text]

    costs = []
    for s, c, pair in zip(selected, control, pairs):
        original = by_id[s["command_id"]]
        source = by_id[c["command_id"]]
        for key in source:
            require(c.get(key) == source[key], f"replacement source record changed: {key}")
        require(s["text"] == original["text"], "selected source text changed")
        require(c["source_file"] == target["source_file"] and c["command_uid"] == target["source_file"] + "::" + c["command_id"], "replacement source file/UID differs")
        require(c["source_span"]["start_byte"] < target_start and c["source_span"]["end_byte"] <= target_start, "replacement not strictly preceding")
        require(command_kind(s) == command_kind(c) == pair["selected_kind"] == pair["replacement_kind"], "exact effective kind differs")
        require(c["introduced_declarations_status"] == "kernel_range_or_certified_ownership_attributed" and c["introduced_declarations"], "replacement identity unresolved")
        require(all(x["basis"] in BASES for x in c["introduced_declarations"]), "replacement ownership uncertified")
        require(not ({x["name"] for x in c["introduced_declarations"]} & exclusion_names), "replacement introduces observed dependency constant")
        a, b = length(s["text"]), length(c["text"])
        require(within_band(a, b) and pair["selected_tokens"] == a and pair["replacement_tokens"] == b, "per-command actual tokenizer band/count differs")
        require(abs(pair["length_ratio"] - b / a) < 1e-12, "length ratio differs")
        sp, cp = profiles[s["command_id"]], profiles[c["command_id"]]
        require(sp["status"] == cp["status"] == "syntactic_valid" and sp["fingerprint"] == cp["fingerprint"] == pair["scope_fingerprint"], "scope fingerprint/status differs")
        for p in (sp, cp):
            payload = {key: p[key] for key in ("profile_version", "namespace_path", "scope_stack", "active_directives", "command_wrappers", "command_attributes", "command_modifiers")}
            require(p["fingerprint"] == object_hash(payload), "scope payload hash differs")
        for key, expected in (("prompt_slot_command_id", s["command_id"]), ("prompt_slot_command_uid", s["command_uid"]), ("prompt_slot_source_order", s["source_order"])):
            require(c[key] == expected, "replacement original slot changed")
        for key, expected in (("selected_command_id", s["command_id"]), ("selected_command_uid", s["command_uid"]), ("replacement_command_id", c["command_id"]), ("replacement_command_uid", c["command_uid"]), ("replacement_source_order", c["source_order"]), ("prompt_slot_source_order", s["source_order"])):
            require(pair[key] == expected, "declaration pairing provenance differs")
        edge = {}
        for candidate in commands:
            cid = candidate["command_id"]
            if (cid in forbidden or candidate["source_file"] != target["source_file"] or candidate["source_span"]["start_byte"] >= target_start or candidate["source_span"]["end_byte"] > target_start or command_kind(candidate) != command_kind(s)):
                continue
            declarations = candidate.get("introduced_declarations") or []
            if (candidate.get("introduced_declarations_status") != "kernel_range_or_certified_ownership_attributed" or not declarations or any(x["basis"] not in BASES for x in declarations) or {x["name"] for x in declarations} & exclusion_names):
                continue
            profile = profiles[cid]
            if profile["status"] == "syntactic_valid" and profile["fingerprint"] == sp["fingerprint"] and within_band(a, length(candidate["text"])):
                edge[cid] = abs(a - length(candidate["text"]))
        costs.append(edge)
    actual_cost = sum(abs(length(s["text"]) - length(c["text"])) for s, c in zip(selected, control))
    require(minimum_cost(costs) == actual_cost == result["assignment_cost"], "assignment is not minimum summed token deviation")
    selected_length = length("\n\n".join(c["text"] for c in selected))
    control_length = length("\n\n".join(c["text"] for c in control))
    require(within_band(selected_length, control_length) and result["selected_MC_tokens"] == selected_length and result["control_MC_tokens"] == control_length, "joined MC actual tokenizer band differs")
    require(encoded["shared_EC_ids"] == [c["command_id"] for c in ec], "shared EC membership differs")
    maxima = {}
    for condition, mc in (("Env", []), ("Selected", selected), ("Control", control)):
        row = encoded["conditions"][condition]
        expected = reconstruct_prompt(target["statement"], ec, mc)
        ids = encode(tokenizer, expected, model)
        require(row["prompt"] == expected and row["input_ids"] == ids, "prompt text/token IDs changed or truncated")
        require(expected.endswith("\n") and not expected.endswith("\n\n"), "prompt boundary newline differs")
        require(row["input_length"] == row["prompt_length"] == len(ids) <= 1024 and row["input_limit"] == 1024, "whole input budget/count differs")
        require(row["output_limit"] == 1024 and row["output_limit_includes_terminal_EOS"] is True, "independent output budget differs")
        require(row["valid_input"] is True and row["reference_proof_in_generator_input"] is False and row["input_ids_contain_prompt_only"] is True, "generator-input safety contract differs")
        require(row["visible_mc_command_ids"] == [c["command_id"] for c in mc] and row["visible_mc_command_uids"] == [c["command_uid"] for c in mc], "visible MC membership differs")
        require(row["visible_ec_command_ids"] == [c["command_id"] for c in ec] and row["visible_ec_command_uids"] == [c["command_uid"] for c in ec], "EC changed across conditions")
        require(row["MC_count"] == len(mc) and row["EC_count"] == len(ec) and row["condition"] == condition, "condition command counts differ")
        maxima[condition] = len(ids)
    return maxima


def validate_source_catalog(commands: list[dict[str, Any]], source_file: str,
                            source: bytes, expected_sha256: str,
                            official: dict[str, Any]) -> None:
    """Check the released parser catalog against the released pinned source.

    This does not pretend to rerun the parser or require private workspace
    paths that are not included in the source dataset.
    """
    require(hashlib.sha256(source).hexdigest() == expected_sha256 == official["sha256"],
            "pinned source bytes/hash differ")
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    require(blob == official["official_git_blob_sha1"], "source differs from pinned official tree")
    seen, previous_end = set(), -1
    for order, c in enumerate(commands):
        cid, span = c["command_id"], c["source_span"]
        require(cid not in seen and c["source_order"] == order, "command UID/order/duplicate differs")
        seen.add(cid)
        require(c["source_file"] == source_file and c["command_uid"] == source_file + "::" + cid,
                "catalog source identity differs")
        a, b = span["start_byte"], span["end_byte"]
        require(0 <= a < b <= len(source) and previous_end <= a and source[a:b] == c["raw_text"].encode(),
                "source command bytes/overlap differ")
        previous_end = b
        require(c["text"] == c["effective_raw_text"].strip(), "full effective command text differs")
        for key, field in (("raw_text_sha256", "raw_text"), ("effective_raw_text_sha256", "effective_raw_text"),
                           ("context_text_sha256", "text")):
            require(c[key] == hashlib.sha256(c[field].encode()).hexdigest(), "catalog text hash differs")


def run(root: Path, extension: Path, audit_root: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from .replacement_scope import scope_profiles

    require(extension != root and root not in extension.parents,
            "output extension must be outside the submitted dataset")
    out = extension / "matched_replacement"
    status, manifest = read_json(out / "build_status.json"), read_json(out / "manifest.json")
    require(status["status"].startswith("complete"), "build not complete; refusing to scan changing outputs")
    population = {r["sample_id"]: r for r in rows(root / "benchmark/test.jsonl") if r["mathematical_context"] and r["environment_context"]}
    sample, final = read_json(out / "sampled_manifest.json"), read_json(out / "frozen_test_manifest.json")
    eligibility = list(rows(out / "eligibility.jsonl"))
    common = validate_sampling(list(population), sample, eligibility, final)
    final_ids, drawn = final["target_ids"], sample["sampled_target_ids"]
    require(sample["source_test_sha256"] == manifest["input_source_test_sha256"] == file_hash(root / "benchmark/test.jsonl"), "input benchmark hash differs")
    common_rows = list(rows(out / "common_eligible_ids.jsonl"))
    require([x["sample_id"] for x in common_rows] == common and all(x["included_in_final"] is (x["sample_id"] in set(final_ids)) for x in common_rows), "common membership artifact differs")
    raw_pairs, targets = index_rows(out / "pairs.jsonl"), index_rows(out / "targets.jsonl")
    require(list(raw_pairs) == list(targets) == final_ids, "raw pair/target membership differs")
    evidence = index_rows(extension / "private_posthoc_audit/matched_replacement_dependency_exclusions.jsonl")
    require(list(evidence) == drawn, "dependency evidence membership/order differs")
    wanted = set(drawn)
    semantic = {r["sample_id"]: r for r in rows(audit_root / "semantic_mc_labels.jsonl") if r["sample_id"] in wanted}
    needed_names = set()
    for r in semantic.values():
        needed_names.add(r.get("target_name"))
        needed_names.update(x["name"] for field in ("expanded_type_auxiliaries", "expanded_value_auxiliaries") for x in r.get(field, []))
        needed_names.update(x["constant"] for field in DEPENDENCY_FIELDS for xs in (r.get(field) or {}).get("constant_evidence", {}).values() for x in xs)
    snapshots = read_json(root / "command_mapping/source_snapshot.json")
    official = read_json(root / "command_mapping/revision_verification.json")
    catalogs = {}
    command_count = 0
    for source_file in sorted({population[sid]["source_file"] for sid in drawn}):
        commands = list(rows(root / "command_mapping/source_commands" / Path(source_file).with_suffix(".jsonl")))
        source = (root / "source_snapshot/mathlib4" / source_file).read_bytes()
        validate_source_catalog(commands, source_file, source, snapshots["source_files"][source_file],
                                official["files"][source_file])
        catalogs[source_file] = (commands, scope_profiles(commands))
        command_count += len(commands)
    for sid in final_ids:
        for model in MODELS:
            needed_names.update(x["name"] for c in raw_pairs[sid]["models"][model]["replacement_MC"] for x in c["introduced_declarations"])
    ownership = {r["name"]: r for r in rows(audit_root / "declaration_ownership.jsonl")}
    while True:
        owners = {owner for name in needed_names if name in ownership for owner in ownership[name]["owners"]}
        if owners <= needed_names:
            break
        needed_names.update(owners)
    declarations = {r["name"]: r for r in rows(audit_root / "declaration_dependencies.jsonl") if r["name"] in needed_names}
    source_hashes = {name: file_hash(audit_root / name) for name in ("semantic_mc_labels.jsonl", "declaration_dependencies.jsonl", "declaration_ownership.jsonl")}
    for sid in drawn:
        e, raw = evidence[sid], semantic.get(sid, {"sample_id": sid})
        for name, expected in source_hashes.items():
            require(e["provenance"]["sources"][name]["sha256"] == expected, "audit dependency source hash differs")
        ids, names = observed_sets(raw, declarations)
        require(set(e["exclusion_command_ids"]) == ids and set(e["exclusion_constant_names"]) == names, "cached exclusion union differs from raw audit/kernel")
        if sid in final_ids:
            require(e["availability_complete"] is True and e["kernel_dump_metadata_verified"] is True, "retained target mapping unavailable")
            check_dependency_certificate(raw, declarations, ownership)
            require(raw["source_file"] == population[sid]["source_file"] and raw["target_command_id"] == population[sid]["target_command_id"], "dependency target provenance differs")
            commands = catalogs[raw["source_file"]][0]
            for field in DEPENDENCY_FIELDS:
                for cid, items in raw[field]["constant_evidence"].items():
                    for item in items:
                        require(attributed_command(item["constant"], item["source_attribution"], declarations, ownership, commands) == cid,
                                "dependency constant attributed to wrong actual source command")
    model_summary = {}
    for model in MODELS:
        acquisition = read_json(root / "tokenizers" / model / "acquisition_manifest.json")
        for name, info in acquisition["files"].items():
            actual = root / "tokenizers" / model / info.get("local_filename", name)
            require(file_hash(actual) == info["sha256"], "actual tokenizer acquisition hash differs")
        tok = AutoTokenizer.from_pretrained(str(root / "tokenizers" / model), local_files_only=True, trust_remote_code=False, use_fast=True)
        require(tok.bos_token_id == (126080 if model == "llada" else 128000) and tok.eos_token_id == (126081 if model == "llada" else 128001), "actual tokenizer special IDs differ")
        base = extension / f"tokenized/{model}/matched_replacement/input_1024"
        encoded = index_rows(base / "pairs.jsonl")
        views = {c: index_rows(base / "views" / c / "test.jsonl") for c in CONDITIONS}
        require(list(encoded) == final_ids and all(list(v) == final_ids for v in views.values()), "tokenized condition/model membership differs")
        maxima = Counter()
        for sid in final_ids:
            target, raw_pair = population[sid], raw_pairs[sid]
            for artifact in (targets[sid], raw_pair, encoded[sid], *(v[sid] for v in views.values())):
                no_proof_fields(artifact)
                require(artifact["sample_id"] == artifact["target_id"] == sid, "target identity differs")
                for key in ("statement", "source_file", "source_module", "target_command_id", "target_command_uid", "target_source_span", "split", "proof_family", "target_kind", "original_name"):
                    require(artifact[key] == target[key], "immutable target field changed: " + key)
            commands, profiles = catalogs[target["source_file"]]
            ids, names = observed_sets(semantic[sid], declarations)
            sizes = validate_model_pair(target, raw_pair, encoded[sid], model, tok, commands, profiles, ids, names)
            for condition in CONDITIONS:
                condition_row = encoded[sid]["conditions"][condition]
                require(all(views[condition][sid].get(k) == value for k, value in condition_row.items()), "individual condition shard differs from pair")
                maxima[condition] = max(maxima[condition], sizes[condition])
            for c in raw_pair["models"][model]["replacement_MC"]:
                for introduced in c["introduced_declarations"]:
                    d = declarations[introduced["name"]]
                    require(d["module"] == target["source_module"], "candidate kernel declaration module differs")
                    require(attributed_command(introduced["name"], introduced["basis"], declarations, ownership, commands) == c["command_id"],
                            "candidate kernel identity attributed to wrong actual source command")
                    if introduced["basis"] in OWNERSHIP_BASES:
                        cert = ownership[introduced["name"]]
                        require(cert["basis"] == introduced["basis"] and cert["owners"], "candidate ownership certificate differs")
        metrics = read_json(base / "tokenization_metrics.json")
        require(metrics["targets"] == len(final_ids) and metrics["input_limit"] == metrics["output_limit"] == 1024 and metrics["prompt_length_max"] == {c: maxima[c] for c in CONDITIONS}, "tokenized metrics differ")
        model_summary[model] = {"targets_checked": len(final_ids), "prompts_reencoded": 3 * len(final_ids), "prompt_length_max": dict(maxima), "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id}
    plan, rules = read_json(out / "experiment_plan.json"), read_json(out / "matching_protocol.json")
    require(plan["targets"] == len(final_ids) and plan["candidate_attempts"] == 32 and plan["input_limit"] == plan["output_limit"] == 1024, "future generation population/attempts/budgets differ")
    require(plan["same_frozen_targets_in_all_comparisons"] is True and set(plan["models"]) == set(MODELS) and set(plan["input_conditions"]) == set(CONDITIONS), "comparison plan membership differs")
    require(rules["implementation_defaults"]["MC_EC_or_statement_truncation"] is False and rules["implementation_defaults"]["reference_proof_length_used_for_inference_eligibility"] is False, "truncation/reference length protocol changed")
    selector_ids = set(read_json(root / "selector/frozen_test_manifest.json")["target_ids"])
    require(final["selector_overlap_count"] == len(selector_ids & set(final_ids)) and final["selector_manifest_separate"] is True, "separate selector overlap accounting differs")
    require(final["families"] == dict(Counter(population[sid]["proof_family"] for sid in final_ids)), "final family counts differ")
    for name, info in manifest["artifacts"].items():
        p = extension / name
        require(p.is_file() and p.stat().st_size == info["bytes"] and file_hash(p) == info["sha256"], "matched artifact manifest/hash differs: " + name)
    for name, expected in manifest["processing_script_sha256"].items():
        require(file_hash(Path(__file__).with_name(name)) == expected, "builder processing source hash differs")
    return {"schema_version": "independent-matched-replacement-validation-v1", "valid": True,
            "sampled_targets_checked": 2500, "population_targets": 11064,
            "common_eligible_targets": len(common), "actual_final_targets": len(final_ids),
            "requested_size_reached": len(final_ids) == 2000, "source_files_checked": len(catalogs),
            "source_commands_checked": command_count, "models": model_summary,
            "checks": {"exact_seeded_sampling_and_no_backfill": True, "common_model_and_condition_membership": True,
                       "source_bytes_pinned_hashes_and_command_spans": True, "raw_kernel_dependency_union_and_certificates": True,
                       "injective_exact_kind_and_true_token_length_bands": True, "minimum_assignment_cost_independently_checked": True,
                       "unchanged_EC_target_original_MC_slots_no_truncation": True, "complete_prompt_tokens_with_correct_BOS_and_budget": True,
                       "generator_rows_contain_no_reference_proof_fields": True, "all_declared_artifact_hashes": True},
            "independence": {"builder_matching_prompt_sampling_or_assignment_helpers_imported": False,
                             "dependency_builder_helpers_imported": False,
                             "scope_parser_shared": "dataset_preparation.replacement_scope.scope_profiles",
                             "scope_payload_hashes_and_pair_equality_rebuilt": True},
            "limitations": ["Source catalog is checked against released pinned bytes; the Lean parser is not rerun.",
                            "Scope recognition is shared syntactic code, not independent Lean elaboration.",
                            "Observed dependency exclusion does not certify mathematical irrelevance.",
                            "Alternative assignments after a serialized-budget failure are not searched; a rejected target may have another feasible matching.",
                            "No model generation, checkpoint equality, decoding settings or Pass@32 result was verified.",
                            "Unavailable target rejection labels are not exhaustively recomputed; every retained target is checked."],
            "dependency_source_sha256": source_hashes, "validator_sha256": file_hash(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, help="Defaults to DATASET/private_posthoc_audit/kernel_inputs")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    extension = args.output_root.resolve()
    if extension == root or root in extension.parents:
        raise ValueError("--output-root must be outside --dataset-root")
    report_path = extension / "matched_replacement/validation_report.json"
    try:
        report = run(root, extension, (args.audit_root or root / "private_posthoc_audit/kernel_inputs").resolve())
    except Exception as exc:
        report = {"schema_version": "independent-matched-replacement-validation-v1", "valid": False,
                  "error": f"{type(exc).__name__}: {exc}", "validator_sha256": file_hash(Path(__file__))}
    report["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
