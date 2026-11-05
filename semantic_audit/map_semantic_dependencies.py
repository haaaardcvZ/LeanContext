"""Join pinned Lean kernel dependencies to exact original source commands.

Main descriptor: constants(value) minus constants(type), restricted to earlier
mathematical declaration commands in the same original source file. This is a
reference-expression descriptor, not a claim that every possible proof needs it.
"""
from __future__ import annotations
import bisect
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def source_position(r, stem):
    return r[stem + "_line"], r[stem + "_col"]

def run():
    inputs = list(read_jsonl(ROOT / "semantic_inputs.jsonl"))
    targets_by_cmd = collections.defaultdict(list)
    target_exact_names = collections.defaultdict(set)
    source_canonical_names = {}
    for item in read_jsonl(ROOT / "source_canonical_name_candidates.jsonl"):
        if item.get("scope_stack_valid"):
            source_canonical_names[(item["source_module"], item["command_id"])] = set(item["canonical_name_candidates"])
    commands_by_module = collections.defaultdict(list)
    command_by_key = {}
    source_name_candidates = collections.defaultdict(set)
    for inp in inputs:
        key = (inp["source_module"], inp["target_command_id"])
        targets_by_cmd[key].append(inp)
        for n in inp.get("target_full_name_candidates", []):
            target_exact_names[key].add(n["full_name"])
    for cmd in read_jsonl(ROOT / "semantic_source_commands.jsonl"):
        key = (cmd["source_module"], cmd["command_id"])
        command_by_key[key] = cmd
        commands_by_module[cmd["source_module"]].append(cmd)
        for hint in cmd.get("full_name_candidates", []):
            source_name_candidates[(cmd["source_module"], hint["full_name"])].add(cmd["command_id"])
    starts = {}
    for mod, cmds in commands_by_module.items():
        cmds.sort(key=lambda c: source_position(c, "start"))
        starts[mod] = [source_position(c, "start") for c in cmds]

    def position_command(mod, pos):
        if mod not in starts or pos is None: return None
        where = (pos["start_line"], pos["start_column"])
        index = bisect.bisect_right(starts[mod], where) - 1
        if index < 0: return None
        cmd = commands_by_module[mod][index]
        if source_position(cmd, "start") <= where < source_position(cmd, "end"):
            return cmd["command_id"]
        return None

    # name -> (module, command_id or None, attribution basis, kind, has_value)
    metadata = {}
    target_candidates = collections.defaultdict(dict)
    own_command_values = collections.defaultdict(dict)
    metadata_counts = collections.Counter()
    for index, row in enumerate(read_jsonl(ROOT / "declaration_dependencies.jsonl"), 1):
        mod, name = row["module"], row["name"]
        command = position_command(mod, row["selection_range"])
        basis = "kernel_selection_range"
        if command is None:
            command = position_command(mod, row["range"])
            basis = "kernel_declaration_range"
        if command is None:
            hints = source_name_candidates.get((mod, name), set())
            if len(hints) == 1:
                command = next(iter(hints)); basis = "exact_source_metadata_name"
            else: basis = "unmapped"
        metadata[name] = (mod, command, basis, row["kind"], row["has_value"])
        metadata_counts[basis] += 1
        key = (mod, command)
        if key in targets_by_cmd and row["has_value"]:
            own_command_values[key][name] = row
        if key in targets_by_cmd and row["kind"] == "theorem" and row["has_value"]:
            target_candidates[key][name] = row
        if index % 200000 == 0: print(f"Indexed {index} kernel declarations.", flush=True)

    # Only compiler-certified ownership may recover a missing declaration range.
    # A dotted namespace prefix by itself is deliberately not sufficient.
    ownership = list(read_jsonl(ROOT / "declaration_ownership.jsonl"))
    for _ in range(3):
        changed = 0
        for owner in ownership:
            name = owner["name"]; meta = metadata.get(name)
            if not meta or meta[1] is not None or meta[0] not in starts: continue
            parent_meta = [metadata.get(p) for p in owner["owners"]]
            if not parent_meta or any(not p or p[0] != meta[0] or p[1] is None for p in parent_meta): continue
            ids = {p[1] for p in parent_meta}
            if len(ids) == 1:
                metadata[name] = (meta[0], next(iter(ids)), owner["basis"], meta[3], meta[4]); changed += 1
        if not changed: break
    final_metadata_counts = collections.Counter(m[2] for m in metadata.values())

    def resolve_target(inp):
        key = (inp["source_module"], inp["target_command_id"])
        candidates = target_candidates.get(key, {})
        # Independent complete source scope reconstruction only disambiguates
        # declarations already certified to occur in this exact source command.
        canonical = [r for n, r in candidates.items() if n in source_canonical_names.get(key, set())]
        if len(canonical) == 1: return canonical[0], "kernel_range_and_complete_source_namespace", []
        hints = target_exact_names[key]
        exact = [r for n, r in candidates.items() if n in hints]
        if len(exact) == 1: return exact[0], "kernel_range_and_exact_source_name", []
        raw = (inp.get("target_raw_name") or "").removeprefix("_root_.")
        suffix = [r for n, r in candidates.items() if raw and (n == raw or n.endswith("." + raw))]
        if len(suffix) == 1: return suffix[0], "kernel_range_and_source_name_suffix", []
        if len(candidates) == 1: return next(iter(candidates.values())), "unique_kernel_theorem_in_source_command", []
        return None, "unresolved_target_identity", sorted(candidates)

    def map_refs(inp, refs):
        found = collections.defaultdict(list)
        unknown, own, later, other_local = [], [], [], []
        for name in sorted(refs):
            md = metadata.get(name)
            # The dump contains every Mathlib-module constant in the imported
            # environment. Missing names therefore belong to imported libraries
            # outside Mathlib and cannot be in this source file's local prefix.
            if not md or md[0] != inp["source_module"]: continue
            if md[1] is None:
                unknown.append(name); continue
            cmd = command_by_key[(md[0], md[1])]
            if md[1] == inp["target_command_id"]:
                own.append(name); continue
            if not (cmd["start_byte"] < inp["target_start_byte"] and cmd["end_byte"] <= inp["target_start_byte"]):
                later.append(name); continue
            if not cmd["is_mathematical_command"]:
                other_local.append({"name": name, "command_id": md[1], "kind": cmd["effective_kind"]}); continue
            found[md[1]].append({"constant": name, "source_attribution": md[2], "declaration_kind": md[3]})
        return {"command_ids": sorted(found), "constant_evidence": dict(found),
                "unmapped_same_file_constants": unknown, "same_target_command_constants": own,
                "nonpreceding_same_file_constants": later, "other_local_commands": other_local}

    def projection_names(rows):
        return {p["projection_name"] or p["structure"] for p in rows}

    def expand_own_auxiliaries(inp, target_name, initial_names, include_projections=False):
        """Undo same-command compilation wrappers, never earlier declarations."""
        own_values = own_command_values[(inp["source_module"], inp["target_command_id"])]
        result = set(initial_names)
        queue = list(result); expanded = []
        visited = set()
        while queue:
            name = queue.pop()
            if name in visited: continue
            visited.add(name)
            if name == target_name: continue
            value = own_values.get(name)
            if value is None: continue
            # Every own_values row was range-attributed or uniquely identified
            # by exact source metadata, not by a guessed namespace prefix.
            extra = set(value["value_constants"])
            if include_projections: extra |= projection_names(value["value_projections"])
            expanded.append({"name": name, "source_attribution": metadata[name][2]})
            queue.extend(extra - result); result.update(extra)
        return result, expanded

    results_path = ROOT / "semantic_mc_labels.jsonl"
    csv_path = ROOT / "semantic_mc_labels.csv"
    count = collections.defaultdict(collections.Counter)
    unknown_examples = []
    csv_fields = ["sample_id", "split", "source_file", "target_name", "target_identity_status", "mc_semantic_status", "mc_dependency_group", "d_m_value_minus_type", "proof_local_commands", "type_local_commands", "proof_extra_constant_commands", "proof_extra_command_commands", "selected_extra_commands", "candidate_extra_commands", "prefilter_extra_commands", "removed_extra_commands", "selected_extra_coverage", "unmapped_extra_constants", "self_auxiliary_extra_constants"]
    with open(results_path, "w", encoding="utf-8") as out, open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields); writer.writeheader()
        for inp in inputs:
            split = inp["split"]; count[split]["N"] += 1
            target, identity_status, ambiguous = resolve_target(inp)
            row = {"sample_id": inp["sample_id"], "split": split, "source_file": inp["source_file"], "source_module": inp["source_module"], "target_command_id": inp["target_command_id"], "target_identity_status": identity_status, "reference_proof_exact_raw_body_match": inp["reference_proof_exact_raw_body_match"]}
            if target is None:
                row.update(target_name=None, mc_semantic_status="unresolved_target", d_m_value_minus_type=None, target_candidates=ambiguous, target_kind=inp["target_kind"])
                count[split]["unresolved_target"] += 1
                count[split]["unresolved_target_kind_" + inp["target_kind"]] += 1
            else:
                outer_type_names = set(target["type_constants"])
                outer_value_names = set(target["value_constants"])
                type_names, type_auxiliaries = expand_own_auxiliaries(inp, target["name"], outer_type_names)
                value_names, value_auxiliaries = expand_own_auxiliaries(inp, target["name"], outer_value_names)
                extra_names = value_names - type_names
                mt = map_refs(inp, type_names); mv = map_refs(inp, value_names); mx = map_refs(inp, extra_names)
                type_plus, _ = expand_own_auxiliaries(inp, target["name"], outer_type_names | projection_names(target["type_projections"]), True)
                value_plus, _ = expand_own_auxiliaries(inp, target["name"], outer_value_names | projection_names(target["value_projections"]), True)
                mxproj = map_refs(inp, value_plus - type_plus)
                selected = set(inp["selected_mc_command_ids"])
                candidate = set(inp["selector_candidate_command_ids"])
                prefilter = set(inp["selected_command_ids_before_context_filter"])
                removed = set(inp["removed_context_command_ids"])
                xcmds = set(mx["command_ids"])
                bad = bool(mx["unmapped_same_file_constants"] or mx["nonpreceding_same_file_constants"])
                status = "unresolved_source_mapping" if bad else "complete"
                dm = True if xcmds else (None if bad else False)
                count[split][status] += 1
                count[split]["positive" if dm is True else "negative" if dm is False else "unknown"] += 1
                count[split]["positive_incomplete_mapping"] += bool(dm is True and bad)
                count[split]["projection_augmented_positive"] += bool(mxproj["command_ids"])
                row.update(target_name=target["name"], mc_semantic_status=status, d_m_value_minus_type=dm,
                    proof_local_commands=len(mv["command_ids"]), type_local_commands=len(mt["command_ids"]),
                    proof_extra_constant_commands=len(xcmds), proof_extra_command_commands=len(set(mv["command_ids"])-set(mt["command_ids"])),
                    selected_extra_commands=len(xcmds & selected), candidate_extra_commands=len(xcmds & candidate),
                    prefilter_extra_commands=len(xcmds & prefilter), removed_extra_commands=len(xcmds & removed),
                    selected_extra_coverage=(len(xcmds & selected)/len(xcmds) if xcmds and not bad else None),
                    unmapped_extra_constants=len(mx["unmapped_same_file_constants"]), self_auxiliary_extra_constants=len(mx["same_target_command_constants"]),
                    selected_mc_command_ids=sorted(selected), type_dependency=mt, value_dependency=mv, proof_extra_dependency=mx,
                    projection_augmented_proof_extra_dependency=mxproj,
                    outer_type_constants=sorted(outer_type_names), outer_value_constants=sorted(outer_value_names),
                    expanded_type_auxiliaries=type_auxiliaries, expanded_value_auxiliaries=value_auxiliaries,
                    type_constants=sorted(type_names), value_constants=sorted(value_names), value_minus_type_constants=sorted(extra_names))
                count[split]["targets_with_expanded_value_auxiliaries"] += bool(value_auxiliaries)
                count[split]["targets_with_self_auxiliaries"] += bool(mx["same_target_command_constants"])
                if bad and len(unknown_examples)<40: unknown_examples.append(row)
                if not bad:
                    count[split]["dependency_commands"] += len(xcmds)
                    count[split]["selected_dependency_commands"] += len(xcmds & selected)
                    count[split]["candidate_dependency_commands"] += len(xcmds & candidate)
                    count[split]["prefilter_dependency_commands"] += len(xcmds & prefilter)
                    count[split]["removed_dependency_commands"] += len(xcmds & removed)
                    if xcmds:
                        count[split]["complete_positive"] += 1
                        count[split]["complete_positive_selected_coverage_sum"] += len(xcmds & selected) / len(xcmds)
                        count[split]["dependent_targets_selected_any"] += bool(xcmds & selected)
                        count[split]["dependent_targets_selected_all"] += xcmds <= selected
            dm = row["d_m_value_minus_type"]
            row["mc_dependency_group"] = "observed_proof_local_dependency" if dm is True else "no_observed_direct_proof_local_dependency" if dm is False else "uncertain"
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            writer.writerow({k:row.get(k) for k in csv_fields})
    exclusive = {}; coverage = {}
    for split, c in count.items():
        exclusive[split] = {"N":c["N"], "observed_proof_local_dependency":c["positive"], "no_observed_direct_proof_local_dependency":c["negative"], "uncertain":c["N"]-c["positive"]-c["negative"], "positive_with_incomplete_other_dependency_mapping":c["positive_incomplete_mapping"]}
        assert sum(exclusive[split][k] for k in ["observed_proof_local_dependency", "no_observed_direct_proof_local_dependency", "uncertain"]) == c["N"]
        coverage[split] = {"population":"complete_positive_only", "targets":c["complete_positive"], "dependency_commands":c["dependency_commands"], "selected_dependency_commands":c["selected_dependency_commands"], "candidate_dependency_commands":c["candidate_dependency_commands"], "prefilter_dependency_commands":c["prefilter_dependency_commands"], "removed_dependency_commands":c["removed_dependency_commands"], "selected_micro_coverage":c["selected_dependency_commands"]/c["dependency_commands"] if c["dependency_commands"] else None, "selected_mean_target_coverage":c["complete_positive_selected_coverage_sum"]/c["complete_positive"] if c["complete_positive"] else None, "targets_selected_any":c["dependent_targets_selected_any"], "targets_selected_all":c["dependent_targets_selected_all"]}
    report = {"version":"kernel-mc-dependencies-v2.1", "method":"kernel_value_constants_minus_target_type_constants after expanding only source-certified same-target-command auxiliary values; same-source-file strictly preceding mathematical command; no unfolding earlier user declarations", "exclusive_label_counts":exclusive, "coverage_complete_positive":coverage, "counts":{s:dict(c) for s,c in count.items()},"kernel_metadata_attribution_before_certified_owner_recovery":dict(metadata_counts),"kernel_metadata_attribution_final":dict(final_metadata_counts),"kernel_declarations":len(metadata),"dump_sha256":hashlib.sha256((ROOT/'declaration_dependencies.jsonl').read_bytes()).hexdigest(),"mapper_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_canonical_candidates_sha256":hashlib.sha256((ROOT/'source_canonical_name_candidates.jsonl').read_bytes()).hexdigest(),"unknown_examples":unknown_examples}
    (ROOT / "semantic_mc_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report["counts"], indent=2))

if __name__ == "__main__": run()
