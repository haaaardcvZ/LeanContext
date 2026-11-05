#!/usr/bin/env python3
"""Aggregate real verification, mapped dependency, and proof-reuse records.

This module does not infer semantic dependencies from lexical name overlap.
Reuse requires recorded kernel-proof-term or elaboration-trace evidence mapped
to a supplied command UID. No verified outputs, dependency records, bootstrap
results, or paper table values are supplied or fabricated by this module.

Paired CLI uses outputs from lean_proof_training/evaluation/verify.py:
  python -m analysis.evidence_analysis paired --targets benchmark/test.jsonl \
      --left env_verified.jsonl --right full_verified.jsonl --both-only \
      --output paired_context_effects.json

Coverage CLI consumes one audited row per target with fields:
  sample_id, dependency_mapping_complete, final_visibility_known,
  eligible_target_dependency_command_uids, visible_mc_command_uids.
The UID sets must come from elaborated constants/source-command mappings and
actual final serialized inputs, respectively, not names found in a raw prompt.

Reuse evidence can be attached to a verification row as ``resource_evidence``:
  [{"command_uid": "Mathlib/...lean::cmd_000001", "constant_name": "Foo.bar",
    "basis": "kernel_proof_term", "used_in_generated_proof": true,
    "target_type_only": false, "explicit_source_reference": true}]
This is an input evidence schema, not a claim that the archived predictions
already contain semantic traces. Collect semantic evidence before reporting
explicit or elaboration-only reuse in the paper.
For paired reuse analysis, target rows must also provide
``visible_mc_command_uids`` from the actual CFull serialized input. Original
unclipped mathematical_context is not an adequate substitute. If visibility is
missing, the target's reuse status is unresolved, not zero reuse.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from .representation_analysis import iter_jsonl, write_json


def pass_at_k(correct: int, attempts: int, k: int) -> float:
    if not (0 <= correct <= attempts and 1 <= k <= attempts):
        raise ValueError("Require 0 <= verified <= attempts and 1 <= k <= attempts")
    if attempts - correct < k:
        return 1.0
    return 1 - math.comb(attempts - correct, k) / math.comb(attempts, k)


def verified_attempt(row: dict) -> bool:
    """Use actual verifier outcome, never a dataset's reference verification flag."""
    if row.get("paper_error_type") == "infrastructure" or row.get("error_type") in {"infrastructure_error", "task_construction_error"}:
        raise ValueError("Infrastructure failures invalidate evaluation; they are not model failures")
    if row.get("passed") is True and row.get("exit_code") != 0:
        raise ValueError("A passed candidate requires a recorded zero Lean exit code")
    return row.get("passed") is True and row.get("exit_code") == 0


def reuse_category(row: dict, supplied_mc_uids: set[str]) -> str | None:
    if not verified_attempt(row):
        return None
    category = None
    for evidence in row.get("resource_evidence", []):
        if evidence.get("command_uid") not in supplied_mc_uids:
            continue
        if evidence.get("basis") not in {"kernel_proof_term", "elaboration_trace"}:
            continue
        if evidence.get("used_in_generated_proof") is not True:
            continue
        if evidence.get("target_type_only") is not False:
            continue
        if not evidence.get("constant_name"):
            raise ValueError("Semantic resource evidence needs a mapped constant name")
        if evidence.get("explicit_source_reference") is True:
            return "explicit_reuse"
        category = "elaboration_only_reuse"
    return category


def target_coverage(row: dict) -> tuple[str, float | None]:
    if row.get("dependency_mapping_complete") is not True or row.get("final_visibility_known") is not True:
        return "unresolved", None
    deps = set(row["eligible_target_dependency_command_uids"])
    visible = set(row["visible_mc_command_uids"])
    if not deps:
        return "no_local_dependencies", None
    retained = len(deps & visible)
    value = retained / len(deps)
    return ("zero" if retained == 0 else "complete" if retained == len(deps) else "partial"), value


def aggregate_coverage(rows) -> dict:
    seen, groups, values = set(), Counter(), []
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate coverage target")
        seen.add(row["sample_id"])
        group, value = target_coverage(row)
        groups[group] += 1
        if value is not None:
            values.append(value)
    return {
        "targets": len(seen), "groups": dict(groups), "auditable_targets": len(values),
        "macro_coverage_percent": 100 * sum(values) / len(values) if values else None,
        "coverage_is_target_weighted_not_dependency_occurrence_weighted": True,
        "no_local_dependencies_and_unresolved_excluded_from_macro": True,
    }


def read_attempts(path: Path, target_ids: set[str], attempts: int) -> dict:
    result, seen = defaultdict(list), set()
    for row in iter_jsonl(path):
        if row.get("mode", "sampled") != "sampled":
            continue
        sample_id = row["sample_id"]
        if sample_id not in target_ids:
            continue
        key = (sample_id, row["candidate_id"])
        if key in seen:
            raise ValueError("Duplicate candidate in verification outputs")
        seen.add(key)
        verified_attempt(row)
        result[sample_id].append(row)
        if len(result[sample_id]) > attempts:
            raise ValueError("More candidates than the declared attempt budget")
    return result


def source_file_bootstrap(differences: list[float], sources: list[str], draws: int, seed: int):
    """Paired cluster bootstrap: resample source files, retain all their targets."""
    import numpy as np

    clusters = defaultdict(list)
    for difference, source in zip(differences, sources):
        clusters[source].append(difference)
    if not clusters or draws < 1:
        return None
    names = sorted(clusters)
    sums = np.array([sum(clusters[name]) for name in names], dtype=float)
    counts = np.array([len(clusters[name]) for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for draw in range(draws):
        selected = rng.integers(0, len(names), size=len(names))
        values[draw] = sums[selected].sum() / counts[selected].sum()
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"lower_pp": 100 * float(lower), "upper_pp": 100 * float(upper),
            "draws": draws, "seed": seed, "resampling_unit": "source_file"}


def aggregate_paired(targets: list[dict], left: dict, right: dict, attempts=32, draws=0, seed=42) -> dict:
    if not targets:
        raise ValueError("A fixed, nonempty target denominator is required")
    output = {"targets": len(targets), "attempts_per_target": attempts,
              "missing_candidates_count_as_failed": True, "metrics": {}}
    newly, lost, reuse = 0, 0, Counter()
    sources = [row["source_file"] for row in targets]
    for k in (1, 8, 16, 32):
        if k > attempts:
            continue
        ls, rs = [], []
        for target in targets:
            sample_id = target["sample_id"]
            ls.append(pass_at_k(sum(verified_attempt(r) for r in left.get(sample_id, [])), attempts, k))
            rs.append(pass_at_k(sum(verified_attempt(r) for r in right.get(sample_id, [])), attempts, k))
        differences = [b-a for a, b in zip(ls, rs)]
        output["metrics"][f"pass_at_{k}"] = {
            "left_percent": 100 * sum(ls) / len(ls), "right_percent": 100 * sum(rs) / len(rs),
            "difference_pp": 100 * sum(differences) / len(differences),
            "paired_95_percent_ci": source_file_bootstrap(differences, sources, draws, seed),
        }
    for target in targets:
        sample_id = target["sample_id"]
        lsolved = any(verified_attempt(r) for r in left.get(sample_id, []))
        rsolved = any(verified_attempt(r) for r in right.get(sample_id, []))
        newly += rsolved and not lsolved
        lost += lsolved and not rsolved
        if rsolved and not lsolved:
            if "visible_mc_command_uids" not in target:
                reuse["visibility_not_supplied"] += 1
                continue
            supplied = set(target["visible_mc_command_uids"])
            verified_rows = [row for row in right.get(sample_id, []) if verified_attempt(row)]
            categories = {reuse_category(row, supplied) for row in verified_rows}
            category = "explicit_reuse" if "explicit_reuse" in categories else (
                "elaboration_only_reuse" if "elaboration_only_reuse" in categories else
                "no_recorded_reuse" if all(isinstance(row.get("resource_evidence"), list) for row in verified_rows) else
                "semantic_evidence_not_supplied"
            )
            reuse[category] += 1
    any_reuse = reuse["explicit_reuse"] + reuse["elaboration_only_reuse"]
    output["paired_solved_targets"] = {"newly_solved": int(newly), "lost": int(lost),
        "net_gain": int(newly-lost), "newly_solved_reuse": dict(reuse),
        "recorded_reuse_evaluable_newly_solved": int(newly-reuse["visibility_not_supplied"]-reuse["semantic_evidence_not_supplied"]),
        "recorded_reuse_share_percent": 100 * any_reuse / newly if newly and not reuse["visibility_not_supplied"] and not reuse["semantic_evidence_not_supplied"] else None,
        "absence_of_recorded_reuse_does_not_imply_absence_of_use": True}
    output["candidate_outcomes"] = {
        name: dict(Counter(row.get("error_type", "unknown") if not verified_attempt(row) else "verified"
                           for rows in condition.values() for row in rows))
        for name, condition in (("left", left), ("right", right))
    }
    output["unsubmitted_or_missing_attempts"] = {
        name: attempts * len(targets) - sum(len(rows) for rows in condition.values())
        for name, condition in (("left", left), ("right", right))
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    coverage = sub.add_parser("coverage")
    coverage.add_argument("--audited-rows", required=True)
    coverage.add_argument("--output", required=True)
    paired = sub.add_parser("paired")
    paired.add_argument("--targets", required=True)
    paired.add_argument("--left", required=True)
    paired.add_argument("--right", required=True)
    paired.add_argument("--both-only", action="store_true")
    paired.add_argument("--attempts", type=int, default=32)
    paired.add_argument("--bootstrap-draws", type=int, default=10000)
    paired.add_argument("--bootstrap-seed", type=int, default=42)
    paired.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "coverage":
        value = aggregate_coverage(iter_jsonl(Path(args.audited_rows)))
    else:
        targets = list(iter_jsonl(Path(args.targets)))
        if args.both_only:
            targets = [r for r in targets if r["mathematical_context"] and r["environment_context"]]
        ids = {r["sample_id"] for r in targets}
        if len(ids) != len(targets):
            raise ValueError("Duplicate denominator targets")
        left = read_attempts(Path(args.left), ids, args.attempts)
        right = read_attempts(Path(args.right), ids, args.attempts)
        value = aggregate_paired(targets, left, right, args.attempts, args.bootstrap_draws, args.bootstrap_seed)
    write_json(Path(args.output), value)


if __name__ == "__main__":
    main()
