"""Pass@1/8/16/32, deterministic accuracy, grouping, and uncertainty."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable


K_VALUES = (1, 8, 16, 32)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator 1-C(n-c,k)/C(n,k)."""
    if n < k or n <= 0:
        raise ValueError(f"pass@{k} requires n >= {k}, found n={n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    failure = 1.0
    for index in range(k):
        failure *= (n - c - index) / (n - index)
    return 1.0 - failure


def mean_stats(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"n": 0, "mean": None, "std": None, "se": None, "ci95": None}
    mean = statistics.fmean(data)
    std = statistics.stdev(data) if len(data) > 1 else 0.0
    se = std / math.sqrt(len(data))
    return {
        "n": len(data),
        "mean": mean,
        "std": std,
        "se": se,
        "ci95": [max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)],
    }


def validate_coverage(
    results: list[dict[str, Any]], index: dict[str, dict[str, Any]]
) -> None:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in results:
        grouped[(str(row["sample_id"]), str(row["mode"]))].append(
            int(row["candidate_id"])
        )
    if {sample_id for sample_id, _ in grouped} != set(index):
        missing = sorted(set(index) - {sample_id for sample_id, _ in grouped})
        extra = sorted({sample_id for sample_id, _ in grouped} - set(index))
        raise ValueError(f"result coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
    for sample_id in sorted(index):
        sampled = sorted(grouped.get((sample_id, "sampled"), []))
        greedy = sorted(grouped.get((sample_id, "greedy_temperature_0"), []))
        if sampled != list(range(32)):
            raise ValueError(f"{sample_id}: sampled candidate IDs are not exactly 0..31")
        if greedy != [0]:
            raise ValueError(f"{sample_id}: greedy candidate IDs are not exactly [0]")


def per_sample_rows(
    results: list[dict[str, Any]], index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(str(row["sample_id"]), str(row["mode"]))].append(row)
    output: list[dict[str, Any]] = []
    for sample_id in sorted(index):
        sampled = grouped[(sample_id, "sampled")]
        greedy = grouped[(sample_id, "greedy_temperature_0")]
        n = len(sampled)
        c = sum(bool(row["passed"]) for row in sampled)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "proof_family": str(index[sample_id]["proof_family"]),
            "sampled_n": n,
            "sampled_c": c,
            "sampled_unique_proofs": len(
                {str(item.get("proof_hash") or "") for item in sampled}
            ),
            "greedy_temperature_0_passed": bool(greedy[0]["passed"]),
        }
        for k in K_VALUES:
            row[f"pass@{k}"] = pass_at_k(n, c, k)
        output.append(row)
    return output


def summarize_group(
    samples: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    sampled_candidates = [row for row in candidate_rows if row["mode"] == "sampled"]
    greedy_candidates = [
        row for row in candidate_rows if row["mode"] == "greedy_temperature_0"
    ]
    result: dict[str, Any] = {
        "samples": len(samples),
        "greedy_temperature_0": mean_stats(
            float(row["greedy_temperature_0_passed"]) for row in samples
        ),
        "sampled_candidate_pass_rate": (
            sum(bool(row["passed"]) for row in sampled_candidates)
            / len(sampled_candidates)
            if sampled_candidates
            else None
        ),
        "mean_unique_sampled_proofs": (
            statistics.fmean(float(row["sampled_unique_proofs"]) for row in samples)
            if samples
            else None
        ),
        "pass_at_k": {
            str(k): mean_stats(float(row[f"pass@{k}"]) for row in samples)
            for k in K_VALUES
        },
        "error_types": {
            "sampled": dict(
                Counter(
                    str(row["error_type"])
                    for row in sampled_candidates
                    if not row["passed"]
                ).most_common()
            ),
            "greedy_temperature_0": dict(
                Counter(
                    str(row["error_type"])
                    for row in greedy_candidates
                    if not row["passed"]
                ).most_common()
            ),
        },
        "paper_error_types": {
            mode: dict(Counter(str(row.get("paper_error_type", "unclassified"))
                               for row in candidate_rows
                               if row["mode"] == mode and not row["passed"]))
            for mode in ("sampled", "greedy_temperature_0")
        },
    }
    return result


def summarize(
    results: list[dict[str, Any]], index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    validate_coverage(results, index)
    samples = per_sample_rows(results, index)
    samples_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        samples_by_family[str(row["proof_family"])].append(row)
    for row in results:
        family = str(index[str(row["sample_id"])]["proof_family"])
        candidates_by_family[family].append(row)
    by_family = {
        family: summarize_group(samples_by_family[family], candidates_by_family[family])
        for family in sorted(samples_by_family)
    }
    non_tactic = [row for row in samples if row["proof_family"] == "Strict Term"]
    tactic = [row for row in samples if row["proof_family"] != "Strict Term"]
    non_tactic_ids = {row["sample_id"] for row in non_tactic}
    tactic_ids = {row["sample_id"] for row in tactic}
    return {
        "protocol": {
            "sampled_candidates": 32,
            "k_values": list(K_VALUES),
            "pass_at_k_estimator": "1 - C(n-c,k) / C(n,k)",
            "deterministic_mode": "greedy argmax; temperature=0 semantics; one candidate",
        },
        "overall": summarize_group(samples, results),
        "by_proof_family": by_family,
        "aggregates": {
            "Non-Tactic": summarize_group(
                non_tactic,
                [row for row in results if str(row["sample_id"]) in non_tactic_ids],
            ),
            "Tactic-Containing": summarize_group(
                tactic,
                [row for row in results if str(row["sample_id"]) in tactic_ids],
            ),
        },
        "per_sample": samples,
    }
