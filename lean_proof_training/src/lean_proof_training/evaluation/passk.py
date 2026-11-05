"""Coverage-aware pass@k and proof-family metrics."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable


PROOF_FAMILIES = ("Strict Term", "Ordinary Tactic", "Hybrid", "Calc")


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased estimator 1-C(n-c,k)/C(n,k); undefined when n < k."""
    if n < k or n <= 0:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # The product form avoids very large intermediate binomial coefficients.
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


def build_per_sample(
    results: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
    k_values: list[int],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["sample_id"])].append(row)
    output = []
    for sample_id in sorted(index):
        items = sorted(grouped.get(sample_id, []), key=lambda row: int(row["candidate_id"]))
        n = len(items)
        c = sum(bool(item.get("passed")) for item in items)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "proof_family": str(index[sample_id]["proof_family"]),
            "n": n,
            "c": c,
            "any_passed": c > 0,
            "unique_proofs": len({str(item.get("proof_hash") or "") for item in items}),
        }
        for k in k_values:
            row[f"top@{k}"] = (
                any(bool(item.get("passed")) for item in items[:k]) if n >= k else None
            )
            row[f"pass@{k}"] = pass_at_k(n, c, k)
        output.append(row)
    return output


def summarize_group(
    sample_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    k_values: list[int],
) -> dict[str, Any]:
    candidate_count = len(candidate_rows)
    passed_candidates = sum(bool(row.get("passed")) for row in candidate_rows)
    result: dict[str, Any] = {
        "samples": len(sample_rows),
        "samples_with_candidates": sum(int(row["n"]) > 0 for row in sample_rows),
        "candidate_count": candidate_count,
        "passed_candidates": passed_candidates,
        "candidate_pass_rate": passed_candidates / candidate_count if candidate_count else None,
        "mean_candidates_per_sample": (
            sum(int(row["n"]) for row in sample_rows) / len(sample_rows) if sample_rows else None
        ),
        "mean_unique_proofs_per_sample": (
            sum(int(row["unique_proofs"]) for row in sample_rows) / len(sample_rows)
            if sample_rows
            else None
        ),
        "solve_rate": mean_stats(float(row["any_passed"]) for row in sample_rows),
        "mean_lean_seconds": (
            statistics.fmean(float(row.get("elapsed_seconds") or 0.0) for row in candidate_rows)
            if candidate_rows
            else None
        ),
        "failure_reasons": dict(
            Counter(
                str(row.get("failure_reason") or "lean_error")
                for row in candidate_rows
                if not row.get("passed")
            ).most_common()
        ),
        "k": {},
    }
    for k in k_values:
        top_values = [float(row[f"top@{k}"]) for row in sample_rows if row[f"top@{k}"] is not None]
        pass_values = [float(row[f"pass@{k}"]) for row in sample_rows if row[f"pass@{k}"] is not None]
        result["k"][str(k)] = {
            "eligible_samples": len(pass_values),
            "coverage": len(pass_values) / len(sample_rows) if sample_rows else 0.0,
            "top_k_pass_rate": mean_stats(top_values),
            "unbiased_pass_at_k": mean_stats(pass_values),
        }
    return result


def summarize(
    results: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
    k_values: list[int],
) -> dict[str, Any]:
    samples = build_per_sample(results, index, k_values)
    results_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    samples_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        results_by_family[str(index[str(row["sample_id"])]["proof_family"])].append(row)
    for row in samples:
        samples_by_family[str(row["proof_family"])].append(row)
    overall = summarize_group(samples, results, k_values)
    by_family = {
        family: summarize_group(samples_by_family[family], results_by_family[family], k_values)
        for family in PROOF_FAMILIES
        if samples_by_family[family]
    }
    non_tactic_samples = samples_by_family.get("Strict Term", [])
    tactic_samples = [row for row in samples if row["proof_family"] != "Strict Term"]
    non_tactic_ids = {row["sample_id"] for row in non_tactic_samples}
    tactic_ids = {row["sample_id"] for row in tactic_samples}
    aggregates = {
        "Non-Tactic": summarize_group(
            non_tactic_samples,
            [row for row in results if str(row["sample_id"]) in non_tactic_ids],
            k_values,
        ),
        "Tactic-Containing": summarize_group(
            tactic_samples,
            [row for row in results if str(row["sample_id"]) in tactic_ids],
            k_values,
        ),
    }
    macro: dict[str, Any] = {"solve_rate": None, "k": {}}
    family_solve = [value["solve_rate"]["mean"] for value in by_family.values()]
    family_solve = [float(value) for value in family_solve if value is not None]
    macro["solve_rate"] = statistics.fmean(family_solve) if family_solve else None
    for k in k_values:
        values = [
            family["k"][str(k)]["unbiased_pass_at_k"]["mean"]
            for family in by_family.values()
        ]
        values = [float(value) for value in values if value is not None]
        macro["k"][str(k)] = statistics.fmean(values) if values else None
    return {
        "overall": overall,
        "by_proof_family": by_family,
        "aggregates": aggregates,
        "macro_over_proof_families": macro,
        "per_sample": samples,
    }
