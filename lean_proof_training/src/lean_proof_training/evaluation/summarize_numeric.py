"""Aggregate per-sample loss or information-gain fields by proof family."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from lean_proof_training.common.io import iter_jsonl


def stats(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"n": 0, "mean": None, "median": None, "std": None, "se": None, "ci95": None}
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    se = std / math.sqrt(len(finite))
    return {
        "n": len(finite),
        "mean": mean,
        "median": statistics.median(finite),
        "std": std,
        "se": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, default=None)
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument(
        "--group-field",
        default=None,
        help="optional row field such as comparison; prevents mixing budgets/controls",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    index = (
        {str(row["sample_id"]): str(row["proof_family"]) for row in iter_jsonl(args.index_file)}
        if args.index_file
        else {}
    )
    values: dict[str, dict[str, dict[str, list[float]]]] = {
        metric: defaultdict(lambda: defaultdict(list)) for metric in args.metric
    }
    seen: set[str] = set()
    for row in iter_jsonl(args.input_file):
        sample_id = str(row["sample_id"])
        family = str(row.get("proof_family") or index.get(sample_id) or "Unknown")
        comparison = str(row.get(args.group_field) or "all") if args.group_field else "all"
        seen.add(sample_id)
        for metric in args.metric:
            if metric in row and row[metric] is not None:
                value = float(row[metric])
                values[metric][comparison]["Overall"].append(value)
                values[metric][comparison][family].append(value)
    report = {
        "samples": len(seen),
        "index_coverage": len(seen & set(index)) / len(index) if index else None,
        "metrics": {
            metric: {
                comparison: {
                    family: stats(group_values)
                    for family, group_values in sorted(families.items())
                }
                for comparison, families in sorted(comparisons.items())
            }
            for metric, comparisons in values.items()
        },
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
