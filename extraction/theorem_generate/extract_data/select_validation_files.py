from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from extract_data.context_pipeline import build_candidates, iter_target_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select source files for validating Flash/Pro candidate-count routing."
    )
    parser.add_argument(
        "--input-root",
        default=(Path(__file__).resolve().parents[1] / "lean_parser" / "command_spans_jsonl" / "Mathlib").as_posix(),
    )
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--below-files", type=int, default=2)
    parser.add_argument("--above-files", type=int, default=1)
    parser.add_argument("--min-targets", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=80)
    parser.add_argument("--output", default="extract_data/validation_files.json")
    parser.add_argument("--source-file-list-output", default="extract_data/validation_source_files.txt")
    parser.add_argument("--recursive-depth", type=int, default=2)
    parser.add_argument("--max-dependency-candidates", type=int, default=60)
    args = parser.parse_args()

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "source_file": "",
            "targets": 0,
            "candidate_counts": [],
            "flash_cases": 0,
            "pro_cases": 0,
            "max_candidate_count": 0,
        }
    )

    for case in iter_target_cases(Path(args.input_root).resolve()):
        source_file = str(case.target.get("source_file"))
        candidates = build_candidates(
            case,
            recursive_depth=args.recursive_depth,
            max_dependency_candidates=args.max_dependency_candidates,
        )
        n = len(candidates)
        item = stats[source_file]
        item["source_file"] = source_file
        item["targets"] += 1
        item["candidate_counts"].append(n)
        item["max_candidate_count"] = max(item["max_candidate_count"], n)
        if n > args.threshold:
            item["pro_cases"] += 1
        else:
            item["flash_cases"] += 1

    rows = []
    for item in stats.values():
        if not (args.min_targets <= item["targets"] <= args.max_targets):
            continue
        counts = item["candidate_counts"]
        rows.append(
            {
                "source_file": item["source_file"],
                "targets": item["targets"],
                "flash_cases": item["flash_cases"],
                "pro_cases": item["pro_cases"],
                "max_candidate_count": item["max_candidate_count"],
                "avg_candidate_count": round(mean(counts), 2),
            }
        )

    below = [row for row in rows if row["pro_cases"] == 0]
    above = [row for row in rows if row["pro_cases"] > 0]

    below.sort(key=lambda row: (-row["targets"], row["avg_candidate_count"], row["source_file"]))
    above.sort(key=lambda row: (row["targets"], -row["max_candidate_count"], row["source_file"]))

    selected_below = below[: args.below_files]
    selected_above = above[: args.above_files]
    selected = selected_below + selected_above

    output = {
        "threshold": args.threshold,
        "below_threshold_files": selected_below,
        "above_threshold_files": selected_above,
        "selected_source_files": [row["source_file"] for row in selected],
        "suggested_command": [
            "python3",
            "-m",
            "extract_data.build_context_deepseek_hybrid",
            "--output",
            "extract_data/validation_deepseek_hybrid.jsonl",
            "--source-file-list",
            args.source_file_list_output,
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    list_path = Path(args.source_file_list_output)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(output["selected_source_files"]) + "\n", encoding="utf-8")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
