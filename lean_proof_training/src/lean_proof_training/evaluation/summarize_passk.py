"""Summarize Lean candidate results overall and by proof family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_proof_training.common.io import iter_jsonl
from lean_proof_training.evaluation.passk import summarize


def parse_k(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or result[0] <= 0:
        raise ValueError("k values must be positive")
    return result


def write_report(payload: dict, output_dir: Path, k_values: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample = payload["per_sample"]
    summary_payload = {key: value for key, value in payload.items() if key != "per_sample"}
    with (output_dir / "pass_at_k_per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "pass_at_k_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Lean pass@k", "", "## Overall", ""]
    overall = payload["overall"]
    lines.extend(
        [
            f"- Samples: {overall['samples']}",
            f"- Candidate pass rate: {overall['candidate_pass_rate']}",
            f"- Solve rate: {overall['solve_rate']['mean']}",
            "",
            "| Group | Samples | Solve rate | " + " | ".join(f"pass@{k}" for k in k_values) + " |",
            "|---|---:|---:|" + "---:|" * len(k_values),
        ]
    )
    groups = {
        "Overall": overall,
        **summary_payload["by_proof_family"],
        **summary_payload["aggregates"],
    }
    for name, group in groups.items():
        pass_values = [group["k"][str(k)]["unbiased_pass_at_k"]["mean"] for k in k_values]
        lines.append(
            f"| {name} | {group['samples']} | {group['solve_rate']['mean']} | "
            + " | ".join(str(value) for value in pass_values)
            + " |"
        )
    (output_dir / "pass_at_k_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", default="1,4,8,16,32,64")
    args = parser.parse_args()
    k_values = parse_k(args.k_values)
    index = {str(row["sample_id"]): row for row in iter_jsonl(args.index_file)}
    results = list(iter_jsonl(args.results_file))
    unknown = sorted({str(row["sample_id"]) for row in results} - set(index))
    if unknown:
        raise ValueError(f"results contain unknown sample IDs: {unknown[:5]}")
    payload = summarize(results, index, k_values)
    write_report(payload, args.output_dir, k_values)
    print(json.dumps({key: value for key, value in payload.items() if key != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
