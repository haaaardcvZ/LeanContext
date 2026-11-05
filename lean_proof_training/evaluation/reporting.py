"""Persist machine-readable and paper-friendly evaluation reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io_utils import write_json, write_jsonl


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * float(value):.2f}"


def write_reports(
    output_dir: Path,
    payload: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    per_sample = payload["per_sample"]
    summary = {key: value for key, value in payload.items() if key != "per_sample"}
    write_jsonl(output_dir / "per_sample_metrics.jsonl", per_sample)
    write_json(output_dir / "metrics.json", summary)

    groups = {
        "Overall": payload["overall"],
        **payload["by_proof_family"],
        **payload["aggregates"],
    }
    lines = [
        "# Lean proof-generation evaluation",
        "",
        "All Pass@k values use the same 32 stochastic candidates per theorem. ",
        "Greedy@1 is a separate deterministic argmax generation with temperature=0 semantics.",
        "",
        "| Group | Samples | Greedy@1 (%) | Pass@1 (%) | Pass@8 (%) | Pass@16 (%) | Pass@32 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        lines.append(
            "| {name} | {samples} | {greedy} | {p1} | {p8} | {p16} | {p32} |".format(
                name=name,
                samples=group["samples"],
                greedy=_percent(group["greedy_temperature_0"]["mean"]),
                p1=_percent(group["pass_at_k"]["1"]["mean"]),
                p8=_percent(group["pass_at_k"]["8"]["mean"]),
                p16=_percent(group["pass_at_k"]["16"]["mean"]),
                p32=_percent(group["pass_at_k"]["32"]["mean"]),
            )
        )
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals: Counter[str] = Counter()
    for row in results:
        mode = str(row["mode"])
        totals[mode] += 1
        error_type = str(row["error_type"])
        counts[mode][error_type] += 1
        if error_type != "success" and len(examples[error_type]) < 10:
            examples[error_type].append(
                {
                    "sample_id": row["sample_id"],
                    "mode": mode,
                    "candidate_id": row["candidate_id"],
                    "proof_family": row["proof_family"],
                    "diagnostic": row.get("diagnostic", ""),
                }
            )
    error_payload = {
        "counts_by_mode": {mode: dict(counter.most_common()) for mode, counter in counts.items()},
        "rates_by_mode": {
            mode: {
                category: count / totals[mode]
                for category, count in counter.items()
            }
            for mode, counter in counts.items()
        },
        "examples": dict(examples),
    }
    write_json(output_dir / "error_types.json", error_payload)
    error_lines = [
        "# Error-type distribution",
        "",
        "The taxonomy is mutually exclusive: every candidate has exactly one final category.",
        "",
        "| Mode | Error type | Count | Rate (%) |",
        "|---|---|---:|---:|",
    ]
    for mode in sorted(counts):
        for category, count in counts[mode].most_common():
            error_lines.append(
                f"| {mode} | {category} | {count} | {100.0 * count / totals[mode]:.2f} |"
            )
    (output_dir / "error_types.md").write_text(
        "\n".join(error_lines) + "\n", encoding="utf-8"
    )
