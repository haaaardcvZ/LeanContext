"""Measure original context lengths with the acquired, real model tokenizer.

This diagnostic does not trim or filter targets. Context-only coverage is not
the coverage of a complete generator input, which also contains the statement.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import math
from pathlib import Path
from typing import Any

from .tokenize import SPLITS, iter_jsonl, load_actual_tokenizer, source_order, write_json, write_row


COMPONENTS = ("MC", "EC", "Full")
BUDGETS = (256, 512, 1024)


def describe(values: list[int]) -> dict[str, Any]:
    """Exact counts and nearest-rank quantiles; report both denominator choices."""
    ordered = sorted(values)
    nonempty = [value for value in ordered if value > 0]
    count = len(ordered)
    quantiles = {}
    for name, probability in (("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
        quantiles[name] = ordered[max(0, math.ceil(count * probability) - 1)] if count else None
    bins = {"0": 0, "1-256": 0, "257-512": 0, "513-1024": 0, "1025-2048": 0, "2049-4096": 0, ">4096": 0}
    for value in ordered:
        label = ("0" if value == 0 else "1-256" if value <= 256 else "257-512" if value <= 512 else "513-1024" if value <= 1024 else "1025-2048" if value <= 2048 else "2049-4096" if value <= 4096 else ">4096")
        bins[label] += 1
    return {
        "targets": count,
        "empty": count - len(nonempty),
        "nonempty": len(nonempty),
        "mean": sum(ordered) / count if count else None,
        "maximum": ordered[-1] if count else None,
        "nearest_rank_quantiles": quantiles,
        "histogram_counts": bins,
        "complete_context_coverage": {
            str(budget): {
                "all_targets_count": sum(value <= budget for value in ordered),
                "all_targets_percent": 100 * sum(value <= budget for value in ordered) / count if count else None,
                "nonempty_targets_count": sum(value <= budget for value in nonempty),
                "nonempty_targets_percent": 100 * sum(value <= budget for value in nonempty) / len(nonempty) if nonempty else None,
            }
            for budget in BUDGETS
        },
        "above_1024_count": sum(value > 1024 for value in ordered),
        "above_1024_percent_all": 100 * sum(value > 1024 for value in ordered) / count if count else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer, metadata = load_actual_tokenizer(args)
    raw_root = (args.dataset_root / "training" / "views" / "CFull").resolve()
    output = args.output_dir.resolve()
    if output == raw_root or raw_root in output.parents or output in raw_root.parents:
        raise ValueError("length analysis must not overlap original raw views")
    output.mkdir(parents=True, exist_ok=True)
    # Bounded cache reduces repeated encodings of shared predecessor commands.
    cache: OrderedDict[str, int] = OrderedDict()

    def length(text: str) -> int:
        if not text:
            return 0
        if text in cache:
            cache.move_to_end(text)
            return cache[text]
        value = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(text) <= 65536:
            cache[text] = value
            if len(cache) > 4096:
                cache.popitem(last=False)
        return value

    all_lengths = {component: [] for component in COMPONENTS}
    result: dict[str, Any] = {
        "schema_version": "leancontext-exact-raw-context-lengths-v1",
        "backbone": args.backbone,
        "tokenizer": metadata,
        "model_generation_performed": False,
        "raw_membership_preserved": True,
        "measurement": {
            "text": "complete original context commands in source order joined by two LF characters",
            "includes_statement": False,
            "includes_prompt_role_or_id_headers": False,
            "includes_BOS_or_EOS": False,
            "truncation": "none",
            "warning": "context-only coverage does not imply complete generator input fits the same budget",
        },
        "splits": {},
    }
    for split in SPLITS:
        lengths = {component: [] for component in COMPONENTS}
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for index, row in enumerate(iter_jsonl(raw_root / f"{split}.jsonl"), 1):
                commands = source_order(row["context_items"])
                if any(item.get("role") not in {"MC", "EC"} for item in commands):
                    raise ValueError(f"{row['sample_id']}: unknown context role")
                texts = {
                    "MC": "\n\n".join(item["text"] for item in commands if item["role"] == "MC"),
                    "EC": "\n\n".join(item["text"] for item in commands if item["role"] == "EC"),
                    "Full": "\n\n".join(item["text"] for item in commands),
                }
                measured = {component: length(text) for component, text in texts.items()}
                for component, value in measured.items():
                    lengths[component].append(value)
                    all_lengths[component].append(value)
                write_row(handle, {"sample_id": row["sample_id"], "target_id": row["target_id"], "split": split, "context_token_lengths": measured})
                if index % 10000 == 0:
                    print(f"{args.backbone}: measured {index} original {split} contexts", flush=True)
        result["splits"][split] = {component: describe(values) for component, values in lengths.items()}
    result["all_splits"] = {component: describe(values) for component, values in all_lengths.items()}
    write_json(output / "length_statistics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--backbone", choices=("dllm", "llama31"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    result = run(parser.parse_args())
    print(f"Complete: {result['all_splits']['Full']['targets']} real-tokenizer context measurements", flush=True)


if __name__ == "__main__":
    main()
