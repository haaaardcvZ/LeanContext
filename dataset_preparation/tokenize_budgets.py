"""Real tokenization with independent input and complete-proof output budgets.

Each backbone uses common targets across all input budgets and all four views.
The original statements/proofs are never truncated to manufacture eligibility.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any

from .tokenize import (
    SPLITS, VIEWS, command_id, command_key, encode_answer, encode_prompt,
    iter_jsonl, load_actual_tokenizer, render_training_prompt, source_order,
    unique_commands, write_json, write_row,
)

PROTOCOL = "leancontext-independent-input-proof-budgets-common-membership-v1"
INPUT_BUDGETS = (1024,)


def prepare_budgeted_views(tokenizer: Any, statement: str, proof: str,
                           contexts: dict[str, list[dict[str, Any]]], *,
                           backbone: str, input_budgets: tuple[int, ...] = INPUT_BUDGETS,
                           output_limit: int = 1024) -> dict[str, Any]:
    budgets = tuple(sorted(set(input_budgets)))
    if not budgets or min(budgets) <= 0 or output_limit <= 0:
        raise ValueError("input and output budgets must be positive")
    if set(contexts) != set(VIEWS) or contexts["C0"]:
        raise ValueError("four contexts with empty C0 required")
    full = source_order(unique_commands(contexts["CFull"]))
    keys = {view: {command_key(item) for item in contexts[view]} for view in VIEWS}
    if keys["CMath"] & keys["CEnv"] or keys["CMath"] | keys["CEnv"] != {command_key(item) for item in full}:
        raise ValueError("Full must be disjoint Math/Env union")
    answer = encode_answer(tokenizer, proof)
    bare = encode_prompt(tokenizer, render_training_prompt(statement, []), backbone)
    eligible = {str(budget): len(bare) <= budget and len(answer) <= output_limit for budget in budgets}
    reasons = []
    if len(answer) > output_limit:
        reasons.append("complete_reference_proof_plus_one_EOS_exceeds_output_limit")
    if len(bare) > min(budgets):
        reasons.append("complete_statement_prompt_exceeds_smallest_input_budget")
    common = {
        "eligible_by_input_budget": eligible,
        "bare_prompt_length": len(bare),
        "answer_length": len(answer),
        "input_budgets": list(budgets),
        "output_limit": output_limit,
    }
    if reasons:
        return {**common, "retained": False, "reason": reasons[0], "reasons": reasons}
    # Each prefix is encoded only once per target, even across three budgets.
    cache: dict[int, dict[str, Any]] = {}

    def render(prefix: int) -> dict[str, Any]:
        if prefix in cache:
            return cache[prefix]
        selected = full[:prefix]
        rendered = {}
        for view in VIEWS:
            visible = [item for item in selected if command_key(item) in keys[view]]
            prompt = render_training_prompt(statement, visible)
            prompt_ids = encode_prompt(tokenizer, prompt, backbone)
            rendered[view] = {
                "statement": statement, "prompt": prompt, "target": proof,
                "reference_proof": proof, "input_ids": prompt_ids + answer,
                "prompt_length": len(prompt_ids), "input_length": len(prompt_ids),
                "answer_length": len(answer), "target_length": len(answer),
                "sequence_length": len(prompt_ids) + len(answer),
                "answer_span": {"start": len(prompt_ids), "end_exclusive": len(prompt_ids) + len(answer)},
                "answer_eos_count": 1, "answer_final_token_id": int(tokenizer.eos_token_id),
                "context_items": visible,
                "context_ids": [command_id(item) for item in visible],
                "context_uids": [command_key(item) for item in visible],
                "visible_command_ids": [command_id(item) for item in visible],
                "visible_command_uids": [command_key(item) for item in visible],
            }
        cache[prefix] = rendered
        return rendered

    original = render(len(full))
    outputs = {}
    for budget in budgets:
        prefix = len(full)
        while any(row["prompt_length"] > budget for row in render(prefix).values()):
            prefix -= 1
            if prefix < 0:
                raise AssertionError("bare statement eligibility checked")
        removed = full[prefix:]
        outputs[str(budget)] = {
            view: {
                **row, "input_limit": budget, "output_limit": output_limit,
                "untrimmed_prompt_length": original[view]["prompt_length"],
                "removed_command_ids": [command_id(item) for item in removed],
                "removed_command_uids": [command_key(item) for item in removed],
                "view_removed_command_ids": [command_id(item) for item in removed if command_key(item) in keys[view]],
            }
            for view, row in render(prefix).items()
        }
    return {**common, "retained": True, "views_by_budget": outputs}


def run(args: argparse.Namespace, tokenizer: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    raw = (args.dataset_root / "training/views").resolve()
    output = args.output_dir.resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("outputs must not overlap raw views")
    budgets = tuple(sorted(set(args.input_budgets)))
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "protocol": PROTOCOL, "backbone": args.backbone, "tokenizer": metadata,
        "input_budgets": list(budgets), "output_limit": args.output_limit,
        "output_limit_includes_one_terminal_EOS": True,
        "common_membership_across_views_and_input_budgets": True,
        "prompt_budget_includes_BOS_and_boundary_newline": True,
        "sequence_packing": False, "padding_serialized": False,
        "model_generation_performed": False,
        "splits": {},
    }
    for split in SPLITS:
        stats = {
            "seen": 0, "retained": 0, "rejected": 0, "rejection_reason_counts": Counter(),
            "feasible_targets_by_input_budget": {str(b): 0 for b in budgets},
            "budgets": {str(b): {"context_trimmed_targets": 0, "max_prompt_lengths": {v: 0 for v in VIEWS}, "max_sequence_lengths": {v: 0 for v in VIEWS}, "max_answer_length": 0} for b in budgets},
        }
        with ExitStack() as stack:
            inputs = {view: iter(iter_jsonl(raw / view / f"{split}.jsonl")) for view in VIEWS}
            files = {}
            for budget in budgets:
                for view in VIEWS:
                    path = output / f"input_{budget}" / "views" / view / f"{split}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    files[budget, view] = stack.enter_context(path.open("w", encoding="utf-8"))
            handles = {}
            for name in ("membership", "rejections"):
                path = output / name / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[name] = stack.enter_context(path.open("w", encoding="utf-8"))
            sentinel = object()
            while True:
                rows = {view: next(iterator, sentinel) for view, iterator in inputs.items()}
                if all(row is sentinel for row in rows.values()):
                    break
                if any(row is sentinel for row in rows.values()):
                    raise ValueError("raw four-view membership differs")
                base = rows["C0"]
                for view, row in rows.items():
                    if (row["sample_id"], row["statement"], row["target"], row["split"], row["view"]) != (base["sample_id"], base["statement"], base["target"], split, view):
                        raise ValueError("raw target or view metadata differs")
                prepared = prepare_budgeted_views(tokenizer, base["statement"], base["target"], {view: row["context_items"] for view, row in rows.items()}, backbone=args.backbone, input_budgets=budgets, output_limit=args.output_limit)
                stats["seen"] += 1
                member = {"sample_id": base["sample_id"], "target_id": base["target_id"], "split": split, **{key: value for key, value in prepared.items() if key != "views_by_budget"}}
                for budget, eligible in prepared["eligible_by_input_budget"].items():
                    stats["feasible_targets_by_input_budget"][budget] += eligible
                write_row(handles["membership"], member)
                if not prepared["retained"]:
                    stats["rejected"] += 1
                    stats["rejection_reason_counts"].update(prepared["reasons"])
                    write_row(handles["rejections"], member)
                    continue
                stats["retained"] += 1
                for budget in budgets:
                    budget_stats = stats["budgets"][str(budget)]
                    budget_views = prepared["views_by_budget"][str(budget)]
                    budget_stats["context_trimmed_targets"] += bool(budget_views["CFull"]["removed_command_ids"])
                    budget_stats["max_answer_length"] = max(budget_stats["max_answer_length"], prepared["answer_length"])
                    for view, encoded in budget_views.items():
                        row = {**rows[view], **encoded, "raw_context_ids": rows[view]["context_ids"], "raw_context_uids": rows[view]["context_uids"], "protocol": PROTOCOL, "tokenization_status": "complete_real_tokenizer"}
                        write_row(files[budget, view], row)
                        budget_stats["max_prompt_lengths"][view] = max(budget_stats["max_prompt_lengths"][view], row["prompt_length"])
                        budget_stats["max_sequence_lengths"][view] = max(budget_stats["max_sequence_lengths"][view], row["sequence_length"])
                if stats["seen"] % 10000 == 0:
                    print(f"{args.backbone}: processed {stats['seen']} {split} targets", flush=True)
        stats["rejection_reason_counts"] = dict(stats["rejection_reason_counts"])
        metrics["splits"][split] = stats
    write_json(output / "tokenization_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--backbone", choices=("dllm", "llama31"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-budgets", nargs="+", type=int, default=list(INPUT_BUDGETS))
    parser.add_argument("--output-limit", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    tokenizer, metadata = load_actual_tokenizer(args)
    print(json.dumps(run(args, tokenizer, metadata), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
