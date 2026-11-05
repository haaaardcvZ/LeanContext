"""Model-agnostic materialization of unified Context views."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from lean_proof_training.common.io import iter_jsonl
from lean_proof_training.common.schema import render_prompt


def budget_name(value: int | None) -> str:
    return "full" if value is None else f"b{int(value)}"


def encode_example(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    view: str,
    context_budget: int | None,
    budget_policy: str,
    max_length: int,
    add_bos_token: bool,
) -> dict[str, Any] | None:
    rendered = render_prompt(
        row,
        view,
        tokenizer=tokenizer,
        context_budget=context_budget,
        budget_policy=budget_policy,
    )
    if rendered is None:
        return None
    prompt, selected_items = rendered
    target = str(row["target"])
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if add_bos_token:
        if tokenizer.bos_token_id is None:
            raise ValueError("tokenize.add_bos_token=true but tokenizer has no BOS token")
        if not prompt_ids or prompt_ids[0] != int(tokenizer.bos_token_id):
            prompt_ids.insert(0, int(tokenizer.bos_token_id))
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token")
    if int(eos_id) in target_ids:
        raise ValueError(f"target already contains EOS: {row['sample_id']}")
    target_ids.append(int(eos_id))
    if not prompt_ids or len(target_ids) <= 1:
        raise ValueError(f"empty prompt or proof tokenization: {row['sample_id']}")
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + target_ids,
        "prompt_length": len(prompt_ids),
        "target_length": len(target_ids),
        "sample_id": row["sample_id"],
        "split": row["split"],
        "training_stage": row["training_stage"],
        "proof_family": row["proof_family"],
        "view": view,
        "context_budget": budget_name(context_budget),
        "context_items": len(selected_items),
    }


def materialize_tokenized_views(
    config: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    raw_dir = Path(config["data"]["raw_dir"])
    output_root = Path(config["data"]["tokenized_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    views = list(config["tokenize"]["views"])
    budgets = [None if value in {None, "full"} else int(value) for value in config["tokenize"]["context_budgets"]]
    max_length = int(config["tokenize"]["max_length"])
    policy = str(config["tokenize"].get("budget_policy", "prefix"))
    add_bos_token = bool(config["tokenize"].get("add_bos_token", False))
    overflow = str(config["tokenize"].get("overflow_policy", "drop"))
    if overflow not in {"drop", "error"}:
        raise ValueError("overflow_policy must be drop or error")
    statistics: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        input_path = raw_dir / f"{split}.jsonl"
        counters: dict[tuple[str, str], Counter[str]] = {}
        with ExitStack() as stack:
            outputs: dict[tuple[str, str], Any] = {}
            for view in views:
                view_budgets = [0] if view == "statement_only" else budgets
                for budget in view_budgets:
                    name = budget_name(budget)
                    path = output_root / view / name / f"{split}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    outputs[(view, name)] = stack.enter_context(path.open("w", encoding="utf-8"))
                    counters[(view, name)] = Counter()
            for row in iter_jsonl(input_path):
                for (view, name), output in outputs.items():
                    budget = None if name == "full" else int(name[1:])
                    counters[(view, name)]["seen"] += 1
                    encoded = encode_example(
                        tokenizer,
                        row,
                        view=view,
                        context_budget=budget,
                        budget_policy=policy,
                        max_length=max_length,
                        add_bos_token=add_bos_token,
                    )
                    if encoded is None:
                        unavailable = view in {"random_control", "hard_control"} and row["contexts"][view] is None
                        key = "unavailable" if unavailable else "overflow"
                        counters[(view, name)][key] += 1
                        if overflow == "error" and not unavailable:
                            raise ValueError(
                                f"sequence exceeds max_length={max_length}: {row['sample_id']} {view}/{name}"
                            )
                        continue
                    output.write(json.dumps(encoded, ensure_ascii=False) + "\n")
                    counters[(view, name)]["written"] += 1
        statistics[split] = {
            f"{view}/{name}": dict(counter)
            for (view, name), counter in sorted(counters.items())
        }
    tokenizer.save_pretrained(output_root / "tokenizer")
    payload = {
        "max_length": max_length,
        "budget_policy": policy,
        "add_bos_token": add_bos_token,
        "splits": statistics,
    }
    with (output_root / "tokenization_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload
