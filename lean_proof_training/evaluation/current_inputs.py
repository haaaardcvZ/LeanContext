"""Current four-view prompts; generation eligibility is reference-proof blind.

Use the saved exact tokenized prompt where available. Training rejections are
not generation exclusions: re-create their input using only the statement and
context. An overlong bare statement stays in the denominator with 33 explicit
empty candidates. No reference-proof token length affects generation.
"""
from __future__ import annotations

import hashlib
from typing import Any

VIEWS = ("C0", "CMath", "CEnv", "CFull")
FIELDS = {"C0": None, "CMath": "mathematical_context", "CEnv": "environment_context", "CFull": "full_context"}


def ordered(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda x: (int(x.get("source_order", x.get("position", 0))), str(x.get("command_uid", x.get("command_id", "")))))


def render(statement: str, items: list[dict[str, Any]]) -> str:
    return "\n\n".join([*(item["text"] for item in ordered(items)), statement]).rstrip("\r\n") + "\n"


def encode(tokenizer: Any, prompt: str, backbone: str) -> list[int]:
    if backbone == "native":
        # Released provers use their OWN pinned tokenizer postprocessor. No
        # instruction/chat template is applied and no Llama ID is assumed.
        return list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if backbone == "llama31":
        bos = tokenizer.bos_token_id
        if bos is None:
            raise ValueError("Llama tokenizer must define BOS")
        if not ids or ids[0] != int(bos):
            ids.insert(0, int(bos))
    return ids


def input_for_generation(row: dict[str, Any], saved: dict[str, Any] | None, *,
                         tokenizer: Any, backbone: str, view: str,
                         input_limit: int = 1024) -> tuple[str, list[int], int, str | None]:
    if view not in VIEWS or input_limit not in (256, 512, 1024):
        raise ValueError("expected a current four-view name and input budget 256/512/1024")
    if saved is not None:
        if str(saved["sample_id"]) != str(row["sample_id"]) or saved["view"] != view:
            raise ValueError("saved tokenized row ID/view mismatch")
        if int(saved["input_limit"]) != input_limit:
            raise ValueError("saved tokenized input budget mismatch")
        prompt = str(saved["prompt"])
        prefix = [int(x) for x in saved["input_ids"][:int(saved["prompt_length"])]]
        if encode(tokenizer, prompt, backbone) != prefix:
            raise ValueError("saved prompt/tokenizer mismatch")
        if saved.get("valid_input") is False:
            return prompt, prefix, 0, str(saved.get("invalid_input_reason") or "invalid_prepacked_input")
        if len(prefix) > input_limit:
            raise ValueError("saved prompt input overflow")
        # Deliberately do not read target_length, answer IDs, target, or proof.
        return prompt, prefix, len(saved.get("context_ids", [])), None
    statement = str(row["statement"])
    bare = render(statement, [])
    bare_ids = encode(tokenizer, bare, backbone)
    if len(bare_ids) > input_limit:
        return bare, bare_ids, 0, "complete_statement_exceeds_input_limit"
    full = ordered(list(row["full_context"]))
    math_ids = {x["command_uid"] for x in row["mathematical_context"]}
    env_ids = {x["command_uid"] for x in row["environment_context"]}
    while True:
        views = {"C0": [], "CMath": [x for x in full if x["command_uid"] in math_ids],
                 "CEnv": [x for x in full if x["command_uid"] in env_ids], "CFull": full}
        encoded = {name: (render(statement, items), encode(tokenizer, render(statement, items), backbone))
                   for name, items in views.items()}
        if all(len(ids) <= input_limit for _, ids in encoded.values()):
            prompt, ids = encoded[view]
            return prompt, ids, len(views[view]), None
        full.pop()


def invalid_input_candidates(row: dict[str, Any], view: str, prompt: str,
                             prompt_tokens: int | None, reason: str) -> list[dict[str, Any]]:
    common = {"sample_id": str(row["sample_id"]), "proof_family": str(row["proof_family"]),
              "context_view": view, "proof": "", "input_valid": False,
              "invalid_input_reason": reason, "prompt_tokens": prompt_tokens,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "max_new_tokens": 1024}
    return [{**common, "mode": "sampled", "candidate_id": i} for i in range(32)] + [
        {**common, "mode": "greedy_temperature_0", "candidate_id": 0}]


def selector_input_for_generation(pair: dict[str, Any], *, policy: str, tokenizer: Any,
                                  backbone: str, input_limit: int) -> tuple[str, list[int], int, str | None]:
    """Consume published pair prompts only; never inspect proof dependencies.

    Both policies use the same target, EC, pool and input budget. Selection and
    scope-cohort packing already happened in dataset preparation. A saved
    invalid input remains an explicit failure in the fixed2000denominator.
    """
    if policy not in ("baseline", "dependency_prioritized"):
        raise ValueError("unknown selector policy; only baseline/dependency_prioritized supported")
    if int(pair["input_limit"]) != input_limit or int(pair["output_limit"]) != 1024:
        raise ValueError("selector pair independent input/output budgets do not match")
    if pair.get("EC_same_for_both_policies") is not True:
        raise ValueError("selector policies must share identical EC")
    selected = pair[policy]
    if not selected.get("valid_input", False):
        return str(selected.get("prompt") or ""), [], 0, str(selected.get("reason") or "invalid_prepacked_selector_input")
    prompt = str(selected["prompt"])
    ids = [int(x) for x in selected["input_ids"]]
    if len(ids) != int(selected["prompt_length"]) or len(ids) > input_limit:
        raise ValueError("selector pair token span/length invalid")
    if encode(tokenizer,prompt,backbone) != ids:
        raise ValueError("selector prompt IDs differ from the active tokenizer")
    visible = selected.get("all_visible_command_ids", selected.get("visible_command_ids", []))
    return prompt, ids, len(visible), None
