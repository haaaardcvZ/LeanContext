#!/usr/bin/env python3
"""LEANCONTEXT serialization, length, stopping, and proof-extraction protocol.

The controlled LLaDA-8B-Base and Llama-3.1-8B experiments use independent
1,024-token input and output budgets. The training target contains the complete
reference proof and exactly one terminal EOS; the combined sequence is at most
2,048 tokens. Statements and reference proofs are never token-truncated. Long
contexts are reduced by removing complete commands from the end of CFull, then
deriving the other views without refilling. Ineligible targets are excluded from
all four training views under the same tokenizer.

Inputs contain no instruction, chat template, or section headings. Prompt and
proof are encoded separately, with one boundary newline. Llama receives one
explicit BOS; LLaDA receives none. AR generation stops at EOS or the output
limit. Diffusion generation completes its scheduled denoising and is then cut at
the first EOS. Text markers are extraction rules, not decoder stopping rules.

The optional 256/512-token utilities support separate budget analyses; they do
not define the primary training membership. The compact dataset release stores
raw records and tokenizer assets; its tokenized caches are created by the
preparation tools. Native decoders for other model families are not implied by
this primary-backbone protocol. Actual runs record their resolved settings.

Usage:
  python3 protocol.py --self-test
  load_pinned_tokenizer('llada', '/path/to/leancontext_dataset/tokenizers/llada')
  prepare_budgeted_views(tokenizer, statement, proof, contexts, backbone='dllm')

Only load_pinned_tokenizer requires Transformers. Other functions and the
self-test use the Python standard library. This module does not download model
weights, call an API, or execute Lean verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

VIEWS = ("C0", "CMath", "CEnv", "CFull")
INPUT_BUDGETS = (1024,)
OUTPUT_LIMIT = 1024  # Includes exactly one terminal EOS in training targets.
TOKENIZERS = {
    "llada": {
        "repository": "GSAI-ML/LLaDA-8B-Base",
        "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
        "prompt_BOS": False,
        "eos_token_id": 126081,
        "mask_token_id": 126336,
        "file_sha256": {
            "tokenizer.json": "ee1ef8e5f6d9493ac25480b7b7337ff5d2c1b946190afff18d12c86ca738ae00",
            "tokenizer_config.json": "6e9f41633217287fcf9a58890efb26e91e905bd6ae2234b534b65e0c36f4dd3c",
        },
    },
    "llama31": {
        "repository": "meta-llama/Llama-3.1-8B",
        "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "prompt_BOS": True,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "file_sha256": {
            "tokenizer.json": "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa",
            "tokenizer_config.json": "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a",
            "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
        },
    },
}


def load_pinned_tokenizer(backbone: str, directory: str | Path) -> Any:
    """Load the exact local tokenizer used to build the current data, offline.

    File hashes certify tokenizer bytes, not model weights or a past checkpoint.
    LLaDA's malformed upstream special_tokens_map was archived under a different
    name and excluded from loading in the existing acquisition pipeline.
    """
    from transformers import AutoTokenizer

    key = "llada" if backbone == "dllm" else backbone
    specification = TOKENIZERS[key]
    directory = Path(directory)
    for name, expected in specification["file_sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned tokenizer file mismatch: {directory / name}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, use_fast=True, trust_remote_code=False,
    )
    if tokenizer.eos_token_id != specification["eos_token_id"]:
        raise ValueError("EOS differs from the actual dataset tokenizer")
    if key == "llama31" and tokenizer.bos_token_id != specification["bos_token_id"]:
        raise ValueError("BOS differs from the actual dataset tokenizer")
    if key == "llada" and tokenizer.convert_tokens_to_ids("<|mdm_mask|>") != 126336:
        raise ValueError("LLaDA diffusion mask differs from the actual vocabulary")
    # Llama has no PAD in the source tokenizer; PAD=EOS is a batching alias only.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def command_id(command: dict[str, Any]) -> str:
    value = command.get("command_id", command.get("command_uid"))
    if value is None:
        raise ValueError("context command has no command_id or command_uid")
    return str(value)


def command_key(command: dict[str, Any]) -> str:
    return str(command.get("command_uid") or command_id(command))


def source_order(commands: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[int, str]:
        order = item.get("source_order", item.get("position"))
        if order is None:
            raise ValueError(f"command {command_id(item)} has no source order")
        return int(order), command_key(item)
    return sorted(commands, key=key)


def unique_commands(commands: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    for command in commands:
        key = command_key(command)
        if key not in seen:
            if not isinstance(command.get("text"), str) or not command["text"].strip():
                raise ValueError(f"command {command_id(command)} has no complete source text")
            seen.add(key)
            result.append(command)
    return result


def render_training_prompt(statement: str, commands: Iterable[dict[str, Any]]) -> str:
    """Actual serializer: commands, blank lines, statement, one final newline.

    No instruction, chat template, role headings, or textual stop markers.
    Local binders/hypotheses/typeclasses inside the statement are never removed.
    """
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be nonempty")
    parts = [item["text"] for item in source_order(commands)] + [statement]
    return "\n\n".join(parts).rstrip("\r\n") + "\n"


def encode_prompt(tokenizer: Any, prompt: str, backbone: str) -> list[int]:
    if backbone not in {"dllm", "llama31"}:
        raise ValueError("backbone must be dllm or llama31")
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if backbone == "llama31":
        bos = tokenizer.bos_token_id
        if bos is None:
            raise ValueError("Llama prompt encoding requires the actual tokenizer BOS")
        if not ids or ids[0] != int(bos):
            ids.insert(0, int(bos))
    return ids


def encode_answer(tokenizer: Any, proof: str) -> list[int]:
    """Exact complete proof plus one EOS; prompt owns the boundary newline."""
    if not isinstance(proof, str) or not proof.strip():
        raise ValueError("reference proof must be nonempty")
    if proof.startswith(("\n", "\r")):
        raise ValueError("reference proof has a leading newline; prompt owns the boundary newline")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("actual tokenizer must define EOS")
    ids = list(tokenizer(proof, add_special_tokens=False)["input_ids"])
    if not ids:
        raise ValueError("reference proof tokenization is empty")
    if int(eos) in ids:
        raise ValueError("reference proof already contains an EOS token")
    return ids + [int(eos)]


def prepare_budgeted_views(tokenizer: Any, statement: str, proof: str,
                           contexts: dict[str, list[dict[str, Any]]], *,
                           backbone: str, input_budgets: tuple[int, ...] = INPUT_BUDGETS,
                           output_limit: int = 1024) -> dict[str, Any]:
    """The actual latest dataset algorithm, including common membership.

    Pass all three budgets to reproduce existing data. Passing only (1024,)
    changes membership and must not be described as the existing export.
    """
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
        "eligible_by_input_budget": eligible, "bare_prompt_length": len(bare),
        "answer_length": len(answer), "input_budgets": list(budgets), "output_limit": output_limit,
    }
    if reasons:
        return {**common, "retained": False, "reason": reasons[0], "reasons": reasons}
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


def generator_input_from_training_row(row: dict[str, Any], tokenizer: Any,
                                      backbone: str) -> list[int]:
    """Use the already packed view, not a new independently refilled context.

    Training input_ids includes the reference answer. Never pass the complete
    list to the generator: only input_ids[:prompt_length] is generator input.
    """
    prompt_length = int(row["prompt_length"])
    ids = [int(value) for value in row["input_ids"][:prompt_length]]
    if ids != encode_prompt(tokenizer, row["prompt"], backbone):
        raise ValueError("Generator serializer differs from the saved training view")
    if not 0 < len(ids) <= int(row["input_limit"]):
        raise ValueError("Generator prompt exceeds its complete-input budget")
    return ids


def generation_limits(prompt_ids: list[int], *, input_limit: int = 1024,
                      output_limit: int = OUTPUT_LIMIT) -> dict[str, int]:
    """Independent input and output limits for the controlled evaluation protocol.

    A longer prompt does not reduce the output allowance. The combined sequence
    limit is the sum of the configured input budget and the output budget.
    """
    if input_limit not in (256, 512, 1024) or output_limit != OUTPUT_LIMIT:
        raise ValueError("Use input 256/512/1024 and independent output 1024")
    if not 0 < len(prompt_ids) <= input_limit:
        raise ValueError("Prompt exceeds its complete-input budget")
    return {"max_new_tokens": output_limit, "max_sequence_length": input_limit + output_limit}


def ar_generation_kwargs(tokenizer: Any, prompt_ids: list[int], *,
                         count: int = 1, temperature: float = 0.0,
                         top_p: float = 0.95, top_k: int = 0,
                         input_limit: int = 1024) -> dict[str, Any]:
    """Pass these kwargs with input_ids/attention_mask to model.generate.

    Actual AR stopping semantics: one tokenizer EOS ID or max_new_tokens.
    No stop_strings, chat terminators, END_OF_PROOF criteria, or beam search.
    """
    limits = generation_limits(prompt_ids, input_limit=input_limit)
    if count <= 0 or temperature < 0:
        raise ValueError("Invalid candidate count or temperature")
    sampled = temperature > 0
    if not sampled and count != 1:
        raise ValueError("Greedy/temperature-zero AR generates one candidate")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    kwargs = {
        "max_new_tokens": limits["max_new_tokens"], "do_sample": sampled,
        "num_return_sequences": count, "num_beams": 1,
        "pad_token_id": int(pad), "eos_token_id": int(tokenizer.eos_token_id),
        "use_cache": True,
    }
    if sampled:
        kwargs.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
    return kwargs


def _trim_at_eos(ids: list[int], eos_id: int | None) -> list[int]:
    if eos_id is not None and eos_id in ids:
        return ids[:ids.index(eos_id)]
    return ids


def decode_generated_suffix(tokenizer: Any, full_output_ids: Iterable[int],
                            prompt_length: int, *, output_limit: int = OUTPUT_LIMIT) -> str:
    """AR: slice away the copied prompt, then decode the generated proof only."""
    ids = list(full_output_ids)
    if prompt_length < 0 or prompt_length > len(ids):
        raise ValueError("Invalid prompt boundary")
    suffix = ids[prompt_length:]
    if len(suffix) > output_limit:
        raise ValueError("Backend exceeded output budget; do not silently truncate")
    suffix = _trim_at_eos(suffix, tokenizer.eos_token_id)
    return tokenizer.decode(suffix, skip_special_tokens=True).strip()


def decode_diffusion_window(tokenizer: Any, final_window_ids: Iterable[int], *,
                            output_limit: int = OUTPUT_LIMIT) -> str:
    """DLLM: apply only AFTER the generator finishes its denoising schedule.

    The existing backend starts with an output window of mask tokens and stops
    denoising when its mask-state is empty or its step budget is exhausted. It
    does not stop the whole schedule upon the first provisional EOS. Final
    visible output ends before the FIRST EOS. No EOS means use the full window.
    """
    ids = list(final_window_ids)
    if len(ids) > output_limit:
        raise ValueError("Diffusion window exceeded the independent output budget")
    return tokenizer.decode(
        _trim_at_eos(ids, tokenizer.eos_token_id), skip_special_tokens=True,
    ).strip()


# Exact existing text-cleanup semantics. These are NOT generation stop markers.
TEXT_CLEANUP_MARKERS = ("<END_OF_PROOF>", "END_OF_PROOF", "<|endoftext|>")
FENCE_RE = re.compile(r"```(?:lean|lean4)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
PLACEHOLDER_RE = re.compile(r"(?<![A-Za-z0-9_'.])(?:sorry|admit)(?![A-Za-z0-9_'.])")
COMMAND_INJECTION_RE = re.compile(
    r"(?m)^\s*(?:import|namespace|section|end|theorem|lemma|axiom|opaque|def|"
    r"instance|class|structure|inductive|#eval|#check|#reduce|set_option)\b"
)
DECLARATION_NAME_RE = re.compile(
    r"(?m)(\b(?:private\s+)?(?:protected\s+)?(?:theorem|lemma)\s+)"
    r"([A-Za-z_][A-Za-z0-9_'.]*)"
)


def clean_proof(value: Any) -> str:
    """Actual evaluator cleanup; lexical, NOT a Lean-aware parser or repair.

    Textual markers are processed even inside strings/comments in the old
    implementation; do not describe this routine as semantic proof extraction.
    It never supplies 'by', changes tactics, or repairs a failed proof.
    """
    text = str(value or "").strip()
    if "\n" not in text and "\\n" in text:
        text = text.replace("\\n", "\n").replace("\\t", "\t")
    fenced = FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    for marker in TEXT_CLEANUP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    for prefix in ("PROOF_SUFFIX:", "Proof suffix:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.startswith(":="):
        text = text[2:].strip()
    return text


def precheck_proof(proof: str) -> str | None:
    """Lexical precheck only; actual acceptance still requires Lean verification."""
    if not proof:
        return "empty_generation"
    if PLACEHOLDER_RE.search(proof):
        return "placeholder_sorry_or_admit"
    if COMMAND_INJECTION_RE.search(proof):
        return "top_level_command_injection"
    return None


def restore_original_name(statement: str, *, kind: str, original_name: str) -> str:
    header = statement.strip()
    if kind == "example":
        return header
    if not original_name:
        raise ValueError("named theorem/lemma has no original_name")
    match = DECLARATION_NAME_RE.search(header)
    if match is None:
        raise ValueError("cannot locate anonymized theorem/lemma name in statement")
    return header[:match.start(2)] + original_name + header[match.end(2):]


def replacement_declaration(index_row: dict[str, Any], proof: str) -> str:
    """index_row is the verified source-index row, not the raw training view."""
    header = restore_original_name(
        str(index_row["statement"]), kind=str(index_row["kind"]),
        original_name=str(index_row.get("original_name") or ""),
    ).rstrip()
    return header + "\n" + proof.strip()


def replace_original_declaration(source: bytes, index_row: dict[str, Any],
                                  generated_text: str) -> tuple[bytes, str]:
    """Pure byte-span replacement in the real source; preserve all following code.

    Requires the existing source index's whole-file and original-command hashes.
    Returns a candidate Lean file, not a claim that Lean accepts the proof.
    """
    proof = clean_proof(generated_text)
    error = precheck_proof(proof)
    if error is not None:
        raise ValueError(error)
    if hashlib.sha256(source).hexdigest() != index_row["source_sha256"]:
        raise ValueError("Original source hash mismatch")
    start, end = int(index_row["start_byte"]), int(index_row["end_byte"])
    if not 0 <= start < end <= len(source):
        raise ValueError("Original declaration byte span is outside source")
    if hashlib.sha256(source[start:end]).hexdigest() != index_row["original_command_sha256"]:
        raise ValueError("Original command hash mismatch")
    replacement = replacement_declaration(index_row, proof).encode("utf-8")
    return source[:start] + replacement + source[end:], proof


def self_test() -> None:
    """Offline mechanics checks with a synthetic tokenizer, not model evidence."""
    class ToyTokenizer:
        eos_token_id, bos_token_id, pad_token_id = 999999, 999998, None

        def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}

        def decode(self, ids: Iterable[int], *, skip_special_tokens: bool) -> str:
            return "".join(chr(value) for value in ids if value < 999998)

    tokenizer = ToyTokenizer()
    statement, proof = "theorem anonymous_target : True :=", "by\n  trivial"
    mc = {"command_id": "m", "command_uid": "A::m", "position": 0, "text": "def unitValue := ()"}
    ec = {"command_id": "e", "command_uid": "A::e", "position": 1, "text": "namespace Demo"}
    contexts = {"C0": [], "CMath": [mc], "CEnv": [ec], "CFull": [ec, mc]}
    prompt = render_training_prompt(statement, [ec, mc])
    assert prompt == mc["text"] + "\n\n" + ec["text"] + "\n\n" + statement + "\n"
    assert encode_prompt(tokenizer, prompt, "llama31")[0] == tokenizer.bos_token_id
    assert encode_prompt(tokenizer, prompt, "dllm")[0] != tokenizer.bos_token_id
    assert encode_answer(tokenizer, proof).count(tokenizer.eos_token_id) == 1
    packed = prepare_budgeted_views(tokenizer, statement, proof, contexts, backbone="dllm")
    assert packed["retained"]
    for budget in INPUT_BUDGETS:
        for row in packed["views_by_budget"][str(budget)].values():
            assert row["target"] == proof and row["prompt_length"] <= budget
            assert row["sequence_length"] <= budget + 1024
            assert row["input_ids"][row["prompt_length"]:].count(tokenizer.eos_token_id) == 1
            assert len(generator_input_from_training_row(row, tokenizer, "dllm")) == row["prompt_length"]
    long_mc = dict(mc, text="def longValue := " + "a" * 1100)
    trimmed = prepare_budgeted_views(tokenizer, statement, proof, {"C0": [], "CMath": [long_mc], "CEnv": [], "CFull": [long_mc]}, backbone="dllm")
    assert trimmed["views_by_budget"]["1024"]["CMath"]["context_items"] == []
    assert not prepare_budgeted_views(tokenizer, statement, "a" * 1024, contexts, backbone="dllm")["retained"]
    assert prepare_budgeted_views(tokenizer, "a" * 257, proof, contexts, backbone="dllm")["retained"]
    assert not prepare_budgeted_views(tokenizer, "a" * 1024, proof, contexts, backbone="dllm")["retained"]
    for length in (256, 512, 1024):
        assert generation_limits([1] * length, input_limit=length)["max_new_tokens"] == 1024
    kwargs = ar_generation_kwargs(tokenizer, [1] * 1024)
    assert kwargs["do_sample"] is False and "temperature" not in kwargs
    assert kwargs["max_new_tokens"] == 1024 and kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert ar_generation_kwargs(tokenizer, [1], count=32, temperature=0.8)["do_sample"]
    suffix = list(map(ord, proof)) + [tokenizer.eos_token_id] + list(map(ord, "not proof"))
    assert decode_generated_suffix(tokenizer, [1, 2] + suffix, 2) == proof
    assert decode_diffusion_window(tokenizer, suffix) == proof
    assert clean_proof("```lean\n:= by\n  trivial\n<END_OF_PROOF>\n```") == proof
    assert precheck_proof("by\n  sorry") == "placeholder_sorry_or_admit"
    assert precheck_proof("theorem other : True := by trivial") == "top_level_command_injection"
    original = b"theorem original : True := by trivial"
    source = b"-- prefix\n" + original + b"\n-- preserved suffix\n"
    index = {"statement": statement, "kind": "theorem", "original_name": "original", "start_byte": 10, "end_byte": 10 + len(original), "source_sha256": hashlib.sha256(source).hexdigest(), "original_command_sha256": hashlib.sha256(original).hexdigest()}
    replaced, extracted = replace_original_declaration(source, index, proof)
    assert extracted == proof and replaced.startswith(b"-- prefix\n") and replaced.endswith(b"\n-- preserved suffix\n")
    assert b"theorem original : True :=\nby\n  trivial" in replaced
    print("PASS: offline length, serialization, EOS, four-view packing, stopping, extraction, and byte replacement checks; no model/API/Lean executed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
