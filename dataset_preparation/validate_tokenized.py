"""Independently audit finished real-tokenizer training and selector outputs.

The creator's tokenization/packing helpers are deliberately not imported.
This auditor reloads the local real tokenizer and independently re-encodes all
saved prompts and reference proofs. Its only write is a separate validation
report under tokenized/BACKBONE, never an immutable raw dataset file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .validate import (
    SPLITS, VIEWS, ValidationError, file_hash, object_hash, read_json,
    reject_proof_fields, require, rows, safe_relative, text_hash, unique_ids,
)


def ordered(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (item["source_order"], item["command_uid"]))


def ids(items: list[dict[str, Any]]) -> list[str]:
    return [item["command_id"] for item in items]


def uids(items: list[dict[str, Any]]) -> list[str]:
    return [item["command_uid"] for item in items]


def training_prompt(statement: str, items: list[dict[str, Any]]) -> str:
    return "\n\n".join([*(item["text"] for item in ordered(items)), statement]).rstrip("\r\n") + "\n"


def selector_prompt(statement: str, ec: list[dict[str, Any]], mc: list[dict[str, Any]]) -> str:
    ec, mc = ordered(ec), ordered(mc)
    roles = {uid: "EC" for uid in uids(ec)}
    require(not set(roles) & set(uids(mc)), "selector MC/EC identities overlap")
    return "\n\n".join([*(item["text"] for item in ordered(ec + mc)), statement]).rstrip("\r\n") + "\n"


def scope_closures(environment: list[dict[str, Any]], status: str) -> tuple[dict[str, list[str]], bool]:
    if status != "syntactic_valid":
        return {}, False
    index = {item["command_id"]: item for item in environment}
    require(len(index) == len(environment), "duplicate shared EC IDs")
    required: dict[str, set[str]] = {}
    for cid, item in index.items():
        references = item.get("required_scope_command_ids", [])
        pairs = item.get("scope_pair_command_ids", [])
        if not isinstance(references, list) or not isinstance(pairs, list):
            return {}, False
        dependencies = set(references + pairs)
        if not dependencies <= set(index):
            return {}, False
        if item.get("scope_role") == "block_closer" and not pairs:
            return {}, False
        if item.get("scope_role") == "block_opener" and item.get("target_scope_active") is False and not pairs:
            return {}, False
        required[cid] = dependencies
    closures = {}
    for cid in index:
        seen, pending = set(), [cid]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(required[current] - seen)
        closures[cid] = ids(ordered([index[member] for member in seen]))
    return closures, True


def coverage(dependencies: dict[str, Any], visible: list[str], valid: bool) -> dict[str, Any]:
    dependency_ids = set(dependencies["command_ids"])
    if not valid:
        return {"status": "invalid_input", "coverage": None, "dependency_count": len(dependency_ids), "visible_dependency_count": None}
    status = dependencies["status"]
    if status not in {"complete_nonempty", "complete"}:
        return {"status": status, "coverage": None, "dependency_count": len(dependency_ids), "visible_dependency_count": None}
    if not dependency_ids:
        return {"status": "no_eligible_local_dependencies", "coverage": None, "dependency_count": 0, "visible_dependency_count": 0}
    count = len(dependency_ids & set(visible))
    category = "zero" if count == 0 else "full" if count == len(dependency_ids) else "partial"
    return {"status": category, "coverage": count / len(dependency_ids), "dependency_count": len(dependency_ids), "visible_dependency_count": count}


def length_summary(values: list[int]) -> dict[str, Any]:
    """Independently count exact inclusive threshold bins and nearest ranks."""
    sorted_values = sorted(values)
    size = len(values)
    empty = values.count(0)
    nonempty = size - empty
    bins = {label: 0 for label in ("0", "1-256", "257-512", "513-1024", "1025-2048", "2049-4096", ">4096")}
    for value in values:
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, "invalid measured context length")
        if value == 0:
            label = "0"
        else:
            label = next((label for threshold, label in ((256, "1-256"), (512, "257-512"), (1024, "513-1024"), (2048, "1025-2048"), (4096, "2049-4096")) if value <= threshold), ">4096")
        bins[label] += 1
    coverage_counts = {}
    for budget in (256, 512, 1024):
        all_count = sum(value <= budget for value in values)
        nonempty_count = sum(0 < value <= budget for value in values)
        coverage_counts[str(budget)] = {
            "all_targets_count": all_count, "all_targets_percent": 100 * all_count / size if size else None,
            "nonempty_targets_count": nonempty_count, "nonempty_targets_percent": 100 * nonempty_count / nonempty if nonempty else None,
        }
    above = sum(value > 1024 for value in values)
    return {
        "targets": size, "empty": empty, "nonempty": nonempty,
        "mean": sum(values) / size if size else None, "maximum": max(values) if size else None,
        "nearest_rank_quantiles": {name: sorted_values[max(0, math.ceil(size * probability) - 1)] if size else None for name, probability in (("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99))},
        "histogram_counts": bins, "complete_context_coverage": coverage_counts,
        "above_1024_count": above, "above_1024_percent_all": 100 * above / size if size else None,
    }


def training_eligibility(prompt_length: int, answer_length: int,
                         budgets: tuple[int, ...], answer_limit: int) -> tuple[dict[str, bool], list[str]]:
    eligible = {str(budget): prompt_length <= budget and answer_length <= answer_limit for budget in budgets}
    reasons = []
    if answer_length > answer_limit:
        reasons.append("complete_reference_proof_plus_one_EOS_exceeds_output_limit")
    if prompt_length > min(budgets):
        reasons.append("complete_statement_prompt_exceeds_smallest_input_budget")
    return eligible, reasons


class TokenizedValidator:
    def __init__(self, root: Path, backbone: str, tokenizer_dir: Path,
                 training_directory: str = "training", selector_directory: str = "selector") -> None:
        self.root = root.resolve()
        self.name = "llada" if backbone in {"llada", "dllm"} else "llama31"
        self.backbone = "dllm" if self.name == "llada" else "llama31"
        self.output = self.root / "tokenized" / self.name
        self.training_output = self.output / safe_relative(training_directory)
        self.selector_output = self.output / safe_relative(selector_directory)
        self.tokenizer_dir = tokenizer_dir.resolve()
        # Metrics are written only after each corresponding stream completes.
        self.training_metrics = read_json(self.training_output / "tokenization_metrics.json")
        self.input_budgets = tuple(self.training_metrics["input_budgets"])
        require(bool(self.input_budgets) and self.input_budgets == tuple(sorted(set(self.input_budgets)))
                and set(self.input_budgets) <= {256, 512, 1024}, "invalid saved input budgets")
        self.answer_limit = 1024
        self.selector_metrics = {
            budget: read_json(self.selector_output / f"input_{budget}/tokenization_metrics.json")
            for budget in self.input_budgets
        }
        self.length_metrics = read_json(self.output / "raw_length_statistics/length_statistics.json")
        self.catalogs: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
        self.context_length_cache: OrderedDict[str, int] = OrderedDict()
        self.report: dict[str, Any] = {
            "schema_version": "independent-real-tokenizer-validation-v2", "valid": False,
            "auditor_source_sha256": file_hash(Path(__file__).resolve()),
            "creator_tokenization_or_packing_helpers_imported": False,
            "backbone": self.backbone, "tokenized_directory": str(self.output),
            "training_directory_audited": str(self.training_output),
            "selector_directory_audited": str(self.selector_output),
            "validation_scope": "exact_real_token_ids_training_spans_and_selector_packing_not_model_generation",
            "model_generation_or_Lean_success_validated": False,
        }

    def stage(self, name: str) -> None:
        self.report["stage"] = name
        print(json.dumps({"tokenized_validation_stage": name, "backbone": self.name}), flush=True)

    def load_tokenizer(self) -> None:
        self.stage("reload_exact_local_tokenizer")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir), local_files_only=True, use_fast=True, trust_remote_code=False)
        require(self.tokenizer.is_fast and self.tokenizer.eos_token_id is not None, "actual fast tokenizer/EOS unavailable")
        training_meta = self.training_metrics["tokenizer"]
        require(all(training_meta == metrics["tokenizer"] for metrics in self.selector_metrics.values()), "training/selector tokenizer metadata differ")
        require(training_meta == self.length_metrics["tokenizer"], "raw length statistics tokenizer differs")
        for filename, expected in training_meta["tokenizer_file_sha256"].items():
            require(file_hash(self.tokenizer_dir / safe_relative(filename)) == expected, f"actual tokenizer file hash differs: {filename}")
        backend_hash = text_hash(self.tokenizer.backend_tokenizer.to_str())
        require(backend_hash == training_meta["tokenizer_backend_sha256"], "actual tokenizer backend differs")
        for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
            require(getattr(self.tokenizer, field) == training_meta[field], f"actual tokenizer {field} differs")
        require(type(self.tokenizer).__name__ == training_meta["tokenizer_class"] and training_meta["add_bos_token"] is (self.backbone == "llama31"), "actual tokenizer class/BOS policy differs")
        require(training_meta["local_files_only"] is True, "tokenization was not recorded as local")
        import transformers
        require(transformers.__version__ == training_meta["transformers_version"] and importlib.metadata.version("tokenizers") == training_meta["tokenizers_version"], "audit runtime tokenizer library versions differ from creation")
        self.report["tokenizer"] = {"source": str(self.tokenizer_dir), "file_sha256": training_meta["tokenizer_file_sha256"], "backend_sha256": backend_hash, "class": type(self.tokenizer).__name__, "transformers_version": transformers.__version__, "tokenizers_version": importlib.metadata.version("tokenizers"), "bos_token_id": self.tokenizer.bos_token_id, "eos_token_id": self.tokenizer.eos_token_id, "exact_saved_creation_identity_match": True}

    def encode(self, prompt: str) -> list[int]:
        encoded = list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if self.backbone == "llama31":
            bos = self.tokenizer.bos_token_id
            require(bos is not None, "Llama BOS missing")
            if not encoded or encoded[0] != bos:
                encoded.insert(0, bos)
        return encoded

    def proof_ids(self, proof: str) -> list[int]:
        require(isinstance(proof, str) and bool(proof.strip()) and not proof.startswith(("\n", "\r")), "reference proof/boundary invalid")
        encoded = list(self.tokenizer(proof, add_special_tokens=False)["input_ids"])
        require(bool(encoded) and self.tokenizer.eos_token_id not in encoded, "reference proof already contains EOS or is empty")
        return encoded + [self.tokenizer.eos_token_id]

    def context_length(self, text: str) -> int:
        if not text:
            return 0
        value = self.context_length_cache.pop(text, None)
        if value is None:
            value = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(text) <= 65536:
            self.context_length_cache[text] = value
            while len(self.context_length_cache) > 4096:
                self.context_length_cache.popitem(last=False)
        return value

    def catalog(self, source_file: str) -> dict[str, dict[str, Any]]:
        catalog = self.catalogs.pop(source_file, None)
        if catalog is None:
            path = self.root / "command_mapping/source_commands" / safe_relative(source_file, ".lean").with_suffix(".jsonl")
            catalog = {record["command_id"]: record for record in rows(path)}
        self.catalogs[source_file] = catalog
        while len(self.catalogs) > 32:
            self.catalogs.popitem(last=False)
        return catalog

    def validate_training(self) -> None:
        self.stage("training_saved_input_budgets_four_views_and_separate_proof_EOS_limit")
        metrics = self.training_metrics
        require(metrics["backbone"] == self.backbone and metrics["protocol"] == "leancontext-independent-input-proof-budgets-common-membership-v1", "final separate-budget training protocol differs")
        require(metrics["input_budgets"] == list(self.input_budgets) and metrics["output_limit"] == self.answer_limit, "authoritative training input budgets or independent answer limit differ")
        for field in ("output_limit_includes_one_terminal_EOS", "common_membership_across_views_and_input_budgets", "prompt_budget_includes_BOS_and_boundary_newline"):
            require(metrics[field] is True, f"training budget/fairness protocol differs: {field}")
        require(metrics["sequence_packing"] is False and metrics["padding_serialized"] is False and metrics["model_generation_performed"] is False, "training padding, packing, or generation metadata differs")
        results = {}
        all_context_lengths = {component: [] for component in ("MC", "EC", "Full")}
        require(self.length_metrics["backbone"] == self.backbone and self.length_metrics["raw_membership_preserved"] is True and self.length_metrics["model_generation_performed"] is False, "raw length statistics backbone/membership/generation metadata differs")
        measurement = self.length_metrics["measurement"]
        require(all(measurement[field] is False for field in ("includes_statement", "includes_prompt_role_or_id_headers", "includes_BOS_or_EOS")) and measurement["truncation"] == "none", "raw context lengths incorrectly include prompt/BOS/EOS/statement or trimming")
        global_seen = set()
        label_rows = 0
        for split in SPLITS:
            raw_iter = {view: iter(rows(self.root / f"training/views/{view}/{split}.jsonl")) for view in VIEWS}
            saved_iter = {(budget, view): iter(rows(self.training_output / f"input_{budget}/views/{view}/{split}.jsonl")) for budget in self.input_budgets for view in VIEWS}
            members = iter(rows(self.training_output / f"membership/{split}.jsonl"))
            rejects = iter(rows(self.training_output / f"rejections/{split}.jsonl"))
            measured_rows = iter(rows(self.output / f"raw_length_statistics/{split}.jsonl"))
            split_context_lengths = {component: [] for component in all_context_lengths}
            stats = {
                "seen": 0, "retained": 0, "rejected": 0, "rejection_reason_counts": Counter(),
                "feasible_targets_by_input_budget": {str(budget): 0 for budget in self.input_budgets},
                "budgets": {str(budget): {"context_trimmed_targets": 0, "max_prompt_lengths": {view: 0 for view in VIEWS}, "max_sequence_lengths": {view: 0 for view in VIEWS}, "max_answer_length": 0} for budget in self.input_budgets},
            }
            while True:
                originals = {view: next(iterator, None) for view, iterator in raw_iter.items()}
                if all(value is None for value in originals.values()):
                    break
                require(all(value is not None for value in originals.values()), f"{split}: raw view cardinalities differ")
                base = originals["C0"]
                sid, statement, proof = base["sample_id"], base["statement"], base["target"]
                require(sid not in global_seen, f"duplicate raw training target: {sid}")
                global_seen.add(sid)
                for view, original in originals.items():
                    require(original["sample_id"] == sid and original["target_id"] == sid and original["split"] == split and original["view"] == view and original["statement"] == statement and original["target"] == proof, f"raw four-view target/statement/proof mismatch: {sid}")
                original_full = ordered(originals["CFull"]["context_items"])
                require(all(item["role"] in {"MC", "EC"} for item in original_full), f"raw length unknown context role: {sid}")
                require(original_full == originals["CFull"]["context_items"] and len(uids(original_full)) == len(set(uids(original_full))), f"raw Full duplicate/out-of-order context: {sid}")
                require(originals["CMath"]["context_items"] == [item for item in original_full if item["role"] == "MC"] and originals["CEnv"]["context_items"] == [item for item in original_full if item["role"] == "EC"], f"raw Math/Env not exact shared Full subsets: {sid}")
                for item in original_full:
                    require(item["command_uid"] == base["source_file"] + "::" + item["command_id"] and item["command_uid"] != base["target_command_uid"] and item["source_span"]["start_byte"] < base["target_source_span"]["start_byte"] and item["source_span"]["end_byte"] <= base["target_source_span"]["start_byte"], f"raw/retained training context includes current target, later, or different-source command: {sid}")
                context_texts = {
                    "MC": "\n\n".join(item["text"] for item in original_full if item["role"] == "MC"),
                    "EC": "\n\n".join(item["text"] for item in original_full if item["role"] == "EC"),
                    "Full": "\n\n".join(item["text"] for item in original_full),
                }
                actual_lengths = {component: self.context_length(text) for component, text in context_texts.items()}
                measured = next(measured_rows, None)
                require(measured is not None and measured["sample_id"] == sid and measured["target_id"] == sid and measured["split"] == split and measured["context_token_lengths"] == actual_lengths, f"raw context actual tokenizer lengths/IDs differ: {sid}")
                for component, value in actual_lengths.items():
                    split_context_lengths[component].append(value)
                    all_context_lengths[component].append(value)
                member = next(members, None)
                require(member is not None and member["sample_id"] == sid and member["target_id"] == sid and member["split"] == split, f"joint membership differs: {sid}")
                answer_ids = self.proof_ids(proof)
                bare_ids = self.encode(training_prompt(statement, []))
                eligible, reasons = training_eligibility(len(bare_ids), len(answer_ids), self.input_budgets, self.answer_limit)
                retained = not reasons
                expected_member = {
                    "sample_id": sid, "target_id": sid, "split": split,
                    "eligible_by_input_budget": eligible, "bare_prompt_length": len(bare_ids),
                    "answer_length": len(answer_ids), "input_budgets": list(self.input_budgets),
                    "output_limit": self.answer_limit, "retained": retained,
                }
                if reasons:
                    expected_member.update(reason=reasons[0], reasons=reasons)
                require(member == expected_member, f"joint independent input/answer eligibility metadata differs: {sid}")
                stats["seen"] += 1
                for budget, feasible in eligible.items():
                    stats["feasible_targets_by_input_budget"][budget] += feasible
                if not retained:
                    rejected = next(rejects, None)
                    require(rejected == member, f"rejection metadata differs: {sid}")
                    stats["rejected"] += 1
                    stats["rejection_reason_counts"].update(reasons)
                    continue
                raw_full = original_full
                math_uids = set(uids(originals["CMath"]["context_items"]))
                env_uids = set(uids(originals["CEnv"]["context_items"]))
                require(not math_uids & env_uids and math_uids | env_uids == set(uids(raw_full)), f"raw Full MC/EC union differs: {sid}")
                stats["retained"] += 1
                encoded_prefixes: dict[int, dict[str, tuple[list[dict[str, Any]], str, list[int]]]] = {}
                prompt_token_cache = {training_prompt(statement, []): bare_ids}

                def encoded_prefix(prefix: int) -> dict[str, tuple[list[dict[str, Any]], str, list[int]]]:
                    if prefix not in encoded_prefixes:
                        selected = raw_full[:prefix]
                        contexts = {"C0": [], "CMath": [item for item in selected if item["command_uid"] in math_uids], "CEnv": [item for item in selected if item["command_uid"] in env_uids], "CFull": selected}
                        encoded_prefixes[prefix] = {}
                        for view, visible in contexts.items():
                            prompt = training_prompt(statement, visible)
                            if prompt not in prompt_token_cache:
                                prompt_token_cache[prompt] = self.encode(prompt)
                            encoded_prefixes[prefix][view] = (visible, prompt, prompt_token_cache[prompt])
                    return encoded_prefixes[prefix]

                original_encoded = encoded_prefix(len(raw_full))
                previous_prefix = -1
                for budget in self.input_budgets:
                    prefix = len(raw_full)
                    while any(len(row[2]) > budget for row in encoded_prefix(prefix).values()):
                        prefix -= 1
                        require(prefix >= 0, f"retained bare statement does not fit input budget: {sid}/{budget}")
                    require(prefix >= previous_prefix, f"context retention decreased with a larger input budget: {sid}/{budget}")
                    previous_prefix = prefix
                    removed = raw_full[prefix:]
                    budget_stats = stats["budgets"][str(budget)]
                    budget_stats["context_trimmed_targets"] += bool(removed)
                    budget_stats["max_answer_length"] = max(budget_stats["max_answer_length"], len(answer_ids))
                    for view, (visible, prompt, prompt_ids) in encoded_prefix(prefix).items():
                        saved_row = next(saved_iter[budget, view], None)
                        original = originals[view]
                        require(saved_row is not None, f"missing common retained target: {sid}/{budget}/{view}")
                        require(saved_row["sample_id"] == sid and saved_row["target_id"] == sid and saved_row["split"] == split and saved_row["view"] == view and saved_row["statement"] == statement and saved_row["target"] == proof and saved_row["reference_proof"] == proof, f"saved condition changed membership/statement/full proof: {sid}/{budget}/{view}")
                        for field in ("proof_family", "command_id", "target_command_id", "target_command_uid", "source_file", "source_module", "target_kind", "original_name", "target_source_span", "template_version"):
                            require(saved_row[field] == original[field], f"saved condition changed target provenance/template: {sid}/{budget}/{view}: {field}")
                        require(saved_row["context_items"] == visible and visible == ordered(visible) and len(uids(visible)) == len(set(uids(visible))), f"shared prefix/derived context/refill/source order differs: {sid}/{budget}/{view}")
                        expected_ids = prompt_ids + answer_ids
                        require(saved_row["prompt"] == prompt and prompt.endswith("\n") and not prompt.endswith("\n\n") and saved_row["input_ids"] == expected_ids, f"exact prompt template or actual token IDs differ: {sid}/{budget}/{view}")
                        for field, expected in (("prompt_length", len(prompt_ids)), ("input_length", len(prompt_ids)), ("answer_length", len(answer_ids)), ("target_length", len(answer_ids)), ("sequence_length", len(expected_ids)), ("input_limit", budget), ("output_limit", self.answer_limit), ("answer_eos_count", 1), ("answer_final_token_id", self.tokenizer.eos_token_id)):
                            require(saved_row[field] == expected, f"actual training length/EOS metadata differs: {sid}/{budget}/{view}: {field}")
                        require(len(prompt_ids) <= budget and len(answer_ids) <= self.answer_limit and len(expected_ids) <= budget + self.answer_limit and saved_row["answer_span"] == {"start": len(prompt_ids), "end_exclusive": len(expected_ids)}, f"independent prompt/answer limits or supervised span differ: {sid}/{budget}/{view}")
                        require(expected_ids[-1] == self.tokenizer.eos_token_id and expected_ids[len(prompt_ids):].count(self.tokenizer.eos_token_id) == 1, f"not exactly one terminal supervised EOS: {sid}/{budget}/{view}")
                        if "labels" in saved_row:
                            require(saved_row["labels"] == [-100] * len(prompt_ids) + answer_ids, f"prompt/padding labels supervised or answer labels differ: {sid}/{budget}/{view}")
                            label_rows += 1
                        if "attention_mask" in saved_row:
                            require(saved_row["attention_mask"] == [1] * len(expected_ids), f"serialized padding differs: {sid}/{budget}/{view}")
                        require(saved_row["context_ids"] == ids(visible) and saved_row["context_uids"] == uids(visible) and saved_row["visible_command_ids"] == ids(visible) and saved_row["visible_command_uids"] == uids(visible), f"stale retained context IDs/UIDs: {sid}/{budget}/{view}")
                        require(saved_row["raw_context_ids"] == original["context_ids"] and saved_row["raw_context_uids"] == original["context_uids"], f"raw context trace differs: {sid}/{budget}/{view}")
                        require(saved_row["removed_command_ids"] == ids(removed) and saved_row["removed_command_uids"] == uids(removed), f"shared removed command trace differs: {sid}/{budget}/{view}")
                        require(saved_row["view_removed_command_ids"] == [item["command_id"] for item in removed if item["command_uid"] in set(original["context_uids"])], f"per-view removal differs: {sid}/{budget}/{view}")
                        require(saved_row["untrimmed_prompt_length"] == len(original_encoded[view][2]), f"untrimmed actual prompt length differs: {sid}/{budget}/{view}")
                        require(saved_row["tokenization_status"] == "complete_real_tokenizer" and saved_row["protocol"] == metrics["protocol"], f"training completion/protocol differs: {sid}/{budget}/{view}")
                        budget_stats["max_prompt_lengths"][view] = max(budget_stats["max_prompt_lengths"][view], len(prompt_ids))
                        budget_stats["max_sequence_lengths"][view] = max(budget_stats["max_sequence_lengths"][view], len(expected_ids))
            require(next(members, None) is None and next(rejects, None) is None and next(measured_rows, None) is None and all(next(iterator, None) is None for iterator in saved_iter.values()), f"extra membership/rejection/view/measurement rows: {split}")
            stats["rejection_reason_counts"] = dict(stats["rejection_reason_counts"])
            require(stats == metrics["splits"][split], f"training metrics do not match actual data: {split}")
            require({component: length_summary(values) for component, values in split_context_lengths.items()} == self.length_metrics["splits"][split], f"raw length histogram/threshold/quantile counts differ: {split}")
            results[split] = stats
            print(json.dumps({"audited_training_split": split, "backbone": self.name, **stats}), flush=True)
        self.report["training"] = {"raw_targets": len(global_seen), "input_budgets": list(self.input_budgets), "separate_complete_proof_plus_one_EOS_limit": self.answer_limit, "maximum_sequence_lengths_are_input_budget_plus_answer_limit": True, "split_counts": results, "common_membership_across_configured_budgets_and_four_views": True, "per_budget_feasibility_independently_verified_on_all_raw_targets": True, "exact_real_prompt_and_reference_proof_ids": True, "four_view_shared_minimal_whole_command_trimming_per_budget": True, "one_terminal_EOS_inside_answer_span": True, "serialized_label_rows": label_rows, "supervision_representation": "answer_span; serialized labels checked when present"}
        summaries = {component: length_summary(values) for component, values in all_context_lengths.items()}
        require(summaries == self.length_metrics["all_splits"], "all-split raw context histograms/thresholds/quantiles differ")
        self.report["raw_context_length_statistics"] = {"all_raw_target_actual_context_only_measurements_match": True, "no_statement_BOS_EOS_or_template_headers": True, "histograms_inclusive_thresholds_and_nearest_rank_quantiles_match": True, "all_splits": summaries, "statistics_file_sha256": file_hash(self.output / "raw_length_statistics/length_statistics.json")}

    def independently_pack(self, raw: dict[str, Any], input_limit: int) -> dict[str, Any]:
        statement = raw["statement"]
        raw_env = raw["shared_environment"]
        env_index = {item["command_id"]: item for item in raw_env}
        priority = unique_ids(raw["shared_environment_priority"], "raw environment priority")
        require(set(priority) == set(env_index), "raw environment priority incomplete")
        closures, scope_valid = scope_closures(raw_env, raw["EC_scope_stack_status"])
        bare_length = len(self.encode(selector_prompt(statement, [], [])))
        if not scope_valid or bare_length > input_limit:
            return {"valid": False, "reason": "shared_environment_scope_records_missing_or_invalid" if not scope_valid else "target_and_fixed_template_exceed_input_limit", "required_fixed_input_length": bare_length}
        selected_ec: dict[str, dict[str, Any]] = {}
        skipped_cohorts = []
        for cid in priority:
            proposal = dict(selected_ec)
            proposal.update({member: env_index[member] for member in closures[cid]})
            if len(self.encode(selector_prompt(statement, list(proposal.values()), []))) <= input_limit:
                selected_ec = proposal
            else:
                skipped_cohorts.append({"priority_command_id": cid, "cohort_command_ids": closures[cid]})
        ec = ordered(list(selected_ec.values()))
        result: dict[str, Any] = {"valid": True, "ec": ec, "environment_budget_skipped_command_ids": [cid for cid in priority if cid not in selected_ec], "environment_budget_skipped_cohorts": skipped_cohorts}
        catalog = self.catalog(raw["source_file"])
        for policy, field in (("baseline", "baseline_ranking"), ("dependency_prioritized", "dependency_prioritized_ranking")):
            selected, skipped = [], []
            for cid in raw[field]:
                command = catalog[cid]
                if len(self.encode(selector_prompt(statement, ec, selected + [command]))) <= input_limit:
                    selected.append(command)
                else:
                    skipped.append(cid)
            mc = ordered(selected)
            prompt = selector_prompt(statement, ec, mc)
            result[policy] = {"mc": mc, "prompt": prompt, "input_ids": self.encode(prompt), "budget_skipped_command_ids": skipped}
        return result

    def validate_selector(self) -> None:
        self.stage("selector_saved_input_budgets_frozen_2000_exact_shared_EC_and_packing")
        self.report["selector"] = {"input_budgets": list(self.input_budgets), "separate_output_allowance": self.answer_limit, "budgets": {}}
        for budget in self.input_budgets:
            self.validate_selector_budget(budget)

    def validate_selector_budget(self, budget: int) -> None:
        metrics = self.selector_metrics[budget]
        require(metrics["backbone"] == self.backbone and metrics["input_limit"] == budget and metrics["output_limit"] == self.answer_limit and metrics["model_generation_performed"] is False, f"selector tokenizer protocol/budget/generation metadata differs: {budget}")
        manifest = read_json(self.root / "selector/frozen_test_manifest.json")
        frozen = unique_ids(manifest["target_ids"], "frozen selector IDs")
        require(len(frozen) == 2000, "selector frozen target count differs")
        packed_iter = iter(rows(self.selector_output / f"input_{budget}/pairs.jsonl"))
        actual_ids = []
        counts = {"seen": 0, "both_valid": 0, "both_invalid": 0, "coverage_categories": {"baseline": Counter(), "dependency_prioritized": Counter()}}
        max_lengths = {"baseline": 0, "dependency_prioritized": 0}
        for raw in rows(getattr(self, "selector_raw_pairs_path", self.root / "selector/pairs.jsonl")):
            saved = next(packed_iter, None)
            sid = raw["sample_id"]
            require(saved is not None and saved["sample_id"] == sid and saved["target_id"] == sid, f"packed selector missing/changed ID: {sid}")
            actual_ids.append(sid)
            reject_proof_fields(saved, f"tokenized selector {sid}")
            for field in ("statement", "split", "source_file", "target_command_id", "target_command_uid", "candidate_pool", "baseline_ranking", "dependency_prioritized_ranking", "target_type_dependencies", "EC_same_for_both_policies", "EC_scope_stack_status"):
                require(saved[field] == raw[field], f"selector raw proof-blind field changed: {sid}: {field}")
            require(saved["backbone"] == self.backbone and saved["input_limit"] == budget and saved["output_limit"] == self.answer_limit and saved["EC_same_for_both_policies"] is True, f"selector backbone/shared-EC/budget differs: {sid}/{budget}")
            require(saved["raw_shared_environment_command_ids"] == ids(raw["shared_environment"]), f"selector raw EC trace differs: {sid}")
            expected = self.independently_pack(raw, budget)
            require(saved["baseline"]["valid_input"] is expected["valid"] and saved["dependency_prioritized"]["valid_input"] is expected["valid"], f"selector paired validity differs: {sid}")
            counts["seen"] += 1
            counts["both_valid" if expected["valid"] else "both_invalid"] += 1
            if not expected["valid"]:
                require(saved["shared_environment"] == [] and saved["shared_environment_ids"] == [], f"invalid selector contains visible EC: {sid}")
            else:
                require(saved["shared_environment"] == expected["ec"] and saved["shared_environment_ids"] == ids(expected["ec"]), f"selector shared EC cohort selection differs: {sid}")
                for field in ("environment_budget_skipped_command_ids", "environment_budget_skipped_cohorts"):
                    require(saved[field] == expected[field], f"selector EC skipped cohort trace differs: {sid}")
            for policy in ("baseline", "dependency_prioritized"):
                row = saved[policy]
                if not expected["valid"]:
                    require(row["reason"] == expected["reason"] and row["required_fixed_input_length"] == expected["required_fixed_input_length"] and row["prompt"] is None and row["input_ids"] is None and row["input_length"] is None, f"invalid selector length/reason/prompt differs: {sid}/{policy}")
                    for field in ("selected_command_ids", "visible_command_ids", "visible_mc_command_ids", "visible_ec_command_ids", "all_visible_command_ids"):
                        require(row[field] == [], f"invalid selector has visible IDs: {sid}/{policy}")
                    visible = []
                else:
                    ec, mc = expected["ec"], expected[policy]["mc"]
                    visible = ids(mc)
                    require(row["selected_mathematical_context"] == mc and row["prompt"] == expected[policy]["prompt"] and row["input_ids"] == expected[policy]["input_ids"], f"selector whole commands/template/actual token IDs differ: {sid}/{policy}")
                    length = len(row["input_ids"])
                    require(length <= budget and row["input_length"] == row["prompt_length"] == length, f"selector input budget/actual length differs: {sid}/{policy}/{budget}")
                    for field in ("selected_command_ids", "visible_command_ids", "visible_mc_command_ids"):
                        require(row[field] == visible, f"selector retained MC metadata differs: {sid}/{policy}")
                    require(row["visible_command_uids"] == uids(mc) and row["visible_ec_command_ids"] == ids(ec) and row["all_visible_command_ids"] == ids(ordered(ec + mc)), f"selector shared EC/all visible IDs differ: {sid}/{policy}")
                    require(row["budget_skipped_command_ids"] == expected[policy]["budget_skipped_command_ids"], f"selector skip-and-continue trace differs: {sid}/{policy}")
                    for command in ec + mc:
                        require(command["command_uid"] != raw["target_command_uid"] and command["source_file"] == raw["source_file"] and command["source_span"]["start_byte"] < raw["target_source_span"]["start_byte"] and command["source_span"]["end_byte"] <= raw["target_source_span"]["start_byte"], f"selector target/later/different-source command visible: {sid}/{policy}")
                    max_lengths[policy] = max(max_lengths[policy], length)
                require(row["target_coverage"] == coverage(raw["target_type_dependencies"], visible, expected["valid"]), f"actual tokenizer-visible dependency coverage differs: {sid}/{policy}")
                counts["coverage_categories"][policy][row["target_coverage"]["status"]] += 1
            if expected["valid"]:
                require(saved["paired_input_length_difference"] == saved["dependency_prioritized"]["input_length"] - saved["baseline"]["input_length"], f"paired selector length difference differs: {sid}")
            require(saved["tokenization_status"] == "complete_real_tokenizer" and saved["final_visibility_status"] == "actual_whole_command_tokenizer_packing", f"selector tokenization status differs: {sid}")
        require(actual_ids == frozen and next(packed_iter, None) is None, "packed selector frozen membership/order differs, including invalid targets")
        serialized_counts = {**counts, "coverage_categories": {policy: dict(count) for policy, count in counts["coverage_categories"].items()}}
        for field in ("seen", "both_valid", "both_invalid", "coverage_categories"):
            require(serialized_counts[field] == metrics[field], f"selector actual metrics differ: {field}/{budget}")
        self.report["selector"]["budgets"][str(budget)] = {**serialized_counts, "max_input_lengths": max_lengths, "input_limit": budget, "output_allowance": self.answer_limit, "all_2000_frozen_targets_including_invalid_retained": True, "exact_independent_whole_command_EC_cohort_and_MC_reconstruction": True, "same_EC_for_both_policies": True, "no_current_target_proof_fields_copied": True}
        print(json.dumps({"audited_selector_input_budget": budget, "backbone": self.name, **serialized_counts}), flush=True)

    def run(self) -> dict[str, Any]:
        self.load_tokenizer()
        self.validate_training()
        self.validate_selector()
        self.report.update(valid=True, stage="complete", validated_at_utc=datetime.now(timezone.utc).isoformat(), training_metrics_sha256=file_hash(self.training_output / "tokenization_metrics.json"), selector_metrics_sha256={str(budget): file_hash(self.selector_output / f"input_{budget}/tokenization_metrics.json") for budget in self.input_budgets})
        return self.report


def self_test() -> None:
    item = {"source_order": 1, "command_uid": "F::c", "command_id": "c", "text": "def historical := 1"}
    require(training_prompt("theorem anonymous_target : True :=", [item]).endswith(" :=\n"), "training template self-test failed")
    require(selector_prompt("statement", [], [item]) == "def historical := 1\n\nstatement\n", "selector template self-test failed")
    require(coverage({"status": "complete_nonempty", "command_ids": ["a", "b"]}, ["a"], True)["coverage"] == 0.5, "coverage self-test failed")
    require(scope_closures([], "syntactic_valid") == ({}, True) and scope_closures([], "syntactic_unresolved") == ({}, False), "scope self-test failed")
    boundaries = length_summary([0, 1, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 4096, 4097])
    require(boundaries["histogram_counts"] == {"0": 1, "1-256": 2, "257-512": 2, "513-1024": 2, "1025-2048": 2, "2049-4096": 2, ">4096": 1} and boundaries["above_1024_count"] == 5 and boundaries["complete_context_coverage"]["1024"]["all_targets_count"] == 7, "length histogram boundary self-test failed")
    require(training_eligibility(256, 1024, (256, 512, 1024), 1024) == ({"256": True, "512": True, "1024": True}, []), "independent prompt/proof limits must allow combined lengths above 1024")
    require(training_eligibility(257, 20, (256, 512, 1024), 1024) == ({"256": False, "512": True, "1024": True}, ["complete_statement_prompt_exceeds_smallest_input_budget"]), "fair common membership and per-budget feasibility self-test failed")
    require(training_eligibility(1, 1025, (256, 512, 1024), 1024) == ({"256": False, "512": False, "1024": False}, ["complete_reference_proof_plus_one_EOS_exceeds_output_limit"]), "complete-proof output limit self-test failed")
    print(json.dumps({"self_test": "passed", "actual_dataset_outputs_read_or_written": False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "leancontext_dataset")
    parser.add_argument("--backbone", choices=("llada", "dllm", "llama31"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--training-directory", default="training", help="Finished training directory relative to tokenized/BACKBONE (for example training_final before promotion)")
    parser.add_argument("--selector-directory", default="selector", help="Finished selector directory relative to tokenized/BACKBONE (for example selector_final before promotion)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.backbone is not None, "--backbone is required")
    name = "llada" if args.backbone in {"llada", "dllm"} else "llama31"
    tokenizer_dir = args.tokenizer or args.dataset_root / "tokenizers" / name
    validator = TokenizedValidator(args.dataset_root, args.backbone, tokenizer_dir, args.training_directory, args.selector_directory)
    exit_code = 0
    try:
        report = validator.run()
    except (ValidationError, KeyError, TypeError, OSError, ValueError, ImportError) as exc:
        report = validator.report
        report.update(valid=False, error=str(exc), validated_at_utc=datetime.now(timezone.utc).isoformat())
        exit_code = 1
    report_path = validator.output / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "report": str(report_path), "error": report.get("error")}), flush=True)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
