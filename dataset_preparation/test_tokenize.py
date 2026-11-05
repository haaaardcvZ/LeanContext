"""Offline tests of exact whole-command packing, with a test-only tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from dataset_preparation import tokenize as prep


class CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def __call__(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(char) + 1000 for char in text]}


def command(cid, text, order, role="MC"):
    return {"command_id": cid, "command_uid": f"Mathlib/Test.lean:{cid}", "text": text, "source_order": order, "position": order, "role": role, "kind": "def" if role == "MC" else "variable", "source_file": "Mathlib/Test.lean"}


class TrainingPackingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.math = command("m", "def M := 1", 0)
        self.env = command("e", "variable (x : Nat)", 1, "EC")
        self.contexts = {"C0": [], "CMath": [self.math], "CEnv": [self.env], "CFull": [self.math, self.env]}

    def test_full_proof_one_eos_and_llama_bos(self):
        proof = "by\n  exact rfl  "
        result = prep.prepare_training_views(self.tokenizer, "example : 1 = 1 :=", proof, self.contexts, backbone="llama31")
        self.assertTrue(result["retained"])
        for row in result["views"].values():
            self.assertEqual(row["reference_proof"], proof)
            self.assertEqual(row["target"], proof)
            self.assertEqual(row["input_ids"][0], 1)
            answer = row["input_ids"][row["prompt_length"]:]
            self.assertEqual(answer[-1], 2)
            self.assertEqual(answer.count(2), 1)
            self.assertEqual(row["answer_length"], len(proof) + 1)
            self.assertEqual(row["answer_span"]["end_exclusive"], len(row["input_ids"]))
            self.assertTrue(row["prompt"].endswith("\n"))
            self.assertFalse(row["prompt"].endswith("\n\n"))

    def test_shared_full_end_trim_no_refill(self):
        statement, proof = "S :=", "by rfl"
        limit = len(prep.render_training_prompt(statement, [self.math])) + len(proof) + 1
        result = prep.prepare_training_views(self.tokenizer, statement, proof, self.contexts, backbone="dllm", max_training_length=limit)
        self.assertTrue(result["retained"])
        self.assertEqual(result["removed_command_ids"], ["e"])
        self.assertEqual(result["views"]["CFull"]["context_ids"], ["m"])
        self.assertEqual(result["views"]["CMath"]["context_ids"], ["m"])
        self.assertEqual(result["views"]["CEnv"]["context_ids"], [])
        for row in result["views"].values():
            self.assertEqual(row["context_ids"], [prep.command_id(item) for item in row["context_items"]])
            self.assertEqual(row["context_uids"], [prep.command_key(item) for item in row["context_items"]])
        self.assertEqual(result["views"]["CEnv"]["prompt"], result["views"]["C0"]["prompt"])
        self.assertTrue(all(row["sequence_length"] <= limit for row in result["views"].values()))

    def test_joint_reject_not_proof_truncation(self):
        result = prep.prepare_training_views(self.tokenizer, "S :=", "by " + "x" * 100, self.contexts, backbone="dllm", max_training_length=30)
        self.assertFalse(result["retained"])
        self.assertNotIn("views", result)
        self.assertEqual(result["reason"], "bare_statement_and_complete_proof_exceed_training_limit")

    def test_leading_newline_is_error_not_normalized(self):
        with self.assertRaisesRegex(ValueError, "leading newline"):
            prep.prepare_training_views(self.tokenizer, "S :=", "\nby rfl", self.contexts, backbone="dllm")

    def test_math_env_must_partition_full(self):
        with self.assertRaisesRegex(ValueError, "union"):
            prep.prepare_training_views(self.tokenizer, "S :=", "by rfl", {**self.contexts, "CEnv": []}, backbone="dllm")


class SelectorPackingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.statement = "example : True :="
        self.small = command("a", "def A := 1", 4)
        self.small2 = command("b", "def B := 2", 2)
        self.large = command("z", "def Z := " + "x" * 500, 1)
        self.ec = command("e", "variable (x : Nat)", 0, "EC")

    def test_skip_continue_and_source_order(self):
        limit = len(prep.render_selector_prompt(self.statement, [], [self.small, self.small2]))
        result = prep.pack_selector_pair(self.tokenizer, self.statement, [], [self.large, self.small, self.small2], [self.small2, self.large, self.small], backbone="dllm", input_limit=limit)
        for policy in ("baseline", "dependency_prioritized"):
            row = result[policy]
            self.assertEqual(row["selected_command_ids"], ["b", "a"])
            self.assertEqual(row["budget_skipped_command_ids"], ["z"])
            self.assertEqual(row["input_length"], limit)
            self.assertEqual(row["prompt"].count(self.small["text"]), 1)
            self.assertLess(row["prompt"].index(self.small2["text"]), row["prompt"].index(self.small["text"]))

    def test_ec_is_frozen_before_both_mc_rankings(self):
        too_large_ec = command("q", "variable " + "x" * 1000, -1, "EC")
        limit = len(prep.render_selector_prompt(self.statement, [self.ec], [self.small]))
        result = prep.pack_selector_pair(self.tokenizer, self.statement, [too_large_ec, self.ec], [self.small, self.small2], [self.small2, self.small], backbone="dllm", input_limit=limit)
        self.assertEqual(result["shared_environment_ids"], ["e"])
        self.assertEqual(result["environment_budget_skipped_command_ids"], ["q"])
        for policy in ("baseline", "dependency_prioritized"):
            self.assertIn(self.ec["text"], result[policy]["prompt"])
            self.assertNotIn("# EC IDs:", result[policy]["prompt"])
            self.assertNotIn("# Target:", result[policy]["prompt"])
            self.assertEqual(result[policy]["prompt"].count(self.ec["text"]), 1)

    def test_selector_uses_training_serializer_with_metadata_outside_prompt(self):
        prompt = prep.render_selector_prompt(self.statement, [self.ec], [self.small])
        self.assertEqual(prompt, prep.render_training_prompt(self.statement, [self.ec, self.small]))
        self.assertNotIn("# MC", prompt)

    def test_independent_dependency_denominator(self):
        result = prep.pack_selector_pair(self.tokenizer, self.statement, [], [self.small], [self.small], backbone="dllm", target_type_dependencies={"status": "complete_nonempty", "command_ids": ["a", "missing_from_pool"]})
        coverage = result["baseline"]["target_coverage"]
        self.assertEqual(coverage["dependency_count"], 2)
        self.assertEqual(coverage["coverage"], 0.5)
        self.assertEqual(coverage["status"], "partial")

    def test_no_dependencies_or_uncertain_not_one_or_zero(self):
        for status in ("no_eligible_local_dependencies", "incomplete_mapping", "unresolved_target"):
            result = prep.dependency_coverage({"status": status, "command_ids": []}, ["a"], valid_input=True)
            self.assertIsNone(result["coverage"])

    def test_invalid_common_input_retains_both_records(self):
        result = prep.pack_selector_pair(self.tokenizer, self.statement, [], [self.small], [self.small], backbone="llama31", input_limit=10, target_type_dependencies={"status": "complete_nonempty", "command_ids": ["a"]})
        for policy in ("baseline", "dependency_prioritized"):
            self.assertFalse(result[policy]["valid_input"])
            self.assertIsNone(result[policy]["input_ids"])
            self.assertIsNone(result[policy]["target_coverage"]["coverage"])

    def test_selector_bos_is_counted(self):
        dllm = prep.pack_selector_pair(self.tokenizer, self.statement, [], [], [], backbone="dllm")
        llama = prep.pack_selector_pair(self.tokenizer, self.statement, [], [], [], backbone="llama31")
        self.assertEqual(llama["baseline"]["input_length"], dllm["baseline"]["input_length"] + 1)
        self.assertEqual(llama["baseline"]["input_ids"][0], 1)

    def test_closed_scope_ec_fits_or_omits_complete_cohort(self):
        opener = {**command("o", "section", 0, "EC"), "scope_role": "block_opener", "target_scope_active": False, "scope_pair_command_ids": ["c"], "required_scope_command_ids": ["c"]}
        effect = {**command("v", "variable (x : Nat)", 1, "EC"), "scope_role": "scope_effect", "required_scope_command_ids": ["o"]}
        closer = {**command("c", "end", 2, "EC"), "scope_role": "block_closer", "scope_pair_command_ids": ["o"], "required_scope_command_ids": ["o"]}
        tail = command("u", "universe u", 3, "EC")
        environment = [effect, opener, closer, tail]
        # Only the independent tail fits; no part of the larger closed-scope
        # cohort may consume the space needed for that later complete command.
        limit = len(prep.render_selector_prompt(self.statement, [tail], []))
        result = prep.pack_selector_pair(self.tokenizer, self.statement, environment, [], [], backbone="dllm", input_limit=limit, environment_scope_status="syntactic_valid")
        self.assertNotIn("v", result["shared_environment_ids"])
        self.assertIn("u", result["shared_environment_ids"])
        self.assertEqual("o" in result["shared_environment_ids"], "c" in result["shared_environment_ids"])
        self.assertEqual(result["baseline"]["prompt"], result["dependency_prioritized"]["prompt"])
        full_limit = len(prep.render_selector_prompt(self.statement, [opener, effect, closer, tail], []))
        full = prep.pack_selector_pair(self.tokenizer, self.statement, environment, [], [], backbone="dllm", input_limit=full_limit)
        self.assertEqual(full["shared_environment_ids"], ["o", "v", "c", "u"])

    def test_missing_scope_reference_invalidates_both_without_silent_drop(self):
        effect = {**self.ec, "required_scope_command_ids": ["missing_opener"]}
        result = prep.pack_selector_pair(self.tokenizer, self.statement, [effect], [self.small], [self.small], backbone="dllm")
        for policy in ("baseline", "dependency_prioritized"):
            self.assertFalse(result[policy]["valid_input"])
            self.assertEqual(result[policy]["reason"], "shared_environment_scope_records_missing_or_invalid")
            self.assertIn("missing_opener", result[policy]["scope_error"])

    def test_unresolved_scope_status_retains_invalid_pair(self):
        result = prep.pack_selector_pair(self.tokenizer, self.statement, [self.ec], [], [], backbone="dllm", environment_scope_status="syntactic_unresolved")
        self.assertFalse(result["baseline"]["valid_input"])
        self.assertFalse(result["dependency_prioritized"]["valid_input"])


class FileWorkflowTests(unittest.TestCase):
    def test_training_output_views_cannot_overwrite_raw_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "training" / "views" / "C0" / "train.jsonl"
            original.parent.mkdir(parents=True)
            original.write_text("ORIGINAL_RAW_BYTES\n")
            before = original.read_bytes()
            args = argparse.Namespace(dataset_root=root, output_dir=root / "training", backbone="dllm", max_training_length=4096)
            with self.assertRaisesRegex(ValueError, "raw training views"):
                prep.run_training(args, CharacterTokenizer(), {"test_only": True})
            self.assertEqual(original.read_bytes(), before)
            self.assertFalse((root / "training" / "membership").exists())

    def test_training_memberships_joint_and_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            math = command("a", "def A := 1", 0)
            contexts = {"C0": [], "CMath": [math], "CEnv": [], "CFull": [math]}
            for split in prep.SPLITS:
                for view in prep.VIEWS:
                    path = root / "training" / "views" / view / f"{split}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rows = [
                        {"sample_id": "ok", "target_id": "T1", "split": split, "view": view, "statement": "S :=", "target": "by rfl", "context_items": contexts[view], "context_ids": [prep.command_id(item) for item in contexts[view]], "context_uids": [prep.command_key(item) for item in contexts[view]]},
                        {"sample_id": "bad", "target_id": "T2", "split": split, "view": view, "statement": "S :=", "target": "by " + "x" * 100, "context_items": contexts[view]},
                    ]
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = argparse.Namespace(dataset_root=root, output_dir=root / "encoded", backbone="dllm", max_training_length=12)
            result = prep.run_training(args, CharacterTokenizer(), {"test_only": True})
            for split in prep.SPLITS:
                self.assertEqual(result["splits"][split]["retained"], 1)
                self.assertEqual(result["splits"][split]["rejected"], 1)
                for view in prep.VIEWS:
                    rows = list(prep.iter_jsonl(root / "encoded" / "views" / view / f"{split}.jsonl"))
                    self.assertEqual([row["sample_id"] for row in rows], ["ok"])
                    self.assertEqual(rows[0]["reference_proof"], "by rfl")
                    self.assertEqual(rows[0]["context_ids"], [])
                    self.assertEqual(rows[0]["context_uids"], [])
                    self.assertEqual(rows[0]["raw_context_ids"], [prep.command_id(item) for item in contexts[view]])
                    self.assertEqual(rows[0]["raw_context_uids"], [prep.command_key(item) for item in contexts[view]])
                self.assertEqual(len(list(prep.iter_jsonl(root / "encoded" / "membership" / f"{split}.jsonl"))), 2)

    def test_catalog_reads_per_source_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "command_mapping" / "source_commands" / "Mathlib" / "Test.jsonl"
            path.parent.mkdir(parents=True)
            item = command("a", "def A := 1", 0)
            path.write_text(json.dumps(item) + "\n")
            catalog = prep.CommandCatalog(root)
            index = catalog.for_source("Mathlib/Test.lean")
            self.assertEqual(index["a"], item)
            self.assertEqual(index[item["command_uid"]], item)

    def test_selector_file_workflow_honors_ec_priority_and_excludes_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mc = command("a", "def A := 1", 1)
            ec = command("e", "variable (x : Nat)", 0, "EC")
            catalog = root / "command_mapping" / "source_commands" / "Mathlib" / "Test.jsonl"
            catalog.parent.mkdir(parents=True)
            catalog.write_text(json.dumps(mc) + "\n")
            pair_path = root / "selector" / "pairs.jsonl"
            pair_path.parent.mkdir()
            pair = {"sample_id": "s", "target_id": "t", "split": "test", "source_file": "Mathlib/Test.lean", "statement": "example : True :=", "reference_proof": "SECRET_CURRENT_REFERENCE_PROOF", "candidate_pool": [{"command_id": "a"}], "baseline_ranking": ["a"], "dependency_prioritized_ranking": ["a"], "shared_environment": [ec], "shared_environment_priority": ["e"], "EC_scope_stack_status": "syntactic_valid", "target_type_dependencies": {"status": "complete_nonempty", "command_ids": ["a", "missing"]}}
            pair_path.write_text(json.dumps(pair) + "\n")
            args = argparse.Namespace(dataset_root=root, output_dir=root / "packed", backbone="dllm", input_limit=1024, output_limit=1024)
            metrics = prep.run_selector(args, CharacterTokenizer(), {"test_only": True})
            self.assertEqual(metrics["seen"], 1)
            output_text = (root / "packed" / "pairs.jsonl").read_text()
            self.assertNotIn("SECRET_CURRENT_REFERENCE_PROOF", output_text)
            row = json.loads(output_text)
            self.assertEqual(row["shared_environment_priority"], ["e"])
            self.assertEqual(row["baseline"]["target_coverage"]["coverage"], 0.5)
            self.assertEqual(row["baseline"]["all_visible_command_ids"], ["e", "a"])

    def test_remote_revision_required_without_import_or_download(self):
        args = argparse.Namespace(tokenizer="GSAI-ML/LLaDA-8B-Base", tokenizer_revision="main", trust_remote_code=False, backbone="dllm")
        with self.assertRaisesRegex(ValueError, "40-hex"):
            prep.load_actual_tokenizer(args)


if __name__ == "__main__":
    unittest.main()
