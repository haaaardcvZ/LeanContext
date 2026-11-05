"""CPU synthetic fixtures for independent D.4 reconstruction, not results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .reconstruction import (binding_terms, new_output, original_views,
                             reconstruct_ec, reconstructed_views, view_records)
from .selector import build_pair


class Tokenizer:
    bos_token_id = 100000
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class ProofAndOriginalContextTrap(dict):
    def get(self, key, default=None):
        if key in {"target", "reference_proof", "proof", "gold", "proof_dependencies",
                   "target_type_dependencies", "mathematical_context", "environment_context", "full_context"}:
            raise AssertionError("alternative attempted forbidden input: " + key)
        return super().get(key, default)


class LaterTextTrap(dict):
    def get(self, key, default=None):
        if key in {"text", "effective_raw_text", "raw_text"}:
            raise AssertionError("target/later command body read")
        return super().get(key, default)


def command(cid, kind, text, order):
    return {"command_id": cid, "command_uid": "Mathlib/F.lean::" + cid,
            "source_file": "Mathlib/F.lean", "source_module": "Mathlib.F",
            "kind": kind, "effective_kind": kind, "parser_kind": kind,
            "text": text, "source_order": order,
            "source_span": {"start_byte": order * 100, "end_byte": order * 100 + 99}}


def target(statement="theorem anonymous_target (x : Nat) : useful x = x :="):
    return ProofAndOriginalContextTrap({"sample_id": "id", "target_id": "id", "split": "test",
        "proof_family": "Strict Term", "statement": statement, "source_file": "Mathlib/F.lean",
        "source_module": "Mathlib.F", "target_command_id": "target", "target_command_uid": "Mathlib/F.lean::target",
        "target_source_span": {"start_byte": 2000, "end_byte": 2099}, "original_name": "target",
        "target_kind": "theorem", "target": "FORBIDDEN_TARGET_GOLD"})


class ReconstructionTests(unittest.TestCase):
    def test_reconstruction_ignores_gold_original_context_type_evidence_and_later_text(self):
        commands = [command("x", "variable", "variable (x : Nat)", 1),
                    command("useful", "def", "def useful (x : Nat) : Nat := x", 2),
                    LaterTextTrap(command("target", "theorem", "FORBIDDEN_TARGET_GOLD", 20)),
                    LaterTextTrap(command("later", "def", "LATER_BODY", 21))]
        result = reconstructed_views(Tokenizer(), target(), commands, backbone="dllm")
        self.assertTrue(result["reconstruction_valid"])
        self.assertTrue(result["EC_rebuilt_for_each_MC_trial"])
        self.assertFalse(result["D5_fixed_EC_used"])
        self.assertNotIn("FORBIDDEN_TARGET_GOLD", json.dumps(result))
        self.assertNotIn("LATER_BODY", json.dumps(result))

    def test_ec_depends_on_selected_mc_not_entire_candidate_pool(self):
        commands = [command("A", "namespace", "namespace A", 1),
                    command("xa", "variable", "variable (a : Nat)", 2),
                    command("fa", "def", "def fa (a : Nat) : Nat := a", 3),
                    command("endA", "end", "end A", 4),
                    command("B", "namespace", "namespace B", 5),
                    command("xb", "variable", "variable (b : Nat)", 6),
                    command("fb", "def", "def fb (b : Nat) : Nat := b", 7),
                    command("endB", "end", "end B", 8)]
        t = target("theorem anonymous_target : True :=")
        pair = build_pair(t, commands, None)
        pool = {c["command_id"]: c for c in pair["candidate_pool"]}
        initial = reconstruct_ec(t["statement"], pair, [])
        with_a = reconstruct_ec(t["statement"], pair, [pool["fa"]])
        with_b = reconstruct_ec(t["statement"], pair, [pool["fb"]])
        self.assertEqual(initial, [])
        self.assertEqual({c["command_id"] for c in with_a}, {"A", "xa", "endA"})
        self.assertEqual({c["command_id"] for c in with_b}, {"B", "xb", "endB"})

    def test_mc_body_does_not_affect_ranking_or_binding_relevance(self):
        commands = [command("x", "variable", "variable (x : Nat)", 1),
                    command("a", "def", "def alpha : Nat := 0", 2),
                    command("u", "def", "def useful (x : Nat) : Nat := x", 3)]
        first = build_pair(target(), commands, None)
        commands[1]["text"] = "def alpha : Nat := useful x x x x x"
        second = build_pair(target(), commands, None)
        self.assertEqual(first["baseline_ranking"], second["baseline_ranking"])
        chosen = [c for c in first["candidate_pool"] if c["command_id"] == "a"]
        ec = reconstruct_ec("theorem anonymous_target : True :=", first, chosen)
        self.assertEqual(ec, [])

    def test_binding_names_and_implicit_instance_type_symbols(self):
        item = {"kind": "variable", "text": "variable (x y : Nat) [inst : Group G] {z : Type u}"}
        terms = binding_terms(item)
        self.assertTrue({"x", "y", "inst", "group", "g", "z"} <= terms)
        self.assertNotIn("nat", terms)

    def test_original_uses_shared_full_prefix_without_view_refill(self):
        mc = [command("m", "def", "m" * 700, 1)]
        ec = [command("e", "variable", "e" * 700, 2)]
        result = original_views(Tokenizer(), target()["statement"], mc, ec, backbone="dllm")
        self.assertEqual(result["removed_command_ids"], ["e"])
        self.assertEqual(result["views"]["CFull"]["prompt"], result["views"]["CMath"]["prompt"])
        self.assertEqual(result["views"]["CEnv"], result["views"]["C0"] | {"view": "CEnv"})
        self.assertTrue(all(v["valid_input"] for v in result["views"].values()))

    def test_invalid_context_keeps_same_valid_c0_and_failed_other_views(self):
        # A named scope closer mismatch makes the conservative reconstruction
        # invalid; it does not change the bare theorem or remove this target.
        commands = [command("A", "namespace", "namespace A", 1),
                    command("endB", "end", "end B", 2)]
        t = target()
        result = reconstructed_views(Tokenizer(), t, commands, backbone="dllm")
        original = original_views(Tokenizer(), t["statement"], [], [], backbone="dllm")
        self.assertFalse(result["reconstruction_valid"])
        self.assertEqual(result["views"]["C0"], original["views"]["C0"])
        self.assertTrue(result["views"]["C0"]["valid_input"])
        self.assertTrue(all(not result["views"][v]["valid_input"] for v in ("CMath", "CEnv", "CFull")))

    def test_overlong_bare_target_is_not_truncated(self):
        t = target("x" * 2000)
        result = reconstructed_views(Tokenizer(), t, [], backbone="dllm")
        self.assertTrue(all(not v["valid_input"] for v in result["views"].values()))
        self.assertEqual(result["views"]["C0"]["prompt"], "x" * 2000 + "\n")

    def test_output_guards_and_stdlib_help(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            root.mkdir()
            with self.assertRaises(ValueError):
                new_output(root, root / "extension", create=False)
            out = Path(temp) / "new"
            new_output(root, out, create=False)
            self.assertFalse(out.exists())
        code_root = Path(__file__).resolve().parents[1]
        process = subprocess.run([sys.executable, "-m", "dataset_preparation.reconstruction", "--help"],
                                 cwd=code_root, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--plan", process.stdout)


if __name__ == "__main__":
    unittest.main()
