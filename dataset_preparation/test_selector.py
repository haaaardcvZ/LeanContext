"""Small proof-blind selector fixtures, independent of local dataset files."""

import copy
import json
import unittest

from dataset_preparation.selector import build_pair, choose_manifest, identifier_terms


class ProofTrap(dict):
    def get(self, key, default=None):
        if key in {"target", "reference_proof", "proof", "gold", "proof_dependencies", "results"}:
            raise AssertionError(f"forbidden gold/proof access: {key}")
        return super().get(key, default)


class CommandTextTrap(dict):
    def get(self, key, default=None):
        if key in {"text", "effective_raw_text", "raw_text"}:
            raise AssertionError(f"forbidden current-target/later source body access: {key}")
        return super().get(key, default)


def command(cid, kind, text, start, *, file="Mathlib/Test.lean", **extra):
    return {"command_id": cid, "kind": kind, "parser_kind": kind,
            "text": text, "source_file": file,
            "source_span": {"start_byte": start, "end_byte": start + 5}, **extra}


def target(**extra):
    return ProofTrap({"sample_id": "target-id", "split": "test", "proof_family": "Strict Term",
        "statement": "theorem anonymous_target : useful x = x :=", "source_file": "Mathlib/Test.lean",
        "target_command_id": "cmd_9", "target_source_span": {"start_byte": 90, "end_byte": 99},
        "reference_proof": "CURRENT_TARGET_SECRET", **extra})


def complete(ids):
    return {"status": "complete_nonempty", "command_ids": ids, "constant_evidence": {}}


class SelectorTests(unittest.TestCase):
    def test_nonbinding_EC_uses_distance_not_extra_structure_tier(self):
        cmds = [command("ns", "namespace", "namespace N", 10),
                command("binding", "variable", "variable (x : Nat)", 20),
                command("near_open", "open", "open Classical", 30),
                command("resource", "def", "def useful (x : Nat) := x", 40)]
        pair = build_pair(target(), cmds, None)
        priority = pair["shared_environment_priority"]
        self.assertEqual(priority[0], "binding")
        self.assertLess(priority.index("near_open"), priority.index("ns"))

    def test_no_gold_access_and_signature_only_ranking(self):
        cmds = [command("cmd_1", "def", "def alpha : Nat := useful x x x", 10),
                command("cmd_2", "def", "def useful (x : Nat) : Nat := bodySecret", 20)]
        row = build_pair(target(), cmds, complete(["cmd_1"]))
        self.assertEqual(row["baseline_ranking"][0], "cmd_2")
        self.assertEqual(row["dependency_prioritized_ranking"][0], "cmd_1")
        self.assertEqual({c["command_id"] for c in row["candidate_pool"]}, {"cmd_1", "cmd_2"})
        self.assertNotIn("CURRENT_TARGET_SECRET", json.dumps(row))
        for candidate in row["candidate_pool"]:
            self.assertNotIn("bodySecret", candidate["signature"])
            self.assertNotIn("bodysecret", candidate["lexical_terms"])
        self.assertEqual(row["tokenization_status"], "pending")
        self.assertNotIn("prompt", row)
        self.assertNotIn("input_token_count", row)

    def test_target_later_and_other_file_excluded(self):
        row = build_pair(target(), [command("cmd_1", "def", "def a := 0", 10),
            command("cmd_9", "theorem", "theorem target : True := CURRENT_TARGET_SECRET", 90),
            command("cmd_10", "def", "def later := 0", 100),
            command("other", "def", "def other := 0", 20, file="Mathlib/Other.lean")], None)
        self.assertEqual(row["baseline_ranking"], ["cmd_1"])
        exclusions = {r["command_id"]: r["reasons"] for r in row["candidate_exclusions"]}
        self.assertIn("target_command", exclusions["cmd_9"])
        self.assertIn("not_strictly_preceding", exclusions["cmd_10"])
        self.assertIn("different_source_file", exclusions["other"])
        self.assertNotIn("CURRENT_TARGET_SECRET", json.dumps(row))

    def test_current_target_and_later_source_text_never_read(self):
        cmds = [command("cmd_1", "def", "def a := 0", 10),
                CommandTextTrap(command("cmd_9", "theorem", "GOLD", 90)),
                CommandTextTrap(command("cmd_10", "def", "LATER_GOLD", 100))]
        row = build_pair(target(), cmds, None)
        self.assertEqual(row["baseline_ranking"], ["cmd_1"])

    def test_direct_theorem_filter_shared_by_both_policies(self):
        row = build_pair(target(), [command("cmd_1", "lemma", "lemma hidden : True := by trivial", 10),
            command("cmd_2", "theorem", "theorem useful : True := by trivial", 20),
            command("cmd_3", "open", "open Classical in theorem wrapped : True := by trivial", 30)], complete(["cmd_1"]))
        self.assertEqual(set(row["baseline_ranking"]), {"cmd_2", "cmd_3"})
        self.assertEqual(set(row["dependency_prioritized_ranking"]), {"cmd_2", "cmd_3"})
        self.assertEqual(row["dependency_pool_coverage"]["missing_from_pool_command_ids"], ["cmd_1"])

    def test_signature_excludes_wrapper_and_attributes(self):
        row = build_pair(target(statement="theorem anonymous_target : Foo.useful x = x :="),
            [command("cmd_1", "theorem", "@[simp] theorem useful : True := by trivial", 10),
             command("cmd_2", "open", "open Secret in def bar : Nat := BODY", 20)], None)
        self.assertIn("cmd_1", row["baseline_ranking"])
        candidates = {c["command_id"]: c for c in row["candidate_pool"]}
        self.assertNotIn("simp", candidates["cmd_1"]["lexical_terms"])
        self.assertNotIn("secret", candidates["cmd_2"]["lexical_terms"])
        self.assertEqual(candidates["cmd_2"]["signature"], "def bar : Nat")

    def test_fallback_and_independent_denominator(self):
        cmds = [command("cmd_1", "def", "def useful := 0", 10), command("cmd_2", "def", "def z := 0", 20)]
        for evidence in (None, {"status": "no_eligible_local_dependencies", "command_ids": []},
                         {"status": "incomplete_mapping", "command_ids": ["cmd_2"]},
                         {"command_ids": ["cmd_2"]}):
            row = build_pair(target(), cmds, evidence)
            self.assertTrue(row["dependency_priority_fallback"])
            self.assertEqual(row["baseline_ranking"], row["dependency_prioritized_ranking"])
        row = build_pair(target(), cmds, complete(["cmd_1", "missing"]))
        self.assertEqual(row["dependency_pool_coverage"]["independent_dependency_count"], 2)
        self.assertEqual(row["dependency_pool_coverage"]["status"], "partial_dependencies_in_pool")

    def test_ties_distance_then_command_id(self):
        cmds = [command("cmd_b", "def", "def bravo : Nat := 0", 10, source_order=2),
                command("cmd_a", "def", "def alpha : Nat := 0", 20, source_order=2),
                command("cmd_c", "def", "def charlie : Nat := 0", 30, source_order=3)]
        row = build_pair(target(statement="theorem anonymous_target : True :=", source_order=9), cmds, None)
        self.assertEqual(row["baseline_ranking"], ["cmd_c", "cmd_a", "cmd_b"])

    def test_local_instance_mc_variable_ec_and_wrapper_not_leaked(self):
        cmds = [command("cmd_1", "variable", "variable [inst : Inhabited Nat]", 10),
                command("cmd_2", "local", "local instance named : Inhabited Nat := ⟨0⟩", 20),
                command("cmd_3", "open", "open Nat in #check succ", 30)]
        row = build_pair(target(), cmds, None)
        self.assertEqual(row["candidate_pool"][0]["kind"], "local_instance")
        self.assertEqual(row["shared_environment_command_ids"], ["cmd_1"])
        self.assertTrue(row["EC_same_for_both_policies"])
        self.assertIn("cmd_3", row["candidate_exclusion_ids"])

    def test_scope_closers_preserved_and_closed_bindings_not_target_priority(self):
        cmds = [command("cmd_1", "namespace", "namespace A", 10),
                command("cmd_2", "variable", "variable (x : Nat)", 20),
                command("cmd_3", "def", "def foo := x", 30),
                command("cmd_4", "end", "end A", 40),
                command("cmd_5", "variable", "variable (y : Nat)", 50)]
        row = build_pair(target(), cmds, None)
        env = {r["command_id"]: r for r in row["shared_environment"]}
        self.assertEqual(env["cmd_1"]["scope_pair_command_ids"], ["cmd_4"])
        self.assertEqual(env["cmd_4"]["scope_role"], "block_closer")
        self.assertFalse(env["cmd_1"]["target_scope_active"])
        self.assertFalse(env["cmd_2"]["target_scope_active"])
        self.assertEqual(row["shared_environment_priority"][0], "cmd_5")
        self.assertEqual(row["candidate_pool"][0]["syntactic_name_candidates"], ["A.foo"])

    def test_noncomputable_named_section_matches_its_end(self):
        cmds = [command("cmd_1", "namespace", "namespace Foo", 10, name="Foo"),
                command("cmd_2", "section", "noncomputable section ProdLemmas", 20, name="ProdLemmas"),
                command("cmd_3", "variable", "variable (x : Nat)", 30),
                command("cmd_4", "def", "def bar := x", 40),
                command("cmd_5", "end", "end ProdLemmas", 50),
                command("cmd_6", "section", "noncomputable section AnonymousMetadataMissing", 60),
                command("cmd_7", "end", "end AnonymousMetadataMissing", 70)]
        row = build_pair(target(), cmds, None)
        self.assertEqual(row["EC_scope_stack_status"], "syntactic_valid")
        env = {r["command_id"]: r for r in row["shared_environment"]}
        self.assertEqual(env["cmd_2"]["scope_name"], "ProdLemmas")
        self.assertEqual(env["cmd_2"]["scope_pair_command_ids"], ["cmd_5"])
        self.assertEqual(env["cmd_6"]["scope_pair_command_ids"], ["cmd_7"])
        self.assertFalse(env["cmd_3"]["target_scope_active"])
        self.assertEqual(row["candidate_pool"][0]["syntactic_name_candidates"], ["Foo.bar"])

    def test_manifest_stratified_deterministic_fixed_subset(self):
        rows = [target(sample_id=f"a{i}", proof_family="A", both_nonempty=True) for i in range(7)]
        rows += [target(sample_id=f"b{i}", proof_family="B", both_nonempty=True) for i in range(3)]
        rows += [target(sample_id="not-test", split="train", both_nonempty=True),
                 target(sample_id="empty", both_nonempty=False)]
        before = copy.deepcopy(rows)
        manifest = choose_manifest(rows, n=5)
        self.assertEqual(manifest["family_quotas"], {"A": 4, "B": 1})
        self.assertEqual(manifest["eligible_population"], 10)
        self.assertEqual(manifest, choose_manifest(list(reversed(rows)), n=5))
        self.assertEqual(rows, before)
        self.assertNotIn("CURRENT_TARGET_SECRET", json.dumps(manifest))
        self.assertEqual(manifest["manifest_origin"], "deterministic_stratified_fixed_subset")
        with self.assertRaises(ValueError): choose_manifest(rows, n=11)

    def test_validation_manifest_preserves_original_split(self):
        rows = [target(sample_id=f"v{i}", split="validation", both_nonempty=True) for i in range(4)]
        rows.append(target(sample_id="test", split="test", both_nonempty=True))
        manifest = choose_manifest(rows, n=2, split="validation")
        self.assertEqual(manifest["population"], "original_both_nonempty_validation_only")
        self.assertTrue(all(row["split"] == "validation" for row in manifest["targets"]))
        self.assertNotIn("test", manifest["target_ids"])

    def test_command_uid_alias_and_source_names_not_generated_kernel_terms(self):
        cmds = [command("cmd_1", "namespace", "namespace Foo", 10),
                command("cmd_2", "def", "def x : Nat := 0", 20, command_uid="stable-uid",
                    introduced_declarations=[{"name": "Foo.x", "basis": "kernel_selection_range"},
                        {"name": "Foo.x.generatedAux", "basis": "kernel_selection_range"}]),
                command("cmd_3", "instance", "instance : Inhabited Nat := ⟨0⟩", 30,
                    introduced_declarations=[{"name": "Foo.instInhabited", "basis": "kernel_selection_range"}])]
        row = build_pair(target(target_command_uid="target-stable-uid", original_name="some_target"), cmds, None)
        candidates = {c["command_id"]: c for c in row["candidate_pool"]}
        self.assertEqual(candidates["cmd_2"]["command_uid"], "stable-uid")
        self.assertEqual(candidates["cmd_2"]["uid"], "stable-uid")
        self.assertEqual(candidates["cmd_2"]["names"], ["x"])
        self.assertNotIn("generated", candidates["cmd_2"]["lexical_terms"])
        self.assertEqual(candidates["cmd_3"]["syntactic_name_candidates"], ["Foo.instInhabited"])
        self.assertEqual(row["target_command_uid"], "target-stable-uid")

    def test_identifier_terms_split_names_camel_case_and_comments(self):
        terms = identifier_terms('theorem anonymous_target : Foo.Bar_myHTTPServer /- SECRET -/ := "GOLD"')
        self.assertEqual(terms, {"foo", "bar", "my", "http", "server"})


if __name__ == "__main__":
    unittest.main()
