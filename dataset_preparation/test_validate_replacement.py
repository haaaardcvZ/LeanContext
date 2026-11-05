from copy import deepcopy
import itertools
import random
import unittest

from .validate_replacement import (
    attributed_command, command_kind, encode, minimum_cost, no_proof_fields, object_hash, observed_sets,
    reconstruct_prompt, validate_model_pair, validate_sampling, within_band,
)


class FakeTokenizer:
    bos_token_id = 10000
    eos_token_id = 10001

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def command(cid, order, text, kind="def"):
    return {"command_id": cid, "command_uid": "Mathlib/F.lean::" + cid,
            "source_file": "Mathlib/F.lean", "source_order": order,
            "source_span": {"start_byte": 20 * order, "end_byte": 20 * order + 19},
            "text": text, "kind": kind, "effective_kind": kind,
            "introduced_declarations_status": "kernel_range_or_certified_ownership_attributed",
            "introduced_declarations": [{"name": "N." + cid, "basis": "kernel_selection_range"}]}


def fixture():
    ec = command("e", 0, "variable (n : Nat)", "variable")
    alternate = command("r", 1, "def otherxxx : Nat := 0")
    selected = command("s", 2, "def original : Nat := 1")
    target = {"sample_id": "id", "source_file": "Mathlib/F.lean", "statement": "theorem target : True :=",
              "target_command_id": "t", "target_source_span": {"start_byte": 200},
              "mathematical_context": [selected], "environment_context": [ec]}
    payload = {"profile_version": "v", "namespace_path": [], "scope_stack": [],
               "active_directives": [], "command_wrappers": [],
               "command_attributes": [], "command_modifiers": []}
    fingerprint = object_hash(payload)
    profiles = {c["command_id"]: {**payload, "status": "syntactic_valid", "fingerprint": fingerprint}
                for c in (ec, alternate, selected)}
    control = {**alternate, "role": "MC", "prompt_slot_command_id": "s",
               "prompt_slot_command_uid": selected["command_uid"], "prompt_slot_source_order": 2}
    pair = {"selected_command_id": "s", "selected_command_uid": selected["command_uid"],
            "replacement_command_id": "r", "replacement_command_uid": alternate["command_uid"],
            "selected_kind": "def", "replacement_kind": "def", "selected_tokens": len(selected["text"]),
            "replacement_tokens": len(alternate["text"]), "length_ratio": len(alternate["text"]) / len(selected["text"]),
            "scope_fingerprint": fingerprint, "replacement_source_order": 1, "prompt_slot_source_order": 2}
    result = {"eligible": True, "reason": None, "declaration_pairs": [pair], "replacement_MC": [control],
              "assignment_cost": 0, "selected_MC_tokens": len(selected["text"]),
              "control_MC_tokens": len(alternate["text"])}
    raw = {"statement": target["statement"], "selected_MC": [selected], "shared_EC": [ec], "models": {"llada": result}}
    conditions = {}
    tokenizer = FakeTokenizer()
    for name, mc in (("Env", []), ("Selected", [selected]), ("Control", [control])):
        prompt = reconstruct_prompt(target["statement"], [ec], mc)
        ids = encode(tokenizer, prompt, "llada")
        conditions[name] = {"condition": name, "prompt": prompt, "input_ids": ids,
                            "input_length": len(ids), "prompt_length": len(ids), "input_limit": 1024,
                            "output_limit": 1024, "output_limit_includes_terminal_EOS": True,
                            "valid_input": True, "reference_proof_in_generator_input": False,
                            "input_ids_contain_prompt_only": True,
                            "visible_mc_command_ids": [c["command_id"] for c in mc],
                            "visible_mc_command_uids": [c["command_uid"] for c in mc],
                            "visible_ec_command_ids": ["e"], "visible_ec_command_uids": [ec["command_uid"]],
                            "MC_count": len(mc), "EC_count": 1}
    encoded = {"statement": target["statement"], "declaration_pairs": [pair],
               "conditions": conditions, "shared_EC_ids": ["e"]}
    return target, raw, encoded, tokenizer, [ec, alternate, selected], profiles


class ReplacementValidatorTests(unittest.TestCase):
    def test_min_cost_matches_exhaustive_assignments(self):
        columns = ["a", "b", "c", "d"]
        for seed in range(25):
            rng = random.Random(seed)
            edges = [{c: rng.randrange(10) for c in columns if rng.random() < .6} for _ in range(3)]
            actual = [sum(edges[i][c] for i, c in enumerate(choice)) for choice in itertools.permutations(columns, 3)
                      if all(c in edges[i] for i, c in enumerate(choice))]
            self.assertEqual(minimum_cost(edges), min(actual) if actual else None)

    def test_flow_can_undo_greedy_assignment_and_reject_Hall_failure(self):
        self.assertEqual(minimum_cost([{"a": 0, "b": 1}, {"a": 0}]), 1)
        self.assertIsNone(minimum_cost([{"a": 0}, {"a": 1}, {"b": 2, "c": 3}]))
        self.assertIsNone(minimum_cost([{}]))

    def test_inclusive_band_integer_boundaries(self):
        self.assertTrue(within_band(100, 90))
        self.assertTrue(within_band(100, 110))
        self.assertFalse(within_band(100, 89))
        self.assertFalse(within_band(100, 111))
        self.assertFalse(within_band(0, 0))

    def test_local_instance_whitespace_does_not_merge_effective_kinds(self):
        self.assertEqual(command_kind({"effective_kind": "instance", "text": "local\n  instance : X := x"}), "local_instance")
        self.assertEqual(command_kind({"effective_kind": "instance", "text": "instance : X := x"}), "instance")

    def test_independent_prompt_uses_original_slots_and_one_newline(self):
        ec = command("e", 2, "variable (n : Nat)", "variable")
        mc = {**command("a", 10, "def a := 0"), "prompt_slot_source_order": 1,
              "prompt_slot_command_uid": "original"}
        p = reconstruct_prompt("theorem target : True :=\n\n", [ec], [mc])
        self.assertLess(p.index("def"), p.index("variable"))
        self.assertTrue(p.endswith(":=\n"))
        self.assertEqual(mc["source_order"], 10)

    def test_exact_tokenizer_encode_BOS_only_for_llama(self):
        tok = FakeTokenizer()
        self.assertEqual(encode(tok, "abc\n", "llada"), [97, 98, 99, 10])
        self.assertEqual(encode(tok, "abc\n", "llama31"), [10000, 97, 98, 99, 10])
        self.assertNotIn(tok.eos_token_id, encode(tok, "abc\n", "llama31"))

    def test_nested_generator_proof_fields_rejected(self):
        no_proof_fields({"target_id": "a", "target_kind": "theorem"})
        for key in ("target", "reference_proof", "labels", "answer_ids"):
            with self.assertRaisesRegex(ValueError, "proof/answer"):
                no_proof_fields({"conditions": [{key: "secret"}]})

    def test_type_and_value_union_retains_projections(self):
        raw = {"target_command_id": "t", "target_name": "N.t", "outer_type_constants": ["N.Type"],
               "outer_value_constants": ["N.used"], "type_constants": ["N.Type"],
               "value_constants": ["N.used"], "value_minus_type_constants": ["N.used"],
               "type_dependency": {"command_ids": ["a"], "constant_evidence": {"a": [{"constant": "N.Type"}]}},
               "value_dependency": {"command_ids": ["b"], "constant_evidence": {"b": [{"constant": "N.used"}]}},
               "expanded_type_auxiliaries": [], "expanded_value_auxiliaries": []}
        declarations = {"N.t": {"type_projections": [], "value_projections": [{"structure": "N.S", "projection_name": "N.S.p"}]}}
        ids, names = observed_sets(raw, declarations)
        self.assertEqual(ids, {"a", "b", "t"})
        self.assertTrue({"N.Type", "N.used", "N.S", "N.S.p"} <= names)

    def sampling_fixture(self):
        population = [f"id{i:05d}" for i in range(11064)]
        drawn = random.Random(2026).sample(sorted(population), 2500)
        sampled = {"sampling_seed": 2026, "sample_size": 2500, "population_size": 11064,
                   "population_sha256": object_hash(sorted(population)), "sampled_target_ids": drawn}
        eligibility = [{"sample_id": sid, "sample_rank": rank, "both_models_eligible": rank < 100,
                        "models": {m: {"eligible": rank < 100, "reason": None if rank < 100 else "no_match"} for m in ("llada", "llama31")}}
                       for rank, sid in enumerate(drawn)]
        final = {"requested_targets": 2000, "sampled_targets": 2500, "target_ids": drawn[:100],
                 "common_eligible_targets": 100, "actual_final_targets": 100,
                 "requested_size_reached": False, "no_backfill_or_matching_rule_relaxation": True,
                 "model_failure_counts": {m: {"eligible": 100, "no_match": 2400} for m in ("llada", "llama31")}}
        return population, sampled, eligibility, final

    def test_partial_frozen_population_is_truthful_not_forced_2000(self):
        values = self.sampling_fixture()
        self.assertEqual(len(validate_sampling(*values)), 100)

    def test_sample_drop_backfill_reorder_and_fabricated_count_rejected(self):
        for variant in ("drop", "backfill", "reorder", "count"):
            population, sampled, eligibility, final = self.sampling_fixture()
            if variant == "drop": eligibility.pop()
            elif variant == "backfill": final["target_ids"].append(sampled["sampled_target_ids"][101])
            elif variant == "reorder": sampled["sampled_target_ids"][:2] = sampled["sampled_target_ids"][:2][::-1]
            else: final["actual_final_targets"] = 2000
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_sampling(population, sampled, eligibility, final)

    def test_valid_pair_and_all_condition_tokens(self):
        target, raw, encoded, tok, commands, profiles = fixture()
        result = validate_model_pair(target, raw, encoded, "llada", tok, commands, profiles, {"t"}, {"N.t"})
        self.assertEqual(set(result), {"Env", "Selected", "Control"})

    def test_kernel_range_and_certified_ownership_source_identity(self):
        commands = [{"source_module": "Mathlib.F", "command_id": cid,
                     "source_span": {"start_line": line, "start_col": 0, "end_line": line + 1, "end_col": 0}}
                    for cid, line in (("a", 1), ("b", 3))]
        declarations = {name: {"module": "Mathlib.F", "selection_range": {"start_line": line, "start_column": 4}}
                        for name, line in (("N.a", 1), ("N.b", 3))}
        declarations["N.a.helper"] = {"module": "Mathlib.F", "selection_range": None}
        owners = {"N.a.helper": {"basis": "registered_structure_projection", "owners": ["N.a"]}}
        self.assertEqual(attributed_command("N.a", "kernel_selection_range", declarations, owners, commands), "a")
        self.assertEqual(attributed_command("N.a.helper", "registered_structure_projection", declarations, owners, commands), "a")
        owners["N.a.helper"]["owners"].append("N.b")
        with self.assertRaisesRegex(ValueError, "multiple source"):
            attributed_command("N.a.helper", "registered_structure_projection", declarations, owners, commands)

    def test_tampered_ids_ec_slots_kind_scope_and_dependency_rejected(self):
        for variant in ("ids", "ec", "slot", "kind", "scope", "dependency", "proof", "source"):
            target, raw, encoded, tok, commands, profiles = deepcopy(fixture())
            excluded = {"N.t"}
            if variant == "ids": encoded["conditions"]["Selected"]["input_ids"].pop()
            elif variant == "ec": raw["shared_EC"][0]["text"] = "changed"
            elif variant == "slot": raw["models"]["llada"]["replacement_MC"][0]["prompt_slot_source_order"] = 3
            elif variant == "kind": raw["models"]["llada"]["declaration_pairs"][0]["replacement_kind"] = "abbrev"
            elif variant == "scope": profiles["r"]["fingerprint"] = "not the same"
            elif variant == "dependency": excluded.add("N.r")
            elif variant == "proof": encoded["target"] = "by trivial"
            else: raw["models"]["llada"]["replacement_MC"][0]["source_order"] = 9
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_model_pair(target, raw, encoded, "llada", tok, commands, profiles, {"t"}, excluded)


if __name__ == "__main__":
    unittest.main()
