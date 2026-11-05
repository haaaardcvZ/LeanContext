import copy
import unittest

from .paper_audits import (coverage_row, coverage_summary, final_visibility,
                          make_length_groups, mapped_dependencies, pool_coverage,
                          recorded_mixed_groups)


def mapping(entries):
    return {"command_ids": list(entries), "constant_evidence": {
        cid: [{"constant": constant, "source_attribution": "kernel_selection_range"}
              for constant in constants] for cid, constants in entries.items()},
        "unmapped_same_file_constants": [], "same_target_command_constants": [],
        "nonpreceding_same_file_constants": [], "other_local_commands": []}


class PaperAuditTests(unittest.TestCase):
    def setUp(self):
        self.target = {"sample_id": "x", "source_file": "A.lean", "target_command_id": "cmd_3",
                       "target_source_span": {"start_byte": 100}, "statement": "theorem t : P :="}
        self.catalog = {f"cmd_{i}": {"source_file": "A.lean", "effective_kind": "def",
            "source_span": {"start_byte": 10*i, "end_byte": 10*i+5}} for i in (1, 2)}
        self.kernel = {"sample_id": "x", "source_file": "A.lean", "target_command_id": "cmd_3",
            "target_name": "t", "reference_proof_exact_raw_body_match": True,
            "outer_type_constants": ["A"], "type_constants": ["A", "ownedAux"],
            "value_constants": ["A", "ownedAux", "B"], "value_minus_type_constants": ["B"],
            "type_dependency": mapping({"cmd_1": ["A"], "cmd_2": ["ownedAux"]}),
            "proof_extra_dependency": mapping({"cmd_1": ["B"]}),
            "projection_augmented_proof_extra_dependency": mapping({"cmd_1": ["B"]})}

    def test_direct_type_excludes_auxiliary_expansion(self):
        value = mapped_dependencies(self.target, self.kernel, self.catalog)
        self.assertTrue(value["complete"])
        self.assertEqual(value["command_ids"], ["cmd_1"])
        self.assertFalse(value["reference_proof_used"])

    def test_constant_difference_before_command_mapping(self):
        # A and B share cmd_1. Subtracting command sets would wrongly erase B.
        value = mapped_dependencies(self.target, self.kernel, self.catalog, mode="proof")
        self.assertTrue(value["complete"])
        self.assertEqual(value["command_ids"], ["cmd_1"])
        self.assertEqual(value["constant_evidence"]["cmd_1"][0]["constant"], "B")

    def test_inconsistent_constant_difference_stays_unresolved(self):
        kernel = copy.deepcopy(self.kernel)
        kernel["value_minus_type_constants"] = []
        value = mapped_dependencies(self.target, kernel, self.catalog, mode="proof")
        self.assertFalse(value["complete"])
        self.assertIn("stored_constant_difference_is_inconsistent", value["reasons"])

    def test_unknown_is_not_empty_or_zero(self):
        kernel = copy.deepcopy(self.kernel)
        kernel["outer_type_constants"].append("UnknownLocal")
        kernel["type_dependency"]["unmapped_same_file_constants"] = ["UnknownLocal"]
        deps = mapped_dependencies(self.target, kernel, self.catalog)
        value = coverage_summary([coverage_row(self.target, deps, True, [])])
        self.assertEqual(value["auditable_targets"], 0)
        self.assertEqual(value["groups"], {"unresolved": 1})
        self.assertIsNone(value["macro_coverage_percent"])
        self.assertEqual(pool_coverage(deps, set()), "unresolved_mapping")

    def test_absent_completeness_evidence_stays_unresolved(self):
        kernel = copy.deepcopy(self.kernel)
        del kernel["type_dependency"]["unmapped_same_file_constants"]
        self.assertFalse(mapped_dependencies(self.target, kernel, self.catalog)["complete"])

    def test_nonpreceding_or_non_math_command_is_not_counted(self):
        catalog = copy.deepcopy(self.catalog)
        catalog["cmd_1"]["source_span"] = {"start_byte": 101, "end_byte": 105}
        value = mapped_dependencies(self.target, self.kernel, catalog)
        self.assertFalse(value["complete"])
        self.assertEqual(value["command_ids"], [])
        catalog["cmd_1"]["source_span"] = {"start_byte": 10, "end_byte": 15}
        catalog["cmd_1"]["effective_kind"] = "namespace"
        self.assertFalse(mapped_dependencies(self.target, self.kernel, catalog)["complete"])

    def test_final_visibility_uses_MC_role_not_raw_context(self):
        row = {"sample_id": "x", "statement": self.target["statement"], "view": "CFull",
               "prompt_length": 10, "input_ids": list(range(15)), "context_items": [
                   {"role": "EC", "command_uid": "A.lean::cmd_2"},
                   {"role": "MC", "command_uid": "A.lean::cmd_1"}]}
        self.assertEqual(final_visibility(row, self.target), (True, ["A.lean::cmd_1"], None))
        self.assertFalse(final_visibility(None, self.target)[0])

    def test_shared_quartiles_have_2766_targets(self):
        targets = [{"sample_id": f"{i:05d}"} for i in range(11064)]
        lengths = {row["sample_id"]: (int(row["sample_id"]) % 100) + 1 for row in targets}
        groups = make_length_groups(targets, lengths)
        self.assertEqual([len(ids) for ids in groups.values()], [2766]*4)
        self.assertEqual(len({sid for ids in groups.values() for sid in ids}), 11064)

    def test_mixed_groups_require_real_complete_annotations(self):
        self.assertEqual(recorded_mixed_groups({"x"}, None)["status"], "not_available")
        with self.assertRaises(ValueError):
            recorded_mixed_groups({"x", "y"}, {"x": {"has_mixed_role_command": True, "annotation_basis": "recorded"}})
        result = recorded_mixed_groups({"x"}, {"x": {"has_mixed_role_command": True, "annotation_basis": "human_recorded_v1"}})
        self.assertEqual(result["groups"]["at_least_one"], ["x"])


if __name__ == "__main__":
    unittest.main()
