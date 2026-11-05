import unittest

from .evidence_analysis import aggregate_coverage, aggregate_paired, pass_at_k, reuse_category, target_coverage, verified_attempt


class EvidenceTests(unittest.TestCase):
    def test_pass_estimator(self):
        self.assertEqual(pass_at_k(0, 32, 32), 0)
        self.assertEqual(pass_at_k(1, 32, 32), 1)
        self.assertAlmostEqual(pass_at_k(4, 32, 1), 0.125)

    def test_coverage_deduplicates_command_uids(self):
        row = {"sample_id": "x", "dependency_mapping_complete": True, "final_visibility_known": True,
               "eligible_target_dependency_command_uids": ["a", "a", "b"], "visible_mc_command_uids": ["a"]}
        self.assertEqual(target_coverage(row), ("partial", .5))
        self.assertEqual(aggregate_coverage([row])["macro_coverage_percent"], 50)
        row["dependency_mapping_complete"] = False
        self.assertEqual(target_coverage(row), ("unresolved", None))

    def test_lexical_reference_is_not_semantic_reuse(self):
        record = {"passed": True, "exit_code": 0, "resource_evidence": [{
            "command_uid": "a", "constant_name": "Foo.bar", "basis": "lexical_match",
            "used_in_generated_proof": True, "target_type_only": False, "explicit_source_reference": True}]}
        self.assertIsNone(reuse_category(record, {"a"}))
        record["resource_evidence"][0]["basis"] = "kernel_proof_term"
        self.assertEqual(reuse_category(record, {"a"}), "explicit_reuse")
        record["resource_evidence"][0]["target_type_only"] = True
        self.assertIsNone(reuse_category(record, {"a"}))

    def test_unsubmitted_target_stays_in_denominator(self):
        targets = [{"sample_id": "a", "source_file": "A.lean", "mathematical_context": []},
                   {"sample_id": "b", "source_file": "B.lean", "mathematical_context": []}]
        right = {"a": [{"passed": True, "exit_code": 0}]}
        result = aggregate_paired(targets, {}, right)
        self.assertEqual(result["targets"], 2)
        self.assertEqual(result["metrics"]["pass_at_32"]["right_percent"], 50)
        self.assertEqual(result["paired_solved_targets"]["newly_solved"], 1)

    def test_absent_semantic_trace_is_not_zero_recorded_reuse(self):
        target = {"sample_id": "a", "source_file": "A.lean", "visible_mc_command_uids": ["A.lean::cmd_1"]}
        right = {"a": [{"passed": True, "exit_code": 0}]}
        result = aggregate_paired([target], {}, right)["paired_solved_targets"]
        self.assertEqual(result["newly_solved_reuse"]["semantic_evidence_not_supplied"], 1)
        self.assertIsNone(result["recorded_reuse_share_percent"])

    def test_infrastructure_failure_is_not_a_model_failure(self):
        with self.assertRaises(ValueError):
            verified_attempt({"passed": False, "exit_code": 1, "error_type": "infrastructure_error"})


if __name__ == "__main__":
    unittest.main()
