from __future__ import annotations

import unittest

from lean_proof_training.evaluation.passk import pass_at_k, summarize


class PassAtKTests(unittest.TestCase):
    def test_unbiased_estimator_and_undefined_coverage(self) -> None:
        self.assertAlmostEqual(pass_at_k(10, 2, 1), 0.2)
        self.assertAlmostEqual(pass_at_k(10, 2, 2), 1.0 - (8 * 7) / (10 * 9))
        self.assertIsNone(pass_at_k(3, 1, 4))

    def test_family_and_non_tactic_summaries(self) -> None:
        index = {
            "a": {"sample_id": "a", "proof_family": "Strict Term"},
            "b": {"sample_id": "b", "proof_family": "Ordinary Tactic"},
        }
        results = [
            {"sample_id": "a", "candidate_id": 0, "passed": True, "proof_hash": "x"},
            {"sample_id": "b", "candidate_id": 0, "passed": False, "proof_hash": "y"},
        ]
        report = summarize(results, index, [1])
        self.assertEqual(report["overall"]["solve_rate"]["mean"], 0.5)
        self.assertEqual(report["aggregates"]["Non-Tactic"]["solve_rate"]["mean"], 1.0)
        self.assertEqual(report["macro_over_proof_families"]["solve_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
