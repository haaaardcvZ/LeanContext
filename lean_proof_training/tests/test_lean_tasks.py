from __future__ import annotations

import unittest

from lean_proof_training.evaluation.candidates import Candidate, precheck_proof
from lean_proof_training.evaluation.lean_tasks import candidate_declaration


class LeanTaskTests(unittest.TestCase):
    def test_statement_with_assignment_gets_one_assignment(self) -> None:
        candidate = Candidate("a", 0, "by\n  trivial", "hash", {})
        declaration = candidate_declaration(
            "@[simp]\ntheorem anonymous_target : True :=", candidate, "benchmark_eval_a_0"
        )
        self.assertEqual(declaration.count(":="), 1)
        self.assertNotIn("@[simp]", declaration)
        self.assertIn("theorem benchmark_eval_a_0", declaration)

    def test_rejects_placeholders_and_command_injection(self) -> None:
        self.assertEqual(precheck_proof("by sorry"), "sorry_or_admit")
        self.assertEqual(
            precheck_proof("by trivial\ntheorem injected : True := by trivial"),
            "top_level_command_injection",
        )

    def test_anonymous_example_becomes_named_theorem(self) -> None:
        candidate = Candidate("a", 0, "by\n  trivial", "hash", {})
        declaration = candidate_declaration(
            "example : True :=", candidate, "benchmark_eval_example"
        )
        self.assertIn("theorem benchmark_eval_example : True :=", declaration)


if __name__ == "__main__":
    unittest.main()
