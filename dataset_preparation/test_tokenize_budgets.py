import unittest

from dataset_preparation.test_tokenize import CharacterTokenizer, command
from dataset_preparation.tokenize_budgets import prepare_budgeted_views


class IndependentBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.math = command("m", "def M := 1", 0)
        self.env = command("e", "variable (x : Nat)", 1, "EC")
        self.contexts = {"C0": [], "CMath": [self.math], "CEnv": [self.env], "CFull": [self.math, self.env]}

    def test_input_and_proof_limits_are_independent(self):
        result = prepare_budgeted_views(self.tokenizer, "S :=", "x" * 20, self.contexts, backbone="dllm", input_budgets=(5, 20, 50), output_limit=21)
        self.assertTrue(result["retained"])
        for budget, views in result["views_by_budget"].items():
            for row in views.values():
                self.assertLessEqual(row["prompt_length"], int(budget))
                self.assertEqual(row["answer_length"], 21)
                self.assertEqual(row["target"], "x" * 20)
                self.assertEqual(row["input_ids"][row["prompt_length"]:].count(2), 1)
                self.assertEqual(row["input_ids"][-1], 2)
                self.assertLessEqual(row["sequence_length"], int(budget) + 21)
        self.assertEqual(result["views_by_budget"]["5"]["C0"]["sequence_length"], 26)

    def test_smallest_input_budget_drives_common_membership(self):
        result = prepare_budgeted_views(self.tokenizer, "S :=", "x", self.contexts, backbone="llama31", input_budgets=(5, 20, 50), output_limit=2)
        self.assertFalse(result["retained"])
        self.assertEqual(result["eligible_by_input_budget"], {"5": False, "20": True, "50": True})
        self.assertEqual(result["bare_prompt_length"], 6)  # BOS + five text tokens
        self.assertNotIn("views_by_budget", result)

    def test_proof_not_shortened_to_create_fit(self):
        result = prepare_budgeted_views(self.tokenizer, "S :=", "xxxx", self.contexts, backbone="dllm", input_budgets=(50, 60), output_limit=4)
        self.assertFalse(result["retained"])
        self.assertEqual(result["answer_length"], 5)
        self.assertFalse(any(result["eligible_by_input_budget"].values()))

    def test_whole_commands_nested_and_math_env_derived_from_full(self):
        result = prepare_budgeted_views(self.tokenizer, "S :=", "x", self.contexts, backbone="dllm", input_budgets=(5, 20, 50), output_limit=2)
        previous = set()
        for budget, views in result["views_by_budget"].items():
            full = set(views["CFull"]["context_uids"])
            self.assertTrue(previous <= full)
            self.assertEqual(full, set(views["CMath"]["context_uids"]) | set(views["CEnv"]["context_uids"]))
            self.assertFalse(set(views["CMath"]["context_uids"]) & set(views["CEnv"]["context_uids"]))
            self.assertEqual(views["C0"]["context_ids"], [])
            for view in views.values():
                self.assertEqual(view["prompt"][-1], "\n")
                self.assertFalse(view["prompt"].endswith("\n\n"))
            previous = full
        self.assertEqual(result["views_by_budget"]["20"]["CFull"]["context_ids"], ["m"])
        self.assertEqual(result["views_by_budget"]["50"]["CFull"]["context_ids"], ["m", "e"])


if __name__ == "__main__":
    unittest.main()
