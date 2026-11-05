from __future__ import annotations

import unittest

from lean_proof_training.common.schema import render_prompt, select_budgeted_items


class CharacterTokenizer:
    eos_token_id = 99
    bos_token_id = 98

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": list(range(len(text)))}


ROW = {
    "sample_id": "x",
    "split": "train",
    "training_stage": "warmup",
    "proof_family": "Strict Term",
    "statement": "theorem t : True :=",
    "target": "by trivial",
    "contexts": {
        "matched": [
            {"kind": "namespace", "text": "namespace A"},
            {"kind": "variable", "text": "variable (n : Nat)"},
        ],
        "random_control": None,
        "hard_control": None,
    },
}


class SchemaTests(unittest.TestCase):
    def test_prompt_owns_exactly_one_separator_newline(self) -> None:
        prompt, _ = render_prompt(ROW, "matched")
        self.assertTrue(prompt.endswith("\n"))
        self.assertFalse(prompt.endswith("\n\n"))
        self.assertFalse(ROW["target"].startswith("\n"))

    def test_prefix_budget_does_not_skip_oversized_item(self) -> None:
        selected = select_budgeted_items(
            CharacterTokenizer(), ROW["contexts"]["matched"], 10, policy="prefix"
        )
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
