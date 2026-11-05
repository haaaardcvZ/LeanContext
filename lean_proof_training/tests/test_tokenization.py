from __future__ import annotations

import unittest

from lean_proof_training.common.tokenization import encode_example
from tests.test_schema import CharacterTokenizer, ROW


class TokenizationTests(unittest.TestCase):
    def test_single_eos_and_optional_bos(self) -> None:
        encoded = encode_example(
            CharacterTokenizer(),
            ROW,
            view="statement_only",
            context_budget=0,
            budget_policy="prefix",
            max_length=4096,
            add_bos_token=True,
        )
        assert encoded is not None
        prompt_length = encoded["prompt_length"]
        target = encoded["input_ids"][prompt_length:]
        self.assertEqual(encoded["input_ids"][0], 98)
        self.assertEqual(target[-1], 99)
        self.assertEqual(target.count(99), 1)


if __name__ == "__main__":
    unittest.main()
