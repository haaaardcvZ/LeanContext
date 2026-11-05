import itertools
import unittest

from .matched_replacement import in_length_band, kind, minimum_assignment, render_prompt, sampled_ids


class MatchedReplacementTests(unittest.TestCase):
    def test_length_band_is_inclusive_integer_exact(self):
        for size in (90, 100, 110):
            self.assertTrue(in_length_band(100, size))
        self.assertFalse(in_length_band(100, 89))
        self.assertFalse(in_length_band(100, 111))
        self.assertFalse(in_length_band(0, 0))
        self.assertTrue(in_length_band(17, 16))
        self.assertFalse(in_length_band(17, 15))

    def test_assignment_is_injective_and_not_greedy(self):
        result = minimum_assignment([{"a": 0, "b": 1}, {"a": 0}])
        self.assertEqual(result["candidate_ids"], ["b", "a"])
        self.assertEqual(result["total_absolute_token_deviation"], 1)

    def test_assignment_failure(self):
        self.assertIsNone(minimum_assignment([{"a": 0}, {"a": 0}]))
        self.assertIsNone(minimum_assignment([{}, {"a": 0}]))
        self.assertIsNone(minimum_assignment([{"a": 0}, {"a": 0}, {"b": 0, "c": 0}]))

    def test_matches_brute_force_cost(self):
        columns = ["a", "b", "c", "d"]
        for seed in range(20):
            import random
            r = random.Random(seed)
            edges = [{c: r.randrange(9) for c in columns if r.random() < .7} for _ in range(3)]
            costs = [sum(edges[i][c] for i, c in enumerate(p)) for p in itertools.permutations(columns, 3)
                     if all(c in edges[i] for i, c in enumerate(p))]
            result = minimum_assignment(edges)
            if costs:
                self.assertEqual(result["total_absolute_token_deviation"], min(costs))
            else:
                self.assertIsNone(result)

    def test_prompt_replacement_keeps_EC_slot_positions(self):
        ec = [{"source_order": 0, "command_uid": "e", "text": "namespace N"},
              {"source_order": 3, "command_uid": "v", "text": "variable (n : Nat)"}]
        mc = [{"source_order": 90, "command_uid": "r", "text": "def alternative := 1",
               "prompt_slot_source_order": 2, "prompt_slot_command_uid": "s"}]
        prompt = render_prompt("theorem target : True :=", ec, mc)
        self.assertLess(prompt.index("alternative"), prompt.index("variable"))
        self.assertEqual(mc[0]["source_order"], 90)
        self.assertTrue(prompt.endswith(":=\n"))
        self.assertFalse(prompt.endswith("\n\n"))

    def test_sample_independent_of_population_file_order(self):
        self.assertEqual(sampled_ids(["c", "a", "b"], 2, 2026), sampled_ids(["a", "b", "c"], 2, 2026))
        with self.assertRaises(ValueError):
            sampled_ids(["a", "a"], 1, 2026)

    def test_local_instance_is_distinct_kind(self):
        self.assertEqual(kind({"effective_kind": "instance", "text": "local instance : X := x"}), "local_instance")
        self.assertEqual(kind({"effective_kind": "instance", "text": "instance : X := x"}), "instance")


if __name__ == "__main__":
    unittest.main()
