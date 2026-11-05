import unittest

from dataset_preparation.summarize_lengths import describe


class LengthStatisticsTests(unittest.TestCase):
    def test_exact_boundaries_and_empty_context(self):
        values = [0, 1, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 4096, 4097]
        stats = describe(values)
        self.assertEqual(stats["empty"], 1)
        self.assertEqual(stats["complete_context_coverage"]["256"]["all_targets_count"], 3)
        self.assertEqual(stats["complete_context_coverage"]["256"]["nonempty_targets_count"], 2)
        self.assertEqual(stats["complete_context_coverage"]["512"]["all_targets_count"], 5)
        self.assertEqual(stats["complete_context_coverage"]["1024"]["all_targets_count"], 7)
        self.assertEqual(stats["above_1024_count"], 5)
        self.assertEqual(sum(stats["histogram_counts"].values()), len(values))
        self.assertEqual(stats["nearest_rank_quantiles"]["median"], 513)

    def test_empty_population_not_zero_percent(self):
        stats = describe([])
        self.assertIsNone(stats["mean"])
        self.assertIsNone(stats["complete_context_coverage"]["1024"]["all_targets_percent"])
        self.assertIsNone(stats["complete_context_coverage"]["1024"]["nonempty_targets_percent"])


if __name__ == "__main__":
    unittest.main()
