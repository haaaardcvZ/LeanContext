"""Offline numerical and indexing tests; no model checkpoint or Lean needed."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from .cka import central_proof_positions, centered_linear_cka, conditional_mask_positions, fixed_mask_positions
from .representation_analysis import (make_sample_manifest, model_loading_config, reference_ids,
                                     resolve_block_modules, theorem_mask_seed, validate_sample_manifest)


class CkaTests(unittest.TestCase):
    def test_central_window_is_one_based(self):
        self.assertEqual(central_proof_positions(8), [1, 2, 3, 4, 5])
        self.assertEqual(central_proof_positions(2), [0])
        with self.assertRaises(ValueError):
            central_proof_positions(1)

    def test_fixed_masks_repeat_across_conditions(self):
        seed = theorem_mask_seed("same-target", 42)
        left = fixed_mask_positions(11, seed)
        right = fixed_mask_positions(11, seed)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 5)
        self.assertEqual(len(left), len(set(left)))

    def test_conditional_short_proof_mask_is_retried_and_stable(self):
        self.assertEqual(central_proof_positions(2), [0])
        self.assertEqual(fixed_mask_positions(2, 0), [1])  # first draw has no central token
        expected = conditional_mask_positions(2, 0)
        self.assertEqual(expected[0], [0])
        self.assertGreater(expected[1], 1)
        self.assertEqual([conditional_mask_positions(2, 0) for _ in range(16)], [expected]*16)

    def test_joint_nonempty_window_pool_excludes_either_backbone_empty(self):
        lengths = {"llada": {"a": 2, "b": 3, "c": 5, "d": 3},
                   "llama31": {"a": 3, "b": 2, "c": 5, "d": 3}}
        manifest = make_sample_manifest(lengths, sources={}, seed=42, size=2)
        self.assertEqual(manifest["shared_population_count"], 4)
        self.assertEqual(manifest["central_empty_excluded_by_backbone"], {"llada": 1, "llama31": 1})
        self.assertEqual(manifest["shared_central_window_eligible_population_count"], 2)
        self.assertEqual(set(manifest["target_ids"]), {"c", "d"})
        self.assertEqual(set(lengths["llada"]), {"a", "b", "c", "d"})
        validate_sample_manifest(manifest, expected_size=2)

    def test_exact_4000_ids_and_old_strict_manifest_incompatible(self):
        lengths = {name: {f"{i:05d}": 5 for i in range(4002)} for name in ("llada", "llama31")}
        lengths["llada"]["00000"] = 2
        lengths["llama31"]["00001"] = 2
        manifest = make_sample_manifest(lengths, sources={})
        validate_sample_manifest(manifest)
        self.assertEqual(len(set(manifest["target_ids"])), 4000)
        self.assertEqual(manifest["sampling_seed"], 42)
        with self.assertRaises(ValueError):
            validate_sample_manifest({"target_ids": manifest["target_ids"], "sample_count": 4000})
        manifest["policy"]["mask_count"] = "ceil_0.5_times_proof_token_length"
        with self.assertRaises(ValueError):
            validate_sample_manifest(manifest)

    def test_reference_eos_removed(self):
        row = {"input_ids": [11, 12, 21, 22, 128001], "prompt_length": 2,
               "answer_span": {"start": 2, "end_exclusive": 5}}
        self.assertEqual(reference_ids(row, "llama31"), ([11, 12], [21, 22]))

    def test_default_full_state_loading_plan_is_pinned(self):
        config = model_loading_config(backbone="llada", training_view="CFull",
            dataset_root=Path("/release/dataset"), checkpoint="/runs/full_sft_CFull/best", dtype="bfloat16")
        self.assertEqual(config["architecture"], "dllm")
        self.assertEqual(config["model"]["name_or_path"], "GSAI-ML/LLaDA-8B-Base")
        self.assertEqual(len(config["model"]["revision"]), 40)
        self.assertEqual(config["model"]["checkpoint_type"], "auto")
        self.assertEqual(config["model"]["checkpoint"], "/runs/full_sft_CFull/best")
        self.assertTrue(config["model"]["trust_remote_code"])

    def test_base_config_is_not_mutated_and_checkpoint_overridden(self):
        base = {"architecture": "ar", "model": {"name_or_path": "local-base",
            "revision": "a" * 40, "checkpoint": "/old/best", "checkpoint_type": "auto",
            "trust_remote_code": False}}
        config = model_loading_config(backbone="llama31", training_view="CEnv",
            dataset_root=Path("/data"), checkpoint="/new/best", dtype="float32", base_config=base)
        self.assertEqual(config["model"]["checkpoint"], "/new/best")
        self.assertEqual(base["model"]["checkpoint"], "/old/best")
        self.assertEqual(config["model"]["revision"], "a" * 40)

    def test_lora_wrapper_exposes_real_blocks(self):
        blocks = [object() for _ in range(32)]
        underlying = SimpleNamespace(model=SimpleNamespace(layers=blocks))
        wrapper = SimpleNamespace(get_base_model=lambda: underlying)
        actual, path = resolve_block_modules(wrapper)
        self.assertIs(actual, blocks)
        self.assertEqual(path, "get_base_model().model.layers")

    def test_cka_identity_centering_and_rotation(self):
        import numpy as np
        x = np.array([[1, 2, 0], [3, 5, 2], [2, 0, 3], [0, 1, 1]], dtype=float)
        self.assertAlmostEqual(centered_linear_cka(x, x), 1.0)
        self.assertAlmostEqual(centered_linear_cka(x, x + 100), 1.0)
        q = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
        self.assertAlmostEqual(centered_linear_cka(x, 2 * x @ q), 1.0)

    def test_cka_rejects_constant_and_nonfinite(self):
        import numpy as np
        with self.assertRaises(ValueError):
            centered_linear_cka(np.ones((4, 2)), np.eye(4))
        with self.assertRaises(ValueError):
            centered_linear_cka(np.array([[1, float("nan")], [2, 3]]), np.eye(2))


if __name__ == "__main__":
    unittest.main()
