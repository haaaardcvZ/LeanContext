"""Tiny CPU D.4 fixtures, separate from D.5 selector and replacement results."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from evaluation.current_inputs import VIEWS, encode, invalid_input_candidates, ordered, render
from evaluation.reconstruction_inputs import reconstruction_input_for_generation, validate_pair
from evaluation.run_interventions import reconstruction_config, reuse_candidate_predictions
from evaluation.tests.test_replacement_interventions import Tokenizer, TargetWithUnreadableGold


def fixture(backbone="llada", invalid=False):
    def command(cid, order, text):
        return {"command_id": cid, "command_uid": "Mathlib/F.lean::" + cid,
                "source_order": order, "text": text}
    e, m = command("e", 0, "namespace N"), command("m", 1, "def x : Nat := 0")
    re, rm = command("re", 2, "open Nat"), command("rm", 3, "def y : Nat := 0")
    immutable = {"sample_id": "id", "target_id": "id", "statement": "theorem target : True :=",
                 "source_file": "Mathlib/F.lean", "source_module": "Mathlib.F",
                 "target_command_id": "t", "target_command_uid": "Mathlib/F.lean::t",
                 "target_source_span": {"start_byte": 1000}, "split": "test",
                 "proof_family": "Ordinary Tactic", "target_kind": "theorem", "original_name": "N.target"}
    target = TargetWithUnreadableGold({**immutable, "target": "FORBIDDEN_TARGET_GOLD",
        "full_context": [e, m], "mathematical_context": [m], "environment_context": [e]})
    pair = {**immutable, "input_limit": 1024, "output_limit": 1024,
            "reference_proof_fields_accessed": False, "model_outcomes_used": False, "variants": {}}
    for variant, mc, ec in (("original", [m], [e]), ("reconstructed", [] if invalid else [rm], [] if invalid else [re])):
        full = ordered([*mc, *ec])
        math_keys = {c["command_uid"] for c in mc}
        env_keys = {c["command_uid"] for c in ec}
        views = {}
        for view, items in (("C0", []), ("CMath", mc), ("CEnv", ec), ("CFull", full)):
            prompt = render(target["statement"], items)
            ids = encode(Tokenizer(), prompt, backbone)
            bad = variant == "reconstructed" and invalid and view != "C0"
            views[view] = {"view": view, "prompt": prompt, "input_ids": ids,
                "prompt_length": len(ids), "input_length": len(ids), "input_limit": 1024, "output_limit": 1024,
                "output_limit_includes_terminal_EOS": True, "input_ids_contain_prompt_only": True,
                "reference_proof_in_generator_input": False, "valid_input": not bad,
                "reason": "source_scope_unresolved" if bad else None,
                "visible_command_ids": [c["command_id"] for c in items],
                "visible_command_uids": [c["command_uid"] for c in items],
                "visible_mc_command_ids": [c["command_id"] for c in items if c["command_uid"] in math_keys],
                "visible_ec_command_ids": [c["command_id"] for c in items if c["command_uid"] in env_keys],
                "context_items": len(items)}
        pair["variants"][variant] = {"views": views, "mathematical_context": mc, "environment_context": ec,
                                      "EC_rebuilt_for_each_MC_trial": True, "D5_fixed_EC_used": False}
    return target, pair


class ReconstructionInterventionTests(unittest.TestCase):
    def test_original_and_reconstructed_both_change_mc_and_ec_not_target(self):
        for backbone in ("llada", "llama31"):
            target, pair = fixture(backbone)
            validate_pair(target, pair)
            prompts = {}
            for variant in ("original", "reconstructed"):
                for view in VIEWS:
                    prompt, ids, _, reason = reconstruction_input_for_generation(target, pair,
                        variant=variant, view=view, tokenizer=Tokenizer(), backbone=backbone)
                    self.assertIsNone(reason)
                    self.assertEqual(ids, encode(Tokenizer(), prompt, backbone))
                    self.assertNotIn("FORBIDDEN_TARGET_GOLD", prompt)
                    prompts[variant, view] = prompt
            self.assertEqual(prompts["original", "C0"], prompts["reconstructed", "C0"])
            self.assertNotEqual(prompts["original", "CMath"], prompts["reconstructed", "CMath"])
            self.assertNotEqual(prompts["original", "CEnv"], prompts["reconstructed", "CEnv"])

    def test_invalid_reconstruction_does_not_drop_target_or_change_c0(self):
        target, pair = fixture(invalid=True)
        for view in VIEWS:
            prompt, ids, _, reason = reconstruction_input_for_generation(target, pair,
                variant="reconstructed", view=view, tokenizer=Tokenizer(), backbone="llada")
            if view == "C0":
                self.assertIsNone(reason)
            else:
                self.assertEqual(reason, "source_scope_unresolved")
                candidates = invalid_input_candidates(target, view, prompt, len(ids), reason)
                self.assertEqual(len(candidates), 33)
                self.assertTrue(all(c["sample_id"] == "id" and c["proof"] == "" for c in candidates))

    def test_original_inputs_cannot_be_reselected_or_refilled(self):
        target, pair = fixture()
        pair["variants"]["original"]["environment_context"] = []
        with self.assertRaisesRegex(ValueError, "Full prefix"):
            validate_pair(target, pair)

    def test_four_views_must_project_one_shared_full_input(self):
        target, pair = fixture()
        pair["variants"]["reconstructed"]["views"]["CMath"]["prompt"] += "\nEXTRA"
        with self.assertRaisesRegex(ValueError, "unrefilled projection"):
            validate_pair(target, pair)

    def test_d5_fixed_ec_baseline_cannot_be_renamed_d4(self):
        target, pair = fixture()
        pair["variants"]["reconstructed"]["D5_fixed_EC_used"] = True
        with self.assertRaisesRegex(ValueError, "D.5"):
            validate_pair(target, pair)

    def test_appended_gold_ids_and_changed_c0_rejected(self):
        target, pair = fixture()
        pair["variants"]["original"]["views"]["CFull"]["input_ids"].append(999)
        with self.assertRaisesRegex(ValueError, "appended answers"):
            validate_pair(target, pair)
        target, pair = fixture()
        pair["variants"]["reconstructed"]["views"]["C0"]["valid_input"] = False
        pair["variants"]["reconstructed"]["views"]["C0"]["reason"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "C0 must be identical"):
            validate_pair(target, pair)

    def test_configs_use_same_cfull_checkpoint_eight_cells_and_fixed2000(self):
        base = {"model": {"checkpoint": "/checkpoints/llada/full_sft_CFull/best"}, "data": {},
                "generation": {}, "evaluation": {}, "output": {}}
        extension = {"pairs_file": "/pairs.jsonl", "allowed_ids_file": "/ids.jsonl",
                     "extension_root": "/extension", "manifest_sha256": "abc", "population": 2000}
        configs = [reconstruction_config(base, backbone="llada", variant=variant, view=view,
                                        extension=extension, output_root=Path("/new"))
                   for variant in ("original", "reconstructed") for view in VIEWS]
        self.assertEqual(len({c["output"]["directory"] for c in configs}), 8)
        self.assertEqual(len({c["model"]["checkpoint"] for c in configs}), 1)
        self.assertTrue(all(c["evaluation"]["expected_population_size"] == 2000 for c in configs))
        self.assertTrue(all(c["intervention"]["train_view"] == "CFull" for c in configs))
        self.assertTrue(all(not c["intervention"]["is_D5_shared_pool_selector"] for c in configs))

    def test_identical_c0_reuses_exact_candidates_not_new_fake_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids_file = root / "ids.jsonl"
            ids_file.write_text(json.dumps({"sample_id": "fixture"}) + "\n", encoding="utf-8")
            base = {"model": {"checkpoint": "/checkpoints/llada/full_sft_CFull/best"},
                    "data": {"test_file": "/original/test.jsonl"}, "generation": {}, "evaluation": {}, "output": {}}
            extension = {"pairs_file": "/pairs.jsonl", "allowed_ids_file": str(ids_file),
                         "extension_root": "/extension", "manifest_sha256": "abc", "population": 2000}
            configs = [reconstruction_config(base, backbone="llada", variant=v, view="C0",
                                            extension=extension, output_root=root) for v in ("original", "reconstructed")]
            source, target = configs
            source_dir = Path(source["output"]["directory"])
            target_dir = Path(target["output"]["directory"])
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            candidates = [{"sample_id": "fixture", "mode": "sampled", "candidate_id": i,
                           "context_view": "original", "proof": f"fixture-proof-{i}"} for i in range(32)]
            candidates.append({"sample_id": "fixture", "mode": "greedy_temperature_0", "candidate_id": 0,
                               "context_view": "original", "proof": "fixture-greedy"})
            (source_dir / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in candidates))
            (source_dir / "evaluation_targets.jsonl").write_text(ids_file.read_text())
            (source_dir / "generation_manifest.json").write_text(json.dumps({"model": {"fixture": True}}))
            reuse_candidate_predictions(source, target, config_file=target_dir / "config.yaml")
            actual = [json.loads(line) for line in (target_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([r["proof"] for r in actual], [r["proof"] for r in candidates])
            self.assertTrue(all(r["context_view"] == "reconstruction_reconstructed_C0" for r in actual))
            manifest = json.loads((target_dir / "generation_manifest.json").read_text())
            self.assertTrue(manifest["predictions_reused"])
            self.assertFalse(manifest["model_generation_performed"])
            target["generation"]["sampled"]["temperature"] = 0.7
            with self.assertRaisesRegex(ValueError, "decoding settings"):
                reuse_candidate_predictions(source, target, config_file=target_dir / "config.yaml")


if __name__ == "__main__":
    unittest.main()
