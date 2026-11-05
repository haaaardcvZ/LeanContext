"""CPU-only contract tests, with tiny synthetic fixtures, not benchmark results."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.current_inputs import encode
from evaluation.replacement_inputs import (
    CONDITIONS, IMMUTABLE, digest, extension_info, render,
    replacement_input_for_generation, validate_pair_contract, validate_replacement_artifacts,
)
from evaluation.run_interventions import replacement_config
from evaluation.run_interventions import main as interventions_main


class Tokenizer:
    bos_token_id = 100000
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class TargetWithUnreadableGold(dict):
    def __getitem__(self, key):
        if key == "target":
            raise AssertionError("generator read the target reference proof")
        return super().__getitem__(key)
    def get(self, key, default=None):
        if key == "target":
            raise AssertionError("generator read the target reference proof")
        return super().get(key, default)


def fixture(backbone="llada"):
    def command(cid, order, text):
        return {"command_id": cid, "command_uid": "Mathlib/F.lean::" + cid,
                "source_order": order, "text": text}
    ec = command("e", 0, "namespace N")
    selected = command("s", 1, "def x : Nat := 0")
    control = {**command("r", 10, "def y : Nat := 0"), "prompt_slot_command_id": "s",
               "prompt_slot_command_uid": selected["command_uid"], "prompt_slot_source_order": 1}
    immutable = {"sample_id": "id", "target_id": "id", "statement": "theorem target : True :=",
                 "source_file": "Mathlib/F.lean", "source_module": "Mathlib.F",
                 "target_command_id": "t", "target_command_uid": "Mathlib/F.lean::t",
                 "target_source_span": {"start_byte": 1000}, "split": "test",
                 "proof_family": "Ordinary Tactic", "target_kind": "theorem", "original_name": "N.target"}
    target = TargetWithUnreadableGold({**immutable, "target": "NEVER_GENERATOR_INPUT",
                                      "environment_context": [ec], "mathematical_context": [selected]})
    declaration_pairs = [{"selected_command_id": "s", "replacement_command_id": "r"}]
    raw = {**immutable, "shared_EC": [ec], "selected_MC": [selected],
           "models": {backbone: {"eligible": True, "replacement_MC": [control],
                                  "declaration_pairs": declaration_pairs}}}
    pair = {**immutable, "declaration_pairs": declaration_pairs, "shared_EC_ids": ["e"], "conditions": {}}
    for condition, mc in (("Env", []), ("Selected", [selected]), ("Control", [control])):
        prompt = render(immutable["statement"], [ec], mc)
        ids = encode(Tokenizer(), prompt, backbone)
        pair["conditions"][condition] = {
            "condition": condition, "prompt": prompt, "input_ids": ids,
            "input_length": len(ids), "prompt_length": len(ids), "input_limit": 1024,
            "output_limit": 1024, "output_limit_includes_terminal_EOS": True,
            "valid_input": True, "input_ids_contain_prompt_only": True,
            "reference_proof_in_generator_input": False,
            "visible_ec_command_ids": ["e"], "visible_ec_command_uids": [ec["command_uid"]],
            "visible_mc_command_ids": [c["command_id"] for c in mc],
            "visible_mc_command_uids": [c["command_uid"] for c in mc], "EC_count": 1, "MC_count": len(mc)}
    return target, pair, raw


def base_config():
    return {"architecture": "dllm", "model": {"checkpoint": "/checkpoints/llada/full_sft_CFull/best"},
            "data": {"test_file": "/original/benchmark/test.jsonl"},
            "source": {"mathlib_root": "/original/mathlib"}, "generation": {},
            "evaluation": {}, "output": {}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def extension_fixture(parent):
    target, pair, raw = fixture()
    dataset, extension = parent / "dataset", parent / "extension"
    test_file = dataset / "benchmark/test.jsonl"
    write_row(test_file, target)
    frozen = {"target_ids": ["id"], "actual_final_targets": 1, "requested_targets": 2000,
              "requested_size_reached": False}
    frozen_path = extension / "matched_replacement/frozen_test_manifest.json"
    write_json(frozen_path, frozen)
    targets_file = extension / "matched_replacement/targets.jsonl"
    write_row(targets_file, {key: target[key] for key in IMMUTABLE})
    raw_file = extension / "matched_replacement/pairs.jsonl"
    write_row(raw_file, raw)
    pairs_file = extension / "tokenized/llada/matched_replacement/input_1024/pairs.jsonl"
    write_row(pairs_file, pair)
    release = {"independent_validation_status": "passed", "input_source_test_sha256": digest(test_file),
               "artifacts": {str(p.relative_to(extension)): {"sha256": digest(p), "bytes": p.stat().st_size}
                             for p in (frozen_path, targets_file, raw_file, pairs_file)}}
    release_path = extension / "matched_replacement/manifest.json"
    write_json(release_path, release)
    write_json(extension / "manifest.json", {"validation_status": "passed", "sha256": digest(release_path),
        "actual_targets": 1, "requested_targets": 2000, "requested_size_reached": False})
    base = base_config()
    base["data"]["test_file"] = str(test_file)
    info = extension_info(extension, "llada")
    config = replacement_config(base, backbone="llada", train_view="CFull", condition="Selected",
                                extension=info, output_root=parent / "results")
    return config, extension, test_file


class ReplacementInterventionTests(unittest.TestCase):
    def test_exact_prompt_only_inputs_and_reference_proof_not_read(self):
        for backbone in ("llada", "llama31"):
            target, pair, raw = fixture(backbone)
            for condition in CONDITIONS:
                prompt, ids, count, invalid = replacement_input_for_generation(
                    target, pair, raw, condition=condition, tokenizer=Tokenizer(), backbone=backbone)
                self.assertIsNone(invalid)
                self.assertEqual(ids, encode(Tokenizer(), prompt, backbone))
                self.assertEqual(count, 1 if condition == "Env" else 2)
                self.assertNotIn("NEVER_GENERATOR_INPUT", prompt)

    def test_three_by_three_grid_has_fixed_checkpoints_and_actual_denominator(self):
        info = {"extension_root": "/extension", "frozen_manifest": "/frozen.json",
                "frozen_manifest_sha256": "abc", "population": 113, "requested_population": 2000,
                "requested_size_reached": False, "pairs_file": "/pairs.jsonl",
                "raw_pairs_file": "/raw.jsonl", "allowed_ids_file": "/actual_targets.jsonl"}
        base = base_config()
        configs = [replacement_config(base, backbone="llada", train_view=train, condition=condition,
                                      extension=info, output_root=Path("/new"))
                   for train in ("CMath", "CEnv", "CFull") for condition in CONDITIONS]
        self.assertEqual(len({c["output"]["directory"] for c in configs}), 9)
        self.assertEqual(len({c["model"]["checkpoint"] for c in configs}), 3)
        for config in configs:
            self.assertEqual(config["evaluation"]["expected_population_size"], 113)
            self.assertEqual(config["data"]["test_file"], base["data"]["test_file"])
            self.assertEqual(config["source"], base["source"])
            self.assertEqual(config["generation"]["sampled"], {"num_candidates": 32, "temperature": 0.6})
            self.assertEqual(config["generation"]["greedy"], {"num_candidates": 1, "temperature": 0.0})
            self.assertFalse(config["intervention"]["requested_size_reached"])
        self.assertEqual(base["model"]["checkpoint"], "/checkpoints/llada/full_sft_CFull/best")

    def test_c0_training_checkpoint_is_not_replacement_protocol(self):
        with self.assertRaisesRegex(ValueError, "CMath"):
            replacement_config(base_config(), backbone="llada", train_view="C0", condition="Env",
                               extension={}, output_root=Path("/new"))

    def test_changed_target_or_original_ec_rejected(self):
        for mutation, message in (("target", "immutable"), ("ec", "EC"), ("prompt", "serialized"),
                                  ("slot", "slot"), ("answer", "appended"), ("proof_field", "proof/answer")):
            target, pair, raw = fixture()
            if mutation == "target":
                pair["statement"] += " changed"
            elif mutation == "ec":
                raw["shared_EC"] = []
            elif mutation == "prompt":
                pair["conditions"]["Control"]["prompt"] += "extra"
            elif mutation == "slot":
                raw["models"]["llada"]["replacement_MC"][0]["prompt_slot_source_order"] += 1
            elif mutation == "answer":
                pair["conditions"]["Selected"]["input_ids"].append(999)
            else:
                pair["target"] = "forbidden proof"
            with self.assertRaisesRegex(ValueError, message):
                validate_pair_contract(target, pair, raw, backbone="llada")

    def test_changed_active_tokenizer_is_rejected(self):
        target, pair, raw = fixture("llama31")
        tokenizer = Tokenizer()
        tokenizer.bos_token_id = 100001
        with self.assertRaisesRegex(ValueError, "active tokenizer"):
            replacement_input_for_generation(target, pair, raw, condition="Selected",
                                               tokenizer=tokenizer, backbone="llama31")

    def test_validated_extension_checks_every_frozen_target(self):
        with tempfile.TemporaryDirectory() as temp:
            config, extension, test_file = extension_fixture(Path(temp))
            validate_replacement_artifacts(config)
            self.assertEqual(config["evaluation"]["expected_population_size"], 1)
            self.assertFalse(config["intervention"]["requested_size_reached"])
            config["evaluation"]["expected_population_size"] = 2000
            with self.assertRaisesRegex(ValueError, "denominator"):
                validate_replacement_artifacts(config)

    def test_read_only_plan_dispatches_grid_without_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            config, extension, _ = extension_fixture(Path(temp))
            base = base_config()
            base["data"]["test_file"] = config["data"]["test_file"]
            results = Path(temp) / "new_results"
            arguments = ["run_interventions", "replacement", "--backbone", "llada",
                         "--extension-root", str(extension), "--output-root", str(results), "--grid", "--plan"]
            output = io.StringIO()
            with patch("sys.argv", arguments), patch("evaluation.run_interventions.load_config", return_value=base), \
                    patch("evaluation.run_interventions.run_configs") as execute, redirect_stdout(output):
                interventions_main()
            execute.assert_not_called()
            plan = json.loads(output.getvalue())
            self.assertFalse(plan["execution_performed"])
            self.assertEqual(len(plan["planned_runs"]), 9)
            self.assertTrue(all(c["evaluation"]["expected_population_size"] == 1 for c in plan["planned_runs"]))
            self.assertFalse(results.exists())

    def test_changed_decoding_or_population_filter_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _, _ = extension_fixture(Path(temp))
            config["generation"]["sampled"]["temperature"] = 0.8
            with self.assertRaisesRegex(ValueError, "temperature 0.6"):
                validate_replacement_artifacts(config)
            config["generation"]["sampled"]["temperature"] = 0.6
            config["evaluation"]["max_samples"] = 1
            with self.assertRaisesRegex(ValueError, "forbids filtering"):
                validate_replacement_artifacts(config)

    def test_changed_or_missing_source_membership_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, extension, test_file = extension_fixture(Path(temp))
            test_file.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test file differs"):
                validate_replacement_artifacts(config)

    def test_modified_extension_artifact_rejected_even_before_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            config, extension, _ = extension_fixture(Path(temp))
            Path(config["data"]["tokenized_input_file"]).write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                validate_replacement_artifacts(config)

    def test_unfinalized_extension_cannot_be_evaluated(self):
        with tempfile.TemporaryDirectory() as temp:
            _, extension, _ = extension_fixture(Path(temp))
            manifest = json.loads((extension / "manifest.json").read_text())
            manifest["validation_status"] = "pending"
            write_json(extension / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "validation and finalization"):
                extension_info(extension, "llada")


if __name__ == "__main__":
    unittest.main()
