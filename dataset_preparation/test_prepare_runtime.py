"""Offline tests for the canonical release-to-runtime preparation interface."""

from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from .prepare_runtime import (CANONICAL, VIEWS, export_saved_views, iter_array,
                              mark_validated, plan, prepare, refresh_cka_sources,
                              validate_record)


def record(sid: str = "test-id", split: str = "test") -> dict:
    statement, proof = "theorem anonymous_target : True :=", "by trivial"
    math = {"command_id": "cmd_1", "command_uid": "Mathlib/Test.lean::cmd_1",
            "source_order": 0, "role": "MC", "text": "lemma helper : True := by trivial"}
    env = {"command_id": "cmd_2", "command_uid": "Mathlib/Test.lean::cmd_2",
           "source_order": 1, "role": "EC", "text": "namespace Example"}
    contexts = {"C0": [], "CMath": [math], "CEnv": [env], "CFull": [math, env]}
    views = {view: {"sample_id": sid, "target_id": sid, "split": split,
                    "view": view, "statement": statement, "target": proof,
                    "context_items": items,
                    "context_ids": [item["command_id"] for item in items],
                    "context_uids": [item["command_uid"] for item in items],
                    "prompt": "\n\n".join([*(item["text"] for item in items), statement]) + "\n"}
             for view, items in contexts.items()}
    return {"sample_id": sid, "split": split, "statement": statement,
            "target": proof, "verified_by_lean": True, "views": views}


class StreamingArrayTests(unittest.TestCase):
    def test_small_chunks_unicode_and_escaped_text(self):
        expected = [{"id": 1, "value": "α\n\"brackets ] [\""}, {"id": 2, "value": "long" * 500}]
        for size in (1, 2, 7, 64, 65536):
            self.assertEqual(list(iter_array(io.StringIO(json.dumps(expected, ensure_ascii=False)), size)), expected)

    def test_empty_and_whitespace(self):
        self.assertEqual(list(iter_array(io.StringIO(" \n [ ] \n "), 1)), [])

    def test_invalid_json_is_not_silently_accepted(self):
        for text in ("{}", "[{\"id\": 1}", "[{\"id\": 1},]", "[] trailing", "[1]", "[{\"id\":1}{\"id\":2}]"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                list(iter_array(io.StringIO(text), 3))


class SavedViewTests(unittest.TestCase):
    def test_exact_saved_views_are_exported_without_reselection(self):
        original = record()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / CANONICAL
            source.write_text(json.dumps([original]), encoding="utf-8")
            counts = export_saved_views(source, root / "runtime", {"train": 0, "validation": 0, "test": 1})
            self.assertEqual(counts["test"], 1)
            for view in VIEWS:
                saved = json.loads((root / "runtime/training/views" / view / "test.jsonl").read_text())
                self.assertEqual(saved, original["views"][view])
            benchmark = json.loads((root / "runtime/benchmark/test.jsonl").read_text())
            self.assertNotIn("views", benchmark)
            self.assertEqual(benchmark, {key: value for key, value in original.items() if key != "views"})

    def test_duplicate_across_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / CANONICAL
            source.write_text(json.dumps([record(split="train"), record(split="test")]))
            with self.assertRaisesRegex(ValueError, "Duplicate target"):
                export_saved_views(source, root / "runtime", {"train": 1, "validation": 0, "test": 1})

    def test_changed_proof_prompt_or_projection_is_rejected(self):
        for mutation in ("proof", "prompt", "projection"):
            altered = deepcopy(record())
            if mutation == "proof":
                altered["views"]["C0"]["target"] = "by exact True.intro"
            elif mutation == "prompt":
                altered["views"]["C0"]["prompt"] += "\n"
            else:
                altered["views"]["CMath"]["context_items"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_record(altered)

    def test_marking_validation_preserves_numeric_token_ids(self):
        original = {"input_ids": [128000, 987, 128001], "prompt": "statement\n", "target": "proof"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text(json.dumps(original) + "\n")
            mark_validated(path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["input_ids"], original["input_ids"])
            self.assertTrue(saved["tokenization_validated"])
            self.assertEqual(saved["prompt"], original["prompt"])


class PlanTests(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        source = root / "submission"
        for name in ("selector", "source_snapshot", "tokenizers/llada", "tokenizers/llama31", "command_mapping/source_commands"):
            (source / name).mkdir(parents=True, exist_ok=True)
        for name in (CANONICAL, "selector/pairs.jsonl", "selector/frozen_test_manifest.json"):
            (source / name).write_text("[]\n")
        for model in ("llada", "llama31"):
            for filename in ("tokenizer.json", "tokenizer_config.json"):
                (source / "tokenizers" / model / filename).write_text("{}\n")
        return source

    def test_default_plan_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            output = root / "runtime"
            result = prepare(argparse.Namespace(dataset_root=source, output_root=output,
                models=["llada", "llama31"], materialization="copy", execute=False))
            self.assertEqual(result["mode"], "plan")
            self.assertFalse(result["writes_performed"])
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            output = root / "runtime"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "new output directory"):
                plan(source, output, ["llada"], "copy")

    def test_overlapping_source_and_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                plan(source, source / "runtime", ["llada"], "copy")

    def test_missing_model_tokenizer_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            (source / "tokenizers/llama31/tokenizer.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing release artifacts"):
                plan(source, root / "runtime", ["llama31"], "copy")


class FrozenCKATests(unittest.TestCase):
    def fixture(self, root: Path, extra: bool = False):
        from analysis.representation_analysis import make_sample_manifest
        lengths = {name: {f"id-{index:04d}": 5 for index in range(4000)}
                   for name in ("llada", "llama31")}
        old_sources = {f"{name}/{view}": {"sha256": "0" * 64, "records": 4000}
                       for name in lengths for view in VIEWS}
        manifest = make_sample_manifest(lengths, sources=old_sources)
        path = root / "analysis_subsets/cka_4000_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest))
        artifacts = {}
        for name, mapping in lengths.items():
            member = root / "tokenized" / name / "training/membership/test.jsonl"
            member.parent.mkdir(parents=True)
            if extra:
                mapping["extra-id"] = 5
            member.write_text("".join(json.dumps({"sample_id": sid, "answer_length": length,
                                                   "retained": True}) + "\n"
                                      for sid, length in mapping.items()))
            for view in VIEWS:
                relative = f"tokenized/{name}/training/input_1024/views/{view}/test.jsonl"
                artifacts[relative] = {"sha256": "f" * 64}
        return path, manifest, artifacts

    def test_source_rebinding_preserves_all_4000_ids_and_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, original, artifacts = self.fixture(root)
            refresh_cka_sources(root, ["llada", "llama31"], artifacts)
            updated = json.loads(path.read_text())
            self.assertEqual(updated["target_ids"], original["target_ids"])
            self.assertEqual(updated["policy"], original["policy"])
            self.assertEqual(updated["sampling_seed"], 42)
            self.assertEqual(updated["sources"]["llada/CFull"]["sha256"], "f" * 64)

    def test_changed_membership_cannot_silently_resample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, artifacts = self.fixture(root, extra=True)
            with self.assertRaisesRegex(ValueError, "do not silently resample"):
                refresh_cka_sources(root, ["llada", "llama31"], artifacts)


if __name__ == "__main__":
    unittest.main()
