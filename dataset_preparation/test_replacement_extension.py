"""Portable release checks; no Transformers, GPU, Lean, or original workspace."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .finalize_replacement import finalize
from .matched_replacement import new_extension_root, sha256
from .validate_replacement import validate_source_catalog


class ReplacementExtensionTests(unittest.TestCase):
    def test_output_never_nested_in_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            root.mkdir()
            for output in (root, root / "new_extension"):
                with self.assertRaisesRegex(ValueError, "outside"):
                    new_extension_root(root, output)
            self.assertEqual(list(root.iterdir()), [])

    def test_nonempty_output_refused_empty_directory_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "dataset", Path(temp) / "extension"
            root.mkdir()
            self.assertEqual(new_extension_root(root, out), out.resolve())
            (out / "user.txt").write_text("unchanged", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                new_extension_root(root, out)
            self.assertEqual((out / "user.txt").read_text(), "unchanged")

    def test_released_source_catalog_validates_and_detects_change(self):
        source = b"def x : Nat := 0\n"
        text = source.decode()
        command = {"command_id": "cmd_0", "command_uid": "Mathlib/F.lean::cmd_0",
                   "source_order": 0, "source_file": "Mathlib/F.lean",
                   "source_span": {"start_byte": 0, "end_byte": len(source)},
                   "raw_text": text, "effective_raw_text": text, "text": text.strip()}
        for key, field in (("raw_text_sha256", "raw_text"), ("effective_raw_text_sha256", "effective_raw_text"),
                           ("context_text_sha256", "text")):
            command[key] = hashlib.sha256(command[field].encode()).hexdigest()
        digest = hashlib.sha256(source).hexdigest()
        official = {"sha256": digest,
                    "official_git_blob_sha1": hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()}
        validate_source_catalog([command], "Mathlib/F.lean", source, digest, official)
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_source_catalog([command], "Mathlib/F.lean", source + b" ", digest, official)

    def test_finalize_without_dataset_manifest_writes_only_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            root, extension = Path(temp) / "dataset", Path(temp) / "extension"
            (root / "benchmark").mkdir(parents=True)
            test = root / "benchmark/test.jsonl"
            test.write_text("source-data\n", encoding="utf-8")
            original_bytes = test.read_bytes()
            out = extension / "matched_replacement"
            out.mkdir(parents=True)
            def put(path, value):
                path.write_text(json.dumps(value), encoding="utf-8")
            put(out / "validation_report.json", {"valid": True, "actual_final_targets": 0,
                "validator_sha256": sha256(Path(__file__).with_name("validate_replacement.py"))})
            put(out / "frozen_test_manifest.json", {"target_ids": [], "actual_final_targets": 0,
                "sampled_targets": 2500, "requested_targets": 2000, "requested_size_reached": False})
            put(out / "manifest.json", {"artifacts": {}, "processing_script_sha256": {},
                "input_source_test_sha256": sha256(test)})
            put(out / "build_status.json", {"status": "complete_partial_requested_size_not_reached"})
            for model in ("llada", "llama31"):
                for condition in ("Env", "Selected", "Control"):
                    path = extension / f"tokenized/{model}/matched_replacement/input_1024/views/{condition}/test.jsonl"
                    path.parent.mkdir(parents=True)
                    path.write_text("", encoding="utf-8")
            summary = finalize(root, extension)
            self.assertFalse(summary["requested_size_reached"])
            self.assertFalse(summary["submitted_dataset_modified"])
            self.assertFalse((root / "manifest.json").exists())
            self.assertEqual(test.read_bytes(), original_bytes)
            release = json.loads((out / "manifest.json").read_text())
            self.assertIn("tokenize.py", release["processing_package_sha256"])
            self.assertIn("selector.py", release["processing_package_sha256"])
            self.assertFalse((extension / "processing").exists())

    def test_help_requires_no_model_or_tokenizer_packages(self):
        code_root = Path(__file__).resolve().parents[1]
        for module in ("matched_replacement", "validate_replacement", "finalize_replacement"):
            completed = subprocess.run([sys.executable, "-m", "dataset_preparation." + module, "--help"],
                cwd=code_root, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--output-root", completed.stdout)

    def test_wrapper_preserves_relative_paths_from_calling_directory(self):
        code_root = Path(__file__).resolve().parents[1]
        script = code_root / "dataset_preparation/run_matched_replacement.sh"
        with tempfile.TemporaryDirectory() as temp:
            caller = Path(temp).resolve()
            logger = caller / "mock_python"
            log = caller / "commands.jsonl"
            logger.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "with open(os.environ['COMMAND_LOG'], 'a') as f:\n"
                "    f.write(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}) + '\\n')\n",
                encoding="utf-8")
            logger.chmod(0o755)
            env = {**os.environ, "PYTHON_BIN": str(logger), "COMMAND_LOG": str(log)}
            completed = subprocess.run(["bash", str(script), "dataset with spaces", "new extension"],
                cwd=caller, env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(commands), 3)
            self.assertEqual([c["args"][1] for c in commands], [
                "dataset_preparation.matched_replacement", "dataset_preparation.validate_replacement",
                "dataset_preparation.finalize_replacement"])
            for command in commands:
                self.assertEqual(Path(command["cwd"]).resolve(), code_root.resolve())
                args = command["args"]
                self.assertEqual(args[args.index("--dataset-root") + 1], str(caller / "dataset with spaces"))
                self.assertEqual(args[args.index("--output-root") + 1], str(caller / "new extension"))


if __name__ == "__main__":
    unittest.main()
