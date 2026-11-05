"""Offline regression tests for the newly added convenience entry point."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pipeline import check_classifier, nonnegative, record_identity, require_new, unresolved_failures


class DriverTests(unittest.TestCase):
    def test_zero_recursive_depth_is_supported(self):
        self.assertEqual(nonnegative("0"), 0)
        with self.assertRaises(argparse.ArgumentTypeError):
            nonnegative("-1")

    def test_identity_supports_both_original_schemas(self):
        self.assertEqual(record_identity({"lean_file_name": "A.lean", "target_command_id": "cmd_1"}), record_identity({"source_file": "A.lean", "target_command_id": "cmd_1"}))
        self.assertIsNone(record_identity({"source_file": "A.lean"}))

    def failures(self, successes, failures):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.jsonl"
            failed = Path(temporary) / "failed.jsonl"
            output.write_text("".join(json.dumps(row) + "\n" for row in successes))
            failed.write_text("".join(json.dumps(row) + "\n" for row in failures))
            return unresolved_failures(output, failed)

    def test_resume_does_not_count_recovered_failure(self):
        self.assertEqual(self.failures([{"lean_file_name": "A.lean", "target_command_id": "cmd_1"}], [{"source_file": "A.lean", "target_command_id": "cmd_1"}]), 0)

    def test_unresolved_failure_is_counted_once_per_target(self):
        row = {"source_file": "A.lean", "target_command_id": "cmd_1"}
        self.assertEqual(self.failures([], [row, row]), 1)

    def test_failure_without_identity_is_not_silently_dropped(self):
        self.assertEqual(self.failures([], [{"error": "unknown input"}]), 1)

    def test_overwrite_guard_and_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.jsonl"
            output.touch()
            with self.assertRaises(FileExistsError):
                require_new(output)
            require_new(output, resume=True)
            require_new(output, force=True)

    def test_legitimate_excluded_surface_is_not_a_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "surface.jsonl"
            output.write_text(json.dumps({"surface_group": "E00_invalid_or_excluded", "exclude_reason": "empty_body"}) + "\n")
            check_classifier(output)

    def test_classifier_exception_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "surface.jsonl"
            output.write_text(json.dumps({"exclude_reason": "classifier_exception"}) + "\n")
            with self.assertRaises(RuntimeError):
                check_classifier(output)


if __name__ == "__main__":
    unittest.main()
