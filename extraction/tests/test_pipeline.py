"""Offline regression tests for the bundled, actual extraction implementation.

Run from this directory's parent: python -m unittest discover -s tests -v
No API request, Lean process, external dataset, or installed package is needed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT / "theorem_generate"))

from extract_data import context_pipeline as pipeline
from extract_data.build_context_deepseek_hybrid import make_deepseek_client, select_with_model
from extract_data.classify_proof_surface import strip_lean_comments
from extract_data.llm_client import parse_json_object
from extract_data.remove_proof_leakage_context import filter_kept_ids
from lean_parser.batch_extract_mathlib_command_spans import (
    mask_lean_comments_bytes, normalize_effective_raw_text, strip_lean_comments_text,
)


def command(cid: str, text: str, *, kind: str = "def", name: str | None = None,
            order: int = 0) -> dict:
    return {"command_id": cid, "kind": kind, "name": name, "effective_raw_text": text,
            "source_file": "Mathlib/Fixture.lean", "source_module": "Mathlib.Fixture",
            "start_byte": order * 100, "start_line": order + 1, "end_line": order + 1}


def fixture_case() -> pipeline.TargetCase:
    target = command("target", "theorem original_name : True := by\n  exact True.intro\n",
                     kind="theorem", name="original_name", order=9)
    return pipeline.TargetCase(Path("fixture.jsonl"), target, [], [])


def client_args(mode: str = "llm") -> argparse.Namespace:
    return argparse.Namespace(mode=mode, base_url="https://example.invalid", model=None,
                              temperature=0.0, timeout=120, max_retries=3,
                              max_chars_per_command=5000)


class OfflinePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        # Any accidental external call turns into a test failure immediately.
        network = patch("urllib.request.urlopen", side_effect=AssertionError("offline test attempted network"))
        process = patch("subprocess.run", side_effect=AssertionError("offline test attempted a subprocess"))
        network.start(); process.start()
        self.addCleanup(network.stop)
        self.addCleanup(process.stop)

    def test_imported_code_is_inside_the_bundle(self) -> None:
        self.assertTrue(Path(pipeline.__file__).resolve().is_relative_to(BUNDLE_ROOT))

    def test_prompt_contains_original_gold_proof_and_candidate_ids(self) -> None:
        case = fixture_case()
        candidates = [command("c1", "def localFact := True", name="localFact", order=1),
                      command("c2", "variable {α : Type*}", kind="variable", order=2)]
        messages = pipeline.make_llm_messages(case, candidates, max_chars_per_command=5000)
        self.assertEqual([row["role"] for row in messages], ["system", "user"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["target"]["command_id"], "target")
        self.assertIn("exact True.intro", payload["target"]["text"])
        self.assertIn("theorem original_name", payload["target"]["text"])
        self.assertEqual([row["command_id"] for row in payload["candidate_commands"]], ["c1", "c2"])
        self.assertIn('"keep_command_ids"', messages[0]["content"])
        self.assertIn('"drop_command_ids"', messages[0]["content"])
        self.assertIn('"reason"', messages[0]["content"])
        self.assertIn("Do not include the target command_id", messages[0]["content"])

    def test_prompt_truncation_is_character_based_and_target_gets_double(self) -> None:
        case = fixture_case(); case.target["effective_raw_text"] = "α" * 30
        messages = pipeline.make_llm_messages(case, [command("c", "β" * 20)], max_chars_per_command=5)
        payload = json.loads(messages[1]["content"])
        suffix = "\n/- TRUNCATED FOR LLM INPUT -/"
        self.assertEqual(payload["target"]["text"], "α" * 10 + suffix)
        self.assertEqual(payload["candidate_commands"][0]["text"], "β" * 5 + suffix)

    def test_normalize_keep_ids_filters_unknown_self_and_duplicates(self) -> None:
        candidates = [command("a", ""), command("b", ""), command("target", "")]
        self.assertEqual(pipeline.normalize_keep_ids(
            {"keep_command_ids": ["b", "bogus", "b", "target", "a"]}, candidates, "target"), ["b", "a"])

    def test_normalize_keep_ids_ignores_drop_list_and_nonlist_keep(self) -> None:
        candidates = [command("a", "")]
        self.assertEqual(pipeline.normalize_keep_ids(
            {"keep_command_ids": ["a"], "drop_command_ids": ["a"]}, candidates, "target"), ["a"])
        for value in (None, "a", {"a": True}):
            self.assertEqual(pipeline.normalize_keep_ids({"keep_command_ids": value}, candidates, "target"), [])

    def test_split_preserves_newline_and_indentation_in_reference_proof(self) -> None:
        result = pipeline.split_target_text("theorem t : True :=\n  rfl\n")
        self.assertEqual(result["target_header_text"], "theorem t : True :=")
        self.assertEqual(result["target_body_text"], "\n  rfl\n")
        self.assertEqual(result["target_split_method"], "assignment")
        self.assertTrue(result["target_split_found"])

    def test_split_inline_reference_removes_only_one_boundary_space(self) -> None:
        result = pipeline.split_target_text("lemma t : True := by trivial\n")
        self.assertEqual(result["target_body_text"], "by trivial\n")
        result = pipeline.split_target_text("lemma t : True :=  by trivial\n")
        self.assertEqual(result["target_body_text"], " by trivial\n")

    def test_split_ignores_attribute_and_binder_assignments(self) -> None:
        text = "@[attr (priority := 42)]\ntheorem t (n : Nat := 1) : n = n := by rfl\n"
        result = pipeline.split_target_text(text)
        self.assertTrue(result["target_header_text"].endswith(": n = n :="))
        self.assertEqual(result["target_body_text"], "by rfl\n")

    def test_split_ignores_assignment_and_where_inside_comments_strings(self) -> None:
        text = 'theorem t (s : String := ":= where") /- := /- where -/ -/ : True := by trivial\n'
        result = pipeline.split_target_text(text)
        self.assertEqual(result["target_body_text"], "by trivial\n")
        self.assertEqual(result["target_split_method"], "assignment")

    def test_split_equation_alternatives_and_missing_separator(self) -> None:
        result = pipeline.split_target_text("theorem t : Nat → True\n  | 0 => by trivial\n  | n+1 => by trivial\n")
        self.assertEqual(result["target_split_method"], "equation_alternatives")
        self.assertTrue(result["target_body_text"].startswith("  | 0"))
        result = pipeline.split_target_text("theorem t : True")
        self.assertFalse(result["target_split_found"])
        self.assertEqual(result["target_body_text"], "")

    def test_nested_comment_strip_preserves_strings_and_escaped_quotes(self) -> None:
        text = 'def s := "-- literal /- block -/ \\\"quoted\\\"" /- outer /- nested -/ tail -/\n-- removed\ndef n := 1\n'
        stripped = strip_lean_comments_text(text)
        self.assertIn('"-- literal /- block -/ \\\"quoted\\\""', stripped)
        self.assertNotIn("outer", stripped)
        self.assertNotIn("nested", stripped)
        self.assertNotIn("removed", stripped)
        self.assertIn("def n := 1", stripped)

    def test_byte_comment_mask_preserves_unicode_offsets_and_newlines(self) -> None:
        text = 'def α := "-- \u5b57\u7b26"\n/- \u5916 /- \u5185 -/\n\u5c3e -/\ndef β := 1 -- \u6ce8\u91ca\n'
        data = text.encode("utf-8")
        masked = mask_lean_comments_bytes(data)
        self.assertEqual(len(masked), len(data))
        self.assertEqual([i for i, b in enumerate(masked) if b == 10], [i for i, b in enumerate(data) if b == 10])
        self.assertIn('"-- \u5b57\u7b26"'.encode(), masked)
        self.assertNotIn("\u6ce8\u91ca".encode(), masked)
        self.assertEqual(masked.index("def β".encode()), data.index("def β".encode()))

    def test_surface_comment_strip_keeps_comment_newline_structure(self) -> None:
        self.assertEqual(strip_lean_comments("by\n/- first\n/- nested -/\nlast -/\nexact True.intro"),
                         "by\n\n\n\nexact True.intro")

    def test_effective_text_normalization_compacts_empty_lines(self) -> None:
        self.assertEqual(normalize_effective_raw_text("\n /- note -/\n\n theorem t : True :=\n\n  by trivial\n\n"),
                         "theorem t : True :=\n  by trivial\n")
        self.assertEqual(normalize_effective_raw_text("/- only /- nested -/ comment -/"), "")

    def test_lexical_filter_drops_only_direct_theorem_lemma_names_absent(self) -> None:
        target = command("t", "theorem t : N.used = N.used := by rfl", kind="theorem", name="t", order=9)
        source = [command("used", "lemma N.used : True := by trivial", kind="lemma", name="N.used"),
                  command("unused", "theorem unrelated : True := by trivial", kind="theorem", name="unrelated"),
                  command("definition", "def unrelatedDef := True", kind="def", name="unrelatedDef"), target]
        kept, removed = filter_kept_ids(rec={"target_command_id": "t", "kept_command_ids": ["used", "unused", "definition"]},
                                       source_records=source, filter_kinds={"lemma", "theorem"})
        self.assertEqual(kept, ["used", "definition"])
        self.assertEqual([row["command_id"] for row in removed], ["unused"])

    def test_lexical_filter_requires_identifier_boundary_not_substring(self) -> None:
        target = command("t", "theorem t : used_extra = used_extra := by rfl", kind="theorem", order=9)
        source = [command("u", "lemma used : True := by trivial", kind="lemma", name="used"), target]
        kept, removed = filter_kept_ids(rec={"target_command_id": "t", "kept_command_ids": ["u"]},
                                       source_records=source, filter_kinds={"lemma", "theorem"})
        self.assertEqual(kept, [])
        self.assertEqual(len(removed), 1)

    def test_merge_restores_source_order_not_selection_order(self) -> None:
        first = command("a", "namespace N", kind="namespace", name="N", order=1)
        second = command("b", "variable {α : Type*}", kind="variable", order=2)
        third = command("c", "def localFact := True", name="localFact", order=3)
        context, closing, merged = pipeline.merge_commands([third, first, second], fixture_case().target,
            ["c", "b", "a"], include_import_mathlib=False, include_closing_ends=True)
        self.assertEqual(context, "namespace N\n\nvariable {α : Type*}\n\ndef localFact := True\n")
        self.assertEqual(closing, "end N\n")
        self.assertLess(merged.index("def localFact"), merged.index("theorem original_name"))
        self.assertTrue(merged.endswith("end N\n"))

    def test_merge_closes_nested_scopes_inside_out_with_optional_import(self) -> None:
        commands = [command("n", "namespace N", kind="namespace", name="N", order=1),
                    command("s", "section", kind="section", order=2)]
        context, closing, _ = pipeline.merge_commands(commands, fixture_case().target, ["s", "n"],
            include_import_mathlib=True, include_closing_ends=True)
        self.assertTrue(context.startswith("import Mathlib\n\nnamespace N"))
        self.assertEqual(closing, "end\n\nend N\n")

    def test_hybrid_client_missing_key_fails_before_network(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                make_deepseek_client(client_args(), model="fixture-model")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "   "}, clear=True):
            with self.assertRaises(RuntimeError):
                make_deepseek_client(client_args(), model="fixture-model")

    def test_hybrid_client_reads_environment_without_calling_api(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": " fixture-not-a-real-key "}, clear=True):
            client = make_deepseek_client(client_args(), model="fixture-model")
        self.assertEqual(client.config.api_key, "fixture-not-a-real-key")
        self.assertEqual(client.config.model, "fixture-model")
        self.assertEqual(client.config.temperature, 0.0)

    def test_heuristic_and_candidate_modes_do_not_require_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            for mode in ("heuristic", "candidates"):
                self.assertIsNone(make_deepseek_client(client_args(mode), model="fixture-model"))
            selected, reason = select_with_model(args=client_args("heuristic"), client=None,
                case=fixture_case(), candidates=[command("a", "def a := 1"), command("b", "def b := 2")])
        self.assertEqual(selected, ["a", "b"])
        self.assertIn("heuristic", reason)

    def test_generic_client_env_defaults_and_no_key_in_heuristic_mode(self) -> None:
        args = client_args("heuristic"); args.base_url = None
        with patch.dict(os.environ, {}, clear=True):
            client, url, model = pipeline.make_client_from_env(provider="fixture", api_key_envs=["FIXTURE_KEY"],
                base_url_env="FIXTURE_URL", default_base_url="https://example.invalid", model_env="FIXTURE_MODEL",
                default_model="fixture-model", args=args)
        self.assertIsNone(client)
        self.assertEqual((url, model), ("https://example.invalid", "fixture-model"))

    def test_parse_json_object_handles_object_and_fenced_response(self) -> None:
        expected = {"keep_command_ids": ["a"], "reason": "needed"}
        self.assertEqual(parse_json_object(json.dumps(expected)), expected)
        self.assertEqual(parse_json_object("```json\n" + json.dumps(expected) + "\n```"), expected)

    def test_parse_json_object_rejects_array_and_non_json(self) -> None:
        with self.assertRaises(TypeError):
            parse_json_object('["a"]')
        with self.assertRaises(json.JSONDecodeError):
            parse_json_object("not JSON")


if __name__ == "__main__":
    unittest.main()
