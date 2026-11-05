"""Small, entirely temporary fixtures for the strict provenance joins."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from dataset_preparation.provenance import (
    SourceRepository, command_record, enrich_target, iter_jsonl,
    load_private_rows, load_public_rows, load_selections,
)


class ProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.relpath = "Mathlib/Fixture.lean"
        self.source = (
            "variable (n : Nat)\n\n"
            "variable (n : Nat)\n\n"
            "open Classical in\nnoncomputable def café : Nat := 1\n\n"
            "theorem original : True := by trivial\n"
        )
        self.parts = (
            "variable (n : Nat)",
            "variable (n : Nat)",
            "open Classical in\nnoncomputable def café : Nat := 1",
            "theorem original : True := by trivial",
        )
        self.commands = []
        cursor = 0
        encoded = self.source.encode("utf-8")
        for index, text in enumerate(self.parts, 1):
            raw = text.encode("utf-8")
            start = encoded.index(raw, cursor)
            end = start + len(raw)
            cursor = end
            start_line = encoded[:start].count(b"\n") + 1
            end_line = encoded[:end].count(b"\n") + 1
            last_newline = encoded[:end].rfind(b"\n")
            end_col = len(encoded[last_newline + 1:end].decode("utf-8"))
            self.commands.append({
                "command_id": f"cmd_{index:06d}",
                "kind": ("variable", "variable", "open", "theorem")[index - 1],
                "syntax_kind": "Fixture.syntax",
                "name": (None, None, "Classical", "original")[index - 1],
                "start_byte": start, "end_byte": end,
                "start_line": start_line, "start_col": 0,
                "end_line": end_line, "end_col": end_col,
                "raw_text": text, "effective_raw_text": text + "\n",
                # Deliberately unsuitable for original-byte splicing.
                "effective_end_byte": start + 1,
                "source_file": self.relpath,
                "source_module": "Mathlib.Fixture",
            })
        source_path = self.root / "mathlib" / self.relpath
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(encoded)
        self.command_path = self.root / "commands" / "Mathlib/Fixture.jsonl"
        self.command_path.parent.mkdir(parents=True)
        self.write_rows(self.command_path, self.commands)
        self.repo = SourceRepository(self.root, "commands", "mathlib")
        self.context = [
            {"position": 0, "kind": "variable", "text": self.parts[1]},
            {"position": 1, "kind": "def", "text": self.parts[2]},
        ]
        self.public = {
            "sample_id": "sample", "split": "validation", "proof_family": "Tactic",
            "statement": "theorem anonymous_target : True :=", "target": "by trivial",
            "mathematical_context": [self.context[1]],
            "environment_context": [self.context[0]],
            "full_context": self.context,
        }
        self.context_text = "\n\n".join(item["text"] for item in self.context)
        self.private = {
            "sample_id": "sample", "split": "train", "relpath": self.relpath,
            "module": "Mathlib.Fixture", "kind": "theorem", "full_name": "original",
            "statement": self.public["statement"], "target": self.public["target"],
            "prompt": self.context_text + "\n\n" + self.public["statement"] + "\n",
            "metadata": {
                "target_command_id": "cmd_000004", "proof_family": "Tactic",
                "target_start_line": self.commands[3]["start_line"],
                "target_end_line": self.commands[3]["end_line"],
                "mathlib4_version": "fixture",
            },
        }
        self.raw = {
            "lean_file_name": self.relpath, "target_command_id": "cmd_000004",
            # The original selection list need not already be source sorted.
            "kept_command_ids": ["cmd_000003", "cmd_000002"],
            "candidate_command_ids": ["cmd_000001", "cmd_000002", "cmd_000003"],
            "context_text": self.context_text + "\n", "selection_provider": "fixture",
            "selection_reason": "fixture reason", "target_body_text": "DO NOT COPY",
        }
        self.raw_path = self.root / "raw.jsonl"
        self.write_rows(self.raw_path, [self.raw])
        self.selection = load_selections(self.raw_path, {"sample": self.private})["sample"]

    @staticmethod
    def write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_exact_ids_resolve_duplicate_text_and_preserve_current_split(self) -> None:
        untouched = copy.deepcopy(self.public)
        enriched, mapping, history = enrich_target(self.public, self.private, self.selection, self.repo)
        self.assertEqual(self.public, untouched)
        self.assertEqual(enriched["split"], "validation")
        self.assertEqual(enriched["target_id"], "sample")
        self.assertEqual(enriched["original_name"], "original")
        self.assertEqual(enriched["target_command_uid"], self.relpath + "::cmd_000004")
        self.assertEqual([i["command_id"] for i in enriched["full_context"]], ["cmd_000002", "cmd_000003"])
        self.assertEqual([i["source_order"] for i in enriched["full_context"]], [1, 2])
        self.assertEqual(enriched["full_context"][1]["parser_kind"], "open")
        self.assertEqual(enriched["full_context"][1]["effective_kind"], "def")
        self.assertEqual(enriched["target_source_span"]["end_byte"], self.commands[3]["end_byte"])
        self.assertEqual(history["kept_command_ids"], ["cmd_000003", "cmd_000002"])
        self.assertEqual(history["ordered_kept_command_ids"], ["cmd_000002", "cmd_000003"])
        self.assertEqual(history["original_prepared_split"], "train")
        serialized = json.dumps([mapping, history])
        self.assertNotIn("by trivial", serialized)
        self.assertNotIn("DO NOT COPY", serialized)
        self.assertNotIn("original_context_text", history)
        self.assertEqual(len(mapping["target_raw_text_sha256"]), 64)

    def test_command_record_has_hashes_but_no_fabricated_declaration(self) -> None:
        record = command_record(self.commands[2])
        self.assertEqual(record["original_name"], "Classical")
        self.assertEqual(record["parser_kind"], "open")
        self.assertEqual(record["effective_kind"], "def")
        self.assertIn("not_elaborated", record["name_identity"])
        self.assertNotIn("raw_text", record)
        self.assertNotIn("effective_raw_text", record)
        self.assertEqual(len(record["raw_text_sha256"]), 64)

    def test_public_and_private_loaders_use_sample_id_not_old_split(self) -> None:
        public_dir = self.root / "public"
        private_dir = self.root / "private"
        public_dir.mkdir()
        private_dir.mkdir()
        for split in ("train", "validation", "test"):
            self.write_rows(public_dir / f"{split}.jsonl", [self.public] if split == "validation" else [])
        for split in ("train", "val", "test"):
            self.write_rows(private_dir / f"{split}.jsonl", [self.private] if split == "train" else [])
        public = load_public_rows(public_dir)
        private = load_private_rows(private_dir, set(public))
        self.assertEqual(public["sample"]["split"], "validation")
        self.assertEqual(private["sample"]["split"], "train")

    def test_nonmatching_raw_version_is_ignored(self) -> None:
        wrong = dict(self.raw, context_text="different context")
        self.write_rows(self.raw_path, [wrong, self.raw])
        selections = load_selections(self.raw_path, {"sample": self.private})
        self.assertEqual(selections["sample"]["raw_selection_row"], 2)

    def test_duplicate_exact_raw_selection_is_rejected(self) -> None:
        self.write_rows(self.raw_path, [self.raw, self.raw])
        with self.assertRaisesRegex(ValueError, "duplicate exact raw selection"):
            load_selections(self.raw_path, {"sample": self.private})

    def test_missing_exact_raw_selection_is_rejected(self) -> None:
        self.write_rows(self.raw_path, [dict(self.raw, context_text="wrong")])
        with self.assertRaisesRegex(ValueError, "missing exact raw selections"):
            load_selections(self.raw_path, {"sample": self.private})

    def test_duplicate_kept_id_is_rejected(self) -> None:
        selection = dict(self.selection, kept_command_ids=["cmd_000002", "cmd_000002"])
        with self.assertRaisesRegex(ValueError, "duplicate command ID"):
            enrich_target(self.public, self.private, selection, self.repo)

    def test_later_selected_command_is_rejected(self) -> None:
        selection = dict(self.selection, kept_command_ids=["cmd_000004"], candidate_command_ids=["cmd_000004"])
        with self.assertRaisesRegex(ValueError, "strictly before target"):
            enrich_target(self.public, self.private, selection, self.repo)

    def test_exact_context_text_mismatch_is_rejected(self) -> None:
        public = copy.deepcopy(self.public)
        public["full_context"][0]["text"] += " "
        with self.assertRaisesRegex(ValueError, "exact context text mismatch"):
            enrich_target(public, self.private, self.selection, self.repo)

    def test_kind_or_category_mismatch_is_rejected(self) -> None:
        public = copy.deepcopy(self.public)
        public["full_context"][1]["kind"] = "theorem"
        with self.assertRaisesRegex(ValueError, "effective context kind mismatch"):
            enrich_target(public, self.private, self.selection, self.repo)

    def test_original_source_mismatch_is_rejected(self) -> None:
        altered = copy.deepcopy(self.commands)
        altered[2]["raw_text"] += "wrong"
        self.write_rows(self.command_path, altered)
        with self.assertRaisesRegex(ValueError, "original source span mismatch"):
            self.repo.commands(self.relpath)

    def test_duplicate_source_local_command_id_is_rejected(self) -> None:
        self.write_rows(self.command_path, self.commands + [self.commands[-1]])
        with self.assertRaisesRegex(ValueError, "duplicate command_id"):
            self.repo.commands(self.relpath)

    def test_noncanonical_source_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid source relpath"):
            self.repo.commands("../Mathlib/Fixture.lean")

    def test_public_private_proof_mismatch_is_rejected(self) -> None:
        private = dict(self.private, target="another proof")
        with self.assertRaisesRegex(ValueError, "public/private target mismatch"):
            enrich_target(self.public, private, self.selection, self.repo)

    def test_private_source_range_mismatch_is_rejected(self) -> None:
        private = copy.deepcopy(self.private)
        private["metadata"]["source_range"] = {"start": {"line": 999}}
        with self.assertRaisesRegex(ValueError, "private source_range line mismatch"):
            enrich_target(self.public, private, self.selection, self.repo)

    def test_history_ignores_unexpected_target_proof_fields(self) -> None:
        selection = dict(self.selection, target_body_text="DO NOT COPY", target="DO NOT COPY")
        _, mapping, history = enrich_target(self.public, self.private, selection, self.repo)
        self.assertNotIn("DO NOT COPY", json.dumps([mapping, history]))

    def test_exact_prefilter_and_removed_history_is_preserved(self) -> None:
        raw = dict(
            self.raw,
            kept_command_ids_before_context_filter=["cmd_000001", "cmd_000003", "cmd_000002"],
            removed_context_command_ids=["cmd_000001"],
            context_filter_method="remove_kept_lemma_theorem_not_named_in_target_statement",
            selection_method="parser_candidates_plus_llm_minimized",
            verified_by_lean=False,
        )
        self.write_rows(self.raw_path, [raw])
        selection = load_selections(self.raw_path, {"sample": self.private})["sample"]
        _, _, history = enrich_target(self.public, self.private, selection, self.repo)
        for field in (
            "kept_command_ids_before_context_filter",
            "removed_context_command_ids", "context_filter_method",
        ):
            self.assertEqual(selection[field], raw[field])
            self.assertEqual(history[field], raw[field])
            self.assertEqual(history["context_filter_history_availability"][field], "present")
        self.assertIs(selection["reference_assisted"], True)
        self.assertIs(history["reference_assisted"], True)
        self.assertIn("truncation", history["reference_assisted_method"])
        # Assistance and actual proof verification are separate facts.
        self.assertIs(history["verified_by_lean"], False)

    def test_missing_null_and_empty_filter_history_remain_distinct(self) -> None:
        raw = dict(self.raw, removed_context_command_ids=None, context_filter_method="")
        self.write_rows(self.raw_path, [raw])
        selection = load_selections(self.raw_path, {"sample": self.private})["sample"]
        _, _, history = enrich_target(self.public, self.private, selection, self.repo)
        self.assertIsNone(history["kept_command_ids_before_context_filter"])
        self.assertIsNone(history["removed_context_command_ids"])
        self.assertEqual(history["context_filter_method"], "")
        self.assertEqual(history["context_filter_history_availability"], {
            "kept_command_ids_before_context_filter": "missing",
            "removed_context_command_ids": "null",
            "context_filter_method": "present",
        })
        self.assertIsNone(history["reference_assisted"])
        self.assertEqual(history["reference_assisted_availability"], "unknown")
        self.assertNotIn("verified_by_lean", history)
        raw["removed_context_command_ids"] = []
        self.write_rows(self.raw_path, [raw])
        selection = load_selections(self.raw_path, {"sample": self.private})["sample"]
        self.assertEqual(selection["removed_context_command_ids"], [])
        self.assertEqual(selection["context_filter_history_availability"]["removed_context_command_ids"], "present")

    def test_explicit_reference_assistance_is_not_overwritten_by_lean_status(self) -> None:
        raw = dict(self.raw, reference_assisted=False, reference_assisted_method="statement_only", verified_by_lean=True)
        self.write_rows(self.raw_path, [raw])
        selection = load_selections(self.raw_path, {"sample": self.private})["sample"]
        _, _, history = enrich_target(self.public, self.private, selection, self.repo)
        self.assertIs(history["reference_assisted"], False)
        self.assertEqual(history["reference_assisted_method"], "statement_only")
        self.assertIs(history["verified_by_lean"], True)

    def test_invalid_jsonl_row_is_rejected(self) -> None:
        path = self.root / "bad.jsonl"
        path.write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON row is not an object"):
            list(iter_jsonl(path))


if __name__ == "__main__":
    unittest.main()
