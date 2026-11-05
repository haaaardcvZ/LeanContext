from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from evaluation.error_types import classify
from evaluation.lean_replacement import replace_original_declaration
from evaluation.metrics import pass_at_k, summarize
from evaluation.proofs import Candidate, decode_proof_tokens, load_candidates, precheck_proof, replacement_declaration


class EvaluationTests(unittest.TestCase):
    def test_pass_at_k(self) -> None:
        self.assertAlmostEqual(pass_at_k(32, 8, 1), 0.25)
        self.assertEqual(pass_at_k(32, 1, 32), 1.0)
        self.assertEqual(pass_at_k(32, 0, 16), 0.0)

    def test_formal_32_plus_greedy_protocol(self) -> None:
        index = {
            "a": {"sample_id": "a", "proof_family": "Strict Term"},
            "b": {"sample_id": "b", "proof_family": "Ordinary Tactic"},
        }
        results = []
        for sample_id in index:
            for candidate_id in range(32):
                results.append(
                    {
                        "sample_id": sample_id,
                        "mode": "sampled",
                        "candidate_id": candidate_id,
                        "passed": sample_id == "a" and candidate_id == 0,
                        "proof_hash": f"{sample_id}-{candidate_id}",
                        "error_type": "success" if sample_id == "a" and candidate_id == 0 else "unsolved_goals",
                    }
                )
            results.append(
                {
                    "sample_id": sample_id,
                    "mode": "greedy_temperature_0",
                    "candidate_id": 0,
                    "passed": sample_id == "a",
                    "proof_hash": f"{sample_id}-greedy",
                    "error_type": "success" if sample_id == "a" else "unsolved_goals",
                }
            )
        report = summarize(results, index)
        self.assertAlmostEqual(report["overall"]["pass_at_k"]["1"]["mean"], 1 / 64)
        self.assertEqual(report["overall"]["pass_at_k"]["32"]["mean"], 0.5)
        self.assertEqual(report["overall"]["greedy_temperature_0"]["mean"], 0.5)

    def test_error_taxonomy_is_ordered(self) -> None:
        result = classify(
            passed=False,
            stderr="file.lean:4:2: error: unsolved goals\n⊢ True",
        )
        self.assertEqual(result["error_type"], "unsolved_goals")
        timeout = classify(passed=False, timed_out=True)
        self.assertEqual(timeout["error_type"], "lean_timeout")

    def test_precheck(self) -> None:
        self.assertEqual(precheck_proof(""), "empty_generation")
        self.assertEqual(precheck_proof("by sorry"), "placeholder_sorry_or_admit")
        self.assertEqual(
            precheck_proof("by trivial\ntheorem injected : True := by trivial"),
            "top_level_command_injection",
        )

    def test_non_cot_reasoning_checked_before_fence_extraction(self) -> None:
        row = {"sample_id": "a", "mode": "sampled", "candidate_id": 0,
               "proof": "<think>reason</think>\n```lean\nby trivial\n```",
               "forbid_reasoning_markers": ["<think>", "</think>"]}
        candidate = load_candidates([row])[0]
        self.assertEqual(candidate.proof, "by trivial")
        self.assertEqual(candidate.protocol_error, "forbidden_reasoning_marker")
        self.assertEqual(classify(passed=False, precheck=candidate.protocol_error)["paper_error_type"], "extraction")

    def test_decode_keeps_special_reasoning_markers(self) -> None:
        class Tokenizer:
            bos_token_id, pad_token_id, eos_token_id = 1, 0, 2
            def decode(self, ids, skip_special_tokens):
                self_seen.append((ids, skip_special_tokens))
                return "<think>reason</think>"
        self_seen = []
        self.assertEqual(decode_proof_tokens(Tokenizer(), [1, 7, 8, 2, 0]), "<think>reason</think>")
        self.assertEqual(self_seen, [([7, 8], False)])

    def test_parser_errors_precede_timeout_name_and_instance_errors(self) -> None:
        result = classify(passed=False, timed_out=True,
                          stderr="unexpected token; unknown identifier; failed to synthesize")
        self.assertEqual(result["error_type"], "syntax_or_parser_error")
        self.assertEqual(result["paper_error_type"], "extraction")
        infrastructure = classify(passed=False, timed_out=True,
                                  stderr="unknown module prefix Mathlib; unexpected token")
        self.assertEqual(infrastructure["paper_error_type"], "infrastructure")

    def test_exact_original_declaration_replacement(self) -> None:
        source = b"namespace N\n\ntheorem old : True := by trivial\n\n#check old\nend N\n"
        original = b"theorem old : True := by trivial"
        start = source.index(original)
        end = start + len(original)
        index = {
            "sample_id": "abc",
            "statement": "theorem anonymous_target : True :=",
            "kind": "theorem",
            "original_name": "old",
            "relpath": "Demo.lean",
            "start_byte": start,
            "end_byte": end,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "original_command_sha256": hashlib.sha256(original).hexdigest(),
        }
        candidate = Candidate("abc", "sampled", 0, "by\n  exact True.intro", "h", {})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Demo.lean").write_bytes(source)
            path = replace_original_declaration(
                index_row=index,
                candidate=candidate,
                mathlib_root=root,
                task_root=root / "tasks",
            )
            generated = path.read_text(encoding="utf-8")
        self.assertIn("theorem old : True :=\nby\n  exact True.intro", generated)
        self.assertNotIn("theorem old : True := by trivial", generated)
        self.assertIn("#check old", generated)

    def test_example_stays_anonymous(self) -> None:
        index = {
            "statement": "example : True :=",
            "kind": "example",
            "original_name": "",
        }
        self.assertEqual(
            replacement_declaration(index, "by trivial"),
            "example : True :=\nby trivial",
        )


if __name__ == "__main__":
    unittest.main()
