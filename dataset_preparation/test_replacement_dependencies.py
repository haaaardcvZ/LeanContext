from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .replacement_dependencies import (
    candidate_exclusion_reason, dependency_evidence, load_evidence,
)


def dependency(commands):
    return {"command_ids": list(commands), "constant_evidence": {
        cid: [{"constant": name, "source_attribution": "kernel_selection_range", "declaration_kind": "definition"}]
        for cid, name in commands.items()}, "unmapped_same_file_constants": [],
        "same_target_command_constants": [], "nonpreceding_same_file_constants": [], "other_local_commands": []}


def semantic():
    return {"sample_id": "s", "source_file": "Mathlib/F.lean", "source_module": "Mathlib.F",
            "target_command_id": "t", "target_name": "N.target",
            "target_identity_status": "kernel_range_and_complete_source_namespace",
            "reference_proof_exact_raw_body_match": True, "mc_semantic_status": "complete",
            "type_dependency": dependency({"a": "N.Type"}),
            "value_dependency": dependency({"a": "N.Type", "b": "N.used"}),
            "proof_extra_dependency": dependency({"b": "N.used"}),
            "projection_augmented_proof_extra_dependency": dependency({"b": "N.used"}),
            "outer_type_constants": ["N.Type", "Imported"],
            "outer_value_constants": ["N.Type", "N.used"],
            "type_constants": ["N.Type", "Imported"], "value_constants": ["N.Type", "N.used"],
            "value_minus_type_constants": ["N.used"],
            "expanded_type_auxiliaries": [], "expanded_value_auxiliaries": []}


def candidate(cid="safe", name="N.safe"):
    return {"command_id": cid, "introduced_declarations_status": "kernel_range_or_certified_ownership_attributed",
            "introduced_declarations": [{"name": name, "basis": "kernel_selection_range"}]}


def kernel(name, type_names=(), value_names=(), **kwargs):
    return {"name": name, "module": "Mathlib.F", "kind": "theorem", "has_value": True,
            "type_constants": list(type_names), "value_constants": list(value_names),
            "type_projections": [], "value_projections": [], **kwargs}


class ProofTextTrap(dict):
    def get(self, key, default=None):
        if key in {"target", "reference_proof", "proof", "gold", "statement"}:
            raise AssertionError(f"forbidden text access: {key}")
        return super().get(key, default)


class ReplacementDependencyTests(unittest.TestCase):
    def test_union_does_not_subtract_types(self):
        e = dependency_evidence(ProofTextTrap(semantic()))
        self.assertTrue(e["availability_complete"])
        self.assertEqual(e["exclusion_command_ids"], ["a", "b", "t"])
        self.assertIn("N.Type", e["exclusion_constant_names"])
        self.assertIn("N.used", e["exclusion_constant_names"])
        self.assertIn("Imported", e["exclusion_constant_names"])
        self.assertFalse(e["reference_proof_text_read"])
        json.dumps(e)

    def test_extra_only_complete_does_not_certify_type_or_value(self):
        for field in ("type_dependency", "value_dependency"):
            q = semantic()
            q[field]["unmapped_same_file_constants"] = ["N.hidden"]
            e = dependency_evidence(q)
            self.assertFalse(e["availability_complete"])
            self.assertIn(f"{field}_unmapped_same_file_constants", e["rejection_reasons"])

    def test_exact_reference_match_is_required(self):
        for value in (None, False):
            q = semantic(); q["reference_proof_exact_raw_body_match"] = value
            self.assertFalse(dependency_evidence(q)["availability_complete"])

    def test_unresolved_target_and_nonpreceding_are_not_empty(self):
        q = semantic(); q["target_identity_status"] = "unresolved_target_identity"
        self.assertFalse(dependency_evidence(q)["availability_complete"])
        q = semantic(); q["value_dependency"]["nonpreceding_same_file_constants"] = ["N.future"]
        self.assertFalse(dependency_evidence(q)["availability_complete"])

    def test_missing_value_metadata_rejects(self):
        q = semantic(); del q["value_dependency"]
        self.assertFalse(dependency_evidence(q)["availability_complete"])

    def test_certified_auxiliary_expansion_is_not_rejected(self):
        q = semantic()
        q["expanded_value_auxiliaries"] = [{"name": "N.target.aux", "source_attribution": "kernel_selection_range"}]
        q["value_dependency"]["same_target_command_constants"] = ["N.target.aux"]
        e = dependency_evidence(q)
        self.assertTrue(e["availability_complete"])
        self.assertIn("N.target.aux", e["exclusion_constant_names"])

    def test_unexpanded_self_auxiliary_is_incomplete(self):
        q = semantic(); q["value_dependency"]["same_target_command_constants"] = ["N.target.unknown"]
        self.assertFalse(dependency_evidence(q)["availability_complete"])

    def test_other_local_commands_are_still_excluded(self):
        q = semantic(); q["type_dependency"]["other_local_commands"] = [{"name": "N.macro", "command_id": "m", "kind": "attribute"}]
        e = dependency_evidence(q)
        self.assertTrue(e["availability_complete"])
        self.assertIn("m", e["exclusion_command_ids"])
        self.assertIn("N.macro", e["exclusion_constant_names"])

    def test_projection_augmented_dependencies_are_excluded(self):
        q = semantic(); q["projection_augmented_proof_extra_dependency"] = dependency({"p": "N.projection"})
        e = dependency_evidence(q)
        self.assertIn("p", e["exclusion_command_ids"])
        self.assertIn("N.projection", e["exclusion_constant_names"])

    def test_candidate_rejects_ids_names_and_uncertain_identity(self):
        e = dependency_evidence(semantic())
        self.assertIsNone(candidate_exclusion_reason(candidate(), e))
        self.assertIsNotNone(candidate_exclusion_reason(candidate("a"), e))
        self.assertIsNotNone(candidate_exclusion_reason(candidate(name="N.used"), e))
        c = candidate(); c["introduced_declarations_status"] = "not_resolved_not_asserted_empty"
        self.assertEqual(candidate_exclusion_reason(c, e), "candidate_introduced_declarations_unresolved")
        c = candidate(); c["introduced_declarations"] = []
        self.assertIsNotNone(candidate_exclusion_reason(c, e))
        c = candidate(); c["introduced_declarations"][0]["basis"] = "namespace_guess"
        self.assertIsNotNone(candidate_exclusion_reason(c, e))
        q = deepcopy(e); q["availability_complete"] = False
        self.assertEqual(candidate_exclusion_reason(candidate(), q), "target_dependency_evidence_incomplete")

    def load_fixture(self, q, declarations, wanted=None, owners=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {"semantic_mc_labels.jsonl": [q], "declaration_dependencies.jsonl": declarations,
                        "declaration_ownership.jsonl": owners or []}
            hashes = {}
            for name, values in contents.items():
                data = b"".join((json.dumps(value)+"\n").encode() for value in values)
                (root / name).write_bytes(data)
                hashes[name] = hashlib.sha256(data).hexdigest()
            return load_evidence(root, wanted or {"s"}), hashes

    def target_dump(self, q):
        return [kernel("N.target", q["outer_type_constants"], q["outer_value_constants"]),
                kernel("N.Type"), kernel("N.used")]

    def test_loader_checks_real_dump_and_actual_byte_hashes(self):
        q = semantic(); result, hashes = self.load_fixture(q, self.target_dump(q), {"s", "missing"})
        self.assertTrue(result["s"]["availability_complete"])
        self.assertTrue(result["s"]["kernel_dump_metadata_verified"])
        self.assertFalse(result["missing"]["availability_complete"])
        self.assertEqual(result["s"]["provenance"]["semantic_record_index_1_based"], 1)
        for name, expected in hashes.items():
            self.assertEqual(result["s"]["provenance"]["sources"][name]["sha256"], expected)

    def test_loader_rejects_different_target_kernel_constants(self):
        q = semantic(); dump = self.target_dump(q); dump[0]["value_constants"] = ["Wrong"]
        result, _ = self.load_fixture(q, dump)
        self.assertFalse(result["s"]["availability_complete"])
        self.assertIn("outer_value_constants_differ_from_actual_kernel_dump", result["s"]["rejection_reasons"])

    def test_loader_validates_nested_recorded_auxiliary_expansion(self):
        q = semantic()
        q["outer_value_constants"] = ["N.target.aux"]
        q["value_constants"] = ["N.target.aux", "N.target.inner", "N.used"]
        q["value_minus_type_constants"] = list(q["value_constants"])
        q["value_dependency"] = dependency({"b": "N.used"})
        q["expanded_value_auxiliaries"] = [{"name": name, "source_attribution": "kernel_selection_range"}
                                             for name in ("N.target.aux", "N.target.inner")]
        dump = self.target_dump(q) + [kernel("N.target.aux", value_names=["N.target.inner"]),
                                    kernel("N.target.inner", value_names=["N.used"])]
        result, _ = self.load_fixture(q, dump)
        self.assertTrue(result["s"]["availability_complete"])
        self.assertIn("N.target.inner", result["s"]["exclusion_constant_names"])

    def test_loader_checks_projection_names_without_inventing_constants(self):
        q = semantic(); dump = self.target_dump(q)
        dump[0]["value_projections"] = [{"structure": "N.Struct", "index": 0, "projection_name": "N.Struct.field"}]
        result, _ = self.load_fixture(q, dump)
        self.assertTrue(result["s"]["availability_complete"])
        self.assertEqual(result["s"]["exclusion_projection_reference_names"], ["N.Struct", "N.Struct.field"])

    def test_loader_requires_actual_ownership_certificate(self):
        q = semantic(); q["type_dependency"]["constant_evidence"]["a"][0]["source_attribution"] = "registered_structure_projection"
        result, _ = self.load_fixture(q, self.target_dump(q))
        self.assertFalse(result["s"]["availability_complete"])
        result, _ = self.load_fixture(q, self.target_dump(q), owners=[{"name": "N.Type", "owners": ["N.Struct"], "basis": "registered_structure_projection"}])
        self.assertTrue(result["s"]["availability_complete"])


if __name__ == "__main__":
    unittest.main()
