import unittest

from .replacement_scope import match_reason, match_scope, scope_profiles


def command(cid, kind, text, name=None):
    return {"command_id": cid, "kind": kind, "name": name, "text": text, "source_file": "Mathlib/Fixture.lean"}


class ReplacementScopeTests(unittest.TestCase):
    def test_adjacent_unrelated_declarations_ignore_names_and_bodies(self):
        p = scope_profiles([command("a", "def", "def apples : Nat := 1"), command("b", "def", "def oranges : String := \"unrelated\"")])
        self.assertTrue(match_scope(p["a"], p["b"])["compatible"])
        self.assertNotIn("apples", str(p["a"]))
        self.assertFalse(p["a"]["Lean_elaboration_checked"])

    def test_named_noncomputable_section_and_end_restore_variables(self):
        p = scope_profiles([
            command("a", "def", "def a := 1"),
            command("s", "section", "noncomputable section Named", "Named"),
            command("v", "variable", "variable (n : Nat)"),
            command("b", "def", "def b := n"),
            command("c", "def", "def c := n + 1"),
            command("e", "end", "end Named", "Named"),
            command("d", "def", "def d := 2"),
        ])
        self.assertTrue(match_scope(p["b"], p["c"])["compatible"])
        self.assertTrue(match_scope(p["a"], p["d"])["compatible"])
        self.assertFalse(match_scope(p["a"], p["b"])["compatible"])
        self.assertTrue(p["b"]["section_stack"][0]["noncomputable"])
        self.assertEqual([item["command_id"] for item in p["b"]["active_directives"]], ["v"])

    def test_namespace_paths_and_nested_end(self):
        p = scope_profiles([command("n", "namespace", "namespace A.B", "A.B"), command("a", "def", "def a := 1"), command("e", "end", "end A.B", "A.B"), command("b", "def", "def b := 1")])
        self.assertEqual(p["a"]["namespace_path"], ["A", "B"])
        self.assertEqual(p["b"]["namespace_path"], [])
        self.assertFalse(match_scope(p["a"], p["b"])["compatible"])

    def test_isolated_open_wrapper_does_not_leak(self):
        p = scope_profiles([command("a", "def", "def a := 1"), command("w", "open", "open Nat in\ndef wrapped := 2"), command("b", "def", "def b := 3"), command("o", "open", "open Nat"), command("c", "def", "def c := 4")])
        self.assertTrue(match_scope(p["a"], p["b"])["compatible"])
        self.assertFalse(match_scope(p["a"], p["w"])["compatible"])
        self.assertEqual(p["w"]["command_wrappers"], ["open Nat"])
        self.assertFalse(match_scope(p["a"], p["c"])["compatible"])
        self.assertEqual([item["command_id"] for item in p["c"]["active_directives"]], ["o"])

    def test_variable_and_nested_option_wrappers_are_isolated(self):
        p = scope_profiles([command("a", "def", "def a := 1"), command("w", "variable", "variable (R A : Type) in\nset_option autoImplicit false in\ndef wrapped := 1"), command("b", "def", "def b := 2")])
        self.assertEqual(p["w"]["command_wrappers"], ["variable (R A : Type)", "set_option autoImplicit false"])
        self.assertFalse(match_scope(p["w"], p["b"])["compatible"])
        self.assertTrue(match_scope(p["a"], p["b"])["compatible"])

    def test_persistent_bindings_and_options_change_profiles(self):
        for kind, text in [("universe", "universe u"), ("variable", "variable (n : Nat)"), ("include", "include n"), ("omit", "omit n"), ("open_scoped", "open scoped BigOperators"), ("set_option", "set_option autoImplicit false"), ("attribute", "attribute [simp] Nat.add_zero")]:
            with self.subTest(kind=kind):
                p = scope_profiles([command("a", "def", "def a := 1"), command("effect", kind, text), command("b", "def", "def b := 2")])
                self.assertFalse(match_scope(p["a"], p["b"])["compatible"])

    def test_local_notation_restored_but_global_notation_persists(self):
        for kind, text, equal in [("local_notation", 'local notation "foo" => Nat', True), ("notation", 'notation "foo" => Nat', False), ("scoped_notation", 'scoped notation "foo" => Nat', False)]:
            with self.subTest(kind=kind):
                p = scope_profiles([command("a", "def", "def a := 1"), command("s", "section", "section"), command("n", kind, text), command("b", "def", "def b := 1"), command("e", "end", "end"), command("c", "def", "def c := 1")])
                self.assertEqual(match_scope(p["a"], p["c"])["compatible"], equal)

    def test_successive_instances_same_bindings_can_match(self):
        p = scope_profiles([command("i", "instance", "instance secretInstance : Inhabited Nat := ⟨0⟩"), command("j", "instance", "instance otherInstance : Inhabited Int := ⟨1⟩")])
        self.assertTrue(match_scope(p["i"], p["j"])["compatible"])
        self.assertNotIn("secretInstance", str(p["j"]))
        self.assertEqual(p["j"]["active_directives"], [])
        self.assertFalse(p["j"]["instance_set_equivalence_checked"])

    def test_wrapped_instance_does_not_leak_MC_or_open_into_EC(self):
        p = scope_profiles([command("i", "open", "open Nat in\ninstance hidden : Inhabited Nat := ⟨0⟩"), command("a", "def", "def a := 1")])
        self.assertEqual(p["i"]["command_wrappers"], ["open Nat"])
        self.assertEqual(p["a"]["command_wrappers"], [])
        self.assertEqual(p["a"]["active_directives"], [])
        self.assertNotIn("hidden", str(p["a"]))

    def test_local_instance_is_current_command_modifier_not_MC_history(self):
        p = scope_profiles([command("a", "def", "def a := 1"), command("s", "section", "section"), command("i", "instance", "local instance hidden : Inhabited Nat := ⟨0⟩"), command("b", "def", "def b := 2"), command("e", "end", "end"), command("c", "def", "def c := 3")])
        self.assertEqual(p["i"]["command_modifiers"], ["local"])
        self.assertEqual(p["b"]["active_directives"], [])
        self.assertTrue(match_scope(p["a"], p["c"])["compatible"])

    def test_intervening_attributed_theorem_not_part_of_EC_history(self):
        p = scope_profiles([command("a", "def", "def a := 1"), command("t", "theorem", "@[simp] theorem secret : True := by trivial"), command("b", "def", "def b := 2")])
        self.assertTrue(match_scope(p["a"], p["b"])["compatible"])
        self.assertEqual(p["b"]["active_directives"], [])

    def test_known_example_does_not_poison_later_profile(self):
        p = scope_profiles([command("a", "def", "def a := 1"), command("x", "example", "example : True := by trivial"), command("b", "def", "def b := 2")])
        self.assertEqual(p["x"]["status"], "syntactic_valid")
        self.assertTrue(match_scope(p["a"], p["b"])["compatible"])

    def test_attributes_and_modifiers_are_not_declaration_names(self):
        p = scope_profiles([command("a", "def", "@[simp] noncomputable def original := 1"), command("b", "def", "def candidate := 2")])
        self.assertEqual(p["a"]["command_attributes"], ["@[simp]"])
        self.assertEqual(p["a"]["command_modifiers"], ["noncomputable"])
        self.assertNotIn("original", str(p["a"]))
        self.assertFalse(match_scope(p["a"], p["b"])["compatible"])

    def test_malformed_end_unavailable(self):
        for commands in [[command("e", "end", "end Missing"), command("a", "def", "def a := 1")], [command("s", "section", "section Named", "Named"), command("e", "end", "end Other"), command("a", "def", "def a := 1")]]:
            with self.subTest(commands=commands):
                p = scope_profiles(commands)
                self.assertEqual(p["a"]["status"], "malformed")
                self.assertIsNone(p["a"]["fingerprint"])
                self.assertFalse(match_scope(p["a"], p["a"])["compatible"])

    def test_Lean_valid_scopes_may_close_implicitly_at_EOF(self):
        for kind, text, name in [("section", "noncomputable section", None), ("section", "section Named", "Named"), ("namespace", "namespace Named", "Named")]:
            with self.subTest(text=text):
                p = scope_profiles([command("s", kind, text, name), command("a", "def", "def a := 1"), command("b", "def", "def b := 2")])
                self.assertEqual(p["a"]["status"], "syntactic_valid")
                self.assertIsNotNone(p["a"]["fingerprint"])
                self.assertTrue(match_scope(p["a"], p["b"])["compatible"])
                self.assertEqual(p["b"]["scope_stack"][0]["command_id"], "s")

    def test_unknown_effects_are_opaque_not_assumed_safe(self):
        p = scope_profiles([command("x", "custom_extension", "do_something_unknown"), command("a", "def", "def a := 1")])
        self.assertEqual(p["a"]["status"], "opaque")
        self.assertIn("unavailable", match_reason(p["a"], p["a"]))

    def test_bad_declaration_header_and_parser_scope_name(self):
        self.assertEqual(scope_profiles([command("a", "def", "not_a_definition")])["a"]["status"], "opaque")
        self.assertEqual(scope_profiles([command("s", "section", "section Real", "Wrong")])["s"]["status"], "malformed")

    def test_duplicate_ids_and_unordered_input_rejected(self):
        with self.assertRaises(ValueError):
            scope_profiles([command("a", "def", "def a := 1"), command("a", "def", "def b := 2")])
        with self.assertRaises(ValueError):
            scope_profiles([{**command("a", "def", "def a := 1"), "source_order": 2}, {**command("b", "def", "def b := 2"), "source_order": 1}])


if __name__ == "__main__":
    unittest.main()
