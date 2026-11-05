"""Conservative source-environment compatibility for replacement commands.

Profiles describe the source prefix *before* a command and its isolated
``... in`` wrappers. Ordinary declaration names and bodies are not profile
features. This is deterministic syntactic compatibility, not Lean elaboration,
typechecking, dependency equivalence, or a claim that a replacement will work.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from .selector import DECL, IDENT, MC_KINDS, _kind, _mask

PROFILE_VERSION = "conservative-source-environment-v1"
VALID_STATUS = "syntactic_valid"
_MODIFIERS = {"private", "protected", "noncomputable", "unsafe", "partial", "nonrec", "local", "scoped"}
_WRAPPER_HEADS = {"open", "set_option", "variable", "include", "omit", "attribute", "local", "scoped"}
_DIRECTIVES = {
    "variable", "universe", "include", "omit", "open", "open_scoped", "set_option",
    "attribute", "notation", "local_notation", "scoped_notation", "infix", "infixl",
    "infixr", "prefix", "postfix", "syntax", "macro", "macro_rules", "export",
}
_LOCAL_DIRECTIVES = {"variable", "universe", "include", "omit", "open", "open_scoped", "set_option"}
_DOC_KINDS = {"moduleDoc", "addDocString", "adaptationNoteCmd", "library_note"}


def _text(command: dict[str, Any]) -> str:
    for field in ("text", "effective_raw_text", "raw_text"):
        value = command.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _word(text: str) -> str:
    match = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)\b", text)
    return match.group(1) if match else ""


def _top_level_in(masked: str) -> tuple[int | None, bool]:
    """Do not confuse a binder/string/RHS term's `in` with a wrapper."""
    depth = []
    closing = {")": "(", "]": "[", "}": "{"}
    i = 0
    while i < len(masked):
        ch = masked[i]
        if ch in "([{":
            depth.append(ch)
        elif ch in closing:
            if not depth or depth.pop() != closing[ch]:
                return None, False
        elif not depth:
            if masked.startswith((":=", "=>"), i):
                return None, True
            if masked.startswith("in", i) and (i == 0 or not (masked[i - 1].isalnum() or masked[i - 1] in "_'.")) and (i + 2 == len(masked) or not (masked[i + 2].isalnum() or masked[i + 2] in "_'.")):
                return i, True
        i += 1
    return None, not depth


def _isolated_wrappers(text: str) -> tuple[list[str], str, str | None]:
    wrappers = []
    remaining = text
    while _word(_mask(remaining)) in _WRAPPER_HEADS:
        offset, balanced = _top_level_in(_mask(remaining))
        if not balanced:
            return wrappers, remaining, "malformed_wrapper_delimiters"
        if offset is None:
            break
        prefix = remaining[:offset].strip()
        if not prefix or not remaining[offset + 2:].strip():
            return wrappers, remaining, "malformed_empty_wrapper"
        wrappers.append(prefix)
        remaining = remaining[offset + 2:].strip()
    return wrappers, remaining, None


def _declaration_header(text: str) -> tuple[list[str], list[str], str | None]:
    """Read only attributes/modifiers preceding a recognized declaration."""
    masked = _mask(text)
    match = DECL.search(masked)
    if match is None:
        return [], [], "opaque_declaration_header"
    prefix = text[:match.start("kind")]
    masked_prefix = _mask(prefix)
    attributes = []
    for attribute in re.finditer(r"@\[[^\]]*\]", masked_prefix):
        attributes.append(prefix[attribute.start():attribute.end()])
    without_attributes = re.sub(r"@\[[^\]]*\]", " ", masked_prefix)
    modifiers = without_attributes.split()
    if any(modifier not in _MODIFIERS for modifier in modifiers):
        return [], [], "opaque_declaration_prefix"
    return attributes, modifiers, None


def _name_parts(name: str) -> list[str] | None:
    if not name:
        return []
    if IDENT.fullmatch(name) is None:
        return None
    return re.findall(r"«[^»]+»|[^\W\d][\w']*", name, re.UNICODE)


def _scope_name(command: dict[str, Any], text: str, kind: str) -> tuple[str, bool, str | None]:
    masked = _mask(text)
    match = re.fullmatch(r"\s*(?:(noncomputable)\s+)?" + kind + r"\b\s*([^\n]*)\s*", masked)
    if match is None:
        return "", False, "malformed_scope_command"
    spelling = match.group(2).strip()
    parser_name = command.get("name", command.get("original_name"))
    if isinstance(parser_name, str) and parser_name != spelling:
        return spelling, bool(match.group(1)), "parser_scope_name_disagrees_with_source"
    if _name_parts(spelling) is None or (kind == "namespace" and not spelling):
        return spelling, bool(match.group(1)), "malformed_scope_name"
    if match.group(1) and kind != "section":
        return spelling, True, "unsupported_noncomputable_scope"
    return spelling, bool(match.group(1)), None


def _directive_kind(kind: str, text: str) -> str | None:
    if kind in _DIRECTIVES:
        return kind
    head = _word(_mask(text))
    if kind in {"local", "scoped"} and head == kind:
        tail = re.sub(r"^\s*" + kind + r"\b", "", _mask(text), count=1)
        inner = _word(tail)
        if inner in _DIRECTIVES:
            return inner
    return None


def scope_profiles(commands: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return strict before-command source profiles for one ordered file.

    Malformed/unknown effects make fingerprints unavailable. Scope-local known
    effects are restored on ``end``; global syntax/notation/explicit attribute
    commands remain in the ledger. All MC predecessor names, bodies, identities,
    instance registrations and attached attributes are excluded from the ledger.
    Current-command modifiers/attributes still participate in its profile.
    Consequently this does not assert full instance-set/environment equivalence.
    Lean permits namespace/section scopes to remain open through file EOF;
    these are active prefix scopes, not malformed missing-end errors.
    """
    commands = list(commands)
    command_ids = [command.get("command_id") for command in commands]
    if any(not isinstance(cid, str) or not cid for cid in command_ids) or len(set(command_ids)) != len(command_ids):
        raise ValueError("commands require unique nonempty command_id values")
    files = {command.get("source_file") for command in commands if command.get("source_file")}
    if len(files) > 1:
        raise ValueError("scope_profiles requires exactly one source file")
    source_file = next(iter(files), None)
    orders = [command.get("source_order") for command in commands]
    if all(order is not None for order in orders) and (any(not isinstance(order, int) or isinstance(order, bool) for order in orders) or any(left >= right for left, right in zip(orders, orders[1:]))):
        raise ValueError("commands must be in strict source order")
    root_local: list[dict[str, Any]] = []
    global_effects: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    namespace: list[str] = []
    issues: list[str] = []
    state_status = VALID_STATUS
    profiles: dict[str, dict[str, Any]] = {}

    def fail(problem: str, status: str = "malformed") -> None:
        nonlocal state_status
        issues.append(problem)
        if state_status != "malformed":
            state_status = status

    def active_effects() -> list[dict[str, Any]]:
        return sorted([*global_effects, *root_local, *(entry for frame in frames for entry in frame["effects"])], key=lambda entry: entry["source_order"])

    def record(effect: dict[str, Any], local: bool) -> None:
        destination = (frames[-1]["effects"] if frames else root_local) if local else global_effects
        destination.append(effect)

    for ordinal, command in enumerate(commands):
        cid = command["command_id"]
        text = _text(command)
        parser, kind = _kind(command, text)
        wrappers, inner_text, wrapper_issue = _isolated_wrappers(text)
        attributes: list[str] = []
        modifiers: list[str] = []
        current_issue = wrapper_issue
        doc_kind = parser.rsplit(".", 1)[-1] in _DOC_KINDS
        known_example = kind == "example" or (wrappers and _word(_mask(inner_text)) == "example")
        if kind in MC_KINDS or (doc_kind and DECL.search(_mask(inner_text))):
            attributes, modifiers, header_issue = _declaration_header(inner_text)
            current_issue = current_issue or header_issue
        elif wrappers and not known_example:
            current_issue = current_issue or "opaque_wrapped_non_declaration_command"
        elif not text:
            current_issue = current_issue or "missing_source_text"
        is_mc = kind in MC_KINDS or known_example or (doc_kind and bool(DECL.search(_mask(inner_text))))
        opening = None
        ending = None
        directive = None
        if not wrappers and kind in {"namespace", "section", "end"}:
            name, noncomputable, scope_issue = _scope_name(command, text, kind)
            current_issue = current_issue or scope_issue
            if kind == "end":
                ending = _name_parts(name) or [None]
                simulated = frames[:]
                for part in reversed(ending):
                    if not simulated:
                        current_issue = "end_without_scope"
                        break
                    frame = simulated.pop()
                    if part is not None and part != frame["name"]:
                        current_issue = "end_name_mismatch"
                        break
            else:
                opening = (kind, name, noncomputable)
        elif not wrappers and not is_mc and not doc_kind:
            directive = _directive_kind(kind, text)
            if directive is None and kind not in {"import", "prelude"}:
                current_issue = current_issue or "opaque_command_effect:" + str(kind)
            elif directive and not _mask(text).strip().removeprefix(_word(_mask(text))).strip():
                current_issue = current_issue or "malformed_empty_directive"
        if current_issue:
            fail(cid + ":" + current_issue, "malformed" if current_issue.startswith(("malformed", "end_", "parser_scope")) else "opaque")
        scope_stack = [{key: frame[key] for key in ("command_id", "kind", "name", "noncomputable")} for frame in frames]
        payload = {
            "profile_version": PROFILE_VERSION,
            "namespace_path": namespace[:], "scope_stack": scope_stack,
            "active_directives": active_effects(), "command_wrappers": wrappers,
            "command_attributes": attributes, "command_modifiers": modifiers,
        }
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if state_status == VALID_STATUS else None
        profiles[cid] = {
            **payload, "command_id": cid, "source_file": source_file,
            "status": state_status, "fingerprint": fingerprint, "issues": issues[:],
            "section_stack": [frame for frame in scope_stack if frame["kind"] == "section"],
            "namespace_stack": [frame for frame in scope_stack if frame["kind"] == "namespace"],
            "profile_point": "source_prefix_before_command_plus_isolated_command_wrappers",
            "compatibility_basis": "conservative_syntactic_only_not_Lean_elaboration",
            "Lean_elaboration_checked": False,
            "instance_set_equivalence_checked": False,
            "MC_predecessor_effects_included": False,
        }
        if current_issue:
            continue
        if wrappers:
            # Isolated lexical effects expire; MC availability is deliberately
            # outside this EC/lexical-scope profile, including inner instances.
            continue
        elif opening:
            frame_kind, name, noncomputable = opening
            for part in _name_parts(name) or [None]:
                frames.append({"command_id": cid, "kind": frame_kind, "name": part,
                               "noncomputable": noncomputable, "previous_namespace": namespace[:],
                               "effects": [], "opener_ordinal": ordinal})
                if frame_kind == "namespace":
                    namespace = [] if part == "_root_" else [*namespace, part]
        elif ending:
            for _ in ending:
                namespace = frames.pop()["previous_namespace"]
        elif directive or kind in {"import", "prelude"}:
            directive = directive or kind
            local = directive in _LOCAL_DIRECTIVES or bool(re.match(r"\s*local\b", _mask(text))) or (directive == "attribute" and bool(re.search(r"\[\s*local\b", _mask(text))))
            record({"command_id": cid, "source_order": ordinal, "kind": directive,
                    "source_text_sha256": hashlib.sha256(text.encode()).hexdigest()}, local)
    # Verified with Lean v4.14.0-rc1: sections (named or anonymous, including
    # noncomputable sections) and namespaces may close implicitly at file EOF.
    # Do not retroactively invalidate their legitimate before-command profiles.
    # Explicit malformed/mismatched `end` commands are still rejected above.
    return profiles


def match_scope(original: dict[str, Any] | None, replacement: dict[str, Any] | None) -> dict[str, Any]:
    if original is None or replacement is None:
        reason = "scope_profile_missing"
    elif original.get("status") != VALID_STATUS or not original.get("fingerprint"):
        reason = "original_scope_profile_unavailable:" + str(original.get("status"))
    elif replacement.get("status") != VALID_STATUS or not replacement.get("fingerprint"):
        reason = "replacement_scope_profile_unavailable:" + str(replacement.get("status"))
    elif original.get("profile_version") != PROFILE_VERSION or replacement.get("profile_version") != PROFILE_VERSION:
        reason = "scope_profile_version_mismatch"
    elif original.get("source_file") and replacement.get("source_file") and original["source_file"] != replacement["source_file"]:
        reason = "different_source_file"
    elif original["fingerprint"] != replacement["fingerprint"]:
        reason = "different_conservative_source_environment"
    else:
        reason = "compatible_conservative_syntactic_environment"
    return {"compatible": reason == "compatible_conservative_syntactic_environment", "reason": reason,
            "compatibility_basis": "syntactic_only_not_Lean_elaboration", "Lean_elaboration_checked": False,
            "instance_set_equivalence_checked": False}


def match_reason(original: dict[str, Any] | None, replacement: dict[str, Any] | None) -> str:
    return match_scope(original, replacement)["reason"]
