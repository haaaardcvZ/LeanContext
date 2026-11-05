"""Independent, streaming validation of the completed raw LeanContext release.

This module deliberately does not import the builder, selector, or provenance
implementation. It checks saved artifacts against each other and against the
unchanged original dataset, parser commands, source bytes, and compiler type
evidence. It does not claim actual token lengths, Lean correctness, or pass rates.

Run after build_status.json records completion:
    python -m dataset_preparation.validate --workspace /path/to/workspace
The only file this CLI writes is OUTPUT/validation_report.json. --self-test
runs small in-memory checks without reading or writing dataset artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


SPLITS = ("train", "validation", "test")
VIEWS = ("C0", "CMath", "CEnv", "CFull")
SPAN_KEYS = ("start_byte", "end_byte", "start_line", "start_col", "end_line", "end_col")
MC_KINDS = frozenset("theorem lemma def abbrev opaque axiom constant structure class inductive instance local_instance alias".split())
EC_KINDS = frozenset("namespace section end variable universe include omit open open_scoped notation local_notation scoped_notation infix infixl infixr prefix postfix attribute set_option export local scoped syntax macro macro_rules".split())
WRAPPERS = frozenset("open open_scoped set_option variable namespace section local scoped include omit".split())
KEYWORDS = frozenset("""theorem lemma def abbrev opaque axiom constant structure class inductive instance
local scoped alias private protected noncomputable unsafe partial nonrec where by do have let show from fun forall exists if then else match with return true false
Type Sort Prop namespace section end variable universe include omit open in set_option attribute export notation infix infixl infixr prefix postfix syntax macro macro_rules deriving extends example anonymous_target""".lower().split())
IDENT = re.compile(r"(?:«[^»]+»|[^\W\d][\w']*)(?:\.(?:«[^»]+»|[^\W\d][\w']*))*", re.UNICODE)
DECL = re.compile(r"(?:^|\bin\s+|\n\s*)(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|unsafe|partial|nonrec|local|scoped)\s+)*(?P<kind>theorem|lemma|def|abbrev|opaque|axiom|constant|structure|class|inductive|instance|alias)\b", re.MULTILINE)
CORE_ITEM_KEYS = ("position", "kind", "text", "command_id", "command_uid", "source_order", "source_span", "parser_kind", "effective_kind")
TYPE_STATUSES = {"complete_nonempty", "no_eligible_local_dependencies", "incomplete_mapping", "unresolved_target"}
FILTER_FIELDS = ("kept_command_ids_before_context_filter", "removed_context_command_ids", "context_filter_method")


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON") from exc
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            yield value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    require(isinstance(text, str), "expected exact string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def object_hash(value: Any) -> str:
    return text_hash(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def safe_relative(value: str, suffix: str | None = None) -> Path:
    require(isinstance(value, str) and bool(value), "invalid relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == value, f"unsafe relative path: {value}")
    if suffix is not None:
        require(path.suffix == suffix, f"incorrect relative path suffix: {value}")
    return Path(value)


def unique_ids(values: Any, label: str) -> list[str]:
    require(isinstance(values, list) and all(isinstance(v, str) and v for v in values), f"{label}: invalid IDs")
    require(len(values) == len(set(values)), f"{label}: duplicate IDs")
    return values


def reject_proof_fields(value: Any, label: str) -> None:
    """Inspect field names, never preceding-declaration text."""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            forbidden = lowered in {"target", "reference_proof", "target_body_text", "sft_target_text", "target_text", "proof_extra_dependency", "reference_proof_extra_dependencies"}
            forbidden = forbidden or lowered.startswith(("gold_", "value_", "outer_value", "expanded_value"))
            require(not forbidden, f"{label}: proof-bearing field {key}")
            reject_proof_fields(child, label + "." + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_proof_fields(child, f"{label}[{index}]")


def reject_fake_measurements(row: dict[str, Any], label: str) -> None:
    forbidden = {"input_ids", "labels", "attention_mask", "token_count", "prompt_token_count", "sequence_length", "pass_rate", "passrate", "generation_results", "model_output"}
    require(not (forbidden & row.keys()) and not any(key.lower().startswith("pass@") for key in row), f"{label}: raw artifact contains tokenized/model-result fields")


def _mask(text: str) -> str:
    """Independently hide string/comment contents while retaining positions."""
    out = list(text)
    position, depth = 0, 0
    while position < len(text):
        if depth:
            if text.startswith("/-", position):
                out[position:position + 2] = [" ", " "]
                depth += 1
                position += 2
            elif text.startswith("-/", position):
                out[position:position + 2] = [" ", " "]
                depth -= 1
                position += 2
            else:
                if text[position] != "\n":
                    out[position] = " "
                position += 1
        elif text.startswith("/-", position):
            out[position:position + 2] = [" ", " "]
            depth = 1
            position += 2
        elif text.startswith("--", position):
            while position < len(text) and text[position] != "\n":
                out[position] = " "
                position += 1
        elif text[position] == '"':
            out[position] = " "
            position += 1
            while position < len(text):
                char = text[position]
                out[position] = "\n" if char == "\n" else " "
                position += 1
                if char == "\\" and position < len(text):
                    out[position] = " "
                    position += 1
                elif char == '"':
                    break
        else:
            position += 1
    return "".join(out)


def terms(text: str) -> set[str]:
    result: set[str] = set()
    for identifier in IDENT.findall(_mask(text)):
        if identifier.lower() == "anonymous_target":
            continue
        for part in re.split(r"[._']+", identifier.replace("«", "").replace("»", "")):
            part = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", part)
            part = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
            result.update(token for token in part.lower().split() if token and token not in KEYWORDS)
    return result


def signature(text: str) -> str:
    masked = _mask(text)
    declaration = DECL.search(masked)
    if declaration:
        text = text[declaration.start("kind"):]
        masked = masked[declaration.start("kind"):]
    depth = 0
    for position, char in enumerate(masked):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        word_start = position == 0 or not (masked[position - 1].isalnum() or masked[position - 1] in "_'.")
        if depth == 0 and (masked.startswith(":=", position) or (word_start and re.match(r"where\b", masked[position:])) or char == "|"):
            return text[:position].strip()
    return text.strip()


def source_names(text: str, kind: str) -> list[str]:
    if kind not in MC_KINDS:
        return []
    match = DECL.search(_mask(text))
    if not match:
        return []
    tail = _mask(text)[match.end():].lstrip()
    if kind == "alias":
        arrow = re.search(r"(?:←|→|<--|-->)\s*(.*)", tail)
        if arrow:
            return [name for name in IDENT.findall(arrow.group(1).split("\n", 1)[0]) if name != "_"]
    name = IDENT.match(tail)
    return [name.group()] if name else []


def name_in_statement(name: str, statement: str) -> bool:
    short = name.rsplit(".", 1)[-1]
    return any(identifier == name or identifier == short or identifier.endswith("." + name) or identifier.rsplit(".", 1)[-1] == short for identifier in IDENT.findall(_mask(statement)))


def selector_kind(command: dict[str, Any]) -> str:
    parser, effective = command["parser_kind"], command["effective_kind"]
    if parser.startswith("Lean.Parser.Command."):
        parser = parser.rsplit(".", 1)[-1]
    if parser == "instance" and re.match(r"\s*local\s+instance\b", _mask(command["text"])):
        return "local_instance"
    if effective.startswith("Lean.Parser.Command."):
        effective = effective.rsplit(".", 1)[-1]
    if effective in WRAPPERS:
        match = DECL.search(_mask(command["text"]))
        if match:
            effective = match.group("kind")
            if effective == "instance" and re.search(r"\blocal\s+instance\s*$", _mask(command["text"])[:match.end()]):
                effective = "local_instance"
    return effective


def source_scope_metadata(source_prefix: list[dict[str, Any]], pool_ids: set[str],
                          target_order: int) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str, list[str]]:
    """Reconstruct syntactic scopes from original source, not saved annotations.

    This is a source-spelling audit, deliberately not a Lean elaboration or
    binding certificate. A named ``noncomputable section`` is a real opener.
    """
    stack: list[tuple[str, str | None, str]] = []
    namespace = ""
    valid = True
    ec: dict[str, dict[str, Any]] = {}
    pool: dict[str, dict[str, Any]] = {}
    for command in sorted(source_prefix, key=lambda item: (item["source_order"], item["command_id"])):
        cid, kind = command["command_id"], selector_kind(command)
        text = command["text"]
        if not text:
            continue
        base = {"scope_path": [frame[0] for frame in stack], "active_namespace": namespace}
        if kind in MC_KINDS:
            if cid in pool_ids:
                pool[cid] = {**base, "scope_identity_status": "syntactic_valid" if valid else "syntactic_unresolved"}
            continue
        if (kind not in EC_KINDS and not any(word in kind.lower() for word in ("notation", "syntax", "attribute"))) or (kind in WRAPPERS and re.search(r"\bin\s+\S", _mask(text))):
            continue
        annotation = {**base, "scope_role": "scope_effect"}
        masked = _mask(text)
        if kind in {"namespace", "section"}:
            opener = re.match(r"^\s*(?:noncomputable\s+)?(namespace|section)\b([^\n]*)", masked)
            parser_name = command.get("name")
            scope_name = parser_name if isinstance(parser_name, str) else opener.group(2).strip() if opener else ""
            if opener is None and parser_name is None:
                valid = False
            annotation.update(scope_role="block_opener", scope_name=scope_name, scope_pair_command_ids=[])
            for name in scope_name.split(".") if scope_name else [None]:
                stack.append((cid, name, namespace))
                if kind == "namespace":
                    if name == "_root_":
                        namespace = ""
                    elif name:
                        namespace = ".".join(filter(None, (namespace, name)))
                    else:
                        valid = False
        elif kind == "end":
            closer = re.match(r"^\s*end\b([^\n]*)", masked)
            scope_name = closer.group(1).strip() if closer else ""
            annotation.update(scope_role="block_closer", scope_name=scope_name, scope_pair_command_ids=[])
            for name in reversed(scope_name.split(".")) if scope_name else [None]:
                if not stack:
                    valid = False
                    break
                opener_id, opener_name, previous_namespace = stack.pop()
                if name and name != opener_name:
                    valid = False
                namespace = previous_namespace
                annotation["scope_pair_command_ids"].append(opener_id)
                ec[opener_id]["scope_pair_command_ids"].append(cid)
        ec[cid] = annotation
    active = {frame[0] for frame in stack}
    required = active | {cid for item in pool.values() for cid in item["scope_path"]}
    status = "syntactic_valid" if valid else "syntactic_unresolved"
    source_orders = {command["command_id"]: command["source_order"] for command in source_prefix}
    kinds = {command["command_id"]: selector_kind(command) for command in source_prefix}
    for cid, annotation in ec.items():
        role = annotation["scope_role"]
        target_active = cid in active if role == "block_opener" else False if role == "block_closer" else set(annotation["scope_path"]) <= active
        binding = kinds[cid] in {"variable", "universe", "include", "omit"} and target_active
        structure = cid in required
        annotation.update(
            target_scope_active=target_active, scope_stack_status=status,
            required_scope_command_ids=list(dict.fromkeys(annotation["scope_path"] + annotation.get("scope_pair_command_ids", []))),
            priority_group=0 if binding else 1,
            priority_reason="target_active_scope_binding" if binding else "target_or_pool_scope_structure" if structure else "conservative_pretarget_scope_effect",
        )
    priority = sorted(ec, key=lambda cid: (ec[cid]["priority_group"], target_order - source_orders[cid], cid))
    for rank, cid in enumerate(priority):
        ec[cid]["selection_priority"] = rank
    return ec, pool, status, priority


def frozen_ids(population: list[dict[str, Any]], n: int, seed: int) -> tuple[list[str], dict[str, int], dict[str, int]]:
    groups: dict[str, list[str]] = {}
    for target in population:
        groups.setdefault(target["proof_family"], []).append(target["sample_id"])
    sizes = {family: len(ids) for family, ids in sorted(groups.items())}
    total = sum(sizes.values())
    require(0 <= n <= total, "frozen manifest exceeds eligible population")
    quotas = {family: n * count // total if total else 0 for family, count in sizes.items()}
    remainder = sorted(sizes, key=lambda family: (-(n * sizes[family] % total) if total else 0, family))
    for family in remainder[:n - sum(quotas.values())]:
        quotas[family] += 1
    chosen: list[str] = []
    for family, ids in sorted(groups.items()):
        ordered = sorted(ids, key=lambda sid: (text_hash(f"{seed}\0{sid}"), sid))
        chosen.extend(ordered[:quotas[family]])
    chosen.sort(key=lambda sid: (text_hash(f"{seed}\0manifest-order\0{sid}"), sid))
    return chosen, sizes, quotas


def scrub_type_source(raw: dict[str, Any]) -> dict[str, Any]:
    dependency = raw.get("type_dependency")
    if dependency is None:
        status, dependency = "unresolved_target", {}
    elif dependency.get("unmapped_same_file_constants") or dependency.get("nonpreceding_same_file_constants") or raw.get("expanded_type_auxiliaries"):
        status = "incomplete_mapping"
    elif dependency.get("command_ids"):
        status = "complete_nonempty"
    else:
        status = "no_eligible_local_dependencies"
    return {
        "sample_id": raw["sample_id"], "source_file": raw["source_file"], "source_module": raw["source_module"],
        "target_command_id": raw["target_command_id"], "target_name": raw.get("target_name"),
        "target_identity_status": raw.get("target_identity_status"), "status": status,
        "method": "direct_elaborated_target_type_kernel_constants_mapped_to_preceding_same_file_MC_commands",
        "command_ids": list(dependency.get("command_ids", [])), "constant_evidence": dependency.get("constant_evidence", {}),
        "unmapped_same_file_constants": dependency.get("unmapped_same_file_constants", []),
        "nonpreceding_same_file_constants": dependency.get("nonpreceding_same_file_constants", []),
        "same_target_command_constants": dependency.get("same_target_command_constants", []),
        "other_local_commands": dependency.get("other_local_commands", []),
        "direct_type_constants": raw.get("outer_type_constants", []), "expanded_type_auxiliaries": raw.get("expanded_type_auxiliaries", []),
        "reference_proof_used": False, "candidate_pool_used_to_define_dependencies": False,
    }


class Validator:
    def __init__(self, workspace: Path, output: Path, audit_root: Path) -> None:
        self.workspace, self.output, self.audit_root = workspace.resolve(), output.resolve(), audit_root.resolve()
        self.manifest = read_json(self.output / "manifest.json")
        self.summary: dict[str, dict[str, Any]] = {}
        self.catalog_cache: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
        self.command_metadata: dict[str, tuple[dict[str, int], str]] = {}
        self.source_split: dict[str, str] = {}
        self.report: dict[str, Any] = {
            "schema_version": "leancontext-independent-raw-validation-v1", "valid": False,
            "workspace": str(self.workspace), "output": str(self.output),
            "validation_scope": "exact_raw_artifacts_and_original_source_provenance_not_model_correctness",
            "not_validated": ["real_tokenizer_lengths_and_trimming", "budgeted_selector_visibility", "model_generation", "Lean_proof_success", "empirical_selector_rule_tuning"],
        }

    def stage(self, name: str) -> None:
        self.report["stage"] = name
        print(json.dumps({"validation_stage": name}), flush=True)

    def catalog(self, source_file: str) -> dict[str, dict[str, Any]]:
        cached = self.catalog_cache.pop(source_file, None)
        if cached is None:
            path = self.output / "command_mapping/source_commands" / safe_relative(source_file, ".lean").with_suffix(".jsonl")
            cached = {row["command_id"]: row for row in rows(path)}
        self.catalog_cache[source_file] = cached
        while len(self.catalog_cache) > 32:
            self.catalog_cache.popitem(last=False)
        return cached

    def validate_artifacts(self) -> None:
        self.stage("artifact_hashes")
        artifacts = self.manifest.get("artifacts")
        require(isinstance(artifacts, dict) and bool(artifacts), "manifest has no artifact hashes")
        total_bytes = 0
        for relative, expected in artifacts.items():
            path = self.output / safe_relative(relative)
            require(path.is_file(), f"manifest artifact missing: {relative}")
            require(path.stat().st_size == expected["bytes"], f"artifact byte count mismatch: {relative}")
            require(file_hash(path) == expected["sha256"], f"artifact SHA256 mismatch: {relative}")
            total_bytes += expected["bytes"]
        excluded = {"manifest.json", "build_status.json", "validation_report.json"}
        unlisted = sorted(str(path.relative_to(self.output)) for path in self.output.rglob("*") if path.is_file() and str(path.relative_to(self.output)) not in artifacts and path.name not in excluded)
        self.report["artifacts"] = {"hashed_files": len(artifacts), "hashed_bytes": total_bytes, "unlisted_files": unlisted}
        immutable = {relative: expected["sha256"] for relative, expected in artifacts.items()
                     if relative.startswith(("benchmark/", "training/", "command_mapping/", "selector/", "private_posthoc_audit/"))
                     or relative == "processing/protocol.json"}
        previous_immutable = self.report.get("validated_immutable_artifact_hashes")
        if previous_immutable is not None:
            require(all(immutable.get(relative) == digest for relative, digest in previous_immutable.items()), "a previously validated immutable data artifact changed; full validation is required")
            require(set(immutable) - set(previous_immutable) <= {"command_mapping/revision_verification.json"}, "new semantic data artifacts require full validation")
        self.report["validated_immutable_artifact_hashes"] = immutable
        self.report["validated_manifest_sha256"] = file_hash(self.output / "manifest.json")
        original_manifest = self.workspace / "sft/context_benchmark/semantic_environment_dataset_split/manifest.json"
        require(file_hash(original_manifest) == self.manifest["original_split_manifest_sha256"], "original split manifest hash changed")
        for field in ("actual_model_generation_performed", "generation_or_pass_rate_results_included"):
            require(self.manifest.get(field) is False, f"raw manifest incorrectly claims {field}")
        tokenized_created = self.manifest.get("tokenized_outputs_created")
        require(isinstance(tokenized_created, bool), "manifest tokenization state is not explicit")
        if tokenized_created:
            require((self.output / "tokenization_manifest.json").is_file(), "manifest claims separate tokenization without a completed tokenization manifest")
        self.report["separate_tokenization_outputs_created"] = tokenized_created

    def validate_benchmark(self) -> None:
        self.stage("original_and_new_benchmark")
        original: dict[str, tuple[str, str]] = {}
        old_counts = Counter()
        for split in SPLITS:
            for row in rows(self.workspace / f"sft/context_benchmark/semantic_environment_dataset_split/{split}.jsonl"):
                sid = row["sample_id"]
                require(sid not in original and row["split"] == split, f"original duplicate/split mismatch: {sid}")
                fields = {key: row[key] for key in ("sample_id", "split", "proof_family", "statement", "target", "mathematical_context", "environment_context", "full_context")}
                original[sid] = (split, object_hash(fields))
                old_counts[split] += 1
        counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
        statement_splits: dict[str, str] = {}
        command_uids: set[str] = set()
        for split in SPLITS:
            for row in rows(self.output / f"benchmark/{split}.jsonl"):
                sid = row["sample_id"]
                require(sid in original and sid not in self.summary, f"new benchmark extra/duplicate ID: {sid}")
                require(row["split"] == split == original[sid][0] and row["target_id"] == sid, f"benchmark split/target ID changed: {sid}")
                fields = {key: row[key] for key in ("sample_id", "split", "proof_family", "statement", "target")}
                for category in ("mathematical_context", "environment_context", "full_context"):
                    fields[category] = [{key: item[key] for key in ("position", "kind", "text")} for item in row[category]]
                require(object_hash(fields) == original[sid][1], f"original statement/proof/context changed: {sid}")
                source_file = row["source_file"]
                safe_relative(source_file, ".lean")
                uid = source_file + "::" + row["target_command_id"]
                require(row["target_command_uid"] == uid and row["command_id"] == row["target_command_id"] and uid not in command_uids, f"target UID collision/mismatch: {sid}")
                command_uids.add(uid)
                require(self.source_split.setdefault(source_file, split) == split, f"source file crosses splits: {source_file}")
                normalized_hash = text_hash(" ".join(row["statement"].split()))
                require(statement_splits.setdefault(normalized_hash, split) == split, f"normalized statement crosses splits: {sid}")
                math, env, full = row["mathematical_context"], row["environment_context"], row["full_context"]
                require(sorted(math + env, key=lambda i: i["position"]) == full, f"Full is not exact MC/EC union: {sid}")
                require([item["position"] for item in full] == list(range(len(full))), f"Full positions changed: {sid}")
                self.check_source_items(full, row, sid)
                math_uids = {item["command_uid"] for item in math}
                env_uids = {item["command_uid"] for item in env}
                require(not math_uids & env_uids, f"MC/EC identity overlap: {sid}")
                require(all(i["effective_kind"] in MC_KINDS for i in math) and all(i["effective_kind"] not in MC_KINDS for i in env), f"MC/EC role mismatch: {sid}")
                target_command = self.catalog(source_file).get(row["target_command_id"])
                require(target_command is not None and target_command["source_span"] == row["target_source_span"], f"target catalog/span mismatch: {sid}")
                require(row["source_revision"]["source_file_sha256"] == text_hash_bytes_source(self.workspace, source_file), f"target source revision mismatch: {sid}")
                view_items = {"C0": [], "CMath": math, "CEnv": env, "CFull": full}
                item_hashes = {}
                for view, items in view_items.items():
                    expected = [{**{key: item[key] for key in CORE_ITEM_KEYS}, "role": "MC" if item["command_uid"] in math_uids else "EC"} for item in items]
                    item_hashes[view] = object_hash(expected)
                self.summary[sid] = {
                    "sample_id": sid, "target_id": sid, "split": split, "proof_family": row["proof_family"],
                    "statement_hash": text_hash(row["statement"]), "target_hash": text_hash(row["target"]),
                    "source_file": source_file, "source_module": row["source_module"], "target_command_id": row["target_command_id"],
                    "target_command_uid": uid, "target_source_span": row["target_source_span"], "target_kind": row["target_kind"], "original_name": row["original_name"],
                    "both": bool(math and env), "context_items_hashes": item_hashes,
                    "context_provenance_hash": object_hash([{key: item[key] for key in CORE_ITEM_KEYS if key != "text"} for item in full]),
                    "original_context_sha256": text_hash("\n\n".join(item["text"] for item in full)),
                    "source_sha256": row["source_revision"]["source_file_sha256"],
                }
                counts[split].update(targets=1, MC_nonempty=bool(math), EC_nonempty=bool(env), both_nonempty=bool(math and env), MC_commands=len(math), EC_commands=len(env))
                reject_fake_measurements(row, f"benchmark {sid}")
        require(set(self.summary) == set(original), "new benchmark sample membership changed")
        require({split: c["targets"] for split, c in counts.items()} == dict(old_counts), "original split counts changed")
        require({split: dict(c) for split, c in counts.items()} == self.manifest["split_counts"], "manifest split/context counts incorrect")
        require(len(self.summary) == self.manifest["raw_targets"] and len(self.source_split) == self.manifest["source_files"], "manifest target/source cardinality mismatch")
        self.report["benchmark"] = {"targets": len(self.summary), "split_counts": {split: dict(c) for split, c in counts.items()}, "source_files": len(self.source_split), "source_file_disjoint": True, "normalized_statement_disjoint": True, "exact_original_values_preserved": True}

    def check_source_items(self, items: list[dict[str, Any]], target: dict[str, Any], label: str) -> None:
        require(isinstance(items, list), f"{label}: invalid context list")
        ids: set[str] = set()
        orders = []
        catalog = self.catalog(target["source_file"])
        target_start = target["target_source_span"]["start_byte"]
        for item in items:
            cid = item["command_id"]
            uid = target["source_file"] + "::" + cid
            require(cid not in ids and item["command_uid"] == uid, f"{label}: duplicate/incorrect context identity")
            ids.add(cid)
            original = catalog.get(cid)
            require(original is not None, f"{label}: context command absent from catalog")
            require(item["source_span"] == original["source_span"] and item["source_order"] == original["source_order"] and item["text"] == original["text"], f"{label}: context source differs")
            span = item["source_span"]
            require(cid != target["target_command_id"] and span["start_byte"] < target_start and span["end_byte"] <= target_start, f"{label}: target/later command in context")
            orders.append(item["source_order"])
        require(orders == sorted(orders) and len(orders) == len(set(orders)), f"{label}: contexts not strictly source ordered")

    def validate_catalogs(self) -> None:
        self.stage("original_source_catalogs")
        snapshot = read_json(self.output / "command_mapping/source_snapshot.json")
        require(set(snapshot["source_files"]) == set(self.source_split), "source snapshot/file membership differs")
        found = {str(path.relative_to(self.output / "command_mapping/source_commands").with_suffix(".lean")) for path in (self.output / "command_mapping/source_commands").rglob("*.jsonl")}
        require(found == set(self.source_split), "catalog file membership differs")
        count, introduced_statuses = 0, Counter()
        mathlib = self.workspace / "theorem_generate/lean_parser/mathlib4-4.14.0-rc1"
        for source_file in sorted(self.source_split):
            source = (mathlib / source_file).read_bytes()
            require(hashlib.sha256(source).hexdigest() == snapshot["source_files"][source_file], f"source snapshot SHA mismatch: {source_file}")
            original_path = self.workspace / "theorem_generate/command_spans_jsonl" / Path(source_file).with_suffix(".jsonl")
            new_path = self.output / "command_mapping/source_commands" / Path(source_file).with_suffix(".jsonl")
            seen: set[str] = set()
            previous_end = -1
            for order, pair in enumerate(zip_longest(rows(original_path), rows(new_path))):
                original, catalog = pair
                require(original is not None and catalog is not None, f"catalog/original command cardinality differs: {source_file}")
                cid = original["command_id"]
                span = {key: original[key] for key in SPAN_KEYS}
                require(cid not in seen and catalog["command_id"] == cid and catalog["source_order"] == order, f"catalog command identity/order differs: {source_file}")
                seen.add(cid)
                require(catalog["source_file"] == source_file and catalog["command_uid"] == source_file + "::" + cid and catalog["source_module"] == original["source_module"], f"catalog source UID/module differs: {source_file}::{cid}")
                require(catalog["source_span"] == span and catalog["parser_kind"] == original["kind"], f"catalog original span/kind differs: {source_file}::{cid}")
                start, end = span["start_byte"], span["end_byte"]
                require(0 <= start < end <= len(source) and start >= previous_end, f"invalid/overlapping original span: {source_file}::{cid}")
                previous_end = end
                require(source[start:end] == original["raw_text"].encode("utf-8") == catalog["raw_text"].encode("utf-8"), f"catalog original raw bytes differ: {source_file}::{cid}")
                require(catalog["effective_raw_text"] == original["effective_raw_text"] and catalog["text"] == original["effective_raw_text"].strip(), f"catalog effective text differs: {source_file}::{cid}")
                for field, value in (("raw_text_sha256", catalog["raw_text"]), ("effective_raw_text_sha256", catalog["effective_raw_text"]), ("context_text_sha256", catalog["text"])):
                    require(catalog[field] == text_hash(value), f"catalog text SHA differs: {source_file}::{cid}")
                require(catalog["original_name"] == original.get("name") and catalog["name"] == original.get("name"), f"catalog surface name differs: {source_file}::{cid}")
                self.command_metadata[source_file + "::" + cid] = (span, catalog["effective_kind"])
                introduced_statuses[catalog["introduced_declarations_status"]] += 1
                count += 1
        require(count == self.manifest["source_commands"], "catalog command total differs from manifest")
        require(file_hash(mathlib / "lake-manifest.json") == snapshot["lake_manifest_sha256"], "pinned lake manifest hash differs")
        require((mathlib / "lean-toolchain").read_text().strip() == snapshot["lean_toolchain"], "pinned Lean toolchain differs")
        self.report["source_catalog"] = {"commands": count, "files": len(found), "all_original_raw_byte_spans_and_hashes_match": True, "introduced_declaration_status_counts": dict(introduced_statuses)}

    def check_target_fields(self, row: dict[str, Any], sid: str, label: str, proof: bool = False) -> dict[str, Any]:
        require(sid in self.summary and row.get("sample_id") == sid and row.get("target_id") == sid, f"{label}: unknown/incorrect target identity")
        expected = self.summary[sid]
        for field in ("split", "proof_family", "source_file", "source_module", "target_command_id", "target_command_uid", "target_kind", "original_name", "target_source_span"):
            if field in row:
                require(row[field] == expected[field], f"{label}: target {field} differs")
        if "statement" in row:
            require(text_hash(row["statement"]) == expected["statement_hash"], f"{label}: statement differs")
        if proof:
            require(text_hash(row["target"]) == expected["target_hash"], f"{label}: reference proof differs")
        return expected

    def validate_training(self) -> None:
        self.stage("four_training_views")
        view_counts: dict[str, dict[str, int]] = {}
        for view in VIEWS:
            view_counts[view] = {}
            seen: set[str] = set()
            for split in SPLITS:
                count = 0
                for row in rows(self.output / f"training/views/{view}/{split}.jsonl"):
                    sid = row["sample_id"]
                    require(sid not in seen and row["view"] == view, f"{view}: duplicate/incorrect view row")
                    seen.add(sid)
                    expected = self.check_target_fields(row, sid, f"{view} {sid}", proof=True)
                    require(row["split"] == split, f"{view}: shard split differs")
                    items = row["context_items"]
                    self.check_source_items(items, row, f"{view} {sid}")
                    require(object_hash([{**{key: item[key] for key in CORE_ITEM_KEYS}, "role": item["role"]} for item in items]) == expected["context_items_hashes"][view], f"{view} {sid}: context/role/subset differs")
                    require(row["context_ids"] == [item["command_id"] for item in items] and row["context_uids"] == [item["command_uid"] for item in items], f"{view} {sid}: context ID list differs")
                    prompt = "\n\n".join([*(item["text"] for item in items), row["statement"]]).rstrip("\r\n") + "\n"
                    require(row["prompt"] == prompt and row["prompt"].endswith("\n") and not row["prompt"].endswith("\n\n"), f"{view} {sid}: prompt template/newline differs")
                    require(row["template_version"] == "training_source_order_context_then_statement_one_newline_v1" and row["tokenization_status"] == "pending_real_backbone_tokenizer", f"{view} {sid}: raw template/status differs")
                    reject_fake_measurements(row, f"{view} {sid}")
                    count += 1
                view_counts[view][split] = count
            require(seen == set(self.summary), f"{view}: target membership differs")
        for kind in ("targets", "split_ids"):
            seen = set()
            for split in SPLITS:
                for row in rows(self.output / f"training/{kind}/{split}.jsonl"):
                    sid = row["sample_id"]
                    require(sid not in seen and self.summary[sid]["split"] == split, f"training {kind}: duplicate/changed membership")
                    seen.add(sid)
                    expected = self.check_target_fields(row, sid, f"training {kind} {sid}", proof=kind == "targets")
                    require(row["command_id"] == expected["target_command_id"], f"training {kind}: command ID differs")
            require(seen == set(self.summary), f"training {kind}: target membership differs")
        self.report["training_views"] = {"counts": view_counts, "same_target_membership_statements_reference_proofs": True, "exact_context_subsets_roles_and_prompt_templates": True}

    def validate_mappings_and_history(self) -> None:
        self.stage("target_mappings_and_original_selection_history")
        seen = set()
        for mapping in rows(self.output / "command_mapping/targets.jsonl"):
            sid = mapping["sample_id"]
            require(sid not in seen, f"duplicate mapping target: {sid}")
            seen.add(sid)
            expected = self.check_target_fields(mapping, sid, f"mapping {sid}")
            require(mapping["statement_sha256"] == expected["statement_hash"] and mapping["proof_sha256"] == expected["target_hash"] and mapping["original_context_sha256"] == expected["original_context_sha256"] and mapping["source_sha256"] == expected["source_sha256"], f"mapping hash differs: {sid}")
            catalog = self.catalog(expected["source_file"])
            target = catalog[expected["target_command_id"]]
            require(mapping["target_raw_text_sha256"] == target["raw_text_sha256"] and mapping["target_effective_raw_text_sha256"] == target["effective_raw_text_sha256"] and mapping["target_source_order"] == target["source_order"], f"mapping target catalog hashes/order differ: {sid}")
            contexts = mapping["context_commands"]
            basic = [{"position": item["position"], "kind": item["effective_kind"], "command_id": item["command_id"], "command_uid": item["command_uid"], "source_order": item["source_order"], "source_span": item["source_span"], "parser_kind": item["parser_kind"], "effective_kind": item["effective_kind"]} for item in contexts]
            require(object_hash(basic) == expected["context_provenance_hash"], f"mapping context differs: {sid}")
            for item in contexts:
                original = catalog[item["command_id"]]
                require(item["raw_text_sha256"] == original["raw_text_sha256"] and item["effective_raw_text_sha256"] == original["effective_raw_text_sha256"] and item["context_text_sha256"] == original["context_text_sha256"], f"mapping context hashes differ: {sid}")
            require("target" not in mapping and "gold_target" not in mapping and "raw_text" not in mapping, f"mapping unexpectedly contains proof text: {sid}")
        require(seen == set(self.summary), "target mapping cardinality/membership differs")
        reverse_seen = set()
        reverse_uids = set()
        for reverse in rows(self.output / "command_mapping/command_to_targets.jsonl"):
            require(reverse["command_uid"] not in reverse_uids, "reverse mapping duplicate command UID")
            reverse_uids.add(reverse["command_uid"])
            require(len(reverse["target_ids"]) == 1, "reverse mapping target cardinality differs")
            sid = reverse["target_ids"][0]
            require(sid not in reverse_seen and sid in self.summary, "reverse mapping duplicate/unknown target")
            reverse_seen.add(sid)
            expected = self.summary[sid]
            require(reverse["source_file"] == expected["source_file"] and reverse["command_id"] == expected["target_command_id"] and reverse["command_uid"] == expected["target_command_uid"] and reverse["split"] == expected["split"], f"reverse mapping differs: {sid}")
        require(reverse_seen == set(self.summary), "reverse mapping target membership differs")
        history = {}
        for record in rows(self.output / "command_mapping/original_selection_history.jsonl"):
            sid = record["sample_id"]
            require(sid not in history, "duplicate selection history")
            expected = self.check_target_fields(record, sid, f"selection history {sid}")
            require(record["original_context_sha256"] == expected["original_context_sha256"], f"selection context hash differs: {sid}")
            history[sid] = record
        require(set(history) == set(self.summary), "selection history target membership differs")
        by_tuple = {(item["source_file"], item["target_command_id"]): sid for sid, item in self.summary.items()}
        matched = set()
        assistance_counts, availability_counts = Counter(), Counter()
        raw_path = Path(self.manifest["inputs"]["raw_selection"])
        for raw in rows(raw_path):
            sid = by_tuple.get((raw.get("lean_file_name") or raw.get("source_file"), raw.get("target_command_id")))
            if sid is None or text_hash(str(raw.get("context_text") or "").strip()) != self.summary[sid]["original_context_sha256"]:
                continue
            require(sid not in matched, f"duplicate exact raw selection: {sid}")
            matched.add(sid)
            record = history[sid]
            for field in ("kept_command_ids", "candidate_command_ids", *FILTER_FIELDS):
                require(record.get(field) == raw.get(field), f"original selection/filter field changed: {sid}: {field}")
            catalog = self.catalog(self.summary[sid]["source_file"])
            kept = unique_ids(raw["kept_command_ids"], f"raw kept IDs {sid}")
            require(all(cid in catalog for cid in kept), f"raw kept source command missing: {sid}")
            ordered = sorted(kept, key=lambda cid: catalog[cid]["source_order"])
            require(record["ordered_kept_command_ids"] == ordered and record["ordered_kept_command_uids"] == [self.summary[sid]["source_file"] + "::" + cid for cid in ordered], f"history kept source order/UIDs differ: {sid}")
            original_context = []
            for position, cid in enumerate(ordered):
                command = catalog[cid]
                original_context.append({"position": position, "kind": command["effective_kind"], "command_id": cid, "command_uid": command["command_uid"], "source_order": command["source_order"], "source_span": command["source_span"], "parser_kind": command["parser_kind"], "effective_kind": command["effective_kind"]})
            require(object_hash(original_context) == self.summary[sid]["context_provenance_hash"], f"raw selected context identities differ from benchmark: {sid}")
            availability = {field: "missing" if field not in raw else "null" if raw[field] is None else "present" for field in FILTER_FIELDS}
            require(record["context_filter_history_availability"] == availability, f"filter missing/null tracking differs: {sid}")
            for field in ("selection_provider", "selection_model", "selection_method", "selection_reason", "verified_by_lean"):
                require((field in record) == (field in raw) and record.get(field) == raw.get(field), f"selection provenance field changed: {sid}: {field}")
            if raw.get("selection_method") == "parser_candidates_plus_llm_minimized" and "reference_assisted" not in raw:
                require(record["reference_assisted"] is True and "truncation" in record["reference_assisted_method"], f"original reference assistance not disclosed: {sid}")
            assistance_counts[str(record.get("reference_assisted"))] += 1
            availability_counts.update(f"{field}:{status}" for field, status in availability.items())
        require(matched == set(self.summary), "original selection history missing exact raw records")
        self.report["mapping_and_history"] = {"target_mappings": len(seen), "unique_target_command_uids": len(reverse_uids), "exact_raw_selection_histories": len(matched), "reference_assistance_counts": dict(assistance_counts), "filter_history_availability": dict(availability_counts)}

    def validate_type_evidence(self) -> dict[str, dict[str, Any]]:
        self.stage("independent_scrubbed_compiler_type_evidence")
        evidence = {}
        statuses = Counter()
        for row in rows(self.output / "selector/target_type_dependencies.jsonl"):
            sid = row["sample_id"]
            require(sid in self.summary and sid not in evidence, "type evidence duplicate/unknown target")
            expected = self.summary[sid]
            require(row["source_file"] == expected["source_file"] and row["target_command_id"] == expected["target_command_id"], f"type evidence source tuple differs: {sid}")
            require(row["reference_proof_used"] is False and row["candidate_pool_used_to_define_dependencies"] is False, f"type evidence not independent/proof blind: {sid}")
            reject_proof_fields(row, f"type evidence {sid}")
            require(row["status"] in TYPE_STATUSES, f"unknown type evidence status: {sid}")
            unique_ids(row["command_ids"], f"type evidence {sid}")
            for cid in row["command_ids"]:
                metadata = self.command_metadata.get(expected["source_file"] + "::" + cid)
                require(metadata is not None and metadata[1] in MC_KINDS and metadata[0]["end_byte"] <= expected["target_source_span"]["start_byte"] and cid != expected["target_command_id"], f"type evidence is not preceding source-local MC: {sid}: {cid}")
            evidence[sid] = row
            statuses[row["status"]] += 1
        # The compiler audit covers a recorded subset, not every
        # raw training target. Missing evidence must not be fabricated as empty.
        require(set(evidence) <= set(self.summary), "type evidence contains non-benchmark target IDs")
        original = self.audit_root / "semantic_mc_labels.jsonl"
        require(original.is_file(), f"independent compiler type evidence input unavailable: {original}")
        require(file_hash(original) == self.manifest["inputs"]["type_mapping_source_sha256"], "original compiler type mapping hash differs")
        matched = set()
        for raw in rows(original):
            sid = raw["sample_id"]
            require(sid in evidence and sid not in matched, "original compiler evidence duplicate/unknown target")
            matched.add(sid)
            require(scrub_type_source(raw) == evidence[sid], f"saved type evidence differs from independently scrubbed source: {sid}")
        require(matched == set(evidence), "original compiler type mapping target membership differs")
        compiler_dump = self.audit_root / "declaration_dependencies.jsonl"
        require(file_hash(compiler_dump) == self.manifest["inputs"]["compiler_dependency_dump_sha256"], "original compiler dependency dump hash differs")
        self.report["target_type_evidence"] = {"targets_with_recorded_evidence": len(evidence), "benchmark_targets_without_recorded_evidence": len(set(self.summary) - set(evidence)), "status_counts": dict(statuses), "exact_independent_type_only_source_comparison": True}
        return evidence

    def validate_selector_manifest(self, filename: str, split: str, n: int) -> list[str]:
        manifest = read_json(self.output / "selector" / filename)
        ids = unique_ids(manifest["target_ids"], filename)
        require(manifest["split"] == split and manifest["n"] == n and len(ids) == n, f"{filename}: wrong split/frozen size")
        population = [row for row in self.summary.values() if row["split"] == split and row["both"]]
        expected_ids, sizes, quotas = frozen_ids(population, n, manifest["selection_seed"])
        require(ids == expected_ids and manifest["eligible_population"] == len(population) and manifest["family_population_counts"] == sizes and manifest["family_quotas"] == quotas, f"{filename}: independent frozen sampling differs")
        require(manifest["selection_uses_dependencies_or_results"] is False, f"{filename}: sampling is not outcome independent")
        targets = manifest["targets"]
        require([row["sample_id"] for row in targets] == ids, f"{filename}: frozen target list differs")
        for target in targets:
            sid = target["sample_id"]
            reject_proof_fields(target, f"manifest target {sid}")
            self.check_target_fields(target, sid, f"manifest target {sid}")
            require(self.summary[sid]["split"] == split and self.summary[sid]["both"], f"{filename}: ineligible frozen target")
        return ids

    def validate_pairs(self, filename: str, ids: list[str], evidence: dict[str, dict[str, Any]]) -> dict[str, int]:
        counts = Counter()
        actual_ids = []
        coverage = Counter()
        ec_statuses = Counter()
        for pair in rows(self.output / "selector" / filename):
            sid = pair["sample_id"]
            actual_ids.append(sid)
            expected = self.check_target_fields(pair, sid, f"pair {sid}")
            reject_proof_fields(pair, f"pair {sid}")
            reject_fake_measurements(pair, f"pair {sid}")
            require(pair["current_target_reference_proof_read"] is False and pair["EC_same_for_both_policies"] is True, f"pair {sid}: proof-blind/shared-EC flag incorrect")
            require(pair["candidate_pool_origin"] == "reconstructed_from_full_original_source_commands", f"pair {sid}: pool origin differs")
            catalog = self.catalog(expected["source_file"])
            target_command = catalog[expected["target_command_id"]]
            target_order = target_command["source_order"]
            source_prefix = [row for row in catalog.values() if row["source_span"]["start_byte"] < expected["target_source_span"]["start_byte"] and row["source_span"]["end_byte"] <= expected["target_source_span"]["start_byte"] and row["command_id"] != expected["target_command_id"]]
            query_terms = terms(pair["statement"])
            expected_pool, expected_ec = {}, []
            for command in source_prefix:
                kind, text = selector_kind(command), command["text"]
                parser = command["parser_kind"]
                if parser.startswith("Lean.Parser.Command."):
                    parser = parser.rsplit(".", 1)[-1]
                if not text:
                    continue
                if kind in MC_KINDS:
                    names = source_names(text, kind)
                    if not names and isinstance(command.get("name"), str) and IDENT.fullmatch(command["name"]):
                        names = [command["name"]]
                    certified = [str(item["name"]) for item in command.get("introduced_declarations", []) if isinstance(item, dict) and item.get("name") and item.get("basis")]
                    lexical_names = names[:]
                    names = list(dict.fromkeys(names or certified))
                    if not names or (parser in {"theorem", "lemma"} and not any(name_in_statement(name, pair["statement"]) for name in lexical_names)):
                        continue
                    header = signature(text)
                    candidate_terms = terms(" ".join(names) + "\n" + header)
                    numerator, denominator = len(query_terms & candidate_terms), len(query_terms | candidate_terms)
                    expected_pool[command["command_id"]] = {"names": names, "signature": header, "lexical_terms": sorted(candidate_terms), "score_numerator": numerator, "score_denominator": denominator, "source_distance": target_order - command["source_order"], "kind": kind, "lexical_filter_source_names": lexical_names}
                elif (kind in EC_KINDS or any(word in kind.lower() for word in ("notation", "syntax", "attribute"))) and not (kind in WRAPPERS and re.search(r"\bin\s+\S", _mask(text))):
                    expected_ec.append(command["command_id"])
            pool, ec = pair["candidate_pool"], pair["shared_environment"]
            self.check_source_items(pool, pair, f"pair pool {sid}")
            self.check_source_items(ec, pair, f"pair EC {sid}")
            pool_ids = [command["command_id"] for command in pool]
            ec_ids = [command["command_id"] for command in ec]
            require(set(pool_ids) == set(expected_pool) and ec_ids == expected_ec and not set(pool_ids) & set(ec_ids), f"pair {sid}: independently reconstructed pool/EC membership differs")
            expected_ec_scopes, expected_pool_scopes, scope_status, ec_priority = source_scope_metadata(source_prefix, set(pool_ids), target_order)
            require(pair["EC_scope_stack_status"] == scope_status and pair["shared_environment_priority"] == ec_priority, f"pair {sid}: independent source scope validity/EC priority differs")
            for command in pool:
                detail = expected_pool[command["command_id"]]
                require(all(command[field] == value for field, value in detail.items()), f"pair {sid}: lexical source/signature features differ")
                require(all(command[field] == value for field, value in expected_pool_scopes[command["command_id"]].items()), f"pair {sid}: candidate source namespace/scope annotation differs")
                require(command["score"] == (detail["score_numerator"] / detail["score_denominator"] if detail["score_denominator"] else 0.0), f"pair {sid}: lexical score differs")
            baseline = unique_ids(pair["baseline_ranking"], f"pair baseline {sid}")
            priority = unique_ids(pair["dependency_prioritized_ranking"], f"pair priority {sid}")
            require(set(baseline) == set(priority) == set(pool_ids), f"pair {sid}: rankings do not share exact pool")
            expected_baseline = sorted(expected_pool, key=lambda cid: (-Fraction(expected_pool[cid]["score_numerator"], expected_pool[cid]["score_denominator"] or 1), expected_pool[cid]["source_distance"], cid))
            require(baseline == expected_baseline, f"pair {sid}: independent lexical baseline order differs")
            require(sid in evidence, f"pair {sid}: no independent recorded type evidence")
            original_type = evidence[sid]
            dependencies = pair["target_type_dependencies"]
            require(dependencies["status"] == original_type["status"] and dependencies["command_ids"] == original_type["command_ids"], f"pair {sid}: independent dependencies/status differ")
            for field in ("constant_evidence", "unmapped_same_file_constants", "nonpreceding_same_file_constants", "same_target_command_constants"):
                require(dependencies[field] == original_type[field], f"pair {sid}: type-only evidence changed")
            require(dependencies["provenance"] == "independently_supplied_kernel_target_type_evidence", f"pair {sid}: dependency origin differs")
            dep_ids = set(dependencies["command_ids"])
            fallback = dependencies["status"] != "complete_nonempty"
            expected_priority = baseline if fallback else [cid for cid in baseline if cid in dep_ids] + [cid for cid in baseline if cid not in dep_ids]
            require(priority == expected_priority and pair["dependency_priority_fallback"] is fallback and pair["dependency_priority_fallback_reason"] == (dependencies["status"] if fallback else None), f"pair {sid}: stable priority partition/fallback differs")
            pool_set = set(pool_ids)
            status = "mapping_uncertain" if dependencies["status"] in {"incomplete_mapping", "unresolved_target"} else "no_eligible_local_dependencies" if not dep_ids else "all_dependencies_in_pool" if dep_ids <= pool_set else "partial_dependencies_in_pool" if dep_ids & pool_set else "no_dependencies_in_pool"
            require(pair["dependency_pool_coverage"] == {"status": status, "independent_dependency_count": len(dep_ids), "in_pool_command_ids": sorted(dep_ids & pool_set), "missing_from_pool_command_ids": sorted(dep_ids - pool_set)}, f"pair {sid}: raw pool coverage fabricated/incorrect")
            require(pair["shared_environment_command_ids"] == ec_ids and set(unique_ids(pair["shared_environment_priority"], f"EC priority {sid}")) == set(ec_ids), f"pair {sid}: shared EC IDs/priority membership differs")
            for command in ec:
                require(command["scope_stack_status"] == pair["EC_scope_stack_status"], f"pair {sid}: shared EC status differs")
                require(all(command[field] == value for field, value in expected_ec_scopes[command["command_id"]].items()), f"pair {sid}: independent EC opener/closer/active-scope metadata differs")
            require(pair["tokenization_status"] == "pending" and pair["final_visibility_status"] == "pending_server_tokenization" and pair["final_target_coverage_status"] == "pending_actual_whole_command_visibility", f"pair {sid}: fake final/tokenizer coverage claim")
            counts.update(targets=1, empty_candidate_pool=not bool(pool), candidate_pool_commands=len(pool))
            counts[original_type["status"]] += 1
            coverage[status] += 1
            ec_statuses[pair["EC_scope_stack_status"]] += 1
        require(actual_ids == ids, f"{filename}: frozen pair IDs/order/membership differs")
        self.report.setdefault("selector_raw_coverage", {})[filename] = {"target_count": len(actual_ids), "dependency_pool_status_counts": dict(coverage), "EC_scope_status_counts": dict(ec_statuses), "final_visibility": "pending_actual_tokenizer_not_measured"}
        return dict(counts)

    def validate_selectors(self, evidence: dict[str, dict[str, Any]]) -> None:
        self.stage("frozen_proof_blind_selector_pairs")
        test_ids = self.validate_selector_manifest("frozen_test_manifest.json", "test", 2000)
        validation_population = sum(row["split"] == "validation" and row["both"] for row in self.summary.values())
        val_ids = self.validate_selector_manifest("validation_manifest.json", "validation", min(200, validation_population))
        require(not set(test_ids) & set(val_ids), "test and validation selector IDs overlap")
        seen = []
        for row in rows(self.output / "selector/targets.jsonl"):
            sid = row["sample_id"]
            seen.append(sid)
            reject_proof_fields(row, f"selector target {sid}")
            self.check_target_fields(row, sid, f"selector target {sid}")
        require(seen == test_ids, "selector target table IDs/order differ from frozen manifest")
        eligible = {sid for sid, row in self.summary.items() if row["split"] == "test" and row["both"]}
        population = set()
        for row in rows(self.output / "selector/test_population_ids.jsonl"):
            sid = row["sample_id"]
            require(sid in eligible and sid not in population and row["source_file"] == self.summary[sid]["source_file"] and row["command_id"] == self.summary[sid]["target_command_id"], "selector population IDs/source differ")
            population.add(sid)
        require(population == eligible, "selector both-nonempty test population differs")
        counts = {"pairs.jsonl": self.validate_pairs("pairs.jsonl", test_ids, evidence), "validation_pairs.jsonl": self.validate_pairs("validation_pairs.jsonl", val_ids, evidence)}
        require(counts == self.manifest["selector_counts"], "selector manifest counts differ from actual raw statuses")
        self.report["selector"] = {"frozen_test_targets": len(test_ids), "structural_validation_targets": len(val_ids), "eligible_both_nonempty_test_population": len(population), "test_validation_disjoint": True, "exact_frozen_pair_and_target_membership": True, "source_prefix_pool_and_shared_EC_independently_reconstructed": True, "priority_is_stable_independent_type_partition_or_fallback": True, "status_counts": counts}

    def run(self) -> dict[str, Any]:
        self.validate_artifacts()
        self.validate_benchmark()
        self.validate_catalogs()
        self.validate_training()
        self.validate_mappings_and_history()
        evidence = self.validate_type_evidence()
        self.validate_selectors(evidence)
        self.report.update(valid=True, stage="complete", validated_at_utc=datetime.now(timezone.utc).isoformat())
        return self.report


_SOURCE_HASH_CACHE: dict[tuple[str, str], str] = {}


def text_hash_bytes_source(workspace: Path, source_file: str) -> str:
    key = (str(workspace), source_file)
    if key not in _SOURCE_HASH_CACHE:
        _SOURCE_HASH_CACHE[key] = file_hash(workspace / "theorem_generate/lean_parser/mathlib4-4.14.0-rc1" / safe_relative(source_file, ".lean"))
    return _SOURCE_HASH_CACHE[key]


def self_test() -> None:
    require(signature("open Classical in\nnoncomputable def café : Nat := by exact 1") == "def café : Nat", "signature self-test failed")
    require(terms('theorem anonymous_target : Foo.bar := "secret" /- nested /- leak -/ leak -/') == {"foo", "bar"}, "masked lexical self-test failed")
    population = [{"sample_id": str(i), "proof_family": "a" if i < 3 else "b"} for i in range(5)]
    ids, sizes, quotas = frozen_ids(population, 3, 7)
    require(len(ids) == 3 and sizes == {"a": 3, "b": 2} and quotas == {"a": 2, "b": 1}, "sampling self-test failed")
    reject_proof_fields({"candidate_pool": [{"text": "theorem historical : True := by trivial"}], "current_target_reference_proof_read": False}, "allowed historical text")
    try:
        reject_proof_fields({"nested": {"value_dependency": []}}, "forbidden")
    except ValidationError:
        pass
    else:
        raise ValidationError("proof-field self-test failed")
    scope_fixture = [
        {"command_id": "s", "source_order": 0, "name": "Named", "parser_kind": "section", "effective_kind": "section", "text": "noncomputable section Named"},
        {"command_id": "v", "source_order": 1, "name": None, "parser_kind": "variable", "effective_kind": "variable", "text": "variable (n : Nat)"},
        {"command_id": "d", "source_order": 2, "name": "old", "parser_kind": "def", "effective_kind": "def", "text": "def old := n"},
        {"command_id": "e", "source_order": 3, "name": "Named", "parser_kind": "end", "effective_kind": "end", "text": "end Named"},
    ]
    scope, pool, status, priority = source_scope_metadata(scope_fixture, {"d"}, 4)
    require(status == "syntactic_valid" and scope["s"]["scope_pair_command_ids"] == ["e"] and scope["e"]["scope_pair_command_ids"] == ["s"] and pool["d"]["scope_path"] == ["s"] and set(priority) == {"s", "v", "e"}, "named noncomputable section source-scope self-test failed")
    broken = [*scope_fixture[:-1], {**scope_fixture[-1], "text": "end Other"}]
    require(source_scope_metadata(broken, {"d"}, 4)[2] == "syntactic_unresolved", "mismatched named end self-test failed")
    print(json.dumps({"self_test": "passed", "dataset_files_read_or_written": False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-root", type=Path,
                        help="Kernel-dependency audit input directory (required for full validation)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--artifacts-only", action="store_true", help="Recheck all hashes after processing-code changes; requires a prior valid full report and unchanged semantic data hashes")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_root is None:
        parser.error("--audit-root is required unless --self-test is selected")
    output = (args.output or args.workspace / "leancontext_dataset").resolve()
    status = read_json(output / "build_status.json")
    require(status.get("status") in {"complete_raw_pending_real_tokenization", "complete_real_tokenization"}, "build not complete: validator will not write a report during construction")
    validator = Validator(args.workspace, output, args.audit_root)
    if args.artifacts_only:
        previous = read_json(output / "validation_report.json")
        require(previous.get("valid") is True and "validated_immutable_artifact_hashes" in previous, "artifact-only refresh requires a prior valid full report with immutable hashes")
        validator.report = previous
    exit_code = 0
    try:
        if args.artifacts_only:
            validator.validate_artifacts()
            report = validator.report
            report.update(valid=True, stage="complete", artifact_revalidated_at_utc=datetime.now(timezone.utc).isoformat())
        else:
            report = validator.run()
    except (ValidationError, KeyError, TypeError, OSError, ValueError) as exc:
        report = validator.report
        report.update(valid=False, error=str(exc), validated_at_utc=datetime.now(timezone.utc).isoformat())
        exit_code = 1
    report_path = output / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "report": str(report_path), "error": report.get("error")}), flush=True)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
