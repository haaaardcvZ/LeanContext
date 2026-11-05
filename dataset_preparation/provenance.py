"""Strict, read-only provenance joins for the Lean context dataset.

Bare command IDs are file-local.  The stable command identity used here is
``source_file::command_id``.  Original byte spans refer to ``raw_text`` in the
pinned Lean source; public context text comes from ``effective_raw_text``.
No function writes dataset or original source files.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .context_classification import (
    DIRECT_MATHEMATICAL_KINDS,
    effective_context_kind,
)


PUBLIC_SPLITS = ("train", "validation", "test")
PRIVATE_SPLITS = ("train", "val", "test")
SPAN_FIELDS = (
    "start_byte", "end_byte", "start_line", "start_col", "end_line", "end_col"
)
CONTEXT_FILTER_FIELDS = (
    "kept_command_ids_before_context_filter",
    "removed_context_command_ids",
    "context_filter_method",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _string(row: dict[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: missing or invalid {field}")
    return value


def _relpath(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.suffix != ".lean":
        raise ValueError(f"invalid source relpath: {value!r}")
    if path.as_posix() != value:
        raise ValueError(f"source relpath is not canonical: {value!r}")
    return value


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read object rows and report malformed input with its file/line location."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSON row is not an object")
            yield row


def load_public_rows(public_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Read current public shards without changing their split assignments."""
    public_dir = Path(public_dir)
    rows: dict[str, dict[str, Any]] = {}
    for split in PUBLIC_SPLITS:
        path = public_dir / f"{split}.jsonl"
        for row in iter_jsonl(path):
            sample_id = _string(row, "sample_id", str(path))
            if sample_id in rows:
                raise ValueError(f"duplicate public sample_id: {sample_id}")
            if row.get("split") != split:
                raise ValueError(f"{sample_id}: public shard/split mismatch")
            for field in ("proof_family", "statement", "target"):
                _string(row, field, sample_id)
            rows[sample_id] = row
    return rows


def _private_key(row: dict[str, Any]) -> tuple[str, str]:
    sample_id = _string(row, "sample_id", "private row")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{sample_id}: missing private metadata")
    return (
        _relpath(_string(row, "relpath", sample_id)),
        _string(metadata, "target_command_id", sample_id),
    )


def load_private_rows(
    private_dir: str | Path, needed_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Join source provenance shards by ID, independently of their stored split."""
    private_dir = Path(private_dir)
    needed_ids = set(needed_ids)
    rows: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    seen_keys: dict[tuple[str, str], str] = {}
    for split in PRIVATE_SPLITS:
        path = private_dir / f"{split}.jsonl"
        if split == "val" and not path.is_file():
            path = private_dir / "validation.jsonl"
        for row in iter_jsonl(path):
            sample_id = _string(row, "sample_id", str(path))
            if sample_id in seen_ids:
                raise ValueError(f"duplicate private sample_id: {sample_id}")
            seen_ids.add(sample_id)
            if sample_id not in needed_ids:
                continue
            key = _private_key(row)
            if key in seen_keys:
                raise ValueError(
                    f"duplicate private target tuple: {key} ({seen_keys[key]}, {sample_id})"
                )
            seen_keys[key] = sample_id
            for field in ("statement", "target", "prompt"):
                _string(row, field, sample_id)
            _string(row["metadata"], "proof_family", sample_id)
            rows[sample_id] = row
    missing = sorted(needed_ids - rows.keys())
    if missing:
        raise ValueError(f"private prepared data misses public IDs: {missing[:10]}")
    return rows


def _private_context(row: dict[str, Any]) -> str:
    sample_id = _string(row, "sample_id", "private row")
    statement = _string(row, "statement", sample_id).strip()
    prompt = _string(row, "prompt", sample_id).strip()
    if not statement or not prompt.endswith(statement):
        raise ValueError(f"{sample_id}: private prompt does not end in its statement")
    return prompt[:-len(statement)].strip()


def _command_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(i, str) or not i for i in value):
        raise ValueError(f"{label}: invalid command ID list")
    if len(value) != len(set(value)):
        raise ValueError(f"{label}: duplicate command ID")
    return list(value)


def _context_filter_history(raw: dict[str, Any], sample_id: str) -> dict[str, Any]:
    """Keep exact filter metadata; distinguish absent, explicit null, and empty."""
    history: dict[str, Any] = {}
    availability: dict[str, str] = {}
    for field in CONTEXT_FILTER_FIELDS:
        value = raw.get(field)
        availability[field] = (
            "missing" if field not in raw else "null" if value is None else "present"
        )
        if value is not None:
            if field.endswith("command_ids") or field == "kept_command_ids_before_context_filter":
                _command_ids(value, f"{sample_id}: {field}")
            elif not isinstance(value, str):
                raise ValueError(f"{sample_id}: invalid {field}")
        # Missing does not mean that no commands were removed: retain None.
        history[field] = value
    history["context_filter_history_availability"] = availability
    return history


def _reference_assistance(raw: dict[str, Any], sample_id: str) -> dict[str, Any]:
    """Describe selector access to original proof separately from Lean status."""
    if "reference_assisted" in raw:
        value = raw["reference_assisted"]
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{sample_id}: invalid reference_assisted")
        return {
            "reference_assisted": value,
            "reference_assisted_method": raw.get("reference_assisted_method"),
            "reference_assisted_availability": "null" if value is None else "explicit",
        }
    if raw.get("selection_method") == "parser_candidates_plus_llm_minimized":
        # context_pipeline.make_llm_messages supplies command_text(target),
        # i.e. the original declaration, with a configurable truncation limit.
        return {
            "reference_assisted": True,
            "reference_assisted_method": (
                "original_target_command_text_supplied_to_llm_selector_"
                "including_reference_proof_when_within_truncation_limit"
            ),
            "reference_assisted_availability": "inferred_from_known_original_pipeline",
        }
    return {
        "reference_assisted": None,
        "reference_assisted_method": "unknown_for_unrecorded_or_unrecognized_selection_method",
        "reference_assisted_availability": "unknown",
    }


def load_selections(
    raw_path: str | Path, private_rows: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Recover original selection IDs by source tuple AND exact context text.

    Other-context versions of a target are ignored, as in the original builder.
    More than one exact match is rejected, even if the duplicate is identical.
    The selected row contains only selection/provenance fields, not target text.
    """
    raw_path = Path(raw_path)
    by_key: dict[tuple[str, str], str] = {}
    contexts: dict[str, str] = {}
    for sample_id, row in private_rows.items():
        if row.get("sample_id") != sample_id:
            raise ValueError(f"private dictionary/sample_id mismatch: {sample_id}")
        key = _private_key(row)
        if key in by_key:
            raise ValueError(f"duplicate private target tuple: {key}")
        by_key[key] = sample_id
        contexts[sample_id] = _private_context(row)
    selections: dict[str, dict[str, Any]] = {}
    for row_number, raw in enumerate(iter_jsonl(raw_path), 1):
        key = (
            raw.get("lean_file_name") or raw.get("source_file") or "",
            raw.get("target_command_id") or "",
        )
        sample_id = by_key.get(key)
        if sample_id is None:
            continue
        context = raw.get("context_text")
        if not isinstance(context, str) or context.strip() != contexts[sample_id]:
            continue
        if sample_id in selections:
            raise ValueError(f"{sample_id}: duplicate exact raw selection")
        kept = _command_ids(raw.get("kept_command_ids", []), sample_id)
        candidates = _command_ids(raw.get("candidate_command_ids", []), sample_id)
        if not set(kept).issubset(candidates):
            raise ValueError(f"{sample_id}: selected command absent from candidate IDs")
        if key[1] in candidates or key[1] in kept:
            raise ValueError(f"{sample_id}: target appears in its context candidates")
        selection = {
            "sample_id": sample_id,
            "source_file": key[0],
            "target_command_id": key[1],
            "candidate_command_ids": candidates,
            "kept_command_ids": kept,
            "original_context_text": contexts[sample_id],
            "raw_selection_file": str(raw_path),
            "raw_selection_row": row_number,
        }
        for field in (
            "selection_provider", "selection_model", "selection_method",
            "selection_reason", "verified_by_lean", "mathlib4_version",
        ):
            if field in raw:
                selection[field] = raw[field]
        selection.update(_context_filter_history(raw, sample_id))
        selection.update(_reference_assistance(raw, sample_id))
        selections[sample_id] = selection
    missing = sorted(private_rows.keys() - selections.keys())
    if missing:
        raise ValueError(f"missing exact raw selections for {len(missing)} samples: {missing[:10]}")
    return selections


def source_span(row: dict[str, Any]) -> dict[str, int]:
    """Original-source span, never the post-comment-removal effective span."""
    result: dict[str, int] = {}
    for field in SPAN_FIELDS:
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{row.get('command_id')}: invalid {field}")
        result[field] = value
    if result["start_byte"] >= result["end_byte"]:
        raise ValueError(f"{row.get('command_id')}: empty or reversed source byte span")
    if result["start_line"] < 1 or result["end_line"] < result["start_line"]:
        raise ValueError(f"{row.get('command_id')}: invalid source line span")
    return result


def command_record(row: dict[str, Any]) -> dict[str, Any]:
    """Proof-text-free command identity; names are surface parser names only."""
    command_id = _string(row, "command_id", "command row")
    source_file = _relpath(_string(row, "source_file", command_id))
    raw_text = row.get("raw_text")
    effective_text = row.get("effective_raw_text")
    if not isinstance(raw_text, str) or not isinstance(effective_text, str):
        raise ValueError(f"{source_file}::{command_id}: missing raw/effective text")
    if row.get("name") is not None and not isinstance(row["name"], str):
        raise ValueError(f"{source_file}::{command_id}: invalid parser surface name")
    parser_kind = _string(row, "kind", command_id)
    return {
        "command_id": command_id,
        "command_uid": f"{source_file}::{command_id}",
        "source_file": source_file,
        "source_module": str(row.get("source_module") or ""),
        "parser_kind": parser_kind,
        "effective_kind": effective_context_kind(parser_kind, effective_text.strip()),
        "syntax_kind": str(row.get("syntax_kind") or ""),
        "original_name": row.get("name"),
        "name_identity": "parser_surface_name_not_elaborated_declaration_identity",
        "source_span": source_span(row),
        "raw_text_sha256": _sha256(raw_text),
        "effective_raw_text_sha256": _sha256(effective_text),
        "context_text_sha256": _sha256(effective_text.strip()),
    }


class SourceRepository:
    """Validated original source records, with a bounded read-only file cache.

    Paths may be absolute or relative to ``root``. ``commands`` preserves the
    original parser order; its zero-based ordinal is ``source_order`` in public
    enrichment. Returned original rows are not modified by this module.
    """

    def __init__(
        self, root: str | Path, command_root: str | Path, mathlib_root: str | Path
    ) -> None:
        self.root = Path(root).resolve()
        command_root, mathlib_root = Path(command_root), Path(mathlib_root)
        self.command_root = command_root if command_root.is_absolute() else self.root / command_root
        self.mathlib_root = mathlib_root if mathlib_root.is_absolute() else self.root / mathlib_root
        self._cache: OrderedDict[str, tuple[list[dict[str, Any]], bytes]] = OrderedDict()
        self.cache_size = 64

    def _load(self, relpath: str) -> tuple[list[dict[str, Any]], bytes]:
        relpath = _relpath(relpath)
        cached = self._cache.pop(relpath, None)
        if cached is not None:
            self._cache[relpath] = cached
            return cached
        source = (self.mathlib_root / relpath).read_bytes()
        rows = list(iter_jsonl(self.command_root / Path(relpath).with_suffix(".jsonl")))
        seen: set[str] = set()
        previous_start = -1
        for row in rows:
            command_id = _string(row, "command_id", relpath)
            if command_id in seen:
                raise ValueError(f"{relpath}: duplicate command_id {command_id}")
            seen.add(command_id)
            if row.get("source_file") != relpath:
                raise ValueError(f"{relpath}::{command_id}: source_file mismatch")
            span = source_span(row)
            start, end = span["start_byte"], span["end_byte"]
            if start < previous_start:
                raise ValueError(f"{relpath}: parser source order is not increasing")
            previous_start = start
            if end > len(source):
                raise ValueError(f"{relpath}::{command_id}: source byte span exceeds file")
            raw_text = row.get("raw_text")
            if not isinstance(raw_text, str) or source[start:end] != raw_text.encode("utf-8"):
                raise ValueError(f"{relpath}::{command_id}: original source span mismatch")
            if not isinstance(row.get("effective_raw_text"), str):
                raise ValueError(f"{relpath}::{command_id}: missing effective_raw_text")
        result = (rows, source)
        self._cache[relpath] = result
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def commands(self, relpath: str) -> list[dict[str, Any]]:
        return self._load(relpath)[0]

    def source_bytes(self, relpath: str) -> bytes:
        return self._load(relpath)[1]


def _public_context(row: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = row["sample_id"]
    mathematical = row.get("mathematical_context")
    environment = row.get("environment_context")
    full = row.get("full_context")
    if not all(isinstance(value, list) for value in (mathematical, environment, full)):
        raise ValueError(f"{sample_id}: public context fields must be lists")
    merged = list(mathematical) + list(environment)
    for item in merged:
        if not isinstance(item, dict):
            raise ValueError(f"{sample_id}: context item is not an object")
        position = item.get("position")
        if not isinstance(position, int) or isinstance(position, bool):
            raise ValueError(f"{sample_id}: invalid context position")
        _string(item, "kind", sample_id)
        _string(item, "text", sample_id)
    merged.sort(key=lambda item: item["position"])
    if [item["position"] for item in merged] != list(range(len(merged))):
        raise ValueError(f"{sample_id}: context positions are not contiguous")
    if merged != full:
        raise ValueError(f"{sample_id}: full_context differs from merged context")
    if any(item["kind"] not in DIRECT_MATHEMATICAL_KINDS for item in mathematical):
        raise ValueError(f"{sample_id}: mathematical context category mismatch")
    if any(item["kind"] in DIRECT_MATHEMATICAL_KINDS for item in environment):
        raise ValueError(f"{sample_id}: environment context category mismatch")
    return merged


def enrich_target(
    public: dict[str, Any], private: dict[str, Any],
    selection: dict[str, Any], repo: SourceRepository,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return enriched public data, proof-free mapping, and selection history."""
    sample_id = _string(public, "sample_id", "public row")
    if private.get("sample_id") != sample_id or selection.get("sample_id") != sample_id:
        raise ValueError(f"{sample_id}: public/private/selection ID mismatch")
    for field in ("statement", "target"):
        if _string(public, field, sample_id) != _string(private, field, sample_id):
            raise ValueError(f"{sample_id}: public/private {field} mismatch")
    metadata = private.get("metadata") or {}
    if public.get("proof_family") != metadata.get("proof_family"):
        raise ValueError(f"{sample_id}: public/private proof_family mismatch")
    if public.get("split") not in PUBLIC_SPLITS:
        raise ValueError(f"{sample_id}: invalid current public split")
    relpath, target_id = _private_key(private)
    if selection.get("source_file") != relpath or selection.get("target_command_id") != target_id:
        raise ValueError(f"{sample_id}: selection target tuple mismatch")
    private_context = _private_context(private)
    if selection.get("original_context_text") != private_context:
        raise ValueError(f"{sample_id}: selection/private context mismatch")
    commands = repo.commands(relpath)
    by_id = {row["command_id"]: row for row in commands}
    orders = {row["command_id"]: ordinal for ordinal, row in enumerate(commands)}
    if target_id not in by_id:
        raise ValueError(f"{sample_id}: missing target command {target_id}")
    target = by_id[target_id]
    if target["kind"] not in {"theorem", "lemma", "example"}:
        raise ValueError(f"{sample_id}: unsupported target kind {target['kind']!r}")
    if private.get("kind") and private["kind"] != target["kind"]:
        raise ValueError(f"{sample_id}: private/parser target kind mismatch")
    for metadata_field, command_field in (
        ("target_start_line", "start_line"), ("target_end_line", "end_line")
    ):
        expected = metadata.get(metadata_field)
        if expected is not None and expected != target[command_field]:
            raise ValueError(f"{sample_id}: target line mismatch")
    source_range = metadata.get("source_range")
    if source_range is not None:
        if not isinstance(source_range, dict):
            raise ValueError(f"{sample_id}: invalid private source_range")
        for endpoint, command_field in (("start", "start_line"), ("finish", "end_line")):
            coordinate = source_range.get(endpoint)
            if coordinate is not None:
                if not isinstance(coordinate, dict):
                    raise ValueError(f"{sample_id}: invalid private source_range {endpoint}")
                expected = coordinate.get("line")
                if expected is not None and expected != target[command_field]:
                    raise ValueError(f"{sample_id}: private source_range line mismatch")
    kept_ids = _command_ids(selection.get("kept_command_ids"), sample_id)
    candidate_ids = _command_ids(selection.get("candidate_command_ids"), sample_id)
    if not set(kept_ids).issubset(candidate_ids):
        raise ValueError(f"{sample_id}: selected command absent from candidate IDs")
    for command_id in candidate_ids:
        if command_id not in by_id:
            raise ValueError(f"{sample_id}: missing candidate command {command_id}")
    selected: list[dict[str, Any]] = []
    for command_id in kept_ids:
        command = by_id[command_id]
        if command["start_byte"] >= target["start_byte"] or command["end_byte"] > target["start_byte"]:
            raise ValueError(f"{sample_id}: selected context is not strictly before target")
        selected.append(command)
    selected.sort(key=lambda row: row["start_byte"])
    original_items = _public_context(public)
    if len(original_items) != len(selected):
        raise ValueError(f"{sample_id}: selected/public context length mismatch")
    items: list[dict[str, Any]] = []
    context_mapping: list[dict[str, Any]] = []
    for item, command in zip(original_items, selected):
        text = command["effective_raw_text"].strip()
        effective_kind = effective_context_kind(command["kind"], text)
        if item["text"] != text:
            raise ValueError(f"{sample_id}: exact context text mismatch at {item['position']}")
        if item["kind"] != effective_kind:
            raise ValueError(f"{sample_id}: effective context kind mismatch at {item['position']}")
        record = command_record(command)
        enriched_item = dict(item)
        enriched_item.update({
            "command_id": record["command_id"],
            "command_uid": record["command_uid"],
            "source_order": orders[command["command_id"]],
            "source_span": record["source_span"],
            "parser_kind": command["kind"],
            "effective_kind": effective_kind,
        })
        items.append(enriched_item)
        context_mapping.append({
            "position": item["position"],
            "source_order": orders[command["command_id"]],
            **record,
        })
    if "\n\n".join(item["text"] for item in items) != private_context:
        raise ValueError(f"{sample_id}: original context reconstruction mismatch")
    target_record = command_record(target)
    original_name = private.get("full_name") or target.get("name")
    enriched = dict(public)
    enriched.update({
        "target_id": sample_id,
        "target_command_id": target_id,
        "target_command_uid": target_record["command_uid"],
        "source_file": relpath,
        "source_module": target_record["source_module"] or str(private.get("module") or ""),
        "target_kind": target["kind"],
        "original_name": original_name,
        "target_source_span": target_record["source_span"],
        "mathematical_context": [items[item["position"]] for item in public["mathematical_context"]],
        "environment_context": [items[item["position"]] for item in public["environment_context"]],
        "full_context": items,
    })
    mapping = {
        "sample_id": sample_id,
        "target_id": sample_id,
        "split": public["split"],
        "proof_family": public["proof_family"],
        "target_command_id": target_id,
        "target_command_uid": target_record["command_uid"],
        "source_file": relpath,
        "source_module": enriched["source_module"],
        "target_kind": target["kind"],
        "original_name": original_name,
        "target_source_order": orders[target_id],
        "target_source_span": target_record["source_span"],
        "source_sha256": hashlib.sha256(repo.source_bytes(relpath)).hexdigest(),
        "target_raw_text_sha256": target_record["raw_text_sha256"],
        "target_effective_raw_text_sha256": target_record["effective_raw_text_sha256"],
        "statement_sha256": _sha256(public["statement"]),
        "proof_sha256": _sha256(public["target"]),
        "original_context_sha256": _sha256(private_context),
        "mathlib_version": str(metadata.get("mathlib4_version") or ""),
        "context_commands": context_mapping,
    }
    # Whitelist provenance fields even if a caller supplies a richer raw row.
    history = {
        key: selection[key] for key in (
            "sample_id", "source_file", "target_command_id",
            "candidate_command_ids", "kept_command_ids",
            "raw_selection_file", "raw_selection_row",
            "selection_provider", "selection_model", "selection_method",
            "selection_reason", "verified_by_lean", "mathlib4_version",
            *CONTEXT_FILTER_FIELDS, "context_filter_history_availability",
            "reference_assisted", "reference_assisted_method",
            "reference_assisted_availability",
        ) if key in selection
    }
    history.update({
        "target_id": sample_id,
        "split": public["split"],
        "target_command_uid": target_record["command_uid"],
        "ordered_kept_command_ids": [row["command_id"] for row in selected],
        "ordered_kept_command_uids": [f"{relpath}::{row['command_id']}" for row in selected],
        "original_context_sha256": _sha256(private_context),
        "original_prepared_split": private.get("split"),
    })
    return enriched, mapping, history
