from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from extract_data.context_pipeline import (
    anonymize_target_header,
    appears_in_text,
    command_text,
    command_key,
    command_meta,
    join_lean_parts,
    merge_commands,
    read_jsonl,
    split_target_text,
)


DEFAULT_FILTER_KINDS = {"lemma", "theorem"}


def resolve_source_jsonl(rec: dict[str, Any], input_root: Path) -> Path:
    source_jsonl = rec.get("source_jsonl")
    if isinstance(source_jsonl, str) and source_jsonl:
        path = Path(source_jsonl)
        if path.exists():
            return path

    lean_file_name = rec.get("lean_file_name") or rec.get("source_file")
    if not isinstance(lean_file_name, str) or not lean_file_name:
        raise ValueError("record is missing lean_file_name/source_file and usable source_jsonl")

    rel = Path(lean_file_name)
    if rel.parts and rel.parts[0] == input_root.name:
        rel = Path(*rel.parts[1:])
    return input_root / rel.with_suffix(".jsonl")


def load_source_records(
    rec: dict[str, Any],
    input_root: Path,
    cache: dict[Path, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    path = resolve_source_jsonl(rec, input_root)
    if path not in cache:
        cache[path] = read_jsonl(path)
    return cache[path]


def has_name_in_statement(cmd: dict[str, Any], target_header_text: str) -> bool:
    name = cmd.get("name")
    return isinstance(name, str) and bool(name) and appears_in_text(name, target_header_text)


def filter_kept_ids(
    *,
    rec: dict[str, Any],
    source_records: list[dict[str, Any]],
    filter_kinds: set[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    by_id = {command_key(cmd): cmd for cmd in source_records}
    target = by_id.get(str(rec.get("target_command_id")))
    if target is not None:
        target_header_text = split_target_text(command_text(target))["target_header_text"]
    else:
        target_header_text = str(rec.get("target_header_text") or "")

    filtered: list[str] = []
    removed: list[dict[str, Any]] = []
    for cid in rec.get("kept_command_ids") or []:
        cid = str(cid)
        cmd = by_id.get(cid)
        if cmd is None:
            continue
        kind = str(cmd.get("kind") or "")
        if kind in filter_kinds and not has_name_in_statement(cmd, target_header_text):
            removed.append(cmd)
            continue
        filtered.append(cid)
    return filtered, removed


def rebuild_record(
    *,
    rec: dict[str, Any],
    source_records: list[dict[str, Any]],
    filtered_keep_ids: list[str],
    removed_commands: list[dict[str, Any]],
    filter_kinds: set[str],
) -> dict[str, Any]:
    by_id = {command_key(cmd): cmd for cmd in source_records}
    target = by_id.get(str(rec.get("target_command_id")))
    if target is None:
        raise ValueError(f"missing target command in source_jsonl: {rec.get('target_command_id')}")

    include_import_mathlib = str(rec.get("context_text") or "").startswith("import Mathlib\n")
    include_closing_ends = bool(rec.get("closing_text"))

    context_text, closing_text, merged_text = merge_commands(
        source_records,
        target,
        filtered_keep_ids,
        include_import_mathlib=include_import_mathlib,
        include_closing_ends=include_closing_ends,
    )

    target_split = split_target_text(command_text(target))
    target_header_text = target_split["target_header_text"]
    target_header_text_anonymized = anonymize_target_header(
        target_header_text, target.get("kind"), target.get("name")
    )

    kept_set = set(filtered_keep_ids)
    kept_commands = [cmd for cmd in source_records if command_key(cmd) in kept_set]

    out = dict(rec)
    out["kept_command_ids_before_context_filter"] = rec.get("kept_command_ids") or []
    out["kept_command_ids"] = filtered_keep_ids
    out["kept_commands"] = [command_meta(cmd) for cmd in kept_commands]
    out["target_text"] = command_text(target)
    out["target_header_text"] = target_header_text
    out["target_header_text_anonymized"] = target_header_text_anonymized
    out["target_body_text"] = target_split["target_body_text"]
    out["target_split_found"] = target_split["target_split_found"]
    out["target_split_method"] = target_split["target_split_method"]
    out["context_text"] = context_text
    out["closing_text"] = closing_text
    out["merged_text"] = merged_text
    out["sft_input_text"] = join_lean_parts([context_text, target_header_text_anonymized])
    out["sft_input_text_without_context"] = join_lean_parts([target_header_text_anonymized])
    out["sft_target_text"] = target_split["target_body_text"] + str(rec.get("proof_end_marker") or "<END_OF_PROOF>")
    out["removed_context_command_ids"] = [command_key(cmd) for cmd in removed_commands]
    out["removed_context_commands"] = [command_meta(cmd) for cmd in removed_commands]
    out["context_filter_method"] = "remove_kept_lemma_theorem_not_named_in_target_statement"
    out["context_filter_kinds"] = sorted(filter_kinds)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove proof-leakage context from generated SFT JSONL by dropping kept "
            "lemma/theorem commands whose names do not appear in the target statement."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--input-root",
        default=(Path(__file__).resolve().parents[1] / "lean_parser" / "command_spans_jsonl" / "Mathlib").as_posix(),
        help="Parser JSONL Mathlib root used when source_jsonl paths are not valid on this machine.",
    )
    parser.add_argument(
        "--filter-kinds",
        default="lemma,theorem",
        help="Comma-separated command kinds to remove when absent from target statement.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    input_root = Path(args.input_root).resolve()
    filter_kinds = {item.strip() for item in args.filter_kinds.split(",") if item.strip()}
    if not filter_kinds:
        filter_kinds = set(DEFAULT_FILTER_KINDS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, list[dict[str, Any]]] = {}

    total = 0
    changed = 0
    removed_total = 0
    failed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
                source_records = load_source_records(rec, input_root, cache)
                filtered_keep_ids, removed_commands = filter_kept_ids(
                    rec=rec,
                    source_records=source_records,
                    filter_kinds=filter_kinds,
                )
                if removed_commands:
                    changed += 1
                    removed_total += len(removed_commands)
                out = rebuild_record(
                    rec=rec,
                    source_records=source_records,
                    filtered_keep_ids=filtered_keep_ids,
                    removed_commands=removed_commands,
                    filter_kinds=filter_kinds,
                )
            except Exception as e:
                failed += 1
                out = {
                    "postprocess_status": "failed",
                    "line_no": line_no,
                    "error": repr(e),
                    "raw_line": line.rstrip("\n"),
                }

            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            if args.progress_every and total % args.progress_every == 0:
                print(
                    f"[progress] total={total} changed={changed} "
                    f"removed={removed_total} failed={failed}"
                )

    print(
        json.dumps(
            {
                "input": input_path.as_posix(),
                "output": output_path.as_posix(),
                "input_root": input_root.as_posix(),
                "total": total,
                "changed_records": changed,
                "removed_context_commands": removed_total,
                "failed": failed,
                "filter_kinds": sorted(filter_kinds),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
