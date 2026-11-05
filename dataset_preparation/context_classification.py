#!/usr/bin/env python3
"""Build one merged semantic/environment Context dataset.

The positive Context in ``matched_dataset`` is the authoritative input.  The
old train/val/test files are read only as storage shards and are merged into a
single ``all.jsonl``.  No data split is retained in the public rows.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


SOURCE_SHARDS = ("train", "val", "test")
PUBLIC_FIELDS = (
    "sample_id",
    "proof_family",
    "statement",
    "target",
    "mathematical_context",
    "environment_context",
)
CONTEXT_ITEM_FIELDS = ("position", "kind", "text")

# Commands that introduce propositions, reusable mathematical objects, data
# types, structures/classes, instances, or aliases.  ``export`` is deliberately
# environmental because it changes name visibility rather than mathematical
# content.
DIRECT_MATHEMATICAL_KINDS = {
    "theorem",
    "lemma",
    "axiom",
    "def",
    "abbrev",
    "opaque",
    "constant",
    "structure",
    "class",
    "inductive",
    "instance",
    "local_instance",
    "alias",
}

# Lean's parser sometimes records the outer command, e.g.
# ``open Classical in noncomputable def ...``.  For these wrappers, classify by
# the declaration introduced inside the wrapper.
DECLARATION_WRAPPER_KINDS = {
    "open",
    "open_scoped",
    "set_option",
    "variable",
    "section",
    "namespace",
    "local",
    "scoped",
}
DECLARATION_WORDS = (
    "theorem",
    "lemma",
    "axiom",
    "def",
    "abbrev",
    "opaque",
    "constant",
    "structure",
    "class",
    "inductive",
    "instance",
    "alias",
)
WRAPPED_DECLARATION_RE = re.compile(
    r"(?:^|\bin\s+|\n\s*)"
    r"(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:private|protected|noncomputable|unsafe|local|scoped)\s+)*"
    r"(?P<kind>" + "|".join(DECLARATION_WORDS) + r")\b",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=root / "sft/context_benchmark/matched_dataset",
        help="Existing shards whose positive matched Context is authoritative.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "sft/context_benchmark/semantic_environment_dataset",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            yield row


def source_rows(source_dir: Path) -> Iterator[dict[str, Any]]:
    for shard in SOURCE_SHARDS:
        yield from read_jsonl(source_dir / f"{shard}.jsonl")


def effective_context_kind(parser_kind: str, text: str) -> str:
    """Return the effective declaration/environment kind for classification."""

    if parser_kind in DIRECT_MATHEMATICAL_KINDS:
        return parser_kind
    if parser_kind not in DECLARATION_WRAPPER_KINDS:
        return parser_kind
    match = WRAPPED_DECLARATION_RE.search(text)
    return match.group("kind") if match else parser_kind


def is_mathematical_context(parser_kind: str, text: str) -> bool:
    return effective_context_kind(parser_kind, text) in DIRECT_MATHEMATICAL_KINDS


def partition_context(
    items: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mathematical: list[dict[str, Any]] = []
    environment: list[dict[str, Any]] = []
    for position, item in enumerate(items):
        parser_kind = str(item.get("kind") or "unknown")
        text = str(item.get("text") or "").strip()
        if not text:
            raise ValueError(f"empty Context command at position {position}")
        effective_kind = effective_context_kind(parser_kind, text)
        public_item = {
            "position": position,
            "kind": effective_kind,
            "text": text,
        }
        if effective_kind in DIRECT_MATHEMATICAL_KINDS:
            mathematical.append(public_item)
        else:
            environment.append(public_item)
    return mathematical, environment


def merge_context(
    mathematical: Iterable[dict[str, Any]],
    environment: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore the original mixed command order from the two public fields."""

    merged = list(mathematical) + list(environment)
    merged.sort(key=lambda item: int(item["position"]))
    return merged


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = args.out_dir / ".all.jsonl.tmp"
    output_path = args.out_dir / "all.jsonl"

    mathematical_kind_counts: Counter[str] = Counter()
    environment_kind_counts: Counter[str] = Counter()
    proof_family_counts: Counter[str] = Counter()
    presence_counts: Counter[str] = Counter()
    source_shard_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    with temporary_path.open("w", encoding="utf-8") as output:
        for source_shard in SOURCE_SHARDS:
            source_path = args.source_dir / f"{source_shard}.jsonl"
            for old_row in read_jsonl(source_path):
                expected_source_fields = (
                    "sample_id",
                    "split",
                    "training_stage",
                    "proof_family",
                    "statement",
                    "target",
                    "contexts",
                )
                if tuple(old_row.keys()) != expected_source_fields:
                    raise ValueError(f"{source_path}: unexpected source schema")
                sample_id = str(old_row["sample_id"])
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate source sample_id: {sample_id}")
                seen_ids.add(sample_id)

                contexts = old_row.get("contexts")
                if not isinstance(contexts, dict) or "matched" not in contexts:
                    raise ValueError(f"source sample has no positive Context: {sample_id}")
                positive = contexts["matched"]
                if not isinstance(positive, list):
                    raise ValueError(f"matched Context is not a list: {sample_id}")
                mathematical, environment = partition_context(positive)
                reconstructed = merge_context(mathematical, environment)
                if [item["position"] for item in reconstructed] != list(range(len(positive))):
                    raise AssertionError(f"non-contiguous Context positions: {sample_id}")
                if [item["text"] for item in reconstructed] != [item["text"] for item in positive]:
                    raise AssertionError(f"Context text/order changed: {sample_id}")

                row = {
                    "sample_id": sample_id,
                    "proof_family": str(old_row["proof_family"]),
                    "statement": str(old_row["statement"]),
                    "target": str(old_row["target"]),
                    "mathematical_context": mathematical,
                    "environment_context": environment,
                }
                if tuple(row.keys()) != PUBLIC_FIELDS:
                    raise AssertionError("unexpected output schema")
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

                source_shard_counts[source_shard] += 1
                proof_family_counts[row["proof_family"]] += 1
                mathematical_kind_counts.update(item["kind"] for item in mathematical)
                environment_kind_counts.update(item["kind"] for item in environment)
                presence_counts["both_nonempty"] += bool(mathematical and environment)
                presence_counts["mathematical_only"] += bool(mathematical and not environment)
                presence_counts["environment_only"] += bool(environment and not mathematical)
                presence_counts["both_empty"] += not mathematical and not environment

    temporary_path.replace(output_path)
    # Remove outputs from the superseded split-based build, while preserving the
    # original matched_dataset source shards.
    removed_split_files: list[str] = []
    for split in SOURCE_SHARDS:
        stale_path = args.out_dir / f"{split}.jsonl"
        if stale_path.exists():
            stale_path.unlink()
            removed_split_files.append(stale_path.name)

    manifest = {
        "schema_version": "lean-semantic-environment-context-merged-v1",
        "dataset_file": "all.jsonl",
        "dataset_is_split": False,
        "public_fields": list(PUBLIC_FIELDS),
        "context_item_fields": list(CONTEXT_ITEM_FIELDS),
        "context_order": (
            "Merge mathematical_context and environment_context, then sort by position."
        ),
        "context_categories": {
            "mathematical_context": (
                "Commands introducing propositions, definitions, constants, data/structure/class "
                "declarations, instances, or aliases. Wrapped declarations are classified by "
                "their inner declaration kind."
            ),
            "environment_context": (
                "Commands controlling Lean scope, variables, namespaces, notation, attributes, "
                "parser/elaborator behavior, options, or name visibility without introducing "
                "new mathematical content."
            ),
            "mathematical_kinds": sorted(DIRECT_MATHEMATICAL_KINDS),
            "export_is_environment": True,
        },
        "removed_fields": [
            "split",
            "training_stage",
            "contexts",
            "matched",
            "random_control",
            "hard_control",
        ],
        "statistics": {
            "total_samples": len(seen_ids),
            "source_shard_counts_before_merge": dict(source_shard_counts),
            "proof_family_counts": dict(sorted(proof_family_counts.items())),
            "mathematical_command_count": sum(mathematical_kind_counts.values()),
            "environment_command_count": sum(environment_kind_counts.values()),
            "mathematical_kind_counts": dict(sorted(mathematical_kind_counts.items())),
            "environment_kind_counts": dict(sorted(environment_kind_counts.items())),
            "sample_context_presence": dict(sorted(presence_counts.items())),
        },
        "provenance": {
            "positive_context_source": str(args.source_dir),
            "source_shards_merged_in_order": list(SOURCE_SHARDS),
            "positive_context_text_and_order_preserved": True,
            "negative_controls_included": False,
            "superseded_split_files_removed": removed_split_files,
        },
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
