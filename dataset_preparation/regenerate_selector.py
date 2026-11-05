"""Rebuild selector pairs for existing fixed IDs after a syntactic scope fix.

Reads only the existing proof-free manifest targets, source-command catalogs,
and independently exported target-type evidence.  No resampling, benchmark gold
proofs, private proof descriptors, model results, or dependency-based membership
selection occurs.  The root orchestrator refreshes release hashes afterwards.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from .selector import TARGET_KEYS, RULE_HASH, RULES, build_pair


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def regenerate(root: Path) -> dict:
    folder = root / "selector"
    plans = []
    all_ids = set()
    for manifest_name, output_name, split in (
            ("frozen_test_manifest.json", "pairs.jsonl", "test"),
            ("validation_manifest.json", "validation_pairs.jsonl", "validation")):
        manifest_path = folder / manifest_name
        manifest = json.loads(manifest_path.read_text())
        ids = manifest["target_ids"]
        if len(ids) != len(set(ids)) or len(ids) != manifest["n"]:
            raise ValueError("fixed manifest has inconsistent membership")
        lookup = {}
        for record in manifest["targets"]:
            target = {key: record[key] for key in TARGET_KEYS if key in record}
            sid = str(target.get("sample_id") or target.get("target_id"))
            if sid in lookup or target.get("split") != split:
                raise ValueError("fixed manifest target identity/split mismatch")
            lookup[sid] = target
        if set(lookup) != set(ids) or all_ids & set(ids):
            raise ValueError("fixed test/validation membership mismatch or overlap")
        all_ids.update(ids)
        old_rows = {row["sample_id"]: row for row in rows(folder / output_name)}
        if list(old_rows) != ids:
            raise ValueError("old pair membership differs from fixed manifest order")
        plans.append((manifest_path, digest(manifest_path), output_name, ids, lookup, old_rows))
    type_path = folder / "target_type_dependencies.jsonl"
    evidence = {row["sample_id"]: row for row in rows(type_path) if row["sample_id"] in all_ids}
    if set(evidence) != all_ids:
        raise ValueError("fixed targets are missing independent target-type evidence")
    catalogs = {}
    report = {"schema_version": "fixed-selector-scope-regeneration-v1",
        "regeneration_reason": "correct_noncomputable_named_section_scope_matching",
        "target_membership_resampled": False, "benchmark_or_reference_proofs_read": False,
        "model_results_read": False, "type_evidence_sha256": digest(type_path),
        "selector_source_sha256": digest(Path(__file__).with_name("selector.py")),
        "rule_revision": RULES["revision"], "rule_revision_sha256": RULE_HASH, "files": {}}
    built = []
    for manifest_path, manifest_hash, output_name, ids, lookup, old_rows in plans:
        output_path = folder / output_name
        before_hash = digest(output_path)
        counts, new_rows = Counter(), []
        for sid in ids:
            target = lookup[sid]
            file = target["source_file"]
            relative = Path(file)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe source path")
            if file not in catalogs:
                catalogs[file] = list(rows(root / "command_mapping" / "source_commands" / relative.with_suffix(".jsonl")))
            pair = build_pair(target, catalogs[file], evidence[sid])
            old = old_rows[sid]
            for field in ("baseline_ranking", "dependency_prioritized_ranking", "target_type_dependencies"):
                if pair[field] != old[field]:
                    raise ValueError(f"scope-only rebuild changed {field}: {sid}")
            if [c["command_id"] for c in pair["candidate_pool"]] != [c["command_id"] for c in old["candidate_pool"]]:
                raise ValueError(f"scope-only rebuild changed candidate pool: {sid}")
            counts[pair["EC_scope_stack_status"]] += 1
            counts["scope_status_changed"] += pair["EC_scope_stack_status"] != old["EC_scope_stack_status"]
            new_rows.append(pair)
        built.append((output_path, new_rows))
        report["files"][output_name] = {"targets": len(ids), "manifest_sha256_unchanged": manifest_hash,
            "before_sha256": before_hash, "counts": dict(counts), "pool_rankings_and_type_denominators_unchanged": True}
    # All targets/denominators/rankings are checked before either file is changed.
    for path, new_rows in built:
        temporary = path.with_suffix(".scopefix.jsonl")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
        report["files"][path.name]["after_sha256"] = digest(path)
    for manifest_path, old_hash, *_ in plans:
        if digest(manifest_path) != old_hash:
            raise AssertionError("manifest unexpectedly changed")
    report_path = folder / "scope_regeneration_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("leancontext_dataset"))
    args = parser.parse_args()
    print(json.dumps(regenerate(args.dataset_root), ensure_ascii=False, indent=2))
