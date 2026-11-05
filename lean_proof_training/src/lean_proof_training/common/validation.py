"""Integrity checks for the publishable unified raw dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from lean_proof_training.common.io import iter_jsonl
from lean_proof_training.common.schema import validate_raw_row


def validate_raw_directory(raw_dir: str | Path) -> dict[str, Any]:
    root = Path(raw_dir)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        for row in iter_jsonl(path):
            validate_raw_row(row)
            if row["split"] != split:
                raise ValueError(f"split mismatch for {row['sample_id']}: {row['split']} != {split}")
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id across splits: {sample_id}")
            seen.add(sample_id)
            counts[split] += 1
            families[str(row["proof_family"])] += 1
            stages[str(row["training_stage"])] += 1
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        expected = int(manifest["stats"][f"split_{split}"])
        if counts[split] != expected:
            raise ValueError(f"manifest count mismatch for {split}: {counts[split]} != {expected}")
    return {
        "samples": len(seen),
        "splits": dict(counts),
        "proof_families": dict(sorted(families.items())),
        "training_stages": dict(sorted(stages.items())),
    }
