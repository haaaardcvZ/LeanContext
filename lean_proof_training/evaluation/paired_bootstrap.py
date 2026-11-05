"""Paired source-file cluster bootstrap of matched per-theorem metrics.

Example: python -m evaluation.paired_bootstrap --left C0/per_sample_metrics.jsonl
  --right CFull/per_sample_metrics.jsonl --source-index CFull/source_index.jsonl
  --metric pass@32 --output comparison.json
All values are evaluated data, never predicted/synthetic paper results.
"""
from __future__ import annotations
import argparse
import random
from collections import defaultdict
from pathlib import Path
from .io_utils import read_jsonl, write_json


def paired_file_bootstrap(left: dict[str, float], right: dict[str, float],
                          source_files: dict[str, str], *, replicates: int = 2000,
                          seed: int = 2026) -> dict:
    if not left or set(left) != set(right) or not set(left) <= set(source_files):
        raise ValueError("paired bootstrap requires identical nonempty theorem IDs and their source files")
    if replicates < 100:
        raise ValueError("at least100bootstrap replicates required")
    groups: dict[str, list[float]] = defaultdict(list)
    for sid in sorted(left):
        groups[source_files[sid]].append(float(right[sid]) - float(left[sid]))
    clusters = [(sum(groups[name]), len(groups[name])) for name in sorted(groups)]
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values.append(sum(x[0] for x in chosen) / sum(x[1] for x in chosen))
    values.sort()
    def percentile(p: float) -> float:
        index = (len(values)-1)*p
        lo, hi = int(index), min(int(index)+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(index-lo)
    return {"targets": len(left), "source_files": len(groups), "replicates": replicates,
            "seed": seed, "resampling_unit": "source_file; paired conditions",
            "aggregation": "macro theorem mean; variable cluster sizes retained",
            "difference_pp": 100*sum(sum(v) for v in groups.values())/len(left),
            "ci95_pp": [100*percentile(.025), 100*percentile(.975)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("left", "right", "source-index", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--metric", default="pass@32")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    def values(path: Path) -> dict[str, float]:
        rows = list(read_jsonl(path))
        result = {str(r["sample_id"]): float(r[args.metric]) for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate theorem ID")
        return result
    mapping = {str(r["sample_id"]): str(r["relpath"]) for r in read_jsonl(args.source_index)}
    write_json(args.output, paired_file_bootstrap(values(args.left), values(args.right), mapping,
               replicates=args.replicates, seed=args.seed))


if __name__ == "__main__":
    main()
