"""Create gold proof candidates for mandatory Lean-harness sanity checking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_proof_training.common.io import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as output:
        for row in iter_jsonl(args.dataset_file):
            output.write(
                json.dumps(
                    {
                        "sample_id": str(row["sample_id"]),
                        "candidate_id": 0,
                        "proof": str(row["target"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
