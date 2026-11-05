from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def record_key(rec: dict[str, Any]) -> str | None:
    source_file = rec.get("lean_file_name") or rec.get("source_file")
    target_command_id = rec.get("target_command_id")
    if not source_file or not target_command_id:
        return None
    return f"{source_file}::{target_command_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate theorem-context JSONL by target identity.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="last",
        help="Which duplicate record to keep. Default keeps the latest appended record.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: OrderedDict[str, str] = OrderedDict()
    passthrough: list[str] = []
    total = 0
    invalid = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                passthrough.append(line.rstrip("\n"))
                continue
            key = record_key(rec)
            if key is None:
                passthrough.append(json.dumps(rec, ensure_ascii=False))
                continue
            if args.keep == "first" and key in records:
                continue
            records[key] = json.dumps(rec, ensure_ascii=False)

    with output_path.open("w", encoding="utf-8") as out:
        for line in passthrough:
            out.write(line + "\n")
        for line in records.values():
            out.write(line + "\n")

    print(
        json.dumps(
            {
                "input": input_path.as_posix(),
                "output": output_path.as_posix(),
                "total_nonempty_lines": total,
                "unique_target_records": len(records),
                "passthrough_records": len(passthrough),
                "duplicates_removed": total - len(records) - len(passthrough),
                "invalid_json_lines": invalid,
                "keep": args.keep,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
