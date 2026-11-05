from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from extract_data.context_pipeline import (
    anonymize_target_header,
    join_lean_parts,
    split_target_text,
)


def repair_record(rec: dict[str, Any]) -> dict[str, Any]:
    target_text = str(rec.get("target_text") or "")
    target_split = split_target_text(target_text)
    target_kind = rec.get("target_kind") or rec.get("kind")
    target_name = rec.get("theorem_name")
    anonymized_header = anonymize_target_header(
        target_split["target_header_text"],
        target_kind,
        target_name,
    )
    context_text = str(rec.get("context_text") or "")
    proof_end_marker = str(rec.get("proof_end_marker") or "<END_OF_PROOF>")

    out = dict(rec)
    out["target_header_text"] = target_split["target_header_text"]
    out["target_header_text_anonymized"] = anonymized_header
    out["target_body_text"] = target_split["target_body_text"]
    out["target_split_found"] = target_split["target_split_found"]
    out["target_split_method"] = target_split["target_split_method"]
    out["sft_input_text"] = join_lean_parts([context_text, anonymized_header])
    out["sft_input_text_without_context"] = join_lean_parts([anonymized_header])
    out["sft_target_text"] = target_split["target_body_text"] + proof_end_marker
    out["split_repair_method"] = "recomputed_from_target_text_ignore_attribute_assignment"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair SFT header/body splits using target_text and the robust splitter."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0
    failed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
                old_target = rec.get("sft_target_text")
                out = repair_record(rec)
                if out.get("sft_target_text") != old_target:
                    changed += 1
            except Exception as e:
                failed += 1
                out = {
                    "repair_status": "failed",
                    "line_no": line_no,
                    "error": repr(e),
                    "raw_line": line.rstrip("\n"),
                }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            if args.progress_every and total % args.progress_every == 0:
                print(f"[progress] total={total} changed={changed} failed={failed}")

    print(
        json.dumps(
            {
                "input": input_path.as_posix(),
                "output": output_path.as_posix(),
                "total": total,
                "changed_records": changed,
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
