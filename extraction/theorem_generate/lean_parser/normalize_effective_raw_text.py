from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BLANK_LINE_GAP_RE = re.compile(r"\n(?:[ \t\r]*\n)+")


def normalize_effective_raw_text(text: str) -> str:
    """
    Normalize only whitespace introduced around command boundaries.

    Rules:
    - If two newlines appear with only spaces/tabs/CRs between them, keep the
      first newline and remove the whitespace plus the second newline.
    - At the very end, remove all trailing spaces/tabs/CRs/newlines, then keep
      exactly one final newline.

    Whitespace before the first newline and after the second newline is kept.
    """
    text = BLANK_LINE_GAP_RE.sub("\n", text)
    return text.rstrip(" \t\r\n") + "\n"


def iter_jsonl_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*.jsonl") if p.is_file()]
    files.sort(key=lambda p: p.as_posix())
    return files


def normalize_file(path: Path, *, dry_run: bool) -> dict[str, Any]:
    changed_lines = 0
    total_lines = 0
    output_lines: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1

            if not line.strip():
                output_lines.append(line)
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{line_no}: invalid JSONL: {e}") from e

            value = rec.get("effective_raw_text")
            if isinstance(value, str):
                normalized = normalize_effective_raw_text(value)
                if normalized != value:
                    rec["effective_raw_text"] = normalized
                    changed_lines += 1

            output_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")

    if changed_lines and not dry_run:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            f.writelines(output_lines)
        tmp_path.replace(path)

    return {
        "path": path.as_posix(),
        "total_lines": total_lines,
        "changed_lines": changed_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize effective_raw_text in Lean command-span JSONL files in place."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "command_spans_jsonl" / "Mathlib",
        help="Root directory containing JSONL files. Defaults to lean_parser/command_spans_jsonl/Mathlib.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report changes; do not overwrite files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"root is not a directory: {root}")

    files = iter_jsonl_files(root)
    changed_files = 0
    changed_lines = 0
    total_lines = 0

    for path in files:
        result = normalize_file(path, dry_run=args.dry_run)
        total_lines += int(result["total_lines"])
        changed_lines += int(result["changed_lines"])
        if result["changed_lines"]:
            changed_files += 1
            if not args.quiet:
                print(
                    f"[changed] {result['path']} "
                    f"lines={result['changed_lines']}/{result['total_lines']}"
                )

    mode = "dry-run" if args.dry_run else "in-place"
    print(
        json.dumps(
            {
                "mode": mode,
                "root": root.as_posix(),
                "files_seen": len(files),
                "files_changed": changed_files,
                "lines_seen": total_lines,
                "lines_changed": changed_lines,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
