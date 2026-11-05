from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def mask_lean_comments_bytes(data: bytes) -> bytes:
    """
    Mask Lean comments with spaces while preserving byte length and newlines.

    Handles:
    - line comments: -- ...
    - nested block comments: /- ... -/
    - doc comments/module docs: /-- ... -/, /-! ... -/
    - ordinary string literals: " ... "

    This is for parser input only. It does not overwrite original files.
    """
    out = bytearray(data)
    n = len(data)
    i = 0

    NORMAL = 0
    STRING = 1
    state = NORMAL

    while i < n:
        b = data[i]

        if state == STRING:
            if b == ord("\\"):
                i += 2
                continue
            if b == ord('"'):
                state = NORMAL
                i += 1
                continue
            i += 1
            continue

        if b == ord('"'):
            state = STRING
            i += 1
            continue

        # Line comment: -- ...
        if i + 1 < n and data[i] == ord("-") and data[i + 1] == ord("-"):
            while i < n and data[i] not in (ord("\n"), ord("\r")):
                out[i] = ord(" ")
                i += 1
            continue

        # Nested block comment: /- ... -/
        if i + 1 < n and data[i] == ord("/") and data[i + 1] == ord("-"):
            depth = 1
            out[i] = ord(" ")
            out[i + 1] = ord(" ")
            i += 2

            while i < n and depth > 0:
                # Nested start
                if i + 1 < n and data[i] == ord("/") and data[i + 1] == ord("-"):
                    out[i] = ord(" ")
                    out[i + 1] = ord(" ")
                    depth += 1
                    i += 2
                    continue

                # End
                if i + 1 < n and data[i] == ord("-") and data[i + 1] == ord("/"):
                    out[i] = ord(" ")
                    out[i + 1] = ord(" ")
                    depth -= 1
                    i += 2
                    continue

                # Preserve line structure
                if data[i] in (ord("\n"), ord("\r")):
                    out[i] = data[i]
                else:
                    out[i] = ord(" ")
                i += 1

            continue

        i += 1

    return bytes(out)


def strip_lean_comments_text(text: str) -> str:
    """
    Remove Lean comments from extracted command text by replacing every
    comment token with the empty string.

    This is intentionally a post-processing step. Parser input should remain
    the original Lean source so command spans and parser scopes are computed
    from the real file.
    """
    out: list[str] = []
    n = len(text)
    i = 0
    in_string = False

    while i < n:
        ch = text[i]

        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if i + 1 < n and text[i] == "-" and text[i + 1] == "-":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue

        if i + 1 < n and text[i] == "/" and text[i + 1] == "-":
            depth = 1
            i += 2
            while i < n and depth > 0:
                if i + 1 < n and text[i] == "/" and text[i + 1] == "-":
                    depth += 1
                    i += 2
                    continue
                if i + 1 < n and text[i] == "-" and text[i + 1] == "/":
                    depth -= 1
                    i += 2
                    continue
                i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def normalize_effective_raw_text(text: str) -> str:
    text = strip_lean_comments_text(text)
    text = re.sub(r"\n(?:[ \t\r]*\n)+", "\n", text)
    text = text.strip(" \t\r\n")
    if not text:
        return ""
    return text + "\n"


def module_name_from_relpath(rel: Path) -> str:
    if rel.suffix == ".lean":
        rel = rel.with_suffix("")
    return ".".join(rel.parts)


def output_path_for(out_root: Path, rel: Path) -> Path:
    return out_root / rel.with_suffix(".jsonl")


def tmp_commentfree_path_for(tmp_root: Path, rel: Path) -> Path:
    return tmp_root / rel.with_suffix(".commentfree.lean")


def should_skip_lean_file(path: Path) -> bool:
    name = path.name
    if name.endswith(".commentfree.lean"):
        return True
    return False


def iter_lean_files(
    mathlib_root: Path,
    include_archive: bool,
    path_include: str | None,
) -> list[Path]:
    roots = [mathlib_root / "Mathlib"]
    if include_archive:
        roots.append(mathlib_root / "Archive")
        roots.append(mathlib_root / "Counterexamples")

    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.lean"):
            if should_skip_lean_file(p):
                continue
            rel = p.relative_to(mathlib_root)
            if path_include and path_include not in rel.as_posix():
                continue
            files.append(p)

    files.sort(key=lambda p: p.relative_to(mathlib_root).as_posix())
    return files


def run_extractor_on_file(
    *,
    mathlib_root: Path,
    extractor: Path,
    source_file: Path,
    rel: Path,
    tmp_root: Path,
    out_root: Path,
    timeout: int,
    force: bool,
    keep_commentfree: bool,
) -> dict[str, Any]:
    out_path = output_path_for(out_root, rel)
    tmp_path = tmp_commentfree_path_for(tmp_root, rel)
    log_path = out_root / "_logs" / rel.with_suffix(".stderr.txt")

    started = time.time()

    if out_path.exists() and not force:
        return {
            "status": "skipped_existing",
            "source_file": rel.as_posix(),
            "output_file": out_path.as_posix(),
            "elapsed_sec": 0.0,
            "num_commands": None,
        }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "lake",
            "env",
            "lean",
            "--run",
            str(extractor),
            str(source_file),
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(mathlib_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )

        stderr = proc.stderr or ""
        if stderr.strip():
            log_path.write_text(stderr, encoding="utf-8")

        if proc.returncode != 0:
            return {
                "status": "failed",
                "source_file": rel.as_posix(),
                "output_file": out_path.as_posix(),
                "stderr_file": log_path.as_posix(),
                "returncode": proc.returncode,
                "elapsed_sec": round(time.time() - started, 3),
            }

        source_rel = rel.as_posix()
        source_abs = str(source_file)
        module_name = module_name_from_relpath(rel)

        num_commands = 0
        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")

        with tmp_out.open("w", encoding="utf-8") as f:
            for line_no, line in enumerate(proc.stdout.splitlines(), start=1):
                if not line.strip():
                    continue

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Extractor produced non-JSON line for {source_rel} line {line_no}: {e}\n"
                        f"line={line[:300]}"
                    ) from e

                raw_text = str(rec.get("raw_text") or "")
                effective_raw_text = normalize_effective_raw_text(raw_text)
                rec["effective_raw_text"] = effective_raw_text

                start_byte = int(rec.get("start_byte", 0))
                rec["effective_end_byte"] = start_byte + len(
                    str(effective_raw_text).encode("utf-8")
                )

                rec["source_file"] = source_rel
                rec["source_file_abs"] = source_abs
                rec["source_module"] = module_name
                rec["source_root"] = str(mathlib_root)
                rec["comment_policy"] = {
                    "parser_input": "original_source",
                    "effective_raw_text": "comments_replaced_with_empty_string_after_parsing",
                    "original_source_not_modified": True,
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                num_commands += 1

        tmp_out.replace(out_path)

        return {
            "status": "ok",
            "source_file": source_rel,
            "output_file": out_path.as_posix(),
            "num_commands": num_commands,
            "elapsed_sec": round(time.time() - started, 3),
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "source_file": rel.as_posix(),
            "output_file": out_path.as_posix(),
            "elapsed_sec": round(time.time() - started, 3),
            "timeout_sec": timeout,
        }
    except Exception as e:
        return {
            "status": "failed_exception",
            "source_file": rel.as_posix(),
            "output_file": out_path.as_posix(),
            "elapsed_sec": round(time.time() - started, 3),
            "error": repr(e),
        }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_failed(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch extract Lean command spans to JSONL, preserving mathlib directory structure."
    )
    parser.add_argument("--mathlib-root", required=True, help="Path to mathlib4 root")
    parser.add_argument("--extractor", required=True, help="Path to ExtractCommandSpans.lean")
    parser.add_argument("--out-root", required=True, help="Output root for mirrored JSONL files")
    parser.add_argument("--tmp-root", default=None, help="Temporary root for comment-masked files")
    parser.add_argument("--include-archive", action="store_true", help="Also scan Archive/ and Counterexamples/")
    parser.add_argument("--path-include", default=None, help="Only process files whose relative path contains this string")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel Lean extractor processes")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per Lean file in seconds")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output JSONL files")
    parser.add_argument("--keep-commentfree", action="store_true", help="Keep temporary comment-masked .lean files")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first failed file")
    args = parser.parse_args()

    mathlib_root = Path(args.mathlib_root).resolve()
    extractor = Path(args.extractor).resolve()
    out_root = Path(args.out_root).resolve()
    tmp_root = Path(args.tmp_root).resolve() if args.tmp_root else out_root / "_commentfree_tmp"

    if not mathlib_root.exists():
        raise FileNotFoundError(f"mathlib root does not exist: {mathlib_root}")
    if not extractor.exists():
        raise FileNotFoundError(f"extractor does not exist: {extractor}")
    if not (mathlib_root / "lakefile.lean").exists():
        raise FileNotFoundError(f"not a mathlib/lake root, missing lakefile.lean: {mathlib_root}")
    if not (mathlib_root / "lean-toolchain").exists():
        raise FileNotFoundError(f"missing lean-toolchain: {mathlib_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    files = iter_lean_files(
        mathlib_root=mathlib_root,
        include_archive=args.include_archive,
        path_include=args.path_include,
    )

    manifest: dict[str, Any] = {
        "mathlib_root": str(mathlib_root),
        "extractor": str(extractor),
        "out_root": str(out_root),
        "tmp_root": str(tmp_root),
        "include_archive": args.include_archive,
        "path_include": args.path_include,
        "jobs": args.jobs,
        "timeout": args.timeout,
        "num_files_discovered": len(files),
        "started_at_unix": time.time(),
        "results": [],
    }

    failed_path = out_root / "_failed.jsonl"
    if failed_path.exists() and args.force:
        failed_path.unlink()

    print(f"[batch] mathlib_root = {mathlib_root}")
    print(f"[batch] extractor    = {extractor}")
    print(f"[batch] out_root     = {out_root}")
    print(f"[batch] tmp_root     = {tmp_root}")
    print(f"[batch] files        = {len(files)}")
    print(f"[batch] jobs         = {args.jobs}")

    ok = 0
    skipped = 0
    failed = 0

    def submit_one(p: Path) -> dict[str, Any]:
        rel = p.relative_to(mathlib_root)
        return run_extractor_on_file(
            mathlib_root=mathlib_root,
            extractor=extractor,
            source_file=p,
            rel=rel,
            tmp_root=tmp_root,
            out_root=out_root,
            timeout=args.timeout,
            force=args.force,
            keep_commentfree=args.keep_commentfree,
        )

    if args.jobs <= 1:
        for idx, p in enumerate(files, start=1):
            rel = p.relative_to(mathlib_root).as_posix()
            print(f"[{idx}/{len(files)}] {rel}")
            rec = submit_one(p)
            manifest["results"].append(rec)

            if rec["status"] == "ok":
                ok += 1
            elif rec["status"] == "skipped_existing":
                skipped += 1
            else:
                failed += 1
                append_failed(failed_path, rec)
                print(f"[failed] {rel}: {rec}")
                if args.fail_fast:
                    break
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(submit_one, p): p for p in files}
            for idx, fut in enumerate(as_completed(futs), start=1):
                p = futs[fut]
                rel = p.relative_to(mathlib_root).as_posix()

                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "status": "failed_future_exception",
                        "source_file": rel,
                        "error": repr(e),
                    }

                manifest["results"].append(rec)

                if rec["status"] == "ok":
                    ok += 1
                    print(f"[{idx}/{len(files)} ok] {rel}")
                elif rec["status"] == "skipped_existing":
                    skipped += 1
                    print(f"[{idx}/{len(files)} skipped] {rel}")
                else:
                    failed += 1
                    append_failed(failed_path, rec)
                    print(f"[{idx}/{len(files)} failed] {rel}: {rec}")
                    if args.fail_fast:
                        # ThreadPool cannot reliably cancel already running Lean processes,
                        # but we stop collecting further work as soon as possible.
                        break

    manifest["finished_at_unix"] = time.time()
    manifest["summary"] = {
        "ok": ok,
        "skipped_existing": skipped,
        "failed": failed,
        "processed_or_seen": ok + skipped + failed,
        "num_files_discovered": len(files),
    }

    write_json(out_root / "_manifest.json", manifest)

    print("[batch] done")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
