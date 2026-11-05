#!/usr/bin/env python3
"""Convenient entry points around the actual, separately preserved source code.

No API call is made unless the user explicitly runs match --call-api.
Lexical statement-name filtering is optional and runs after deduplication.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "theorem_generate"


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def invoke(module: str, arguments: list[str]) -> None:
    env = os.environ.copy()
    original_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SOURCE_ROOT) + (os.pathsep + original_path if original_path else "")
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=SOURCE_ROOT,
        env=env,
        check=True,
    )


def require_new(path: Path, *, force: bool = False, resume: bool = False) -> None:
    if path.exists() and not (force or resume):
        raise FileExistsError(f"Refusing to overwrite {path}. Use --force or --resume where supported.")


def paths(args: argparse.Namespace) -> tuple[Path, Path]:
    work = args.work_dir.resolve()
    parser_root = args.parser_root.resolve() if args.parser_root else work / "command_spans_jsonl" / "Mathlib"
    return work, parser_root


def check_failed(path: Path, status_fields: tuple[str, ...]) -> None:
    """Original scripts sometimes exit zero while recording failures in JSONL."""
    failures = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if any(row.get(field) == "failed" for field in status_fields):
                failures += 1
    if failures:
        raise RuntimeError(f"{failures} failed rows recorded in {path}; inspect before using this output.")


def record_identity(row: dict) -> tuple[str, str] | None:
    source = row.get("lean_file_name") or row.get("source_file")
    target = row.get("target_command_id")
    return (str(source), str(target)) if source and target else None


def unresolved_failures(output: Path, failed: Path) -> int:
    """Resume appends failures; an old failure can already have been recovered."""
    successful = set()
    with output.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                identity = record_identity(json.loads(line))
                if identity:
                    successful.add(identity)
    unresolved = set()
    unidentified = 0
    with failed.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            identity = record_identity(json.loads(line))
            if identity is None:
                unidentified += 1
            elif identity not in successful:
                unresolved.add(identity)
    return len(unresolved) + unidentified


def check_classifier(path: Path) -> None:
    with path.open(encoding="utf-8") as stream:
        errors = sum(
            bool(row.get("classifier_error")) or row.get("exclude_reason") == "classifier_exception"
            for row in (json.loads(line) for line in stream if line.strip())
        )
    if errors:
        raise RuntimeError(f"{errors} classifier exceptions recorded in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    parse = stages.add_parser("parse", help="Original Lean frontend and batch command-span extraction")
    parse.add_argument("--mathlib-root", type=Path, required=True)
    parse.add_argument("--work-dir", type=Path, required=True)
    parse.add_argument("--jobs", type=positive, default=1)
    parse.add_argument("--timeout", type=positive, default=300)
    parse.add_argument("--path-include")
    parse.add_argument("--include-archive", action="store_true")
    parse.add_argument("--force", action="store_true")

    for name in ("candidates", "match"):
        stage = stages.add_parser(name)
        stage.add_argument("--work-dir", type=Path, required=True)
        stage.add_argument("--parser-root", type=Path, help="Directory of parser JSONL files, normally .../Mathlib")
        stage.add_argument("--limit", type=positive, default=50 if name == "candidates" else None)
        stage.add_argument("--source-file", action="append", default=[])
        stage.add_argument("--recursive-depth", type=nonnegative, default=2)
        stage.add_argument("--max-dependency-candidates", type=positive, default=60)
        stage.add_argument("--max-chars-per-command", type=positive, default=5000)
        stage.add_argument("--flash-model", default=os.environ.get("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"))
        stage.add_argument("--pro-model", default=os.environ.get("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"))
        stage.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        stage.add_argument("--pro-threshold", type=int, default=30)
        stage.add_argument("--workers", type=positive, default=1)
        if name == "match":
            stage.add_argument("--call-api", action="store_true", required=True, help="Explicitly authorize paid API calls for this run")
            stage.add_argument("--all-targets", action="store_true", help="Explicitly select the full corpus instead of --limit")
            stage.add_argument("--resume", action="store_true")
            stage.add_argument("--temperature", type=float, default=0.0)
            stage.add_argument("--timeout", type=positive, default=120)
            stage.add_argument("--max-retries", type=positive, default=3)
        else:
            stage.add_argument("--force", action="store_true")

    post = stages.add_parser("postprocess")
    post.add_argument("--work-dir", type=Path, required=True)
    post.add_argument("--parser-root", type=Path)
    post.add_argument("--input", type=Path, help="Defaults to work-dir/selected.raw.jsonl")
    post.add_argument("--historical-statement-filter", action="store_true", help="Reproduce original lexical lemma/theorem filtering; this can remove useful mathematical resources")
    post.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage == "parse":
        mathlib = args.mathlib_root.resolve()
        expected = json.loads((ROOT / "versions.json").read_text())["lean_toolchain"]
        if (mathlib / "lean-toolchain").read_text().strip() != expected:
            raise ValueError(f"Wrong Lean toolchain; expected {expected}")
        from check_environment import inspect
        revision = inspect(mathlib)
        if revision["git_commit_matches"] is False:
            raise ValueError(f"Wrong Mathlib checkout commit: {revision['observed_checkout_commit']}")
        if revision["git_commit_matches"] is None:
            print("[warning] Mathlib source revision is not certified: no standalone Git checkout detected.", file=sys.stderr)
        output = args.work_dir.resolve() / "command_spans_jsonl"
        arguments = ["--mathlib-root", str(mathlib), "--extractor", str(SOURCE_ROOT / "lean_parser/ExtractCommandSpans.lean"), "--out-root", str(output), "--jobs", str(args.jobs), "--timeout", str(args.timeout)]
        for option in ("path_include", "include_archive", "force"):
            value = getattr(args, option)
            if value:
                arguments.append("--" + option.replace("_", "-"))
                if isinstance(value, str):
                    arguments.append(value)
        invoke("lean_parser.batch_extract_mathlib_command_spans", arguments)
        summary = json.loads((output / "_manifest.json").read_text())["summary"]
        if summary["num_files_discovered"] == 0 or summary["failed"]:
            raise RuntimeError(f"Parser did not finish successfully: {summary}. Inspect _manifest.json and _logs.")
        return

    work, parser_root = paths(args)
    if args.stage in {"candidates", "match"}:
        if not parser_root.is_dir() or not any(parser_root.rglob("*.jsonl")):
            raise FileNotFoundError(f"No parser JSONL files under {parser_root}")
        is_match = args.stage == "match"
        if is_match:
            if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
                raise RuntimeError("Set DEEPSEEK_API_KEY in the environment; no key is bundled.")
            if args.limit is None and not args.all_targets:
                parser.error("match requires --limit N or --all-targets to make corpus size explicit")
            if args.limit is not None and args.all_targets:
                parser.error("--limit and --all-targets are mutually exclusive")
        output = work / ("selected.raw.jsonl" if is_match else "candidates.jsonl")
        require_new(output, force=getattr(args, "force", False), resume=getattr(args, "resume", False))
        arguments = ["--input-root", str(parser_root), "--mode", "llm" if is_match else "candidates", "--output", str(output)]
        for name in ("recursive_depth", "max_dependency_candidates", "max_chars_per_command", "flash_model", "pro_model", "base_url", "pro_threshold", "workers"):
            arguments += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if args.limit is not None:
            arguments += ["--limit", str(args.limit)]
        for source_file in args.source_file:
            arguments += ["--source-file", source_file]
        if is_match:
            for name in ("temperature", "timeout", "max_retries"):
                arguments += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            if args.resume:
                arguments.append("--resume")
        invoke("extract_data.build_context_deepseek_hybrid", arguments)
        failed = output.with_suffix(output.suffix + ".failed.jsonl")
        if failed.exists() and failed.stat().st_size:
            remaining = unresolved_failures(output, failed)
            if remaining:
                raise RuntimeError(f"{remaining} targets have unresolved failures; inspect {failed}. Successful rows remain in {output}.")
        return

    original = args.input.resolve() if args.input else work / "selected.raw.jsonl"
    if not original.is_file():
        raise FileNotFoundError(original)
    deduplicated = work / "selected.deduplicated.jsonl"
    filtered = work / "selected.statement_context.jsonl"
    repaired = work / "selected.repaired.jsonl"
    classified = work / "selected.surface.jsonl"
    outputs = [deduplicated, repaired, classified] + ([filtered] if args.historical_statement_filter else [])
    for output in outputs:
        if output.resolve() == original.resolve():
            raise ValueError("Input and output must be different files")
        require_new(output, force=args.force)
    if args.historical_statement_filter and not parser_root.is_dir():
        raise FileNotFoundError(parser_root)
    invoke("extract_data.dedupe_jsonl_by_target", ["--input", str(original), "--output", str(deduplicated), "--keep", "last"])
    repair_input = deduplicated
    if args.historical_statement_filter:
        invoke("extract_data.remove_proof_leakage_context", ["--input", str(deduplicated), "--output", str(filtered), "--input-root", str(parser_root), "--filter-kinds", "lemma,theorem"])
        check_failed(filtered, ("postprocess_status",))
        repair_input = filtered
    invoke("extract_data.repair_sft_target_split", ["--input", str(repair_input), "--output", str(repaired)])
    check_failed(repaired, ("repair_status",))
    invoke("extract_data.classify_proof_surface", ["--input", str(repaired), "--output", str(classified)])
    check_classifier(classified)
    print(json.dumps({"repaired_with_command_mapping": str(repaired), "surface_classification": str(classified), "historical_statement_filter_applied": args.historical_statement_filter}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)
