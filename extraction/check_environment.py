#!/usr/bin/env python3
"""Read-only checks; does not install tools, download Mathlib, or call an LLM."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def inspect(mathlib: Path) -> dict:
    expected = json.loads((Path(__file__).resolve().parent / "versions.json").read_text())
    toolchain_path = mathlib / "lean-toolchain"
    toolchain = toolchain_path.read_text().strip() if toolchain_path.is_file() else None
    cache_paths = [mathlib / ".lake/build/lib/Mathlib.olean", mathlib / ".lake/build/lib/lean/Mathlib.olean"]
    result = {
        "python": sys.version.split()[0],
        "python_supported": sys.version_info >= (3, 10),
        "lake": shutil.which("lake"),
        "elan": shutil.which("elan"),
        "mathlib_root": str(mathlib),
        "expected_toolchain": expected["lean_toolchain"],
        "observed_toolchain": toolchain,
        "toolchain_matches": toolchain == expected["lean_toolchain"],
        "lakefile_exists": (mathlib / "lakefile.lean").is_file(),
        "lake_manifest_exists": (mathlib / "lake-manifest.json").is_file(),
        "mathlib_olean_exists": any(path.is_file() for path in cache_paths),
        "expected_mathlib_commit": expected["mathlib_commit"],
        "observed_checkout_commit": None,
        "git_commit_matches": None,
        "API_called": False,
        "lean_extraction_executed": False,
    }
    # Never mistake a parent Git repository for the Mathlib checkout.
    if shutil.which("git") and mathlib.is_dir():
        top = subprocess.run(["git", "-C", str(mathlib), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == mathlib:
            commit = subprocess.run(["git", "-C", str(mathlib), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            result["observed_checkout_commit"] = commit
            result["git_commit_matches"] = commit == expected["mathlib_commit"]
    if result["observed_checkout_commit"] is None:
        result["source_revision_note"] = "No standalone Mathlib Git checkout detected; commit identity is not certified by this check."
    result["ready_for_extraction"] = all(result[field] for field in ("python_supported", "lake", "toolchain_matches", "lakefile_exists", "lake_manifest_exists", "mathlib_olean_exists")) and result["git_commit_matches"] is not False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mathlib-root", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.mathlib_root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready_for_extraction"] else 1)


if __name__ == "__main__":
    main()
