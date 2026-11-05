"""Independently compare local source blobs to the pinned official Git tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


COMMIT = "a7fc949f1b05c2a01e01c027fd9f480496a1253e"
API = "https://api.github.com/repos/leanprover-community/mathlib4"


def request_json(url: str):
    if not url.startswith(API + "/"):
        raise ValueError("Only the official pinned Mathlib GitHub API is allowed")
    # macOS system curl uses the OS trust store; never disable TLS verification.
    process = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                              "--max-time", "30", "--header", "Accept: application/vnd.github+json",
                              "--user-agent", "LeanContext-source-provenance-audit", url],
                             capture_output=True, text=True, check=True, timeout=35)
    return json.loads(process.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mathlib-root", type=Path, required=True)
    args = parser.parse_args()
    snapshot_path = args.dataset_root / "command_mapping/source_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    tag_url = API + "/git/ref/tags/v4.14.0-rc1"
    tag = request_json(tag_url)
    if tag["object"]["type"] != "commit" or tag["object"]["sha"] != COMMIT:
        raise RuntimeError("Official Mathlib tag does not identify the recorded commit")
    tree_url = API + "/git/trees/" + COMMIT + "?recursive=1"
    tree = request_json(tree_url)
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated source tree")
    blobs = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    checked, failures = {}, []
    for relative, expected_sha256 in snapshot["source_files"].items():
        data = (args.mathlib_root / relative).read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()
        if sha256 != expected_sha256 or blobs.get(relative) != git_blob:
            failures.append(relative)
        checked[relative] = {"sha256": sha256, "local_git_blob_sha1": git_blob, "official_git_blob_sha1": blobs.get(relative)}
    evidence = {
        "status": "verified" if not failures else "mismatch",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_tag": "v4.14.0-rc1", "official_commit": COMMIT,
        "official_tag_url": tag_url, "official_tree_url": tree_url,
        "official_tree_sha": tree["sha"], "source_files_checked": len(checked),
        "method": "Compare Git blob SHA1 (blob length NUL content) of every exported original source file with the official pinned Git tree, plus dataset SHA256",
        "mismatches": failures, "files": checked,
    }
    destination = args.dataset_root / "command_mapping/revision_verification.json"
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Pinned source revision mismatches: {failures[:10]}")
    print(json.dumps({key: value for key, value in evidence.items() if key != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
