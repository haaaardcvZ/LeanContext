"""Acquire only exact, pinned Llama-3.1-8B tokenizer files.

Use official public metadata to verify any legitimately public redistribution.
No model weights, remote Python code, credentials, or dataset processing are used.
The JSON printed at completion is the acquisition manifest to preserve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

OFFICIAL_REPO = "meta-llama/Llama-3.1-8B"
OFFICIAL_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
PUBLIC_REPO = "NousResearch/Meta-Llama-3.1-8B"
PUBLIC_REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def curl(url: str, proxy: str, destination: Path | None = None) -> bytes:
    args = ["curl", "-4", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "10", "--max-time", "60", "--proxy", proxy, url]
    if destination is not None:
        args += ["--output", str(destination)]
    result = subprocess.run(args, capture_output=True)
    if result.returncode:
        # Do not echo request headers, environment, or credential material.
        raise RuntimeError(f"public curl failed with exit status {result.returncode}")
    return result.stdout


def metadata(repo: str, revision: str, proxy: str) -> tuple[str, dict]:
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    obj = json.loads(curl(url, proxy))
    if obj.get("sha") != revision or obj.get("private"):
        raise RuntimeError("public repository metadata identity mismatch")
    selected = {item["rfilename"]: item for item in obj.get("siblings", [])
                if item.get("rfilename") in FILES}
    if set(selected) != set(FILES):
        raise RuntimeError("missing tokenizer metadata")
    return url, {"id": obj["id"], "sha": obj["sha"], "gated": obj.get("gated"),
                 "private": obj.get("private"), "files": selected}


def fingerprints(path: Path) -> dict:
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}


def matches(actual: dict, official: dict) -> bool:
    if actual["size"] != official["size"]:
        return False
    lfs = official.get("lfs")
    return actual["sha256"] == lfs["sha256"] if lfs else actual["git_blob_sha1"] == official["blobId"]


def acquire(out: Path, proxy: str) -> dict:
    official_url, official = metadata(OFFICIAL_REPO, OFFICIAL_REVISION, proxy)
    out.mkdir(parents=True, exist_ok=True)
    attempt = {"mechanism": "huggingface_hub_automatic_existing_authentication",
               "credentials_read_or_printed_by_script": False}
    official_ok = False
    try:
        import requests
        from huggingface_hub import configure_http_backend, hf_hub_download

        def backend():
            session = requests.Session()
            session.trust_env = False
            session.proxies = {"https": proxy, "http": proxy}
            return session

        configure_http_backend(backend_factory=backend)
        for name in FILES:
            hf_hub_download(OFFICIAL_REPO, name, revision=OFFICIAL_REVISION,
                            local_dir=str(out), etag_timeout=15)
        official_ok = True
        attempt["status"] = "success"
    except Exception as exc:
        attempt.update(status="failed", error_type=type(exc).__name__,
                       http_status=getattr(getattr(exc, "response", None), "status_code", None))
    mirror_url, mirror = (None, None)
    repo, revision = OFFICIAL_REPO, OFFICIAL_REVISION
    if not official_ok:
        mirror_url, mirror = metadata(PUBLIC_REPO, PUBLIC_REVISION, proxy)
        if mirror["gated"] or mirror["private"]:
            raise RuntimeError("redistribution must be legitimately public, not gated/private")
        for name in FILES:
            reference, public = official["files"][name], mirror["files"][name]
            if reference["size"] != public["size"] or reference["blobId"] != public["blobId"] or reference.get("lfs") != public.get("lfs"):
                raise RuntimeError(f"public file differs from pinned official metadata: {name}")
        repo, revision = PUBLIC_REPO, PUBLIC_REVISION
        for name in FILES:
            curl(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", proxy, out / name)
    files = {}
    for name in FILES:
        actual = fingerprints(out / name)
        if not matches(actual, official["files"][name]):
            raise RuntimeError(f"download failed pinned official byte-identity check: {name}")
        files[name] = {**actual, "official_reference": official["files"][name],
            "official_hash_match": True, "source_url": f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
            "verification": "Git_blob_SHA1_over_exact_file_bytes_matches_official_public_API"}
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(out.resolve()), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    sample = "theorem anonymous_target : True :=\nby trivial"
    plain = tokenizer.encode(sample, add_special_tokens=False)
    wrapped = tokenizer.encode(sample, add_special_tokens=True)
    return {"schema_version": "exact-pinned-tokenizer-acquisition-v1", "model": "llama31",
        "status": "downloaded_verified_and_locally_loaded", "official_repo": OFFICIAL_REPO,
        "official_revision": OFFICIAL_REVISION, "official_metadata_url": official_url,
        "official_metadata": official, "official_download_attempt": attempt,
        "download_repo": repo, "download_revision": revision,
        "download_metadata_url": mirror_url, "download_metadata": mirror,
        "redistribution_identity": "exact_official_bytes_not_approximate_family_tokenizer",
        "tokenizer_directory": str(out.resolve()), "downloaded_model_weights": False,
        "remote_code_executed": False, "TLS_certificate_verification_disabled": False,
        "files": files, "license_url": f"https://huggingface.co/{OFFICIAL_REPO}/blob/{OFFICIAL_REVISION}/LICENSE",
        "loading": {"transformers_version": transformers.__version__, "tokenizers_version": tokenizers.__version__,
            "class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
            "bos_token": tokenizer.bos_token, "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
            "vocabulary_size": len(tokenizer), "sample_plain_tokens": len(plain),
            "sample_with_special_tokens": len(wrapped), "sample_special_token_prefix": wrapped[:1],
            "sample_special_token_suffix": wrapped[-1:]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("leancontext_dataset/tokenizers/llama31"))
    parser.add_argument("--proxy", required=True, help="existing macOS HTTPS proxy, no TLS downgrade")
    args = parser.parse_args()
    print(json.dumps(acquire(args.out, args.proxy), ensure_ascii=False, indent=2))
