"""Acquire only the pinned public LLaDA tokenizer artifacts, never model weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from urllib.request import getproxies


REPOSITORY = "GSAI-ML/LLaDA-8B-Base"
ALLOWLIST = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json")


def curl(url: str, destination: Path | None = None) -> bytes:
    command = ["curl", "--location", "--fail", "--silent", "--show-error", "--connect-timeout", "20", "--max-time", "180", "--retry", "2", "--retry-delay", "2", "--proto", "=https", "--proto-redir", "=https"]
    if destination is not None:
        command += ["--output", str(destination)]
    command += [url]
    # curl does not otherwise inherit macOS's configured network proxy.
    # Preserve that user-configured route without logging credentials or URLs.
    environment = os.environ.copy()
    for scheme, proxy in getproxies().items():
        if scheme in {"http", "https"}:
            environment.setdefault(scheme + "_proxy", proxy)
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, env=environment).stdout


def acquire(output: Path) -> dict:
    api_url = f"https://huggingface.co/api/models/{REPOSITORY}"
    metadata = json.loads(curl(api_url))
    revision = metadata.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("public model API did not return an immutable full commit SHA")
    siblings = {item["rfilename"] for item in metadata.get("siblings", [])}
    if "tokenizer.json" not in siblings or "tokenizer_config.json" not in siblings:
        raise ValueError("public model repository lacks the required fast tokenizer artifacts")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename in ALLOWLIST:
        if filename not in siblings:
            continue
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{quote(filename)}"
        path = output / filename
        temporary = output / (filename + ".download")
        curl(url, temporary)
        data = temporary.read_bytes()
        valid_json = True
        parsing_error = None
        try:
            json.loads(data)
        except json.JSONDecodeError as exc:
            if filename != "special_tokens_map.json":
                raise
            # Preserve defective optional upstream bytes without changing them.
            # AutoTokenizer can use the genuine tokenizer_config + tokenizer.json;
            # an invalid optional map must not silently be repaired or loaded.
            valid_json = False
            parsing_error = str(exc)
            path = output / "special_tokens_map.upstream.json"
        temporary.replace(path)
        files[filename] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "local_filename": path.name, "valid_json": valid_json, "excluded_from_AutoTokenizer_loading": not valid_json, "json_parsing_error": parsing_error}
    card = metadata.get("cardData") or {}
    manifest = {
        "schema_version": "tokenizer-only-acquisition-v1",
        "repository": REPOSITORY,
        "revision": revision,
        "metadata_url": api_url,
        "model_page": f"https://huggingface.co/{REPOSITORY}",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "filename_allowlist": list(ALLOWLIST),
        "files": files,
        "license_metadata": {key: card.get(key) for key in ("license", "license_name", "license_link")},
        "weights_downloaded": False,
        "model_code_downloaded": False,
        "credentials_used": False,
    }
    (output / "acquisition_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def smoke_test(output: Path, sample_path: Path) -> dict:
    """Load only the acquired tokenizer and encode one genuine raw target."""
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(output), local_files_only=True, trust_remote_code=False)
    with sample_path.open(encoding="utf-8") as handle:
        row = next(json.loads(line) for line in handle if line.strip())
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    proof_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is None or int(eos) in proof_ids:
        raise ValueError("genuine sample has no usable unique final EOS convention")
    answer_ids = list(proof_ids) + [int(eos)]
    result = {
        "repository": REPOSITORY,
        "revision": json.loads((output / "acquisition_manifest.json").read_text())["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
        "sample_id": row["sample_id"],
        "sample_file": str(sample_path.resolve()),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": eos,
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "mdm_mask_vocabulary_id": tokenizer.convert_tokens_to_ids("<|mdm_mask|>"),
        "vocabulary_size_with_added_tokens": len(tokenizer),
        "prompt_length": len(prompt_ids),
        "proof_length_without_eos": len(proof_ids),
        "answer_length_with_one_eos": len(answer_ids),
        "sequence_length": len(prompt_ids) + len(answer_ids),
        "answer_eos_count": answer_ids.count(int(eos)),
        "answer_last_token_is_eos": answer_ids[-1] == int(eos),
        "backbone_bos_added": False,
        "reference_proof_preserved": tokenizer.decode(proof_ids, skip_special_tokens=False) == row["target"],
        "model_weights_or_model_code_loaded": False,
        "full_dataset_tokenization_performed": False,
    }
    (output / "tokenizer_smoke_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path, help="one genuine training view file for an offline smoke test")
    parser.add_argument("--smoke-only", action="store_true", help="reuse already acquired files; no network")
    args = parser.parse_args()
    if not args.smoke_only:
        print(json.dumps(acquire(args.output), ensure_ascii=False, indent=2, sort_keys=True))
    if args.sample:
        print(json.dumps(smoke_test(args.output, args.sample), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
