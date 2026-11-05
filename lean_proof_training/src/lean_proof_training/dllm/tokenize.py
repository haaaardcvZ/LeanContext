#!/usr/bin/env python3
"""Tokenize unified Matched-Context data for LLaDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_proof_training.common.config import load_config
from lean_proof_training.common.modeling import load_tokenizer
from lean_proof_training.common.tokenization import materialize_tokenized_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    tokenizer = load_tokenizer(config)
    if tokenizer.mask_token_id is None and config.get("loss", {}).get("mask_token_id") is None:
        raise ValueError("LLaDA tokenizer/config must provide a mask token ID")
    result = materialize_tokenized_views(config, tokenizer)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
