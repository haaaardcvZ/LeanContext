#!/usr/bin/env python3
"""Tokenize unified Matched-Context data for Llama-3.1."""

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
    result = materialize_tokenized_views(config, load_tokenizer(config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
