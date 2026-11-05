#!/usr/bin/env python3
"""Validate the raw unified dataset before tokenization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_proof_training.common.validation import validate_raw_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_raw_directory(args.raw_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
