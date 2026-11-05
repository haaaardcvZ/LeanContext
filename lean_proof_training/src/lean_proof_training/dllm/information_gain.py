#!/usr/bin/env python3
"""Compute same-checkpoint LLaDA Context information gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_proof_training.common.config import load_config
from lean_proof_training.common.information_gain import run_information_gain
from lean_proof_training.dllm.loss import compute_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    result = run_information_gain(
        load_config(args.config, args.set), architecture="dllm", loss_function=compute_loss
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
