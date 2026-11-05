#!/usr/bin/env python3
"""Train Llama-3.1 with DDP LoRA or FSDP full SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_proof_training.common.config import load_config
from lean_proof_training.common.training import run_training
from lean_proof_training.llama31.loss import compute_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    result = run_training(
        load_config(args.config, args.set), architecture="llama31", loss_function=compute_loss
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
