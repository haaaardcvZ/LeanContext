"""Multi-GPU Accelerate runtime for Distributed Data Parallel training."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs


def build_accelerator(config: dict[str, Any]) -> Accelerator:
    distributed = config.get("distributed") or {}
    strategy = str(distributed.get("strategy", "ddp")).lower()
    handlers = [
        DistributedDataParallelKwargs(
            find_unused_parameters=bool(distributed.get("find_unused_parameters", False)),
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        ),
        InitProcessGroupKwargs(
            timeout=timedelta(seconds=int(distributed.get("timeout_seconds", 7200)))
        ),
    ]
    if strategy != "ddp":
        raise ValueError("This training entry point supports distributed.strategy=ddp")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["training"].get("gradient_accumulation_steps", 1)),
        mixed_precision=str(config["training"].get("mixed_precision", "bf16")),
        project_dir=str(config["output_dir"]),
        kwargs_handlers=handlers,
    )
    expected = int(distributed.get("expected_processes", 16))
    if accelerator.num_processes != expected:
        raise RuntimeError(
            f"expected {expected} distributed processes, got {accelerator.num_processes}"
        )
    return accelerator
