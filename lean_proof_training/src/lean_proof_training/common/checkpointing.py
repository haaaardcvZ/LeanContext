"""Distributed checkpoint and append-only metric persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(
    *,
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    name: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    destination = output_dir / name
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and destination.exists():
        shutil.rmtree(destination)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(destination / "state"))
    if accelerator.is_main_process:
        (destination / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tokenizer.save_pretrained(destination / "tokenizer")
    if str(config.get("tuning", {}).get("mode")) == "lora":
        unwrapped = accelerator.unwrap_model(model)
        state = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped.save_pretrained(destination / "model", state_dict=state)
    accelerator.wait_for_everyone()
