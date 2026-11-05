"""Load AR/DLLM base, full-SFT, or LoRA checkpoints for inference."""

from __future__ import annotations

import json
import importlib.metadata
from pathlib import Path
from typing import Any

import torch


def resolve_dtype(value: str) -> torch.dtype:
    key = str(value).lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _adapter_dir(checkpoint: Path) -> Path | None:
    for candidate in (checkpoint / "model", checkpoint):
        if (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def _tokenizer_source(model_config: dict[str, Any], checkpoint: Path | None) -> str:
    if checkpoint is not None:
        for candidate in (checkpoint / "tokenizer", checkpoint):
            if candidate.is_dir() and any(
                (candidate / name).exists()
                for name in ("tokenizer_config.json", "tokenizer.json", "spiece.model")
            ):
                return str(candidate)
    return str(
        model_config.get("tokenizer_name_or_path")
        or model_config["name_or_path"]
    )


def _state_file(checkpoint: Path) -> Path | None:
    candidates = (
        checkpoint / "state" / "model.safetensors",
        checkpoint / "state" / "pytorch_model.bin",
        checkpoint / "model.safetensors",
        checkpoint / "pytorch_model.bin",
    )
    return next((path for path in candidates if path.is_file()), None)


def _load_state_file(model: Any, path: Path) -> None:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "full checkpoint does not match base model: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )


def load_model_and_tokenizer(
    config: dict[str, Any], *, architecture: str, device: torch.device
) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    minimum = model_config.get("minimum_transformers_version")
    if minimum:
        from packaging.version import Version
        actual = importlib.metadata.version("transformers")
        if Version(actual) < Version(str(minimum)):
            raise RuntimeError(f"released model requires transformers>={minimum}; installed={actual}. Use its separate released-model environment.")
    checkpoint_value = model_config.get("checkpoint")
    checkpoint = Path(str(checkpoint_value)) if checkpoint_value else None
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    tokenizer_source = _tokenizer_source(model_config, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=(model_config.get("tokenizer_revision") or model_config.get("revision")) if not Path(tokenizer_source).is_dir() else None,
        local_files_only=bool(model_config.get("tokenizer_local_files_only", False)),
        cache_dir=model_config.get("cache_dir"),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        use_fast=bool(model_config.get("use_fast_tokenizer", True)),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    architecture = str(architecture).lower()
    if architecture not in {"ar", "dllm"}:
        raise ValueError("architecture must be ar or dllm")
    model_class = AutoModelForCausalLM if architecture == "ar" else AutoModel
    adapter = _adapter_dir(checkpoint) if checkpoint else None
    checkpoint_type = str(model_config.get("checkpoint_type", "auto"))
    hf_source = str(model_config["name_or_path"])
    if checkpoint and checkpoint_type in {"hf", "merged"}:
        hf_source = str(checkpoint)
    elif checkpoint and checkpoint_type == "auto" and adapter is None:
        if (checkpoint / "config.json").is_file() and any(
            checkpoint.glob("*.safetensors")
        ):
            hf_source = str(checkpoint)
    model = model_class.from_pretrained(
        hf_source,
        revision=model_config.get("revision") if hf_source == str(model_config["name_or_path"]) else None,
        cache_dir=model_config.get("cache_dir"),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=resolve_dtype(model_config.get("dtype", "bfloat16")),
        low_cpu_mem_usage=True,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    loaded_as = "base_or_hf"
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        loaded_as = "lora_adapter"
    elif checkpoint is not None and hf_source != str(checkpoint):
        state_file = _state_file(checkpoint)
        if state_file is None:
            raise FileNotFoundError(
                "checkpoint is neither a PEFT adapter, a Hugging Face model, nor an "
                f"Accelerate full-model state: {checkpoint}"
            )
        _load_state_file(model, state_file)
        loaded_as = "accelerate_full_state"

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = architecture == "ar"
    model.requires_grad_(False)
    model.eval().to(device)
    metadata = {
        "architecture": architecture,
        "base_model": str(model_config["name_or_path"]),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_loading": loaded_as,
        "tokenizer": tokenizer_source,
        "dtype": str(model_config.get("dtype", "bfloat16")),
        "revision": model_config.get("revision"),
        "tokenizer_mode": config.get("generation", {}).get("tokenizer_mode", "primary_backbone_contract"),
        "released_evaluation": config.get("released_evaluation"),
    }
    return model, tokenizer, metadata
