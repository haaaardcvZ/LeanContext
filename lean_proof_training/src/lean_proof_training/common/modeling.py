"""Tokenizer/model loading and full-SFT/LoRA adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(value: str) -> torch.dtype:
    key = str(value).lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def load_tokenizer(config: dict[str, Any], *, tokenized: bool = False) -> Any:
    model_config = config["model"]
    source = model_config.get("tokenizer_name_or_path") or model_config["name_or_path"]
    if tokenized:
        candidate = Path(config["data"]["tokenized_root"]) / "tokenizer"
        if candidate.is_dir():
            source = str(candidate)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=str(model_config["cache_dir"]),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        use_fast=bool(model_config.get("use_fast_tokenizer", True)),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must provide an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    config: dict[str, Any],
    *,
    architecture: str,
    tokenizer: Any,
    trainable: bool,
) -> Any:
    model_config = config["model"]
    cls = AutoModel if architecture == "dllm" else AutoModelForCausalLM
    model = cls.from_pretrained(
        model_config["name_or_path"],
        revision=model_config.get("revision"),
        cache_dir=str(model_config["cache_dir"]),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=resolve_dtype(model_config.get("dtype", "bfloat16")),
        low_cpu_mem_usage=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    tuning = config.get("tuning") or {"mode": "full"}
    mode = str(tuning.get("mode", "full"))
    initial_adapter = tuning.get("initial_adapter_path")
    if initial_adapter:
        model = PeftModel.from_pretrained(model, str(initial_adapter), is_trainable=trainable)
    elif mode == "lora":
        lora_kwargs: dict[str, Any] = {
            "r": int(tuning.get("r", 16)),
            "lora_alpha": int(tuning.get("alpha", 32)),
            "lora_dropout": float(tuning.get("dropout", 0.05)),
            "target_modules": list(tuning["target_modules"]),
            "bias": "none",
        }
        if architecture == "llama31":
            lora_kwargs["task_type"] = "CAUSAL_LM"
        model = get_peft_model(
            model,
            LoraConfig(**lora_kwargs),
        )
    elif mode != "full":
        raise ValueError("tuning.mode must be full or lora")

    if not trainable:
        model.requires_grad_(False)
    elif bool(model_config.get("gradient_checkpointing", True)):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads") and mode == "lora":
            model.enable_input_require_grads()
    return model


def resolved_loss_config(
    config: dict[str, Any], *, architecture: str, tokenizer: Any
) -> dict[str, Any]:
    """Bind tokenizer-owned special IDs once for every loss consumer."""
    result = dict(config.get("loss") or {})
    if architecture == "dllm" and result.get("mask_token_id") is None:
        if tokenizer.mask_token_id is None:
            raise ValueError("DLLM tokenizer must provide mask_token_id")
        result["mask_token_id"] = int(tokenizer.mask_token_id)
    return result
