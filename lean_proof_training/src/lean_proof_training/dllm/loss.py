"""Answer-only exact-count masked-diffusion loss."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F

from lean_proof_training.common.loss_types import LossOutput


def _forward(model: Any, ids: torch.Tensor, attention: torch.Tensor) -> Any:
    kwargs: dict[str, Any] = {"input_ids": ids, "use_cache": False}
    if not bool(attention.bool().all().item()):
        kwargs["attention_mask"] = attention
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        return model(**kwargs)


def _seed(sample_id: str, repeat_index: int, base_seed: int) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{repeat_index}:{sample_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def compute_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    config: dict[str, Any],
    evaluating: bool,
    repeat_index: int,
) -> LossOutput:
    ids = batch["input_ids"]
    attention = batch["attention_mask"]
    prompt_lengths = batch["prompt_lengths"]
    target_lengths = batch["target_lengths"]
    batch_size, sequence_length = ids.shape
    positions = torch.arange(sequence_length, device=ids.device).unsqueeze(0)
    target_mask = (
        (positions >= prompt_lengths.unsqueeze(1))
        & (positions < (prompt_lengths + target_lengths).unsqueeze(1))
        & attention.bool()
    )
    mask_token_id = config.get("mask_token_id")
    if mask_token_id is None:
        mask_token_id = getattr(getattr(model, "config", None), "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("DLLM loss requires loss.mask_token_id or model.config.mask_token_id")

    masked = torch.zeros_like(target_mask)
    base_seed = int(config.get("eval_mask_seed", 20260721))
    fixed_ratio = float(config.get("eval_mask_ratio", 0.5))
    if evaluating and not 0.0 < fixed_ratio <= 1.0:
        raise ValueError("eval_mask_ratio must be in (0, 1]")
    for row in range(batch_size):
        candidates = torch.nonzero(target_mask[row], as_tuple=False).flatten()
        target_count = int(candidates.numel())
        if target_count == 0:
            raise ValueError(f"sample has zero supervised target tokens: {batch['sample_ids'][row]}")
        if evaluating:
            selected_count = max(1, min(target_count, math.ceil(fixed_ratio * target_count)))
            generator = torch.Generator(device=ids.device)
            generator.manual_seed(_seed(batch["sample_ids"][row], repeat_index, base_seed))
        else:
            selected_count = int(torch.randint(1, target_count + 1, (), device=ids.device))
            generator = None
        order = torch.randperm(target_count, device=ids.device, generator=generator)
        masked[row, candidates[order[:selected_count]]] = True

    noisy_ids = ids.masked_fill(masked, int(mask_token_id))
    logits = _forward(model, noisy_ids, attention).logits
    selected_logits = logits[masked]
    labels = ids[masked]
    token_losses = F.cross_entropy(selected_logits, labels, reduction="none")
    sample_indices = (
        torch.arange(batch_size, device=ids.device).unsqueeze(1).expand(batch_size, sequence_length)[masked]
    )
    loss_sums = torch.zeros(batch_size, device=ids.device, dtype=token_losses.dtype)
    masked_counts = masked.sum(dim=1)
    loss_sums.scatter_add_(0, sample_indices, token_losses)
    per_sample_correct = torch.zeros(batch_size, device=ids.device, dtype=torch.long)
    per_sample_correct.scatter_add_(
        0, sample_indices, (selected_logits.argmax(dim=-1) == labels).long()
    )
    # Exact-count uniform masking makes the masked-token mean an unbiased
    # estimator of each proof's target-token mean. Every theorem has weight 1.
    per_sample = loss_sums / masked_counts.clamp_min(1)
    correct = (selected_logits.argmax(dim=-1) == labels).sum()
    return LossOutput(
        loss=per_sample.mean(),
        per_sample_loss=per_sample,
        per_sample_correct=per_sample_correct,
        per_sample_tokens=masked_counts,
        correct_tokens=correct,
        supervised_tokens=masked.sum(),
        extras={
            "mean_mask_ratio": (
                masked_counts.float() / target_mask.sum(dim=1).clamp_min(1).float()
            ).mean()
        },
    )
