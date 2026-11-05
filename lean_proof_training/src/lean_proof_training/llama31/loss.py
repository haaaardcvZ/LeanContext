"""Answer-only macro-averaged causal language-model loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from lean_proof_training.common.loss_types import LossOutput


def compute_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    config: dict[str, Any],
    evaluating: bool,
    repeat_index: int,
) -> LossOutput:
    del config, evaluating, repeat_index
    ids = batch["input_ids"]
    attention = batch["attention_mask"]
    prompt_lengths = batch["prompt_lengths"]
    target_lengths = batch["target_lengths"]
    outputs = model(input_ids=ids, attention_mask=attention, use_cache=False)
    logits = outputs.logits
    batch_size, sequence_length = ids.shape
    positions = torch.arange(sequence_length, device=ids.device).unsqueeze(0)
    target_mask = (
        (positions >= prompt_lengths.unsqueeze(1))
        & (positions < (prompt_lengths + target_lengths).unsqueeze(1))
        & attention.bool()
    )
    # logits[t-1] predicts token[t], including the single supervised EOS.
    shift_logits = logits[:, :-1]
    shift_labels = ids[:, 1:]
    shift_mask = target_mask[:, 1:]
    selected_logits = shift_logits[shift_mask]
    labels = shift_labels[shift_mask]
    token_losses = F.cross_entropy(selected_logits, labels, reduction="none")
    sample_indices = (
        torch.arange(batch_size, device=ids.device)
        .unsqueeze(1)
        .expand(batch_size, sequence_length - 1)[shift_mask]
    )
    loss_sums = torch.zeros(batch_size, device=ids.device, dtype=token_losses.dtype)
    token_counts = shift_mask.sum(dim=1)
    if not bool((token_counts > 0).all().item()):
        raise ValueError("every AR sample must contain supervised target tokens")
    loss_sums.scatter_add_(0, sample_indices, token_losses)
    correct_mask = (selected_logits.argmax(dim=-1) == labels).long()
    per_sample_correct = torch.zeros(batch_size, device=ids.device, dtype=torch.long)
    per_sample_correct.scatter_add_(0, sample_indices, correct_mask)
    per_sample = loss_sums / token_counts
    return LossOutput(
        loss=per_sample.mean(),
        per_sample_loss=per_sample,
        per_sample_correct=per_sample_correct,
        per_sample_tokens=token_counts,
        correct_tokens=correct_mask.sum(),
        supervised_tokens=shift_mask.sum(),
    )
