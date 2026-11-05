"""Shared loss output type."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LossOutput:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    per_sample_correct: torch.Tensor
    per_sample_tokens: torch.Tensor
    correct_tokens: torch.Tensor
    supervised_tokens: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
