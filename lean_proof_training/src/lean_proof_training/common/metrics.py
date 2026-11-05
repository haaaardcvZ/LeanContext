"""Exact distributed theorem-macro metrics, including proof-family slices."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from lean_proof_training.common.batching import move_batch


PROOF_FAMILIES = ("Strict Term", "Ordinary Tactic", "Hybrid", "Calc")


@torch.no_grad()
def evaluate_loader(
    model: Any,
    loader: DataLoader,
    accelerator: Any,
    loss_function: Callable[..., Any],
    loss_config: dict[str, Any],
    local_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model.eval()
    totals = torch.zeros((len(PROOF_FAMILIES) + 1, 4), dtype=torch.float64)
    family_to_index = {family: index for index, family in enumerate(PROOF_FAMILIES)}
    for batch in loader:
        batch = move_batch(batch, accelerator.device)
        output = loss_function(
            model,
            batch,
            config=loss_config,
            evaluating=True,
            repeat_index=0,
        )
        family_codes = torch.tensor(
            [family_to_index[family] for family in batch["proof_families"]],
            device=accelerator.device,
            dtype=torch.long,
        )
        if local_rows is not None:
            for index, sample_id in enumerate(batch["sample_ids"]):
                token_count = int(output.per_sample_tokens[index].detach().item())
                correct_count = int(output.per_sample_correct[index].detach().item())
                local_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "proof_family": str(batch["proof_families"][index]),
                        "view": str(batch["views"][index]),
                        "loss": float(output.per_sample_loss[index].detach().item()),
                        "correct_tokens": correct_count,
                        "supervised_tokens": token_count,
                        "token_accuracy": correct_count / max(token_count, 1),
                    }
                )
        losses, correct, tokens, family_codes = accelerator.gather_for_metrics(
            (
                output.per_sample_loss.detach(),
                output.per_sample_correct.detach(),
                output.per_sample_tokens.detach(),
                family_codes,
            )
        )
        for index in range(losses.numel()):
            values = torch.tensor(
                [float(losses[index]), 1.0, float(correct[index]), float(tokens[index])],
                dtype=torch.float64,
            )
            totals[-1] += values
            totals[int(family_codes[index])] += values
    model.train()

    def metrics(values: torch.Tensor) -> dict[str, float | int]:
        samples = max(values[1].item(), 1.0)
        tokens = max(values[3].item(), 1.0)
        return {
            "macro_loss": values[0].item() / samples,
            "token_accuracy": values[2].item() / tokens,
            "samples": int(values[1].item()),
            "supervised_tokens": int(values[3].item()),
        }

    result: dict[str, Any] = metrics(totals[-1])
    result["by_proof_family"] = {
        family: metrics(totals[index])
        for index, family in enumerate(PROOF_FAMILIES)
        if totals[index, 1].item() > 0
    }
    return result
