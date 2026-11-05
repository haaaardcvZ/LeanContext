"""AR and discrete-diffusion proof-suffix generation backends."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from .proofs import decode_proof_tokens


def _trim_at_eos(ids: list[int], eos_id: int | None) -> list[int]:
    if eos_id is not None and eos_id in ids:
        return ids[: ids.index(eos_id)]
    return ids


class ARGenerator:
    def __init__(self, model: Any, tokenizer: Any, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        count: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
    ) -> list[str]:
        if count <= 0:
            return []
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        # Transformers 4.45 does not reliably accept a `generator=` kwarg in
        # GenerationMixin. Each rank generates sequentially, so seeding the
        # rank-local torch RNG here is deterministic and does not race.
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        sampled = temperature > 0
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": sampled,
            "num_return_sequences": int(count),
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "eos_token_id": int(self.tokenizer.eos_token_id),
            "use_cache": True,
        }
        if sampled:
            kwargs.update(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
        elif count != 1:
            raise ValueError("greedy AR decoding produces exactly one candidate")
        output = self.model.generate(**kwargs)
        start = len(prompt_ids)
        return [
            decode_proof_tokens(self.tokenizer, row[start:])
            for row in output
        ]


def top_k_top_p_filter(
    logits: torch.Tensor, *, top_k: int, top_p: float
) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0 and top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits.masked_fill_(remove, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


class DLLMGenerator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        *,
        mask_token_id: int | None,
        num_steps: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        resolved_mask = mask_token_id
        if resolved_mask is None:
            resolved_mask = getattr(tokenizer, "mask_token_id", None)
        if resolved_mask is None:
            raise ValueError("DLLM generation requires mask_token_id")
        self.mask_token_id = int(resolved_mask)
        self.num_steps = int(num_steps)
        if self.num_steps <= 0:
            raise ValueError("num_diffusion_steps must be positive")

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            probabilities = F.softmax(logits.float(), dim=-1)
            tokens = probabilities.argmax(dim=-1)
        else:
            scaled = top_k_top_p_filter(
                logits.float() / float(temperature), top_k=top_k, top_p=top_p
            )
            probabilities = F.softmax(scaled, dim=-1)
            tokens = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            ).squeeze(-1)
        confidence = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return tokens, confidence

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        count: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
    ) -> list[str]:
        if count <= 0:
            return []
        generation_length = int(max_new_tokens)
        if generation_length <= 0:
            raise ValueError("max_new_tokens must be positive")
        prompt_length = len(prompt_ids)
        ids = torch.tensor(
            [prompt_ids + [self.mask_token_id] * generation_length] * count,
            dtype=torch.long,
            device=self.device,
        )
        masked = torch.ones(
            (count, generation_length), dtype=torch.bool, device=self.device
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))

        for step in range(self.num_steps):
            if not bool(masked.any().item()):
                break
            output = self.model(input_ids=ids, use_cache=False)
            logits = output.logits[:, prompt_length:, :]
            steps_left = max(1, self.num_steps - step)
            for row_index in range(count):
                positions = torch.nonzero(masked[row_index], as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                sampled, confidence = self._sample_tokens(
                    logits[row_index, positions],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=generator,
                )
                commit_count = max(1, math.ceil(positions.numel() / steps_left))
                chosen = torch.topk(
                    confidence, k=min(commit_count, confidence.numel())
                ).indices
                absolute = positions[chosen]
                ids[row_index, prompt_length + absolute] = sampled[chosen]
                masked[row_index, absolute] = False

        suffixes = ids[:, prompt_length:].tolist()
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        return [
            decode_proof_tokens(self.tokenizer, _trim_at_eos(list(suffix), eos_id))
            for suffix in suffixes
        ]
