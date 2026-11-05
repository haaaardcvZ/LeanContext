"""Tokenized datasets, paired view sampling, and exact EOS-aware collation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from lean_proof_training.common.jsonl import IndexedJsonlDataset


class PairedViewDataset(Dataset):
    """Aligned without/with views with deterministic complementary sampling."""

    def __init__(
        self,
        without_path: str | Path,
        with_path: str | Path,
        *,
        stage: str | None,
        seed: int,
        context_probability: float = 0.5,
    ) -> None:
        self.without = IndexedJsonlDataset(without_path, stage=stage)
        self.with_context = IndexedJsonlDataset(with_path, stage=stage)
        with_index = {sample_id: index for index, sample_id in enumerate(self.with_context.sample_ids)}
        self.pairs = [
            (index, with_index[sample_id])
            for index, sample_id in enumerate(self.without.sample_ids)
            if sample_id in with_index
        ]
        if not self.pairs:
            raise ValueError("paired views have no sample-ID intersection")
        self.seed = int(seed)
        self.context_probability = float(context_probability)
        if not 0.0 <= self.context_probability <= 1.0:
            raise ValueError("context_probability must be in [0, 1]")
        self.epoch = 0

    @property
    def sample_ids(self) -> list[str]:
        return [self.without.sample_ids[left] for left, _ in self.pairs]

    @property
    def proof_family_counts(self) -> Any:
        return self.without.proof_family_counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def _use_context(self, sample_id: str) -> bool:
        # At p=0.5, adjacent epochs are exact complements for every theorem.
        digest = hashlib.blake2b(
            f"{self.seed}:{sample_id}".encode("utf-8"), digest_size=8
        ).digest()
        unit = int.from_bytes(digest, "big") / float(1 << 64)
        if self.context_probability == 0.5:
            return bool((int.from_bytes(digest, "big") & 1) ^ (self.epoch & 1))
        shifted = (unit + self.epoch * 0.6180339887498949) % 1.0
        return shifted < self.context_probability

    def __getitem__(self, index: int) -> dict[str, Any]:
        without_index, with_index = self.pairs[index]
        sample_id = self.without.sample_ids[without_index]
        if self._use_context(sample_id):
            return self.with_context[with_index]
        return self.without[without_index]


class AlignedPairDataset(Dataset):
    """Return both paired views for information-gain evaluation."""

    def __init__(
        self,
        left_path: str | Path,
        right_path: str | Path,
        *,
        stage: str | None = None,
        allow_ids: set[str] | None = None,
    ) -> None:
        self.left = IndexedJsonlDataset(left_path, stage=stage, allow_ids=allow_ids)
        self.right = IndexedJsonlDataset(right_path, stage=stage, allow_ids=allow_ids)
        right_index = {sample_id: index for index, sample_id in enumerate(self.right.sample_ids)}
        self.pairs = [
            (index, right_index[sample_id])
            for index, sample_id in enumerate(self.left.sample_ids)
            if sample_id in right_index
        ]
        if not self.pairs:
            raise ValueError("aligned evaluation views have no intersection")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        left, right = self.pairs[index]
        left_row = self.left[left]
        right_row = self.right[right]
        if left_row["sample_id"] != right_row["sample_id"]:
            raise AssertionError("aligned-pair sample ID mismatch")
        return {"left": left_row, "right": right_row}


class TokenCollator:
    def __init__(self, pad_token_id: int, eos_token_id: int, multiple: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.multiple = int(multiple)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(row["input_ids"]) for row in rows)
        if self.multiple > 1:
            max_length = ((max_length + self.multiple - 1) // self.multiple) * self.multiple
        ids_batch: list[list[int]] = []
        masks: list[list[int]] = []
        prompt_lengths: list[int] = []
        target_lengths: list[int] = []
        for row in rows:
            ids = [int(token) for token in row["input_ids"]]
            prompt_length = int(row["prompt_length"])
            target_length = int(row["target_length"])
            if prompt_length + target_length != len(ids):
                raise ValueError(f"non-contiguous prompt/target span: {row['sample_id']}")
            target_ids = ids[prompt_length:]
            if target_ids[-1] != self.eos_token_id or target_ids.count(self.eos_token_id) != 1:
                raise ValueError(f"target must end in exactly one EOS: {row['sample_id']}")
            padding = max_length - len(ids)
            ids_batch.append(ids + [self.pad_token_id] * padding)
            masks.append([1] * len(ids) + [0] * padding)
            prompt_lengths.append(prompt_length)
            target_lengths.append(target_length)
        return {
            "input_ids": torch.tensor(ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "proof_families": [str(row["proof_family"]) for row in rows],
            "views": [str(row["view"]) for row in rows],
        }


class PairCollator:
    def __init__(self, collator: TokenCollator) -> None:
        self.collator = collator

    def __call__(self, rows: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
        return {
            "left": self.collator([row["left"] for row in rows]),
            "right": self.collator([row["right"] for row in rows]),
        }
