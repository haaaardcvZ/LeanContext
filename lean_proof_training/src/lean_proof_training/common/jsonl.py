"""Streaming and indexed JSONL utilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

from torch.utils.data import Dataset

from lean_proof_training.common.io import iter_jsonl


class IndexedJsonlDataset(Dataset):
    """Random-access JSONL with stage and sample-ID filtering."""

    def __init__(
        self,
        path: str | Path,
        *,
        stage: str | None = None,
        allow_ids: set[str] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.offsets: list[int] = []
        self.sample_ids: list[str] = []
        self.proof_family_counts: Counter[str] = Counter()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if stage is not None and str(row.get("training_stage")) != stage:
                    continue
                if allow_ids is not None and sample_id not in allow_ids:
                    continue
                self.offsets.append(offset)
                self.sample_ids.append(sample_id)
                self.proof_family_counts[str(row["proof_family"])] += 1
        if not self.offsets:
            raise ValueError(f"empty filtered dataset: {self.path}")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError(f"duplicate sample IDs: {self.path}")
        self._handle: BinaryIO | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline())


def load_id_set(path: str | Path) -> set[str]:
    return {str(row["sample_id"]) for row in iter_jsonl(path)}
