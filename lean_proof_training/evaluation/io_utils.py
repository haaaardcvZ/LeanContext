"""Small dependency-free helpers shared by the evaluation commands."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    config = _expand(config)
    config["_config_path"] = str(path)
    return config


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_int(*parts: Any, bits: int = 63) -> int:
    text = "\0".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << bits) - 1)


def select_rows(
    rows: Iterable[dict[str, Any]], *, max_samples: int, seed: int
) -> list[dict[str, Any]]:
    selected = list(rows)
    if max_samples <= 0 or max_samples >= len(selected):
        return selected
    # A seeded sample is less biased than taking the first N source-ordered rows.
    generator = random.Random(seed)
    indices = sorted(generator.sample(range(len(selected)), max_samples))
    return [selected[index] for index in indices]


def require_path(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError("missing configuration key: " + ".".join(keys))
        value = value[key]
    path = Path(str(value))
    if "$" in str(path):
        raise ValueError(
            f"unexpanded environment variable in {'.'.join(keys)}: {path}"
        )
    return path
