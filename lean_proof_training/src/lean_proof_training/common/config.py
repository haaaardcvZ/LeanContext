"""YAML loading, inheritance, overrides, and environment expansion."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    active = set() if seen is None else seen
    if path in active:
        raise ValueError(f"cyclic YAML inheritance: {path}")
    active.add(path)
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise ValueError(f"config is not a mapping: {path}")
    parent = current.pop("extends", None)
    if parent is None:
        return current
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge(load_yaml(parent_path, active), current)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be KEY=VALUE: {override!r}")
        dotted, raw = override.split("=", 1)
        keys = [part for part in dotted.strip().split(".") if part]
        if not keys:
            raise ValueError(f"empty override key: {override!r}")
        target = result
        for key in keys[:-1]:
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"cannot descend through non-mapping key: {dotted}")
            target = child
        target[keys[-1]] = yaml.safe_load(raw)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = apply_overrides(load_yaml(path), overrides or [])
    config = _expand(config)
    config["_config_path"] = str(Path(path).expanduser().resolve())
    return config
