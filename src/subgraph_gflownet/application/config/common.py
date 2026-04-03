from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf


def to_plain_mapping(node: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(node, DictConfig):
        container = OmegaConf.to_container(node, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"Expected {field_name} to resolve to a mapping.")
        return dict(container)
    if isinstance(node, Mapping):
        return {str(key): deepcopy(value) for key, value in node.items()}
    raise TypeError(f"Expected {field_name} to be a mapping, got {type(node)!r}.")


def deep_merge(
    *, base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(base=merged[key], override=value)
            continue
        merged[key] = deepcopy(value)
    return merged


__all__ = ["deep_merge", "to_plain_mapping"]
