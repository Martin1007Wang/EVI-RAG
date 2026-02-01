from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable


def require_cfg_mapping(raw: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return raw


def validate_cfg_keys(
    raw: Mapping[str, Any],
    *,
    required: set[str],
    optional: Iterable[str] = (),
    name: str,
) -> None:
    missing = set(required) - set(raw.keys())
    if missing:
        raise ValueError(f"{name} missing keys: {sorted(missing)}")
    optional = set(optional)
    extra = set(raw.keys()) - set(required) - optional
    if extra:
        raise ValueError(f"{name} has unsupported keys: {sorted(extra)}")


__all__ = [
    "require_cfg_mapping",
    "validate_cfg_keys",
]
