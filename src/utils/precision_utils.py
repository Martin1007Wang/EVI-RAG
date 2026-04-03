from __future__ import annotations

import torch


_PRECISION_ALIASES = {
    "16": "16-mixed",
    "32": "32-true",
    "64": "64-true",
    "bf16": "bf16-mixed",
}


def normalize_precision(precision: object) -> str | None:
    if precision is None:
        return None
    normalized = str(precision).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return _PRECISION_ALIASES.get(normalized, normalized)


def infer_float_dtype_from_precision(precision: object) -> torch.dtype | None:
    normalized = normalize_precision(precision)
    if normalized is None:
        return None
    if normalized.startswith("bf16"):
        return torch.bfloat16
    if normalized.startswith("16"):
        return torch.float16
    if normalized.startswith("32"):
        return torch.float32
    if normalized.startswith("64"):
        return torch.float64
    return None


__all__ = [
    "infer_float_dtype_from_precision",
    "normalize_precision",
]
