from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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


def resolve_module_float_dtype(module: nn.Module) -> torch.dtype | None:
    for tensor in module.parameters(recurse=False):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    for tensor in module.buffers(recurse=False):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return None


def align_float_input_dtype(tensor: torch.Tensor, *, module: nn.Module) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor
    target_dtype = resolve_module_float_dtype(module)
    if target_dtype is None or tensor.dtype == target_dtype:
        return tensor
    if torch.is_autocast_enabled():
        return tensor
    return tensor.to(dtype=target_dtype)


def masked_softmax_in_float32(
    values: torch.Tensor,
    *,
    mask: torch.Tensor,
    dim: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    scores = values.to(dtype=torch.float32).masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=dim)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    if output_dtype is not None and probs.dtype != output_dtype:
        probs = probs.to(dtype=output_dtype)
    return probs


def logsigmoid_in_float32(
    values: torch.Tensor, *, output_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    out = F.logsigmoid(values.to(dtype=torch.float32))
    if out.dtype != output_dtype:
        out = out.to(dtype=output_dtype)
    return out


def sigmoid_in_float32(
    values: torch.Tensor, *, output_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    out = torch.sigmoid(values.to(dtype=torch.float32))
    if out.dtype != output_dtype:
        out = out.to(dtype=output_dtype)
    return out


def binary_cross_entropy_with_logits_in_float32(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits.to(dtype=torch.float32),
        targets.to(dtype=torch.float32),
        reduction=reduction,
    )


__all__ = [
    "align_float_input_dtype",
    "binary_cross_entropy_with_logits_in_float32",
    "infer_float_dtype_from_precision",
    "logsigmoid_in_float32",
    "masked_softmax_in_float32",
    "normalize_precision",
    "resolve_module_float_dtype",
    "sigmoid_in_float32",
]
