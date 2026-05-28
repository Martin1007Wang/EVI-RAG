from __future__ import annotations

from typing import TypeAlias

import torch

Scalar: TypeAlias = torch.Tensor | float | int


def require_scalar_tensor(
    value: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}.")
    return value


def validate_scalar(
    value: Scalar,
    *,
    name: str,
) -> Scalar:
    if isinstance(value, torch.Tensor):
        return require_scalar_tensor(value, name=name)
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be bool.")
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"{name} must be Tensor, float, or int; got {type(value)!r}.")


def detach_scalar(
    value: Scalar,
    *,
    name: str,
) -> Scalar:
    value = validate_scalar(value, name=name)
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value
