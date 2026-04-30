from __future__ import annotations

from typing import Any

import torch


def step_ratio(counts: torch.Tensor, lengths: torch.Tensor) -> float:
    denom = float(lengths.float().sum().item())
    if denom <= 0.0:
        return 0.0
    return float(counts.float().sum().item()) / denom


def mean_or_zero(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def std_or_zero(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return float(values.float().std(unbiased=False).item())


def scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def flatten_metric_groups(
    results: dict[str, dict[str, float]],
    *,
    prefix: str,
) -> dict[str, float]:
    return {
        f"{prefix}/{group}/{name}": value
        for group, metrics in results.items()
        for name, value in metrics.items()
    }


__all__ = [
    "flatten_metric_groups",
    "mean_or_zero",
    "scalar_float",
    "std_or_zero",
    "step_ratio",
]
