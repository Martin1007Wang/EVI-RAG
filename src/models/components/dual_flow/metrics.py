from __future__ import annotations

import torch

from .constants import _STANDARD_METRICS


class DualFlowMetricsMixin:
    @staticmethod
    def _filter_metrics(metrics: dict[str, torch.Tensor], keep: set[str]) -> dict[str, torch.Tensor]:
        if not metrics:
            return {}
        return {name: value for name, value in metrics.items() if name in keep}

    def _get_standard_metrics(self, stage: str) -> set[str]:
        key = str(stage).strip().lower()
        if key not in _STANDARD_METRICS:
            raise ValueError(f"Unsupported metrics stage: {stage!r}.")
        return _STANDARD_METRICS[key]


__all__ = ["DualFlowMetricsMixin"]
