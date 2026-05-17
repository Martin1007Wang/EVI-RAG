"""Callback exports (lazy to avoid importing heavy deps at module import time)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "LoggedMetricEarlyStopping",
    "LocalMetricsWriter",
    "StepEarlyStopping",
]

if TYPE_CHECKING:  # pragma: no cover
    from .step_early_stopping import LoggedMetricEarlyStopping
    from .local_metrics_writer import LocalMetricsWriter
    from .step_early_stopping import StepEarlyStopping


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "LoggedMetricEarlyStopping":
        from .step_early_stopping import LoggedMetricEarlyStopping

        return LoggedMetricEarlyStopping
    if name == "LocalMetricsWriter":
        from .local_metrics_writer import LocalMetricsWriter

        return LocalMetricsWriter
    if name == "StepEarlyStopping":
        from .step_early_stopping import StepEarlyStopping

        return StepEarlyStopping
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
