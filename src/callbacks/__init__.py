"""Callback exports (lazy to avoid importing heavy deps at module import time)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["GFlowNetEvalMetrics", "GFlowNetRolloutArtifactWriter"]

if TYPE_CHECKING:  # pragma: no cover
    from .gflownet_eval_metrics import GFlowNetEvalMetrics
    from .gflownet_rollout_artifact_writer import GFlowNetRolloutArtifactWriter


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "GFlowNetEvalMetrics":
        from .gflownet_eval_metrics import GFlowNetEvalMetrics

        return GFlowNetEvalMetrics
    if name == "GFlowNetRolloutArtifactWriter":
        from .gflownet_rollout_artifact_writer import GFlowNetRolloutArtifactWriter

        return GFlowNetRolloutArtifactWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
