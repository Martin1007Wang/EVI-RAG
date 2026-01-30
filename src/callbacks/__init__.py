"""Callback exports (lazy to avoid importing heavy deps at module import time)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["DualFlowEvalMetrics", "DualFlowRolloutArtifactWriter"]

if TYPE_CHECKING:  # pragma: no cover
    from .dual_flow_eval_metrics import DualFlowEvalMetrics
    from .dual_flow_rollout_artifact_writer import DualFlowRolloutArtifactWriter


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "DualFlowEvalMetrics":
        from .dual_flow_eval_metrics import DualFlowEvalMetrics

        return DualFlowEvalMetrics
    if name == "DualFlowRolloutArtifactWriter":
        from .dual_flow_rollout_artifact_writer import DualFlowRolloutArtifactWriter

        return DualFlowRolloutArtifactWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
