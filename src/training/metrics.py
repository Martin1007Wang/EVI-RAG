from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.eval.rollout import evaluate_rollout_samples

if TYPE_CHECKING:
    from src.data.schema import RetrievalBatch
    from src.weaver.context import GraphContext
    from src.weaver.rollout.trajectory import TrajectoryBatch


class WeaverMetricSuite:
    """
    Logging-facing metric facade.

    It evaluates rollout retrieval quality only.
    """

    def __init__(
        self,
        *,
        k_windows: Sequence[int],
        exclude_anchors_from_retrieved: bool,
        use_reachable_targets: bool,
        enable_terminal_diagnostics: bool,
    ) -> None:
        self.k_windows = tuple(int(k) for k in k_windows)
        if not self.k_windows or any(k < 1 for k in self.k_windows):
            raise ValueError(f"k_windows must contain positive integers, got {self.k_windows}.")
        self.exclude_anchors_from_retrieved = bool(exclude_anchors_from_retrieved)
        self.use_reachable_targets = bool(use_reachable_targets)
        self.enable_terminal_diagnostics = bool(enable_terminal_diagnostics)

    def eval_metrics(
        self,
        *,
        rollout_samples: TrajectoryBatch,
        batch: RetrievalBatch,
        stage: str,
        context: GraphContext | None = None,
    ) -> dict[str, float]:
        if context is None:
            raise ValueError("context is required to evaluate rollout metrics.")

        metrics = evaluate_rollout_samples(
            trajectories=rollout_samples,
            batch=batch,
            context=context,
            k_windows=self.k_windows,
            exclude_anchors_from_retrieved=self.exclude_anchors_from_retrieved,
            use_reachable_targets=self.use_reachable_targets,
            enable_terminal_diagnostics=self.enable_terminal_diagnostics,
        )

        if not stage:
            return metrics
        return {f"{stage}/{name}": float(value) for name, value in metrics.items()}


__all__ = [
    "WeaverMetricSuite",
]
