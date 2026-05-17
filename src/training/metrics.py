from __future__ import annotations

from typing import TYPE_CHECKING

from src.eval.rollout import evaluate_rollout_samples

if TYPE_CHECKING:
    from src.data.schema import RetrievalBatch
    from src.weaver.rollout.result import RolloutResult


class WeaverMetricSuite:
    """
    Logging-facing metric facade.

    It evaluates rollout retrieval quality only.
    """

    def __init__(
        self,
        *,
        best_of_k: int,
        exclude_anchors_from_retrieved: bool,
        use_reachable_targets: bool,
    ) -> None:
        self.best_of_k = int(best_of_k)
        if self.best_of_k < 1:
            raise ValueError(f"best_of_k must be positive, got {self.best_of_k}.")
        self.exclude_anchors_from_retrieved = bool(exclude_anchors_from_retrieved)
        self.use_reachable_targets = bool(use_reachable_targets)

    def eval_metrics(
        self,
        *,
        rollout_samples: Sequence[RolloutResult],
        batch: RetrievalBatch,
        stage: str,
    ) -> dict[str, float]:
        metrics = evaluate_rollout_samples(
            rollout_samples=rollout_samples,
            batch=batch,
            best_of_k=self.best_of_k,
            exclude_anchors_from_retrieved=self.exclude_anchors_from_retrieved,
            use_reachable_targets=self.use_reachable_targets,
        )

        if not stage:
            return metrics
        return {f"{stage}/{name}": float(value) for name, value in metrics.items()}


__all__ = [
    "WeaverMetricSuite",
]
