from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.eval.groups import flatten_metric_groups
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
        best_of_k_values: Sequence[int],
        utility_k: int,
        utility_lambda: float,
        exclude_anchors_from_retrieved: bool,
        use_reachable_targets: bool,
    ) -> None:
        self.best_of_k_values = _positive_int_tuple(
            best_of_k_values,
            name="best_of_k_values",
        )
        self.utility_k = int(utility_k)
        if self.utility_k < 1:
            raise ValueError(f"utility_k must be positive, got {self.utility_k}.")
        self.utility_lambda = float(utility_lambda)
        if self.utility_lambda < 0.0:
            raise ValueError(
                f"utility_lambda must be non-negative, got {self.utility_lambda}."
            )
        self.exclude_anchors_from_retrieved = bool(exclude_anchors_from_retrieved)
        self.use_reachable_targets = bool(use_reachable_targets)

    def eval_metrics(
        self,
        *,
        rollout_samples: Sequence[RolloutResult],
        batch: RetrievalBatch,
        stage: str,
    ) -> dict[str, float]:
        grouped = evaluate_rollout_samples(
            rollout_samples=rollout_samples,
            batch=batch,
            best_of_k_values=self.best_of_k_values,
            utility_k=self.utility_k,
            utility_lambda=self.utility_lambda,
            exclude_anchors_from_retrieved=self.exclude_anchors_from_retrieved,
            use_reachable_targets=self.use_reachable_targets,
        )

        return flatten_metric_groups(
            grouped,
            prefix=str(stage),
        )


def _positive_int_tuple(
    values: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values if int(value) >= 1}))

    if not result:
        raise ValueError(f"{name} must contain at least one positive integer.")

    return result


__all__ = [
    "WeaverMetricSuite",
]
