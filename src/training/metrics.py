from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.eval.rollout import evaluate_rollout_samples

if TYPE_CHECKING:
    from src.data.schema import RetrievalBatch
    from src.weaver.context import GraphContext, TargetContext
    from src.weaver.nn.feature_encoder import EncodedFeatures
    from src.weaver.rollout.result import RolloutResult
    from src.weaver.utility import TrueTerminalReward


class WeaverMetricSuite:
    """
    Logging-facing metric facade.

    It evaluates rollout retrieval quality only.
    """

    def __init__(
        self,
        *,
        k_windows: Sequence[int] | None = None,
        best_of_k: int | None = None,
        exclude_anchors_from_retrieved: bool,
        use_reachable_targets: bool,
    ) -> None:
        if k_windows is None:
            if best_of_k is None:
                k_windows = (1, 2, 4, 8, 16)
            else:
                k = int(best_of_k)
                if k < 1:
                    raise ValueError(f"best_of_k must be positive, got {k}.")
                values = [1]
                while values[-1] < k:
                    values.append(values[-1] * 2)
                k_windows = tuple(x for x in values if x <= k)
        self.k_windows = tuple(int(k) for k in k_windows)
        if not self.k_windows or any(k < 1 for k in self.k_windows):
            raise ValueError(f"k_windows must contain positive integers, got {self.k_windows}.")
        self.exclude_anchors_from_retrieved = bool(exclude_anchors_from_retrieved)
        self.use_reachable_targets = bool(use_reachable_targets)

    def eval_metrics(
        self,
        *,
        rollout_samples: Sequence[RolloutResult],
        batch: RetrievalBatch,
        stage: str,
        context: GraphContext | None = None,
        features: EncodedFeatures | None = None,
        reward_model: TrueTerminalReward | None = None,
        target_context: TargetContext | None = None,
    ) -> dict[str, float]:
        metrics = evaluate_rollout_samples(
            rollout_samples=rollout_samples,
            batch=batch,
            k_windows=self.k_windows,
            exclude_anchors_from_retrieved=self.exclude_anchors_from_retrieved,
            use_reachable_targets=self.use_reachable_targets,
            context=context,
            features=features,
            reward_model=reward_model,
            target_context=target_context,
        )

        if not stage:
            return metrics
        return {f"{stage}/{name}": float(value) for name, value in metrics.items()}


__all__ = [
    "WeaverMetricSuite",
]
