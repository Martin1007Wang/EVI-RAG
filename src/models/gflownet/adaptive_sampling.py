from __future__ import annotations

from dataclasses import dataclass
import math

from src.models.configs import AdaptiveSamplingConfig


@dataclass(frozen=True)
class AdaptiveSamplingDecision:
    rollout_batch_size: int
    temperature_multiplier: float
    sampling_temperature: float


@dataclass(frozen=True)
class AdaptiveSamplingMetrics:
    success_rate: float
    unique_success_paths_per_100_rollouts: float
    subtb_residual_variance_per_batch: float
    start_node_entropy_normalized: float


class AdaptiveSamplingController:
    def __init__(
        self,
        *,
        config: AdaptiveSamplingConfig,
        base_rollout_batch_size: int,
    ) -> None:
        self.config = config
        self.base_rollout_batch_size = int(base_rollout_batch_size)
        if self.base_rollout_batch_size < 1:
            raise ValueError("base_rollout_batch_size must be >= 1.")
        min_rollout = (
            self.base_rollout_batch_size
            if config.min_rollout_batch_size is None
            else int(config.min_rollout_batch_size)
        )
        max_rollout = (
            max(self.base_rollout_batch_size, min_rollout)
            if config.max_rollout_batch_size is None
            else int(config.max_rollout_batch_size)
        )
        if min_rollout > self.base_rollout_batch_size:
            min_rollout = self.base_rollout_batch_size
        if max_rollout < self.base_rollout_batch_size:
            max_rollout = self.base_rollout_batch_size
        self.min_rollout_batch_size = min_rollout
        self.max_rollout_batch_size = max_rollout
        self._current_rollout_batch_size = min(
            max(self.base_rollout_batch_size, self.min_rollout_batch_size),
            self.max_rollout_batch_size,
        )
        self._temperature_multiplier = 1.0
        self._observed_steps = 0
        self._ema_success_rate: float | None = None
        self._ema_unique_success_paths_per_100_rollouts: float | None = None
        self._ema_subtb_residual_variance_per_batch: float | None = None
        self._ema_start_node_entropy_normalized: float | None = None

    @property
    def current_rollout_batch_size(self) -> int:
        return int(self._current_rollout_batch_size)

    @property
    def temperature_multiplier(self) -> float:
        return float(self._temperature_multiplier)

    def decision(self, *, base_temperature: float) -> AdaptiveSamplingDecision:
        resolved_temperature = float(base_temperature) * float(
            self._temperature_multiplier
        )
        return AdaptiveSamplingDecision(
            rollout_batch_size=int(self._current_rollout_batch_size),
            temperature_multiplier=float(self._temperature_multiplier),
            sampling_temperature=max(resolved_temperature, 1.0e-6),
        )

    def observe(self, metrics: AdaptiveSamplingMetrics) -> None:
        if not self.config.enabled:
            return
        self._observed_steps += 1
        self._ema_success_rate = self._update_ema(
            current=self._ema_success_rate,
            value=metrics.success_rate,
        )
        self._ema_unique_success_paths_per_100_rollouts = self._update_ema(
            current=self._ema_unique_success_paths_per_100_rollouts,
            value=metrics.unique_success_paths_per_100_rollouts,
        )
        self._ema_subtb_residual_variance_per_batch = self._update_ema(
            current=self._ema_subtb_residual_variance_per_batch,
            value=metrics.subtb_residual_variance_per_batch,
        )
        self._ema_start_node_entropy_normalized = self._update_ema(
            current=self._ema_start_node_entropy_normalized,
            value=metrics.start_node_entropy_normalized,
        )
        if self._observed_steps <= int(self.config.warmup_steps):
            return
        self._maybe_adjust_rollout_batch_size()
        self._maybe_adjust_temperature_multiplier()

    def snapshot_metrics(self) -> dict[str, float]:
        metrics = {
            "rollout_batch_size": float(self._current_rollout_batch_size),
            "sampling_temperature_multiplier": float(self._temperature_multiplier),
        }
        if self._ema_success_rate is not None:
            metrics["adaptive_success_rate_ema"] = float(self._ema_success_rate)
        if self._ema_unique_success_paths_per_100_rollouts is not None:
            metrics["adaptive_unique_success_paths_per_100_rollouts_ema"] = float(
                self._ema_unique_success_paths_per_100_rollouts
            )
        if self._ema_subtb_residual_variance_per_batch is not None:
            metrics["adaptive_subtb_residual_variance_ema"] = float(
                self._ema_subtb_residual_variance_per_batch
            )
        if self._ema_start_node_entropy_normalized is not None:
            metrics["adaptive_start_node_entropy_normalized_ema"] = float(
                self._ema_start_node_entropy_normalized
            )
        return metrics

    def _maybe_adjust_rollout_batch_size(self) -> None:
        success_rate = self._require_metric(self._ema_success_rate, "success_rate")
        unique_success = self._require_metric(
            self._ema_unique_success_paths_per_100_rollouts,
            "unique_success_paths_per_100_rollouts",
        )
        subtb_variance = self._require_metric(
            self._ema_subtb_residual_variance_per_batch,
            "subtb_residual_variance_per_batch",
        )
        need_more_budget = (
            success_rate < float(self.config.low_success_rate_threshold)
            or unique_success
            < float(self.config.low_unique_success_paths_per_100_rollouts)
            or subtb_variance > float(self.config.high_subtb_residual_variance)
        )
        can_use_less_budget = (
            success_rate > float(self.config.high_success_rate_threshold)
            and unique_success
            > float(self.config.high_unique_success_paths_per_100_rollouts)
            and subtb_variance < float(self.config.low_subtb_residual_variance)
        )
        if need_more_budget:
            grown = int(
                math.ceil(
                    float(self._current_rollout_batch_size)
                    * float(self.config.rollout_growth_factor)
                )
            )
            self._current_rollout_batch_size = min(grown, self.max_rollout_batch_size)
            return
        if can_use_less_budget:
            shrunk = int(
                math.floor(
                    float(self._current_rollout_batch_size)
                    * float(self.config.rollout_shrink_factor)
                )
            )
            self._current_rollout_batch_size = max(shrunk, self.min_rollout_batch_size)

    def _maybe_adjust_temperature_multiplier(self) -> None:
        entropy = self._require_metric(
            self._ema_start_node_entropy_normalized,
            "start_node_entropy_normalized",
        )
        success_rate = self._require_metric(self._ema_success_rate, "success_rate")
        unique_success = self._require_metric(
            self._ema_unique_success_paths_per_100_rollouts,
            "unique_success_paths_per_100_rollouts",
        )
        if entropy < float(self.config.low_start_entropy_normalized):
            self._temperature_multiplier = min(
                self._temperature_multiplier
                * float(self.config.temperature_multiplier_up),
                float(self.config.max_temperature_multiplier),
            )
            return
        if entropy > float(self.config.high_start_entropy_normalized) and (
            success_rate < float(self.config.high_success_rate_threshold)
            or unique_success
            > float(self.config.high_unique_success_paths_per_100_rollouts)
        ):
            self._temperature_multiplier = max(
                self._temperature_multiplier
                * float(self.config.temperature_multiplier_down),
                float(self.config.min_temperature_multiplier),
            )

    def _update_ema(self, *, current: float | None, value: float) -> float:
        if current is None or not math.isfinite(current):
            return float(value)
        alpha = float(self.config.ema_alpha)
        return (alpha * float(value)) + ((1.0 - alpha) * current)

    @staticmethod
    def _require_metric(value: float | None, name: str) -> float:
        if value is None or not math.isfinite(value):
            raise RuntimeError(f"AdaptiveSamplingController missing finite {name}.")
        return float(value)


__all__ = [
    "AdaptiveSamplingController",
    "AdaptiveSamplingDecision",
    "AdaptiveSamplingMetrics",
]
