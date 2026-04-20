from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch


@dataclass(frozen=True)
class TrajectoryPlan:
    """Deprecated teacher-trace plan kept for compatibility in helper code."""

    sample_id: str
    edge_trace_local: tuple[int, ...]
    mode: str = "prefix"
    forced_prefix_len: int = 1
    source: str = "teacher"

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("TrajectoryPlan.sample_id must be non-empty.")
        if self.mode not in {"prefix", "full"}:
            raise ValueError(
                f"TrajectoryPlan.mode must be 'prefix' or 'full', got {self.mode!r}."
            )
        if self.forced_prefix_len < 0:
            raise ValueError(
                "TrajectoryPlan.forced_prefix_len must be >= 0, got "
                f"{self.forced_prefix_len}."
            )
        if not self.source:
            raise ValueError("TrajectoryPlan.source must be non-empty.")

    def forced_expand_count(self) -> int:
        if self.mode == "full":
            return len(self.edge_trace_local)
        return min(len(self.edge_trace_local), self.forced_prefix_len)


@dataclass(frozen=True)
class TrajectoryTrace:
    sample_id: str
    edge_trace_local: tuple[int, ...]
    traj_len: int
    terminal_log_reward: float
    priority: float
    insert_step: int
    source: str = "online"
    positive_edge_hit_count: int = 0
    positive_prefix_hit_len: int = 0
    relation_only_score_mean: float = 0.0
    relation_only_score_max: float = 0.0
    final_score_mean: float = 0.0
    teacher_forced_action_count: int = 0

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("TrajectoryTrace.sample_id must be non-empty.")
        if self.traj_len < 1:
            raise ValueError(
                f"TrajectoryTrace.traj_len must be >= 1, got {self.traj_len}."
            )
        if len(self.edge_trace_local) != self.traj_len - 1:
            raise ValueError(
                "TrajectoryTrace.edge_trace_local length must equal traj_len - 1, "
                f"got len={len(self.edge_trace_local)} and traj_len={self.traj_len}."
            )
        if self.priority <= 0.0:
            raise ValueError(
                f"TrajectoryTrace.priority must be > 0, got {self.priority}."
            )
        if not torch.isfinite(
            torch.tensor(self.terminal_log_reward, dtype=torch.float32)
        ):
            raise ValueError(
                "TrajectoryTrace.terminal_log_reward must be finite, "
                f"got {self.terminal_log_reward}."
            )
        if not self.source:
            raise ValueError("TrajectoryTrace.source must be non-empty.")
        if self.positive_edge_hit_count < 0:
            raise ValueError(
                "TrajectoryTrace.positive_edge_hit_count must be >= 0, got "
                f"{self.positive_edge_hit_count}."
            )
        if self.positive_prefix_hit_len < 0:
            raise ValueError(
                "TrajectoryTrace.positive_prefix_hit_len must be >= 0, got "
                f"{self.positive_prefix_hit_len}."
            )
        if self.teacher_forced_action_count < 0:
            raise ValueError(
                "TrajectoryTrace.teacher_forced_action_count must be >= 0, got "
                f"{self.teacher_forced_action_count}."
            )
        for name, value in (
            ("relation_only_score_mean", self.relation_only_score_mean),
            ("relation_only_score_max", self.relation_only_score_max),
            ("final_score_mean", self.final_score_mean),
        ):
            if not torch.isfinite(torch.tensor(value, dtype=torch.float32)):
                raise ValueError(f"TrajectoryTrace.{name} must be finite, got {value}.")


@dataclass(frozen=True)
class ReplayConfig:
    enabled: bool = False
    capacity: int = 4096
    sample_size: int = 4
    warmup_steps: int = 0
    min_size: int = 32
    loss_coef: float = 1.0
    reset_on_fit_start: bool = True
    priority_epsilon: float = 1e-6
    priority_exponent: float = 0.6
    importance_sampling_exponent: float = 0.4
    age_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"replay.capacity must be >= 1, got {self.capacity}.")
        if self.sample_size < 1:
            raise ValueError(
                f"replay.sample_size must be >= 1, got {self.sample_size}."
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"replay.warmup_steps must be >= 0, got {self.warmup_steps}."
            )
        if self.min_size < 1:
            raise ValueError(f"replay.min_size must be >= 1, got {self.min_size}.")
        if self.loss_coef < 0.0:
            raise ValueError(f"replay.loss_coef must be >= 0, got {self.loss_coef}.")
        if self.priority_epsilon <= 0.0:
            raise ValueError(
                f"replay.priority_epsilon must be > 0, got {self.priority_epsilon}."
            )
        if self.priority_exponent <= 0.0:
            raise ValueError(
                "replay.priority_exponent must be > 0, got "
                f"{self.priority_exponent}."
            )
        if self.importance_sampling_exponent < 0.0:
            raise ValueError(
                "replay.importance_sampling_exponent must be >= 0, got "
                f"{self.importance_sampling_exponent}."
            )
        if not 0.0 < self.age_decay <= 1.0:
            raise ValueError(
                f"replay.age_decay must be in (0, 1], got {self.age_decay}."
            )


@dataclass(frozen=True)
class ReplaySample:
    traces: list[TrajectoryTrace]
    indices: torch.Tensor
    importance_weights: torch.Tensor


class OnlineReplayBuffer:
    """Fixed-capacity online replay with true round-robin insertion."""

    def __init__(self, *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}.")
        self.capacity = int(capacity)
        self._items: list[TrajectoryTrace] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._items)

    def add_many(self, traces: Sequence[TrajectoryTrace]) -> None:
        for trace in traces:
            if len(self._items) < self.capacity:
                self._items.append(trace)
            else:
                self._items[self._next_index] = trace
            self._next_index = (self._next_index + 1) % self.capacity

    def sample(
        self,
        k: int,
        *,
        current_step: int | None = None,
        age_decay: float = 1.0,
        importance_sampling_exponent: float = 0.0,
    ) -> ReplaySample:
        if k < 1:
            raise ValueError(f"sample size k must be >= 1, got {k}.")
        if not self._items:
            return ReplaySample(
                traces=[],
                indices=torch.empty((0,), dtype=torch.long),
                importance_weights=torch.empty((0,), dtype=torch.float32),
            )
        priorities = self._effective_priorities(
            current_step=current_step,
            age_decay=age_decay,
        )
        probs = priorities / priorities.sum().clamp_min(torch.finfo(priorities.dtype).eps)
        sampled_idx = torch.multinomial(probs, num_samples=k, replacement=True)
        sampled_probs = probs.index_select(0, sampled_idx)
        if importance_sampling_exponent > 0.0:
            weights = (len(self._items) * sampled_probs).pow(
                -float(importance_sampling_exponent)
            )
            weights = weights / weights.max().clamp_min(torch.finfo(weights.dtype).eps)
        else:
            weights = torch.ones_like(sampled_probs)
        return ReplaySample(
            traces=[self._items[int(idx)] for idx in sampled_idx.tolist()],
            indices=sampled_idx,
            importance_weights=weights.to(dtype=torch.float32),
        )

    def update_priorities(
        self, indices: torch.Tensor | Sequence[int], priorities: torch.Tensor | Sequence[float]
    ) -> None:
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        priority_tensor = torch.as_tensor(priorities, dtype=torch.float32)
        if idx_tensor.ndim != 1 or priority_tensor.ndim != 1:
            raise ValueError("indices and priorities must be 1-D.")
        if idx_tensor.numel() != priority_tensor.numel():
            raise ValueError(
                f"indices length {idx_tensor.numel()} != priorities length {priority_tensor.numel()}."
            )
        for idx, priority in zip(idx_tensor.tolist(), priority_tensor.tolist()):
            if not 0 <= idx < len(self._items):
                continue
            self._items[idx] = replace(self._items[idx], priority=float(priority))

    def clear(self) -> None:
        self._items.clear()
        self._next_index = 0

    def _effective_priorities(
        self,
        *,
        current_step: int | None,
        age_decay: float,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                self._effective_priority(
                    trace,
                    current_step=current_step,
                    age_decay=age_decay,
                )
                for trace in self._items
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _effective_priority(
        trace: TrajectoryTrace,
        *,
        current_step: int | None,
        age_decay: float,
    ) -> float:
        priority = float(trace.priority)
        if current_step is None or age_decay >= 1.0:
            return priority
        age = max(int(current_step) - int(trace.insert_step), 0)
        return priority * (float(age_decay) ** age)


def residual_priority(
    loss_value: torch.Tensor | float,
    *,
    epsilon: float,
    exponent: float,
) -> torch.Tensor:
    loss_tensor = torch.as_tensor(loss_value, dtype=torch.float32)
    return loss_tensor.abs().clamp_min(float(epsilon)).pow(float(exponent))


def reward_priority_from_log_reward(
    log_reward: torch.Tensor,
    *,
    epsilon: float,
    max_priority_log_reward: float,
    min_priority_log_reward: float = -20.0,
) -> torch.Tensor:
    clipped = log_reward.detach().float().clamp(
        min=float(min_priority_log_reward),
        max=float(max_priority_log_reward),
    )
    reward_range = float(max_priority_log_reward) - float(min_priority_log_reward)
    normalized = (clipped - float(min_priority_log_reward)) / max(reward_range, 1e-8)
    return normalized.clamp_min(0.0) + float(epsilon)


__all__ = [
    "OnlineReplayBuffer",
    "ReplayConfig",
    "ReplaySample",
    "TrajectoryPlan",
    "TrajectoryTrace",
    "reward_priority_from_log_reward",
    "residual_priority",
]
