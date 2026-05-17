from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from omegaconf import DictConfig

from src.data.schema import RetrievalBatch
from src.training.config import RolloutRuntimeConfig
from src.weaver.policy import Policy
from src.weaver.rollout.engine import RolloutContext, RolloutEngine
from src.weaver.rollout.replay import (
    ReplayBatch,
    ReplaySampleBudget,
    ShortestPathReplaySource,
    transitions_from_rollouts,
)
from src.weaver.rollout.result import RolloutResult
from src.weaver.transitions import TransitionBatch


@dataclass(frozen=True, slots=True)
class RolloutChunk:
    rollouts: tuple[RolloutResult, ...]
    transitions: TransitionBatch | None = None

    @property
    def has_rollouts(self) -> bool:
        return len(self.rollouts) > 0

    @property
    def has_replay(self) -> bool:
        return self.transitions is not None and self.transitions.num_transitions > 0

    @property
    def num_policy_rollouts(self) -> int:
        return len(self.rollouts)

    @property
    def num_replay_transitions(self) -> int:
        if self.transitions is None:
            return 0
        return int(self.transitions.num_transitions)


@dataclass(frozen=True, slots=True)
class ReplayScheduleRow:
    until_progress: float
    policy_rollout: float
    replay_expand: float


@dataclass(frozen=True, slots=True)
class ReplaySchedule:
    enabled: bool
    rows: tuple[ReplayScheduleRow, ...]

    def weights_at(self, progress: float) -> ReplayScheduleRow:
        for row in self.rows:
            if progress <= row.until_progress:
                return row
        return self.rows[-1]


class RolloutRunner:
    """
    Thin execution wrapper around RolloutEngine.

    Owns:
        - train/eval rollout counts;
        - chunking;
        - optional replay sampling from policy-visited states.
    """

    def __init__(
        self,
        *,
        engine: RolloutEngine,
        rollout_cfg: RolloutRuntimeConfig,
        replay_source: ShortestPathReplaySource | None = None,
        replay_schedule: DictConfig | None = None,
        progress_fn: Callable[[], float] | None = None,
    ) -> None:
        self.engine = engine
        self.replay_source = replay_source

        self.train_num_rollout = int(rollout_cfg.train_num_rollout)
        self.eval_num_rollout = int(rollout_cfg.eval_num_rollout)
        self.train_chunk_size = int(rollout_cfg.train_chunk_size)
        self.eval_chunk_size = int(rollout_cfg.eval_chunk_size)

        self.replay_schedule = parse_replay_schedule(replay_schedule)
        self.progress_fn = progress_fn or zero_progress

    def train_chunks(
        self,
        *,
        policy: Policy,
        batch: RetrievalBatch,
        context: RolloutContext,
        temperature: float,
    ) -> Iterator[RolloutChunk]:
        for num_samples in chunk_sizes(
            total=self.train_num_rollout,
            chunk_size=self.train_chunk_size,
        ):
            yield self.train_chunk(
                policy=policy,
                batch=batch,
                context=context,
                num_samples=num_samples,
                temperature=temperature,
            )

    def train_chunk(
        self,
        *,
        policy: Policy,
        batch: RetrievalBatch,
        context: RolloutContext,
        num_samples: int,
        temperature: float,
    ) -> RolloutChunk:
        budget = self.sample_budget(num_samples)

        rollouts: tuple[RolloutResult, ...] = ()
        if budget.policy_rollout > 0:
            rollouts = tuple(
                self.engine.sample_rollouts(
                    policy=policy,
                    context=context,
                    num_rollouts=budget.policy_rollout,
                    temperature=temperature,
                )
            )

        replay: ReplayBatch | None = None
        if budget.replay_expand > 0:
            if self.replay_source is None:
                raise RuntimeError("replay_source is required when replay_expand > 0.")
            replay = self.replay_source.sample_from_rollouts(
                batch=batch,
                rollouts=rollouts,
                num_transitions=budget.replay_expand,
                device=context.device,
            )

        transition_parts: list[TransitionBatch] = []
        policy_transitions = transitions_from_rollouts(
            batch=batch,
            rollouts=rollouts,
            budget=self.engine.expand_budget,
            rollout_context=context,
            backward_kernel=self.replay_source.backward_kernel if self.replay_source is not None else ShortestPathReplaySource(expand_budget=self.engine.expand_budget).backward_kernel,
            device=context.device,
        )
        if policy_transitions is not None and policy_transitions.num_transitions > 0:
            transition_parts.append(policy_transitions)
        if replay is not None and replay.num_transitions > 0:
            transition_parts.append(replay.transitions)

        return RolloutChunk(
            rollouts=rollouts,
            transitions=None if not transition_parts else TransitionBatch.concat(transition_parts),
        )

    def eval_rollouts(
        self,
        *,
        policy: Policy,
        batch: RetrievalBatch,
        context: RolloutContext,
        temperature: float,
        num_rollouts: int | None = None,
        chunk_size: int | None = None,
    ) -> tuple[RolloutResult, ...]:
        total = self.eval_num_rollout if num_rollouts is None else int(num_rollouts)
        size = self.eval_chunk_size if chunk_size is None else int(chunk_size)

        rollouts: list[RolloutResult] = []
        for current_size in chunk_sizes(
            total=total,
            chunk_size=size,
        ):
            rollouts.extend(
                self.engine.sample_rollouts(
                    policy=policy,
                    context=context,
                    num_rollouts=current_size,
                    temperature=temperature,
                )
            )

        return tuple(rollouts)

    def sample_budget(self, total: int) -> ReplaySampleBudget:
        total = int(total)
        if not self.replay_schedule.enabled:
            return ReplaySampleBudget(
                policy_rollout=total,
                replay_expand=0,
            )

        weights = self.replay_schedule.weights_at(
            progress=float(self.progress_fn()),
        )
        return allocate_replay_budget(
            total=total,
            policy_weight=weights.policy_rollout,
            replay_weight=weights.replay_expand,
        )


def parse_replay_schedule(
    cfg: DictConfig | None,
) -> ReplaySchedule:
    if cfg is None or not bool(cfg.enabled):
        return ReplaySchedule(
            enabled=False,
            rows=(
                ReplayScheduleRow(
                    until_progress=1.0,
                    policy_rollout=1.0,
                    replay_expand=0.0,
                ),
            ),
        )

    rows = tuple(
        ReplayScheduleRow(
            until_progress=float(row.until_progress),
            policy_rollout=float(row.policy_rollout),
            replay_expand=float(row.replay_expand),
        )
        for row in cfg.schedule
    )
    if len(rows) == 0:
        raise ValueError("replay_schedule.schedule must not be empty.")

    return ReplaySchedule(
        enabled=True,
        rows=rows,
    )


def allocate_replay_budget(
    *,
    total: int,
    policy_weight: float,
    replay_weight: float,
) -> ReplaySampleBudget:
    total = int(total)
    if total <= 0:
        return ReplaySampleBudget(
            policy_rollout=0,
            replay_expand=0,
        )

    weight_sum = float(policy_weight) + float(replay_weight)
    if weight_sum <= 0.0:
        raise ValueError("policy_rollout and replay_expand weights cannot both be zero.")

    policy_raw = total * float(policy_weight) / weight_sum
    replay_raw = total * float(replay_weight) / weight_sum

    policy_count = int(policy_raw)
    replay_count = int(replay_raw)
    remainder = total - policy_count - replay_count
    if remainder > 0:
        policy_fraction = policy_raw - policy_count
        replay_fraction = replay_raw - replay_count
        if policy_fraction >= replay_fraction:
            policy_count += remainder
        else:
            replay_count += remainder

    return ReplaySampleBudget(
        policy_rollout=policy_count,
        replay_expand=replay_count,
    )


def chunk_sizes(
    *,
    total: int,
    chunk_size: int,
) -> Iterator[int]:
    remaining = int(total)
    size = int(chunk_size)

    if remaining <= 0:
        return

    if size <= 0:
        raise ValueError(f"chunk_size must be positive, got {size}.")

    while remaining > 0:
        current = min(size, remaining)
        remaining -= current
        yield current


def zero_progress() -> float:
    return 0.0


__all__ = [
    "ReplaySampleBudget",
    "ReplaySchedule",
    "ReplayScheduleRow",
    "RolloutChunk",
    "RolloutRunner",
    "allocate_replay_budget",
    "chunk_sizes",
    "parse_replay_schedule",
]
