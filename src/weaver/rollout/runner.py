from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.data.schema import RetrievalBatch
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.policy import ForwardPolicy
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.rollout.replay import (
    ReplayBatch,
    ReplayBuilder,
    ReplaySampleBudget,
    ReplaySource,
)
from src.weaver.rollout.result import RolloutResult
from src.weaver.transition import (
    SRC_POLICY,
    SRC_REPLAY,
    TrainingBatch,
)
from src.weaver.utility import TrueTerminalReward


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    rollouts: tuple[RolloutResult, ...]
    training: TrainingBatch | None = None
    replay: ReplayBatch | None = None

    @property
    def num_rollouts(self) -> int:
        return len(self.rollouts)

    @property
    def num_transitions(self) -> int:
        if self.training is None:
            return 0
        return int(self.training.num_items)

    @property
    def has_transitions(self) -> bool:
        return self.num_transitions > 0

    @property
    def num_replay_trajectories(self) -> int:
        if self.replay is None:
            return 0
        return int(self.replay.num_trajectories)

    @property
    def num_replay_transitions(self) -> int:
        if self.training is None:
            return 0
        return int(self.training.expansions.meta.source_ids.eq(SRC_REPLAY).sum().item())

    @property
    def num_replay_terminal_transitions(self) -> int:
        if self.training is None:
            return 0
        return int(self.training.terminals.meta.source_ids.eq(SRC_REPLAY).sum().item())


@dataclass(frozen=True, slots=True)
class ReplayScheduleRow:
    until_progress: float
    policy_rollout: float
    replay_expand: float


@dataclass(frozen=True, slots=True)
class ReplaySchedule:
    rows: tuple[ReplayScheduleRow, ...]

    def weights_at(self, progress: float) -> ReplayScheduleRow:
        for row in self.rows:
            if progress <= row.until_progress:
                return row
        return self.rows[-1]


class RolloutRunner:
    def __init__(
        self,
        *,
        engine: RolloutEngine,
        train_num_rollouts: int,
        eval_num_rollouts: int,
        replay_source: ReplaySource | None = None,
        replay_builder: ReplayBuilder | None = None,
        replay_schedule: ReplaySchedule | None = None,
        progress_fn: Callable[[], float] | None = None,
    ) -> None:
        self.engine = engine
        self.train_num_rollouts = int(train_num_rollouts)
        self.eval_num_rollouts = int(eval_num_rollouts)
        self.replay_source = replay_source
        self.replay_builder = replay_builder
        self.replay_schedule = replay_schedule
        self.progress_fn = progress_fn or zero_progress

    def train_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        batch: RetrievalBatch,
        context: GraphContext,
        features: EncodedFeatures,
        reward_model: TrueTerminalReward | None = None,
        target_context: TargetContext | None = None,
    ) -> RolloutBatch:
        progress = float(self.progress_fn())
        budget = self.sample_budget(
            self.train_num_rollouts,
            progress=progress,
        )
        rollouts, policy_training = self.policy_rollouts(
            policy=policy,
            context=context,
            features=features,
            num_rollouts=budget.policy_rollout,
        )
        replay = self.replay_trajectories(
            batch=batch,
            rollouts=rollouts,
            context=context,
            num_trajectories=budget.replay_expand,
            reward_model=reward_model,
            target_context=target_context,
            progress=progress,
        )
        training = self.training_batch(
            rollouts=rollouts,
            policy_training=policy_training,
            replay=replay,
            context=context,
        )
        return RolloutBatch(
            rollouts=rollouts,
            training=training,
            replay=replay,
        )

    def eval_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: EncodedFeatures,
        num_rollouts: int | None = None,
    ) -> tuple[RolloutResult, ...]:
        total = self.eval_num_rollouts if num_rollouts is None else int(num_rollouts)
        rollouts, _ = self.policy_rollouts(
            policy=policy,
            context=context,
            features=features,
            num_rollouts=total,
        )
        return rollouts

    def policy_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: EncodedFeatures,
        num_rollouts: int,
    ) -> tuple[tuple[RolloutResult, ...], TrainingBatch | None]:
        num_rollouts = int(num_rollouts)
        if num_rollouts <= 0:
            return (), None
        rollouts, training = self.engine.sample_rollouts(
            policy=policy,
            context=context,
            features=features,
            num_rollouts=num_rollouts,
        )
        return tuple(rollouts), training

    def replay_trajectories(
        self,
        *,
        batch: RetrievalBatch,
        rollouts: tuple[RolloutResult, ...],
        context: GraphContext,
        num_trajectories: int,
        reward_model: TrueTerminalReward | None = None,
        target_context: TargetContext | None = None,
        progress: float = 0.0,
    ) -> ReplayBatch | None:
        if int(num_trajectories) <= 0:
            return None
        if self.replay_source is None:
            raise RuntimeError("replay_source is required when replay_expand > 0.")
        return self.replay_source.sample_from_rollouts(
            batch=batch,
            context=context,
            rollouts=rollouts,
            num_trajectories=int(num_trajectories),
            reward_model=reward_model,
            target_context=target_context,
            progress=float(progress),
        )

    def training_batch(
        self,
        *,
        rollouts: tuple[RolloutResult, ...],
        policy_training: TrainingBatch | None,
        replay: ReplayBatch | None,
        context: GraphContext,
    ) -> TrainingBatch | None:
        parts: list[TrainingBatch] = []

        if policy_training is not None and policy_training.num_items > 0:
            parts.append(policy_training.with_source_id(SRC_POLICY))

        if replay is not None and replay.num_trajectories > 0:
            if self.replay_builder is None:
                raise RuntimeError("replay_builder is required when replay is enabled.")
            replay_training = self.replay_builder.build(
                graph=context,
                trajectories=replay,
            )
            if replay_training.num_items > 0:
                parts.append(replay_training.with_source_id(SRC_REPLAY))

        if not parts:
            return None
        return TrainingBatch.concat_reindex_trajectories(parts)

    def sample_budget(
        self,
        total: int,
        *,
        progress: float | None = None,
    ) -> ReplaySampleBudget:
        total = int(total)
        if self.replay_schedule is None:
            return ReplaySampleBudget(
                policy_rollout=total,
                replay_expand=0,
            )

        if progress is None:
            progress = float(self.progress_fn())
        weights = self.replay_schedule.weights_at(
            progress=float(progress),
        )
        return allocate_replay_budget(
            total=total,
            policy_weight=weights.policy_rollout,
            replay_weight=weights.replay_expand,
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

    policy_weight = float(policy_weight)
    replay_weight = float(replay_weight)
    weight_sum = policy_weight + replay_weight
    if weight_sum <= 0.0:
        raise ValueError("policy_weight and replay_weight cannot both be zero.")

    policy_raw = total * policy_weight / weight_sum
    replay_raw = total * replay_weight / weight_sum
    policy_count = int(policy_raw)
    replay_count = int(replay_raw)
    remainder = total - policy_count - replay_count
    if remainder > 0:
        if policy_raw - policy_count >= replay_raw - replay_count:
            policy_count += remainder
        else:
            replay_count += remainder
    return ReplaySampleBudget(
        policy_rollout=policy_count,
        replay_expand=replay_count,
    )


def zero_progress() -> float:
    return 0.0


__all__ = [
    "ReplaySchedule",
    "ReplayScheduleRow",
    "RolloutBatch",
    "RolloutRunner",
    "allocate_replay_budget",
]
