from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.policy import ForwardPolicy
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.rollout.replay import ReplaySource
from src.weaver.rollout.trajectory import SRC_POLICY, SRC_REPLAY, TrajectoryBatch


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """
    Output of train rollout generation.

    trajectories:
    - policy and replay trajectories concatenated together.
    """

    trajectories: TrajectoryBatch

    @property
    def num_trajectories(self) -> int:
        return int(self.trajectories.num_trajectories)

    @property
    def num_policy_trajectories(self) -> int:
        if self.trajectories.num_trajectories == 0:
            return 0
        return int(self.trajectories.source.eq(int(SRC_POLICY)).sum().item())

    @property
    def num_replay_trajectories(self) -> int:
        if self.trajectories.num_trajectories == 0:
            return 0
        return int(self.trajectories.source.eq(int(SRC_REPLAY)).sum().item())


class RolloutRunner:
    """
    Thin rollout orchestrator.

    Responsibilities:
    - sample policy trajectories;
    - optionally attach precomputed replay trajectories;

    Non-responsibilities:
    - no replay reward gating;
    - no replay stats;
    - no dynamic replay schedule;
    - no progress_fn;
    - no loss construction;
    - no metric aggregation;
    - no State / StateOps manipulation.
    """

    def __init__(
        self,
        *,
        engine: RolloutEngine,
        train_policy_rollouts: int,
        train_replay_rollouts: int = 0,
        eval_rollouts: int = 1,
        replay_source: ReplaySource | None = None,
    ) -> None:
        self.engine = engine
        self.replay_source = replay_source

        self.train_policy_rollouts = int(train_policy_rollouts)
        self.train_replay_rollouts = int(train_replay_rollouts)
        self.eval_rollouts_count = int(eval_rollouts)

        if self.train_policy_rollouts < 0:
            raise ValueError("train_policy_rollouts must be nonnegative.")
        if self.train_replay_rollouts < 0:
            raise ValueError("train_replay_rollouts must be nonnegative.")
        if self.eval_rollouts_count < 0:
            raise ValueError("eval_rollouts must be nonnegative.")
        if self.train_replay_rollouts > 0 and replay_source is None:
            raise ValueError("replay_source is required when train_replay_rollouts > 0.")

        self._validate_budget_consistency()

    @property
    def budget(self) -> int:
        return int(self.engine.budget)

    def _validate_budget_consistency(self) -> None:
        if self.replay_source is None:
            return

        replay_budget = int(self.replay_source.budget)
        if replay_budget != self.budget:
            raise ValueError("replay_source budget must match rollout engine budget, got " f"{replay_budget} and {self.budget}.")

    @torch.no_grad()
    def train_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        batch: RetrievalBatch,
        context: GraphContext,
        features: FeatureBank,
        target_context: TargetContext,
    ) -> RolloutBatch:
        if int(policy.budget) != self.budget:
            raise ValueError("policy budget must match rollout engine budget, got " f"{int(policy.budget)} and {self.budget}.")

        policy_trajectories = self.policy_rollouts(
            policy=policy,
            context=context,
            features=features,
            num_rollouts=self.train_policy_rollouts,
        )

        replay_trajectories = self.replay_rollouts(
            batch=batch,
            context=context,
            target_context=target_context,
        )

        trajectories = concat_trajectory_batches(
            [
                policy_trajectories,
                replay_trajectories,
            ],
            device=context.device,
            budget=self.budget,
        )

        if trajectories.num_trajectories <= 0:
            raise RuntimeError("RolloutRunner produced zero training trajectories.")

        return RolloutBatch(
            trajectories=trajectories,
        )

    @torch.no_grad()
    def eval_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: FeatureBank,
        num_rollouts: int | None = None,
    ) -> TrajectoryBatch:
        count = self.eval_rollouts_count if num_rollouts is None else int(num_rollouts)
        if int(policy.budget) != self.budget:
            raise ValueError("policy budget must match rollout engine budget, got " f"{int(policy.budget)} and {self.budget}.")

        return self.policy_rollouts(
            policy=policy,
            context=context,
            features=features,
            num_rollouts=count,
        )

    @torch.no_grad()
    def policy_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: FeatureBank,
        num_rollouts: int,
    ) -> TrajectoryBatch:
        num_rollouts = int(num_rollouts)

        if num_rollouts <= 0:
            return TrajectoryBatch.empty(
                device=context.device,
                budget=self.budget,
            )

        graph_ids = repeated_graph_ids(
            num_graphs=int(context.num_graphs),
            repeats=num_rollouts,
            device=context.device,
        )

        return self.engine.sample(
            policy=policy,
            context=context,
            features=features,
            graph_ids=graph_ids,
        )

    @torch.no_grad()
    def replay_rollouts(
        self,
        *,
        batch: RetrievalBatch,
        context: GraphContext,
        target_context: TargetContext,
    ) -> TrajectoryBatch:
        if self.train_replay_rollouts <= 0:
            return TrajectoryBatch.empty(
                device=context.device,
                budget=self.budget,
            )

        if self.replay_source is None:
            raise RuntimeError("replay_source is required when train_replay_rollouts > 0.")

        return self.replay_source.sample(batch=batch, graph=context, target=target_context, trajectories_per_graph=self.train_replay_rollouts)


def repeated_graph_ids(
    *,
    num_graphs: int,
    repeats: int,
    device: torch.device,
) -> torch.Tensor:
    num_graphs = int(num_graphs)
    repeats = int(repeats)

    if num_graphs < 0:
        raise ValueError("num_graphs must be nonnegative.")
    if repeats < 0:
        raise ValueError("repeats must be nonnegative.")

    if num_graphs == 0 or repeats == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=device,
        )

    return torch.arange(
        num_graphs,
        dtype=torch.long,
        device=device,
    ).repeat_interleave(repeats)


def concat_trajectory_batches(
    batches: list[TrajectoryBatch],
    *,
    device: torch.device,
    budget: int,
) -> TrajectoryBatch:
    non_empty = [batch for batch in batches if batch.num_trajectories > 0]

    if not non_empty:
        return TrajectoryBatch.empty(
            device=device,
            budget=int(budget),
        )

    return TrajectoryBatch.concat(non_empty)


__all__ = [
    "RolloutBatch",
    "RolloutRunner",
    "concat_trajectory_batches",
    "repeated_graph_ids",
]
