from __future__ import annotations

from dataclasses import dataclass

import torch
from src.weaver.context import GraphContext
from src.weaver.feature import FeatureBank
from src.weaver.objectives.transition_batch import NonterminalTransitionBatch
from src.weaver.policy import ForwardPolicy
from src.weaver.rollout.replay import ReplaySource
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.context import ReplayContext, TargetContext
from src.weaver.state import ExpansionBatch, StateBatch, cat_state_batches


@dataclass(frozen=True, slots=True)
class TrainRolloutBatch:
    trajectories: TrajectoryBatch
    replay_transitions: NonterminalTransitionBatch | None
    metrics: dict[str, torch.Tensor]

    @property
    def num_trajectories(self) -> int:
        return int(self.trajectories.num_trajectories)

    @property
    def num_policy_trajectories(self) -> int:
        if int(self.trajectories.num_trajectories) == 0:
            return 0
        return int(self.trajectories.is_policy.sum().item())


class RolloutRunner:
    def __init__(
        self,
        *,
        engine: RolloutEngine,
        train_policy_rollouts: int,
        replay_source: ReplaySource | None = None,
        eval_rollouts: int = 1,
    ) -> None:
        self.engine = engine
        self.train_policy_rollouts = int(train_policy_rollouts)
        self.replay_source = replay_source
        self.eval_rollouts_count = int(eval_rollouts)

        if self.train_policy_rollouts < 0:
            raise ValueError("train_policy_rollouts must be non-negative.")
        if self.eval_rollouts_count < 0:
            raise ValueError("eval_rollouts must be non-negative.")

    @torch.no_grad()
    def train_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        target_context: TargetContext,
        replay_context: ReplayContext,
        features: FeatureBank,
        budget: int,
    ) -> TrainRolloutBatch:
        trajectories = self.policy_rollouts(
            policy=policy,
            context=context,
            features=features,
            num_rollouts=self.train_policy_rollouts,
            budget=budget,
        )

        replay_transitions = None
        replay_stats = None
        if self.replay_source is not None:
            replay_initial_state = _policy_prefix_states(
                trajectories=trajectories,
                graph_context=context,
            )
            replay = self.replay_source.collect(
                graph_context=context,
                target_context=target_context,
                replay_context=replay_context,
                initial_state=replay_initial_state,
            )
            replay_transitions = replay.nonterminal
            replay_stats = replay.stats

        if int(trajectories.num_trajectories) <= 0:
            raise RuntimeError("RolloutRunner produced zero training trajectories.")

        metrics = _train_rollout_metrics(
            trajectories=trajectories,
            device=context.device,
            replay_transitions=replay_transitions,
            replay_stats=replay_stats,
        )

        return TrainRolloutBatch(
            trajectories=trajectories,
            replay_transitions=replay_transitions,
            metrics=metrics,
        )

    @torch.no_grad()
    def eval_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: FeatureBank,
        budget: int,
        num_rollouts: int | None = None,
    ) -> TrajectoryBatch:
        count = self.eval_rollouts_count if num_rollouts is None else int(num_rollouts)
        return self.policy_rollouts(
            policy=policy,
            context=context,
            features=features,
            num_rollouts=count,
            budget=budget,
        )

    @torch.no_grad()
    def policy_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: FeatureBank,
        num_rollouts: int,
        budget: int,
    ) -> TrajectoryBatch:
        num_rollouts = int(num_rollouts)
        if num_rollouts == 0:
            return TrajectoryBatch.empty(
                device=context.device,
                budget=budget,
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
            budget=budget,
        )


def _policy_prefix_states(
    *,
    trajectories: TrajectoryBatch,
    graph_context: GraphContext,
) -> StateBatch:
    if int(trajectories.num_trajectories) == 0:
        return StateBatch.initial(
            graph_ids=torch.empty(0, dtype=torch.long, device=graph_context.device),
            budget=int(trajectories.budget),
            graph_context=graph_context,
        )
    budget = int(trajectories.budget)

    counts = trajectories.edge_count.to(dtype=torch.long)
    prefix_total = int(counts.sum().item())
    if prefix_total == 0:
        return StateBatch.initial(
            graph_ids=torch.empty(0, dtype=torch.long, device=graph_context.device),
            budget=budget,
            graph_context=graph_context,
        )
    row_ids = torch.repeat_interleave(
        torch.arange(int(trajectories.num_trajectories), device=trajectories.device, dtype=torch.long),
        counts,
        output_size=prefix_total,
    )
    prefix_lens = torch.cat(
        [
            torch.arange(int(count.item()), device=trajectories.device, dtype=torch.long)
            for count in counts
            if int(count.item()) > 0
        ],
        dim=0,
    )
    prefix_edge_ids = torch.full(
        (prefix_total, budget),
        -1,
        dtype=torch.long,
        device=trajectories.device,
    )
    for slot in range(budget):
        mask = prefix_lens.gt(slot)
        if not bool(mask.any()):
            continue
        prefix_edge_ids[mask, slot] = trajectories.edge_ids.index_select(0, row_ids[mask])[:, slot]
    return StateBatch.from_selected_edges(
        graph_ids=trajectories.graph_ids.index_select(0, row_ids),
        edge_ids=prefix_edge_ids,
        edge_count=prefix_lens,
        budget=budget,
        graph_context=graph_context,
    )

def _train_rollout_metrics(
    *,
    trajectories: TrajectoryBatch,
    device: torch.device,
    replay_transitions: NonterminalTransitionBatch | None,
    replay_stats,
) -> dict[str, torch.Tensor]:
    if int(trajectories.num_trajectories) == 0:
        num_policy = 0
    else:
        num_policy = int(trajectories.is_policy.sum().item())
    replay_count = 0 if replay_transitions is None else int(replay_transitions.num_transitions)
    return {
        "num_trajectories": torch.tensor(
            float(trajectories.num_trajectories),
            device=device,
        ),
        "num_policy_trajectories": torch.tensor(
            float(num_policy),
            device=device,
        ),
        "replay_nonterminal_count": torch.tensor(float(replay_count), device=device),
        "replay_prefix_count": torch.tensor(
            float(0 if replay_stats is None else replay_stats.prefix_count),
            device=device,
        ),
        "replay_positive_transition_count": torch.tensor(
            float(0 if replay_stats is None else replay_stats.positive_transition_count),
            device=device,
        ),
        "replay_prefix_with_positive_rate": torch.tensor(
            float(0.0 if replay_stats is None else replay_stats.prefix_with_positive_rate),
            device=device,
        ),
        "replay_mean_positive_edges_per_prefix": torch.tensor(
            float(0.0 if replay_stats is None else replay_stats.mean_positive_edges_per_prefix),
            device=device,
        ),
    }


def repeated_graph_ids(
    *,
    num_graphs: int,
    repeats: int,
    device: torch.device,
) -> torch.Tensor:
    num_graphs = int(num_graphs)
    repeats = int(repeats)
    if num_graphs < 0:
        raise ValueError("num_graphs must be non-negative.")
    if repeats < 0:
        raise ValueError("repeats must be non-negative.")
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
    non_empty = [batch for batch in batches if int(batch.num_trajectories) > 0]
    if not non_empty:
        return TrajectoryBatch.empty(
            device=device,
            budget=budget,
        )
    return TrajectoryBatch.concat(non_empty)


__all__ = [
    "TrainRolloutBatch",
    "RolloutRunner",
    "_policy_prefix_states",
    "concat_trajectory_batches",
    "repeated_graph_ids",
]
