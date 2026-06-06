from __future__ import annotations

from dataclasses import dataclass

import torch
from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack
from src.weaver.policy import ForwardPolicy, PolicyInput
from src.weaver.rollout.replay import ReplaySource
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.rollout.trajectory import TrajectoryBatch
from src.weaver.context import ReplayContext, TargetContext
from src.weaver.state import StateBatch


@dataclass(frozen=True, slots=True)
class TrainRolloutBatch:
    trajectories: TrajectoryBatch
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
        features: FeaturePack,
        policy_input: PolicyInput,
        budget: int,
        global_step: int = 0,
        replay_round: int = 0,
    ) -> TrainRolloutBatch:
        trajectories = self.policy_rollouts(
            policy=policy,
            context=context,
            features=features,
            policy_input=policy_input,
            num_rollouts=self.train_policy_rollouts,
            budget=budget,
        )

        replay_trajectories = TrajectoryBatch.empty(device=context.device, budget=budget)
        replay_fraction = torch.tensor(0.0, device=context.device)
        replay_raw_count = 0
        replay_priority_mean = torch.tensor(0.0, device=context.device)
        if self.replay_source is not None:
            replay_fraction = torch.tensor(
                float(self.replay_source.current_fraction()),
                dtype=torch.float32,
                device=context.device,
            )
            replay_raw_count = self.replay_source.raw_trajectory_count(
                replay_context=replay_context,
                budget=budget,
                replay_round=int(replay_round),
            )
            replay_trajectories = self.replay_source.sample_trajectories(
                graph_context=context,
                target_context=target_context,
                replay_context=replay_context,
                budget=budget,
                replay_round=int(replay_round),
            )
            valid_priority = replay_context.priority[replay_context.edge_count.ge(0) & replay_context.edge_count.le(budget)]
            if int(valid_priority.numel()) > 0:
                replay_priority_mean = valid_priority.float().mean()
        all_trajectories = concat_trajectory_batches(
            [trajectories, replay_trajectories],
            device=context.device,
            budget=budget,
        )

        if int(all_trajectories.num_trajectories) <= 0:
            raise RuntimeError("RolloutRunner produced zero training trajectories.")

        metrics = _train_rollout_metrics(
            trajectories=all_trajectories,
            device=context.device,
            context=context,
            target_context=target_context,
            replay_fraction=replay_fraction,
            replay_raw_count=replay_raw_count,
            replay_priority_mean=replay_priority_mean,
        )

        return TrainRolloutBatch(
            trajectories=all_trajectories,
            metrics=metrics,
        )

    @torch.no_grad()
    def eval_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: FeaturePack,
        policy_input: PolicyInput,
        budget: int,
        num_rollouts: int | None = None,
        diversity_edge_penalty: float = 0.0,
    ) -> TrajectoryBatch:
        count = self.eval_rollouts_count if num_rollouts is None else int(num_rollouts)
        if float(diversity_edge_penalty) > 0.0:
            return self.diverse_policy_rollouts(
                policy=policy,
                context=context,
                features=features,
                policy_input=policy_input,
                num_rollouts=count,
                budget=budget,
                edge_penalty=float(diversity_edge_penalty),
            )
        return self.policy_rollouts(
            policy=policy,
            context=context,
            features=features,
            policy_input=policy_input,
            num_rollouts=count,
            budget=budget,
        )

    @torch.no_grad()
    def diverse_policy_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: FeaturePack,
        policy_input: PolicyInput,
        num_rollouts: int,
        budget: int,
        edge_penalty: float,
    ) -> TrajectoryBatch:
        num_rollouts = int(num_rollouts)
        if num_rollouts == 0:
            return TrajectoryBatch.empty(
                device=context.device,
                budget=budget,
            )
        edge_use_count = torch.zeros(int(context.num_edges), dtype=torch.float32, device=context.device)
        batches: list[TrajectoryBatch] = []
        graph_ids = repeated_graph_ids(
            num_graphs=int(context.num_graphs),
            repeats=1,
            device=context.device,
        )
        for _ in range(num_rollouts):
            trajectories = self.engine.sample(
                policy=policy,
                context=context,
                features=features,
                policy_input=policy_input,
                graph_ids=graph_ids,
                budget=budget,
                edge_logit_bias=-float(edge_penalty) * edge_use_count,
            )
            batches.append(trajectories)
            selected = trajectories.edge_ids[trajectories.valid_edge_mask()]
            if int(selected.numel()) > 0:
                edge_use_count.scatter_add_(
                    0,
                    selected,
                    torch.ones_like(selected, dtype=torch.float32),
                )
        combined = concat_trajectory_batches(
            batches,
            device=context.device,
            budget=budget,
        )
        if int(combined.num_trajectories) == 0:
            return combined
        return combined.select_rows(torch.argsort(combined.graph_ids, stable=True))

    @torch.no_grad()
    def policy_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: FeaturePack,
        policy_input: PolicyInput,
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
            policy_input=policy_input,
            graph_ids=graph_ids,
            budget=budget,
        )


def _train_rollout_metrics(
    *,
    trajectories: TrajectoryBatch,
    device: torch.device,
    context: GraphContext,
    target_context: TargetContext,
    replay_fraction: torch.Tensor,
    replay_raw_count: int,
    replay_priority_mean: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if int(trajectories.num_trajectories) == 0:
        num_policy = 0
    else:
        num_policy = int(trajectories.is_policy.sum().item())
    num_replay = int(trajectories.is_replay.sum().item()) if int(trajectories.num_trajectories) > 0 else 0
    replay_edge_counts = trajectories.edge_count[trajectories.is_replay] if num_replay > 0 else torch.empty(0, dtype=torch.long, device=device)
    replay_rows = trajectories.is_replay.nonzero(as_tuple=False).flatten()
    replay_recall = _replay_recall(
        trajectories=trajectories.select_rows(replay_rows),
        context=context,
        target_context=target_context,
    )
    replay_graph_count = int(torch.unique(trajectories.graph_ids.index_select(0, replay_rows)).numel()) if num_replay > 0 else 0
    target_diversity, anchor_diversity = _replay_diversity(
        trajectories=trajectories.select_rows(replay_rows),
        context=context,
        target_context=target_context,
    )
    replay_trajectories = trajectories.select_rows(replay_rows)
    replay_sets = torch.unique(
        torch.cat(
            [
                replay_trajectories.graph_ids.view(-1, 1),
                replay_trajectories.edge_count.view(-1, 1),
                replay_trajectories.edge_ids,
            ],
            dim=1,
        ),
        dim=0,
    )
    policy_mask = trajectories.is_policy
    budget_truncated = (trajectories.is_budget_truncated & policy_mask).sum().item() if int(trajectories.num_trajectories) > 0 else 0
    model_stop = (trajectories.is_policy_stop & policy_mask).sum().item() if int(trajectories.num_trajectories) > 0 else 0
    structural_stop = (trajectories.is_no_frontier & policy_mask).sum().item() if int(trajectories.num_trajectories) > 0 else 0
    denom = float(max(num_policy, 1))
    return {
        "num_trajectories": torch.tensor(
            float(trajectories.num_trajectories),
            device=device,
        ),
        "num_policy_trajectories": torch.tensor(
            float(num_policy),
            device=device,
        ),
        "num_replay_trajectories": torch.tensor(float(num_replay), device=device),
        "replay_fraction": replay_fraction,
        "replay_raw_trajectory_count": torch.tensor(float(replay_raw_count), device=device),
        "replay_kept_trajectory_count": torch.tensor(float(num_replay), device=device),
        "replay/trajectory_count": torch.tensor(float(num_replay), device=device),
        "replay/priority_mean": replay_priority_mean,
        "replay/edge_count_mean": replay_edge_counts.float().mean() if int(replay_edge_counts.numel()) > 0 else torch.tensor(0.0, device=device),
        "replay/coverage_recall_mean": replay_recall,
        "replay/unique_edge_set_rate": torch.tensor(float(replay_sets.size(0)) / float(max(num_replay, 1)), device=device),
        "replay/target_diversity_rate": target_diversity,
        "replay/anchor_pair_diversity_rate": anchor_diversity,
        "replay/no_feasible_path_rate": torch.tensor(
            1.0 - float(replay_graph_count) / float(max(int(context.num_graphs), 1)),
            device=device,
        ),
        "terminal/budget_truncated_rate": torch.tensor(float(budget_truncated) / denom, device=device),
        "terminal/model_stop_rate": torch.tensor(float(model_stop) / denom, device=device),
        "terminal/structural_stop_rate": torch.tensor(float(structural_stop) / denom, device=device),
        "terminal/budget_truncated_count": torch.tensor(float(budget_truncated), device=device),
    }


def _replay_recall(*, trajectories: TrajectoryBatch, context: GraphContext, target_context: TargetContext) -> torch.Tensor:
    if int(trajectories.num_trajectories) == 0:
        return torch.tensor(0.0, device=context.device)
    state = _replay_states(trajectories=trajectories, context=context)
    active = state.active_node_index(context)
    hit = target_context.target_mask.index_select(0, active.node_ids)
    hit_count = torch.zeros(state.num_states, dtype=torch.long, device=context.device)
    hit_count.scatter_add_(0, active.row_ids, hit.long())
    target_count = target_context.target_count_by_graph.index_select(0, state.graph_ids).clamp_min(1)
    return (hit_count.float() / target_count.float()).mean()


def _replay_diversity(*, trajectories: TrajectoryBatch, context: GraphContext, target_context: TargetContext) -> tuple[torch.Tensor, torch.Tensor]:
    if int(trajectories.num_trajectories) == 0:
        zero = torch.tensor(0.0, device=context.device)
        return zero, zero
    graph_ids = torch.unique(trajectories.graph_ids)
    state = _replay_states(trajectories=trajectories, context=context)
    active = state.active_node_index(context)
    active_target = target_context.target_mask.index_select(0, active.node_ids)
    target_nodes = active.node_ids[active_target]
    target_graphs = state.graph_ids.index_select(0, active.row_ids[active_target])
    covered_target_count = _unique_node_count_by_graph(
        graph_ids=target_graphs,
        node_ids=target_nodes,
        num_graphs=context.num_graphs,
        num_nodes=context.num_nodes,
    )

    selected = state.selected_edge_index()
    src = context.edge_src.index_select(0, selected.edge_ids)
    anchor_src = context.anchor_mask.index_select(0, src)
    used_anchor_count = _unique_node_count_by_graph(
        graph_ids=state.graph_ids.index_select(0, selected.row_ids[anchor_src]),
        node_ids=src[anchor_src],
        num_graphs=context.num_graphs,
        num_nodes=context.num_nodes,
    )
    target_count = target_context.target_count_by_graph.index_select(0, graph_ids).clamp_min(1)
    anchor_count = (context.anchor_ptr[1:] - context.anchor_ptr[:-1]).index_select(0, graph_ids).clamp_min(1)
    return (
        (covered_target_count.index_select(0, graph_ids).float() / target_count.float()).mean(),
        (used_anchor_count.index_select(0, graph_ids).float() / anchor_count.float()).mean(),
    )


def _replay_states(*, trajectories: TrajectoryBatch, context: GraphContext) -> StateBatch:
    return StateBatch.from_selected_edges(
        graph_ids=trajectories.graph_ids,
        edge_ids=trajectories.edge_ids,
        edge_count=trajectories.edge_count,
        budget=trajectories.budget,
        graph_context=context,
    )


def _unique_node_count_by_graph(*, graph_ids: torch.Tensor, node_ids: torch.Tensor, num_graphs: int, num_nodes: int) -> torch.Tensor:
    keys = torch.unique(graph_ids * max(int(num_nodes), 1) + node_ids)
    return torch.bincount(torch.div(keys, max(int(num_nodes), 1), rounding_mode="floor"), minlength=int(num_graphs))


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
    "concat_trajectory_batches",
    "repeated_graph_ids",
]
