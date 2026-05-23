from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import State


@dataclass(frozen=True, slots=True)
class RewardOutput:
    """
    Terminal reward values for a batch of states.

    Only `log_reward` is optimization-facing.
    `raw_log_reward` and `metrics` are diagnostics.
    """

    log_reward: torch.Tensor
    raw_log_reward: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


class EvidenceUtilityReward(nn.Module):
    """
    Terminal evidence utility reward.

    Shortest-path edge labels are support priors, not complete gold evidence.
    """

    def __init__(
        self,
        *,
        answer_weight: float = 1.0,
        support_weight: float = 0.25,
        connect_weight: float = 0.5,
        edge_cost: float = 0.05,
        unsupported_edge_cost: float = 0.05,
        fail_cost: float = 1.0,
        reward_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if answer_weight <= 0.0:
            raise ValueError(f"answer_weight must be positive, got {answer_weight}.")
        for name, value in {
            "support_weight": support_weight,
            "connect_weight": connect_weight,
            "edge_cost": edge_cost,
            "unsupported_edge_cost": unsupported_edge_cost,
            "fail_cost": fail_cost,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if reward_temperature <= 0.0:
            raise ValueError(f"reward_temperature must be positive, got {reward_temperature}.")

        self.answer_weight = float(answer_weight)
        self.support_weight = float(support_weight)
        self.connect_weight = float(connect_weight)
        self.edge_cost = float(edge_cost)
        self.unsupported_edge_cost = float(unsupported_edge_cost)
        self.fail_cost = float(fail_cost)
        self.reward_temperature = float(reward_temperature)

    @torch.no_grad()
    def forward(
        self,
        *,
        state: State,
        graph_context: GraphContext,
        target_context: TargetContext,
    ) -> RewardOutput:
        answer_count = (state.active_node_mask & target_context.target_mask.view(1, -1)).sum(dim=1)
        selected_edge_mask = state.edge_mask
        selected_edge_count = selected_edge_mask.sum(dim=1).float()
        failed = answer_count.eq(0)

        support_edge_mask = shortest_path_support_union(
            graph_context=graph_context,
            target_context=target_context,
        )
        row_support_mask = support_edge_mask.index_select(0, state.row_to_graph)
        support_total = support_edge_mask.sum(dim=1).index_select(0, state.row_to_graph).float()
        selected_support_count = (selected_edge_mask & row_support_mask).sum(dim=1).float()
        support_coverage = selected_support_count / support_total.clamp_min(1.0)
        unsupported_edge_count = (selected_edge_mask & ~row_support_mask).sum(dim=1).float()

        connected_answer = answer_count.gt(0) & (
            selected_support_count.gt(0) | selected_edge_count.eq(0)
        )

        answer_gain = self.answer_weight * torch.log1p(answer_count.float())
        support_gain = self.support_weight * support_coverage
        connect_gain = self.connect_weight * connected_answer.float()
        edge_penalty = self.edge_cost * selected_edge_count
        unsupported_penalty = self.unsupported_edge_cost * unsupported_edge_count
        fail_penalty = self.fail_cost * failed.float()

        raw_log_reward = (
            answer_gain
            + support_gain
            + connect_gain
            - edge_penalty
            - unsupported_penalty
            - fail_penalty
        )
        log_reward = raw_log_reward / self.reward_temperature

        target_count = target_context.target_count_by_graph.index_select(0, state.row_to_graph)
        answer_recall = answer_count.float() / target_count.clamp_min(1).float()

        return RewardOutput(
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
            metrics={
                "reward_answer_gain_mean": answer_gain.mean().detach(),
                "reward_answer_count_mean": answer_count.float().mean().detach(),
                "reward_answer_recall_mean": answer_recall.mean().detach(),
                "reward_target_count_mean": target_count.float().mean().detach(),
                "reward_support_gain_mean": support_gain.mean().detach(),
                "reward_support_coverage_mean": support_coverage.mean().detach(),
                "reward_connect_gain_mean": connect_gain.mean().detach(),
                "reward_connected_answer_rate": connected_answer.float().mean().detach(),
                "reward_selected_edge_count_mean": selected_edge_count.mean().detach(),
                "reward_edge_penalty_mean": edge_penalty.mean().detach(),
                "reward_unsupported_edge_count_mean": unsupported_edge_count.mean().detach(),
                "reward_unsupported_penalty_mean": unsupported_penalty.mean().detach(),
                "reward_fail_rate": failed.float().mean().detach(),
                "reward_raw_log_reward_mean": raw_log_reward.mean().detach(),
                "reward_log_reward_mean": log_reward.mean().detach(),
            },
        )


class TrueTerminalReward(EvidenceUtilityReward):
    """
    Backward-compatible compact answer-reaching reward.
    """

    def __init__(
        self,
        *,
        answer_weight: float = 1.0,
        edge_cost: float = 0.05,
        fail_cost: float = 1.0,
        reward_temperature: float = 1.0,
    ) -> None:
        super().__init__(
            answer_weight=answer_weight,
            support_weight=0.0,
            connect_weight=0.0,
            edge_cost=edge_cost,
            unsupported_edge_cost=0.0,
            fail_cost=fail_cost,
            reward_temperature=reward_temperature,
        )


def shortest_path_support_union(
    *,
    graph_context: GraphContext,
    target_context: TargetContext,
) -> torch.Tensor:
    device = graph_context.device
    out = torch.zeros(
        (int(graph_context.num_graphs), int(graph_context.num_edges)),
        dtype=torch.bool,
        device=device,
    )
    flat = target_context.node_target_shortest_path_edge_mask_flat.to(device=device, dtype=torch.bool)
    if flat.numel() == 0:
        return out

    edge_counts = torch.bincount(
        graph_context.edge_to_graph.to(device=device, dtype=torch.long),
        minlength=int(graph_context.num_graphs),
    )
    edge_ptr = torch.empty(int(graph_context.num_graphs) + 1, dtype=torch.long, device=device)
    edge_ptr[0] = 0
    edge_ptr[1:] = torch.cumsum(edge_counts, dim=0)
    target_ptr = target_context.reachable_target_node_ids_ptr.to(device=device, dtype=torch.long)

    offset = 0
    for graph_id in range(int(graph_context.num_graphs)):
        num_targets = int((target_ptr[graph_id + 1] - target_ptr[graph_id]).item())
        num_edges = int((edge_ptr[graph_id + 1] - edge_ptr[graph_id]).item())
        block_size = num_targets * num_edges
        if block_size <= 0:
            continue
        block = flat[offset : offset + block_size].view(num_targets, num_edges)
        edge_start = int(edge_ptr[graph_id].item())
        edge_end = int(edge_ptr[graph_id + 1].item())
        out[graph_id, edge_start:edge_end] = block.any(dim=0)
        offset += block_size
    return out


__all__ = [
    "EvidenceUtilityReward",
    "RewardOutput",
    "shortest_path_support_union",
    "TrueTerminalReward",
]
