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


class TrueTerminalReward(nn.Module):
    """
    Terminal graph reward.

    The generated object is a graph state:

        z = (V_z, E_z)

    where:

        V_z = active nodes in the generated graph
        E_z = selected edges in the generated graph

    The utility of a graph is the number of answer targets contained in V_z:

        answer_count(z) = |V_z ∩ Y|

    Reward:

        raw_log_R(z)
            = answer_weight * log(1 + answer_count(z))
              - edge_cost * |E_z|
              - fail_cost * 1[answer_count(z) = 0]

        log_R(z) = raw_log_R(z) / reward_temperature

    This reward does not use shortest-path edge labels.
    Shortest-path labels may still be used for replay, diagnostics, or auxiliary
    evaluation, but they are not part of the terminal reward.
    """

    def __init__(
        self,
        *,
        answer_weight: float = 1.0,
        edge_cost: float = 0.05,
        fail_cost: float = 1.0,
        reward_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if answer_weight <= 0.0:
            raise ValueError(f"answer_weight must be positive, got {answer_weight}.")
        if edge_cost < 0.0:
            raise ValueError(f"edge_cost must be non-negative, got {edge_cost}.")
        if fail_cost < 0.0:
            raise ValueError(f"fail_cost must be non-negative, got {fail_cost}.")
        if reward_temperature <= 0.0:
            raise ValueError(f"reward_temperature must be positive, got {reward_temperature}.")

        self.answer_weight = float(answer_weight)
        self.edge_cost = float(edge_cost)
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
        del graph_context

        answer_count = (state.active_node_mask & target_context.target_mask.view(1, -1)).sum(dim=1)

        selected_edge_count = state.edge_mask.sum(dim=1).float()
        failed = answer_count.eq(0)

        answer_gain = self.answer_weight * torch.log1p(answer_count.float())
        edge_penalty = self.edge_cost * selected_edge_count
        fail_penalty = self.fail_cost * failed.float()

        raw_log_reward = answer_gain - edge_penalty - fail_penalty
        log_reward = raw_log_reward / self.reward_temperature

        target_count = target_context.target_count_by_graph.index_select(
            0,
            state.row_to_graph,
        )
        answer_recall = answer_count.float() / target_count.clamp_min(1).float()

        return RewardOutput(
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
            metrics={
                "reward_true_answer_gain_mean": answer_gain.mean().detach(),
                "reward_true_answer_count_mean": answer_count.float().mean().detach(),
                "reward_true_answer_recall_mean": answer_recall.mean().detach(),
                "reward_true_target_count_mean": target_count.float().mean().detach(),
                "reward_true_selected_edge_count_mean": selected_edge_count.mean().detach(),
                "reward_true_edge_penalty_mean": edge_penalty.mean().detach(),
                "reward_true_fail_rate": failed.float().mean().detach(),
                "reward_true_raw_log_reward_mean": raw_log_reward.mean().detach(),
                "reward_true_log_reward_mean": log_reward.mean().detach(),
            },
        )


__all__ = [
    "RewardOutput",
    "TrueTerminalReward",
]
