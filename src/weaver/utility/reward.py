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

    `metrics` is optional in the sense that callers should only rely on
    `log_reward`; metrics are logging-facing diagnostics.
    """

    log_reward: torch.Tensor
    raw_log_reward: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


class TrueTerminalReward(nn.Module):
    """
    Label-dependent terminal reward:

        supported_count(z) = |S_V ∩ Y|
        recall(z)          = supported_count(z) / |Y|

        raw_log_R(z)
            = log(epsilon + answer_weight * recall(z))
              - edge_cost * |S_E|
              - fail_cost * 1[supported_count(z) = 0]

        log_R(z) = raw_log_R(z) / log_reward_scale
    """

    def __init__(
        self,
        *,
        epsilon: float = 1.0e-6,
        answer_weight: float = 1.0,
        edge_cost: float = 0.05,
        fail_cost: float = 1.0,
        log_reward_scale: float = 5.0,
    ) -> None:
        super().__init__()

        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if answer_weight <= 0.0:
            raise ValueError(f"answer_weight must be positive, got {answer_weight}.")
        if edge_cost < 0.0:
            raise ValueError(f"edge_cost must be non-negative, got {edge_cost}.")
        if fail_cost < 0.0:
            raise ValueError(f"fail_cost must be non-negative, got {fail_cost}.")
        if log_reward_scale <= 0.0:
            raise ValueError(
                f"log_reward_scale must be positive, got {log_reward_scale}."
            )

        self.epsilon = float(epsilon)
        self.answer_weight = float(answer_weight)
        self.edge_cost = float(edge_cost)
        self.fail_cost = float(fail_cost)
        self.log_reward_scale = float(log_reward_scale)

    @torch.no_grad()
    def forward(
        self,
        *,
        state: State,
        graph_context: GraphContext,
        target_context: TargetContext,
    ) -> RewardOutput:
        del graph_context
        target_count = target_context.target_count_by_graph.index_select(
            0,
            state.row_to_graph,
        )
        supported_count = (
            state.active_node_mask & target_context.target_mask.view(1, -1)
        ).sum(dim=1)
        answer_recall = supported_count / target_count.clamp_min(1)

        selected_edge_count = state.edge_mask.sum(dim=1)
        answer_term = self.answer_weight * answer_recall
        edge_penalty = self.edge_cost * selected_edge_count
        fail_penalty = self.fail_cost * supported_count.eq(0)

        raw_log_reward = (
            torch.log(self.epsilon + answer_term) - edge_penalty - fail_penalty
        )
        log_reward = raw_log_reward / self.log_reward_scale

        return RewardOutput(
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
            metrics={
                "reward/true/recall_mean": answer_recall.float().mean().detach(),
                "reward/true/supported_count_mean": supported_count.float().mean().detach(),
                "reward/true/target_count_mean": target_count.float().mean().detach(),
                "reward/true/edge_penalty_mean": edge_penalty.float().mean().detach(),
                "reward/true/fail_rate": supported_count.eq(0).float().mean().detach(),
            },
        )


__all__ = [
    "RewardOutput",
    "TrueTerminalReward",
]
