from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import NodeSelection, StateBatch

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class TerminalRewardOutput:
    log_reward: Tensor
    raw_log_reward: Tensor
    answer_count: Tensor
    candidate_count: Tensor
    target_count: Tensor
    target_recall: Tensor
    edge_count: Tensor
    valid_mask: Tensor
    success_mask: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)


class TerminalRewardModel(nn.Module):
    def __init__(
        self,
        *,
        eps: float = 1.0,
        answer_weight: float = 6.0,
        coverage_weight: float = 1.0,
        edge_cost: float = 0.15,
        fail_cost: float = 6.0,
        answer_prize: float = 2.0,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.answer_weight = float(answer_weight)
        self.coverage_weight = float(coverage_weight)
        self.edge_cost = float(edge_cost)
        self.fail_cost = float(fail_cost)
        self.answer_prize = float(answer_prize)

    @torch.no_grad()
    def forward(
        self,
        *,
        state: StateBatch,
        target_context: TargetContext,
        graph_context: GraphContext,
        active: NodeSelection | None = None,
    ) -> TerminalRewardOutput:
        active = state.active_node_index(graph_context) if active is None else active
        answer_count = torch.zeros(state.num_states, dtype=torch.float32, device=state.device)
        candidate_count = torch.zeros_like(answer_count)
        if int(active.node_ids.numel()) > 0:
            answer_count.scatter_add_(0, active.row_ids, target_context.target_mask.index_select(0, active.node_ids).float())
            candidate_count.scatter_add_(0, active.row_ids, torch.ones_like(active.row_ids, dtype=torch.float32))
        target_count = target_context.target_count_by_graph.index_select(0, state.graph_ids).float()
        valid = target_count.gt(0)
        success = answer_count.gt(0) & valid
        recall = answer_count / target_count.clamp_min(1.0)
        coverage = torch.log(answer_count * float(torch.exp(torch.tensor(self.answer_prize))) + self.eps)
        edge_count = state.edge_count.float()
        log_reward = (
            self.answer_weight * recall
            + self.coverage_weight * coverage
            - self.edge_cost * edge_count
            - self.fail_cost * (~success).float()
        )
        return TerminalRewardOutput(
            log_reward=log_reward,
            raw_log_reward=log_reward,
            answer_count=answer_count,
            candidate_count=candidate_count,
            target_count=target_count,
            target_recall=recall,
            edge_count=edge_count,
            valid_mask=valid,
            success_mask=success,
            metrics={
                "reward/log_reward_mean": _mean(log_reward),
                "reward/log_reward_std": _std(log_reward),
                "reward/log_reward_min": _min(log_reward),
                "reward/log_reward_max": _max(log_reward),
                "reward/answer_recall_mean": _masked_mean(recall, valid),
                "reward/edge_count_mean": _masked_mean(edge_count, valid),
                "reward/hit_rate": _mean(success.float()),
                "reward/valid_rate": _mean(valid.float()),
            },
        )


RewardOutput = TerminalRewardOutput
EvidenceSubgraphReward = TerminalRewardModel


def _mean(value: Tensor) -> Tensor:
    return value.mean().detach() if int(value.numel()) else value.new_zeros(())


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return value[mask].mean().detach() if bool(mask.any()) else value.new_zeros(())


def _std(value: Tensor) -> Tensor:
    return value.std(unbiased=False).detach() if int(value.numel()) else value.new_zeros(())


def _min(value: Tensor) -> Tensor:
    return value.min().detach() if int(value.numel()) else value.new_zeros(())


def _max(value: Tensor) -> Tensor:
    return value.max().detach() if int(value.numel()) else value.new_zeros(())


__all__ = [
    "EvidenceSubgraphReward",
    "RewardOutput",
    "TerminalRewardModel",
    "TerminalRewardOutput",
]
