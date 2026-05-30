from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.weaver.context import TargetContext
from src.weaver.state import StateBatch

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class RewardOutput:
    log_reward: Tensor  # [S]
    raw_log_reward: Tensor  # [S]
    answer_count: Tensor  # [S]
    candidate_count: Tensor  # [S]
    target_count: Tensor  # [S]
    answer_precision: Tensor  # [S]
    target_recall: Tensor  # [S]
    answer_f_score: Tensor  # [S]
    edge_count: Tensor  # [S]
    valid_mask: Tensor  # [S]
    success_mask: Tensor  # [S]
    metrics: dict[str, Tensor] = field(default_factory=dict)


class AnswerGroundedCoverageReward(nn.Module):
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

        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        for name, value in (
            ("answer_weight", answer_weight),
            ("coverage_weight", coverage_weight),
            ("edge_cost", edge_cost),
            ("fail_cost", fail_cost),
            ("answer_prize", answer_prize),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

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
    ) -> RewardOutput:
        target_mask = target_context.target_mask.to(
            device=state.device,
            dtype=torch.bool,
        )
        anchor_target_count_by_graph = _anchor_count_by_graph(
            cached=target_context.anchor_target_count_by_graph,
            fallback_mask=target_mask,
            state=state,
        )

        answer_count = (
            anchor_target_count_by_graph.index_select(0, state.graph_ids)
            + _sum_state_prefix_mask_hits(
                state_node_ids=state.activated_node_ids,
                state_node_count=state.activated_node_count,
                mask=target_mask,
            )
        )
        target_count = target_context.target_count_by_graph.index_select(
            0,
            state.graph_ids,
        ).to(dtype=torch.float32)
        candidate_count = state.anchor_count + state.activated_node_count

        valid_mask = target_count.gt(0)
        success_mask = answer_count.gt(0) & valid_mask

        answer_precision = answer_count.float() / candidate_count.clamp_min(1).float()
        target_recall = answer_count.float() / target_count.clamp_min(1.0)
        answer_f_score = torch.where(
            valid_mask,
            target_recall,
            torch.zeros_like(target_recall),
        )

        coverage_sum = answer_count.float() * float(torch.exp(torch.tensor(self.answer_prize)).item())
        coverage_utility = torch.log(
            coverage_sum + float(self.eps)
        )
        answer_term = self.answer_weight * target_recall
        structure_term = self.coverage_weight * coverage_utility
        edge_count = state.edge_count.float()
        edge_penalty = self.edge_cost * edge_count
        fail_penalty = self.fail_cost * (~success_mask).float()

        raw_log_reward = answer_term + structure_term - edge_penalty - fail_penalty
        log_reward = raw_log_reward

        return RewardOutput(
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
            answer_count=answer_count.float(),
            candidate_count=candidate_count.float(),
            target_count=target_count,
            answer_precision=answer_precision,
            target_recall=target_recall,
            answer_f_score=answer_f_score,
            edge_count=edge_count,
            valid_mask=valid_mask,
            success_mask=success_mask,
            metrics={
                "reward/log_reward_mean": _mean(log_reward),
                "reward/coverage_utility_mean": _masked_mean(coverage_utility, valid_mask),
                "reward/answer_recall_mean": _masked_mean(target_recall, valid_mask),
                "reward/edge_count_mean": _masked_mean(edge_count, valid_mask),
                "reward/hit_rate": _mean(success_mask.float()),
                "reward/valid_rate": _mean(valid_mask.float()),
            },
        )


EvidenceSubgraphReward = AnswerGroundedCoverageReward


def _mean(value: Tensor) -> Tensor:
    if int(value.numel()) == 0:
        return value.new_zeros(())
    return value.mean().detach()


def _masked_mean(
    value: Tensor,
    mask: Tensor,
) -> Tensor:
    if int(value.numel()) == 0:
        return value.new_zeros(())
    mask = mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        return value.new_zeros(())
    return value[mask].mean().detach()


__all__ = [
    "AnswerGroundedCoverageReward",
    "EvidenceSubgraphReward",
    "RewardOutput",
]


def _sum_state_prefix_mask_hits(
    *,
    state_node_ids: Tensor,
    state_node_count: Tensor,
    mask: Tensor,
) -> Tensor:
    if int(state_node_ids.numel()) == 0:
        return torch.zeros_like(state_node_count)
    budget = int(state_node_ids.size(1))
    valid = torch.arange(
        budget,
        device=state_node_ids.device,
        dtype=torch.long,
    ).view(1, -1).lt(state_node_count.view(-1, 1))
    if not bool(valid.any()):
        return torch.zeros_like(state_node_count)
    row_ids = valid.nonzero(as_tuple=True)[0]
    hits = mask.index_select(0, state_node_ids[valid]).to(dtype=torch.long)
    out = torch.zeros_like(state_node_count)
    out.scatter_add_(0, row_ids, hits)
    return out


def _anchor_count_by_graph(
    *,
    cached: Tensor,
    fallback_mask: Tensor,
    state: StateBatch,
) -> Tensor:
    if state.num_states == 0:
        return torch.empty(0, dtype=torch.long, device=state.device)
    graph_count = int(state.graph_ids.max().item()) + 1
    if int(cached.numel()) >= graph_count:
        return cached.to(device=state.device, dtype=torch.long)
    del fallback_mask
    return torch.zeros(graph_count, dtype=torch.long, device=state.device)


def _activated_prefix_mask(state: StateBatch) -> Tensor:
    width = int(state.activated_node_ids.max().item()) + 1 if int(state.activated_node_ids.numel()) > 0 and bool(state.activated_node_ids.ge(0).any()) else 0
    out = torch.zeros((state.num_states, width), dtype=torch.bool, device=state.device)
    if int(state.activated_node_ids.numel()) == 0:
        return out
    budget = int(state.activated_node_ids.size(1))
    valid = torch.arange(
        budget,
        device=state.device,
        dtype=torch.long,
    ).view(1, -1).lt(state.activated_node_count.view(-1, 1))
    if not bool(valid.any()):
        return out
    rows = valid.nonzero(as_tuple=True)[0]
    out[rows, state.activated_node_ids[valid]] = True
    return out
