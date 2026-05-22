from __future__ import annotations

import torch

from src.weaver.context import GraphContext, TargetContext
from src.weaver.policy import (
    ForwardPolicyOutput,
)
from src.weaver.state import Frontier, State
from src.weaver.transition import SRC_POLICY
from src.weaver.utility import RewardOutput, TrueTerminalReward

def stop_expand_margin(policy_out: ForwardPolicyOutput) -> torch.Tensor:
    return policy_out.stop_vs_continue_log_ratio.float()


def policy_stop_expand_margin_mean(policy_out: ForwardPolicyOutput) -> torch.Tensor:
    margin = stop_expand_margin(policy_out)
    return masked_mean_or_zero(margin, torch.isfinite(margin))


def policy_stop_expand_margin_by_mask(
    policy_out: ForwardPolicyOutput,
    mask: torch.Tensor,
) -> torch.Tensor:
    margin = stop_expand_margin(policy_out)
    mask = mask.to(device=margin.device, dtype=torch.bool).view(-1) & torch.isfinite(margin)
    return masked_mean_or_zero(margin, mask)


def reward_delta_after_hit(
    *,
    reward_model: TrueTerminalReward,
    child: State,
    graph_context: GraphContext,
    target_context: TargetContext | None,
    features,
    parent_log_reward: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    rows = mask.nonzero(as_tuple=False).flatten()
    if rows.numel() == 0 or target_context is None:
        return parent_log_reward.new_empty((0,))

    child_reward = call_reward_model(
        reward_model=reward_model,
        state=child.select_rows(rows),
        graph_context=graph_context,
        target_context=target_context,
        features=features,
    )
    return child_reward.log_reward.float() - parent_log_reward.index_select(0, rows)


def call_reward_model(
    *,
    reward_model: TrueTerminalReward,
    state: State,
    graph_context: GraphContext,
    target_context: TargetContext | None,
    features,
) -> RewardOutput:
    if isinstance(reward_model, TrueTerminalReward):
        if target_context is None:
            raise ValueError("TrueTerminalReward requires target_context.")
        return reward_model(
            state=state,
            graph_context=graph_context,
            target_context=target_context,
        )

    raise TypeError(
        "reward_model must be TrueTerminalReward, "
        f"got {type(reward_model).__name__}."
    )


def supported_target_count(
    *,
    state: State,
    graph_context: GraphContext,
    target_context: TargetContext,
) -> torch.Tensor:
    return (
        state.active_node_mask & target_context.target_mask.view(1, -1)
    ).sum(dim=1)


def stop_probability_by_depth(
    *,
    stop_log_prob: torch.Tensor,
    step_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    if step_ids.numel() == 0:
        return metrics

    step_ids = step_ids.to(device=stop_log_prob.device, dtype=torch.long).view(-1)
    stop_probability = stop_log_prob.float().exp()
    for depth in torch.unique(step_ids, sorted=True).tolist():
        mask = step_ids.eq(int(depth))
        metrics[f"policy_stop_prob_depth{int(depth)}"] = masked_mean_or_zero(
            stop_probability,
            mask,
        ).detach()
    return metrics


def stop_expand_margin_by_depth(
    *,
    policy_out: ForwardPolicyOutput,
    step_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    if step_ids.numel() == 0:
        return metrics

    margin = stop_expand_margin(policy_out)
    step_ids = step_ids.to(device=margin.device, dtype=torch.long).view(-1)
    finite = torch.isfinite(margin)
    for depth in torch.unique(step_ids, sorted=True).tolist():
        mask = step_ids.eq(int(depth)) & finite
        metrics[f"policy_stop_expand_margin_depth{int(depth)}"] = masked_mean_or_zero(
            margin,
            mask,
        ).detach()
    return metrics


def rollout_metrics(
    *,
    source_ids: torch.Tensor,
    action_edge_ids: torch.Tensor,
    step_ids: torch.Tensor,
    frontier: Frontier,
    num_rows: int,
) -> dict[str, torch.Tensor]:
    policy = source_ids.eq(SRC_POLICY)
    stop = policy & action_edge_ids.lt(0)
    structural = stop & ~rows_with_frontier(frontier=frontier, num_rows=num_rows)

    return {
        "rollout_stop_rate": masked_fraction(stop, policy).detach(),
        "rollout_structural_stop_rate": masked_fraction(structural, stop).detach(),
        "rollout_mean_stop_depth": masked_mean_or_zero(step_ids.float(), stop).detach(),
    }


def source_rollout_metrics(
    *,
    source_ids: torch.Tensor,
    action_edge_ids: torch.Tensor,
    step_ids: torch.Tensor,
    parent_frontier: Frontier,
    num_rows: int,
) -> dict[str, torch.Tensor]:
    metrics = rollout_metrics(
        source_ids=source_ids,
        action_edge_ids=action_edge_ids,
        step_ids=step_ids,
        frontier=parent_frontier,
        num_rows=int(num_rows),
    )
    return {
        **metrics,
    }


def rows_with_frontier(
    *,
    frontier: Frontier,
    num_rows: int,
) -> torch.Tensor:
    out = torch.zeros(
        int(num_rows),
        dtype=torch.bool,
        device=frontier.row_ids.device,
    )
    if frontier.row_ids.numel() > 0:
        out.index_fill_(0, frontier.row_ids, True)
    return out


def summarize_policy_diagnostics(
    *,
    policy_out: ForwardPolicyOutput,
) -> dict[str, torch.Tensor]:
    return {
        "policy_stop_log_prob_mean": policy_out.stop_log_prob.float().mean().detach(),
        "policy_action_log_prob_mean": policy_out.action_log_prob().float().mean().detach(),
        "policy_stop_prob_mean": policy_out.stop_prob().mean().detach(),
        "policy_edge_prob_mass_mean": policy_out.edge_prob_mass().mean().detach(),
        "policy_stop_expand_margin_mean": policy_stop_expand_margin_mean(policy_out).detach(),
    }


def masked_mean_or_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.numel() == 0 or not bool(mask.any()):
        return values.new_zeros(())
    return values.float()[mask].mean()


def masked_fraction(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    count = denominator.float().sum()
    if not bool(count.gt(0)):
        return count.new_zeros(())
    return numerator.float().sum() / count


__all__ = [
    "call_reward_model",
    "policy_stop_expand_margin_by_mask",
    "policy_stop_expand_margin_mean",
    "reward_delta_after_hit",
    "rollout_metrics",
    "source_rollout_metrics",
    "stop_expand_margin",
    "stop_expand_margin_by_depth",
    "stop_probability_by_depth",
    "summarize_policy_diagnostics",
    "supported_target_count",
]
