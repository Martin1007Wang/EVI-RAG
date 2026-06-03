from __future__ import annotations

import torch
from torch import nn

from src.weaver.objectives.output import ObjectiveOutput
from src.weaver.reward import TerminalRewardOutput

from .batch import SubTBBatch, SubTBTermTable
from .scoring import SubTBPolicyScores


class SubTrajectoryBalanceObjective(nn.Module):
    def __init__(
        self,
        *,
        subtb_lambda: float = 1.0,
        max_subtrajectory_length: int | None = None,
    ) -> None:
        super().__init__()
        self.subtb_lambda = float(subtb_lambda)
        self.max_subtrajectory_length = None if max_subtrajectory_length is None else int(max_subtrajectory_length)
        if self.subtb_lambda < 0.0:
            raise ValueError("subtb_lambda must be nonnegative.")
        if self.max_subtrajectory_length is not None and self.max_subtrajectory_length <= 0:
            raise ValueError("max_subtrajectory_length must be positive or None.")

    def forward(
        self,
        *,
        batch: SubTBBatch,
        scores: SubTBPolicyScores,
        reward: TerminalRewardOutput,
    ) -> ObjectiveOutput:
        loss, metrics = _subtb_loss_and_metrics(
            batch=batch,
            scores=scores,
            reward=reward,
            subtb_lambda=self.subtb_lambda,
        )
        return ObjectiveOutput(loss=loss, metrics=metrics, num_states=batch.states.num_states)


def _subtb_loss_and_metrics(
    *,
    batch: SubTBBatch,
    scores: SubTBPolicyScores,
    reward: TerminalRewardOutput,
    subtb_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = scores.log_flow.sum() * 0.0
    total_loss = zero
    total_weight = scores.log_flow.new_zeros(())

    policy_loss, policy_weight, policy_stats = _transition_terms(
        terms=batch.policy_transition_terms,
        batch=batch,
        scores=scores,
        subtb_lambda=subtb_lambda,
    )
    replay_loss, replay_weight, replay_stats = _transition_terms(
        terms=batch.replay_transition_terms,
        batch=batch,
        scores=scores,
        subtb_lambda=subtb_lambda,
    )
    terminal_loss, terminal_weight, terminal_stats = _terminal_terms(
        terms=batch.terminal_terms,
        batch=batch,
        scores=scores,
        reward=reward,
        subtb_lambda=subtb_lambda,
    )
    total_loss = total_loss + policy_loss + replay_loss + terminal_loss
    total_weight = total_weight + policy_weight + replay_weight + terminal_weight
    loss = total_loss / total_weight.clamp_min(1.0)

    stop_log_prob_terminal = scores.stop_log_prob_by_state.index_select(0, batch.terminal_state_ids[batch.trainable_terminal_mask])
    stop_action_log_prob_terminal = scores.stop_log_prob_by_state.index_select(0, batch.terminal_state_ids[batch.terminal_stop_action_mask])
    skipped_budget = batch.trajectories.is_budget_boundary.sum().item()
    skipped_external = batch.trajectories.is_external_terminal.sum().item()
    supervised_external = (batch.terminal_stop_action_mask & batch.trajectories.is_external_terminal).sum().item()
    metrics = {
        "objective/loss": float(loss.detach()),
        "objective/state_count": float(batch.states.num_states),
        "objective/trajectory_count": float(batch.trajectories.num_trajectories),
        "objective/frontier_size_total": float(scores.action_count.sum().item()),
        "objective/frontier_size_max": float(scores.action_count.max().item()) if int(scores.action_count.numel()) else 0.0,
        "objective/frontier_size_mean": _mean(scores.frontier_count),
        "objective/log_flow_mean": _mean(scores.log_flow),
        "objective/log_flow_std": _std(scores.log_flow),
        "objective/stop_log_prob_all_mean": _mean(scores.stop_log_prob_by_state),
        "objective/stop_log_prob_terminal_mean": _mean(stop_log_prob_terminal),
        "objective/forward_log_prob_mean": _mean(scores.step_log_prob[batch.valid_steps]),
        "objective/backward_log_prob_abs_mean": _abs_mean(scores.log_backward_by_state[batch.states.edge_count.gt(0)]),
        "objective/subtb_transition_abs_residual_mean": _weighted_abs_mean_from_stats(policy_stats, replay_stats),
        "objective/subtb_transition_abs_residual_mean_policy": _stats_mean(policy_stats),
        "objective/subtb_transition_abs_residual_mean_replay": _stats_mean(replay_stats),
        "objective/subtb_terminal_abs_residual_mean": _stats_mean(terminal_stats),
        "objective/terminal_reward_trajectory_count": float(torch.unique(batch.terminal_terms.traj_ids).numel()),
        "objective/terminal_reward_term_count": float(batch.terminal_terms.num_terms),
        "objective/terminal_trainable_trajectory_count": float(batch.trainable_terminal_mask.sum().item()),
        "objective/terminal_stop_action_count": float(batch.terminal_stop_action_mask.sum().item()),
        "objective/terminal_stop_action_external_count": float(supervised_external),
        "objective/terminal_skipped_budget_count": float(skipped_budget),
        "objective/terminal_skipped_external_count": float(skipped_external),
        "objective/stop_log_prob_supervised_terminal_mean": _mean(stop_action_log_prob_terminal),
    }
    metrics.update({str(name): float(value.detach()) for name, value in reward.metrics.items()})
    metrics["subtb/unique_state_count"] = float(batch.states.num_states)
    return loss, metrics


def _transition_terms(
    *,
    terms: SubTBTermTable,
    batch: SubTBBatch,
    scores: SubTBPolicyScores,
    subtb_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, int]]:
    if terms.num_terms == 0:
        zero = scores.log_flow.sum() * 0.0
        return zero, zero, (0.0, 0)
    residual = (
        scores.log_flow.index_select(0, terms.start_state_ids)
        + (
            scores.forward_prefix_by_traj[terms.traj_ids, terms.end_steps]
            - scores.forward_prefix_by_traj[terms.traj_ids, terms.start_steps]
        )
        - (
            scores.backward_prefix_by_traj[terms.traj_ids, terms.end_steps]
            - scores.backward_prefix_by_traj[terms.traj_ids, terms.start_steps]
        )
        - scores.log_flow.index_select(0, terms.end_state_ids)
    )
    weight = _lambda_weight(reference=residual, exponents=terms.lambda_exponent, subtb_lambda=subtb_lambda)
    return (weight * residual.square()).sum(), weight.sum(), _residual_stats(residual)


def _terminal_terms(
    *,
    terms: SubTBTermTable,
    batch: SubTBBatch,
    scores: SubTBPolicyScores,
    reward: TerminalRewardOutput,
    subtb_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, int]]:
    if terms.num_terms == 0:
        zero = scores.log_flow.sum() * 0.0
        return zero, zero, (0.0, 0)
    reward_valid = reward.valid_mask.index_select(0, terms.end_state_ids)
    if not bool(reward_valid.any()):
        zero = scores.log_flow.sum() * 0.0
        return zero, zero, (0.0, 0)
    valid_idx = reward_valid.nonzero(as_tuple=False).flatten()
    traj_ids = terms.traj_ids.index_select(0, valid_idx)
    start_steps = terms.start_steps.index_select(0, valid_idx)
    end_steps = terms.end_steps.index_select(0, valid_idx)
    start_state_ids = terms.start_state_ids.index_select(0, valid_idx)
    end_state_ids = terms.end_state_ids.index_select(0, valid_idx)
    exponents = terms.lambda_exponent.index_select(0, valid_idx)
    terminal_stop_logp = scores.terminal_stop_logp_by_traj.index_select(0, traj_ids)
    terminal_stop_action = batch.terminal_stop_action_mask.index_select(0, traj_ids)
    terminal_action_logp = torch.where(
        terminal_stop_action,
        terminal_stop_logp,
        terminal_stop_logp.new_zeros(terminal_stop_logp.shape),
    )
    residual = (
        scores.log_flow.index_select(0, start_state_ids)
        + (scores.forward_prefix_by_traj[traj_ids, end_steps] - scores.forward_prefix_by_traj[traj_ids, start_steps])
        + terminal_action_logp
        - (scores.backward_prefix_by_traj[traj_ids, end_steps] - scores.backward_prefix_by_traj[traj_ids, start_steps])
        - reward.log_reward.float().index_select(0, end_state_ids)
    )
    weight = _lambda_weight(reference=residual, exponents=exponents, subtb_lambda=subtb_lambda)
    return (weight * residual.square()).sum(), weight.sum(), _residual_stats(residual)


def _lambda_weight(*, reference: torch.Tensor, exponents: torch.Tensor, subtb_lambda: float) -> torch.Tensor:
    if subtb_lambda == 1.0:
        return reference.new_ones(int(exponents.numel()))
    return reference.new_full((int(exponents.numel()),), subtb_lambda).pow(exponents.float())


def _residual_stats(residual: torch.Tensor) -> tuple[float, int]:
    if int(residual.numel()) == 0:
        return 0.0, 0
    detached = residual.detach()
    return float(detached.abs().sum().item()), int(detached.numel())


def _stats_mean(stats: tuple[float, int]) -> float:
    abs_sum, count = stats
    return abs_sum / float(count) if count > 0 else 0.0


def _weighted_abs_mean_from_stats(*stats: tuple[float, int]) -> float:
    total_sum = 0.0
    total_count = 0
    for abs_sum, count in stats:
        total_sum += abs_sum
        total_count += count
    return total_sum / float(total_count) if total_count > 0 else 0.0


def _abs_mean(value: torch.Tensor) -> float:
    return float(value.abs().mean().detach()) if int(value.numel()) else 0.0


def _mean(value: torch.Tensor) -> float:
    return float(value.mean().detach()) if int(value.numel()) else 0.0


def _std(value: torch.Tensor) -> float:
    return float(value.std(unbiased=False).detach()) if int(value.numel()) else 0.0
