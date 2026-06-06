from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.graph.segments import segment_logsumexp
from src.weaver.objectives.output import ObjectiveOutput
from src.weaver.reward import EvidenceStateScoreOutput

from .batch import SubTBBatch, SubTBTermTable
from .scoring import SubTBPolicyScores


@dataclass(frozen=True, slots=True)
class ResidualStats:
    abs_sum: float
    count: int
    abs_p95: float
    abs_max: float


class ForwardLookingSubTBObjective(nn.Module):
    def __init__(
        self,
        *,
        subtb_lambda: float = 1.0,
        terminal_loss_weight: float = 2.0,
        path_nce_weight: float = 0.1,
        path_nce_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.subtb_lambda = float(subtb_lambda)
        if self.subtb_lambda < 0.0:
            raise ValueError("subtb_lambda must be nonnegative.")
        self.terminal_loss_weight = float(terminal_loss_weight)
        self.path_nce_weight = float(path_nce_weight)
        self.path_nce_temperature = float(path_nce_temperature)
        if self.terminal_loss_weight < 0.0:
            raise ValueError("terminal_loss_weight must be nonnegative.")
        if self.path_nce_weight < 0.0:
            raise ValueError("path_nce_weight must be nonnegative.")
        if self.path_nce_temperature <= 0.0:
            raise ValueError("path_nce_temperature must be positive.")

    def forward(
        self,
        *,
        batch: SubTBBatch,
        scores: SubTBPolicyScores,
        reward: EvidenceStateScoreOutput,
        global_step: int = 0,
        path_gold_mask: torch.Tensor | None = None,
    ) -> ObjectiveOutput:
        del global_step
        forward_prefix_by_traj = _prefix_sums(scores.step_log_prob)
        backward_prefix_by_traj = _prefix_sums(scores.backward_step_log_prob)
        transition_loss, transition_weight, transition_stats = _transition_terms(
            terms=batch.transition_terms,
            scores=scores,
            forward_prefix_by_traj=forward_prefix_by_traj,
            backward_prefix_by_traj=backward_prefix_by_traj,
            subtb_lambda=self.subtb_lambda,
        )
        terminal_loss, terminal_weight, terminal_stats = _terminal_terms(
            terms=batch.terminal_terms,
            batch=batch,
            scores=scores,
            reward=reward,
            forward_prefix_by_traj=forward_prefix_by_traj,
            backward_prefix_by_traj=backward_prefix_by_traj,
            subtb_lambda=self.subtb_lambda,
        )
        path_nce_loss, path_nce_stats = _path_nce_terms(
            scores=scores,
            gold_mask=path_gold_mask,
            temperature=self.path_nce_temperature,
        )

        total_loss = transition_loss + self.terminal_loss_weight * terminal_loss
        total_weight = transition_weight + self.terminal_loss_weight * terminal_weight
        base_loss = total_loss / total_weight.clamp_min(1.0)
        loss = base_loss + self.path_nce_weight * path_nce_loss

        terminal_valid_by_traj = reward.terminal_valid_mask.index_select(0, batch.terminal_state_ids)
        terminal_stop_metric_mask = batch.terminal_trainable_stop_mask & terminal_valid_by_traj
        stop_log_prob_terminal = scores.stop_log_prob_by_state.index_select(
            0,
            batch.terminal_state_ids[terminal_stop_metric_mask],
        )
        backward_metric_values = scores.backward_step_log_prob[batch.valid_steps]

        metrics: dict[str, float] = {
            "objective/loss": float(loss.detach()),
            "objective/base_loss": float(base_loss.detach()),
            "objective/state_count": float(batch.states.num_states),
            "objective/trajectory_count": float(batch.trajectories.num_trajectories),
            "objective/transition_term_count": float(batch.transition_terms.num_terms),
            "objective/terminal_term_count": float(batch.terminal_terms.num_terms),
            "objective/frontier_size_total": float(scores.frontier_count.sum().item()),
            "objective/frontier_size_max": float(scores.frontier_count.max().item()) if int(scores.frontier_count.numel()) else 0.0,
            "objective/frontier_size_mean": _mean(scores.frontier_count),
            "objective/log_flow_mean": _mean(scores.log_flow),
            "objective/stop_log_prob_all_mean": _mean(scores.stop_log_prob_by_state),
            "objective/terminal_stop_log_prob_mean": _mean(stop_log_prob_terminal),
            "objective/forward_log_prob_mean": _mean(scores.step_log_prob[batch.valid_steps]),
            "objective/backward_log_prob_abs_mean": _abs_mean(backward_metric_values),
            "objective/subtb_transition_abs_residual_mean": _stats_mean(transition_stats),
            "objective/subtb_transition_abs_residual_p95": transition_stats.abs_p95,
            "objective/subtb_transition_abs_residual_max": transition_stats.abs_max,
            "objective/subtb_terminal_abs_residual_mean": _stats_mean(terminal_stats),
            "objective/subtb_terminal_abs_residual_p95": terminal_stats.abs_p95,
            "objective/subtb_terminal_abs_residual_max": terminal_stats.abs_max,
            "objective/terminal_loss_weight": float(self.terminal_loss_weight),
            "objective/path_nce_loss": float(path_nce_loss.detach()),
            "objective/path_nce_weight": float(self.path_nce_weight),
            "objective/path_nce_temperature": float(self.path_nce_temperature),
            "objective/path_nce_state_count": float(path_nce_stats[0]),
            "objective/path_nce_positive_count": float(path_nce_stats[1]),
            "objective/terminal_stop_trainable_count": float(batch.terminal_trainable_stop_mask.sum().item()),
            "objective/terminal_budget_truncated_count": float(batch.trajectories.is_budget_truncated.sum().item()),
            "objective/terminal_external_count": float(batch.trajectories.is_external_terminal.sum().item()),
        }
        metrics.update({str(name): float(value.detach()) for name, value in reward.metrics.items()})
        return ObjectiveOutput(
            loss=loss,
            metrics=metrics,
            num_states=batch.states.num_states,
        )


def _transition_terms(
    *,
    terms: SubTBTermTable,
    scores: SubTBPolicyScores,
    forward_prefix_by_traj: torch.Tensor,
    backward_prefix_by_traj: torch.Tensor,
    subtb_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, ResidualStats]:
    if terms.num_terms == 0:
        zero = scores.log_flow.sum() * 0.0
        return zero, zero, _empty_residual_stats()

    log_pf = (
        forward_prefix_by_traj[terms.traj_ids, terms.end_steps]
        - forward_prefix_by_traj[terms.traj_ids, terms.start_steps]
    )
    log_pb = (
        backward_prefix_by_traj[terms.traj_ids, terms.end_steps]
        - backward_prefix_by_traj[terms.traj_ids, terms.start_steps]
    )
    residual = (
        scores.log_flow.index_select(0, terms.start_state_ids)
        + log_pf
        - log_pb
        - scores.log_flow.index_select(0, terms.end_state_ids)
    )
    weight = _lambda_weight(reference=residual, exponents=terms.lambda_exponent, subtb_lambda=subtb_lambda)
    return (weight * _subtb_residual_penalty(residual)).sum(), weight.sum(), _residual_stats(residual)


def _terminal_terms(
    *,
    terms: SubTBTermTable,
    batch: SubTBBatch,
    scores: SubTBPolicyScores,
    reward: EvidenceStateScoreOutput,
    forward_prefix_by_traj: torch.Tensor,
    backward_prefix_by_traj: torch.Tensor,
    subtb_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, ResidualStats]:
    if terms.num_terms == 0:
        zero = scores.log_flow.sum() * 0.0
        return zero, zero, _empty_residual_stats()

    reward_valid = reward.terminal_valid_mask.index_select(0, terms.end_state_ids)
    if not bool(reward_valid.any()):
        zero = scores.log_flow.sum() * 0.0
        return zero, zero, _empty_residual_stats()

    valid_idx = reward_valid.nonzero(as_tuple=False).flatten()
    traj_ids = terms.traj_ids.index_select(0, valid_idx)
    start_steps = terms.start_steps.index_select(0, valid_idx)
    end_steps = terms.end_steps.index_select(0, valid_idx)
    start_state_ids = terms.start_state_ids.index_select(0, valid_idx)
    end_state_ids = terms.end_state_ids.index_select(0, valid_idx)
    exponents = terms.lambda_exponent.index_select(0, valid_idx)

    log_pf = (
        forward_prefix_by_traj[traj_ids, end_steps]
        - forward_prefix_by_traj[traj_ids, start_steps]
    )
    log_pb = (
        backward_prefix_by_traj[traj_ids, end_steps]
        - backward_prefix_by_traj[traj_ids, start_steps]
    )

    terminal_stop_logp = scores.terminal_stop_logp_by_traj.index_select(0, traj_ids)
    train_stop = batch.terminal_trainable_stop_mask.index_select(0, traj_ids)
    terminal_action_logp = torch.where(
        train_stop,
        terminal_stop_logp,
        terminal_stop_logp.new_zeros(terminal_stop_logp.shape),
    )

    terminal_log_reward = reward.log_reward.detach().float().index_select(0, end_state_ids)
    residual = (
        scores.log_flow.index_select(0, start_state_ids)
        + log_pf
        + terminal_action_logp
        - log_pb
        - terminal_log_reward
    )
    weight = _lambda_weight(reference=residual, exponents=exponents, subtb_lambda=subtb_lambda)
    return (weight * _subtb_residual_penalty(residual)).sum(), weight.sum(), _residual_stats(residual)


def _path_nce_terms(
    *,
    scores: SubTBPolicyScores,
    gold_mask: torch.Tensor | None,
    temperature: float,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if (
        gold_mask is None
        or scores.frontier_row_ids is None
        or scores.frontier_log_prob is None
        or int(scores.frontier_log_prob.numel()) == 0
    ):
        zero = scores.log_flow.sum() * 0.0
        return zero, (0, 0)

    gold_mask = gold_mask.to(device=scores.frontier_log_prob.device, dtype=torch.bool).view(-1)
    if int(gold_mask.numel()) != int(scores.frontier_log_prob.numel()):
        raise ValueError("path_gold_mask must have one value per frontier action.")
    if not bool(gold_mask.any()):
        zero = scores.log_flow.sum() * 0.0
        return zero, (0, 0)

    row_ids = scores.frontier_row_ids.to(device=scores.frontier_log_prob.device, dtype=torch.long).view(-1)
    if int(row_ids.numel()) != int(scores.frontier_log_prob.numel()):
        raise ValueError("frontier_row_ids must have one value per frontier action.")

    num_states = int(scores.log_flow.numel())
    positives_by_row = torch.bincount(row_ids[gold_mask], minlength=num_states)
    total_by_row = torch.bincount(row_ids, minlength=num_states)
    trainable_rows = positives_by_row.gt(0) & positives_by_row.lt(total_by_row)
    trainable_mask = trainable_rows.index_select(0, row_ids)
    if not bool(trainable_mask.any()):
        zero = scores.log_flow.sum() * 0.0
        return zero, (0, 0)

    scaled_log_prob = scores.frontier_log_prob.float() / float(temperature)
    all_log_z = segment_logsumexp(
        values=scaled_log_prob[trainable_mask],
        segment_ids=row_ids[trainable_mask],
        num_segments=num_states,
    )
    positive_mask = trainable_mask & gold_mask
    positive_log_z = segment_logsumexp(
        values=scaled_log_prob[positive_mask],
        segment_ids=row_ids[positive_mask],
        num_segments=num_states,
    )
    rows = trainable_rows.nonzero(as_tuple=False).flatten()
    loss = all_log_z.index_select(0, rows) - positive_log_z.index_select(0, rows)
    return loss.mean(), (int(rows.numel()), int(gold_mask[trainable_mask].sum().item()))


def _lambda_weight(*, reference: torch.Tensor, exponents: torch.Tensor, subtb_lambda: float) -> torch.Tensor:
    if subtb_lambda == 1.0:
        return reference.new_ones(int(exponents.numel()))
    return reference.new_full((int(exponents.numel()),), subtb_lambda).pow(exponents.float())


def _subtb_residual_penalty(residual: torch.Tensor) -> torch.Tensor:
    return residual.square()


def _residual_stats(residual: torch.Tensor) -> ResidualStats:
    if int(residual.numel()) == 0:
        return _empty_residual_stats()
    abs_residual = residual.detach().abs().float()
    return ResidualStats(
        abs_sum=float(abs_residual.sum().item()),
        count=int(abs_residual.numel()),
        abs_p95=float(torch.quantile(abs_residual, 0.95).item()),
        abs_max=float(abs_residual.max().item()),
    )


def _empty_residual_stats() -> ResidualStats:
    return ResidualStats(abs_sum=0.0, count=0, abs_p95=0.0, abs_max=0.0)


def _stats_mean(stats: ResidualStats) -> float:
    return stats.abs_sum / float(stats.count) if stats.count > 0 else 0.0


def _abs_mean(value: torch.Tensor) -> float:
    return float(value.abs().mean().detach()) if int(value.numel()) else 0.0


def _mean(value: torch.Tensor) -> float:
    return float(value.mean().detach()) if int(value.numel()) else 0.0


def _prefix_sums(step_log_prob: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            step_log_prob.new_zeros((step_log_prob.size(0), 1)),
            torch.cumsum(step_log_prob, dim=1),
        ],
        dim=1,
    )
