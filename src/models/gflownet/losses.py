from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from src.models.configs import AnswerQuotientConfig, SubTrajectoryBalanceConfig

if TYPE_CHECKING:
    from .sampler import TrajectoryGFNSampleBatch


def _zero_metric_tensor() -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float32)


def _select_unique_terminal_path_indices(
    sample_batch: "TrajectoryGFNSampleBatch",
) -> torch.Tensor:
    termination_steps = getattr(sample_batch, "termination_action_steps", None)
    if termination_steps is None:
        termination_steps = sample_batch.terminal_num_steps
    trace_stop_mask = sample_batch.trace_stop_mask
    if trace_stop_mask is None:
        trace_stop_mask = torch.zeros_like(
            sample_batch.trace_edge_ids, dtype=torch.bool
        )
    batch_size, num_rollouts = sample_batch.start_nodes.shape
    max_actions = int(sample_batch.trace_edge_ids.size(-1))
    flat_graph_ids = (
        torch.arange(
            batch_size, device=sample_batch.start_nodes.device, dtype=torch.long
        )
        .unsqueeze(1)
        .expand(batch_size, num_rollouts)
        .reshape(-1)
        .detach()
        .cpu()
        .tolist()
    )
    flat_start_nodes = (
        sample_batch.start_nodes.reshape(-1)
        .detach()
        .cpu()
        .to(dtype=torch.long)
        .tolist()
    )
    flat_termination_steps = (
        termination_steps.reshape(-1).detach().cpu().to(dtype=torch.long).tolist()
    )
    flat_edge_rows = (
        sample_batch.trace_edge_ids.reshape(-1, max_actions)
        .detach()
        .cpu()
        .to(dtype=torch.long)
        .tolist()
    )
    flat_stop_rows = (
        trace_stop_mask.reshape(-1, max_actions)
        .detach()
        .cpu()
        .to(dtype=torch.long)
        .tolist()
    )
    unique_positions: list[int] = []
    seen_keys: set[tuple[int, ...]] = set()
    for idx in range(len(flat_graph_ids)):
        key = (
            int(flat_graph_ids[idx]),
            int(flat_start_nodes[idx]),
            int(flat_termination_steps[idx]),
            *[int(value) for value in flat_edge_rows[idx]],
            *[int(value) for value in flat_stop_rows[idx]],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_positions.append(idx)
    return torch.tensor(
        unique_positions,
        device=sample_batch.start_nodes.device,
        dtype=torch.long,
    )


@dataclass(frozen=True)
class SubTrajectoryBalanceLossOutput:
    loss: torch.Tensor
    subtb_loss: torch.Tensor
    residual_abs: torch.Tensor
    residual_variance: torch.Tensor
    root_abs: torch.Tensor
    success_rate: torch.Tensor
    log_z_mean: torch.Tensor
    log_z_variance: torch.Tensor
    root_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)
    pairwise_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)
    terminal_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)
    answer_quotient_component_loss: torch.Tensor = field(
        default_factory=_zero_metric_tensor
    )
    answer_quotient_residual_abs: torch.Tensor = field(
        default_factory=_zero_metric_tensor
    )
    answer_quotient_observed_sink_count: torch.Tensor = field(
        default_factory=_zero_metric_tensor
    )


class SubTrajectoryBalanceLoss:
    """Forward-prefix SubTB specialized to the prefix-tree state space.

    The current objective fits the multiplicatively shaped trajectory measure
    defined by the fixed terminal energy table plus any step-level log rewards
    carried in ``log_reward_steps``. Root, pairwise, and terminal residuals are
    always aggregated as separate components so long-horizon pairwise terms do
    not dominate the root boundary or suffix-return anchors. When configured,
    an additional answer-sink component groups terminal path mass by answer
    equivalence class before applying the terminal boundary constraint.

    Importantly, the loss is evaluated under whichever coverage distribution
    produced ``sample_batch``. Online rollouts may come from a proposal policy
    and replayed rollouts may come from teacher-forced buffer trajectories, but
    the residuals themselves always use target-policy ``log P_F`` and ``log F``.
    This implementation therefore fits SubTB constraints under the sampled
    coverage measure instead of importance-reweighting back to a target-policy
    expectation.
    """

    def __init__(
        self,
        *,
        config: SubTrajectoryBalanceConfig | None = None,
        answer_quotient_config: AnswerQuotientConfig | None = None,
        root_weight: float | None = None,
        move_weight: float | None = None,
        terminal_weight: float | None = None,
    ) -> None:
        del root_weight, move_weight, terminal_weight
        self.config = config or SubTrajectoryBalanceConfig()
        self.answer_quotient_config = answer_quotient_config or AnswerQuotientConfig()

    @staticmethod
    def _weighted_mean(
        *,
        values: torch.Tensor,
        weights: torch.Tensor,
        normalize: bool,
        reduce_dims: tuple[int, ...],
    ) -> torch.Tensor:
        weighted = values * weights
        if normalize:
            denom = weights.sum(dim=reduce_dims).clamp_min(1.0)
            return weighted.sum(dim=reduce_dims) / denom
        return weighted.sum(dim=reduce_dims)

    @staticmethod
    def _build_step_prefix(
        *,
        step_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_rollouts, max_steps = step_values.shape
        state_horizon = max_steps + 1
        step_prefix = step_values.new_zeros((batch_size, num_rollouts, state_horizon))
        if max_steps > 0:
            step_prefix[:, :, 1:] = torch.cumsum(step_values, dim=-1)
        return step_prefix

    @staticmethod
    def _component_means(
        *,
        root_loss: torch.Tensor,
        pairwise_loss: torch.Tensor,
        terminal_loss: torch.Tensor,
        state_weights: torch.Tensor,
        terminal_weights: torch.Tensor,
        normalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pairwise_component = SubTrajectoryBalanceLoss._weighted_mean(
            values=pairwise_loss,
            weights=state_weights,
            normalize=normalize,
            reduce_dims=(-2, -1),
        )
        terminal_component = SubTrajectoryBalanceLoss._weighted_mean(
            values=terminal_loss,
            weights=terminal_weights,
            normalize=normalize,
            reduce_dims=(-1,),
        )
        return root_loss, pairwise_component, terminal_component

    def _combine_components(
        self,
        *,
        root_loss: torch.Tensor,
        pairwise_loss: torch.Tensor,
        terminal_loss: torch.Tensor,
        state_weights: torch.Tensor,
        terminal_weights: torch.Tensor,
        answer_quotient_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_weight = float(self.config.root_loss_weight)
        pairwise_weight = float(self.config.pairwise_loss_weight)
        terminal_weight = float(self.config.terminal_loss_weight)
        answer_weight = float(self.answer_quotient_config.weight)

        root_component, pairwise_component, terminal_component = self._component_means(
            root_loss=root_loss,
            pairwise_loss=pairwise_loss,
            terminal_loss=terminal_loss,
            state_weights=state_weights,
            terminal_weights=terminal_weights,
            normalize=bool(self.config.normalize),
        )
        weighted_root = root_component * root_weight
        weighted_pairwise = pairwise_component * pairwise_weight
        weighted_terminal = terminal_component * terminal_weight
        weighted_answer = root_component.new_zeros(root_component.shape)
        if answer_quotient_loss is not None:
            weighted_answer = answer_quotient_loss * answer_weight
        total = weighted_root + weighted_pairwise + weighted_answer
        if not bool(self.answer_quotient_config.replace_terminal_loss):
            total = total + weighted_terminal
        return (
            total,
            weighted_root,
            weighted_pairwise,
            weighted_terminal,
            weighted_answer,
        )

    def _compute_answer_quotient_loss(
        self,
        *,
        sample_batch: "TrajectoryGFNSampleBatch",
        terminal_log_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        terminal_sink_ids = getattr(sample_batch, "terminal_sink_ids", None)
        terminal_sink_log_rewards = getattr(
            sample_batch, "terminal_sink_log_rewards", None
        )
        if terminal_sink_ids is None or terminal_sink_log_rewards is None:
            zero = terminal_log_mass.new_zeros(())
            return zero, zero, zero
        batch_size, num_rollouts = terminal_log_mass.shape
        unique_path_indices = _select_unique_terminal_path_indices(sample_batch)
        flat_graph_ids = (
            torch.arange(batch_size, device=terminal_log_mass.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(batch_size, num_rollouts)
            .reshape(-1)
        )
        flat_terminal_log_mass = terminal_log_mass.reshape(-1).index_select(
            0, unique_path_indices
        )
        flat_graph_ids = flat_graph_ids.index_select(0, unique_path_indices)
        flat_sink_ids = (
            terminal_sink_ids.to(dtype=torch.long)
            .reshape(-1)
            .index_select(0, unique_path_indices)
        )
        flat_sink_log_rewards = (
            terminal_sink_log_rewards.to(dtype=torch.float32)
            .reshape(-1)
            .index_select(0, unique_path_indices)
        )
        group_to_positions: dict[tuple[int, int], list[int]] = {}
        key_graph_ids = flat_graph_ids.detach().cpu().tolist()
        key_sink_ids = flat_sink_ids.detach().cpu().tolist()
        for position, (graph_id, sink_id) in enumerate(
            zip(key_graph_ids, key_sink_ids)
        ):
            group_to_positions.setdefault((int(graph_id), int(sink_id)), []).append(
                position
            )
        if not group_to_positions:
            zero = terminal_log_mass.new_zeros(())
            return zero, zero, zero
        sink_residuals: list[torch.Tensor] = []
        for positions in group_to_positions.values():
            position_index = torch.tensor(
                positions,
                device=terminal_log_mass.device,
                dtype=torch.long,
            )
            grouped_log_mass = torch.logsumexp(
                flat_terminal_log_mass.index_select(0, position_index), dim=0
            )
            sink_target = flat_sink_log_rewards[positions[0]]
            sink_residuals.append(grouped_log_mass - sink_target)
        residual_tensor = torch.stack(sink_residuals, dim=0)
        return (
            residual_tensor.square().mean(),
            residual_tensor.abs().mean(),
            residual_tensor.new_tensor(float(residual_tensor.numel())),
        )

    def compute(
        self, sample_batch: TrajectoryGFNSampleBatch
    ) -> SubTrajectoryBalanceLossOutput:
        log_pf_steps = sample_batch.log_pf_steps.to(dtype=torch.float32)
        if tuple(sample_batch.log_pb_steps.shape) != tuple(log_pf_steps.shape):
            raise ValueError(
                "log_pb_steps must match log_pf_steps shape for SubTB. "
                f"log_pb_steps={tuple(sample_batch.log_pb_steps.shape)} "
                f"log_pf_steps={tuple(log_pf_steps.shape)}."
            )

        batch_size, num_rollouts, max_steps = log_pf_steps.shape
        sequence_horizon = max_steps + 1
        graph_log_z_values = sample_batch.graph_log_z.to(dtype=torch.float32)
        start_log_probs = sample_batch.start_log_probs.to(dtype=torch.float32)
        start_state_log_f = sample_batch.start_state_log_f.to(dtype=torch.float32)
        start_log_rewards = getattr(sample_batch, "start_log_rewards", None)
        if start_log_rewards is None:
            start_log_rewards = torch.zeros_like(start_log_probs)
        else:
            start_log_rewards = start_log_rewards.to(dtype=torch.float32)
            if tuple(start_log_rewards.shape) != tuple(start_log_probs.shape):
                raise ValueError(
                    "start_log_rewards must match start_log_probs shape for SubTB. "
                    f"start_log_rewards={tuple(start_log_rewards.shape)} "
                    f"start_log_probs={tuple(start_log_probs.shape)}."
                )

        state_values = log_pf_steps.new_zeros(
            (batch_size, num_rollouts, sequence_horizon)
        )
        state_values[:, :, 0] = start_state_log_f
        if max_steps > 0:
            state_values[:, :, 1:] = sample_batch.next_state_log_f_steps.to(
                dtype=torch.float32
            )

        log_reward_steps = getattr(sample_batch, "log_reward_steps", None)
        if log_reward_steps is None:
            log_reward_steps = torch.zeros_like(log_pf_steps)
        else:
            log_reward_steps = log_reward_steps.to(dtype=torch.float32)
            if tuple(log_reward_steps.shape) != tuple(log_pf_steps.shape):
                raise ValueError(
                    "log_reward_steps must match log_pf_steps shape for SubTB. "
                    f"log_reward_steps={tuple(log_reward_steps.shape)} "
                    f"log_pf_steps={tuple(log_pf_steps.shape)}."
                )

        # MS-SubTB fits a shaped trajectory measure whose log-mass is the sum of
        # the terminal log-reward anchor and any per-step log-reward increments.
        # We keep `log_pb_steps` in the sample batch for interface compatibility,
        # but the loss itself only uses target forward prefixes and shaped reward
        # sums under the sampled coverage measure; there is no importance-ratio
        # correction against the online proposal policy in this implementation.
        forward_prefix = self._build_step_prefix(step_values=log_pf_steps)
        reward_prefix = self._build_step_prefix(step_values=log_reward_steps)

        terminal_counts = getattr(sample_batch, "termination_action_steps", None)
        if terminal_counts is None:
            terminal_index = sample_batch.terminal_num_steps.to(dtype=torch.long)
        else:
            terminal_index = terminal_counts.to(dtype=torch.long)
        if bool((terminal_index < 0).any().item()) or bool(
            (terminal_index >= sequence_horizon).any().item()
        ):
            raise ValueError(
                "terminal action index produced an out-of-range terminal state index for SubTB. "
                f"sequence_horizon={sequence_horizon}"
            )
        terminal_forward_prefix = forward_prefix.gather(
            -1, terminal_index.unsqueeze(-1)
        ).squeeze(-1)
        terminal_reward_prefix = reward_prefix.gather(
            -1, terminal_index.unsqueeze(-1)
        ).squeeze(-1)
        terminal_values = (
            sample_batch.terminal_log_rewards.to(dtype=torch.float32)
            - terminal_forward_prefix
            + terminal_reward_prefix
        )
        terminal_log_mass = (
            graph_log_z_values.unsqueeze(1)
            + start_log_probs
            - start_log_rewards
            + terminal_forward_prefix
            - terminal_reward_prefix
        )

        position_ids = torch.arange(sequence_horizon, device=state_values.device)
        position_ids = position_ids.view(1, 1, -1)
        state_position_mask = position_ids < terminal_index.unsqueeze(-1)
        terminal_start_mask = state_position_mask
        if not torch.isfinite(state_values[state_position_mask]).all():
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check log_pf/log_f/log_reward."
            )
        if not torch.isfinite(log_reward_steps).all():
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check step-level log rewards."
            )
        if not torch.isfinite(terminal_values).all():
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check log_pf/log_f/log_reward."
            )
        root_residual = (
            graph_log_z_values.unsqueeze(1)
            + start_log_probs
            - start_log_rewards
            - start_state_log_f
        )
        if not torch.isfinite(root_residual).all():
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check root log_z/log_pf/log_f."
            )

        start_positions = position_ids.unsqueeze(-1)
        end_positions = position_ids.unsqueeze(-2)
        pair_mask = (
            state_position_mask.unsqueeze(-1)
            & state_position_mask.unsqueeze(-2)
            & (start_positions < end_positions)
        )
        pair_lengths = (
            (end_positions - start_positions).clamp_min(0).to(dtype=torch.float32)
        )
        pairwise_forward = forward_prefix.unsqueeze(-2) - forward_prefix.unsqueeze(-1)
        pairwise_reward = reward_prefix.unsqueeze(-2) - reward_prefix.unsqueeze(-1)
        pairwise_residual = (
            state_values.unsqueeze(-1)
            + pairwise_forward
            - pairwise_reward
            - state_values.unsqueeze(-2)
        )
        terminal_residual = (
            state_values
            + terminal_forward_prefix.unsqueeze(-1)
            - terminal_reward_prefix.unsqueeze(-1)
            - forward_prefix
            + reward_prefix
            - sample_batch.terminal_log_rewards.to(dtype=torch.float32).unsqueeze(-1)
        )

        lambda_weight = float(self.config.lambda_weight)
        if lambda_weight == 1.0:
            state_weights = torch.ones_like(pairwise_residual)
            terminal_weights = torch.ones_like(terminal_residual)
        else:
            state_weights = torch.pow(
                torch.full_like(pairwise_residual, fill_value=lambda_weight),
                (pair_lengths - 1.0).clamp_min(0.0),
            )
            terminal_lengths = terminal_index.unsqueeze(-1).to(
                dtype=torch.float32
            ) - position_ids.to(dtype=torch.float32)
            terminal_weights = torch.pow(
                torch.full_like(terminal_residual, fill_value=lambda_weight),
                (terminal_lengths - 1.0).clamp_min(0.0),
            )
        state_weights = torch.where(
            pair_mask, state_weights, torch.zeros_like(state_weights)
        )
        terminal_weights = torch.where(
            terminal_start_mask,
            terminal_weights,
            torch.zeros_like(terminal_weights),
        )

        pairwise_loss = pairwise_residual.square()
        terminal_loss = terminal_residual.square()
        root_loss = root_residual.square()
        answer_quotient_loss = root_loss.new_zeros(())
        answer_quotient_residual_abs = root_loss.new_zeros(())
        answer_quotient_observed_sink_count = root_loss.new_zeros(())
        if self.answer_quotient_config.active:
            (
                answer_quotient_loss,
                answer_quotient_residual_abs,
                answer_quotient_observed_sink_count,
            ) = self._compute_answer_quotient_loss(
                sample_batch=sample_batch,
                terminal_log_mass=terminal_log_mass,
            )
        (
            per_rollout_loss,
            root_component_loss,
            pairwise_component_loss,
            terminal_component_loss,
            answer_quotient_component_loss,
        ) = self._combine_components(
            root_loss=root_loss,
            pairwise_loss=pairwise_loss,
            terminal_loss=terminal_loss,
            state_weights=state_weights,
            terminal_weights=terminal_weights,
            answer_quotient_loss=answer_quotient_loss,
        )
        loss = per_rollout_loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check log_pf/log_f/log_reward."
            )

        valid_residuals: list[torch.Tensor] = []
        if int(root_residual.numel()) > 0:
            valid_residuals.append(root_residual.reshape(-1))
        if bool(pair_mask.any().item()):
            valid_residuals.append(pairwise_residual[pair_mask])
        if bool(terminal_start_mask.any().item()):
            valid_residuals.append(terminal_residual[terminal_start_mask])
        if valid_residuals:
            all_residuals = torch.cat(valid_residuals, dim=0)
            residual_abs = all_residuals.abs().mean()
            residual_variance = all_residuals.var(unbiased=False)
        else:
            residual_abs = torch.zeros((), device=loss.device)
            residual_variance = torch.zeros((), device=loss.device)

        return SubTrajectoryBalanceLossOutput(
            loss=loss,
            subtb_loss=loss.detach(),
            residual_abs=residual_abs.detach(),
            residual_variance=residual_variance.detach(),
            root_abs=root_residual.abs().mean().detach(),
            success_rate=sample_batch.success_mask.to(dtype=torch.float32)
            .mean()
            .detach(),
            log_z_mean=graph_log_z_values.mean().detach(),
            log_z_variance=graph_log_z_values.var(unbiased=False).detach(),
            root_component_loss=root_component_loss.mean().detach(),
            pairwise_component_loss=pairwise_component_loss.mean().detach(),
            terminal_component_loss=terminal_component_loss.mean().detach(),
            answer_quotient_component_loss=answer_quotient_component_loss.mean().detach(),
            answer_quotient_residual_abs=answer_quotient_residual_abs.detach(),
            answer_quotient_observed_sink_count=answer_quotient_observed_sink_count.detach(),
        )


__all__ = [
    "SubTrajectoryBalanceLoss",
    "SubTrajectoryBalanceLossOutput",
]
