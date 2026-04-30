from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .buffers import RolloutBuffer


@dataclass(frozen=True)
class StepResult:
    log_pf: torch.Tensor
    log_pb: torch.Tensor

    action_type: torch.Tensor
    continue_mask: torch.Tensor
    stop_mask: torch.Tensor
    selected_edge_ids: torch.Tensor

    terminal_log_reward: torch.Tensor
    terminal_answer_f1: torch.Tensor

    proposal_intervention_mask: torch.Tensor

    terminal_edge_penalty: torch.Tensor
    terminal_base_log_reward: torch.Tensor
    terminal_utility: torch.Tensor
    terminal_expanded_edge_count: torch.Tensor
    terminal_minimal_edge_count: torch.Tensor
    terminal_minimality_gap: torch.Tensor
    terminal_minimality_penalty: torch.Tensor


@dataclass(frozen=True)
class RolloutTraces:
    """
    Step-level traces.

    Shape convention:
        all tensors: [B, T]
    """

    state_log_flows: torch.Tensor
    log_pf: torch.Tensor
    log_pb: torch.Tensor

    action_type: torch.Tensor
    continue_mask: torch.Tensor
    stop_mask: torch.Tensor
    selected_edge_ids: torch.Tensor

    stop_now_log_reward: torch.Tensor
    stop_now_answer_f1: torch.Tensor
    stop_now_valid_mask: torch.Tensor

    stop_log_pf: torch.Tensor
    stop_tb_valid_mask: torch.Tensor

    target_stop_prob: torch.Tensor
    target_continue_prob: torch.Tensor
    policy_action_valid_mask: torch.Tensor

    edge_action_entropy: torch.Tensor
    edge_action_entropy_valid_mask: torch.Tensor

    advantage_aux_loss: torch.Tensor
    advantage_aux_valid_mask: torch.Tensor

    proposal_intervention_mask: torch.Tensor
    budget_exhausted_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class RolloutStats:
    """
    Trajectory-level quantities.

    Shape convention:
        all tensors: [B]
    """

    root_log_z: torch.Tensor
    trajectory_length: torch.Tensor

    terminal_log_reward: torch.Tensor
    terminal_answer_f1: torch.Tensor

    proposal_intervention_count: torch.Tensor

    edge_action_entropy: torch.Tensor
    edge_action_entropy_valid_mask: torch.Tensor

    terminal_edge_penalty: torch.Tensor | None = None
    terminal_base_log_reward: torch.Tensor | None = None
    terminal_utility: torch.Tensor | None = None
    terminal_expanded_edge_count: torch.Tensor | None = None
    terminal_minimal_edge_count: torch.Tensor | None = None
    terminal_minimality_gap: torch.Tensor | None = None
    terminal_minimality_penalty: torch.Tensor | None = None


@dataclass(frozen=True)
class RolloutBatch:
    stats: RolloutStats
    traces: RolloutTraces

    @classmethod
    def from_buffer(
        cls,
        *,
        buffer: RolloutBuffer,
        root_log_z: torch.Tensor,
    ) -> RolloutBatch:
        return cls(
            stats=RolloutStats(
                root_log_z=root_log_z,
                trajectory_length=buffer.traj_len,
                terminal_log_reward=buffer.terminal_log_reward,
                terminal_answer_f1=buffer.terminal_answer_f1,
                terminal_edge_penalty=buffer.terminal_edge_penalty,
                terminal_base_log_reward=buffer.terminal_base_log_reward,
                terminal_utility=buffer.terminal_utility,
                terminal_expanded_edge_count=buffer.terminal_expanded_edge_count,
                terminal_minimal_edge_count=buffer.terminal_minimal_edge_count,
                terminal_minimality_gap=buffer.terminal_minimality_gap,
                terminal_minimality_penalty=buffer.terminal_minimality_penalty,
                proposal_intervention_count=buffer.proposal_intervention_mask.to(
                    dtype=torch.float32
                ).sum(dim=1),
                edge_action_entropy=buffer.edge_action_entropy.sum(dim=1),
                edge_action_entropy_valid_mask=buffer.edge_action_entropy_valid_mask.to(
                    dtype=torch.float32
                ).sum(dim=1),
            ),
            traces=RolloutTraces(
                state_log_flows=buffer.state_log_flows,
                log_pf=buffer.step_log_pf,
                log_pb=buffer.step_log_pb,
                action_type=buffer.action_type,
                continue_mask=buffer.continue_mask,
                stop_mask=buffer.stop_mask,
                selected_edge_ids=buffer.selected_edge_ids,
                stop_now_log_reward=buffer.stop_now_log_reward,
                stop_now_answer_f1=buffer.stop_now_answer_f1,
                stop_now_valid_mask=buffer.stop_now_valid_mask,
                stop_log_pf=buffer.stop_log_pf,
                stop_tb_valid_mask=buffer.stop_tb_valid_mask,
                target_stop_prob=buffer.target_stop_prob,
                target_continue_prob=buffer.target_continue_prob,
                policy_action_valid_mask=buffer.policy_action_valid_mask,
                edge_action_entropy=buffer.edge_action_entropy,
                edge_action_entropy_valid_mask=buffer.edge_action_entropy_valid_mask,
                advantage_aux_loss=buffer.advantage_aux_loss,
                advantage_aux_valid_mask=buffer.advantage_aux_valid_mask,
                proposal_intervention_mask=buffer.proposal_intervention_mask,
                budget_exhausted_mask=buffer.budget_exhausted_mask,
            ),
        )


__all__ = [
    "RolloutBatch",
    "RolloutStats",
    "RolloutTraces",
    "StepResult",
]
