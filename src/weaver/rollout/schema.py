from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .buffer import RolloutBuffer


@dataclass(frozen=True, slots=True)
class StepResult:
    """
    One executed rollout transition.

    Shape convention:
        all tensors: [B]

    terminal_* fields are only meaningful for graphs whose stop_mask is true
    at this transition. RolloutBuffer copies them into trajectory-level fields.
    """

    log_pf: torch.Tensor
    log_pb: torch.Tensor

    action_type: torch.Tensor
    continue_mask: torch.Tensor
    stop_mask: torch.Tensor
    selected_edge_ids: torch.Tensor

    terminal_log_reward: torch.Tensor
    terminal_answer_f1: torch.Tensor

    terminal_complexity_penalty: torch.Tensor
    terminal_base_log_reward: torch.Tensor
    terminal_utility: torch.Tensor
    terminal_expanded_edge_count: torch.Tensor
    terminal_answer_degree_excess: torch.Tensor


@dataclass(frozen=True, slots=True)
class RolloutTraces:
    """
    Step-level rollout traces.

    Shape convention:
        all tensors: [B, T]

    StopAdv convention:
        stop_adv_target:
            Soft stop target y_stop(s) in [0, 1].

        stop_adv_valid_mask:
            True where the StopAdv oracle was evaluated.

        stop_adv_continue_log_reward:
            Counterfactual pooled continue value J_continue(s), used only for
            diagnostics.

    stop_adv_loss is a backward-compatible diagnostic field. When omitted, it is
    computed from stop_log_pf and stop_adv_target during initialization.
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

    budget_exhausted_mask: torch.Tensor | None = None

    stop_adv_loss: torch.Tensor | None = None
    stop_adv_target: torch.Tensor | None = None
    stop_adv_valid_mask: torch.Tensor | None = None
    stop_adv_continue_log_reward: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.stop_adv_loss is None:
            object.__setattr__(self, "stop_adv_loss", self._compute_stop_adv_loss())

    def _compute_stop_adv_loss(self) -> torch.Tensor | None:
        if self.stop_adv_target is None or self.stop_adv_valid_mask is None:
            return None

        stop_prob = torch.where(
            torch.isfinite(self.stop_log_pf),
            self.stop_log_pf.exp(),
            torch.zeros_like(self.stop_log_pf),
        ).clamp(1e-6, 1.0 - 1e-6)
        target = self.stop_adv_target.to(
            device=stop_prob.device,
            dtype=stop_prob.dtype,
        ).clamp(0.0, 1.0)
        loss = F.binary_cross_entropy(stop_prob, target, reduction="none")
        valid = self.stop_adv_valid_mask.to(device=loss.device, dtype=torch.bool)
        return torch.where(valid, loss, torch.zeros_like(loss))


@dataclass(frozen=True, slots=True)
class RolloutStats:
    """
    Trajectory-level rollout quantities.

    Shape convention:
        all tensors: [B]
    """

    root_log_z: torch.Tensor
    trajectory_length: torch.Tensor

    terminal_log_reward: torch.Tensor
    terminal_answer_f1: torch.Tensor

    edge_action_entropy: torch.Tensor
    edge_action_count: torch.Tensor

    terminal_complexity_penalty: torch.Tensor | None = None
    terminal_base_log_reward: torch.Tensor | None = None
    terminal_utility: torch.Tensor | None = None
    terminal_expanded_edge_count: torch.Tensor | None = None
    terminal_answer_degree_excess: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """
    Immutable rollout result consumed by losses and metrics.
    """

    stats: RolloutStats
    traces: RolloutTraces

    @classmethod
    def from_buffer(
        cls,
        *,
        buffer: RolloutBuffer,
    ) -> "RolloutBatch":
        if not buffer.root_log_z_written:
            raise RuntimeError("RolloutBuffer.root_log_z was never written.")

        edge_entropy = buffer.edge_action_entropy
        edge_entropy_valid = buffer.edge_action_entropy_valid_mask.to(
            dtype=torch.float32
        )

        return cls(
            stats=RolloutStats(
                root_log_z=buffer.root_log_z,
                trajectory_length=buffer.traj_len,
                terminal_log_reward=buffer.terminal_log_reward,
                terminal_answer_f1=buffer.terminal_answer_f1,
                edge_action_entropy=edge_entropy.sum(dim=1),
                edge_action_count=edge_entropy_valid.sum(dim=1),
                terminal_complexity_penalty=buffer.terminal_complexity_penalty,
                terminal_base_log_reward=buffer.terminal_base_log_reward,
                terminal_utility=buffer.terminal_utility,
                terminal_expanded_edge_count=buffer.terminal_expanded_edge_count,
                terminal_answer_degree_excess=buffer.terminal_answer_degree_excess,
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
                edge_action_entropy=edge_entropy,
                edge_action_entropy_valid_mask=buffer.edge_action_entropy_valid_mask,
                budget_exhausted_mask=buffer.budget_exhausted_mask,
                stop_adv_target=buffer.stop_adv_target,
                stop_adv_valid_mask=buffer.stop_adv_valid_mask,
                stop_adv_continue_log_reward=buffer.stop_adv_continue_log_reward,
            ),
        )


__all__ = [
    "RolloutBatch",
    "RolloutStats",
    "RolloutTraces",
    "StepResult",
]
