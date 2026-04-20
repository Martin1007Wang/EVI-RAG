from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from src.models.replay import TrajectoryTrace

ROLLOUT_DTYPE = torch.float32


@dataclass(frozen=True)
class RolloutBatch:
    """Batched rollout output. Shape: B = batch size, T = max_steps + 1."""

    root_log_z: torch.Tensor
    traj_len: torch.Tensor
    trajectory_log_pf: torch.Tensor
    trajectory_log_pb: torch.Tensor
    terminal_log_rewards: torch.Tensor

    state_log_flows: torch.Tensor | None = None
    step_log_pf: torch.Tensor | None = None
    step_log_pb: torch.Tensor | None = None
    step_log_shaping: torch.Tensor | None = None

    terminal_active_nodes: torch.Tensor | None = None
    terminal_active_edges: torch.Tensor | None = None
    terminal_pre_stop_active_nodes: torch.Tensor | None = None
    terminal_pre_stop_active_edges: torch.Tensor | None = None
    terminal_stop_mask: torch.Tensor | None = None
    terminal_stop_log_pb: torch.Tensor | None = None
    selected_edge_ids: torch.Tensor | None = None
    selected_relation_only_logits: torch.Tensor | None = None
    selected_final_logits: torch.Tensor | None = None
    teacher_forced_action_count: torch.Tensor | None = None
    trajectory_traces: tuple[TrajectoryTrace, ...] | None = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            **{
                field.name: (
                    getattr(self, field.name).to(device)
                    if isinstance(getattr(self, field.name), torch.Tensor)
                    else getattr(self, field.name)
                )
                for field in dataclasses.fields(self)
            }
        )


@dataclass
class StepResult:
    """Single-step rollout outputs written into history buffers."""

    log_pf: torch.Tensor
    log_pb: torch.Tensor
    log_shaping: torch.Tensor
    stop_mask: torch.Tensor
    terminal_log_rewards: torch.Tensor
    terminal_stop_log_pb: torch.Tensor
    chosen_edge_ids: torch.Tensor
    chosen_relation_only_logits: torch.Tensor
    chosen_final_logits: torch.Tensor


__all__ = ["ROLLOUT_DTYPE", "RolloutBatch", "StepResult"]
