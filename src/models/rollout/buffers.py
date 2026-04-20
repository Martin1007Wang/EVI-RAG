from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .types import ROLLOUT_DTYPE, StepResult


@dataclass
class RolloutAccumulators:
    """Cross-step history buffers for one rollout over a batched graph set."""

    B: int
    T: int
    device: torch.device

    is_terminated: torch.Tensor = field(init=False)
    traj_len: torch.Tensor = field(init=False)
    terminal_log_rewards: torch.Tensor = field(init=False)
    terminal_stop_mask: torch.Tensor = field(init=False)
    terminal_stop_log_pb: torch.Tensor = field(init=False)
    trajectory_log_pf: torch.Tensor = field(init=False)
    trajectory_log_pb: torch.Tensor = field(init=False)
    state_log_flows: torch.Tensor = field(init=False)
    step_log_pf: torch.Tensor = field(init=False)
    step_log_pb: torch.Tensor = field(init=False)
    step_log_shaping: torch.Tensor = field(init=False)
    selected_edge_ids: torch.Tensor = field(init=False)
    selected_relation_only_logits: torch.Tensor = field(init=False)
    selected_final_logits: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        B, T, device = self.B, self.T, self.device
        self.is_terminated = torch.zeros(B, dtype=torch.bool, device=device)
        self.traj_len = torch.zeros(B, dtype=torch.long, device=device)
        self.terminal_log_rewards = torch.zeros(B, dtype=ROLLOUT_DTYPE, device=device)
        self.terminal_stop_mask = torch.zeros(B, dtype=torch.bool, device=device)
        self.terminal_stop_log_pb = torch.zeros(B, dtype=ROLLOUT_DTYPE, device=device)
        self.trajectory_log_pf = torch.zeros(B, dtype=ROLLOUT_DTYPE, device=device)
        self.trajectory_log_pb = torch.zeros(B, dtype=ROLLOUT_DTYPE, device=device)
        self.state_log_flows = torch.zeros((B, T), dtype=ROLLOUT_DTYPE, device=device)
        self.step_log_pf = torch.zeros((B, T), dtype=ROLLOUT_DTYPE, device=device)
        self.step_log_pb = torch.zeros((B, T), dtype=ROLLOUT_DTYPE, device=device)
        self.step_log_shaping = torch.zeros((B, T), dtype=ROLLOUT_DTYPE, device=device)
        self.selected_edge_ids = torch.full((B, T), -1, dtype=torch.long, device=device)
        self.selected_relation_only_logits = torch.zeros(
            (B, T), dtype=ROLLOUT_DTYPE, device=device
        )
        self.selected_final_logits = torch.zeros(
            (B, T), dtype=ROLLOUT_DTYPE, device=device
        )

    def write_step(self, t: int, result: StepResult) -> None:
        self.trajectory_log_pf = self.trajectory_log_pf + result.log_pf
        self.trajectory_log_pb = self.trajectory_log_pb + result.log_pb
        self.step_log_pf[:, t] = result.log_pf
        self.step_log_pb[:, t] = result.log_pb
        self.step_log_shaping[:, t] = result.log_shaping
        self.selected_edge_ids[:, t] = result.chosen_edge_ids
        self.selected_relation_only_logits[:, t] = result.chosen_relation_only_logits
        self.selected_final_logits[:, t] = result.chosen_final_logits

        stop = result.stop_mask
        if stop.any():
            self.is_terminated[stop] = True
            self.terminal_stop_mask[stop] = True
            self.traj_len[stop] = t + 1
            self.terminal_log_rewards[stop] = result.terminal_log_rewards[stop]
            self.terminal_stop_log_pb[stop] = result.terminal_stop_log_pb[stop]


__all__ = ["RolloutAccumulators"]
