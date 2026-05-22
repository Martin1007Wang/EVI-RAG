from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .action import StepAction, TERMINAL_EDGE_ID

NO_STEP = -1


@dataclass(slots=True)
class RolloutTape:
    R: int
    T: int
    device: torch.device
    dtype: torch.dtype = torch.float32

    selected_edge_ids: torch.Tensor = field(init=False)
    policy_action_log_prob: torch.Tensor = field(init=False)
    behavior_action_log_prob: torch.Tensor = field(init=False)
    terminal_step: torch.Tensor = field(init=False)
    stop_reason: torch.Tensor = field(init=False)
    is_stopped: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.R = int(self.R)
        self.T = int(self.T)
        rollout_shape = (self.R, self.T)
        row_shape = (self.R,)
        self.selected_edge_ids = torch.full(
            rollout_shape,
            TERMINAL_EDGE_ID,
            dtype=torch.long,
            device=self.device,
        )
        self.policy_action_log_prob = torch.zeros(
            rollout_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.behavior_action_log_prob = torch.zeros(
            rollout_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.terminal_step = torch.full(
            row_shape,
            NO_STEP,
            dtype=torch.long,
            device=self.device,
        )
        self.stop_reason = torch.full(
            row_shape,
            NO_STEP,
            dtype=torch.long,
            device=self.device,
        )
        self.is_stopped = torch.zeros(
            row_shape,
            dtype=torch.bool,
            device=self.device,
        )

    def write(
        self,
        t: int,
        action: StepAction,
    ) -> None:
        step = int(t)
        rows = action.row_ids.to(device=self.device, dtype=torch.long)
        if rows.numel() == 0:
            return

        self.selected_edge_ids[rows, step] = action.edge_ids.to(
            device=self.device,
            dtype=torch.long,
        )
        self.policy_action_log_prob[rows, step] = action.policy_log_prob.to(
            device=self.device,
            dtype=self.dtype,
        )
        self.behavior_action_log_prob[rows, step] = action.behavior_log_prob.to(
            device=self.device,
            dtype=self.dtype,
        )

        terminal_mask = action.terminal_mask
        terminal_rows = rows[terminal_mask]
        if terminal_rows.numel() == 0:
            return

        self.terminal_step[terminal_rows] = step
        self.stop_reason[terminal_rows] = action.stop_reason[terminal_mask].to(
            device=self.device,
            dtype=torch.long,
        )
        self.is_stopped[terminal_rows] = True


__all__ = [
    "NO_STEP",
    "RolloutTape",
]
