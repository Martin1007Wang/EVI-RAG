from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .action import StepAction, STOP_EDGE_ID

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
    stop_step: torch.Tensor = field(init=False)
    forced_stop: torch.Tensor = field(init=False)
    is_stopped: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.R = int(self.R)
        self.T = int(self.T)
        self.selected_edge_ids = torch.full(
            (self.R, self.T),
            STOP_EDGE_ID,
            dtype=torch.long,
            device=self.device,
        )
        self.policy_action_log_prob = torch.zeros(
            (self.R, self.T),
            dtype=self.dtype,
            device=self.device,
        )
        self.behavior_action_log_prob = torch.zeros(
            (self.R, self.T),
            dtype=self.dtype,
            device=self.device,
        )
        self.stop_step = torch.full(
            (self.R,),
            NO_STEP,
            dtype=torch.long,
            device=self.device,
        )
        self.forced_stop = torch.zeros(
            self.R,
            dtype=torch.bool,
            device=self.device,
        )
        self.is_stopped = torch.zeros(
            self.R,
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

        stop_mask = action.stop_mask
        stop_rows = rows[stop_mask]
        if stop_rows.numel() == 0:
            return

        self.stop_step[stop_rows] = step
        self.forced_stop[stop_rows] = action.forced[stop_mask].to(
            device=self.device,
            dtype=torch.bool,
        )
        self.is_stopped[stop_rows] = True


__all__ = [
    "NO_STEP",
    "RolloutTape",
]
