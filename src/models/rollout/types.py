"""
rollout/types.py — Rollout 对外数据契约（纯 DTO，无业务逻辑）
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .buffers import RolloutBuffer

def _dc_to(obj, device: torch.device):
    if obj is None:
        return None
    updates = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, torch.Tensor):
            updates[f.name] = val.to(device)
        elif dataclasses.is_dataclass(val) and not isinstance(val, type):
            updates[f.name] = _dc_to(val, device)
    return dataclasses.replace(obj, **updates)

@dataclass(frozen=True)
class StepResult:
    log_pf: torch.Tensor
    log_pb: torch.Tensor
    stop_mask: torch.Tensor
    terminal_log_rewards: torch.Tensor
    selected_edge_ids: torch.Tensor

@dataclass(frozen=True)
class TrajectoryStats:
    root_log_z: torch.Tensor
    traj_len: torch.Tensor
    trajectory_log_pf: torch.Tensor
    trajectory_log_pb: torch.Tensor
    terminal_log_rewards: torch.Tensor
    teacher_forced_action_count: torch.Tensor

@dataclass(frozen=True)
class StepTraces:
    state_log_flows: torch.Tensor
    step_log_pf: torch.Tensor
    step_log_pb: torch.Tensor
    selected_edge_ids: torch.Tensor

@dataclass(frozen=True)
class RolloutBatch:
    stats: TrajectoryStats
    traces: StepTraces

    @classmethod
    def from_run(
        cls,
        *,
        buffer: "RolloutBuffer",
        root_log_z: torch.Tensor,
    ) -> RolloutBatch:
        return cls(
            stats=TrajectoryStats(
                root_log_z=root_log_z,
                traj_len=buffer.traj_len,
                trajectory_log_pf=buffer.trajectory_log_pf,
                trajectory_log_pb=buffer.trajectory_log_pb,
                terminal_log_rewards=buffer.terminal_log_rewards,
                teacher_forced_action_count=buffer.teacher_forced_action_count,
            ),
            traces=StepTraces(
                state_log_flows=buffer.state_log_flows,
                step_log_pf=buffer.step_log_pf,
                step_log_pb=buffer.step_log_pb,
                selected_edge_ids=buffer.selected_edge_ids,
            ),
        )

__all__ = ["StepResult", "TrajectoryStats", "StepTraces", "RolloutBatch"]