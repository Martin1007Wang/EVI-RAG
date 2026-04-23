"""
buffers.py — GFlowNet Rollout 轨迹缓冲区 (极简重构版)

职责：
作为单次 Rollout 过程中的唯一状态容器，负责预分配显存、记录单步转移张量，并管理终止状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .types import StepResult


@dataclass
class RolloutBuffer:
    """
    统一的 Rollout 缓冲区与状态机。
    持有轨迹上的所有 [B, T] 和 [B] 级张量，管理图的终止状态。
    """

    B: int
    T: int  # 通常等于 expand_budget + 1
    device: torch.device

    # ── 控制流与终止状态 [B] ──────────────────────────────────────
    is_terminated: torch.Tensor = field(init=False)
    traj_len: torch.Tensor = field(init=False)  # num_decisions; stopped traj = num_expands + 1
    terminal_log_rewards: torch.Tensor = field(init=False)
    teacher_forced_action_count: torch.Tensor = field(init=False)

    # ── 轨迹级累积量 [B] ──────────────────────────────────────────
    trajectory_log_pf: torch.Tensor = field(init=False)
    trajectory_log_pb: torch.Tensor = field(init=False)

    # ── 步级时间序列 [B, T] ───────────────────────────────────────
    state_log_flows: torch.Tensor = field(init=False)
    step_log_pf: torch.Tensor = field(init=False)
    step_log_pb: torch.Tensor = field(init=False)
    selected_edge_ids: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        B, T, dev = self.B, self.T, self.device
        zeros_B  = lambda: torch.zeros(B, dtype=torch.float32, device=dev)
        zeros_BT = lambda: torch.zeros((B, T), dtype=torch.float32, device=dev)

        # 状态机初始化
        self.is_terminated = torch.zeros(B, dtype=torch.bool, device=dev)
        self.traj_len = torch.zeros(B, dtype=torch.long, device=dev)
        self.terminal_log_rewards = zeros_B()
        self.teacher_forced_action_count = torch.zeros(B, dtype=torch.long, device=dev)

        # 全局概率累积
        self.trajectory_log_pf = zeros_B()
        self.trajectory_log_pb = zeros_B()

        # 时间序列初始化
        self.state_log_flows = zeros_BT()
        self.step_log_pf = zeros_BT()
        self.step_log_pb = zeros_BT()
        self.selected_edge_ids = torch.full((B, T), -1, dtype=torch.long, device=dev)

    def write_step(self, t: int, result: StepResult) -> None:
        """
        原子化写入单步结果。
        包含两大操作：1. 写入活跃图的转移数据；2. 锁存新终止图的奖励与状态。
        """
        active = ~self.is_terminated
        if not active.any():
            return

        # 1. 安全地只为 Active 图累积概率和记录路径 (防止悬空梯度或脏数据)
        self.trajectory_log_pf[active] += result.log_pf[active]
        self.trajectory_log_pb[active] += result.log_pb[active]
        
        self.step_log_pf[active, t] = result.log_pf[active]
        self.step_log_pb[active, t] = result.log_pb[active]
        self.selected_edge_ids[active, t] = result.selected_edge_ids[active]

        # 2. 处理 Stop 事件：仅对“在本步刚刚触发 Stop”的图进行结算
        newly_stopped = result.stop_mask & active
        if newly_stopped.any():
            self.is_terminated[newly_stopped] = True
            self.traj_len[newly_stopped] = t + 1
            self.terminal_log_rewards[newly_stopped] = result.terminal_log_rewards[newly_stopped]

    def finalize_teacher_counts(self, counts: torch.Tensor) -> None:
        """由 Engine 在 Rollout 结束时统一写入 Teacher 统计。"""
        self.teacher_forced_action_count.copy_(counts)


__all__ = ["RolloutBuffer"]
