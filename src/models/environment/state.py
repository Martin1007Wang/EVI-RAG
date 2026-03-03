from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class DynamicAgentState:
    """
    [推演状态] 智能体在时间步 t 的动态游走状态。
    """

    step_t: int
    # 智能体绝对物理坐标 (考虑了 node_ptr 的偏移)
    current_nodes: torch.Tensor  # [B, num_agents_per_graph]
    # 显式流向标签，避免通过外部变量隐式注入方向语义。
    flow_direction: Literal["forward", "backward"]
    # 当前持有的隐状态 (例如 LSTM/GRU 隐藏层，或历史路径向量)
    # 初始时等同于 question_emb 的扩展
    hidden_states: torch.Tensor  # [B, num_agents_per_graph, d]
    # 访问历史与遮罩，防止死循环 (True 表示已访问)
    visited_mask: torch.Tensor  # [B*num_agents_per_graph, N_total]
    # 轨迹记录与结算
    cumulative_rewards: torch.Tensor  # [B, num_agents_per_graph]
    done_mask: torch.Tensor  # [B, num_agents_per_graph], 命中或超步数标记
    # 历史中真实 move 动作计数（不含 STOP），用于 stop_min_steps 等马尔可夫约束。
    num_moves: torch.Tensor  # [B, num_agents_per_graph]
    # 轨迹 token 历史，采用 (node, relation, node, ...) 序列语义。
    # path_token_types: False=node token, True=relation token
    path_token_ids: torch.Tensor | None = None  # [B, num_agents_per_graph, T]
    path_token_types: torch.Tensor | None = None  # [B, num_agents_per_graph, T]
    path_lengths: torch.Tensor | None = None  # [B, num_agents_per_graph]


__all__ = ["DynamicAgentState"]
