from dataclasses import dataclass
from typing import Literal

import torch

RECENT_NODE_WINDOW = 3


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
    # 当前持有的隐状态；初始时等同于 question_emb 的扩展。
    hidden_states: torch.Tensor  # [B, num_agents_per_graph, d]
    # 最近访问节点窗口（[current, prev1, prev2]），用于禁止短回退
    visited_mask: torch.Tensor  # [B*num_agents_per_graph, window]
    # 轨迹记录与结算
    cumulative_rewards: torch.Tensor  # [B, num_agents_per_graph]
    done_mask: torch.Tensor  # [B, num_agents_per_graph], 命中或超步数标记
    # 历史中真实 move 动作计数（不含 STOP），用于 stop_min_steps 等马尔可夫约束。
    num_moves: torch.Tensor  # [B, num_agents_per_graph]
    # Node-Markov contract: 仅保留当前节点 token，不允许路径历史。
    # path_token_types: False=node token, True=relation token
    path_token_ids: torch.Tensor | None = None  # [B, num_agents_per_graph, T]
    path_token_types: torch.Tensor | None = None  # [B, num_agents_per_graph, T]
    path_lengths: torch.Tensor | None = None  # [B, num_agents_per_graph]


__all__ = ["DynamicAgentState", "RECENT_NODE_WINDOW"]
