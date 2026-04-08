from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from src.utils.reward_utils import (
    build_anchor_induced_edge_mask,
    per_graph_mask_count,
    prune_to_protected_core,
)


class RewardModel(nn.Module):
    """
    经过数学修正的子图级联合奖励模型。
    所有超参数在 Linear 空间定义，最终输出 Log Reward 供 TB Loss 使用。
    """

    def __init__(
        self,
        hit_reward_base: float = 10.0,  # 基础命中奖励 (Linear)
        hit_reward_max: float = 100.0,  # 找齐所有答案的最大奖励 (Linear)
        miss_reward: float = 1e-4,  # 未命中的保底奖励 (Linear, 必须 > 0)
        penalty_step: float = 0.95,  # 每新增一条边的惩罚衰减因子 (乘性, 0~1)
        penalty_dead: float = 0.80,  # 每新增一条悬空边的惩罚衰减因子 (乘性, 0~1)
        max_prune_iters: int = 64,
    ):
        super().__init__()
        # 强制类型转换与断言，防止负数进入 Linear 空间导致 log(负数) 崩盘
        assert miss_reward > 0, "miss_reward must be strictly positive."
        self.hit_reward_base = float(hit_reward_base)
        self.hit_reward_max = float(hit_reward_max)
        self.miss_reward = float(miss_reward)

        # 将乘性衰减因子转换为 Log 空间的加性惩罚 (更稳定)
        # 例如: penalty_step = 0.95 -> log(0.95) ≈ -0.051
        self.log_penalty_step = torch.log(torch.tensor(float(penalty_step)))
        self.log_penalty_dead = torch.log(torch.tensor(float(penalty_dead)))

        self.max_prune_iters = int(max_prune_iters)

    def forward(
        self,
        base_graph: RetrievalBatch,
        active_nodes: torch.Tensor,  # [Total_Nodes]
        active_edges: torch.Tensor,  # [Total_Edges]
        root_active_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        计算终止子图的联合 Log Reward.
        """
        B = base_graph.num_graphs
        device = base_graph.edge_index.device

        if root_active_edges is None:
            root_active_edges = build_anchor_induced_edge_mask(
                base_graph.edge_index,
                base_graph.is_anchor_mask,
            )

        # -------------------------------------------------------------
        # 1. 语义召回：连续比例奖励 (Proportional Reward)
        # -------------------------------------------------------------
        target_mask = base_graph.is_target_mask
        active_gold = active_nodes & target_mask

        hits_per_graph = per_graph_mask_count(active_gold, base_graph.batch, B, dtype=torch.float32)
        gold_per_graph = per_graph_mask_count(target_mask, base_graph.batch, B, dtype=torch.float32)

        # 计算召回比例 [0.0, 1.0]
        recall_ratio = hits_per_graph / gold_per_graph.clamp(min=1.0)

        # 构建基础 Log Reward
        # 没命中: log(miss_reward)
        # 命中: log( base + (max - base) * recall_ratio )
        linear_recall_reward = torch.where(
            recall_ratio > 0,
            self.hit_reward_base + (self.hit_reward_max - self.hit_reward_base) * recall_ratio,
            torch.tensor(self.miss_reward, device=device),
        )
        base_log_reward = torch.log(linear_recall_reward)

        # -------------------------------------------------------------
        # 2. 结构惩罚计算 (在 Log 空间做加法，等价于在 Linear 空间做乘法衰减)
        # -------------------------------------------------------------
        edge_batch_idx = base_graph.edge_batch
        added_edges = active_edges & ~root_active_edges

        # 2.1 步数惩罚 (鼓励最短路径)
        added_edge_count = scatter_sum(added_edges.int(), edge_batch_idx, dim=0, dim_size=B)
        log_sparse_penalty = added_edge_count.float() * self.log_penalty_step.to(device)

        # 2.2 悬空边惩罚 (严厉打击无效探索)
        protected_nodes = base_graph.is_anchor_mask | active_gold
        _core_nodes, core_edges = prune_to_protected_core(
            active_nodes=active_nodes,
            active_edges=active_edges,
            edge_index=base_graph.edge_index,
            protected_nodes=protected_nodes,
            max_iters=self.max_prune_iters,
        )
        dangling_added_edges = added_edges & ~core_edges
        dead_edge_count = scatter_sum(dangling_added_edges.int(), edge_batch_idx, dim=0, dim_size=B)
        log_dead_penalty = dead_edge_count.float() * self.log_penalty_dead.to(device)

        # -------------------------------------------------------------
        # 3. 最终组合
        # -------------------------------------------------------------
        # Log 空间加法: log(R_base * (0.95^step) * (0.8^dead))
        final_log_reward = base_log_reward + log_sparse_penalty + log_dead_penalty

        return final_log_reward


__all__ = ["RewardModel"]
