from __future__ import annotations
import torch
from torch import nn
from torch_scatter import scatter_sum, scatter_min

from src.data.schema import RetrievalBatch


class RewardModel(nn.Module):
    """
    增量式子图 MDP 联合全局奖励评估模块
    公式: \log R(G_T, y) = U(y, Y^*) - (\lambda_1 |E_T| + \lambda_2 \max(0, K_T - 1))
    全过程 100% GPU 张量化，零 CPU 阻塞。
    """

    def __init__(
        self,
        hit_reward: float = 10.0,  # 命中答案的对数奖励 (U)
        miss_reward: float = -1.0,  # 未命中的惩罚 (U)
        lambda_1: float = 0.1,  # 边数量惩罚系数
        lambda_2: float = 1.0,  # 弱连通分量惩罚系数
        max_cc_iters: int = 20,  # 连通分量标签传播的最大迭代次数 (图直径)
    ):
        super().__init__()
        self.hit_reward = float(hit_reward)
        self.miss_reward = float(miss_reward)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.max_cc_iters = int(max_cc_iters)

    def _compute_weakly_connected_components(
        self,
        num_nodes: int,
        batch_idx: torch.Tensor,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        [黑科技] 纯 GPU 并行连通分量计算 (Label Propagation)
        返回: 每个图的弱连通分量数量 K_T [B]
        """
        device = active_nodes.device
        B = int(batch_idx.max().item()) + 1

        # 1. 初始化标签：每个节点的标签就是它的全局 ID
        # 对于未激活的节点，给一个极大值（num_nodes）使其不参与求极小值
        node_labels = torch.arange(num_nodes, device=device)
        node_labels[~active_nodes] = num_nodes

        # 2. 提取当前子图的有效边，并构建无向图（求弱连通分量必须无向）
        src = edge_index[0][active_edges]
        dst = edge_index[1][active_edges]
        if src.numel() == 0:
            # 如果整批图一条边都没有，K_T 就是 active_nodes 的数量
            return scatter_sum(active_nodes.int(), batch_idx, dim=0, dim_size=B)

        u = torch.cat([src, dst])
        v = torch.cat([dst, src])

        # 3. 标签传播循环 (向邻居广播自己的最小 ID)
        for _ in range(self.max_cc_iters):
            # 将源节点的标签发给目标节点，取最小值
            min_neighbor_labels, _ = scatter_min(node_labels[u], v, dim_size=num_nodes)

            # scatter_min 对没有收到的节点会返回该数据类型的最大值，将其替换为 num_nodes
            min_neighbor_labels[min_neighbor_labels > num_nodes] = num_nodes

            # 更新当前节点标签
            new_labels = torch.min(node_labels, min_neighbor_labels)

            # 如果全量标签不再变化，提前收敛 (通常 3-5 次迭代即可收敛)
            if torch.equal(new_labels, node_labels):
                break
            node_labels = new_labels

        # 4. 统计 K_T：如果一个节点的标签还是它自己，且它是活跃节点，那它就是一个连通分量的 Root
        is_component_root = (
            node_labels == torch.arange(num_nodes, device=device)
        ) & active_nodes

        # 聚合每个图内的连通分量数
        K_T = scatter_sum(is_component_root.int(), batch_idx, dim=0, dim_size=B)
        return K_T

    def forward(
        self,
        base_graph: RetrievalBatch,
        sampled_sinks: torch.Tensor,  # [B] 智能体最后停下的目标节点 y
        active_nodes: torch.Tensor,  # [Total_Nodes]
        active_edges: torch.Tensor,  # [Total_Edges]
    ) -> torch.Tensor:
        """
        计算终止状态的联合 Log Reward
        """
        B = base_graph.num_graphs
        device = base_graph.edge_index.device

        # ==========================================
        # 项 1: 效用收益 U(y, Y*)
        # ==========================================
        # 检查采样到的 y 是否在金标准答案集中
        valid_sink_mask = sampled_sinks >= 0
        is_hit = torch.zeros(B, dtype=torch.bool, device=device)
        if valid_sink_mask.any():
            is_hit[valid_sink_mask] = base_graph.is_target_mask[
                sampled_sinks[valid_sink_mask]
            ]

        # 组装效用分数：命中了拿 hit_reward，没命中拿 miss_reward
        U = torch.where(
            is_hit,
            torch.tensor(self.hit_reward, device=device),
            torch.tensor(self.miss_reward, device=device),
        )

        # ==========================================
        # 项 2: 结构正则化 \Psi(G_T) = \lambda_1 |E_T| + \lambda_2 \max(0, K_T - 1)
        # ==========================================

        # 2a. 计算边数惩罚
        # 取 active_edges 的源节点，看它们属于哪个图，然后聚合
        edge_batch_idx = base_graph.edge_batch
        E_T = scatter_sum(active_edges.int(), edge_batch_idx, dim=0, dim_size=B)  # [B]

        # 2b. 计算弱连通分量数惩罚
        if self.lambda_2 > 0.0:
            K_T = self._compute_weakly_connected_components(
                num_nodes=base_graph.num_nodes,
                batch_idx=base_graph.batch,
                active_nodes=active_nodes,
                active_edges=active_edges,
                edge_index=base_graph.edge_index,
            )
            k_penalty = self.lambda_2 * torch.clamp(K_T - 1, min=0).float()
        else:
            k_penalty = torch.zeros(B, device=device)

        # 组装结构惩罚
        Psi = self.lambda_1 * E_T.float() + k_penalty

        # ==========================================
        # 最终组合
        # ==========================================
        log_reward = U - Psi

        return log_reward


__all__ = ["RewardModel"]
