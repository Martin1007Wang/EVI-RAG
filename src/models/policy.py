from __future__ import annotations
import torch
from typing import Any
from torch import nn
from torch_geometric.nn import global_mean_pool

from .modules.backbone import GNNBackbone
from .modules.heads import ActionHead, FlowHead
from src.data.schema import RetrievalBatch


class Policy(nn.Module):
    """
    增量式子图扩张 MDP 策略网络
    """

    def __init__(self, backbone_cfg: dict[str, Any], hidden_dim: int = 512):
        super().__init__()
        self.backbone = GNNBackbone(**backbone_cfg)
        self.flow_head = FlowHead(hidden_dim=hidden_dim)
        self.action_head = ActionHead(hidden_dim=hidden_dim)

    def forward(
        self,
        batch: RetrievalBatch,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 1. 骨干网络：提取被掩码后的拓扑特征
        node_h, edge_relation_h, _ = self.backbone(batch, active_edges=active_edges)

        # 2. 状态表征：严格定义为当前已探索子图的聚合 S_t
        active_node_h = node_h[active_nodes]
        active_batch_idx = batch.batch[active_nodes]
        subgraph_h = global_mean_pool(
            active_node_h, active_batch_idx, size=batch.num_graphs
        )

        # 3. 状态流动量打分
        log_flows = self.flow_head(subgraph_h)

        # 4. 动作空间独立打分
        action_logits = self.action_head(
            node_h=node_h,
            edge_relation_h=edge_relation_h,
            subgraph_h=subgraph_h,
            batch_index=batch.batch,
            edge_batch_index=batch.edge_batch,
        )

        return log_flows, action_logits
