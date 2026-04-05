from __future__ import annotations
from typing import Any
import torch
from torch import nn


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(max(num_layers - 1, 0)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class FlowHead(nn.Module):
    """
    计算子图状态流动量 log F(S_t)
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mlp = _build_mlp(
            input_dim=hidden_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, subgraph_h: torch.Tensor) -> torch.Tensor:
        # subgraph_h: [Num_Graphs, hidden_dim]
        return self.mlp(subgraph_h).squeeze(-1)  # [Num_Graphs]


class ActionHead(nn.Module):
    """
    动作策略头 (Forward Policy Head)
    将动作空间解耦为：
    1. 宏观决策：Expand vs Sink
    2. 若 Expand，三步独立打分：u, r, v
    3. 若 Sink，一步打分：y
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # 1. 宏观决策: P_type (Expand, Sink)
        self.type_scorer = _build_mlp(
            input_dim=hidden_dim,
            output_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # 2. 扩张三部曲 (Expand: u, r, v)
        # 节点上下文维度 = node_h + subgraph_h = hidden_dim * 2
        self.expand_u_scorer = _build_mlp(
            input_dim=hidden_dim * 2,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.expand_r_scorer = _build_mlp(
            input_dim=hidden_dim * 2,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.expand_v_scorer = _build_mlp(
            input_dim=hidden_dim * 2,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # 3. 沉汇打分 (Sink: y)
        self.sink_scorer = _build_mlp(
            input_dim=hidden_dim * 2,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        node_h: torch.Tensor,
        edge_relation_h: torch.Tensor,
        subgraph_h: torch.Tensor,
        batch_index: torch.Tensor,
        edge_batch_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 将图的全局状态 S_t 广播到每个节点，形成条件语义
        node_ctx = torch.cat([node_h, subgraph_h[batch_index]], dim=-1)

        # 关系特征已经在 collate 阶段按边对齐，这里直接做逐边条件打分。
        edge_rel_ctx = torch.cat(
            [edge_relation_h, subgraph_h[edge_batch_index]],
            dim=-1,
        )

        # 返回所有未归一化的原始潜力分数 (Logits)
        return {
            "type_logits": self.type_scorer(subgraph_h),  # [Num_Graphs, 2]
            "expand_u_logits": self.expand_u_scorer(node_ctx).squeeze(
                -1
            ),  # [Total_Nodes]
            "expand_edge_rel_logits": self.expand_r_scorer(edge_rel_ctx).squeeze(
                -1
            ),  # [Total_Edges]
            "expand_v_logits": self.expand_v_scorer(node_ctx).squeeze(
                -1
            ),  # [Total_Nodes]
            "sink_logits": self.sink_scorer(node_ctx).squeeze(-1),  # [Total_Nodes]
        }
