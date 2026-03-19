# src/models/components/embedding.py
"""
[系统实体] 策略网络 Backbone
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.configs import BackboneConfig
from src.utils.nn_init import init_linear_xavier

from .gnn import RelationalGNNLayer


@dataclass(frozen=True)
class BackboneInput:
    """Structured inputs for graph/question encoding modules."""

    node_features: torch.Tensor
    relation_features: torch.Tensor
    question_embedding: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    num_nodes: int
    question_context: torch.Tensor | None = None


@dataclass(frozen=True)
class BackboneOutput:
    """Encoded graph/question features produced by embedding modules."""

    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    question_context_tokens: torch.Tensor


class EmbeddingAdapter(nn.Module):
    """
    嵌入适配器 (低秩残差注入)
    数学本质：x' = x + W_up(GELU(W_down(Norm(x))))
    保证微调后的特征依然停留在原有的预训练流形空间中。
    """

    def __init__(self, *, emb_dim: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.down = nn.Linear(emb_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, emb_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        init_linear_xavier(self.down)
        # 向上投影初始化为 0，确保训练初期表现等价于恒等映射 (Identity)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings
        # 预归一化残差流
        delta = self.drop(self.up(self.act(self.down(self.norm(embeddings)))))
        return embeddings + delta


class EmbeddingBackbone(nn.Module):
    """
    嵌入 Backbone
    职责：
    1. 注入 Adapter 微调信号 (同流形微调)
    2. 投影到隐空间 (跨空间映射，严格保证投影矩阵可导)
    3. 执行关系驱动的 GNN 编码
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.emb_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_gnn_layers = config.gnn_layers
        self.use_adapter = config.use_adapter
        if self.use_adapter:
            self.node_adapter = EmbeddingAdapter(
                emb_dim=self.emb_dim,
                adapter_dim=config.adapter_dim,
                dropout=config.adapter_dropout,
            )
            self.rel_adapter = EmbeddingAdapter(
                emb_dim=self.emb_dim,
                adapter_dim=config.adapter_dim,
                dropout=config.adapter_dropout,
            )
        else:
            self.node_adapter = self.rel_adapter = None

        self.node_norm = nn.LayerNorm(self.emb_dim)
        self.rel_norm = nn.LayerNorm(self.emb_dim)
        self.node_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.rel_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.q_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        init_linear_xavier(self.node_proj)
        init_linear_xavier(self.rel_proj)
        init_linear_xavier(self.q_proj)

        self.gnn_layers = nn.ModuleList(
            [
                RelationalGNNLayer(
                    hidden_dim=self.hidden_dim, dropout=config.gnn_dropout
                )
                for _ in range(self.num_gnn_layers)
            ]
        )

    def project_features(
        self,
        *,
        node_features: torch.Tensor,
        relation_features: torch.Tensor,
        question_embedding: torch.Tensor,
        question_context: torch.Tensor | None = None,
    ) -> BackboneOutput:
        if question_context is None:
            question_context = question_embedding.unsqueeze(1)
        return BackboneOutput(
            node_tokens=self.project_node_embeddings(node_features),
            relation_tokens=self.project_relation_embeddings(relation_features),
            question_tokens=self.project_question_embeddings(question_embedding),
            question_context_tokens=self.project_question_context_embeddings(
                question_context
            ),
        )

    def encode(self, inputs: BackboneInput) -> BackboneOutput:
        projected = self.project_features(
            node_features=inputs.node_features,
            relation_features=inputs.relation_features,
            question_embedding=inputs.question_embedding,
            question_context=inputs.question_context,
        )
        node_tokens = self.encode_graph(
            node_tokens=projected.node_tokens,
            relation_tokens=projected.relation_tokens,
            edge_index=inputs.edge_index,
            edge_relations=inputs.edge_relations,
            num_nodes=int(inputs.num_nodes),
        )
        return BackboneOutput(
            node_tokens=node_tokens,
            relation_tokens=projected.relation_tokens,
            question_tokens=projected.question_tokens,
            question_context_tokens=projected.question_context_tokens,
        )

    def forward(self, inputs: BackboneInput) -> BackboneOutput:
        return self.encode(inputs)

    def project_node_embeddings(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if self.node_adapter is not None:
            node_embeddings = self.node_adapter(node_embeddings)
        return self.node_proj(self.node_norm(node_embeddings))

    def project_relation_embeddings(
        self, relation_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.rel_adapter is not None:
            relation_embeddings = self.rel_adapter(relation_embeddings)
        return self.rel_proj(self.rel_norm(relation_embeddings))

    def project_question_embeddings(self, question_emb: torch.Tensor) -> torch.Tensor:
        return self.q_proj(question_emb)

    def project_question_context_embeddings(
        self, question_context: torch.Tensor
    ) -> torch.Tensor:
        return self.q_proj(question_context)

    def encode_graph(
        self,
        *,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,  # <-- 物理链路修补：必须传入关系 ID
        num_nodes: int,
        question_tokens: torch.Tensor | None = None,
        node_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        [数学实体] 图消息传递流

        `question_tokens` 与 `node_batch` 仅为兼容旧调用方保留，当前实现不会使用。
        """
        del question_tokens, node_batch
        # [系统级修正] 使用数值变量进行逻辑判断
        if self.num_gnn_layers == 0:
            return node_tokens

        out = node_tokens
        for layer in self.gnn_layers:
            out = layer(
                node_tokens=out,
                relation_tokens=relation_tokens,
                edge_index=edge_index,
                edge_relations=edge_relations,  # 传递给底层 GNN 进行异构计算
                num_nodes=num_nodes,
            )
        return out


__all__ = [
    "BackboneInput",
    "BackboneOutput",
    "EmbeddingAdapter",
    "EmbeddingBackbone",
]
