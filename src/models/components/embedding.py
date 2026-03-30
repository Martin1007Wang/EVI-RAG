# src/models/components/embedding.py
"""
[系统实体] 策略网络 Backbone
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.nn_init import init_linear_xavier
from src.utils.precision_utils import align_float_input_dtype


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
        embeddings = align_float_input_dtype(embeddings, module=self.norm)
        # 预归一化残差流
        delta = self.drop(self.up(self.act(self.down(self.norm(embeddings)))))
        return embeddings + delta


class EmbeddingBackbone(nn.Module):
    """
    嵌入 Backbone
    职责：
    1. 注入 Adapter 微调信号 (同流形微调)
    2. 投影到隐空间 (跨空间映射，严格保证投影矩阵可导)
    3. 输出供策略层消费的节点/关系/问题表征
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 1024,
        hidden_dim: int = 512,
        use_adapter: bool = True,
        adapter_dim: int = 128,
        adapter_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_adapter = bool(use_adapter)
        adapter_dim = int(adapter_dim)
        adapter_dropout = float(adapter_dropout)
        if self.emb_dim < 1:
            raise ValueError("backbone.embedding_dim must be >= 1.")
        if self.hidden_dim < 1:
            raise ValueError("backbone.hidden_dim must be >= 1.")
        if adapter_dim < 1:
            raise ValueError("backbone.adapter_dim must be >= 1.")
        if adapter_dropout < 0.0 or adapter_dropout >= 1.0:
            raise ValueError("backbone.adapter_dropout must be in [0, 1).")
        if self.use_adapter:
            self.node_adapter = EmbeddingAdapter(
                emb_dim=self.emb_dim,
                adapter_dim=adapter_dim,
                dropout=adapter_dropout,
            )
            self.rel_adapter = EmbeddingAdapter(
                emb_dim=self.emb_dim,
                adapter_dim=adapter_dim,
                dropout=adapter_dropout,
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
        _ = inputs.edge_index, inputs.edge_relations, inputs.num_nodes
        return self.project_features(
            node_features=inputs.node_features,
            relation_features=inputs.relation_features,
            question_embedding=inputs.question_embedding,
            question_context=inputs.question_context,
        )

    def forward(self, inputs: BackboneInput) -> BackboneOutput:
        return self.encode(inputs)

    def project_node_embeddings(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if self.node_adapter is not None:
            node_embeddings = self.node_adapter(node_embeddings)
        node_embeddings = align_float_input_dtype(
            node_embeddings, module=self.node_norm
        )
        return self.node_proj(self.node_norm(node_embeddings))

    def project_relation_embeddings(
        self, relation_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.rel_adapter is not None:
            relation_embeddings = self.rel_adapter(relation_embeddings)
        relation_embeddings = align_float_input_dtype(
            relation_embeddings, module=self.rel_norm
        )
        return self.rel_proj(self.rel_norm(relation_embeddings))

    def project_question_embeddings(self, question_emb: torch.Tensor) -> torch.Tensor:
        question_emb = align_float_input_dtype(question_emb, module=self.q_proj)
        return self.q_proj(question_emb)

    def project_question_context_embeddings(
        self, question_context: torch.Tensor
    ) -> torch.Tensor:
        question_context = align_float_input_dtype(question_context, module=self.q_proj)
        return self.q_proj(question_context)


__all__ = [
    "BackboneInput",
    "BackboneOutput",
    "EmbeddingAdapter",
    "EmbeddingBackbone",
]
