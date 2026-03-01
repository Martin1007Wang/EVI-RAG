# src/models/components/backbone.py
"""
[系统实体] 策略网络 Backbone
"""
from __future__ import annotations
import torch
from torch import nn

from src.models.configs.policy import BackboneConfig
from .gnn import RelationalGNNLayer


def _init_linear(layer: nn.Linear) -> None:
    """Xavier 均匀初始化，防止梯度弥散"""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


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

        _init_linear(self.down)
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
        self.use_film = bool(config.use_film)
        if self.use_adapter:
            self.node_adapter = EmbeddingAdapter(
                emb_dim=self.emb_dim, adapter_dim=config.adapter_dim, dropout=config.adapter_dropout
            )
            self.rel_adapter = EmbeddingAdapter(
                emb_dim=self.emb_dim, adapter_dim=config.adapter_dim, dropout=config.adapter_dropout
            )
        else:
            self.node_adapter = self.rel_adapter = None

        if self.emb_dim == self.hidden_dim:
            self.node_proj = nn.Identity()
            self.rel_proj = nn.Identity()
            self.q_proj = nn.Identity()
            self.node_norm = nn.LayerNorm(self.emb_dim)
            self.rel_norm = nn.LayerNorm(self.emb_dim)
        else:
            self.node_norm = nn.LayerNorm(self.emb_dim)
            self.rel_norm = nn.LayerNorm(self.emb_dim)
            self.node_proj = nn.Linear(self.emb_dim, self.hidden_dim)
            self.rel_proj = nn.Linear(self.emb_dim, self.hidden_dim)
            self.q_proj = nn.Linear(self.emb_dim, self.hidden_dim)
            _init_linear(self.node_proj)
            _init_linear(self.rel_proj)
            _init_linear(self.q_proj)

        self.gnn_layers = nn.ModuleList(
            [
                RelationalGNNLayer(hidden_dim=self.hidden_dim, dropout=config.gnn_dropout)
                for _ in range(self.num_gnn_layers)
            ]
        )
        if self.use_film and self.num_gnn_layers > 0:
            self.film_norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_gnn_layers)]
            )
            self.film_gamma = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_gnn_layers)]
            )
            self.film_beta = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_gnn_layers)]
            )
            for gamma, beta in zip(self.film_gamma, self.film_beta):
                nn.init.zeros_(gamma.weight)
                nn.init.ones_(gamma.bias)
                nn.init.zeros_(beta.weight)
                nn.init.zeros_(beta.bias)
        else:
            self.film_norms = None
            self.film_gamma = None
            self.film_beta = None

    def project_node_embeddings(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if self.node_adapter is not None:
            node_embeddings = self.node_adapter(node_embeddings)
        return self.node_proj(self.node_norm(node_embeddings))

    def project_relation_embeddings(self, relation_embeddings: torch.Tensor) -> torch.Tensor:
        if self.rel_adapter is not None:
            relation_embeddings = self.rel_adapter(relation_embeddings)
        return self.rel_proj(self.rel_norm(relation_embeddings))

    def project_question_embeddings(self, question_emb: torch.Tensor) -> torch.Tensor:
        return self.q_proj(question_emb)

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
        """
        # [系统级修正] 使用数值变量进行逻辑判断
        if self.num_gnn_layers == 0:
            return node_tokens
        if self.use_film:
            if question_tokens is None or node_batch is None:
                raise ValueError("FiLM conditioning requires question_tokens and node_batch.")
            q_node = question_tokens.index_select(0, node_batch)
        else:
            q_node = None

        out = node_tokens
        for idx, layer in enumerate(self.gnn_layers):
            out = layer(
                node_tokens=out,
                relation_tokens=relation_tokens,
                edge_index=edge_index,
                edge_relations=edge_relations,  # 传递给底层 GNN 进行异构计算
                num_nodes=num_nodes,
            )
            if self.use_film and q_node is not None:
                gamma = self.film_gamma[idx](q_node)
                beta = self.film_beta[idx](q_node)
                out = gamma * self.film_norms[idx](out) + beta
        return out


__all__ = ["EmbeddingBackbone", "EmbeddingAdapter"]
