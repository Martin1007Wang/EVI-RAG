# src/models/components/embedding.py
"""
[系统实体] 策略网络 Backbone
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dtypes import align_float_input_dtype


def _init_linear_xavier(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


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
    node_graph_index: torch.Tensor | None = None


@dataclass(frozen=True)
class BackboneOutput:
    """Encoded graph/question features produced by embedding modules."""

    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    question_context_tokens: torch.Tensor


def _build_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
    ]
    if float(dropout) > 0.0:
        layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(int(hidden_dim), int(output_dim)))
    mlp = nn.Sequential(*layers)
    for module in mlp:
        if isinstance(module, nn.Linear):
            _init_linear_xavier(module)
    return mlp


def _resolve_node_graph_index(
    *,
    node_graph_index: torch.Tensor | None,
    num_nodes: int,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    if node_graph_index is None:
        if int(num_graphs) != 1:
            raise ValueError(
                "Structured backbone encoding requires node_graph_index when num_graphs > 1."
            )
        return torch.zeros((int(num_nodes),), device=device, dtype=torch.long)
    node_graph_index = node_graph_index.to(device=device, dtype=torch.long).view(-1)
    if int(node_graph_index.numel()) != int(num_nodes):
        raise ValueError(
            "node_graph_index must align with num_nodes in BackboneInput. "
            f"Expected {int(num_nodes)}, got {int(node_graph_index.numel())}."
        )
    if int(node_graph_index.numel()) > 0:
        min_graph_index = int(node_graph_index.min().item())
        max_graph_index = int(node_graph_index.max().item())
        if min_graph_index < 0 or max_graph_index >= int(num_graphs):
            raise ValueError(
                "node_graph_index contains out-of-range graph ids in BackboneInput. "
                f"min={min_graph_index} max={max_graph_index} num_graphs={int(num_graphs)}"
            )
    return node_graph_index


def _mean_aggregate_messages(
    *, index: torch.Tensor, values: torch.Tensor, dim_size: int
) -> torch.Tensor:
    aggregated = values.new_zeros((int(dim_size), int(values.size(-1))))
    if int(values.numel()) == 0:
        return aggregated
    aggregated.index_add_(0, index, values)
    counts = values.new_zeros((int(dim_size),))
    counts.index_add_(
        0,
        index,
        torch.ones((int(index.numel()),), device=index.device, dtype=values.dtype),
    )
    return aggregated / counts.clamp_min(1.0).unsqueeze(-1)


class _GraphPropagationLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        dropout: float,
        use_question_conditioning: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_question_conditioning = bool(use_question_conditioning)
        self.relation_gate = nn.Linear(self.hidden_dim, 1)
        update_width = self.hidden_dim * (4 if self.use_question_conditioning else 3)
        self.update_norm = nn.LayerNorm(update_width)
        self.update = _build_mlp(
            input_dim=update_width,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.hidden_dim,
            dropout=float(dropout),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.question_proj = None
        _init_linear_xavier(self.relation_gate)
        if self.use_question_conditioning:
            self.question_proj = nn.Linear(self.hidden_dim, 1)
            _init_linear_xavier(self.question_proj)

    @staticmethod
    def _build_sparse_adjacency(
        *,
        row_index: torch.Tensor,
        col_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        normalization = edge_weight.new_zeros((int(num_nodes),))
        normalization.index_add_(0, row_index, edge_weight)
        normalized_weight = edge_weight / normalization.index_select(
            0, row_index
        ).clamp_min(1e-6)
        adjacency = torch.sparse_coo_tensor(
            torch.stack((row_index, col_index), dim=0),
            normalized_weight,
            size=(int(num_nodes), int(num_nodes)),
            device=edge_weight.device,
            dtype=edge_weight.dtype,
        )
        return adjacency.coalesce()

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        node_graph_index: torch.Tensor,
    ) -> torch.Tensor:
        if int(node_tokens.size(0)) == 0 or int(edge_index.size(1)) == 0:
            return node_tokens
        src_index = edge_index[0].to(device=node_tokens.device, dtype=torch.long)
        dst_index = edge_index[1].to(device=node_tokens.device, dtype=torch.long)
        relation_index = edge_relations.to(device=node_tokens.device, dtype=torch.long)
        relation_tokens = relation_tokens.index_select(0, relation_index)
        relation_tokens = align_float_input_dtype(
            relation_tokens, module=self.relation_gate
        )
        edge_weight_logits = self.relation_gate(relation_tokens).squeeze(-1)
        node_question_tokens = None
        if self.use_question_conditioning:
            assert self.question_proj is not None
            edge_question_tokens = question_tokens.index_select(
                0, node_graph_index.index_select(0, src_index)
            )
            edge_question_tokens = align_float_input_dtype(
                edge_question_tokens, module=self.question_proj
            )
            edge_weight_logits = edge_weight_logits + self.question_proj(
                edge_question_tokens
            ).squeeze(-1)
            node_question_tokens = question_tokens.index_select(0, node_graph_index)
        propagation_dtype = (
            torch.float32
            if node_tokens.dtype in {torch.float16, torch.bfloat16}
            else node_tokens.dtype
        )
        edge_weight = torch.sigmoid(edge_weight_logits.to(dtype=torch.float32)).to(
            dtype=propagation_dtype
        )
        incoming_adj = self._build_sparse_adjacency(
            row_index=dst_index,
            col_index=src_index,
            edge_weight=edge_weight,
            num_nodes=int(node_tokens.size(0)),
        )
        outgoing_adj = self._build_sparse_adjacency(
            row_index=src_index,
            col_index=dst_index,
            edge_weight=edge_weight,
            num_nodes=int(node_tokens.size(0)),
        )
        with torch.autocast(device_type=node_tokens.device.type, enabled=False):
            propagation_nodes = node_tokens.to(dtype=propagation_dtype)
            incoming_pool = torch.sparse.mm(incoming_adj, propagation_nodes)
            outgoing_pool = torch.sparse.mm(outgoing_adj, propagation_nodes)
        if incoming_pool.dtype != node_tokens.dtype:
            incoming_pool = incoming_pool.to(dtype=node_tokens.dtype)
            outgoing_pool = outgoing_pool.to(dtype=node_tokens.dtype)
        update_inputs = [node_tokens, incoming_pool, outgoing_pool]
        if node_question_tokens is not None:
            update_inputs.append(node_question_tokens)
        stacked_inputs = torch.cat(update_inputs, dim=-1)
        stacked_inputs = align_float_input_dtype(
            stacked_inputs, module=self.update_norm
        )
        stacked_inputs = self.update_norm(stacked_inputs)
        stacked_inputs = align_float_input_dtype(stacked_inputs, module=self.update[0])
        delta = self.update(stacked_inputs)
        return node_tokens + self.dropout(delta)


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

        _init_linear_xavier(self.down)
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
        gnn_num_layers: int = 0,
        gnn_dropout: float = 0.1,
        gnn_use_question_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.emb_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_adapter = bool(use_adapter)
        self.gnn_num_layers = int(gnn_num_layers)
        self.gnn_use_question_conditioning = bool(gnn_use_question_conditioning)
        adapter_dim = int(adapter_dim)
        adapter_dropout = float(adapter_dropout)
        gnn_dropout = float(gnn_dropout)
        if self.emb_dim < 1:
            raise ValueError("backbone.embedding_dim must be >= 1.")
        if self.hidden_dim < 1:
            raise ValueError("backbone.hidden_dim must be >= 1.")
        if adapter_dim < 1:
            raise ValueError("backbone.adapter_dim must be >= 1.")
        if adapter_dropout < 0.0 or adapter_dropout >= 1.0:
            raise ValueError("backbone.adapter_dropout must be in [0, 1).")
        if self.gnn_num_layers < 0:
            raise ValueError("backbone.gnn_num_layers must be >= 0.")
        if gnn_dropout < 0.0 or gnn_dropout >= 1.0:
            raise ValueError("backbone.gnn_dropout must be in [0, 1).")
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
        _init_linear_xavier(self.node_proj)
        _init_linear_xavier(self.rel_proj)
        _init_linear_xavier(self.q_proj)
        self.graph_layers = nn.ModuleList(
            _GraphPropagationLayer(
                hidden_dim=self.hidden_dim,
                dropout=gnn_dropout,
                use_question_conditioning=self.gnn_use_question_conditioning,
            )
            for _ in range(self.gnn_num_layers)
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
        encoded = self.project_features(
            node_features=inputs.node_features,
            relation_features=inputs.relation_features,
            question_embedding=inputs.question_embedding,
            question_context=inputs.question_context,
        )
        if self.gnn_num_layers <= 0:
            return encoded
        node_graph_index = _resolve_node_graph_index(
            node_graph_index=inputs.node_graph_index,
            num_nodes=int(inputs.num_nodes),
            num_graphs=int(encoded.question_tokens.size(0)),
            device=encoded.node_tokens.device,
        )
        node_tokens = encoded.node_tokens
        for layer in self.graph_layers:
            node_tokens = layer(
                node_tokens=node_tokens,
                relation_tokens=encoded.relation_tokens,
                question_tokens=encoded.question_tokens,
                edge_index=inputs.edge_index,
                edge_relations=inputs.edge_relations,
                node_graph_index=node_graph_index,
            )
        return BackboneOutput(
            node_tokens=node_tokens,
            relation_tokens=encoded.relation_tokens,
            question_tokens=encoded.question_tokens,
            question_context_tokens=encoded.question_context_tokens,
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
