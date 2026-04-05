from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.data.schema import RetrievalBatch


def _init_xavier(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    mlp = nn.Sequential(*layers)
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            _init_xavier(m)
    return mlp


def _module_float_dtype(module: nn.Module) -> Optional[torch.dtype]:
    """Return the dtype of the first floating-point parameter/buffer (non-recursive)."""
    for t in module.parameters(recurse=False):
        if torch.is_floating_point(t):
            return t.dtype
    for t in module.buffers(recurse=False):
        if torch.is_floating_point(t):
            return t.dtype
    return None


def build_sparse_adjacency(
    row_index: torch.Tensor,
    col_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Build a row-normalised sparse COO adjacency matrix.

    Normalisation: each row is divided by the sum of its outgoing weights
    (clamped to ≥1e-6 to avoid division by zero).
    """
    norm = edge_weight.new_zeros(num_nodes).index_add_(0, row_index, edge_weight)
    w = edge_weight / norm.index_select(0, row_index).clamp_min_(1e-6)
    adj = torch.sparse_coo_tensor(
        torch.stack([row_index, col_index]),
        w,
        size=(num_nodes, num_nodes),
        device=edge_weight.device,
        dtype=edge_weight.dtype,
    )
    return adj.coalesce()


# ---------------------------------------------------------------------------
# EmbeddingAdapter
# ---------------------------------------------------------------------------


class EmbeddingAdapter(nn.Module):
    """
    Low-rank residual adapter: x' = x + W_up(GELU(W_down(Norm(x)))).

    ``up`` is zero-initialised so the adapter is an identity map at the start
    of training, keeping activations on the pre-trained manifold.
    """

    def __init__(self, *, emb_dim: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.down = nn.Linear(emb_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, emb_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        _init_xavier(self.down)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

        # Cache target dtype to avoid scanning parameters on every forward call.
        self._target_dtype: Optional[torch.dtype] = None

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        if self._target_dtype is None:
            self._target_dtype = _module_float_dtype(self.norm)
        if self._target_dtype is not None and x.dtype != self._target_dtype:
            return x.to(self._target_dtype)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        x = self._cast(x)
        return x + self.drop(self.up(self.act(self.down(self.norm(x)))))


# ---------------------------------------------------------------------------
# Graph propagation layer
# ---------------------------------------------------------------------------


class _GNNLayer(nn.Module):
    """
    Single GNN message-passing layer with learned, relation-gated edge weights.

    Aggregates incoming and outgoing neighbour features via sparse matrix
    multiplication, then updates each node with a residual MLP.  Optionally
    conditions edge weights on the per-graph question embedding.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        dropout: float,
        use_question_conditioning: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_question_conditioning = use_question_conditioning

        self.relation_gate = nn.Linear(hidden_dim, 1)
        _init_xavier(self.relation_gate)

        self.question_proj: Optional[nn.Linear] = None
        if use_question_conditioning:
            self.question_proj = nn.Linear(hidden_dim, 1)
            _init_xavier(self.question_proj)

        update_width = hidden_dim * (4 if use_question_conditioning else 3)
        self.update_norm = nn.LayerNorm(update_width)
        self.update = _build_mlp(
            input_dim=update_width,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,  # [N, hidden_dim]
        edge_relation_tokens: torch.Tensor,  # [E, hidden_dim]
        question_tokens: torch.Tensor,  # [G, hidden_dim]  (one per graph in batch)
        edge_index: torch.Tensor,  # [2, E]
        node_graph_index: torch.Tensor,  # [N]  PyG batch.batch
    ) -> torch.Tensor:
        num_nodes = node_tokens.size(0)
        if num_nodes == 0 or edge_index.size(1) == 0:
            return node_tokens

        device = node_tokens.device
        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)

        edge_rel_tok = edge_relation_tokens
        if edge_rel_tok.dtype != self.relation_gate.weight.dtype:
            edge_rel_tok = edge_rel_tok.to(self.relation_gate.weight.dtype)

        logits = self.relation_gate(edge_rel_tok).squeeze(-1)  # [E]

        node_q_tok: Optional[torch.Tensor] = None
        if self.use_question_conditioning:
            assert self.question_proj is not None
            edge_q = question_tokens.index_select(
                0, node_graph_index.index_select(0, src)
            )
            if edge_q.dtype != self.question_proj.weight.dtype:
                edge_q = edge_q.to(self.question_proj.weight.dtype)
            logits = logits + self.question_proj(edge_q).squeeze(-1)
            node_q_tok = question_tokens.index_select(0, node_graph_index)

        # Compute edge weights in float32 for numerical stability,
        # then cast back to match node features.
        edge_weight = torch.sigmoid(logits.float()).to(node_tokens.dtype)

        in_adj = build_sparse_adjacency(dst, src, edge_weight, num_nodes)
        out_adj = build_sparse_adjacency(src, dst, edge_weight, num_nodes)

        # Sparse mm requires float32 when node features are fp16/bf16.
        prop_dtype = (
            torch.float32
            if node_tokens.dtype in {torch.float16, torch.bfloat16}
            else node_tokens.dtype
        )
        nodes_fp32 = node_tokens.to(prop_dtype)
        incoming = torch.sparse.mm(in_adj.to(prop_dtype), nodes_fp32).to(
            node_tokens.dtype
        )
        outgoing = torch.sparse.mm(out_adj.to(prop_dtype), nodes_fp32).to(
            node_tokens.dtype
        )

        parts = [node_tokens, incoming, outgoing]
        if node_q_tok is not None:
            parts.append(node_q_tok)

        combined = torch.cat(parts, dim=-1)
        norm_dtype = self.update_norm.weight.dtype
        if combined.dtype != norm_dtype:
            combined = combined.to(norm_dtype)
        combined = self.update_norm(combined)

        lin_dtype = self.update[0].weight.dtype
        if combined.dtype != lin_dtype:
            combined = combined.to(lin_dtype)

        return node_tokens + self.dropout(self.update(combined))


# ---------------------------------------------------------------------------
# GNNBackbone
# ---------------------------------------------------------------------------


class GNNBackbone(nn.Module):
    """
    Graph feature encoder for the GFlowNet retrieval agent.

    Consumes a ``RetrievalBatch`` produced by the data module and an optional
    boolean or index mask ``active_edges`` that restricts the visible subgraph
    to the MDP's current state (no information leakage across time steps).

    Returns
    -------
    node_h : Tensor  [TotalNodes, hidden_dim]
    edge_relation_h : Tensor  [TotalEdges, hidden_dim]
    q_h    : Tensor  [NumGraphs, hidden_dim]
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
        self.emb_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gnn_num_layers = gnn_num_layers

        if use_adapter:
            self.node_adapter: Optional[EmbeddingAdapter] = EmbeddingAdapter(
                emb_dim=embedding_dim, adapter_dim=adapter_dim, dropout=adapter_dropout
            )
            self.rel_adapter: Optional[EmbeddingAdapter] = EmbeddingAdapter(
                emb_dim=embedding_dim, adapter_dim=adapter_dim, dropout=adapter_dropout
            )
        else:
            self.node_adapter = None
            self.rel_adapter = None

        self.node_norm = nn.LayerNorm(embedding_dim)
        self.rel_norm = nn.LayerNorm(embedding_dim)
        self.node_proj = nn.Linear(embedding_dim, hidden_dim)
        self.rel_proj = nn.Linear(embedding_dim, hidden_dim)
        self.q_proj = nn.Linear(embedding_dim, hidden_dim)

        for proj in (self.node_proj, self.rel_proj, self.q_proj):
            _init_xavier(proj)

        self.graph_layers = nn.ModuleList(
            [
                _GNNLayer(
                    hidden_dim=hidden_dim,
                    dropout=gnn_dropout,
                    use_question_conditioning=gnn_use_question_conditioning,
                )
                for _ in range(gnn_num_layers)
            ]
        )

    def _project_nodes(self, node_tokens: torch.Tensor) -> torch.Tensor:
        if self.node_adapter is not None:
            node_tokens = self.node_adapter(node_tokens)
        t = node_tokens.to(self.node_norm.weight.dtype)
        return self.node_proj(self.node_norm(t))

    def _project_edge_relations(
        self, edge_relation_tokens: torch.Tensor
    ) -> torch.Tensor:
        if self.rel_adapter is not None:
            edge_relation_tokens = self.rel_adapter(edge_relation_tokens)
        t = edge_relation_tokens.to(self.rel_norm.weight.dtype)
        return self.rel_proj(self.rel_norm(t))

    def _project_question(self, question_emb: torch.Tensor) -> torch.Tensor:
        t = question_emb.to(self.q_proj.weight.dtype)
        return self.q_proj(t)

    def forward(
        self,
        batch: RetrievalBatch,
        active_edges: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_h = self._project_nodes(batch.node_tokens)
        edge_relation_h = self._project_edge_relations(batch.edge_relation_tokens)
        q_h = self._project_question(batch.question_emb)

        if self.gnn_num_layers == 0:
            return node_h, edge_relation_h, q_h

        if active_edges is not None:
            edge_index = batch.edge_index[:, active_edges]
            gnn_edge_relation_tokens = edge_relation_h[active_edges]
        else:
            edge_index = batch.edge_index
            gnn_edge_relation_tokens = edge_relation_h

        for layer in self.graph_layers:
            node_h = layer(
                node_tokens=node_h,
                edge_relation_tokens=gnn_edge_relation_tokens,
                question_tokens=q_h,
                edge_index=edge_index,
                node_graph_index=batch.batch,
            )

        return node_h, edge_relation_h, q_h


__all__ = ["EmbeddingAdapter", "GNNBackbone", "build_sparse_adjacency"]
