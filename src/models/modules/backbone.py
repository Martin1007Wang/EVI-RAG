"""
GNN backbone for GFlowNet-based knowledge-graph retrieval.

Optimisations over the original version
----------------------------------------
1. EmbeddingAdapter and backbone MLPs rely on autocast instead of manually
   forcing large activations to parameter dtype.
2. _GNNLayer:
   - Softmax-based edge-weight normalisation via PyG's built-in
     ``torch_geometric.utils.softmax`` (replaces hand-written sigmoid +
     sum-normalise; more stable gradients, no clamp needed).
   - Edge features updated every layer with a lightweight residual MLP so
     that multi-hop reasoning can condition on context-aware relation
     representations (previously edges were static across all GNN layers).
   - Forward and reverse neighbourhoods are aggregated explicitly so the node
     update MLP can consume them as separate feature blocks.
   - Named constants replace magic multipliers (3×/4× hidden_dim).
3. GNNBackbone:
   - active_edges mask applied once; augmented index built from the result.
   - Minor readability fixes (named variables, inline comments).
4. All hand-written index_add_ / clamp_min_ normalisation removed.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax

from src.data.schema import RetrievalBatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# EmbeddingAdapter
# ---------------------------------------------------------------------------


class EmbeddingAdapter(nn.Module):
    """
    Low-rank residual adapter: x' = x + W_up(GELU(W_down(Norm(x)))).

    ``up`` is zero-initialised so the adapter is an identity map at the start
    of training, keeping activations on the pre-trained manifold.

    Large activations are left in their current dtype so AMP/autocast can pick
    the execution dtype instead of being forced back to parameter precision.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return x + self.drop(self.up(self.act(self.down(self.norm(x)))))


# ---------------------------------------------------------------------------
# GNN message-passing layer
# ---------------------------------------------------------------------------

# Number of feature groups concatenated before the update MLP.
# [self, agg_fwd, agg_rev] = 3, plus optional [question] = 4.
_AGG_GROUPS_BASE = 3
_AGG_GROUPS_WITH_Q = 4


class _GNNLayer(MessagePassing):
    """
    Single GNN layer with:
      - Relation- and question-gated edge weights (softmax-normalised via PyG).
      - Separate incoming/outgoing aggregations for node updates.
      - Per-layer edge feature update (residual MLP) so relation
        representations can accumulate structural context across hops.

    Aggregation scheme
    ------------------
    Incoming and outgoing neighbourhoods are propagated separately so the
    update MLP sees them as distinct feature groups. This costs one extra
    propagate relative to a packed representation, but keeps the node update
    path simple and explicit.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        dropout: float,
        use_question_conditioning: bool,
    ) -> None:
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.use_question_conditioning = use_question_conditioning

        # Edge logit: relation contribution.
        self.relation_gate = nn.Linear(hidden_dim, 1, bias=False)
        _init_xavier(self.relation_gate)

        # Edge logit: question contribution (optional).
        self.question_proj: Optional[nn.Linear] = None
        if use_question_conditioning:
            self.question_proj = nn.Linear(hidden_dim, 1, bias=False)
            _init_xavier(self.question_proj)

        # Node update MLP input:
        #   [node_self | agg_fwd | agg_rev | (question)]
        # Each agg block has hidden_dim; fwd+rev are packed in one propagate
        # output of size 2*hidden_dim, so split later.
        n_groups = _AGG_GROUPS_WITH_Q if use_question_conditioning else _AGG_GROUPS_BASE
        update_width = hidden_dim * n_groups
        self.update_norm = nn.LayerNorm(update_width)
        self.update_mlp = _build_mlp(
            input_dim=update_width,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.node_dropout = nn.Dropout(dropout)

        # Edge update MLP: relation_h' = relation_h + MLP([relation_h, src_h, dst_h])
        edge_update_width = hidden_dim * 3  # relation | src | dst
        self.edge_update_norm = nn.LayerNorm(edge_update_width)
        self.edge_update_mlp = _build_mlp(
            input_dim=edge_update_width,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.edge_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # PyG message hook
    # ------------------------------------------------------------------

    def message(  # type: ignore[override]
        self,
        x_j: torch.Tensor,  # [E_aug, hidden_dim]  source node features
        edge_weight: torch.Tensor,  # [E_aug]
    ) -> torch.Tensor:
        return edge_weight.unsqueeze(-1) * x_j  # [E_aug, hidden_dim]

    # ------------------------------------------------------------------
    # Edge weight computation
    # ------------------------------------------------------------------

    def _compute_edge_logits(
        self,
        edge_relation_tokens: torch.Tensor,  # [E, hidden_dim]
        question_tokens: torch.Tensor,  # [G, hidden_dim]
        src: torch.Tensor,  # [E]
        node_graph_index: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """Return unnormalised logits of shape [E]."""
        logits = self.relation_gate(edge_relation_tokens).squeeze(-1)  # [E]

        if self.use_question_conditioning:
            assert self.question_proj is not None
            graph_per_src = node_graph_index[src]  # [E]
            q_per_edge = question_tokens[graph_per_src]  # [E, hidden_dim]
            logits = logits + self.question_proj(q_per_edge).squeeze(-1)

        return logits  # [E]

    # ------------------------------------------------------------------
    # Edge feature update
    # ------------------------------------------------------------------

    def _update_edge_features(
        self,
        edge_relation_h: torch.Tensor,  # [E, hidden_dim]
        node_h: torch.Tensor,  # [N, hidden_dim]
        src: torch.Tensor,  # [E]  long
        dst: torch.Tensor,  # [E]  long
    ) -> torch.Tensor:
        """Residual update: edge_h' = edge_h + MLP([edge_h | src_h | dst_h])."""
        combined = torch.cat([edge_relation_h, node_h[src], node_h[dst]], dim=-1)
        combined = self.edge_update_norm(combined)
        delta = self.edge_dropout(self.edge_update_mlp(combined))
        return edge_relation_h + delta.to(edge_relation_h.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,  # [N, hidden_dim]
        edge_relation_tokens: torch.Tensor,  # [E, hidden_dim]
        question_tokens: torch.Tensor,  # [G, hidden_dim]
        edge_index: torch.Tensor,  # [2, E]
        node_graph_index: torch.Tensor,  # [N]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        node_h_new        : [N, hidden_dim]
        edge_relation_new : [E, hidden_dim]
        """
        num_nodes = node_tokens.size(0)
        num_edges = edge_index.size(1)

        if num_nodes == 0 or num_edges == 0:
            return node_tokens, edge_relation_tokens

        src = edge_index[0].long()
        dst = edge_index[1].long()

        # ---- edge logits & softmax normalisation (per destination node) ----
        logits = self._compute_edge_logits(
            edge_relation_tokens, question_tokens, src, node_graph_index
        )  # [E]

        # PyG softmax: for each destination node, weights over its incoming edges sum to 1.
        # Compute in float32 for stability, cast back for mixed-precision safety.
        fwd_weight = pyg_softmax(logits.float(), dst, num_nodes=num_nodes).to(
            node_tokens.dtype
        )

        # Reversed edges share the same logits but normalise per *source* node.
        rev_weight = pyg_softmax(logits.float(), src, num_nodes=num_nodes).to(
            node_tokens.dtype
        )

        # ---- we need separate fwd / rev aggregations for the MLP ----
        # Two lightweight propagates (no MLP yet; just weighted sums). The
        # earlier augmented-edge path was unused and only increased memory.
        incoming = self.propagate(
            edge_index=torch.stack([src, dst]),
            x=node_tokens,
            edge_weight=fwd_weight,
            size=(num_nodes, num_nodes),
        )
        outgoing = self.propagate(
            edge_index=torch.stack([dst, src]),
            x=node_tokens,
            edge_weight=rev_weight,
            size=(num_nodes, num_nodes),
        )

        # ---- node update ----
        parts = [node_tokens, incoming, outgoing]
        if self.use_question_conditioning:
            parts.append(question_tokens[node_graph_index])

        combined = torch.cat(parts, dim=-1)
        combined = self.update_norm(combined)
        node_h_new = node_tokens + self.node_dropout(self.update_mlp(combined))

        # ---- edge feature update ----
        edge_relation_new = self._update_edge_features(
            edge_relation_tokens, node_tokens, src, dst
        )

        return node_h_new, edge_relation_new


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
    node_h          : Tensor  [TotalNodes, hidden_dim]
    edge_relation_h : Tensor  [TotalEdges, hidden_dim]  (context-updated per layer)
    q_h             : Tensor  [NumGraphs,  hidden_dim]
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

        # ---- optional pre-trained embedding adapters ----
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

        # ---- input projections ----
        self.node_norm = nn.LayerNorm(embedding_dim)
        self.rel_norm = nn.LayerNorm(embedding_dim)
        self.node_proj = nn.Linear(embedding_dim, hidden_dim)
        self.rel_proj = nn.Linear(embedding_dim, hidden_dim)
        self.q_proj = nn.Linear(embedding_dim, hidden_dim)
        for proj in (self.node_proj, self.rel_proj, self.q_proj):
            _init_xavier(proj)

        # ---- early question fusion ----
        # Applied before GNN layers so step-0 node representations already
        # carry question semantics even when no edge has been activated yet.
        fusion_width = hidden_dim * 2  # [node_h | q_h]
        self.node_question_fusion_norm = nn.LayerNorm(fusion_width)
        self.node_question_fusion = _build_mlp(
            input_dim=fusion_width,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            dropout=gnn_dropout,
        )
        self.node_question_fusion_dropout = nn.Dropout(gnn_dropout)

        # ---- GNN layers ----
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

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project_nodes(self, node_tokens: torch.Tensor) -> torch.Tensor:
        if self.node_adapter is not None:
            node_tokens = self.node_adapter(node_tokens)
        return self.node_proj(self.node_norm(node_tokens))

    def _project_edge_relations(
        self, edge_relation_tokens: torch.Tensor
    ) -> torch.Tensor:
        if self.rel_adapter is not None:
            edge_relation_tokens = self.rel_adapter(edge_relation_tokens)
        return self.rel_proj(self.rel_norm(edge_relation_tokens))

    def _project_question(self, question_emb: torch.Tensor) -> torch.Tensor:
        return self.q_proj(question_emb)

    def _snapshot_active_edges(
        self,
        *,
        batch: RetrievalBatch,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Snapshot the caller-provided active-edge mask at the backbone boundary.

        Policy already passes a `SubgraphState` snapshot, but backbone is also a
        public module and may be called directly in tests or future utilities.
        Cloning here prevents direct callers from leaking a live rollout mask
        into autograd-tracked indexing inside this forward path.
        """
        if active_edges.dtype != torch.bool:
            raise TypeError("active_edges must be a torch.bool tensor.")
        if active_edges.dim() != 1:
            raise ValueError("active_edges must be a 1D tensor.")
        if active_edges.device != batch.edge_index.device:
            raise ValueError(
                "active_edges and batch.edge_index must live on the same device."
            )

        num_edges = int(batch.edge_index.size(1))
        if int(active_edges.numel()) != num_edges:
            raise ValueError(
                f"active_edges length mismatch: expected {num_edges}, got {active_edges.numel()}."
            )

        return active_edges.detach().clone()

    # ------------------------------------------------------------------
    # Early question fusion
    # ------------------------------------------------------------------

    def _fuse_question_into_nodes(
        self,
        node_h: torch.Tensor,
        question_h: torch.Tensor,
        node_graph_index: torch.Tensor,
    ) -> torch.Tensor:
        if node_h.numel() == 0:
            return node_h

        # Broadcast graph-level question embedding to each node.
        node_q_h = question_h[node_graph_index]  # [N, hidden_dim]
        combined = torch.cat([node_h, node_q_h], dim=-1)  # [N, 2*hidden_dim]

        combined = self.node_question_fusion_norm(combined)

        delta = self.node_question_fusion_dropout(self.node_question_fusion(combined))
        return node_h + delta.to(node_h.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: RetrievalBatch,
        active_edges: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode the batch under an optional active-edge mask snapshot.

        If `active_edges` is provided, backbone snapshots it defensively before
        any masked gather/scatter so callers cannot pass a live mutable rollout
        mask that will later be updated in-place.
        """
        # Project all modalities into the shared hidden space.
        node_h = self._project_nodes(batch.node_tokens)
        edge_relation_h = self._project_edge_relations(batch.edge_relation_tokens)
        q_h = self._project_question(batch.question_emb)

        # Early question fusion: inject query semantics into node features
        # before any edge is activated (critical for GFlowNet step 0).
        node_h = self._fuse_question_into_nodes(node_h, q_h, batch.batch)

        if self.gnn_num_layers == 0:
            return node_h, edge_relation_h, q_h

        edge_mask = None
        if active_edges is not None:
            edge_mask = self._snapshot_active_edges(
                batch=batch, active_edges=active_edges
            )

        # Apply the active-edge mask once; both node and edge tensors use it.
        if edge_mask is not None:
            edge_index = batch.edge_index[:, edge_mask]
            gnn_edge_relation_h = edge_relation_h[edge_mask]
        else:
            edge_index = batch.edge_index
            gnn_edge_relation_h = edge_relation_h

        # Iterative graph propagation with per-layer edge updates.
        for layer in self.graph_layers:
            node_h, gnn_edge_relation_h = layer(
                node_tokens=node_h,
                edge_relation_tokens=gnn_edge_relation_h,
                question_tokens=q_h,
                edge_index=edge_index,
                node_graph_index=batch.batch,
            )

        # Write updated edge features back to the full edge_relation_h tensor.
        if edge_mask is not None:
            edge_relation_h = edge_relation_h.clone()
            edge_relation_h[edge_mask] = gnn_edge_relation_h
        else:
            edge_relation_h = gnn_edge_relation_h

        return node_h, edge_relation_h, q_h


__all__ = ["EmbeddingAdapter", "GNNBackbone"]
