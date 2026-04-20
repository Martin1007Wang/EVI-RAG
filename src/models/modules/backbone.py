from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_softmax, scatter_sum

from src.data.schema import RetrievalBatch
from src.utils.nn_utils import (
    init_xavier,
    validate_bool_mask,
)


@dataclass(frozen=True)
class PreparedGNNInput:
    """Static features computed once per batch / rollout.

    All tensors live in the same encoder semantic space because the
    preprocessing pipeline already emits aligned L2-normalized text
    embeddings for nodes, relations, and queries. Cosine similarities
    between node_h, rel_h, and query_h are therefore meaningful from step 0.

    node_h:       [N, H]  node semantics in encoder space
    rel_h:        [E, H]  forward relation semantics in encoder space
    query_h:      [G, H]  query semantics in encoder space
    node_input_h: [N, H]  NBF initial states after anchor-gated query fusion

    NOTE: inv_rel_h removed. The dataset contains only directed edges with no
    inverse-relation augmentation. Injecting a backward pass over non-existent
    edges would fabricate directional signals absent from the data.
    """

    node_h: torch.Tensor
    rel_h: torch.Tensor
    query_h: torch.Tensor
    node_input_h: torch.Tensor


class _NBFLayer(nn.Module):
    """One untied NBF-style Bellman-Ford message-passing layer.

    Changes vs. original:
    - Removed bwd_msg_mlp and inv_rel_h: dataset is strictly directed; backward
      messages over non-existent edges introduce spurious gradients.
    - Replaced scatter_mean with query-conditioned softmax attention aggregation:
      scatter_mean dilutes sparse correct-path signals on high-degree hub nodes
      (common in Freebase PPR subgraphs). Attention lets the model upweight
      query-relevant neighbours regardless of degree.
    """

    def __init__(self, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        # Forward message MLP: [src_node, rel, query] → message
        self.fwd_msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Attention scoring: projects message to scalar logit
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        for m in self.fwd_msg_mlp:
            if isinstance(m, nn.Linear):
                init_xavier(m)
        init_xavier(self.attn_score)
        init_xavier(self.update)

    def forward(
        self,
        *,
        node_h: torch.Tensor,
        rel_h: torch.Tensor,
        query_h: torch.Tensor,
        edge_index: torch.Tensor,
        node_graph_index: torch.Tensor,
        edge_state_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = int(node_h.size(0))
        if num_nodes == 0 or edge_index.size(1) == 0:
            return node_h

        src = edge_index[0].long()
        dst = edge_index[1].long()

        if edge_state_ids is not None:
            if edge_state_ids.dtype != torch.long:
                raise TypeError("edge_state_ids must be torch.long.")
            if edge_state_ids.shape != (edge_index.size(1),):
                raise ValueError(
                    f"edge_state_ids shape mismatch: expected ({edge_index.size(1)},), "
                    f"got {tuple(edge_state_ids.shape)}."
                )
            # state=0 (inactive): fully masked — irrelevant to current subgraph.
            # state=1 (frontier): included — lets policy sense reachable neighbours.
            # state=2 (traversed): included — propagates established path context.
            active_mask = edge_state_ids > 0
            if not bool(active_mask.any()):
                return node_h
            src = src[active_mask]
            dst = dst[active_mask]
            rel_h = rel_h[active_mask]

        query_per_edge = query_h.index_select(0, node_graph_index.index_select(0, src))

        # Compute forward messages: [src_node || rel || query] → [E, H]
        msg_fwd = self.fwd_msg_mlp(torch.cat([node_h.index_select(0, src), rel_h, query_per_edge], dim=-1))

        # Query-conditioned softmax attention over incoming neighbours.
        # attn_logit: scalar per edge; softmax normalises over all edges sharing dst.
        # This prevents hub nodes from diluting sparse query-relevant signals.
        attn_logit = self.attn_score(msg_fwd)  # [E, 1]
        attn_weight = scatter_softmax(attn_logit, dst, dim=0)  # [E, 1]
        agg = scatter_sum(msg_fwd * attn_weight, dst, dim=0, dim_size=num_nodes)  # [N, H]

        return self.norm(node_h + self.dropout(self.update(agg)))


@dataclass(frozen=True)
class BackboneOutput:
    node_h: torch.Tensor
    rel_h: torch.Tensor
    query_h: torch.Tensor
    feature_bank: PreparedGNNInput
    edge_state_ids: torch.Tensor

    def as_triple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.node_h, self.rel_h, self.query_h

    def __iter__(self):
        yield from self.as_triple()


class NBFBackbone(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int = 1024,
        hidden_dim: int = 1024,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.1,
        dde_max_distance: int = 4,
    ) -> None:
        super().__init__()
        self.emb_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.gnn_num_layers = int(gnn_num_layers)
        self.dde_max_distance = int(dde_max_distance)

        if self.gnn_num_layers < 0:
            raise ValueError(f"gnn_num_layers must be >= 0, got {self.gnn_num_layers}.")
        if self.dde_max_distance < 0:
            raise ValueError(f"dde_max_distance must be >= 0, got {self.dde_max_distance}.")
        if self.hidden_dim != self.emb_dim:
            raise ValueError(
                "NBFBackbone no longer applies a shared projection, so "
                f"hidden_dim must equal embedding_dim. Got hidden_dim={self.hidden_dim} "
                f"and embedding_dim={self.emb_dim}."
            )
        self.non_text_embedding = nn.Parameter(torch.randn(self.emb_dim) * 0.02)
        self.anchor_gate = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.nbf_layers = nn.ModuleList(
            _NBFLayer(hidden_dim=self.hidden_dim, dropout=gnn_dropout) for _ in range(self.gnn_num_layers)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        init_xavier(self.anchor_gate)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocessed embeddings are already aligned in encoder space."""
        return x

    def _resolve_node_tokens(self, batch: RetrievalBatch) -> torch.Tensor:
        tokens = batch.node_tokens
        mask = getattr(batch, "non_text_node_mask", None)
        if mask is None or not bool(mask.any()):
            return tokens
        if mask.dtype != torch.bool:
            raise TypeError("non_text_node_mask must be torch.bool.")
        if mask.shape != (tokens.size(0),):
            raise ValueError(
                f"non_text_node_mask shape mismatch: expected ({tokens.size(0)},), " f"got {tuple(mask.shape)}."
            )
        placeholder = self.non_text_embedding.to(device=tokens.device, dtype=tokens.dtype)
        return torch.where(mask.unsqueeze(-1), placeholder.view(1, -1), tokens)

    def _build_node_input_h(
        self,
        *,
        node_h: torch.Tensor,
        query_h: torch.Tensor,
        batch_index: torch.Tensor,
        anchor_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        node_input_h = node_h.clone()
        if anchor_mask is None:
            return node_input_h
        if anchor_mask.dtype != torch.bool:
            raise TypeError("is_anchor_mask must be torch.bool.")
        if anchor_mask.shape != (node_h.size(0),):
            raise ValueError(
                f"is_anchor_mask shape mismatch: expected ({node_h.size(0)},), " f"got {tuple(anchor_mask.shape)}."
            )
        if not bool(anchor_mask.any()):
            return node_input_h
        anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).view(-1)
        anchor_batch_idx = batch_index.index_select(0, anchor_idx)
        anchor_node_h = node_h.index_select(0, anchor_idx)
        anchor_query_h = query_h.index_select(0, anchor_batch_idx)
        anchor_gate = torch.sigmoid(
            self.anchor_gate(
                torch.cat(
                    [anchor_node_h, anchor_query_h, anchor_node_h * anchor_query_h],
                    dim=-1,
                )
            )
        )
        fused_anchor_h = (1.0 - anchor_gate) * anchor_node_h + anchor_gate * anchor_query_h
        node_input_h[anchor_mask] = F.normalize(fused_anchor_h, p=2, dim=-1)
        return node_input_h

    def _derive_edge_state_ids(
        self,
        *,
        batch: RetrievalBatch,
        active_edges: Optional[torch.Tensor],
        active_nodes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_edges = int(batch.edge_index.size(1))
        device = batch.edge_index.device

        if num_edges == 0 or (active_edges is None and active_nodes is None):
            return torch.zeros(num_edges, dtype=torch.long, device=device)

        if active_edges is None:
            active_edges = torch.zeros(num_edges, dtype=torch.bool, device=device)

        if active_nodes is None:
            if not hasattr(batch, "is_anchor_mask"):
                raise ValueError(
                    "_derive_edge_state_ids: active_nodes=None but batch has no "
                    "is_anchor_mask. Cannot infer visited nodes."
                )
            visited = batch.is_anchor_mask.clone()
            if active_edges.any():
                visited[batch.edge_index[0][active_edges]] = True
                visited[batch.edge_index[1][active_edges]] = True
        else:
            visited = validate_bool_mask(active_nodes, int(batch.num_nodes), "active_nodes", batch.batch.device)

        traversed = validate_bool_mask(active_edges, num_edges, "active_edges", device)
        src, dst = batch.edge_index[0], batch.edge_index[1]
        frontier = (~traversed) & (visited[src] | visited[dst])

        state_ids = torch.zeros(num_edges, dtype=torch.long, device=device)
        state_ids[frontier] = 1
        state_ids[traversed] = 2
        return state_ids

    # ── Public API ────────────────────────────────────────────────────────

    def project(self, batch: RetrievalBatch) -> PreparedGNNInput:
        """Prepare node / relation / query features once per batch.

        The preprocessing pipeline already produces aligned L2-normalized
        embeddings, so the backbone now consumes them directly without an
        additional shared LayerNorm or projection.
        """
        node_h = self._project(self._resolve_node_tokens(batch))
        rel_h = self._project(batch.relation_tokens)
        query_h = self._project(batch.question_emb)
        node_input_h = self._build_node_input_h(
            node_h=node_h,
            query_h=query_h,
            batch_index=batch.batch,
            anchor_mask=getattr(batch, "is_anchor_mask", None),
        )
        return PreparedGNNInput(
            node_h=node_h,
            rel_h=rel_h,
            query_h=query_h,
            node_input_h=node_input_h,
        )

    def run_layers(
        self,
        *,
        feature_bank: PreparedGNNInput,
        edge_index: torch.Tensor,
        node_graph_index: torch.Tensor,
        edge_state_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_h = feature_bank.node_input_h
        for layer in self.nbf_layers:
            node_h = layer(
                node_h=node_h,
                rel_h=feature_bank.rel_h,
                query_h=feature_bank.query_h,
                edge_index=edge_index,
                node_graph_index=node_graph_index,
                edge_state_ids=edge_state_ids,
            )
        return self.output_norm(node_h)

    # Alias kept for call-site compatibility.
    def precompute_static(self, batch: RetrievalBatch) -> PreparedGNNInput:
        return self.project(batch)

    def forward(
        self,
        batch: RetrievalBatch,
        active_edges: Optional[torch.Tensor] = None,
        active_nodes: Optional[torch.Tensor] = None,
        static_context: Optional[PreparedGNNInput] = None,
    ) -> BackboneOutput:
        fb = static_context if static_context is not None else self.project(batch)

        if not (fb.node_input_h.shape[-1] == fb.rel_h.shape[-1] == fb.query_h.shape[-1] == self.hidden_dim):
            raise ValueError(
                f"Feature dim mismatch: node={fb.node_input_h.shape[-1]}, "
                f"rel={fb.rel_h.shape[-1]}, query={fb.query_h.shape[-1]}, "
                f"expected={self.hidden_dim}."
            )

        edge_state_ids = self._derive_edge_state_ids(
            batch=batch,
            active_edges=active_edges,
            active_nodes=active_nodes,
        )
        node_h = self.run_layers(
            feature_bank=fb,
            edge_index=batch.edge_index,
            node_graph_index=batch.batch,
            edge_state_ids=edge_state_ids,
        )
        return BackboneOutput(
            node_h=node_h,
            rel_h=fb.rel_h,
            query_h=fb.query_h,
            feature_bank=fb,
            edge_state_ids=edge_state_ids,
        )


__all__ = [
    "PreparedGNNInput",
    "BackboneOutput",
    "NBFBackbone",
]
