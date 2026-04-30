from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_scatter import scatter_logsumexp, scatter_sum

from src.data.schema import RetrievalBatch
from src.weaver.state import State

from .backbone import FeatureBank


@dataclass(frozen=True)
class EvidenceContext:
    """
    Query-conditioned evidence state shared by flow, Stop, and edge scoring.
    """

    query_h: torch.Tensor
    state_h: torch.Tensor
    node_h: torch.Tensor
    rel_h: torch.Tensor
    progress: torch.Tensor


class StateReadout(nn.Module):
    """
    Query-conditioned readout for canonical subgraph states.

    State semantics:
        s = (V_s, E_s)

    Readout semantics:
        anchors are fixed query-conditioned roots;
        active nodes and selected non-root edges are evidence;
        attention is conditioned on the question, not rollout order.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        state_feature_dim: int = 0,
        dropout: float = 0.0,
        use_state_features: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if use_state_features or state_feature_dim:
            raise ValueError(
                "StateReadout no longer accepts handcrafted state_features. "
                "Use the graph state and progress encoded in EvidenceContext instead."
            )

        self.attention_scale = self.hidden_dim**-0.5

        self.edge_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.edge_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.node_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.node_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.node_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        input_dim = hidden_dim * 4 + 1

        self.out = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        state: State,
        state_features: torch.Tensor | None = None,
    ) -> EvidenceContext:
        if state_features is not None:
            raise ValueError("StateReadout does not consume external state_features.")

        device = fb.node_h.device
        dtype = fb.node_h.dtype
        num_graphs = int(batch.num_graphs)

        node_batch = batch.batch.to(device=device, dtype=torch.long)
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        active_edges = state.active_edges.to(device=device, dtype=torch.bool)
        root_edges = state.root_active_edges.to(device=device, dtype=torch.bool)
        anchor_mask = fb.anchor_mask.to(device=device, dtype=torch.bool)

        anchor_pool = self._mean_nodes(
            node_h=fb.node_h,
            node_batch=node_batch,
            mask=anchor_mask,
            num_graphs=num_graphs,
        )

        node_evidence = self._query_node_pool(
            query_h=fb.query_h,
            node_h=fb.node_h,
            node_batch=node_batch,
            node_mask=active_nodes,
            num_graphs=num_graphs,
        )

        edge_evidence = self._query_edge_pool(
            fb=fb,
            batch=batch,
            edge_batch=edge_batch,
            selected_edges=active_edges & ~root_edges,
            num_graphs=num_graphs,
        )

        progress = state.expand_ratio_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        ).to(device=device, dtype=dtype)

        state_h = self.state_norm(
            self.out(
                torch.cat(
                    [
                        fb.query_h,
                        anchor_pool,
                        edge_evidence,
                        node_evidence,
                        progress.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
        )

        return EvidenceContext(
            query_h=fb.query_h,
            state_h=state_h,
            node_h=fb.node_h,
            rel_h=fb.rel_h,
            progress=progress,
        )

    def _query_edge_pool(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        edge_batch: torch.Tensor,
        selected_edges: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if selected_edges.numel() == 0 or not bool(selected_edges.any()):
            return fb.node_h.new_zeros((num_graphs, self.hidden_dim))

        edge_ids = selected_edges.nonzero(as_tuple=False).view(-1)
        edge_index = batch.edge_index.to(device=fb.node_h.device, dtype=torch.long)

        src = edge_index[0].index_select(0, edge_ids)
        dst = edge_index[1].index_select(0, edge_ids)
        graph_id = edge_batch.index_select(0, edge_ids)

        src_h = fb.node_h.index_select(0, src)
        rel_h = fb.rel_h.index_select(0, edge_ids)
        dst_h = fb.node_h.index_select(0, dst)

        edge_h = self.edge_encoder(torch.cat([src_h, rel_h, dst_h], dim=-1))

        return self._query_attention_pool(
            query_h=fb.query_h,
            values=edge_h,
            batch_index=graph_id,
            q_proj=self.edge_q,
            k_proj=self.edge_k,
            v_proj=self.edge_v,
            num_graphs=num_graphs,
        )

    def _query_node_pool(
        self,
        *,
        query_h: torch.Tensor,
        node_h: torch.Tensor,
        node_batch: torch.Tensor,
        node_mask: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if node_mask.numel() == 0 or not bool(node_mask.any()):
            return node_h.new_zeros((num_graphs, self.hidden_dim))

        node_ids = node_mask.nonzero(as_tuple=False).view(-1)
        values = node_h.index_select(0, node_ids)
        graph_id = node_batch.index_select(0, node_ids)

        return self._query_attention_pool(
            query_h=query_h,
            values=values,
            batch_index=graph_id,
            q_proj=self.node_q,
            k_proj=self.node_k,
            v_proj=self.node_v,
            num_graphs=num_graphs,
        )

    def _query_attention_pool(
        self,
        *,
        query_h: torch.Tensor,
        values: torch.Tensor,
        batch_index: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        num_graphs: int,
    ) -> torch.Tensor:
        if values.numel() == 0:
            return query_h.new_zeros((int(num_graphs), self.hidden_dim))

        batch_index = batch_index.to(device=values.device, dtype=torch.long)

        q = q_proj(query_h)
        k = k_proj(values)
        v = v_proj(values)

        scores = (q.index_select(0, batch_index) * k).sum(dim=-1)
        scores = scores * self.attention_scale

        log_norm = scatter_logsumexp(
            scores,
            batch_index,
            dim=0,
            dim_size=int(num_graphs),
        )
        weight = (scores - log_norm.index_select(0, batch_index)).exp()

        return scatter_sum(
            v * weight.unsqueeze(-1),
            batch_index,
            dim=0,
            dim_size=int(num_graphs),
        )

    @staticmethod
    def _mean_nodes(
        *,
        node_h: torch.Tensor,
        node_batch: torch.Tensor,
        mask: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if mask.numel() == 0 or not bool(mask.any()):
            return node_h.new_zeros((num_graphs, node_h.size(-1)))

        node_ids = mask.nonzero(as_tuple=False).view(-1)
        values = node_h.index_select(0, node_ids)
        graph_id = node_batch.index_select(0, node_ids)

        return StateReadout._mean_by_graph(
            values=values,
            graph_id=graph_id,
            num_graphs=num_graphs,
        )

    @staticmethod
    def _mean_by_graph(
        *,
        values: torch.Tensor,
        graph_id: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros((num_graphs, values.size(-1)))

        total = scatter_sum(values, graph_id, dim=0, dim_size=num_graphs)
        count = torch.bincount(graph_id, minlength=num_graphs).to(
            device=values.device,
            dtype=values.dtype,
        )

        return total / count.clamp_min(1).unsqueeze(-1)


__all__ = ["EvidenceContext", "StateReadout"]
