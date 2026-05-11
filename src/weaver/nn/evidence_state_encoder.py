from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_scatter import scatter_logsumexp, scatter_max, scatter_sum

from src.data.schema import RetrievalBatch
from src.weaver.state import RolloutState, State

from .edge_encoder import EdgeEncoder
from .feature_encoder import FeatureBank


@dataclass(frozen=True)
class StateOnlyContext:
    """
    Query-conditioned state representation shared by flow, Stop, and edge scoring.

    The input state is expected to be canonical:

        V_s = anchors union endpoints(E_s)

    StateReadout consumes that invariant but does not repair stale mutable node
    masks. Executor owns rollout-time state validation.

    state_h:
        Readout of the current subgraph state s = (V_s, E_s).

    query_h / node_h / rel_h:
        Model-space features reused by downstream heads.

    progress:
        Per-graph expansion progress, kept outside state_h.
    """

    state_h: torch.Tensor
    query_h: torch.Tensor
    node_h: torch.Tensor
    rel_h: torch.Tensor
    progress: torch.Tensor
    relation_path_h: torch.Tensor | None = None


@dataclass(frozen=True)
class StateContext(StateOnlyContext):
    """
    Full policy context for a state and its current action frontier.
    """

    frontier_summary: torch.Tensor | None = None
    frontier_edge_ids: torch.Tensor | None = None
    frontier_edge_batch: torch.Tensor | None = None
    frontier_edge_h: torch.Tensor | None = None

    @classmethod
    def merge(
        cls,
        state: StateOnlyContext,
        frontier: "FrontierReadout",
        *,
        frontier_summary: torch.Tensor | None,
    ) -> "StateContext":
        return cls(
            state_h=state.state_h,
            query_h=state.query_h,
            node_h=state.node_h,
            rel_h=state.rel_h,
            progress=state.progress,
            relation_path_h=state.relation_path_h,
            frontier_summary=frontier_summary,
            frontier_edge_ids=frontier.edge_ids,
            frontier_edge_batch=frontier.edge_batch,
            frontier_edge_h=frontier.edge_h,
        )


@dataclass(frozen=True)
class FrontierReadout:
    """
    Frontier under the canonical active-node set.

    edge_ids are exactly inactive edges incident to V_s. The corresponding
    edge_h is cached here so Policy does not rescan or re-encode the frontier.
    """

    summary: torch.Tensor
    edge_ids: torch.Tensor
    edge_batch: torch.Tensor
    edge_h: torch.Tensor


class StateReadout(nn.Module):
    """
    Query-conditioned readout for subgraph states.

    State:
        s = (V_s, E_s)
        V_s = anchors union endpoints(E_s)

    Readout:
        h_s = LN(MLP([h_q, Pool_q(V_s), Pool_q(E_s)]))

    All active edges are treated as evidence, including root edges.
    This module does not perform message passing and does not consume labels,
    rewards, shortest-path distances, or handcrafted state features.

    Frontier source of truth:
        frontier(s) = {e=(u,r,v) not in E_s : u in V_s or v in V_s}

    Policy.forward must use the frontier returned here; otherwise the readout
    summary and action space can silently diverge.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        edge_encoder: EdgeEncoder | None = None,
        use_path_memory: bool = True,
        use_frontier_summary: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        num_layers = int(num_layers)
        if num_layers not in {1, 2}:
            raise ValueError(
                f"state_readout.num_layers must be 1 or 2, got {num_layers}."
            )

        self.attention_scale = self.hidden_dim**-0.5
        self.edge_encoder = edge_encoder or EdgeEncoder(hidden_dim=self.hidden_dim)
        self.use_path_memory = bool(use_path_memory)
        self.use_frontier_summary = bool(use_frontier_summary)

        input_dim = self.hidden_dim * 3
        if self.use_path_memory:
            input_dim += self.hidden_dim

        if num_layers == 1:
            self.out = nn.Linear(input_dim, self.hidden_dim)
        else:
            self.out = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        state: State | RolloutState,
    ) -> StateContext:
        state_ctx = self.forward_state(fb=fb, batch=batch, state=state)
        frontier = self.forward_frontier(
            fb=fb,
            batch=batch,
            state=state,
            state_ctx=state_ctx,
        )
        frontier_summary = frontier.summary if self.use_frontier_summary else None
        return StateContext.merge(
            state_ctx,
            frontier,
            frontier_summary=frontier_summary,
        )

    def forward_state(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        state: State | RolloutState,
    ) -> StateOnlyContext:
        device = fb.node_h.device
        dtype = fb.node_h.dtype
        num_graphs = int(batch.num_graphs)

        if state.active_nodes.ndim == 2:
            if not isinstance(state, RolloutState):
                raise TypeError(
                    "2D active state masks require RolloutState for "
                    "rollout_to_graph mapping."
                )
            return self._forward_rollout_state(
                fb=fb,
                batch=batch,
                state=state,
            )

        node_batch = batch.batch.to(device=device, dtype=torch.long)
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        active_edges = state.active_edges.to(device=device, dtype=torch.bool)

        node_evidence = self._pool_nodes(
            query_h=fb.query_h,
            node_h=fb.node_h,
            node_batch=node_batch,
            node_mask=active_nodes,
            num_graphs=num_graphs,
        )

        edge_evidence = self._pool_edges(
            fb=fb,
            batch=batch,
            edge_batch=edge_batch,
            edge_mask=active_edges,
            num_graphs=num_graphs,
        )

        relation_path_h = None
        if self.use_path_memory:
            relation_path_h = self._pool_relations(
                fb=fb,
                edge_batch=edge_batch,
                edge_mask=active_edges,
                num_graphs=num_graphs,
            )

        state_pieces = [fb.query_h, node_evidence, edge_evidence]
        if relation_path_h is not None:
            state_pieces.append(relation_path_h)

        state_h = self.norm(self.out(torch.cat(state_pieces, dim=-1)))

        progress = state.expand_ratio_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        ).to(device=device, dtype=dtype)

        return StateOnlyContext(
            state_h=state_h,
            query_h=fb.query_h,
            node_h=fb.node_h,
            rel_h=fb.rel_h,
            progress=progress,
            relation_path_h=relation_path_h,
        )

    def forward_frontier(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        state: State | RolloutState,
        state_ctx: StateOnlyContext | None = None,
    ) -> FrontierReadout:
        device = fb.node_h.device

        if state.active_nodes.ndim == 2:
            if not isinstance(state, RolloutState):
                raise TypeError(
                    "2D active state masks require RolloutState for "
                    "rollout_to_graph mapping."
                )
            rollout_to_graph = state.rollout_to_graph.to(
                device=device,
                dtype=torch.long,
            )
            query_h = (
                state_ctx.query_h.to(device=device, dtype=fb.node_h.dtype)
                if state_ctx is not None
                else fb.query_h.index_select(0, rollout_to_graph)
            )
            return self._frontier_rollout_readout(
                fb=fb,
                batch=batch,
                query_h=query_h,
                edge_batch=batch.edge_batch.to(device=device, dtype=torch.long),
                active_nodes=state.active_nodes.to(device=device, dtype=torch.bool),
                active_edges=state.active_edges.to(device=device, dtype=torch.bool),
                rollout_to_graph=rollout_to_graph,
                num_rollouts=int(state.num_rollouts),
            )

        return self._frontier_readout(
            fb=fb,
            batch=batch,
            edge_batch=batch.edge_batch.to(device=device, dtype=torch.long),
            active_nodes=state.active_nodes.to(device=device, dtype=torch.bool),
            active_edges=state.active_edges.to(device=device, dtype=torch.bool),
            num_graphs=int(batch.num_graphs),
        )

    def forward_successor_state_delta(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        state: State | RolloutState,
        candidate_edge_ids: torch.Tensor,
        candidate_batch_ids: torch.Tensor,
    ) -> StateOnlyContext:
        """
        Exact state-only readout for candidate successors s + e.

        This avoids materializing one dense successor mask row per candidate.
        It relies on StateReadout's query-softmax pooling being decomposable
        into parent active-set statistics plus the candidate edge/endpoints.
        """
        device = fb.node_h.device
        dtype = fb.node_h.dtype
        edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)
        parent_ids = candidate_batch_ids.to(device=device, dtype=torch.long).view(-1)
        if edge_ids.shape != parent_ids.shape:
            raise ValueError(
                "candidate_edge_ids and candidate_batch_ids must have matching shape: "
                f"{tuple(edge_ids.shape)} != {tuple(parent_ids.shape)}."
            )

        num_candidates = int(edge_ids.numel())
        if num_candidates == 0:
            query_h = fb.query_h.new_zeros((0, self.hidden_dim))
            relation_path_h = (
                fb.node_h.new_zeros((0, self.hidden_dim))
                if self.use_path_memory
                else None
            )
            return StateOnlyContext(
                state_h=fb.node_h.new_zeros((0, self.hidden_dim)),
                query_h=query_h,
                node_h=fb.node_h,
                rel_h=fb.rel_h,
                progress=fb.node_h.new_zeros((0,), dtype=dtype),
                relation_path_h=relation_path_h,
            )

        num_edges = int(batch.edge_index.size(1))
        if bool((edge_ids < 0).any()) or bool((edge_ids >= num_edges).any()):
            raise ValueError("candidate_edge_ids must contain original edge ids.")

        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        src_all, dst_all = edge_index
        src = src_all.index_select(0, edge_ids)
        dst = dst_all.index_select(0, edge_ids)
        row_ids = torch.arange(num_candidates, dtype=torch.long, device=device)

        if state.active_nodes.ndim == 2:
            if not isinstance(state, RolloutState):
                raise TypeError(
                    "2D active state masks require RolloutState for "
                    "rollout_to_graph mapping."
                )
            num_parents = int(state.num_rollouts)
            if bool((parent_ids < 0).any()) or bool((parent_ids >= num_parents).any()):
                raise ValueError("candidate_batch_ids must contain rollout row ids.")

            rollout_to_graph = state.rollout_to_graph.to(
                device=device,
                dtype=torch.long,
            )
            parent_static_ids = rollout_to_graph.index_select(0, parent_ids)
            parent_query_h = fb.query_h.index_select(0, rollout_to_graph)
            query_h = fb.query_h.index_select(0, parent_static_ids)

            active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
            active_edges = state.active_edges.to(device=device, dtype=torch.bool)
            root_edges = state.root_edges.to(device=device, dtype=torch.bool)

            parent_node_pool, parent_node_log_norm = (
                self._pool_rollout_nodes_with_log_norm(
                    query_h=parent_query_h,
                    node_h=fb.node_h,
                    node_mask=active_nodes,
                    num_rollouts=num_parents,
                )
            )
            parent_edge_pool, parent_edge_log_norm = (
                self._pool_rollout_edges_with_log_norm(
                    fb=fb,
                    batch=batch,
                    query_h=parent_query_h,
                    edge_mask=active_edges,
                    num_rollouts=num_parents,
                )
            )
            if self.use_path_memory:
                parent_rel_pool, parent_rel_log_norm = (
                    self._pool_rollout_relations_with_log_norm(
                        fb=fb,
                        query_h=parent_query_h,
                        edge_mask=active_edges,
                        num_rollouts=num_parents,
                    )
                )
            else:
                parent_rel_pool = None
                parent_rel_log_norm = None

            src_active = active_nodes[parent_ids, src]
            dst_active = active_nodes[parent_ids, dst]
            edge_active = active_edges[parent_ids, edge_ids]
            edge_root = root_edges[parent_ids, edge_ids]
            parent_used = (active_edges & ~root_edges).sum(dim=1).to(dtype=torch.long)
        else:
            num_parents = int(batch.num_graphs)
            if bool((parent_ids < 0).any()) or bool((parent_ids >= num_parents).any()):
                raise ValueError("candidate_batch_ids must contain graph ids.")

            query_h = fb.query_h.index_select(0, parent_ids)
            node_batch = batch.batch.to(device=device, dtype=torch.long)
            edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
            active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
            active_edges = state.active_edges.to(device=device, dtype=torch.bool)
            root_edges = state.root_edges.to(device=device, dtype=torch.bool)

            parent_node_pool, parent_node_log_norm = self._pool_nodes_with_log_norm(
                query_h=fb.query_h,
                node_h=fb.node_h,
                node_batch=node_batch,
                node_mask=active_nodes,
                num_graphs=num_parents,
            )
            parent_edge_pool, parent_edge_log_norm = self._pool_edges_with_log_norm(
                fb=fb,
                batch=batch,
                edge_batch=edge_batch,
                edge_mask=active_edges,
                num_graphs=num_parents,
            )
            if self.use_path_memory:
                parent_rel_pool, parent_rel_log_norm = (
                    self._pool_relations_with_log_norm(
                        fb=fb,
                        edge_batch=edge_batch,
                        edge_mask=active_edges,
                        num_graphs=num_parents,
                    )
                )
            else:
                parent_rel_pool = None
                parent_rel_log_norm = None

            src_active = active_nodes.index_select(0, src)
            dst_active = active_nodes.index_select(0, dst)
            edge_active = active_edges.index_select(0, edge_ids)
            edge_root = root_edges.index_select(0, edge_ids)
            parent_used = state.expanded_edge_count_per_graph(
                edge_batch=edge_batch,
                num_graphs=num_parents,
            )

        src_new = ~src_active
        dst_new = (~dst_active) & dst.ne(src)
        node_event_rows = torch.cat([row_ids[src_new], row_ids[dst_new]], dim=0)
        node_event_ids = torch.cat([src[src_new], dst[dst_new]], dim=0)
        node_event_values = fb.node_h.index_select(0, node_event_ids)
        node_evidence = self._combine_query_pool_delta(
            parent_pool=parent_node_pool.index_select(0, parent_ids),
            parent_log_norm=parent_node_log_norm.index_select(0, parent_ids),
            query_h=query_h,
            event_values=node_event_values,
            event_batch=node_event_rows,
        )

        edge_h = self.edge_encoder(
            src_h=fb.node_h.index_select(0, src),
            rel_h=fb.rel_h.index_select(0, edge_ids),
            dst_h=fb.node_h.index_select(0, dst),
        )
        edge_new = ~edge_active
        edge_event_rows = row_ids[edge_new]
        edge_evidence = self._combine_query_pool_delta(
            parent_pool=parent_edge_pool.index_select(0, parent_ids),
            parent_log_norm=parent_edge_log_norm.index_select(0, parent_ids),
            query_h=query_h,
            event_values=edge_h.index_select(0, edge_event_rows),
            event_batch=edge_event_rows,
        )

        relation_path_h = None
        if self.use_path_memory:
            if parent_rel_pool is None or parent_rel_log_norm is None:
                raise RuntimeError("Missing parent relation pool for path memory.")
            relation_path_h = self._combine_query_pool_delta(
                parent_pool=parent_rel_pool.index_select(0, parent_ids),
                parent_log_norm=parent_rel_log_norm.index_select(0, parent_ids),
                query_h=query_h,
                event_values=fb.rel_h.index_select(0, edge_ids).index_select(
                    0,
                    edge_event_rows,
                ),
                event_batch=edge_event_rows,
            )

        state_pieces = [query_h, node_evidence, edge_evidence]
        if relation_path_h is not None:
            state_pieces.append(relation_path_h)

        state_h = self.norm(self.out(torch.cat(state_pieces, dim=-1)))

        if state.expand_budget <= 0:
            progress = fb.node_h.new_zeros((num_candidates,), dtype=dtype)
        else:
            new_expanded = (edge_new & ~edge_root).to(dtype=torch.long)
            used = parent_used.index_select(0, parent_ids) + new_expanded
            progress = (used.to(dtype=dtype) / float(state.expand_budget)).clamp(
                0.0,
                1.0,
            )

        return StateOnlyContext(
            state_h=state_h,
            query_h=query_h,
            node_h=fb.node_h,
            rel_h=fb.rel_h,
            progress=progress,
            relation_path_h=relation_path_h,
        )

    def _forward_rollout_state(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        state: RolloutState,
    ) -> StateOnlyContext:
        device = fb.node_h.device
        dtype = fb.node_h.dtype
        num_rollouts = int(state.num_rollouts)

        rollout_to_graph = state.rollout_to_graph.to(device=device, dtype=torch.long)
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        active_edges = state.active_edges.to(device=device, dtype=torch.bool)

        query_h = fb.query_h.index_select(0, rollout_to_graph)

        node_evidence = self._pool_rollout_nodes(
            query_h=query_h,
            node_h=fb.node_h,
            node_mask=active_nodes,
            num_rollouts=num_rollouts,
        )

        edge_evidence = self._pool_rollout_edges(
            fb=fb,
            batch=batch,
            query_h=query_h,
            edge_mask=active_edges,
            num_rollouts=num_rollouts,
        )

        relation_path_h = None
        if self.use_path_memory:
            relation_path_h = self._pool_rollout_relations(
                fb=fb,
                query_h=query_h,
                edge_mask=active_edges,
                num_rollouts=num_rollouts,
            )

        state_pieces = [query_h, node_evidence, edge_evidence]
        if relation_path_h is not None:
            state_pieces.append(relation_path_h)

        state_h = self.norm(self.out(torch.cat(state_pieces, dim=-1)))

        progress = state.expand_ratio_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_rollouts,
        ).to(device=device, dtype=dtype)

        return StateOnlyContext(
            state_h=state_h,
            query_h=query_h,
            node_h=fb.node_h,
            rel_h=fb.rel_h,
            progress=progress,
            relation_path_h=relation_path_h,
        )

    def _pool_nodes(
        self,
        *,
        query_h: torch.Tensor,
        node_h: torch.Tensor,
        node_batch: torch.Tensor,
        node_mask: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled, _ = self._pool_nodes_with_log_norm(
            query_h=query_h,
            node_h=node_h,
            node_batch=node_batch,
            node_mask=node_mask,
            num_graphs=num_graphs,
        )
        return pooled

    def _pool_nodes_with_log_norm(
        self,
        *,
        query_h: torch.Tensor,
        node_h: torch.Tensor,
        node_batch: torch.Tensor,
        node_mask: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_ids = node_mask.nonzero(as_tuple=False).flatten()
        if node_ids.numel() == 0:
            return self._empty_pool_with_log_norm(
                values=node_h,
                num_graphs=num_graphs,
            )

        return self._query_pool_with_log_norm(
            query_h=query_h,
            values=node_h.index_select(0, node_ids),
            batch_index=node_batch.index_select(0, node_ids),
            num_graphs=num_graphs,
        )

    def _pool_edges(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        edge_batch: torch.Tensor,
        edge_mask: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled, _ = self._pool_edges_with_log_norm(
            fb=fb,
            batch=batch,
            edge_batch=edge_batch,
            edge_mask=edge_mask,
            num_graphs=num_graphs,
        )
        return pooled

    def _pool_edges_with_log_norm(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        edge_batch: torch.Tensor,
        edge_mask: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_ids = edge_mask.nonzero(as_tuple=False).flatten()
        if edge_ids.numel() == 0:
            return self._empty_pool_with_log_norm(
                values=fb.node_h,
                num_graphs=num_graphs,
            )

        edge_index = batch.edge_index.to(device=fb.node_h.device, dtype=torch.long)

        src = edge_index[0].index_select(0, edge_ids)
        dst = edge_index[1].index_select(0, edge_ids)

        edge_h = self.edge_encoder(
            src_h=fb.node_h.index_select(0, src),
            rel_h=fb.rel_h.index_select(0, edge_ids),
            dst_h=fb.node_h.index_select(0, dst),
        )

        return self._query_pool_with_log_norm(
            query_h=fb.query_h,
            values=edge_h,
            batch_index=edge_batch.index_select(0, edge_ids),
            num_graphs=num_graphs,
        )

    def _pool_relations(
        self,
        *,
        fb: FeatureBank,
        edge_batch: torch.Tensor,
        edge_mask: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled, _ = self._pool_relations_with_log_norm(
            fb=fb,
            edge_batch=edge_batch,
            edge_mask=edge_mask,
            num_graphs=num_graphs,
        )
        return pooled

    def _pool_relations_with_log_norm(
        self,
        *,
        fb: FeatureBank,
        edge_batch: torch.Tensor,
        edge_mask: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_ids = edge_mask.nonzero(as_tuple=False).flatten()
        if edge_ids.numel() == 0:
            return self._empty_pool_with_log_norm(
                values=fb.node_h,
                num_graphs=num_graphs,
            )

        return self._query_pool_with_log_norm(
            query_h=fb.query_h,
            values=fb.rel_h.index_select(0, edge_ids),
            batch_index=edge_batch.index_select(0, edge_ids),
            num_graphs=num_graphs,
        )

    def _pool_rollout_nodes(
        self,
        *,
        query_h: torch.Tensor,
        node_h: torch.Tensor,
        node_mask: torch.Tensor,
        num_rollouts: int,
    ) -> torch.Tensor:
        pooled, _ = self._pool_rollout_nodes_with_log_norm(
            query_h=query_h,
            node_h=node_h,
            node_mask=node_mask,
            num_rollouts=num_rollouts,
        )
        return pooled

    def _pool_rollout_nodes_with_log_norm(
        self,
        *,
        query_h: torch.Tensor,
        node_h: torch.Tensor,
        node_mask: torch.Tensor,
        num_rollouts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_ids, node_ids = node_mask.nonzero(as_tuple=True)
        if node_ids.numel() == 0:
            return self._empty_pool_with_log_norm(
                values=node_h,
                num_graphs=num_rollouts,
            )

        return self._query_pool_with_log_norm(
            query_h=query_h,
            values=node_h.index_select(0, node_ids),
            batch_index=rollout_ids,
            num_graphs=num_rollouts,
        )

    def _pool_rollout_edges(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        query_h: torch.Tensor,
        edge_mask: torch.Tensor,
        num_rollouts: int,
    ) -> torch.Tensor:
        pooled, _ = self._pool_rollout_edges_with_log_norm(
            fb=fb,
            batch=batch,
            query_h=query_h,
            edge_mask=edge_mask,
            num_rollouts=num_rollouts,
        )
        return pooled

    def _pool_rollout_edges_with_log_norm(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        query_h: torch.Tensor,
        edge_mask: torch.Tensor,
        num_rollouts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_ids, edge_ids = edge_mask.nonzero(as_tuple=True)
        if edge_ids.numel() == 0:
            return self._empty_pool_with_log_norm(
                values=fb.node_h,
                num_graphs=num_rollouts,
            )

        edge_index = batch.edge_index.to(device=fb.node_h.device, dtype=torch.long)
        src = edge_index[0].index_select(0, edge_ids)
        dst = edge_index[1].index_select(0, edge_ids)

        edge_h = self.edge_encoder(
            src_h=fb.node_h.index_select(0, src),
            rel_h=fb.rel_h.index_select(0, edge_ids),
            dst_h=fb.node_h.index_select(0, dst),
        )

        return self._query_pool_with_log_norm(
            query_h=query_h,
            values=edge_h,
            batch_index=rollout_ids,
            num_graphs=num_rollouts,
        )

    def _pool_rollout_relations(
        self,
        *,
        fb: FeatureBank,
        query_h: torch.Tensor,
        edge_mask: torch.Tensor,
        num_rollouts: int,
    ) -> torch.Tensor:
        pooled, _ = self._pool_rollout_relations_with_log_norm(
            fb=fb,
            query_h=query_h,
            edge_mask=edge_mask,
            num_rollouts=num_rollouts,
        )
        return pooled

    def _pool_rollout_relations_with_log_norm(
        self,
        *,
        fb: FeatureBank,
        query_h: torch.Tensor,
        edge_mask: torch.Tensor,
        num_rollouts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_ids, edge_ids = edge_mask.nonzero(as_tuple=True)
        if edge_ids.numel() == 0:
            return self._empty_pool_with_log_norm(
                values=fb.node_h,
                num_graphs=num_rollouts,
            )

        return self._query_pool_with_log_norm(
            query_h=query_h,
            values=fb.rel_h.index_select(0, edge_ids),
            batch_index=rollout_ids,
            num_graphs=num_rollouts,
        )

    def _frontier_readout(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        edge_batch: torch.Tensor,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
        num_graphs: int,
    ) -> FrontierReadout:
        edge_index = batch.edge_index.to(device=fb.node_h.device, dtype=torch.long)
        src_all, dst_all = edge_index

        frontier_mask = (
            active_nodes.index_select(0, src_all)
            | active_nodes.index_select(0, dst_all)
        ) & ~active_edges
        edge_ids = frontier_mask.nonzero(as_tuple=False).flatten()

        if edge_ids.numel() == 0:
            return FrontierReadout(
                summary=fb.node_h.new_zeros((num_graphs, 3)),
                edge_ids=edge_ids,
                edge_batch=edge_batch.new_empty((0,)),
                edge_h=fb.node_h.new_zeros((0, self.hidden_dim)),
            )

        src = src_all.index_select(0, edge_ids)
        dst = dst_all.index_select(0, edge_ids)
        graph_id = edge_batch.index_select(0, edge_ids)

        edge_h = self.edge_encoder(
            src_h=fb.node_h.index_select(0, src),
            rel_h=fb.rel_h.index_select(0, edge_ids),
            dst_h=fb.node_h.index_select(0, dst),
        )

        query = fb.query_h.index_select(0, graph_id)
        scores = (query * edge_h).sum(dim=-1) * self.attention_scale

        counts = torch.bincount(graph_id, minlength=num_graphs).to(
            device=fb.node_h.device,
            dtype=fb.node_h.dtype,
        )
        has_edge = counts.gt(0)

        logsumexp = scatter_logsumexp(
            scores,
            graph_id,
            dim=0,
            dim_size=num_graphs,
        )
        logmeanexp = logsumexp - counts.clamp_min(1.0).log()
        max_score = scatter_max(
            scores,
            graph_id,
            dim=0,
            dim_size=num_graphs,
        )[0]
        log_size = counts.clamp_min(1.0).log()

        zeros = fb.node_h.new_zeros(num_graphs)
        return FrontierReadout(
            summary=torch.stack(
                [
                    torch.where(has_edge, max_score, zeros),
                    torch.where(has_edge, logmeanexp, zeros),
                    torch.where(has_edge, log_size, zeros),
                ],
                dim=-1,
            ),
            edge_ids=edge_ids,
            edge_batch=graph_id,
            edge_h=edge_h,
        )

    def _frontier_rollout_readout(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        query_h: torch.Tensor,
        edge_batch: torch.Tensor,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
        rollout_to_graph: torch.Tensor,
        num_rollouts: int,
    ) -> FrontierReadout:
        edge_index = batch.edge_index.to(device=fb.node_h.device, dtype=torch.long)
        src_all, dst_all = edge_index

        belongs = edge_batch.view(1, -1).eq(rollout_to_graph.view(-1, 1))
        frontier_mask = (
            (
                active_nodes.index_select(1, src_all)
                | active_nodes.index_select(1, dst_all)
            )
            & ~active_edges
            & belongs
        )
        rollout_ids, edge_ids = frontier_mask.nonzero(as_tuple=True)

        if edge_ids.numel() == 0:
            return FrontierReadout(
                summary=fb.node_h.new_zeros((num_rollouts, 3)),
                edge_ids=edge_ids,
                edge_batch=rollout_ids,
                edge_h=fb.node_h.new_zeros((0, self.hidden_dim)),
            )

        src = src_all.index_select(0, edge_ids)
        dst = dst_all.index_select(0, edge_ids)

        edge_h = self.edge_encoder(
            src_h=fb.node_h.index_select(0, src),
            rel_h=fb.rel_h.index_select(0, edge_ids),
            dst_h=fb.node_h.index_select(0, dst),
        )

        query = query_h.index_select(0, rollout_ids)
        scores = (query * edge_h).sum(dim=-1) * self.attention_scale

        counts = torch.bincount(rollout_ids, minlength=num_rollouts).to(
            device=fb.node_h.device,
            dtype=fb.node_h.dtype,
        )
        has_edge = counts.gt(0)

        logsumexp = scatter_logsumexp(
            scores,
            rollout_ids,
            dim=0,
            dim_size=num_rollouts,
        )
        logmeanexp = logsumexp - counts.clamp_min(1.0).log()
        max_score = scatter_max(
            scores,
            rollout_ids,
            dim=0,
            dim_size=num_rollouts,
        )[0]
        log_size = counts.clamp_min(1.0).log()

        zeros = fb.node_h.new_zeros(num_rollouts)
        return FrontierReadout(
            summary=torch.stack(
                [
                    torch.where(has_edge, max_score, zeros),
                    torch.where(has_edge, logmeanexp, zeros),
                    torch.where(has_edge, log_size, zeros),
                ],
                dim=-1,
            ),
            edge_ids=edge_ids,
            edge_batch=rollout_ids,
            edge_h=edge_h,
        )

    def _query_pool(
        self,
        *,
        query_h: torch.Tensor,
        values: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled, _ = self._query_pool_with_log_norm(
            query_h=query_h,
            values=values,
            batch_index=batch_index,
            num_graphs=num_graphs,
        )
        return pooled

    def _query_pool_with_log_norm(
        self,
        *,
        query_h: torch.Tensor,
        values: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_index = batch_index.to(device=values.device, dtype=torch.long)

        query = query_h.index_select(0, batch_index)
        scores = (query * values).sum(dim=-1) * self.attention_scale

        log_norm = scatter_logsumexp(
            scores,
            batch_index,
            dim=0,
            dim_size=num_graphs,
        )
        weights = (scores - log_norm.index_select(0, batch_index)).exp()

        pooled = scatter_sum(
            values * weights.unsqueeze(-1),
            batch_index,
            dim=0,
            dim_size=num_graphs,
        )
        return pooled, log_norm

    def _combine_query_pool_delta(
        self,
        *,
        parent_pool: torch.Tensor,
        parent_log_norm: torch.Tensor,
        query_h: torch.Tensor,
        event_values: torch.Tensor,
        event_batch: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(parent_pool.size(0))
        if event_values.numel() == 0:
            return parent_pool

        event_batch = event_batch.to(device=parent_pool.device, dtype=torch.long)
        if event_batch.numel() != event_values.size(0):
            raise ValueError(
                "event_batch and event_values must have matching first dimension: "
                f"{event_batch.numel()} != {event_values.size(0)}."
            )

        event_query = query_h.index_select(0, event_batch)
        event_scores = (
            event_query * event_values
        ).sum(dim=-1) * self.attention_scale
        event_log_norm = scatter_logsumexp(
            event_scores,
            event_batch,
            dim=0,
            dim_size=num_graphs,
        )
        combined_log_norm = torch.logaddexp(parent_log_norm, event_log_norm)
        parent_weight = (parent_log_norm - combined_log_norm).exp().unsqueeze(-1)
        event_weights = (
            event_scores - combined_log_norm.index_select(0, event_batch)
        ).exp()
        event_pool = scatter_sum(
            event_values * event_weights.unsqueeze(-1),
            event_batch,
            dim=0,
            dim_size=num_graphs,
        )
        return parent_pool * parent_weight + event_pool

    def _empty_pool_with_log_norm(
        self,
        *,
        values: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            values.new_zeros((int(num_graphs), self.hidden_dim)),
            values.new_full((int(num_graphs),), -torch.inf),
        )


__all__ = ["FrontierReadout", "StateContext", "StateOnlyContext", "StateReadout"]
