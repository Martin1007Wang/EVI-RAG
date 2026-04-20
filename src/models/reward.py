from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from src.utils.nn_utils import cosine_scores
from src.utils.reward_utils import per_graph_mask_count


class RewardModel(nn.Module):
    """Terminal log-reward for KG subgraph retrieval.

    The terminal objective is hit-first and semantics-backed:

        log R(x) = log(recall(x))                         if recall(x) > 0
        log R(x) = log_r_min + eta * soft_answer_sim(x)  otherwise

    where ``soft_answer_sim(x)`` is a weak semantic fallback derived from the
    static entity embedding space already attached to the batch. It only
    activates when recall is zero, so any answer hit still dominates the score.

    This keeps the primary objective aligned with answer recall while avoiding
    the sparse-reward collapse where every failed trajectory receives the exact
    same terminal score.

    Potential-based Shaping
    -----------------------
    Potential-based shaping is still available for experiments, but it is
    disabled by default so the effective terminal objective remains the
    hit-first reward above.
    When enabled, a potential function ϕ(s) is defined over intermediate states:

        ϕ(s_t) = relation_shaping_scale · max_{e ∈ frontier(s_t)} cosine(q, rel(e))

    The shaped step reward is the *difference*:

        r_t^shaped = ϕ(s_{t+1}) - ϕ(s_t)

    This telescopes to ϕ(x) - ϕ(s_0) over the full trajectory, which is a
    pure function of the terminal state x and the fixed initial state s_0.
    Flow conservation is therefore preserved: the effective reward

        R_eff(x) = R(x) · exp(ϕ(x) - ϕ(s_0))

    is still a function of x alone, not of the path taken to reach x.

    This shaping favours states whose currently available frontier exposes a
    stronger relation match to the query, but still remains a pure state
    function and therefore telescopes safely.

    Parameters
    ----------
    log_r_min : float
        Finite floor used when recall is zero. Must be < 0.
    semantic_fallback_scale : float
        eta - scale of the semantic fallback bonus applied only when recall is
        zero. Set to 0.0 to recover the old hard floor.
    relation_shaping_scale : float
        λ_ϕ — scale of the relation-matching potential ϕ(s).
        Shaped step reward = ϕ(s_{t+1}) - ϕ(s_t).
        Set to 0.0 to disable shaping entirely (evaluation / inference).
    """

    def __init__(
        self,
        log_r_min: float = -5.0,
        semantic_fallback_scale: float = 0.25,
        relation_shaping_scale: float = 0.0,
        positive_coverage_shaping_scale: float = 0.0,
        distance_progress_shaping_scale: float = 0.0,
    ) -> None:
        super().__init__()

        if log_r_min >= 0.0:
            raise ValueError(f"log_r_min must be < 0 (a penalty floor), got {log_r_min}.")
        if semantic_fallback_scale < 0.0:
            raise ValueError(
                "semantic_fallback_scale must be >= 0, got "
                f"{semantic_fallback_scale}."
            )
        if relation_shaping_scale < 0.0:
            raise ValueError(
                "relation_shaping_scale must be >= 0, got "
                f"{relation_shaping_scale}."
            )
        if positive_coverage_shaping_scale < 0.0:
            raise ValueError(
                "positive_coverage_shaping_scale must be >= 0, got "
                f"{positive_coverage_shaping_scale}."
            )
        if distance_progress_shaping_scale < 0.0:
            raise ValueError(
                "distance_progress_shaping_scale must be >= 0, got "
                f"{distance_progress_shaping_scale}."
            )

        self.log_r_min = float(log_r_min)
        self.semantic_fallback_scale = float(semantic_fallback_scale)
        self.relation_shaping_scale = float(relation_shaping_scale)
        self.positive_coverage_shaping_scale = float(positive_coverage_shaping_scale)
        self.distance_progress_shaping_scale = float(distance_progress_shaping_scale)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _semantic_fallback_bonus(
        self,
        base_graph: RetrievalBatch,
        active_nodes: torch.Tensor,
    ) -> torch.Tensor:
        if self.semantic_fallback_scale == 0.0:
            return torch.zeros(
                base_graph.num_graphs,
                dtype=torch.float32,
                device=active_nodes.device,
            )

        non_text_mask = getattr(base_graph, "non_text_node_mask", None)
        if non_text_mask is None:
            text_mask = torch.ones_like(active_nodes, dtype=torch.bool)
        else:
            text_mask = ~non_text_mask

        active_text_nodes = active_nodes & text_mask
        target_text_nodes = base_graph.is_target_mask & text_mask
        node_embeddings = base_graph.node_tokens
        graph_bonus = torch.zeros(
            base_graph.num_graphs,
            dtype=torch.float32,
            device=active_nodes.device,
        )

        for graph_id in range(base_graph.num_graphs):
            graph_mask = base_graph.batch == graph_id
            graph_active = active_text_nodes & graph_mask
            graph_targets = target_text_nodes & graph_mask
            if not bool(graph_active.any()) or not bool(graph_targets.any()):
                continue

            active_h = node_embeddings[graph_active]
            target_h = node_embeddings[graph_targets].mean(dim=0, keepdim=True)
            target_h = target_h.expand(active_h.size(0), -1)
            similarity = cosine_scores(active_h, target_h).max().clamp(min=0.0, max=1.0)
            graph_bonus[graph_id] = similarity.to(dtype=torch.float32)

        return self.semantic_fallback_scale * graph_bonus

    def forward(
        self,
        base_graph: RetrievalBatch,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-graph terminal log R(x).

        Returns
        -------
        log_reward : Tensor, shape (num_graphs,), dtype float32
            log R(x) for each graph. Guaranteed: values >= log_r_min.
        """
        num_graphs = base_graph.num_graphs
        node_batch = base_graph.batch
        dtype = torch.float32

        with torch.no_grad():
            target_mask = base_graph.is_target_mask
            active_gold = active_nodes & target_mask

            hits_per_graph = per_graph_mask_count(active_gold, node_batch, num_graphs, dtype=dtype)
            gold_per_graph = per_graph_mask_count(target_mask, node_batch, num_graphs, dtype=dtype)
            recall = torch.zeros(num_graphs, dtype=dtype, device=active_nodes.device)
            valid_gold = gold_per_graph.gt(0.0)
            recall[valid_gold] = (
                hits_per_graph[valid_gold] / gold_per_graph[valid_gold]
            )

            log_reward = torch.full_like(recall, self.log_r_min)
            positive_recall = recall.gt(0.0)
            log_reward[positive_recall] = recall[positive_recall].log()
            if bool((~positive_recall).any()):
                semantic_bonus = self._semantic_fallback_bonus(base_graph, active_nodes)
                log_reward[~positive_recall] = self.log_r_min + semantic_bonus[~positive_recall]

        log_reward = log_reward.clamp(min=self.log_r_min)
        return log_reward.to(dtype=torch.float32)

    @torch.no_grad()
    def _infer_active_nodes(
        self,
        base_graph: RetrievalBatch,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        active_nodes = base_graph.is_anchor_mask.clone()
        if active_edges.numel() == 0 or not bool(active_edges.any()):
            return active_nodes

        src, dst = base_graph.edge_index[0], base_graph.edge_index[1]
        active_src = src[active_edges]
        active_dst = dst[active_edges]
        active_nodes[active_src] = True
        active_nodes[active_dst] = True
        return active_nodes

    @torch.no_grad()
    def _relation_frontier_potential(
        self,
        base_graph: RetrievalBatch,
        active_edges: torch.Tensor,
        *,
        query_h: torch.Tensor,
        rel_h: torch.Tensor,
    ) -> torch.Tensor:
        if self.relation_shaping_scale == 0.0:
            return torch.zeros(
                base_graph.num_graphs,
                dtype=torch.float32,
                device=active_edges.device,
            )

        active_nodes = self._infer_active_nodes(base_graph, active_edges)
        src, dst = base_graph.edge_index[0], base_graph.edge_index[1]
        frontier_mask = (~active_edges) & (active_nodes[src] | active_nodes[dst])
        if not bool(frontier_mask.any()):
            return torch.zeros(
                base_graph.num_graphs,
                dtype=torch.float32,
                device=active_edges.device,
            )

        frontier_edge_ids = torch.nonzero(frontier_mask, as_tuple=False).view(-1)
        frontier_batch = base_graph.edge_batch.index_select(0, frontier_edge_ids)
        frontier_query_h = query_h.index_select(0, frontier_batch)
        frontier_rel_h = rel_h.index_select(0, frontier_edge_ids)
        frontier_scores = cosine_scores(frontier_query_h, frontier_rel_h)

        phi = torch.zeros(
            base_graph.num_graphs,
            dtype=torch.float32,
            device=active_edges.device,
        )
        for graph_id in range(base_graph.num_graphs):
            graph_mask = frontier_batch == graph_id
            if bool(graph_mask.any()):
                phi[graph_id] = frontier_scores[graph_mask].max().to(dtype=torch.float32)
        return self.relation_shaping_scale * phi

    @torch.no_grad()
    def _positive_coverage_potential(
        self,
        base_graph: RetrievalBatch,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        if self.positive_coverage_shaping_scale == 0.0:
            return torch.zeros(
                base_graph.num_graphs,
                dtype=torch.float32,
                device=active_edges.device,
            )
        positive_edges = base_graph.positive_edge_mask.to(dtype=torch.float32)
        active_positive = active_edges.to(dtype=torch.float32) * positive_edges
        hit_count = scatter_sum(
            active_positive,
            base_graph.edge_batch,
            dim=0,
            dim_size=base_graph.num_graphs,
        )
        total_positive = scatter_sum(
            positive_edges,
            base_graph.edge_batch,
            dim=0,
            dim_size=base_graph.num_graphs,
        ).clamp_min(1.0)
        return self.positive_coverage_shaping_scale * (hit_count / total_positive)

    @torch.no_grad()
    def _distance_progress_potential(
        self,
        base_graph: RetrievalBatch,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        if self.distance_progress_shaping_scale == 0.0:
            return torch.zeros(
                base_graph.num_graphs,
                dtype=torch.float32,
                device=active_edges.device,
            )
        active_nodes = self._infer_active_nodes(base_graph, active_edges)
        distance = base_graph.node_to_target_distance
        max_path = base_graph.max_path_length.to(device=active_edges.device).float()
        phi = torch.zeros(
            base_graph.num_graphs,
            dtype=torch.float32,
            device=active_edges.device,
        )
        for graph_id in range(base_graph.num_graphs):
            graph_node_mask = base_graph.batch == graph_id
            active_graph_nodes = active_nodes & graph_node_mask
            if not bool(active_graph_nodes.any().item()):
                continue
            active_distances = distance[active_graph_nodes]
            reachable = active_distances.ge(0)
            if not bool(reachable.any().item()):
                continue
            max_len = float(max(max_path[graph_id].item(), 1.0))
            min_distance = float(active_distances[reachable].min().item())
            phi[graph_id] = 1.0 - min_distance / max_len
        return self.distance_progress_shaping_scale * phi

    @torch.no_grad()
    def potential(
        self,
        base_graph: RetrievalBatch,
        active_edges: torch.Tensor,
        *,
        query_h: torch.Tensor,
        rel_h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-graph potential ϕ(s) for reward shaping.

        ϕ(s) = relation_shaping_scale · max_{e ∈ frontier(s)} cosine(q, rel(e))

        This is a pure function of the current state (active_edges), the fixed
        graph structure, and static projected query/relation features. It does
        NOT depend on the path taken to reach the state, so the shaped step reward

            r_t^shaped = ϕ(s_{t+1}) - ϕ(s_t)

        preserves flow conservation (Ng et al. 1999 potential-based shaping).

        Parameters
        ----------
        base_graph : RetrievalBatch
        active_edges : BoolTensor, shape [num_edges]
            Current active edge mask for the batch.
        query_h : FloatTensor, shape [num_graphs, hidden_dim]
            Shared-space projected query features.
        rel_h : FloatTensor, shape [num_edges, hidden_dim]
            Shared-space projected relation features.

        Returns
        -------
        phi : Tensor, shape (num_graphs,), dtype float32
            ϕ(s) per graph. Zero when relation_shaping_scale == 0.0 or when
            the graph has no frontier edges.
        """
        return (
            self._relation_frontier_potential(
                base_graph,
                active_edges,
                query_h=query_h,
                rel_h=rel_h,
            )
            + self._positive_coverage_potential(base_graph, active_edges)
            + self._distance_progress_potential(base_graph, active_edges)
        )

    @torch.no_grad()
    def step_shaping(
        self,
        base_graph: RetrievalBatch,
        active_edges_before: torch.Tensor,
        active_edges_after: torch.Tensor,
        *,
        query_h: torch.Tensor,
        rel_h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the shaped step reward: ϕ(s_{t+1}) - ϕ(s_t).

        Convenience wrapper around ``potential()``. Call this once per
        rollout step and store the result in the trajectory buffer as
        ``step_shaping`` (replacing the old ``step_bonus`` field).

        For a ``stop`` action, active_edges_before == active_edges_after, so
        the difference is automatically zero — no special-casing needed.

        Parameters
        ----------
        base_graph : RetrievalBatch
        active_edges_before : BoolTensor [num_edges]   ← s_t
        active_edges_after  : BoolTensor [num_edges]   ← s_{t+1}

        Returns
        -------
        shaping : Tensor, shape (num_graphs,), dtype float32
            ϕ(s_{t+1}) - ϕ(s_t), one scalar per graph.
        """
        phi_before = self.potential(
            base_graph,
            active_edges_before,
            query_h=query_h,
            rel_h=rel_h,
        )
        phi_after = self.potential(
            base_graph,
            active_edges_after,
            query_h=query_h,
            rel_h=rel_h,
        )
        return phi_after - phi_before


__all__ = ["RewardModel"]
