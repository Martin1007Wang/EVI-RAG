from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.state import State

from .feature_encoder import FeatureBank


@dataclass(frozen=True, slots=True)
class CandidateContext:
    """
    State-local candidate edge coordinates shared by policy scoring.

    graph_id is the dynamic state row used for policy grouping. In the physical
    path it is also the original graph id. In fused static rollouts,
    static_graph_id stores the original graph id used to index FeatureBank query
    tensors.
    """

    edge_ids: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    graph_id: torch.Tensor
    src_active: torch.Tensor
    dst_active: torch.Tensor
    static_graph_id: torch.Tensor | None = None

    @property
    def num_candidates(self) -> int:
        return int(self.edge_ids.numel())


@dataclass(frozen=True, slots=True)
class CandidateSemanticScores:
    query_relation_score: torch.Tensor
    query_src_node_score: torch.Tensor
    query_dst_node_score: torch.Tensor
    query_new_node_score: torch.Tensor
    new_text_mask: torch.Tensor


def build_candidate_context(
    *,
    batch: RetrievalBatch,
    state: State,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor | None = None,
    candidate_static_graph_ids: torch.Tensor | None = None,
    device: torch.device,
) -> CandidateContext:
    edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)

    empty_long = torch.empty((0,), dtype=torch.long, device=device)
    empty_bool = torch.empty((0,), dtype=torch.bool, device=device)
    if edge_ids.numel() == 0:
        return CandidateContext(
            edge_ids=edge_ids,
            src=empty_long,
            dst=empty_long,
            graph_id=empty_long,
            src_active=empty_bool,
            dst_active=empty_bool,
            static_graph_id=empty_long,
        )

    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    src = edge_index[0].index_select(0, edge_ids)
    dst = edge_index[1].index_select(0, edge_ids)
    if candidate_batch_ids is None:
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
        graph_id = edge_batch.index_select(0, edge_ids)
    else:
        graph_id = candidate_batch_ids.to(device=device, dtype=torch.long).view(-1)
        if graph_id.shape != edge_ids.shape:
            raise ValueError(
                "candidate_batch_ids must have the same shape as candidate_edge_ids: "
                f"{tuple(graph_id.shape)} != {tuple(edge_ids.shape)}."
            )

    active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
    if active_nodes.ndim == 1:
        src_active = active_nodes.index_select(0, src)
        dst_active = active_nodes.index_select(0, dst)
    elif active_nodes.ndim == 2:
        src_active = active_nodes[graph_id, src]
        dst_active = active_nodes[graph_id, dst]
    else:
        raise ValueError(
            "state.active_nodes must have shape [N] or [R, N], "
            f"got {tuple(active_nodes.shape)}."
        )

    static_graph_id: torch.Tensor | None
    if candidate_static_graph_ids is not None:
        static_graph_id = candidate_static_graph_ids.to(
            device=device,
            dtype=torch.long,
        ).view(-1)
        if static_graph_id.shape != edge_ids.shape:
            raise ValueError(
                "candidate_static_graph_ids must have the same shape as "
                "candidate_edge_ids: "
                f"{tuple(static_graph_id.shape)} != {tuple(edge_ids.shape)}."
            )
    elif active_nodes.ndim == 2 and hasattr(state, "rollout_to_graph"):
        rollout_to_graph = state.rollout_to_graph.to(device=device, dtype=torch.long)
        static_graph_id = rollout_to_graph.index_select(0, graph_id)
    else:
        static_graph_id = graph_id

    return CandidateContext(
        edge_ids=edge_ids,
        src=src,
        dst=dst,
        graph_id=graph_id,
        src_active=src_active,
        dst_active=dst_active,
        static_graph_id=static_graph_id,
    )


def candidate_semantic_scores(
    *,
    fb: FeatureBank,
    candidates: CandidateContext,
) -> CandidateSemanticScores:
    device = fb.query_sem_h.device
    edge_ids = candidates.edge_ids.to(device=device, dtype=torch.long)
    src = candidates.src.to(device=device, dtype=torch.long)
    dst = candidates.dst.to(device=device, dtype=torch.long)
    graph_id = candidates.graph_id.to(device=device, dtype=torch.long)
    static_graph_id = (
        candidates.static_graph_id.to(device=device, dtype=torch.long)
        if candidates.static_graph_id is not None
        else graph_id
    )

    if edge_ids.numel() == 0:
        empty_score = fb.query_sem_h.new_zeros((0,))
        empty_mask = torch.zeros((0,), dtype=torch.bool, device=device)
        return CandidateSemanticScores(
            query_relation_score=empty_score,
            query_src_node_score=empty_score,
            query_dst_node_score=empty_score,
            query_new_node_score=empty_score,
            new_text_mask=empty_mask,
        )

    query_sem_h = fb.query_sem_h.index_select(0, static_graph_id)
    rel_sem_h = fb.rel_sem_h.index_select(0, edge_ids)
    src_sem_h = fb.node_sem_h.index_select(0, src)
    dst_sem_h = fb.node_sem_h.index_select(0, dst)

    query_relation_score = (query_sem_h * rel_sem_h).sum(dim=-1)
    query_src_node_score = (query_sem_h * src_sem_h).sum(dim=-1)
    query_dst_node_score = (query_sem_h * dst_sem_h).sum(dim=-1)

    if fb.node_is_non_text is None:
        new_text_mask = torch.zeros(src.shape, dtype=torch.bool, device=device)
        query_new_node_score = query_sem_h.new_zeros((src.numel(),))
    else:
        node_is_non_text = fb.node_is_non_text.to(device=device, dtype=torch.bool)
        src_is_text = ~node_is_non_text.index_select(0, src)
        dst_is_text = ~node_is_non_text.index_select(0, dst)

        src_active = candidates.src_active.to(device=device, dtype=torch.bool)
        dst_active = candidates.dst_active.to(device=device, dtype=torch.bool)
        src_is_new_text = dst_active & ~src_active & src_is_text
        dst_is_new_text = src_active & ~dst_active & dst_is_text
        new_text_mask = src_is_new_text | dst_is_new_text
        query_new_node_score = torch.where(
            src_is_new_text,
            query_src_node_score,
            torch.where(
                dst_is_new_text,
                query_dst_node_score,
                query_sem_h.new_zeros((src.numel(),)),
            ),
        )

    return CandidateSemanticScores(
        query_relation_score=query_relation_score,
        query_src_node_score=query_src_node_score,
        query_dst_node_score=query_dst_node_score,
        query_new_node_score=query_new_node_score,
        new_text_mask=new_text_mask,
    )


__all__ = [
    "CandidateContext",
    "CandidateSemanticScores",
    "build_candidate_context",
    "candidate_semantic_scores",
]
