from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class _PreparedBatch:
    num_graphs: int
    num_nodes_total: int
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor
    question_emb_raw: torch.Tensor
    edge_embeddings_raw: torch.Tensor
    node_embeddings: torch.Tensor
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    context_tokens: torch.Tensor
    node_batch: torch.Tensor
    q_local_indices: torch.Tensor
    a_local_indices: torch.Tensor
    q_ptr: torch.Tensor
    a_ptr: torch.Tensor
    dummy_mask: torch.Tensor
    node_global_ids: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor
    sample_ids: list[str]
    start_nodes_fwd: torch.Tensor
    start_tokens_fwd: torch.Tensor
    edge_ids_by_head_fwd: torch.Tensor
    edge_ptr_by_head_fwd: torch.Tensor
    edge_ids_by_tail_fwd: torch.Tensor
    edge_ptr_by_tail_fwd: torch.Tensor
    edge_ids_by_head_bwd: torch.Tensor
    edge_ptr_by_head_bwd: torch.Tensor
    edge_ids_by_tail_bwd: torch.Tensor
    edge_ptr_by_tail_bwd: torch.Tensor
    edge_inverse_map: torch.Tensor


@dataclass
class _BeamState:
    beam_nodes: torch.Tensor
    beam_scores: torch.Tensor
    beam_paths: torch.Tensor
    beam_lengths: torch.Tensor
    beam_done: torch.Tensor
    flat_graph_ids: torch.Tensor
    flat_beam_ids: torch.Tensor
    beam_context: torch.Tensor
    beam_prev_rel: torch.Tensor
    num_graphs: int
    beam_size: int
    max_steps: int
    neg_inf: float


@dataclass
class _BeamCandidates:
    cand_scores: torch.Tensor
    cand_nodes: torch.Tensor
    cand_graph: torch.Tensor
    cand_src_beam: torch.Tensor
    cand_edge_id: torch.Tensor
    cand_is_edge: torch.Tensor
    cand_done: torch.Tensor


@dataclass
class _BeamCandidateMatrix:
    scores: torch.Tensor
    nodes: torch.Tensor
    src_beam: torch.Tensor
    edge_id: torch.Tensor
    is_edge: torch.Tensor
    done: torch.Tensor
    counts: torch.Tensor


@dataclass(frozen=True)
class _RolloutResult:
    log_pf_sum: torch.Tensor
    stop_nodes: torch.Tensor
    num_moves: torch.Tensor
    stop_reason: torch.Tensor
    actions: Optional[torch.Tensor]
    log_pf_steps: Optional[torch.Tensor]
    policy_metrics: Optional[dict[str, torch.Tensor]] = None


__all__ = [
    "_PreparedBatch",
    "_BeamState",
    "_BeamCandidates",
    "_BeamCandidateMatrix",
    "_RolloutResult",
]
