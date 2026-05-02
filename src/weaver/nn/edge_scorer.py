from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .candidate_context import CandidateContext, candidate_semantic_scores
from .feature_encoder import FeatureBank
from .state_readout import StateContext


@dataclass(frozen=True, slots=True)
class EdgeScoreBreakdown:
    query_relation_score: torch.Tensor
    query_new_node_score: torch.Tensor
    semantic_score: torch.Tensor
    new_text_mask: torch.Tensor
    semantic_logits: torch.Tensor
    final_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class _EdgeBaseMeasure:
    query_relation_score: torch.Tensor
    query_new_node_score: torch.Tensor
    semantic_score: torch.Tensor
    new_text_mask: torch.Tensor
    semantic_logits: torch.Tensor


class EdgeScorer(nn.Module):
    """
    Semantic-prior scorer for candidate Expand edges.

    For candidate edge e=(u,r,v):

        semantic(e)
            = <q_sem, r_sem>
              + alpha * 1[new endpoint is text] * <q_sem, new_sem>

        logit(e | s) = tau * semantic(e)

    This scorer intentionally has no state-conditioned residual branch. It
    keeps the same high-level forward contract as the policy scorer while making
    semantic logits and final logits identical.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        type: str = "semantic_prior",
        entity_weight_init: float = 0.1,
        logit_scale_init: float = 5.0,
        trainable_entity_weight: bool = True,
        trainable_logit_scale: bool = True,
        **removed_kwargs: Any,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.scorer_type = str(type)
        if self.scorer_type != "semantic_prior":
            raise ValueError(
                "edge_scorer.type must be 'semantic_prior', " f"got {type!r}."
            )

        if removed_kwargs:
            raise ValueError(
                "edge_scorer no longer accepts residual config keys: "
                f"{sorted(removed_kwargs)}."
            )

        self.entity_weight = nn.Parameter(torch.tensor(float(entity_weight_init)))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.entity_weight.requires_grad_(bool(trainable_entity_weight))
        self.logit_scale.requires_grad_(bool(trainable_logit_scale))

    def forward(
        self,
        *,
        fb: FeatureBank,
        context: StateContext,
        edge_index: torch.Tensor,
        edge_batch_index: torch.Tensor,
        active_nodes: torch.Tensor,
        candidate_edge_ids: torch.Tensor,
        candidate_context: CandidateContext | None = None,
        return_breakdown: bool = False,
    ) -> torch.Tensor | EdgeScoreBreakdown:
        del context

        device = fb.node_h.device
        dtype = fb.node_h.dtype

        candidate_edge_ids = candidate_edge_ids.to(
            device=device,
            dtype=torch.long,
        ).view(-1)

        if candidate_edge_ids.numel() == 0:
            empty = fb.node_h.new_zeros((0,))
            if not return_breakdown:
                return empty
            empty_bool = torch.zeros((0,), dtype=torch.bool, device=device)
            return EdgeScoreBreakdown(
                query_relation_score=empty,
                query_new_node_score=empty,
                semantic_score=empty,
                new_text_mask=empty_bool,
                semantic_logits=empty,
                final_logits=empty,
            )

        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_batch_index = edge_batch_index.to(device=device, dtype=torch.long)
        active_nodes = active_nodes.to(device=device, dtype=torch.bool)

        candidates = candidate_context or CandidateContext(
            edge_ids=candidate_edge_ids,
            src=edge_index[0].index_select(0, candidate_edge_ids),
            dst=edge_index[1].index_select(0, candidate_edge_ids),
            graph_id=edge_batch_index.index_select(0, candidate_edge_ids),
            src_active=active_nodes.index_select(
                0,
                edge_index[0].index_select(0, candidate_edge_ids),
            ),
            dst_active=active_nodes.index_select(
                0,
                edge_index[1].index_select(0, candidate_edge_ids),
            ),
        )

        base = self._base_measure(
            fb=fb,
            candidates=candidates,
            num_candidates=candidate_edge_ids.numel(),
            device=device,
            dtype=dtype,
        )
        final_logits = base.semantic_logits

        if not return_breakdown:
            return final_logits

        return EdgeScoreBreakdown(
            query_relation_score=base.query_relation_score,
            query_new_node_score=base.query_new_node_score,
            semantic_score=base.semantic_score,
            new_text_mask=base.new_text_mask,
            semantic_logits=base.semantic_logits,
            final_logits=final_logits,
        )

    def _base_measure(
        self,
        *,
        fb: FeatureBank,
        candidates: CandidateContext,
        num_candidates: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _EdgeBaseMeasure:
        semantic = candidate_semantic_scores(fb=fb, candidates=candidates)
        query_relation_score = semantic.query_relation_score.to(
            device=device,
            dtype=dtype,
        )
        query_new_node_score = semantic.query_new_node_score.to(
            device=device,
            dtype=dtype,
        )
        new_text_mask = semantic.new_text_mask.to(device=device, dtype=torch.bool)

        if query_relation_score.shape != (int(num_candidates),):
            raise ValueError(
                "query_relation_score must have shape "
                f"[{int(num_candidates)}], got "
                f"{tuple(query_relation_score.shape)}."
            )
        if query_new_node_score.shape != (int(num_candidates),):
            raise ValueError(
                "query_new_node_score must have shape "
                f"[{int(num_candidates)}], got "
                f"{tuple(query_new_node_score.shape)}."
            )

        semantic_score = (
            query_relation_score
            + self.entity_weight.to(device=device, dtype=dtype) * query_new_node_score
        )
        semantic_logits = (
            self.logit_scale.to(device=device, dtype=dtype) * semantic_score
        )
        return _EdgeBaseMeasure(
            query_relation_score=query_relation_score,
            query_new_node_score=query_new_node_score,
            semantic_score=semantic_score,
            new_text_mask=new_text_mask,
            semantic_logits=semantic_logits,
        )


__all__ = [
    "EdgeScoreBreakdown",
    "EdgeScorer",
]
