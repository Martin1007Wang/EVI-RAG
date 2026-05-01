from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.schema import RetrievalBatch
from src.weaver.state import State

from .candidate_context import (
    CandidateContext,
    build_candidate_context,
    candidate_semantic_scores,
)
from .feature_encoder import FeatureBank
from .state_readout import StateContext


@dataclass(frozen=True, slots=True)
class TransitionFeatureOutput:
    values: torch.Tensor
    names: tuple[str, ...]
    query_relation_score: torch.Tensor
    query_src_node_score: torch.Tensor
    query_dst_node_score: torch.Tensor
    query_new_node_score: torch.Tensor
    new_text_mask: torch.Tensor


class TransitionFeatureBuilder(nn.Module):
    """
    Encode only the MDP transition type for candidate edges.

    Semantic relevance belongs to the detached base logit. Structural context
    belongs to node/edge/state representations. This module only tells the
    residual whether a frontier edge expands from src, expands from dst, or
    closes an internal edge.
    """

    names = ("src_active_dst_new", "dst_active_src_new", "both_active")

    def __init__(
        self,
        *,
        hidden_dim: int,
        dde_dim: int = 0,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        # Kept only because Policy knows the encoder DDE width at construction.
        # DDE already enters node_h through FeatureEncoder structural bias.
        self.dde_dim = max(0, int(dde_dim))

    @property
    def feature_dim(self) -> int:
        return len(self.names)

    def forward(
        self,
        *,
        fb: FeatureBank,
        context: StateContext,
        batch: RetrievalBatch,
        state: State,
        candidate_edge_ids: torch.Tensor,
        candidate_context: CandidateContext | None = None,
    ) -> TransitionFeatureOutput:
        del context

        device = fb.node_h.device
        dtype = fb.node_h.dtype

        candidates = candidate_context or build_candidate_context(
            batch=batch,
            state=state,
            candidate_edge_ids=candidate_edge_ids,
            device=device,
        )

        empty_bool = torch.zeros((0,), dtype=torch.bool, device=device)
        empty_score = fb.node_h.new_zeros((0,))
        if candidates.num_candidates == 0:
            return TransitionFeatureOutput(
                values=fb.node_h.new_zeros((0, self.feature_dim)),
                names=self.names,
                query_relation_score=empty_score,
                query_src_node_score=empty_score,
                query_dst_node_score=empty_score,
                query_new_node_score=empty_score,
                new_text_mask=empty_bool,
            )

        src_active = candidates.src_active.to(device=device, dtype=torch.bool)
        dst_active = candidates.dst_active.to(device=device, dtype=torch.bool)
        frontier_valid = src_active | dst_active
        if not bool(frontier_valid.all()):
            raise RuntimeError(
                "TransitionFeatureBuilder received a non-frontier edge: every "
                "candidate must have at least one active endpoint."
            )

        values = torch.stack(
            [
                src_active & ~dst_active,
                dst_active & ~src_active,
                src_active & dst_active,
            ],
            dim=-1,
        ).to(dtype=dtype)

        semantic = candidate_semantic_scores(fb=fb, candidates=candidates)
        return TransitionFeatureOutput(
            values=values,
            names=self.names,
            query_relation_score=semantic.query_relation_score,
            query_src_node_score=semantic.query_src_node_score,
            query_dst_node_score=semantic.query_dst_node_score,
            query_new_node_score=semantic.query_new_node_score,
            new_text_mask=semantic.new_text_mask,
        )


__all__ = ["TransitionFeatureBuilder", "TransitionFeatureOutput"]
