from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.nn_utils import (
    build_mlp,
    init_xavier,
    require_finite,
    zero_last_linear,
)


class _ProjectedDotScalar(nn.Module):
    """Projected dot-product scorer with a small residual MLP."""

    def __init__(
        self,
        hidden_dim: int,
        num_residual_layers: int = 1,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.score_scale = hidden_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residual = build_mlp(
            hidden_dim * 3,
            1,
            max(hidden_dim // 2, 1),
            num_residual_layers,
            dropout,
        )

        init_xavier(self.q_proj)
        init_xavier(self.k_proj)
        if zero_init:
            zero_last_linear(self.residual)

    def _score(self, ctx_q: torch.Tensor, ctx_s: torch.Tensor) -> torch.Tensor:
        projected_q = self.q_proj(ctx_q)
        projected_s = self.k_proj(ctx_s)
        bilinear = (projected_q * projected_s).sum(dim=-1) * self.score_scale
        residual = self.residual(
            torch.cat([ctx_q, ctx_s, ctx_q * ctx_s], dim=-1)
        ).squeeze(-1)
        return bilinear + residual


class ZHead(_ProjectedDotScalar):
    """log Z(q, s0)."""

    def forward(self, query_h: torch.Tensor, root_state_h: torch.Tensor) -> torch.Tensor:
        return self._score(query_h, root_state_h)


class FlowHead(_ProjectedDotScalar):
    """log F(s | q)."""

    def forward(self, query_h: torch.Tensor, state_h: torch.Tensor) -> torch.Tensor:
        return self._score(query_h, state_h)


@dataclass
class EdgeScorerInputs:
    """Inputs for ExpandEdgeScorer.forward.

    src_dyn_node_h : [E, H]  dynamic source-node state from backbone_out.node_h[src]
    edge_batch_index: [E]    graph index per edge
    dst_stat_node_h: [E, H]  static destination entity semantics from feature_bank.node_h[dst]
    rel_h          : [E, H]  relation semantics
    query_h        : [G, H]  query semantics
    """

    src_dyn_node_h: torch.Tensor
    edge_batch_index: torch.Tensor
    dst_stat_node_h: torch.Tensor
    rel_h: torch.Tensor
    query_h: torch.Tensor


@dataclass(frozen=True)
class EdgeScoreBreakdown:
    relation_only_logits: torch.Tensor
    residual_logits: torch.Tensor
    final_logits: torch.Tensor


class ExpandEdgeScorer(nn.Module):
    """Candidate-edge scorer.

    Design principle
    ----------------
    The task is retrieval: find the 1-2 semantically relevant edges among
    ~5000 candidates.  The dominant signal is semantic similarity between
    the query and the relation semantics.

    The relation-only cosine prior stays as the dominant zero-shot signal.
    A zero-initialized residual MLP consumes explicit first-principles edge
    factors ``[query, src_dyn, rel, dst_stat]`` to learn context-sensitive
    reranking without destroying the prior at initialization.
    """

    def __init__(
        self,
        hidden_dim: int,
        prior_scale_init: float = 5.0,
        prior_scale_trainable: bool = True,
        num_residual_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale_init)))
        self.prior_scale.requires_grad_(bool(prior_scale_trainable))
        self.residual_scorer = build_mlp(
            self.hidden_dim * 4,
            1,
            self.hidden_dim,
            num_residual_layers,
            dropout,
        )
        zero_last_linear(self.residual_scorer)

    def forward(
        self,
        inp: EdgeScorerInputs,
        *,
        return_breakdown: bool = False,
    ) -> torch.Tensor | EdgeScoreBreakdown:
        if inp.src_dyn_node_h.numel() == 0:
            empty = inp.src_dyn_node_h.new_zeros((0,))
            if not return_breakdown:
                return empty
            return EdgeScoreBreakdown(
                relation_only_logits=empty,
                residual_logits=empty,
                final_logits=empty,
            )

        query_h = inp.query_h
        rel_h = inp.rel_h

        query_per_edge = query_h.index_select(0, inp.edge_batch_index)  # [E, H]
        relation_only_logits = self.prior_scale * F.cosine_similarity(
            query_per_edge,
            rel_h,
            dim=-1,
        )
        residual_logits = self.residual_scorer(
            torch.cat(
                [query_per_edge, inp.src_dyn_node_h, inp.rel_h, inp.dst_stat_node_h],
                dim=-1,
            )
        ).squeeze(-1)
        final_logits = relation_only_logits + residual_logits
        if not return_breakdown:
            return final_logits
        return EdgeScoreBreakdown(
            relation_only_logits=relation_only_logits,
            residual_logits=residual_logits,
            final_logits=final_logits,
        )


class ActionHead(nn.Module):
    """Graph-level action-type scorer returning (B, 2) logits.

    Column convention: col 0 = expand, col 1 = stop.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        type_feature_dim: int = 0,
        zero_init_type_output: bool = True,
    ) -> None:
        super().__init__()
        if type_feature_dim < 0:
            raise ValueError(f"type_feature_dim must be >= 0, got {type_feature_dim}.")
        self.type_feature_dim = int(type_feature_dim)
        self.type_scorer = build_mlp(
            hidden_dim + self.type_feature_dim,
            2,
            hidden_dim,
            num_layers,
            dropout,
        )
        if zero_init_type_output:
            zero_last_linear(self.type_scorer)

    def forward(
        self,
        state_h: torch.Tensor,
        type_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        finite_state_h = require_finite(state_h, name="action_state_h")
        if self.type_feature_dim > 0:
            if type_features is None:
                raise ValueError(f"type_feature_dim={self.type_feature_dim} but type_features is None.")
            if type_features.shape != (state_h.size(0), self.type_feature_dim):
                raise ValueError(
                    f"type_features shape mismatch: expected " f"({state_h.size(0)}, {self.type_feature_dim}), " f"got {tuple(type_features.shape)}."
                )
            type_ctx = torch.cat(
                [finite_state_h, require_finite(type_features.to(finite_state_h.dtype), name="type_features")],
                dim=-1,
            )
        else:
            type_ctx = finite_state_h
        return {"type_logits": self.type_scorer(type_ctx)}


def build_edge_scorer_inputs(
    backbone_out,
    edge_index: torch.Tensor,
    edge_batch_index: torch.Tensor,
) -> EdgeScorerInputs:
    """Construct EdgeScorerInputs from a BackboneOutput.

    Single authoritative mapping from backbone tensors to scorer fields.
    Call once per rollout step.
    """
    src = edge_index[0]
    dst = edge_index[1]
    return EdgeScorerInputs(
        src_dyn_node_h=backbone_out.node_h.index_select(0, src),
        edge_batch_index=edge_batch_index,
        dst_stat_node_h=backbone_out.feature_bank.node_h.index_select(0, dst),
        rel_h=backbone_out.rel_h,
        query_h=backbone_out.query_h,
    )


__all__ = [
    "ActionHead",
    "EdgeScoreBreakdown",
    "EdgeScorerInputs",
    "ExpandEdgeScorer",
    "FlowHead",
    "ZHead",
    "build_edge_scorer_inputs",
]
 
