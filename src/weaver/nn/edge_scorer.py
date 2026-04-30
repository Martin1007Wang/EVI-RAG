from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.nn_utils import zero_last_linear

from .backbone import FeatureBank
from .state_readout import EvidenceContext


@dataclass(frozen=True)
class EdgeScoreBreakdown:
    """
    Candidate-edge semantic diagnostics.

    q_rel:
        cos(q, relation). Main ranking signal.

    q_new:
        cos(q, new_text_endpoint), zero when the new endpoint has no text feature.

    q_candidate:
        q_rel + alpha * q_new. This is the scalar semantic score before logit scale.

    new_text_mask:
        1 if the candidate introduces a text endpoint, else 0.

    semantic_logits / final_logits:
        semantic_logits is the semantic prior. final_logits may also include
        a state-conditioned residual.
    """

    q_rel: torch.Tensor
    q_new: torch.Tensor
    q_candidate: torch.Tensor
    new_text_mask: torch.Tensor
    semantic_logits: torch.Tensor
    final_logits: torch.Tensor
    prior_logits: torch.Tensor | None = None
    residual_logits: torch.Tensor | None = None


class ExpandEdgeScorer(nn.Module):
    """
    Semantic-prior plus state-conditioned residual scorer for Expand actions.

    For candidate edge e=(u,r,v), under current state s:

        known endpoint = endpoint already in V_s
        new endpoint   = endpoint newly introduced by e

    Main assumption:
        question semantics primarily determine which relation should be followed.

        Semantic prior:

        prior(e) = tau * (
            <h_q_sem, h_r_sem>
            + alpha * 1[new endpoint is text] * <h_q_sem, h_new_sem>
        )

        Residual:

        residual(e) = MLP([h_q, h_s, EdgeEnc(h_u, h_r, h_v, b_e)])

    Final logit:

        logit(e) = prior(e) + lambda_res * residual(e)

    The semantic prior keeps the initial policy close to relation-first frontier
    ranking; the residual lets the current evidence graph influence transitions.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        entity_weight_init: float = 0.1,
        logit_scale_init: float = 10.0,
        residual_scale_init: float = 0.1,
        residual_dropout: float = 0.0,
        trainable_scales: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.entity_weight = nn.Parameter(torch.tensor(float(entity_weight_init)))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.use_residual = bool(use_residual)

        self.entity_weight.requires_grad_(bool(trainable_scales))
        self.logit_scale.requires_grad_(bool(trainable_scales))
        self.residual_scale.requires_grad_(bool(trainable_scales))

        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 + 5, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(residual_dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(residual_dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        zero_last_linear(self.residual_head)

    def forward(
        self,
        *,
        fb: FeatureBank,
        state_h: torch.Tensor,
        context: EvidenceContext | None = None,
        edge_index: torch.Tensor,
        edge_batch_index: torch.Tensor,
        active_nodes: torch.Tensor,
        candidate_edge_ids: torch.Tensor,
        graph_expand_ratio: torch.Tensor | None = None,
        node_is_non_text: torch.Tensor | None = None,
        return_breakdown: bool = False,
    ) -> torch.Tensor | EdgeScoreBreakdown:
        del graph_expand_ratio

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
            return EdgeScoreBreakdown(
                q_rel=empty,
                q_new=empty,
                q_candidate=empty,
                new_text_mask=empty,
                semantic_logits=empty,
                final_logits=empty,
                prior_logits=empty,
                residual_logits=empty,
            )

        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_batch_index = edge_batch_index.to(device=device, dtype=torch.long)
        active_nodes = active_nodes.to(device=device, dtype=torch.bool)
        anchor_mask = fb.anchor_mask.to(device=device, dtype=torch.bool)

        src = edge_index[0].index_select(0, candidate_edge_ids)
        dst = edge_index[1].index_select(0, candidate_edge_ids)
        graph_id = edge_batch_index.index_select(0, candidate_edge_ids)

        src_active = active_nodes.index_select(0, src)
        dst_active = active_nodes.index_select(0, dst)

        src_sem_h = fb.node_sem_h.index_select(0, src)
        dst_sem_h = fb.node_sem_h.index_select(0, dst)
        rel_sem_h = fb.rel_sem_h.index_select(0, candidate_edge_ids)
        query_sem_h = fb.query_sem_h.index_select(0, graph_id)

        src_text, dst_text = text_endpoint_masks(
            node_is_non_text=node_is_non_text,
            src=src,
            dst=dst,
            like=query_sem_h[:, 0],
        )

        src_new_text = dst_active & ~src_active & src_text.bool()
        dst_new_text = src_active & ~dst_active & dst_text.bool()

        new_sem_h = torch.zeros_like(rel_sem_h)
        new_sem_h[src_new_text] = src_sem_h[src_new_text]
        new_sem_h[dst_new_text] = dst_sem_h[dst_new_text]

        new_text_mask = (src_new_text | dst_new_text).to(dtype=dtype)

        q_rel = (query_sem_h * rel_sem_h).sum(dim=-1)
        q_new = (query_sem_h * new_sem_h).sum(dim=-1) * new_text_mask

        q_candidate = q_rel + self.entity_weight.to(dtype=dtype) * q_new
        semantic_logits = self.logit_scale.to(dtype=dtype) * q_candidate

        if self.use_residual:
            src_h = fb.node_h.index_select(0, src)
            dst_h = fb.node_h.index_select(0, dst)
            rel_h = fb.rel_h.index_select(0, candidate_edge_ids)
            query_h = fb.query_h.index_select(0, graph_id)

            if context is not None:
                state_h_by_edge = context.state_h.index_select(0, graph_id)
            else:
                state_h_by_edge = state_h.index_select(0, graph_id)

            endpoint_flags = torch.stack(
                [
                    src_active.to(dtype=dtype),
                    dst_active.to(dtype=dtype),
                    anchor_mask.index_select(0, src).to(dtype=dtype),
                    anchor_mask.index_select(0, dst).to(dtype=dtype),
                    new_text_mask,
                ],
                dim=-1,
            )

            candidate_h = self.candidate_encoder(
                torch.cat([src_h, rel_h, dst_h, endpoint_flags], dim=-1)
            )
            residual_logits = self.residual_head(
                torch.cat([query_h, state_h_by_edge, candidate_h], dim=-1)
            ).squeeze(-1)

            final_logits = semantic_logits + self.residual_scale.to(
                dtype=dtype
            ) * residual_logits.to(dtype=dtype)
        else:
            residual_logits = torch.zeros_like(semantic_logits)
            final_logits = semantic_logits

        if not return_breakdown:
            return final_logits

        return EdgeScoreBreakdown(
            q_rel=q_rel,
            q_new=q_new,
            q_candidate=q_candidate,
            new_text_mask=new_text_mask,
            semantic_logits=semantic_logits,
            final_logits=final_logits,
            prior_logits=semantic_logits,
            residual_logits=residual_logits,
        )


def text_endpoint_masks(
    *,
    node_is_non_text: torch.Tensor | None,
    src: torch.Tensor,
    dst: torch.Tensor,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if node_is_non_text is None:
        ones = torch.ones_like(like)
        return ones, ones

    node_is_non_text = node_is_non_text.to(device=like.device, dtype=torch.bool)

    src_text = ~node_is_non_text.index_select(0, src)
    dst_text = ~node_is_non_text.index_select(0, dst)

    return src_text.to(dtype=like.dtype), dst_text.to(dtype=like.dtype)


__all__ = [
    "EdgeScoreBreakdown",
    "ExpandEdgeScorer",
]
