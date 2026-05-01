from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.nn_utils import zero_last_linear

from .candidate_context import CandidateContext, candidate_semantic_scores
from .edge_encoder import EdgeEncoder
from .feature_encoder import FeatureBank
from .state_readout import StateContext
from .transition_features import TransitionFeatureOutput


@dataclass(frozen=True, slots=True)
class EdgeScoreBreakdown:
    query_relation_score: torch.Tensor
    query_new_node_score: torch.Tensor
    semantic_score: torch.Tensor
    new_text_mask: torch.Tensor
    semantic_logits: torch.Tensor
    residual_logits: torch.Tensor
    final_logits: torch.Tensor
    residual_scale: torch.Tensor


@dataclass(frozen=True, slots=True)
class _EdgeBaseMeasure:
    query_relation_score: torch.Tensor
    query_new_node_score: torch.Tensor
    semantic_score: torch.Tensor
    new_text_mask: torch.Tensor
    semantic_logits: torch.Tensor


class EdgeScorer(nn.Module):
    """
    Semantic-prior residual scorer for candidate Expand edges.

    For candidate edge e=(u,r,v):

        semantic(e)
            = <q_sem, r_sem>
              + alpha * 1[new endpoint is text] * <q_sem, new_sem>

        phi_E(e)
            = W_E [h_u, h_r, h_v]

        residual(e | s)
            = MLP([h_s, phi_E(e), transition_type(e,s),
                   stop_gradient(semantic_logit(e))])

        logit(e | s)
            = tau * semantic(e) + lambda_eff * residual(e | s)

    The semantic prior is the base logit, not a weak feature. The residual is
    initialized and scheduled so early policy rankings preserve the prior.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_encoder: EdgeEncoder | None = None,
        type: str = "semantic_prior_residual",
        action_feature_dim: int = 0,
        transition_feature_dim: int | None = None,
        action_dropout: float = 0.0,
        entity_weight_init: float = 0.1,
        logit_scale_init: float = 5.0,
        residual_scale_init: float = 1.0,
        use_residual: bool = True,
        trainable_entity_weight: bool = True,
        trainable_logit_scale: bool = True,
        trainable_residual: bool = True,
        residual_warmup_start_step: int = 0,
        residual_warmup_steps: int = 0,
        residual_max_multiplier: float = 1.0,
        freeze_residual_until_step: int = 0,
        zero_init_residual_output: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.scorer_type = str(type)
        if self.scorer_type != "semantic_prior_residual":
            raise ValueError(
                "edge_scorer.type must be 'semantic_prior_residual', " f"got {type!r}."
            )

        self.edge_encoder = edge_encoder or EdgeEncoder(hidden_dim=self.hidden_dim)
        legacy_action_feature_dim = _non_negative_int(
            action_feature_dim,
            "action_feature_dim",
        )
        if transition_feature_dim is None:
            transition_feature_dim = legacy_action_feature_dim
        else:
            transition_feature_dim = _non_negative_int(
                transition_feature_dim,
                "transition_feature_dim",
            )
            if (
                legacy_action_feature_dim != 0
                and legacy_action_feature_dim != transition_feature_dim
            ):
                raise ValueError(
                    "action_feature_dim and transition_feature_dim disagree: "
                    f"{legacy_action_feature_dim} != {transition_feature_dim}."
                )
        self.transition_feature_dim = int(transition_feature_dim)
        self.action_feature_dim = self.transition_feature_dim
        self.residual_features = ("state", "edge", "transition_type", "semantic_prior")
        self.use_residual = bool(use_residual)
        self._trainable_residual = bool(trainable_residual)
        self.residual_warmup_start_step = _non_negative_int(
            residual_warmup_start_step,
            "residual_warmup_start_step",
        )
        self.residual_warmup_steps = _non_negative_int(
            residual_warmup_steps,
            "residual_warmup_steps",
        )
        self.residual_max_multiplier = _non_negative_float(
            residual_max_multiplier,
            "residual_max_multiplier",
        )
        self.freeze_residual_until_step = _non_negative_int(
            freeze_residual_until_step,
            "freeze_residual_until_step",
        )
        self.register_buffer(
            "_residual_scale_multiplier",
            torch.tensor(
                self._scheduled_residual_multiplier(step=0),
                dtype=torch.float32,
            ),
        )

        self.entity_weight = nn.Parameter(torch.tensor(float(entity_weight_init)))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.entity_weight.requires_grad_(bool(trainable_entity_weight))
        self.logit_scale.requires_grad_(bool(trainable_logit_scale))
        self.residual_scale.requires_grad_(self._trainable_residual)

        if not bool(zero_init_residual_output):
            raise ValueError(
                "zero_init_residual_output=false is not supported. The residual "
                "head must be zero-initialized so step-0 logits preserve the "
                "semantic prior."
            )

        self.residual_head = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 2 + self.transition_feature_dim + 1,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(float(action_dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        zero_last_linear(self.residual_head)
        for parameter in self.residual_head.parameters():
            parameter.requires_grad_(self._trainable_residual)

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
        transition_features: TransitionFeatureOutput | None = None,
        action_features: TransitionFeatureOutput | None = None,
        return_breakdown: bool = False,
    ) -> torch.Tensor | EdgeScoreBreakdown:
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
                residual_logits=empty,
                final_logits=empty,
                residual_scale=self.effective_residual_scale(
                    device=device,
                    dtype=dtype,
                ),
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
        graph_id = candidates.graph_id.to(device=device, dtype=torch.long)

        if transition_features is not None and action_features is not None:
            raise ValueError(
                "Pass only transition_features; action_features is a legacy alias."
            )
        if transition_features is None:
            transition_features = action_features

        base = self._base_measure(
            fb=fb,
            candidates=candidates,
            num_candidates=candidate_edge_ids.numel(),
            transition_features=transition_features,
            device=device,
            dtype=dtype,
        )

        edge_h = self._edge_h(
            fb=fb,
            context=context,
            candidates=candidates,
        )

        residual_logits = self._residual_logits_from_edge_h(
            fb=fb,
            context=context,
            edge_h=edge_h,
            graph_id=graph_id,
            semantic_logits=base.semantic_logits,
            transition_features=transition_features,
        )

        residual_scale = self.effective_residual_scale(device=device, dtype=dtype)
        if not self.use_residual:
            residual_logits = torch.zeros_like(residual_logits)
        final_logits = base.semantic_logits + residual_scale * residual_logits

        if not return_breakdown:
            return final_logits

        return EdgeScoreBreakdown(
            query_relation_score=base.query_relation_score,
            query_new_node_score=base.query_new_node_score,
            semantic_score=base.semantic_score,
            new_text_mask=base.new_text_mask,
            semantic_logits=base.semantic_logits,
            residual_logits=residual_logits,
            final_logits=final_logits,
            residual_scale=residual_scale,
        )

    def _base_measure(
        self,
        *,
        fb: FeatureBank,
        candidates: CandidateContext,
        num_candidates: int,
        transition_features: TransitionFeatureOutput | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _EdgeBaseMeasure:
        if transition_features is None:
            semantic = candidate_semantic_scores(fb=fb, candidates=candidates)
            query_relation_score = semantic.query_relation_score
            query_new_node_score = semantic.query_new_node_score
            new_text_mask = semantic.new_text_mask
        else:
            query_relation_score = transition_features.query_relation_score.to(
                device=device,
                dtype=dtype,
            )
            query_new_node_score = transition_features.query_new_node_score.to(
                device=device,
                dtype=dtype,
            )
            new_text_mask = transition_features.new_text_mask.to(
                device=device,
                dtype=torch.bool,
            )
            if query_relation_score.shape != (int(num_candidates),):
                raise ValueError(
                    "transition_features.query_relation_score must have shape "
                    f"[{int(num_candidates)}], got "
                    f"{tuple(query_relation_score.shape)}."
                )
            if query_new_node_score.shape != (int(num_candidates),):
                raise ValueError(
                    "transition_features.query_new_node_score must have shape "
                    f"[{int(num_candidates)}], got "
                    f"{tuple(query_new_node_score.shape)}."
                )

        semantic_score = (
            query_relation_score
            + self.entity_weight.to(device=device, dtype=dtype) * query_new_node_score
        )
        semantic_logits = self.logit_scale.to(device=device, dtype=dtype) * semantic_score
        return _EdgeBaseMeasure(
            query_relation_score=query_relation_score,
            query_new_node_score=query_new_node_score,
            semantic_score=semantic_score,
            new_text_mask=new_text_mask,
            semantic_logits=semantic_logits,
        )

    def _edge_h(
        self,
        *,
        fb: FeatureBank,
        context: StateContext,
        candidates: CandidateContext,
    ) -> torch.Tensor:
        cached = self._cached_frontier_edge_h(
            context=context,
            candidates=candidates,
            device=fb.node_h.device,
            dtype=fb.node_h.dtype,
        )
        if cached is not None:
            return cached

        src = candidates.src.to(device=fb.node_h.device, dtype=torch.long)
        dst = candidates.dst.to(device=fb.node_h.device, dtype=torch.long)
        edge_ids = candidates.edge_ids.to(device=fb.node_h.device, dtype=torch.long)

        src_h = fb.node_h.index_select(0, src)
        rel_h = fb.rel_h.index_select(0, edge_ids)
        dst_h = fb.node_h.index_select(0, dst)

        return self.edge_encoder(
            src_h=src_h,
            rel_h=rel_h,
            dst_h=dst_h,
        )

    @staticmethod
    def _cached_frontier_edge_h(
        *,
        context: StateContext,
        candidates: CandidateContext,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if context.frontier_edge_ids is None or context.frontier_edge_h is None:
            return None

        cached_ids = context.frontier_edge_ids.to(device=device, dtype=torch.long)
        edge_ids = candidates.edge_ids.to(device=device, dtype=torch.long)
        if cached_ids.shape != edge_ids.shape:
            return None
        if not bool(torch.equal(cached_ids, edge_ids)):
            return None

        return context.frontier_edge_h.to(device=device, dtype=dtype)

    def _residual_logits_from_edge_h(
        self,
        *,
        fb: FeatureBank,
        context: StateContext,
        edge_h: torch.Tensor,
        graph_id: torch.Tensor,
        semantic_logits: torch.Tensor,
        transition_features: TransitionFeatureOutput | None,
    ) -> torch.Tensor:
        state_h = context.state_h.index_select(0, graph_id)
        semantic_prior_h = (
            semantic_logits.detach()
            .to(
                device=fb.node_h.device,
                dtype=fb.node_h.dtype,
            )
            .unsqueeze(-1)
        )
        if transition_features is None:
            if self.transition_feature_dim > 0:
                raise ValueError(
                    "transition_features are required when "
                    f"transition_feature_dim={self.transition_feature_dim}."
                )
            transition_feature_values = fb.node_h.new_zeros((edge_h.size(0), 0))
        else:
            transition_feature_values = transition_features.values.to(
                device=fb.node_h.device,
                dtype=fb.node_h.dtype,
            )
            expected_shape = (edge_h.size(0), self.transition_feature_dim)
            if transition_feature_values.shape != expected_shape:
                raise ValueError(
                    "transition_features.values must have shape "
                    f"{expected_shape}, got "
                    f"{tuple(transition_feature_values.shape)}."
                )

        residual_input = torch.cat(
            [state_h, edge_h, transition_feature_values, semantic_prior_h],
            dim=-1,
        )

        return self.residual_head(residual_input).squeeze(-1)

    def effective_residual_scale(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        scale = self.residual_scale * self._residual_scale_multiplier.to(
            device=self.residual_scale.device,
            dtype=self.residual_scale.dtype,
        )
        if device is not None or dtype is not None:
            scale = scale.to(
                device=device or scale.device,
                dtype=dtype or scale.dtype,
            )
        return scale

    def update_residual_schedule(self, *, step: int) -> dict[str, float]:
        multiplier = self._scheduled_residual_multiplier(step=int(step))
        self._residual_scale_multiplier.fill_(float(multiplier))

        residual_trainable = (
            self._trainable_residual and int(step) >= self.freeze_residual_until_step
        )
        self.residual_scale.requires_grad_(residual_trainable)
        for parameter in self.residual_head.parameters():
            parameter.requires_grad_(residual_trainable)

        effective_scale = float(
            (
                self.residual_scale.detach()
                * self._residual_scale_multiplier.detach().to(
                    device=self.residual_scale.device,
                    dtype=self.residual_scale.dtype,
                )
            )
            .cpu()
            .item()
        )
        return {
            "residual_multiplier": float(multiplier),
            "residual_effective_scale": effective_scale,
            "residual_trainable": float(residual_trainable),
        }

    def _scheduled_residual_multiplier(self, *, step: int) -> float:
        step = int(step)
        if step < self.residual_warmup_start_step:
            return 0.0

        if self.residual_warmup_steps <= 0:
            return float(self.residual_max_multiplier)

        progress = (step - self.residual_warmup_start_step) / float(
            self.residual_warmup_steps
        )
        progress = max(0.0, min(1.0, progress))
        return float(self.residual_max_multiplier) * progress


def _non_negative_int(value: object, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out}.")
    return out


def _non_negative_float(value: object, name: str) -> float:
    out = float(value)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out}.")
    return out


__all__ = [
    "EdgeScoreBreakdown",
    "EdgeScorer",
]
