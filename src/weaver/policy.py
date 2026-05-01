from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_scatter import scatter_logsumexp, scatter_max

from src.data.schema import RetrievalBatch

from .nn.candidate_context import build_candidate_context
from .nn.edge_encoder import EdgeEncoder
from .nn.edge_scorer import EdgeScoreBreakdown, EdgeScorer
from .nn.feature_encoder import FeatureBank, FeatureEncoder
from .nn.flow_head import FlowHead
from .nn.state_readout import StateReadout
from .nn.stop_gate import StopExpandGate
from .nn.transition_features import TransitionFeatureBuilder
from .state import RolloutState, State


@dataclass(frozen=True)
class PolicyOutput:
    """
    Policy evaluation at one subgraph state.

    The policy is purely neural. It does not know rewards, StopAdv targets,
    teachers, or oracle child states.

    candidate_edge_ids / candidate_batch_ids are the frontier returned by
    StateReadout for the same state snapshot. They must be interpreted under
    the canonical state invariant:

        V_s = anchors union endpoints(E_s)
    """

    state_log_flow: torch.Tensor

    stop_logits: torch.Tensor
    expand_logits: torch.Tensor

    edge_logits: torch.Tensor
    candidate_edge_ids: torch.Tensor
    candidate_batch_ids: torch.Tensor

    edge_score_breakdown: EdgeScoreBreakdown | None = None

    @property
    def type_logits(self) -> torch.Tensor:
        return torch.stack([self.stop_logits, self.expand_logits], dim=-1)


class Policy(nn.Module):
    """
    Forward policy and state-flow estimator for subgraph-state GFlowNet.

    State:
        s = (V_s, E_s)
        V_s = anchors union endpoints(E_s)

    Readout:
        h_s = StateReadout(q, V_s, E_s)

    State flow:
        log F(s | q) = FlowHead(h_s)

    Option policy:
        P(Stop | s), P(Expand | s)
            = softmax([z_stop(h_s), z_expand(h_s)])

    Conditional edge policy:
        P(edge | s, Expand)
            = softmax_{e in frontier(s)} z_e(s, q)

    Root edges are part of E_s and are read by StateReadout. They are not
    excluded from evidence. Expansion budget accounting is handled by State.

    Policy consumes the state snapshot as provided by RolloutEngine. It does not
    rebuild V_s from E_s; Executor validates that mutable state masks satisfy the
    canonical node closure before transitions are applied.
    """

    def __init__(
        self,
        *,
        feature_encoder_cfg: dict[str, Any],
        hidden_dim: int = 1024,
        state_readout_dropout: float = 0.0,
        state_readout_cfg: dict[str, Any] | None = None,
        transition_features_cfg: dict[str, Any] | None = None,
        action_features_cfg: dict[str, Any] | None = None,
        stop_scorer_cfg: dict[str, Any] | None = None,
        edge_scorer_cfg: dict[str, Any] | None = None,
        flow_head_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.feature_encoder = FeatureEncoder(**feature_encoder_cfg)

        self.edge_encoder = EdgeEncoder(hidden_dim=self.hidden_dim)

        state_readout_kwargs = dict(state_readout_cfg or {})
        state_readout_kwargs.setdefault("dropout", float(state_readout_dropout))

        self.state_readout = StateReadout(
            hidden_dim=self.hidden_dim,
            edge_encoder=self.edge_encoder,
            **state_readout_kwargs,
        )

        self.flow_head = FlowHead(
            hidden_dim=self.hidden_dim,
            **(flow_head_cfg or {}),
        )

        if transition_features_cfg is not None and action_features_cfg is not None:
            raise ValueError(
                "Use only transition_features_cfg; action_features_cfg is a legacy alias."
            )
        transition_cfg = (
            transition_features_cfg
            if transition_features_cfg is not None
            else action_features_cfg
        )
        if transition_cfg:
            raise ValueError(
                "transition_features_cfg no longer accepts handcrafted feature "
                f"switches; residual transition features are fixed. Got: {sorted(transition_cfg)}."
            )

        self.transition_feature_builder = TransitionFeatureBuilder(
            hidden_dim=self.hidden_dim,
            dde_dim=int(getattr(self.feature_encoder, "dde_dim", 0)),
        )

        self.stop_gate = StopExpandGate(
            hidden_dim=self.hidden_dim,
            **(stop_scorer_cfg or {}),
        )

        edge_scorer_kwargs = dict(edge_scorer_cfg or {})
        share_edge_encoder = bool(
            edge_scorer_kwargs.pop("share_edge_encoder_with_readout", True)
        )
        if not share_edge_encoder:
            raise ValueError(
                "edge_scorer.share_edge_encoder_with_readout=false is no longer "
                "supported; Policy always shares edge_encoder."
            )
        edge_scorer_kwargs.setdefault(
            "transition_feature_dim",
            self.transition_feature_builder.feature_dim,
        )
        self.edge_scorer = EdgeScorer(
            hidden_dim=self.hidden_dim,
            edge_encoder=self.edge_encoder,
            **edge_scorer_kwargs,
        )

    def prepare_rollout_context(self, batch: RetrievalBatch) -> FeatureBank:
        return self.feature_encoder(batch)

    def forward(
        self,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: FeatureBank | None = None,
        *,
        return_edge_breakdown: bool = False,
        edge_logit_mode: str = "final",
    ) -> PolicyOutput:
        if edge_logit_mode not in {"final", "semantic"}:
            raise ValueError(
                "edge_logit_mode must be 'final' or 'semantic', "
                f"got {edge_logit_mode!r}."
            )

        fb = (
            rollout_context
            if rollout_context is not None
            else self.feature_encoder(batch)
        )

        device = fb.node_h.device
        num_graphs = int(batch.num_graphs)

        self._validate_feature_bank(
            fb=fb,
            batch=batch,
            num_graphs=num_graphs,
        )

        context = self.state_readout(
            fb=fb,
            batch=batch,
            state=state,
        )
        num_policy_graphs = int(context.state_h.size(0))

        state_log_flow = self.flow_head(state_h=context.state_h)

        if context.frontier_edge_ids is None or context.frontier_edge_batch is None:
            raise RuntimeError("StateReadout did not return frontier candidates.")

        candidate_edge_ids = context.frontier_edge_ids.to(
            device=device,
            dtype=torch.long,
        )
        candidate_batch_ids = context.frontier_edge_batch.to(
            device=device,
            dtype=torch.long,
        )
        candidate_context = build_candidate_context(
            batch=batch,
            state=state,
            candidate_edge_ids=candidate_edge_ids,
            candidate_batch_ids=candidate_batch_ids,
            device=device,
        )

        transition_features = self.transition_feature_builder(
            fb=fb,
            context=context,
            batch=batch,
            state=state,
            candidate_edge_ids=candidate_edge_ids,
            candidate_context=candidate_context,
        )

        edge_score = self.edge_scorer(
            fb=fb,
            context=context,
            edge_index=batch.edge_index,
            edge_batch_index=batch.edge_batch,
            active_nodes=state.active_nodes,
            candidate_edge_ids=candidate_edge_ids,
            candidate_context=candidate_context,
            transition_features=transition_features,
            return_breakdown=return_edge_breakdown or edge_logit_mode == "semantic",
        )

        if return_edge_breakdown or edge_logit_mode == "semantic":
            if not isinstance(edge_score, EdgeScoreBreakdown):
                raise TypeError(
                    "Expected EdgeScoreBreakdown when edge breakdown is requested."
                )

            edge_logits = (
                edge_score.semantic_logits
                if edge_logit_mode == "semantic"
                else edge_score.final_logits
            )
            edge_breakdown = edge_score if return_edge_breakdown else None
        else:
            if not isinstance(edge_score, torch.Tensor):
                raise TypeError(
                    "Expected tensor edge logits when edge breakdown is not requested."
                )

            edge_logits = edge_score
            edge_breakdown = None

        if candidate_batch_ids.numel() == 0:
            has_candidate_edge = torch.zeros(
                num_policy_graphs,
                dtype=torch.bool,
                device=device,
            )
        else:
            has_candidate_edge = torch.bincount(
                candidate_batch_ids,
                minlength=num_policy_graphs,
            ).gt(0)
        frontier_summary = frontier_logit_summary(
            edge_logits=edge_logits,
            edge_batch=candidate_batch_ids,
            num_graphs=num_policy_graphs,
            device=device,
        )

        stop_logits, expand_logits = self.stop_gate(
            state_h=context.state_h,
            edge_logits=edge_logits,
            edge_batch_index=candidate_batch_ids,
            num_graphs=num_policy_graphs,
            progress_ratio=context.progress,
            frontier_summary=frontier_summary.as_tensor(),
            has_candidate_edge=has_candidate_edge,
        )

        return PolicyOutput(
            state_log_flow=state_log_flow,
            stop_logits=stop_logits,
            expand_logits=expand_logits,
            edge_logits=edge_logits,
            candidate_edge_ids=candidate_edge_ids,
            candidate_batch_ids=candidate_batch_ids,
            edge_score_breakdown=edge_breakdown,
        )

    def _validate_feature_bank(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        num_graphs: int,
    ) -> None:
        expected = {
            "query_h": (fb.query_h, int(num_graphs), self.hidden_dim),
            "node_h": (fb.node_h, int(batch.num_nodes_total), self.hidden_dim),
            "rel_h": (fb.rel_h, int(batch.edge_index.size(1)), self.hidden_dim),
            "query_sem_h": (fb.query_sem_h, int(num_graphs), None),
            "node_sem_h": (fb.node_sem_h, int(batch.num_nodes_total), None),
            "rel_sem_h": (fb.rel_sem_h, int(batch.edge_index.size(1)), None),
        }

        for name, (tensor, first_dim, hidden_dim) in expected.items():
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must have shape [N, H], got {tuple(tensor.shape)}."
                )
            if tensor.size(0) != first_dim:
                raise ValueError(
                    f"{name} first dimension mismatch: expected {first_dim}, "
                    f"got {tensor.size(0)}."
                )
            if hidden_dim is not None and tensor.size(-1) != hidden_dim:
                raise ValueError(
                    f"{name} hidden dimension mismatch: expected {hidden_dim}, "
                    f"got {tensor.size(-1)}."
                )

        if fb.node_dde is not None:
            if fb.node_dde.ndim != 2:
                raise ValueError(
                    f"node_dde must have shape [num_nodes, D], got {tuple(fb.node_dde.shape)}."
                )
            if fb.node_dde.size(0) != int(batch.num_nodes_total):
                raise ValueError(
                    "node_dde first dimension mismatch: expected "
                    f"{int(batch.num_nodes_total)}, got {fb.node_dde.size(0)}."
                )

        if fb.node_is_non_text is not None:
            if fb.node_is_non_text.ndim != 1:
                raise ValueError(
                    "node_is_non_text must have shape [num_nodes], got "
                    f"{tuple(fb.node_is_non_text.shape)}."
                )
            if fb.node_is_non_text.numel() != int(batch.num_nodes_total):
                raise ValueError(
                    "node_is_non_text length mismatch: expected "
                    f"{int(batch.num_nodes_total)}, got {fb.node_is_non_text.numel()}."
                )


@dataclass(frozen=True)
class FrontierLogitSummary:
    edge_logmeanexp: torch.Tensor
    edge_max: torch.Tensor
    edge_sharpness: torch.Tensor
    edge_log_size: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(
            [self.edge_max, self.edge_logmeanexp, self.edge_log_size],
            dim=-1,
        )


def frontier_logit_summary(
    *,
    edge_logits: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> FrontierLogitSummary:
    edge_logits = edge_logits.to(device=device)
    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    num_graphs = int(num_graphs)

    if edge_logits.numel() == 0:
        zeros = edge_logits.new_zeros(num_graphs)
        return FrontierLogitSummary(
            edge_logmeanexp=zeros,
            edge_max=zeros,
            edge_sharpness=zeros,
            edge_log_size=zeros,
        )

    counts = torch.bincount(edge_batch, minlength=num_graphs).to(
        device=device,
        dtype=edge_logits.dtype,
    )
    has_edge = counts.gt(0)

    logsumexp = scatter_logsumexp(
        edge_logits,
        edge_batch,
        dim=0,
        dim_size=num_graphs,
    )
    logmeanexp = logsumexp - counts.clamp_min(1.0).log()

    edge_max = scatter_max(
        edge_logits,
        edge_batch,
        dim=0,
        dim_size=num_graphs,
    )[0]

    zeros = edge_logits.new_zeros(num_graphs)
    edge_logmeanexp = torch.where(has_edge, logmeanexp, zeros)
    edge_max = torch.where(has_edge, edge_max, zeros)
    edge_sharpness = torch.where(has_edge, edge_max - edge_logmeanexp, zeros)
    edge_log_size = torch.where(has_edge, counts.clamp_min(1.0).log(), zeros)

    return FrontierLogitSummary(
        edge_logmeanexp=edge_logmeanexp,
        edge_max=edge_max,
        edge_sharpness=edge_sharpness,
        edge_log_size=edge_log_size,
    )


__all__ = [
    "FrontierLogitSummary",
    "Policy",
    "PolicyOutput",
    "frontier_logit_summary",
]
