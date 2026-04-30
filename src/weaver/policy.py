from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_scatter import scatter_logsumexp, scatter_max

from src.data.schema import RetrievalBatch

from .action_ops import has_segment
from .nn.action_head import StopExpandGate
from .nn.backbone import FeatureBank, SemanticFeatureEncoder
from .nn.edge_scorer import EdgeScoreBreakdown, ExpandEdgeScorer
from .nn.flow_head import FlowHead
from .nn.state_readout import StateReadout
from .state import State
from .state_ops import frontier_edges


@dataclass(frozen=True)
class CandidateEdges:
    """
    Candidate Expand actions for the current batched state.

    edge_ids:
        Physical edge ids in the current RetrievalBatch.

    expand_logits:
        Conditional edge logits for P(edge | state, Expand).

    batch_index:
        Physical graph id for each candidate edge.
    """

    edge_ids: torch.Tensor
    expand_logits: torch.Tensor
    batch_index: torch.Tensor

    def __len__(self) -> int:
        return int(self.edge_ids.numel())

    @property
    def edge_logits(self) -> torch.Tensor:
        return self.expand_logits

    @property
    def candidate_batch_ids(self) -> torch.Tensor:
        return self.batch_index


@dataclass(frozen=True)
class PolicyOutput:
    """
    One forward-policy and flow evaluation at a subgraph state.
    """

    state_log_flow: torch.Tensor

    # Option-level logits.
    stop_logits: torch.Tensor
    expand_logits: torch.Tensor

    # Conditional edge logits for P(edge | s, Expand).
    edge_logits: torch.Tensor
    candidate_batch_ids: torch.Tensor
    candidate_edge_ids: torch.Tensor

    root_log_z: torch.Tensor | None = None
    edge_score_breakdown: EdgeScoreBreakdown | None = None

    @property
    def candidates(self) -> CandidateEdges:
        return CandidateEdges(
            edge_ids=self.candidate_edge_ids,
            expand_logits=self.edge_logits,
            batch_index=self.candidate_batch_ids,
        )

    @property
    def type_logits(self) -> torch.Tensor:
        return torch.stack([self.stop_logits, self.expand_logits], dim=-1)


PolicyStepOutput = PolicyOutput


class Policy(nn.Module):
    """
    Forward policy and state-flow estimator for subgraph-state GFlowNet.

    State:
        s = (V_s, E_s)

    Option-level policy:

        P(Stop | s), P(Expand | s)
            = softmax([z_stop(s), z_expand(s)])

    Conditional edge policy:

        P(edge | s, Expand)
            = softmax_{e in frontier(s)} z_e(s, q)

    Full action probability:

        P(Stop | s)
            = P(option=Stop | s)

        P(Expand(e) | s)
            = P(option=Expand | s) * P(e | s, Expand)

    The edge logits are conditional ranking logits. They are not calibrated
    joint transition-flow logits and must not be directly normalized against
    Stop. Stop/Expand is handled by StopExpandGate.

    State flow:

        log F(s | q) = FlowHead(q, StateReadout(s, q))
    """

    def __init__(
        self,
        *,
        feature_encoder_cfg: dict[str, Any],
        hidden_dim: int = 1024,
        state_readout_dropout: float = 0.0,
        stop_scorer_cfg: dict[str, Any] | None = None,
        edge_scorer_cfg: dict[str, Any] | None = None,
        flow_head_cfg: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.feature_encoder = SemanticFeatureEncoder(**feature_encoder_cfg)

        self.state_readout = StateReadout(
            hidden_dim=self.hidden_dim,
            state_feature_dim=0,
            dropout=float(state_readout_dropout),
            use_state_features=False,
        )

        self.flow_head = FlowHead(
            hidden_dim=self.hidden_dim,
            **(flow_head_cfg or {}),
        )

        self.action_scorer = StopExpandGate(
            hidden_dim=self.hidden_dim,
            **(stop_scorer_cfg or {}),
        )

        self.expand_edge_scorer = ExpandEdgeScorer(
            hidden_dim=self.hidden_dim,
            **(edge_scorer_cfg or {}),
        )

    def prepare_rollout_context(self, batch: RetrievalBatch) -> FeatureBank:
        return self.feature_encoder(batch)

    def forward(
        self,
        batch: RetrievalBatch,
        state: State,
        rollout_context: FeatureBank | None = None,
        *,
        return_edge_breakdown: bool = False,
        **_: Any,
    ) -> PolicyOutput:
        """
        Evaluate flow, Stop/Expand logits, and conditional frontier-edge logits.

        Extra keyword arguments are ignored intentionally to remain compatible
        with older callers while removing reward-aware Stop features from the
        policy interface.
        """
        fb = (
            rollout_context
            if rollout_context is not None
            else self.feature_encoder(batch)
        )

        device = fb.node_h.device
        num_graphs = int(batch.num_graphs)

        self._validate_feature_bank(fb=fb, num_graphs=num_graphs)

        context = self.state_readout(
            fb=fb,
            batch=batch,
            state=state,
            state_features=None,
        )
        state_h = context.state_h

        state_log_flow = self.flow_head(
            query_h=context.query_h,
            state_h=state_h,
        )

        edge_ids, edge_batch = frontier_edges(
            batch=batch,
            state=state,
            device=device,
        )

        edge_score = self.expand_edge_scorer(
            fb=fb,
            state_h=state_h,
            context=context,
            edge_index=batch.edge_index,
            edge_batch_index=batch.edge_batch,
            active_nodes=state.active_nodes,
            candidate_edge_ids=edge_ids,
            node_is_non_text=fb.node_is_non_text,
            return_breakdown=return_edge_breakdown,
        )

        if return_edge_breakdown:
            if not isinstance(edge_score, EdgeScoreBreakdown):
                raise TypeError(
                    "Expected EdgeScoreBreakdown when return_edge_breakdown=True."
                )
            edge_logits = edge_score.final_logits
            breakdown = edge_score
        else:
            if not isinstance(edge_score, torch.Tensor):
                raise TypeError(
                    "Expected tensor logits when return_edge_breakdown=False."
                )
            edge_logits = edge_score
            breakdown = None

        has_candidate_edge = has_segment(
            batch_index=edge_batch,
            num_segments=num_graphs,
            device=device,
        )

        edge_logmeanexp: torch.Tensor | None = None
        edge_max: torch.Tensor | None = None
        if getattr(self.action_scorer, "use_frontier_summary", True):
            frontier_summary = frontier_logit_summary(
                edge_logits=edge_logits,
                edge_batch=edge_batch,
                num_graphs=num_graphs,
                device=device,
            )
            edge_logmeanexp = frontier_summary.edge_logmeanexp
            edge_max = frontier_summary.edge_max
            if getattr(self.action_scorer, "detach_frontier_summary", False):
                edge_logmeanexp = edge_logmeanexp.detach()
                edge_max = edge_max.detach()

        stop_logits, option_expand_logits = self.action_scorer(
            query_h=context.query_h,
            state_h=state_h,
            has_candidate_edge=has_candidate_edge,
            progress_ratio=context.progress,
            edge_logmeanexp=edge_logmeanexp,
            edge_max=edge_max,
        )

        return PolicyOutput(
            state_log_flow=state_log_flow,
            # RolloutEngine decides whether the current step is the root.
            # Avoid checking state.is_root_state here because that synchronizes CUDA.
            root_log_z=state_log_flow,
            stop_logits=stop_logits,
            expand_logits=option_expand_logits,
            edge_logits=edge_logits,
            candidate_batch_ids=edge_batch,
            candidate_edge_ids=edge_ids,
            edge_score_breakdown=breakdown,
        )

    def _validate_feature_bank(
        self,
        *,
        fb: FeatureBank,
        num_graphs: int,
    ) -> None:
        if fb.query_h.ndim != 2:
            raise ValueError(
                f"query_h must have shape [B, H], got {tuple(fb.query_h.shape)}."
            )
        if fb.node_h.ndim != 2:
            raise ValueError(
                f"node_h must have shape [N, H], got {tuple(fb.node_h.shape)}."
            )
        if fb.rel_h.ndim != 2:
            raise ValueError(
                f"rel_h must have shape [E, H], got {tuple(fb.rel_h.shape)}."
            )

        if fb.query_h.size(0) != int(num_graphs):
            raise ValueError(
                f"query_h has {fb.query_h.size(0)} graphs, expected {num_graphs}."
            )

        for name, tensor in {
            "query_h": fb.query_h,
            "node_h": fb.node_h,
            "rel_h": fb.rel_h,
        }.items():
            if tensor.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"{name} hidden dimension mismatch: "
                    f"got {tensor.size(-1)}, expected {self.hidden_dim}."
                )


@dataclass(frozen=True)
class FrontierLogitSummary:
    """
    Size-normalized summaries of conditional frontier-edge logits.

    edge_logmeanexp:
        log mean exp over frontier edge logits per graph. Unlike logsumexp,
        this does not reward having many candidate edges.

    edge_max:
        maximum frontier edge logit per graph.

    edge_sharpness:
        edge_max - edge_logmeanexp. Measures whether the frontier has a
        standout candidate, without exposing raw frontier size.
    """

    edge_logmeanexp: torch.Tensor
    edge_max: torch.Tensor
    edge_sharpness: torch.Tensor


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

    return FrontierLogitSummary(
        edge_logmeanexp=edge_logmeanexp,
        edge_max=edge_max,
        edge_sharpness=edge_sharpness,
    )


__all__ = [
    "CandidateEdges",
    "FrontierLogitSummary",
    "Policy",
    "PolicyOutput",
    "PolicyStepOutput",
    "frontier_logit_summary",
]
