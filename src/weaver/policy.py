from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_scatter import scatter_logsumexp, scatter_max

from src.data.schema import RetrievalBatch
from src.utils.nn_utils import zero_last_linear
from .nn.candidate_context import build_candidate_context
from .nn.edge_encoder import EdgeEncoder
from .nn.edge_scorer import EdgeScoreBreakdown, EdgeScorer
from .nn.feature_encoder import FeatureBank, FeatureEncoder
from .nn.flow_head import FlowHead
from .nn.state_readout import StateReadout
from .nn.stop_head import LearnedStopHead
from .nn.transition_features import TransitionFeatureBuilder, TransitionFeatureOutput
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

    For fused rollout states, candidate_batch_ids are dynamic rollout row ids,
    not physical RetrievalBatch graph ids.
    """

    state_log_flow: torch.Tensor

    stop_logits: torch.Tensor
    # Diagnostic aggregate logsumexp_e z_e. The target action distribution is
    # still the flat softmax over Stop and every Expand(e), not a two-stage gate.
    expand_logits: torch.Tensor

    edge_logits: torch.Tensor
    candidate_edge_ids: torch.Tensor
    candidate_batch_ids: torch.Tensor

    edge_score_breakdown: EdgeScoreBreakdown | None = None


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

    Semantic base-measure target policy:
        z_e(s, q) = z0(e | s, q) + r_theta(s, e, q)
        z_stop(s, q) = learned_stop_theta(h_s, progress, frontier(z0 + r))

    z0 is the PLM semantic base measure from EdgeScorer. r_theta is a
    state-conditioned density correction. Uniform removable P_B is used only in
    SubTB accounting after an edge is selected; it is not part of forward logits.

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
        edge_residual_cfg: dict[str, Any] | None = None,
        flow_head_cfg: dict[str, Any] | None = None,
        action_parameterization: str = "semantic_base_gfn",
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.action_parameterization = str(action_parameterization)
        if self.action_parameterization != "semantic_base_gfn":
            raise ValueError(
                "action_parameterization must be 'semantic_base_gfn', "
                f"got {action_parameterization!r}."
            )

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

        stop_head_kwargs = dict(stop_scorer_cfg or {})
        stop_head_kwargs.setdefault("use_progress", True)
        stop_head_kwargs.setdefault("use_frontier_summary", True)
        stop_head_kwargs.setdefault("state_stat_dim", 2)
        self.stop_head = LearnedStopHead(
            hidden_dim=self.hidden_dim,
            **stop_head_kwargs,
        )

        edge_scorer_kwargs = dict(edge_scorer_cfg or {})
        if "share_edge_encoder_with_readout" in edge_scorer_kwargs:
            raise ValueError(
                "edge_scorer.share_edge_encoder_with_readout was removed with the "
                "residual edge scorer."
            )
        self.edge_scorer = EdgeScorer(
            hidden_dim=self.hidden_dim,
            **edge_scorer_kwargs,
        )
        self.edge_residual_head = EdgeResidualHead(
            hidden_dim=self.hidden_dim,
            transition_feature_dim=self.transition_feature_builder.feature_dim,
            **(edge_residual_cfg or {}),
        )

    @property
    def requires_stop_log_reward(self) -> bool:
        return False

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
        stop_log_reward: torch.Tensor | None = None,
    ) -> PolicyOutput:
        del stop_log_reward

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
        base_breakdown = self._semantic_edge_breakdown(
            fb=fb,
            batch=batch,
            state=state,
            context=context,
            candidate_edge_ids=candidate_edge_ids,
            candidate_batch_ids=candidate_batch_ids,
            candidate_context=candidate_context,
        )
        transition_features = self.transition_feature_builder(
            fb=fb,
            context=context,
            batch=batch,
            state=state,
            candidate_edge_ids=candidate_edge_ids,
            candidate_batch_ids=candidate_batch_ids,
            candidate_context=candidate_context,
        )

        residual_logits = self.edge_residual_head(
            context=context,
            candidate_edge_h=context.frontier_edge_h,
            candidate_batch_ids=candidate_batch_ids,
            semantic_logits=base_breakdown.semantic_logits,
            transition_features=transition_features,
        )
        final_edge_logits = base_breakdown.semantic_logits + residual_logits
        edge_logits = (
            base_breakdown.semantic_logits
            if edge_logit_mode == "semantic"
            else final_edge_logits
        )
        edge_breakdown = (
            EdgeScoreBreakdown(
                query_relation_score=base_breakdown.query_relation_score,
                query_new_node_score=base_breakdown.query_new_node_score,
                semantic_score=base_breakdown.semantic_score,
                new_text_mask=base_breakdown.new_text_mask,
                semantic_logits=base_breakdown.semantic_logits,
                residual_logits=residual_logits,
                final_logits=final_edge_logits,
            )
            if return_edge_breakdown or edge_logit_mode == "semantic"
            else None
        )

        expand_logits = _segment_logsumexp_or_neg_inf(
            values=edge_logits,
            batch_ids=candidate_batch_ids,
            num_graphs=num_policy_graphs,
        )
        has_candidate_edge = _has_candidate_edge(
            candidate_batch_ids=candidate_batch_ids,
            num_graphs=num_policy_graphs,
            device=device,
        )
        frontier_summary = frontier_logit_summary(
            edge_logits=edge_logits,
            edge_batch=candidate_batch_ids,
            num_graphs=num_policy_graphs,
            device=device,
        )
        state_stats = self._state_stats(
            state=state,
            batch=batch,
            num_graphs=num_policy_graphs,
            device=device,
            dtype=state_log_flow.dtype,
        )
        stop_logits = self.stop_head(
            state_h=context.state_h,
            num_graphs=num_policy_graphs,
            progress_ratio=context.progress,
            frontier_summary=frontier_summary.as_tensor(),
            state_stats=state_stats,
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

    def _semantic_edge_breakdown(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        state: State | RolloutState,
        context: Any,
        candidate_edge_ids: torch.Tensor,
        candidate_batch_ids: torch.Tensor,
        candidate_context: Any | None = None,
    ) -> EdgeScoreBreakdown:
        device = fb.node_h.device
        candidate_context = candidate_context or build_candidate_context(
            batch=batch,
            state=state,
            candidate_edge_ids=candidate_edge_ids,
            candidate_batch_ids=candidate_batch_ids,
            device=device,
        )
        edge_score = self.edge_scorer(
            fb=fb,
            context=context,
            edge_index=batch.edge_index,
            edge_batch_index=batch.edge_batch,
            active_nodes=state.active_nodes,
            candidate_edge_ids=candidate_edge_ids,
            candidate_context=candidate_context,
            return_breakdown=True,
        )
        if not isinstance(edge_score, EdgeScoreBreakdown):
            raise TypeError(
                "Expected EdgeScoreBreakdown when edge breakdown is requested."
            )
        return edge_score

    def _state_stats(
        self,
        *,
        state: State | RolloutState,
        batch: RetrievalBatch,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expanded = state.expanded_edge_count_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=int(num_graphs),
        ).to(device=device, dtype=dtype)
        remaining = state.remaining_budget_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=int(num_graphs),
        ).to(device=device, dtype=dtype)
        return torch.stack([expanded, remaining], dim=-1)

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


def _segment_logsumexp_or_neg_inf(
    *,
    values: torch.Tensor,
    batch_ids: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    values = values.view(-1)
    batch_ids = batch_ids.to(device=values.device, dtype=torch.long).view(-1)
    num_graphs = int(num_graphs)
    if values.numel() == 0:
        return values.new_full((num_graphs,), -torch.inf)

    counts = torch.bincount(batch_ids, minlength=num_graphs).to(device=values.device)
    raw = scatter_logsumexp(values, batch_ids, dim=0, dim_size=num_graphs)
    return torch.where(
        counts.gt(0),
        raw,
        values.new_full((num_graphs,), -torch.inf),
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


class EdgeResidualHead(nn.Module):
    """
    Learned density correction r_theta(s,e,q) added to semantic base logits.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        transition_feature_dim: int,
        dropout: float = 0.0,
        trainable: bool = True,
        zero_init: bool = True,
        include_semantic_logit: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.transition_feature_dim = int(transition_feature_dim)
        if self.transition_feature_dim < 0:
            raise ValueError(
                f"transition_feature_dim must be >= 0, got {transition_feature_dim}."
            )
        self.include_semantic_logit = bool(include_semantic_logit)

        input_dim = self.hidden_dim * 2 + self.transition_feature_dim
        if self.include_semantic_logit:
            input_dim += 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        if zero_init:
            zero_last_linear(self.net)
        for parameter in self.parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(
        self,
        *,
        context: Any,
        candidate_edge_h: torch.Tensor | None,
        candidate_batch_ids: torch.Tensor,
        semantic_logits: torch.Tensor,
        transition_features: TransitionFeatureOutput,
    ) -> torch.Tensor:
        semantic_logits = semantic_logits.view(-1)
        num_candidates = int(semantic_logits.numel())
        if num_candidates == 0:
            return semantic_logits.new_empty((0,))

        if candidate_edge_h is None:
            raise RuntimeError("StateReadout did not return frontier edge features.")
        edge_h = candidate_edge_h.to(
            device=semantic_logits.device,
            dtype=context.state_h.dtype,
        )
        if edge_h.shape != (num_candidates, self.hidden_dim):
            raise ValueError(
                "candidate_edge_h must have shape "
                f"[{num_candidates}, {self.hidden_dim}], got {tuple(edge_h.shape)}."
            )

        batch_ids = candidate_batch_ids.to(
            device=context.state_h.device,
            dtype=torch.long,
        ).view(-1)
        if batch_ids.shape != (num_candidates,):
            raise ValueError(
                "candidate_batch_ids must have shape "
                f"[{num_candidates}], got {tuple(batch_ids.shape)}."
            )
        state_h = context.state_h.index_select(0, batch_ids)

        transition_values = transition_features.values.to(
            device=context.state_h.device,
            dtype=context.state_h.dtype,
        )
        expected = (num_candidates, self.transition_feature_dim)
        if transition_values.shape != expected:
            raise ValueError(
                f"transition_features.values must have shape {expected}, "
                f"got {tuple(transition_values.shape)}."
            )

        pieces = [state_h, edge_h, transition_values]
        if self.include_semantic_logit:
            pieces.append(
                semantic_logits.detach()
                .to(device=context.state_h.device, dtype=context.state_h.dtype)
                .unsqueeze(-1)
            )

        return self.net(torch.cat(pieces, dim=-1)).squeeze(-1)


def _has_candidate_edge(
    *,
    candidate_batch_ids: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    candidate_batch_ids = candidate_batch_ids.to(device=device, dtype=torch.long)
    if candidate_batch_ids.numel() == 0:
        return torch.zeros(int(num_graphs), dtype=torch.bool, device=device)
    return torch.bincount(candidate_batch_ids, minlength=int(num_graphs)).gt(0)


def _candidate_successor_state(
    *,
    batch: RetrievalBatch,
    state: State | RolloutState,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
) -> RolloutState:
    if isinstance(state, RolloutState) or state.active_nodes.ndim == 2:
        if not isinstance(state, RolloutState):
            raise TypeError("2D active masks require RolloutState.")
        return _rollout_candidate_successor_state(
            batch=batch,
            state=state,
            candidate_edge_ids=candidate_edge_ids,
            candidate_batch_ids=candidate_batch_ids,
        )

    return _single_batch_candidate_successor_state(
        batch=batch,
        state=state,
        candidate_edge_ids=candidate_edge_ids,
        candidate_batch_ids=candidate_batch_ids,
    )


def _single_batch_candidate_successor_state(
    *,
    batch: RetrievalBatch,
    state: State,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
) -> RolloutState:
    device = state.active_nodes.device
    edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)
    graph_ids = candidate_batch_ids.to(device=device, dtype=torch.long).view(-1)
    if edge_ids.shape != graph_ids.shape:
        raise ValueError(
            "candidate_edge_ids and candidate_batch_ids must have matching shape: "
            f"{tuple(edge_ids.shape)} != {tuple(graph_ids.shape)}."
        )

    num_candidates = int(edge_ids.numel())
    node_batch = batch.batch.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

    node_belongs = node_batch.view(1, -1).eq(graph_ids.view(-1, 1))
    edge_belongs = edge_batch.view(1, -1).eq(graph_ids.view(-1, 1))

    active_nodes = state.active_nodes.view(1, -1).expand_as(node_belongs) & node_belongs
    active_edges = state.active_edges.view(1, -1).expand_as(edge_belongs) & edge_belongs
    root_edges = state.root_edges.view(1, -1).expand_as(edge_belongs) & edge_belongs

    anchor_mask = torch.zeros_like(state.active_nodes, dtype=torch.bool, device=device)
    anchors = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
    valid_anchors = anchors.ge(0) & anchors.lt(anchor_mask.numel())
    if bool(valid_anchors.any()):
        anchor_mask[anchors[valid_anchors]] = True
    anchor_nodes = anchor_mask.view(1, -1).expand_as(node_belongs) & node_belongs

    next_state = RolloutState(
        active_nodes=active_nodes.clone(),
        active_edges=active_edges.clone(),
        root_edges=root_edges.clone(),
        anchor_nodes=anchor_nodes.clone(),
        rollout_to_graph=graph_ids,
        expand_budget=int(state.expand_budget),
    )
    next_state.apply_expansion(
        rollout_ids=torch.arange(num_candidates, dtype=torch.long, device=device),
        chosen_edges=edge_ids,
        edge_index=batch.edge_index,
    )
    return next_state


def _rollout_candidate_successor_state(
    *,
    batch: RetrievalBatch,
    state: RolloutState,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
) -> RolloutState:
    device = state.active_nodes.device
    edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)
    rollout_ids = candidate_batch_ids.to(device=device, dtype=torch.long).view(-1)
    if edge_ids.shape != rollout_ids.shape:
        raise ValueError(
            "candidate_edge_ids and candidate_batch_ids must have matching shape: "
            f"{tuple(edge_ids.shape)} != {tuple(rollout_ids.shape)}."
        )

    num_candidates = int(edge_ids.numel())
    next_state = RolloutState(
        active_nodes=state.active_nodes.index_select(0, rollout_ids).clone(),
        active_edges=state.active_edges.index_select(0, rollout_ids).clone(),
        root_edges=state.root_edges.index_select(0, rollout_ids).clone(),
        anchor_nodes=state.anchor_nodes.index_select(0, rollout_ids).clone(),
        rollout_to_graph=state.rollout_to_graph.index_select(0, rollout_ids),
        expand_budget=int(state.expand_budget),
    )
    next_state.apply_expansion(
        rollout_ids=torch.arange(num_candidates, dtype=torch.long, device=device),
        chosen_edges=edge_ids,
        edge_index=batch.edge_index,
    )
    return next_state


__all__ = [
    "EdgeResidualHead",
    "FrontierLogitSummary",
    "Policy",
    "PolicyOutput",
    "frontier_logit_summary",
]
