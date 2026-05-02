from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_scatter import scatter_logsumexp

from src.data.schema import RetrievalBatch
from .nn.candidate_context import build_candidate_context
from .nn.edge_encoder import EdgeEncoder
from .nn.edge_scorer import EdgeScoreBreakdown, EdgeScorer
from .nn.feature_encoder import FeatureBank, FeatureEncoder
from .nn.flow_head import FlowHead
from .nn.state_readout import StateReadout
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

    Standard backward-flow target policy:
        logit(Stop | s) = log R_stop(s)
        logit(Expand e | s) = log F_theta(s + e)

    The executor adds the candidate-level log P_B(s | s + e) term before
    sampling and before writing log P_F into rollout traces. The semantic edge
    scorer is kept only for diagnostics/proposals and does not enter target
    logits.

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
        stop_scorer_cfg: dict[str, Any] | None = None,
        edge_scorer_cfg: dict[str, Any] | None = None,
        flow_head_cfg: dict[str, Any] | None = None,
        action_parameterization: str = "gfn_backward_flow",
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.action_parameterization = str(action_parameterization)
        if self.action_parameterization != "gfn_backward_flow":
            raise ValueError(
                "action_parameterization must be 'gfn_backward_flow', "
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

        if stop_scorer_cfg:
            raise ValueError(
                "policy_cfg.stop_scorer is not used by "
                "action_parameterization='gfn_backward_flow'."
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

    @property
    def requires_stop_log_reward(self) -> bool:
        return True

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
        edge_breakdown = (
            self._semantic_edge_breakdown(
                fb=fb,
                batch=batch,
                state=state,
                context=context,
                candidate_edge_ids=candidate_edge_ids,
                candidate_batch_ids=candidate_batch_ids,
            )
            if return_edge_breakdown or edge_logit_mode == "semantic"
            else None
        )

        if edge_logit_mode == "semantic":
            if edge_breakdown is None:
                raise RuntimeError("Semantic edge logits require edge breakdown.")
            edge_logits = edge_breakdown.semantic_logits
        else:
            edge_logits = self._successor_flow_logits(
                fb=fb,
                batch=batch,
                state=state,
                candidate_edge_ids=candidate_edge_ids,
                candidate_batch_ids=candidate_batch_ids,
            )

        stop_logits = _validate_stop_log_reward(
            stop_log_reward,
            num_graphs=int(num_policy_graphs),
            device=device,
            dtype=edge_logits.dtype if edge_logits.numel() else state_log_flow.dtype,
        )
        expand_logits = _segment_logsumexp_or_neg_inf(
            values=edge_logits,
            batch_ids=candidate_batch_ids,
            num_graphs=num_policy_graphs,
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
    ) -> EdgeScoreBreakdown:
        device = fb.node_h.device
        candidate_context = build_candidate_context(
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

    def _successor_flow_logits(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        state: State | RolloutState,
        candidate_edge_ids: torch.Tensor,
        candidate_batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        device = fb.node_h.device
        if candidate_edge_ids.numel() == 0:
            return fb.node_h.new_empty((0,))
        successor_context = self.state_readout.forward_successor_state_delta(
            fb=fb,
            batch=batch,
            state=state,
            candidate_edge_ids=candidate_edge_ids,
            candidate_batch_ids=candidate_batch_ids,
        )
        return self.flow_head(
            state_h=successor_context.state_h,
        ).to(device=device, dtype=fb.node_h.dtype)

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


def _validate_stop_log_reward(
    value: torch.Tensor | None,
    *,
    num_graphs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        raise ValueError(
            "Policy action_parameterization='gfn_backward_flow' requires "
            "stop_log_reward for every visited state."
        )
    value = value.to(device=device, dtype=dtype).view(-1)
    if value.shape != (int(num_graphs),):
        raise ValueError(
            f"stop_log_reward must have shape [{int(num_graphs)}], "
            f"got {tuple(value.shape)}."
        )
    return value


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
    "Policy",
    "PolicyOutput",
]
