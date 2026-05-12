from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_max

from src.data.schema import RetrievalBatch
from src.graph.ops import (
    uniform_backward_parent_counts_for_selected_nonroot_edge_trace_tensors,
    uniform_log_pb_for_selected_nonroot_edge_trace_tensors,
)
from src.graph.segments import segment_logsumexp, subtract_log_normalizer
from src.weaver.reward.context import build_reward_batch_context
from src.weaver.reward.model import (
    _evidence_support_stats_for_edge_traces,
    _set_reward_terms,
)
from src.weaver.reward.utility import sparse_rollout_active_node_trace

from .nn.evidence_tokens import build_evidence_tokens
from .nn.edge_residual_scorer import EdgeResidualScorer
from .nn.feature_encoder import FeatureBank, FeatureEncoder
from .nn.flow_head import FlowHead
from .nn.frontier_builder import build_frontier
from .nn.evidence_state_encoder import EvidenceStateEncoder
from .nn.frontier_context import FrontierContext, frontier_semantic_scores
from .nn.frontier_pointer import FrontierPointerDiagnostics, FrontierPointerPolicy
from .nn.relation_residual_edge_scorer import (
    RelationResidualEdgeDiagnostics,
    RelationResidualEdgeScorer,
)
from .nn.successor_policy import SuccessorEdgeAdvantageScorer, SuccessorValueHead
from .nn.terminal_head import TerminalHead
from .nn.transition_features import TransitionFeatureBuilder
from .state import RolloutState, State


@dataclass(frozen=True)
class PolicyOutput:
    """
    Policy evaluation at one subgraph state.

    frontier_edge_ids / frontier_batch_ids are the global frontier returned by
    FrontierBuilder for the same state snapshot. For fused rollout states,
    frontier_batch_ids are dynamic rollout row ids.
    """

    stop_logits: torch.Tensor
    edge_logits: torch.Tensor
    state_log_flow: torch.Tensor

    log_p_stop: torch.Tensor
    log_p_continue: torch.Tensor
    edge_cond_logprob: torch.Tensor
    edge_expand_logprob: torch.Tensor

    frontier_edge_ids: torch.Tensor
    frontier_batch_ids: torch.Tensor

    edge_policy_diagnostics: FrontierPointerDiagnostics | RelationResidualEdgeDiagnostics | None = None
    log_c_continue: torch.Tensor | None = None
    log_z_action: torch.Tensor | None = None
    terminal_energy: torch.Tensor | None = None
    continue_energy: torch.Tensor | None = None
    value_energy: torch.Tensor | None = None
    # REMOVED: TE-BFM trace fields kept only for old serialized buffers — see methodology.md §3.9
    te_bfm_loss: torch.Tensor | None = None
    te_bfm_valid_mask: torch.Tensor | None = None
    te_bfm_residual_abs: torch.Tensor | None = None
    te_bfm_target_log_value: torch.Tensor | None = None
    te_bfm_log_reward: torch.Tensor | None = None
    te_bfm_stop_prob: torch.Tensor | None = None
    te_bfm_frontier_edge_count: torch.Tensor | None = None
    te_bfm_counterfactual_child_loss: torch.Tensor | None = None
    te_bfm_frontier_cap_used: torch.Tensor | None = None
    te_bfm_frontier_cap_dropped_edge_count: torch.Tensor | None = None
    bdb_stop_loss: torch.Tensor | None = None
    bdb_edge_loss: torch.Tensor | None = None
    bdb_base_loss: torch.Tensor | None = None
    bdb_stop_valid_mask: torch.Tensor | None = None
    bdb_edge_valid_mask: torch.Tensor | None = None
    bdb_base_valid_mask: torch.Tensor | None = None
    bdb_delta_stop: torch.Tensor | None = None
    bdb_delta_edge: torch.Tensor | None = None
    bdb_delta_base: torch.Tensor | None = None
    bdb_frontier_size: torch.Tensor | None = None
    bdb_parent_count: torch.Tensor | None = None
    bdb_log_reward: torch.Tensor | None = None
    bdb_log_flow: torch.Tensor | None = None
    terminal_quotient_backup_used_rate: torch.Tensor | None = None
    terminal_quotient_parent_count: torch.Tensor | None = None
    terminal_quotient_edge_count: torch.Tensor | None = None
    terminal_quotient_group_count_mean: torch.Tensor | None = None
    terminal_quotient_floor_edge_count_mean: torch.Tensor | None = None
    terminal_quotient_positive_edge_count_mean: torch.Tensor | None = None
    terminal_quotient_speedup_estimate: torch.Tensor | None = None


@dataclass(frozen=True)
class PolicyContext:
    fb: FeatureBank


@dataclass(frozen=True)
class _TerminalQuotientBackup:
    continuation: torch.Tensor
    edge_terms: torch.Tensor
    row_ids: torch.Tensor
    parent_count: int
    edge_count: int
    group_count_mean: float
    floor_edge_count_mean: float
    positive_edge_count_mean: float
    speedup_estimate: float


@dataclass(frozen=True)
class _FrontierCapStats:
    used_rate: float
    dropped_edge_count_mean: float


@dataclass(frozen=True)
class _TerminalEdgeTerms:
    log_reward: torch.Tensor
    log_pb: torch.Tensor
    supported: torch.Tensor

    @property
    def edge_terms(self) -> torch.Tensor:
        return self.log_reward + self.log_pb


class Policy(nn.Module):
    """
    Forward policy for evidence subgraph construction.

    State:
        s = (V_s, E_s), V_s = anchors union endpoints(E_s)

    Action:
        Stop or Add(e), where e belongs to the global frontier C(s).

    Expand policy:
        configurable frontier edge continuation logits over all legal frontier
        edges. The default successor_value scorer evaluates each successor state
        with policy-only value and edge-advantage heads.

    Stop policy:
        learned Stop head normalized against frontier continuation mass.
        FrontierPointerPolicy is diagnostic-only unless the pointer scorer is
        explicitly selected for ablation.
    """

    def __init__(
        self,
        *,
        feature_encoder_cfg: dict[str, Any],
        hidden_dim: int = 1024,
        mode: str = "bdb",
        max_budget: int = 8,
        flow_budget_conditioning: str = "none",
        te_bfm_lookahead_depth: int = 2,
        te_bfm_child_chunk_size: int = 4096,
        te_bfm_terminal_chunk_size: int = 4096,
        te_bfm_max_backup_edges_per_state: int | None = None,
        te_bfm_include_counterfactual_internal_states: bool = False,
        te_bfm_max_expanded_states: int = 1000000,
        bdb_child_chunk_size: int = 2048,
        edge_scorer: str = "relation_residual",
        continuation_logit_bias_init: float = -5.912023,
        continuation_mass_reduction: str = "logsumexp",
        evidence_state_encoder_dropout: float = 0.0,
        evidence_state_encoder_cfg: dict[str, Any] | None = None,
        flow_head_cfg: dict[str, Any] | None = None,
        frontier_pointer_cfg: dict[str, Any] | None = None,
        relation_residual_edge_scorer_cfg: dict[str, Any] | None = None,
        stop_head_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.mode = str(mode).lower()
        if self.mode not in {"bdb"}:
            raise ValueError(f"Unsupported policy mode {mode!r}.")
        self.max_budget = int(max_budget)
        if self.max_budget < 0:
            raise ValueError(f"max_budget must be non-negative, got {max_budget}.")
        self.flow_budget_conditioning = str(flow_budget_conditioning).lower()
        if self.flow_budget_conditioning not in {"none", "additive"}:
            raise ValueError(
                "flow_budget_conditioning must be 'none' or 'additive', "
                f"got {flow_budget_conditioning!r}."
            )
        self.te_bfm_lookahead_depth = int(te_bfm_lookahead_depth)
        self.te_bfm_child_chunk_size = int(te_bfm_child_chunk_size)
        self.te_bfm_terminal_chunk_size = int(te_bfm_terminal_chunk_size)
        self.te_bfm_max_backup_edges_per_state = (
            None
            if te_bfm_max_backup_edges_per_state is None
            else int(te_bfm_max_backup_edges_per_state)
        )
        self.te_bfm_include_counterfactual_internal_states = bool(
            te_bfm_include_counterfactual_internal_states
        )
        self.te_bfm_max_expanded_states = int(te_bfm_max_expanded_states)
        self.bdb_child_chunk_size = int(bdb_child_chunk_size)
        # REMOVED: TE-BFM backup hyperparameter validation — see methodology.md §3.9
        if self.bdb_child_chunk_size < 1:
            raise ValueError("bdb_child_chunk_size must be >= 1.")
        self._last_te_bfm_terminal_quotient_diagnostics: dict[str, float] = {
            "terminal_quotient_backup_used_rate": 0.0,
            "terminal_quotient_parent_count": 0.0,
            "terminal_quotient_edge_count": 0.0,
            "terminal_quotient_group_count_mean": 0.0,
            "terminal_quotient_floor_edge_count_mean": 0.0,
            "terminal_quotient_positive_edge_count_mean": 0.0,
            "terminal_quotient_speedup_estimate": 0.0,
            "frontier_cap_used_rate": 0.0,
            "frontier_cap_dropped_edge_count_mean": 0.0,
        }
        self.edge_scorer = str(edge_scorer)
        allowed_edge_scorers = {"relation_residual", "pointer"}
        if self.edge_scorer not in allowed_edge_scorers:
            raise ValueError(
                "edge_scorer must be one of "
                f"{sorted(allowed_edge_scorers)}, got {self.edge_scorer!r}."
            )
        self.continuation_logit_bias = nn.Parameter(
            torch.tensor(float(continuation_logit_bias_init), dtype=torch.float32)
        )
        self.continuation_mass_reduction = str(continuation_mass_reduction)
        allowed_mass_reductions = {"logsumexp", "logmeanexp"}
        if self.continuation_mass_reduction not in allowed_mass_reductions:
            raise ValueError(
                "continuation_mass_reduction must be one of "
                f"{sorted(allowed_mass_reductions)}, "
                f"got {self.continuation_mass_reduction!r}."
            )

        self.feature_encoder = FeatureEncoder(**feature_encoder_cfg)

        state_encoder_kwargs = dict(evidence_state_encoder_cfg or {})
        state_encoder_kwargs.setdefault(
            "dropout",
            float(evidence_state_encoder_dropout),
        )
        self.state_encoder = EvidenceStateEncoder(
            hidden_dim=self.hidden_dim,
            **state_encoder_kwargs,
        )

        self.flow_head = FlowHead(
            hidden_dim=self.hidden_dim,
            **(flow_head_cfg or {}),
        )
        self.budget_embedding = (
            nn.Embedding(self.max_budget + 1, self.hidden_dim)
            if self.flow_budget_conditioning == "additive"
            else None
        )
        if stop_head_cfg:
            raise ValueError("policy.stop_head was replaced by TerminalHead.")
        self.terminal_head = TerminalHead(hidden_dim=self.hidden_dim)
        self.transition_feature_builder = TransitionFeatureBuilder(
            hidden_dim=self.hidden_dim,
        )
        self.edge_residual_scorer = EdgeResidualScorer(
            hidden_dim=self.hidden_dim,
            feature_dim=self.transition_feature_builder.feature_dim,
        )
        self.successor_value_head = SuccessorValueHead(hidden_dim=self.hidden_dim)
        self.successor_edge_advantage_scorer = SuccessorEdgeAdvantageScorer(
            hidden_dim=self.hidden_dim,
            feature_dim=self.transition_feature_builder.feature_dim,
        )
        self.frontier_pointer = FrontierPointerPolicy(
            hidden_dim=self.hidden_dim,
            **(frontier_pointer_cfg or {}),
        )
        self.relation_residual_edge_scorer = RelationResidualEdgeScorer(
            hidden_dim=self.hidden_dim,
            **(relation_residual_edge_scorer_cfg or {}),
        )

    def set_residual_warmup_step(self, step: int) -> None:
        self.relation_residual_edge_scorer.set_warmup_step(int(step))

    def edge_prior_diagnostics_summary(self) -> dict[str, float]:
        return self.relation_residual_edge_scorer.diagnostics_summary()

    def prepare_rollout_context(self, batch: RetrievalBatch) -> PolicyContext:
        fb = self.feature_encoder(batch)
        return PolicyContext(fb=fb)

    def forward(
        self,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext | FeatureBank | None = None,
        *,
        reward_model: Any | None = None,
        remaining_budget: torch.Tensor | None = None,
        return_edge_diagnostics: bool = False,
        compute_bdb_trace: bool = False,
    ) -> PolicyOutput:
        policy_context = self._coerce_policy_context(batch, rollout_context)
        fb = policy_context.fb
        device = fb.node_h.device

        self._validate_feature_bank(
            fb=fb,
            batch=batch,
            num_graphs=int(batch.num_graphs),
        )

        context = self.state_encoder(fb=fb, batch=batch, state=state)
        num_policy_graphs = int(context.state_h.size(0))
        if remaining_budget is None:
            remaining_budget = state.remaining_budget_per_graph(
                edge_batch=batch.edge_batch,
                num_graphs=num_policy_graphs,
            )
        remaining_budget = remaining_budget.to(device=device, dtype=torch.long).view(-1)
        if remaining_budget.shape != (num_policy_graphs,):
            raise ValueError(
                "remaining_budget must have shape "
                f"[{num_policy_graphs}], got {tuple(remaining_budget.shape)}."
            )
        frontier_context = build_frontier(
            fb=fb,
            batch=batch,
            state=state,
            frontier_mode=self._frontier_mode(),
        )
        frontier_edge_ids = frontier_context.edge_ids.to(
            device=device,
            dtype=torch.long,
        )
        frontier_batch_ids = frontier_context.graph_id.to(
            device=device,
            dtype=torch.long,
        )
        full_frontier_counts = torch.bincount(
            frontier_batch_ids,
            minlength=num_policy_graphs,
        ).to(device=device, dtype=torch.float32)
        if frontier_edge_ids.numel() > 0:
            frontier_has_budget = remaining_budget.gt(0).index_select(
                0,
                frontier_batch_ids,
            )
            frontier_edge_ids = frontier_edge_ids[frontier_has_budget]
            frontier_batch_ids = frontier_batch_ids[frontier_has_budget]
            frontier_context = _filter_frontier_context(
                frontier_context,
                frontier_has_budget,
                device=device,
            )
        state_log_flow = self.evaluate_state_log_flow(
            batch=batch,
            state=state,
            remaining_budget=remaining_budget,
            rollout_context=policy_context,
            context=context,
        )
        # REMOVED: reward/backup-derived inference logits — see methodology.md §3.6
        depth = (
            torch.full_like(remaining_budget, int(state.expand_budget))
            - remaining_budget
        ).clamp_min(0)
        (
            expand_logits,
            continue_energy,
            value_energy,
            terminal_energy,
        ) = self._learned_action_terms(
            batch=batch,
            state=state,
            rollout_context=policy_context,
            frontier_batch_ids=frontier_batch_ids,
            frontier_edge_ids=frontier_edge_ids,
            context=context,
            frontier_context=frontier_context,
            depth=depth,
            remaining_budget=remaining_budget,
            num_graphs=num_policy_graphs,
            dtype=state_log_flow.dtype,
            device=state_log_flow.device,
        )
        edge_diagnostics = None
        if return_edge_diagnostics:
            edge_diagnostics = self._edge_policy_diagnostics(
                batch=batch,
                state=state,
                fb=policy_context.fb,
                context=context,
                frontier_edge_ids=frontier_edge_ids,
                frontier_batch_ids=frontier_batch_ids,
                frontier_context=frontier_context,
                depth=depth,
                remaining_budget=remaining_budget,
                num_graphs=num_policy_graphs,
                dtype=state_log_flow.dtype,
                device=state_log_flow.device,
            )
        (
            log_p_stop,
            log_p_continue,
            edge_cond_logprob,
            edge_expand_logprob,
        ) = budgeted_energy_policy_log_probs(
            terminal_energy=terminal_energy,
            expand_logits=expand_logits,
            frontier_batch_ids=frontier_batch_ids,
            remaining_budget=remaining_budget,
            num_graphs=num_policy_graphs,
        )
        bdb_trace: dict[str, torch.Tensor] = {}
        if self.mode == "bdb" and compute_bdb_trace:
            if reward_model is None:
                raise ValueError("reward_model is required when compute_bdb_trace=True.")
            bdb_trace = self._bdb_trace(
                batch=batch,
                state=state,
                rollout_context=policy_context,
                frontier_edge_ids=frontier_edge_ids,
                frontier_batch_ids=frontier_batch_ids,
                remaining_budget=remaining_budget,
                state_log_flow=state_log_flow,
                log_p_stop=log_p_stop,
                edge_expand_logprob=edge_expand_logprob,
                terminal_energy=terminal_energy,
                value_energy=value_energy,
                full_frontier_counts=full_frontier_counts.to(
                    device=state_log_flow.device,
                    dtype=state_log_flow.dtype,
                ),
                reward_model=reward_model,
                dtype=state_log_flow.dtype,
                device=state_log_flow.device,
            )
        return PolicyOutput(
            stop_logits=terminal_energy,
            edge_logits=expand_logits,
            state_log_flow=state_log_flow,
            log_p_stop=log_p_stop,
            log_p_continue=log_p_continue,
            edge_cond_logprob=edge_cond_logprob,
            edge_expand_logprob=edge_expand_logprob,
            frontier_edge_ids=frontier_edge_ids,
            frontier_batch_ids=frontier_batch_ids,
            edge_policy_diagnostics=edge_diagnostics,
            log_c_continue=continue_energy,
            log_z_action=value_energy,
            terminal_energy=terminal_energy,
            continue_energy=continue_energy,
            value_energy=value_energy,
            **bdb_trace,
        )

    def _bdb_trace(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        state_log_flow: torch.Tensor,
        log_p_stop: torch.Tensor,
        edge_expand_logprob: torch.Tensor,
        terminal_energy: torch.Tensor,
        value_energy: torch.Tensor,
        full_frontier_counts: torch.Tensor,
        reward_model: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        num_graphs = int(state_log_flow.numel())
        with torch.no_grad():
            reward = reward_model.evaluate_terminal_state(
                retrieval_batch=batch,
                state=state,
                diagnostics="basic",
            )
        log_reward = (
            reward.log_reward.to(device=device, dtype=dtype).view(num_graphs).detach()
        )
        remaining = remaining_budget.to(device=device, dtype=torch.long).view(
            num_graphs
        )
        frontier_counts = full_frontier_counts.to(device=device, dtype=dtype).view(
            num_graphs
        )
        base_valid = remaining.eq(0) | frontier_counts.eq(0)
        non_base = ~base_valid

        delta_base = terminal_energy.to(device=device, dtype=dtype) - log_reward
        base_loss = torch.where(
            base_valid,
            delta_base.square(),
            state_log_flow.new_zeros((num_graphs,)),
        )
        delta_stop = terminal_energy.to(device=device, dtype=dtype) - log_reward
        stop_loss = torch.where(
            non_base,
            delta_stop.square(),
            state_log_flow.new_zeros((num_graphs,)),
        )

        edge_loss_sum = state_log_flow.new_zeros((num_graphs,))
        edge_delta_sum = state_log_flow.new_zeros((num_graphs,))
        parent_count_sum = state_log_flow.new_zeros((num_graphs,))
        edge_counts = state_log_flow.new_zeros((num_graphs,))
        if frontier_edge_ids.numel() > 0:
            row_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
            edge_ids = frontier_edge_ids.to(device=device, dtype=torch.long).view(-1)
            active_edge = non_base.index_select(0, row_ids)
            if bool(active_edge.any()):
                active_pos = active_edge.nonzero(as_tuple=False).flatten()
                for chunk_pos in active_pos.split(self.bdb_child_chunk_size):
                    chunk_edges = edge_ids.index_select(0, chunk_pos)
                    chunk_rows = row_ids.index_select(0, chunk_pos)
                    child_state = _successor_state_for_frontier(
                        batch=batch,
                        state=state,
                        frontier_edge_ids=chunk_edges,
                        frontier_batch_ids=chunk_rows,
                    )
                    child_budget = remaining.index_select(0, chunk_rows) - 1
                    child_context = self.state_encoder(
                        fb=rollout_context.fb,
                        batch=batch,
                        state=child_state,
                    )
                    child_terminal = self.terminal_head(
                        state_h=child_context.state_h.to(device=device, dtype=dtype),
                    ).to(device=device, dtype=dtype)
                    child_target = child_terminal
                    parent_counts = _uniform_parent_counts_for_successor_edges(
                        batch=batch,
                        successor=child_state,
                        frontier_edge_ids=chunk_edges,
                    ).to(device=device, dtype=dtype)
                    log_pb = -parent_counts.clamp_min(1.0).log()
                    delta_edge = (
                        value_energy.to(device=device, dtype=dtype).index_select(0, chunk_rows)
                        + edge_expand_logprob.index_select(0, chunk_pos).to(
                            device=device,
                            dtype=dtype,
                        )
                        - child_target.detach()
                        - log_pb
                    )
                    delta_edge = delta_edge.to(dtype=edge_loss_sum.dtype)
                    contribution = delta_edge.square().to(dtype=edge_loss_sum.dtype)
                    parent_counts = parent_counts.to(dtype=parent_count_sum.dtype)
                    edge_loss_sum.scatter_add_(0, chunk_rows, contribution)
                    edge_delta_sum.scatter_add_(0, chunk_rows, delta_edge.detach())
                    parent_count_sum.scatter_add_(0, chunk_rows, parent_counts.detach())
                    edge_counts.scatter_add_(
                        0,
                        chunk_rows,
                        torch.ones_like(edge_counts.index_select(0, chunk_rows)),
                    )

        edge_valid = non_base & edge_counts.gt(0)
        edge_loss = torch.where(
            edge_valid,
            edge_loss_sum / edge_counts.clamp_min(1.0),
            state_log_flow.new_zeros((num_graphs,)),
        )
        delta_edge = torch.where(
            edge_valid,
            edge_delta_sum / edge_counts.clamp_min(1.0),
            state_log_flow.new_zeros((num_graphs,)),
        )
        parent_count = torch.where(
            edge_valid,
            parent_count_sum / edge_counts.clamp_min(1.0),
            state_log_flow.new_zeros((num_graphs,)),
        )

        return {
            "bdb_stop_loss": stop_loss,
            "bdb_edge_loss": edge_loss,
            "bdb_base_loss": base_loss,
            "bdb_stop_valid_mask": non_base,
            "bdb_edge_valid_mask": edge_valid,
            "bdb_base_valid_mask": base_valid,
            "bdb_delta_stop": delta_stop.detach(),
            "bdb_delta_edge": delta_edge.detach(),
            "bdb_delta_base": delta_base.detach(),
            "bdb_frontier_size": frontier_counts.detach(),
            "bdb_parent_count": parent_count.detach(),
            "bdb_log_reward": log_reward.detach(),
            "bdb_log_flow": value_energy.detach(),
        }

    def evaluate_state_log_flow(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        remaining_budget: torch.Tensor,
        rollout_context: PolicyContext | FeatureBank | None = None,
        context: Any | None = None,
    ) -> torch.Tensor:
        policy_context = self._coerce_policy_context(batch, rollout_context)
        if context is None:
            context = self.state_encoder(fb=policy_context.fb, batch=batch, state=state)
        state_h = context.state_h
        if self.budget_embedding is not None:
            budget = remaining_budget.to(
                device=state_h.device,
                dtype=torch.long,
            ).view(-1)
            budget = budget.clamp(0, self.max_budget)
            if budget.shape != (int(state_h.size(0)),):
                raise ValueError(
                    "remaining_budget must match state rows for flow evaluation: "
                    f"{tuple(budget.shape)} != {(int(state_h.size(0)),)}."
                )
            state_h = state_h + self.budget_embedding(budget).to(
                dtype=state_h.dtype,
            )
        return self.flow_head(state_h=state_h)

    def _te_bfm_policy_output(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        context: Any,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        state_log_flow: torch.Tensor,
        reward_model: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> PolicyOutput:
        # REMOVED: reward/backup-derived action logits — see methodology.md §3.6
        raise RuntimeError("TE-BFM policy output was removed; BDB uses learned logits.")

    def _removed_te_bfm_policy_output(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        context: Any,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        state_log_flow: torch.Tensor,
        reward_model: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> PolicyOutput:
        num_graphs = int(state_log_flow.numel())
        reward = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            state=state,
            diagnostics="basic",
        )
        log_reward = reward.log_reward.to(device=device, dtype=dtype).view(num_graphs)
        cap_used_rate = self._last_te_bfm_terminal_quotient_diagnostics.get(
            "frontier_cap_used_rate",
            0.0,
        )
        cap_dropped_mean = self._last_te_bfm_terminal_quotient_diagnostics.get(
            "frontier_cap_dropped_edge_count_mean",
            0.0,
        )
        self._last_te_bfm_terminal_quotient_diagnostics = {
            "terminal_quotient_backup_used_rate": 0.0,
            "terminal_quotient_parent_count": 0.0,
            "terminal_quotient_edge_count": 0.0,
            "terminal_quotient_group_count_mean": 0.0,
            "terminal_quotient_floor_edge_count_mean": 0.0,
            "terminal_quotient_positive_edge_count_mean": 0.0,
            "terminal_quotient_speedup_estimate": 0.0,
            "frontier_cap_used_rate": cap_used_rate,
            "frontier_cap_dropped_edge_count_mean": cap_dropped_mean,
        }
        edge_logits = self._te_bfm_edge_logits(
            batch=batch,
            state=state,
            rollout_context=rollout_context,
            frontier_edge_ids=frontier_edge_ids,
            frontier_batch_ids=frontier_batch_ids,
            remaining_budget=remaining_budget,
            reward_model=reward_model,
            dtype=dtype,
            device=device,
        )
        log_c = segment_logsumexp_or_neg_inf(
            values=edge_logits,
            segment_ids=frontier_batch_ids,
            num_graphs=num_graphs,
            device=device,
            dtype=dtype,
        )
        expandable = remaining_budget.gt(0)
        stop_logits = log_reward - log_c
        stop_logits = torch.where(
            expandable & torch.isfinite(log_c),
            stop_logits,
            torch.full_like(stop_logits, torch.inf),
        )
        log_z = torch.logaddexp(log_reward, log_c)
        target = torch.where(expandable, log_z, log_reward)
        residual = state_log_flow - target.detach()
        valid = remaining_budget.gt(0)
        parent_loss = torch.where(
            valid,
            residual.square(),
            residual.new_zeros(residual.shape),
        )
        if self.te_bfm_include_counterfactual_internal_states:
            internal_loss = self._te_bfm_counterfactual_child_loss(
                batch=batch,
                state=state,
                rollout_context=rollout_context,
                frontier_edge_ids=frontier_edge_ids,
                frontier_batch_ids=frontier_batch_ids,
                remaining_budget=remaining_budget,
                reward_model=reward_model,
                dtype=dtype,
                device=device,
            )
        else:
            internal_loss = torch.zeros_like(parent_loss)
        loss = parent_loss + internal_loss

        (
            log_p_stop,
            log_p_continue,
            edge_cond_logprob,
            edge_expand_logprob,
        ) = hazard_policy_log_probs(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            frontier_batch_ids=frontier_batch_ids,
            num_graphs=num_graphs,
        )
        frontier_counts = torch.bincount(
            frontier_batch_ids.to(device=device, dtype=torch.long),
            minlength=num_graphs,
        ).to(device=device, dtype=dtype)

        return PolicyOutput(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            state_log_flow=state_log_flow,
            log_p_stop=log_p_stop,
            log_p_continue=log_p_continue,
            edge_cond_logprob=edge_cond_logprob,
            edge_expand_logprob=edge_expand_logprob,
            frontier_edge_ids=frontier_edge_ids,
            frontier_batch_ids=frontier_batch_ids,
            edge_policy_diagnostics=None,
            log_c_continue=log_c,
            log_z_action=target,
            te_bfm_loss=loss,
            te_bfm_valid_mask=valid,
            te_bfm_residual_abs=residual.abs(),
            te_bfm_target_log_value=target.detach(),
            te_bfm_log_reward=log_reward.detach(),
            te_bfm_stop_prob=log_p_stop.exp().detach(),
            te_bfm_frontier_edge_count=frontier_counts.detach(),
            te_bfm_counterfactual_child_loss=internal_loss.detach(),
            te_bfm_frontier_cap_used=torch.full(
                (num_graphs,),
                float(
                    self._last_te_bfm_terminal_quotient_diagnostics[
                        "frontier_cap_used_rate"
                    ]
                ),
                dtype=dtype,
                device=device,
            ),
            te_bfm_frontier_cap_dropped_edge_count=torch.full(
                (num_graphs,),
                float(
                    self._last_te_bfm_terminal_quotient_diagnostics[
                        "frontier_cap_dropped_edge_count_mean"
                    ]
                ),
                dtype=dtype,
                device=device,
            ),
            terminal_quotient_backup_used_rate=edge_logits.new_tensor(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "terminal_quotient_backup_used_rate"
                ]
            ),
            terminal_quotient_parent_count=edge_logits.new_tensor(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "terminal_quotient_parent_count"
                ]
            ),
            terminal_quotient_edge_count=edge_logits.new_tensor(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "terminal_quotient_edge_count"
                ]
            ),
            terminal_quotient_group_count_mean=edge_logits.new_tensor(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "terminal_quotient_group_count_mean"
                ]
            ),
            terminal_quotient_floor_edge_count_mean=edge_logits.new_tensor(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "terminal_quotient_floor_edge_count_mean"
                ]
            ),
            terminal_quotient_positive_edge_count_mean=edge_logits.new_tensor(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "terminal_quotient_positive_edge_count_mean"
                ]
            ),
            terminal_quotient_speedup_estimate=edge_logits.new_tensor(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "terminal_quotient_speedup_estimate"
                ]
            ),
        )

    def _te_bfm_counterfactual_child_loss(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        reward_model: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if frontier_edge_ids.numel() == 0 or self.te_bfm_lookahead_depth <= 1:
            return torch.zeros(
                remaining_budget.shape,
                dtype=dtype,
                device=device,
            )
        if int(frontier_edge_ids.numel()) > self.te_bfm_max_expanded_states:
            raise RuntimeError(
                "TE-BFM counterfactual child fit exceeded max_expanded_states: "
                f"{int(frontier_edge_ids.numel())} > "
                f"{self.te_bfm_max_expanded_states}."
            )
        parent_budget = remaining_budget.to(device=device, dtype=torch.long)
        edge_budget = parent_budget.index_select(
            0,
            frontier_batch_ids.to(device=device, dtype=torch.long),
        )
        child_budget = edge_budget - 1
        train_child = child_budget.gt(0)
        if not bool(train_child.any()):
            return torch.zeros(
                remaining_budget.shape,
                dtype=dtype,
                device=device,
            )
        edge_ids = frontier_edge_ids[train_child]
        row_ids = frontier_batch_ids[train_child]
        child_budget = child_budget[train_child]
        sums = torch.zeros(
            int(remaining_budget.numel()),
            dtype=dtype,
            device=device,
        )
        counts = torch.zeros_like(sums)
        positions = torch.arange(int(edge_ids.numel()), dtype=torch.long, device=device)
        for chunk_pos in positions.split(self.te_bfm_child_chunk_size):
            chunk_edges = edge_ids.index_select(0, chunk_pos)
            chunk_rows = row_ids.index_select(0, chunk_pos)
            chunk_budget = child_budget.index_select(0, chunk_pos)
            child_state = _successor_state_for_frontier(
                batch=batch,
                state=state,
                frontier_edge_ids=chunk_edges,
                frontier_batch_ids=chunk_rows,
            )
            target = self._te_bfm_value(
                batch=batch,
                state=child_state,
                remaining_budget=chunk_budget,
                lookahead_depth=self.te_bfm_lookahead_depth - 1,
                reward_model=reward_model,
                rollout_context=rollout_context,
                detach_bootstrap=True,
            ).to(device=device, dtype=dtype)
            pred = self.evaluate_state_log_flow(
                batch=batch,
                state=child_state,
                remaining_budget=chunk_budget,
                rollout_context=rollout_context,
            ).to(device=device, dtype=dtype)
            child_loss = (pred - target.detach()).square()
            scatter_rows = chunk_rows.to(device=device, dtype=torch.long)
            sums.scatter_add_(0, scatter_rows, child_loss)
            counts.scatter_add_(
                0,
                scatter_rows,
                torch.ones_like(child_loss),
            )
        return torch.where(counts.gt(0), sums / counts.clamp_min(1.0), sums)

    def _te_bfm_edge_logits(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        reward_model: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if frontier_edge_ids.numel() == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        parent_budget = remaining_budget.to(device=device, dtype=torch.long)
        edge_budget = parent_budget.index_select(
            0,
            frontier_batch_ids.to(device=device, dtype=torch.long),
        )
        active_edge = edge_budget.gt(0)
        if not bool(active_edge.any()):
            return torch.empty((0,), dtype=dtype, device=device)
        out = torch.full(
            frontier_edge_ids.shape,
            -torch.inf,
            dtype=dtype,
            device=device,
        )
        active_positions = active_edge.nonzero(as_tuple=False).flatten()
        terminal_positions = (
            active_positions[edge_budget.index_select(0, active_positions).eq(1)]
            if isinstance(state, RolloutState)
            else active_positions.new_empty((0,))
        )
        if terminal_positions.numel() > 0:
            terminal = self._te_bfm_terminal_backup_quotient(
                batch=batch,
                state=state,
                frontier_edge_ids=frontier_edge_ids.index_select(0, terminal_positions),
                frontier_batch_ids=frontier_batch_ids.index_select(
                    0,
                    terminal_positions,
                ),
                reward_model=reward_model,
                dtype=dtype,
                device=device,
            )
            out[terminal_positions] = terminal.edge_terms.to(device=device, dtype=dtype)

        recursive_positions = (
            active_positions[edge_budget.index_select(0, active_positions).gt(1)]
            if isinstance(state, RolloutState)
            else active_positions
        )
        if int(recursive_positions.numel()) > self.te_bfm_max_expanded_states:
            raise RuntimeError(
                "TE-BFM frontier expansion exceeded max_expanded_states: "
                f"{int(recursive_positions.numel())} > "
                f"{self.te_bfm_max_expanded_states}."
            )
        if recursive_positions.numel() > 0:
            for chunk_pos in recursive_positions.split(self.te_bfm_child_chunk_size):
                edge_ids = frontier_edge_ids.index_select(0, chunk_pos)
                row_ids = frontier_batch_ids.index_select(0, chunk_pos)
                child_state = _successor_state_for_frontier(
                    batch=batch,
                    state=state,
                    frontier_edge_ids=edge_ids,
                    frontier_batch_ids=row_ids,
                )
                child_budget = edge_budget.index_select(0, chunk_pos) - 1
                child_value = self._te_bfm_value(
                    batch=batch,
                    state=child_state,
                    remaining_budget=child_budget,
                    lookahead_depth=self.te_bfm_lookahead_depth - 1,
                    reward_model=reward_model,
                    rollout_context=rollout_context,
                    detach_bootstrap=True,
                ).to(device=device, dtype=dtype)
                log_pb = _uniform_log_pb_for_successor_edges(
                    batch=batch,
                    successor=child_state,
                    frontier_edge_ids=edge_ids,
                ).to(device=device, dtype=dtype)
                out[chunk_pos] = child_value + log_pb
        return out

    def _te_bfm_terminal_backup_quotient(
        self,
        *,
        batch: RetrievalBatch,
        state: RolloutState,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        reward_model: Any,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _TerminalQuotientBackup:
        row_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
        edge_term_chunks: list[torch.Tensor] = []
        log_reward_chunks: list[torch.Tensor] = []
        log_pb_chunks: list[torch.Tensor] = []
        supported_chunks: list[torch.Tensor] = []
        edge_ids = frontier_edge_ids.to(device=device, dtype=torch.long).view(-1)
        for chunk_pos in torch.arange(
            int(edge_ids.numel()),
            dtype=torch.long,
            device=device,
        ).split(self.te_bfm_terminal_chunk_size):
            terminal_terms = _te_bfm_terminal_edge_terms(
                batch=batch,
                state=state,
                frontier_edge_ids=edge_ids.index_select(0, chunk_pos),
                frontier_batch_ids=row_ids.index_select(0, chunk_pos),
                reward_model=reward_model,
                dtype=dtype,
                device=device,
            )
            edge_term_chunks.append(terminal_terms.edge_terms)
            log_reward_chunks.append(terminal_terms.log_reward)
            log_pb_chunks.append(terminal_terms.log_pb)
            supported_chunks.append(terminal_terms.supported)
        edge_terms = (
            torch.cat(edge_term_chunks, dim=0)
            if edge_term_chunks
            else torch.empty((0,), dtype=dtype, device=device)
        )
        supported = (
            torch.cat(supported_chunks, dim=0)
            if supported_chunks
            else torch.empty((0,), dtype=dtype, device=device)
        )
        log_rewards = (
            torch.cat(log_reward_chunks, dim=0)
            if log_reward_chunks
            else torch.empty((0,), dtype=dtype, device=device)
        )
        log_pbs = (
            torch.cat(log_pb_chunks, dim=0)
            if log_pb_chunks
            else torch.empty((0,), dtype=dtype, device=device)
        )
        group_masses, group_rows = _terminal_quotient_group_masses(
            row_ids=row_ids,
            log_rewards=log_rewards,
            log_pbs=log_pbs,
            dtype=dtype,
            device=device,
        )
        continuation = segment_logsumexp_or_neg_inf(
            values=group_masses,
            segment_ids=group_rows,
            num_graphs=state.num_rollouts,
            device=device,
            dtype=dtype,
        )
        counts = torch.bincount(row_ids, minlength=state.num_rollouts).to(
            device=device,
            dtype=dtype,
        )
        positive = supported.gt(0)
        positive_counts = torch.zeros_like(counts)
        floor_counts = torch.zeros_like(counts)
        positive_counts.scatter_add_(
            0,
            row_ids,
            positive.to(device=device, dtype=dtype),
        )
        floor_counts.scatter_add_(
            0,
            row_ids,
            (~positive).to(device=device, dtype=dtype),
        )
        active = counts.gt(0)
        parent_count = int(active.sum().item())
        edge_count = int(row_ids.numel())
        group_counts = torch.bincount(group_rows, minlength=state.num_rollouts).to(
            device=device,
            dtype=dtype,
        )
        group_count_mean = (
            float(group_counts[active].mean().item()) if bool(active.any()) else 0.0
        )
        floor_mean = (
            float(floor_counts[active].mean().item()) if bool(active.any()) else 0.0
        )
        positive_mean = (
            float(positive_counts[active].mean().item()) if bool(active.any()) else 0.0
        )
        group_count_total = int(group_rows.numel())
        speedup = float(edge_count) / float(max(group_count_total, 1))
        cap_used_rate = self._last_te_bfm_terminal_quotient_diagnostics.get(
            "frontier_cap_used_rate",
            0.0,
        )
        cap_dropped_mean = self._last_te_bfm_terminal_quotient_diagnostics.get(
            "frontier_cap_dropped_edge_count_mean",
            0.0,
        )
        self._last_te_bfm_terminal_quotient_diagnostics = {
            "terminal_quotient_backup_used_rate": 1.0 if edge_count > 0 else 0.0,
            "terminal_quotient_parent_count": float(parent_count),
            "terminal_quotient_edge_count": float(edge_count),
            "terminal_quotient_group_count_mean": group_count_mean,
            "terminal_quotient_floor_edge_count_mean": floor_mean,
            "terminal_quotient_positive_edge_count_mean": positive_mean,
            "terminal_quotient_speedup_estimate": speedup,
            "frontier_cap_used_rate": cap_used_rate,
            "frontier_cap_dropped_edge_count_mean": cap_dropped_mean,
        }
        return _TerminalQuotientBackup(
            continuation=continuation,
            edge_terms=edge_terms,
            row_ids=row_ids,
            parent_count=parent_count,
            edge_count=edge_count,
            group_count_mean=group_count_mean,
            floor_edge_count_mean=floor_mean,
            positive_edge_count_mean=positive_mean,
            speedup_estimate=speedup,
        )

    def _te_bfm_value(
        self,
        *,
        batch: RetrievalBatch,
        state: RolloutState,
        remaining_budget: torch.Tensor,
        lookahead_depth: int,
        reward_model: Any,
        rollout_context: PolicyContext,
        detach_bootstrap: bool,
    ) -> torch.Tensor:
        device = remaining_budget.device
        budget = remaining_budget.to(device=device, dtype=torch.long).view(-1)
        if budget.shape != (state.num_rollouts,):
            raise ValueError(
                "remaining_budget must match TE-BFM state rows: "
                f"{tuple(budget.shape)} != {(state.num_rollouts,)}."
            )
        reward = reward_model.evaluate_terminal_state(
            retrieval_batch=batch,
            state=state,
            diagnostics="basic",
        )
        log_reward = reward.log_reward.to(device=device, dtype=torch.float32).view(-1)
        if int(lookahead_depth) <= 0:
            flow = self.evaluate_state_log_flow(
                batch=batch,
                state=state,
                remaining_budget=budget,
                rollout_context=rollout_context,
            ).to(device=device, dtype=torch.float32)
            if detach_bootstrap:
                flow = flow.detach()
            return torch.where(budget.eq(0), log_reward, flow)
        if bool(budget.eq(0).all()):
            return log_reward

        frontier = build_frontier(
            fb=rollout_context.fb,
            batch=batch,
            state=state,
            frontier_mode="boundary",
        )
        if self.te_bfm_max_backup_edges_per_state is not None:
            frontier, cap_stats = _cap_frontier_context_by_semantic_topk(
                fb=rollout_context.fb,
                frontier=frontier,
                max_edges_per_state=self.te_bfm_max_backup_edges_per_state,
                device=device,
            )
            self._last_te_bfm_terminal_quotient_diagnostics[
                "frontier_cap_used_rate"
            ] = max(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "frontier_cap_used_rate"
                ],
                cap_stats.used_rate,
            )
            self._last_te_bfm_terminal_quotient_diagnostics[
                "frontier_cap_dropped_edge_count_mean"
            ] = max(
                self._last_te_bfm_terminal_quotient_diagnostics[
                    "frontier_cap_dropped_edge_count_mean"
                ],
                cap_stats.dropped_edge_count_mean,
            )
        edge_ids = frontier.edge_ids.to(device=device, dtype=torch.long)
        row_ids = frontier.graph_id.to(device=device, dtype=torch.long)
        if edge_ids.numel() > 0:
            has_budget = budget.gt(0).index_select(0, row_ids)
            edge_ids = edge_ids[has_budget]
            row_ids = row_ids[has_budget]
        if edge_ids.numel() == 0:
            return log_reward

        child_terms: list[torch.Tensor] = []
        child_rows: list[torch.Tensor] = []
        terminal_edge = budget.index_select(0, row_ids).eq(1)
        if bool(terminal_edge.any()):
            terminal = self._te_bfm_terminal_backup_quotient(
                batch=batch,
                state=state,
                frontier_edge_ids=edge_ids[terminal_edge],
                frontier_batch_ids=row_ids[terminal_edge],
                reward_model=reward_model,
                dtype=log_reward.dtype,
                device=device,
            )
            child_terms.append(terminal.edge_terms)
            child_rows.append(terminal.row_ids)

        recursive_edge = ~terminal_edge
        recursive_edge_ids = edge_ids[recursive_edge]
        recursive_row_ids = row_ids[recursive_edge]
        if int(recursive_edge_ids.numel()) > self.te_bfm_max_expanded_states:
            diagnostics = _te_bfm_recursive_expansion_diagnostics(
                batch=batch,
                state=state,
                row_ids=recursive_row_ids,
                expanded_state_count=int(recursive_edge_ids.numel()),
                max_expanded_states=self.te_bfm_max_expanded_states,
                remaining_budget=budget,
                lookahead_depth=int(lookahead_depth),
            )
            raise RuntimeError(
                "TE-BFM recursive expansion exceeded max_expanded_states: "
                f"{int(recursive_edge_ids.numel())} > {self.te_bfm_max_expanded_states}. "
                "This usually indicates a high-fanout exact lookahead outlier. "
                "diagnostics="
                f"{json.dumps(diagnostics, sort_keys=True)}"
            )

        if recursive_edge_ids.numel() > 0:
            positions = torch.arange(
                int(recursive_edge_ids.numel()),
                dtype=torch.long,
                device=device,
            )
            for chunk_pos in positions.split(self.te_bfm_child_chunk_size):
                chunk_edges = recursive_edge_ids.index_select(0, chunk_pos)
                chunk_rows = recursive_row_ids.index_select(0, chunk_pos)
                child_state = _successor_state_for_frontier(
                    batch=batch,
                    state=state,
                    frontier_edge_ids=chunk_edges,
                    frontier_batch_ids=chunk_rows,
                )
                child_budget = budget.index_select(0, chunk_rows) - 1
                child_value = self._te_bfm_value(
                    batch=batch,
                    state=child_state,
                    remaining_budget=child_budget,
                    lookahead_depth=int(lookahead_depth) - 1,
                    reward_model=reward_model,
                    rollout_context=rollout_context,
                    detach_bootstrap=detach_bootstrap,
                )
                log_pb = _uniform_log_pb_for_successor_edges(
                    batch=batch,
                    successor=child_state,
                    frontier_edge_ids=chunk_edges,
                ).to(device=device, dtype=child_value.dtype)
                child_terms.append(child_value + log_pb)
                child_rows.append(chunk_rows)
        if not child_terms:
            return log_reward
        terms = torch.cat(child_terms, dim=0)
        term_rows = torch.cat(child_rows, dim=0)
        continuation = segment_logsumexp_or_neg_inf(
            values=terms,
            segment_ids=term_rows,
            num_graphs=state.num_rollouts,
            device=device,
            dtype=terms.dtype,
        )
        target = torch.logaddexp(log_reward, continuation)
        return torch.where(budget.gt(0), target, log_reward)

    def _learned_action_terms(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        frontier_batch_ids: torch.Tensor,
        frontier_edge_ids: torch.Tensor,
        context: Any,
        frontier_context: FrontierContext,
        depth: torch.Tensor,
        remaining_budget: torch.Tensor,
        num_graphs: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.edge_scorer == "relation_residual":
            edge_logits = self._relation_residual_edge_logits(
                batch=batch,
                fb=rollout_context.fb,
                context=context,
                frontier_edge_ids=frontier_edge_ids,
                frontier_batch_ids=frontier_batch_ids,
                frontier_context=frontier_context,
                depth=depth,
                remaining_budget=remaining_budget,
                num_graphs=num_graphs,
                dtype=dtype,
                device=device,
            )
        elif self.edge_scorer == "pointer":
            edge_logits = self._pointer_edge_logits(
                batch=batch,
                state=state,
                fb=rollout_context.fb,
                context=context,
                frontier_edge_ids=frontier_edge_ids,
                frontier_batch_ids=frontier_batch_ids,
                frontier_context=frontier_context,
                dtype=dtype,
                device=device,
            )
        else:
            raise RuntimeError(f"Unknown edge_scorer {self.edge_scorer!r}.")
        # REMOVED: reward/backup/semantic-derived action logits — see methodology.md §3.6
        if self.edge_scorer == "pointer" and edge_logits.numel() > 0:
            edge_logits = edge_logits + self.continuation_logit_bias.to(
                device=edge_logits.device,
                dtype=edge_logits.dtype,
            )
            edge_logits = _reduce_frontier_size_bias(
                continuation_logits=edge_logits,
                frontier_batch_ids=frontier_batch_ids,
                num_graphs=int(num_graphs),
                mode=self.continuation_mass_reduction,
            )
        terminal_energy = self.terminal_head(
            state_h=context.state_h.to(device=device, dtype=dtype)
        )
        child_value = self._exact1_child_terminal_values(
            batch=batch,
            state=state,
            rollout_context=rollout_context,
            frontier_edge_ids=frontier_edge_ids,
            frontier_batch_ids=frontier_batch_ids,
            remaining_budget=remaining_budget,
            dtype=dtype,
            device=device,
        )
        expand_logits = edge_logits + child_value
        log_c = segment_logsumexp_or_neg_inf(
            values=expand_logits,
            segment_ids=frontier_batch_ids,
            num_graphs=int(num_graphs),
            device=expand_logits.device,
            dtype=expand_logits.dtype,
        )
        can_continue = remaining_budget.to(device=device, dtype=torch.long).gt(0)
        log_c = torch.where(
            can_continue,
            log_c,
            torch.full_like(log_c, -torch.inf),
        )
        log_z = torch.logaddexp(terminal_energy, log_c)
        log_z = torch.where(can_continue, log_z, terminal_energy)
        return expand_logits, log_c, log_z, terminal_energy

    def _exact1_child_terminal_values(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if frontier_edge_ids.numel() == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        row_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
        edge_ids = frontier_edge_ids.to(device=device, dtype=torch.long).view(-1)
        child_values = torch.empty(edge_ids.shape, dtype=dtype, device=device)
        for chunk_pos in torch.arange(
            int(edge_ids.numel()),
            dtype=torch.long,
            device=device,
        ).split(self.bdb_child_chunk_size):
            chunk_edges = edge_ids.index_select(0, chunk_pos)
            chunk_rows = row_ids.index_select(0, chunk_pos)
            child_state = _successor_state_for_frontier(
                batch=batch,
                state=state,
                frontier_edge_ids=chunk_edges,
                frontier_batch_ids=chunk_rows,
            )
            child_context = self.state_encoder(
                fb=rollout_context.fb,
                batch=batch,
                state=child_state,
            )
            child_terminal = self.terminal_head(
                state_h=child_context.state_h.to(device=device, dtype=dtype),
            ).to(device=device, dtype=dtype)
            child_values.index_copy_(0, chunk_pos, child_terminal)
        parent_budget = remaining_budget.to(device=device, dtype=torch.long).index_select(
            0,
            row_ids,
        )
        return torch.where(
            parent_budget.gt(0),
            child_values,
            torch.full_like(child_values, -torch.inf),
        )

    def _relation_residual_edge_logits(
        self,
        *,
        batch: RetrievalBatch,
        fb: FeatureBank,
        context: Any,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        frontier_context: FrontierContext,
        depth: torch.Tensor,
        remaining_budget: torch.Tensor,
        num_graphs: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if frontier_edge_ids.numel() == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        frontier_size = torch.bincount(
            frontier_batch_ids.to(device=device, dtype=torch.long),
            minlength=int(num_graphs),
        ).to(device=device, dtype=dtype)
        logits = self.relation_residual_edge_scorer(
            fb=fb,
            batch=batch,
            context=context,
            frontier=frontier_context,
            frontier_batch_ids=frontier_batch_ids,
            depth=depth,
            remaining_budget=remaining_budget,
            frontier_size=frontier_size,
        )
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("RelationResidualEdgeScorer must return logits tensor.")
        return logits.to(device=device, dtype=dtype)

    def _edge_policy_diagnostics(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        fb: FeatureBank,
        context: Any,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        frontier_context: FrontierContext,
        depth: torch.Tensor,
        remaining_budget: torch.Tensor,
        num_graphs: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> FrontierPointerDiagnostics | RelationResidualEdgeDiagnostics | None:
        if self.edge_scorer == "relation_residual":
            frontier_size = torch.bincount(
                frontier_batch_ids.to(device=device, dtype=torch.long),
                minlength=int(num_graphs),
            ).to(device=device, dtype=dtype)
            output = self.relation_residual_edge_scorer(
                fb=fb,
                batch=batch,
                context=context,
                frontier=frontier_context,
                frontier_batch_ids=frontier_batch_ids,
                depth=depth,
                remaining_budget=remaining_budget,
                frontier_size=frontier_size,
                return_diagnostics=True,
            )
            if not isinstance(output, RelationResidualEdgeDiagnostics):
                raise RuntimeError(
                    "RelationResidualEdgeScorer must return diagnostics when "
                    "return_diagnostics=True."
                )
            return output

        if self.edge_scorer == "pointer":
            evidence_tokens, evidence_mask = build_evidence_tokens(
                fb=fb,
                batch=batch,
                state=state,
                query_h=context.query_h,
            )
            output = self.frontier_pointer(
                fb=fb,
                context=context,
                frontier_edge_ids=frontier_edge_ids,
                frontier_batch_ids=frontier_batch_ids,
                frontier_context=frontier_context,
                evidence_tokens=evidence_tokens,
                evidence_mask=evidence_mask,
                return_diagnostics=True,
            )
            if not isinstance(output, FrontierPointerDiagnostics):
                raise RuntimeError(
                    "FrontierPointerPolicy must return diagnostics when "
                    "return_edge_diagnostics=True."
                )
            return output

        return None

    def _semantic_residual_edge_logits(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        fb: FeatureBank,
        context: Any,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        frontier_context: FrontierContext,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if frontier_edge_ids.numel() == 0:
            return torch.empty((0,), dtype=dtype, device=device)

        edge_ids = frontier_edge_ids.to(device=device, dtype=torch.long).view(-1)
        row_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
        semantic = frontier_semantic_scores(fb=fb, frontier=frontier_context)
        semantic_score = (
            semantic.query_relation_score + semantic.query_new_node_score
        ).to(device=device, dtype=dtype)
        transition_features = self.transition_feature_builder(
            fb=fb,
            context=context,
            batch=batch,
            state=state,
            frontier_edge_ids=edge_ids,
            frontier_batch_ids=row_ids,
            frontier_context=frontier_context,
        ).values.to(device=device, dtype=dtype)
        residual = self.edge_residual_scorer(
            state_h=context.state_h.to(device=device, dtype=dtype),
            edge_h=fb.edge_h.index_select(0, edge_ids).to(device=device, dtype=dtype),
            query_h=context.query_h.to(device=device, dtype=dtype),
            row_ids=row_ids,
            edge_feat=transition_features,
        )
        return semantic_score + residual.to(device=device, dtype=dtype)

    def _pointer_edge_logits(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        fb: FeatureBank,
        context: Any,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        frontier_context: FrontierContext,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if frontier_edge_ids.numel() == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        evidence_tokens, evidence_mask = build_evidence_tokens(
            fb=fb,
            batch=batch,
            state=state,
            query_h=context.query_h,
        )
        output = self.frontier_pointer(
            fb=fb,
            context=context,
            frontier_edge_ids=frontier_edge_ids,
            frontier_batch_ids=frontier_batch_ids,
            frontier_context=frontier_context,
            evidence_tokens=evidence_tokens,
            evidence_mask=evidence_mask,
            return_diagnostics=False,
        )
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("FrontierPointerPolicy must return logits tensor.")
        return output.to(device=device, dtype=dtype)

    def _successor_edge_log_mass(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        context: Any | None = None,
        frontier_context: FrontierContext | None = None,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return self._successor_value_edge_log_mass(
            batch=batch,
            state=state,
            rollout_context=rollout_context,
            context=context,
            frontier_context=frontier_context,
            frontier_edge_ids=frontier_edge_ids,
            frontier_batch_ids=frontier_batch_ids,
            dtype=dtype,
            device=device,
        )

    def _successor_value_edge_log_mass(
        self,
        *,
        batch: RetrievalBatch,
        state: State | RolloutState,
        rollout_context: PolicyContext,
        context: Any | None,
        frontier_context: FrontierContext | None,
        frontier_edge_ids: torch.Tensor,
        frontier_batch_ids: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if frontier_edge_ids.numel() == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        edge_ids = frontier_edge_ids.to(device=device, dtype=torch.long).view(-1)
        row_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
        if context is None:
            context = self.state_encoder(
                fb=rollout_context.fb,
                batch=batch,
                state=state,
            )
        successor = _successor_state_for_frontier(
            batch=batch,
            state=state,
            frontier_edge_ids=frontier_edge_ids,
            frontier_batch_ids=frontier_batch_ids,
        )
        successor_context = self.state_encoder(
            fb=rollout_context.fb,
            batch=batch,
            state=successor,
        )
        successor_h = successor_context.state_h.to(device=device, dtype=dtype)
        successor_value = self.successor_value_head(successor_h=successor_h)
        transition_features = self.transition_feature_builder(
            fb=rollout_context.fb,
            context=context,
            batch=batch,
            state=state,
            frontier_edge_ids=edge_ids,
            frontier_batch_ids=row_ids,
            frontier_context=frontier_context,
        ).values.to(device=device, dtype=dtype)
        edge_advantage = self.successor_edge_advantage_scorer(
            state_h=context.state_h.to(device=device, dtype=dtype),
            edge_h=rollout_context.fb.edge_h.index_select(0, edge_ids).to(
                device=device,
                dtype=dtype,
            ),
            successor_h=successor_h,
            query_h=context.query_h.to(device=device, dtype=dtype),
            row_ids=row_ids,
            edge_feat=transition_features,
        )
        log_pb = _uniform_log_pb_for_successor_edges(
            batch=batch,
            successor=successor,
            frontier_edge_ids=frontier_edge_ids,
        )
        return successor_value + edge_advantage + log_pb.to(
            device=successor_value.device,
            dtype=successor_value.dtype,
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

        if fb.edge_h.ndim != 2:
            raise ValueError(
                f"edge_h must have shape [num_edges, H], got {tuple(fb.edge_h.shape)}."
            )
        if fb.edge_h.size(0) != int(batch.edge_index.size(1)):
            raise ValueError(
                "edge_h first dimension mismatch: expected "
                f"{int(batch.edge_index.size(1))}, got {fb.edge_h.size(0)}."
            )
        if fb.edge_h.size(-1) != self.hidden_dim:
            raise ValueError(
                "edge_h hidden dimension mismatch: expected "
                f"{self.hidden_dim}, got {fb.edge_h.size(-1)}."
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

        if fb.node_text_row_ids is None or fb.entity_text_sem_h is None:
            raise ValueError(
                "FeatureBank must expose raw PLM fields node_text_row_ids and "
                "entity_text_sem_h for relation_residual semantic priors."
            )
        if fb.node_text_row_ids.ndim != 1:
            raise ValueError(
                "node_text_row_ids must have shape [num_nodes], got "
                f"{tuple(fb.node_text_row_ids.shape)}."
            )
        if fb.node_text_row_ids.numel() != int(batch.num_nodes_total):
            raise ValueError(
                "node_text_row_ids length mismatch: expected "
                f"{int(batch.num_nodes_total)}, got {fb.node_text_row_ids.numel()}."
            )
        if fb.entity_text_sem_h.ndim != 2:
            raise ValueError(
                "entity_text_sem_h must have shape [num_text_entities, D], got "
                f"{tuple(fb.entity_text_sem_h.shape)}."
            )

    def _coerce_policy_context(
        self,
        batch: RetrievalBatch,
        rollout_context: PolicyContext | FeatureBank | None,
    ) -> PolicyContext:
        if rollout_context is None:
            return self.prepare_rollout_context(batch)
        if isinstance(rollout_context, PolicyContext):
            return rollout_context
        if isinstance(rollout_context, FeatureBank):
            return PolicyContext(fb=rollout_context)
        raise TypeError(
            "rollout_context must be a PolicyContext, FeatureBank, or None, "
            f"got {type(rollout_context).__name__}."
        )

    def _frontier_mode(self) -> str:
        return "boundary"


def budgeted_energy_policy_log_probs(
    *,
    terminal_energy: torch.Tensor,
    expand_logits: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    remaining_budget: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize terminal energy against budgeted expand energies.

    For b > 0:
        P(stop | s,b) = exp(T(s) - V_b(s))
        P(edge e | s,b) = exp(expand_logit(e) - V_b(s))

    For b == 0 or empty frontier, Stop is forced.
    """
    num_graphs = int(num_graphs)
    terminal_energy = terminal_energy.to(dtype=torch.float32).view(num_graphs)
    expand_logits = expand_logits.to(
        device=terminal_energy.device,
        dtype=torch.float32,
    ).view(-1)
    frontier_batch_ids = frontier_batch_ids.to(
        device=terminal_energy.device,
        dtype=torch.long,
    ).view(-1)
    remaining_budget = remaining_budget.to(
        device=terminal_energy.device,
        dtype=torch.long,
    ).view(num_graphs)
    if expand_logits.numel() != frontier_batch_ids.numel():
        raise ValueError(
            "expand_logits and frontier_batch_ids must have matching length: "
            f"{expand_logits.numel()} != {frontier_batch_ids.numel()}."
        )
    _validate_hazard_logits(stop_logits=terminal_energy, edge_logits=expand_logits)

    continue_energy = segment_logsumexp(
        values=expand_logits,
        segment_ids=frontier_batch_ids,
        num_segments=num_graphs,
    )
    can_expand = remaining_budget.gt(0) & torch.isfinite(continue_energy)
    value_energy = torch.logaddexp(terminal_energy, continue_energy)
    log_p_stop = torch.where(
        can_expand,
        terminal_energy - value_energy,
        torch.zeros_like(terminal_energy),
    )
    log_p_continue = torch.where(
        can_expand,
        continue_energy - value_energy,
        torch.full_like(terminal_energy, -torch.inf),
    )
    edge_expand_logprob = expand_logits - value_energy.index_select(
        0,
        frontier_batch_ids,
    )
    edge_allowed = can_expand.index_select(0, frontier_batch_ids)
    edge_expand_logprob = torch.where(
        edge_allowed,
        edge_expand_logprob,
        torch.full_like(edge_expand_logprob, -torch.inf),
    )
    edge_cond_logprob = subtract_log_normalizer(
        values=expand_logits,
        log_normalizer=continue_energy.index_select(0, frontier_batch_ids),
    )
    edge_cond_logprob = torch.where(
        edge_allowed,
        edge_cond_logprob,
        torch.full_like(edge_cond_logprob, -torch.inf),
    )
    finite_expand_logits = torch.where(
        torch.isfinite(expand_logits),
        expand_logits,
        torch.zeros_like(expand_logits),
    )
    log_p_stop = log_p_stop + finite_expand_logits.sum() * 0.0
    prob_sum = _hazard_probability_sum(
        log_p_stop=log_p_stop,
        edge_expand_logprob=edge_expand_logprob,
        frontier_batch_ids=frontier_batch_ids,
    )
    prob_tolerance = _normalization_probability_tolerance(
        prob_sum=prob_sum,
        frontier_batch_ids=frontier_batch_ids,
    )
    if bool(_normalization_bad_mask(prob_sum=prob_sum, tolerance=prob_tolerance).any()):
        raise RuntimeError(
            _hazard_normalization_error(
                prob_sum=prob_sum,
                prob_tolerance=prob_tolerance,
                stop_logits=terminal_energy,
                edge_logits=expand_logits,
                edge_log_z=continue_energy,
                action_log_z=value_energy,
                frontier_batch_ids=frontier_batch_ids,
            )
        )
    return log_p_stop, log_p_continue, edge_cond_logprob, edge_expand_logprob


def hazard_policy_log_probs(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return factorized log-probs:

        P(Stop | s) = sigmoid(-continue_logit)
        P(Add(e) | s) = sigmoid(continue_logit) * softmax(edge_logits)_e
    """
    num_graphs = int(num_graphs)
    stop_logits = stop_logits.to(dtype=torch.float32).view(num_graphs)
    edge_logits = edge_logits.to(device=stop_logits.device, dtype=torch.float32).view(-1)
    frontier_batch_ids = frontier_batch_ids.to(
        device=stop_logits.device,
        dtype=torch.long,
    ).view(-1)

    if edge_logits.numel() != frontier_batch_ids.numel():
        raise ValueError(
            "edge_logits and frontier_batch_ids must have matching length: "
            f"{edge_logits.numel()} != {frontier_batch_ids.numel()}."
        )
    _validate_hazard_logits(stop_logits=stop_logits, edge_logits=edge_logits)

    edge_log_z = segment_logsumexp(
        values=edge_logits,
        segment_ids=frontier_batch_ids,
        num_segments=num_graphs,
    )
    can_expand = torch.isfinite(edge_log_z)
    raw_log_p_stop = F.logsigmoid(-stop_logits)
    raw_log_p_continue = F.logsigmoid(stop_logits)
    log_p_stop = torch.where(
        can_expand,
        raw_log_p_stop,
        torch.zeros_like(stop_logits),
    )
    log_p_continue = torch.where(
        can_expand,
        raw_log_p_continue,
        torch.full_like(stop_logits, -torch.inf),
    )
    edge_cond_logprob = subtract_log_normalizer(
        values=edge_logits,
        log_normalizer=edge_log_z.index_select(0, frontier_batch_ids),
    )
    edge_expand_logprob = (
        log_p_continue.index_select(0, frontier_batch_ids) + edge_cond_logprob
    )
    finite_edge_logits = torch.where(
        torch.isfinite(edge_logits),
        edge_logits,
        torch.zeros_like(edge_logits),
    )
    log_p_stop = log_p_stop + finite_edge_logits.sum() * 0.0
    prob_sum = _hazard_probability_sum(
        log_p_stop=log_p_stop,
        edge_expand_logprob=edge_expand_logprob,
        frontier_batch_ids=frontier_batch_ids,
    )
    prob_tolerance = _normalization_probability_tolerance(
        prob_sum=prob_sum,
        frontier_batch_ids=frontier_batch_ids,
    )
    if bool(_normalization_bad_mask(prob_sum=prob_sum, tolerance=prob_tolerance).any()):
        raise RuntimeError(
            _hazard_normalization_error(
                prob_sum=prob_sum,
                prob_tolerance=prob_tolerance,
                stop_logits=stop_logits,
                edge_logits=edge_logits,
                edge_log_z=edge_log_z,
                action_log_z=log_p_continue,
                frontier_batch_ids=frontier_batch_ids,
            )
        )

    return log_p_stop, log_p_continue, edge_cond_logprob, edge_expand_logprob


def _normalization_probability_tolerance(
    *,
    prob_sum: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
) -> torch.Tensor:
    frontier_counts = torch.bincount(
        frontier_batch_ids.to(device=prob_sum.device, dtype=torch.long),
        minlength=int(prob_sum.numel()),
    ).to(device=prob_sum.device, dtype=prob_sum.dtype)
    scatter_roundoff = (
        frontier_counts.clamp_min(1.0).sqrt() * torch.finfo(torch.float32).eps * 8.0
    )
    return prob_sum.new_full(prob_sum.shape, 5.0e-5) + scatter_roundoff


def _hazard_probability_sum(
    *,
    log_p_stop: torch.Tensor,
    edge_expand_logprob: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
) -> torch.Tensor:
    check_log_p_stop = log_p_stop.detach().to(dtype=torch.float64)
    check_edge_logprob = edge_expand_logprob.detach().to(dtype=torch.float64)
    row_ids = frontier_batch_ids.to(
        device=check_log_p_stop.device,
        dtype=torch.long,
    )
    prob_sum = check_log_p_stop.exp()
    finite_edge_prob = torch.where(
        torch.isfinite(check_edge_logprob),
        check_edge_logprob.exp(),
        check_edge_logprob.new_zeros(check_edge_logprob.shape),
    )
    return prob_sum.scatter_add(0, row_ids, finite_edge_prob)


def _edge_logit_stats(
    *,
    edge_logits: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    num_graphs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    edge_logits = edge_logits.to(device=device, dtype=dtype).view(-1)
    row_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
    if edge_logits.numel() != row_ids.numel():
        raise ValueError(
            "edge_logits and frontier_batch_ids must have matching length: "
            f"{edge_logits.numel()} != {row_ids.numel()}."
        )
    num_graphs = int(num_graphs)
    frontier_size = torch.bincount(row_ids, minlength=num_graphs).to(
        device=device,
        dtype=dtype,
    )
    if edge_logits.numel() == 0:
        zeros = torch.zeros(num_graphs, device=device, dtype=dtype)
        return {
            "frontier_size": zeros,
            "max_edge_logit": zeros,
            "mean_edge_logit": zeros,
            "log_frontier_size": zeros,
        }

    max_edge_logit = edge_logits.new_full((num_graphs,), -torch.inf).scatter_reduce(
        0,
        row_ids,
        edge_logits,
        reduce="amax",
        include_self=True,
    )
    sum_edge_logit = edge_logits.new_zeros((num_graphs,)).scatter_add(
        0,
        row_ids,
        edge_logits,
    )
    mean_edge_logit = sum_edge_logit / frontier_size.clamp_min(1.0)
    max_edge_logit = torch.where(
        frontier_size.gt(0),
        max_edge_logit,
        torch.zeros_like(max_edge_logit),
    )
    return {
        "frontier_size": frontier_size,
        "max_edge_logit": max_edge_logit,
        "mean_edge_logit": mean_edge_logit,
        "log_frontier_size": frontier_size.clamp_min(1.0).log(),
    }


def _normalization_bad_mask(
    *,
    prob_sum: torch.Tensor,
    tolerance: torch.Tensor,
) -> torch.Tensor:
    return (prob_sum - torch.ones_like(prob_sum)).abs() > tolerance


def _validate_hazard_logits(
    *,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
) -> None:
    if not bool(torch.isfinite(stop_logits).all()):
        raise RuntimeError(
            "stop_logits must be finite before hazard normalization. "
            f"stop={_tensor_finiteness_summary(stop_logits)}"
        )
    bad_edge = torch.isnan(edge_logits) | torch.isposinf(edge_logits)
    if bool(bad_edge.any()):
        raise RuntimeError(
            "edge_logits may be finite or -inf mask sentinels, but not NaN/+inf. "
            f"edge={_tensor_finiteness_summary(edge_logits)}"
        )


def _hazard_normalization_error(
    *,
    prob_sum: torch.Tensor,
    prob_tolerance: torch.Tensor,
    stop_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    edge_log_z: torch.Tensor,
    action_log_z: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
) -> str:
    prob_error = (prob_sum - torch.ones_like(prob_sum)).abs()
    bad = _normalization_bad_mask(prob_sum=prob_sum, tolerance=prob_tolerance)
    bad_ids = bad.nonzero(as_tuple=False).flatten()
    preview = bad_ids[:8].detach().cpu().tolist()
    counts = torch.bincount(
        frontier_batch_ids.to(device=prob_sum.device, dtype=torch.long),
        minlength=int(prob_sum.numel()),
    )
    return (
        "Action probabilities must sum to 1 for every state. "
        f"bad_rows={preview}, bad_count={int(bad.sum().item())}, "
        f"prob_sum_min={float(prob_sum.nan_to_num().min().item()):.6g}, "
        f"prob_sum_max={float(prob_sum.nan_to_num().max().item()):.6g}, "
        f"prob_error_max={float(prob_error.nan_to_num().max().item()):.6g}, "
        f"prob_tolerance_bad={prob_tolerance.index_select(0, bad_ids[:8]).detach().cpu().tolist()}, "
        f"stop={_tensor_finiteness_summary(stop_logits)}, "
        f"edge={_tensor_finiteness_summary(edge_logits)}, "
        f"edge_log_z={_tensor_finiteness_summary(edge_log_z)}, "
        f"action_log_z={_tensor_finiteness_summary(action_log_z)}, "
        f"frontier_count_bad={counts.index_select(0, bad_ids[:8]).detach().cpu().tolist()}"
    )


def _tensor_finiteness_summary(tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    total = int(tensor.numel())
    if total == 0:
        return "total=0"
    finite = torch.isfinite(tensor)
    pos_inf = torch.isposinf(tensor)
    neg_inf = torch.isneginf(tensor)
    nan = torch.isnan(tensor)
    if bool(finite.any()):
        finite_values = tensor[finite].to(dtype=torch.float32)
        finite_range = (
            f", finite_min={float(finite_values.min().item()):.6g}, "
            f"finite_max={float(finite_values.max().item()):.6g}"
        )
    else:
        finite_range = ""
    return (
        f"total={total}, finite={int(finite.sum().item())}, "
        f"nan={int(nan.sum().item())}, +inf={int(pos_inf.sum().item())}, "
        f"-inf={int(neg_inf.sum().item())}{finite_range}"
    )


def segment_logsumexp_or_neg_inf(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_graphs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if values.numel() == 0:
        return torch.full(
            (int(num_graphs),),
            -torch.inf,
            dtype=dtype,
            device=device,
        )
    return segment_logsumexp(
        values=values.to(device=device, dtype=dtype),
        segment_ids=segment_ids.to(device=device, dtype=torch.long),
        num_segments=int(num_graphs),
    )


def _reduce_frontier_size_bias(
    *,
    continuation_logits: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    num_graphs: int,
    mode: str,
) -> torch.Tensor:
    if mode == "logsumexp":
        return continuation_logits
    if mode != "logmeanexp":
        raise RuntimeError(f"Unknown continuation mass reduction {mode!r}.")
    counts = torch.bincount(
        frontier_batch_ids.to(device=continuation_logits.device, dtype=torch.long),
        minlength=int(num_graphs),
    ).to(device=continuation_logits.device, dtype=continuation_logits.dtype)
    edge_counts = counts.index_select(0, frontier_batch_ids).clamp_min(1.0)
    return continuation_logits - edge_counts.log()


def _successor_state_for_frontier(
    *,
    batch: RetrievalBatch,
    state: State | RolloutState,
    frontier_edge_ids: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
) -> RolloutState:
    edge_ids = frontier_edge_ids.to(
        device=batch.edge_index.device,
        dtype=torch.long,
    ).view(-1)
    row_ids = frontier_batch_ids.to(
        device=batch.edge_index.device,
        dtype=torch.long,
    ).view(-1)
    if edge_ids.shape != row_ids.shape:
        raise ValueError(
            "frontier_edge_ids and frontier_batch_ids must have matching shape: "
            f"{tuple(edge_ids.shape)} != {tuple(row_ids.shape)}."
        )
    if edge_ids.numel() == 0:
        raise ValueError("Cannot build successor state for an empty frontier.")

    if isinstance(state, RolloutState):
        parent = state.select_rollouts(row_ids)
        rollout_to_graph = parent.rollout_to_graph
        expand_budget = int(state.expand_budget)
    else:
        node_batch = batch.batch.to(device=edge_ids.device, dtype=torch.long)
        edge_batch = batch.edge_batch.to(device=edge_ids.device, dtype=torch.long)
        rollout_to_graph = edge_batch.index_select(0, edge_ids)
        node_belongs = node_batch.view(1, -1).eq(rollout_to_graph.view(-1, 1))
        edge_belongs = edge_batch.view(1, -1).eq(rollout_to_graph.view(-1, 1))
        parent = RolloutState(
            rollout_to_graph=rollout_to_graph,
            expand_budget=int(state.expand_budget),
            edge_index=batch.edge_index.to(device=edge_ids.device, dtype=torch.long),
            num_nodes=int(batch.num_nodes_total),
            num_edges=int(batch.edge_index.size(1)),
            active_nodes=state.active_nodes.to(device=edge_ids.device, dtype=torch.bool)
            .unsqueeze(0)
            .expand(int(edge_ids.numel()), -1)
            & node_belongs,
            active_edges=state.active_edges.to(device=edge_ids.device, dtype=torch.bool)
            .unsqueeze(0)
            .expand(int(edge_ids.numel()), -1)
            & edge_belongs,
            root_edges=state.root_edges.to(device=edge_ids.device, dtype=torch.bool)
            .unsqueeze(0)
            .expand(int(edge_ids.numel()), -1)
            & edge_belongs,
            boundary_nodes=(
                state.boundary_nodes.to(device=edge_ids.device, dtype=torch.bool)
                .unsqueeze(0)
                .expand(int(edge_ids.numel()), -1)
                & node_belongs
                if state.boundary_nodes is not None
                else None
            ),
        )
        expand_budget = int(state.expand_budget)

    successor = parent.snapshot()
    successor.expand_budget = expand_budget
    _ensure_expanded_trace_capacity(successor, capacity=expand_budget)
    successor.apply_expansion(
        rollout_ids=torch.arange(
            int(edge_ids.numel()),
            dtype=torch.long,
            device=edge_ids.device,
        ),
        chosen_edges=edge_ids,
        edge_index=batch.edge_index,
        validate=True,
    )
    return successor


def _te_bfm_terminal_edge_terms(
    *,
    batch: RetrievalBatch,
    state: RolloutState,
    frontier_edge_ids: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    reward_model: Any,
    dtype: torch.dtype,
    device: torch.device,
) -> _TerminalEdgeTerms:
    edge_ids = frontier_edge_ids.to(device=state.device, dtype=torch.long).view(-1)
    row_ids = frontier_batch_ids.to(device=state.device, dtype=torch.long).view(-1)
    if edge_ids.shape != row_ids.shape:
        raise ValueError(
            "frontier_edge_ids and frontier_batch_ids must have matching shape: "
            f"{tuple(edge_ids.shape)} != {tuple(row_ids.shape)}."
        )
    if edge_ids.numel() == 0:
        empty = torch.empty((0,), dtype=dtype, device=device)
        return _TerminalEdgeTerms(
            log_reward=empty,
            log_pb=empty,
            supported=empty,
        )

    context = build_reward_batch_context(
        retrieval_batch=batch,
        device=state.device,
        dtype=torch.float32,
        debug_checks=bool(getattr(reward_model, "debug_checks", False)),
    )
    child_expanded_trace, child_expanded_lengths = (
        state.expanded_edge_trace_for_rollouts_tensor(
            row_ids,
            selected_edge_ids=edge_ids,
        )
    )
    anchor_node_trace, anchor_node_lengths = state.anchor_node_trace_for_rollouts_tensor(
        row_ids,
    )
    active_node_trace, active_node_valid = sparse_rollout_active_node_trace(
        anchor_node_trace=anchor_node_trace,
        anchor_node_lengths=anchor_node_lengths,
        expanded_edge_trace=child_expanded_trace,
        expanded_edge_lengths=child_expanded_lengths,
        edge_index=context.edge_index,
    )
    root_trace, root_lengths = state.root_edge_trace_for_rollouts_tensor(row_ids)
    active_edge_trace = _concat_trace_rows(
        left_trace=root_trace,
        left_lengths=root_lengths,
        right_trace=child_expanded_trace,
        right_lengths=child_expanded_lengths,
    )
    active_edge_lengths = root_lengths + child_expanded_lengths
    row_to_graph = state.rollout_to_graph.to(
        device=state.device,
        dtype=torch.long,
    ).index_select(0, row_ids)
    support = _evidence_support_stats_for_edge_traces(
        active_node_trace=active_node_trace,
        active_node_valid=active_node_valid,
        active_edge_trace=active_edge_trace,
        active_edge_lengths=active_edge_lengths,
        row_to_graph=row_to_graph,
        context=context,
    )
    reward_terms = _set_reward_terms(
        supported_answer_count=support.supported_answer_count,
        path_utility=support.supported_answer_count.new_zeros(
            support.supported_answer_count.shape
        ),
        expanded_edge_count=child_expanded_lengths.to(
            device=context.edge_index.device,
            dtype=context.dtype,
        ),
        answer_credit=float(getattr(reward_model, "answer_credit")),
        edge_cost=float(getattr(reward_model, "edge_cost")),
        fail_penalty=float(getattr(reward_model, "fail_penalty")),
    )
    local_rows = torch.arange(
        int(edge_ids.numel()),
        dtype=torch.long,
        device=state.device,
    )
    log_pb = uniform_log_pb_for_selected_nonroot_edge_trace_tensors(
        active_nonroot_edge_trace=child_expanded_trace,
        active_nonroot_edge_lengths=child_expanded_lengths,
        anchor_node_trace=anchor_node_trace,
        anchor_node_lengths=anchor_node_lengths,
        edge_index=batch.edge_index,
        edge_batch=batch.edge_batch,
        row_ids=local_rows,
        selected_edge_ids=edge_ids,
        row_to_graph=row_to_graph,
        validate=False,
        append_selected_if_missing=True,
    )
    return _TerminalEdgeTerms(
        log_reward=reward_terms.log_reward.to(device=device, dtype=dtype),
        log_pb=log_pb.to(device=device, dtype=dtype),
        supported=support.supported_answer_count.to(device=device, dtype=dtype),
    )


def _terminal_quotient_group_masses(
    *,
    row_ids: torch.Tensor,
    log_rewards: torch.Tensor,
    log_pbs: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_ids = row_ids.to(device=device, dtype=torch.long).view(-1)
    log_rewards = log_rewards.to(device=device, dtype=dtype).view(-1)
    log_pbs = log_pbs.to(device=device, dtype=dtype).view(-1)
    if row_ids.numel() == 0:
        empty_mass = torch.empty((0,), dtype=dtype, device=device)
        empty_rows = torch.empty((0,), dtype=torch.long, device=device)
        return empty_mass, empty_rows
    if log_rewards.shape != row_ids.shape or log_pbs.shape != row_ids.shape:
        raise ValueError(
            "row_ids, log_rewards, and log_pbs must have matching shape for "
            "terminal quotient grouping."
        )

    groups: dict[tuple[int, float, float], int] = {}
    rewards_cpu = log_rewards.detach().cpu().tolist()
    pbs_cpu = log_pbs.detach().cpu().tolist()
    rows_cpu = row_ids.detach().cpu().tolist()
    for row_id, log_reward, log_pb in zip(rows_cpu, rewards_cpu, pbs_cpu):
        key = (int(row_id), float(log_reward), float(log_pb))
        groups[key] = groups.get(key, 0) + 1

    group_rows: list[int] = []
    group_masses: list[float] = []
    for (row_id, log_reward, log_pb), count in groups.items():
        group_rows.append(row_id)
        group_masses.append(math.log(float(count)) + log_reward + log_pb)

    return (
        torch.tensor(group_masses, dtype=dtype, device=device),
        torch.tensor(group_rows, dtype=torch.long, device=device),
    )


def _concat_trace_rows(
    *,
    left_trace: torch.Tensor,
    left_lengths: torch.Tensor,
    right_trace: torch.Tensor,
    right_lengths: torch.Tensor,
) -> torch.Tensor:
    width = int(left_trace.size(1) + right_trace.size(1))
    out = torch.full(
        (int(left_trace.size(0)), width),
        -1,
        dtype=torch.long,
        device=left_trace.device,
    )
    if left_trace.size(1) > 0:
        out[:, : left_trace.size(1)] = left_trace
    if right_trace.size(1) > 0:
        right_cols = torch.arange(
            right_trace.size(1),
            dtype=torch.long,
            device=left_trace.device,
        ).view(1, -1)
        target_cols = left_lengths.to(device=left_trace.device, dtype=torch.long).view(
            -1,
            1,
        ) + right_cols
        out.scatter_(1, target_cols, right_trace.to(device=left_trace.device))
    return out


def _te_bfm_recursive_expansion_diagnostics(
    *,
    batch: RetrievalBatch,
    state: RolloutState,
    row_ids: torch.Tensor,
    expanded_state_count: int,
    max_expanded_states: int,
    remaining_budget: torch.Tensor,
    lookahead_depth: int,
) -> dict[str, Any]:
    row_ids_cpu = row_ids.detach().cpu().to(dtype=torch.long).view(-1)
    counts_by_row = torch.bincount(row_ids_cpu, minlength=state.num_rollouts)
    active_rows = counts_by_row.nonzero(as_tuple=False).flatten()
    if active_rows.numel() == 0:
        child_counts = torch.empty(0, dtype=torch.long)
    else:
        child_counts = counts_by_row.index_select(0, active_rows)

    rollout_to_graph = state.rollout_to_graph.detach().cpu().to(dtype=torch.long)
    active_graphs = (
        rollout_to_graph.index_select(0, active_rows)
        if active_rows.numel() > 0
        else torch.empty(0, dtype=torch.long)
    )
    graph_contrib: dict[int, int] = {}
    for graph_id, count in zip(active_graphs.tolist(), child_counts.tolist()):
        graph_contrib[int(graph_id)] = graph_contrib.get(int(graph_id), 0) + int(count)

    top_graph_ids = sorted(
        graph_contrib,
        key=lambda graph_id: graph_contrib[graph_id],
        reverse=True,
    )[:5]

    budget_cpu = remaining_budget.detach().cpu().to(dtype=torch.long).view(-1)
    graph_records = [
        _te_bfm_graph_diagnostics(
            batch=batch,
            state=state,
            graph_id=graph_id,
            active_rows=active_rows,
            active_graphs=active_graphs,
            counts_by_row=counts_by_row,
            remaining_budget=budget_cpu,
            frontier_count=graph_contrib[graph_id],
        )
        for graph_id in top_graph_ids
    ]

    return {
        "expanded_state_count": int(expanded_state_count),
        "max_expanded_states": int(max_expanded_states),
        "lookahead_depth": int(lookahead_depth),
        "remaining_budget_min": _safe_int(budget_cpu.min()) if budget_cpu.numel() else None,
        "remaining_budget_max": _safe_int(budget_cpu.max()) if budget_cpu.numel() else None,
        "current_state_frontier_count": int(expanded_state_count),
        "frontier_count": int(expanded_state_count),
        "child_frontier_count_sum": int(expanded_state_count),
        "child_frontier_count_mean": _safe_mean(child_counts),
        "child_frontier_count_max": _safe_int(child_counts.max())
        if child_counts.numel()
        else 0,
        "num_rollout_rows": int(state.num_rollouts),
        "num_frontier_rows": int(active_rows.numel()),
        "top_graphs": graph_records,
    }


def _te_bfm_graph_diagnostics(
    *,
    batch: RetrievalBatch,
    state: RolloutState,
    graph_id: int,
    active_rows: torch.Tensor,
    active_graphs: torch.Tensor,
    counts_by_row: torch.Tensor,
    remaining_budget: torch.Tensor,
    frontier_count: int,
) -> dict[str, Any]:
    graph_id = int(graph_id)
    ptr = getattr(batch, "ptr", None)
    edge_ptr = getattr(batch, "edge_ptr", None)
    node_lo = _ptr_value(ptr, graph_id, default=0)
    node_hi = _ptr_value(ptr, graph_id + 1, default=int(getattr(batch, "num_nodes_total", 0)))
    edge_lo = _ptr_value(edge_ptr, graph_id, default=0)
    edge_hi = _ptr_value(
        edge_ptr,
        graph_id + 1,
        default=int(getattr(batch, "num_edges_total", 0)),
    )

    row_mask = active_graphs.eq(graph_id) if active_graphs.numel() else torch.empty(0, dtype=torch.bool)
    rows = active_rows[row_mask] if row_mask.numel() else torch.empty(0, dtype=torch.long)
    row_counts = counts_by_row.index_select(0, rows) if rows.numel() else torch.empty(0, dtype=torch.long)
    row_budgets = remaining_budget.index_select(0, rows) if rows.numel() else torch.empty(0, dtype=torch.long)

    sample_id = _batch_graph_field(batch, graph_id, "sample_id", "question_id")
    split = _batch_graph_field(batch, graph_id, "split")
    if split is None and sample_id is not None:
        parts = str(sample_id).split("/")
        if len(parts) >= 3:
            split = parts[1]

    return {
        "split": split,
        "sample_id": sample_id,
        "question_id": _batch_graph_field(batch, graph_id, "question_id") or sample_id,
        "batch_index": _batch_graph_field(batch, graph_id, "batch_index"),
        "graph_index": graph_id,
        "graph_index_in_batch": graph_id,
        "num_nodes": max(0, node_hi - node_lo),
        "num_edges": max(0, edge_hi - edge_lo),
        "num_anchors": _node_id_count_for_graph(batch, "anchor_node_ids", graph_id),
        "num_targets": _node_id_count_for_graph(batch, "target_node_ids", graph_id),
        "remaining_budget_b_min": _safe_int(row_budgets.min()) if row_budgets.numel() else None,
        "remaining_budget_b_max": _safe_int(row_budgets.max()) if row_budgets.numel() else None,
        "frontier_count": int(frontier_count),
        "child_frontier_count_sum": int(frontier_count),
        "child_frontier_count_mean": _safe_mean(row_counts),
        "child_frontier_count_max": _safe_int(row_counts.max()) if row_counts.numel() else 0,
        "rollout_row_count": int(rows.numel()),
    }


def _batch_graph_field(
    batch: RetrievalBatch,
    graph_id: int,
    *names: str,
) -> Any | None:
    for name in names:
        if not hasattr(batch, name):
            continue
        value = getattr(batch, name)
        item = _graph_field_item(value, int(graph_id))
        if item is not None:
            return item
    return None


def _graph_field_item(value: Any, graph_id: int) -> Any | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[graph_id] if 0 <= graph_id < len(value) else None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _tensor_scalar(value)
        if 0 <= graph_id < int(value.size(0)):
            return _tensor_scalar(value[graph_id])
        return None
    if isinstance(value, str):
        return value if graph_id == 0 else None
    return value if graph_id == 0 else None


def _tensor_scalar(value: torch.Tensor) -> Any:
    if value.numel() != 1:
        return value.detach().cpu().tolist()
    item = value.detach().cpu().item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def _node_id_count_for_graph(
    batch: RetrievalBatch,
    field_name: str,
    graph_id: int,
) -> int:
    if not hasattr(batch, field_name) or not hasattr(batch, "batch"):
        return 0
    ids = getattr(batch, field_name)
    if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
        return 0
    node_batch = batch.batch.detach().cpu().to(dtype=torch.long)
    ids_cpu = ids.detach().cpu().to(dtype=torch.long).view(-1)
    valid = ids_cpu.ge(0) & ids_cpu.lt(int(node_batch.numel()))
    if not bool(valid.any()):
        return 0
    graph_ids = node_batch.index_select(0, ids_cpu[valid])
    return int(graph_ids.eq(int(graph_id)).sum().item())


def _ptr_value(ptr: Any, index: int, *, default: int) -> int:
    if not isinstance(ptr, torch.Tensor):
        return int(default)
    ptr_cpu = ptr.detach().cpu().to(dtype=torch.long).view(-1)
    if index < 0 or index >= int(ptr_cpu.numel()):
        return int(default)
    return int(ptr_cpu[index].item())


def _safe_int(value: torch.Tensor) -> int:
    return int(value.item())


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.to(dtype=torch.float32).mean().item())


def _ensure_expanded_trace_capacity(
    state: RolloutState,
    *,
    capacity: int,
) -> None:
    if state.expanded_edge_trace is None:
        return
    capacity = int(capacity)
    current_width = int(state.expanded_edge_trace.size(1))
    if current_width >= capacity:
        return
    padded = torch.full(
        (int(state.expanded_edge_trace.size(0)), capacity),
        -1,
        dtype=state.expanded_edge_trace.dtype,
        device=state.expanded_edge_trace.device,
    )
    if current_width > 0:
        padded[:, :current_width] = state.expanded_edge_trace
    state.expanded_edge_trace = padded


def _uniform_log_pb_for_successor_edges(
    *,
    batch: RetrievalBatch,
    successor: RolloutState,
    frontier_edge_ids: torch.Tensor,
) -> torch.Tensor:
    edge_ids = frontier_edge_ids.to(
        device=batch.edge_index.device,
        dtype=torch.long,
    ).view(-1)
    row_ids = torch.arange(
        int(edge_ids.numel()),
        dtype=torch.long,
        device=edge_ids.device,
    )
    expanded_edge_trace, expanded_edge_lengths = (
        successor.expanded_edge_trace_for_rollouts_tensor(row_ids)
    )
    anchor_node_trace, anchor_node_lengths = (
        successor.anchor_node_trace_for_rollouts_tensor(row_ids)
    )
    return uniform_log_pb_for_selected_nonroot_edge_trace_tensors(
        active_nonroot_edge_trace=expanded_edge_trace,
        active_nonroot_edge_lengths=expanded_edge_lengths,
        anchor_node_trace=anchor_node_trace,
        anchor_node_lengths=anchor_node_lengths,
        edge_index=batch.edge_index,
        edge_batch=batch.edge_batch,
        row_ids=row_ids,
        selected_edge_ids=edge_ids,
        row_to_graph=successor.rollout_to_graph,
        validate=False,
        append_selected_if_missing=True,
    )


def _uniform_parent_counts_for_successor_edges(
    *,
    batch: RetrievalBatch,
    successor: RolloutState,
    frontier_edge_ids: torch.Tensor,
) -> torch.Tensor:
    edge_ids = frontier_edge_ids.to(
        device=batch.edge_index.device,
        dtype=torch.long,
    ).view(-1)
    row_ids = torch.arange(
        int(edge_ids.numel()),
        dtype=torch.long,
        device=edge_ids.device,
    )
    expanded_edge_trace, expanded_edge_lengths = (
        successor.expanded_edge_trace_for_rollouts_tensor(row_ids)
    )
    anchor_node_trace, anchor_node_lengths = (
        successor.anchor_node_trace_for_rollouts_tensor(row_ids)
    )
    return uniform_backward_parent_counts_for_selected_nonroot_edge_trace_tensors(
        active_nonroot_edge_trace=expanded_edge_trace,
        active_nonroot_edge_lengths=expanded_edge_lengths,
        anchor_node_trace=anchor_node_trace,
        anchor_node_lengths=anchor_node_lengths,
        edge_index=batch.edge_index,
        edge_batch=batch.edge_batch,
        row_ids=row_ids,
        selected_edge_ids=edge_ids,
        row_to_graph=successor.rollout_to_graph,
        validate=False,
        append_selected_if_missing=True,
    )


def _filter_frontier_context(
    frontier: Any,
    mask: torch.Tensor,
    *,
    device: torch.device,
) -> Any:
    mask = mask.to(device=device, dtype=torch.bool).view(-1)
    return type(frontier)(
        **{
            field: value[mask] if isinstance(value, torch.Tensor) else value
            for field, value in (
                (item.name, getattr(frontier, item.name)) for item in fields(frontier)
            )
        }
    )


def _cap_frontier_context_by_semantic_topk(
    *,
    fb: FeatureBank,
    frontier: FrontierContext,
    max_edges_per_state: int,
    device: torch.device,
) -> tuple[FrontierContext, _FrontierCapStats]:
    max_edges = int(max_edges_per_state)
    if max_edges < 1:
        raise ValueError(
            f"max_edges_per_state must be >= 1, got {max_edges_per_state}."
        )
    if frontier.edge_ids.numel() <= max_edges:
        return frontier, _FrontierCapStats(
            used_rate=0.0,
            dropped_edge_count_mean=0.0,
        )

    row_ids = frontier.graph_id.to(device=device, dtype=torch.long).view(-1)
    if row_ids.numel() == 0:
        return frontier, _FrontierCapStats(
            used_rate=0.0,
            dropped_edge_count_mean=0.0,
        )
    semantic = frontier_semantic_scores(fb=fb, frontier=frontier)
    scores = (
        semantic.query_relation_score + semantic.query_new_node_score
    ).to(device=device, dtype=torch.float32)
    keep = torch.zeros(row_ids.shape, dtype=torch.bool, device=device)
    dropped_counts: list[float] = []
    for row_id in torch.unique(row_ids, sorted=True):
        row_mask = row_ids.eq(row_id)
        positions = row_mask.nonzero(as_tuple=False).flatten()
        if int(positions.numel()) <= max_edges:
            keep[positions] = True
            dropped_counts.append(0.0)
            continue
        row_scores = scores.index_select(0, positions)
        _, local_topk = torch.topk(row_scores, k=max_edges, largest=True, sorted=False)
        keep[positions.index_select(0, local_topk)] = True
        dropped_counts.append(float(int(positions.numel()) - max_edges))
    dropped = [count for count in dropped_counts if count > 0.0]
    stats = _FrontierCapStats(
        used_rate=1.0 if dropped else 0.0,
        dropped_edge_count_mean=(
            float(sum(dropped) / max(len(dropped), 1)) if dropped else 0.0
        ),
    )
    return _filter_frontier_context(frontier, keep, device=device), stats


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

    logsumexp = segment_logsumexp(
        values=edge_logits,
        segment_ids=edge_batch,
        num_segments=num_graphs,
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
    "PolicyContext",
    "PolicyOutput",
    "budgeted_energy_policy_log_probs",
    "frontier_logit_summary",
    "hazard_policy_log_probs",
    "segment_logsumexp_or_neg_inf",
]
