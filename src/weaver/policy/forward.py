from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.weaver.context import GraphContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.state import (
    FrontierEncoding,
    NodeSelection,
    StateBatch,
    frontier_from_graph,
)

from .edge_scorer import QuestionConditionedEdgeScorer
from .output import PolicyOutput, STOP_EDGE_ID


@dataclass(frozen=True, slots=True)
class PolicyInput:
    question_h_by_graph: torch.Tensor  # [G, H]
    edge_h: torch.Tensor  # [E, H]
    frontier_prune_score: torch.Tensor | None = None  # [E]
    # align_score removed: query-relation similarity is consumed exclusively
    # by frontier pruning; using it again in edge scoring causes double-biasing
    # and suppresses GFlowNet exploration diversity.


@dataclass(frozen=True, slots=True)
class PolicyActionSpace:
    active: NodeSelection
    frontier: FrontierEncoding


@dataclass(frozen=True, slots=True)
class FrontierPruningConfig:
    enabled: bool = False
    threshold: float = 0.0
    min_keep_per_state: int = 0
    apply_train: bool = False
    apply_eval: bool = False
    apply_scoring: bool = False
    keep_recorded_edges_in_train: bool = True


class StateFlowHead(nn.Module):
    """state_h -> log F_theta(z)."""

    def __init__(self, *, state_dim: int) -> None:
        super().__init__()
        state_dim = int(state_dim)
        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        head_dim = min(state_dim, 256)
        self.net = nn.Sequential(
            nn.Linear(state_dim, head_dim, bias=False),
            nn.LayerNorm(head_dim),
            nn.SiLU(),
            nn.Linear(head_dim, 1, bias=True),
        )

    def forward(self, *, state_h: torch.Tensor) -> torch.Tensor:  # [S]
        return self.net(state_h.float()).squeeze(-1)


class FlowEstimator(nn.Module):
    """
    Flat-energy forward policy head.

    For each state z, the action set is:

        A(z) = {STOP} union C(z)

    The head outputs unnormalized action logits:

        L_STOP(z) = stop_head(h_z)

        L_e(z) = c_theta(z, e)
               - beta * log |C(z)|

    where beta = frontier_size_correction.

    align_score is no longer used here. Query-relation similarity is consumed
    exclusively by frontier pruning upstream to avoid double-biasing.

    Final policy is computed outside this module by PolicyOutput:

        P_F(a | z) = softmax_{a in A(z)} L_a(z)

    CONTINUE is not a sampled action. Its probability mass is:

        P_F(continue | z) = sum_{e in C(z)} P_F(e | z)
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        stop_initial_bias: float = 0.0,
        frontier_size_correction: float = 0.0,
        edge_state_hidden_dim: int | None = None,
        edge_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        frontier_size_correction = float(frontier_size_correction)
        if frontier_size_correction < 0.0:
            raise ValueError("frontier_size_correction must be nonnegative.")

        self.hidden_dim = hidden_dim
        self.frontier_size_correction = frontier_size_correction

        self.edge_scorer = QuestionConditionedEdgeScorer(
            hidden_dim=hidden_dim,
            state_hidden_dim=edge_state_hidden_dim,
            dropout=edge_dropout,
        )

        # align_scale removed along with align_score channel.

        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        nn.init.constant_(self.stop_head[-1].bias, float(stop_initial_bias))

    def score_edges(
        self,
        *,
        state_h: torch.Tensor,  # [F, H]
        frontier_edge_h: torch.Tensor,  # [F, H]
    ) -> torch.Tensor:  # [F]
        return self.edge_scorer.score_state(
            state_h=state_h,
            edge_h=frontier_edge_h,
        ).float()

    def score_stop(self, *, state_h: torch.Tensor) -> torch.Tensor:  # [S] -> [S]
        return self.stop_head(state_h.float()).squeeze(-1)

    def forward(
        self,
        *,
        state_h: torch.Tensor,  # [S, H]
        frontier_row_ids: torch.Tensor,  # [F]
        frontier_edge_h: torch.Tensor,  # [F, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        S = int(state_h.size(0))

        stop_logits = state_h.new_zeros((S,), dtype=torch.float32)

        if int(frontier_row_ids.numel()) == 0:
            return state_h.new_empty((0,), dtype=torch.float32), stop_logits

        _validate_row_ids(frontier_row_ids, upper=S, name="frontier_row_ids")

        edge_logits = self.score_edges(
            state_h=state_h.index_select(0, frontier_row_ids),
            frontier_edge_h=frontier_edge_h,
        )

        frontier_count = torch.bincount(frontier_row_ids, minlength=S)

        if self.frontier_size_correction != 0.0:
            count = (
                frontier_count.index_select(0, frontier_row_ids).clamp_min(1).float()
            )
            edge_logits = edge_logits - self.frontier_size_correction * count.log()

        has_frontier = frontier_count.gt(0)
        if bool(has_frontier.any()):
            stop_logits[has_frontier] = self.score_stop(
                state_h=state_h.index_select(
                    0, has_frontier.nonzero(as_tuple=False).flatten()
                )
            ).float()

        _require_finite(edge_logits, "edge_logits")
        _require_finite(stop_logits, "stop_logits")

        return edge_logits, stop_logits


class ForwardPolicy(nn.Module):
    """
    GFlowNet forward policy for KG evidence-subgraph retrieval.

    Data flow:

      FeaturePack:
        question_h           [G, H]
        edge_h               [E, H]
        frontier_prune_score [E]   <- query-relation similarity, pruning only

      StateEncoder:
        question_h_per_state + selected_edge_h -> state_h [S, H]

      FlowEstimator:
        STOP action:
            L_STOP(z) = MLP_stop(h_z)

        Edge action:
            L_e(z) = c_theta(z, e)
                   - beta * log |C(z)|

      PolicyOutput:
        action_logits  = [STOP logits, edge logits]
        action_row_ids = [state rows, frontier row ids]
        action_edge_ids = [-1, physical edge ids]

    Final softmax is over the unified action set:

        {STOP} union C(z)
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        flow_estimator: FlowEstimator,
        state_flow_head: StateFlowHead,
        frontier_pruning: FrontierPruningConfig | None = None,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.flow_estimator = flow_estimator
        self.state_flow_head = state_flow_head
        self.frontier_pruning = frontier_pruning or FrontierPruningConfig()

    def build_policy_input(
        self,
        features: FeaturePack,
        graph_context: GraphContext | None = None,
        *,
        compute_align_score: bool = False,  # retained for call-site compat, ignored
    ) -> PolicyInput:
        return PolicyInput(
            question_h_by_graph=features.question_h.float(),
            edge_h=features.edge_h.float(),
            frontier_prune_score=(
                features.frontier_prune_score.float()
                if features.frontier_prune_score is not None
                else None
            ),
        )

    def prepare_action_space(
        self,
        *,
        state: StateBatch,
        graph_context: GraphContext,
        policy_input: PolicyInput | None = None,
        training: bool = False,
        scoring: bool = False,
        recorded_edge_ids_by_state: torch.Tensor | None = None,
    ) -> PolicyActionSpace:
        active = state.active_node_index(graph_context)
        frontier = frontier_from_graph(
            state=state,
            graph=graph_context,
            active=active,
        )
        frontier = self._maybe_prune_frontier(
            frontier=frontier,
            state=state,
            policy_input=policy_input,
            training=training,
            scoring=scoring,
            recorded_edge_ids_by_state=recorded_edge_ids_by_state,
        )
        return PolicyActionSpace(active=active, frontier=frontier)

    def _maybe_prune_frontier(
        self,
        *,
        frontier: FrontierEncoding,
        state: StateBatch,
        policy_input: PolicyInput | None,
        training: bool,
        scoring: bool,
        recorded_edge_ids_by_state: torch.Tensor | None,
    ) -> FrontierEncoding:
        cfg = self.frontier_pruning
        if not cfg.enabled:
            return frontier
        if scoring and not cfg.apply_scoring:
            return frontier
        if training and not cfg.apply_train:
            return frontier
        if (not training) and not cfg.apply_eval:
            return frontier
        if int(frontier.edge_ids.numel()) == 0:
            return frontier
        if policy_input is None:
            return frontier

        prune_scores = policy_input.frontier_prune_score
        if prune_scores is None:
            return frontier

        row_ids = frontier.row_ids
        edge_ids = frontier.edge_ids
        relation_scores = prune_scores.index_select(0, edge_ids).float()

        keep_mask = relation_scores.ge(float(cfg.threshold))
        k = int(cfg.min_keep_per_state) if int(cfg.min_keep_per_state) > 0 else 1
        keep_mask = keep_mask | _topk_keep_mask(
            row_ids=row_ids,
            scores=relation_scores,
            num_states=state.num_states,
            k=k,
        )

        if (
            training
            and cfg.keep_recorded_edges_in_train
            and recorded_edge_ids_by_state is not None
        ):
            keep_mask = keep_mask | _recorded_edge_keep_mask(
                frontier=frontier,
                recorded_edge_ids_by_state=recorded_edge_ids_by_state,
            )

        if bool(keep_mask.all()):
            return frontier
        return FrontierEncoding(
            row_ids=row_ids[keep_mask],
            edge_ids=edge_ids[keep_mask],
            graph_ids=frontier.graph_ids[keep_mask],
        )

    def _build_state_h_batched(
        self,
        *,
        S: int,
        question_h_per_state: torch.Tensor,  # [S, H]
        selected_row_ids: torch.Tensor,  # [E_sel]
        selected_edge_ids: torch.Tensor,  # [E_sel]
        edge_h: torch.Tensor,  # [E_total, H]
        device: torch.device,
    ) -> torch.Tensor:  # [S, H]
        H = int(question_h_per_state.shape[-1])

        if int(selected_row_ids.numel()) == 0:
            is_empty = torch.ones(S, dtype=torch.bool, device=device)
            dummy_kv = torch.zeros(S, 1, H, device=device)
            full_mask = torch.ones(S, 1, dtype=torch.bool, device=device)
            state_h = self.state_encoder(
                question_h=question_h_per_state,
                selected_edge_h=dummy_kv,
                key_padding_mask=full_mask,
                is_empty=is_empty,
            )
            _require_finite(state_h, "state_h")
            return state_h

        _validate_row_ids(selected_row_ids, upper=S, name="selected_row_ids")

        order = torch.argsort(selected_row_ids, stable=True)
        selected_row_ids = selected_row_ids.index_select(0, order)
        selected_edge_ids = selected_edge_ids.index_select(0, order)

        edge_counts = torch.bincount(selected_row_ids, minlength=S)
        L_max = int(edge_counts.max().item())
        is_empty = edge_counts.eq(0)

        state_offsets = torch.zeros(S + 1, dtype=torch.long, device=device)
        state_offsets[1:] = edge_counts.cumsum(0)

        local_pos = torch.arange(
            selected_row_ids.numel(), device=device
        ) - state_offsets.index_select(0, selected_row_ids)

        if bool(
            (
                local_pos.lt(0)
                | local_pos.ge(edge_counts.index_select(0, selected_row_ids))
            ).any()
        ):
            raise ValueError("Failed to construct local selected-edge positions.")

        padded = torch.zeros(S, L_max, H, device=device)
        key_padding_mask = torch.ones(S, L_max, dtype=torch.bool, device=device)

        sel_edge_h = edge_h.index_select(0, selected_edge_ids)
        padded[selected_row_ids, local_pos] = sel_edge_h
        key_padding_mask[selected_row_ids, local_pos] = False

        state_h = self.state_encoder(
            question_h=question_h_per_state,
            selected_edge_h=padded,
            key_padding_mask=key_padding_mask,
            is_empty=is_empty,
        )
        _require_finite(state_h, "state_h")
        return state_h

    def forward(
        self,
        *,
        state: StateBatch,
        features: FeaturePack,
        graph_context: GraphContext,
        policy_input: PolicyInput | None = None,
        action_space: PolicyActionSpace | None = None,
        compute_log_flow: bool = False,
    ) -> PolicyOutput:
        if policy_input is None:
            policy_input = self.build_policy_input(
                features,
                graph_context=graph_context,
            )

        if action_space is None:
            action_space = self.prepare_action_space(
                state=state,
                graph_context=graph_context,
                policy_input=policy_input,
                training=self.training,
            )

        frontier = action_space.frontier
        S = int(state.num_states)
        dev = state.device

        _validate_row_ids(frontier.row_ids, upper=S, name="frontier.row_ids")

        question_h_per_state = policy_input.question_h_by_graph.index_select(
            0, state.graph_ids
        )  # [S, H]

        selected = state.selected_edge_index()

        state_h = self._build_state_h_batched(
            S=S,
            question_h_per_state=question_h_per_state,
            selected_row_ids=selected.row_ids,
            selected_edge_ids=selected.edge_ids,
            edge_h=policy_input.edge_h,
            device=dev,
        )

        f_edge_ids = frontier.edge_ids
        frontier_edge_h = policy_input.edge_h.index_select(0, f_edge_ids)

        edge_logits, stop_logits = self.flow_estimator(
            state_h=state_h,
            frontier_row_ids=frontier.row_ids,
            frontier_edge_h=frontier_edge_h,
        )

        frontier_count = torch.bincount(frontier.row_ids, minlength=S)

        empty_with_frontier = state.edge_count.eq(0) & frontier_count.gt(0)
        if bool(empty_with_frontier.any()):
            stop_logits = stop_logits.clone()
            stop_logits[empty_with_frontier] = -1.0e9

        _require_finite(edge_logits, "edge_logits")
        _require_finite(stop_logits, "stop_logits")

        log_flow_base = None
        if compute_log_flow:
            log_flow_base = self.state_flow_head(state_h=state_h).float()
            _require_finite(log_flow_base, "log_flow_base")

        rows = torch.arange(S, dtype=torch.long, device=dev)

        return PolicyOutput(
            action_logits=torch.cat(
                [stop_logits.float(), edge_logits.float()],
                dim=0,
            ),
            action_row_ids=torch.cat(
                [rows, frontier.row_ids],
                dim=0,
            ),
            action_edge_ids=torch.cat(
                [torch.full_like(rows, STOP_EDGE_ID), f_edge_ids],
                dim=0,
            ),
            frontier=frontier,
            log_flow_base=log_flow_base,
            state_h=state_h if compute_log_flow else None,
        )


__all__ = [
    "FrontierPruningConfig",
    "FlowEstimator",
    "ForwardPolicy",
    "PolicyActionSpace",
    "PolicyInput",
    "StateFlowHead",
]


def _topk_keep_mask(
    *,
    row_ids: torch.Tensor,
    scores: torch.Tensor,
    num_states: int,
    k: int,
) -> torch.Tensor:
    keep = torch.zeros_like(row_ids, dtype=torch.bool)
    if k <= 0 or int(row_ids.numel()) == 0:
        return keep
    for row in range(int(num_states)):
        positions = torch.nonzero(row_ids.eq(row), as_tuple=False).flatten()
        if int(positions.numel()) == 0:
            continue
        take = min(int(k), int(positions.numel()))
        top_positions = torch.topk(
            scores.index_select(0, positions), k=take, largest=True, sorted=False
        ).indices
        keep[positions.index_select(0, top_positions)] = True
    return keep


def _recorded_edge_keep_mask(
    *,
    frontier: FrontierEncoding,
    recorded_edge_ids_by_state: torch.Tensor,
) -> torch.Tensor:
    keep = torch.zeros_like(frontier.edge_ids, dtype=torch.bool)
    if (
        int(frontier.edge_ids.numel()) == 0
        or int(recorded_edge_ids_by_state.numel()) == 0
    ):
        return keep
    valid_recorded = recorded_edge_ids_by_state.ge(0)
    for row in torch.unique(frontier.row_ids).tolist():
        frontier_positions = torch.nonzero(
            frontier.row_ids.eq(row), as_tuple=False
        ).flatten()
        recorded = recorded_edge_ids_by_state[row]
        recorded = recorded[valid_recorded[row]]
        if int(frontier_positions.numel()) == 0 or int(recorded.numel()) == 0:
            continue
        frontier_edges = frontier.edge_ids.index_select(0, frontier_positions)
        row_keep = torch.isin(frontier_edges, recorded)
        keep[frontier_positions] = row_keep
    return keep


def _validate_row_ids(
    row_ids: torch.Tensor,
    *,
    upper: int,
    name: str,
) -> None:
    if int(row_ids.numel()) == 0:
        return
    if bool(row_ids.lt(0).any()) or bool(row_ids.ge(int(upper)).any()):
        bad = ((row_ids.lt(0)) | (row_ids.ge(int(upper)))).nonzero(as_tuple=False)[:8]
        raise ValueError(
            f"{name} contains out-of-range rows; sample positions={bad.tolist()}."
        )


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if bool(torch.isfinite(tensor).all()):
        return
    bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
    preview = bad[:8].tolist()
    raise ValueError(f"{name} contains non-finite values at indices {preview}.")
