from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import init

from src.graph.segments import segment_logsumexp
from src.utils.nn_utils import init_xavier
from src.weaver.context import GraphContext
from src.weaver.feature import (
    FeatureBank,
    select_node_embedding,
    select_node_text_mask,
    select_query_embedding,
    select_relation_embedding,
)
from src.weaver.nn.state_encoder import StateEncoder
from src.weaver.state import ActionSpace, StateBatch

from .output import PolicyOutput

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class StateRepresentation:
    query_h: Tensor
    selected_h: Tensor
    covered_h: Tensor

    @property
    def num_states(self) -> int:
        return int(self.query_h.size(0))

    @property
    def hidden_dim(self) -> int:
        return int(self.query_h.size(1))


@dataclass(frozen=True, slots=True)
class FrontierEncoding:
    row_ids: Tensor
    edge_ids: Tensor
    dst_node_ids: Tensor

    edge_h: Tensor
    relation_h: Tensor
    dst_h: Tensor
    dst_text_mask: Tensor

    @property
    def num_actions(self) -> int:
        return int(self.edge_ids.numel())


class StopLogFlowPredictor(nn.Module):
    """
    Parametric reward predictor over current states.

    Semantics:
        stop_log_flow(z) = log R_hat_psi(z)

    The true reward calculator remains parameter-free and outside this module.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_hidden_dim: int,
        initial_log_reward: float = -4.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.initial_log_reward = float(initial_log_reward)

        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.action_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.action_hidden_dim, 1),
        )

        self.reset_parameters()

    def forward(self, state_repr: StateRepresentation) -> Tensor:
        x = torch.cat(
            (
                state_repr.query_h,
                state_repr.selected_h,
                state_repr.covered_h,
            ),
            dim=-1,
        )

        return self.net(x).squeeze(-1).float()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                init_xavier(module)

        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError(f"Expected final nn.Linear, got {type(last).__name__}.")

        init.zeros_(last.weight)
        init.constant_(last.bias, self.initial_log_reward)


class SemanticEdgePrior(nn.Module):
    """
    Fixed semantic prior in the original semantic embedding space.

    No learned projection is used here.
    """

    def __init__(
        self,
        *,
        relation_weight: float = 1.0,
        dst_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.relation_weight = float(relation_weight)
        self.dst_weight = float(dst_weight)

    def forward(
        self,
        *,
        state_repr: StateRepresentation,
        frontier: FrontierEncoding,
    ) -> Tensor:
        if int(frontier.num_actions) == 0:
            return state_repr.query_h.new_empty((0,), dtype=torch.float32)

        query_h = state_repr.query_h.index_select(0, frontier.row_ids).float()

        relation_prior = (query_h * frontier.relation_h.float()).sum(dim=-1)

        dst_text_mask = frontier.dst_text_mask.to(dtype=torch.float32)
        dst_prior = dst_text_mask * (query_h * frontier.dst_h.float()).sum(dim=-1)

        return (float(self.relation_weight) * relation_prior + float(self.dst_weight) * dst_prior).float()


class LowRankEdgeResidualScorer(nn.Module):
    """
    State-conditioned residual scorer.

    residual(z, e) =
        < f_state([q, selected, covered])[z], f_edge(edge_h)[e] >
        / sqrt(action_dim)
        + edge_bias(edge_h)

    This avoids a per-frontier-edge 4H concat MLP.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        use_edge_bias: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.use_edge_bias = bool(use_edge_bias)

        self.state_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.action_dim),
            nn.SiLU(),
            nn.Linear(self.action_dim, self.action_dim),
        )

        self.edge_proj = nn.Linear(
            self.hidden_dim,
            self.action_dim,
            bias=False,
        )

        self.edge_bias = nn.Linear(self.hidden_dim, 1) if self.use_edge_bias else None

        self.reset_parameters()

    def forward(
        self,
        *,
        state_repr: StateRepresentation,
        frontier: FrontierEncoding,
    ) -> Tensor:
        if int(frontier.num_actions) == 0:
            return state_repr.query_h.new_empty((0,), dtype=torch.float32)

        if frontier.edge_h.ndim != 2 or int(frontier.edge_h.size(1)) != self.hidden_dim:
            raise ValueError(f"frontier.edge_h must have shape [F, {self.hidden_dim}], " f"got {tuple(frontier.edge_h.shape)}.")

        state_x = torch.cat(
            (
                state_repr.query_h,
                state_repr.selected_h,
                state_repr.covered_h,
            ),
            dim=-1,
        )

        state_h = self.state_proj(state_x)  # [S, D]
        edge_h = self.edge_proj(frontier.edge_h)  # [F, D]
        state_h_f = state_h.index_select(0, frontier.row_ids)

        score = (state_h_f * edge_h).sum(dim=-1)
        score = score / float(self.action_dim) ** 0.5

        if self.edge_bias is not None:
            score = score + self.edge_bias(frontier.edge_h).squeeze(-1)

        return score.float()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_xavier(module)

        if self.edge_bias is not None:
            init.zeros_(self.edge_bias.weight)
            init.zeros_(self.edge_bias.bias)


class ForwardPolicy(nn.Module):
    """
    Forward flow model.

    STOP flow:
        stop_log_flow(z) = log R_hat_psi(z)

    EXPAND flow:
        edge_log_flow(z,e)
          = semantic_prior(q,e)
          + edge_residual_weight * residual_theta(z,e)
          - frontier_size_correction * log |C(z)|

    STATE flow:
        state_log_flow(z)
          = logaddexp(stop_log_flow(z), logsumexp_e edge_log_flow(z,e))
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        semantic_relation_weight: float = 1.0,
        semantic_dst_weight: float = 0.5,
        edge_residual_weight: float = 0.25,
        frontier_size_correction: float = 1.0,
        action_hidden_dim: int | None = None,
        residual_action_dim: int | None = None,
        initial_stop_log_reward: float = -4.0,
    ) -> None:
        super().__init__()

        self.state_encoder = state_encoder

        hidden_dim = int(state_encoder.hidden_dim)
        self.action_hidden_dim = int(action_hidden_dim or hidden_dim)
        self.residual_action_dim = int(residual_action_dim or min(256, hidden_dim))

        self.edge_residual_weight = float(edge_residual_weight)
        self.frontier_size_correction = float(frontier_size_correction)

        if self.edge_residual_weight < 0.0:
            raise ValueError("edge_residual_weight must be nonnegative.")
        if self.frontier_size_correction < 0.0:
            raise ValueError("frontier_size_correction must be nonnegative.")

        self.stop_log_flow_predictor = StopLogFlowPredictor(
            hidden_dim=hidden_dim,
            action_hidden_dim=self.action_hidden_dim,
            initial_log_reward=initial_stop_log_reward,
        )

        self.semantic_prior = SemanticEdgePrior(
            relation_weight=semantic_relation_weight,
            dst_weight=semantic_dst_weight,
        )

        self.edge_residual_scorer = LowRankEdgeResidualScorer(
            hidden_dim=hidden_dim,
            action_dim=self.residual_action_dim,
            use_edge_bias=True,
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.state_encoder.hidden_dim)

    def forward(
        self,
        *,
        features: FeatureBank,
        state: StateBatch,
        context: GraphContext,
        action_space: ActionSpace,
    ) -> PolicyOutput:
        if int(action_space.num_states) != int(state.num_states):
            raise ValueError("action_space.num_states must match state.num_states: " f"{int(action_space.num_states)} != {int(state.num_states)}.")

        state_repr = self.encode_state(
            features=features,
            state=state,
            context=context,
        )

        stop_log_flow = self.stop_log_flow_predictor(state_repr)

        edge_log_flow, edge_raw_score = self.score_edge_flows(
            features=features,
            context=context,
            action_space=action_space,
            state_repr=state_repr,
        )

        state_log_flow, continue_log_flow = combine_action_flows(
            stop_log_flow=stop_log_flow,
            edge_log_flow=edge_log_flow,
            action_space=action_space,
        )

        return PolicyOutput(
            action_space=action_space,
            state_log_flow=state_log_flow,
            stop_log_flow=stop_log_flow,
            continue_log_flow=continue_log_flow,
            edge_log_flow=edge_log_flow,
            edge_raw_score=edge_raw_score,
        )

    def encode_state(
        self,
        *,
        features: FeatureBank,
        state: StateBatch,
        context: GraphContext,
    ) -> StateRepresentation:
        query_h = select_query_embedding(
            features,
            state.graph_ids,
        )

        selected_h = self.state_encoder.selected_edge_summary(
            features=features,
            state=state,
            context=context,
            query_h=query_h,
        )

        covered_h = self.state_encoder.covered_node_summary(
            features=features,
            state=state,
            context=context,
            query_h=query_h,
        )

        return StateRepresentation(
            query_h=query_h,
            selected_h=selected_h,
            covered_h=covered_h,
        )

    def encode_frontier(
        self,
        *,
        features: FeatureBank,
        context: GraphContext,
        action_space: ActionSpace,
    ) -> FrontierEncoding:
        row_ids = action_space.expand_state_ids
        edge_ids = action_space.expand_edge_ids

        if int(edge_ids.numel()) == 0:
            raise ValueError("encode_frontier requires at least one expansion action.")

        dst_node_ids = context.edge_dst.index_select(0, edge_ids)

        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            context=context,
            edge_ids=edge_ids,
        )

        relation_h = select_relation_embedding(
            features,
            edge_ids,
        )

        dst_h = select_node_embedding(
            features,
            dst_node_ids,
        )

        dst_text_mask = select_node_text_mask(
            features,
            dst_node_ids,
        ).to(dtype=torch.float32)

        return FrontierEncoding(
            row_ids=row_ids,
            edge_ids=edge_ids,
            dst_node_ids=dst_node_ids,
            edge_h=edge_h,
            relation_h=relation_h,
            dst_h=dst_h,
            dst_text_mask=dst_text_mask,
        )

    def score_edge_flows(
        self,
        *,
        features: FeatureBank,
        context: GraphContext,
        action_space: ActionSpace,
        state_repr: StateRepresentation,
    ) -> tuple[Tensor, Tensor]:
        if int(action_space.num_expansions) == 0:
            empty = state_repr.query_h.new_empty((0,), dtype=torch.float32)
            return empty, empty

        frontier = self.encode_frontier(
            features=features,
            context=context,
            action_space=action_space,
        )

        semantic_prior = self.semantic_prior(
            state_repr=state_repr,
            frontier=frontier,
        )

        residual = self.edge_residual_scorer(
            state_repr=state_repr,
            frontier=frontier,
        )

        edge_raw_score = (semantic_prior + float(self.edge_residual_weight) * residual).float()

        edge_log_flow = apply_frontier_mass_correction(
            edge_raw_score=edge_raw_score,
            action_space=action_space,
            frontier_size_correction=self.frontier_size_correction,
        )

        return edge_log_flow.float(), edge_raw_score.float()


def combine_action_flows(
    *,
    stop_log_flow: Tensor,
    edge_log_flow: Tensor,
    action_space: ActionSpace,
) -> tuple[Tensor, Tensor]:
    num_states = int(action_space.num_states)

    if stop_log_flow.shape != (num_states,):
        raise ValueError(f"stop_log_flow must have shape [{num_states}], " f"got {tuple(stop_log_flow.shape)}.")

    continue_log_flow = segment_logsumexp(
        values=edge_log_flow,
        segment_ids=action_space.expand_state_ids,
        num_segments=num_states,
    ).float()

    state_log_flow = torch.logaddexp(
        stop_log_flow,
        continue_log_flow,
    ).float()

    return state_log_flow, continue_log_flow


def apply_frontier_mass_correction(
    *,
    edge_raw_score: Tensor,
    action_space: ActionSpace,
    frontier_size_correction: float = 1.0,
) -> Tensor:
    """
    Apply per-state frontier-size correction.

    For every e in C(z):
        edge_log_flow(z,e)
          = edge_raw_score(z,e) - alpha * log |C(z)|

    This does not change P(e | continue, z). It only changes total CONTINUE
    mass relative to STOP.
    """
    if int(action_space.num_expansions) == 0:
        return edge_raw_score

    row_ids = action_space.expand_state_ids
    expand_count = action_space.expand_count.index_select(0, row_ids)

    log_expand_count = expand_count.to(dtype=edge_raw_score.dtype).clamp_min(1).log()

    return (edge_raw_score - float(frontier_size_correction) * log_expand_count).float()


__all__ = [
    "ForwardPolicy",
    "FrontierEncoding",
    "LowRankEdgeResidualScorer",
    "SemanticEdgePrior",
    "StateRepresentation",
    "StopLogFlowPredictor",
    "apply_frontier_mass_correction",
    "combine_action_flows",
]
