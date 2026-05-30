from __future__ import annotations

import math

import torch
from torch import nn

from src.graph.segments import segment_logsumexp, segment_softmax
from src.weaver.feature import FeaturePack
from src.weaver.nn import (
    PolicyCache,
    PolicyCacheBuilder,
    StateEncoder,
)
from src.weaver.context import GraphContext
from src.weaver.state import FrontierEncoding, StateBatch, frontier_from_graph

from .output import PolicyOutput

Tensor = torch.Tensor


class LowRankInteraction(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        rank: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.left_norm = nn.LayerNorm(self.hidden_dim)
        self.right_norm = nn.LayerNorm(self.hidden_dim)
        self.left_proj = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.right_proj = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.out = nn.Linear(self.rank, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(
        self,
        *,
        left_h: torch.Tensor,
        right_h: torch.Tensor,
    ) -> torch.Tensor:
        left = self.left_proj(self.left_norm(left_h.float()))
        right = self.right_proj(self.right_norm(right_h.float()))
        return self.out(left * right).squeeze(-1)


class EdgeFlowHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        interaction: LowRankInteraction,
        initial_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_unary = nn.Linear(hidden_dim, 1)
        self.interaction = interaction
        nn.init.zeros_(self.edge_unary.weight)
        nn.init.constant_(self.edge_unary.bias, float(initial_bias))

    def forward(
        self,
        *,
        state_selected_h: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        edge_h = edge_h.float()
        unary = self.edge_unary(edge_h).squeeze(-1)
        cross = self.interaction(left_h=state_selected_h, right_h=edge_h)
        return unary + cross


class StopFlowHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        interaction: LowRankInteraction,
        initial_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.sel_unary = nn.Linear(hidden_dim, 1)
        self.frontier_unary = nn.Linear(hidden_dim, 1)
        self.interaction = interaction
        self.bias = nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))
        nn.init.zeros_(self.sel_unary.weight)
        nn.init.zeros_(self.sel_unary.bias)
        nn.init.zeros_(self.frontier_unary.weight)
        nn.init.zeros_(self.frontier_unary.bias)

    def forward(
        self,
        *,
        state_selected_h: torch.Tensor,
        state_frontier_h: torch.Tensor,
    ) -> torch.Tensor:
        selected_h = state_selected_h.float()
        frontier_h = state_frontier_h.float()
        sel_term = self.sel_unary(selected_h).squeeze(-1)
        frontier_penalty = torch.nn.functional.softplus(
            self.frontier_unary(frontier_h).squeeze(-1)
        ) - math.log(2.0)
        cross_term = self.interaction(left_h=selected_h, right_h=frontier_h)
        return self.bias + sel_term - frontier_penalty + cross_term


class ForwardPolicy(nn.Module):
    def __init__(
        self,
        *,
        cache_builder: PolicyCacheBuilder,
        state_encoder: StateEncoder,
        stop_head: StopFlowHead,
        edge_head: EdgeFlowHead,
    ) -> None:
        super().__init__()
        if stop_head.interaction is not edge_head.interaction:
            raise ValueError("stop_head and edge_head must share the same interaction module.")
        self.cache_builder = cache_builder
        self.state_encoder = state_encoder
        self.stop_head = stop_head
        self.edge_head = edge_head

    def compute_cache(self, features: FeaturePack) -> PolicyCache:
        return self.cache_builder(features)

    def forward(
        self,
        *,
        state: StateBatch,
        features: FeaturePack,
        graph_context: GraphContext | None = None,
        remaining_budget: torch.Tensor | None = None,
        cache: PolicyCache | None = None,
        potentials: PolicyCache | None = None,
    ) -> PolicyOutput:
        if cache is None:
            cache = potentials
        if cache is None:
            cache = self.compute_cache(features)

        if remaining_budget is None:
            remaining_budget = state.budget_left

        if graph_context is not None:
            frontier = frontier_from_graph(
                state=state,
                graph=graph_context,
                remaining_budget=remaining_budget,
            )
        else:
            frontier = state.frontier(
                edge_src=features.edge_src,
                edge_dst=features.edge_dst,
                remaining_budget=remaining_budget,
            )

        state_encoding = self.state_encoder(
            state=state,
            cache=cache,
            frontier=frontier,
            remaining_budget=remaining_budget,
        )
        state_selected_h = state_encoding.state_selected_h.float()

        frontier_selected_h = state_selected_h.index_select(0, frontier.row_ids)
        frontier_edge_h = cache.edge_h.index_select(0, frontier.edge_ids).float()
        edge_log_flow = self.edge_head(
            state_selected_h=frontier_selected_h,
            edge_h=frontier_edge_h,
        ).float()
        state_frontier_h = _pool_frontier_opportunities(
            edge_log_flow=edge_log_flow,
            edge_h=frontier_edge_h,
            frontier=frontier,
            num_states=state.num_states,
            hidden_dim=state_selected_h.size(-1),
        )
        stop_log_flow = self.stop_head(
            state_selected_h=state_selected_h,
            state_frontier_h=state_frontier_h,
        ).float()

        continue_log_flow = _continue_log_flow_from_frontier(
            edge_log_flow=edge_log_flow,
            frontier=frontier,
            num_states=state.num_states,
        )
        state_log_flow = torch.logaddexp(
            stop_log_flow,
            continue_log_flow,
        )

        return PolicyOutput(
            state_log_flow=state_log_flow,
            stop_log_flow=stop_log_flow,
            continue_log_flow=continue_log_flow,
            edge_log_flow=edge_log_flow,
            frontier=frontier,
            state_selected_h=state_selected_h,
            state_frontier_h=state_frontier_h,
        )


def _pool_frontier_opportunities(
    *,
    edge_log_flow: Tensor,
    edge_h: Tensor,
    frontier: FrontierEncoding,
    num_states: int,
    hidden_dim: int,
) -> Tensor:
    if int(edge_log_flow.numel()) == 0:
        return torch.zeros(
            (num_states, hidden_dim),
            dtype=torch.float32,
            device=frontier.remaining_budget.device,
        )
    weights = segment_softmax(
        edge_log_flow,
        segment_ids=frontier.row_ids,
        num_segments=num_states,
    ).view(-1, 1)
    out = torch.zeros(
        (num_states, hidden_dim),
        dtype=torch.float32,
        device=edge_h.device,
    )
    out.scatter_add_(
        0,
        frontier.row_ids.view(-1, 1).expand(-1, hidden_dim),
        edge_h.float() * weights,
    )
    return out


def _continue_log_flow_from_frontier(
    *,
    edge_log_flow: Tensor,
    frontier: FrontierEncoding,
    num_states: int,
) -> Tensor:
    if int(edge_log_flow.numel()) == 0:
        return torch.full(
            (num_states,),
            float("-inf"),
            dtype=torch.float32,
            device=frontier.remaining_budget.device,
        )
    return segment_logsumexp(
        values=edge_log_flow,
        segment_ids=frontier.row_ids,
        num_segments=num_states,
    )


__all__ = [
    "EdgeFlowHead",
    "ForwardPolicy",
    "LowRankInteraction",
    "StopFlowHead",
]
