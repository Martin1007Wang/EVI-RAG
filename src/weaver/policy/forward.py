from __future__ import annotations

import torch
from torch.nn import init
from torch import nn

from src.graph.segments import segment_logsumexp
from src.utils.nn_utils import init_xavier

from ..context import GraphContext
from ..nn.feature_encoder import EncodedFeatures
from ..nn.state_encoder import StateEncoder
from ..state import Frontier, State
from .output import ForwardPolicyOutput


class ForwardPolicy(nn.Module):
    """
    Forward action-flow policy over STOP plus legal frontier edges.
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        max_expand_budget: int = 3,
    ) -> None:
        super().__init__()

        self.state_encoder = state_encoder
        self.max_expand_budget = int(max_expand_budget)

        hidden_dim = state_encoder.hidden_dim
        edge_dim = state_encoder.edge_encoder.output_dim

        self.budget_embedding = nn.Embedding(self.max_expand_budget + 1, hidden_dim)

        self.stop_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.edge_advantage_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    @property
    def hidden_dim(self) -> int:
        return self.state_encoder.hidden_dim

    def forward(
        self,
        *,
        features: EncodedFeatures,
        state: State,
        context: GraphContext,
        frontier: Frontier,
    ) -> ForwardPolicyOutput:
        encoding = self.state_encoder(
            features=features,
            state=state,
            context=context,
        )
        budget_h = self.encode_budget(state)

        (
            stop_log_flow,
            continue_log_flow,
            continue_log_gain,
            edge_log_flow,
            edge_log_reference,
            edge_log_advantage,
            frontier_row_ids,
            frontier_edge_ids,
        ) = self.action_log_flows(
            features=features,
            context=context,
            query_h=encoding.query_h,
            state_h=encoding.row_state_h,
            budget_h=budget_h,
            frontier=frontier,
        )
        state_log_flow = stop_log_flow + torch.nn.functional.softplus(continue_log_gain)
        stop_log_prob = -torch.nn.functional.softplus(continue_log_gain)
        edge_log_prob = edge_log_flow - state_log_flow.index_select(
            0,
            frontier_row_ids.to(device=state_log_flow.device, dtype=torch.long),
        )

        return ForwardPolicyOutput(
            frontier_row_ids=frontier_row_ids,
            frontier_edge_ids=frontier_edge_ids,
            stop_log_flow=stop_log_flow,
            continue_log_flow=continue_log_flow,
            continue_log_gain=continue_log_gain,
            edge_log_flow=edge_log_flow,
            edge_log_reference=edge_log_reference,
            edge_log_advantage=edge_log_advantage,
            state_log_flow=state_log_flow,
            stop_log_prob=stop_log_prob,
            edge_log_prob=edge_log_prob,
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )

    def action_log_flows(
        self,
        *,
        features: EncodedFeatures,
        context: GraphContext,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        budget_h: torch.Tensor,
        frontier: Frontier,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        stop_log_flow = self.score_stop_flow(
            query_h=query_h,
            state_h=state_h,
            budget_h=budget_h,
        ).float()
        continue_log_gain = stop_log_flow.new_full((query_h.size(0),), -torch.inf)
        continue_log_flow = stop_log_flow.new_full((query_h.size(0),), -torch.inf)

        if frontier.edge_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=query_h.device)
            empty_float = stop_log_flow.new_empty((0,))
            return (
                stop_log_flow,
                continue_log_flow,
                continue_log_gain,
                empty_float,
                empty_float,
                empty_float,
                empty,
                empty,
            )

        row_ids = frontier.row_ids.to(device=query_h.device, dtype=torch.long)
        row_frontier_size = torch.bincount(
            row_ids,
            minlength=int(query_h.size(0)),
        ).to(dtype=torch.float32, device=query_h.device)
        edge_log_reference = -torch.log(
            row_frontier_size.index_select(0, row_ids).clamp_min(1.0)
        )

        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            src_node_ids=torch.empty(0, dtype=torch.long, device=query_h.device),
            edge_ids=frontier.edge_ids,
            dst_node_ids=torch.empty(0, dtype=torch.long, device=query_h.device),
        )

        edge_log_advantage = self.score_edge_advantage(
            query_h=query_h.index_select(0, row_ids),
            state_h=state_h.index_select(0, row_ids),
            budget_h=budget_h.index_select(0, row_ids),
            edge_h=edge_h,
        ).float()
        edge_log_measure = edge_log_reference + edge_log_advantage
        continue_log_gain = segment_logsumexp(
            values=edge_log_measure,
            segment_ids=row_ids,
            num_segments=int(query_h.size(0)),
        )
        continue_log_flow = stop_log_flow + continue_log_gain
        edge_log_flow = stop_log_flow.index_select(0, row_ids) + edge_log_measure

        return (
            stop_log_flow,
            continue_log_flow,
            continue_log_gain,
            edge_log_flow,
            edge_log_reference,
            edge_log_advantage,
            frontier.row_ids,
            frontier.edge_ids,
        )

    def encode_budget(
        self,
        state: State,
    ) -> torch.Tensor:
        remaining = torch.clamp(
            self.max_expand_budget - state.depth.to(dtype=torch.long),
            min=0,
            max=self.max_expand_budget,
        )
        return self.budget_embedding(remaining)

    def score_stop_flow(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        budget_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.stop_flow_head(torch.cat([query_h, state_h, budget_h], dim=-1)).squeeze(-1)

    def score_edge_advantage(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        budget_h: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_advantage_head(
            torch.cat(
                [
                    query_h,
                    state_h,
                    budget_h,
                    edge_h,
                ],
                dim=-1,
            )
        ).squeeze(-1)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.budget_embedding.weight, mean=0.0, std=0.02)

        for module in self.stop_flow_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.stop_flow_head[-1])

        for module in self.edge_advantage_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.edge_advantage_head[-1])


__all__ = [
    "ForwardPolicy",
]


def _zero_linear(module: nn.Module) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}.")
    init.zeros_(module.weight)
    init.zeros_(module.bias)
