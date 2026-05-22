from __future__ import annotations

import torch
from torch.nn import init
from torch import nn

from src.graph.segments import segment_log_softmax
from src.utils.nn_utils import init_xavier

from ..context import GraphContext
from ..nn.feature_encoder import EncodedFeatures
from ..nn.state_encoder import StateEncoder
from ..state import Frontier, State
from .output import ForwardPolicyOutput


class ForwardPolicy(nn.Module):
    """
    Forward action-flow policy over TERMINAL plus legal frontier edges.
    """

    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
    ) -> None:
        super().__init__()

        self.state_encoder = state_encoder

        hidden_dim = state_encoder.hidden_dim
        edge_dim = state_encoder.edge_encoder.output_dim

        self.terminal_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.continuation_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.edge_policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
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

        (
            terminal_log_flow,
            continuation_log_flow,
            edge_log_flow,
            edge_log_policy,
            frontier_row_ids,
            frontier_edge_ids,
        ) = self.action_log_flows(
            features=features,
            context=context,
            query_h=encoding.query_h,
            state_h=encoding.row_state_h,
            frontier=frontier,
        )
        state_log_flow = torch.logaddexp(terminal_log_flow, continuation_log_flow)
        terminal_log_prob = terminal_log_flow - state_log_flow
        edge_log_prob = edge_log_flow - state_log_flow.index_select(
            0,
            frontier_row_ids.to(device=state_log_flow.device, dtype=torch.long),
        )

        return ForwardPolicyOutput(
            frontier_row_ids=frontier_row_ids,
            frontier_edge_ids=frontier_edge_ids,
            terminal_log_flow=terminal_log_flow,
            continuation_log_flow=continuation_log_flow,
            edge_log_flow=edge_log_flow,
            edge_log_policy=edge_log_policy,
            state_log_flow=state_log_flow,
            terminal_log_prob=terminal_log_prob,
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
        frontier: Frontier,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        terminal_log_flow = self.score_terminal_flow(
            query_h=query_h,
            state_h=state_h,
        ).float()
        continuation_log_flow = terminal_log_flow.new_full((query_h.size(0),), -torch.inf)

        if frontier.edge_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=query_h.device)
            empty_float = continuation_log_flow.new_empty((0,))
            return terminal_log_flow, continuation_log_flow, empty_float, empty_float, empty, empty

        row_ids = frontier.row_ids.to(device=query_h.device, dtype=torch.long)

        src_node_ids = context.edge_index[0].index_select(0, frontier.edge_ids)
        dst_node_ids = context.edge_index[1].index_select(0, frontier.edge_ids)

        continuation_rows = self.score_continuation_flow(
            query_h=query_h.index_select(0, row_ids),
            state_h=state_h.index_select(0, row_ids),
        ).float()
        continuation_log_flow.scatter_reduce_(
            0,
            row_ids,
            continuation_rows,
            reduce="amax",
            include_self=True,
        )

        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            src_node_ids=src_node_ids,
            edge_ids=frontier.edge_ids,
            dst_node_ids=dst_node_ids,
        )
        edge_log_policy = self.score_edge_policy(
            query_h=query_h.index_select(0, row_ids),
            state_h=state_h.index_select(0, row_ids),
            edge_h=edge_h,
        ).float()
        edge_log_policy = segment_log_softmax(
            edge_log_policy,
            row_ids,
            num_segments=int(query_h.size(0)),
        )
        edge_log_flow = continuation_log_flow.index_select(0, row_ids) + edge_log_policy

        return (
            terminal_log_flow,
            continuation_log_flow,
            edge_log_flow,
            edge_log_policy,
            frontier.row_ids,
            frontier.edge_ids,
        )

    def score_terminal_flow(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.terminal_flow_head(torch.cat([query_h, state_h], dim=-1)).squeeze(-1)

    def score_continuation_flow(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.continuation_flow_head(torch.cat([query_h, state_h], dim=-1)).squeeze(-1)

    def score_edge_policy(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_policy_head(
            torch.cat(
                [
                    query_h,
                    state_h,
                    edge_h,
                ],
                dim=-1,
            )
        ).squeeze(-1)

    def reset_parameters(self) -> None:
        for module in self.terminal_flow_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.terminal_flow_head[-1])

        for module in self.continuation_flow_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.continuation_flow_head[-1])

        for module in self.edge_policy_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)


__all__ = [
    "ForwardPolicy",
]


def _zero_linear(module: nn.Module) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}.")
    init.zeros_(module.weight)
    init.zeros_(module.bias)
