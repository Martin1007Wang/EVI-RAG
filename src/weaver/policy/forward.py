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

        self.terminal_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.continue_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.edge_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    @property
    def hidden_dim(self) -> int:
        return self.state_encoder.hidden_dim

    @property
    def stop_flow_head(self) -> nn.Module:
        return self.terminal_flow_head

    @property
    def edge_advantage_head(self) -> nn.Module:
        return self.edge_score_head

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
            terminal_log_flow,
            continue_log_flow,
            state_log_flow,
            edge_logit,
            edge_log_prob,
            edge_log_flow,
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
        stop_log_prob = terminal_log_flow - state_log_flow
        expand_log_prob = continue_log_flow - state_log_flow
        edge_action_log_prob = edge_log_flow - state_log_flow.index_select(
            0,
            frontier_row_ids.to(device=state_log_flow.device, dtype=torch.long),
        )

        return ForwardPolicyOutput(
            frontier_row_ids=frontier_row_ids,
            frontier_edge_ids=frontier_edge_ids,
            terminal_log_flow=terminal_log_flow,
            continue_log_flow=continue_log_flow,
            state_log_flow=state_log_flow,
            edge_logit=edge_logit,
            edge_log_prob=edge_log_prob,
            edge_log_flow=edge_log_flow,
            stop_log_prob=stop_log_prob,
            expand_log_prob=expand_log_prob,
            edge_action_log_prob=edge_action_log_prob,
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
        terminal_log_flow = self.score_terminal_flow(
            query_h=query_h,
            state_h=state_h,
            budget_h=budget_h,
        ).float()
        continue_log_flow = self.score_continue_flow(
            query_h=query_h,
            state_h=state_h,
            budget_h=budget_h,
        ).float()
        can_continue = torch.zeros(query_h.size(0), dtype=torch.bool, device=query_h.device)
        if frontier.row_ids.numel() > 0:
            can_continue.index_fill_(0, frontier.row_ids.to(device=query_h.device, dtype=torch.long), True)
        continue_log_flow = continue_log_flow.masked_fill(~can_continue, -torch.inf)
        state_log_flow = torch.logaddexp(terminal_log_flow, continue_log_flow)

        if frontier.edge_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=query_h.device)
            empty_float = terminal_log_flow.new_empty((0,))
            return (
                terminal_log_flow,
                continue_log_flow,
                state_log_flow,
                empty_float,
                empty_float,
                empty_float,
                empty,
                empty,
            )

        row_ids = frontier.row_ids.to(device=query_h.device, dtype=torch.long)
        src_node_ids = context.edge_index[0].index_select(0, frontier.edge_ids)
        dst_node_ids = context.edge_index[1].index_select(0, frontier.edge_ids)
        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            src_node_ids=src_node_ids,
            edge_ids=frontier.edge_ids,
            dst_node_ids=dst_node_ids,
            query_h=query_h.index_select(0, row_ids),
        )

        edge_logit = self.score_edge(
            query_h=query_h.index_select(0, row_ids),
            state_h=state_h.index_select(0, row_ids),
            budget_h=budget_h.index_select(0, row_ids),
            edge_h=edge_h,
        ).float()
        edge_log_prob = segment_log_softmax(
            edge_logit,
            row_ids,
            num_segments=int(query_h.size(0)),
        ).float()
        edge_log_flow = continue_log_flow.index_select(0, row_ids) + edge_log_prob

        return (
            terminal_log_flow,
            continue_log_flow,
            state_log_flow,
            edge_logit,
            edge_log_prob,
            edge_log_flow,
            frontier.row_ids,
            frontier.edge_ids,
        )

    def encode_budget(
        self,
        state: State,
    ) -> torch.Tensor:
        remaining = torch.clamp(
            state.remaining_budget.to(dtype=torch.long),
            min=0,
            max=self.max_expand_budget,
        )
        return self.budget_embedding(remaining)

    def score_terminal_flow(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        budget_h: torch.Tensor,
    ) -> torch.Tensor:
        del query_h
        return self.terminal_flow_head(torch.cat([state_h, budget_h], dim=-1)).squeeze(-1)

    def score_continue_flow(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        budget_h: torch.Tensor,
    ) -> torch.Tensor:
        del query_h
        return self.continue_flow_head(torch.cat([state_h, budget_h], dim=-1)).squeeze(-1)

    def score_edge(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        budget_h: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_score_head(
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

        for module in self.terminal_flow_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.terminal_flow_head[-1])

        for module in self.continue_flow_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.continue_flow_head[-1])

        for module in self.edge_score_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.edge_score_head[-1])


__all__ = [
    "ForwardPolicy",
]


def _zero_linear(module: nn.Module) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module).__name__}.")
    init.zeros_(module.weight)
    init.zeros_(module.bias)
