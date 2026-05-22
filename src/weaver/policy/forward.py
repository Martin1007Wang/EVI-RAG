from __future__ import annotations

import torch
from torch.nn import init
from torch import nn

from src.utils.nn_utils import init_xavier

from ..context import GraphContext
from ..nn.feature_encoder import EncodedFeatures
from ..nn.state_encoder import StateEncoder
from ..state import Frontier, State
from .output import PolicyOutput


class ForwardPolicy(nn.Module):
    """
    Forward policy scores for a hierarchical STOP/CONTINUE distribution.
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

        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.edge_head = nn.Sequential(
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
    ) -> PolicyOutput:
        encoding = self.state_encoder(
            features=features,
            state=state,
            context=context,
        )

        stop_logit, log_flow, edge_logit, edge_row_ids, edge_ids = self.action_logits(
            features=features,
            context=context,
            query_h=encoding.query_h,
            state_h=encoding.row_state_h,
            frontier=frontier,
        )

        return PolicyOutput(
            stop_logit=stop_logit,
            log_flow=log_flow,
            edge_logit=edge_logit,
            frontier=Frontier(
                row_ids=edge_row_ids,
                edge_ids=edge_ids,
            ),
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )

    def action_logits(
        self,
        *,
        features: EncodedFeatures,
        context: GraphContext,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        frontier: Frontier,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stop_logit = self.score_stop(
            query_h=query_h,
            state_h=state_h,
        )
        log_flow = self.score_flow(
            query_h=query_h,
            state_h=state_h,
        )

        if frontier.edge_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=query_h.device)
            return stop_logit, log_flow, stop_logit.new_empty((0,)), empty, empty

        src_node_ids = context.edge_index[0].index_select(0, frontier.edge_ids)
        dst_node_ids = context.edge_index[1].index_select(0, frontier.edge_ids)

        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            src_node_ids=src_node_ids,
            edge_ids=frontier.edge_ids,
            dst_node_ids=dst_node_ids,
        )
        edge_logit = self.score_edges(
            query_h=query_h.index_select(0, frontier.row_ids),
            state_h=state_h.index_select(0, frontier.row_ids),
            edge_h=edge_h,
        )

        return stop_logit, log_flow, edge_logit, frontier.row_ids, frontier.edge_ids

    def score_stop(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
    ) -> torch.Tensor:
        stop_features = torch.cat(
            [
                query_h,
                state_h,
            ],
            dim=-1,
        )
        return self.stop_head(stop_features).squeeze(-1)

    def score_flow(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow_head(torch.cat([query_h, state_h], dim=-1)).squeeze(-1)

    def score_edges(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_head(
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
        for module in self.stop_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.stop_head[-1])

        for module in self.flow_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)
        _zero_linear(self.flow_head[-1])

        for module in self.edge_head:
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
