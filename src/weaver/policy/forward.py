from __future__ import annotations

import torch
from torch import nn

from src.graph.segments import segment_logsumexp
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

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 2, hidden_dim),
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

        stop_logit, edge_logit, edge_row_ids, edge_ids = self.action_logits(
            features=features,
            context=context,
            state=state,
            query_h=encoding.query_h,
            state_h=encoding.row_state_h,
            frontier=frontier,
        )
        state_log_flow = self.compute_state_log_flow(
            stop_logit=stop_logit,
            edge_logit=edge_logit,
            edge_row_ids=edge_row_ids,
            num_rows=state.num_rows,
        )

        return PolicyOutput(
            stop_logit=stop_logit,
            edge_logit=edge_logit,
            state_log_flow=state_log_flow,
            edge_row_ids=edge_row_ids,
            edge_ids=edge_ids,
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )

    def action_logits(
        self,
        *,
        features: EncodedFeatures,
        context: GraphContext,
        state: State,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        frontier: Frontier,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stop_logit = self.score_stop(
            query_h=query_h,
            state_h=state_h,
        )

        if frontier.edge_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=query_h.device)
            return stop_logit, stop_logit.new_empty((0,)), empty, empty

        src_node_ids = context.edge_index[0].index_select(0, frontier.edge_ids)
        dst_node_ids = context.edge_index[1].index_select(0, frontier.edge_ids)
        src_active = state.active_node_mask[frontier.row_ids, src_node_ids].float()
        dst_active = state.active_node_mask[frontier.row_ids, dst_node_ids].float()

        edge_h = self.state_encoder.encode_edge_tokens(
            features=features,
            src_node_ids=src_node_ids,
            edge_ids=frontier.edge_ids,
            dst_node_ids=dst_node_ids,
        )
        edge_type_h = torch.stack(
            [src_active, dst_active],
            dim=-1,
        )
        edge_logit = self.score_edges(
            query_h=query_h.index_select(0, frontier.row_ids),
            state_h=state_h.index_select(0, frontier.row_ids),
            edge_h=edge_h,
            edge_type_h=edge_type_h,
        )

        return stop_logit, edge_logit, frontier.row_ids, frontier.edge_ids

    def score_stop(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.stop_head(torch.cat([query_h, state_h], dim=-1)).squeeze(-1)

    def score_edges(
        self,
        *,
        query_h: torch.Tensor,
        state_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_type_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_head(
            torch.cat(
                [
                    query_h,
                    state_h,
                    edge_h,
                    edge_type_h,
                ],
                dim=-1,
            )
        ).squeeze(-1)

    def compute_state_log_flow(
        self,
        *,
        stop_logit: torch.Tensor,
        edge_logit: torch.Tensor,
        edge_row_ids: torch.Tensor,
        num_rows: int,
    ) -> torch.Tensor:
        state_log_flow = stop_logit.float()
        if edge_logit.numel() == 0:
            return state_log_flow

        edge_row_ids = edge_row_ids.to(device=edge_logit.device, dtype=torch.long)
        edge_log_flow = segment_logsumexp(
            values=edge_logit.float(),
            segment_ids=edge_row_ids,
            num_segments=int(num_rows),
        )
        return torch.logaddexp(state_log_flow, edge_log_flow)

    def reset_parameters(self) -> None:
        for module in self.stop_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)

        for module in self.edge_head:
            if isinstance(module, nn.Linear):
                init_xavier(module)


__all__ = [
    "ForwardPolicy",
]
