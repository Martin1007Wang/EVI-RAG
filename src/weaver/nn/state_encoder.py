from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.nn_utils import init_xavier
from src.weaver.context import GraphContext
from src.weaver.state import State

from .edge_encoder import EdgeEncoder
from .feature_encoder import (
    EncodedFeatures,
    select_edge_relation_model,
    select_node_model,
    select_query_model,
)


@dataclass(frozen=True, slots=True)
class StateEncoding:
    query_h: torch.Tensor
    row_state_h: torch.Tensor
    node_state_h: torch.Tensor
    edge_state_h: torch.Tensor


class SegmentTokenPool(nn.Module):
    """
    Project tokens and mean-pool them by rollout row.

    Contract:
    - tokens already live in model space.
    - row_ids already indexes rollout rows.
    - num_rows is the number of rollout states.
    - No dtype/device/view/detach/normalization repair is performed.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        self.reset_parameters()

    def forward(
        self,
        *,
        tokens: torch.Tensor,
        row_ids: torch.Tensor,
        num_rows: int,
    ) -> torch.Tensor:
        token_h = self.token_proj(tokens)

        out = token_h.new_zeros((num_rows, self.output_dim))
        out.scatter_add_(
            0,
            row_ids[:, None].expand(-1, self.output_dim),
            token_h,
        )
        counts = token_h.new_zeros((num_rows, 1))
        counts.scatter_add_(
            0,
            row_ids[:, None],
            token_h.new_ones((token_h.size(0), 1)),
        )
        return out / counts.clamp_min(1.0)

    def reset_parameters(self) -> None:
        for module in self.token_proj:
            if isinstance(module, nn.Linear):
                init_xavier(module)


class StateEncoder(nn.Module):
    """
    Encode rollout states from model-space features.

    State representation:

        query_h      = query model feature
        node_state_h = mean-pooled active node model tokens
        edge_state_h = mean-pooled selected edge model tokens
        row_state_h  = learned fusion of query/node/edge state

    Contract:
    - FeatureEncoder owns semantic/model-space construction.
    - StateEncoder consumes only model-space features.
    - EdgeEncoder returns role-preserving edge tokens, e.g. concat(src, rel, dst).
    - StateEncoder compresses edge tokens only for state representation.
    - No dtype/device/view/detach/normalization repair is performed.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_encoder: EdgeEncoder | None = None,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_encoder = edge_encoder or EdgeEncoder(hidden_dim=hidden_dim)

        self.node_pool = SegmentTokenPool(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
        )

        self.edge_pool = SegmentTokenPool(
            input_dim=self.edge_encoder.output_dim,
            output_dim=hidden_dim,
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.reset_parameters()

    def forward(
        self,
        *,
        features: EncodedFeatures,
        state: State,
        context: GraphContext,
    ) -> StateEncoding:
        num_rows = state.num_rows

        query_h = select_query_model(
            features,
            state.graph_ids,
        )

        node_state_h = self.encode_active_nodes(
            features=features,
            state=state,
            context=context,
            num_rows=num_rows,
            like=query_h,
        )

        edge_state_h = self.encode_selected_edges(
            features=features,
            state=state,
            context=context,
            num_rows=num_rows,
            like=query_h,
        )

        row_state_h = self.fuse(
            torch.cat(
                [
                    query_h,
                    node_state_h,
                    edge_state_h,
                ],
                dim=-1,
            )
        )

        return StateEncoding(
            query_h=query_h,
            row_state_h=row_state_h,
            node_state_h=node_state_h,
            edge_state_h=edge_state_h,
        )

    def encode_edge_tokens(
        self,
        *,
        features: EncodedFeatures,
        src_node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_encoder(
            src_h=select_node_model(features, src_node_ids),
            rel_h=select_edge_relation_model(features, edge_ids),
            dst_h=select_node_model(features, dst_node_ids),
        )

    def encode_active_nodes(
        self,
        *,
        features: EncodedFeatures,
        state: State,
        context: GraphContext,
        num_rows: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        row_ids, node_ids = state.active_node_mask.nonzero(as_tuple=True)

        if node_ids.numel() == 0:
            return like.new_zeros((num_rows, self.hidden_dim))

        return self.node_pool(
            tokens=select_node_model(features, node_ids),
            row_ids=row_ids,
            num_rows=num_rows,
        )

    def encode_selected_edges(
        self,
        *,
        features: EncodedFeatures,
        state: State,
        context: GraphContext,
        num_rows: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        row_ids, edge_ids = state.selected_edge_mask.nonzero(as_tuple=True)

        if edge_ids.numel() == 0:
            return like.new_zeros((num_rows, self.hidden_dim))

        src_node_ids = context.edge_index[0].index_select(0, edge_ids)
        dst_node_ids = context.edge_index[1].index_select(0, edge_ids)

        edge_h = self.encode_edge_tokens(
            features=features,
            src_node_ids=src_node_ids,
            edge_ids=edge_ids,
            dst_node_ids=dst_node_ids,
        )

        return self.edge_pool(
            tokens=edge_h,
            row_ids=row_ids,
            num_rows=num_rows,
        )

    def reset_parameters(self) -> None:
        for module in self.fuse:
            if isinstance(module, nn.Linear):
                init_xavier(module)


__all__ = [
    "SegmentTokenPool",
    "StateEncoder",
    "StateEncoding",
]
