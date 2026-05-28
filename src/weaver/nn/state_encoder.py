from __future__ import annotations

import math

import torch
from torch import nn

from src.graph.segments import segment_softmax
from src.utils.nn_utils import init_xavier
from src.weaver.context import GraphContext
from src.weaver.state import StateBatch

from .edge_encoder import EdgeEncoder
from ..feature import (
    FeatureBank,
    select_node_embedding,
    select_query_embedding,
    select_relation_embedding,
)

Tensor = torch.Tensor


class QueryTokenPooler(nn.Module):
    """
    Query-conditioned ragged attention pooling.

    Contract:
    - query_h: [S, H]
    - token_h: [T, H]
    - token_row_ids: [T], token_row_ids[t] in [0, S)
    - output: [S, H]

    Mixed precision invariant:
    - projections may run under autocast;
    - score, segment softmax, and scatter reduction run in fp32;
    - returned hidden state uses the projected value dtype.
    """

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.empty_token = nn.Parameter(torch.empty(self.hidden_dim))

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.reset_parameters()

    def forward(
        self,
        *,
        query_h: Tensor,
        token_h: Tensor,
        token_row_ids: Tensor,
    ) -> Tensor:
        self._validate_inputs(
            query_h=query_h,
            token_h=token_h,
            token_row_ids=token_row_ids,
        )

        num_states = int(query_h.size(0))

        token_h = token_h.to(device=query_h.device)
        token_row_ids = token_row_ids.to(
            device=query_h.device,
            dtype=torch.long,
        ).view(-1)

        token_h, token_row_ids = self._add_empty_tokens(
            token_h=token_h,
            token_row_ids=token_row_ids,
            num_states=num_states,
        )

        q = self.q_proj(query_h.index_select(0, token_row_ids))
        k = self.k_proj(token_h)
        v = self.v_proj(token_h)

        output_dtype = v.dtype

        q = q.float()
        k = k.float()
        v = v.float()

        score = (q * k).sum(dim=-1)
        score = score / math.sqrt(float(self.hidden_dim))

        alpha = segment_softmax(
            score,
            token_row_ids,
            num_segments=num_states,
        ).float()

        pooled = torch.zeros(
            (num_states, self.hidden_dim),
            dtype=torch.float32,
            device=v.device,
        )

        pooled.scatter_add_(
            0,
            token_row_ids[:, None].expand(-1, self.hidden_dim),
            alpha[:, None] * v,
        )

        return self.out(pooled.to(dtype=output_dtype))

    def _add_empty_tokens(
        self,
        *,
        token_h: Tensor,
        token_row_ids: Tensor,
        num_states: int,
    ) -> tuple[Tensor, Tensor]:
        if num_states == 0:
            return token_h, token_row_ids

        has_token = torch.zeros(
            num_states,
            dtype=torch.bool,
            device=token_row_ids.device,
        )

        if int(token_row_ids.numel()) > 0:
            has_token[token_row_ids] = True

        missing_row_ids = (~has_token).nonzero(as_tuple=False).flatten()

        if int(missing_row_ids.numel()) == 0:
            return token_h, token_row_ids

        empty_h = self.empty_token.to(
            device=token_h.device,
            dtype=token_h.dtype,
        ).expand(int(missing_row_ids.numel()), -1)

        return (
            torch.cat((token_h, empty_h), dim=0),
            torch.cat((token_row_ids, missing_row_ids), dim=0),
        )

    def _validate_inputs(
        self,
        *,
        query_h: Tensor,
        token_h: Tensor,
        token_row_ids: Tensor,
    ) -> None:
        if query_h.ndim != 2 or int(query_h.size(1)) != self.hidden_dim:
            raise ValueError(f"query_h must have shape [S, {self.hidden_dim}], " f"got {tuple(query_h.shape)}.")

        if token_h.ndim != 2 or int(token_h.size(1)) != self.hidden_dim:
            raise ValueError(f"token_h must have shape [T, {self.hidden_dim}], " f"got {tuple(token_h.shape)}.")

        if token_row_ids.ndim != 1:
            raise ValueError(f"token_row_ids must have shape [T], " f"got {tuple(token_row_ids.shape)}.")

        if int(token_h.size(0)) != int(token_row_ids.numel()):
            raise ValueError("token_h and token_row_ids must have matching first dimension.")

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.empty_token,
            mean=0.0,
            std=0.02,
        )

        init_xavier(self.q_proj)
        init_xavier(self.k_proj)
        init_xavier(self.v_proj)

        for module in self.out:
            if isinstance(module, nn.Linear):
                init_xavier(module)


class StateEncoder(nn.Module):
    """
    Encode state-derived summaries for the forward policy.

    Responsibilities:
    - encode physical KG edges e=(u, r, v);
    - pool selected evidence-edge tokens;
    - pool covered-node tokens.

    Non-responsibilities:
    - action-space enumeration;
    - state transition;
    - STOP scoring;
    - reward computation;
    - budget logic.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_encoder: EdgeEncoder | None = None,
        token_pooler: QueryTokenPooler | None = None,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.edge_encoder = edge_encoder or EdgeEncoder(
            hidden_dim=self.hidden_dim,
        )
        self.token_pooler = token_pooler or QueryTokenPooler(
            hidden_dim=self.hidden_dim,
        )

        edge_output_dim = int(self.edge_encoder.output_dim)
        if edge_output_dim != self.hidden_dim:
            raise ValueError(
                "StateEncoder requires edge_encoder.output_dim == hidden_dim, "
                f"got edge_encoder.output_dim={edge_output_dim}, "
                f"hidden_dim={self.hidden_dim}."
            )

    def encode_edge_tokens(
        self,
        *,
        features: FeatureBank,
        context: GraphContext,
        edge_ids: Tensor,
    ) -> Tensor:
        edge_ids = edge_ids.to(
            device=context.edge_src.device,
            dtype=torch.long,
        ).view(-1)

        if int(edge_ids.numel()) == 0:
            return features.relation_embedding.new_empty(
                (0, self.hidden_dim),
            )

        src_node_ids = context.edge_src.index_select(0, edge_ids)
        dst_node_ids = context.edge_dst.index_select(0, edge_ids)

        return self.edge_encoder(
            src_h=select_node_embedding(features, src_node_ids),
            rel_h=select_relation_embedding(features, edge_ids),
            dst_h=select_node_embedding(features, dst_node_ids),
        )

    def selected_edge_summary(
        self,
        *,
        features: FeatureBank,
        state: StateBatch,
        context: GraphContext,
        query_h: Tensor | None = None,
    ) -> Tensor:
        if query_h is None:
            query_h = select_query_embedding(
                features,
                state.graph_ids,
            )

        row_ids, edge_ids = selected_edge_pairs(state)

        return self.token_pooler(
            query_h=query_h,
            token_h=self.encode_edge_tokens(
                features=features,
                context=context,
                edge_ids=edge_ids,
            ),
            token_row_ids=row_ids,
        )

    def covered_node_summary(
        self,
        *,
        features: FeatureBank,
        state: StateBatch,
        context: GraphContext,
        query_h: Tensor | None = None,
    ) -> Tensor:
        if query_h is None:
            query_h = select_query_embedding(
                features,
                state.graph_ids,
            )

        row_ids, node_ids = state.covered_node_pairs(context)

        if int(node_ids.numel()) == 0:
            token_h = query_h.new_empty((0, self.hidden_dim))
        else:
            token_h = select_node_embedding(features, node_ids)

        return self.token_pooler(
            query_h=query_h,
            token_h=token_h,
            token_row_ids=row_ids,
        )


def selected_edge_pairs(state: StateBatch) -> tuple[Tensor, Tensor]:
    """
    Convert padded selected-edge state into ragged pairs.

    Input:
    - state.edge_ids: [S, B]
    - state.edge_count: [S]

    Output:
    - row_ids: [T]
    - edge_ids: [T]
    """

    _, budget = state.edge_ids.shape

    slot_ids = torch.arange(
        int(budget),
        dtype=torch.long,
        device=state.device,
    )

    valid = slot_ids.view(1, -1).lt(state.edge_count.view(-1, 1))
    row_ids, col_ids = valid.nonzero(as_tuple=True)

    return row_ids, state.edge_ids[row_ids, col_ids]


__all__ = [
    "QueryTokenPooler",
    "StateEncoder",
    "selected_edge_pairs",
]
