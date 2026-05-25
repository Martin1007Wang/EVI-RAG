from __future__ import annotations

import math

import torch
from torch import nn

from src.graph.segments import segment_softmax
from src.utils.nn_utils import init_xavier
from src.weaver.context import GraphContext
from src.weaver.state import StateBatch

from .edge_encoder import EdgeEncoder
from .feature_encoder import (
    FeatureBank,
    select_node_embedding,
    select_query_embedding,
    select_relation_embedding,
)

Tensor = torch.Tensor


class QueryTokenPooler(nn.Module):
    """
    Query-conditioned ragged token pooling.

    Input:
    - query_h: [S, H]
    - token_h: [T, H]
    - token_row_ids[t] gives the state row of token t

    Output:
    - pooled_h: [S, H]

    This module does not know whether tokens are edges, nodes, anchors, or
    anything else. Token construction is owned by StateEncoder methods.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
    ) -> None:
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
        num_states: int,
    ) -> Tensor:
        num_states = int(num_states)

        if query_h.shape != (num_states, self.hidden_dim):
            raise ValueError(f"query_h must have shape [{num_states}, {self.hidden_dim}], " f"got {tuple(query_h.shape)}.")

        if token_h.ndim != 2 or int(token_h.size(1)) != self.hidden_dim:
            raise ValueError(f"token_h must have shape [T, {self.hidden_dim}], " f"got {tuple(token_h.shape)}.")

        if token_row_ids.ndim != 1:
            raise ValueError("token_row_ids must have shape [T].")

        if int(token_h.size(0)) != int(token_row_ids.numel()):
            raise ValueError("token_h and token_row_ids must have matching length.")

        if int(token_h.numel()) == 0:
            token_h = self.empty_token.unsqueeze(0).expand(num_states, -1)
            token_row_ids = torch.arange(
                num_states,
                dtype=torch.long,
                device=query_h.device,
            )

        token_row_ids = token_row_ids.to(
            device=query_h.device,
            dtype=torch.long,
        )

        q = self.q_proj(query_h.index_select(0, token_row_ids))
        k = self.k_proj(token_h)
        v = self.v_proj(token_h)

        score = (q * k).sum(dim=-1) / math.sqrt(float(self.hidden_dim))

        alpha = segment_softmax(
            score,
            token_row_ids,
            num_segments=num_states,
        )

        pooled = v.new_zeros((num_states, self.hidden_dim))
        pooled.scatter_add_(
            0,
            token_row_ids[:, None].expand(-1, self.hidden_dim),
            alpha.unsqueeze(-1) * v,
        )

        return self.out(pooled)

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
    - encode physical KG edges e=(u,r,v) into model-space edge tokens;
    - pool selected-edge tokens into selected_h(z);
    - pool covered-node tokens into covered_h(z).

    This module does not:
    - enumerate legal actions;
    - update state;
    - score STOP;
    - score action probabilities.
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

    @property
    def edge_output_dim(self) -> int:
        return int(self.edge_encoder.output_dim)

    def query_embeddings(
        self,
        *,
        features: FeatureBank,
        state: StateBatch,
    ) -> Tensor:
        return select_query_embedding(
            features,
            state.graph_ids,
        )

    def encode_edge_tokens(
        self,
        *,
        features: FeatureBank,
        context: GraphContext,
        edge_ids: Tensor,
    ) -> Tensor:
        """
        Encode physical KG edges by edge id.

        edge_ids are physical edge ids inside the batched GraphContext.
        """

        edge_ids = edge_ids.to(dtype=torch.long).view(-1)

        if int(edge_ids.numel()) == 0:
            return features.relation_embedding.new_empty(
                (0, self.edge_output_dim),
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
        """
        Query-conditioned summary over selected evidence-edge tokens.

        Tokens:
        - selected physical edges edge_ids[s, :edge_count[s]]

        Empty states receive the learned empty token through QueryTokenPooler.
        """

        query_h = self.query_embeddings(features=features, state=state) if query_h is None else query_h

        row_ids, edge_ids = selected_edge_pairs(state)

        token_h = self.encode_edge_tokens(
            features=features,
            context=context,
            edge_ids=edge_ids,
        )

        return self.token_pooler(
            query_h=query_h,
            token_h=token_h,
            token_row_ids=row_ids,
            num_states=state.num_states,
        )

    def covered_node_summary(
        self,
        *,
        features: FeatureBank,
        state: StateBatch,
        context: GraphContext,
        query_h: Tensor | None = None,
    ) -> Tensor:
        """
        Query-conditioned summary over covered nodes.

        Covered nodes are defined by StateBatch:
        - graph anchors;
        - sources of selected edges;
        - destinations of selected edges.

        No anchor role embedding is added here. If anchor identity matters, it
        should be represented by the node/entity text embedding or introduced
        explicitly in the policy, not hidden inside StateEncoder.
        """

        query_h = self.query_embeddings(features=features, state=state) if query_h is None else query_h

        row_ids, node_ids = state.covered_node_pairs(context)

        if int(node_ids.numel()) == 0:
            token_h = query_h.new_empty((0, self.hidden_dim))
        else:
            token_h = select_node_embedding(features, node_ids)

        return self.token_pooler(
            query_h=query_h,
            token_h=token_h,
            token_row_ids=row_ids,
            num_states=state.num_states,
        )


def selected_edge_pairs(state: StateBatch) -> tuple[Tensor, Tensor]:
    """
    Return padded-free (state_id, selected_edge_id) pairs.

    This is a derived view from StateBatch.edge_ids and StateBatch.edge_count.
    It does not become stored state.
    """

    num_states, budget = state.edge_ids.shape

    if int(num_states) == 0 or int(budget) == 0:
        empty = torch.empty(
            0,
            dtype=torch.long,
            device=state.device,
        )
        return empty, empty

    flat_edge_ids = state.edge_ids.reshape(-1)

    flat_state_ids = torch.repeat_interleave(
        torch.arange(
            int(num_states),
            dtype=torch.long,
            device=state.device,
        ),
        int(budget),
    )

    valid = flat_edge_ids.ge(0)

    return flat_state_ids[valid], flat_edge_ids[valid]


__all__ = [
    "QueryTokenPooler",
    "StateEncoder",
    "selected_edge_pairs",
]
