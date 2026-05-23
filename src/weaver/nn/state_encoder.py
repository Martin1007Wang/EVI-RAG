from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from src.graph.segments import segment_softmax
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
    edge_state_h: torch.Tensor


class QueryEdgeEncoder(nn.Module):
    """
    Query-conditioned edge reader over [src, rel, dst].
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        self.src_role = nn.Parameter(torch.empty(self.hidden_dim))
        self.rel_role = nn.Parameter(torch.empty(self.hidden_dim))
        self.dst_role = nn.Parameter(torch.empty(self.hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self,
        *,
        query_h: torch.Tensor,
        src_h: torch.Tensor,
        rel_h: torch.Tensor,
        dst_h: torch.Tensor,
    ) -> torch.Tensor:
        edge_tokens = torch.stack(
            [
                src_h + self.src_role,
                rel_h + self.rel_role,
                dst_h + self.dst_role,
            ],
            dim=1,
        )
        edge_h = self._manual_attention(
            query=query_h.unsqueeze(1),
            key=edge_tokens,
            value=edge_tokens,
        )
        return self.out(edge_h.squeeze(1))

    def _manual_attention(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q_weight, k_weight, v_weight = self.attn.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = self.attn.in_proj_bias.chunk(3, dim=0)

        q = torch.nn.functional.linear(query, q_weight, q_bias)
        k = torch.nn.functional.linear(key, k_weight, k_bias)
        v = torch.nn.functional.linear(value, v_weight, v_bias)

        batch_size = int(q.size(0))
        head_dim = self.hidden_dim // self.num_heads
        q = q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(head_dim))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.attn.out_proj(out)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.src_role, mean=0.0, std=0.02)
        nn.init.normal_(self.rel_role, mean=0.0, std=0.02)
        nn.init.normal_(self.dst_role, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        for module in self.out:
            if isinstance(module, nn.Linear):
                init_xavier(module)


class QueryStateEncoder(nn.Module):
    """
    Query-conditioned ragged attention over selected evidence-edge tokens and anchors.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.empty_token = nn.Parameter(torch.empty(self.hidden_dim))
        self.anchor_role = nn.Parameter(torch.empty(self.hidden_dim))
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
        query_h: torch.Tensor,
        token_h: torch.Tensor,
        token_row_ids: torch.Tensor,
        num_rows: int,
    ) -> torch.Tensor:
        if token_h.numel() == 0:
            token_h = self.empty_token.unsqueeze(0).expand(num_rows, -1)
            token_row_ids = torch.arange(
                num_rows,
                dtype=torch.long,
                device=query_h.device,
            )

        q = self.q_proj(query_h.index_select(0, token_row_ids))
        k = self.k_proj(token_h)
        v = self.v_proj(token_h)

        scale = math.sqrt(float(self.hidden_dim))
        score = (q * k).sum(dim=-1) / scale
        alpha = segment_softmax(
            score,
            token_row_ids,
            num_segments=int(num_rows),
        )

        pooled = v.new_zeros((num_rows, self.hidden_dim))
        pooled.scatter_add_(
            0,
            token_row_ids[:, None].expand(-1, self.hidden_dim),
            alpha.unsqueeze(-1) * v,
        )
        return self.out(pooled)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.empty_token, mean=0.0, std=0.02)
        nn.init.normal_(self.anchor_role, mean=0.0, std=0.02)
        init_xavier(self.q_proj)
        init_xavier(self.k_proj)
        init_xavier(self.v_proj)
        for module in self.out:
            if isinstance(module, nn.Linear):
                init_xavier(module)


class StateEncoder(nn.Module):
    """
    Encode rollout states from query-conditioned edge and evidence attention.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_encoder: EdgeEncoder | QueryEdgeEncoder | None = None,
        state_encoder: QueryStateEncoder | None = None,
        edge_num_heads: int = 1,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if edge_encoder is None:
            self.edge_encoder = QueryEdgeEncoder(
                hidden_dim=self.hidden_dim,
                num_heads=edge_num_heads,
            )
        elif isinstance(edge_encoder, EdgeEncoder):
            self.edge_encoder = QueryEdgeEncoder(
                hidden_dim=self.hidden_dim,
                num_heads=edge_num_heads,
            )
        else:
            self.edge_encoder = edge_encoder
        self.state_encoder = state_encoder or QueryStateEncoder(hidden_dim=self.hidden_dim)

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
        token_h, token_row_ids = self._state_tokens(
            features=features,
            state=state,
            context=context,
            query_h=query_h,
        )
        row_state_h = self.state_encoder(
            query_h=query_h,
            token_h=token_h,
            token_row_ids=token_row_ids,
            num_rows=num_rows,
        )

        return StateEncoding(
            query_h=query_h,
            row_state_h=row_state_h,
            edge_state_h=row_state_h,
        )

    def encode_edge_tokens(
        self,
        *,
        features: EncodedFeatures,
        src_node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
        query_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_ids = edge_ids.to(dtype=torch.long)
        if edge_ids.numel() == 0:
            return features.query_model.new_empty((0, self.edge_encoder.output_dim))

        if src_node_ids.numel() == 0 or dst_node_ids.numel() == 0:
            src_node_ids = features.edge_relation_model.new_empty((0,), dtype=torch.long)
            dst_node_ids = features.edge_relation_model.new_empty((0,), dtype=torch.long)

        if src_node_ids.numel() == 0:
            raise ValueError("src_node_ids and dst_node_ids must be provided for query-conditioned edge encoding.")
        if int(src_node_ids.numel()) != int(edge_ids.numel()) or int(dst_node_ids.numel()) != int(edge_ids.numel()):
            raise ValueError("src_node_ids, edge_ids, and dst_node_ids must have matching lengths.")

        if query_h is None:
            raise ValueError("query_h is required for query-conditioned edge encoding.")

        return self.edge_encoder(
            query_h=query_h,
            src_h=select_node_model(features, src_node_ids),
            rel_h=select_edge_relation_model(features, edge_ids),
            dst_h=select_node_model(features, dst_node_ids),
        )

    def _state_tokens(
        self,
        *,
        features: EncodedFeatures,
        state: State,
        context: GraphContext,
        query_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_h: list[torch.Tensor] = []
        token_row_ids: list[torch.Tensor] = []

        selected_rows, selected_edge_ids = state.selected_edges()
        if selected_edge_ids.numel() > 0:
            selected_src = context.edge_index[0].index_select(0, selected_edge_ids)
            selected_dst = context.edge_index[1].index_select(0, selected_edge_ids)
            selected_query_h = query_h.index_select(0, selected_rows)
            token_h.append(
                self.encode_edge_tokens(
                    features=features,
                    src_node_ids=selected_src,
                    edge_ids=selected_edge_ids,
                    dst_node_ids=selected_dst,
                    query_h=selected_query_h,
                )
            )
            token_row_ids.append(selected_rows)

        anchor_node_ids = context.anchor_mask.nonzero(as_tuple=True)[0]
        if anchor_node_ids.numel() > 0:
            anchor_graph_ids = context.node_to_graph.index_select(0, anchor_node_ids)
            row_ids = state.graph_ids.view(-1, 1).eq(anchor_graph_ids.view(1, -1)).nonzero(as_tuple=False)[:, 0]
            if row_ids.numel() > 0:
                repeated_anchor_ids = anchor_node_ids.index_select(
                    0,
                    state.graph_ids.view(-1, 1).eq(anchor_graph_ids.view(1, -1)).nonzero(as_tuple=False)[:, 1],
                )
                token_h.append(
                    select_node_model(features, repeated_anchor_ids) + self.state_encoder.anchor_role
                )
                token_row_ids.append(row_ids)

        if token_h:
            return torch.cat(token_h, dim=0), torch.cat(token_row_ids, dim=0)

        num_rows = state.num_rows
        empty_row_ids = torch.arange(num_rows, dtype=torch.long, device=query_h.device)
        empty_h = self.state_encoder.empty_token.unsqueeze(0).expand(num_rows, -1)
        return empty_h, empty_row_ids


__all__ = [
    "QueryEdgeEncoder",
    "QueryStateEncoder",
    "StateEncoder",
    "StateEncoding",
]
