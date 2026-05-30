from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.graph.segments import segment_softmax
from src.weaver.feature import FeaturePack
from src.weaver.state import FrontierEncoding, StateBatch


@dataclass(frozen=True, slots=True)
class PolicyCache:
    edge_h: torch.Tensor  # [E, H]
    node_h: torch.Tensor  # [N, H]
    query_base_h_by_graph: torch.Tensor  # [G, H]


@dataclass(frozen=True, slots=True)
class StateEncoding:
    state_selected_h: torch.Tensor  # [S, H], float32

    @property
    def state_h(self) -> torch.Tensor:
        return self.state_selected_h


class PolicyCacheBuilder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, features: FeaturePack) -> PolicyCache:
        return PolicyCache(
            edge_h=features.edge_h,
            node_h=features.node_h,
            query_base_h_by_graph=features.query_h,
        )


class MultiHeadQueryPool(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads. "
                f"Got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )
        self.head_dim = self.hidden_dim // self.num_heads
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.x_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.score = nn.Linear(self.head_dim, 1, bias=False)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(
        self,
        *,
        query_h: torch.Tensor,
        tokens: torch.Tensor,
        row_ids: torch.Tensor,
        num_rows: int,
    ) -> torch.Tensor:
        query_h = query_h.float()
        if int(tokens.numel()) == 0:
            return torch.zeros(
                (num_rows, self.hidden_dim),
                dtype=torch.float32,
                device=query_h.device,
            )

        row_ids = row_ids.to(device=query_h.device, dtype=torch.long).view(-1)
        pooled_tokens = tokens.float()
        q = self.q_proj(query_h).view(num_rows, self.num_heads, self.head_dim)
        x = self.x_proj(pooled_tokens).view(-1, self.num_heads, self.head_dim)
        q = q.index_select(0, row_ids)
        logits = self.score(torch.nn.functional.silu(q + x)).squeeze(-1)

        head_ids = torch.arange(
            self.num_heads,
            device=row_ids.device,
            dtype=torch.long,
        ).view(1, -1)
        segment_ids = row_ids.view(-1, 1) * self.num_heads + head_ids
        weights = segment_softmax(
            logits.view(-1),
            segment_ids=segment_ids.view(-1),
            num_segments=num_rows * self.num_heads,
        ).view(-1, self.num_heads, 1)

        out = torch.zeros(
            (num_rows, self.hidden_dim),
            dtype=torch.float32,
            device=query_h.device,
        )
        weighted = (
            pooled_tokens.view(-1, self.num_heads, self.head_dim) * weights
        ).view(-1, self.hidden_dim)
        out.scatter_add_(
            0,
            row_ids.view(-1, 1).expand(-1, self.hidden_dim),
            weighted,
        )
        return self.out(out)


class StateEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        max_budget: int,
    ) -> None:
        super().__init__()
        del max_budget
        self.query_pool = MultiHeadQueryPool(hidden_dim=hidden_dim)

    def forward(
        self,
        *,
        state: StateBatch,
        cache: PolicyCache,
        frontier: FrontierEncoding,
        remaining_budget: torch.Tensor,
        features: FeaturePack | None = None,
    ) -> StateEncoding:
        del features, frontier, remaining_budget
        num_states = int(state.num_states)
        hidden_dim = int(cache.edge_h.size(-1))
        if num_states == 0:
            return StateEncoding(
                state_selected_h=torch.empty(
                    0,
                    hidden_dim,
                    dtype=torch.float32,
                    device=cache.edge_h.device,
                )
            )

        query_base_h = cache.query_base_h_by_graph.index_select(0, state.graph_ids).float()
        selected = state.selected_edge_index()
        selected_h = self.query_pool(
            query_h=query_base_h,
            tokens=cache.edge_h.index_select(0, selected.edge_ids),
            row_ids=selected.row_ids,
            num_rows=num_states,
        )
        has_selected = state.selected_edge_count.gt(0).view(-1, 1)
        state_selected_h = torch.where(
            has_selected,
            selected_h,
            query_base_h,
        )
        return StateEncoding(state_selected_h=state_selected_h)

__all__ = [
    "MultiHeadQueryPool",
    "PolicyCache",
    "PolicyCacheBuilder",
    "StateEncoder",
    "StateEncoding",
]
