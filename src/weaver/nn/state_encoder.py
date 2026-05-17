from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.nn_utils import init_xavier
from src.weaver.context import GraphContext
from src.weaver.state import State, derive_remaining_budget

from .feature_encoder import FeatureBank


@dataclass(frozen=True, slots=True)
class StateEncoding:
    """
    Per-rollout state representation.

    state_h:
        evidence-state vector h_z, shape [R, H].

    query_h:
        original query vector h_q aligned to rollout rows, shape [R, H].
    """

    state_h: torch.Tensor
    query_h: torch.Tensor


class StateEncoder(nn.Module):
    """
    Permutation-invariant encoder for z = (V_z, E_z, b_z).

    It encodes the current evidence state only.
    It does not score frontier edges.
    It does not compute stop flow.
    It does not perform query-edge matching.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_budget: int = 8,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.max_budget = int(max_budget)
        if self.max_budget < 0:
            raise ValueError(f"max_budget must be non-negative, got {max_budget}.")

        self.node_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.budget_proj = nn.Linear(3, self.hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

        self._reset_parameters()

    def forward(
        self,
        *,
        features: FeatureBank,
        context: GraphContext,
        state: State,
    ) -> StateEncoding:
        del context

        device = features.node_h.device
        dtype = features.node_h.dtype

        num_rows = int(state.num_rollouts)

        node_summary = self._node_summary(
            features=features,
            state=state,
            num_rows=num_rows,
            device=device,
            dtype=dtype,
        )

        edge_summary = self._edge_summary(
            features=features,
            state=state,
            num_rows=num_rows,
            device=device,
            dtype=dtype,
        )

        budget_feat = self._budget_features(
            state=state,
            device=device,
            dtype=dtype,
        )

        state_h = self.out_norm(self.node_proj(node_summary) + self.edge_proj(edge_summary) + self.budget_proj(budget_feat))

        query_h = features.query_h.index_select(
            0,
            state.row_to_graph.to(device=device, dtype=torch.long),
        )

        return StateEncoding(
            state_h=state_h,
            query_h=query_h.to(device=device, dtype=dtype),
        )

    def _node_summary(
        self,
        *,
        features: FeatureBank,
        state: State,
        num_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows, node_ids = state.active_node_trace_rows()
        rows = rows.to(device=device, dtype=torch.long)
        node_ids = node_ids.to(device=device, dtype=torch.long)

        if node_ids.numel() == 0:
            return torch.zeros((num_rows, self.hidden_dim), device=device, dtype=dtype)

        node_h = features.node_h.to(device=device, dtype=dtype).index_select(0, node_ids)
        return _segment_mean(
            values=node_h,
            row_ids=rows,
            num_rows=num_rows,
        )

    def _edge_summary(
        self,
        *,
        features: FeatureBank,
        state: State,
        num_rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows, edge_ids = state.active_edge_trace_rows()
        rows = rows.to(device=device, dtype=torch.long)
        edge_ids = edge_ids.to(device=device, dtype=torch.long)

        if edge_ids.numel() == 0:
            return torch.zeros((num_rows, self.hidden_dim), device=device, dtype=dtype)

        edge_h = features.edge_h.to(device=device, dtype=dtype).index_select(0, edge_ids)
        return _segment_mean(
            values=edge_h,
            row_ids=rows,
            num_rows=num_rows,
        )

    def _budget_features(
        self,
        *,
        state: State,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        remaining = derive_remaining_budget(state).to(device=device, dtype=dtype).view(-1)
        max_budget = max(float(self.max_budget), 1.0)

        depth = (max_budget - remaining).clamp_min(0.0)
        remaining_ratio = remaining / max_budget
        depth_ratio = depth / max_budget
        exhausted = remaining.le(0.0).to(dtype=dtype)

        return torch.stack(
            [remaining_ratio, depth_ratio, exhausted],
            dim=-1,
        )

    def _reset_parameters(self) -> None:
        init_xavier(self.node_proj)
        init_xavier(self.edge_proj)
        init_xavier(self.budget_proj)


def _segment_mean(
    *,
    values: torch.Tensor,
    row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    out = values.new_zeros((int(num_rows), int(values.size(-1))))
    count = values.new_zeros((int(num_rows), 1))

    out.scatter_add_(
        0,
        row_ids.view(-1, 1).expand(-1, values.size(-1)),
        values,
    )

    count.scatter_add_(
        0,
        row_ids.view(-1, 1),
        values.new_ones((values.size(0), 1)),
    )

    return out / count.clamp_min(1.0)


__all__ = ["StateEncoding", "StateEncoder"]
