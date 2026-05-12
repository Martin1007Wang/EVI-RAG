from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schema import RetrievalBatch
from src.utils.nn_utils import init_xavier, require_finite, zero_last_linear

from .evidence_state_encoder import StateContext
from .feature_encoder import FeatureBank
from .frontier_context import FrontierContext


@dataclass(frozen=True, slots=True)
class BudgetedSuccessorDiagnostics:
    sem_q_rel: torch.Tensor
    sem_q_dst: torch.Tensor
    terminal_logit: torch.Tensor
    edge_branch_logits: torch.Tensor
    value_logit: torch.Tensor
    dst_text_mask: torch.Tensor


class BudgetedSuccessorPolicy(nn.Module):
    """
    Joint stop/expand scorer on one log-value scale.

    The terminal head predicts T_theta(s).  Each frontier branch predicts
    G_theta(s,e,b) from parent state, successor state, raw PLM semantic scalars,
    and budget/structure features.  The state value is
    logsumexp(T_theta(s), {G_theta(s,e,b)}).
    """

    scalar_dim = 14

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_budget: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        zero_init: bool = True,
        terminal_bias_init: float = 0.0,
        edge_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_budget = int(max_budget)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if self.max_budget < 0:
            raise ValueError(f"max_budget must be non-negative, got {max_budget}.")
        num_layers = int(num_layers)
        if num_layers not in {1, 2, 3}:
            raise ValueError("num_layers must be one of {1,2,3}.")

        self.budget_embedding = nn.Embedding(self.max_budget + 1, self.hidden_dim)
        self.terminal = _mlp(
            input_dim=self.hidden_dim * 2,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=float(dropout),
        )
        self.edge = _mlp(
            input_dim=self.hidden_dim * 5 + self.scalar_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=float(dropout),
        )
        self._reset_parameters()
        if zero_init:
            zero_last_linear(self.terminal)
            zero_last_linear(self.edge)
        _set_last_bias(self.terminal, float(terminal_bias_init))
        _set_last_bias(self.edge, float(edge_bias_init))

    def terminal_logit(
        self,
        *,
        state_h: torch.Tensor,
        remaining_budget: torch.Tensor,
    ) -> torch.Tensor:
        state_h = require_finite(state_h, name="state_h")
        budget_h = self._budget_h(
            remaining_budget=remaining_budget,
            rows=int(state_h.size(0)),
            device=state_h.device,
            dtype=state_h.dtype,
        )
        return self.terminal(torch.cat([state_h, budget_h], dim=-1)).squeeze(-1)

    def edge_branch_logits(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        context: StateContext,
        child_state_h: torch.Tensor,
        frontier: FrontierContext,
        frontier_batch_ids: torch.Tensor,
        depth: torch.Tensor,
        remaining_budget: torch.Tensor,
        frontier_size: torch.Tensor,
    ) -> torch.Tensor:
        device = context.state_h.device
        dtype = context.state_h.dtype
        edge_ids = frontier.edge_ids.to(device=device, dtype=torch.long).view(-1)
        row_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
        if edge_ids.numel() == 0:
            return context.state_h.new_empty((0,))
        if edge_ids.shape != row_ids.shape:
            raise ValueError("frontier ids and row ids must have matching shape.")
        if child_state_h.shape != (edge_ids.numel(), self.hidden_dim):
            raise ValueError(
                "child_state_h must have shape "
                f"({edge_ids.numel()}, {self.hidden_dim}), got {tuple(child_state_h.shape)}."
            )

        state_h = require_finite(context.state_h, name="context.state_h").to(
            device=device,
            dtype=dtype,
        )
        query_h = require_finite(context.query_h, name="context.query_h").to(
            device=device,
            dtype=dtype,
        )
        edge_h = require_finite(fb.edge_h, name="fb.edge_h").to(
            device=device,
            dtype=dtype,
        ).index_select(0, edge_ids)
        budget_h = self._budget_h(
            remaining_budget=remaining_budget,
            rows=int(state_h.size(0)),
            device=device,
            dtype=dtype,
        ).index_select(0, row_ids)

        scalar_features = self._scalar_features(
            fb=fb,
            batch=batch,
            frontier=frontier,
            edge_ids=edge_ids,
            row_ids=row_ids,
            state_rows=int(state_h.size(0)),
            depth=depth,
            remaining_budget=remaining_budget,
            frontier_size=frontier_size,
            dtype=dtype,
            device=device,
        )
        parts = [
            state_h.index_select(0, row_ids),
            child_state_h.to(device=device, dtype=dtype),
            edge_h,
            query_h.index_select(0, row_ids),
            budget_h,
            scalar_features,
        ]
        return self.edge(torch.cat(parts, dim=-1)).squeeze(-1)

    def forward(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        context: StateContext,
        child_state_h: torch.Tensor,
        frontier: FrontierContext,
        frontier_batch_ids: torch.Tensor,
        depth: torch.Tensor,
        remaining_budget: torch.Tensor,
        frontier_size: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        terminal = self.terminal_logit(
            state_h=context.state_h,
            remaining_budget=remaining_budget,
        )
        edges = self.edge_branch_logits(
            fb=fb,
            batch=batch,
            context=context,
            child_state_h=child_state_h,
            frontier=frontier,
            frontier_batch_ids=frontier_batch_ids,
            depth=depth,
            remaining_budget=remaining_budget,
            frontier_size=frontier_size,
        )
        value = _joint_value(
            terminal=terminal,
            edge_branch_logits=edges,
            frontier_batch_ids=frontier_batch_ids,
            remaining_budget=remaining_budget,
            num_graphs=int(num_graphs),
        )
        return terminal, edges, value

    def diagnostics(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        context: StateContext,
        child_state_h: torch.Tensor,
        frontier: FrontierContext,
        frontier_batch_ids: torch.Tensor,
        depth: torch.Tensor,
        remaining_budget: torch.Tensor,
        frontier_size: torch.Tensor,
        num_graphs: int,
    ) -> BudgetedSuccessorDiagnostics:
        terminal, edges, value = self(
            fb=fb,
            batch=batch,
            context=context,
            child_state_h=child_state_h,
            frontier=frontier,
            frontier_batch_ids=frontier_batch_ids,
            depth=depth,
            remaining_budget=remaining_budget,
            frontier_size=frontier_size,
            num_graphs=num_graphs,
        )
        sem_q_rel, sem_q_dst, dst_text_mask = _semantic_scalars(
            fb=fb,
            frontier=frontier,
            edge_ids=frontier.edge_ids.to(device=fb.node_h.device, dtype=torch.long),
            dtype=fb.node_h.dtype,
            device=fb.node_h.device,
        )
        return BudgetedSuccessorDiagnostics(
            sem_q_rel=sem_q_rel.detach(),
            sem_q_dst=sem_q_dst.detach(),
            terminal_logit=terminal.detach(),
            edge_branch_logits=edges.detach(),
            value_logit=value.detach(),
            dst_text_mask=dst_text_mask.detach(),
        )

    def _budget_h(
        self,
        *,
        remaining_budget: torch.Tensor,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        budget = remaining_budget.to(device=device, dtype=torch.long).view(-1)
        if budget.shape != (int(rows),):
            raise ValueError(
                f"remaining_budget must have shape [{rows}], got {tuple(budget.shape)}."
            )
        budget = budget.clamp(0, self.max_budget)
        return self.budget_embedding(budget).to(device=device, dtype=dtype)

    def _scalar_features(
        self,
        *,
        fb: FeatureBank,
        batch: RetrievalBatch,
        frontier: FrontierContext,
        edge_ids: torch.Tensor,
        row_ids: torch.Tensor,
        state_rows: int,
        depth: torch.Tensor,
        remaining_budget: torch.Tensor,
        frontier_size: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        src = frontier.src.to(device=device, dtype=torch.long).view(-1)
        dst = frontier.dst.to(device=device, dtype=torch.long).view(-1)
        sem_q_rel, sem_q_dst, dst_text_mask = _semantic_scalars(
            fb=fb,
            frontier=frontier,
            edge_ids=edge_ids,
            dtype=dtype,
            device=device,
        )
        depth = depth.to(device=device, dtype=dtype).view(-1)
        remaining = remaining_budget.to(device=device, dtype=dtype).view(-1)
        frontier_size = frontier_size.to(device=device, dtype=dtype).view(-1)
        if depth.numel() != state_rows or remaining.numel() != state_rows:
            raise ValueError("depth and remaining_budget must match state rows.")
        if frontier_size.numel() != state_rows:
            raise ValueError("frontier_size must match state rows.")

        anchor_mask = torch.zeros(int(fb.node_h.size(0)), dtype=torch.bool, device=device)
        anchors = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
        if anchors.numel() > 0:
            valid = anchors.ge(0) & anchors.lt(anchor_mask.numel())
            anchor_mask[anchors[valid]] = True
        src_is_anchor = anchor_mask.index_select(0, src).to(dtype=dtype)
        node_degree = (
            fb.node_log_degree.to(device=device, dtype=dtype)
            if fb.node_log_degree is not None
            else torch.zeros((int(fb.node_h.size(0)),), dtype=dtype, device=device)
        )
        rel_freq = (
            fb.edge_relation_log_frequency.to(device=device, dtype=dtype)
            if fb.edge_relation_log_frequency is not None
            else torch.zeros((int(fb.edge_h.size(0)),), dtype=dtype, device=device)
        )
        dst_nontext = (
            fb.node_is_non_text.to(device=device, dtype=torch.bool).index_select(0, dst)
            if fb.node_is_non_text is not None
            else ~dst_text_mask
        ).to(dtype=dtype)

        denom = float(max(self.max_budget, 1))
        return torch.stack(
            [
                sem_q_rel.detach(),
                sem_q_dst.detach(),
                depth.index_select(0, row_ids) / denom,
                remaining.index_select(0, row_ids) / denom,
                frontier_size.clamp_min(1).log().index_select(0, row_ids),
                node_degree.index_select(0, src),
                node_degree.index_select(0, dst),
                rel_freq.index_select(0, edge_ids),
                src_is_anchor,
                dst_nontext,
                frontier.src_active.to(device=device, dtype=dtype).view(-1),
                frontier.dst_active.to(device=device, dtype=dtype).view(-1),
                remaining.index_select(0, row_ids).eq(1).to(dtype=dtype),
                remaining.index_select(0, row_ids).gt(1).to(dtype=dtype),
            ],
            dim=-1,
        )

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.budget_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_xavier(module)


def _semantic_scalars(
    *,
    fb: FeatureBank,
    frontier: FrontierContext,
    edge_ids: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_ids.numel() == 0:
        empty = torch.empty((0,), dtype=dtype, device=device)
        return empty, empty, torch.empty((0,), dtype=torch.bool, device=device)
    static_graph_id = (
        frontier.static_graph_id.to(device=device, dtype=torch.long).view(-1)
        if frontier.static_graph_id is not None
        else frontier.graph_id.to(device=device, dtype=torch.long).view(-1)
    )
    q_sem = F.normalize(fb.query_sem_h.to(device=device, dtype=dtype).index_select(0, static_graph_id), p=2, dim=-1)
    rel_sem = F.normalize(fb.rel_sem_h.to(device=device, dtype=dtype).index_select(0, edge_ids), p=2, dim=-1)
    sem_q_rel = (q_sem * rel_sem).sum(dim=-1)
    dst = frontier.dst.to(device=device, dtype=torch.long).view(-1)
    if fb.node_text_row_ids is None or fb.entity_text_sem_h is None:
        return sem_q_rel, sem_q_rel.new_zeros(sem_q_rel.shape), torch.zeros_like(sem_q_rel, dtype=torch.bool)
    text_rows = fb.node_text_row_ids.to(device=device, dtype=torch.long).index_select(0, dst)
    dst_text_mask = text_rows.ge(0)
    sem_q_dst = sem_q_rel.new_zeros(sem_q_rel.shape)
    if bool(dst_text_mask.any()):
        pos = dst_text_mask.nonzero(as_tuple=False).flatten()
        dst_sem = F.normalize(fb.entity_text_sem_h.to(device=device, dtype=dtype).index_select(0, text_rows.index_select(0, pos)), p=2, dim=-1)
        sem_q_dst[pos] = (q_sem.index_select(0, pos) * dst_sem).sum(dim=-1)
    return sem_q_rel, sem_q_dst, dst_text_mask


def _joint_value(
    *,
    terminal: torch.Tensor,
    edge_branch_logits: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    remaining_budget: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    from src.graph.segments import segment_logsumexp

    edge_log_z = segment_logsumexp(
        values=edge_branch_logits,
        segment_ids=frontier_batch_ids,
        num_segments=int(num_graphs),
    )
    can_expand = remaining_budget.to(
        device=terminal.device,
        dtype=torch.long,
    ).view(int(num_graphs)).gt(0) & torch.isfinite(edge_log_z)
    value = torch.logaddexp(terminal.view(int(num_graphs)), edge_log_z)
    return torch.where(can_expand, value, terminal.view(int(num_graphs)))


def _mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if int(num_layers) == 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)]
    for _ in range(int(num_layers) - 2):
        layers.extend([nn.Dropout(float(dropout)), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _set_last_bias(module: nn.Module, value: float) -> None:
    for layer in reversed(tuple(module.modules())):
        if isinstance(layer, nn.Linear):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, float(value))
            return
    raise TypeError("No nn.Linear found.")


__all__ = ["BudgetedSuccessorDiagnostics", "BudgetedSuccessorPolicy"]
