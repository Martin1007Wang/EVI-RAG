from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.graph.segments import segment_log_softmax, segment_logsumexp
from src.weaver.context import GraphContext
from src.weaver.state import FrontierEncoding, StateBatch, root_reachable_mask_from_edges

from .edge_scorer import QuestionConditionedEdgeScorer
from .output import STOP_EDGE_ID

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class BackwardPolicyOutput:
    action_logits: Tensor
    action_row_ids: Tensor
    action_edge_ids: Tensor
    num_states: int

    @classmethod
    def stop_action(cls, *, num_states: int, device: torch.device) -> BackwardPolicyOutput:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return cls(
            action_logits=empty.to(dtype=torch.float32),
            action_row_ids=empty,
            action_edge_ids=empty,
            num_states=int(num_states),
        )

    @property
    def action_log_prob(self) -> Tensor:
        return segment_log_softmax(
            self.action_logits,
            self.action_row_ids,
            num_segments=int(self.num_states),
        )

    @property
    def log_partition(self) -> Tensor:
        return segment_logsumexp(
            values=self.action_logits,
            segment_ids=self.action_row_ids,
            num_segments=int(self.num_states),
        )

    def gather_log_prob(self, *, row_ids: Tensor, edge_ids: Tensor) -> Tensor:
        row_ids = row_ids.to(device=self.action_row_ids.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.action_edge_ids.device, dtype=torch.long).view(-1)
        if int(row_ids.numel()) != int(edge_ids.numel()):
            raise ValueError("row_ids and edge_ids must have the same length.")
        if int(row_ids.numel()) == 0:
            return self.action_logits.new_empty((0,), dtype=torch.float32)
        if bool(row_ids.lt(0).any()) or bool(row_ids.ge(int(self.num_states)).any()):
            raise ValueError("requested backward action row must be in range.")

        stop = edge_ids.eq(int(STOP_EDGE_ID))
        out = self.action_logits.new_zeros((int(row_ids.numel()),), dtype=torch.float32)
        if bool((~stop).any()):
            positions = _find_removable_actions(
                action_row_ids=self.action_row_ids,
                action_edge_ids=self.action_edge_ids,
                num_states=int(self.num_states),
                row_ids=row_ids[~stop],
                edge_ids=edge_ids[~stop],
            )
            out[~stop] = self.action_log_prob.index_select(0, positions)
        return out


class BackwardPolicy(nn.Module):
    """Parameterized P_B(parent | child) over removable child edges."""

    def __init__(self, *, hidden_dim: int, relation_lambda: float = 0.5) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.edge_scorer = QuestionConditionedEdgeScorer(
            hidden_dim=hidden_dim,
            relation_lambda=relation_lambda,
        )

    def forward(
        self,
        *,
        child_state_h: Tensor,
        question_h_by_graph: Tensor,
        edge_h: Tensor,
        relation_h: Tensor,
        removable: FrontierEncoding,
    ) -> BackwardPolicyOutput:
        logits = self.score_edges(
            child_state_h=child_state_h,
            question_h_by_graph=question_h_by_graph,
            edge_h=edge_h,
            relation_h=relation_h,
            removable=removable,
        )
        return BackwardPolicyOutput(
            action_logits=logits.float(),
            action_row_ids=removable.row_ids,
            action_edge_ids=removable.edge_ids,
            num_states=int(child_state_h.size(0)),
        )

    def score_edges(
        self,
        *,
        child_state_h: Tensor,
        question_h_by_graph: Tensor,
        edge_h: Tensor,
        relation_h: Tensor,
        removable: FrontierEncoding,
    ) -> Tensor:
        if int(removable.edge_ids.numel()) == 0:
            return child_state_h.new_empty((0,), dtype=torch.float32)
        state_part = child_state_h.index_select(0, removable.row_ids)
        edge_part = edge_h.index_select(0, removable.edge_ids)
        relation_part = relation_h.index_select(0, removable.edge_ids)
        question_part = question_h_by_graph.index_select(0, removable.graph_ids)
        phi_align = self.edge_scorer.score_alignment(
            question_h=question_part,
            edge_h=edge_part,
            relation_h=relation_part,
        )
        phi_state = self.edge_scorer.score_state(state_h=state_part, edge_h=edge_part)
        return phi_align + phi_state


class UniformBackwardPolicy(nn.Module):
    """Uniform P_B(parent | child) over removable child edges."""

    def __init__(self, *, hidden_dim: object = None, relation_lambda: object = None) -> None:
        super().__init__()
        del hidden_dim, relation_lambda

    def forward(
        self,
        *,
        child_state_h: Tensor,
        question_h_by_graph: Tensor,
        edge_h: Tensor,
        relation_h: Tensor,
        removable: FrontierEncoding,
    ) -> BackwardPolicyOutput:
        del question_h_by_graph, edge_h, relation_h
        return BackwardPolicyOutput(
            action_logits=child_state_h.new_zeros((int(removable.edge_ids.numel()),), dtype=torch.float32),
            action_row_ids=removable.row_ids,
            action_edge_ids=removable.edge_ids,
            num_states=int(child_state_h.size(0)),
        )


def removable_edges(
    *,
    child_state: StateBatch,
    graph_context: GraphContext,
) -> FrontierEncoding:
    """
    Enumerate legally removable edges for the backward policy.

    An edge e is removable from child state z' iff the induced parent
    z = z' \\ {e} is a valid predecessor under the forward policy:
      1. z remains anchor/root-reachable.
      2. e.src is active in z, matching the forward frontier condition.

    The second check intentionally uses e.src, not e.dst. Removing a leaf
    edge is a legal backward step even when e.dst is no longer active in
    the parent state.
    """
    width = int(child_state.edge_capacity)
    valid = torch.arange(width, device=child_state.device).view(1, -1).lt(child_state.edge_count.view(-1, 1))
    child_rows, remove_pos = valid.nonzero(as_tuple=True)
    if int(child_rows.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=child_state.device)
        return FrontierEncoding(row_ids=empty, edge_ids=empty, graph_ids=empty)

    removed_edges = child_state.edge_ids[child_rows, remove_pos]
    parent_edges = child_state.edge_ids.index_select(0, child_rows).clone()
    parent_edges[torch.arange(int(child_rows.numel()), device=child_state.device), remove_pos] = -1
    sentinel = max(int(graph_context.num_edges), 1)
    parent_edges = torch.sort(torch.where(parent_edges.lt(0), sentinel, parent_edges), dim=1).values
    parent_edges = torch.where(parent_edges.eq(sentinel), -1, parent_edges)
    parent_count = child_state.edge_count.index_select(0, child_rows) - 1
    parent_graph_ids = child_state.graph_ids.index_select(0, child_rows)
    parent = StateBatch(
        graph_ids=parent_graph_ids,
        edge_ids=parent_edges,
        edge_count=parent_count,
    )

    root_reachable = root_reachable_mask_from_edges(
        edge_ids=parent_edges,
        edge_count=parent_count,
        graph=graph_context,
    )
    active = parent.active_node_index(graph_context)
    node_span = max(int(graph_context.num_nodes), 1)
    active_keys = active.row_ids * node_span + active.node_ids
    removed_src = graph_context.edge_src.index_select(0, removed_edges)
    request_keys = torch.arange(int(child_rows.numel()), device=child_state.device) * node_span + removed_src
    legal = root_reachable & torch.isin(request_keys, active_keys)
    return FrontierEncoding(
        row_ids=child_rows[legal],
        edge_ids=removed_edges[legal],
        graph_ids=parent_graph_ids[legal],
    )


def legal_predecessor_count(
    *,
    child_state: StateBatch,
    graph_context: GraphContext,
) -> Tensor:
    removable = removable_edges(
        child_state=child_state,
        graph_context=graph_context,
    )
    counts = torch.zeros(child_state.num_states, dtype=torch.long, device=child_state.device)
    if int(removable.row_ids.numel()) == 0:
        return counts
    legal = torch.ones_like(removable.row_ids, dtype=torch.long)
    return counts.scatter_add_(0, removable.row_ids, legal)


def uniform_backward_log_prob(
    *,
    child_state: StateBatch,
    graph_context: GraphContext,
) -> Tensor:
    counts = legal_predecessor_count(
        child_state=child_state,
        graph_context=graph_context,
    )
    if bool(counts.le(0).any()):
        raise ValueError("Every child state must have a legal root-reachable predecessor.")
    return -torch.log(counts.float())


__all__ = [
    "BackwardPolicy",
    "BackwardPolicyOutput",
    "UniformBackwardPolicy",
    "legal_predecessor_count",
    "removable_edges",
    "uniform_backward_log_prob",
]


def _find_removable_actions(
    *,
    action_row_ids: Tensor,
    action_edge_ids: Tensor,
    num_states: int,
    row_ids: Tensor,
    edge_ids: Tensor,
) -> Tensor:
    row_ids = row_ids.to(device=action_row_ids.device, dtype=torch.long).view(-1)
    edge_ids = edge_ids.to(device=action_edge_ids.device, dtype=torch.long).view(-1)
    if int(row_ids.numel()) != int(edge_ids.numel()):
        raise ValueError("row_ids and edge_ids must have the same length.")
    if int(row_ids.numel()) == 0:
        return row_ids
    if bool(row_ids.lt(0).any()) or bool(row_ids.ge(int(num_states)).any()):
        raise ValueError("requested backward action row must be in range.")
    if bool(edge_ids.lt(0).any()):
        raise ValueError("requested backward edge id must be nonnegative.")
    edge_span = max(
        int(action_edge_ids.max().item()) + 1 if int(action_edge_ids.numel()) else 0,
        int(edge_ids.max().item()) + 1 if int(edge_ids.numel()) else 0,
        1,
    )
    action_keys = action_row_ids * edge_span + action_edge_ids
    requested_keys = row_ids * edge_span + edge_ids
    order = torch.argsort(action_keys)
    sorted_keys = action_keys.index_select(0, order)
    positions = torch.searchsorted(sorted_keys, requested_keys)
    in_range = positions.lt(int(sorted_keys.numel()))
    clamped = positions.clamp_max(max(int(sorted_keys.numel()) - 1, 0))
    legal = torch.zeros_like(in_range)
    legal[in_range] = sorted_keys.index_select(0, clamped[in_range]).eq(requested_keys[in_range])
    if not bool(legal.all()):
        raise ValueError("requested backward action must be uniquely removable.")
    return order.index_select(0, positions)
