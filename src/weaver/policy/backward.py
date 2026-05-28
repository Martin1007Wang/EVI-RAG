from __future__ import annotations

import math

import torch

from src.weaver.context import GraphContext
from src.weaver.state import (
    StateBatch,
    remove_selected_edge,
    state_rows_equal,
)

Tensor = torch.Tensor


def canonical_backward_log_prob(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    action_edge_ids: Tensor,
    graph_context: GraphContext,
    validate: bool = False,
) -> Tensor:
    """
    Uniform backward kernel over valid canonical predecessor states.

    For canonical edge-set states, a child can have multiple legal parents:
    remove any selected edge whose removal leaves an anchor-reachable state and
    whose edge is legal in that parent's forward frontier.
    """

    action_edge_ids = action_edge_ids.to(
        device=child_state.device,
        dtype=torch.long,
    ).view(-1)

    _check_batch_shapes(
        parent_state=parent_state,
        child_state=child_state,
        action_edge_ids=action_edge_ids,
    )

    out = torch.empty(
        int(action_edge_ids.numel()),
        dtype=torch.float32,
        device=child_state.device,
    )

    for row in range(int(action_edge_ids.numel())):
        parents = valid_predecessor_count(
            child_state=child_state,
            graph_context=graph_context,
            row=row,
        )
        if parents <= 0:
            raise ValueError("child_state has no valid canonical predecessors.")
        out[row] = -math.log(float(parents))

    if validate:
        validate_canonical_transitions(
            parent_state=parent_state,
            child_state=child_state,
            action_edge_ids=action_edge_ids,
            graph_context=graph_context,
        )

    return out


def deterministic_backward_log_prob(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    action_edge_ids: Tensor,
    graph_context: GraphContext,
    validate: bool = False,
) -> Tensor:
    """
    Backward-compatible name for the canonical uniform backward kernel.
    """

    return canonical_backward_log_prob(
        parent_state=parent_state,
        child_state=child_state,
        action_edge_ids=action_edge_ids,
        graph_context=graph_context,
        validate=validate,
    )


def valid_predecessor_count(
    *,
    child_state: StateBatch,
    graph_context: GraphContext,
    row: int,
) -> int:
    count = int(child_state.edge_count[int(row)].item())
    if count <= 0:
        return 0

    total = 0
    selected = child_state.edge_ids[int(row), :count]
    for edge_id in selected.tolist():
        parent = remove_selected_edge(
            state=child_state,
            row=int(row),
            edge_id=int(edge_id),
        )
        if _edge_is_forward_legal(
            parent=parent,
            edge_id=int(edge_id),
            graph_context=graph_context,
        ):
            total += 1
    return total


def validate_canonical_transitions(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    action_edge_ids: Tensor,
    graph_context: GraphContext,
) -> None:
    _check_batch_shapes(
        parent_state=parent_state,
        child_state=child_state,
        action_edge_ids=action_edge_ids,
    )
    for row, action_edge_id in enumerate(action_edge_ids.tolist()):
        action_edge_id = int(action_edge_id)
        if action_edge_id < 0:
            _validate_terminal_row(
                parent_state=parent_state,
                child_state=child_state,
                row=row,
            )
            continue

        expected_parent = remove_selected_edge(
            state=child_state,
            row=row,
            edge_id=action_edge_id,
        )
        if not state_rows_equal(
            left=parent_state,
            left_row=row,
            right=expected_parent,
            right_row=0,
        ):
            raise ValueError("Expansion parent must equal child minus action edge.")
        if not _edge_is_forward_legal(
            parent=expected_parent,
            edge_id=action_edge_id,
            graph_context=graph_context,
        ):
            raise ValueError("Expansion action is not legal in the canonical parent frontier.")


def _check_batch_shapes(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    action_edge_ids: Tensor,
) -> None:
    if int(parent_state.num_states) != int(child_state.num_states):
        raise ValueError("parent_state and child_state must have the same num_states.")
    if int(action_edge_ids.numel()) != int(child_state.num_states):
        raise ValueError("action_edge_ids must match state rows.")
    if int(parent_state.budget) != int(child_state.budget):
        raise ValueError("parent_state and child_state must have the same budget.")
    if not bool(parent_state.graph_ids.eq(child_state.graph_ids).all()):
        raise ValueError("parent_state and child_state must have identical graph_ids.")


def _validate_terminal_row(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    row: int,
) -> None:
    if not state_rows_equal(
        left=parent_state,
        left_row=int(row),
        right=child_state,
        right_row=int(row),
    ):
        raise ValueError("Terminal rows must not change selected edge set.")


def _edge_is_forward_legal(
    *,
    parent: StateBatch,
    edge_id: int,
    graph_context: GraphContext,
) -> bool:
    action_space = parent.action_space(graph_context)
    if int(action_space.num_expansions) == 0:
        return False
    return bool(action_space.expand_edge_ids.eq(int(edge_id)).any())


__all__ = [
    "canonical_backward_log_prob",
    "deterministic_backward_log_prob",
    "valid_predecessor_count",
    "validate_canonical_transitions",
]
