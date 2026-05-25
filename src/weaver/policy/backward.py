from __future__ import annotations

import torch

from src.weaver.state import StateBatch

Tensor = torch.Tensor


def deterministic_backward_log_prob(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    action_edge_ids: Tensor,
    validate: bool = False,
) -> Tensor:
    """
    Deterministic backward kernel for ordered-prefix states.

    Forward expansion:

        parent.edge_ids[row, :k]
        -- action e -->
        child.edge_ids[row, :k + 1]

    Since StateBatch stores ordered prefixes, every expansion child has exactly
    one predecessor: remove the last appended edge.

    Therefore:

        log P_B(parent | child) = 0

    This function:
    - has no parameters;
    - is not used for rollout inference;
    - does not read rewards;
    - does not enumerate removable edges;
    - does not model unordered-subgraph predecessors.
    """

    action_edge_ids = action_edge_ids.to(
        device=child_state.device,
        dtype=torch.long,
    ).view(-1)

    if validate:
        validate_ordered_prefix_transitions(
            parent_state=parent_state,
            child_state=child_state,
            action_edge_ids=action_edge_ids,
        )

    return torch.zeros(
        int(action_edge_ids.numel()),
        dtype=torch.float32,
        device=child_state.device,
    )


def validate_ordered_prefix_transitions(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    action_edge_ids: Tensor,
) -> None:
    """
    Debug-only validation.

    Expansion rows must satisfy:

        child = parent + action_edge_id

    Terminal diagnostic rows may use negative action ids and must not change
    the prefix.
    """

    action_edge_ids = action_edge_ids.to(
        device=child_state.device,
        dtype=torch.long,
    ).view(-1)

    if int(parent_state.num_states) != int(child_state.num_states):
        raise ValueError("parent_state and child_state must have the same num_states.")

    if int(action_edge_ids.numel()) != int(child_state.num_states):
        raise ValueError("action_edge_ids must match state rows.")

    if int(parent_state.budget) != int(child_state.budget):
        raise ValueError("parent_state and child_state must have the same budget.")

    if not bool(parent_state.graph_ids.eq(child_state.graph_ids).all()):
        raise ValueError("parent_state and child_state must have identical graph_ids.")

    expand_rows = action_edge_ids.ge(0).nonzero(as_tuple=False).flatten()
    terminal_rows = action_edge_ids.lt(0).nonzero(as_tuple=False).flatten()

    if int(expand_rows.numel()) > 0:
        _validate_expansion_rows(
            parent_state=parent_state,
            child_state=child_state,
            rows=expand_rows,
            action_edge_ids=action_edge_ids,
        )

    if int(terminal_rows.numel()) > 0:
        _validate_terminal_rows(
            parent_state=parent_state,
            child_state=child_state,
            rows=terminal_rows,
        )


def _validate_expansion_rows(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    rows: Tensor,
    action_edge_ids: Tensor,
) -> None:
    parent_count = parent_state.edge_count.index_select(0, rows)
    child_count = child_state.edge_count.index_select(0, rows)

    if not bool(child_count.eq(parent_count + 1).all()):
        raise ValueError("Expansion child edge_count must equal parent edge_count + 1.")

    appended_edges = child_state.edge_ids[rows, parent_count]
    selected_actions = action_edge_ids.index_select(0, rows)

    if not bool(appended_edges.eq(selected_actions).all()):
        raise ValueError("Expansion action_edge_ids must equal appended child edge.")

    if not _prefix_equal(
        left=parent_state.edge_ids.index_select(0, rows),
        right=child_state.edge_ids.index_select(0, rows),
        lengths=parent_count,
    ):
        raise ValueError("Expansion parent prefix must match child prefix.")


def _validate_terminal_rows(
    *,
    parent_state: StateBatch,
    child_state: StateBatch,
    rows: Tensor,
) -> None:
    parent_count = parent_state.edge_count.index_select(0, rows)
    child_count = child_state.edge_count.index_select(0, rows)

    if not bool(parent_count.eq(child_count).all()):
        raise ValueError("Terminal rows must not change edge_count.")

    if not _prefix_equal(
        left=parent_state.edge_ids.index_select(0, rows),
        right=child_state.edge_ids.index_select(0, rows),
        lengths=parent_count,
    ):
        raise ValueError("Terminal rows must not change selected edge prefix.")


def _prefix_equal(
    *,
    left: Tensor,
    right: Tensor,
    lengths: Tensor,
) -> bool:
    if left.shape != right.shape:
        return False

    if int(left.numel()) == 0:
        return True

    steps = torch.arange(
        int(left.size(1)),
        dtype=torch.long,
        device=left.device,
    ).unsqueeze(0)

    valid = steps.lt(lengths.view(-1, 1))
    return bool(left[valid].eq(right[valid]).all())


__all__ = [
    "deterministic_backward_log_prob",
    "validate_ordered_prefix_transitions",
]
