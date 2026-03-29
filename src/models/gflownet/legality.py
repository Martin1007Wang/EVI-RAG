from __future__ import annotations

import torch

from .repetition import (
    build_entity_revisit_mask,
    build_entity_revisit_mask_from_flat_state,
)
from .prefix_state import ForwardActionDistribution, SearchState


def build_legal_forward_move_keep_mask(
    *,
    flat_current_abs_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
    node_entity_ids_by_abs_node: torch.Tensor,
    num_nodes: int,
    candidate_target_abs_nodes: torch.Tensor,
    candidate_agent_indices: torch.Tensor,
    child_num_steps: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    """Keep only forward moves that stay inside the legal prefix state space.

    A graph move is legal iff it stays within the horizon and lands on an
    entity that has not appeared earlier in that state's exact prefix history.
    This is a hard support restriction, not reward shaping.
    """

    if int(candidate_target_abs_nodes.numel()) == 0:
        return torch.zeros_like(candidate_target_abs_nodes, dtype=torch.bool)
    within_horizon = child_num_steps.to(dtype=torch.long) <= int(max_steps)
    revisit_mask = build_entity_revisit_mask_from_flat_state(
        flat_current_abs_nodes=flat_current_abs_nodes,
        flat_num_steps=flat_num_steps,
        flat_path_token_ids=flat_path_token_ids,
        node_entity_ids_by_abs_node=node_entity_ids_by_abs_node,
        num_nodes=num_nodes,
        candidate_target_abs_nodes=candidate_target_abs_nodes,
        candidate_agent_indices=candidate_agent_indices,
    )
    return within_horizon & (~revisit_mask)


def build_unique_forward_candidate_keep_mask(
    *,
    flat_current_abs_nodes: torch.Tensor,
    flat_num_steps: torch.Tensor,
    flat_path_token_ids: torch.Tensor | None,
    node_entity_ids_by_abs_node: torch.Tensor,
    num_nodes: int,
    candidate_target_abs_nodes: torch.Tensor,
    candidate_agent_indices: torch.Tensor,
    child_num_steps: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    """Backward-compatible alias for ``build_legal_forward_move_keep_mask``."""

    return build_legal_forward_move_keep_mask(
        flat_current_abs_nodes=flat_current_abs_nodes,
        flat_num_steps=flat_num_steps,
        flat_path_token_ids=flat_path_token_ids,
        node_entity_ids_by_abs_node=node_entity_ids_by_abs_node,
        num_nodes=num_nodes,
        candidate_target_abs_nodes=candidate_target_abs_nodes,
        candidate_agent_indices=candidate_agent_indices,
        child_num_steps=child_num_steps,
        max_steps=max_steps,
    )


def build_forward_invalid_action_mask(
    *,
    distribution: ForwardActionDistribution,
    state: SearchState,
    max_steps: int,
) -> torch.Tensor:
    edge_logits = distribution.edge_logits
    if int(edge_logits.numel()) == 0:
        return torch.zeros_like(edge_logits, dtype=torch.bool)
    edge_agent_batch = distribution.edge_agent_batch
    active_flat = ~state.flatten_done_mask()
    num_steps_flat = state.flatten_num_steps()
    at_horizon = active_flat & (num_steps_flat >= int(max_steps))
    stop_action_mask = (
        distribution.is_stop_action.to(dtype=torch.bool)
        if distribution.is_stop_action is not None
        else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
    )
    move_action_mask = ~stop_action_mask
    invalid_mask = at_horizon.index_select(0, edge_agent_batch) & move_action_mask
    revisit_mask = build_entity_revisit_mask(
        state=state,
        candidate_target_abs_nodes=distribution.target_nodes,
        candidate_agent_indices=edge_agent_batch,
        max_steps=max_steps,
    )
    return invalid_mask | (revisit_mask & move_action_mask)


def apply_forward_legality(
    distribution: ForwardActionDistribution,
    *,
    state: SearchState,
    max_steps: int,
) -> ForwardActionDistribution:
    """Hard-mask forward actions that leave the legal prefix state space."""

    edge_logits = distribution.edge_logits
    if int(edge_logits.numel()) == 0:
        return distribution
    invalid_mask = build_forward_invalid_action_mask(
        distribution=distribution,
        state=state,
        max_steps=max_steps,
    )
    if not bool(invalid_mask.any().item()):
        return distribution
    neg_inf = torch.tensor(
        float("-inf"),
        device=edge_logits.device,
        dtype=edge_logits.dtype,
    )
    return ForwardActionDistribution(
        edge_logits=edge_logits.masked_fill(invalid_mask, neg_inf),
        edge_agent_batch=distribution.edge_agent_batch,
        edge_ids=distribution.edge_ids,
        target_nodes=distribution.target_nodes,
        out_degrees=distribution.out_degrees,
        is_stop_action=distribution.is_stop_action,
        is_root_action=distribution.is_root_action,
        current_log_f=distribution.current_log_f,
        active_agent_count=distribution.active_agent_count,
        unique_active_state_count=distribution.unique_active_state_count,
        raw_graph_candidate_count=distribution.raw_graph_candidate_count,
        scored_graph_candidate_count=distribution.scored_graph_candidate_count,
    )


__all__ = [
    "apply_forward_legality",
    "build_legal_forward_move_keep_mask",
    "build_forward_invalid_action_mask",
    "build_unique_forward_candidate_keep_mask",
]
