from __future__ import annotations

from dataclasses import dataclass

import torch

from .prefix_state import ForwardActionDistribution, PreparedSearchBatch, SearchState


@dataclass(frozen=True)
class BackwardCandidateBatch:
    edge_ids: torch.Tensor
    source_nodes: torch.Tensor
    edge_agent_batch: torch.Tensor
    in_degrees: torch.Tensor
    parent_num_steps: torch.Tensor


def expected_backward_transitions(
    *,
    state: SearchState,
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_num_steps = state.flatten_num_steps()
    flat_path_token_ids = state.flatten_path_token_ids(max_steps=max_steps)
    parent_step_ids = (flat_num_steps - 1).clamp_min(0)
    parent_node_positions = (2 * parent_step_ids).to(dtype=torch.long)
    parent_relation_positions = (parent_node_positions + 1).to(dtype=torch.long)
    row_idx = torch.arange(
        int(flat_num_steps.numel()),
        device=flat_num_steps.device,
        dtype=torch.long,
    )
    expected_parent_nodes = flat_path_token_ids[row_idx, parent_node_positions]
    expected_relation_ids = flat_path_token_ids[row_idx, parent_relation_positions]
    return expected_parent_nodes, expected_relation_ids


def gather_backward_candidates(
    *,
    state: SearchState,
) -> BackwardCandidateBatch:
    batch_size, num_rollouts = state.current_nodes.shape
    flat_current_nodes = state.flatten_current_nodes()
    active_mask = ~state.flatten_done_mask()
    flat_num_steps = state.flatten_num_steps()
    backward_active_mask = active_mask & (flat_num_steps > 0)
    edge_ids, source_nodes, edge_agent_batch, in_degrees = (
        state.topology.gather_incoming_edges(
            current_nodes=flat_current_nodes,
            active_mask=backward_active_mask,
        )
    )
    if int(edge_ids.numel()) > 0:
        parent_num_steps = flat_num_steps.index_select(0, edge_agent_batch) - 1
    else:
        parent_num_steps = flat_num_steps.new_empty((0,))
    return BackwardCandidateBatch(
        edge_ids=edge_ids,
        source_nodes=source_nodes,
        edge_agent_batch=edge_agent_batch,
        in_degrees=in_degrees.view(batch_size, num_rollouts),
        parent_num_steps=parent_num_steps,
    )


def compute_tree_backward_logits(
    *,
    prepared_batch: PreparedSearchBatch,
    state: SearchState,
    edge_ids: torch.Tensor,
    source_nodes: torch.Tensor,
    edge_agent_batch: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    """Recover the unique parent edge on the prefix-tree state space."""

    edge_logits = torch.full(
        (int(edge_ids.numel()),),
        fill_value=float("-inf"),
        device=edge_ids.device,
        dtype=torch.float32,
    )
    if int(edge_ids.numel()) == 0:
        return edge_logits

    expected_parent_nodes, expected_relation_ids = expected_backward_transitions(
        state=state,
        max_steps=max_steps,
    )
    relation_ids = prepared_batch.topology.edge_type.index_select(0, edge_ids)
    valid_edges = (
        source_nodes == expected_parent_nodes.index_select(0, edge_agent_batch)
    ) & (relation_ids == expected_relation_ids.index_select(0, edge_agent_batch))
    edge_logits = torch.where(valid_edges, torch.zeros_like(edge_logits), edge_logits)

    valid_counts = torch.zeros(
        (int(state.current_nodes.numel()),),
        device=edge_ids.device,
        dtype=torch.long,
    )
    if int(valid_edges.numel()) > 0:
        valid_counts.scatter_add_(0, edge_agent_batch, valid_edges.to(dtype=torch.long))
    active_mask = (~state.flatten_done_mask()) & (state.flatten_num_steps() > 0)
    if not bool((valid_counts[active_mask] > 0).all().item()):
        invalid_agents = torch.nonzero(
            active_mask & (valid_counts <= 0),
            as_tuple=False,
        ).view(-1)
        raise RuntimeError(
            "Backward distribution could not recover a parent edge from the encoded path. "
            f"invalid_agents={invalid_agents.tolist()}"
        )
    return edge_logits


def compute_policy_backward_distribution(
    *,
    prepared_batch: PreparedSearchBatch,
    state: SearchState,
    max_steps: int,
) -> ForwardActionDistribution:
    """Compatibility backward distribution used by tests and diagnostics.

    The training hot path does not reconstruct move-step backward logits anymore,
    but several evaluation and invariance tests still depend on the exact prefix
    tree predecessor semantics.
    """

    flat_done_mask = state.flatten_done_mask()
    flat_absorbing_mask = state.flatten_absorbing_mask()
    flat_num_steps = state.flatten_num_steps()
    flat_current_nodes = state.flatten_current_nodes()
    active_non_root = (~flat_done_mask) & (flat_num_steps > 0)
    active_start = (~flat_done_mask) & (flat_num_steps == 0)
    candidates = gather_backward_candidates(state=state)
    if bool(active_non_root.any().item()) and int(candidates.edge_ids.numel()) == 0:
        invalid_agents = torch.nonzero(active_non_root, as_tuple=False).view(-1)
        raise RuntimeError(
            "Backward distribution could not recover a parent edge from the encoded path. "
            f"invalid_agents={invalid_agents.tolist()}"
        )

    edge_ids = candidates.edge_ids
    source_nodes = candidates.source_nodes
    edge_agent_batch = candidates.edge_agent_batch
    edge_logits = torch.empty(
        (0,),
        device=state.current_nodes.device,
        dtype=torch.float32,
    )
    is_stop_action = torch.zeros_like(edge_ids, dtype=torch.bool)
    is_root_action = torch.zeros_like(edge_ids, dtype=torch.bool)
    if int(edge_ids.numel()) > 0:
        edge_logits = compute_tree_backward_logits(
            prepared_batch=prepared_batch,
            state=state,
            edge_ids=edge_ids,
            source_nodes=source_nodes,
            edge_agent_batch=edge_agent_batch,
            max_steps=max_steps,
        )
    if bool(active_start.any().item()):
        root_agents = torch.nonzero(active_start, as_tuple=False).view(-1)
        root_edge_ids = torch.full_like(root_agents, fill_value=-2)
        root_source_nodes = flat_current_nodes.index_select(0, root_agents)
        root_logits = torch.zeros_like(root_agents, dtype=torch.float32)
        edge_ids = torch.cat((edge_ids, root_edge_ids), dim=0)
        source_nodes = torch.cat((source_nodes, root_source_nodes), dim=0)
        edge_agent_batch = torch.cat((edge_agent_batch, root_agents), dim=0)
        edge_logits = torch.cat((edge_logits, root_logits), dim=0)
        is_stop_action = torch.cat(
            (is_stop_action, torch.zeros_like(root_agents, dtype=torch.bool)),
            dim=0,
        )
        is_root_action = torch.cat(
            (is_root_action, torch.ones_like(root_agents, dtype=torch.bool)),
            dim=0,
        )
    if bool(flat_absorbing_mask.any().item()):
        stop_agents = torch.nonzero(flat_absorbing_mask, as_tuple=False).view(-1)
        stop_edge_ids = torch.full_like(stop_agents, fill_value=-1)
        stop_source_nodes = flat_current_nodes.index_select(0, stop_agents)
        stop_logits = torch.zeros_like(stop_agents, dtype=torch.float32)
        edge_ids = torch.cat((edge_ids, stop_edge_ids), dim=0)
        source_nodes = torch.cat((source_nodes, stop_source_nodes), dim=0)
        edge_agent_batch = torch.cat((edge_agent_batch, stop_agents), dim=0)
        edge_logits = torch.cat((edge_logits, stop_logits), dim=0)
        is_stop_action = torch.cat(
            (is_stop_action, torch.ones_like(stop_agents, dtype=torch.bool)),
            dim=0,
        )
        is_root_action = torch.cat(
            (is_root_action, torch.zeros_like(stop_agents, dtype=torch.bool)),
            dim=0,
        )
    if int(edge_agent_batch.numel()) > 0:
        order = torch.argsort(edge_agent_batch, stable=True)
        edge_ids = edge_ids.index_select(0, order)
        source_nodes = source_nodes.index_select(0, order)
        edge_agent_batch = edge_agent_batch.index_select(0, order)
        edge_logits = edge_logits.index_select(0, order)
        is_stop_action = is_stop_action.index_select(0, order)
        is_root_action = is_root_action.index_select(0, order)
    in_degrees = candidates.in_degrees.clone()
    if bool(active_start.any().item()):
        in_degrees.view(-1)[active_start] = 1
    if bool(flat_absorbing_mask.any().item()):
        in_degrees.view(-1)[flat_absorbing_mask] = 1
    return ForwardActionDistribution(
        edge_logits=edge_logits.to(dtype=torch.float32),
        edge_agent_batch=edge_agent_batch,
        edge_ids=edge_ids,
        target_nodes=source_nodes,
        out_degrees=in_degrees,
        is_stop_action=is_stop_action,
        is_root_action=is_root_action,
    )


__all__ = [
    "BackwardCandidateBatch",
    "compute_policy_backward_distribution",
    "compute_tree_backward_logits",
    "expected_backward_transitions",
    "gather_backward_candidates",
]
