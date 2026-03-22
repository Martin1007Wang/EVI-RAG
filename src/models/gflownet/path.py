from __future__ import annotations

import torch


STOP_TOKEN_ID = -1


def max_path_tokens(*, max_steps: int) -> int:
    return (2 * int(max_steps)) + 2


def initialize_path_token_ids(
    *, start_nodes: torch.Tensor, max_steps: int
) -> torch.Tensor:
    token_ids = torch.zeros(
        (*start_nodes.shape, max_path_tokens(max_steps=max_steps)),
        device=start_nodes.device,
        dtype=torch.long,
    )
    token_ids[..., 0] = start_nodes
    return token_ids


def append_relation_and_node_tokens(
    *,
    path_token_ids: torch.Tensor,
    num_steps: torch.Tensor,
    relation_ids: torch.Tensor,
    target_nodes: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    updated = path_token_ids.clone()
    append_relation_and_node_tokens_inplace(
        path_token_ids=updated,
        num_steps=num_steps,
        relation_ids=relation_ids,
        target_nodes=target_nodes,
        active_mask=active_mask,
    )
    return updated


def append_relation_and_node_tokens_inplace(
    *,
    path_token_ids: torch.Tensor,
    num_steps: torch.Tensor,
    relation_ids: torch.Tensor,
    target_nodes: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    flat_updated = path_token_ids.view(-1, int(path_token_ids.size(-1)))
    flat_num_steps = num_steps.reshape(-1)
    flat_relation_ids = relation_ids.reshape(-1)
    flat_target_nodes = target_nodes.reshape(-1)
    if active_mask is None:
        active_indices = torch.arange(
            int(flat_num_steps.numel()), device=flat_num_steps.device, dtype=torch.long
        )
    else:
        active_indices = torch.nonzero(active_mask.reshape(-1), as_tuple=False).view(-1)
    if int(active_indices.numel()) == 0:
        return path_token_ids
    relation_positions = (2 * flat_num_steps.index_select(0, active_indices) + 1).to(
        dtype=torch.long
    )
    node_positions = relation_positions + 1
    flat_updated[active_indices, relation_positions] = flat_relation_ids.index_select(
        0, active_indices
    )
    flat_updated[active_indices, node_positions] = flat_target_nodes.index_select(
        0, active_indices
    )
    return path_token_ids


def append_stop_token_inplace(
    *,
    path_token_ids: torch.Tensor,
    num_steps: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    flat_updated = path_token_ids.view(-1, int(path_token_ids.size(-1)))
    flat_num_steps = num_steps.reshape(-1)
    if active_mask is None:
        active_indices = torch.arange(
            int(flat_num_steps.numel()), device=flat_num_steps.device, dtype=torch.long
        )
    else:
        active_indices = torch.nonzero(active_mask.reshape(-1), as_tuple=False).view(-1)
    if int(active_indices.numel()) == 0:
        return path_token_ids
    stop_positions = (2 * flat_num_steps.index_select(0, active_indices) + 1).to(
        dtype=torch.long
    )
    flat_updated[active_indices, stop_positions] = STOP_TOKEN_ID
    return path_token_ids


def derive_parent_path_token_ids(
    *,
    child_path_token_ids: torch.Tensor,
    child_num_steps: torch.Tensor,
    parent_nodes: torch.Tensor,
) -> torch.Tensor:
    parent_num_steps = (child_num_steps - 1).clamp_min(0)
    updated = child_path_token_ids.clone()
    token_length = int(updated.size(-1))
    positions = torch.arange(token_length, device=updated.device, dtype=torch.long)
    parent_token_lengths = 2 * parent_num_steps.reshape(-1) + 1
    flat_updated = updated.view(-1, token_length)
    flat_updated = flat_updated.masked_fill(
        positions.unsqueeze(0) >= parent_token_lengths.unsqueeze(1),
        0,
    )
    row_idx = torch.arange(
        int(flat_updated.size(0)), device=updated.device, dtype=torch.long
    )
    last_node_positions = 2 * parent_num_steps.reshape(-1)
    flat_updated[row_idx, last_node_positions] = parent_nodes.reshape(-1)
    return flat_updated.view_as(child_path_token_ids)


def count_path_node_revisits(
    *, path_token_ids: torch.Tensor, num_steps: torch.Tensor
) -> torch.Tensor:
    flat_paths = path_token_ids.view(-1, int(path_token_ids.size(-1)))
    flat_num_steps = num_steps.reshape(-1).to(dtype=torch.long)
    revisit_counts = torch.zeros_like(flat_num_steps, dtype=torch.long)
    for row_idx in range(int(flat_paths.size(0))):
        step_count = int(flat_num_steps[row_idx].item())
        node_ids = flat_paths[row_idx, : (2 * step_count) + 1 : 2]
        if int(node_ids.numel()) <= 1:
            continue
        _, visit_counts = torch.unique(node_ids, sorted=False, return_counts=True)
        revisit_counts[row_idx] = (visit_counts - 1).clamp_min(0).sum()
    return revisit_counts.view_as(num_steps)


def reconstruct_trace_path_token_ids(
    *,
    start_nodes: torch.Tensor,
    trace_edge_ids: torch.Tensor,
    trace_num_steps: torch.Tensor,
    trace_stop_mask: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    path_token_ids = initialize_path_token_ids(
        start_nodes=start_nodes, max_steps=max_steps
    )
    trace_path_token_ids = torch.zeros(
        (*trace_edge_ids.shape, int(path_token_ids.size(-1))),
        device=trace_edge_ids.device,
        dtype=torch.long,
    )
    current_path_token_ids = path_token_ids
    for step_idx in range(int(trace_edge_ids.size(-1))):
        trace_path_token_ids[:, :, step_idx] = current_path_token_ids
        chosen_edge_ids = trace_edge_ids[:, :, step_idx]
        graph_move_mask = chosen_edge_ids >= 0
        if bool(graph_move_mask.any().item()):
            flat_edges = chosen_edge_ids.reshape(-1)
            safe_edge_ids = flat_edges.clamp(min=0)
            relation_ids = edge_type.index_select(0, safe_edge_ids).view_as(
                chosen_edge_ids
            )
            target_nodes = (
                edge_index[1].index_select(0, safe_edge_ids).view_as(chosen_edge_ids)
            )
            current_path_token_ids = append_relation_and_node_tokens_inplace(
                path_token_ids=current_path_token_ids,
                num_steps=trace_num_steps[:, :, step_idx],
                relation_ids=relation_ids,
                target_nodes=target_nodes,
                active_mask=graph_move_mask,
            )
        if trace_stop_mask is None:
            continue
        stop_action_mask = trace_stop_mask[:, :, step_idx].to(dtype=torch.bool)
        if bool(stop_action_mask.any().item()):
            current_path_token_ids = append_stop_token_inplace(
                path_token_ids=current_path_token_ids,
                num_steps=trace_num_steps[:, :, step_idx],
                active_mask=stop_action_mask,
            )
    return trace_path_token_ids


__all__ = [
    "append_relation_and_node_tokens",
    "append_relation_and_node_tokens_inplace",
    "append_stop_token_inplace",
    "count_path_node_revisits",
    "derive_parent_path_token_ids",
    "initialize_path_token_ids",
    "max_path_tokens",
    "reconstruct_trace_path_token_ids",
    "STOP_TOKEN_ID",
]
