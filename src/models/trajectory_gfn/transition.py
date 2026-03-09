from __future__ import annotations

import torch

from src.utils.segment_ops import compute_has_finite_edges

from .policy import ForwardActionDistribution
from .state import TrajectoryState


def apply_forward_constraints(
    distribution: ForwardActionDistribution,
    *,
    state: TrajectoryState,
    node_is_target: torch.Tensor,
    min_stop_steps: int,
    max_steps: int,
) -> ForwardActionDistribution:
    active_flat = ~state.flatten_done()
    num_moves_flat = state.flatten_num_moves()
    current_nodes = state.flatten_current()
    safe_current = current_nodes.clamp(
        min=0, max=max(int(node_is_target.numel()) - 1, 0)
    )
    on_target = node_is_target.index_select(0, safe_current) & active_flat
    at_horizon = active_flat & (num_moves_flat >= max_steps)
    force_stop = on_target | at_horizon

    stop_logits_flat = distribution.stop_logits.view(-1)
    edge_logits = distribution.edge_logits
    edge_agent_batch = distribution.edge_agent_batch
    out_degrees_flat = distribution.out_degrees.view(-1)
    neg_inf = torch.tensor(
        float("-inf"), device=stop_logits_flat.device, dtype=stop_logits_flat.dtype
    )

    if int(edge_logits.numel()) > 0:
        prefix_nodes = state.flatten_path_nodes()
        visited_nodes = prefix_nodes.index_select(0, edge_agent_batch)
        visited_mask = visited_nodes >= 0
        revisit = (
            distribution.target_nodes.unsqueeze(1) == visited_nodes
        ) & visited_mask
        edge_logits = edge_logits.masked_fill(
            revisit.any(dim=1),
            neg_inf,
        )
        edge_force_stop = force_stop.index_select(0, edge_agent_batch)
        edge_logits = edge_logits.masked_fill(edge_force_stop, neg_inf)

    has_finite_edges = compute_has_finite_edges(
        edge_logits=edge_logits,
        out_degrees=out_degrees_flat,
    )
    ban_stop = active_flat & (~force_stop) & (num_moves_flat < min_stop_steps)
    stop_logits_flat = stop_logits_flat.masked_fill(ban_stop, neg_inf)
    invalid_rows = (
        active_flat
        & (~force_stop)
        & (num_moves_flat < min_stop_steps)
        & (~has_finite_edges)
    )
    return ForwardActionDistribution(
        edge_logits=edge_logits,
        edge_agent_batch=edge_agent_batch,
        stop_logits=stop_logits_flat.view_as(distribution.stop_logits),
        edge_ids=distribution.edge_ids,
        target_nodes=distribution.target_nodes,
        out_degrees=distribution.out_degrees,
        state_log_flows=distribution.state_log_flows,
        invalid_rows=(distribution.invalid_rows.view(-1) | invalid_rows).view_as(
            distribution.invalid_rows
        ),
    )


def advance_state(
    state: TrajectoryState,
    *,
    chosen_target_nodes: torch.Tensor,
    chosen_edge_ids: torch.Tensor,
    is_stop: torch.Tensor,
) -> TrajectoryState:
    if chosen_target_nodes.dim() != 1:
        raise ValueError(
            "chosen_target_nodes must be flattened before advancing state."
        )
    flat_current = state.flatten_current()
    active_flat = ~state.flatten_done()
    is_stop = is_stop.to(device=flat_current.device, dtype=torch.bool)
    active_move = active_flat & (~is_stop)
    if int(is_stop.numel()) != int(flat_current.numel()):
        raise ValueError("is_stop length mismatch with current node batch.")
    if int(chosen_edge_ids.numel()) != int(flat_current.numel()):
        raise ValueError("chosen_edge_ids length mismatch with current node batch.")
    next_current = torch.where(active_move, chosen_target_nodes, flat_current)
    next_done = state.flatten_done() | (active_flat & is_stop)
    next_num_moves = state.flatten_num_moves() + active_move.to(dtype=torch.long)
    prefix_nodes = state.flatten_path_nodes().clone()
    prefix_edges = state.flatten_path_edge_ids().clone()
    if bool(active_move.any().item()):
        row_ids = torch.arange(int(flat_current.numel()), device=flat_current.device)[
            active_move
        ]
        prefix_edges[row_ids, next_num_moves[active_move] - 1] = chosen_edge_ids[
            active_move
        ].to(dtype=torch.long)
        prefix_nodes[row_ids, next_num_moves[active_move]] = chosen_target_nodes[
            active_move
        ].to(dtype=torch.long)
    batch_size, num_rollouts = state.current_node.shape
    return TrajectoryState(
        step_t=state.step_t + 1,
        current_node=next_current.view_as(state.current_node),
        done_mask=next_done.view_as(state.done_mask),
        num_moves=next_num_moves.view_as(state.num_moves),
        path_nodes=prefix_nodes.view(batch_size, num_rollouts, -1),
        path_edge_ids=prefix_edges.view(batch_size, num_rollouts, -1),
    )
