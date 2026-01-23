from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

_STOP_RELATION = -1
_ZERO = 0
_ONE = 1


@dataclass(frozen=True)
class TrajectoryStats:
    stop_idx: torch.Tensor
    num_moves: torch.Tensor
    move_mask: torch.Tensor
    step_mask_incl_stop: torch.Tensor
    terminal_mask: torch.Tensor


def derive_trajectory(*, actions_seq: torch.Tensor, stop_value: int = _STOP_RELATION) -> TrajectoryStats:
    if actions_seq.dim() != 2:
        raise ValueError(f"actions_seq must be [B, T], got shape={tuple(actions_seq.shape)}")
    actions = actions_seq.to(dtype=torch.long)
    batch, max_steps = actions.shape
    if batch == _ZERO:
        empty_idx = torch.zeros((0,), device=actions.device, dtype=torch.long)
        empty_mask = torch.zeros((0, max_steps), device=actions.device, dtype=torch.bool)
        return TrajectoryStats(
            stop_idx=empty_idx,
            num_moves=empty_idx,
            move_mask=empty_mask,
            step_mask_incl_stop=empty_mask,
            terminal_mask=empty_mask,
        )
    stop_mask = actions == stop_value
    if not bool(stop_mask.any(dim=1).all().detach().tolist()):
        raise ValueError("All trajectories must contain a STOP action.")
    stop_idx = stop_mask.to(dtype=torch.long).argmax(dim=1)
    step_ids = torch.arange(max_steps, device=actions.device).view(1, -1)
    after_stop = step_ids > stop_idx.view(-1, 1)
    invalid_after = after_stop & (actions != stop_value)
    if bool(invalid_after.any().detach().tolist()):
        raise ValueError("Actions after STOP must be STOP padding.")
    invalid_neg = (actions < _ZERO) & (actions != stop_value)
    if bool(invalid_neg.any().detach().tolist()):
        raise ValueError("Actions contain invalid negative values.")
    move_mask = step_ids < stop_idx.view(-1, 1)
    step_mask_incl_stop = step_ids <= stop_idx.view(-1, 1)
    terminal_mask = step_ids == stop_idx.view(-1, 1)
    num_moves = stop_idx
    return TrajectoryStats(
        stop_idx=stop_idx,
        num_moves=num_moves,
        move_mask=move_mask,
        step_mask_incl_stop=step_mask_incl_stop,
        terminal_mask=terminal_mask,
    )


def reflect_backward_to_forward(
    *,
    actions_seq: torch.Tensor,
    num_moves: torch.Tensor | None = None,
    stop_idx: torch.Tensor | None = None,
    edge_inverse_map: torch.Tensor,
    stop_value: int = _STOP_RELATION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if actions_seq.dim() != 2:
        raise ValueError(f"actions_seq must be [B, T], got shape={tuple(actions_seq.shape)}")
    if (num_moves is None) == (stop_idx is None):
        raise ValueError("Provide exactly one of num_moves or stop_idx.")
    device = actions_seq.device
    batch, max_steps = actions_seq.shape
    if stop_idx is None:
        stop_idx = num_moves
    if stop_idx is None:
        raise ValueError("stop_idx must be provided.")
    stop_idx = stop_idx.to(device=device, dtype=torch.long)
    if stop_idx.dim() != 1:
        raise ValueError(f"stop_idx must be [B], got shape={tuple(stop_idx.shape)}")
    if stop_idx.numel() != batch:
        raise ValueError("stop_idx batch size mismatch with actions_seq.")
    if bool((stop_idx < _ZERO).any().detach().tolist()) or bool((stop_idx >= max_steps).any().detach().tolist()):
        raise ValueError("stop_idx must be within [0, T).")
    valid_len = stop_idx.clamp(min=_ZERO, max=max_steps)
    base_idx = torch.arange(max_steps, device=device).unsqueeze(0).expand(batch, -1)
    mask = base_idx < valid_len.unsqueeze(1)
    rev_idx = valid_len.unsqueeze(1) - 1 - base_idx
    rev_idx = torch.where(mask, rev_idx, torch.zeros_like(rev_idx))
    gathered = actions_seq.gather(1, rev_idx.clamp(min=_ZERO))
    inv_edges = edge_inverse_map.to(device=device, dtype=torch.long)
    flat_idx = gathered.clamp(min=_ZERO).view(-1)
    mapped = inv_edges.index_select(0, flat_idx).view_as(gathered)
    mapped = torch.where(mask, mapped, torch.full_like(mapped, stop_value))
    reflected = torch.full_like(actions_seq, stop_value)
    reflected = reflected.masked_scatter(mask, mapped[mask])
    batch_ids = torch.arange(batch, device=device, dtype=torch.long)
    reflected[batch_ids, stop_idx] = stop_value
    return reflected, stop_idx


def reflect_forward_to_backward(
    *,
    actions_seq: torch.Tensor,
    num_moves: torch.Tensor | None = None,
    stop_idx: torch.Tensor | None = None,
    edge_inverse_map: torch.Tensor,
    stop_value: int = _STOP_RELATION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return reflect_backward_to_forward(
        actions_seq=actions_seq,
        num_moves=num_moves,
        stop_idx=stop_idx,
        edge_inverse_map=edge_inverse_map,
        stop_value=stop_value,
    )


def resolve_current_stop_locals(
    *,
    curr_nodes: torch.Tensor,
    node_ptr: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_graphs = int(node_ptr.numel() - 1)
    num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
    sentinel = num_nodes_total + _ONE
    out = torch.full((num_graphs,), sentinel, device=node_ptr.device, dtype=dtype)
    if num_graphs <= _ZERO:
        return out, out != sentinel
    curr_nodes = curr_nodes.view(-1)
    if curr_nodes.numel() != num_graphs:
        raise ValueError("curr_nodes length mismatch with batch size.")
    valid = curr_nodes >= _ZERO
    if not bool(valid.any().item()):
        return out, valid
    safe_nodes = curr_nodes.clamp(min=_ZERO)
    graph_ids = torch.arange(num_graphs, device=node_ptr.device)
    local_idx = safe_nodes - node_ptr.index_select(0, graph_ids)
    out = torch.where(valid, local_idx.to(dtype=dtype), out)
    return out, valid


def stack_steps(
    steps: list[torch.Tensor],
    *,
    num_graphs: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    fill_value: float | int,
) -> torch.Tensor:
    if num_steps <= _ZERO:
        return torch.zeros((num_graphs, _ZERO), device=device, dtype=dtype)
    if not steps:
        return torch.full((num_graphs, num_steps), fill_value, device=device, dtype=dtype)
    stacked = torch.stack(steps, dim=1)
    if stacked.size(1) > num_steps:
        raise ValueError("step stack exceeds expected rollout horizon.")
    if stacked.size(1) < num_steps:
        pad = torch.full(
            (num_graphs, num_steps - stacked.size(1)),
            fill_value,
            device=device,
            dtype=dtype,
        )
        stacked = torch.cat([stacked, pad], dim=1)
    return stacked


__all__ = [
    "TrajectoryStats",
    "derive_trajectory",
    "reflect_backward_to_forward",
    "reflect_forward_to_backward",
    "resolve_current_stop_locals",
    "stack_steps",
]
