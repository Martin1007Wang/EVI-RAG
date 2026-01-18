from __future__ import annotations

from typing import Tuple

import torch

_STOP_RELATION = -1
_ZERO = 0


def reflect_backward_to_forward(
    *,
    actions_seq: torch.Tensor,
    lengths: torch.Tensor,
    edge_inverse_map: torch.Tensor,
    stop_value: int = _STOP_RELATION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reflect backward trajectories (t -> ... -> s -> STOP) into forward form (s -> ... -> t -> STOP).

    Assumes lengths include the terminal STOP step. Drops the trailing STOP, reverses remaining edges,
    maps to inverse edges, then appends STOP. Padding beyond `lengths` is filled with STOP.
    """
    if actions_seq.dim() != 2:
        raise ValueError(f"actions_seq must be [B, T], got shape={tuple(actions_seq.shape)}")
    if lengths.dim() != 1:
        raise ValueError(f"lengths must be [B], got shape={tuple(lengths.shape)}")
    if actions_seq.size(0) != lengths.numel():
        raise ValueError("actions_seq batch size mismatch with lengths.")
    device = actions_seq.device
    batch, max_steps = actions_seq.shape
    lengths = lengths.to(device=device, dtype=torch.long)
    # Remove terminal STOP from effective length; clamp to non-negative
    valid_len = (lengths - 1).clamp(min=_ZERO, max=max_steps)
    base_idx = torch.arange(max_steps, device=device).unsqueeze(0).expand(batch, -1)
    mask = base_idx < valid_len.unsqueeze(1)
    # reverse indices within valid lengths
    rev_idx = valid_len.unsqueeze(1) - 1 - base_idx
    rev_idx = torch.where(mask, rev_idx, torch.zeros_like(rev_idx))
    gathered = actions_seq.gather(1, rev_idx.clamp(min=_ZERO))
    # map to inverse edges for valid positions
    inv_edges = edge_inverse_map.to(device=device, dtype=torch.long)
    mapped = inv_edges.gather(0, gathered.clamp(min=_ZERO))
    mapped = torch.where(mask, mapped, torch.full_like(mapped, stop_value))
    # assemble output and append STOP at new tail
    reflected = torch.full_like(actions_seq, stop_value)
    reflected = reflected.masked_scatter(mask, mapped[mask])
    tail_indices = valid_len.clamp(max=max_steps - 1)
    batch_ids = torch.arange(batch, device=device, dtype=torch.long)
    reflected[batch_ids, tail_indices] = stop_value
    new_lengths = valid_len + 1  # add STOP back
    return reflected, new_lengths


__all__ = ["reflect_backward_to_forward"]
