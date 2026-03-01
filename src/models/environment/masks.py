from __future__ import annotations

import torch


def build_node_membership_mask(
    *,
    local_indices: torch.Tensor,
    ptr: torch.Tensor,
    node_ptr: torch.Tensor,
    num_nodes_total: int,
    device: torch.device,
    field_name: str,
) -> torch.Tensor:
    """Build per-node boolean membership mask from CSR-style grouped local indices."""
    mask = torch.zeros((num_nodes_total,), dtype=torch.bool, device=device)
    if local_indices.numel() == 0:
        return mask

    counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
    if int(counts.sum().item()) != int(local_indices.numel()):
        raise ValueError(f"{field_name} ptr mismatch with index length.")

    counts_on_device = counts.to(device=device)
    offsets = node_ptr[:-1].to(device=device).repeat_interleave(counts_on_device)
    absolute_indices = local_indices.to(device=device) + offsets
    if bool((absolute_indices < 0).any().item()) or bool((absolute_indices >= num_nodes_total).any().item()):
        raise ValueError(f"{field_name} out of range for membership mask construction.")
    mask.scatter_(0, absolute_indices, True)
    return mask


__all__ = ["build_node_membership_mask"]
