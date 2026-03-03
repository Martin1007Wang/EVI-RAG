from __future__ import annotations

from typing import Literal

import torch

FlowDirection = Literal["forward", "backward"]


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
    if bool((absolute_indices < 0).any().item()) or bool(
        (absolute_indices >= num_nodes_total).any().item()
    ):
        raise ValueError(f"{field_name} out of range for membership mask construction.")
    mask.scatter_(0, absolute_indices, True)
    return mask


def infer_super_source_absolute_indices(
    *,
    node_ptr: torch.Tensor,
    node_global_ids: torch.Tensor,
    num_nodes_total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if node_ptr.dim() != 1 or int(node_ptr.numel()) < 2:
        raise ValueError(
            "node_ptr must be 1D with at least one graph when inferring super source nodes."
        )
    node_ptr_long = node_ptr.to(device=device, dtype=torch.long)
    if int(node_ptr_long[-1].item()) != int(num_nodes_total):
        raise ValueError(
            "node_ptr[-1] mismatch with num_nodes_total when inferring super source nodes: "
            f"node_ptr[-1]={int(node_ptr_long[-1].item())}, num_nodes_total={int(num_nodes_total)}."
        )
    node_global_long = node_global_ids.to(device=device, dtype=torch.long)
    if int(node_global_long.numel()) != int(num_nodes_total):
        raise ValueError(
            "node_global_ids length mismatch with num_nodes_total when inferring super source nodes: "
            f"node_global_ids={int(node_global_long.numel())}, num_nodes_total={int(num_nodes_total)}."
        )
    counts = node_ptr_long[1:] - node_ptr_long[:-1]
    if bool((counts < 2).any().item()):
        raise ValueError(
            "Each graph must contain at least two nodes when super source is enabled."
        )
    forward_abs = node_ptr_long[1:] - 2
    backward_abs = node_ptr_long[1:] - 1
    super_mask = node_global_long < 0
    num_graphs = int(node_ptr_long.numel()) - 1
    if int(super_mask.sum().item()) != 2 * num_graphs:
        raise ValueError(
            "Super-source layout invariant violated: expected exactly two virtual nodes per graph "
            f"(got {int(super_mask.sum().item())} negatives for {num_graphs} graphs)."
        )
    forward_ok = bool(super_mask.index_select(0, forward_abs).all().item())
    backward_ok = bool(super_mask.index_select(0, backward_abs).all().item())
    if not (forward_ok and backward_ok):
        raise ValueError(
            "Super-source layout invariant violated: trailing per-graph nodes are not both virtual."
        )
    return forward_abs, backward_abs


def has_super_source_layout(
    *,
    node_ptr: torch.Tensor,
    node_global_ids: torch.Tensor,
    num_nodes_total: int,
    device: torch.device,
) -> bool:
    try:
        infer_super_source_absolute_indices(
            node_ptr=node_ptr,
            node_global_ids=node_global_ids,
            num_nodes_total=num_nodes_total,
            device=device,
        )
    except ValueError:
        return False
    return True


def resolve_super_source_absolute_indices(
    *,
    node_ptr: torch.Tensor,
    node_global_ids: torch.Tensor,
    num_nodes_total: int,
    direction: FlowDirection,
    device: torch.device,
) -> torch.Tensor:
    forward_abs, backward_abs = infer_super_source_absolute_indices(
        node_ptr=node_ptr,
        node_global_ids=node_global_ids,
        num_nodes_total=num_nodes_total,
        device=device,
    )
    if direction == "forward":
        return forward_abs
    if direction == "backward":
        return backward_abs
    raise ValueError(f"Unsupported flow direction: {direction!r}")


__all__ = [
    "FlowDirection",
    "build_node_membership_mask",
    "has_super_source_layout",
    "infer_super_source_absolute_indices",
    "resolve_super_source_absolute_indices",
]
