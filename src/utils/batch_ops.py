from __future__ import annotations

from typing import Optional

import torch


def build_dummy_mask(*, answer_ptr: torch.Tensor) -> torch.Tensor:
    return answer_ptr[1:] == answer_ptr[:-1]


def build_node_batch(*, node_ptr: torch.Tensor, device: torch.device) -> torch.Tensor:
    num_graphs = node_ptr.numel() - 1
    if num_graphs <= 0:
        return torch.zeros((0,), device=device, dtype=torch.long)
    counts = node_ptr[1:] - node_ptr[:-1]
    return torch.repeat_interleave(torch.arange(num_graphs, device=device), counts)


def build_node_mask(num_nodes_total: int, indices: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros((num_nodes_total,), device=indices.device, dtype=torch.bool)
    if indices.numel() > 0:
        valid = indices >= 0
        safe = indices[valid]
        if safe.numel() > 0:
            mask[safe] = True
    return mask


def edge_reorder_perm(
    *,
    edge_index: torch.Tensor,
    edge_batch: torch.Tensor,
    edge_relations: torch.Tensor,
    node_ptr: torch.Tensor,
    num_edges_before: int,
) -> Optional[torch.Tensor]:
    if edge_index.size(1) != num_edges_before:
        raise ValueError("edge_index length mismatch before reorder.")
    if edge_relations.numel() != num_edges_before:
        raise ValueError("edge_relations length mismatch before reorder.")
    if edge_batch.numel() != num_edges_before:
        raise ValueError("edge_batch length mismatch before reorder.")
    if edge_index.numel() == 0:
        return None
    num_graphs = node_ptr.numel() - 1
    if num_graphs <= 0:
        return None
    edge_batch = edge_batch.view(-1)
    if edge_batch.numel() <= 1:
        return None
    if (edge_batch[:-1] <= edge_batch[1:]).all().item():
        return None
    return torch.argsort(edge_batch)


def reorder_edge_inverse_map(
    *,
    edge_inverse_map: torch.Tensor,
    perm: Optional[torch.Tensor],
) -> torch.Tensor:
    if perm is None or edge_inverse_map.numel() == 0:
        return edge_inverse_map
    if edge_inverse_map.numel() != perm.numel():
        raise ValueError("edge_inverse_map length mismatch with perm.")
    perm = perm.view(-1)
    edge_inverse_map = edge_inverse_map.index_select(0, perm)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    valid = edge_inverse_map >= 0
    safe = torch.where(valid, edge_inverse_map, torch.zeros_like(edge_inverse_map))
    mapped = inv_perm.index_select(0, safe)
    return torch.where(valid, mapped, edge_inverse_map)


__all__ = [
    "build_dummy_mask",
    "build_node_batch",
    "build_node_mask",
    "edge_reorder_perm",
    "reorder_edge_inverse_map",
]
