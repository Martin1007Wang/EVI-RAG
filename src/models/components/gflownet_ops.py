from __future__ import annotations

from dataclasses import dataclass

import torch

_GUMBEL_EPS = 1e-10
_ZERO = 0
_ONE = 1
_EDGE_HEAD_INDEX = 0
_EDGE_TAIL_INDEX = 1


def neg_inf_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


def segment_max(src: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment-wise max for 1D tensors with argmax indices."""
    if src.numel() == 0:
        max_per = torch.full((num_segments,), neg_inf_value(src), device=src.device, dtype=src.dtype)
        argmax = torch.zeros((num_segments,), device=src.device, dtype=torch.long)
        return max_per, argmax

    segment_ids = segment_ids.to(device=src.device, dtype=torch.long).view(-1)
    max_per = torch.full((num_segments,), neg_inf_value(src), device=src.device, dtype=src.dtype)
    max_per.scatter_reduce_(0, segment_ids, src, reduce="amax", include_self=True)

    positions = torch.arange(src.numel(), device=src.device, dtype=torch.long)
    is_max = src == max_per.index_select(0, segment_ids)
    sentinel = src.numel()
    candidate = torch.where(is_max, positions, torch.full_like(positions, sentinel))
    argmin = torch.full((num_segments,), sentinel, device=src.device, dtype=torch.long)
    argmin.scatter_reduce_(0, segment_ids, candidate, reduce="amin", include_self=True)
    argmax = torch.where(argmin == sentinel, torch.zeros_like(argmin), argmin)
    return max_per, argmax


def segment_logsumexp_1d(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.full((num_segments,), neg_inf_value(logits), device=logits.device, dtype=logits.dtype)

    device = logits.device
    calc_dtype = logits.dtype
    neg_inf = torch.finfo(calc_dtype).min
    max_per = torch.full((num_segments,), neg_inf, device=device, dtype=calc_dtype)
    max_per.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    shifted = logits - max_per[segment_ids]
    exp = torch.exp(shifted)
    sum_per = torch.zeros((num_segments,), device=device, dtype=calc_dtype)
    sum_per.index_add_(0, segment_ids, exp)
    eps = torch.finfo(calc_dtype).eps
    return torch.log(sum_per.clamp(min=eps)) + max_per


def gumbel_noise_like(tensor: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(tensor)
    return -torch.log(-torch.log(u.clamp(min=_GUMBEL_EPS, max=1.0 - _GUMBEL_EPS)))


@dataclass(frozen=True)
class OutgoingEdges:
    edge_ids: torch.Tensor
    edge_batch: torch.Tensor
    edge_counts: torch.Tensor
    has_edge: torch.Tensor


def gather_outgoing_edges(
    *,
    curr_nodes: torch.Tensor,
    edge_ids_by_head: torch.Tensor,
    edge_ptr_by_head: torch.Tensor,
    active_mask: torch.Tensor,
) -> OutgoingEdges:
    curr_nodes = curr_nodes.to(device=edge_ptr_by_head.device, dtype=torch.long).view(-1)
    active_mask = active_mask.to(device=curr_nodes.device, dtype=torch.bool).view(-1)
    num_graphs = int(curr_nodes.numel())
    if num_graphs == _ZERO:
        empty = torch.zeros((_ZERO,), device=curr_nodes.device, dtype=torch.long)
        return OutgoingEdges(edge_ids=empty, edge_batch=empty, edge_counts=empty, has_edge=empty.to(dtype=torch.bool))
    active_mask = active_mask & (curr_nodes >= _ZERO)
    safe_nodes = torch.where(active_mask, curr_nodes, torch.zeros_like(curr_nodes))
    offsets = edge_ptr_by_head.index_select(0, safe_nodes)
    next_offsets = edge_ptr_by_head.index_select(0, safe_nodes + _ONE)
    counts = (next_offsets - offsets).to(dtype=torch.long)
    counts = torch.where(active_mask, counts, torch.zeros_like(counts))
    has_edge = counts > _ZERO
    graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=curr_nodes.device), counts)
    offset_rep = torch.repeat_interleave(offsets, counts)
    starts = counts.cumsum(0) - counts
    start_rep = torch.repeat_interleave(starts, counts)
    local_pos = torch.arange(offset_rep.size(0), device=curr_nodes.device, dtype=offset_rep.dtype) - start_rep
    edge_pos = offset_rep + local_pos
    edge_ids = edge_ids_by_head.index_select(0, edge_pos)
    return OutgoingEdges(edge_ids=edge_ids, edge_batch=graph_ids, edge_counts=counts, has_edge=has_edge)




def build_edge_head_csr(
    *,
    edge_index: torch.Tensor,
    num_nodes_total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == _ZERO:
        edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
        edge_ids = torch.zeros((_ZERO,), device=device, dtype=torch.long)
        return edge_ids, edge_ptr
    heads = edge_index[_EDGE_HEAD_INDEX].to(device=device, dtype=torch.long).view(-1)
    counts = torch.bincount(heads, minlength=int(num_nodes_total)).to(device=device, dtype=torch.long)
    edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
    edge_ptr[_ONE:] = counts.cumsum(0)
    edge_ids = torch.argsort(heads)
    return edge_ids, edge_ptr


def build_edge_head_csr_from_mask(
    *,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes_total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_mask = edge_mask.to(device=edge_index.device, dtype=torch.bool).view(-1)
    edge_ids = edge_mask.nonzero(as_tuple=False).view(-1)
    if edge_ids.numel() == _ZERO:
        edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
        empty = torch.zeros((_ZERO,), device=device, dtype=torch.long)
        return empty, edge_ptr
    heads = edge_index[_EDGE_HEAD_INDEX].index_select(0, edge_ids).to(device=device, dtype=torch.long)
    counts = torch.bincount(heads, minlength=int(num_nodes_total)).to(device=device, dtype=torch.long)
    edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
    edge_ptr[_ONE:] = counts.cumsum(0)
    order = torch.argsort(heads)
    edge_ids = edge_ids.index_select(0, order)
    return edge_ids, edge_ptr


def build_edge_tail_csr(
    *,
    edge_index: torch.Tensor,
    num_nodes_total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == _ZERO:
        edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
        edge_ids = torch.zeros((_ZERO,), device=device, dtype=torch.long)
        return edge_ids, edge_ptr
    tails = edge_index[_EDGE_HEAD_INDEX + _ONE].to(device=device, dtype=torch.long).view(-1)
    counts = torch.bincount(tails, minlength=int(num_nodes_total)).to(device=device, dtype=torch.long)
    edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
    edge_ptr[_ONE:] = counts.cumsum(0)
    edge_ids = torch.argsort(tails)
    return edge_ids, edge_ptr


def build_edge_tail_csr_from_mask(
    *,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes_total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_mask = edge_mask.to(device=edge_index.device, dtype=torch.bool).view(-1)
    edge_ids = edge_mask.nonzero(as_tuple=False).view(-1)
    if edge_ids.numel() == _ZERO:
        edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
        empty = torch.zeros((_ZERO,), device=device, dtype=torch.long)
        return empty, edge_ptr
    tails = edge_index[_EDGE_TAIL_INDEX].index_select(0, edge_ids).to(device=device, dtype=torch.long)
    counts = torch.bincount(tails, minlength=int(num_nodes_total)).to(device=device, dtype=torch.long)
    edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
    edge_ptr[_ONE:] = counts.cumsum(0)
    order = torch.argsort(tails)
    edge_ids = edge_ids.index_select(0, order)
    return edge_ids, edge_ptr


__all__ = [
    "OutgoingEdges",
    "build_edge_head_csr_from_mask",
    "build_edge_tail_csr_from_mask",
    "build_edge_head_csr",
    "build_edge_tail_csr",
    "gather_outgoing_edges",
    "gumbel_noise_like",
    "neg_inf_value",
    "segment_max",
    "segment_logsumexp_1d",
]
