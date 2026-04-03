from __future__ import annotations

from typing import Any

import torch


def require_tensor(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}.")
    return value


def require_1d_long(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = require_tensor(value, name=name, device=device)
    if tensor.dtype != torch.long or tensor.dim() != 1:
        raise ValueError(
            f"{name} must be 1D torch.long, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def require_2d_float(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = require_tensor(value, name=name, device=device)
    if not torch.is_floating_point(tensor) or tensor.dim() != 2:
        raise ValueError(
            f"{name} must be 2D floating point, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def require_3d_float(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = require_tensor(value, name=name, device=device)
    if not torch.is_floating_point(tensor) or tensor.dim() != 3:
        raise ValueError(
            f"{name} must be 3D floating point, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def require_bool_2d(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = require_tensor(value, name=name, device=device)
    if tensor.dtype != torch.bool or tensor.dim() != 2:
        raise ValueError(
            f"{name} must be 2D bool, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def move_float_feature(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if dtype is not None and torch.is_floating_point(tensor):
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def compact_relation_table(
    *,
    edge_rel_global: torch.Tensor,
    relation_embeddings: torch.Tensor,
    edge_rel_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rel_dim = int(relation_embeddings.size(-1))
    if int(edge_rel_local.numel()) == 0:
        return (
            edge_rel_global.new_empty((0,)),
            relation_embeddings.new_empty((0, rel_dim)),
            edge_rel_local.new_empty((0,)),
        )
    used_local_ids, compact_edge_rel_local = torch.unique(
        edge_rel_local, sorted=True, return_inverse=True
    )
    first_occ = torch.full(
        (int(used_local_ids.numel()),),
        fill_value=int(edge_rel_local.numel()),
        device=edge_rel_local.device,
        dtype=torch.long,
    )
    edge_ids = torch.arange(
        int(edge_rel_local.numel()), device=edge_rel_local.device, dtype=torch.long
    )
    first_occ.scatter_reduce_(
        0,
        compact_edge_rel_local,
        edge_ids,
        reduce="amin",
        include_self=True,
    )
    return (
        edge_rel_global.index_select(0, first_occ),
        relation_embeddings.index_select(0, used_local_ids),
        compact_edge_rel_local,
    )


def build_relation_table_from_rows(
    *,
    relation_global_ids: torch.Tensor,
    relation_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_dim = int(relation_embeddings.size(-1))
    if int(relation_global_ids.numel()) == 0:
        return relation_global_ids.new_empty((0,)), relation_embeddings.new_empty(
            (0, rel_dim)
        )
    unique_relation_ids, relation_inverse = torch.unique(
        relation_global_ids, sorted=True, return_inverse=True
    )
    first_occ = torch.full(
        (int(unique_relation_ids.numel()),),
        fill_value=int(relation_global_ids.numel()),
        device=relation_global_ids.device,
        dtype=torch.long,
    )
    relation_row_ids = torch.arange(
        int(relation_global_ids.numel()),
        device=relation_global_ids.device,
        dtype=torch.long,
    )
    first_occ.scatter_reduce_(
        0,
        relation_inverse,
        relation_row_ids,
        reduce="amin",
        include_self=True,
    )
    return unique_relation_ids, relation_embeddings.index_select(0, first_occ)


def require_edge_index(value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = require_tensor(value, name="edge_index", device=device)
    if tensor.dtype != torch.long or tensor.dim() != 2 or int(tensor.size(0)) != 2:
        raise ValueError(
            f"edge_index must be [2, E] torch.long, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def compute_edge_batch_and_ptr(
    edge_index: torch.Tensor,
    *,
    node_ptr: torch.Tensor,
    num_graphs: int,
    device: torch.device,
    validate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if node_ptr.numel() != num_graphs + 1:
        raise ValueError(
            f"node_ptr length mismatch: got {node_ptr.numel()} expected {num_graphs + 1}."
        )
    edge_batch = torch.bucketize(edge_index[0], node_ptr[1:], right=True)
    if validate:
        tail_batch = torch.bucketize(edge_index[1], node_ptr[1:], right=True)
        if edge_batch.numel() > 0:
            min_idx = int(edge_batch.min().item())
            max_idx = int(edge_batch.max().item())
            if min_idx < 0 or max_idx >= num_graphs:
                raise ValueError(
                    "edge_batch contains out-of-range indices; "
                    f"min={min_idx} max={max_idx} num_graphs={num_graphs}."
                )
        if not torch.equal(edge_batch, tail_batch):
            raise ValueError(
                "edge_index crosses graph boundaries; head/tail graph assignments differ."
            )
        if edge_batch.numel() > 1 and not bool(
            (edge_batch[:-1] <= edge_batch[1:]).all().item()
        ):
            raise ValueError(
                "edge_batch is not non-decreasing along the flattened edge list, "
                "which breaks per-graph slicing; ensure edges are concatenated per-graph."
            )
    edge_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
    edge_counts.scatter_add_(
        0, edge_batch, torch.ones_like(edge_batch, dtype=torch.long)
    )
    edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    edge_ptr[1:] = edge_counts.cumsum(0)
    return edge_batch, edge_ptr


def build_edge_ptr_from_edge_batch(
    edge_batch: torch.Tensor,
    *,
    num_graphs: int,
    device: torch.device,
    validate: bool,
) -> torch.Tensor:
    if edge_batch.dtype != torch.long or edge_batch.dim() != 1:
        raise ValueError(
            f"edge_batch must be 1D torch.long, got {edge_batch.dtype} {tuple(edge_batch.shape)}."
        )
    if int(edge_batch.numel()) == 0:
        return torch.zeros((num_graphs + 1,), dtype=torch.long, device=device)
    if validate:
        min_idx = int(edge_batch.min().item())
        max_idx = int(edge_batch.max().item())
        if min_idx < 0 or max_idx >= num_graphs:
            raise ValueError(
                "edge_batch contains out-of-range indices; "
                f"min={min_idx} max={max_idx} num_graphs={num_graphs}."
            )
        if edge_batch.numel() > 1 and not bool(
            (edge_batch[:-1] <= edge_batch[1:]).all().item()
        ):
            raise ValueError(
                "edge_batch is not non-decreasing along the flattened edge list, "
                "which breaks per-graph slicing; ensure edges are concatenated per-graph."
            )
    edge_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
    edge_counts.scatter_add_(
        0, edge_batch, torch.ones_like(edge_batch, dtype=torch.long)
    )
    edge_ptr = torch.zeros((num_graphs + 1,), dtype=torch.long, device=device)
    edge_ptr[1:] = edge_counts.cumsum(0)
    return edge_ptr


def coerce_str_list(value: Any, *, expected_size: int, name: str) -> list[str]:
    if value is None:
        return ["" for _ in range(expected_size)]
    if isinstance(value, str):
        if expected_size != 1:
            raise ValueError(
                f"{name} single string cannot represent {expected_size} items."
            )
        return [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be list/tuple[str], got {type(value)!r}.")
    values = [str(item or "") for item in value]
    if len(values) != expected_size:
        raise ValueError(
            f"{name} length mismatch: expected {expected_size}, got {len(values)}."
        )
    return values


__all__ = [
    "build_edge_ptr_from_edge_batch",
    "build_relation_table_from_rows",
    "coerce_str_list",
    "compact_relation_table",
    "compute_edge_batch_and_ptr",
    "move_float_feature",
    "require_1d_long",
    "require_2d_float",
    "require_3d_float",
    "require_bool_2d",
    "require_edge_index",
    "require_tensor",
]
