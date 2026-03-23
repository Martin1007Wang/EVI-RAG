from __future__ import annotations

import torch

from src.graph import TrajectoryBatch

from .sampler import TrajectoryGFNSampleBatch

_SUCCESS_PATH_HASH_SEED_A = 1469598103934665603
_SUCCESS_PATH_HASH_SEED_B = 7809847782465536322
_SUCCESS_PATH_HASH_MUL_A = 1099511628211
_SUCCESS_PATH_HASH_MUL_B = 1402946736689701973
_SUCCESS_PATH_HASH_COL_SALT = 1000003


def compute_edge_offsets(batch: TrajectoryBatch) -> torch.Tensor:
    edge_counts = torch.bincount(batch.edge_batch, minlength=batch.num_graphs)
    return edge_counts.cumsum(0) - edge_counts


def collect_success_rollout_key_rows(
    *,
    batch: TrajectoryBatch,
    sample_batch: TrajectoryGFNSampleBatch,
) -> torch.Tensor | None:
    success_positions = torch.nonzero(sample_batch.success_mask, as_tuple=False)
    if int(success_positions.numel()) == 0:
        return None
    graph_indices = success_positions[:, 0]
    rollout_indices = success_positions[:, 1]
    node_offsets = batch.node_ptr[:-1].index_select(0, graph_indices)
    edge_offsets = compute_edge_offsets(batch).index_select(0, graph_indices)
    start_local_nodes = (
        sample_batch.start_nodes[graph_indices, rollout_indices] - node_offsets
    )
    terminal_local_nodes = (
        sample_batch.terminal_nodes[graph_indices, rollout_indices] - node_offsets
    )
    terminal_num_steps = sample_batch.terminal_num_steps[graph_indices, rollout_indices]
    trace_edge_ids = sample_batch.trace_edge_ids[graph_indices, rollout_indices]
    local_trace_edge_ids = torch.where(
        trace_edge_ids >= 0,
        trace_edge_ids - edge_offsets.unsqueeze(1),
        torch.full_like(trace_edge_ids, fill_value=-1),
    )
    return torch.cat(
        (
            graph_indices.unsqueeze(1),
            start_local_nodes.unsqueeze(1),
            terminal_local_nodes.unsqueeze(1),
            terminal_num_steps.unsqueeze(1),
            local_trace_edge_ids,
        ),
        dim=1,
    )


def deduplicate_success_rollout_key_rows(
    success_path_rows: torch.Tensor | None,
) -> torch.Tensor | None:
    if success_path_rows is None or int(success_path_rows.numel()) == 0:
        return None
    return torch.unique(success_path_rows, dim=0)


def compute_success_path_hash_pairs(success_path_rows: torch.Tensor) -> torch.Tensor:
    if success_path_rows.dim() != 2:
        raise ValueError(
            "compute_success_path_hash_pairs expects a 2D tensor of path rows. "
            f"Got shape={tuple(success_path_rows.shape)}."
        )
    if int(success_path_rows.size(0)) == 0:
        return success_path_rows.new_empty((0, 2), dtype=torch.long)

    normalized_rows = success_path_rows.to(dtype=torch.long) + 2
    num_rows = int(normalized_rows.size(0))
    hash_a = torch.full(
        (num_rows,),
        fill_value=_SUCCESS_PATH_HASH_SEED_A,
        device=success_path_rows.device,
        dtype=torch.long,
    )
    hash_b = torch.full(
        (num_rows,),
        fill_value=_SUCCESS_PATH_HASH_SEED_B,
        device=success_path_rows.device,
        dtype=torch.long,
    )
    for col_idx in range(int(normalized_rows.size(1))):
        column = normalized_rows[:, col_idx] + (
            (col_idx + 1) * _SUCCESS_PATH_HASH_COL_SALT
        )
        hash_a = (hash_a * _SUCCESS_PATH_HASH_MUL_A) + column
        hash_b = (hash_b * _SUCCESS_PATH_HASH_MUL_B) + (column * ((2 * col_idx) + 1))
    return torch.stack((hash_a, hash_b), dim=1)


__all__ = [
    "collect_success_rollout_key_rows",
    "compute_edge_offsets",
    "compute_success_path_hash_pairs",
    "deduplicate_success_rollout_key_rows",
]
