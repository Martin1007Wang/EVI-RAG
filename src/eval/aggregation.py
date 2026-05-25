from __future__ import annotations

import torch

from src.weaver.rollout.trajectory import TrajectoryBatch


def grouped_sample_ids(trajectories: TrajectoryBatch, *, num_graphs: int) -> torch.Tensor:
    if trajectories.num_trajectories == 0:
        return torch.empty(0, dtype=torch.long, device=trajectories.device)
    graph_ids = trajectories.graph_ids.to(dtype=torch.long)
    counts = torch.bincount(graph_ids, minlength=int(num_graphs))
    expected = torch.repeat_interleave(
        torch.arange(int(num_graphs), dtype=torch.long, device=trajectories.device),
        counts,
    )
    if expected.shape != graph_ids.shape or not bool(expected.eq(graph_ids).all()):
        raise ValueError("trajectory graph_ids must be grouped by graph; pass explicit sample_ids for mixed trajectories.")
    starts = torch.cumsum(counts, dim=0) - counts
    return torch.arange(
        trajectories.num_trajectories,
        dtype=torch.long,
        device=trajectories.device,
    ) - starts.index_select(0, graph_ids)


def trajectory_row_matrix(
    trajectories: TrajectoryBatch,
    values: torch.Tensor,
    *,
    num_graphs: int,
    fill_value: float = 0.0,
    sample_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    values = values.to(device=trajectories.device).view(-1)
    if int(values.numel()) != trajectories.num_trajectories:
        raise ValueError("values must match trajectory count.")
    if trajectories.num_trajectories == 0:
        return values.new_full((0, int(num_graphs)), fill_value)

    graph_ids = trajectories.graph_ids.to(dtype=torch.long)
    if sample_ids is None:
        sample_ids = grouped_sample_ids(trajectories, num_graphs=int(num_graphs))
    else:
        sample_ids = sample_ids.to(device=trajectories.device, dtype=torch.long).view(-1)
        if int(sample_ids.numel()) != trajectories.num_trajectories:
            raise ValueError("sample_ids must match trajectory count.")

    num_samples = int(sample_ids.max().item()) + 1 if sample_ids.numel() > 0 else 0
    out = values.new_full((num_samples, int(num_graphs)), fill_value)
    out[sample_ids, graph_ids] = values
    return out


def num_grouped_samples(trajectories: TrajectoryBatch, *, num_graphs: int) -> int:
    if trajectories.num_trajectories == 0:
        return 0
    counts = torch.bincount(trajectories.graph_ids.to(dtype=torch.long), minlength=int(num_graphs))
    return int(counts.max().item()) if counts.numel() > 0 else 0


__all__ = [
    "grouped_sample_ids",
    "num_grouped_samples",
    "trajectory_row_matrix",
]
