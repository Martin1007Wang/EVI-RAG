from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.graph import TrajectoryBatch

if TYPE_CHECKING:
    from .sampler import SubgraphTrajectorySampleBatch


@dataclass(frozen=True)
class SubgraphReplayRecord:
    trajectory_batch: TrajectoryBatch
    edge_ids: tuple[int, ...]
    source: str

    @property
    def signature(self) -> tuple[str, tuple[int, ...]]:
        sample_id = str(self.trajectory_batch.sample_ids[0])
        return sample_id, tuple(int(edge_id) for edge_id in self.edge_ids)


def _localize_edge_ids(
    *, edge_ids: tuple[int, ...], edge_start: int
) -> tuple[int, ...]:
    localized = tuple(int(edge_id) - int(edge_start) for edge_id in edge_ids)
    if any(int(edge_id) < 0 for edge_id in localized):
        raise RuntimeError(
            "Replay edge ids must be local to the selected graph after reindexing."
        )
    return localized


def _compress_replay_batch(batch: TrajectoryBatch) -> TrajectoryBatch:
    return batch.to(device="cpu", feature_dtype=torch.float16)


class SubgraphSuccessReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        deduplicate: bool,
    ) -> None:
        self.capacity = int(capacity)
        self.deduplicate = bool(deduplicate)
        if self.capacity < 1:
            raise ValueError("training.success_replay.capacity must be >= 1.")
        self._records: deque[SubgraphReplayRecord] = deque()
        self._signatures: set[tuple[str, tuple[int, ...]]] = set()

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: SubgraphReplayRecord) -> bool:
        signature = record.signature
        if self.deduplicate and signature in self._signatures:
            return False
        self._records.append(record)
        self._signatures.add(signature)
        while len(self._records) > self.capacity:
            evicted = self._records.popleft()
            self._signatures.discard(evicted.signature)
        return True

    def add_successful_trajectories(
        self,
        *,
        batch: TrajectoryBatch,
        sample_batch: "SubgraphTrajectorySampleBatch",
    ) -> int:
        if int(batch.num_graphs) != int(sample_batch.num_graphs):
            raise ValueError("Replay buffer batch/sample_batch graph count mismatch.")
        added = 0
        cached_graph_batches: dict[int, TrajectoryBatch] = {}
        for graph_idx in range(int(sample_batch.num_graphs)):
            success_mask = sample_batch.terminal_hit_mask[graph_idx].view(-1)
            if not bool(success_mask.any().item()):
                continue
            edge_start = int(batch.edge_ptr[graph_idx].item())
            graph_batch = cached_graph_batches.get(graph_idx)
            if graph_batch is None:
                graph_batch = _compress_replay_batch(batch.select_graph(graph_idx))
                cached_graph_batches[graph_idx] = graph_batch
            chosen_edges = sample_batch.chosen_edge_ids[graph_idx]
            for rollout_idx in (
                torch.nonzero(success_mask, as_tuple=False).view(-1).tolist()
            ):
                rollout_edges = chosen_edges[int(rollout_idx)]
                edge_ids = _localize_edge_ids(
                    edge_ids=tuple(
                        int(edge_id)
                        for edge_id in rollout_edges.detach().cpu().tolist()
                        if int(edge_id) >= 0
                    ),
                    edge_start=edge_start,
                )
                if not edge_ids:
                    continue
                if self.add(
                    SubgraphReplayRecord(
                        trajectory_batch=graph_batch,
                        edge_ids=edge_ids,
                        source="success",
                    )
                ):
                    added += 1
        return added

    def sample(self, *, max_records: int) -> list[SubgraphReplayRecord]:
        if max_records <= 0 or not self._records:
            return []
        num_records = min(int(max_records), len(self._records))
        permutation = torch.randperm(len(self._records), device="cpu")[
            :num_records
        ].tolist()
        records = list(self._records)
        return [records[int(index)] for index in permutation]


__all__ = [
    "SubgraphReplayRecord",
    "SubgraphSuccessReplayBuffer",
]
