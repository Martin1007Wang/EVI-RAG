from __future__ import annotations

from dataclasses import dataclass

import torch

POLICY_STOP = 0
NO_FRONTIER = 1
BUDGET = 2

SRC_POLICY = 0
SRC_REPLAY = 1


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """
    Completed rollout trajectories.

    Truth source:
    - graph_ids[t] gives the batch-local graph id of trajectory t.
    - edge_ids[t, :edge_count[t]] gives the selected physical KG edges.
    - edge_ids[t, edge_count[t]:] is padding and should be -1 by construction.
    - edge_logp[t, k] is the policy log-prob of edge_ids[t, k] for policy
      trajectories; replay builders may fill zeros.
    - stop_logp[t] is the policy log-prob of STOP only when STOP was sampled
      by the policy; boundary stops may fill zero.
    - source[t] identifies whether the trajectory came from policy or replay.

    This object is a record, not a state machine.
    State reconstruction belongs in rollout/transition utilities.
    """

    graph_ids: torch.Tensor  # [T]
    edge_ids: torch.Tensor  # [T, B]
    edge_logp: torch.Tensor  # [T, B]
    edge_count: torch.Tensor  # [T]
    stop_reason: torch.Tensor  # [T]
    stop_logp: torch.Tensor  # [T]
    source: torch.Tensor  # [T]

    @property
    def device(self) -> torch.device:
        return self.graph_ids.device

    @property
    def num_trajectories(self) -> int:
        return int(self.graph_ids.numel())

    @property
    def budget(self) -> int:
        return int(self.edge_ids.size(1))

    def valid_edge_mask(self) -> torch.Tensor:
        steps = torch.arange(
            self.budget,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        return steps.lt(self.edge_count.unsqueeze(1))

    def select_rows(self, rows: torch.Tensor) -> TrajectoryBatch:
        rows = rows.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)

        return TrajectoryBatch(
            graph_ids=self.graph_ids.index_select(0, rows),
            edge_ids=self.edge_ids.index_select(0, rows),
            edge_logp=self.edge_logp.index_select(0, rows),
            edge_count=self.edge_count.index_select(0, rows),
            stop_reason=self.stop_reason.index_select(0, rows),
            stop_logp=self.stop_logp.index_select(0, rows),
            source=self.source.index_select(0, rows),
        )

    @classmethod
    def empty(
        cls,
        *,
        device: torch.device,
        budget: int,
    ) -> TrajectoryBatch:
        budget = int(budget)
        if budget < 0:
            raise ValueError("budget must be nonnegative.")

        edge_ids = torch.empty(
            (0, budget),
            dtype=torch.long,
            device=device,
        )

        return cls(
            graph_ids=torch.empty(0, dtype=torch.long, device=device),
            edge_ids=edge_ids,
            edge_logp=torch.empty((0, budget), dtype=torch.float32, device=device),
            edge_count=torch.empty(0, dtype=torch.long, device=device),
            stop_reason=torch.empty(0, dtype=torch.long, device=device),
            stop_logp=torch.empty(0, dtype=torch.float32, device=device),
            source=torch.empty(0, dtype=torch.long, device=device),
        )

    @classmethod
    def concat(cls, batches: list[TrajectoryBatch]) -> TrajectoryBatch:
        if not batches:
            raise ValueError("Cannot concatenate an empty trajectory sequence.")

        non_empty = [batch for batch in batches if batch.num_trajectories > 0]

        if not non_empty:
            first = batches[0]
            return cls.empty(
                device=first.device,
                budget=first.budget,
            )

        first = non_empty[0]

        for batch in non_empty[1:]:
            if int(batch.budget) != int(first.budget):
                raise ValueError("Cannot concatenate trajectories with different budgets.")
            if batch.device != first.device:
                raise ValueError("Cannot concatenate trajectories on different devices.")

        return cls(
            graph_ids=torch.cat([batch.graph_ids for batch in non_empty], dim=0),
            edge_ids=torch.cat([batch.edge_ids for batch in non_empty], dim=0),
            edge_logp=torch.cat([batch.edge_logp for batch in non_empty], dim=0),
            edge_count=torch.cat([batch.edge_count for batch in non_empty], dim=0),
            stop_reason=torch.cat([batch.stop_reason for batch in non_empty], dim=0),
            stop_logp=torch.cat([batch.stop_logp for batch in non_empty], dim=0),
            source=torch.cat([batch.source for batch in non_empty], dim=0),
        )


def trajectory_logp(trajectories: TrajectoryBatch) -> torch.Tensor:
    """
    Return trajectory-level policy log-prob.

    This is intentionally a function, not a TrajectoryBatch property.
    Evaluation/scoring code may use it, but the data structure should remain
    a record.
    """

    edge_logp = trajectories.edge_logp.masked_fill(
        ~trajectories.valid_edge_mask(),
        0.0,
    ).sum(dim=1)

    return edge_logp + trajectories.stop_logp


def _check_1d_long(value: torch.Tensor, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)}.")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long.")


def _check_2d_long(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [N, M], got {tuple(value.shape)}.")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long.")


def _check_1d_float(value: torch.Tensor, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)}.")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must have floating dtype.")


def _check_2d_float(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [N, M], got {tuple(value.shape)}.")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must have floating dtype.")


__all__ = [
    "BUDGET",
    "NO_FRONTIER",
    "POLICY_STOP",
    "SRC_POLICY",
    "SRC_REPLAY",
    "TrajectoryBatch",
    "trajectory_logp",
]
