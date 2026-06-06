from __future__ import annotations

from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------- #
#  Terminal-kind codes                                                         #
# --------------------------------------------------------------------------- #
POLICY_STOP: int = 0
NO_FRONTIER: int = 1
BUDGET_TRUNCATED: int = 2
EXTERNAL_TERMINAL: int = 3

# --------------------------------------------------------------------------- #
#  Source codes                                                                #
# --------------------------------------------------------------------------- #
SRC_POLICY: int = 0
SRC_REPLAY: int = 1

# Canonical dtypes — change here to propagate everywhere
_LONG = torch.long  # graph_ids, edge_ids, edge_count
_FLOAT = torch.float32  # edge_logp, stop_logp  (swap to bfloat16 if needed)
_REASON = torch.uint8  # terminal kind: 4 values → uint8 saves space
_SOURCE = torch.bool  # source: 0=policy, 1=replay → bool is 1 byte/elem


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """
    Completed rollout trajectories — a flat tensor record.

    Layout
    ------
    graph_ids  [T]      batch-local graph id for each trajectory
    edge_ids   [T, B]   physical KG edge indices; padding cells are -1
    edge_logp  [T, B]   policy log-prob per edge step; padding cells are 0.0
    edge_count [T]      number of valid (non-padding) edge steps
    stop_reason[T]      uint8  — POLICY_STOP / NO_FRONTIER / BUDGET_TRUNCATED / EXTERNAL_TERMINAL
    stop_logp  [T]      log-prob of the sampled STOP token; 0.0 for forced terminals
    source     [T]      bool   — False = SRC_POLICY, True = SRC_REPLAY

    Every row is a completed trajectory. stop_reason explains why it terminated.
    POLICY_STOP, BUDGET_TRUNCATED, and EXTERNAL_TERMINAL contribute trainable STOP terms.
    NO_FRONTIER is structural-only.

    This object is a record, not a state machine.
    State reconstruction belongs in rollout/transition utilities.
    """

    graph_ids: torch.Tensor  # [T]   long
    edge_ids: torch.Tensor  # [T,B] long,  padding = -1
    edge_logp: torch.Tensor  # [T,B] float, padding = 0.0
    edge_count: torch.Tensor  # [T]   long
    stop_reason: torch.Tensor  # [T]   uint8
    stop_logp: torch.Tensor  # [T]   float
    source: torch.Tensor  # [T]   bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _check_1d_long(self.graph_ids, "graph_ids")
        _check_2d_long(self.edge_ids, "edge_ids")
        _check_2d_float(self.edge_logp, "edge_logp")
        _check_1d_long(self.edge_count, "edge_count")
        _check_1d_uint8(self.stop_reason, "stop_reason")
        _check_1d_float(self.stop_logp, "stop_logp")
        _check_1d_bool(self.source, "source")

        num_trajectories = int(self.graph_ids.numel())
        budget = int(self.edge_ids.size(1))
        if self.edge_ids.shape != (num_trajectories, budget):
            raise ValueError("edge_ids must have shape [num_trajectories, budget].")
        if self.edge_logp.shape != self.edge_ids.shape:
            raise ValueError("edge_logp must match edge_ids shape.")
        for name, value in (
            ("edge_count", self.edge_count),
            ("stop_reason", self.stop_reason),
            ("stop_logp", self.stop_logp),
            ("source", self.source),
        ):
            if int(value.numel()) != num_trajectories:
                raise ValueError(f"{name} must have one value per trajectory.")
            if value.device != self.device:
                raise ValueError(f"{name} must be on the same device as graph_ids.")
        if self.edge_ids.device != self.device or self.edge_logp.device != self.device:
            raise ValueError("edge_ids and edge_logp must be on the same device as graph_ids.")
        if bool(self.edge_count.lt(0).any()) or bool(self.edge_count.gt(budget).any()):
            raise ValueError("edge_count must be in [0, budget].")
        if bool(self.stop_reason.gt(int(EXTERNAL_TERMINAL)).any()):
            raise ValueError("stop_reason contains an unknown terminal kind.")

        valid = self.valid_edge_mask()
        if bool(self.edge_ids[valid].lt(0).any()):
            raise ValueError("valid edge_ids must be nonnegative.")
        if bool(self.edge_ids[~valid].ne(-1).any()):
            raise ValueError("padding edge_ids must be -1.")
        if bool(self.edge_logp[~valid].ne(0.0).any()):
            raise ValueError("padding edge_logp values must be 0.0.")

    # ---------------------------------------------------------------------- #
    #  Shape / device / dtype accessors                                       #
    # ---------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        return self.graph_ids.device

    @property
    def num_trajectories(self) -> int:
        return self.graph_ids.numel()

    @property
    def budget(self) -> int:
        return self.edge_ids.size(1)

    # Convenience: distinguish policy vs. replay without int() casts
    @property
    def is_policy(self) -> torch.Tensor:
        """Boolean mask [T], True where source == SRC_POLICY."""
        return ~self.source  # source is bool: False = policy

    @property
    def is_replay(self) -> torch.Tensor:
        """Boolean mask [T], True where source == SRC_REPLAY."""
        return self.source

    @property
    def is_policy_stop(self) -> torch.Tensor:
        """Boolean mask [T], True where the policy explicitly sampled STOP."""
        return self.stop_reason.eq(int(POLICY_STOP))

    @property
    def is_no_frontier(self) -> torch.Tensor:
        """Boolean mask [T], True where expansion was impossible before budget exhaustion."""
        return self.stop_reason.eq(int(NO_FRONTIER))

    @property
    def is_budget_truncated(self) -> torch.Tensor:
        """Boolean mask [T], True where the edge budget truncated the row."""
        return self.stop_reason.eq(int(BUDGET_TRUNCATED))

    @property
    def is_external_terminal(self) -> torch.Tensor:
        """Boolean mask [T], True where the terminal came from external replay data."""
        return self.stop_reason.eq(int(EXTERNAL_TERMINAL))

    @property
    def has_trainable_stop(self) -> torch.Tensor:
        """Boolean mask [T], True where terminal STOP should contribute to the objective."""
        return self.is_policy_stop | self.is_budget_truncated | self.is_external_terminal

    @property
    def has_terminal_reward(self) -> torch.Tensor:
        """Boolean mask [T], True where terminal reward should anchor the objective."""
        return self.is_policy_stop | self.is_no_frontier | self.is_budget_truncated | self.is_external_terminal

    @property
    def is_forced_terminal(self) -> torch.Tensor:
        """Boolean mask [T], True for terminals without a trainable STOP decision."""
        return self.is_no_frontier | self.is_budget_truncated | self.is_external_terminal

    @property
    def terminal_kind(self) -> torch.Tensor:
        """Alias for stop_reason to make terminal provenance explicit at call sites."""
        return self.stop_reason

    # ---------------------------------------------------------------------- #
    #  Masking                                                                #
    # ---------------------------------------------------------------------- #

    def valid_edge_mask(self) -> torch.Tensor:
        """[T, B] bool mask — True for non-padding edge positions."""
        steps = torch.arange(
            self.budget,
            dtype=_LONG,
            device=self.device,
        ).unsqueeze(0)
        return steps.lt(self.edge_count.unsqueeze(1))

    # ---------------------------------------------------------------------- #
    #  Selection / combination                                                #
    # ---------------------------------------------------------------------- #

    def select_rows(self, rows: torch.Tensor) -> TrajectoryBatch:
        rows = rows.to(device=self.device, dtype=_LONG).view(-1)
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
    def empty(cls, *, device: torch.device, budget: int) -> TrajectoryBatch:
        budget = int(budget)
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        return cls(
            graph_ids=torch.empty(0, dtype=_LONG, device=device),
            edge_ids=torch.full((0, budget), -1, dtype=_LONG, device=device),
            edge_logp=torch.empty((0, budget), dtype=_FLOAT, device=device),
            edge_count=torch.empty(0, dtype=_LONG, device=device),
            stop_reason=torch.empty(0, dtype=_REASON, device=device),
            stop_logp=torch.empty(0, dtype=_FLOAT, device=device),
            source=torch.empty(0, dtype=_SOURCE, device=device),
        )

    @classmethod
    def concat(cls, batches: list[TrajectoryBatch]) -> TrajectoryBatch:
        if not batches:
            raise ValueError("Cannot concatenate an empty list.")

        non_empty = [b for b in batches if b.num_trajectories > 0]
        if not non_empty:
            first = batches[0]
            return cls.empty(device=first.device, budget=first.budget)

        first = non_empty[0]
        for b in non_empty[1:]:
            if b.budget != first.budget:
                raise ValueError(f"Budget mismatch in concat: {b.budget} vs {first.budget}.")
            if b.device != first.device:
                raise ValueError(f"Device mismatch in concat: {b.device} vs {first.device}.")

        return cls(
            graph_ids=torch.cat([b.graph_ids for b in non_empty]),
            edge_ids=torch.cat([b.edge_ids for b in non_empty]),
            edge_logp=torch.cat([b.edge_logp for b in non_empty]),
            edge_count=torch.cat([b.edge_count for b in non_empty]),
            stop_reason=torch.cat([b.stop_reason for b in non_empty]),
            stop_logp=torch.cat([b.stop_logp for b in non_empty]),
            source=torch.cat([b.source for b in non_empty]),
        )


# --------------------------------------------------------------------------- #
#  Standalone scoring utility                                                  #
# --------------------------------------------------------------------------- #


def trajectory_logp(trajectories: TrajectoryBatch) -> torch.Tensor:
    """
    Trajectory-level policy log-prob  [T].

    sum of valid edge log-probs + stop log-prob.
    Intentionally a free function — scoring logic should not live on the
    data record.
    """
    edge_logp = trajectories.edge_logp.masked_fill(
        ~trajectories.valid_edge_mask(),
        0.0,
    ).sum(dim=1)
    return edge_logp + trajectories.stop_logp


# --------------------------------------------------------------------------- #
#  Internal shape / dtype validators (used by builders, not TrajectoryBatch)  #
# --------------------------------------------------------------------------- #


def _check_1d_long(value: torch.Tensor, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(value.shape)}.")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long.")


def _check_2d_long(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(value.shape)}.")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long.")


def _check_1d_float(value: torch.Tensor, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(value.shape)}.")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating-point.")


def _check_2d_float(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(value.shape)}.")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating-point.")


def _check_1d_uint8(value: torch.Tensor, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(value.shape)}.")
    if value.dtype != torch.uint8:
        raise TypeError(f"{name} must be torch.uint8.")


def _check_1d_bool(value: torch.Tensor, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(value.shape)}.")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must be torch.bool.")


__all__ = [
    "BUDGET_TRUNCATED",
    "EXTERNAL_TERMINAL",
    "NO_FRONTIER",
    "POLICY_STOP",
    "SRC_POLICY",
    "SRC_REPLAY",
    "TrajectoryBatch",
    "trajectory_logp",
]
