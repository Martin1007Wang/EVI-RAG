from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import torch

from .subgraph import SubgraphReconstructor, UnionSubgraphMasks

if TYPE_CHECKING:
    from src.data.schema import RetrievalBatch

    from .trace import RolloutTrace


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """
    Immutable finalized rollout result.

    Shape convention:
        R: number of rollout rows
        T: maximum decision steps

        source_graph_id:      [R]

        traj_len:             [R]
        terminal_step:        [R]
        terminal_stop_log_prob: [R]

        valid_mask:           [R, T]
        expand_mask:          [R, T]
        stop_mask:            [R, T]
        forced_stop_mask:     [R, T]

        action_type:          [R, T]
        selected_edge_ids:    [R, T]
    """

    source_graph_id: torch.Tensor

    traj_len: torch.Tensor
    terminal_step: torch.Tensor
    terminal_stop_log_prob: torch.Tensor

    valid_mask: torch.Tensor
    expand_mask: torch.Tensor
    stop_mask: torch.Tensor
    forced_stop_mask: torch.Tensor

    action_type: torch.Tensor
    selected_edge_ids: torch.Tensor

    expand_budget: int | None = None

    def __post_init__(self) -> None:
        self._check_vector(self.source_graph_id, "source_graph_id", dtype=torch.long)
        self._check_vector(self.traj_len, "traj_len", dtype=torch.long)
        self._check_vector(self.terminal_step, "terminal_step", dtype=torch.long)
        self._check_vector(
            self.terminal_stop_log_prob,
            "terminal_stop_log_prob",
            floating=True,
        )

        self._check_matrix(self.valid_mask, "valid_mask", dtype=torch.bool)
        self._check_matrix(self.expand_mask, "expand_mask", dtype=torch.bool)
        self._check_matrix(self.stop_mask, "stop_mask", dtype=torch.bool)
        self._check_matrix(
            self.forced_stop_mask,
            "forced_stop_mask",
            dtype=torch.bool,
        )

        self._check_matrix(self.action_type, "action_type", dtype=torch.long)
        self._check_matrix(
            self.selected_edge_ids,
            "selected_edge_ids",
            dtype=torch.long,
        )

        num_rows = int(self.traj_len.numel())
        max_steps = int(self.valid_mask.size(1))
        expand_budget = max_steps - 1 if self.expand_budget is None else int(self.expand_budget)
        if expand_budget < 0:
            raise ValueError(f"expand_budget must be non-negative, got {expand_budget}.")
        if max_steps != expand_budget + 1:
            raise ValueError(
                "RolloutResult max_steps must equal expand_budget + 1: "
                f"{max_steps} != {expand_budget + 1}."
            )
        object.__setattr__(self, "expand_budget", expand_budget)

        for name, tensor in self._vector_fields().items():
            if tuple(tensor.shape) != (num_rows,):
                raise ValueError(f"{name} must have shape [{num_rows}], got {tuple(tensor.shape)}.")

        for name, tensor in self._matrix_fields().items():
            if tuple(tensor.shape) != (num_rows, max_steps):
                raise ValueError(f"{name} must have shape [{num_rows}, {max_steps}], " f"got {tuple(tensor.shape)}.")

        self._check_same_device()

    @classmethod
    def from_trace(
        cls,
        *,
        trace: RolloutTrace,
        source_graph_id: torch.Tensor,
        expand_budget: int,
    ) -> RolloutResult:
        """
        Freeze a completed RolloutTrace into a RolloutResult.

        Rollout results intentionally contain no differentiable policy outputs.
        """
        trace.assert_complete()

        return cls(
            source_graph_id=source_graph_id,
            traj_len=trace.traj_len,
            terminal_step=trace.terminal_step,
            terminal_stop_log_prob=trace.terminal_stop_log_prob,
            valid_mask=trace.valid_mask,
            expand_mask=trace.expand_mask,
            stop_mask=trace.stop_mask,
            forced_stop_mask=trace.forced_stop_mask,
            action_type=trace.action_type,
            selected_edge_ids=trace.selected_edge_ids,
            expand_budget=expand_budget,
        )

    @property
    def num_rollouts(self) -> int:
        return int(self.traj_len.numel())

    @property
    def max_steps(self) -> int:
        return int(self.valid_mask.size(1))

    @property
    def device(self) -> torch.device:
        return self.valid_mask.device

    @property
    def selected_edges(self) -> torch.Tensor:
        return self.selected_edge_ids

    def select_rows(self, rows: torch.Tensor) -> RolloutResult:
        rows = self._checked_rows(rows)

        return RolloutResult(
            source_graph_id=self.source_graph_id.index_select(0, rows),
            traj_len=self.traj_len.index_select(0, rows),
            terminal_step=self.terminal_step.index_select(0, rows),
            terminal_stop_log_prob=self.terminal_stop_log_prob.index_select(0, rows),
            valid_mask=self.valid_mask.index_select(0, rows),
            expand_mask=self.expand_mask.index_select(0, rows),
            stop_mask=self.stop_mask.index_select(0, rows),
            forced_stop_mask=self.forced_stop_mask.index_select(0, rows),
            action_type=self.action_type.index_select(0, rows),
            selected_edge_ids=self.selected_edge_ids.index_select(0, rows),
            expand_budget=self.expand_budget,
        )

    def detach_to(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> RolloutResult:
        """
        Return a detached copy, optionally moved to device.
        """
        if device is None:
            return self._map_tensors(lambda x: x.detach())

        return self._map_tensors(lambda x: x.detach().to(device=device))

    def to(
        self,
        *,
        device: torch.device | str,
    ) -> RolloutResult:
        """
        Return a copy moved to device without detaching autograd history.
        """
        return self._map_tensors(lambda x: x.to(device=device))

    def _map_tensors(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> RolloutResult:
        return RolloutResult(
            source_graph_id=fn(self.source_graph_id),
            traj_len=fn(self.traj_len),
            terminal_step=fn(self.terminal_step),
            terminal_stop_log_prob=fn(self.terminal_stop_log_prob),
            valid_mask=fn(self.valid_mask),
            expand_mask=fn(self.expand_mask),
            stop_mask=fn(self.stop_mask),
            forced_stop_mask=fn(self.forced_stop_mask),
            action_type=fn(self.action_type),
            selected_edge_ids=fn(self.selected_edge_ids),
            expand_budget=self.expand_budget,
        )

    def _checked_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.device != self.device:
            raise ValueError(f"rows must be on {self.device}, got {rows.device}.")
        if rows.dtype != torch.long:
            raise ValueError(f"rows must be torch.long, got {rows.dtype}.")
        if rows.ndim != 1:
            raise ValueError(f"rows must have shape [M], got {tuple(rows.shape)}.")

        if rows.numel() > 0:
            bad = rows.lt(0) | rows.ge(self.num_rollouts)
            if bool(bad.any()):
                bad_rows = rows[bad]
                raise ValueError(
                    "rows contains ids outside rollout rows: "
                    f"min_bad={int(bad_rows.min())}, "
                    f"max_bad={int(bad_rows.max())}, "
                    f"num_rollouts={self.num_rollouts}."
                )

        return rows

    def _vector_fields(self) -> dict[str, torch.Tensor]:
        return {
            "source_graph_id": self.source_graph_id,
            "traj_len": self.traj_len,
            "terminal_step": self.terminal_step,
            "terminal_stop_log_prob": self.terminal_stop_log_prob,
        }

    def _matrix_fields(self) -> dict[str, torch.Tensor]:
        return {
            "valid_mask": self.valid_mask,
            "expand_mask": self.expand_mask,
            "stop_mask": self.stop_mask,
            "forced_stop_mask": self.forced_stop_mask,
            "action_type": self.action_type,
            "selected_edge_ids": self.selected_edge_ids,
        }

    @staticmethod
    def _check_vector(
        tensor: torch.Tensor,
        name: str,
        *,
        dtype: torch.dtype | None = None,
        floating: bool = False,
    ) -> None:
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape [R], got {tuple(tensor.shape)}.")

        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}.")

        if floating and not tensor.dtype.is_floating_point:
            raise ValueError(f"{name} must be floating point, got {tensor.dtype}.")

    @staticmethod
    def _check_matrix(
        tensor: torch.Tensor,
        name: str,
        *,
        dtype: torch.dtype | None = None,
        floating: bool = False,
    ) -> None:
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [R, T], got {tuple(tensor.shape)}.")

        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}.")

        if floating and not tensor.dtype.is_floating_point:
            raise ValueError(f"{name} must be floating point, got {tensor.dtype}.")

    def _check_same_device(self) -> None:
        device = self.device

        for name, tensor in self._vector_fields().items():
            if tensor.device != device:
                raise ValueError(f"{name} must be on {device}, got {tensor.device}.")

        for name, tensor in self._matrix_fields().items():
            if tensor.device != device:
                raise ValueError(f"{name} must be on {device}, got {tensor.device}.")

    def split_by_rollout_id(
        self,
        *,
        rollouts_per_graph: int,
    ) -> list[RolloutResult]:
        """
        Split a fused rollout result by rollout id.

        Fused row layout:
            row = graph_id * rollouts_per_graph + rollout_id

        Returns:
            list length = rollouts_per_graph.
            Each item contains one rollout per graph.
        """
        rollouts_per_graph = int(rollouts_per_graph)
        if rollouts_per_graph < 1:
            raise ValueError(f"rollouts_per_graph must be positive, got {rollouts_per_graph}.")

        if self.num_rollouts % rollouts_per_graph != 0:
            raise ValueError("num_rollouts must be divisible by rollouts_per_graph: " f"{self.num_rollouts} % {rollouts_per_graph} != 0.")

        graph_ids = torch.arange(
            self.num_rollouts // rollouts_per_graph,
            dtype=torch.long,
            device=self.device,
        )

        return [self.select_rows(graph_ids * rollouts_per_graph + rollout_id) for rollout_id in range(rollouts_per_graph)]

    def terminal_subgraph_mask(
        self,
        batch: RetrievalBatch,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deprecated. Use SubgraphReconstructor(batch, device=device).reconstruct().
        """
        warnings.warn(
            "RolloutResult.terminal_subgraph_mask is deprecated; "
            "use SubgraphReconstructor.reconstruct().",
            DeprecationWarning,
            stacklevel=2,
        )
        return SubgraphReconstructor(batch, device=device).reconstruct(self)

    @staticmethod
    def stack_terminal_subgraph_masks(
        rollouts: Sequence[RolloutResult],
        batch: RetrievalBatch,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deprecated. Use SubgraphReconstructor(batch, device=device).stack().
        """
        device = _default_mask_device() if device is None else device
        warnings.warn(
            "RolloutResult.stack_terminal_subgraph_masks is deprecated; "
            "use SubgraphReconstructor.stack().",
            DeprecationWarning,
            stacklevel=2,
        )
        return SubgraphReconstructor(batch, device=device).stack(rollouts)

    @staticmethod
    def union_terminal_subgraph_masks(
        rollouts: Sequence[RolloutResult],
        batch: RetrievalBatch,
        *,
        device: torch.device | None = None,
    ) -> UnionSubgraphMasks:
        """
        Deprecated. Use SubgraphReconstructor(batch, device=device).union().
        """
        device = _default_mask_device() if device is None else device
        warnings.warn(
            "RolloutResult.union_terminal_subgraph_masks is deprecated; "
            "use SubgraphReconstructor.union().",
            DeprecationWarning,
            stacklevel=2,
        )
        return SubgraphReconstructor(batch, device=device).union(rollouts)

    @classmethod
    def concat(
        cls,
        rollouts: Sequence[RolloutResult],
    ) -> RolloutResult:
        """
        Concatenate rollout results along rollout-row dimension.
        """
        if not rollouts:
            raise ValueError("Cannot concatenate an empty rollout sequence.")

        if len(rollouts) == 1:
            return rollouts[0]

        expand_budget = int(rollouts[0].expand_budget)
        for rollout in rollouts[1:]:
            if int(rollout.expand_budget) != expand_budget:
                raise ValueError(
                    "Cannot concatenate rollout results with different expand_budget values."
                )

        return cls(
            source_graph_id=torch.cat([x.source_graph_id for x in rollouts], dim=0),
            traj_len=torch.cat([x.traj_len for x in rollouts], dim=0),
            terminal_step=torch.cat([x.terminal_step for x in rollouts], dim=0),
            terminal_stop_log_prob=torch.cat(
                [x.terminal_stop_log_prob for x in rollouts],
                dim=0,
            ),
            valid_mask=torch.cat([x.valid_mask for x in rollouts], dim=0),
            expand_mask=torch.cat([x.expand_mask for x in rollouts], dim=0),
            stop_mask=torch.cat([x.stop_mask for x in rollouts], dim=0),
            forced_stop_mask=torch.cat([x.forced_stop_mask for x in rollouts], dim=0),
            action_type=torch.cat([x.action_type for x in rollouts], dim=0),
            selected_edge_ids=torch.cat([x.selected_edge_ids for x in rollouts], dim=0),
            expand_budget=expand_budget,
        )


def _default_mask_device() -> torch.device:
    return torch.device("cpu")


__all__ = [
    "RolloutResult",
    "UnionSubgraphMasks",
]
