from __future__ import annotations

from dataclasses import dataclass, field

import torch

NO_ACTION = -1
EXPAND_ACTION = 0
STOP_ACTION = 1
FORCED_STOP_ACTION = 2

NO_EDGE = -1


@dataclass(slots=True)
class RolloutTrace:
    """
    Mutable trace for one fused vectorized rollout.

    This object records trajectory history. It does not control rollout,
    compute policy, compute reward, compute frontier, or compute loss.

    Shape convention:
        R: number of rollout rows
        T: maximum number of decision steps

    Stored tensors:
        selected_edge_ids[row, t] = selected edge id for expand action, else -1
        action_type[row, t]       = expand / stop / forced-stop / no-action

        valid_mask[row, t]        = whether row has a recorded action at t
        expand_mask[row, t]       = whether action is expand
        stop_mask[row, t]         = whether action is terminal
        forced_stop_mask[row, t]  = whether terminal action is forced

        terminal_stop_log_prob[row] = forward STOP log-prob at termination
        terminal_step[row]        = terminal step index
        traj_len[row]             = terminal_step + 1
        is_terminated[row]        = whether terminal action has been recorded
    """

    R: int
    T: int
    device: torch.device
    dtype: torch.dtype = torch.float32

    valid_mask: torch.Tensor = field(init=False)
    expand_mask: torch.Tensor = field(init=False)
    stop_mask: torch.Tensor = field(init=False)
    forced_stop_mask: torch.Tensor = field(init=False)

    action_type: torch.Tensor = field(init=False)
    selected_edge_ids: torch.Tensor = field(init=False)

    terminal_stop_log_prob: torch.Tensor = field(init=False)
    terminal_step: torch.Tensor = field(init=False)
    traj_len: torch.Tensor = field(init=False)
    is_terminated: torch.Tensor = field(init=False)

    state_written_mask: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.R = int(self.R)
        self.T = int(self.T)

        if self.R <= 0:
            raise ValueError(f"R must be positive, got {self.R}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")
        if not self.dtype.is_floating_point:
            raise ValueError(f"dtype must be floating point, got {self.dtype}.")

        rt = (self.R, self.T)
        r = (self.R,)

        self.valid_mask = torch.zeros(rt, dtype=torch.bool, device=self.device)
        self.expand_mask = torch.zeros(rt, dtype=torch.bool, device=self.device)
        self.stop_mask = torch.zeros(rt, dtype=torch.bool, device=self.device)
        self.forced_stop_mask = torch.zeros(rt, dtype=torch.bool, device=self.device)

        self.action_type = torch.full(
            rt,
            NO_ACTION,
            dtype=torch.long,
            device=self.device,
        )
        self.selected_edge_ids = torch.full(
            rt,
            NO_EDGE,
            dtype=torch.long,
            device=self.device,
        )

        self.terminal_stop_log_prob = torch.zeros(r, dtype=self.dtype, device=self.device)
        self.terminal_step = torch.full(
            r,
            NO_ACTION,
            dtype=torch.long,
            device=self.device,
        )
        self.traj_len = torch.zeros(r, dtype=torch.long, device=self.device)
        self.is_terminated = torch.zeros(r, dtype=torch.bool, device=self.device)

        self.state_written_mask = torch.zeros(rt, dtype=torch.bool, device=self.device)

    @property
    def selected_edges(self) -> torch.Tensor:
        return self.selected_edge_ids

    def write_state(
        self,
        *,
        t: int,
        rows: torch.Tensor,
    ) -> None:
        """
        Record that s_t was visited before sampling/executing action.
        """
        col = self._check_t(t)
        rows = self._rows(rows)

        self._check_rows_not_terminated(rows)

        self.state_written_mask[rows, col] = True

    def write_expand(
        self,
        *,
        t: int,
        rows: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> None:
        """
        Record expand action.

        Pairwise semantics:
            rows[i] expands edge_ids[i]
        """
        col = self._check_t(t)
        rows = self._rows(rows)
        edge_ids = self._long_vector(
            edge_ids,
            expected=rows.numel(),
            name="edge_ids",
        )

        self._check_rows_not_terminated(rows)
        self._check_state_written(rows, col)
        self._check_step_unwritten(rows, col)

        self.valid_mask[rows, col] = True
        self.expand_mask[rows, col] = True
        self.action_type[rows, col] = EXPAND_ACTION
        self.selected_edge_ids[rows, col] = edge_ids

    def write_terminal(
        self,
        *,
        t: int,
        rows: torch.Tensor,
        stop_log_prob: torch.Tensor,
        forced: torch.Tensor | None = None,
    ) -> None:
        """
        Record terminal action.

        The trace stores only the forward STOP log-probability.
        """
        col = self._check_t(t)
        rows = self._rows(rows)

        forced = self._optional_bool_vector(
            forced,
            expected=rows.numel(),
            name="forced",
        )
        stop_log_prob = self._float_vector(
            stop_log_prob,
            expected=rows.numel(),
            name="stop_log_prob",
        )

        self._check_rows_not_terminated(rows)
        self._check_state_written(rows, col)
        self._check_step_unwritten(rows, col)

        self.valid_mask[rows, col] = True
        self.stop_mask[rows, col] = True
        self.forced_stop_mask[rows, col] = forced

        self.action_type[rows, col] = torch.where(
            forced,
            torch.full_like(rows, FORCED_STOP_ACTION),
            torch.full_like(rows, STOP_ACTION),
        )

        self.terminal_stop_log_prob[rows] = stop_log_prob
        self.terminal_step[rows] = col
        self.traj_len[rows] = col + 1
        self.is_terminated[rows] = True

    def assert_complete(self) -> None:
        """
        Debug-only check.

        Every rollout row must terminate exactly once.
        """
        if not bool(self.is_terminated.all()):
            unfinished = (~self.is_terminated).nonzero(as_tuple=False).flatten()
            raise AssertionError(f"Unfinished rollout rows: {unfinished.tolist()}.")

        terminal_count = self.stop_mask.sum(dim=1)
        if not bool(terminal_count.eq(1).all()):
            bad = terminal_count.ne(1).nonzero(as_tuple=False).flatten()
            raise AssertionError("Each rollout row must have exactly one terminal action. " f"Bad rows: {bad.tolist()}.")

        bad_len = self.traj_len.le(0)
        if bool(bad_len.any()):
            bad = bad_len.nonzero(as_tuple=False).flatten()
            raise AssertionError(f"Terminated rows must have positive traj_len. Bad rows: {bad.tolist()}.")

    def _check_t(self, t: int) -> int:
        t = int(t)
        if not 0 <= t < self.T:
            raise IndexError(f"t must be in [0, {self.T}), got {t}.")
        return t

    def _rows(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.device != self.device:
            raise ValueError(f"rows must be on {self.device}, got {rows.device}.")
        if rows.dtype != torch.long:
            raise ValueError(f"rows must be long, got {rows.dtype}.")
        if rows.ndim != 1:
            raise ValueError(f"rows must have shape [M], got {tuple(rows.shape)}.")
        return rows

    def _float_vector(
        self,
        tensor: torch.Tensor,
        *,
        expected: int,
        name: str,
    ) -> torch.Tensor:
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}.")
        if not tensor.dtype.is_floating_point:
            raise ValueError(f"{name} must be floating point, got {tensor.dtype}.")
        if tensor.ndim != 1 or int(tensor.numel()) != int(expected):
            raise ValueError(f"{name} must have shape [{int(expected)}], " f"got {tuple(tensor.shape)}.")
        return tensor.to(dtype=self.dtype)

    def _long_vector(
        self,
        tensor: torch.Tensor,
        *,
        expected: int,
        name: str,
    ) -> torch.Tensor:
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}.")
        if tensor.dtype != torch.long:
            raise ValueError(f"{name} must be long, got {tensor.dtype}.")
        if tensor.ndim != 1 or int(tensor.numel()) != int(expected):
            raise ValueError(f"{name} must have shape [{int(expected)}], " f"got {tuple(tensor.shape)}.")
        return tensor

    def _optional_bool_vector(
        self,
        tensor: torch.Tensor | None,
        *,
        expected: int,
        name: str,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(expected, dtype=torch.bool, device=self.device)
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}.")
        if tensor.dtype != torch.bool:
            raise ValueError(f"{name} must be bool, got {tensor.dtype}.")
        if tensor.ndim != 1 or int(tensor.numel()) != int(expected):
            raise ValueError(f"{name} must have shape [{int(expected)}], " f"got {tuple(tensor.shape)}.")
        return tensor

    def _check_rows_not_terminated(self, rows: torch.Tensor) -> None:
        if rows.numel() == 0:
            return

        terminated = self.is_terminated.index_select(0, rows)
        if bool(terminated.any()):
            bad = rows[terminated]
            raise RuntimeError(f"Cannot write to already terminated rollout rows: {bad.tolist()}.")

    def _check_state_written(self, rows: torch.Tensor, t: int) -> None:
        if rows.numel() == 0:
            return

        written = self.state_written_mask[rows, int(t)]
        if not bool(written.all()):
            bad = rows[~written]
            raise RuntimeError("write_state must be called before writing an action. " f"Missing rows at t={int(t)}: {bad.tolist()}.")

    def _check_step_unwritten(self, rows: torch.Tensor, t: int) -> None:
        if rows.numel() == 0:
            return

        already_written = self.valid_mask[rows, int(t)]
        if bool(already_written.any()):
            bad = rows[already_written]
            raise RuntimeError(f"Action already written at t={int(t)} for rows: {bad.tolist()}.")


__all__ = [
    "EXPAND_ACTION",
    "FORCED_STOP_ACTION",
    "NO_ACTION",
    "NO_EDGE",
    "STOP_ACTION",
    "RolloutTrace",
]
