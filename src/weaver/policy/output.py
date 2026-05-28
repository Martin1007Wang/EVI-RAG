from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_scatter import scatter_max

from src.weaver.state import ActionSpace

STOP_EDGE_ID = -1
INVALID_FRONTIER_POS = -1


def _as_long_1d(
    x: torch.Tensor,
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    return x.to(device=device, dtype=torch.long).view(-1)


def _check_same_length(
    *,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_name: str,
    rhs_name: str,
) -> None:
    if int(lhs.numel()) != int(rhs.numel()):
        raise ValueError(f"{lhs_name} and {rhs_name} must have the same length: " f"got {int(lhs.numel())} and {int(rhs.numel())}.")


@dataclass(frozen=True, slots=True)
class SampledAction:
    row_ids: torch.Tensor  # [B]
    edge_ids: torch.Tensor  # [B], STOP = -1
    frontier_pos: torch.Tensor  # [B], STOP = -1
    log_prob: torch.Tensor  # [B]
    action_log_flow: torch.Tensor  # [B]

    def __post_init__(self) -> None:
        n = int(self.row_ids.numel())
        for name, value in (
            ("edge_ids", self.edge_ids),
            ("frontier_pos", self.frontier_pos),
            ("log_prob", self.log_prob),
            ("action_log_flow", self.action_log_flow),
        ):
            if int(value.numel()) != n:
                raise ValueError(f"SampledAction.{name} must have length {n}, " f"got {int(value.numel())}.")

        if self.row_ids.dtype != torch.long:
            raise TypeError("SampledAction.row_ids must be torch.long.")
        if self.edge_ids.dtype != torch.long:
            raise TypeError("SampledAction.edge_ids must be torch.long.")
        if self.frontier_pos.dtype != torch.long:
            raise TypeError("SampledAction.frontier_pos must be torch.long.")

    @property
    def device(self) -> torch.device:
        return self.row_ids.device

    @property
    def num_actions(self) -> int:
        return int(self.row_ids.numel())

    @property
    def is_stop(self) -> torch.Tensor:
        return self.edge_ids.eq(int(STOP_EDGE_ID))

    @property
    def is_expand(self) -> torch.Tensor:
        return ~self.is_stop

    def detach(self) -> SampledAction:
        return SampledAction(
            row_ids=self.row_ids.detach(),
            edge_ids=self.edge_ids.detach(),
            frontier_pos=self.frontier_pos.detach(),
            log_prob=self.log_prob.detach(),
            action_log_flow=self.action_log_flow.detach(),
        )


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """
    Flat CSR policy output over STOP ∪ frontier edges.

    First-class quantities:
    - stop_log_flow[z]
    - edge_log_flow[pos]
    - continue_log_flow[z] = logsumexp_{pos in C(z)} edge_log_flow[pos]
    - state_log_flow[z] = logaddexp(stop_log_flow[z], continue_log_flow[z])

    Expansion action identity is frontier_pos, not merely edge_id.
    """

    action_space: ActionSpace

    state_log_flow: torch.Tensor  # [S]
    stop_log_flow: torch.Tensor  # [S]
    continue_log_flow: torch.Tensor  # [S]

    edge_log_flow: torch.Tensor  # [F]
    edge_raw_score: torch.Tensor  # [F]

    def __post_init__(self) -> None:
        if self.state_log_flow.ndim != 1:
            raise ValueError("state_log_flow must be rank-1 [S].")
        if self.stop_log_flow.ndim != 1:
            raise ValueError("stop_log_flow must be rank-1 [S].")
        if self.continue_log_flow.ndim != 1:
            raise ValueError("continue_log_flow must be rank-1 [S].")
        if self.edge_log_flow.ndim != 1:
            raise ValueError("edge_log_flow must be rank-1 [F].")
        if self.edge_raw_score.ndim != 1:
            raise ValueError("edge_raw_score must be rank-1 [F].")

        num_states = int(self.state_log_flow.numel())
        num_edges = int(self.edge_log_flow.numel())

        if int(self.stop_log_flow.numel()) != num_states:
            raise ValueError("stop_log_flow must have shape [S].")
        if int(self.continue_log_flow.numel()) != num_states:
            raise ValueError("continue_log_flow must have shape [S].")
        if int(self.edge_raw_score.numel()) != num_edges:
            raise ValueError("edge_raw_score must have shape [F].")

        if int(self.action_space.expand_ptr.numel()) != num_states + 1:
            raise ValueError(
                "action_space.expand_ptr must have shape [S + 1]. " f"Expected {num_states + 1}, got " f"{int(self.action_space.expand_ptr.numel())}."
            )

        if int(self.action_space.expand_state_ids.numel()) != num_edges:
            raise ValueError("action_space.expand_state_ids must have shape [F].")
        if int(self.action_space.expand_edge_ids.numel()) != num_edges:
            raise ValueError("action_space.expand_edge_ids must have shape [F].")

    @property
    def device(self) -> torch.device:
        return self.state_log_flow.device

    @property
    def dtype(self) -> torch.dtype:
        return self.state_log_flow.dtype

    @property
    def num_states(self) -> int:
        return int(self.state_log_flow.numel())

    @property
    def num_expand_actions(self) -> int:
        return int(self.edge_log_flow.numel())

    @property
    def stop_log_prob(self) -> torch.Tensor:
        return self.stop_log_flow - self.state_log_flow

    @property
    def continue_log_prob(self) -> torch.Tensor:
        return self.continue_log_flow - self.state_log_flow

    @property
    def edge_log_prob(self) -> torch.Tensor:
        if self.num_expand_actions == 0:
            return torch.empty(0, dtype=self.dtype, device=self.device)

        row_ids = self.action_space.expand_state_ids.to(
            device=self.device,
            dtype=torch.long,
        )
        return self.edge_log_flow - self.state_log_flow.index_select(0, row_ids)

    @property
    def conditional_edge_log_prob(self) -> torch.Tensor:
        if self.num_expand_actions == 0:
            return torch.empty(0, dtype=self.dtype, device=self.device)

        row_ids = self.action_space.expand_state_ids.to(
            device=self.device,
            dtype=torch.long,
        )
        return self.edge_log_flow - self.continue_log_flow.index_select(0, row_ids)

    def sample(self, *, rows: torch.Tensor) -> SampledAction:
        """
        Sample one independent action for each row entry.

        Repeated rows are allowed and mean independent repeated draws from the
        same physical state row.
        """
        rows = _as_long_1d(rows, device=self.device, name="rows")
        n = int(rows.numel())

        if n == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=self.device)
            empty_float = torch.empty(0, dtype=self.dtype, device=self.device)
            return SampledAction(
                row_ids=empty_long,
                edge_ids=empty_long,
                frontier_pos=empty_long,
                log_prob=empty_float,
                action_log_flow=empty_float,
            )

        self._validate_row_ids(rows, name="rows")

        edge_ids = torch.full(
            (n,),
            int(STOP_EDGE_ID),
            dtype=torch.long,
            device=self.device,
        )
        frontier_pos = torch.full(
            (n,),
            int(INVALID_FRONTIER_POS),
            dtype=torch.long,
            device=self.device,
        )

        stop_continue_log_prob = torch.stack(
            (
                self.stop_log_flow.index_select(0, rows) - self.state_log_flow.index_select(0, rows),
                self.continue_log_flow.index_select(0, rows) - self.state_log_flow.index_select(0, rows),
            ),
            dim=1,
        )

        decision = torch.multinomial(
            stop_continue_log_prob.exp(),
            num_samples=1,
        ).squeeze(1)

        expand_request_mask = decision.eq(1)
        if bool(expand_request_mask.any()):
            expand_request_ids = torch.nonzero(
                expand_request_mask,
                as_tuple=False,
            ).view(-1)

            expand_rows = rows.index_select(0, expand_request_ids)
            sampled_pos = self._sample_expand_frontier_pos(rows=expand_rows)

            frontier_pos[expand_request_ids] = sampled_pos
            edge_ids[expand_request_ids] = self.action_space.expand_edge_ids.to(
                device=self.device,
                dtype=torch.long,
            ).index_select(0, sampled_pos)

        log_prob, action_log_flow = self._gather_sampled_fast(
            row_ids=rows,
            edge_ids=edge_ids,
            frontier_pos=frontier_pos,
        )

        return SampledAction(
            row_ids=rows,
            edge_ids=edge_ids,
            frontier_pos=frontier_pos,
            log_prob=log_prob,
            action_log_flow=action_log_flow,
        )

    def _sample_expand_frontier_pos(self, *, rows: torch.Tensor) -> torch.Tensor:
        """
        Sample expansion frontier positions with Gumbel-max.

        Uses edge_log_flow directly. For each row z, subtracting
        continue_log_flow[z] is a row-wise constant, so it is unnecessary for
        argmax sampling.
        """
        rows = _as_long_1d(rows, device=self.device, name="rows")
        m = int(rows.numel())

        if m == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        ptr = self.action_space.expand_ptr.to(device=self.device, dtype=torch.long)
        starts = ptr.index_select(0, rows)
        ends = ptr.index_select(0, rows + 1)
        counts = ends - starts

        if not bool(counts.gt(0).all()):
            bad = rows[counts.le(0)][:10].detach().cpu().tolist()
            raise RuntimeError("CONTINUE was sampled for a row with no legal expansions. " f"First offending rows: {bad}.")

        total = int(counts.sum().item())

        request_ids = torch.repeat_interleave(
            torch.arange(m, dtype=torch.long, device=self.device),
            counts,
            output_size=total,
        )

        segment_starts = torch.repeat_interleave(
            torch.cumsum(counts, dim=0) - counts,
            counts,
            output_size=total,
        )

        offsets = torch.arange(total, dtype=torch.long, device=self.device) - segment_starts

        candidate_pos = starts.index_select(0, request_ids) + offsets
        candidate_score = self.edge_log_flow.index_select(0, candidate_pos)

        gumbel = -torch.empty_like(candidate_score).exponential_().log()
        noisy_score = candidate_score + gumbel

        _, winner_pos_in_candidate = scatter_max(
            noisy_score,
            request_ids,
            dim=0,
            dim_size=m,
        )

        return candidate_pos.index_select(0, winner_pos_in_candidate)

    def gather_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        frontier_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_ids = _as_long_1d(row_ids, device=self.device, name="row_ids")
        edge_ids = _as_long_1d(edge_ids, device=self.device, name="edge_ids")

        _check_same_length(
            lhs=row_ids,
            rhs=edge_ids,
            lhs_name="row_ids",
            rhs_name="edge_ids",
        )
        self._validate_row_ids(row_ids, name="row_ids")

        pos = self._prepare_frontier_pos(
            row_ids=row_ids,
            edge_ids=edge_ids,
            frontier_pos=frontier_pos,
        )

        log_prob, _ = self._gather_sampled_fast(
            row_ids=row_ids,
            edge_ids=edge_ids,
            frontier_pos=pos,
        )
        return log_prob

    def gather_action_log_flow(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        frontier_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_ids = _as_long_1d(row_ids, device=self.device, name="row_ids")
        edge_ids = _as_long_1d(edge_ids, device=self.device, name="edge_ids")

        _check_same_length(
            lhs=row_ids,
            rhs=edge_ids,
            lhs_name="row_ids",
            rhs_name="edge_ids",
        )
        self._validate_row_ids(row_ids, name="row_ids")

        pos = self._prepare_frontier_pos(
            row_ids=row_ids,
            edge_ids=edge_ids,
            frontier_pos=frontier_pos,
        )

        _, action_log_flow = self._gather_sampled_fast(
            row_ids=row_ids,
            edge_ids=edge_ids,
            frontier_pos=pos,
        )
        return action_log_flow

    def _gather_sampled_fast(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        frontier_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_prob = torch.empty(
            int(row_ids.numel()),
            dtype=self.dtype,
            device=self.device,
        )
        action_log_flow = torch.empty_like(log_prob)

        stop = edge_ids.eq(int(STOP_EDGE_ID))
        expand = ~stop

        if bool(stop.any()):
            stop_rows = row_ids[stop]
            stop_flow = self.stop_log_flow.index_select(0, stop_rows)
            state_flow = self.state_log_flow.index_select(0, stop_rows)

            action_log_flow[stop] = stop_flow
            log_prob[stop] = stop_flow - state_flow

        if bool(expand.any()):
            expand_rows = row_ids[expand]
            expand_pos = frontier_pos[expand]

            edge_flow = self.edge_log_flow.index_select(0, expand_pos)
            state_flow = self.state_log_flow.index_select(0, expand_rows)

            action_log_flow[expand] = edge_flow
            log_prob[expand] = edge_flow - state_flow

        return log_prob, action_log_flow

    def gather_edge_log_flow_by_pos(
        self,
        *,
        frontier_pos: torch.Tensor,
    ) -> torch.Tensor:
        frontier_pos = _as_long_1d(
            frontier_pos,
            device=self.device,
            name="frontier_pos",
        )
        self._validate_frontier_pos(frontier_pos, name="frontier_pos")
        return self.edge_log_flow.index_select(0, frontier_pos)

    def gather_edge_log_prob_by_pos(
        self,
        *,
        frontier_pos: torch.Tensor,
    ) -> torch.Tensor:
        frontier_pos = _as_long_1d(
            frontier_pos,
            device=self.device,
            name="frontier_pos",
        )
        self._validate_frontier_pos(frontier_pos, name="frontier_pos")

        row_ids = self.action_space.expand_state_ids.to(
            device=self.device,
            dtype=torch.long,
        ).index_select(0, frontier_pos)

        return self.edge_log_flow.index_select(0, frontier_pos) - self.state_log_flow.index_select(0, row_ids)

    def gather_conditional_edge_log_prob_by_pos(
        self,
        *,
        frontier_pos: torch.Tensor,
    ) -> torch.Tensor:
        frontier_pos = _as_long_1d(
            frontier_pos,
            device=self.device,
            name="frontier_pos",
        )
        self._validate_frontier_pos(frontier_pos, name="frontier_pos")

        row_ids = self.action_space.expand_state_ids.to(
            device=self.device,
            dtype=torch.long,
        ).index_select(0, frontier_pos)

        return self.edge_log_flow.index_select(0, frontier_pos) - self.continue_log_flow.index_select(0, row_ids)

    def _prepare_frontier_pos(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        frontier_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        if frontier_pos is None:
            return self.resolve_frontier_pos(
                row_ids=row_ids,
                edge_ids=edge_ids,
            )

        frontier_pos = _as_long_1d(
            frontier_pos,
            device=self.device,
            name="frontier_pos",
        )

        _check_same_length(
            lhs=edge_ids,
            rhs=frontier_pos,
            lhs_name="edge_ids",
            rhs_name="frontier_pos",
        )

        stop = edge_ids.eq(int(STOP_EDGE_ID))
        expand = ~stop

        if bool(stop.any()) and bool(frontier_pos[stop].ne(int(INVALID_FRONTIER_POS)).any()):
            raise ValueError("frontier_pos must be INVALID_FRONTIER_POS for STOP actions.")

        if bool(expand.any()):
            expand_pos = frontier_pos[expand]
            self._validate_frontier_pos(expand_pos, name="frontier_pos[expand]")

            actual_rows = self.action_space.expand_state_ids.to(
                device=self.device,
                dtype=torch.long,
            ).index_select(0, expand_pos)

            actual_edges = self.action_space.expand_edge_ids.to(
                device=self.device,
                dtype=torch.long,
            ).index_select(0, expand_pos)

            if not bool(actual_rows.eq(row_ids[expand]).all()):
                raise ValueError("frontier_pos does not match row_ids.")
            if not bool(actual_edges.eq(edge_ids[expand]).all()):
                raise ValueError("frontier_pos does not match edge_ids.")

        return frontier_pos

    def resolve_frontier_pos(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Debug/compatibility resolver from (row_id, edge_id) to frontier_pos.

        Main training path should pass frontier_pos directly.
        """
        row_ids = _as_long_1d(row_ids, device=self.device, name="row_ids")
        edge_ids = _as_long_1d(edge_ids, device=self.device, name="edge_ids")

        _check_same_length(
            lhs=row_ids,
            rhs=edge_ids,
            lhs_name="row_ids",
            rhs_name="edge_ids",
        )

        out = torch.full(
            (int(row_ids.numel()),),
            int(INVALID_FRONTIER_POS),
            dtype=torch.long,
            device=self.device,
        )

        expand = edge_ids.ne(int(STOP_EDGE_ID))
        if not bool(expand.any()):
            return out

        expand_rows = row_ids[expand]
        expand_edges = edge_ids[expand]

        frontier_rows = self.action_space.expand_state_ids.to(
            device=self.device,
            dtype=torch.long,
        )
        frontier_edges = self.action_space.expand_edge_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        if int(frontier_edges.numel()) == 0:
            raise ValueError("ActionSpace contains no legal expansion actions.")

        max_frontier_edge = frontier_edges.max()
        max_query_edge = expand_edges.max()
        base = torch.maximum(max_frontier_edge, max_query_edge) + 1

        max_row = torch.maximum(frontier_rows.max(), expand_rows.max())
        max_safe = torch.iinfo(torch.long).max
        if bool(max_row.gt(max_safe // base)):
            raise OverflowError("Encoded (row, edge) lookup key would overflow int64. " "Pass frontier_pos instead.")

        frontier_keys = frontier_rows * base + frontier_edges
        query_keys = expand_rows * base + expand_edges

        sorted_keys, order = torch.sort(frontier_keys)
        positions = torch.searchsorted(sorted_keys, query_keys)

        in_range = positions.lt(int(sorted_keys.numel()))
        safe_positions = positions.clamp_max(int(sorted_keys.numel()) - 1)

        matched = in_range & sorted_keys.index_select(0, safe_positions).eq(query_keys)

        if not bool(matched.all()):
            missing = query_keys[~matched][:10].detach().cpu().tolist()
            raise ValueError("edge_id is not a legal expansion action for the requested row. " f"First missing encoded keys: {missing}")

        out[expand] = order.index_select(0, safe_positions)
        return out

    def _validate_row_ids(self, row_ids: torch.Tensor, *, name: str) -> None:
        if int(row_ids.numel()) == 0:
            return

        bad_mask = row_ids.lt(0) | row_ids.ge(self.num_states)
        if bool(bad_mask.any()):
            bad = row_ids[bad_mask][:10].detach().cpu().tolist()
            raise IndexError(f"{name} contains out-of-range state rows. " f"num_states={self.num_states}, first bad={bad}.")

    def _validate_frontier_pos(
        self,
        frontier_pos: torch.Tensor,
        *,
        name: str,
    ) -> None:
        if int(frontier_pos.numel()) == 0:
            return

        bad_mask = frontier_pos.lt(0) | frontier_pos.ge(self.num_expand_actions)
        if bool(bad_mask.any()):
            bad = frontier_pos[bad_mask][:10].detach().cpu().tolist()
            raise IndexError(
                f"{name} contains out-of-range frontier positions. " f"num_expand_actions={self.num_expand_actions}, " f"first bad={bad}."
            )


__all__ = [
    "PolicyOutput",
    "SampledAction",
    "STOP_EDGE_ID",
    "INVALID_FRONTIER_POS",
]
