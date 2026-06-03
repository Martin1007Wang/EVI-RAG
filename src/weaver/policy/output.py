from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.graph.segments import sample_segmented_positions, segment_logsumexp
from src.weaver.state import FrontierEncoding

STOP_EDGE_ID = -1


@dataclass(frozen=True, slots=True)
class SampledAction:
    row_ids: torch.Tensor
    edge_ids: torch.Tensor
    log_prob: torch.Tensor

    @property
    def is_stop(self) -> torch.Tensor:
        return self.edge_ids.eq(STOP_EDGE_ID)


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    action_logits: torch.Tensor  # [S + F]
    action_row_ids: torch.Tensor  # [S + F]
    action_edge_ids: torch.Tensor  # [S + F], STOP = -1
    frontier: FrontierEncoding
    log_flow: torch.Tensor | None  # [S], training-only
    _log_partition: torch.Tensor = field(init=False, repr=False)
    _action_log_prob: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        log_partition = segment_logsumexp(
            values=self.action_logits.float(),
            segment_ids=self.action_row_ids,
            num_segments=self.num_states,
        )
        object.__setattr__(self, "_log_partition", log_partition)
        object.__setattr__(
            self,
            "_action_log_prob",
            self.action_logits.float() - log_partition.index_select(0, self.action_row_ids),
        )

    @property
    def device(self) -> torch.device:
        return self.action_logits.device

    @property
    def num_states(self) -> int:
        if int(self.action_row_ids.numel()) == 0:
            return 0
        return int(self.action_row_ids.max().item()) + 1

    @property
    def log_partition(self) -> torch.Tensor:
        return self._log_partition

    @property
    def action_log_prob(self) -> torch.Tensor:
        return self._action_log_prob

    @property
    def forced_terminal_mask(self) -> torch.Tensor:
        frontier_count = torch.bincount(self.frontier.row_ids, minlength=self.num_states)
        return frontier_count.eq(0)

    def require_log_flow(self) -> torch.Tensor:
        if self.log_flow is None:
            raise RuntimeError("log_flow was not computed. Call policy with compute_log_flow=True.")
        return self.log_flow

    def gather_log_prob(self, *, row_ids: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        positions = _find_actions(
            num_states=self.num_states,
            frontier=self.frontier,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )
        return self.action_log_prob.index_select(0, positions)

    def sample(self, *, rows: torch.Tensor) -> SampledAction:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        pos = sample_segmented_positions(
            logits=self.action_logits,
            segment_ids=self.action_row_ids,
            num_segments=self.num_states,
            active_segments=rows,
            temperature=1.0,
        )
        return SampledAction(
            row_ids=rows,
            edge_ids=self.action_edge_ids.index_select(0, pos),
            log_prob=self.action_log_prob.index_select(0, pos),
        )


def _find_actions(
    *,
    num_states: int,
    frontier: FrontierEncoding,
    row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    row_ids = row_ids.to(device=frontier.row_ids.device, dtype=torch.long).view(-1)
    edge_ids = edge_ids.to(device=frontier.edge_ids.device, dtype=torch.long).view(-1)
    if int(row_ids.numel()) != int(edge_ids.numel()):
        raise ValueError("row_ids and edge_ids must have the same length.")
    if int(row_ids.numel()) == 0:
        return row_ids
    if bool(row_ids.lt(0).any()) or bool(row_ids.ge(int(num_states)).any()):
        raise ValueError("requested action row must be in range.")
    stop = edge_ids.eq(STOP_EDGE_ID)
    positions = row_ids.clone()
    if bool((~stop).any()):
        requested_rows = row_ids[~stop]
        requested_edges = edge_ids[~stop]
        if bool(requested_edges.lt(0).any()):
            raise ValueError("requested edge action must be STOP or nonnegative.")
        upper = torch.cat([frontier.edge_ids, requested_edges]).max().add(1)
        frontier_keys = frontier.row_ids * upper + frontier.edge_ids
        requested_keys = requested_rows * upper + requested_edges
        frontier_positions = torch.searchsorted(frontier_keys, requested_keys)
        in_range = frontier_positions.lt(int(frontier_keys.numel()))
        legal = torch.zeros_like(in_range)
        legal[in_range] = frontier_keys.index_select(0, frontier_positions[in_range]).eq(requested_keys[in_range])
        if not bool(legal.all()):
            raise ValueError("requested action must be uniquely legal.")
        positions[~stop] = int(num_states) + frontier_positions
    return positions


__all__ = ["PolicyOutput", "STOP_EDGE_ID", "SampledAction"]
