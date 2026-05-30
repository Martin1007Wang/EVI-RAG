from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.state import FrontierEncoding

STOP_EDGE_ID = -1
INVALID_FRONTIER_POS = -1


def sample_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(x).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(u))


@dataclass(frozen=True, slots=True)
class SampledAction:
    row_ids: torch.Tensor  # [B]
    edge_ids: torch.Tensor  # [B], STOP = -1
    frontier_pos: torch.Tensor  # [B], STOP = -1
    log_prob: torch.Tensor  # [B]
    action_log_flow: torch.Tensor  # [B]

    @property
    def is_stop(self) -> torch.Tensor:
        return self.edge_ids.eq(int(STOP_EDGE_ID))

    @property
    def is_expand(self) -> torch.Tensor:
        return ~self.is_stop


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    state_log_flow: torch.Tensor  # [S]
    stop_log_flow: torch.Tensor  # [S]
    continue_log_flow: torch.Tensor  # [S]
    edge_log_flow: torch.Tensor  # [F]
    frontier: FrontierEncoding
    state_selected_h: torch.Tensor | None = None  # [S, H]
    state_frontier_h: torch.Tensor | None = None  # [S, H]

    def __post_init__(self) -> None:
        flow_dtype = torch.float32
        if self.state_log_flow.dtype != flow_dtype:
            object.__setattr__(self, "state_log_flow", self.state_log_flow.to(dtype=flow_dtype))
        if self.stop_log_flow.dtype != flow_dtype:
            object.__setattr__(self, "stop_log_flow", self.stop_log_flow.to(dtype=flow_dtype))
        if self.continue_log_flow.dtype != flow_dtype:
            object.__setattr__(self, "continue_log_flow", self.continue_log_flow.to(dtype=flow_dtype))
        if self.edge_log_flow.dtype != flow_dtype:
            object.__setattr__(self, "edge_log_flow", self.edge_log_flow.to(dtype=flow_dtype))

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

    def sample(self, *, rows: torch.Tensor) -> SampledAction:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        if int(rows.numel()) == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=self.device)
            empty_float = torch.empty(0, dtype=self.dtype, device=self.device)
            return SampledAction(
                row_ids=empty_long,
                edge_ids=empty_long,
                frontier_pos=empty_long,
                log_prob=empty_float,
                action_log_flow=empty_float,
            )

        stop_score = self.stop_log_flow.index_select(0, rows) + sample_gumbel_like(
            self.stop_log_flow.index_select(0, rows)
        )

        expand_mask = self.frontier.row_ids.unsqueeze(0).eq(rows.unsqueeze(1))
        best_edge_score = torch.full_like(stop_score, float("-inf"))
        best_frontier_pos = torch.full_like(rows, int(INVALID_FRONTIER_POS))

        if self.num_expand_actions > 0:
            edge_score = self.edge_log_flow + sample_gumbel_like(self.edge_log_flow)
            for i in range(int(rows.numel())):
                pos = expand_mask[i].nonzero(as_tuple=True)[0]
                if int(pos.numel()) == 0:
                    continue
                scores = edge_score.index_select(0, pos)
                best = int(torch.argmax(scores).item())
                best_edge_score[i] = scores[best]
                best_frontier_pos[i] = pos[best]

        is_stop = stop_score >= best_edge_score
        edge_ids = torch.full_like(rows, int(STOP_EDGE_ID))
        action_log_flow = self.stop_log_flow.index_select(0, rows)
        log_prob = action_log_flow - self.state_log_flow.index_select(0, rows)

        expand_rows = (~is_stop).nonzero(as_tuple=True)[0]
        if int(expand_rows.numel()) > 0:
            chosen_pos = best_frontier_pos.index_select(0, expand_rows)
            edge_ids[expand_rows] = self.frontier.edge_ids.index_select(0, chosen_pos)
            action_log_flow[expand_rows] = self.edge_log_flow.index_select(0, chosen_pos)
            log_prob[expand_rows] = action_log_flow.index_select(0, expand_rows) - self.state_log_flow.index_select(
                0,
                rows.index_select(0, expand_rows),
            )

        frontier_pos = torch.where(
            is_stop,
            torch.full_like(best_frontier_pos, int(INVALID_FRONTIER_POS)),
            best_frontier_pos,
        )

        return SampledAction(
            row_ids=rows,
            edge_ids=edge_ids,
            frontier_pos=frontier_pos,
            log_prob=log_prob,
            action_log_flow=action_log_flow,
        )

    def gather_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        row_ids = row_ids.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)

        out = self.stop_log_flow.index_select(0, row_ids) - self.state_log_flow.index_select(0, row_ids)
        expand = edge_ids.ge(0)
        if bool(expand.any()):
            pos = find_frontier_positions(
                frontier_row_ids=self.frontier.row_ids,
                frontier_edge_ids=self.frontier.edge_ids,
                query_row_ids=row_ids[expand],
                query_edge_ids=edge_ids[expand],
            )
            out[expand] = self.edge_log_flow.index_select(0, pos) - self.state_log_flow.index_select(0, row_ids[expand])
        return out

    def gather_action_log_flow(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        row_ids = row_ids.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)
        out = self.stop_log_flow.index_select(0, row_ids)
        expand = edge_ids.ge(0)
        if bool(expand.any()):
            pos = find_frontier_positions(
                frontier_row_ids=self.frontier.row_ids,
                frontier_edge_ids=self.frontier.edge_ids,
                query_row_ids=row_ids[expand],
                query_edge_ids=edge_ids[expand],
            )
            out[expand] = self.edge_log_flow.index_select(0, pos)
        return out


def find_frontier_positions(
    *,
    frontier_row_ids: torch.Tensor,
    frontier_edge_ids: torch.Tensor,
    query_row_ids: torch.Tensor,
    query_edge_ids: torch.Tensor,
) -> torch.Tensor:
    if int(query_row_ids.numel()) != int(query_edge_ids.numel()):
        raise ValueError("query_row_ids and query_edge_ids must have the same length.")
    if int(query_row_ids.numel()) == 0:
        return torch.empty(0, dtype=torch.long, device=frontier_row_ids.device)

    base = int(frontier_edge_ids.max().item()) + 2 if int(frontier_edge_ids.numel()) > 0 else 1
    frontier_keys = frontier_row_ids * base + frontier_edge_ids
    query_keys = query_row_ids.to(device=frontier_row_ids.device) * base + query_edge_ids.to(device=frontier_edge_ids.device)

    sorted_keys, order = torch.sort(frontier_keys)
    pos = torch.searchsorted(sorted_keys, query_keys)
    if bool(pos.ge(sorted_keys.numel()).any()):
        raise ValueError("Some requested expansion actions are not legal in the frontier.")
    matched = sorted_keys.index_select(0, pos)
    if not bool(matched.eq(query_keys).all()):
        raise ValueError("Some requested expansion actions are not legal in the frontier.")
    return order.index_select(0, pos)


def gather_stop_log_prob(
    *,
    output: PolicyOutput,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    return output.stop_log_flow.index_select(0, row_ids) - output.state_log_flow.index_select(0, row_ids)


def gather_expand_log_prob(
    *,
    output: PolicyOutput,
    action_row_ids: torch.Tensor,
    action_edge_ids: torch.Tensor,
) -> torch.Tensor:
    pos = find_frontier_positions(
        frontier_row_ids=output.frontier.row_ids,
        frontier_edge_ids=output.frontier.edge_ids,
        query_row_ids=action_row_ids,
        query_edge_ids=action_edge_ids,
    )
    return output.edge_log_flow.index_select(0, pos) - output.state_log_flow.index_select(0, action_row_ids)


__all__ = [
    "INVALID_FRONTIER_POS",
    "PolicyOutput",
    "STOP_EDGE_ID",
    "SampledAction",
    "find_frontier_positions",
    "gather_expand_log_prob",
    "gather_stop_log_prob",
    "sample_gumbel_like",
]
