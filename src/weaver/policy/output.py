from __future__ import annotations

from dataclasses import dataclass

import torch

TERMINAL_EDGE_ID = -1


@dataclass(frozen=True, slots=True)
class ForwardPolicyOutput:
    """
    Forward policy output for a terminal/continue/conditional-edge flow factorization.
    """

    frontier_row_ids: torch.Tensor
    frontier_edge_ids: torch.Tensor
    terminal_log_flow: torch.Tensor
    continue_log_flow: torch.Tensor
    state_log_flow: torch.Tensor
    edge_logit: torch.Tensor
    edge_log_prob: torch.Tensor
    edge_log_flow: torch.Tensor
    stop_log_prob: torch.Tensor
    expand_log_prob: torch.Tensor
    edge_action_log_prob: torch.Tensor
    num_rows: int
    num_edges: int
    frontier_offsets: torch.Tensor | None = None
    action_key_order: torch.Tensor | None = None
    sorted_action_keys: torch.Tensor | None = None

    def __post_init__(self) -> None:
        device = self.terminal_log_flow.device
        float_fields = {
            "terminal_log_flow": self.terminal_log_flow,
            "continue_log_flow": self.continue_log_flow,
            "state_log_flow": self.state_log_flow,
            "edge_logit": self.edge_logit,
            "edge_log_prob": self.edge_log_prob,
            "edge_log_flow": self.edge_log_flow,
            "stop_log_prob": self.stop_log_prob,
            "expand_log_prob": self.expand_log_prob,
            "edge_action_log_prob": self.edge_action_log_prob,
        }
        for name, value in float_fields.items():
            if value.device != device:
                raise ValueError(f"{name} must be on the same device as terminal_log_flow.")
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must use torch.float32.")

        index_fields = {
            "frontier_row_ids": self.frontier_row_ids,
            "frontier_edge_ids": self.frontier_edge_ids,
        }
        for name, value in index_fields.items():
            if value.device != device:
                raise ValueError(f"{name} must be on the same device as terminal_log_flow.")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long.")
        offsets = self.frontier_offsets
        if offsets is None:
            offsets = _frontier_offsets(
                edge_row_ids=self.frontier_row_ids,
                num_rows=int(self.num_rows),
                device=device,
            )
            object.__setattr__(self, "frontier_offsets", offsets)
        action_key_order = self.action_key_order
        sorted_action_keys = self.sorted_action_keys
        if action_key_order is None or sorted_action_keys is None:
            edge_keys = self.frontier_row_ids * int(self.num_edges) + self.frontier_edge_ids
            action_key_order = torch.argsort(edge_keys)
            sorted_action_keys = edge_keys.index_select(0, action_key_order)
            object.__setattr__(self, "action_key_order", action_key_order)
            object.__setattr__(self, "sorted_action_keys", sorted_action_keys)

    @property
    def edge_row_ids(self) -> torch.Tensor:
        return self.frontier_row_ids

    @property
    def edge_ids(self) -> torch.Tensor:
        return self.frontier_edge_ids

    @property
    def stop_log_flow(self) -> torch.Tensor:
        return self.terminal_log_flow

    @property
    def stop_vs_continue_log_ratio(self) -> torch.Tensor:
        return self.terminal_log_flow.float() - self.continue_log_flow

    def has_edge(self) -> torch.Tensor:
        return _rows_with_edges(
            edge_row_ids=self.frontier_row_ids,
            num_rows=int(self.num_rows),
            device=self.terminal_log_flow.device,
        )

    def stop_prob(self) -> torch.Tensor:
        return self.stop_log_prob.exp()

    def action_log_prob(self) -> torch.Tensor:
        return torch.cat(
            [
                self.stop_log_prob,
                self.edge_action_log_prob,
            ],
            dim=0,
        )

    def edge_prob_mass(self) -> torch.Tensor:
        return self.expand_log_prob.exp()

    def conditional_edge_prob_mass(self) -> torch.Tensor:
        mass = self.terminal_log_flow.new_zeros((int(self.num_rows),)).float()
        if self.edge_log_prob.numel() == 0:
            return mass
        probs = self.edge_log_prob.exp()
        mass.scatter_add_(
            0,
            self.frontier_row_ids.to(device=mass.device, dtype=torch.long),
            probs,
        )
        return mass

    def gather_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _gather_by_action_keys(
            stop_values=self.stop_log_prob,
            edge_values=self.edge_action_log_prob,
            out=self,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

    def gather_action_log_flow(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _gather_by_action_keys(
            stop_values=self.stop_log_flow,
            edge_values=self.edge_log_flow,
            out=self,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

    def frontier_size(self) -> torch.Tensor:
        return _frontier_size(
            edge_row_ids=self.frontier_row_ids,
            num_rows=int(self.num_rows),
            device=self.terminal_log_flow.device,
        )

    def edge_action_entropy(self) -> torch.Tensor:
        entropy = self.terminal_log_flow.new_zeros((int(self.num_rows),)).float()
        if self.edge_log_prob.numel() == 0:
            return entropy
        edge_row_ids = self.frontier_row_ids.to(device=self.edge_log_prob.device, dtype=torch.long)
        log_prob = self.edge_log_prob.float()
        prob = log_prob.exp()
        entropy.scatter_add_(0, edge_row_ids, -(prob * log_prob))
        return entropy

    def sample(
        self,
        *,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        rows = rows.to(device=self.terminal_log_flow.device, dtype=torch.long).view(-1)
        if rows.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=rows.device)

        row_ids = self.frontier_row_ids
        edge_ids = self.frontier_edge_ids
        stop_pos = torch.arange(rows.numel(), device=rows.device, dtype=torch.long)
        stop_logits = self.stop_log_prob.index_select(0, rows)
        stop_segment = stop_pos

        offsets = self.frontier_offsets.index_select(0, rows)
        lengths = self.frontier_offsets.index_select(0, rows + 1) - offsets
        total_edges = int(lengths.sum().item())
        if total_edges == 0:
            return torch.full((rows.numel(),), TERMINAL_EDGE_ID, dtype=torch.long, device=rows.device)

        edge_positions = _segment_positions(lengths=lengths) + torch.repeat_interleave(offsets, lengths)
        edge_logits = self.edge_action_log_prob.index_select(0, edge_positions)
        edge_segment = torch.repeat_interleave(stop_pos, lengths)
        edge_choice_ids = edge_ids.index_select(0, edge_positions)

        logits = torch.cat([stop_logits, edge_logits], dim=0)
        segment_ids = torch.cat([stop_segment, edge_segment], dim=0)
        choice_ids = torch.cat(
            [
                torch.full((rows.numel(),), TERMINAL_EDGE_ID, dtype=torch.long, device=rows.device),
                edge_choice_ids,
            ],
            dim=0,
        )
        gumbel = -torch.empty_like(logits).exponential_().log()
        winner_pos = _segment_argmax(
            values=logits + gumbel,
            segment_ids=segment_ids,
            num_segments=int(rows.numel()),
        )
        return choice_ids.index_select(0, winner_pos)


def _frontier_size(
    *,
    edge_row_ids: torch.Tensor,
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    if edge_row_ids.numel() == 0:
        return torch.zeros(int(num_rows), dtype=torch.float32, device=device)
    return torch.bincount(
        edge_row_ids.to(device=device, dtype=torch.long),
        minlength=int(num_rows),
    ).float()


def _frontier_offsets(
    *,
    edge_row_ids: torch.Tensor,
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    counts = torch.bincount(
        edge_row_ids.to(device=device, dtype=torch.long),
        minlength=int(num_rows),
    )
    offsets = torch.empty(int(num_rows) + 1, dtype=torch.long, device=device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def _rows_with_edges(
    *,
    edge_row_ids: torch.Tensor,
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    has_edge = torch.zeros(int(num_rows), dtype=torch.bool, device=device)
    if edge_row_ids.numel() > 0:
        has_edge.index_fill_(0, edge_row_ids.to(device=device, dtype=torch.long), True)
    return has_edge


def _gather_by_action_keys(
    *,
    stop_values: torch.Tensor,
    edge_values: torch.Tensor,
    out: ForwardPolicyOutput,
    row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    row_ids = row_ids.to(device=stop_values.device, dtype=torch.long).view(-1)
    edge_ids = edge_ids.to(device=stop_values.device, dtype=torch.long).view(-1)
    if int(row_ids.numel()) != int(edge_ids.numel()):
        raise ValueError("row_ids and edge_ids must have the same length.")
    if bool(row_ids.lt(0).any()) or bool(row_ids.ge(int(out.num_rows)).any()):
        raise IndexError("row_ids must be in [0, ForwardPolicyOutput.num_rows).")

    out_values = stop_values.new_empty((row_ids.numel(),))
    stop_mask = edge_ids.eq(TERMINAL_EDGE_ID)
    if bool(stop_mask.any()):
        out_values[stop_mask] = stop_values.index_select(0, row_ids[stop_mask])

    edge_mask = edge_ids.ge(0)
    if bool(edge_mask.any()):
        key_width = int(out.num_edges)
        if key_width <= 0:
            raise KeyError("Requested edge action is not present in ForwardPolicyOutput.")
        if bool(edge_ids[edge_mask].ge(key_width).any()):
            raise IndexError("edge_ids must be TERMINAL_EDGE_ID or in [0, ForwardPolicyOutput.num_edges).")
        target_keys = row_ids[edge_mask] * key_width + edge_ids[edge_mask]
        sorted_keys = out.sorted_action_keys
        order = out.action_key_order
        duplicate_keys = (
            sorted_keys[1:].eq(sorted_keys[:-1])
            if sorted_keys.numel() > 1
            else torch.empty(0, dtype=torch.bool, device=sorted_keys.device)
        )
        if bool(duplicate_keys.any()):
            raise ValueError("ForwardPolicyOutput frontier contains duplicate (row_id, edge_id) actions.")
        positions = torch.searchsorted(sorted_keys, target_keys)
        if bool(positions.ge(sorted_keys.numel()).any()):
            raise KeyError("Requested edge action is not present in ForwardPolicyOutput.")
        matched = sorted_keys.index_select(0, positions)
        if not torch.equal(matched, target_keys):
            raise KeyError("Requested edge action is not present in ForwardPolicyOutput.")
        out_values[edge_mask] = edge_values.index_select(0, order.index_select(0, positions))

    return out_values


def _segment_argmax(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    best = values.new_full((int(num_segments),), -torch.inf)
    best.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    matches = values.eq(best.index_select(0, segment_ids))
    candidate_pos = torch.arange(values.numel(), device=values.device, dtype=torch.long)
    fallback = torch.full_like(candidate_pos, values.numel())
    winner = torch.full((int(num_segments),), values.numel(), dtype=torch.long, device=values.device)
    winner.scatter_reduce_(
        0,
        segment_ids,
        torch.where(matches, candidate_pos, fallback),
        reduce="amin",
        include_self=True,
    )
    if bool(winner.eq(values.numel()).any()):
        raise RuntimeError("segment argmax failed to pick an action for at least one segment.")
    return winner


def _segment_positions(*, lengths: torch.Tensor) -> torch.Tensor:
    total = int(lengths.sum().item())
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=lengths.device)
    starts = torch.cumsum(lengths, dim=0) - lengths
    return torch.arange(total, dtype=torch.long, device=lengths.device) - torch.repeat_interleave(starts, lengths)


__all__ = [
    "ForwardPolicyOutput",
    "TERMINAL_EDGE_ID",
]
