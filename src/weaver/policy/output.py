from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.graph.segments import segment_log_softmax, segment_logsumexp
from src.weaver.state import Frontier

STOP_EDGE_ID = -1


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """
    Forward policy output for Bernoulli STOP and conditional edge choice.

        stop_logit = Bernoulli STOP logit
        log_flow   = state log flow
        edge_logit = CONTINUE branch edge policy scores
    """

    stop_logit: torch.Tensor
    log_flow: torch.Tensor
    edge_logit: torch.Tensor
    frontier: Frontier
    num_rows: int
    num_edges: int

    @property
    def edge_row_ids(self) -> torch.Tensor:
        return self.frontier.row_ids

    @property
    def edge_ids(self) -> torch.Tensor:
        return self.frontier.edge_ids

    def has_edge(self) -> torch.Tensor:
        return _rows_with_edges(
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
            device=self.stop_logit.device,
        )

    @property
    def stop_log_prob(self) -> torch.Tensor:
        log_prob = F.logsigmoid(self.stop_logit.float())
        return torch.where(
            self.has_edge(),
            log_prob,
            torch.zeros_like(log_prob),
        )

    def stop_prob(self) -> torch.Tensor:
        return self.stop_log_prob.exp()

    @property
    def continue_log_prob(self) -> torch.Tensor:
        log_prob = F.logsigmoid(-self.stop_logit.float())
        return torch.where(
            self.has_edge(),
            log_prob,
            torch.full_like(log_prob, -torch.inf),
        )

    def continue_prob(self) -> torch.Tensor:
        return self.continue_log_prob.exp()

    @property
    def edge_log_partition(self) -> torch.Tensor:
        if self.edge_logit.numel() == 0:
            return self.log_flow.new_full((int(self.num_rows),), -torch.inf).float()
        return segment_logsumexp(
            values=self.edge_logit.float(),
            segment_ids=self.edge_row_ids.to(device=self.edge_logit.device, dtype=torch.long),
            num_segments=int(self.num_rows),
        )

    @property
    def edge_log_cond_prob(self) -> torch.Tensor:
        if self.edge_logit.numel() == 0:
            return self.edge_logit.new_empty((0,)).float()
        return segment_log_softmax(
            self.edge_logit.float(),
            self.edge_row_ids.to(device=self.edge_logit.device, dtype=torch.long),
            num_segments=int(self.num_rows),
        )

    @property
    def edge_log_prob(self) -> torch.Tensor:
        if self.edge_logit.numel() == 0:
            return self.edge_logit.new_empty((0,)).float()
        edge_row_ids = self.edge_row_ids.to(device=self.stop_logit.device, dtype=torch.long)
        return self.continue_log_prob.index_select(0, edge_row_ids) + self.edge_log_cond_prob

    def action_log_prob(self) -> torch.Tensor:
        return torch.cat(
            [
                self.stop_log_prob,
                self.edge_log_prob,
            ],
            dim=0,
        )

    def edge_prob_mass(self) -> torch.Tensor:
        mass = self.stop_logit.new_zeros((int(self.num_rows),)).float()
        if self.edge_logit.numel() == 0:
            return mass
        probs = self.edge_log_prob.exp()
        mass.scatter_add_(
            0,
            self.edge_row_ids.to(device=mass.device, dtype=torch.long),
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
            edge_values=self.edge_log_prob,
            out=self,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

    def gather_continue_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
    ) -> torch.Tensor:
        row_ids = row_ids.to(device=self.stop_logit.device, dtype=torch.long).view(-1)
        return self.continue_log_prob.index_select(0, row_ids)

    def gather_edge_cond_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _gather_by_action_keys(
            stop_values=self.stop_logit.new_zeros((int(self.num_rows),)).float(),
            edge_values=self.edge_log_cond_prob,
            out=self,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

    def frontier_size(self) -> torch.Tensor:
        return _frontier_size(
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
            device=self.stop_logit.device,
        )

    def edge_cond_entropy(self) -> torch.Tensor:
        entropy = self.stop_logit.new_zeros((int(self.num_rows),)).float()
        if self.edge_logit.numel() == 0:
            return entropy
        edge_row_ids = self.edge_row_ids.to(device=self.edge_logit.device, dtype=torch.long)
        log_prob = self.edge_log_cond_prob
        prob = log_prob.exp()
        entropy.scatter_add_(0, edge_row_ids, -(prob * log_prob))
        return entropy

    def sample(
        self,
        *,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        rows = rows.to(device=self.stop_logit.device, dtype=torch.long).view(-1)
        picked_edge_ids = torch.full(
            (rows.numel(),),
            STOP_EDGE_ID,
            dtype=torch.long,
            device=rows.device,
        )
        if rows.numel() == 0:
            return picked_edge_ids

        row_has_edge = self.has_edge().index_select(0, rows)
        candidate_rows = rows[row_has_edge]
        if candidate_rows.numel() == 0:
            return picked_edge_ids

        stop_prob = self.stop_prob().index_select(0, candidate_rows)
        sampled_stop = torch.bernoulli(stop_prob).to(dtype=torch.bool)
        continue_rows = candidate_rows[~sampled_stop]
        if continue_rows.numel() == 0:
            return picked_edge_ids

        sampled_continue = _sample_edge_actions(
            edge_log_prob=self.edge_log_cond_prob,
            edge_row_ids=self.edge_row_ids,
            edge_ids=self.edge_ids,
            rows=continue_rows,
            num_rows=int(self.num_rows),
        )
        row_positions = _row_positions(
            rows=rows,
            selected_rows=continue_rows,
        )
        picked_edge_ids.index_copy_(0, row_positions, sampled_continue)
        return picked_edge_ids


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
    out: PolicyOutput,
    row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    row_ids = row_ids.to(device=stop_values.device, dtype=torch.long).view(-1)
    edge_ids = edge_ids.to(device=stop_values.device, dtype=torch.long).view(-1)
    if int(row_ids.numel()) != int(edge_ids.numel()):
        raise ValueError("row_ids and edge_ids must have the same length.")
    if bool(row_ids.lt(0).any()) or bool(row_ids.ge(int(out.num_rows)).any()):
        raise IndexError("row_ids must be in [0, PolicyOutput.num_rows).")

    out_values = stop_values.new_empty((row_ids.numel(),))
    stop_mask = edge_ids.eq(STOP_EDGE_ID)
    if bool(stop_mask.any()):
        out_values[stop_mask] = stop_values.index_select(0, row_ids[stop_mask])

    edge_mask = edge_ids.ge(0)
    if bool(edge_mask.any()):
        key_width = int(out.num_edges)
        if key_width <= 0:
            raise KeyError("Requested edge action is not present in PolicyOutput.")
        if bool(edge_ids[edge_mask].ge(key_width).any()):
            raise IndexError("edge_ids must be STOP_EDGE_ID or in [0, PolicyOutput.num_edges).")
        edge_keys = out.edge_row_ids * key_width + out.edge_ids
        target_keys = row_ids[edge_mask] * key_width + edge_ids[edge_mask]
        order = torch.argsort(edge_keys)
        sorted_keys = edge_keys.index_select(0, order)
        duplicate_keys = (
            sorted_keys[1:].eq(sorted_keys[:-1])
            if sorted_keys.numel() > 1
            else torch.empty(0, dtype=torch.bool, device=sorted_keys.device)
        )
        if bool(duplicate_keys.any()):
            raise ValueError("PolicyOutput frontier contains duplicate (row_id, edge_id) actions.")
        positions = torch.searchsorted(sorted_keys, target_keys)
        if bool(positions.ge(sorted_keys.numel()).any()):
            raise KeyError("Requested edge action is not present in PolicyOutput.")
        matched = sorted_keys.index_select(0, positions)
        if not torch.equal(matched, target_keys):
            raise KeyError("Requested edge action is not present in PolicyOutput.")
        out_values[edge_mask] = edge_values.index_select(0, order.index_select(0, positions))

    return out_values


def _sample_edge_actions(
    *,
    edge_log_prob: torch.Tensor,
    edge_row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    rows: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    local_edge_row_ids, kept_edge_ids, values = _select_rows_for_sampling(
        edge_log_prob=edge_log_prob,
        edge_row_ids=edge_row_ids,
        edge_ids=edge_ids,
        rows=rows,
        num_rows=num_rows,
    )
    picked_positions = _segment_gumbel_argmax(
        values=values,
        row_ids=local_edge_row_ids,
        num_rows=int(rows.numel()),
    )
    return kept_edge_ids.index_select(0, picked_positions)


def _select_rows_for_sampling(
    *,
    edge_log_prob: torch.Tensor,
    edge_row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    rows: torch.Tensor,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_row_ids = edge_row_ids.to(device=rows.device, dtype=torch.long)
    edge_ids = edge_ids.to(device=rows.device, dtype=torch.long)
    row_mask = torch.zeros(int(num_rows), dtype=torch.bool, device=rows.device)
    row_mask.index_fill_(0, rows, True)
    keep_edges = row_mask.index_select(0, edge_row_ids)
    kept_edge_row_ids = edge_row_ids[keep_edges]
    kept_edge_ids = edge_ids[keep_edges]
    local_edge_row_ids = _remap_rows(
        row_ids=kept_edge_row_ids,
        rows=rows,
        num_rows=int(num_rows),
    )
    return local_edge_row_ids, kept_edge_ids, edge_log_prob.to(device=rows.device)[keep_edges]


def _row_positions(
    *,
    rows: torch.Tensor,
    selected_rows: torch.Tensor,
) -> torch.Tensor:
    positions = torch.empty(
        rows.numel(),
        dtype=torch.long,
        device=rows.device,
    )
    positions.index_copy_(
        0,
        rows,
        torch.arange(rows.numel(), dtype=torch.long, device=rows.device),
    )
    return positions.index_select(0, selected_rows)


def _remap_rows(
    *,
    row_ids: torch.Tensor,
    rows: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    mapping = torch.empty(
        int(num_rows),
        dtype=torch.long,
        device=rows.device,
    )
    mapping.index_copy_(
        0,
        rows,
        torch.arange(
            rows.numel(),
            dtype=torch.long,
            device=rows.device,
        ),
    )
    return mapping.index_select(0, row_ids)


def _segment_gumbel_argmax(
    *,
    values: torch.Tensor,
    row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    gumbel = -torch.log(-torch.log(torch.rand_like(values).clamp_(1.0e-9, 1.0 - 1.0e-9)))
    perturbed = values + gumbel

    picked = torch.empty(
        int(num_rows),
        dtype=torch.long,
        device=values.device,
    )

    for row in range(int(num_rows)):
        positions = row_ids.eq(row).nonzero(as_tuple=False).flatten()
        if positions.numel() <= 0:
            raise RuntimeError(f"No candidate actions for sampled row {row}.")
        row_values = perturbed.index_select(0, positions)
        picked[row] = positions[int(torch.argmax(row_values).item())]

    return picked


__all__ = [
    "STOP_EDGE_ID",
    "PolicyOutput",
]
