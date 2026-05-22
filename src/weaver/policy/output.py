from __future__ import annotations

from dataclasses import dataclass

import torch

TERMINAL_EDGE_ID = -1


@dataclass(frozen=True, slots=True)
class ForwardPolicyOutput:
    """
    Forward policy output for a flat stop-or-edge action-flow distribution.
    """

    frontier_row_ids: torch.Tensor
    frontier_edge_ids: torch.Tensor
    stop_log_flow: torch.Tensor
    continue_log_flow: torch.Tensor
    continue_log_gain: torch.Tensor
    edge_log_flow: torch.Tensor
    edge_log_reference: torch.Tensor
    edge_log_advantage: torch.Tensor
    state_log_flow: torch.Tensor
    stop_log_prob: torch.Tensor
    edge_log_prob: torch.Tensor
    num_rows: int
    num_edges: int

    def __post_init__(self) -> None:
        device = self.stop_log_flow.device
        float_fields = {
            "stop_log_flow": self.stop_log_flow,
            "continue_log_flow": self.continue_log_flow,
            "continue_log_gain": self.continue_log_gain,
            "edge_log_flow": self.edge_log_flow,
            "edge_log_reference": self.edge_log_reference,
            "edge_log_advantage": self.edge_log_advantage,
            "state_log_flow": self.state_log_flow,
            "stop_log_prob": self.stop_log_prob,
            "edge_log_prob": self.edge_log_prob,
        }
        for name, value in float_fields.items():
            if value.device != device:
                raise ValueError(f"{name} must be on the same device as stop_log_flow.")
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must use torch.float32.")

        index_fields = {
            "frontier_row_ids": self.frontier_row_ids,
            "frontier_edge_ids": self.frontier_edge_ids,
        }
        for name, value in index_fields.items():
            if value.device != device:
                raise ValueError(f"{name} must be on the same device as stop_log_flow.")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long.")

    @property
    def edge_row_ids(self) -> torch.Tensor:
        return self.frontier_row_ids

    @property
    def edge_ids(self) -> torch.Tensor:
        return self.frontier_edge_ids

    @property
    def stop_vs_continue_log_ratio(self) -> torch.Tensor:
        return self.stop_log_flow.float() - self.continue_log_flow

    def has_edge(self) -> torch.Tensor:
        return _rows_with_edges(
            edge_row_ids=self.frontier_row_ids,
            num_rows=int(self.num_rows),
            device=self.stop_log_flow.device,
        )

    def stop_prob(self) -> torch.Tensor:
        return self.stop_log_prob.exp()

    def action_log_prob(self) -> torch.Tensor:
        return torch.cat(
            [
                self.stop_log_prob,
                self.edge_log_prob,
            ],
            dim=0,
        )

    def edge_prob_mass(self) -> torch.Tensor:
        mass = self.stop_log_flow.new_zeros((int(self.num_rows),)).float()
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
            edge_values=self.edge_log_prob,
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
            device=self.stop_log_flow.device,
        )

    def edge_action_entropy(self) -> torch.Tensor:
        entropy = self.stop_log_flow.new_zeros((int(self.num_rows),)).float()
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
        rows = rows.to(device=self.stop_log_flow.device, dtype=torch.long).view(-1)
        picked_edge_ids = torch.full(
            (rows.numel(),),
            TERMINAL_EDGE_ID,
            dtype=torch.long,
            device=rows.device,
        )
        if rows.numel() == 0:
            return picked_edge_ids

        for out_pos, row in enumerate(rows.tolist()):
            edge_positions = self.frontier_row_ids.eq(int(row)).nonzero(as_tuple=False).flatten()
            values = [self.stop_log_prob[int(row)].float()]
            edge_ids = [TERMINAL_EDGE_ID]
            if edge_positions.numel() > 0:
                values.append(self.edge_log_prob.index_select(0, edge_positions).float())
                edge_ids.extend(self.frontier_edge_ids.index_select(0, edge_positions).tolist())
            logits = torch.cat([value.view(-1) for value in values], dim=0)
            gumbel = -torch.empty_like(logits).exponential_().log()
            picked = int(torch.argmax(logits + gumbel).item())
            picked_edge_ids[out_pos] = int(edge_ids[picked])
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
        edge_keys = out.frontier_row_ids * key_width + out.frontier_edge_ids
        target_keys = row_ids[edge_mask] * key_width + edge_ids[edge_mask]
        order = torch.argsort(edge_keys)
        sorted_keys = edge_keys.index_select(0, order)
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


__all__ = [
    "ForwardPolicyOutput",
    "TERMINAL_EDGE_ID",
]
