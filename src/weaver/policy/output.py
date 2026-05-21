from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph.segments import segment_log_softmax, segment_logsumexp

STOP_EDGE_ID = -1


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """
    Forward policy output.

        stop_logit     = T_theta(s)
        edge_logit     = G_theta(s, e)
        state_log_flow = log F_theta(s)

    """

    stop_logit: torch.Tensor
    edge_logit: torch.Tensor
    state_log_flow: torch.Tensor
    edge_row_ids: torch.Tensor
    edge_ids: torch.Tensor
    num_rows: int
    num_edges: int

    def log_flow(self) -> torch.Tensor:
        stop_logit, edge_logit = self._tempered_logits(1.0)
        return _state_log_flow(
            stop_logit=stop_logit,
            edge_logit=edge_logit,
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
        )

    def has_edge(self) -> torch.Tensor:
        return _rows_with_edges(
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
            device=self.stop_logit.device,
        )

    def action_log_prob(self, *, temperature: float = 1.0) -> torch.Tensor:
        return torch.cat(
            [
                self.stop_log_prob(temperature=temperature),
                self.edge_log_prob(temperature=temperature),
            ],
            dim=0,
        )

    def stop_log_prob(self, *, temperature: float = 1.0) -> torch.Tensor:
        stop_logit, edge_logit = self._tempered_logits(temperature)
        return _stop_log_prob(
            stop_logit=stop_logit,
            edge_logit=edge_logit,
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
        )

    def stop_prob(self, *, temperature: float = 1.0) -> torch.Tensor:
        return self.stop_log_prob(temperature=temperature).exp()

    def continue_log_prob(self, *, temperature: float = 1.0) -> torch.Tensor:
        stop_logit, edge_logit = self._tempered_logits(temperature)
        return _continue_log_prob(
            stop_logit=stop_logit,
            edge_logit=edge_logit,
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
        )

    def continue_prob(self, *, temperature: float = 1.0) -> torch.Tensor:
        return self.continue_log_prob(temperature=temperature).exp()

    def edge_cond_log_prob(self, *, temperature: float = 1.0) -> torch.Tensor:
        _, edge_logit = self._tempered_logits(temperature)
        return _edge_cond_log_prob(
            edge_logit=edge_logit,
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
        )

    def edge_log_prob(self, *, temperature: float = 1.0) -> torch.Tensor:
        stop_logit, edge_logit = self._tempered_logits(temperature)
        return _edge_log_prob(
            stop_logit=stop_logit,
            edge_logit=edge_logit,
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
        )

    def edge_prob_mass(self, *, temperature: float = 1.0) -> torch.Tensor:
        mass = self.stop_logit.new_zeros((int(self.num_rows),)).float()
        if self.edge_logit.numel() == 0:
            return mass
        probs = self.edge_log_prob(temperature=temperature).exp()
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
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return _gather_by_action_keys(
            stop_values=self.stop_log_prob(temperature=temperature),
            edge_values=self.edge_log_prob(temperature=temperature),
            out=self,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

    def gather_continue_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        row_ids = row_ids.to(device=self.stop_logit.device, dtype=torch.long).view(-1)
        return self.continue_log_prob(temperature=temperature).index_select(0, row_ids)

    def gather_edge_cond_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return _gather_by_action_keys(
            stop_values=self.stop_logit.new_zeros((int(self.num_rows),)).float(),
            edge_values=self.edge_cond_log_prob(temperature=temperature),
            out=self,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

    def gather_log_flow(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return _gather_by_action_keys(
            stop_values=self.stop_logit.float() / float(temperature),
            edge_values=self.edge_logit.float() / float(temperature),
            out=self,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

    def sample(
        self,
        *,
        rows: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        rows = rows.to(device=self.stop_logit.device, dtype=torch.long).view(-1)
        stop_edge_ids = torch.full(
            (rows.numel(),),
            STOP_EDGE_ID,
            dtype=torch.long,
            device=rows.device,
        )
        if rows.numel() == 0:
            return stop_edge_ids

        sampled = _sample_actions(
            stop_logit=self.stop_logit.float(),
            edge_logit=self.edge_logit.float(),
            edge_row_ids=self.edge_row_ids,
            edge_ids=self.edge_ids,
            rows=rows,
            num_rows=int(self.num_rows),
            temperature=float(temperature),
        )
        return sampled.to(device=stop_edge_ids.device, dtype=stop_edge_ids.dtype)

    def _tempered_logits(self, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        return self.stop_logit.float() / temperature, self.edge_logit.float() / temperature

    def frontier_size(self) -> torch.Tensor:
        return _frontier_size(
            edge_row_ids=self.edge_row_ids,
            num_rows=int(self.num_rows),
            device=self.stop_logit.device,
        )

    def edge_cond_entropy(self, *, temperature: float = 1.0) -> torch.Tensor:
        entropy = self.stop_logit.new_zeros((int(self.num_rows),)).float()
        if self.edge_logit.numel() == 0:
            return entropy
        edge_row_ids = self.edge_row_ids.to(device=self.edge_logit.device, dtype=torch.long)
        log_prob = self.edge_cond_log_prob(temperature=temperature)
        prob = log_prob.exp()
        entropy.scatter_add_(0, edge_row_ids, -(prob * log_prob))
        return entropy


def _state_log_flow(
    *,
    stop_logit: torch.Tensor,
    edge_logit: torch.Tensor,
    edge_row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    state_log_flow = stop_logit.float()
    if edge_logit.numel() == 0:
        return state_log_flow
    edge_row_ids = edge_row_ids.to(device=edge_logit.device, dtype=torch.long)
    edge_log_flow = segment_logsumexp(
        values=edge_logit.float(),
        segment_ids=edge_row_ids,
        num_segments=int(num_rows),
    )
    return torch.logaddexp(state_log_flow, edge_log_flow)


def _stop_log_prob(
    *,
    stop_logit: torch.Tensor,
    edge_logit: torch.Tensor,
    edge_row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    return stop_logit.float() - _state_log_flow(
        stop_logit=stop_logit,
        edge_logit=edge_logit,
        edge_row_ids=edge_row_ids,
        num_rows=int(num_rows),
    )


def _continue_log_prob(
    *,
    stop_logit: torch.Tensor,
    edge_logit: torch.Tensor,
    edge_row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    state_log_flow = _state_log_flow(
        stop_logit=stop_logit,
        edge_logit=edge_logit,
        edge_row_ids=edge_row_ids,
        num_rows=int(num_rows),
    )
    edge_row_ids = edge_row_ids.to(device=state_log_flow.device, dtype=torch.long)
    edge_mass = state_log_flow.new_full((int(num_rows),), -torch.inf)
    if edge_logit.numel() > 0:
        edge_mass = segment_logsumexp(
            values=edge_logit.float(),
            segment_ids=edge_row_ids,
            num_segments=int(num_rows),
        )
    return edge_mass - state_log_flow


def _edge_cond_log_prob(
    *,
    edge_logit: torch.Tensor,
    edge_row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    if edge_logit.numel() == 0:
        return edge_logit.new_empty((0,)).float()
    return segment_log_softmax(
        edge_logit.float(),
        edge_row_ids.to(device=edge_logit.device, dtype=torch.long),
        num_segments=int(num_rows),
    )


def _edge_log_prob(
    *,
    stop_logit: torch.Tensor,
    edge_logit: torch.Tensor,
    edge_row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    if edge_logit.numel() == 0:
        return edge_logit.new_empty((0,)).float()
    edge_row_ids = edge_row_ids.to(device=edge_logit.device, dtype=torch.long)
    return edge_logit.float() - _state_log_flow(
        stop_logit=stop_logit,
        edge_logit=edge_logit,
        edge_row_ids=edge_row_ids,
        num_rows=int(num_rows),
    ).index_select(0, edge_row_ids)


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
        duplicate_keys = sorted_keys[1:].eq(sorted_keys[:-1]) if sorted_keys.numel() > 1 else torch.empty(0, dtype=torch.bool, device=sorted_keys.device)
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


def _sample_actions(
    *,
    stop_logit: torch.Tensor,
    edge_logit: torch.Tensor,
    edge_row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    rows: torch.Tensor,
    num_rows: int,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    picked_edge_ids = torch.full(
        (rows.numel(),),
        STOP_EDGE_ID,
        dtype=torch.long,
        device=rows.device,
    )
    if rows.numel() == 0:
        return picked_edge_ids

    stop_action_row_ids = rows
    stop_action_edge_ids = torch.full_like(rows, STOP_EDGE_ID)
    if edge_ids.numel() == 0:
        action_row_ids = stop_action_row_ids
        action_edge_ids = stop_action_edge_ids
        action_values = stop_logit.index_select(0, rows) / float(temperature)
    else:
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
        action_row_ids = torch.cat([stop_action_row_ids, local_edge_row_ids], dim=0)
        action_edge_ids = torch.cat([stop_action_edge_ids, kept_edge_ids], dim=0)
        action_values = torch.cat(
            [
                stop_logit.index_select(0, rows),
                edge_logit.to(device=rows.device)[keep_edges],
            ],
            dim=0,
        ) / float(temperature)

    picked_positions = _segment_gumbel_argmax(
        values=action_values,
        row_ids=action_row_ids,
        num_rows=int(rows.numel()),
    )
    return action_edge_ids.index_select(0, picked_positions)


__all__ = [
    "STOP_EDGE_ID",
    "PolicyOutput",
]
