from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.state import ActionSpace

STOP_EDGE_ID = -1


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    action_space: ActionSpace

    state_log_flow: torch.Tensor  # [S]
    stop_log_flow: torch.Tensor  # [S]
    continue_log_flow: torch.Tensor  # [S]

    stop_log_prob: torch.Tensor  # [S]
    continue_log_prob: torch.Tensor  # [S]

    edge_log_flow: torch.Tensor  # [F]
    edge_log_prob: torch.Tensor  # [F]
    conditional_edge_log_prob: torch.Tensor  # [F]

    edge_raw_score: torch.Tensor  # [F]

    @property
    def device(self) -> torch.device:
        return self.state_log_flow.device

    @property
    def num_states(self) -> int:
        return int(self.state_log_flow.numel())

    def sample(self, *, rows: torch.Tensor) -> torch.Tensor:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)

        out = torch.empty(
            int(rows.numel()),
            dtype=torch.long,
            device=self.device,
        )

        for i, row in enumerate(rows.tolist()):
            start = int(self.action_space.expand_ptr[row].item())
            end = int(self.action_space.expand_ptr[row + 1].item())

            logits = torch.cat(
                (
                    self.stop_log_prob[row].view(1),
                    self.edge_log_prob[start:end],
                ),
                dim=0,
            )

            selected = torch.multinomial(
                logits.exp(),
                num_samples=1,
            ).squeeze(0)

            if int(selected.item()) == 0:
                out[i] = int(STOP_EDGE_ID)
            else:
                out[i] = self.action_space.expand_edge_ids[start + int(selected.item()) - 1]

        return out

    def gather_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        row_ids = row_ids.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)

        if int(row_ids.numel()) != int(edge_ids.numel()):
            raise ValueError("row_ids and edge_ids must have the same length.")

        out = torch.empty(
            int(row_ids.numel()),
            dtype=self.stop_log_prob.dtype,
            device=self.device,
        )

        stop = edge_ids.eq(int(STOP_EDGE_ID))
        if bool(stop.any()):
            out[stop] = self.stop_log_prob.index_select(0, row_ids[stop])

        expand = ~stop
        if bool(expand.any()):
            out[expand] = self._gather_edge_log_prob(
                row_ids=row_ids[expand],
                edge_ids=edge_ids[expand],
            )

        return out

    def gather_action_log_flow(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        row_ids = row_ids.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)

        if int(row_ids.numel()) != int(edge_ids.numel()):
            raise ValueError("row_ids and edge_ids must have the same length.")

        out = torch.empty(
            int(row_ids.numel()),
            dtype=self.stop_log_flow.dtype,
            device=self.device,
        )

        stop = edge_ids.eq(int(STOP_EDGE_ID))
        if bool(stop.any()):
            out[stop] = self.stop_log_flow.index_select(0, row_ids[stop])

        expand = ~stop
        if bool(expand.any()):
            out[expand] = self._gather_edge_log_flow(
                row_ids=row_ids[expand],
                edge_ids=edge_ids[expand],
            )

        return out

    def _gather_edge_log_prob(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty(
            int(edge_ids.numel()),
            dtype=self.edge_log_prob.dtype,
            device=self.device,
        )

        for i, row in enumerate(row_ids.tolist()):
            start = int(self.action_space.expand_ptr[row].item())
            end = int(self.action_space.expand_ptr[row + 1].item())

            candidates = self.action_space.expand_edge_ids[start:end]
            match = candidates.eq(edge_ids[i]).nonzero(as_tuple=False).flatten()

            if int(match.numel()) != 1:
                raise ValueError("edge_id is not a legal action for the requested row.")

            out[i] = self.edge_log_prob[start + int(match.item())]

        return out

    def _gather_edge_log_flow(
        self,
        *,
        row_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty(
            int(edge_ids.numel()),
            dtype=self.edge_log_flow.dtype,
            device=self.device,
        )

        for i, row in enumerate(row_ids.tolist()):
            start = int(self.action_space.expand_ptr[row].item())
            end = int(self.action_space.expand_ptr[row + 1].item())

            candidates = self.action_space.expand_edge_ids[start:end]
            match = candidates.eq(edge_ids[i]).nonzero(as_tuple=False).flatten()

            if int(match.numel()) != 1:
                raise ValueError("edge_id is not a legal action for the requested row.")

            out[i] = self.edge_log_flow[start + int(match.item())]

        return out


__all__ = [
    "PolicyOutput",
    "STOP_EDGE_ID",
]
