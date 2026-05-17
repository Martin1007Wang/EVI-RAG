from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from src.weaver.state import State


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    parent_state: State
    child_state: State
    action_edge_ids: torch.Tensor
    log_backward_prob: torch.Tensor

    @property
    def num_transitions(self) -> int:
        return int(self.action_edge_ids.numel())

    @property
    def device(self) -> torch.device:
        return self.action_edge_ids.device

    @classmethod
    def concat(
        cls,
        batches: Sequence[TransitionBatch],
    ) -> TransitionBatch:
        if not batches:
            raise ValueError("Cannot concatenate an empty transition sequence.")
        if len(batches) == 1:
            return batches[0]
        return cls(
            parent_state=State.concat([batch.parent_state for batch in batches]),
            child_state=State.concat([batch.child_state for batch in batches]),
            action_edge_ids=torch.cat(
                [batch.action_edge_ids.view(-1) for batch in batches],
                dim=0,
            ),
            log_backward_prob=torch.cat(
                [batch.log_backward_prob.view(-1) for batch in batches],
                dim=0,
            ),
        )

    def select_rows(
        self,
        rows: torch.Tensor,
    ) -> TransitionBatch:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        return TransitionBatch(
            parent_state=self.parent_state.select_rows(rows),
            child_state=self.child_state.select_rows(rows),
            action_edge_ids=self.action_edge_ids.index_select(0, rows),
            log_backward_prob=self.log_backward_prob.index_select(0, rows),
        )


# Temporary compatibility alias for callers not yet updated.
FlowTransitionBatch = TransitionBatch


__all__ = ["FlowTransitionBatch", "TransitionBatch"]
