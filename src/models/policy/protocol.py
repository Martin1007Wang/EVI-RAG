from __future__ import annotations

from typing import Protocol

import torch

from .state import SearchState
from .types import ForwardActionDistribution, PreparedSearchBatch, StartDistribution


class SearchPolicyProtocol(Protocol):
    def prepare_batch(self, batch) -> PreparedSearchBatch: ...

    def compute_start_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> StartDistribution: ...

    def compute_forward_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> ForwardActionDistribution: ...

    def compute_log_state_scores(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> torch.Tensor: ...

    @staticmethod
    def compute_move_log_probs(
        distribution: ForwardActionDistribution,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


__all__ = ["SearchPolicyProtocol"]
