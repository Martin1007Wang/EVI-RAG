from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.schema import RetrievalBatch
from src.weaver.state import State

from .candidate_context import CandidateContext, build_candidate_context
from .feature_encoder import FeatureBank
from .state_readout import StateContext


@dataclass(frozen=True, slots=True)
class TransitionFeatureOutput:
    values: torch.Tensor
    names: tuple[str, ...]


class TransitionFeatureBuilder(nn.Module):
    """
    Minimal transition-type features for the learned density correction.

    Semantic relevance lives in the base measure z0. These features only tell
    the residual whether a frontier edge adds a new endpoint from either side or
    closes an internal edge.
    """

    names = ("src_active_dst_new", "dst_active_src_new", "both_active")

    def __init__(
        self,
        *,
        hidden_dim: int,
        dde_dim: int = 0,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        # DDE already enters node_h through FeatureEncoder; kept for constructor
        # compatibility with policy runtime wiring.
        self.dde_dim = max(0, int(dde_dim))

    @property
    def feature_dim(self) -> int:
        return len(self.names)

    def forward(
        self,
        *,
        fb: FeatureBank,
        context: StateContext,
        batch: RetrievalBatch,
        state: State,
        candidate_edge_ids: torch.Tensor,
        candidate_batch_ids: torch.Tensor | None = None,
        candidate_context: CandidateContext | None = None,
    ) -> TransitionFeatureOutput:
        del context

        device = fb.node_h.device
        dtype = fb.node_h.dtype

        candidates = candidate_context or build_candidate_context(
            batch=batch,
            state=state,
            candidate_edge_ids=candidate_edge_ids,
            candidate_batch_ids=candidate_batch_ids,
            device=device,
        )

        if candidates.num_candidates == 0:
            return TransitionFeatureOutput(
                values=fb.node_h.new_zeros((0, self.feature_dim)),
                names=self.names,
            )

        src_active = candidates.src_active.to(device=device, dtype=torch.bool)
        dst_active = candidates.dst_active.to(device=device, dtype=torch.bool)
        if not bool((src_active | dst_active).all()):
            raise RuntimeError(
                "TransitionFeatureBuilder received a non-frontier edge: every "
                "candidate must have at least one active endpoint."
            )

        values = torch.stack(
            [
                src_active & ~dst_active,
                dst_active & ~src_active,
                src_active & dst_active,
            ],
            dim=-1,
        ).to(dtype=dtype)

        return TransitionFeatureOutput(values=values, names=self.names)


__all__ = ["TransitionFeatureBuilder", "TransitionFeatureOutput"]
