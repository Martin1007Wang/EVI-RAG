from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from src.weaver.context import GraphContext
from src.weaver.state import ExpansionBatch, StateBatch

Tensor = torch.Tensor


class TransitionSource(IntEnum):
    POLICY = 0
    WEAK_REPLAY = 1
    ORACLE = 2


@dataclass(frozen=True, slots=True)
class NonterminalTransitionBatch:
    """
    EXPAND transitions for edge-flow matching.

    Semantics:
    - parent_state contains S parent states.
    - parent_state_ids[k] selects a row in parent_state.
    - edge_ids[k] is the physical KG edge expanded from that parent.
    - child_state[k] should equal parent_state[parent_state_ids[k]] + edge_ids[k].
      If child_state is None, it is constructed by StateBatch.branch(...).
    - log_backward[k] is log P_B(parent | child).
      If None, it is computed from backward_kernel.
    - source is optional metadata for metrics only and must not affect loss.
    """

    parent_state: StateBatch
    parent_state_ids: Tensor  # [K]
    edge_ids: Tensor  # [K]
    child_state: StateBatch | None = None
    graph_context: GraphContext | None = None
    log_backward: Tensor | None = None
    source: Tensor | None = None  # [K]

    @property
    def device(self) -> torch.device:
        return self.parent_state.device

    @property
    def num_transitions(self) -> int:
        return int(self.edge_ids.numel())

    def materialize_child_state(self) -> StateBatch:
        if self.child_state is not None:
            return self.child_state
        if self.graph_context is None:
            raise ValueError("graph_context is required to materialize child_state when child_state is absent.")

        return self.parent_state.branch(
            ExpansionBatch(
                state_ids=self.parent_state_ids,
                edge_ids=self.edge_ids,
            ),
            graph_context=self.graph_context,
        )


@dataclass(frozen=True, slots=True)
class TerminalTransitionBatch:
    """
    STOP transitions for terminal-edge reward matching.

    source is optional metadata for metrics only and must not affect loss.
    """

    state: StateBatch
    source: Tensor | None = None  # [S]

    @property
    def device(self) -> torch.device:
        return self.state.device

    @property
    def num_transitions(self) -> int:
        return int(self.state.num_states)


@dataclass(frozen=True, slots=True)
class EdgeFlowMatchingBatch:
    nonterminal: NonterminalTransitionBatch | None
    terminal: TerminalTransitionBatch | None


__all__ = [
    "EdgeFlowMatchingBatch",
    "NonterminalTransitionBatch",
    "TerminalTransitionBatch",
    "TransitionSource",
]
