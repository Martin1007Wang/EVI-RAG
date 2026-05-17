from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
from torch import nn

from src.weaver.context import FlowContext
from src.weaver.state import Frontier, State

from .feature_encoder import FeatureBank


@dataclass(frozen=True, slots=True)
class FrontierEncoding:
    """
    Action-level tensors aligned by frontier position.

    For every i in [0, F):

        row_ids[i]  = rollout-state row
        edge_ids[i] = physical batch edge id

    The transition is:

        z_{row_ids[i]} -> z_{row_ids[i]} + edge_ids[i]

    All tensors with first dimension F are aligned to the same legal frontier
    action list. No consumer should pass row_ids separately.

    edge_direction is carried as discrete structural metadata. FeatureBank does
    not precompute direction-specific edge features; the scorer is responsible
    for consuming this field if direction should affect action scores.
    """

    row_ids: torch.Tensor  # [F], long
    edge_ids: torch.Tensor  # [F], long

    edge_h: torch.Tensor  # [F, H]
    query_h: torch.Tensor  # [F, H], graph-level query broadcast to actions

    src_sem_h: torch.Tensor  # [F, D]
    rel_sem_h: torch.Tensor  # [F, D]
    dst_sem_h: torch.Tensor  # [F, D]
    query_sem_h: torch.Tensor  # [F, D], graph-level query broadcast to actions
    edge_direction: torch.Tensor  # [F], long: 0=forward, 1=backward, 2=internal

    @property
    def num_actions(self) -> int:
        return int(self.edge_ids.numel())

    @property
    def num_edges(self) -> int:
        warnings.warn(
            "FrontierEncoding.num_edges is deprecated; use num_actions.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_actions

    @property
    def device(self) -> torch.device:
        return self.edge_ids.device


class FrontierEncoder(nn.Module):
    """
    Materializes action-level features for legal frontier transitions.

    Responsibilities:
    - preserve frontier identity: row_ids and edge_ids;
    - gather edge, endpoint, relation, and query features;
    - return one aligned FrontierEncoding object.

    Non-responsibilities:
    - build the frontier;
    - repair malformed frontier shapes;
    - score actions;
    - compute probabilities;
    - mix actions with attention/message passing.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

    def forward(
        self,
        *,
        context: FlowContext,
        features: FeatureBank,
        state: State,
        frontier: Frontier,
    ) -> FrontierEncoding:
        if features.edge_h.shape[-1] != self.hidden_dim:
            raise ValueError(
                "features.edge_h hidden dimension must match FrontierEncoder hidden_dim: "
                f"{features.edge_h.shape[-1]} != {self.hidden_dim}."
            )
        if features.query_h.shape[-1] != self.hidden_dim:
            raise ValueError(
                "features.query_h hidden dimension must match FrontierEncoder hidden_dim: "
                f"{features.query_h.shape[-1]} != {self.hidden_dim}."
            )

        row_ids = frontier.row_ids
        edge_ids = frontier.edge_ids

        endpoints = context.edge_index.index_select(1, edge_ids)
        src_ids = endpoints[0]
        dst_ids = endpoints[1]

        graph_ids = state.row_to_graph.index_select(0, row_ids)

        return FrontierEncoding(
            row_ids=row_ids,
            edge_ids=edge_ids,
            edge_h=features.edge_h.index_select(0, edge_ids),
            query_h=features.query_h.index_select(0, graph_ids),
            src_sem_h=features.node_sem_h.index_select(0, src_ids),
            rel_sem_h=features.rel_sem_h.index_select(0, edge_ids),
            dst_sem_h=features.node_sem_h.index_select(0, dst_ids),
            query_sem_h=features.query_sem_h.index_select(0, graph_ids),
            edge_direction=frontier.edge_direction.to(
                device=edge_ids.device,
                dtype=torch.long,
            ),
        )


__all__ = ["FrontierEncoding", "FrontierEncoder"]
