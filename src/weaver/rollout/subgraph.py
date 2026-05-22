from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.data.schema import RetrievalBatch
from src.graph.masks import anchor_node_mask

if TYPE_CHECKING:
    from .result import RolloutResult


@dataclass(frozen=True, slots=True)
class UnionSubgraphMasks:
    """
    Union of terminal subgraphs over rollout samples.

    node_mask:
        Boolean mask over all nodes in the current batched graph, shape [N].

    edge_mask:
        Boolean mask over all edges in the current batched graph, shape [E].
    """

    node_mask: torch.Tensor
    edge_mask: torch.Tensor


class SubgraphReconstructor:
    """
    Reconstruct terminal node/edge masks from rollout trajectories.

    RolloutResult owns trajectory tensors only; this helper owns the
    RetrievalBatch topology required to interpret selected edge ids.
    """

    def __init__(
        self,
        batch: RetrievalBatch,
        *,
        device: torch.device,
    ) -> None:
        self.batch = batch
        self.device = device

    def reconstruct(self, result: RolloutResult) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct the terminal subgraph represented by one rollout result.

        Returned masks are over the full physical batched graph:
            node_mask: [N]
            edge_mask: [E]

        This preserves the legacy evaluator contract: one RolloutResult must
        contain exactly one rollout row per graph in the batch.
        """
        num_graphs = int(self.batch.num_graphs)
        num_nodes = int(self.batch.num_nodes_total)
        num_edges = int(self.batch.edge_index.size(1))

        edge_index = self.batch.edge_index.to(device=self.device, dtype=torch.long)

        node_mask = anchor_node_mask(self.batch, device=self.device)
        edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=self.device)

        selected_edge_ids = result.selected_edge_ids.to(
            device=self.device,
            dtype=torch.long,
        )
        continue_mask = result.expand_mask.to(
            device=self.device,
            dtype=torch.bool,
        )

        if selected_edge_ids.ndim != 2:
            raise ValueError(
                f"selected_edge_ids must have shape [B, T], "
                f"got {tuple(selected_edge_ids.shape)}."
            )

        if continue_mask.shape != selected_edge_ids.shape:
            raise ValueError(
                "continue_mask must have the same shape as selected_edge_ids: "
                f"{tuple(continue_mask.shape)} != {tuple(selected_edge_ids.shape)}."
            )

        batch_size, horizon = selected_edge_ids.shape
        if batch_size != num_graphs:
            raise ValueError(
                "rollout batch size must match batch.num_graphs: "
                f"{batch_size} != {num_graphs}."
            )

        terminal_step = result.terminal_step.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)

        if terminal_step.shape != (num_graphs,):
            raise ValueError(
                f"terminal_step must have shape [{num_graphs}], "
                f"got {tuple(terminal_step.shape)}."
            )

        step_ids = torch.arange(horizon, device=self.device).unsqueeze(0)
        valid_steps = step_ids <= terminal_step.unsqueeze(1)
        valid_expands = valid_steps & continue_mask & selected_edge_ids.ge(0)

        if not bool(valid_expands.any()):
            return node_mask, edge_mask

        edge_ids = selected_edge_ids[valid_expands].view(-1)
        _check_id_range(edge_ids, upper=num_edges, name="selected_edge_ids")

        edge_mask[edge_ids] = True

        endpoints = edge_index[:, edge_ids].reshape(-1)
        _check_id_range(endpoints, upper=num_nodes, name="selected edge endpoints")
        node_mask[endpoints] = True

        return node_mask, edge_mask

    def stack(
        self,
        rollouts: Sequence[RolloutResult],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stack terminal subgraph masks for rollout samples.

        Returns:
            node_masks: [R, N]
            edge_masks: [R, E]
        """
        num_nodes = int(self.batch.num_nodes_total)
        num_edges = int(self.batch.edge_index.size(1))

        if not rollouts:
            return (
                torch.zeros((0, num_nodes), dtype=torch.bool, device=self.device),
                torch.zeros((0, num_edges), dtype=torch.bool, device=self.device),
            )

        nodes: list[torch.Tensor] = []
        edges: list[torch.Tensor] = []

        for rollout in rollouts:
            node_mask, edge_mask = self.reconstruct(rollout)
            nodes.append(node_mask)
            edges.append(edge_mask)

        return torch.stack(nodes, dim=0), torch.stack(edges, dim=0)

    def union(
        self,
        rollouts: Sequence[RolloutResult],
    ) -> UnionSubgraphMasks:
        """
        Compute node / edge union over terminal subgraphs from rollout samples.
        """
        node_masks, edge_masks = self.stack(rollouts)

        if node_masks.numel() == 0:
            return UnionSubgraphMasks(
                node_mask=torch.zeros(
                    int(self.batch.num_nodes_total),
                    dtype=torch.bool,
                    device=self.device,
                ),
                edge_mask=torch.zeros(
                    int(self.batch.edge_index.size(1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            )

        return UnionSubgraphMasks(
            node_mask=node_masks.any(dim=0),
            edge_mask=edge_masks.any(dim=0),
        )


def _check_id_range(
    ids: torch.Tensor,
    *,
    upper: int,
    name: str,
) -> None:
    if ids.numel() == 0:
        return

    min_id = int(ids.min())
    max_id = int(ids.max())

    if min_id < 0 or max_id >= int(upper):
        raise ValueError(
            f"{name} contains ids outside range [0, {upper}): "
            f"min={min_id}, max={max_id}."
        )


__all__ = [
    "SubgraphReconstructor",
    "UnionSubgraphMasks",
]
