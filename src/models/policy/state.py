from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph_runtime import GraphObservation, GraphTopology


@dataclass(frozen=True)
class SearchState:
    topology: GraphTopology
    observation: GraphObservation
    current_nodes: torch.Tensor
    done_mask: torch.Tensor
    num_steps: torch.Tensor

    @classmethod
    def initialize(
        cls,
        *,
        topology: GraphTopology,
        observation: GraphObservation,
        start_nodes: torch.Tensor,
    ) -> SearchState:
        if start_nodes.dim() != 2:
            raise ValueError(
                "start_nodes must be 2D [num_graphs, num_rollouts], "
                f"got shape={tuple(start_nodes.shape)}."
            )
        return cls(
            topology=topology,
            observation=observation,
            current_nodes=start_nodes.clone(),
            done_mask=torch.zeros_like(start_nodes, dtype=torch.bool),
            num_steps=torch.zeros_like(start_nodes, dtype=torch.long),
        )

    @classmethod
    def from_edge_path(
        cls,
        *,
        topology: GraphTopology,
        observation: GraphObservation,
        start_node: int,
        edge_ids: tuple[int, ...],
        max_steps: int,
        device: torch.device,
    ) -> SearchState:
        num_steps = len(edge_ids)
        if num_steps > max_steps:
            raise ValueError("edge path length cannot exceed max_steps.")
        current_node = int(start_node)
        for edge_id in edge_ids:
            edge_id_int = int(edge_id)
            edge_src = int(topology.edge_index[0, edge_id_int].item())
            edge_dst = int(topology.edge_index[1, edge_id_int].item())
            if edge_src != current_node:
                raise ValueError("edge path is not source-contiguous.")
            current_node = edge_dst
        return cls(
            topology=topology,
            observation=observation,
            current_nodes=torch.tensor(
                [[current_node]], device=device, dtype=torch.long
            ),
            done_mask=torch.zeros((1, 1), device=device, dtype=torch.bool),
            num_steps=torch.tensor([[num_steps]], device=device, dtype=torch.long),
        )

    def flatten_current_nodes(self) -> torch.Tensor:
        return self.current_nodes.view(-1)

    def flatten_done_mask(self) -> torch.Tensor:
        return self.done_mask.view(-1)

    def flatten_num_steps(self) -> torch.Tensor:
        return self.num_steps.view(-1)

    def flatten_graph_index(self) -> torch.Tensor:
        return self.topology.graph_index_from_nodes(self.flatten_current_nodes())


__all__ = ["SearchState"]
