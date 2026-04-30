from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.graph.ops import build_anchor_induced_edge_mask


@dataclass
class State:
    """
    Batched subgraph state.

        s = (V_s, E_s)

    The rollout budget is external configuration. Derived rollout-depth features
    such as the number of selected non-root edges are computed from ``E_s``
    relative to the invariant root edge set ``E_0``.

    active_nodes:
        Boolean mask over batched nodes.
    active_edges:
        Boolean mask over batched edges.
    root_active_edges:
        Anchor-induced initial edges E_0. These are part of the initial
        subgraph but are not counted as learned expansions.
    expand_budget:
        Rollout configuration attached to this batched state handle. This is
        not a mathematical state component.
    """

    root_active_edges: torch.Tensor
    active_nodes: torch.Tensor
    active_edges: torch.Tensor
    expand_budget: int

    @classmethod
    def create_initial(
        cls,
        batch: RetrievalBatch,
        *,
        expand_budget: int,
    ) -> State:
        device = batch.edge_index.device
        num_nodes = int(batch.num_nodes_total)
        active_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        anchors = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
        if anchors.numel() > 0:
            active_nodes[anchors] = True
        root_edges = build_anchor_induced_edge_mask(
            batch.edge_index.to(device=device, dtype=torch.long),
            active_nodes,
        )
        return cls(
            root_active_edges=root_edges,
            active_nodes=active_nodes,
            active_edges=root_edges.clone(),
            expand_budget=int(expand_budget),
        )

    def detach(self) -> State:
        return State(
            root_active_edges=self.root_active_edges.detach().clone(),
            active_nodes=self.active_nodes.detach().clone(),
            active_edges=self.active_edges.detach().clone(),
            expand_budget=self.expand_budget,
        )

    def apply_expansion(
        self,
        *,
        chosen_edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        if chosen_edges.numel() > 0:
            chosen_edges = chosen_edges.to(
                device=self.active_edges.device,
                dtype=torch.long,
            )
            edge_index = edge_index.to(
                device=self.active_edges.device,
                dtype=torch.long,
            )
            src = edge_index[0].index_select(0, chosen_edges)
            dst = edge_index[1].index_select(0, chosen_edges)
            self.active_edges[chosen_edges] = True
            self.active_nodes[src] = True
            self.active_nodes[dst] = True

    def selected_nonroot_edges(self) -> torch.Tensor:
        return self.active_edges & ~self.root_active_edges.to(
            device=self.active_edges.device,
            dtype=torch.bool,
        )

    @property
    def is_root_state(self) -> bool:
        return not bool(self.selected_nonroot_edges().any())

    def per_graph_selected_nonroot_edge_count(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        edge_batch = edge_batch.to(device=self.active_edges.device, dtype=torch.long)
        counts = torch.bincount(
            edge_batch,
            weights=self.selected_nonroot_edges().to(dtype=torch.float32),
            minlength=int(num_graphs),
        )
        return counts.to(dtype=torch.long)

    def remaining_budget_per_graph(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        counts = self.per_graph_selected_nonroot_edge_count(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )
        budget = torch.full(
            (int(num_graphs),),
            int(self.expand_budget),
            dtype=torch.long,
            device=counts.device,
        )
        return (budget - counts).clamp_min(0)

    def expand_ratio_per_graph(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.expand_budget <= 0:
            return torch.zeros(
                int(num_graphs),
                dtype=torch.float32,
                device=self.active_edges.device,
            )

        counts = self.per_graph_selected_nonroot_edge_count(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        ).to(dtype=torch.float32)
        return (counts / float(self.expand_budget)).clamp(max=1.0)

    def synchronous_rollout_depth(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
        active_graphs: torch.Tensor | None = None,
    ) -> int:
        counts = self.per_graph_selected_nonroot_edge_count(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )
        if active_graphs is not None:
            active_graphs = active_graphs.to(device=counts.device, dtype=torch.bool)
            if active_graphs.shape != (int(num_graphs),):
                raise ValueError(
                    f"active_graphs must have shape [{int(num_graphs)}], got {tuple(active_graphs.shape)}."
                )
            counts = counts[active_graphs]

        if counts.numel() == 0:
            return 0

        if not bool(counts.eq(counts[0]).all()):
            raise RuntimeError(
                "Synchronous rollout depth must match across unfinished graphs, "
                f"got per-graph selected non-root edge counts={counts.tolist()}."
            )
        return int(counts[0].item())

    @property
    def device(self) -> torch.device:
        return self.active_nodes.device


__all__ = ["State"]
