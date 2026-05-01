from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.graph.ops import build_anchor_induced_edge_mask


@dataclass(slots=True)
class State:
    """
    Batched subgraph state.

        s = (V_s, E_s)

    active_nodes:
        Boolean mask over batched nodes.

    active_edges:
        Boolean mask over batched edges. This is the full current edge set E_s,
        including root edges.

    root_edges:
        Anchor-induced initial edges E_0. These edges are part of the initial
        state, but they are not counted as learned expansion steps.

    expand_budget:
        Maximum number of learned non-root edge expansions per graph. This is
        rollout configuration, not a mathematical state component.
    """

    active_nodes: torch.Tensor
    active_edges: torch.Tensor
    root_edges: torch.Tensor
    expand_budget: int

    @classmethod
    def create_initial(
        cls,
        batch: RetrievalBatch,
        *,
        expand_budget: int,
        validate_anchor_ids: bool = True,
    ) -> State:
        device = batch.edge_index.device
        num_nodes = int(batch.num_nodes_total)

        active_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        anchors = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
        if anchors.numel() > 0:
            valid = anchors.ge(0) & anchors.lt(num_nodes)
            if bool(validate_anchor_ids) and bool((~valid).any()):
                invalid = anchors[~valid]
                raise ValueError(
                    "anchor_node_ids must contain physical node ids in current batch: "
                    f"min={int(invalid.min())}, max={int(invalid.max())}, "
                    f"num_nodes={num_nodes}."
                )
            if bool(valid.any()):
                active_nodes[anchors[valid]] = True

        root_edges = build_anchor_induced_edge_mask(
            edge_index=batch.edge_index.to(device=device, dtype=torch.long),
            anchor_mask=active_nodes,
        ).to(dtype=torch.bool)

        return cls(
            active_nodes=active_nodes,
            active_edges=root_edges.clone(),
            root_edges=root_edges,
            expand_budget=int(expand_budget),
        )

    @property
    def device(self) -> torch.device:
        return self.active_nodes.device

    @property
    def num_nodes(self) -> int:
        return int(self.active_nodes.numel())

    @property
    def num_edges(self) -> int:
        return int(self.active_edges.numel())

    def detach(self) -> State:
        return State(
            active_nodes=self.active_nodes.detach().clone(),
            active_edges=self.active_edges.detach().clone(),
            root_edges=self.root_edges.detach().clone(),
            expand_budget=int(self.expand_budget),
        )

    def active_node_ids(self) -> torch.Tensor:
        return self.active_nodes.nonzero(as_tuple=False).flatten()

    def active_edge_ids(self) -> torch.Tensor:
        return self.active_edges.nonzero(as_tuple=False).flatten()

    def expanded_edge_mask(self) -> torch.Tensor:
        """
        Edges selected by learned rollout expansion:

            E_s \\ E_0
        """
        return self.active_edges & ~self.root_edges

    def expanded_edge_ids(self) -> torch.Tensor:
        return self.expanded_edge_mask().nonzero(as_tuple=False).flatten()

    @property
    def is_root_state(self) -> bool:
        return self.expanded_edge_ids().numel() == 0

    def apply_expansion(
        self,
        *,
        chosen_edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        """
        Mutate the state by adding selected edges and their endpoints.
        """
        if chosen_edges.numel() == 0:
            return

        device = self.device
        chosen_edges = chosen_edges.to(device=device, dtype=torch.long).view(-1)
        valid = chosen_edges.ge(0) & chosen_edges.lt(self.num_edges)
        if bool((~valid).any()):
            invalid = chosen_edges[~valid]
            raise ValueError(
                "chosen_edges must contain physical edge ids in current batch: "
                f"min={int(invalid.min())}, max={int(invalid.max())}, "
                f"num_edges={self.num_edges}."
            )

        edge_index = edge_index.to(device=device, dtype=torch.long)
        src = edge_index[0].index_select(0, chosen_edges)
        dst = edge_index[1].index_select(0, chosen_edges)

        self.active_edges[chosen_edges] = True
        self.active_nodes[src] = True
        self.active_nodes[dst] = True

    def expanded_edge_count_per_graph(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        edge_ids = self.expanded_edge_ids()
        num_graphs = int(num_graphs)

        if edge_ids.numel() == 0:
            return torch.zeros(num_graphs, dtype=torch.long, device=self.device)

        edge_batch = edge_batch.to(device=self.device, dtype=torch.long)
        graph_ids = edge_batch.index_select(0, edge_ids)

        return torch.bincount(
            graph_ids,
            minlength=num_graphs,
        ).to(dtype=torch.long)

    def per_graph_selected_nonroot_edge_count(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        return self.expanded_edge_count_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

    def remaining_budget_per_graph(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        used = self.expanded_edge_count_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        budget = torch.full(
            (int(num_graphs),),
            int(self.expand_budget),
            dtype=torch.long,
            device=used.device,
        )

        return (budget - used).clamp_min(0)

    def expand_ratio_per_graph(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        num_graphs = int(num_graphs)

        if self.expand_budget <= 0:
            return torch.zeros(num_graphs, dtype=torch.float32, device=self.device)

        used = self.expanded_edge_count_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        ).to(dtype=torch.float32)

        return (used / float(self.expand_budget)).clamp(0.0, 1.0)

    def synchronous_rollout_depth(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
        active_graphs: torch.Tensor | None = None,
    ) -> int:
        counts = self.expanded_edge_count_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        if active_graphs is not None:
            active_graphs = active_graphs.to(device=counts.device, dtype=torch.bool)
            if active_graphs.shape != (int(num_graphs),):
                raise ValueError(
                    f"active_graphs must have shape [{int(num_graphs)}], "
                    f"got {tuple(active_graphs.shape)}."
                )
            counts = counts[active_graphs]

        if counts.numel() == 0:
            return 0

        first = counts[0]
        if not bool(counts.eq(first).all()):
            raise RuntimeError(
                "Synchronous rollout depth must match across unfinished graphs, "
                f"got per-graph expanded edge counts={counts.tolist()}."
            )

        return int(first.item())


@dataclass(slots=True)
class RolloutState:
    """
    Dynamic rollout state over one shared static RetrievalBatch.

    Coordinate convention:
        rollout ids: rows in active_nodes / active_edges, shape [R]
        graph ids: original graph ids in the static RetrievalBatch, shape [B]
        node ids: original node ids in the static RetrievalBatch, shape [N]
        edge ids: original edge ids in the static RetrievalBatch, shape [E]

    This is the static-batch / dynamic-rollout split:

        static graph/query/features: B
        dynamic state: R = K * B

    The canonical state invariant is still:

        V_s = anchors(original_graph) union endpoints(E_s)
    """

    active_nodes: torch.Tensor
    active_edges: torch.Tensor
    root_edges: torch.Tensor
    anchor_nodes: torch.Tensor
    rollout_to_graph: torch.Tensor
    expand_budget: int

    @classmethod
    def create_initial(
        cls,
        batch: RetrievalBatch,
        *,
        expand_budget: int,
        rollout_to_graph: torch.Tensor,
        validate_anchor_ids: bool = True,
    ) -> "RolloutState":
        device = batch.edge_index.device
        num_nodes = int(batch.num_nodes_total)
        num_edges = int(batch.edge_index.size(1))
        num_graphs = int(batch.num_graphs)

        rollout_to_graph = rollout_to_graph.to(device=device, dtype=torch.long).view(-1)
        if rollout_to_graph.numel() == 0:
            raise ValueError("rollout_to_graph must contain at least one rollout row.")
        if bool((rollout_to_graph < 0).any()) or bool(
            (rollout_to_graph >= num_graphs).any()
        ):
            raise ValueError(
                "rollout_to_graph must map each rollout row to an original graph id "
                f"in [0, {num_graphs})."
            )

        num_rollouts = int(rollout_to_graph.numel())
        anchor_nodes = torch.zeros(
            (num_rollouts, num_nodes),
            dtype=torch.bool,
            device=device,
        )

        anchors = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
        if anchors.numel() > 0:
            valid = anchors.ge(0) & anchors.lt(num_nodes)
            if bool(validate_anchor_ids) and bool((~valid).any()):
                invalid = anchors[~valid]
                raise ValueError(
                    "anchor_node_ids must contain physical node ids in current batch: "
                    f"min={int(invalid.min())}, max={int(invalid.max())}, "
                    f"num_nodes={num_nodes}."
                )
            anchors = anchors[valid]
            if anchors.numel() > 0:
                node_batch = batch.batch.to(device=device, dtype=torch.long)
                anchor_graph = node_batch.index_select(0, anchors)
                row_ids, anchor_pos = (
                    rollout_to_graph.view(-1, 1)
                    .eq(anchor_graph.view(1, -1))
                    .nonzero(as_tuple=True)
                )
                if row_ids.numel() > 0:
                    anchor_nodes[row_ids, anchors.index_select(0, anchor_pos)] = True

        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
        src, dst = edge_index
        belongs = edge_batch.view(1, num_edges).eq(rollout_to_graph.view(-1, 1))
        root_edges = (
            anchor_nodes.index_select(1, src)
            & anchor_nodes.index_select(1, dst)
            & belongs
        )

        return cls(
            active_nodes=anchor_nodes.clone(),
            active_edges=root_edges.clone(),
            root_edges=root_edges,
            anchor_nodes=anchor_nodes,
            rollout_to_graph=rollout_to_graph,
            expand_budget=int(expand_budget),
        )

    @property
    def device(self) -> torch.device:
        return self.active_nodes.device

    @property
    def num_rollouts(self) -> int:
        return int(self.active_nodes.size(0))

    @property
    def num_nodes(self) -> int:
        return int(self.active_nodes.size(1))

    @property
    def num_edges(self) -> int:
        return int(self.active_edges.size(1))

    def detach(self) -> "RolloutState":
        return RolloutState(
            active_nodes=self.active_nodes.detach().clone(),
            active_edges=self.active_edges.detach().clone(),
            root_edges=self.root_edges.detach().clone(),
            anchor_nodes=self.anchor_nodes.detach().clone(),
            rollout_to_graph=self.rollout_to_graph.detach().clone(),
            expand_budget=int(self.expand_budget),
        )

    def expanded_edge_mask(self) -> torch.Tensor:
        return self.active_edges & ~self.root_edges

    def expanded_edge_count_per_graph(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        del edge_batch
        num_graphs = int(num_graphs)
        if num_graphs != self.num_rollouts:
            raise ValueError(
                "RolloutState expanded counts are indexed by rollout row: "
                f"expected num_graphs={self.num_rollouts}, got {num_graphs}."
            )
        return self.expanded_edge_mask().sum(dim=1).to(dtype=torch.long)

    def per_graph_selected_nonroot_edge_count(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        return self.expanded_edge_count_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

    def remaining_budget_per_graph(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        used = self.expanded_edge_count_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )
        budget = torch.full(
            (self.num_rollouts,),
            int(self.expand_budget),
            dtype=torch.long,
            device=self.device,
        )
        return (budget - used).clamp_min(0)

    def expand_ratio_per_graph(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        del edge_batch
        if int(num_graphs) != self.num_rollouts:
            raise ValueError(
                "RolloutState progress is indexed by rollout row: "
                f"expected num_graphs={self.num_rollouts}, got {num_graphs}."
            )
        if self.expand_budget <= 0:
            return torch.zeros(
                self.num_rollouts,
                dtype=torch.float32,
                device=self.device,
            )
        used = self.expanded_edge_mask().sum(dim=1).to(dtype=torch.float32)
        return (used / float(self.expand_budget)).clamp(0.0, 1.0)

    def apply_expansion(
        self,
        *,
        rollout_ids: torch.Tensor,
        chosen_edges: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        if chosen_edges.numel() == 0:
            return

        device = self.device
        rollout_ids = rollout_ids.to(device=device, dtype=torch.long).view(-1)
        chosen_edges = chosen_edges.to(device=device, dtype=torch.long).view(-1)
        if rollout_ids.shape != chosen_edges.shape:
            raise ValueError(
                "rollout_ids and chosen_edges must have matching shape: "
                f"{tuple(rollout_ids.shape)} != {tuple(chosen_edges.shape)}."
            )
        if bool((rollout_ids < 0).any()) or bool(
            (rollout_ids >= self.num_rollouts).any()
        ):
            raise ValueError(
                "rollout_ids must contain dynamic rollout ids in current state."
            )
        if bool((chosen_edges < 0).any()) or bool(
            (chosen_edges >= self.num_edges).any()
        ):
            raise ValueError(
                "chosen_edges must contain original edge ids in current batch."
            )

        edge_index = edge_index.to(device=device, dtype=torch.long)
        src = edge_index[0].index_select(0, chosen_edges)
        dst = edge_index[1].index_select(0, chosen_edges)

        self.active_edges[rollout_ids, chosen_edges] = True
        self.active_nodes[rollout_ids, src] = True
        self.active_nodes[rollout_ids, dst] = True

    def synchronous_rollout_depth(
        self,
        *,
        edge_batch: torch.Tensor,
        num_graphs: int,
        active_graphs: torch.Tensor | None = None,
    ) -> int:
        counts = self.expanded_edge_count_per_graph(
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        if active_graphs is not None:
            active_graphs = active_graphs.to(device=counts.device, dtype=torch.bool)
            if active_graphs.shape != (self.num_rollouts,):
                raise ValueError(
                    f"active_graphs must have shape [{self.num_rollouts}], "
                    f"got {tuple(active_graphs.shape)}."
                )
            counts = counts[active_graphs]

        if counts.numel() == 0:
            return 0

        first = counts[0]
        if not bool(counts.eq(first).all()):
            raise RuntimeError(
                "Synchronous rollout depth must match across unfinished rollout rows, "
                f"got expanded edge counts={counts.tolist()}."
            )

        return int(first.item())


__all__ = ["RolloutState", "State"]
