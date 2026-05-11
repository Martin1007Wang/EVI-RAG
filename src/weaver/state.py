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
    boundary_nodes: torch.Tensor | None = None

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
            boundary_nodes=active_nodes.clone(),
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
            boundary_nodes=(
                None
                if self.boundary_nodes is None
                else self.boundary_nodes.detach().clone()
            ),
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
        if self.boundary_nodes is not None:
            self.boundary_nodes[src] = False
            self.boundary_nodes[dst] = True
        self.assert_budget_invariant(edge_batch=None, num_graphs=None)

    def assert_budget_invariant(
        self,
        *,
        edge_batch: torch.Tensor | None,
        num_graphs: int | None,
    ) -> None:
        if edge_batch is None or num_graphs is None:
            used = int(self.expanded_edge_mask().sum().item())
            if used > int(self.expand_budget):
                raise AssertionError(
                    "Budget invariant failed: |E_z \\ E_0| exceeds B."
                )
            return
        remaining = self.remaining_budget_per_graph(
            edge_batch=edge_batch,
            num_graphs=int(num_graphs),
        )
        used = self.expanded_edge_count_per_graph(
            edge_batch=edge_batch,
            num_graphs=int(num_graphs),
        )
        expected = int(self.expand_budget) - used
        if not bool(torch.equal(remaining, expected.clamp_min(0))):
            raise AssertionError("Budget invariant failed: b_z != B - |E_z \\ E_0|.")

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


@dataclass(slots=True, init=False)
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

    rollout_to_graph: torch.Tensor
    expand_budget: int
    edge_index: torch.Tensor | None
    _num_nodes: int
    _num_edges: int
    expanded_edge_trace: torch.Tensor | None = None
    expanded_edge_lengths: torch.Tensor | None = None
    anchor_node_trace: torch.Tensor | None = None
    anchor_node_lengths: torch.Tensor | None = None
    root_edge_trace: torch.Tensor | None = None
    root_edge_lengths: torch.Tensor | None = None
    _active_nodes: torch.Tensor | None = None
    _active_edges: torch.Tensor | None = None
    _root_edges: torch.Tensor | None = None
    _anchor_nodes: torch.Tensor | None = None
    _boundary_nodes: torch.Tensor | None = None

    def __init__(
        self,
        *,
        rollout_to_graph: torch.Tensor,
        expand_budget: int,
        edge_index: torch.Tensor | None = None,
        num_nodes: int | None = None,
        num_edges: int | None = None,
        expanded_edge_trace: torch.Tensor | None = None,
        expanded_edge_lengths: torch.Tensor | None = None,
        anchor_node_trace: torch.Tensor | None = None,
        anchor_node_lengths: torch.Tensor | None = None,
        root_edge_trace: torch.Tensor | None = None,
        root_edge_lengths: torch.Tensor | None = None,
        active_nodes: torch.Tensor | None = None,
        active_edges: torch.Tensor | None = None,
        root_edges: torch.Tensor | None = None,
        anchor_nodes: torch.Tensor | None = None,
        boundary_nodes: torch.Tensor | None = None,
    ) -> None:
        device = rollout_to_graph.device
        if edge_index is not None:
            edge_index = edge_index.to(device=device, dtype=torch.long)
            if edge_index.ndim != 2 or edge_index.size(0) != 2:
                raise ValueError(
                    f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
                )

        rollout_to_graph = rollout_to_graph.to(device=device, dtype=torch.long).view(-1)
        if rollout_to_graph.numel() == 0:
            raise ValueError("rollout_to_graph must contain at least one rollout row.")

        inferred_nodes = _infer_num_nodes(
            num_nodes=num_nodes,
            active_nodes=active_nodes,
            anchor_nodes=anchor_nodes,
            boundary_nodes=boundary_nodes,
            edge_index=edge_index,
        )
        inferred_edges = _infer_num_edges(
            num_edges=num_edges,
            active_edges=active_edges,
            root_edges=root_edges,
            edge_index=edge_index,
        )

        self.rollout_to_graph = rollout_to_graph
        self.expand_budget = int(expand_budget)
        self.edge_index = edge_index
        self._num_nodes = int(inferred_nodes)
        self._num_edges = int(inferred_edges)

        self._active_nodes = _optional_bool_cache(active_nodes, device=device)
        self._active_edges = _optional_bool_cache(active_edges, device=device)
        self._root_edges = _optional_bool_cache(root_edges, device=device)
        self._anchor_nodes = _optional_bool_cache(anchor_nodes, device=device)
        self._boundary_nodes = _optional_bool_cache(boundary_nodes, device=device)

        if anchor_node_trace is None and self._anchor_nodes is not None:
            anchor_node_trace, anchor_node_lengths = _dense_bool_rows_to_trace(
                self._anchor_nodes,
                fill_value=-1,
            )
        if root_edge_trace is None and self._root_edges is not None:
            root_edge_trace, root_edge_lengths = _dense_bool_rows_to_trace(
                self._root_edges,
                fill_value=-1,
            )
        if expanded_edge_trace is None and self._active_edges is not None:
            if self._root_edges is None:
                expanded_mask = self._active_edges
            else:
                expanded_mask = self._active_edges & ~self._root_edges
            expanded_edge_trace, expanded_edge_lengths = _dense_bool_rows_to_trace(
                expanded_mask,
                fill_value=-1,
            )

        self.expanded_edge_trace = _optional_long_matrix(
            expanded_edge_trace,
            device=device,
        )
        self.expanded_edge_lengths = _optional_long_vector(
            expanded_edge_lengths,
            device=device,
            rows=self.num_rollouts,
            name="expanded_edge_lengths",
        )
        self.anchor_node_trace = _optional_long_matrix(
            anchor_node_trace,
            device=device,
        )
        self.anchor_node_lengths = _optional_long_vector(
            anchor_node_lengths,
            device=device,
            rows=self.num_rollouts,
            name="anchor_node_lengths",
        )
        self.root_edge_trace = _optional_long_matrix(
            root_edge_trace,
            device=device,
        )
        self.root_edge_lengths = _optional_long_vector(
            root_edge_lengths,
            device=device,
            rows=self.num_rollouts,
            name="root_edge_lengths",
        )
        self._validate_trace_pair(
            trace=self.expanded_edge_trace,
            lengths=self.expanded_edge_lengths,
            name="expanded_edge",
        )
        self._validate_trace_pair(
            trace=self.anchor_node_trace,
            lengths=self.anchor_node_lengths,
            name="anchor_node",
        )
        self._validate_trace_pair(
            trace=self.root_edge_trace,
            lengths=self.root_edge_lengths,
            name="root_edge",
        )

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

        node_batch = batch.batch.to(device=device, dtype=torch.long)
        anchor_graph = (
            node_batch.index_select(0, anchors)
            if anchors.numel() > 0
            else anchors.new_empty((0,))
        )
        anchor_node_trace, anchor_node_lengths = _items_by_rollout_trace(
            item_ids=anchors,
            item_graph_ids=anchor_graph,
            rollout_to_graph=rollout_to_graph,
            fill_value=-1,
        )

        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        anchor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if anchors.numel() > 0:
            anchor_mask[anchors] = True
        root_edge_ids = build_anchor_induced_edge_mask(
            edge_index=edge_index,
            anchor_mask=anchor_mask,
        ).nonzero(as_tuple=False).flatten()
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
        root_graph = (
            edge_batch.index_select(0, root_edge_ids)
            if root_edge_ids.numel() > 0
            else root_edge_ids.new_empty((0,))
        )
        root_edge_trace, root_edge_lengths = _items_by_rollout_trace(
            item_ids=root_edge_ids,
            item_graph_ids=root_graph,
            rollout_to_graph=rollout_to_graph,
            fill_value=-1,
        )

        expanded_edge_trace = torch.full(
            (int(rollout_to_graph.numel()), max(int(expand_budget), 0)),
            -1,
            dtype=torch.long,
            device=device,
        )
        expanded_edge_lengths = torch.zeros(
            int(rollout_to_graph.numel()),
            dtype=torch.long,
            device=device,
        )

        return cls(
            rollout_to_graph=rollout_to_graph,
            expand_budget=int(expand_budget),
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_edges=num_edges,
            expanded_edge_trace=expanded_edge_trace,
            expanded_edge_lengths=expanded_edge_lengths,
            anchor_node_trace=anchor_node_trace,
            anchor_node_lengths=anchor_node_lengths,
            root_edge_trace=root_edge_trace,
            root_edge_lengths=root_edge_lengths,
        )

    @property
    def device(self) -> torch.device:
        return self.rollout_to_graph.device

    @property
    def num_rollouts(self) -> int:
        return int(self.rollout_to_graph.numel())

    @property
    def num_nodes(self) -> int:
        return int(self._num_nodes)

    @property
    def num_edges(self) -> int:
        return int(self._num_edges)

    @property
    def active_nodes(self) -> torch.Tensor:
        if self._active_nodes is not None:
            return self._active_nodes
        return self.materialize_active_nodes()

    @property
    def active_edges(self) -> torch.Tensor:
        if self._active_edges is not None:
            return self._active_edges
        return self.materialize_active_edges()

    @property
    def root_edges(self) -> torch.Tensor:
        if self._root_edges is not None:
            return self._root_edges
        return self.materialize_root_edges()

    @property
    def anchor_nodes(self) -> torch.Tensor:
        if self._anchor_nodes is not None:
            return self._anchor_nodes
        return self.materialize_anchor_nodes()

    @property
    def boundary_nodes(self) -> torch.Tensor:
        if self._boundary_nodes is not None:
            return self._boundary_nodes
        return self.materialize_boundary_nodes()

    def detach(self) -> "RolloutState":
        return RolloutState(
            rollout_to_graph=self.rollout_to_graph.detach().clone(),
            expand_budget=int(self.expand_budget),
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            expanded_edge_trace=(
                None
                if self.expanded_edge_trace is None
                else self.expanded_edge_trace.detach().clone()
            ),
            expanded_edge_lengths=(
                None
                if self.expanded_edge_lengths is None
                else self.expanded_edge_lengths.detach().clone()
            ),
            anchor_node_trace=(
                None
                if self.anchor_node_trace is None
                else self.anchor_node_trace.detach().clone()
            ),
            anchor_node_lengths=(
                None
                if self.anchor_node_lengths is None
                else self.anchor_node_lengths.detach().clone()
            ),
            root_edge_trace=(
                None
                if self.root_edge_trace is None
                else self.root_edge_trace.detach().clone()
            ),
            root_edge_lengths=(
                None
                if self.root_edge_lengths is None
                else self.root_edge_lengths.detach().clone()
            ),
            boundary_nodes=(
                None
                if self._boundary_nodes is None
                else self._boundary_nodes.detach().clone()
            ),
        )

    def snapshot(self) -> "RolloutState":
        return RolloutState(
            rollout_to_graph=self.rollout_to_graph.clone(),
            expand_budget=int(self.expand_budget),
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            expanded_edge_trace=_clone_optional_tensor(self.expanded_edge_trace),
            expanded_edge_lengths=_clone_optional_tensor(self.expanded_edge_lengths),
            anchor_node_trace=_clone_optional_tensor(self.anchor_node_trace),
            anchor_node_lengths=_clone_optional_tensor(self.anchor_node_lengths),
            root_edge_trace=_clone_optional_tensor(self.root_edge_trace),
            root_edge_lengths=_clone_optional_tensor(self.root_edge_lengths),
            active_nodes=_clone_optional_tensor(self._active_nodes),
            active_edges=_clone_optional_tensor(self._active_edges),
            root_edges=_clone_optional_tensor(self._root_edges),
            anchor_nodes=_clone_optional_tensor(self._anchor_nodes),
            boundary_nodes=_clone_optional_tensor(self._boundary_nodes),
        )

    def select_rollouts(self, rollout_ids: torch.Tensor) -> "RolloutState":
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        if rollout_ids.numel() == 0:
            raise ValueError("rollout_ids must contain at least one row.")
        if bool(rollout_ids.lt(0).any()) or bool(
            rollout_ids.ge(self.num_rollouts).any()
        ):
            raise ValueError(
                "rollout_ids must contain dynamic rollout ids in current state."
            )

        return RolloutState(
            rollout_to_graph=self.rollout_to_graph.index_select(0, rollout_ids),
            expand_budget=int(self.expand_budget),
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            expanded_edge_trace=_index_optional_matrix(
                self.expanded_edge_trace,
                rollout_ids,
            ),
            expanded_edge_lengths=_index_optional_vector(
                self.expanded_edge_lengths,
                rollout_ids,
            ),
            anchor_node_trace=_index_optional_matrix(
                self.anchor_node_trace,
                rollout_ids,
            ),
            anchor_node_lengths=_index_optional_vector(
                self.anchor_node_lengths,
                rollout_ids,
            ),
            root_edge_trace=_index_optional_matrix(
                self.root_edge_trace,
                rollout_ids,
            ),
            root_edge_lengths=_index_optional_vector(
                self.root_edge_lengths,
                rollout_ids,
            ),
            active_nodes=_index_optional_matrix(self._active_nodes, rollout_ids),
            active_edges=_index_optional_matrix(self._active_edges, rollout_ids),
            root_edges=_index_optional_matrix(self._root_edges, rollout_ids),
            anchor_nodes=_index_optional_matrix(self._anchor_nodes, rollout_ids),
            boundary_nodes=_index_optional_matrix(self._boundary_nodes, rollout_ids),
        )

    def expanded_edge_mask(self) -> torch.Tensor:
        return self.materialize_expanded_edges()

    def expanded_edge_ids_for_rollout(self, rollout_id: int) -> torch.Tensor:
        if self.expanded_edge_trace is None or self.expanded_edge_lengths is None:
            return self.expanded_edge_mask()[int(rollout_id)].nonzero(
                as_tuple=False
            ).flatten()
        length = int(self.expanded_edge_lengths[int(rollout_id)].item())
        return self.expanded_edge_trace[int(rollout_id), :length]

    def active_edge_trace_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.root_edge_trace is None
            or self.root_edge_lengths is None
            or self.expanded_edge_trace is None
            or self.expanded_edge_lengths is None
        ):
            return _dense_bool_rows_to_positions(self.active_edges)

        root_rows, root_edges = _trace_rows_to_positions(
            trace=self.root_edge_trace,
            lengths=self.root_edge_lengths,
        )
        expanded_rows, expanded_edges = _trace_rows_to_positions(
            trace=self.expanded_edge_trace,
            lengths=self.expanded_edge_lengths,
        )
        if root_edges.numel() == 0:
            return expanded_rows, expanded_edges
        if expanded_edges.numel() == 0:
            return root_rows, root_edges
        return (
            torch.cat([root_rows, expanded_rows], dim=0),
            torch.cat([root_edges, expanded_edges], dim=0),
        )

    def active_node_trace_rows(self, *, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.anchor_node_trace is None
            or self.anchor_node_lengths is None
            or self.expanded_edge_trace is None
            or self.expanded_edge_lengths is None
        ):
            return _dense_bool_rows_to_positions(self.active_nodes)

        anchor_rows, anchor_nodes = _trace_rows_to_positions(
            trace=self.anchor_node_trace,
            lengths=self.anchor_node_lengths,
        )
        expanded_rows, expanded_edges = _trace_rows_to_positions(
            trace=self.expanded_edge_trace,
            lengths=self.expanded_edge_lengths,
        )
        if expanded_edges.numel() == 0:
            return anchor_rows, anchor_nodes

        edge_index = edge_index.to(device=self.device, dtype=torch.long)
        src = edge_index[0].index_select(0, expanded_edges)
        dst = edge_index[1].index_select(0, expanded_edges)
        node_rows = torch.cat([anchor_rows, expanded_rows, expanded_rows], dim=0)
        node_ids = torch.cat([anchor_nodes, src, dst], dim=0)
        if node_ids.numel() == 0:
            return node_rows, node_ids

        unique_key = node_rows * int(self.num_nodes) + node_ids
        unique_key = torch.unique(unique_key, sorted=True)
        return unique_key.div(self.num_nodes, rounding_mode="floor"), unique_key.remainder(
            self.num_nodes
        )

    def boundary_node_trace_rows(self, *, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _dense_bool_rows_to_positions(
            self.materialize_boundary_nodes(edge_index=edge_index)
        )

    def anchor_node_ids_for_rollout(self, rollout_id: int) -> torch.Tensor:
        return self._trace_values_for_row(
            trace=self.anchor_node_trace,
            lengths=self.anchor_node_lengths,
            row=int(rollout_id),
            fallback_mask=self._anchor_nodes,
        )

    def root_edge_ids_for_rollout(self, rollout_id: int) -> torch.Tensor:
        return self._trace_values_for_row(
            trace=self.root_edge_trace,
            lengths=self.root_edge_lengths,
            row=int(rollout_id),
            fallback_mask=self._root_edges,
        )

    def materialize_anchor_nodes(self) -> torch.Tensor:
        return _trace_to_dense_bool(
            trace=self.anchor_node_trace,
            lengths=self.anchor_node_lengths,
            rows=self.num_rollouts,
            cols=self.num_nodes,
            fallback=self._anchor_nodes,
            device=self.device,
        )

    def materialize_anchor_nodes_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _trace_to_dense_bool_for_rows(
            trace=self.anchor_node_trace,
            lengths=self.anchor_node_lengths,
            row_ids=rollout_ids,
            cols=self.num_nodes,
            fallback=self._anchor_nodes,
            device=self.device,
        )

    def materialize_root_edges(self) -> torch.Tensor:
        return _trace_to_dense_bool(
            trace=self.root_edge_trace,
            lengths=self.root_edge_lengths,
            rows=self.num_rollouts,
            cols=self.num_edges,
            fallback=self._root_edges,
            device=self.device,
        )

    def materialize_root_edges_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _trace_to_dense_bool_for_rows(
            trace=self.root_edge_trace,
            lengths=self.root_edge_lengths,
            row_ids=rollout_ids,
            cols=self.num_edges,
            fallback=self._root_edges,
            device=self.device,
        )

    def materialize_expanded_edges(self) -> torch.Tensor:
        return _trace_to_dense_bool(
            trace=self.expanded_edge_trace,
            lengths=self.expanded_edge_lengths,
            rows=self.num_rollouts,
            cols=self.num_edges,
            fallback=(
                None
                if self._active_edges is None
                else (
                    self._active_edges
                    if self._root_edges is None
                    else self._active_edges & ~self._root_edges
                )
            ),
            device=self.device,
        )

    def materialize_expanded_edges_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
    ) -> torch.Tensor:
        expanded_fallback = None
        if self._active_edges is not None:
            expanded_fallback = (
                self._active_edges
                if self._root_edges is None
                else self._active_edges & ~self._root_edges
            )
        return _trace_to_dense_bool_for_rows(
            trace=self.expanded_edge_trace,
            lengths=self.expanded_edge_lengths,
            row_ids=rollout_ids,
            cols=self.num_edges,
            fallback=expanded_fallback,
            device=self.device,
        )

    def materialize_active_edges(self) -> torch.Tensor:
        if self._active_edges is not None:
            return self._active_edges
        return self.materialize_root_edges() | self.materialize_expanded_edges()

    def materialize_active_edges_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._active_edges is not None:
            rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
            return self._active_edges.index_select(0, rollout_ids)
        return self.materialize_root_edges_for_rollouts(
            rollout_ids,
        ) | self.materialize_expanded_edges_for_rollouts(rollout_ids)

    def anchor_node_trace_for_rollouts_tensor(
        self,
        rollout_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._trace_matrix_for_rollouts(
            trace=self.anchor_node_trace,
            lengths=self.anchor_node_lengths,
            rollout_ids=rollout_ids,
            fallback_mask=self._anchor_nodes,
            cols=self.num_nodes,
        )

    def root_edge_trace_for_rollouts_tensor(
        self,
        rollout_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._trace_matrix_for_rollouts(
            trace=self.root_edge_trace,
            lengths=self.root_edge_lengths,
            rollout_ids=rollout_ids,
            fallback_mask=self._root_edges,
            cols=self.num_edges,
        )

    def expanded_edge_trace_for_rollouts_tensor(
        self,
        rollout_ids: torch.Tensor,
        selected_edge_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        if (
            selected_edge_ids is None
            and self.expanded_edge_trace is not None
            and self.expanded_edge_lengths is not None
        ):
            return (
                self.expanded_edge_trace.index_select(0, rollout_ids),
                self.expanded_edge_lengths.index_select(0, rollout_ids),
            )

        trace, lengths = self._trace_matrix_for_rollouts(
            trace=self.expanded_edge_trace,
            lengths=self.expanded_edge_lengths,
            rollout_ids=rollout_ids,
            fallback_mask=(
                None
                if self._active_edges is None
                else (
                    self._active_edges
                    if self._root_edges is None
                    else self._active_edges & ~self._root_edges
                )
            ),
            cols=self.num_edges,
        )
        if selected_edge_ids is None:
            return trace, lengths

        selected_edge_ids = selected_edge_ids.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)
        if selected_edge_ids.shape != lengths.shape:
            raise ValueError(
                "selected_edge_ids must match rollout_ids shape: "
                f"{tuple(selected_edge_ids.shape)} != {tuple(lengths.shape)}."
            )
        if selected_edge_ids.numel() == 0:
            return trace, lengths

        if trace.size(1) == 0:
            contains = torch.zeros_like(lengths, dtype=torch.bool)
        else:
            valid = torch.arange(
                trace.size(1),
                dtype=torch.long,
                device=self.device,
            ).view(1, -1) < lengths.view(-1, 1)
            contains = (valid & trace.eq(selected_edge_ids.view(-1, 1))).any(dim=1)
        missing = ~contains
        if not bool(missing.any()):
            return trace, lengths

        expanded = torch.full(
            (trace.size(0), trace.size(1) + 1),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        if trace.numel() > 0:
            expanded[:, : trace.size(1)] = trace
        rows = missing.nonzero(as_tuple=False).flatten()
        expanded[rows, lengths.index_select(0, rows)] = selected_edge_ids.index_select(
            0,
            rows,
        )
        updated_lengths = lengths + missing.to(dtype=torch.long)
        return expanded, updated_lengths

    def active_edge_trace_for_rollouts_tensor(
        self,
        rollout_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_trace, root_lengths = self.root_edge_trace_for_rollouts_tensor(
            rollout_ids,
        )
        expanded_trace, expanded_lengths = self.expanded_edge_trace_for_rollouts_tensor(
            rollout_ids,
        )
        width = int(root_trace.size(1) + expanded_trace.size(1))
        trace = torch.full(
            (root_trace.size(0), width),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        if root_trace.size(1) > 0:
            trace[:, : root_trace.size(1)] = root_trace
        if expanded_trace.size(1) > 0:
            expanded_cols = torch.arange(
                expanded_trace.size(1),
                dtype=torch.long,
                device=self.device,
            ).view(1, -1)
            target_cols = root_lengths.view(-1, 1) + expanded_cols
            trace.scatter_(1, target_cols, expanded_trace)
        return trace, root_lengths + expanded_lengths

    def materialize_active_nodes(
        self,
        *,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._active_nodes is not None:
            return self._active_nodes
        edge_index = self._resolve_edge_index(edge_index)
        rows, node_ids = self.active_node_trace_rows(edge_index=edge_index)
        dense = torch.zeros(
            (self.num_rollouts, self.num_nodes),
            dtype=torch.bool,
            device=self.device,
        )
        if node_ids.numel() > 0:
            dense[rows, node_ids] = True
        return dense

    def materialize_active_nodes_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
        *,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        if self._active_nodes is not None:
            return self._active_nodes.index_select(0, rollout_ids)

        dense = self.materialize_anchor_nodes_for_rollouts(rollout_ids)
        if rollout_ids.numel() == 0:
            return dense
        if (
            self.expanded_edge_trace is None
            or self.expanded_edge_lengths is None
            or self.expanded_edge_trace.size(1) == 0
        ):
            return dense

        edge_index = self._resolve_edge_index(edge_index)
        selected_edges = self.expanded_edge_trace.index_select(0, rollout_ids)
        selected_lengths = self.expanded_edge_lengths.index_select(0, rollout_ids)
        valid = torch.arange(
            selected_edges.size(1),
            dtype=torch.long,
            device=self.device,
        ).view(1, -1) < selected_lengths.view(-1, 1)
        local_rows, cols = valid.nonzero(as_tuple=True)
        if local_rows.numel() == 0:
            return dense

        edge_ids = selected_edges[local_rows, cols]
        src = edge_index[0].index_select(0, edge_ids)
        dst = edge_index[1].index_select(0, edge_ids)
        dense[local_rows, src] = True
        dense[local_rows, dst] = True
        return dense

    def materialize_boundary_nodes(
        self,
        *,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._boundary_nodes is not None:
            return self._boundary_nodes

        dense = self.materialize_anchor_nodes()
        if (
            self.expanded_edge_trace is None
            or self.expanded_edge_lengths is None
            or self.expanded_edge_trace.size(1) == 0
        ):
            return dense

        edge_index = self._resolve_edge_index(edge_index)
        valid = torch.arange(
            self.expanded_edge_trace.size(1),
            dtype=torch.long,
            device=self.device,
        ).view(1, -1) < self.expanded_edge_lengths.view(-1, 1)
        rows, cols = valid.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return dense

        edge_ids = self.expanded_edge_trace[rows, cols]
        src = edge_index[0].index_select(0, edge_ids)
        dst = edge_index[1].index_select(0, edge_ids)
        dense[rows, dst] = True
        dense[rows, src] = False
        return dense

    def trace_active_nonroot_edges_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
        selected_edge_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        if selected_edge_ids is not None:
            selected_edge_ids = selected_edge_ids.to(
                device=self.device,
                dtype=torch.long,
            ).view(-1)
            if selected_edge_ids.shape != rollout_ids.shape:
                raise ValueError(
                    "selected_edge_ids must match rollout_ids shape: "
                    f"{tuple(selected_edge_ids.shape)} != {tuple(rollout_ids.shape)}."
                )
        values_by_row: list[torch.Tensor] = []
        selected: list[torch.Tensor | None] = (
            [None for _ in range(int(rollout_ids.numel()))]
            if selected_edge_ids is None
            else [edge for edge in selected_edge_ids.unbind(dim=0)]
        )
        for row, maybe_edge in zip(rollout_ids.detach().cpu().tolist(), selected):
            values = self._trace_values_for_row(
                trace=self.expanded_edge_trace,
                lengths=self.expanded_edge_lengths,
                row=int(row),
                fallback_mask=None,
            )
            if values.numel() == 0 and self.expanded_edge_trace is None:
                expanded = self.materialize_expanded_edges_for_rollouts(
                    rollout_ids.new_tensor([int(row)]),
                )[0]
                values = expanded.nonzero(as_tuple=False).flatten()
            if maybe_edge is not None and not bool(values.eq(maybe_edge).any()):
                values = torch.cat([values, maybe_edge.view(1)], dim=0)
            values_by_row.append(values)
        return values_by_row

    def trace_anchor_nodes_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        return [
            self._trace_values_for_row(
                trace=self.anchor_node_trace,
                lengths=self.anchor_node_lengths,
                row=int(row),
                fallback_mask=self._anchor_nodes,
            )
            for row in rollout_ids.detach().cpu().tolist()
        ]

    def trace_active_edges_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        values_by_row: list[torch.Tensor] = []
        for row in rollout_ids.detach().cpu().tolist():
            root = self._trace_values_for_row(
                trace=self.root_edge_trace,
                lengths=self.root_edge_lengths,
                row=int(row),
                fallback_mask=self._root_edges,
            )
            expanded = self._trace_values_for_row(
                trace=self.expanded_edge_trace,
                lengths=self.expanded_edge_lengths,
                row=int(row),
                fallback_mask=None,
            )
            if expanded.numel() == 0 and self.expanded_edge_trace is None:
                expanded = self.materialize_expanded_edges_for_rollouts(
                    rollout_ids.new_tensor([int(row)]),
                )[0].nonzero(as_tuple=False).flatten()
            if root.numel() == 0:
                values_by_row.append(expanded)
            elif expanded.numel() == 0:
                values_by_row.append(root)
            else:
                values_by_row.append(torch.unique(torch.cat([root, expanded], dim=0)))
        return values_by_row

    def trace_root_edges_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        return [
            self._trace_values_for_row(
                trace=self.root_edge_trace,
                lengths=self.root_edge_lengths,
                row=int(row),
                fallback_mask=self._root_edges,
            )
            for row in rollout_ids.detach().cpu().tolist()
        ]

    def trace_active_nodes_for_rollouts(
        self,
        rollout_ids: torch.Tensor,
        *,
        edge_index: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        edge_index = self._resolve_edge_index(edge_index)
        values_by_row: list[torch.Tensor] = []
        for row in rollout_ids.detach().cpu().tolist():
            anchors = self._trace_values_for_row(
                trace=self.anchor_node_trace,
                lengths=self.anchor_node_lengths,
                row=int(row),
                fallback_mask=self._anchor_nodes,
            )
            expanded_edges = self._trace_values_for_row(
                trace=self.expanded_edge_trace,
                lengths=self.expanded_edge_lengths,
                row=int(row),
                fallback_mask=None,
            )
            if expanded_edges.numel() == 0:
                values_by_row.append(anchors)
                continue
            src = edge_index[0].index_select(0, expanded_edges)
            dst = edge_index[1].index_select(0, expanded_edges)
            values_by_row.append(torch.unique(torch.cat([anchors, src, dst], dim=0)))
        return values_by_row

    def contains_active_edges(
        self,
        *,
        rollout_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)
        if rollout_ids.shape != edge_ids.shape:
            raise ValueError(
                "rollout_ids and edge_ids must have matching shape: "
                f"{tuple(rollout_ids.shape)} != {tuple(edge_ids.shape)}."
            )
        if self._active_edges is not None:
            return self._active_edges[rollout_ids, edge_ids]
        return self._trace_contains(
            trace=self.root_edge_trace,
            lengths=self.root_edge_lengths,
            rows=rollout_ids,
            values=edge_ids,
        ) | self._trace_contains(
            trace=self.expanded_edge_trace,
            lengths=self.expanded_edge_lengths,
            rows=rollout_ids,
            values=edge_ids,
        )

    def contains_active_nodes(
        self,
        *,
        rollout_ids: torch.Tensor,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        node_ids = node_ids.to(device=self.device, dtype=torch.long).view(-1)
        if rollout_ids.shape != node_ids.shape:
            raise ValueError(
                "rollout_ids and node_ids must have matching shape: "
                f"{tuple(rollout_ids.shape)} != {tuple(node_ids.shape)}."
            )
        if self._active_nodes is not None:
            return self._active_nodes[rollout_ids, node_ids]

        anchor_hit = self._trace_contains(
            trace=self.anchor_node_trace,
            lengths=self.anchor_node_lengths,
            rows=rollout_ids,
            values=node_ids,
        )
        if rollout_ids.numel() == 0 or self.expanded_edge_trace is None:
            return anchor_hit
        if self.expanded_edge_lengths is None or self.expanded_edge_trace.size(1) == 0:
            return anchor_hit

        edge_index = self._resolve_edge_index(edge_index)
        lengths = self.expanded_edge_lengths.index_select(0, rollout_ids)
        edge_rows = self.expanded_edge_trace.index_select(0, rollout_ids)
        valid = torch.arange(
            edge_rows.size(1),
            dtype=torch.long,
            device=self.device,
        ).view(1, -1) < lengths.view(-1, 1)
        safe_edges = edge_rows.clamp_min(0).view(-1)
        src = edge_index[0].index_select(0, safe_edges).view_as(edge_rows)
        dst = edge_index[1].index_select(0, safe_edges).view_as(edge_rows)
        endpoint_hit = valid & (
            src.eq(node_ids.view(-1, 1)) | dst.eq(node_ids.view(-1, 1))
        )
        return anchor_hit | endpoint_hit.any(dim=1)

    def root_edge_mask_for_pairs(
        self,
        *,
        rollout_ids: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)
        if self._root_edges is not None:
            return self._root_edges[rollout_ids, edge_ids]
        return self._trace_contains(
            trace=self.root_edge_trace,
            lengths=self.root_edge_lengths,
            rows=rollout_ids,
            values=edge_ids,
        )

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
        if self.expanded_edge_lengths is not None:
            return self.expanded_edge_lengths.to(dtype=torch.long)
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
        if self.expanded_edge_lengths is not None:
            used = self.expanded_edge_lengths.to(dtype=torch.float32)
        else:
            used = self.expanded_edge_mask().sum(dim=1).to(dtype=torch.float32)
        return (used / float(self.expand_budget)).clamp(0.0, 1.0)

    def apply_expansion(
        self,
        *,
        rollout_ids: torch.Tensor,
        chosen_edges: torch.Tensor,
        edge_index: torch.Tensor,
        validate: bool = True,
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
        if validate and (
            bool((rollout_ids < 0).any())
            or bool((rollout_ids >= self.num_rollouts).any())
        ):
            raise ValueError(
                "rollout_ids must contain dynamic rollout ids in current state."
            )
        if validate and (
            bool((chosen_edges < 0).any())
            or bool((chosen_edges >= self.num_edges).any())
        ):
            raise ValueError(
                "chosen_edges must contain original edge ids in current batch."
            )

        edge_index = edge_index.to(device=device, dtype=torch.long)
        src = edge_index[0].index_select(0, chosen_edges)
        dst = edge_index[1].index_select(0, chosen_edges)

        if self._active_edges is not None:
            self._active_edges[rollout_ids, chosen_edges] = True
        if self._active_nodes is not None:
            self._active_nodes[rollout_ids, src] = True
            self._active_nodes[rollout_ids, dst] = True
        if self._boundary_nodes is not None:
            self._boundary_nodes[rollout_ids, src] = False
            self._boundary_nodes[rollout_ids, dst] = True

        self._record_expanded_edges(
            rollout_ids=rollout_ids,
            chosen_edges=chosen_edges,
            validate=validate,
        )
        self.assert_budget_invariant()

    def assert_budget_invariant(self) -> None:
        used = self.expanded_edge_count_per_graph(
            edge_batch=torch.empty(0, dtype=torch.long, device=self.device),
            num_graphs=self.num_rollouts,
        )
        remaining = self.remaining_budget_per_graph(
            edge_batch=torch.empty(0, dtype=torch.long, device=self.device),
            num_graphs=self.num_rollouts,
        )
        expected = (int(self.expand_budget) - used).clamp_min(0)
        if not bool(torch.equal(remaining, expected)):
            raise AssertionError("Budget invariant failed: b_z != B - |E_z \\ E_0|.")

    def _record_expanded_edges(
        self,
        *,
        rollout_ids: torch.Tensor,
        chosen_edges: torch.Tensor,
        validate: bool = True,
    ) -> None:
        if self.expanded_edge_trace is None or self.expanded_edge_lengths is None:
            return
        if self.expanded_edge_trace.size(1) == 0:
            return

        if validate:
            already_expanded = self._trace_contains(
                trace=self.expanded_edge_trace,
                lengths=self.expanded_edge_lengths,
                rows=rollout_ids,
                values=chosen_edges,
            )
            nonroot = ~self.root_edge_mask_for_pairs(
                rollout_ids=rollout_ids,
                edge_ids=chosen_edges,
            ) & ~already_expanded
            if not bool(nonroot.any()):
                return
            rows = rollout_ids[nonroot]
            edges = chosen_edges[nonroot]
        else:
            rows = rollout_ids
            edges = chosen_edges

        current_lengths = self.expanded_edge_lengths.index_select(0, rows)
        if validate and bool(current_lengths.ge(self.expanded_edge_trace.size(1)).any()):
            raise RuntimeError(
                "expanded edge trace exceeded expand_budget; check can_expand masks."
            )

        self.expanded_edge_trace[rows, current_lengths] = edges
        self.expanded_edge_lengths[rows] = current_lengths + 1

    def _trace_contains(
        self,
        *,
        trace: torch.Tensor | None,
        lengths: torch.Tensor | None,
        rows: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        values = values.to(device=self.device, dtype=torch.long).view(-1)
        if rows.shape != values.shape:
            raise ValueError(
                "rows and values must have matching shape: "
                f"{tuple(rows.shape)} != {tuple(values.shape)}."
            )
        if rows.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        if trace is None or lengths is None or trace.size(1) == 0:
            return torch.zeros(rows.shape, dtype=torch.bool, device=self.device)
        selected = trace.index_select(0, rows)
        selected_lengths = lengths.index_select(0, rows)
        valid = torch.arange(
            selected.size(1),
            dtype=torch.long,
            device=self.device,
        ).view(1, -1) < selected_lengths.view(-1, 1)
        return (valid & selected.eq(values.view(-1, 1))).any(dim=1)

    def _trace_matrix_for_rollouts(
        self,
        *,
        trace: torch.Tensor | None,
        lengths: torch.Tensor | None,
        rollout_ids: torch.Tensor,
        fallback_mask: torch.Tensor | None,
        cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_ids = rollout_ids.to(device=self.device, dtype=torch.long).view(-1)
        if trace is not None and lengths is not None:
            return (
                trace.to(device=self.device, dtype=torch.long).index_select(
                    0,
                    rollout_ids,
                ),
                lengths.to(device=self.device, dtype=torch.long).index_select(
                    0,
                    rollout_ids,
                ),
            )
        dense = _trace_to_dense_bool_for_rows(
            trace=None,
            lengths=None,
            row_ids=rollout_ids,
            cols=int(cols),
            fallback=fallback_mask,
            device=self.device,
        )
        return _dense_bool_rows_to_trace(dense, fill_value=-1)

    def _trace_values_for_row(
        self,
        *,
        trace: torch.Tensor | None,
        lengths: torch.Tensor | None,
        row: int,
        fallback_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        row = int(row)
        if row < 0 or row >= self.num_rollouts:
            raise IndexError(f"rollout_id must be in [0, {self.num_rollouts}), got {row}.")
        if trace is not None and lengths is not None:
            length = int(lengths[row].item())
            return trace[row, :length]
        if fallback_mask is not None:
            return fallback_mask[row].nonzero(as_tuple=False).flatten()
        return torch.empty(0, dtype=torch.long, device=self.device)

    def _resolve_edge_index(self, edge_index: torch.Tensor | None) -> torch.Tensor:
        resolved = edge_index if edge_index is not None else self.edge_index
        if resolved is None:
            raise RuntimeError(
                "RolloutState does not carry edge_index; pass edge_index explicitly."
            )
        return resolved.to(device=self.device, dtype=torch.long)

    @staticmethod
    def _validate_trace_pair(
        *,
        trace: torch.Tensor | None,
        lengths: torch.Tensor | None,
        name: str,
    ) -> None:
        if (trace is None) != (lengths is None):
            raise ValueError(f"{name}_trace and {name}_lengths must be provided together.")
        if trace is None:
            return
        if trace.ndim != 2:
            raise ValueError(f"{name}_trace must be 2D, got {tuple(trace.shape)}.")
        if lengths.ndim != 1 or lengths.numel() != trace.size(0):
            raise ValueError(
                f"{name}_lengths must have shape [{trace.size(0)}], "
                f"got {tuple(lengths.shape)}."
            )

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


def _index_optional_matrix(
    tensor: torch.Tensor | None,
    row_ids: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.index_select(0, row_ids.to(device=tensor.device, dtype=torch.long))


def _index_optional_vector(
    tensor: torch.Tensor | None,
    row_ids: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.index_select(0, row_ids.to(device=tensor.device, dtype=torch.long))


def _clone_optional_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.clone()


def _dense_bool_rows_to_trace(
    mask: torch.Tensor,
    *,
    fill_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, values = _dense_bool_rows_to_positions(mask)
    num_rows = int(mask.size(0))
    lengths = torch.bincount(rows, minlength=num_rows).to(dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    trace = torch.full(
        (num_rows, max_len),
        int(fill_value),
        dtype=torch.long,
        device=mask.device,
    )
    if values.numel() == 0:
        return trace, lengths

    offsets = _positions_in_sorted_rows(rows, num_rows=num_rows)
    trace[rows, offsets] = values
    return trace, lengths


def _dense_bool_rows_to_positions(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [R, C], got {tuple(mask.shape)}.")
    return mask.nonzero(as_tuple=True)


def _trace_rows_to_positions(
    *,
    trace: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if trace.ndim != 2:
        raise ValueError(f"trace must have shape [R, K], got {tuple(trace.shape)}.")
    lengths = lengths.to(device=trace.device, dtype=torch.long).view(-1)
    num_rows = int(trace.size(0))
    if lengths.shape != (num_rows,):
        raise ValueError(
            f"lengths must have shape [{num_rows}], got {tuple(lengths.shape)}."
        )
    if trace.size(1) == 0 or lengths.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=trace.device)
        return empty, empty

    valid = torch.arange(trace.size(1), dtype=torch.long, device=trace.device).view(
        1,
        -1,
    ) < lengths.view(-1, 1)
    rows, cols = valid.nonzero(as_tuple=True)
    return rows, trace[rows, cols]


def _positions_in_sorted_rows(rows: torch.Tensor, *, num_rows: int) -> torch.Tensor:
    if rows.numel() == 0:
        return rows.new_empty((0,))
    counts = torch.bincount(rows, minlength=int(num_rows))
    starts = torch.cumsum(counts, dim=0) - counts
    return torch.arange(rows.numel(), dtype=torch.long, device=rows.device) - starts.index_select(
        0,
        rows,
    )


def _items_by_rollout_trace(
    *,
    item_ids: torch.Tensor,
    item_graph_ids: torch.Tensor,
    rollout_to_graph: torch.Tensor,
    fill_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    item_ids = item_ids.to(device=rollout_to_graph.device, dtype=torch.long).view(-1)
    item_graph_ids = item_graph_ids.to(
        device=rollout_to_graph.device,
        dtype=torch.long,
    ).view(-1)
    if item_ids.shape != item_graph_ids.shape:
        raise ValueError(
            "item_ids and item_graph_ids must have matching shape: "
            f"{tuple(item_ids.shape)} != {tuple(item_graph_ids.shape)}."
        )

    num_rows = int(rollout_to_graph.numel())
    if item_ids.numel() == 0:
        return (
            torch.full(
                (num_rows, 0),
                int(fill_value),
                dtype=torch.long,
                device=rollout_to_graph.device,
            ),
            torch.zeros(num_rows, dtype=torch.long, device=rollout_to_graph.device),
        )

    max_graph = int(
        torch.cat([rollout_to_graph, item_graph_ids]).max().item()
    ) + 1
    counts_by_graph = torch.bincount(
        item_graph_ids,
        minlength=max_graph,
    ).to(dtype=torch.long)
    lengths = counts_by_graph.index_select(0, rollout_to_graph)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    trace = torch.full(
        (num_rows, max_len),
        int(fill_value),
        dtype=torch.long,
        device=rollout_to_graph.device,
    )
    if max_len == 0:
        return trace, lengths

    order = torch.argsort(item_graph_ids)
    sorted_items = item_ids.index_select(0, order)
    sorted_graph_ids = item_graph_ids.index_select(0, order)
    sorted_counts = torch.bincount(sorted_graph_ids, minlength=max_graph)
    starts = torch.cumsum(sorted_counts, dim=0) - sorted_counts
    for row, graph_id in enumerate(rollout_to_graph.detach().cpu().tolist()):
        length = int(lengths[row].item())
        if length == 0:
            continue
        start = int(starts[int(graph_id)].item())
        trace[row, :length] = sorted_items[start : start + length]
    return trace, lengths


def _trace_to_dense_bool(
    *,
    trace: torch.Tensor | None,
    lengths: torch.Tensor | None,
    rows: int,
    cols: int,
    fallback: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if fallback is not None:
        return fallback.to(device=device, dtype=torch.bool)
    dense = torch.zeros(
        (int(rows), int(cols)),
        dtype=torch.bool,
        device=device,
    )
    if trace is None or lengths is None:
        return dense
    row_ids, values = _trace_rows_to_positions(
        trace=trace.to(device=device, dtype=torch.long),
        lengths=lengths.to(device=device, dtype=torch.long),
    )
    if values.numel() > 0:
        dense[row_ids, values] = True
    return dense


def _trace_to_dense_bool_for_rows(
    *,
    trace: torch.Tensor | None,
    lengths: torch.Tensor | None,
    row_ids: torch.Tensor,
    cols: int,
    fallback: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    row_ids = row_ids.to(device=device, dtype=torch.long).view(-1)
    if fallback is not None:
        return fallback.to(device=device, dtype=torch.bool).index_select(0, row_ids)
    dense = torch.zeros(
        (int(row_ids.numel()), int(cols)),
        dtype=torch.bool,
        device=device,
    )
    if row_ids.numel() == 0 or trace is None or lengths is None:
        return dense
    selected = trace.to(device=device, dtype=torch.long).index_select(0, row_ids)
    selected_lengths = lengths.to(device=device, dtype=torch.long).index_select(
        0,
        row_ids,
    )
    if selected.size(1) == 0:
        return dense
    valid = torch.arange(
        selected.size(1),
        dtype=torch.long,
        device=device,
    ).view(1, -1) < selected_lengths.view(-1, 1)
    local_rows, cols_in_trace = valid.nonzero(as_tuple=True)
    if local_rows.numel() > 0:
        dense[local_rows, selected[local_rows, cols_in_trace]] = True
    return dense


def _optional_bool_cache(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device=device, dtype=torch.bool)


def _optional_long_matrix(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    tensor = tensor.to(device=device, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"trace tensors must be 2D, got {tuple(tensor.shape)}.")
    return tensor


def _optional_long_vector(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    rows: int,
    name: str,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    tensor = tensor.to(device=device, dtype=torch.long).view(-1)
    if tensor.shape != (int(rows),):
        raise ValueError(f"{name} must have shape [{int(rows)}], got {tuple(tensor.shape)}.")
    return tensor


def _infer_num_nodes(
    *,
    num_nodes: int | None,
    active_nodes: torch.Tensor | None,
    anchor_nodes: torch.Tensor | None,
    boundary_nodes: torch.Tensor | None,
    edge_index: torch.Tensor | None,
) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    for tensor in (active_nodes, anchor_nodes, boundary_nodes):
        if tensor is not None:
            if tensor.ndim != 2:
                raise ValueError(
                    "dense node state tensors must have shape [R, N], "
                    f"got {tuple(tensor.shape)}."
                )
            return int(tensor.size(1))
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1
    raise ValueError("num_nodes is required when node masks and edge_index are absent.")


def _infer_num_edges(
    *,
    num_edges: int | None,
    active_edges: torch.Tensor | None,
    root_edges: torch.Tensor | None,
    edge_index: torch.Tensor | None,
) -> int:
    if num_edges is not None:
        return int(num_edges)
    for tensor in (active_edges, root_edges):
        if tensor is not None:
            if tensor.ndim != 2:
                raise ValueError(
                    "dense edge state tensors must have shape [R, E], "
                    f"got {tuple(tensor.shape)}."
                )
            return int(tensor.size(1))
    if edge_index is not None:
        return int(edge_index.size(1))
    raise ValueError("num_edges is required when edge masks and edge_index are absent.")


__all__ = ["RolloutState", "State"]
