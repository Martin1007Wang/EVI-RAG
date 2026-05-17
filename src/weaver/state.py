from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import torch
from src.data.schema import RetrievalBatch
from src.weaver.context import GraphContext


@dataclass(frozen=True, slots=True)
class Frontier:
    """
    Expandable edge actions induced by State.

    row_ids[i] expands the directed KG edge edge_ids[i].

    row_ids:
        rollout-state rows, local to this State.

    edge_ids:
        physical edge ids in the batched RetrievalBatch graph.
    """

    row_ids: torch.Tensor  # [F], long
    edge_ids: torch.Tensor  # [F], long

    @property
    def num_actions(self) -> int:
        return int(self.edge_ids.numel())

    @property
    def is_empty(self) -> bool:
        return self.num_actions == 0


class FrontierBuilder:
    """
    Builds C(z), the expandable edge set.

    Frontier rule:

        e = (u, r, v) is expandable from row z iff:

            e not in S_z
            |S_z| < B
            graph(e) == graph(z)
            exactly one of u, v is in V_z

    V_z is read from State.node_mask. In rollout, node_mask is a cache
    maintained by State.apply_edges_(). For backward-kernel/debug code that
    deletes edges, rebuild node_mask from edge_mask before using this builder.
    """

    def __init__(self, graph_context: GraphContext) -> None:
        self.graph_context = graph_context
        self.incident_ptr, self.edge_ids_by_node = _build_incident_edges(graph_context)

    @classmethod
    def from_batch(
        cls,
        batch: RetrievalBatch,
        *,
        device: torch.device | None = None,
    ) -> FrontierBuilder:
        return cls(GraphContext.from_batch(batch, device=device))

    @classmethod
    def from_graph_context(
        cls,
        graph_context: GraphContext,
    ) -> FrontierBuilder:
        return cls(graph_context)

    def build(self, state: State) -> Frontier:
        rows, nodes = _mask_to_pairs(state.node_mask)

        if rows.numel() == 0:
            return _empty_frontier(state.device)

        keep = state.remaining_budget.index_select(0, rows).gt(0)
        rows = rows[keep]
        nodes = nodes[keep]

        if rows.numel() == 0:
            return _empty_frontier(state.device)

        ptr = self.incident_ptr
        starts = ptr.index_select(0, nodes)
        ends = ptr.index_select(0, nodes + 1)
        degrees = ends - starts

        keep = degrees.gt(0)
        rows = rows[keep]
        starts = starts[keep]
        degrees = degrees[keep]

        if rows.numel() == 0:
            return _empty_frontier(state.device)

        frontier_rows = torch.repeat_interleave(rows, degrees)
        edge_positions = torch.repeat_interleave(starts, degrees) + _segment_arange(degrees)
        edge_ids = self.edge_ids_by_node.index_select(
            0,
            edge_positions,
        )

        same_graph = self.graph_context.edge_to_graph.index_select(
            0,
            edge_ids,
        ).eq(state.row_to_graph.index_select(0, frontier_rows))

        frontier_rows = frontier_rows[same_graph]
        edge_ids = edge_ids[same_graph]

        if edge_ids.numel() == 0:
            return _empty_frontier(state.device)

        unselected = ~state.edge_mask[frontier_rows, edge_ids]
        frontier_rows = frontier_rows[unselected]
        edge_ids = edge_ids[unselected]

        if edge_ids.numel() == 0:
            return _empty_frontier(state.device)

        keys = frontier_rows * state.num_edges + edge_ids
        keys = torch.unique(keys, sorted=True)

        row_ids = keys // state.num_edges
        edge_ids = keys % state.num_edges
        src = self.graph_context.edge_index[0].index_select(0, edge_ids)
        dst = self.graph_context.edge_index[1].index_select(0, edge_ids)
        src_active = state.node_mask[row_ids, src]
        dst_active = state.node_mask[row_ids, dst]
        expands_new_node = src_active ^ dst_active
        row_ids = row_ids[expands_new_node]
        edge_ids = edge_ids[expands_new_node]

        if edge_ids.numel() == 0:
            return _empty_frontier(state.device)

        return Frontier(
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

    def contains(
        self,
        *,
        state: State,
        row: int,
        edge_id: int,
    ) -> bool:
        return is_frontier_edge(
            state=state,
            row=row,
            edge_id=edge_id,
            graph_context=self.graph_context,
        )


@dataclass(slots=True)
class State:
    """
    Dynamic rollout state.

    Mathematical truth:
        edge_mask:         [R, E], bool
        max_budget_by_row: [R], long
        row_to_graph:      [R], long

    Cache:
        node_mask:         [R, N], bool

    remaining_budget is derived, not stored.
    """

    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    max_budget_by_row: torch.Tensor
    row_to_graph: torch.Tensor

    @classmethod
    def initial(
        cls,
        batch: RetrievalBatch,
        *,
        budget: int,
        rollouts_per_graph: int = 1,
        device: torch.device | None = None,
    ) -> State:
        device = batch.edge_index.device if device is None else device

        num_nodes = int(batch.num_nodes_total)
        num_edges = int(batch.num_edges_total)
        num_graphs = int(batch.num_graphs_total)
        num_rows = num_graphs * int(rollouts_per_graph)

        node_mask = torch.zeros(
            (num_rows, num_nodes),
            dtype=torch.bool,
            device=device,
        )
        edge_mask = torch.zeros(
            (num_rows, num_edges),
            dtype=torch.bool,
            device=device,
        )
        max_budget_by_row = torch.full(
            (num_rows,),
            int(budget),
            dtype=torch.long,
            device=device,
        )
        row_to_graph = build_row_to_graph(
            batch,
            rollouts_per_graph=rollouts_per_graph,
            device=device,
        )

        anchors = batch.anchor_node_ids.to(device=device, dtype=torch.long)
        if anchors.numel() > 0:
            node_to_graph = batch.batch.to(device=device, dtype=torch.long)
            rows, cols = _anchor_rows_and_cols(
                anchors=anchors,
                node_to_graph=node_to_graph,
                rollouts_per_graph=rollouts_per_graph,
                num_rows=num_rows,
            )

            node_mask[rows, cols] = True

        return cls(
            node_mask=node_mask,
            edge_mask=edge_mask,
            max_budget_by_row=max_budget_by_row,
            row_to_graph=row_to_graph,
        )

    @classmethod
    def initial_from_graph_context(
        cls,
        graph_context: GraphContext,
        *,
        budget: int,
        rollouts_per_graph: int = 1,
    ) -> State:
        device = graph_context.device
        num_rows = int(graph_context.num_graphs) * int(rollouts_per_graph)
        node_mask = torch.zeros(
            (num_rows, int(graph_context.num_nodes)),
            dtype=torch.bool,
            device=device,
        )
        edge_mask = torch.zeros(
            (num_rows, int(graph_context.num_edges)),
            dtype=torch.bool,
            device=device,
        )
        max_budget_by_row = torch.full(
            (num_rows,),
            int(budget),
            dtype=torch.long,
            device=device,
        )
        row_to_graph = torch.arange(
            int(graph_context.num_graphs),
            dtype=torch.long,
            device=device,
        ).repeat_interleave(int(rollouts_per_graph))

        anchors = graph_context.anchor_mask.nonzero(as_tuple=False).view(-1)
        if anchors.numel() > 0:
            rows, cols = _anchor_rows_and_cols(
                anchors=anchors,
                node_to_graph=graph_context.node_to_graph,
                rollouts_per_graph=rollouts_per_graph,
                num_rows=num_rows,
            )
            node_mask[rows, cols] = True

        return cls(
            node_mask=node_mask,
            edge_mask=edge_mask,
            max_budget_by_row=max_budget_by_row,
            row_to_graph=row_to_graph,
        )

    @classmethod
    def initial_from_graph_ids(
        cls,
        batch: RetrievalBatch,
        *,
        graph_ids: torch.Tensor,
        budget: int,
        device: torch.device | None = None,
    ) -> State:
        device = batch.edge_index.device if device is None else device
        graph_ids = graph_ids.to(device=device, dtype=torch.long).view(-1)

        state = cls.initial(
            batch,
            budget=budget,
            rollouts_per_graph=1,
            device=device,
        )
        return state.select_rows(graph_ids)

    @property
    def device(self) -> torch.device:
        return self.edge_mask.device

    @property
    def num_rollouts(self) -> int:
        return int(self.edge_mask.size(0))

    @property
    def num_nodes(self) -> int:
        return int(self.node_mask.size(1))

    @property
    def num_edges(self) -> int:
        return int(self.edge_mask.size(1))

    @property
    def remaining_budget(self) -> torch.Tensor:
        return derive_remaining_budget(self)

    @property
    def active_node_rows(self) -> torch.Tensor:
        return self.active_node_trace_rows()[0]

    @property
    def active_node_ids(self) -> torch.Tensor:
        return self.active_node_trace_rows()[1]

    @property
    def selected_edge_rows(self) -> torch.Tensor:
        return self.active_edge_trace_rows()[0]

    @property
    def selected_edge_ids(self) -> torch.Tensor:
        return self.active_edge_trace_rows()[1]

    def clone(self) -> State:
        return State(
            node_mask=self.node_mask.clone(),
            edge_mask=self.edge_mask.clone(),
            max_budget_by_row=self.max_budget_by_row.clone(),
            row_to_graph=self.row_to_graph.clone(),
        )

    @classmethod
    def concat(cls, states: Sequence[State]) -> State:
        if not states:
            raise ValueError("Cannot concatenate an empty state sequence.")

        if len(states) == 1:
            return states[0]

        return cls(
            node_mask=torch.cat([state.node_mask for state in states], dim=0),
            edge_mask=torch.cat([state.edge_mask for state in states], dim=0),
            max_budget_by_row=torch.cat(
                [state.max_budget_by_row.view(-1) for state in states],
                dim=0,
            ),
            row_to_graph=torch.cat(
                [state.row_to_graph.view(-1) for state in states],
                dim=0,
            ),
        )

    def select_rows(self, rows: torch.Tensor) -> State:
        return State(
            node_mask=self.node_mask.index_select(0, rows),
            edge_mask=self.edge_mask.index_select(0, rows),
            max_budget_by_row=self.max_budget_by_row.index_select(0, rows),
            row_to_graph=self.row_to_graph.index_select(0, rows),
        )

    def active_node_trace_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _mask_to_pairs(self.node_mask)

    def active_edge_trace_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _mask_to_pairs(self.edge_mask)

    def contains_active_nodes(
        self,
        *,
        row_ids: torch.Tensor,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.node_mask[row_ids, node_ids]

    def apply_edges_(
        self,
        *,
        edge_index: torch.Tensor,
        rows: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> None:
        """
        Hot-path transition.

        Preconditions owned by caller:
            rows:     [K], long, same device
            edge_ids: [K], long, same device
            each row appears at most once
            every pair comes from FrontierBuilder.build(state)
        """
        if edge_ids.numel() == 0:
            return

        src = edge_index[0, edge_ids]
        dst = edge_index[1, edge_ids]

        self.edge_mask[rows, edge_ids] = True
        self.node_mask[rows, src] = True
        self.node_mask[rows, dst] = True

    def rebuild_node_mask_(
        self,
        *,
        graph_context: GraphContext,
    ) -> None:
        """
        Rebuild node_mask from edge_mask truth.

        Use this after deletion-style operations, backward parent construction,
        or debugging. Rollout expansion should not need this.
        """
        self.node_mask = derive_node_mask(
            state=self,
            graph_context=graph_context,
        )

    def with_rebuilt_node_mask(
        self,
        *,
        graph_context: GraphContext,
    ) -> State:
        return State(
            node_mask=derive_node_mask(
                state=self,
                graph_context=graph_context,
            ),
            edge_mask=self.edge_mask,
            max_budget_by_row=self.max_budget_by_row,
            row_to_graph=self.row_to_graph,
        )


def build_row_to_graph(
    batch: RetrievalBatch,
    *,
    rollouts_per_graph: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Row-to-static-graph mapping consistent with State.initial.

        row = graph_id * rollouts_per_graph + rollout_id
    """
    device = batch.edge_index.device if device is None else device

    return torch.arange(
        int(batch.num_graphs_total),
        dtype=torch.long,
        device=device,
    ).repeat_interleave(int(rollouts_per_graph))


def derive_remaining_budget(state: State) -> torch.Tensor:
    selected_count = state.edge_mask.long().sum(dim=1)
    return state.max_budget_by_row - selected_count


def derive_node_mask(
    *,
    state: State,
    graph_context: GraphContext,
) -> torch.Tensor:
    """
    Rebuild active nodes from edge_mask truth.

    Active nodes are exactly:

        anchors(graph(row)) union endpoints(selected_edges(row))

    This matches the canonical state cache:

        X_n(S) = A_n union Vtx_n(S)
    """
    out = torch.zeros_like(state.node_mask)

    for row in range(state.num_rollouts):
        graph_id = state.row_to_graph[row]
        graph_nodes = graph_context.node_to_graph.eq(graph_id)
        anchors = (graph_context.anchor_mask & graph_nodes).nonzero(as_tuple=True)[0]

        if anchors.numel() > 0:
            out[row, anchors] = True

        selected_edges = state.edge_mask[row].nonzero(as_tuple=True)[0]
        if selected_edges.numel() == 0:
            continue

        src = graph_context.edge_index[0].index_select(0, selected_edges)
        dst = graph_context.edge_index[1].index_select(0, selected_edges)
        out[row, src] = True
        out[row, dst] = True

    return out


def assert_anchor_connected_state(
    *,
    state: State,
    graph_context: GraphContext,
) -> None:
    """
    Debug-only assertion.

    Verify that every selected edge can be generated by the recursive frontier
    expansion rule starting from anchors.
    """
    for row in range(state.num_rollouts):
        selected_edges = state.edge_mask[row].nonzero(as_tuple=True)[0]
        graph_id = state.row_to_graph[row]

        selected_graph = graph_context.edge_to_graph.index_select(
            0,
            selected_edges,
        )
        if not bool(selected_graph.eq(graph_id).all()):
            raise AssertionError("Every selected edge must belong to the rollout graph.")
        if not _is_recursively_frontier_reachable(
            selected_edges=selected_edges,
            graph_id=int(graph_id.item()),
            graph_context=graph_context,
        ):
            raise AssertionError("Every selected edge must be reachable by recursive frontier expansion from anchors.")


def is_frontier_edge(
    *,
    state: State,
    row: int,
    edge_id: int,
    graph_context: GraphContext,
) -> bool:
    """
    Scalar frontier predicate using the same incident rule as FrontierBuilder.

    Caller owns row/edge validity.
    """
    if bool(state.edge_mask[row, edge_id]):
        return False

    if int(state.remaining_budget[row].item()) <= 0:
        return False

    if int(graph_context.edge_to_graph[edge_id].item()) != int(state.row_to_graph[row].item()):
        return False

    src = int(graph_context.edge_index[0, edge_id].item())
    dst = int(graph_context.edge_index[1, edge_id].item())

    return bool(state.node_mask[row, src] ^ state.node_mask[row, dst])


def _anchor_component_nodes(
    *,
    anchors: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    active = torch.zeros(
        int(num_nodes),
        dtype=torch.bool,
        device=device,
    )

    if anchors.numel() == 0:
        return active

    active[anchors] = True

    if src.numel() == 0:
        return active

    # Relax the undirected selected-edge component one hop per pass. An
    # E-edge subgraph reaches its fixed point within E propagation rounds.
    active_long = active.to(dtype=torch.long)
    for _ in range(int(src.numel())):
        propagated = (
            active_long.index_select(0, src).gt(0)
            | active_long.index_select(0, dst).gt(0)
        ).to(dtype=torch.long)
        next_active = active_long.clone()
        next_active.index_add_(0, src, propagated)
        next_active.index_add_(0, dst, propagated)
        active_long = next_active.clamp_max_(1)

    return active_long.bool()


def _is_recursively_frontier_reachable(
    *,
    selected_edges: torch.Tensor,
    graph_id: int,
    graph_context: GraphContext,
) -> bool:
    if selected_edges.numel() == 0:
        return True

    graph_nodes = graph_context.node_to_graph.eq(int(graph_id))
    anchor_nodes = set(
        (graph_context.anchor_mask & graph_nodes)
        .nonzero(as_tuple=False)
        .view(-1)
        .detach()
        .cpu()
        .tolist()
    )
    if not anchor_nodes:
        return False

    src_nodes = (
        graph_context.edge_index[0]
        .index_select(0, selected_edges)
        .detach()
        .cpu()
        .tolist()
    )
    dst_nodes = (
        graph_context.edge_index[1]
        .index_select(0, selected_edges)
        .detach()
        .cpu()
        .tolist()
    )

    parent: dict[int, int] = {}

    def find(node: int) -> int:
        node = int(node)
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    component_nodes: set[int] = set()
    for src_node, dst_node in zip(src_nodes, dst_nodes):
        component_nodes.add(int(src_node))
        component_nodes.add(int(dst_node))
        if not union(int(src_node), int(dst_node)):
            return False

    anchors_by_component: dict[int, int] = {}
    for node in component_nodes:
        if node in anchor_nodes:
            root = find(node)
            anchors_by_component[root] = anchors_by_component.get(root, 0) + 1

    for node in component_nodes:
        if anchors_by_component.get(find(node), 0) != 1:
            return False

    return True


def _mask_to_pairs(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = mask.nonzero(as_tuple=True)
    return rows, cols


def _build_incident_edges(
    graph_context: GraphContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = graph_context.edge_index
    src = edge_index[0]
    dst = edge_index[1]

    endpoint_nodes = torch.cat([src, dst], dim=0)
    endpoint_edges = torch.arange(
        int(graph_context.num_edges),
        dtype=torch.long,
        device=graph_context.device,
    ).repeat(2)

    order = torch.argsort(endpoint_nodes)
    sorted_nodes = endpoint_nodes.index_select(0, order)
    sorted_edges = endpoint_edges.index_select(0, order)

    incident_count = torch.bincount(
        sorted_nodes,
        minlength=int(graph_context.num_nodes),
    )

    incident_ptr = torch.empty(
        int(graph_context.num_nodes) + 1,
        dtype=torch.long,
        device=graph_context.device,
    )
    incident_ptr[0] = 0
    incident_ptr[1:] = torch.cumsum(incident_count, dim=0)

    return incident_ptr, sorted_edges


def _anchor_rows_and_cols(
    *,
    anchors: torch.Tensor,
    node_to_graph: torch.Tensor,
    rollouts_per_graph: int,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rollouts_per_graph = int(rollouts_per_graph)
    anchor_graph = node_to_graph.index_select(0, anchors)
    rollout_offsets = torch.arange(
        rollouts_per_graph,
        dtype=torch.long,
        device=anchors.device,
    )
    rows = (
        anchor_graph.repeat_interleave(rollouts_per_graph) * rollouts_per_graph
        + rollout_offsets.repeat(int(anchors.numel()))
    )
    cols = anchors.repeat_interleave(rollouts_per_graph)
    if rows.numel() > 0:
        assert int(rows.max().item()) < int(num_rows), (
            "Anchor-to-row mapping overflows num_rows. "
            "Expected row = graph_id * rollouts_per_graph + rollout_id."
        )
    return rows, cols


def _segment_arange(lengths: torch.Tensor) -> torch.Tensor:
    """
    For lengths [3, 2], return [0, 1, 2, 0, 1].
    """
    if lengths.numel() == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=lengths.device,
        )

    total = int(lengths.sum().item())
    if total == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=lengths.device,
        )

    starts = torch.cumsum(lengths, dim=0) - lengths

    return torch.arange(
        total,
        dtype=torch.long,
        device=lengths.device,
    ) - torch.repeat_interleave(starts, lengths)


def _empty_frontier(device: torch.device) -> Frontier:
    empty = torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )
    return Frontier(
        row_ids=empty,
        edge_ids=empty,
    )


__all__ = [
    "Frontier",
    "FrontierBuilder",
    "State",
    "assert_anchor_connected_state",
    "build_row_to_graph",
    "derive_node_mask",
    "derive_remaining_budget",
    "is_frontier_edge",
]
