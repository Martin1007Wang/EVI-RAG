from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import torch
from src.data.schema import RetrievalBatch
from src.weaver.context import FlowContext


@dataclass(frozen=True, slots=True)
class Frontier:
    """
    Expandable action list induced by State.

    row_ids[i] expands edge_ids[i].

    row_ids:
        rollout-state rows, local to this State.

    edge_ids:
        physical edge ids in the batched RetrievalBatch graph.
    """

    row_ids: torch.Tensor  # [F], long
    edge_ids: torch.Tensor  # [F], long
    edge_direction: torch.Tensor  # [F], long: 0=forward, 1=backward, 2=internal

    @property
    def num_actions(self) -> int:
        return int(self.edge_ids.numel())

    @property
    def num_edges(self) -> int:
        return self.num_actions

    @property
    def is_empty(self) -> bool:
        return self.num_actions == 0


@dataclass(frozen=True, slots=True)
class GraphTopology:
    """
    Public static graph topology view shared by rollout utilities.

    This is the minimal topology contract required by derive_node_mask() and
    debug assertions. It intentionally excludes incident-edge indexing.
    """

    edge_index: torch.Tensor  # [2, E], long
    edge_to_graph: torch.Tensor  # [E], long
    node_to_graph: torch.Tensor  # [N], long
    anchor_mask: torch.Tensor  # [N], bool

    @classmethod
    def from_flow_context(
        cls,
        flow_context: FlowContext,
    ) -> GraphTopology:
        return cls(
            edge_index=flow_context.edge_index,
            edge_to_graph=_edge_to_graph(
                edge_index=flow_context.edge_index,
                node_to_graph=flow_context.node_to_graph,
            ),
            node_to_graph=flow_context.node_to_graph,
            anchor_mask=flow_context.anchor_mask,
        )

    @classmethod
    def from_tensors(
        cls,
        *,
        edge_index: torch.Tensor,
        node_to_graph: torch.Tensor,
        anchor_mask: torch.Tensor,
        edge_to_graph: torch.Tensor | None = None,
    ) -> GraphTopology:
        edge_index = edge_index.to(dtype=torch.long)
        node_to_graph = node_to_graph.to(device=edge_index.device, dtype=torch.long)
        anchor_mask = anchor_mask.to(device=edge_index.device, dtype=torch.bool)

        if edge_to_graph is None:
            edge_to_graph = _edge_to_graph(
                edge_index=edge_index,
                node_to_graph=node_to_graph,
            )
        else:
            edge_to_graph = edge_to_graph.to(
                device=edge_index.device,
                dtype=torch.long,
            )

        return cls(
            edge_index=edge_index,
            edge_to_graph=edge_to_graph,
            node_to_graph=node_to_graph,
            anchor_mask=anchor_mask,
        )

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    @property
    def num_nodes(self) -> int:
        return int(self.node_to_graph.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))


@dataclass(frozen=True, slots=True)
class IncidentEdgeIndex(GraphTopology):
    """
    Node-to-incident-edge index over a batched graph.

    Selected triples remain directed. Incidence is used only for frontier
    eligibility from the active anchor-connected evidence component.
    """

    incident_ptr: torch.Tensor  # [N + 1], long
    edge_ids_by_node: torch.Tensor  # [2E], long

    @classmethod
    def from_batch(
        cls,
        batch: RetrievalBatch,
        *,
        device: torch.device | None = None,
    ) -> IncidentEdgeIndex:
        return cls.from_flow_context(
            FlowContext.from_batch(batch, device=device),
        )

    @classmethod
    def from_flow_context(
        cls,
        flow_context: FlowContext,
    ) -> IncidentEdgeIndex:
        return cls.from_graph_topology(
            GraphTopology.from_flow_context(flow_context),
        )

    @classmethod
    def from_graph_topology(
        cls,
        topology: GraphTopology,
    ) -> IncidentEdgeIndex:
        edge_index = topology.edge_index
        num_nodes = topology.num_nodes
        num_edges = topology.num_edges
        src = edge_index[0]
        dst = edge_index[1]

        endpoint_nodes = torch.cat([src, dst], dim=0)
        endpoint_edges = torch.arange(
            num_edges,
            dtype=torch.long,
            device=topology.device,
        ).repeat(2)

        order = torch.argsort(endpoint_nodes)
        sorted_nodes = endpoint_nodes.index_select(0, order)
        sorted_edges = endpoint_edges.index_select(0, order)

        incident_count = torch.bincount(
            sorted_nodes,
            minlength=num_nodes,
        )

        incident_ptr = torch.empty(
            num_nodes + 1,
            dtype=torch.long,
            device=topology.device,
        )
        incident_ptr[0] = 0
        incident_ptr[1:] = torch.cumsum(incident_count, dim=0)

        return cls(
            incident_ptr=incident_ptr,
            edge_ids_by_node=sorted_edges,
            edge_index=topology.edge_index,
            edge_to_graph=topology.edge_to_graph,
            node_to_graph=topology.node_to_graph,
            anchor_mask=topology.anchor_mask,
        )

    @property
    def num_nodes(self) -> int:
        return int(self.incident_ptr.numel()) - 1


_IncidentView = GraphTopology


class FrontierBuilder:
    """
    Builds C(z), the expandable edge set.

    Frontier rule:

        e = (u, r, v) is expandable from row z iff:

            e not in S_z
            |S_z| < B
            graph(e) == graph(z)
            u in V_z or v in V_z

    V_z is read from State.node_mask. In rollout, node_mask is a cache
    maintained by State.apply_edges_(). For backward-kernel/debug code that
    deletes edges, rebuild node_mask from edge_mask before using this builder.
    """

    def __init__(self, edge_index: IncidentEdgeIndex) -> None:
        self.edge_index = edge_index

    @classmethod
    def from_batch(
        cls,
        batch: RetrievalBatch,
        *,
        device: torch.device | None = None,
    ) -> FrontierBuilder:
        return cls(IncidentEdgeIndex.from_batch(batch, device=device))

    @classmethod
    def from_flow_context(
        cls,
        flow_context: FlowContext,
    ) -> FrontierBuilder:
        return cls(IncidentEdgeIndex.from_flow_context(flow_context))

    def build(self, state: State) -> Frontier:
        assert_node_cache_consistent(
            state=state,
            edge_index=self.edge_index,
        )
        rows, nodes = _mask_to_pairs(state.node_mask)

        if rows.numel() == 0:
            return _empty_frontier(state.device)

        keep = state.remaining_budget.index_select(0, rows).gt(0)
        rows = rows[keep]
        nodes = nodes[keep]

        if rows.numel() == 0:
            return _empty_frontier(state.device)

        ptr = self.edge_index.incident_ptr
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
        edge_ids = self.edge_index.edge_ids_by_node.index_select(
            0,
            edge_positions,
        )

        same_graph = self.edge_index.edge_to_graph.index_select(
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
        edge_direction = _edge_direction(
            state=state,
            edge_index=self.edge_index.edge_index,
            row_ids=row_ids,
            edge_ids=edge_ids,
        )

        return Frontier(
            row_ids=row_ids,
            edge_ids=edge_ids,
            edge_direction=edge_direction,
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
            edge_index=self.edge_index,
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
    def initial_from_flow_context(
        cls,
        flow_context: FlowContext,
        *,
        budget: int,
        rollouts_per_graph: int = 1,
    ) -> State:
        device = flow_context.device
        num_rows = int(flow_context.num_graphs) * int(rollouts_per_graph)
        node_mask = torch.zeros(
            (num_rows, int(flow_context.num_nodes)),
            dtype=torch.bool,
            device=device,
        )
        edge_mask = torch.zeros(
            (num_rows, int(flow_context.num_edges)),
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
            int(flow_context.num_graphs),
            dtype=torch.long,
            device=device,
        ).repeat_interleave(int(rollouts_per_graph))

        anchors = flow_context.anchor_mask.nonzero(as_tuple=False).view(-1)
        if anchors.numel() > 0:
            rows, cols = _anchor_rows_and_cols(
                anchors=anchors,
                node_to_graph=flow_context.node_to_graph,
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
        edge_index: IncidentEdgeIndex,
    ) -> None:
        """
        Rebuild node_mask from edge_mask truth.

        Use this after deletion-style operations, backward parent construction,
        or debugging. Rollout expansion should not need this.
        """
        self.node_mask = derive_node_mask(
            state=self,
            edge_index=edge_index,
        )

    def with_rebuilt_node_mask(
        self,
        *,
        edge_index: IncidentEdgeIndex,
    ) -> State:
        return State(
            node_mask=derive_node_mask(
                state=self,
                edge_index=edge_index,
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
    edge_index: IncidentEdgeIndex | GraphTopology | torch.Tensor,
    node_to_graph: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Rebuild active nodes from edge_mask truth.

    Active nodes are exactly:

        anchors(graph(row)) union endpoints(selected_edges(row))

    This matches the canonical state cache:

        X_n(S) = A_n union Vtx_n(S)
    """
    incident = _as_graph_topology(
        edge_index=edge_index,
        node_to_graph=node_to_graph,
        anchor_mask=anchor_mask,
    )
    out = torch.zeros_like(state.node_mask)

    for row in range(state.num_rollouts):
        graph_id = state.row_to_graph[row]
        graph_nodes = incident.node_to_graph.eq(graph_id)
        anchors = (incident.anchor_mask & graph_nodes).nonzero(as_tuple=True)[0]

        if anchors.numel() > 0:
            out[row, anchors] = True

        selected_edges = state.edge_mask[row].nonzero(as_tuple=True)[0]
        if selected_edges.numel() == 0:
            continue

        src = incident.edge_index[0].index_select(0, selected_edges)
        dst = incident.edge_index[1].index_select(0, selected_edges)
        out[row, src] = True
        out[row, dst] = True

    return out


def assert_anchor_connected_state(
    *,
    state: State,
    edge_index: IncidentEdgeIndex | GraphTopology | torch.Tensor,
    edge_to_graph: torch.Tensor | None = None,
    node_to_graph: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> None:
    """
    Debug-only assertion.

    Verify that every selected edge can be generated by the recursive frontier
    expansion rule starting from anchors.
    """
    incident = _as_graph_topology(
        edge_index=edge_index,
        edge_to_graph=edge_to_graph,
        node_to_graph=node_to_graph,
        anchor_mask=anchor_mask,
    )

    for row in range(state.num_rollouts):
        selected_edges = state.edge_mask[row].nonzero(as_tuple=True)[0]
        graph_id = state.row_to_graph[row]

        selected_graph = incident.edge_to_graph.index_select(
            0,
            selected_edges,
        )
        if not bool(selected_graph.eq(graph_id).all()):
            raise AssertionError("Every selected edge must belong to the rollout graph.")
        if not _is_recursively_frontier_reachable(
            selected_edges=selected_edges,
            graph_id=int(graph_id.item()),
            incident=incident,
        ):
            raise AssertionError("Every selected edge must be reachable by recursive frontier expansion from anchors.")


def assert_node_cache_consistent(
    *,
    state: State,
    edge_index: IncidentEdgeIndex,
) -> None:
    """
    Debug-only assertion.
    """
    derived = derive_node_mask(
        state=state,
        edge_index=edge_index,
    )

    if not bool(torch.equal(state.node_mask, derived)):
        raise AssertionError("node_mask cache differs from edge_mask-derived active nodes.")


def is_frontier_edge(
    *,
    state: State,
    row: int,
    edge_id: int,
    edge_index: IncidentEdgeIndex,
) -> bool:
    """
    Scalar frontier predicate using the same incident rule as FrontierBuilder.

    Caller owns row/edge validity.
    """
    if bool(state.edge_mask[row, edge_id]):
        return False

    if int(state.remaining_budget[row].item()) <= 0:
        return False

    if int(edge_index.edge_to_graph[edge_id].item()) != int(state.row_to_graph[row].item()):
        return False

    src = int(edge_index.edge_index[0, edge_id].item())
    dst = int(edge_index.edge_index[1, edge_id].item())

    return bool(state.node_mask[row, src] or state.node_mask[row, dst])


def _as_graph_topology(
    *,
    edge_index: IncidentEdgeIndex | GraphTopology | torch.Tensor,
    edge_to_graph: torch.Tensor | None = None,
    node_to_graph: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> IncidentEdgeIndex | GraphTopology:
    if isinstance(edge_index, GraphTopology):
        return edge_index

    if node_to_graph is None or anchor_mask is None:
        raise TypeError(
            "node_to_graph and anchor_mask are required when edge_index is a tensor."
        )

    return GraphTopology.from_tensors(
        edge_index=edge_index,
        node_to_graph=node_to_graph,
        anchor_mask=anchor_mask,
        edge_to_graph=edge_to_graph,
    )


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
    incident: IncidentEdgeIndex | GraphTopology,
) -> bool:
    if selected_edges.numel() == 0:
        return True

    graph_nodes = incident.node_to_graph.eq(int(graph_id))
    active = (incident.anchor_mask & graph_nodes).clone()
    remaining = selected_edges.clone()

    while remaining.numel() > 0:
        src = incident.edge_index[0].index_select(0, remaining)
        dst = incident.edge_index[1].index_select(0, remaining)
        expandable = active.index_select(0, src) | active.index_select(0, dst)
        if not bool(expandable.any()):
            return False
        chosen = remaining.index_select(0, expandable.nonzero(as_tuple=False).view(-1))
        chosen_src = incident.edge_index[0].index_select(0, chosen)
        chosen_dst = incident.edge_index[1].index_select(0, chosen)
        active[chosen_src] = True
        active[chosen_dst] = True
        remaining = remaining.index_select(0, (~expandable).nonzero(as_tuple=False).view(-1))

    return True


def _mask_to_pairs(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = mask.nonzero(as_tuple=True)
    return rows, cols


def _edge_to_graph(
    *,
    edge_index: torch.Tensor,
    node_to_graph: torch.Tensor,
) -> torch.Tensor:
    return node_to_graph.index_select(0, edge_index[0])


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
        edge_direction=empty,
    )


def _edge_direction(
    *,
    state: State,
    edge_index: torch.Tensor,
    row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    if edge_ids.numel() == 0:
        return edge_ids.new_empty(0)

    src = edge_index[0].index_select(0, edge_ids)
    dst = edge_index[1].index_select(0, edge_ids)
    src_active = state.node_mask[row_ids, src]
    dst_active = state.node_mask[row_ids, dst]

    forward = torch.zeros_like(edge_ids)
    backward = torch.ones_like(edge_ids)
    internal = torch.full_like(edge_ids, 2)
    return torch.where(src_active & dst_active, internal, torch.where(src_active, forward, backward))


__all__ = [
    "Frontier",
    "GraphTopology",
    "IncidentEdgeIndex",
    "FrontierBuilder",
    "State",
    "_IncidentView",
    "assert_anchor_connected_state",
    "assert_node_cache_consistent",
    "build_row_to_graph",
    "derive_node_mask",
    "derive_remaining_budget",
    "is_frontier_edge",
]
