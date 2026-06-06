from __future__ import annotations

from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ExpansionBatch:
    state_ids: Tensor  # [K]
    edge_ids: Tensor  # [K]

    @property
    def num_actions(self) -> int:
        return int(self.edge_ids.numel())


@dataclass(frozen=True, slots=True)
class FrontierEncoding:
    row_ids: Tensor  # [F]
    edge_ids: Tensor  # [F]
    graph_ids: Tensor  # [F]

    @property
    def num_actions(self) -> int:
        return int(self.edge_ids.numel())


@dataclass(frozen=True, slots=True)
class EdgeSelection:
    row_ids: Tensor  # [K]
    edge_ids: Tensor  # [K]


@dataclass(frozen=True, slots=True)
class NodeSelection:
    row_ids: Tensor  # [K]
    node_ids: Tensor  # [K]


@dataclass(frozen=True, slots=True)
class StateBatch:
    graph_ids: Tensor  # [S]
    edge_ids: Tensor  # [S, B], sorted ascending with -1 padding
    edge_count: Tensor  # [S]

    @classmethod
    def initial(
        cls,
        *,
        graph_ids: Tensor,
        budget: int,
        graph_context: GraphContext,
    ) -> StateBatch:
        graph_ids = graph_ids.to(device=graph_context.device, dtype=torch.long).view(-1)
        edge_capacity = int(budget)
        if edge_capacity < 0:
            raise ValueError("edge capacity must be nonnegative.")
        return cls(
            graph_ids=graph_ids,
            edge_ids=torch.full(
                (int(graph_ids.numel()), edge_capacity),
                -1,
                dtype=torch.long,
                device=graph_context.device,
            ),
            edge_count=torch.zeros(
                int(graph_ids.numel()),
                dtype=torch.long,
                device=graph_context.device,
            ),
        )

    @classmethod
    def from_selected_edges(
        cls,
        *,
        graph_ids: Tensor,
        edge_ids: Tensor,
        edge_count: Tensor,
        budget: int,
        graph_context: GraphContext,
    ) -> StateBatch:
        graph_ids = graph_ids.to(device=graph_context.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=graph_context.device, dtype=torch.long)
        edge_count = edge_count.to(device=graph_context.device, dtype=torch.long).view(-1)
        edge_capacity = int(edge_ids.size(1)) if edge_ids.ndim == 2 else int(budget)
        if int(budget) != edge_capacity:
            raise ValueError("budget must match edge_ids storage width.")
        if edge_ids.shape != (int(graph_ids.numel()), edge_capacity):
            raise ValueError("edge_ids must have shape [num_states, edge_capacity].")
        if int(edge_count.numel()) != int(graph_ids.numel()):
            raise ValueError("edge_count must have one value per state.")
        if bool(edge_count.lt(0).any()) or bool(edge_count.gt(edge_capacity).any()):
            raise ValueError("edge_count must be in [0, edge_capacity].")

        valid = _prefix_mask(edge_count, edge_capacity)
        selected = edge_ids[valid]
        if bool(selected.lt(0).any()):
            raise ValueError("selected edge ids must be nonnegative.")
        if bool(selected.ge(int(graph_context.num_edges)).any()):
            raise ValueError("selected edge ids must be smaller than graph_context.num_edges.")
        sentinel = max(int(graph_context.num_edges), 1)
        out = torch.sort(torch.where(valid, edge_ids, sentinel), dim=1).values
        out = torch.where(out.eq(sentinel), -1, out)
        duplicate = valid[:, 1:] & out[:, 1:].eq(out[:, :-1])
        if bool(duplicate.any()):
            raise ValueError("selected edge ids must be unique.")
        selected_rows = valid.nonzero(as_tuple=True)[0]
        if selected.numel() and not bool(graph_context.edge_to_graph.index_select(0, selected).eq(graph_ids.index_select(0, selected_rows)).all()):
            raise ValueError("selected edges must belong to the state graph.")

        state = cls(graph_ids=graph_ids, edge_ids=out, edge_count=edge_count)
        if not bool(state.root_reachable_mask(graph_context).all()):
            raise ValueError("selected edge set must be root-reachable.")
        return state

    @property
    def device(self) -> torch.device:
        return self.graph_ids.device

    @property
    def num_states(self) -> int:
        return int(self.graph_ids.numel())

    @property
    def edge_capacity(self) -> int:
        return int(self.edge_ids.size(1))

    def take(self, rows: Tensor) -> StateBatch:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        return StateBatch(
            graph_ids=self.graph_ids.index_select(0, rows),
            edge_ids=self.edge_ids.index_select(0, rows),
            edge_count=self.edge_count.index_select(0, rows),
        )

    def selected_edge_index(self) -> EdgeSelection:
        valid = _prefix_mask(self.edge_count, self.edge_capacity)
        rows = valid.nonzero(as_tuple=True)[0]
        return EdgeSelection(row_ids=rows, edge_ids=self.edge_ids[valid])

    def active_node_index(self, graph: GraphContext) -> NodeSelection:
        return _active_nodes(state=self, graph=graph)

    def covered_node_pairs(self, graph: GraphContext) -> tuple[Tensor, Tensor]:
        active = self.active_node_index(graph)
        return active.row_ids, active.node_ids

    def frontier(
        self,
        *,
        graph_context: GraphContext,
        active: NodeSelection | None = None,
    ) -> FrontierEncoding:
        return frontier_from_graph(
            state=self,
            graph=graph_context,
            active=active,
        )

    def branch(
        self,
        expansion: ExpansionBatch,
        *,
        graph_context: GraphContext,
    ) -> StateBatch:
        rows = expansion.state_ids.to(device=self.device, dtype=torch.long).view(-1)
        return self.take(rows).advance(
            ExpansionBatch(
                state_ids=torch.arange(int(rows.numel()), device=self.device),
                edge_ids=expansion.edge_ids,
            ),
            graph_context=graph_context,
        )

    def advance(
        self,
        expansion: ExpansionBatch,
        *,
        graph_context: GraphContext,
        trusted: bool = False,
    ) -> StateBatch:
        rows = expansion.state_ids.to(device=self.device, dtype=torch.long).view(-1)
        new_edges = expansion.edge_ids.to(device=self.device, dtype=torch.long).view(-1)
        if int(rows.numel()) != int(new_edges.numel()):
            raise ValueError("state_ids and edge_ids must have the same length.")
        if int(torch.unique(rows).numel()) != int(rows.numel()):
            raise ValueError("advance() requires unique state rows.")
        if int(rows.numel()) == 0:
            return self
        if bool(self.edge_count.index_select(0, rows).ge(self.edge_capacity).any()):
            raise ValueError("advance() received a state with no remaining edge storage capacity.")

        if not trusted:
            frontier = frontier_from_graph(state=self.take(rows), graph=graph_context)
            requested = torch.arange(int(rows.numel()), device=self.device) * max(int(graph_context.num_edges), 1) + new_edges
            legal = frontier.row_ids * max(int(graph_context.num_edges), 1) + frontier.edge_ids
            if not bool(torch.isin(requested, legal).all()):
                raise ValueError("advance() received an edge outside the legal frontier.")

        next_ids = self.edge_ids.clone()
        next_count = self.edge_count.clone()
        pos = next_count.index_select(0, rows)
        next_ids[rows, pos] = new_edges
        next_count[rows] += 1
        changed = next_ids.index_select(0, rows)
        sentinel = max(int(graph_context.num_edges), 1)
        changed = torch.sort(torch.where(changed.lt(0), sentinel, changed), dim=1).values
        next_ids[rows] = torch.where(changed.eq(sentinel), -1, changed)
        return StateBatch(
            graph_ids=self.graph_ids,
            edge_ids=next_ids,
            edge_count=next_count,
        )

    def root_reachable_mask(self, graph: GraphContext) -> Tensor:
        return root_reachable_mask_from_edges(
            edge_ids=self.edge_ids,
            edge_count=self.edge_count,
            graph=graph,
        )


def frontier_from_graph(
    *,
    state: StateBatch,
    graph: GraphContext,
    active: NodeSelection | None = None,
) -> FrontierEncoding:
    active = state.active_node_index(graph) if active is None else active
    expandable = state.edge_count.lt(state.edge_capacity)
    if int(active.row_ids.numel()) > 0:
        keep_active = expandable.index_select(0, active.row_ids)
        active = NodeSelection(
            row_ids=active.row_ids[keep_active],
            node_ids=active.node_ids[keep_active],
        )
    if int(active.row_ids.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=state.device)
        return FrontierEncoding(empty, empty, empty)
    counts = graph.adjacency.out_ptr.index_select(0, active.node_ids + 1) - graph.adjacency.out_ptr.index_select(0, active.node_ids)
    total = int(counts.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=state.device)
        return FrontierEncoding(empty, empty, empty)
    rows = torch.repeat_interleave(active.row_ids, counts, output_size=total)
    positions = torch.repeat_interleave(graph.adjacency.out_ptr.index_select(0, active.node_ids), counts, output_size=total) + _segment_arange(counts)
    edges = graph.adjacency.edge_ids_by_src.index_select(0, positions)
    selected = state.edge_ids.index_select(0, rows)
    valid = _prefix_mask(state.edge_count.index_select(0, rows), state.edge_capacity)
    keep = ~(selected.eq(edges.view(-1, 1)) & valid).any(dim=1)
    rows, edges = rows[keep], edges[keep]
    if int(edges.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=state.device)
        return FrontierEncoding(empty, empty, empty)
    keys = rows * max(int(graph.num_edges), 1) + edges
    order = torch.argsort(keys)
    keep = torch.ones(int(order.numel()), dtype=torch.bool, device=state.device)
    sorted_keys = keys.index_select(0, order)
    if int(keep.numel()) > 1:
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    order = order[keep]
    rows = rows.index_select(0, order)
    edges = edges.index_select(0, order)
    graph_ids = state.graph_ids.index_select(0, rows)
    return FrontierEncoding(rows, edges, graph_ids)


def remove_selected_edge(
    *,
    state: StateBatch,
    row: int,
    edge_id: int,
    graph_context: GraphContext,
) -> StateBatch:
    child = state.take(torch.tensor([int(row)], device=state.device))
    count = int(child.edge_count[0].item())
    selected = child.edge_ids[0, :count]
    keep = selected.ne(int(edge_id))
    if int(keep.sum().item()) == count:
        raise ValueError("edge_id must be selected.")
    edge_ids = torch.full_like(child.edge_ids, -1)
    remaining = selected[keep]
    edge_ids[0, : int(remaining.numel())] = remaining
    return StateBatch(
        graph_ids=child.graph_ids,
        edge_ids=edge_ids,
        edge_count=torch.tensor([int(remaining.numel())], dtype=torch.long, device=state.device),
    )


def _active_nodes(*, state: StateBatch, graph: GraphContext) -> NodeSelection:
    anchor_start = graph.anchor_ptr.index_select(0, state.graph_ids)
    anchor_count = graph.anchor_ptr.index_select(0, state.graph_ids + 1) - anchor_start
    anchor_total = int(anchor_count.sum().item())
    rows = torch.repeat_interleave(torch.arange(state.num_states, device=state.device), anchor_count, output_size=anchor_total)
    nodes = graph.anchor_node_ids.index_select(
        0,
        torch.repeat_interleave(anchor_start, anchor_count, output_size=anchor_total) + _segment_arange(anchor_count),
    )
    selected = state.selected_edge_index()
    if int(selected.edge_ids.numel()) > 0:
        rows = torch.cat([rows, selected.row_ids, selected.row_ids])
        nodes = torch.cat([nodes, graph.edge_src.index_select(0, selected.edge_ids), graph.edge_dst.index_select(0, selected.edge_ids)])
    keys = rows * max(int(graph.num_nodes), 1) + nodes
    order = torch.argsort(keys)
    sorted_keys = keys.index_select(0, order)
    keep = torch.ones(int(order.numel()), dtype=torch.bool, device=state.device)
    if int(keep.numel()) > 1:
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    order = order[keep]
    return NodeSelection(row_ids=rows.index_select(0, order), node_ids=nodes.index_select(0, order))


def root_reachable_mask_from_edges(*, edge_ids: Tensor, edge_count: Tensor, graph: GraphContext) -> Tensor:
    valid = _prefix_mask(edge_count, int(edge_ids.size(1)))
    if not bool(valid.any()):
        return torch.ones(int(edge_ids.size(0)), dtype=torch.bool, device=edge_ids.device)

    rows = valid.nonzero(as_tuple=True)[0]
    selected_edges = edge_ids[valid]
    src = graph.edge_src.index_select(0, selected_edges)
    dst = graph.edge_dst.index_select(0, selected_edges)
    node_span = max(int(graph.num_nodes), 1)
    row_src_keys = rows * node_span + src
    row_dst_keys = rows * node_span + dst

    order = torch.argsort(row_src_keys)
    sorted_src_keys = row_src_keys.index_select(0, order)
    reachable_edges = torch.zeros(int(selected_edges.numel()), dtype=torch.bool, device=edge_ids.device)
    processed_nodes = torch.empty(0, dtype=torch.long, device=edge_ids.device)
    frontier_nodes = torch.unique(row_src_keys[graph.anchor_mask.index_select(0, src)])

    while int(frontier_nodes.numel()) > 0:
        if int(processed_nodes.numel()) > 0:
            positions = torch.searchsorted(processed_nodes, frontier_nodes)
            in_range = positions.lt(int(processed_nodes.numel()))
            already_seen = torch.zeros_like(in_range)
            already_seen[in_range] = processed_nodes.index_select(0, positions[in_range]).eq(frontier_nodes[in_range])
            frontier_nodes = frontier_nodes[~already_seen]
            if int(frontier_nodes.numel()) == 0:
                break
        processed_nodes = torch.unique(torch.cat([processed_nodes, frontier_nodes]))

        starts = torch.searchsorted(sorted_src_keys, frontier_nodes, right=False)
        ends = torch.searchsorted(sorted_src_keys, frontier_nodes, right=True)
        counts = ends - starts
        has_edges = counts.gt(0)
        if not bool(has_edges.any()):
            frontier_nodes = torch.empty(0, dtype=torch.long, device=edge_ids.device)
            continue
        starts = starts[has_edges]
        counts = counts[has_edges]
        positions = torch.repeat_interleave(starts, counts, output_size=int(counts.sum().item())) + _segment_arange(counts)
        edge_positions = order.index_select(0, positions)
        edge_positions = edge_positions[~reachable_edges.index_select(0, edge_positions)]
        if int(edge_positions.numel()) == 0:
            frontier_nodes = torch.empty(0, dtype=torch.long, device=edge_ids.device)
            continue
        reachable_edges[edge_positions] = True
        frontier_nodes = torch.unique(row_dst_keys.index_select(0, edge_positions))

    reachable_count = torch.zeros(int(edge_ids.size(0)), dtype=torch.long, device=edge_ids.device)
    reachable_count.scatter_add_(0, rows, reachable_edges.long())
    return reachable_count.eq(edge_count)


def _prefix_mask(count: Tensor, width: int) -> Tensor:
    return torch.arange(width, device=count.device).view(1, -1).lt(count.view(-1, 1))


def _segment_arange(count: Tensor) -> Tensor:
    total = int(count.sum().item())
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=count.device)
    starts = torch.cumsum(count, dim=0) - count
    return torch.arange(total, device=count.device) - torch.repeat_interleave(starts, count, output_size=total)


__all__ = [
    "EdgeSelection",
    "ExpansionBatch",
    "FrontierEncoding",
    "NodeSelection",
    "StateBatch",
    "frontier_from_graph",
    "remove_selected_edge",
    "root_reachable_mask_from_edges",
]
