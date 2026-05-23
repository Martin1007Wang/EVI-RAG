from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext


@dataclass(frozen=True, slots=True)
class Frontier:
    """
    Candidate physical KG edges for expansion.

    The terminal action is not included here.
    The full policy action space is:

        A(z) = {TERMINAL} ∪ Frontier(z)
    """

    row_ids: torch.Tensor
    edge_ids: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.edge_ids.numel())

    @property
    def is_empty(self) -> bool:
        return self.edge_ids.numel() == 0


@dataclass(frozen=True, slots=True)
class State:
    """
    Dynamic evidence-subgraph state.

    For each rollout row r:

        S_r = selected evidence edge set
        X_r = active node set

    The hot path stores selected edges sparsely as padded per-row ids.
    Dense edge/node masks remain available as lazy compatibility views.
    """

    graph_ids: torch.Tensor  # [R]
    selected_edge_ids: torch.Tensor  # [R, K] padded with -1
    active_node_ids: torch.Tensor  # [R, K_nodes] padded with -1
    step: torch.Tensor  # [R]
    remaining_budget: torch.Tensor  # [R]
    num_graph_edges: int
    num_graph_nodes: int
    _selected_edge_mask_cache: torch.Tensor | None = None
    _active_node_mask_cache: torch.Tensor | None = None
    _frontier_cache: Frontier | None = None
    _edge_state_h_cache: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.graph_ids.ndim != 1:
            raise ValueError(f"graph_ids must have shape [R], got {tuple(self.graph_ids.shape)}.")
        if self.selected_edge_ids.ndim != 2:
            raise ValueError(
                f"selected_edge_ids must have shape [R, K], got {tuple(self.selected_edge_ids.shape)}."
            )
        if self.active_node_ids.ndim != 2:
            raise ValueError(
                f"active_node_ids must have shape [R, K_nodes], got {tuple(self.active_node_ids.shape)}."
            )
        num_rows = int(self.graph_ids.numel())
        if int(self.selected_edge_ids.size(0)) != num_rows:
            raise ValueError("selected_edge_ids rows must match graph_ids length.")
        if int(self.active_node_ids.size(0)) != num_rows:
            raise ValueError("active_node_ids rows must match graph_ids length.")
        if self.step.shape != (num_rows,):
            raise ValueError(f"step must have shape [{num_rows}], got {tuple(self.step.shape)}.")
        if self.remaining_budget.shape != (num_rows,):
            raise ValueError(
                f"remaining_budget must have shape [{num_rows}], got {tuple(self.remaining_budget.shape)}."
            )
        if bool(self.remaining_budget.lt(0).any()):
            raise ValueError("remaining_budget must be non-negative.")

    @classmethod
    def initial(
        cls,
        graph: GraphContext,
        graph_ids: torch.Tensor,
        expand_budget: int = 0,
    ) -> State:
        graph_ids = graph_ids.to(device=graph.device, dtype=torch.long).view(-1)
        num_rows = int(graph_ids.numel())
        empty_edges = torch.empty((num_rows, 0), dtype=torch.long, device=graph.device)
        active_node_ids = active_nodes_for_graph_rows(
            graph=graph,
            graph_ids=graph_ids,
        )
        step = torch.zeros(
            num_rows,
            dtype=torch.long,
            device=graph.device,
        )
        remaining_budget = torch.full(
            (num_rows,),
            int(expand_budget),
            dtype=torch.long,
            device=graph.device,
        )
        return cls(
            graph_ids=graph_ids,
            selected_edge_ids=empty_edges,
            active_node_ids=active_node_ids,
            step=step,
            remaining_budget=remaining_budget,
            num_graph_edges=int(graph.num_edges),
            num_graph_nodes=int(graph.num_nodes),
        )

    @classmethod
    def from_selected_edges(
        cls,
        *,
        graph: GraphContext,
        graph_ids: torch.Tensor,
        selected_edge_mask: torch.Tensor,
        expand_budget: int | None = None,
    ) -> State:
        graph_ids = graph_ids.to(device=graph.device, dtype=torch.long).view(-1)
        selected_edge_mask = selected_edge_mask.to(device=graph.device, dtype=torch.bool)
        if selected_edge_mask.ndim != 2:
            raise ValueError(f"selected_edge_mask must have shape [R, E], got {tuple(selected_edge_mask.shape)}.")
        if int(selected_edge_mask.size(0)) != int(graph_ids.numel()):
            raise ValueError("selected_edge_mask rows must match graph_ids length.")
        if int(selected_edge_mask.size(1)) != int(graph.num_edges):
            raise ValueError(
                "selected_edge_mask edge dimension must match graph.num_edges: "
                f"{int(selected_edge_mask.size(1))} != {int(graph.num_edges)}."
            )

        rows, edge_ids = selected_edge_mask.nonzero(as_tuple=True)
        if edge_ids.numel() > 0:
            edge_graph_ids = graph.edge_to_graph.index_select(0, edge_ids)
            row_graph_ids = graph_ids.index_select(0, rows)
            if not bool(edge_graph_ids.eq(row_graph_ids).all()):
                raise ValueError("selected_edge_mask contains edges outside their row graph.")

        selected_edge_ids = padded_ids_from_mask(selected_edge_mask, pad_value=-1)
        active_node_ids = active_node_ids_from_selected_edges(
            graph=graph,
            graph_ids=graph_ids,
            selected_edge_ids=selected_edge_ids,
        )
        step = selected_edge_mask.sum(dim=1).to(dtype=torch.long)
        if expand_budget is None:
            remaining_budget = torch.zeros_like(step)
        else:
            remaining_budget = (torch.full_like(step, int(expand_budget)) - step).clamp_min(0)
        return cls(
            graph_ids=graph_ids,
            selected_edge_ids=selected_edge_ids,
            active_node_ids=active_node_ids,
            step=step,
            remaining_budget=remaining_budget,
            num_graph_edges=int(graph.num_edges),
            num_graph_nodes=int(graph.num_nodes),
            _selected_edge_mask_cache=selected_edge_mask.contiguous(),
        )

    @property
    def device(self) -> torch.device:
        return self.graph_ids.device

    @property
    def num_rows(self) -> int:
        return int(self.graph_ids.numel())

    @property
    def num_edges(self) -> int:
        return int(self.num_graph_edges)

    @property
    def selected_edge_count(self) -> torch.Tensor:
        return self.step

    @property
    def selected_edge_mask(self) -> torch.Tensor:
        cached = self._selected_edge_mask_cache
        if cached is not None:
            return cached
        mask = torch.zeros(
            (self.num_rows, int(self.num_graph_edges)),
            dtype=torch.bool,
            device=self.device,
        )
        row_ids, edge_ids = self.selected_edges()
        if edge_ids.numel() > 0:
            mask[row_ids, edge_ids] = True
        object.__setattr__(self, "_selected_edge_mask_cache", mask)
        return mask

    @property
    def active_node_mask(self) -> torch.Tensor:
        cached = self._active_node_mask_cache
        if cached is not None:
            return cached
        mask = torch.zeros(
            (self.num_rows, int(self.num_graph_nodes)),
            dtype=torch.bool,
            device=self.device,
        )
        row_ids, node_ids = self.active_nodes()
        if node_ids.numel() > 0:
            mask[row_ids, node_ids] = True
        object.__setattr__(self, "_active_node_mask_cache", mask)
        return mask

    @property
    def edge_mask(self) -> torch.Tensor:
        return self.selected_edge_mask

    @property
    def row_to_graph(self) -> torch.Tensor:
        return self.graph_ids

    @property
    def depth(self) -> torch.Tensor:
        return self.step

    def selected_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        return non_padded_ids(self.selected_edge_ids)

    def active_nodes(self) -> tuple[torch.Tensor, torch.Tensor]:
        return non_padded_ids(self.active_node_ids)

    def select_rows(
        self,
        rows: torch.Tensor,
    ) -> State:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        selected_edge_mask = self._selected_edge_mask_cache
        active_node_mask = self._active_node_mask_cache
        frontier = self._frontier_cache
        return State(
            graph_ids=self.graph_ids.index_select(0, rows),
            selected_edge_ids=self.selected_edge_ids.index_select(0, rows),
            active_node_ids=self.active_node_ids.index_select(0, rows),
            step=self.step.index_select(0, rows),
            remaining_budget=self.remaining_budget.index_select(0, rows),
            num_graph_edges=self.num_graph_edges,
            num_graph_nodes=self.num_graph_nodes,
            _selected_edge_mask_cache=(
                selected_edge_mask.index_select(0, rows)
                if selected_edge_mask is not None
                else None
            ),
            _active_node_mask_cache=(
                active_node_mask.index_select(0, rows)
                if active_node_mask is not None
                else None
            ),
            _frontier_cache=select_frontier_rows(frontier, rows) if frontier is not None else None,
            _edge_state_h_cache=None,
        )

    def clone(self) -> State:
        return State(
            graph_ids=self.graph_ids.clone(),
            selected_edge_ids=self.selected_edge_ids.clone(),
            active_node_ids=self.active_node_ids.clone(),
            step=self.step.clone(),
            remaining_budget=self.remaining_budget.clone(),
            num_graph_edges=self.num_graph_edges,
            num_graph_nodes=self.num_graph_nodes,
            _selected_edge_mask_cache=(
                self._selected_edge_mask_cache.clone()
                if self._selected_edge_mask_cache is not None
                else None
            ),
            _active_node_mask_cache=(
                self._active_node_mask_cache.clone()
                if self._active_node_mask_cache is not None
                else None
            ),
            _frontier_cache=clone_frontier(self._frontier_cache),
            _edge_state_h_cache=(
                self._edge_state_h_cache.clone()
                if self._edge_state_h_cache is not None
                else None
            ),
        )

    @classmethod
    def concat(
        cls,
        states: Sequence[State],
    ) -> State:
        if not states:
            raise ValueError("Cannot concatenate an empty state sequence.")

        num_edges = int(states[0].num_graph_edges)
        num_nodes = int(states[0].num_graph_nodes)
        edge_width = max(int(state.selected_edge_ids.size(1)) for state in states)
        node_width = max(int(state.active_node_ids.size(1)) for state in states)
        selected_edge_ids = torch.cat(
            [pad_width(state.selected_edge_ids, edge_width, pad_value=-1) for state in states],
            dim=0,
        )
        active_node_ids = torch.cat(
            [pad_width(state.active_node_ids, node_width, pad_value=-1) for state in states],
            dim=0,
        )
        selected_edge_mask = [state._selected_edge_mask_cache for state in states]
        active_node_mask = [state._active_node_mask_cache for state in states]
        return cls(
            graph_ids=torch.cat([state.graph_ids for state in states], dim=0),
            selected_edge_ids=selected_edge_ids,
            active_node_ids=active_node_ids,
            step=torch.cat([state.step for state in states], dim=0),
            remaining_budget=torch.cat([state.remaining_budget for state in states], dim=0),
            num_graph_edges=num_edges,
            num_graph_nodes=num_nodes,
            _selected_edge_mask_cache=(
                torch.cat(selected_edge_mask, dim=0)
                if all(mask is not None for mask in selected_edge_mask)
                else None
            ),
            _active_node_mask_cache=(
                torch.cat(active_node_mask, dim=0)
                if all(mask is not None for mask in active_node_mask)
                else None
            ),
            _edge_state_h_cache=None,
        )

    def frontier(
        self,
        graph: GraphContext,
        *,
        expand_budget: int | None = None,
    ) -> Frontier:
        """
        Return the physical directed outgoing frontier.

        A KG edge e=(u,r,v) is legal for row i iff:

            u ∈ X_i
            e ∉ S_i
            edge_to_graph[e] == graph_ids[i]
            remaining_budget(i) > 0
        """
        cacheable = expand_budget is None
        if not bool(self.remaining_budget.gt(0).any()):
            return empty_frontier(graph.device)

        cached = self._frontier_cache
        if cached is not None and cacheable:
            return cached

        rows, active_nodes = self.active_nodes()
        if rows.numel() == 0:
            return empty_frontier(graph.device)

        out_rows, out_edges = incident_edges_from_nodes(
            rows=rows,
            node_ids=active_nodes,
            ptr=graph.adjacency.out_ptr,
            edge_ids_by_node=graph.adjacency.edge_ids_by_src,
        )
        if out_edges.numel() == 0:
            return empty_frontier(graph.device)

        frontier_rows, frontier_edge_ids = filter_edges_in_same_graph(
            graph=graph,
            state=self,
            rows=out_rows,
            edge_ids=out_edges,
        )
        if frontier_edge_ids.numel() == 0:
            return empty_frontier(graph.device)

        selected_keys = selected_edge_keys(self)
        if selected_keys.numel() > 0:
            candidate_keys = frontier_rows * int(self.num_graph_edges) + frontier_edge_ids
            keep = ~membership_mask(
                query_ids=candidate_keys,
                candidate_ids=selected_keys,
            )
            frontier_rows = frontier_rows[keep]
            frontier_edge_ids = frontier_edge_ids[keep]
            if frontier_edge_ids.numel() == 0:
                return empty_frontier(graph.device)

        frontier_rows, frontier_edge_ids = unique_row_edge_pairs(
            rows=frontier_rows,
            edge_ids=frontier_edge_ids,
            num_edges=self.num_edges,
        )

        before_horizon = self.remaining_budget.gt(0).index_select(0, frontier_rows)
        frontier_rows = frontier_rows[before_horizon]
        frontier_edge_ids = frontier_edge_ids[before_horizon]
        if frontier_edge_ids.numel() == 0:
            return empty_frontier(graph.device)

        frontier = Frontier(
            row_ids=frontier_rows,
            edge_ids=frontier_edge_ids,
        )
        if cacheable:
            object.__setattr__(self, "_frontier_cache", frontier)
        return frontier

    def expand(
        self,
        graph: GraphContext,
        rows: torch.Tensor,
        edge_ids: torch.Tensor,
        *,
        expand_budget: int,
        validate: bool = False,
    ) -> State:
        """
        Return the child state after selecting one physical edge per row.

        Terminal actions must not enter this method.
        """

        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)
        if rows.numel() != edge_ids.numel():
            raise ValueError("rows and edge_ids must have the same length.")
        if rows.numel() == 0:
            return self
        if validate:
            self.validate_expansion_actions(
                graph=graph,
                rows=rows,
                edge_ids=edge_ids,
                expand_budget=expand_budget,
            )
        else:
            validate_expansion_inputs(
                state=self,
                rows=rows,
                edge_ids=edge_ids,
            )

        selected_edge_ids = append_unique_row_ids(
            padded_ids=self.selected_edge_ids,
            rows=rows,
            values=edge_ids,
        )
        src_node_ids = graph.edge_index[0].index_select(0, edge_ids)
        dst_node_ids = graph.edge_index[1].index_select(0, edge_ids)
        active_node_ids = append_unique_row_pairs(
            padded_ids=self.active_node_ids,
            rows=torch.cat([rows, rows], dim=0),
            values=torch.cat([src_node_ids, dst_node_ids], dim=0),
        )
        step = self.step.clone()
        step.index_add_(
            0,
            rows,
            torch.ones(rows.numel(), dtype=torch.long, device=self.device),
        )
        remaining_budget = self.remaining_budget.clone()
        remaining_budget.index_add_(
            0,
            rows,
            -torch.ones(rows.numel(), dtype=torch.long, device=self.device),
        )
        return State(
            graph_ids=self.graph_ids,
            selected_edge_ids=selected_edge_ids,
            active_node_ids=active_node_ids,
            step=step,
            remaining_budget=remaining_budget,
            num_graph_edges=self.num_graph_edges,
            num_graph_nodes=self.num_graph_nodes,
        )

    def validate_expansion_actions(
        self,
        *,
        graph: GraphContext,
        rows: torch.Tensor,
        edge_ids: torch.Tensor,
        expand_budget: int,
    ) -> None:
        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)
        validate_expansion_inputs(
            state=self,
            rows=rows,
            edge_ids=edge_ids,
        )

        frontier = self.frontier(
            graph,
            expand_budget=int(expand_budget),
        )
        if frontier.edge_ids.numel() == 0:
            raise ValueError("Expansion action is not in the current frontier.")

        key_width = int(self.num_edges)
        frontier_keys = frontier.row_ids * key_width + frontier.edge_ids
        target_keys = rows * key_width + edge_ids
        if not bool(membership_mask(query_ids=target_keys, candidate_ids=frontier_keys).all()):
            raise ValueError("Expansion action is not in the current frontier.")


def validate_expansion_inputs(
    *,
    state: State,
    rows: torch.Tensor,
    edge_ids: torch.Tensor,
) -> None:
    if rows.numel() != edge_ids.numel():
        raise ValueError("rows and edge_ids must have the same length.")
    if rows.numel() == 0:
        return
    if bool(rows.lt(0).any()) or bool(rows.ge(state.num_rows).any()):
        raise ValueError("Expansion rows must be valid state row ids.")
    if bool(edge_ids.lt(0).any()) or bool(edge_ids.ge(state.num_edges).any()):
        raise ValueError("Expansion edge ids must be valid non-terminal edge ids.")
    if int(torch.unique(rows).numel()) != int(rows.numel()):
        raise ValueError("At most one expansion action is allowed per row.")
    if bool(state.remaining_budget.index_select(0, rows).le(0).any()):
        raise ValueError("Expansion rows must have positive remaining budget.")


def active_nodes_for_graph_rows(
    *,
    graph: GraphContext,
    graph_ids: torch.Tensor,
) -> torch.Tensor:
    anchor_node_ids = graph.anchor_mask.nonzero(as_tuple=True)[0]
    if anchor_node_ids.numel() == 0 or graph_ids.numel() == 0:
        return torch.empty((int(graph_ids.numel()), 0), dtype=torch.long, device=graph.device)

    anchor_graph_ids = graph.node_to_graph.index_select(0, anchor_node_ids)
    grouped: list[torch.Tensor] = []
    max_width = 0
    for graph_id in graph_ids.tolist():
        nodes = anchor_node_ids[anchor_graph_ids.eq(int(graph_id))]
        grouped.append(nodes)
        max_width = max(max_width, int(nodes.numel()))
    out = torch.full((int(graph_ids.numel()), max_width), -1, dtype=torch.long, device=graph.device)
    for row, node_ids in enumerate(grouped):
        if node_ids.numel() > 0:
            out[row, : node_ids.numel()] = node_ids
    return out


def active_node_ids_from_selected_edges(
    *,
    graph: GraphContext,
    graph_ids: torch.Tensor,
    selected_edge_ids: torch.Tensor,
) -> torch.Tensor:
    active_node_ids = active_nodes_for_graph_rows(
        graph=graph,
        graph_ids=graph_ids,
    )
    rows, edge_ids = non_padded_ids(selected_edge_ids)
    if edge_ids.numel() == 0:
        return active_node_ids
    src_node_ids = graph.edge_index[0].index_select(0, edge_ids)
    dst_node_ids = graph.edge_index[1].index_select(0, edge_ids)
    return append_unique_row_pairs(
        padded_ids=active_node_ids,
        rows=torch.cat([rows, rows], dim=0),
        values=torch.cat([src_node_ids, dst_node_ids], dim=0),
    )


def selected_edge_keys(state: State) -> torch.Tensor:
    rows, edge_ids = state.selected_edges()
    if edge_ids.numel() == 0:
        return empty_long(state.device)
    return rows * int(state.num_graph_edges) + edge_ids


def select_frontier_rows(frontier: Frontier, rows: torch.Tensor) -> Frontier:
    rows = rows.to(device=frontier.row_ids.device, dtype=torch.long).view(-1)
    if rows.numel() == 0 or frontier.edge_ids.numel() == 0:
        return empty_frontier(frontier.row_ids.device)
    source = frontier.row_ids
    remapped = torch.full((int(source.numel()),), -1, dtype=torch.long, device=source.device)
    for new_row, old_row in enumerate(rows.tolist()):
        remapped[source.eq(int(old_row))] = int(new_row)
    keep = remapped.ge(0)
    if not bool(keep.any()):
        return empty_frontier(source.device)
    return Frontier(
        row_ids=remapped[keep],
        edge_ids=frontier.edge_ids[keep],
    )


def clone_frontier(frontier: Frontier | None) -> Frontier | None:
    if frontier is None:
        return None
    return Frontier(
        row_ids=frontier.row_ids.clone(),
        edge_ids=frontier.edge_ids.clone(),
    )


def non_padded_ids(
    padded_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if padded_ids.numel() == 0:
        return empty_long(padded_ids.device), empty_long(padded_ids.device)
    return padded_ids.ge(0).nonzero(as_tuple=True)[0], padded_ids[padded_ids.ge(0)]


def pad_width(
    tensor: torch.Tensor,
    width: int,
    *,
    pad_value: int,
) -> torch.Tensor:
    width = int(width)
    if int(tensor.size(1)) == width:
        return tensor
    out = torch.full(
        (int(tensor.size(0)), width),
        int(pad_value),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if tensor.numel() > 0:
        out[:, : int(tensor.size(1))] = tensor
    return out


def append_unique_row_ids(
    *,
    padded_ids: torch.Tensor,
    rows: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    return append_unique_row_pairs(
        padded_ids=padded_ids,
        rows=rows,
        values=values,
    )


def append_unique_row_pairs(
    *,
    padded_ids: torch.Tensor,
    rows: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    rows = rows.to(device=padded_ids.device, dtype=torch.long).view(-1)
    values = values.to(device=padded_ids.device, dtype=torch.long).view(-1)
    if rows.numel() != values.numel():
        raise ValueError("rows and values must have the same length.")
    existing_counts = padded_ids.ge(0).sum(dim=1) if padded_ids.numel() > 0 else torch.zeros(
        int(padded_ids.size(0)), dtype=torch.long, device=padded_ids.device
    )
    additions_per_row = torch.zeros_like(existing_counts)
    memberships = []
    for row, value in zip(rows.tolist(), values.tolist(), strict=True):
        row_ids = padded_ids[row, : int(existing_counts[row].item())] if int(existing_counts[row].item()) > 0 else empty_long(
            padded_ids.device
        )
        exists = bool(row_ids.eq(int(value)).any())
        memberships.append(exists)
        if not exists:
            additions_per_row[row] += 1
    if not bool(additions_per_row.any()):
        return padded_ids

    target_width = int((existing_counts + additions_per_row).max().item())
    out = pad_width(padded_ids, target_width, pad_value=-1)
    next_pos = existing_counts.clone()
    for exists, row, value in zip(memberships, rows.tolist(), values.tolist(), strict=True):
        if exists:
            continue
        out[row, int(next_pos[row].item())] = int(value)
        next_pos[row] += 1
    return out


def padded_ids_from_mask(
    mask: torch.Tensor,
    *,
    pad_value: int,
) -> torch.Tensor:
    row_counts = mask.sum(dim=1).to(dtype=torch.long)
    width = int(row_counts.max().item()) if row_counts.numel() > 0 else 0
    out = torch.full(
        (int(mask.size(0)), width),
        int(pad_value),
        dtype=torch.long,
        device=mask.device,
    )
    if width == 0:
        return out
    for row in range(int(mask.size(0))):
        edge_ids = mask[row].nonzero(as_tuple=False).flatten()
        if edge_ids.numel() > 0:
            out[row, : edge_ids.numel()] = edge_ids
    return out


def incident_edges_from_nodes(
    *,
    rows: torch.Tensor,
    node_ids: torch.Tensor,
    ptr: torch.Tensor,
    edge_ids_by_node: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Enumerate edge ids grouped by active nodes.
    """

    starts = ptr.index_select(0, node_ids)
    ends = ptr.index_select(0, node_ids + 1)
    degrees = ends - starts

    keep = degrees.gt(0)
    if not bool(keep.any()):
        return empty_long(rows.device), empty_long(rows.device)

    kept_rows = rows[keep]
    kept_starts = starts[keep]
    kept_degrees = degrees[keep]

    expanded_rows = torch.repeat_interleave(
        kept_rows,
        kept_degrees,
    )
    positions = torch.repeat_interleave(
        kept_starts,
        kept_degrees,
    ) + segment_arange(kept_degrees)

    expanded_edge_ids = edge_ids_by_node.index_select(
        0,
        positions,
    )

    return expanded_rows, expanded_edge_ids


def filter_edges_in_same_graph(
    *,
    graph: GraphContext,
    state: State,
    rows: torch.Tensor,
    edge_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_graph_ids = graph.edge_to_graph.index_select(
        0,
        edge_ids,
    )
    row_graph_ids = state.graph_ids.index_select(
        0,
        rows,
    )

    keep = edge_graph_ids.eq(row_graph_ids)

    return rows[keep], edge_ids[keep]


def unique_row_edge_pairs(
    *,
    rows: torch.Tensor,
    edge_ids: torch.Tensor,
    num_edges: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_keys = rows * int(num_edges) + edge_ids
    pair_keys = torch.unique(
        pair_keys,
        sorted=True,
    )

    unique_rows = torch.div(
        pair_keys,
        int(num_edges),
        rounding_mode="floor",
    )
    unique_edge_ids = pair_keys.remainder(
        int(num_edges),
    )

    return unique_rows, unique_edge_ids


def membership_mask(
    *,
    query_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    query_ids = query_ids.view(-1)
    candidate_ids = candidate_ids.view(-1)
    if query_ids.numel() == 0 or candidate_ids.numel() == 0:
        return torch.zeros(query_ids.numel(), dtype=torch.bool, device=query_ids.device)
    sorted_candidates = torch.sort(candidate_ids).values
    positions = torch.searchsorted(sorted_candidates, query_ids)
    in_bounds = positions.lt(sorted_candidates.numel())
    matched = torch.zeros(query_ids.numel(), dtype=torch.bool, device=query_ids.device)
    if bool(in_bounds.any()):
        matched[in_bounds] = sorted_candidates.index_select(0, positions[in_bounds]).eq(query_ids[in_bounds])
    return matched


def empty_frontier(
    device: torch.device,
) -> Frontier:
    empty = empty_long(device)
    return Frontier(
        row_ids=empty,
        edge_ids=empty,
    )


def empty_long(
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )


def segment_arange(
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    For lengths [a, b, c], return:

        [0, ..., a-1, 0, ..., b-1, 0, ..., c-1]
    """

    total = int(lengths.sum().item())
    if total == 0:
        return empty_long(lengths.device)

    starts = (
        torch.cumsum(
            lengths,
            dim=0,
        )
        - lengths
    )

    return torch.arange(
        total,
        dtype=torch.long,
        device=lengths.device,
    ) - torch.repeat_interleave(
        starts,
        lengths,
    )


__all__ = [
    "Frontier",
    "State",
    "empty_frontier",
]
