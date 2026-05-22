from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext


@dataclass(frozen=True, slots=True)
class Frontier:
    """
    Candidate physical KG edges for expansion.

    STOP is not included here.
    The full policy action space is:

        A(z) = {STOP} ∪ Frontier(z)
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

    selected_edge_mask is the canonical evidence-edge set S_r.

    active_node_mask is a maintained cache:

        X_r = anchors(graph_ids[r]) ∪ endpoints(S_r)

    step is the number of expansion actions already applied.
    Under valid transitions, step equals |S_r|.
    """

    graph_ids: torch.Tensor  # [R]
    selected_edge_mask: torch.Tensor  # [R, E]
    active_node_mask: torch.Tensor  # [R, N]
    step: torch.Tensor  # [R]

    @classmethod
    def initial(
        cls,
        graph: GraphContext,
        graph_ids: torch.Tensor,
    ) -> State:
        num_rows = int(graph_ids.numel())

        selected_edge_mask = torch.zeros(
            (num_rows, int(graph.num_edges)),
            dtype=torch.bool,
            device=graph.device,
        )
        active_node_mask = anchor_mask_for_graph_rows(
            graph=graph,
            graph_ids=graph_ids,
        )
        step = torch.zeros(
            num_rows,
            dtype=torch.long,
            device=graph.device,
        )

        return cls(
            graph_ids=graph_ids,
            selected_edge_mask=selected_edge_mask,
            active_node_mask=active_node_mask,
            step=step,
        )

    @classmethod
    def from_selected_edges(
        cls,
        *,
        graph: GraphContext,
        graph_ids: torch.Tensor,
        selected_edge_mask: torch.Tensor,
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

        active_node_mask = anchor_mask_for_graph_rows(
            graph=graph,
            graph_ids=graph_ids,
        )
        rows, edge_ids = selected_edge_mask.nonzero(as_tuple=True)
        if edge_ids.numel() > 0:
            edge_graph_ids = graph.edge_to_graph.index_select(0, edge_ids)
            row_graph_ids = graph_ids.index_select(0, rows)
            if not bool(edge_graph_ids.eq(row_graph_ids).all()):
                raise ValueError("selected_edge_mask contains edges outside their row graph.")
            src_node_ids = graph.edge_index[0].index_select(0, edge_ids)
            dst_node_ids = graph.edge_index[1].index_select(0, edge_ids)
            active_node_mask[rows, src_node_ids] = True
            active_node_mask[rows, dst_node_ids] = True

        return cls(
            graph_ids=graph_ids,
            selected_edge_mask=selected_edge_mask,
            active_node_mask=active_node_mask,
            step=selected_edge_mask.sum(dim=1).to(dtype=torch.long),
        )

    @property
    def device(self) -> torch.device:
        return self.selected_edge_mask.device

    @property
    def num_rows(self) -> int:
        return int(self.graph_ids.numel())

    @property
    def num_edges(self) -> int:
        return int(self.selected_edge_mask.size(1))

    @property
    def selected_edge_count(self) -> torch.Tensor:
        return self.step

    @property
    def edge_mask(self) -> torch.Tensor:
        return self.selected_edge_mask

    @property
    def row_to_graph(self) -> torch.Tensor:
        return self.graph_ids

    @property
    def depth(self) -> torch.Tensor:
        return self.step

    def selected_edges(self) -> tuple[torch.Tensor, ...]:
        """
        Return selected edges as:

            row_ids, edge_ids
        """

        return self.selected_edge_mask.nonzero(as_tuple=True)

    def select_rows(
        self,
        rows: torch.Tensor,
    ) -> State:
        return State(
            graph_ids=self.graph_ids.index_select(0, rows),
            selected_edge_mask=self.selected_edge_mask.index_select(0, rows),
            active_node_mask=self.active_node_mask.index_select(0, rows),
            step=self.step.index_select(0, rows),
        )

    def clone(self) -> State:
        return State(
            graph_ids=self.graph_ids.clone(),
            selected_edge_mask=self.selected_edge_mask.clone(),
            active_node_mask=self.active_node_mask.clone(),
            step=self.step.clone(),
        )

    @classmethod
    def concat(
        cls,
        states: Sequence[State],
    ) -> State:
        if not states:
            raise ValueError("Cannot concatenate an empty state sequence.")

        return cls(
            graph_ids=torch.cat(
                [state.graph_ids for state in states],
                dim=0,
            ),
            selected_edge_mask=torch.cat(
                [state.selected_edge_mask for state in states],
                dim=0,
            ),
            active_node_mask=torch.cat(
                [state.active_node_mask for state in states],
                dim=0,
            ),
            step=torch.cat(
                [state.step for state in states],
                dim=0,
            ),
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
            depth(i) < expand_budget, when expand_budget is provided

        This method does not fabricate inverse edges.

        STOP is not included.
        """
        if expand_budget is not None and not bool(self.depth.lt(int(expand_budget)).any()):
            return empty_frontier(graph.device)

        rows, active_nodes = self.active_node_mask.nonzero(as_tuple=True)
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

        frontier_rows = out_rows
        frontier_edge_ids = out_edges

        frontier_rows, frontier_edge_ids = filter_edges_in_same_graph(
            graph=graph,
            state=self,
            rows=frontier_rows,
            edge_ids=frontier_edge_ids,
        )
        if frontier_edge_ids.numel() == 0:
            return empty_frontier(graph.device)

        unselected = ~self.selected_edge_mask[
            frontier_rows,
            frontier_edge_ids,
        ]
        frontier_rows = frontier_rows[unselected]
        frontier_edge_ids = frontier_edge_ids[unselected]
        if frontier_edge_ids.numel() == 0:
            return empty_frontier(graph.device)

        frontier_rows, frontier_edge_ids = unique_row_edge_pairs(
            rows=frontier_rows,
            edge_ids=frontier_edge_ids,
            num_edges=self.num_edges,
        )

        if expand_budget is not None:
            before_horizon = self.depth.lt(int(expand_budget)).index_select(
                0,
                frontier_rows,
            )
            frontier_rows = frontier_rows[before_horizon]
            frontier_edge_ids = frontier_edge_ids[before_horizon]
            if frontier_edge_ids.numel() == 0:
                return empty_frontier(graph.device)

        return Frontier(
            row_ids=frontier_rows,
            edge_ids=frontier_edge_ids,
        )

    def expand(
        self,
        graph: GraphContext,
        rows: torch.Tensor,
        edge_ids: torch.Tensor,
        *,
        expand_budget: int,
    ) -> State:
        """
        Return the child state after selecting one physical edge per row.

        STOP actions must not enter this method.
        """

        if rows.numel() != edge_ids.numel():
            raise ValueError("rows and edge_ids must have the same length.")

        rows = rows.to(device=self.device, dtype=torch.long).view(-1)
        edge_ids = edge_ids.to(device=self.device, dtype=torch.long).view(-1)

        if rows.numel() == 0:
            return self

        self.validate_expansion_actions(
            graph=graph,
            rows=rows,
            edge_ids=edge_ids,
            expand_budget=expand_budget,
        )

        selected_edge_mask = self.selected_edge_mask.clone()
        active_node_mask = self.active_node_mask.clone()
        step = self.step.clone()

        selected_edge_mask[rows, edge_ids] = True

        src_node_ids = graph.edge_index[0].index_select(0, edge_ids)
        dst_node_ids = graph.edge_index[1].index_select(0, edge_ids)

        active_node_mask[rows, src_node_ids] = True
        active_node_mask[rows, dst_node_ids] = True

        step.index_add_(
            0,
            rows,
            torch.ones(
                rows.numel(),
                dtype=torch.long,
                device=self.device,
            ),
        )

        return State(
            graph_ids=self.graph_ids,
            selected_edge_mask=selected_edge_mask,
            active_node_mask=active_node_mask,
            step=step,
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

        if rows.numel() != edge_ids.numel():
            raise ValueError("rows and edge_ids must have the same length.")
        if rows.numel() == 0:
            return
        if bool(rows.lt(0).any()) or bool(rows.ge(self.num_rows).any()):
            raise ValueError("Expansion rows must be valid state row ids.")
        if bool(edge_ids.lt(0).any()) or bool(edge_ids.ge(self.num_edges).any()):
            raise ValueError("Expansion edge ids must be valid non-STOP edge ids.")
        if int(torch.unique(rows).numel()) != int(rows.numel()):
            raise ValueError("At most one expansion action is allowed per row.")

        frontier = self.frontier(
            graph,
            expand_budget=int(expand_budget),
        )
        if frontier.edge_ids.numel() == 0:
            raise ValueError("Expansion action is not in the current frontier.")

        key_width = int(self.num_edges)
        frontier_keys = frontier.row_ids * key_width + frontier.edge_ids
        target_keys = rows * key_width + edge_ids
        order = torch.argsort(frontier_keys)
        sorted_keys = frontier_keys.index_select(0, order)
        positions = torch.searchsorted(sorted_keys, target_keys)
        in_bounds = positions.lt(sorted_keys.numel())
        if not bool(in_bounds.all()):
            raise ValueError("Expansion action is not in the current frontier.")
        matched = sorted_keys.index_select(0, positions)
        if not torch.equal(matched, target_keys):
            raise ValueError("Expansion action is not in the current frontier.")


def anchor_mask_for_graph_rows(
    *,
    graph: GraphContext,
    graph_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Build initial active-node masks for rollout rows.

    Row i activates all anchors whose graph id equals graph_ids[i].
    """

    out = torch.zeros(
        (int(graph_ids.numel()), int(graph.num_nodes)),
        dtype=torch.bool,
        device=graph.device,
    )

    anchor_node_ids = graph.anchor_mask.nonzero(as_tuple=True)[0]
    if anchor_node_ids.numel() == 0:
        return out

    anchor_graph_ids = graph.node_to_graph.index_select(
        0,
        anchor_node_ids,
    )

    rows, anchor_positions = (
        graph_ids[:, None]
        .eq(
            anchor_graph_ids[None, :],
        )
        .nonzero(as_tuple=True)
    )

    if rows.numel() > 0:
        out[
            rows,
            anchor_node_ids.index_select(0, anchor_positions),
        ] = True

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

    Inputs:
        rows:
            Row id for each active node occurrence.

        node_ids:
            Active physical node ids.

        ptr / edge_ids_by_node:
            CSR node -> edge-id index.
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
    "anchor_mask_for_graph_rows",
    "empty_frontier",
]
