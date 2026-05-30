from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.weaver.context import GraphContext

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ExpansionBatch:
    state_ids: Tensor  # [K]
    edge_ids: Tensor  # [K]

    @property
    def device(self) -> torch.device:
        return self.state_ids.device

    @property
    def num_actions(self) -> int:
        return int(self.edge_ids.numel())

    @classmethod
    def empty(cls, *, device: torch.device) -> ExpansionBatch:
        empty = _empty_long(device)
        return cls(state_ids=empty, edge_ids=empty)


@dataclass(frozen=True, slots=True)
class ActionSpace:
    num_states: int
    expand_state_ids: Tensor  # [F]
    expand_edge_ids: Tensor  # [F]
    expand_ptr: Tensor  # [S + 1]

    @property
    def device(self) -> torch.device:
        return self.expand_ptr.device

    @property
    def num_expansions(self) -> int:
        return int(self.expand_edge_ids.numel())

    @property
    def expand_count(self) -> Tensor:
        return self.expand_ptr[1:] - self.expand_ptr[:-1]


@dataclass(frozen=True, slots=True)
class FrontierEncoding:
    row_ids: Tensor  # [F]
    edge_ids: Tensor  # [F]
    dst_ids: Tensor  # [F]
    remaining_budget: Tensor  # [S]

    @property
    def device(self) -> torch.device:
        return self.row_ids.device

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
    selected_edge_count: Tensor  # [S]
    selected_edge_ids: Tensor  # [S, B]
    activated_node_count: Tensor  # [S]
    activated_node_ids: Tensor  # [S, B]
    anchor_count: Tensor  # [S]
    budget: int

    @classmethod
    def initial(
        cls,
        *,
        graph_ids: Tensor,
        budget: int,
        graph_context: GraphContext,
    ) -> StateBatch:
        graph_ids = graph_ids.to(
            device=graph_context.device,
            dtype=torch.long,
        ).view(-1)
        budget = int(budget)
        if budget < 0:
            raise ValueError("budget must be nonnegative.")

        num_states = int(graph_ids.numel())
        selected_edge_count = torch.zeros(
            num_states,
            dtype=torch.long,
            device=graph_context.device,
        )
        selected_edge_ids = torch.full(
            (num_states, budget),
            -1,
            dtype=torch.long,
            device=graph_context.device,
        )
        activated_node_count = torch.zeros(
            num_states,
            dtype=torch.long,
            device=graph_context.device,
        )
        activated_node_ids = torch.full(
            (num_states, budget),
            -1,
            dtype=torch.long,
            device=graph_context.device,
        )
        anchor_count = (
            graph_context.anchor_ptr.index_select(0, graph_ids + 1)
            - graph_context.anchor_ptr.index_select(0, graph_ids)
        )
        return cls(
            graph_ids=graph_ids,
            selected_edge_count=selected_edge_count,
            selected_edge_ids=selected_edge_ids,
            activated_node_count=activated_node_count,
            activated_node_ids=activated_node_ids,
            anchor_count=anchor_count,
            budget=budget,
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
        state = cls.initial(
            graph_ids=graph_ids,
            budget=budget,
            graph_context=graph_context,
        )
        edge_ids = edge_ids.to(device=graph_context.device, dtype=torch.long)
        edge_count = edge_count.to(device=graph_context.device, dtype=torch.long).view(-1)

        if edge_ids.ndim != 2:
            raise ValueError(f"edge_ids must have shape [S, B], got {tuple(edge_ids.shape)}.")
        if int(edge_ids.size(0)) != int(state.num_states):
            raise ValueError("edge_ids must have one row per state.")
        if int(edge_ids.size(1)) != int(budget):
            raise ValueError("edge_ids must have one column per budget slot.")
        if int(edge_count.numel()) != int(state.num_states):
            raise ValueError("edge_count must have one item per state.")
        if bool(edge_count.lt(0).any()) or bool(edge_count.gt(int(budget)).any()):
            raise ValueError("edge_count must be in [0, budget] for every state.")

        next_state = state
        for slot in range(int(edge_ids.size(1))):
            valid_rows = edge_count.gt(slot).nonzero(as_tuple=True)[0]
            if int(valid_rows.numel()) == 0:
                continue
            next_state = next_state.advance(
                ExpansionBatch(
                    state_ids=valid_rows,
                    edge_ids=edge_ids[valid_rows, slot],
                ),
                graph_context=graph_context,
            )
        return next_state

    @property
    def device(self) -> torch.device:
        return self.graph_ids.device

    @property
    def num_states(self) -> int:
        return int(self.graph_ids.numel())

    @property
    def edge_count(self) -> Tensor:
        return self.selected_edge_count

    @property
    def edge_ids(self) -> Tensor:
        budget = int(self.budget)
        out = self.selected_edge_ids.clone()
        if budget == 0 or self.num_states == 0:
            return out
        valid = torch.arange(
            budget,
            dtype=torch.long,
            device=self.device,
        ).view(1, -1).lt(self.selected_edge_count.view(-1, 1))
        if not bool(valid.any()):
            return out
        sortable = torch.where(valid, out, torch.full_like(out, torch.iinfo(out.dtype).max))
        sortable, _ = torch.sort(sortable, dim=1)
        out = torch.where(valid, sortable, torch.full_like(out, -1))
        return out

    @property
    def budget_left(self) -> Tensor:
        return int(self.budget) - self.edge_count

    def take(self, state_ids: Tensor) -> StateBatch:
        state_ids = state_ids.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)
        return StateBatch(
            graph_ids=self.graph_ids.index_select(0, state_ids),
            selected_edge_count=self.selected_edge_count.index_select(0, state_ids),
            selected_edge_ids=self.selected_edge_ids.index_select(0, state_ids),
            activated_node_count=self.activated_node_count.index_select(0, state_ids),
            activated_node_ids=self.activated_node_ids.index_select(0, state_ids),
            anchor_count=self.anchor_count.index_select(0, state_ids),
            budget=int(self.budget),
        )

    def branch(
        self,
        expansion: ExpansionBatch,
        *,
        graph_context: GraphContext,
    ) -> StateBatch:
        parent_state_ids = expansion.state_ids.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)
        new_edge_ids = expansion.edge_ids.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)
        if int(parent_state_ids.numel()) != int(new_edge_ids.numel()):
            raise ValueError("expansion.state_ids and expansion.edge_ids must have the same length.")
        next_state = StateBatch(
            graph_ids=self.graph_ids.index_select(0, parent_state_ids),
            selected_edge_count=self.selected_edge_count.index_select(0, parent_state_ids).clone(),
            selected_edge_ids=self.selected_edge_ids.index_select(0, parent_state_ids).clone(),
            activated_node_count=self.activated_node_count.index_select(0, parent_state_ids).clone(),
            activated_node_ids=self.activated_node_ids.index_select(0, parent_state_ids).clone(),
            anchor_count=self.anchor_count.index_select(0, parent_state_ids),
            budget=int(self.budget),
        )
        return next_state.advance(
            ExpansionBatch(
                state_ids=torch.arange(
                    int(parent_state_ids.numel()),
                    dtype=torch.long,
                    device=self.device,
                ),
                edge_ids=new_edge_ids,
            ),
            graph_context=graph_context,
        )

    def advance(
        self,
        expansion: ExpansionBatch,
        *,
        graph_context: GraphContext,
    ) -> StateBatch:
        rows = expansion.state_ids.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)
        new_edge_ids = expansion.edge_ids.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)
        if int(rows.numel()) != int(new_edge_ids.numel()):
            raise ValueError("expansion.state_ids and expansion.edge_ids must have the same length.")
        if int(rows.numel()) == 0:
            return self
        if int(torch.unique(rows).numel()) != int(rows.numel()):
            raise ValueError("advance() requires each state row to appear at most once.")
        if bool(self.budget_left.index_select(0, rows).le(0).any()):
            raise ValueError("advance() received rows with no remaining budget.")

        parent_graph_ids = self.graph_ids.index_select(0, rows)
        edge_graph_ids = graph_context.edge_to_graph.index_select(0, new_edge_ids)
        if not bool(parent_graph_ids.eq(edge_graph_ids).all()):
            raise ValueError("advance() received cross-graph edges.")

        selected_ids = self.selected_edge_ids.index_select(0, rows)
        selected_count = self.selected_edge_count.index_select(0, rows)
        if bool(_rows_have_edge(selected_ids, selected_count, new_edge_ids).any()):
            raise ValueError("advance() received an already selected edge.")

        active_nodes = _active_node_id_matrix(
            state=self.take(rows),
            graph_context=graph_context,
        )
        src_ids = graph_context.edge_src.index_select(0, new_edge_ids)
        if not bool(_rows_have_node(active_nodes, src_ids).all()):
            raise ValueError("advance() received edges whose source is inactive.")

        next_selected_edge_count = self.selected_edge_count.clone()
        next_selected_edge_ids = self.selected_edge_ids.clone()
        next_activated_node_count = self.activated_node_count.clone()
        next_activated_node_ids = self.activated_node_ids.clone()

        dst_ids = graph_context.edge_dst.index_select(0, new_edge_ids)
        dst_was_inactive = ~_rows_have_node(active_nodes, dst_ids)
        insert_pos = next_selected_edge_count.index_select(0, rows)
        next_selected_edge_ids[rows, insert_pos] = new_edge_ids
        next_selected_edge_count[rows] = next_selected_edge_count[rows] + 1

        if bool(dst_was_inactive.any()):
            active_rows = rows[dst_was_inactive]
            active_dst_ids = dst_ids[dst_was_inactive]
            active_insert_pos = next_activated_node_count.index_select(0, active_rows)
            next_activated_node_ids[active_rows, active_insert_pos] = active_dst_ids
            next_activated_node_count[active_rows] = next_activated_node_count[active_rows] + 1

        return StateBatch(
            graph_ids=self.graph_ids,
            selected_edge_count=next_selected_edge_count,
            selected_edge_ids=next_selected_edge_ids,
            activated_node_count=next_activated_node_count,
            activated_node_ids=next_activated_node_ids,
            anchor_count=self.anchor_count,
            budget=int(self.budget),
        )

    def covered_node_pairs(self, graph: GraphContext) -> tuple[Tensor, Tensor]:
        return _covered_node_pairs(state=self, graph=graph)

    def selected_edge_index(self) -> EdgeSelection:
        budget = int(self.budget)
        if budget == 0 or self.num_states == 0:
            empty = _empty_long(self.device)
            return EdgeSelection(row_ids=empty, edge_ids=empty)
        valid = torch.arange(
            budget,
            dtype=torch.long,
            device=self.device,
        ).view(1, -1).lt(self.selected_edge_count.view(-1, 1))
        if not bool(valid.any()):
            empty = _empty_long(self.device)
            return EdgeSelection(row_ids=empty, edge_ids=empty)
        row_ids = valid.nonzero(as_tuple=True)[0]
        edge_ids = self.selected_edge_ids[valid]
        return EdgeSelection(row_ids=row_ids, edge_ids=edge_ids)

    def active_node_index(self, graph: GraphContext | None = None) -> NodeSelection:
        if graph is None:
            raise ValueError("graph is required for sparse active node reconstruction.")
        row_ids, node_ids = _covered_node_pairs(state=self, graph=graph)
        return NodeSelection(row_ids=row_ids, node_ids=node_ids)

    def active_activated_node_index(self) -> NodeSelection:
        budget = int(self.budget)
        if budget == 0 or self.num_states == 0:
            empty = _empty_long(self.device)
            return NodeSelection(row_ids=empty, node_ids=empty)
        valid = torch.arange(
            budget,
            dtype=torch.long,
            device=self.device,
        ).view(1, -1).lt(self.activated_node_count.view(-1, 1))
        if not bool(valid.any()):
            empty = _empty_long(self.device)
            return NodeSelection(row_ids=empty, node_ids=empty)
        row_ids = valid.nonzero(as_tuple=True)[0]
        node_ids = self.activated_node_ids[valid]
        return NodeSelection(row_ids=row_ids, node_ids=node_ids)

    def frontier(
        self,
        *,
        edge_src: Tensor,
        edge_dst: Tensor,
        remaining_budget: Tensor | None = None,
    ) -> FrontierEncoding:
        if remaining_budget is None:
            remaining_budget = self.budget_left
        remaining_budget = remaining_budget.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)
        if int(remaining_budget.numel()) != int(self.num_states):
            raise ValueError("remaining_budget must have one item per state.")
        if self.num_states == 0:
            empty = _empty_long(self.device)
            return FrontierEncoding(
                row_ids=empty,
                edge_ids=empty,
                dst_ids=empty,
                remaining_budget=remaining_budget,
            )
        active_nodes = _active_node_id_matrix_from_edges(
            state=self,
            edge_src=edge_src,
            edge_dst=edge_dst,
        )
        if int(active_nodes.numel()) == 0:
            empty = _empty_long(self.device)
            return FrontierEncoding(
                row_ids=empty,
                edge_ids=empty,
                dst_ids=empty,
                remaining_budget=remaining_budget,
            )

        edge_ids_full = torch.arange(
            int(edge_src.numel()),
            dtype=torch.long,
            device=self.device,
        )
        candidate_rows = torch.arange(
            self.num_states,
            dtype=torch.long,
            device=self.device,
        ).view(-1, 1).expand(-1, int(edge_src.numel()))
        src_match = active_nodes.unsqueeze(-1).eq(edge_src.view(1, 1, -1)).any(dim=1)
        has_budget = remaining_budget.gt(0).view(self.num_states, 1)
        selected = self.selected_edge_ids
        selected_count = self.selected_edge_count
        if int(selected.size(1)) == 0:
            not_selected = torch.ones_like(src_match)
        else:
            steps = torch.arange(
                int(selected.size(1)),
                dtype=torch.long,
                device=self.device,
            ).view(1, -1, 1)
            valid = steps.lt(selected_count.view(-1, 1, 1))
            not_selected = ~(selected.unsqueeze(-1).eq(edge_ids_full.view(1, 1, -1)) & valid).any(dim=1)
        legal = src_match & has_budget & not_selected
        row_ids, edge_ids = legal.nonzero(as_tuple=True)
        if int(edge_ids.numel()) == 0:
            empty = _empty_long(self.device)
            return FrontierEncoding(
                row_ids=empty,
                edge_ids=empty,
                dst_ids=empty,
                remaining_budget=remaining_budget,
            )
        dst_ids = edge_dst.to(device=self.device, dtype=torch.long).index_select(0, edge_ids)
        return FrontierEncoding(
            row_ids=row_ids,
            edge_ids=edge_ids,
            dst_ids=dst_ids,
            remaining_budget=remaining_budget,
        )

    def action_space(self, graph: GraphContext) -> ActionSpace:
        frontier = frontier_from_graph(
            state=self,
            graph=graph,
            remaining_budget=self.budget_left,
        )
        counts = torch.bincount(
            frontier.row_ids,
            minlength=self.num_states,
        ).to(dtype=torch.long)
        ptr = torch.empty(
            self.num_states + 1,
            dtype=torch.long,
            device=self.device,
        )
        ptr[0] = 0
        ptr[1:] = torch.cumsum(counts, dim=0)
        return ActionSpace(
            num_states=self.num_states,
            expand_state_ids=frontier.row_ids,
            expand_edge_ids=frontier.edge_ids,
            expand_ptr=ptr,
        )


def frontier_from_graph(
    *,
    state: StateBatch,
    graph: GraphContext,
    remaining_budget: Tensor | None = None,
) -> FrontierEncoding:
    if remaining_budget is None:
        remaining_budget = state.budget_left
    remaining_budget = remaining_budget.to(
        device=state.device,
        dtype=torch.long,
    ).view(-1)
    if int(remaining_budget.numel()) != int(state.num_states):
        raise ValueError("remaining_budget must have one item per state.")
    if state.num_states == 0:
        empty = _empty_long(state.device)
        return FrontierEncoding(
            row_ids=empty,
            edge_ids=empty,
            dst_ids=empty,
            remaining_budget=remaining_budget,
        )

    active_row_ids, active_node_ids = _covered_node_pairs(state=state, graph=graph)
    if int(active_node_ids.numel()) == 0:
        empty = _empty_long(state.device)
        return FrontierEncoding(
            row_ids=empty,
            edge_ids=empty,
            dst_ids=empty,
            remaining_budget=remaining_budget,
        )

    out_ptr = graph.adjacency.out_ptr
    edge_ids_by_src = graph.adjacency.edge_ids_by_src
    counts = out_ptr.index_select(0, active_node_ids + 1) - out_ptr.index_select(0, active_node_ids)
    total = int(counts.sum().item())
    if total == 0:
        empty = _empty_long(state.device)
        return FrontierEncoding(
            row_ids=empty,
            edge_ids=empty,
            dst_ids=empty,
            remaining_budget=remaining_budget,
        )

    row_ids = torch.repeat_interleave(
        active_row_ids,
        counts,
        output_size=total,
    )
    starts = out_ptr.index_select(0, active_node_ids)
    offsets = _segment_arange(counts)
    positions = torch.repeat_interleave(
        starts,
        counts,
        output_size=total,
    ) + offsets
    candidate_edge_ids = edge_ids_by_src.index_select(0, positions)

    has_budget = remaining_budget.index_select(0, row_ids).gt(0)
    not_selected = ~_rows_have_edge(
        state.selected_edge_ids.index_select(0, row_ids),
        state.selected_edge_count.index_select(0, row_ids),
        candidate_edge_ids,
    )
    keep = has_budget & not_selected
    if not bool(keep.any()):
        empty = _empty_long(state.device)
        return FrontierEncoding(
            row_ids=empty,
            edge_ids=empty,
            dst_ids=empty,
            remaining_budget=remaining_budget,
        )

    kept_row_ids = row_ids[keep]
    kept_edge_ids = candidate_edge_ids[keep]
    kept_dst_ids = graph.edge_dst.index_select(0, kept_edge_ids)
    dedup_keys = kept_row_ids * int(graph.num_edges) + kept_edge_ids
    order = torch.argsort(dedup_keys)
    sorted_keys = dedup_keys.index_select(0, order)
    keep_first = torch.ones(
        int(sorted_keys.numel()),
        dtype=torch.bool,
        device=state.device,
    )
    if int(sorted_keys.numel()) > 1:
        keep_first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    unique_order = order[keep_first]
    return FrontierEncoding(
        row_ids=kept_row_ids.index_select(0, unique_order),
        edge_ids=kept_edge_ids.index_select(0, unique_order),
        dst_ids=kept_dst_ids.index_select(0, unique_order),
        remaining_budget=remaining_budget,
    )


def cat_state_batches(states: Sequence[StateBatch]) -> StateBatch:
    if not states:
        raise ValueError("Cannot concatenate an empty sequence of StateBatch objects.")
    first = states[0]
    for state in states[1:]:
        if int(state.budget) != int(first.budget):
            raise ValueError("Cannot concatenate StateBatch objects with different budgets.")
        if state.device != first.device:
            raise ValueError("Cannot concatenate StateBatch objects on different devices.")
    return StateBatch(
        graph_ids=torch.cat([state.graph_ids for state in states], dim=0),
        selected_edge_count=torch.cat([state.selected_edge_count for state in states], dim=0),
        selected_edge_ids=torch.cat([state.selected_edge_ids for state in states], dim=0),
        activated_node_count=torch.cat([state.activated_node_count for state in states], dim=0),
        activated_node_ids=torch.cat([state.activated_node_ids for state in states], dim=0),
        anchor_count=torch.cat([state.anchor_count for state in states], dim=0),
        budget=int(first.budget),
    )


def canonicalize_state_batch(state: StateBatch) -> StateBatch:
    return state


def remove_selected_edge(
    *,
    state: StateBatch,
    row: int,
    edge_id: int,
    graph_context: GraphContext,
) -> StateBatch:
    row = int(row)
    edge_id = int(edge_id)
    child = state.take(torch.tensor([row], dtype=torch.long, device=state.device))
    count = int(child.edge_count[0].item())
    if count <= 0:
        raise ValueError("edge_id must be selected in the row.")
    selected = child.selected_edge_ids[0, :count]
    keep = selected.ne(edge_id)
    if int(keep.sum().item()) == count:
        raise ValueError("edge_id must be selected in the row.")
    next_edge_ids = torch.full(
        (1, int(child.budget)),
        -1,
        dtype=torch.long,
        device=child.device,
    )
    kept_edges = selected[keep]
    if int(kept_edges.numel()) > 0:
        next_edge_ids[0, : int(kept_edges.numel())] = kept_edges
    return StateBatch(
        graph_ids=child.graph_ids,
        selected_edge_count=torch.tensor([int(kept_edges.numel())], dtype=torch.long, device=child.device),
        selected_edge_ids=next_edge_ids,
        activated_node_count=_recompute_activated_node_count(
            graph_ids=child.graph_ids,
            selected_edge_ids=next_edge_ids,
            selected_edge_count=torch.tensor([int(kept_edges.numel())], dtype=torch.long, device=child.device),
            budget=int(child.budget),
            graph_context=graph_context,
        ),
        activated_node_ids=_recompute_activated_node_ids(
            graph_ids=child.graph_ids,
            selected_edge_ids=next_edge_ids,
            selected_edge_count=torch.tensor([int(kept_edges.numel())], dtype=torch.long, device=child.device),
            budget=int(child.budget),
            graph_context=graph_context,
        ),
        anchor_count=child.anchor_count,
        budget=int(child.budget),
    )


def state_rows_equal(
    *,
    left: StateBatch,
    left_row: int,
    right: StateBatch,
    right_row: int,
) -> bool:
    if int(left.budget) != int(right.budget):
        return False
    if int(left.graph_ids[int(left_row)].item()) != int(right.graph_ids[int(right_row)].item()):
        return False
    left_count = int(left.edge_count[int(left_row)].item())
    right_count = int(right.edge_count[int(right_row)].item())
    if left_count != right_count:
        return False
    return bool(left.edge_ids[int(left_row), :left_count].eq(right.edge_ids[int(right_row), :right_count]).all())


def _covered_node_pairs(
    *,
    state: StateBatch,
    graph: GraphContext,
) -> tuple[Tensor, Tensor]:
    if state.num_states == 0:
        empty = _empty_long(state.device)
        return empty, empty

    anchor_starts = graph.anchor_ptr.index_select(0, state.graph_ids)
    anchor_ends = graph.anchor_ptr.index_select(0, state.graph_ids + 1)
    anchor_counts = anchor_ends - anchor_starts
    anchor_total = int(anchor_counts.sum().item())

    pieces_row: list[Tensor] = []
    pieces_node: list[Tensor] = []
    if anchor_total > 0:
        anchor_rows = torch.repeat_interleave(
            torch.arange(state.num_states, device=state.device),
            anchor_counts,
            output_size=anchor_total,
        )
        anchor_offsets = _segment_arange(anchor_counts)
        anchor_positions = torch.repeat_interleave(
            anchor_starts,
            anchor_counts,
            output_size=anchor_total,
        ) + anchor_offsets
        pieces_row.append(anchor_rows)
        pieces_node.append(graph.anchor_node_ids.index_select(0, anchor_positions))

    activated = state.active_activated_node_index()
    if int(activated.node_ids.numel()) > 0:
        pieces_row.append(activated.row_ids)
        pieces_node.append(activated.node_ids)

    selected = state.selected_edge_index()
    if int(selected.edge_ids.numel()) > 0:
        src = graph.edge_src.index_select(0, selected.edge_ids)
        dst = graph.edge_dst.index_select(0, selected.edge_ids)
        pieces_row.extend((selected.row_ids, selected.row_ids))
        pieces_node.extend((src, dst))

    if not pieces_node:
        empty = _empty_long(state.device)
        return empty, empty

    row_ids = torch.cat(pieces_row, dim=0)
    node_ids = torch.cat(pieces_node, dim=0)
    keys = row_ids * int(graph.num_nodes) + node_ids
    order = torch.argsort(keys)
    sorted_keys = keys.index_select(0, order)
    keep = torch.ones(
        int(sorted_keys.numel()),
        dtype=torch.bool,
        device=state.device,
    )
    if int(sorted_keys.numel()) > 1:
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    unique_order = order[keep]
    return row_ids.index_select(0, unique_order), node_ids.index_select(0, unique_order)


def _active_node_id_matrix(
    *,
    state: StateBatch,
    graph_context: GraphContext,
) -> Tensor:
    max_anchor = int((graph_context.anchor_ptr[1:] - graph_context.anchor_ptr[:-1]).max().item()) if int(graph_context.num_graphs) > 0 else 0
    width = max_anchor + int(state.budget)
    if width == 0 or state.num_states == 0:
        return torch.empty((state.num_states, 0), dtype=torch.long, device=state.device)
    matrix = torch.full(
        (state.num_states, width),
        -1,
        dtype=torch.long,
        device=state.device,
    )

    anchor_starts = graph_context.anchor_ptr.index_select(0, state.graph_ids)
    anchor_ends = graph_context.anchor_ptr.index_select(0, state.graph_ids + 1)
    anchor_counts = anchor_ends - anchor_starts
    anchor_total = int(anchor_counts.sum().item())
    if anchor_total > 0:
        anchor_rows = torch.repeat_interleave(
            torch.arange(state.num_states, device=state.device),
            anchor_counts,
            output_size=anchor_total,
        )
        anchor_offsets = _segment_arange(anchor_counts)
        anchor_positions = torch.repeat_interleave(
            anchor_starts,
            anchor_counts,
            output_size=anchor_total,
        ) + anchor_offsets
        matrix[anchor_rows, anchor_offsets] = graph_context.anchor_node_ids.index_select(0, anchor_positions)

    if int(state.activated_node_count.numel()) > 0 and int(state.budget) > 0:
        valid = torch.arange(
            int(state.budget),
            dtype=torch.long,
            device=state.device,
        ).view(1, -1).lt(state.activated_node_count.view(-1, 1))
        if bool(valid.any()):
            rows, cols = valid.nonzero(as_tuple=True)
            base = anchor_counts.index_select(0, rows)
            matrix[rows, base + cols] = state.activated_node_ids[valid]
    return matrix


def _active_node_id_matrix_from_edges(
    *,
    state: StateBatch,
    edge_src: Tensor,
    edge_dst: Tensor,
) -> Tensor:
    width = 1 + int(state.budget) + int(state.budget) * 2
    if width == 0 or state.num_states == 0:
        return torch.empty((state.num_states, 0), dtype=torch.long, device=state.device)
    matrix = torch.full(
        (state.num_states, width),
        -1,
        dtype=torch.long,
        device=state.device,
    )
    matrix[:, 0] = 0

    activated = state.active_activated_node_index()
    if int(activated.node_ids.numel()) > 0:
        matrix[activated.row_ids, 1 + _within_row_positions(activated.row_ids)] = activated.node_ids

    selected = state.selected_edge_index()
    if int(selected.edge_ids.numel()) > 0:
        src = edge_src.to(device=state.device, dtype=torch.long).index_select(0, selected.edge_ids)
        dst = edge_dst.to(device=state.device, dtype=torch.long).index_select(0, selected.edge_ids)
        base = 1 + int(state.budget)
        pos = _within_row_positions(selected.row_ids) * 2
        matrix[selected.row_ids, base + pos] = src
        matrix[selected.row_ids, base + pos + 1] = dst
    return matrix


def _rows_have_node(active_nodes: Tensor, query_node_ids: Tensor) -> Tensor:
    if int(query_node_ids.numel()) == 0:
        return torch.empty(0, dtype=torch.bool, device=active_nodes.device)
    if int(active_nodes.numel()) == 0:
        return torch.zeros(int(query_node_ids.numel()), dtype=torch.bool, device=query_node_ids.device)
    return active_nodes.eq(query_node_ids.view(-1, 1)).any(dim=1)


def _rows_have_edge(selected_edge_ids: Tensor, selected_edge_count: Tensor, query_edge_ids: Tensor) -> Tensor:
    if int(query_edge_ids.numel()) == 0:
        return torch.empty(0, dtype=torch.bool, device=selected_edge_ids.device)
    if int(selected_edge_ids.size(1)) == 0:
        return torch.zeros(int(query_edge_ids.numel()), dtype=torch.bool, device=query_edge_ids.device)
    steps = torch.arange(
        int(selected_edge_ids.size(1)),
        dtype=torch.long,
        device=selected_edge_ids.device,
    ).view(1, -1)
    valid = steps.lt(selected_edge_count.view(-1, 1))
    return (selected_edge_ids.eq(query_edge_ids.view(-1, 1)) & valid).any(dim=1)


def _segment_arange(lengths: Tensor) -> Tensor:
    lengths = lengths.to(dtype=torch.long).view(-1)
    if int(lengths.numel()) == 0:
        return _empty_long(lengths.device)
    total = int(lengths.sum().item())
    starts = torch.cumsum(lengths, dim=0) - lengths
    return torch.arange(total, dtype=torch.long, device=lengths.device) - torch.repeat_interleave(
        starts,
        lengths,
        output_size=total,
    )


def _within_row_positions(row_ids: Tensor) -> Tensor:
    if int(row_ids.numel()) == 0:
        return _empty_long(row_ids.device)
    unique, counts = torch.unique_consecutive(row_ids, return_counts=True)
    del unique
    return _segment_arange(counts)


def _recompute_activated_node_ids(
    *,
    graph_ids: Tensor,
    selected_edge_ids: Tensor,
    selected_edge_count: Tensor,
    budget: int,
    graph_context: GraphContext,
) -> Tensor:
    out = torch.full(
        (int(graph_ids.numel()), int(budget)),
        -1,
        dtype=torch.long,
        device=graph_ids.device,
    )
    for row in range(int(graph_ids.numel())):
        graph_id = int(graph_ids[row].item())
        anchor_start = int(graph_context.anchor_ptr[graph_id].item())
        anchor_end = int(graph_context.anchor_ptr[graph_id + 1].item())
        anchors = graph_context.anchor_node_ids[anchor_start:anchor_end]
        count = int(selected_edge_count[row].item())
        if count <= 0:
            continue
        selected = selected_edge_ids[row, :count]
        dst = graph_context.edge_dst.index_select(0, selected)
        keep = ~dst.view(-1, 1).eq(anchors.view(1, -1)).any(dim=1) if int(anchors.numel()) > 0 else torch.ones_like(dst, dtype=torch.bool)
        kept = dst[keep]
        if int(kept.numel()) > 0:
            unique = torch.unique_consecutive(torch.sort(kept).values)
            take = min(int(unique.numel()), int(budget))
            out[row, :take] = unique[:take]
    return out


def _recompute_activated_node_count(
    *,
    graph_ids: Tensor,
    selected_edge_ids: Tensor,
    selected_edge_count: Tensor,
    budget: int,
    graph_context: GraphContext,
) -> Tensor:
    activated = _recompute_activated_node_ids(
        graph_ids=graph_ids,
        selected_edge_ids=selected_edge_ids,
        selected_edge_count=selected_edge_count,
        budget=budget,
        graph_context=graph_context,
    )
    return activated.ge(0).sum(dim=1, dtype=torch.long)


def _empty_long(device: torch.device) -> Tensor:
    return torch.empty(0, dtype=torch.long, device=device)


__all__ = [
    "ActionSpace",
    "EdgeSelection",
    "ExpansionBatch",
    "FrontierEncoding",
    "NodeSelection",
    "StateBatch",
    "canonicalize_state_batch",
    "cat_state_batches",
    "frontier_from_graph",
    "remove_selected_edge",
    "state_rows_equal",
]
