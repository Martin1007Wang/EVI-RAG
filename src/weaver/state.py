from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch

from src.weaver.context import GraphContext

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ExpansionBatch:
    """
    Chosen EXPAND actions.

    Contract:
    - state_ids[k] is the local parent-state id inside a StateBatch.
    - edge_ids[k] is a physical KG edge id.
    - STOP actions are not represented here.
    - Multiple rows may reference the same parent state; this supports branching.
    """

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
    """
    Legal actions derived from StateBatch and GraphContext.

    STOP:
    - STOP is legal for every state.
    - STOP is represented implicitly by state id, not by a fake edge id.

    EXPAND:
    - expand_state_ids[k] is the parent state id.
    - expand_edge_ids[k] is the legal physical KG edge id.
    - expand_ptr[s]:expand_ptr[s + 1] gives the contiguous expansion slice
      for state s.
    - expansion actions exclude already selected edges.
    - expansion actions only exist for states with remaining budget.

    Invariant:
    - expand_state_ids is grouped by state id.
    - expand_ptr has shape [num_states + 1].
    """

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
    def stop_state_ids(self) -> Tensor:
        return torch.arange(
            int(self.num_states),
            dtype=torch.long,
            device=self.device,
        )

    @property
    def expand_count(self) -> Tensor:
        return self.expand_ptr[1:] - self.expand_ptr[:-1]

    @classmethod
    def empty(
        cls,
        *,
        num_states: int,
        device: torch.device,
    ) -> ActionSpace:
        return cls(
            num_states=int(num_states),
            expand_state_ids=_empty_long(device),
            expand_edge_ids=_empty_long(device),
            expand_ptr=torch.zeros(
                int(num_states) + 1,
                dtype=torch.long,
                device=device,
            ),
        )


@dataclass(frozen=True, slots=True)
class StateBatch:
    """
    Batched canonical evidence-subgraph states.

    Truth source:
    - graph_ids[s] gives the graph id of state s.
    - edge_ids[s, :edge_count[s]] gives the selected physical KG edge set.
    - selected edge ids are stored sorted so construction order is not state.
    - edge_ids is the only selected-edge truth source.

    Derived views:
    - covered_node_pairs(graph): anchors plus endpoints of selected edges.
    - action_space(graph): STOP plus legal EXPAND actions.

    STOP is an action, not an edge, and is never written into edge_ids.

    Expansion semantics:
    - Current implementation enumerates outgoing edges from covered nodes.
    - If the intended theory is incident-edge expansion, GraphContext must expose
      incoming CSR too; do not hide that semantic change here.
    """

    graph_ids: Tensor  # [S] 当前 state s 属于哪个原始的 KGQA 子图
    edge_ids: Tensor  # [S, B], sorted selected physical edges padded with -1
    edge_count: Tensor  # [S] 当前 state s 已经选择了多少条边
    budget: int  # 所有 state 最大可选择的边的数量

    @classmethod
    def initial(
        cls,
        *,
        graph_ids: Tensor,
        budget: int,
    ) -> StateBatch:
        graph_ids = graph_ids.to(dtype=torch.long).view(-1)
        budget = int(budget)

        if budget < 0:
            raise ValueError("budget must be nonnegative.")

        num_states = int(graph_ids.numel())
        device = graph_ids.device

        return cls(
            graph_ids=graph_ids,
            edge_ids=torch.full(
                (num_states, budget),
                -1,
                dtype=torch.long,
                device=device,
            ),
            edge_count=torch.zeros(
                num_states,
                dtype=torch.long,
                device=device,
            ),
            budget=budget,
        )

    @property
    def device(self) -> torch.device:
        return self.graph_ids.device

    @property
    def num_states(self) -> int:
        return int(self.graph_ids.numel())

    @property
    def budget_left(self) -> Tensor:
        return int(self.budget) - self.edge_count

    def take(self, state_ids: Tensor) -> StateBatch:
        """
        Select prefix states by local state ids.

        Use this for branching, terminal extraction, candidate pruning, or
        evaluation selection.
        """

        state_ids = state_ids.to(
            device=self.device,
            dtype=torch.long,
        ).view(-1)

        return StateBatch(
            graph_ids=self.graph_ids.index_select(0, state_ids),
            edge_ids=self.edge_ids.index_select(0, state_ids),
            edge_count=self.edge_count.index_select(0, state_ids),
            budget=int(self.budget),
        )

    def branch(self, expansion: ExpansionBatch) -> StateBatch:
        """
        Apply chosen expansion actions and return child states.

        This is not an in-place row update.

        If expansion has K actions, the returned StateBatch has K child states.
        Multiple expansion actions may branch from the same parent state.
        """

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

        next_graph_ids = self.graph_ids.index_select(0, parent_state_ids)
        next_edge_ids = self.edge_ids.index_select(0, parent_state_ids).clone()
        next_edge_count = self.edge_count.index_select(0, parent_state_ids).clone()

        child_ids = torch.arange(
            int(parent_state_ids.numel()),
            dtype=torch.long,
            device=self.device,
        )

        if bool(next_edge_ids.eq(new_edge_ids[:, None]).any(dim=1).any()):
            raise ValueError("branch() received an already selected edge.")

        next_edge_ids[child_ids, next_edge_count] = new_edge_ids
        next_edge_count = next_edge_count + 1
        next_edge_ids = _canonicalize_edge_ids(
            edge_ids=next_edge_ids,
            edge_count=next_edge_count,
        )

        return StateBatch(
            graph_ids=next_graph_ids,
            edge_ids=next_edge_ids,
            edge_count=next_edge_count,
            budget=int(self.budget),
        )

    def advance(self, expansion: ExpansionBatch) -> StateBatch:
        """
        Advance selected state rows by one chosen expansion edge.

        Contract:
        - returned StateBatch has the same number of states as self.
        - expansion.state_ids are local row ids inside this StateBatch.
        - each row may appear at most once.
        - each row must have remaining budget.
        - expansion.edge_ids should come from this state's ActionSpace.
        """

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

        # Rollout advance is fixed-row update. Repeated rows would make the update ambiguous.
        if int(torch.unique(rows).numel()) != int(rows.numel()):
            raise ValueError("advance() requires each state row to appear at most once.")

        if bool(self.budget_left.index_select(0, rows).le(0).any()):
            raise ValueError("advance() received rows with no remaining budget.")

        next_edge_ids = self.edge_ids.clone()
        next_edge_count = self.edge_count.clone()

        pos = next_edge_count.index_select(0, rows)
        next_edge_ids[rows, pos] = new_edge_ids
        next_edge_count[rows] = next_edge_count[rows] + 1
        next_edge_ids = _canonicalize_edge_ids(
            edge_ids=next_edge_ids,
            edge_count=next_edge_count,
        )

        return StateBatch(
            graph_ids=self.graph_ids,
            edge_ids=next_edge_ids,
            edge_count=next_edge_count,
            budget=int(self.budget),
        )

    def covered_node_pairs(self, graph: GraphContext) -> tuple[Tensor, Tensor]:
        """
        Return unique (state_id, node_id) pairs covered by each prefix state.

        Covered nodes are:
        - graph anchors;
        - sources of selected edges;
        - destinations of selected edges.

        This is a derived view, not a stored state field.
        """

        return _covered_node_pairs(state=self, graph=graph)

    def action_space(self, graph: GraphContext) -> ActionSpace:
        """
        Derive STOP plus legal EXPAND actions from this state batch.

        Expansion frontier:

            covered nodes
            -> outgoing physical KG edges
            -> same parent graph
            -> parent has remaining budget
            -> edge not already selected by that parent state
        """

        if self.num_states == 0:
            return ActionSpace.empty(
                num_states=0,
                device=self.device,
            )

        covered_state_ids, covered_node_ids = _covered_node_pairs(
            state=self,
            graph=graph,
        )

        expand_state_ids, expand_edge_ids = _outgoing_edges_from_nodes(
            state_ids=covered_state_ids,
            node_ids=covered_node_ids,
            ptr=graph.adjacency.out_ptr,
            edge_ids_by_src=graph.adjacency.edge_ids_by_src,
        )

        expand_state_ids, expand_edge_ids = _filter_legal_expansions(
            state=self,
            graph=graph,
            state_ids=expand_state_ids,
            edge_ids=expand_edge_ids,
        )

        expand_ptr = _grouped_state_ids_to_ptr(
            state_ids=expand_state_ids,
            num_states=self.num_states,
            device=self.device,
        )

        return ActionSpace(
            num_states=self.num_states,
            expand_state_ids=expand_state_ids,
            expand_edge_ids=expand_edge_ids,
            expand_ptr=expand_ptr,
        )


def cat_state_batches(states: Sequence[StateBatch]) -> StateBatch:
    """
    Concatenate StateBatch objects.

    This is a batch-boundary operation, not a StateBatch method.
    """

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
        edge_ids=torch.cat([state.edge_ids for state in states], dim=0),
        edge_count=torch.cat([state.edge_count for state in states], dim=0),
        budget=int(first.budget),
    )


def canonicalize_state_batch(state: StateBatch) -> StateBatch:
    """
    Return a state batch with sorted selected edges and clean -1 padding.

    This is useful at boundaries that wrap ordered trajectory records as states.
    State construction itself remains a plain dataclass operation.
    """

    return StateBatch(
        graph_ids=state.graph_ids,
        edge_ids=_canonicalize_edge_ids(
            edge_ids=state.edge_ids,
            edge_count=state.edge_count,
        ),
        edge_count=state.edge_count,
        budget=int(state.budget),
    )


def remove_selected_edge(
    *,
    state: StateBatch,
    row: int,
    edge_id: int,
) -> StateBatch:
    """
    Return a one-row canonical state with ``edge_id`` removed from ``row``.
    """

    row = int(row)
    edge_id = int(edge_id)
    count = int(state.edge_count[row].item())
    selected = state.edge_ids[row, :count]
    keep = selected.ne(int(edge_id))
    if int(keep.sum().item()) != count - 1:
        raise ValueError("edge_id must appear exactly once in the selected edge set.")

    edge_ids = torch.full(
        (1, int(state.budget)),
        -1,
        dtype=torch.long,
        device=state.device,
    )
    remaining = selected[keep]
    if int(remaining.numel()) > 0:
        edge_ids[0, : int(remaining.numel())] = torch.sort(remaining).values

    return StateBatch(
        graph_ids=state.graph_ids[row : row + 1],
        edge_ids=edge_ids,
        edge_count=torch.tensor(
            [int(remaining.numel())],
            dtype=torch.long,
            device=state.device,
        ),
        budget=int(state.budget),
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
    if int(left.edge_count[int(left_row)].item()) != int(right.edge_count[int(right_row)].item()):
        return False
    count = int(left.edge_count[int(left_row)].item())
    if count == 0:
        return True
    left_edges = _selected_edges_for_row(left, int(left_row))
    right_edges = _selected_edges_for_row(right, int(right_row))
    return bool(left_edges.eq(right_edges).all())


def _covered_node_pairs(
    *,
    state: StateBatch,
    graph: GraphContext,
) -> tuple[Tensor, Tensor]:
    anchor_state_ids, anchor_node_ids = _anchor_node_pairs(
        state=state,
        graph=graph,
    )

    selected_state_ids, selected_edge_ids = _selected_edge_pairs(state)

    if int(selected_edge_ids.numel()) == 0:
        return _unique_sorted_pairs(
            left_ids=anchor_state_ids,
            right_ids=anchor_node_ids,
            right_size=int(graph.num_nodes),
        )

    selected_src_ids = graph.edge_src.index_select(0, selected_edge_ids)
    selected_dst_ids = graph.edge_dst.index_select(0, selected_edge_ids)

    state_ids = torch.cat(
        (
            anchor_state_ids,
            selected_state_ids,
            selected_state_ids,
        ),
        dim=0,
    )

    node_ids = torch.cat(
        (
            anchor_node_ids,
            selected_src_ids,
            selected_dst_ids,
        ),
        dim=0,
    )

    return _unique_sorted_pairs(
        left_ids=state_ids,
        right_ids=node_ids,
        right_size=int(graph.num_nodes),
    )


def _anchor_node_pairs(
    *,
    state: StateBatch,
    graph: GraphContext,
) -> tuple[Tensor, Tensor]:
    starts = graph.anchor_ptr.index_select(0, state.graph_ids)
    ends = graph.anchor_ptr.index_select(0, state.graph_ids + 1)
    counts = ends - starts
    total = counts.sum()

    state_ids = _repeat_interleave(
        torch.arange(
            state.num_states,
            dtype=torch.long,
            device=state.device,
        ),
        counts,
        output_size=total,
    )

    positions = _repeat_interleave(
        starts,
        counts,
        output_size=total,
    ) + _segment_arange(counts)

    node_ids = graph.anchor_node_ids.index_select(0, positions)

    return state_ids, node_ids


def _selected_edge_pairs(state: StateBatch) -> tuple[Tensor, Tensor]:
    """
    Return padded-free (state_id, selected_edge_id) pairs.

    edge_count is the semantic selected-edge count.
    """

    num_states, budget = state.edge_ids.shape

    if int(num_states) == 0 or int(budget) == 0:
        empty = _empty_long(state.device)
        return empty, empty

    steps = torch.arange(
        int(budget),
        dtype=torch.long,
        device=state.device,
    ).view(1, int(budget))
    valid = steps.lt(state.edge_count.view(int(num_states), 1))

    state_ids = torch.repeat_interleave(
        torch.arange(
            int(num_states),
            dtype=torch.long,
            device=state.device,
        ),
        state.edge_count.to(dtype=torch.long),
    )

    return state_ids, state.edge_ids[valid]


def _outgoing_edges_from_nodes(
    *,
    state_ids: Tensor,
    node_ids: Tensor,
    ptr: Tensor,
    edge_ids_by_src: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Expand (state_id, node_id) pairs into outgoing (state_id, edge_id) pairs.

    CSR contract:
    - ptr[node]:ptr[node + 1] gives the slice inside edge_ids_by_src.
    """

    if int(node_ids.numel()) == 0:
        device = state_ids.device
        empty = _empty_long(device)
        return empty, empty

    starts = ptr.index_select(0, node_ids)
    ends = ptr.index_select(0, node_ids + 1)
    degrees = ends - starts
    total = degrees.sum()

    out_state_ids = _repeat_interleave(
        state_ids,
        degrees,
        output_size=total,
    )

    positions = _repeat_interleave(
        starts,
        degrees,
        output_size=total,
    ) + _segment_arange(degrees)

    out_edge_ids = edge_ids_by_src.index_select(0, positions)

    return out_state_ids, out_edge_ids


def _filter_legal_expansions(
    *,
    state: StateBatch,
    graph: GraphContext,
    state_ids: Tensor,
    edge_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Filter candidate expansion edges.

    Conditions:
    - edge belongs to the same graph as the parent state;
    - parent state has remaining budget;
    - edge is not already selected by that parent state.

    Selected-edge exclusion is O(F * B).
    This is deliberate: B is the expansion budget and should stay small.
    """

    if int(edge_ids.numel()) == 0:
        return state_ids, edge_ids

    parent_graph_ids = state.graph_ids.index_select(0, state_ids)
    edge_graph_ids = graph.edge_to_graph.index_select(0, edge_ids)

    same_graph = edge_graph_ids.eq(parent_graph_ids)
    has_budget = state.budget_left.gt(0).index_select(0, state_ids)

    selected_by_parent = state.edge_ids.index_select(0, state_ids)
    not_selected = selected_by_parent.ne(edge_ids[:, None]).all(dim=1)

    keep = same_graph & has_budget & not_selected

    return state_ids[keep], edge_ids[keep]


def _canonicalize_edge_ids(
    *,
    edge_ids: Tensor,
    edge_count: Tensor,
) -> Tensor:
    edge_ids = edge_ids.to(dtype=torch.long)
    edge_count = edge_count.to(dtype=torch.long).view(-1)

    if edge_ids.ndim != 2:
        raise ValueError(f"edge_ids must have shape [S, B], got {tuple(edge_ids.shape)}.")
    if int(edge_ids.size(0)) != int(edge_count.numel()):
        raise ValueError("edge_count must have one item per state row.")

    out = edge_ids.new_full(edge_ids.shape, -1)
    budget = int(edge_ids.size(1))
    if budget == 0:
        return out

    for row in range(int(edge_ids.size(0))):
        count = int(edge_count[row].item())
        if count < 0 or count > budget:
            raise ValueError("edge_count must be in [0, budget].")
        if count == 0:
            continue

        selected = edge_ids[row, :count]
        if bool(selected.lt(0).any()):
            raise ValueError("selected edge ids must be nonnegative.")
        unique = torch.unique(selected)
        if int(unique.numel()) != count:
            raise ValueError("canonical states cannot contain duplicate selected edges.")
        out[row, :count] = torch.sort(selected).values

    return out


def _selected_edges_for_row(
    state: StateBatch,
    row: int,
) -> Tensor:
    row = int(row)
    count = int(state.edge_count[row].item())
    if count <= 0:
        return _empty_long(state.device)
    return torch.sort(state.edge_ids[row, :count].to(dtype=torch.long)).values


def _is_legal_state_row(
    *,
    state: StateBatch,
    graph: GraphContext,
    row: int,
) -> bool:
    row = int(row)
    graph_id = int(state.graph_ids[row].item())
    selected = _selected_edges_for_row(state, row)

    if int(selected.numel()) == 0:
        return True

    if int(torch.unique(selected).numel()) != int(selected.numel()):
        return False

    edge_graph = graph.edge_to_graph.index_select(0, selected)
    if not bool(edge_graph.eq(graph_id).all()):
        return False

    anchor_start = int(graph.anchor_ptr[graph_id].item())
    anchor_end = int(graph.anchor_ptr[graph_id + 1].item())
    reachable: set[int] = {int(node_id) for node_id in graph.anchor_node_ids[anchor_start:anchor_end].detach().cpu().tolist()}

    remaining = set(int(edge_id) for edge_id in selected.detach().cpu().tolist())
    src = graph.edge_src.detach().cpu()
    dst = graph.edge_dst.detach().cpu()

    changed = True
    while changed and remaining:
        changed = False
        for edge_id in list(remaining):
            if int(src[edge_id].item()) in reachable:
                reachable.add(int(dst[edge_id].item()))
                remaining.remove(edge_id)
                changed = True

    return not remaining


def _unique_sorted_pairs(
    *,
    left_ids: Tensor,
    right_ids: Tensor,
    right_size: int,
) -> tuple[Tensor, Tensor]:
    """
    Unique pair compression.

    Encodes each pair as:

        key = left_id * right_size + right_id

    This is appropriate for covered-node pairs:
    O(S * (num_anchors + 2 * budget)).
    Do not use this for large frontier deduplication unless measured.
    """

    if int(left_ids.numel()) == 0:
        device = left_ids.device
        empty = _empty_long(device)
        return empty, empty

    keys = left_ids.to(dtype=torch.long) * int(right_size) + right_ids.to(dtype=torch.long)
    keys = torch.unique(keys, sorted=True)

    return (
        torch.div(keys, int(right_size), rounding_mode="floor"),
        keys.remainder(int(right_size)),
    )


def _grouped_state_ids_to_ptr(
    *,
    state_ids: Tensor,
    num_states: int,
    device: torch.device,
) -> Tensor:
    """
    Convert grouped expansion state ids to CSR-style ptr.

    state_ids is expected to be grouped by state id. The ptr remains valid when
    some states have zero legal expansion actions.
    """

    counts = torch.bincount(
        state_ids,
        minlength=int(num_states),
    ).to(dtype=torch.long)

    ptr = torch.empty(
        int(num_states) + 1,
        dtype=torch.long,
        device=device,
    )

    ptr[0] = 0
    ptr[1:] = torch.cumsum(counts, dim=0)

    return ptr


def _segment_arange(lengths: Tensor) -> Tensor:
    """
    For lengths [l0, l1, ...], return:

        [0, ..., l0 - 1, 0, ..., l1 - 1, ...]

    Used to expand CSR segments without Python loops.
    """

    lengths = lengths.to(dtype=torch.long).view(-1)

    if int(lengths.numel()) == 0:
        return _empty_long(lengths.device)

    total = lengths.sum()
    starts = torch.cumsum(lengths, dim=0) - lengths

    return torch.arange(
        _torch_scalar_as_int(total),
        dtype=torch.long,
        device=lengths.device,
    ) - _repeat_interleave(
        starts,
        lengths,
        output_size=total,
    )


def _repeat_interleave(
    values: Tensor,
    repeats: Tensor,
    *,
    output_size: Tensor,
) -> Tensor:
    """
    Type-checker-safe wrapper around torch.repeat_interleave.

    Runtime PyTorch accepts a 0-d integral Tensor for output_size.
    Pylance's overload usually requires int. The cast is intentionally local:
    it avoids scattering `# type: ignore` or `.item()` across the code.

    Do not replace this with int(output_size.item()) in hot CUDA paths unless
    you accept the synchronization.
    """

    return torch.repeat_interleave(
        values,
        repeats.to(dtype=torch.long),
        output_size=_torch_scalar_as_int(output_size),
    )


def _torch_scalar_as_int(value: Tensor) -> int:
    """
    Static typing adapter.

    This is a no-op at runtime. It exists because PyTorch accepts 0-d scalar
    tensors in places where its public typing stubs often only declare int.
    """

    return cast(int, value)


def _check_1d_long(value: Tensor, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)}.")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long.")


def _check_2d_long(value: Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [N, M], got {tuple(value.shape)}.")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long.")


def _empty_long(device: torch.device) -> Tensor:
    return torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )


__all__ = [
    "ActionSpace",
    "ExpansionBatch",
    "StateBatch",
    "canonicalize_state_batch",
    "cat_state_batches",
    "remove_selected_edge",
    "state_rows_equal",
]
