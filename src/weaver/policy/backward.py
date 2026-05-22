from __future__ import annotations

import torch
from torch import nn

from ..context import GraphContext
from ..state import State


class BackwardPolicy(nn.Module):
    """
    Backward kernel P_B for ordered subtrajectory supervision.

    It is not the rollout policy.
    It is not the inference policy.
    It should not read rewards.
    """

    def log_prob(
        self,
        *,
        child_state: State,
        context: GraphContext,
        action_edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class UniformValidPredecessorBackwardPolicy(BackwardPolicy):
    """
    Uniform backward policy over exact forward predecessors.

    For child state:

        z' = S'

    A removable edge e in S' is a valid predecessor action iff:

        S = S' \\ {e}

    satisfies:

        1. S is forward-reachable under the current frontier semantics.
        2. e is in Frontier(S).

    Then:

        P_B(S | S') = 1 / |Pred(S')|
    """

    def log_prob(
        self,
        *,
        child_state: State,
        context: GraphContext,
        action_edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = action_edge_ids.new_zeros(action_edge_ids.numel()).float()
        expand = action_edge_ids.ge(0)
        if not bool(expand.any()):
            return out

        counts = valid_predecessor_count(
            state=child_state,
            context=context,
        ).float()
        if bool(counts[expand].le(0).any()):
            raise ValueError("Expansion child state has no exact forward predecessor.")

        expanded_rows = expand.nonzero(as_tuple=False).flatten()
        expanded_edge_ids = action_edge_ids[expand]
        for local_pos, row in enumerate(expanded_rows.tolist()):
            selected = child_state.edge_mask[row].nonzero(as_tuple=True)[0]
            edge_id = int(expanded_edge_ids[local_pos].item())
            parent_edges = _parent_edge_ids_after_removal(
                selected_edge_ids=selected,
                removed_edge_id=edge_id,
            )
            if not is_valid_predecessor_edge(
                context=context,
                graph_id=int(child_state.row_to_graph[row].item()),
                parent_edge_ids=parent_edges,
                removed_edge_id=edge_id,
            ):
                raise ValueError(
                    f"Edge {edge_id} is not an exact forward predecessor for child row {row}."
                )

        out[expand] = -counts[expand].log()
        return out


def valid_predecessor_count(
    *,
    state: State,
    context: GraphContext,
) -> torch.Tensor:
    counts = torch.zeros(
        state.num_rows,
        dtype=torch.long,
        device=state.device,
    )

    for row in range(state.num_rows):
        selected = state.edge_mask[row].nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            continue

        count = 0
        for pos in range(int(selected.numel())):
            edge_id = int(selected[pos].item())
            parent_edges = torch.cat(
                [
                    selected[:pos],
                    selected[pos + 1 :],
                ],
                dim=0,
            )
            if is_valid_predecessor_edge(
                context=context,
                graph_id=int(state.row_to_graph[row].item()),
                parent_edge_ids=parent_edges,
                removed_edge_id=edge_id,
            ):
                count += 1

        counts[row] = count

    return counts


def is_valid_predecessor_edge(
    *,
    context: GraphContext,
    graph_id: int,
    parent_edge_ids: torch.Tensor,
    removed_edge_id: int,
) -> bool:
    if removed_edge_id < 0 or removed_edge_id >= int(context.num_edges):
        return False

    removed_edge_tensor = torch.tensor(
        [removed_edge_id],
        dtype=torch.long,
        device=context.device,
    )
    removed_edge_graph = context.edge_to_graph.index_select(0, removed_edge_tensor)
    if not bool(removed_edge_graph.eq(int(graph_id)).all()):
        return False

    parent_state = exact_forward_parent_state(
        context=context,
        graph_id=graph_id,
        edge_ids=parent_edge_ids,
    )
    if parent_state is None:
        return False

    frontier = parent_state.frontier(
        context,
        expand_budget=None,
    )
    return bool(frontier.edge_ids.eq(int(removed_edge_id)).any())


def exact_forward_parent_state(
    *,
    context: GraphContext,
    graph_id: int,
    edge_ids: torch.Tensor,
) -> State | None:
    edge_ids = edge_ids.to(
        device=context.device,
        dtype=torch.long,
    ).view(-1)
    if edge_ids.numel() == 0:
        return State.initial(
            graph=context,
            graph_ids=torch.tensor(
                [graph_id],
                dtype=torch.long,
                device=context.device,
            ),
        )

    if int(torch.unique(edge_ids).numel()) != int(edge_ids.numel()):
        return None

    edge_graph = context.edge_to_graph.index_select(0, edge_ids)
    if not bool(edge_graph.eq(int(graph_id)).all()):
        return None

    budget = int(edge_ids.numel())
    state = State.initial(
        graph=context,
        graph_ids=torch.tensor(
            [graph_id],
            dtype=torch.long,
            device=context.device,
        ),
    )
    row = torch.zeros(1, dtype=torch.long, device=context.device)
    remaining = edge_ids

    while remaining.numel() > 0:
        frontier = state.frontier(
            context,
            expand_budget=budget,
        )
        frontier_mask = _membership_mask(
            query_ids=frontier.edge_ids,
            candidate_ids=remaining,
        )
        if not bool(frontier_mask.any()):
            return None

        next_edge_id = frontier.edge_ids[frontier_mask][:1]
        state = state.expand(
            graph=context,
            rows=row,
            edge_ids=next_edge_id,
            expand_budget=budget,
        )
        remaining = remaining[remaining.ne(next_edge_id.view(()))]

    return state


def _parent_edge_ids_after_removal(
    *,
    selected_edge_ids: torch.Tensor,
    removed_edge_id: int,
) -> torch.Tensor:
    keep = selected_edge_ids.ne(int(removed_edge_id))
    if int(keep.sum().item()) == int(selected_edge_ids.numel()):
        raise ValueError(f"Removed edge {removed_edge_id} is not selected in the child state.")
    return selected_edge_ids[keep]


def _membership_mask(
    *,
    query_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    query_ids = query_ids.view(-1)
    candidate_ids = candidate_ids.view(-1)
    if query_ids.numel() == 0 or candidate_ids.numel() == 0:
        return torch.zeros(
            query_ids.numel(),
            dtype=torch.bool,
            device=query_ids.device,
        )

    sorted_candidates = torch.sort(candidate_ids).values
    positions = torch.searchsorted(sorted_candidates, query_ids)
    in_bounds = positions.lt(sorted_candidates.numel())
    matched = torch.zeros(
        query_ids.numel(),
        dtype=torch.bool,
        device=query_ids.device,
    )
    if bool(in_bounds.any()):
        matched[in_bounds] = sorted_candidates.index_select(
            0,
            positions[in_bounds],
        ).eq(query_ids[in_bounds])
    return matched


__all__ = [
    "BackwardPolicy",
    "UniformValidPredecessorBackwardPolicy",
    "valid_predecessor_count",
]
