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
    Uniform backward policy over valid predecessor edges.

    For child state:

        z' = S'

    A removable edge e in S' is a valid predecessor action iff:

        S = S' \\ {e}

    satisfies:

        1. S is still anchor-connected.
        2. e touches the active node set of S.

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

        counts = valid_predecessor_count(
            state=child_state,
            context=context,
        ).float()
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
    if not is_anchor_connected_edge_set(
        context=context,
        graph_id=graph_id,
        edge_ids=parent_edge_ids,
    ):
        return False

    active = active_nodes_from_edge_set(
        context=context,
        graph_id=graph_id,
        edge_ids=parent_edge_ids,
    )

    src = int(context.edge_index[0, removed_edge_id].item())
    dst = int(context.edge_index[1, removed_edge_id].item())
    return src in active or dst in active


def active_nodes_from_edge_set(
    *,
    context: GraphContext,
    graph_id: int,
    edge_ids: torch.Tensor,
) -> set[int]:
    active = set(anchor_nodes_for_graph(
        context=context,
        graph_id=graph_id,
    ))

    for edge_id in edge_ids.tolist():
        active.add(int(context.edge_index[0, edge_id].item()))
        active.add(int(context.edge_index[1, edge_id].item()))

    return active


def is_anchor_connected_edge_set(
    *,
    context: GraphContext,
    graph_id: int,
    edge_ids: torch.Tensor,
) -> bool:
    if edge_ids.numel() == 0:
        return True

    anchors = anchor_nodes_for_graph(
        context=context,
        graph_id=graph_id,
    )
    adjacency: dict[int, set[int]] = {}

    for edge_id in edge_ids.tolist():
        src = int(context.edge_index[0, edge_id].item())
        dst = int(context.edge_index[1, edge_id].item())
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    frontier = list(anchors)
    seen = set(anchors)

    while frontier:
        node = frontier.pop()
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)

    endpoints: set[int] = set()
    for edge_id in edge_ids.tolist():
        endpoints.add(int(context.edge_index[0, edge_id].item()))
        endpoints.add(int(context.edge_index[1, edge_id].item()))

    return endpoints.issubset(seen)


def anchor_nodes_for_graph(
    *,
    context: GraphContext,
    graph_id: int,
) -> set[int]:
    anchors = context.anchor_mask.nonzero(as_tuple=True)[0]
    anchor_graph = context.node_to_graph.index_select(0, anchors)
    keep = anchor_graph.eq(graph_id)
    return set(anchors[keep].tolist())


__all__ = [
    "BackwardPolicy",
    "UniformValidPredecessorBackwardPolicy",
    "valid_predecessor_count",
]
