from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch

unreachable_distance = -1
node_target_unreachable_distance = 1_000_000_000


@dataclass(frozen=True, slots=True)
class PathLabels:
    reachable_target_node_ids: torch.Tensor
    node_target_distance: torch.Tensor
    node_target_shortest_path_edge_mask_flat: torch.Tensor


@dataclass(frozen=True, slots=True)
class TargetPathLabels:
    target_node_ids: torch.Tensor
    node_target_distances_flat: torch.Tensor
    node_target_shortest_path_edge_mask_flat: torch.Tensor


@dataclass(frozen=True, slots=True)
class AnchorPathLabels:
    anchor_node_forward_distances_flat: torch.Tensor
    anchor_node_backward_distances_flat: torch.Tensor


@dataclass(frozen=True, slots=True)
class _Graph:
    src: list[int]
    dst: list[int]
    src_tensor: torch.Tensor
    dst_tensor: torch.Tensor
    adjacency: list[list[int]]
    reverse_adjacency: list[list[int]]

    @property
    def num_edges(self) -> int:
        return len(self.src)


def compute_path_labels(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    target_node_ids: torch.Tensor,
    num_nodes: int,
) -> PathLabels:
    graph = _graph_from_edge_index(edge_index=edge_index, num_nodes=num_nodes)
    anchors = _valid_unique_nodes(anchor_node_ids, num_nodes=num_nodes)
    targets = _valid_unique_nodes(target_node_ids, num_nodes=num_nodes)

    anchor_distances = None
    if num_nodes > 0 and anchors and targets:
        anchor_distances = [_bfs(graph.adjacency, anchor) for anchor in anchors]

    target = _target_path_labels(
        graph=graph,
        anchors=anchors,
        targets=targets,
        num_nodes=num_nodes,
        anchor_distances=anchor_distances,
    )

    num_targets = int(target.target_node_ids.numel())
    return PathLabels(
        reachable_target_node_ids=target.target_node_ids,
        node_target_distance=_nearest_target_distance(
            node_target_distances_flat=target.node_target_distances_flat,
            num_targets=num_targets,
            num_nodes=num_nodes,
        ),
        node_target_shortest_path_edge_mask_flat=target.node_target_shortest_path_edge_mask_flat,
    )


def compute_target_path_labels(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    target_node_ids: torch.Tensor,
    num_nodes: int,
) -> TargetPathLabels:
    graph = _graph_from_edge_index(edge_index=edge_index, num_nodes=num_nodes)
    return _target_path_labels(
        graph=graph,
        anchors=_valid_unique_nodes(anchor_node_ids, num_nodes=num_nodes),
        targets=_valid_unique_nodes(target_node_ids, num_nodes=num_nodes),
        num_nodes=num_nodes,
    )


def compute_anchor_path_labels(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    num_nodes: int,
) -> AnchorPathLabels:
    graph = _graph_from_edge_index(edge_index=edge_index, num_nodes=num_nodes)
    return _anchor_path_labels(
        graph=graph,
        anchors=_valid_unique_nodes(anchor_node_ids, num_nodes=num_nodes),
        num_nodes=num_nodes,
    )


def _target_path_labels(
    *,
    graph: _Graph,
    anchors: Sequence[int],
    targets: Sequence[int],
    num_nodes: int,
    anchor_distances: Sequence[Sequence[int]] | None = None,
) -> TargetPathLabels:
    if num_nodes <= 0 or not anchors or not targets:
        return _empty_target_labels()

    anchor_distances = anchor_distances or [
        _bfs(graph.adjacency, anchor) for anchor in anchors
    ]
    reachable_targets = [
        target
        for target in targets
        if any(dist[target] != unreachable_distance for dist in anchor_distances)
    ]
    if not reachable_targets:
        return _empty_target_labels()

    target_distances = torch.tensor(
        [_bfs(graph.reverse_adjacency, target) for target in reachable_targets],
        dtype=torch.long,
    )
    edge_mask = _shortest_path_edge_mask(
        graph=graph,
        anchor_distances=anchor_distances,
        targets=reachable_targets,
        target_distances=target_distances,
    )

    return TargetPathLabels(
        target_node_ids=torch.tensor(reachable_targets, dtype=torch.long),
        node_target_distances_flat=target_distances.reshape(-1).contiguous(),
        node_target_shortest_path_edge_mask_flat=edge_mask.reshape(-1).contiguous(),
    )


def _anchor_path_labels(
    *,
    graph: _Graph,
    anchors: Sequence[int],
    num_nodes: int,
    anchor_distances: Sequence[Sequence[int]] | None = None,
) -> AnchorPathLabels:
    if num_nodes <= 0:
        return _empty_anchor_labels()
    if not anchors:
        unreachable = torch.full((num_nodes,), unreachable_distance, dtype=torch.long)
        return AnchorPathLabels(unreachable, unreachable.clone())

    forward = (
        _min_distances(anchor_distances, num_nodes=num_nodes)
        if anchor_distances is not None
        else _multi_source_min_dist(graph.adjacency, anchors)
    )
    backward = _multi_source_min_dist(graph.reverse_adjacency, anchors)
    return AnchorPathLabels(
        anchor_node_forward_distances_flat=forward.contiguous(),
        anchor_node_backward_distances_flat=backward.contiguous(),
    )


def _graph_from_edge_index(*, edge_index: torch.Tensor, num_nodes: int) -> _Graph:
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}.")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
        )
    if edge_index.numel() > 0:
        min_id = int(edge_index.min())
        max_id = int(edge_index.max())
        if min_id < 0 or max_id >= num_nodes:
            raise ValueError(
                f"edge_index contains node id outside [0, {num_nodes}): "
                f"min={min_id}, max={max_id}."
            )

    edge_index = edge_index.to(dtype=torch.long, device="cpu").contiguous()
    src_tensor = edge_index[0]
    dst_tensor = edge_index[1]
    src = [int(x) for x in src_tensor.tolist()]
    dst = [int(x) for x in dst_tensor.tolist()]

    adjacency = [[] for _ in range(num_nodes)]
    reverse_adjacency = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adjacency[u].append(v)
        reverse_adjacency[v].append(u)

    return _Graph(
        src=src,
        dst=dst,
        src_tensor=src_tensor,
        dst_tensor=dst_tensor,
        adjacency=adjacency,
        reverse_adjacency=reverse_adjacency,
    )


def _shortest_path_edge_mask(
    *,
    graph: _Graph,
    anchor_distances: Sequence[Sequence[int]],
    targets: Sequence[int],
    target_distances: torch.Tensor,
) -> torch.Tensor:
    num_targets = len(targets)
    num_edges = graph.num_edges
    mask = torch.zeros((num_targets, num_edges), dtype=torch.bool)
    if num_targets == 0 or num_edges == 0 or not anchor_distances:
        return mask

    anchor_matrix = torch.tensor(anchor_distances, dtype=torch.long)

    max_elements_per_chunk = 4_000_000
    for target_idx, target in enumerate(targets):
        edge_to_target = target_distances[target_idx, graph.dst_tensor]
        candidate_edges = edge_to_target.ne(unreachable_distance)
        if not bool(candidate_edges.any()):
            continue

        anchor_to_target = anchor_matrix[:, int(target)]
        active_anchors = anchor_to_target.ne(unreachable_distance)
        if not bool(active_anchors.any()):
            continue

        edge_ids = torch.nonzero(candidate_edges, as_tuple=False).view(-1)
        active_anchor_dist = anchor_matrix[active_anchors]
        active_anchor_to_target = anchor_to_target[active_anchors]
        chunk_size = max(
            1,
            max_elements_per_chunk // max(1, int(active_anchor_dist.size(0))),
        )

        for start in range(0, int(edge_ids.numel()), chunk_size):
            edge_chunk = edge_ids[start : start + chunk_size]
            src = graph.src_tensor[edge_chunk]
            dst = graph.dst_tensor[edge_chunk]
            suffix_distance = target_distances[target_idx, dst]

            anchor_to_src = active_anchor_dist[:, src]
            on_path = anchor_to_src.ne(unreachable_distance) & (
                anchor_to_src + 1 + suffix_distance.view(1, -1)
                == active_anchor_to_target.view(-1, 1)
            )

            mask[target_idx, edge_chunk] = on_path.any(dim=0)

    return mask


def _multi_source_min_dist(
    adjacency: list[list[int]],
    starts: Sequence[int],
) -> torch.Tensor:
    dist = [unreachable_distance] * len(adjacency)
    queue: deque[int] = deque()
    for start in starts:
        if dist[start] == 0:
            continue
        dist[start] = 0
        queue.append(start)

    while queue:
        u = queue.popleft()
        next_dist = dist[u] + 1
        for v in adjacency[u]:
            if dist[v] == unreachable_distance:
                dist[v] = next_dist
                queue.append(v)
    return torch.tensor(dist, dtype=torch.long)


def _min_distances(
    distances: Sequence[Sequence[int]] | None,
    *,
    num_nodes: int,
) -> torch.Tensor:
    if not distances:
        return torch.full((num_nodes,), unreachable_distance, dtype=torch.long)

    out: list[int] = []
    for node in range(num_nodes):
        best = min(
            (dist[node] for dist in distances if dist[node] != unreachable_distance),
            default=unreachable_distance,
        )
        out.append(best)
    return torch.tensor(out, dtype=torch.long)


def _nearest_target_distance(
    *,
    node_target_distances_flat: torch.Tensor,
    num_targets: int,
    num_nodes: int,
) -> torch.Tensor:
    if num_nodes <= 0:
        return torch.empty((0,), dtype=torch.long)
    if num_targets <= 0:
        return torch.full(
            (num_nodes,),
            node_target_unreachable_distance,
            dtype=torch.long,
        )

    distances = node_target_distances_flat.view(num_targets, num_nodes)
    distances = distances.masked_fill(
        distances.eq(unreachable_distance),
        node_target_unreachable_distance,
    )
    return distances.min(dim=0).values.long().contiguous()


def _bfs(adjacency: list[list[int]], start: int) -> list[int]:
    dist = [unreachable_distance] * len(adjacency)
    dist[start] = 0
    queue: deque[int] = deque([start])

    while queue:
        u = queue.popleft()
        next_dist = dist[u] + 1
        for v in adjacency[u]:
            if dist[v] == unreachable_distance:
                dist[v] = next_dist
                queue.append(v)
    return dist


def _valid_unique_nodes(node_ids: torch.Tensor, *, num_nodes: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in node_ids.view(-1).tolist():
        node = int(value)
        if node in seen or not 0 <= node < num_nodes:
            continue
        seen.add(node)
        out.append(node)
    return out


def _empty_target_labels() -> TargetPathLabels:
    return TargetPathLabels(
        target_node_ids=torch.empty((0,), dtype=torch.long),
        node_target_distances_flat=torch.empty((0,), dtype=torch.long),
        node_target_shortest_path_edge_mask_flat=torch.empty((0,), dtype=torch.bool),
    )


def _empty_anchor_labels() -> AnchorPathLabels:
    return AnchorPathLabels(
        anchor_node_forward_distances_flat=torch.empty((0,), dtype=torch.long),
        anchor_node_backward_distances_flat=torch.empty((0,), dtype=torch.long),
    )


__all__ = [
    "AnchorPathLabels",
    "PathLabels",
    "TargetPathLabels",
    "compute_anchor_path_labels",
    "compute_path_labels",
    "compute_target_path_labels",
    "node_target_unreachable_distance",
    "unreachable_distance",
]
