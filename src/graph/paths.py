from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Sequence
import torch

unreachable_distance = -1
node_target_unreachable_distance = 1_000_000_000


@dataclass(frozen=True)
class PathLabels:
    reachable_target_node_ids: torch.Tensor
    anchor_node_forward_distances_flat: torch.Tensor
    anchor_node_backward_distances_flat: torch.Tensor
    node_target_distance: torch.Tensor
    target_node_distances_flat: torch.Tensor
    target_shortest_path_count_flat: torch.Tensor
    target_shortest_path_edge_mask_flat: torch.Tensor


def compute_path_labels(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    target_node_ids: torch.Tensor,
    num_nodes: int,
) -> PathLabels:
    target = compute_target_path_labels(
        edge_index=edge_index,
        anchor_node_ids=anchor_node_ids,
        target_node_ids=target_node_ids,
        num_nodes=num_nodes,
    )
    anchor = compute_anchor_path_labels(
        edge_index=edge_index,
        anchor_node_ids=anchor_node_ids,
        num_nodes=num_nodes,
    )
    return PathLabels(
        reachable_target_node_ids=target.target_node_ids.long().contiguous(),
        anchor_node_forward_distances_flat=anchor.anchor_node_forward_distances_flat.contiguous(),
        anchor_node_backward_distances_flat=anchor.anchor_node_backward_distances_flat.contiguous(),
        node_target_distance=_nearest_target_distance(
            target_node_distances_flat=target.target_node_distances_flat,
            num_targets=int(target.target_node_ids.numel()),
            num_nodes=num_nodes,
        ),
        target_node_distances_flat=target.target_node_distances_flat.contiguous(),
        target_shortest_path_count_flat=target.target_shortest_path_count_flat.contiguous(),
        target_shortest_path_edge_mask_flat=target.target_shortest_path_edge_mask_flat.contiguous(),
    )


@dataclass(frozen=True)
class TargetPathLabels:
    target_node_ids: torch.Tensor
    target_node_distances_flat: torch.Tensor
    target_shortest_path_count_flat: torch.Tensor
    target_shortest_path_edge_mask_flat: torch.Tensor


@dataclass(frozen=True)
class AnchorPathLabels:
    anchor_node_forward_distances_flat: torch.Tensor
    anchor_node_backward_distances_flat: torch.Tensor


def compute_target_path_labels(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    target_node_ids: torch.Tensor,
    num_nodes: int,
) -> TargetPathLabels:
    """
    Compute target-conditioned shortest-path supervision.
    Semantics:
    - target_node_ids:
        Reachable answer target local node ids.
    - target_node_distances_flat:
        Flattened [num_targets, num_nodes].
        Entry [t, v] is d(v -> target_t).
    - target_shortest_path_count_flat:
        Flattened [num_targets, num_nodes].
        Entry [t, v] is the number of shortest suffixes from v to target_t.
    - target_shortest_path_edge_mask_flat:
        Flattened [num_targets, num_edges].
        Entry [t, e] is true iff edge e lies on at least one shortest path
        from some anchor to target_t.
    """
    if num_nodes <= 0:
        return _empty_target_labels()
    anchors = _valid_unique_nodes(anchor_node_ids, num_nodes=num_nodes)
    targets = _valid_unique_nodes(target_node_ids, num_nodes=num_nodes)
    if not anchors or not targets:
        return _empty_target_labels()
    src, dst = _edge_lists(edge_index)
    adjacency, reverse_adjacency = _build_adjacency(
        num_nodes=num_nodes,
        src=src,
        dst=dst,
    )
    anchor_distances = [_bfs(adjacency, anchor) for anchor in anchors]
    reachable_targets = [
        target
        for target in targets
        if any(dist[target] != unreachable_distance for dist in anchor_distances)
    ]
    if not reachable_targets:
        return _empty_target_labels()
    target_distances = torch.tensor(
        [_bfs(reverse_adjacency, target) for target in reachable_targets],
        dtype=torch.long,
    )
    target_counts = _shortest_suffix_count_matrix(
        adjacency=adjacency,
        target_node_ids=reachable_targets,
        target_distances=target_distances,
    )
    target_edge_mask = _shortest_path_edge_mask(
        src=src,
        dst=dst,
        anchor_distances=anchor_distances,
        target_node_ids=reachable_targets,
        target_distances=target_distances,
    )
    return TargetPathLabels(
        target_node_ids=torch.tensor(reachable_targets, dtype=torch.long),
        target_node_distances_flat=target_distances.reshape(-1).contiguous(),
        target_shortest_path_count_flat=target_counts.reshape(-1).contiguous(),
        target_shortest_path_edge_mask_flat=target_edge_mask.reshape(-1).contiguous(),
    )


def compute_anchor_path_labels(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    num_nodes: int,
) -> AnchorPathLabels:
    """
    Compute anchor-conditioned structural distances.
    Semantics:
    - anchor_node_forward_distances_flat:
        Shape [num_nodes]. Entry [v] is min_a d(a -> v).
    - anchor_node_backward_distances_flat:
        Shape [num_nodes]. Entry [v] is min_a d(v -> a).
    """
    if num_nodes <= 0:
        return _empty_anchor_labels()
    anchors = _valid_unique_nodes(anchor_node_ids, num_nodes=num_nodes)
    if not anchors:
        unreachable = torch.full((num_nodes,), unreachable_distance, dtype=torch.long)
        return AnchorPathLabels(
            anchor_node_forward_distances_flat=unreachable,
            anchor_node_backward_distances_flat=unreachable.clone(),
        )
    src, dst = _edge_lists(edge_index)
    adjacency, reverse_adjacency = _build_adjacency(
        num_nodes=num_nodes,
        src=src,
        dst=dst,
    )
    return AnchorPathLabels(
        anchor_node_forward_distances_flat=_multi_source_min_dist(adjacency, anchors),
        anchor_node_backward_distances_flat=_multi_source_min_dist(
            reverse_adjacency,
            anchors,
        ),
    )


def _shortest_suffix_count_matrix(
    *,
    adjacency: list[list[int]],
    target_node_ids: Sequence[int],
    target_distances: torch.Tensor,
) -> torch.Tensor:
    rows = [
        _shortest_suffix_counts(
            adjacency=adjacency,
            target_node_id=int(target),
            node_to_target_dist=target_distances[target_idx].tolist(),
        )
        for target_idx, target in enumerate(target_node_ids)
    ]
    return torch.stack(rows, dim=0)


def _shortest_suffix_counts(
    *,
    adjacency: list[list[int]],
    target_node_id: int,
    node_to_target_dist: Sequence[int],
) -> torch.Tensor:
    """
    Count shortest suffixes from every node to one target.
    Dynamic program over distance layers:
    count[target] = 1
    count[u] = sum count[v] for edges u -> v with d(v, target) = d(u, target) - 1
    """
    num_nodes = len(adjacency)
    counts = torch.zeros(num_nodes, dtype=torch.float32)
    if not 0 <= target_node_id < num_nodes:
        return counts
    if node_to_target_dist[target_node_id] != 0:
        return counts
    counts[target_node_id] = 1.0
    max_dist = max((d for d in node_to_target_dist if d >= 0), default=0)
    for dist in range(1, max_dist + 1):
        for u, u_dist in enumerate(node_to_target_dist):
            if u_dist != dist:
                continue
            total = 0.0
            for v in adjacency[u]:
                if node_to_target_dist[v] == dist - 1:
                    total += float(counts[v])
            counts[u] = total
    return counts


def _shortest_path_edge_mask(
    *,
    src: Sequence[int],
    dst: Sequence[int],
    anchor_distances: Sequence[Sequence[int]],
    target_node_ids: Sequence[int],
    target_distances: torch.Tensor,
) -> torch.Tensor:
    """
    Mark edges lying on at least one anchor-to-target shortest path.
    """
    mask = torch.zeros((len(target_node_ids), len(src)), dtype=torch.bool)
    target_distance_rows = target_distances.tolist()
    for target_idx, target in enumerate(target_node_ids):
        node_to_target = target_distance_rows[target_idx]
        for anchor_to_node in anchor_distances:
            anchor_to_target = anchor_to_node[int(target)]
            if anchor_to_target == unreachable_distance:
                continue
            for edge_id, (u, v) in enumerate(zip(src, dst)):
                if anchor_to_node[u] == unreachable_distance:
                    continue
                if node_to_target[v] == unreachable_distance:
                    continue
                if anchor_to_node[u] + 1 + node_to_target[v] == anchor_to_target:
                    mask[target_idx, edge_id] = True
    return mask


def _multi_source_min_dist(
    adjacency: list[list[int]],
    starts: Sequence[int],
) -> torch.Tensor:
    """
    Compute min_s d(s -> v) for all v.
    """
    dist = [unreachable_distance] * len(adjacency)
    queue: deque[int] = deque()
    for start in starts:
        if dist[start] == 0:
            continue
        dist[start] = 0
        queue.append(start)
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if dist[v] != unreachable_distance:
                continue
            dist[v] = dist[u] + 1
            queue.append(v)
    return torch.tensor(dist, dtype=torch.long)


def _nearest_target_distance(
    *,
    target_node_distances_flat: torch.Tensor,
    num_targets: int,
    num_nodes: int,
) -> torch.Tensor:
    """
    Collapse per-target d(v -> target_t) labels into one nearest-target distance.
    """
    if num_nodes <= 0:
        return torch.empty((0,), dtype=torch.long)
    if num_targets <= 0:
        return torch.full(
            (num_nodes,),
            node_target_unreachable_distance,
            dtype=torch.long,
        )

    distances = target_node_distances_flat.view(num_targets, num_nodes)
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
        for v in adjacency[u]:
            if dist[v] != unreachable_distance:
                continue
            dist[v] = dist[u] + 1
            queue.append(v)
    return dist


def _build_adjacency(
    *,
    num_nodes: int,
    src: Sequence[int],
    dst: Sequence[int],
) -> tuple[list[list[int]], list[list[int]]]:
    adjacency = [[] for _ in range(num_nodes)]
    reverse_adjacency = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        adjacency[u].append(v)
        reverse_adjacency[v].append(u)
    return adjacency, reverse_adjacency


def _edge_lists(edge_index: torch.Tensor) -> tuple[list[int], list[int]]:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}."
        )
    if edge_index.numel() == 0:
        return [], []
    return (
        [int(x) for x in edge_index[0].tolist()],
        [int(x) for x in edge_index[1].tolist()],
    )


def _valid_unique_nodes(
    node_ids: torch.Tensor,
    *,
    num_nodes: int,
) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in node_ids.view(-1).tolist():
        node_id = int(value)
        if node_id in seen:
            continue
        if not 0 <= node_id < num_nodes:
            continue
        seen.add(node_id)
        out.append(node_id)
    return out


def _empty_target_labels() -> TargetPathLabels:
    return TargetPathLabels(
        target_node_ids=torch.empty((0,), dtype=torch.long),
        target_node_distances_flat=torch.empty((0,), dtype=torch.long),
        target_shortest_path_count_flat=torch.empty((0,), dtype=torch.float32),
        target_shortest_path_edge_mask_flat=torch.empty((0,), dtype=torch.bool),
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
