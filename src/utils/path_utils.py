from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import torch


signed_anchor_unreachable = 1_000_000_000
unreachable_distance = -1


@dataclass(frozen=True)
class ReachableTargets:
    """
    Reachable answer targets discovered from anchors before any budget filtering.

    Semantics:
    - `target_node_ids` contains local node ids of reachable answer targets.
    - `target_node_distances_flat` is a flattened tensor with shape semantics
      (num_targets, num_nodes).
    - The order of `target_node_ids` matches the first dimension of
      `target_node_distances_flat`.
    """

    target_node_ids: torch.Tensor
    target_node_distances_flat: torch.Tensor


@dataclass(frozen=True)
class TeacherLabels:
    """
    Final train-target-conditioned shortest-path supervision.

    Semantics:
    - `target_node_ids` contains local node ids of the final train targets.
    - `target_node_distances_flat` has shape semantics (num_targets, num_nodes).
    - `target_shortest_path_count_flat` has shape semantics (num_targets, num_nodes).
    - `target_shortest_path_edge_mask_flat` has shape semantics
      (num_targets, num_edges).
    - The order of `target_node_ids` matches the first dimension of all flattened
      per-target tensors in this object.
    """

    target_node_ids: torch.Tensor
    target_node_distances_flat: torch.Tensor
    target_shortest_path_count_flat: torch.Tensor
    target_shortest_path_edge_mask_flat: torch.Tensor


def compute_signed_anchor_distances(
    *,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Compute signed directed distances relative to the nearest anchor.

    Output semantics:
    - 0: anchor node
    - positive: reachable from an anchor in the forward graph
    - negative: can reach an anchor in the reverse graph
    - signed_anchor_unreachable: disconnected from all anchors both ways
    """
    if num_nodes <= 0:
        return torch.empty((0,), dtype=torch.long)

    signed = torch.full(
        (num_nodes,),
        signed_anchor_unreachable,
        dtype=torch.long,
    )

    anchor_nodes = torch.nonzero(
        is_anchor_mask.view(-1), as_tuple=False
    ).view(-1).tolist()
    if not anchor_nodes:
        return signed

    if edge_index.numel() == 0:
        signed[torch.as_tensor(anchor_nodes, dtype=torch.long)] = 0
        return signed

    adjacency, reverse_adjacency = _build_adjacency(
        num_nodes=num_nodes,
        src=edge_index[0].tolist(),
        dst=edge_index[1].tolist(),
    )

    forward_dist = _multi_source_bfs_dist(adjacency, anchor_nodes)
    backward_dist = _multi_source_bfs_dist(reverse_adjacency, anchor_nodes)

    for node_idx in range(num_nodes):
        fwd = forward_dist[node_idx]
        bwd = backward_dist[node_idx]

        if fwd == 0 or bwd == 0:
            signed[node_idx] = 0
        elif fwd == unreachable_distance and bwd == unreachable_distance:
            continue
        elif bwd == unreachable_distance:
            signed[node_idx] = int(fwd)
        elif fwd == unreachable_distance:
            signed[node_idx] = -int(bwd)
        else:
            signed[node_idx] = int(fwd) if fwd <= bwd else -int(bwd)

    return signed


def compute_reachable_targets(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    answer_node_ids: torch.Tensor,
    num_nodes: int,
) -> ReachableTargets:
    """
    Discover reachable answer targets and compute per-target node distances.

    Intended use:
    - graph collection
    - prepared-sample construction

    This function does not compute shortest-path counts or edge masks.
    """
    if num_nodes <= 0 or edge_index.numel() == 0:
        return _empty_reachable_targets()

    anchors = _unique_valid_node_ids(anchor_node_ids, num_nodes=num_nodes)
    answers = _unique_valid_node_ids(answer_node_ids, num_nodes=num_nodes)

    if not anchors or not answers:
        return _empty_reachable_targets()

    adjacency, reverse_adjacency = _build_adjacency(
        num_nodes=num_nodes,
        src=edge_index[0].tolist(),
        dst=edge_index[1].tolist(),
    )

    dist_from_anchors = {anchor: _bfs_dist(adjacency, anchor) for anchor in anchors}

    reachable_target_ids = sorted(
        {
            target
            for target in answers
            if any(
                dist_from_anchors[anchor][target] != unreachable_distance
                for anchor in anchors
            )
        }
    )
    if not reachable_target_ids:
        return _empty_reachable_targets()

    target_node_distances = _compute_target_node_distance_matrix(
        reverse_adjacency=reverse_adjacency,
        target_node_ids=reachable_target_ids,
        num_nodes=num_nodes,
    )

    return ReachableTargets(
        target_node_ids=torch.as_tensor(reachable_target_ids, dtype=torch.long),
        target_node_distances_flat=target_node_distances.reshape(-1).contiguous(),
    )


def compute_teacher_labels(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    target_node_ids: torch.Tensor,
    num_nodes: int,
) -> TeacherLabels:
    """
    Compute final train-target-conditioned shortest-path supervision.

    Intended use:
    - materialization after budget filtering
    - final teacher guidance labels

    Only returns the per-target labels actually needed by downstream teacher guidance.
    """
    if num_nodes <= 0 or edge_index.numel() == 0:
        return _empty_teacher_labels()

    anchors = _unique_valid_node_ids(anchor_node_ids, num_nodes=num_nodes)
    targets = _unique_valid_node_ids(target_node_ids, num_nodes=num_nodes)

    if not anchors or not targets:
        return _empty_teacher_labels()

    adjacency, reverse_adjacency = _build_adjacency(
        num_nodes=num_nodes,
        src=edge_index[0].tolist(),
        dst=edge_index[1].tolist(),
    )

    dist_from_anchors = {anchor: _bfs_dist(adjacency, anchor) for anchor in anchors}

    reachable_targets = sorted(
        {
            target
            for target in targets
            if any(
                dist_from_anchors[anchor][target] != unreachable_distance
                for anchor in anchors
            )
        }
    )
    if not reachable_targets:
        return _empty_teacher_labels()

    target_node_distances = _compute_target_node_distance_matrix(
        reverse_adjacency=reverse_adjacency,
        target_node_ids=reachable_targets,
        num_nodes=num_nodes,
    )

    target_shortest_path_edge_mask = _compute_target_shortest_path_edge_mask(
        src=edge_index[0].tolist(),
        dst=edge_index[1].tolist(),
        dist_from_anchors=dist_from_anchors,
        target_node_ids=reachable_targets,
        target_node_distances=target_node_distances,
    )

    target_shortest_path_count = _compute_target_shortest_path_counts(
        adjacency=adjacency,
        target_node_ids=reachable_targets,
        target_node_distances=target_node_distances,
    )

    return TeacherLabels(
        target_node_ids=torch.as_tensor(reachable_targets, dtype=torch.long),
        target_node_distances_flat=target_node_distances.reshape(-1).contiguous(),
        target_shortest_path_count_flat=target_shortest_path_count.reshape(-1).contiguous(),
        target_shortest_path_edge_mask_flat=target_shortest_path_edge_mask.reshape(-1).contiguous(),
    )


def _empty_reachable_targets() -> ReachableTargets:
    return ReachableTargets(
        target_node_ids=torch.empty((0,), dtype=torch.long),
        target_node_distances_flat=torch.empty((0,), dtype=torch.long),
    )


def _empty_teacher_labels() -> TeacherLabels:
    return TeacherLabels(
        target_node_ids=torch.empty((0,), dtype=torch.long),
        target_node_distances_flat=torch.empty((0,), dtype=torch.long),
        target_shortest_path_count_flat=torch.empty((0,), dtype=torch.float32),
        target_shortest_path_edge_mask_flat=torch.empty((0,), dtype=torch.bool),
    )


def _compute_target_node_distance_matrix(
    *,
    reverse_adjacency: list[list[int]],
    target_node_ids: Sequence[int],
    num_nodes: int,
) -> torch.Tensor:
    """
    Return a tensor with shape [num_targets, num_nodes], where row t stores the
    shortest directed distance from every node to target t.
    """
    target_node_distances = torch.full(
        (len(target_node_ids), num_nodes),
        unreachable_distance,
        dtype=torch.long,
    )

    for target_pos, target in enumerate(target_node_ids):
        target_node_distances[target_pos] = torch.as_tensor(
            _bfs_dist(reverse_adjacency, int(target)),
            dtype=torch.long,
        )

    return target_node_distances


def _compute_target_shortest_path_edge_mask(
    *,
    src: Sequence[int],
    dst: Sequence[int],
    dist_from_anchors: dict[int, list[int]],
    target_node_ids: Sequence[int],
    target_node_distances: torch.Tensor,
) -> torch.Tensor:
    """
    Return a tensor with shape [num_targets, num_edges], where entry [t, e] is True
    iff edge e lies on at least one shortest path from any anchor to target t.
    """
    num_edges = len(src)
    mask = torch.zeros((len(target_node_ids), num_edges), dtype=torch.bool)

    distance_rows = target_node_distances.tolist()

    for anchor_distances in dist_from_anchors.values():
        for target_pos, target in enumerate(target_node_ids):
            total_distance = anchor_distances[int(target)]
            if total_distance == unreachable_distance:
                continue

            target_distances = distance_rows[target_pos]
            for edge_id, (u, v) in enumerate(zip(src, dst)):
                if (
                    anchor_distances[int(u)] != unreachable_distance
                    and target_distances[int(v)] != unreachable_distance
                    and anchor_distances[int(u)] + 1 + target_distances[int(v)] == total_distance
                ):
                    mask[target_pos, edge_id] = True

    return mask


def _compute_target_shortest_path_counts(
    *,
    adjacency: list[list[int]],
    target_node_ids: Sequence[int],
    target_node_distances: torch.Tensor,
) -> torch.Tensor:
    """
    Return a tensor with shape [num_targets, num_nodes], where entry [t, v] is the
    number of shortest suffix paths from node v to target t.
    """
    counts = torch.zeros((len(target_node_ids), len(adjacency)), dtype=torch.float32)

    distance_rows = target_node_distances.tolist()
    for target_pos, target in enumerate(target_node_ids):
        counts[target_pos] = _count_shortest_suffixes(
            adjacency=adjacency,
            node_to_target_distance=distance_rows[target_pos],
            target_node_ids=[int(target)],
        )

    return counts


def _count_shortest_suffixes(
    *,
    adjacency: list[list[int]],
    node_to_target_distance: Sequence[int],
    target_node_ids: Sequence[int],
) -> torch.Tensor:
    """
    Count shortest suffix paths to the provided target set under a fixed distance field.

    For a single target:
    - suffix_count[target] = 1
    - suffix_count[node] = sum of suffix_count[neighbor] over outgoing neighbors that
      move exactly one step closer to the target
    """
    suffix_count = torch.zeros((len(adjacency),), dtype=torch.float32)

    for target in target_node_ids:
        if 0 <= int(target) < suffix_count.numel():
            suffix_count[int(target)] = 1.0

    max_distance = max(
        (distance for distance in node_to_target_distance if distance >= 0),
        default=unreachable_distance,
    )
    if max_distance <= 0:
        return suffix_count

    for distance in range(1, max_distance + 1):
        for node_idx, node_distance in enumerate(node_to_target_distance):
            if node_distance != distance:
                continue

            total = 0.0
            for neighbor in adjacency[node_idx]:
                if node_to_target_distance[neighbor] == distance - 1:
                    total += float(suffix_count[neighbor].item())

            suffix_count[node_idx] = total

    return suffix_count


def _unique_valid_node_ids(
    node_ids: torch.Tensor,
    *,
    num_nodes: int,
) -> list[int]:
    """
    Keep valid local node ids, remove duplicates, preserve first-seen order.
    """
    seen: set[int] = set()
    ordered: list[int] = []

    for raw in node_ids.view(-1).tolist():
        node_id = int(raw)
        if not (0 <= node_id < num_nodes):
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)

    return ordered


def _build_adjacency(
    *,
    num_nodes: int,
    src: Sequence[int],
    dst: Sequence[int],
) -> tuple[list[list[int]], list[list[int]]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    reverse_adjacency: list[list[int]] = [[] for _ in range(num_nodes)]

    for src_raw, dst_raw in zip(src, dst):
        u = int(src_raw)
        v = int(dst_raw)
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        adjacency[u].append(v)
        reverse_adjacency[v].append(u)

    return adjacency, reverse_adjacency


def _bfs_dist(adjacency: list[list[int]], start: int) -> list[int]:
    num_nodes = len(adjacency)
    dist = [unreachable_distance] * num_nodes
    dist[start] = 0

    queue = deque([start])
    while queue:
        node = queue.popleft()
        next_distance = dist[node] + 1
        for neighbor in adjacency[node]:
            if dist[neighbor] == unreachable_distance:
                dist[neighbor] = next_distance
                queue.append(neighbor)

    return dist


def _multi_source_bfs_dist(
    adjacency: list[list[int]],
    starts: Sequence[int],
) -> list[int]:
    num_nodes = len(adjacency)
    dist = [unreachable_distance] * num_nodes
    queue = deque()

    for start in starts:
        node = int(start)
        if 0 <= node < num_nodes and dist[node] == unreachable_distance:
            dist[node] = 0
            queue.append(node)

    while queue:
        node = queue.popleft()
        next_distance = dist[node] + 1
        for neighbor in adjacency[node]:
            if dist[neighbor] == unreachable_distance:
                dist[neighbor] = next_distance
                queue.append(neighbor)

    return dist


__all__ = [
    "signed_anchor_unreachable",
    "ReachableTargets",
    "TeacherLabels",
    "compute_reachable_targets",
    "compute_teacher_labels",
    "compute_signed_anchor_distances",
]