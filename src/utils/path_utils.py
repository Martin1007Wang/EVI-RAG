from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Set

import torch

_DIST_UNREACHABLE = -1
_PATH_MODE_UNDIRECTED = "undirected"
_PATH_MODE_QA_DIRECTED = "qa_directed"
SIGNED_ANCHOR_DISTANCE_UNREACHABLE = 1_000_000_000


@dataclass(frozen=True)
class ShortestPathLabels:
    """SubgraphRAG 风格的弱监督标签"""

    num_edges: int
    positive_edge_ids: torch.Tensor  # 位于最短路径上的边索引
    reachable_target_node_ids: torch.Tensor
    max_path_length: Optional[int]


@dataclass(frozen=True)
class ShortestPathTeacherTargets:
    """Teacher targets for state-conditioned shortest-path supervision."""

    num_edges: int
    positive_edge_ids: torch.Tensor
    reachable_target_node_ids: torch.Tensor
    node_to_target_distance: torch.Tensor
    shortest_suffix_count: torch.Tensor
    bounded_suffix_count: torch.Tensor
    max_path_length: Optional[int]


def compute_signed_anchor_distances(
    *,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    num_nodes: int,
    path_mode: str = _PATH_MODE_QA_DIRECTED,
) -> torch.Tensor:
    """Compute signed directed distances to the nearest anchor."""
    path_mode = str(path_mode or _PATH_MODE_QA_DIRECTED).strip().lower()
    if path_mode not in {_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED}:
        raise ValueError(
            f"Unsupported path_mode={path_mode!r}; expected one of "
            f"{(_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED)}."
        )

    if num_nodes <= 0:
        return torch.empty((0,), dtype=torch.long)

    signed = torch.full(
        (num_nodes,),
        SIGNED_ANCHOR_DISTANCE_UNREACHABLE,
        dtype=torch.long,
    )
    if edge_index.numel() == 0:
        anchor_ids = torch.nonzero(is_anchor_mask.view(-1), as_tuple=False).view(-1)
        if anchor_ids.numel() > 0:
            signed[anchor_ids] = 0
        return signed

    anchor_nodes = (
        torch.nonzero(is_anchor_mask.view(-1), as_tuple=False).view(-1).tolist()
    )
    if not anchor_nodes:
        return signed

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adjacency, reverse_adjacency = _build_adjacency(
        num_nodes=num_nodes,
        src=src,
        dst=dst,
        path_mode=path_mode,
    )
    forward_dist = _multi_source_bfs_dist(adjacency, anchor_nodes)
    backward_dist = _multi_source_bfs_dist(reverse_adjacency, anchor_nodes)

    for node_idx in range(num_nodes):
        fwd = forward_dist[node_idx]
        bwd = backward_dist[node_idx]
        if fwd == 0 or bwd == 0:
            signed[node_idx] = 0
            continue
        if fwd == _DIST_UNREACHABLE and bwd == _DIST_UNREACHABLE:
            continue
        if bwd == _DIST_UNREACHABLE:
            signed[node_idx] = int(fwd)
            continue
        if fwd == _DIST_UNREACHABLE:
            signed[node_idx] = -int(bwd)
            continue
        signed[node_idx] = int(fwd) if fwd <= bwd else -int(bwd)
    return signed


def compute_shortest_path_teacher_targets(
    *,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    is_target_mask: torch.Tensor,
    num_nodes: int,
    path_mode: str = _PATH_MODE_QA_DIRECTED,
    budget_max_steps: int | None = None,
) -> ShortestPathTeacherTargets:
    """Compute shortest-path teacher targets and suffix statistics."""
    path_mode = str(path_mode or _PATH_MODE_QA_DIRECTED).strip().lower()
    if path_mode not in {_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED}:
        raise ValueError(
            f"Unsupported path_mode={path_mode!r}; expected one of "
            f"{(_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED)}."
        )

    node_to_target_distance = torch.full(
        (num_nodes,), _DIST_UNREACHABLE, dtype=torch.long
    )
    shortest_suffix_count = torch.zeros((num_nodes,), dtype=torch.float32)
    resolved_budget_max_steps = _resolve_budget_max_steps(
        budget_max_steps=budget_max_steps
    )
    bounded_suffix_count = torch.zeros(
        (resolved_budget_max_steps + 1, max(num_nodes, 0)), dtype=torch.float32
    )
    if edge_index.numel() == 0 or num_nodes <= 0:
        return ShortestPathTeacherTargets(
            num_edges=0,
            positive_edge_ids=torch.empty((0,), dtype=torch.long),
            reachable_target_node_ids=torch.empty((0,), dtype=torch.long),
            node_to_target_distance=node_to_target_distance,
            shortest_suffix_count=shortest_suffix_count,
            bounded_suffix_count=bounded_suffix_count,
            max_path_length=None,
        )

    anchor_nodes = torch.nonzero(is_anchor_mask.view(-1)).view(-1).tolist()
    answer_nodes = torch.nonzero(is_target_mask.view(-1)).view(-1).tolist()
    if not anchor_nodes or not answer_nodes:
        return ShortestPathTeacherTargets(
            num_edges=edge_index.size(1),
            positive_edge_ids=torch.empty((0,), dtype=torch.long),
            reachable_target_node_ids=torch.empty((0,), dtype=torch.long),
            node_to_target_distance=node_to_target_distance,
            shortest_suffix_count=shortest_suffix_count,
            bounded_suffix_count=bounded_suffix_count,
            max_path_length=None,
        )

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adj, rev_adj = _build_adjacency(
        num_nodes=num_nodes,
        src=src,
        dst=dst,
        path_mode=path_mode,
    )

    dist_from_anchors = {a: _bfs_dist(adj, a) for a in anchor_nodes}
    reachable_target_ids = sorted(
        {
            target
            for target in answer_nodes
            if any(
                dist_from_anchors[anchor][target] != _DIST_UNREACHABLE
                for anchor in anchor_nodes
            )
        }
    )
    if not reachable_target_ids:
        return ShortestPathTeacherTargets(
            num_edges=edge_index.size(1),
            positive_edge_ids=torch.empty((0,), dtype=torch.long),
            reachable_target_node_ids=torch.empty((0,), dtype=torch.long),
            node_to_target_distance=node_to_target_distance,
            shortest_suffix_count=shortest_suffix_count,
            bounded_suffix_count=bounded_suffix_count,
            max_path_length=None,
        )

    dist_to_answers = {ans: _bfs_dist(rev_adj, ans) for ans in reachable_target_ids}
    for node_idx in range(num_nodes):
        best_distance: Optional[int] = None
        for target in reachable_target_ids:
            distance = dist_to_answers[target][node_idx]
            if distance == _DIST_UNREACHABLE:
                continue
            best_distance = (
                distance if best_distance is None else min(best_distance, distance)
            )
        if best_distance is not None:
            node_to_target_distance[node_idx] = int(best_distance)

    positive_edge_ids: Set[int] = set()
    max_len: Optional[int] = None
    undirected = path_mode == _PATH_MODE_UNDIRECTED
    for anchor in anchor_nodes:
        for ans in reachable_target_ids:
            d_total = dist_from_anchors[anchor][ans]
            if d_total == _DIST_UNREACHABLE:
                continue
            max_len = d_total if max_len is None else max(max_len, d_total)
            for e_id, (u, v) in enumerate(zip(src, dst)):
                if (
                    dist_from_anchors[anchor][u] != _DIST_UNREACHABLE
                    and dist_to_answers[ans][v] != _DIST_UNREACHABLE
                    and dist_from_anchors[anchor][u] + 1 + dist_to_answers[ans][v]
                    == d_total
                ):
                    positive_edge_ids.add(e_id)
                    continue
                if (
                    undirected
                    and dist_from_anchors[anchor][v] != _DIST_UNREACHABLE
                    and dist_to_answers[ans][u] != _DIST_UNREACHABLE
                    and dist_from_anchors[anchor][v] + 1 + dist_to_answers[ans][u]
                    == d_total
                ):
                    positive_edge_ids.add(e_id)

    shortest_suffix_count = _count_shortest_suffixes(
        adjacency=adj,
        node_to_target_distance=node_to_target_distance.tolist(),
        reachable_target_ids=reachable_target_ids,
    )
    bounded_suffix_count = compute_bounded_suffix_count(
        adjacency=adj,
        is_target_mask=is_target_mask,
        budget_max_steps=resolved_budget_max_steps,
    )
    return ShortestPathTeacherTargets(
        num_edges=edge_index.size(1),
        positive_edge_ids=torch.as_tensor(sorted(positive_edge_ids), dtype=torch.long),
        reachable_target_node_ids=torch.as_tensor(
            reachable_target_ids, dtype=torch.long
        ),
        node_to_target_distance=node_to_target_distance,
        shortest_suffix_count=shortest_suffix_count,
        bounded_suffix_count=bounded_suffix_count,
        max_path_length=max_len,
    )


def compute_shortest_path_labels(
    *,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    is_target_mask: torch.Tensor,
    num_nodes: int,
    path_mode: str = _PATH_MODE_QA_DIRECTED,
) -> ShortestPathLabels:
    """
    计算严格的最短路径标签。
    算法逻辑：
    1. 建立有向图索引。
    2. 计算所有锚点到所有答案的双向最短距离。
    3. 标记所有满足 d(anchor, u) + 1 + d(v, answer) == d(anchor, answer) 的边 (u, v)。
    """
    path_mode = str(path_mode or _PATH_MODE_QA_DIRECTED).strip().lower()
    if path_mode not in {_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED}:
        raise ValueError(
            f"Unsupported path_mode={path_mode!r}; expected one of "
            f"{(_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED)}."
        )

    teacher_targets = compute_shortest_path_teacher_targets(
        edge_index=edge_index,
        is_anchor_mask=is_anchor_mask,
        is_target_mask=is_target_mask,
        num_nodes=num_nodes,
        path_mode=path_mode,
    )
    return ShortestPathLabels(
        num_edges=teacher_targets.num_edges,
        positive_edge_ids=teacher_targets.positive_edge_ids,
        reachable_target_node_ids=teacher_targets.reachable_target_node_ids,
        max_path_length=teacher_targets.max_path_length,
    )


def _count_shortest_suffixes(
    *,
    adjacency: list[list[int]],
    node_to_target_distance: Sequence[int],
    reachable_target_ids: Sequence[int],
) -> torch.Tensor:
    suffix_count = torch.zeros((len(adjacency),), dtype=torch.float32)
    for target in reachable_target_ids:
        if 0 <= int(target) < suffix_count.numel():
            suffix_count[int(target)] = 1.0

    max_distance = max(
        (distance for distance in node_to_target_distance if distance >= 0), default=-1
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


def compute_bounded_suffix_count(
    *,
    adjacency: Sequence[Sequence[int]],
    is_target_mask: torch.Tensor,
    budget_max_steps: int,
) -> torch.Tensor:
    """Count answer-reaching suffix mass within each remaining budget.

    ``bounded_suffix_count[b, u]`` is the number of answer-reaching suffixes that
    can start from node ``u`` and terminate within at most ``b`` additional
    expansion steps. Answer nodes are treated as absorbing terminals: once the
    rollout reaches an answer-bearing node, the teacher may stop immediately, so
    each target contributes a base mass of 1 for every budget.

    This tensor is intentionally a static graph-level teacher score rather than a
    full dynamic-state oracle. It ignores the visited-set/history component of
    the rollout state and therefore approximates bounded simple-path support by a
    budget-conditioned suffix mass over the static graph.
    """
    if budget_max_steps < 0:
        raise ValueError(
            f"budget_max_steps must be >= 0, got {budget_max_steps}."
        )
    num_nodes = len(adjacency)
    counts = torch.zeros(
        (budget_max_steps + 1, num_nodes),
        dtype=torch.float32,
    )
    if num_nodes == 0:
        return counts

    target_mask = is_target_mask.view(-1).bool()
    if target_mask.numel() != num_nodes:
        raise ValueError(
            "is_target_mask length must match adjacency size, got "
            f"{target_mask.numel()} and {num_nodes}."
        )

    counts[:, target_mask] = 1.0
    if budget_max_steps == 0:
        return counts

    for budget in range(1, budget_max_steps + 1):
        prev = counts[budget - 1]
        for node_idx, neighbors in enumerate(adjacency):
            if bool(target_mask[node_idx].item()):
                continue
            total = 0.0
            for neighbor in neighbors:
                if 0 <= int(neighbor) < num_nodes:
                    total += float(prev[int(neighbor)].item())
            counts[budget, node_idx] = total
    return counts


def _resolve_budget_max_steps(*, budget_max_steps: int | None) -> int:
    if budget_max_steps is None:
        return 0
    resolved = int(budget_max_steps)
    if resolved < 0:
        raise ValueError(
            f"budget_max_steps must be >= 0 when set, got {resolved}."
        )
    return resolved


def _build_adjacency(
    *,
    num_nodes: int,
    src: Sequence[int],
    dst: Sequence[int],
    path_mode: str,
) -> tuple[list[list[int]], list[list[int]]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    reverse_adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    undirected = path_mode == _PATH_MODE_UNDIRECTED
    for src_raw, dst_raw in zip(src, dst):
        u = int(src_raw)
        v = int(dst_raw)
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        adjacency[u].append(v)
        reverse_adjacency[v].append(u)
        if undirected and u != v:
            adjacency[v].append(u)
            reverse_adjacency[u].append(v)
    return adjacency, reverse_adjacency


def _bfs_dist(adjacency: list[list[int]], start: int) -> list[int]:
    num_nodes = len(adjacency)
    dist = [_DIST_UNREACHABLE] * num_nodes
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        d_next = dist[u] + 1
        for v in adjacency[u]:
            if dist[v] == _DIST_UNREACHABLE:
                dist[v] = d_next
                q.append(v)
    return dist


def _multi_source_bfs_dist(
    adjacency: list[list[int]], starts: Sequence[int]
) -> list[int]:
    num_nodes = len(adjacency)
    dist = [_DIST_UNREACHABLE] * num_nodes
    q = deque()
    for start in starts:
        start_idx = int(start)
        if 0 <= start_idx < num_nodes and dist[start_idx] == _DIST_UNREACHABLE:
            dist[start_idx] = 0
            q.append(start_idx)
    while q:
        u = q.popleft()
        d_next = dist[u] + 1
        for v in adjacency[u]:
            if dist[v] == _DIST_UNREACHABLE:
                dist[v] = d_next
                q.append(v)
    return dist


__all__ = [
    "SIGNED_ANCHOR_DISTANCE_UNREACHABLE",
    "compute_bounded_suffix_count",
    "compute_shortest_path_labels",
    "compute_shortest_path_teacher_targets",
    "compute_signed_anchor_distances",
    "ShortestPathLabels",
    "ShortestPathTeacherTargets",
]
