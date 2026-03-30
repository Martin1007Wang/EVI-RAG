from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_DIST_UNREACHABLE = -1


# ---------------------------------------------------------------------------
# Strict shortest-path labeling (SubgraphRAG-style)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortestPathLabels:
    """Weak supervision labels for SubgraphRAG-style triple scoring."""

    num_edges: int
    positive_edge_ids: torch.Tensor  # Long[P]
    max_path_length: Optional[int]


@dataclass(frozen=True)
class ForwardShortestPathTrajectory:
    """Deterministic forward shortest path used for guidance replay."""

    anchor_node: int
    path_nodes: tuple[int, ...]
    path_edge_ids: tuple[int, ...]
    hop_length: int


@dataclass(frozen=True)
class ForwardMultiAnchorUnionTrajectory:
    """Deterministic forward teacher subgraph that covers all anchors to one answer."""

    answer_node: int
    anchor_nodes: tuple[int, ...]
    anchor_path_nodes: tuple[tuple[int, ...], ...]
    anchor_path_edge_ids: tuple[tuple[int, ...], ...]
    ordered_edge_ids: tuple[int, ...]
    union_edge_ids: tuple[int, ...]
    total_hop_length: int


def _unique_valid_indices(raw: torch.Tensor, *, num_nodes: int) -> list[int]:
    if raw.numel() == 0:
        return []
    vals = raw.view(-1).detach().to(dtype=torch.long, device="cpu").tolist()
    return sorted({int(v) for v in vals if 0 <= int(v) < int(num_nodes)})


def _bfs_dist(adjacency: list[list[int]], start: int) -> list[int]:
    num_nodes = len(adjacency)
    dist = [_DIST_UNREACHABLE] * num_nodes
    if not (0 <= start < num_nodes):
        return dist
    dist[start] = 0
    q: deque[int] = deque([int(start)])
    while q:
        u = q.popleft()
        du = dist[u] + 1
        for v in adjacency[u]:
            if dist[v] != _DIST_UNREACHABLE:
                continue
            dist[v] = du
            q.append(v)
    return dist


def _multi_source_bfs_dist(adjacency: list[list[int]], starts: list[int]) -> list[int]:
    num_nodes = len(adjacency)
    dist = [_DIST_UNREACHABLE] * num_nodes
    q: deque[int] = deque()
    for start in starts:
        if not (0 <= start < num_nodes) or dist[start] != _DIST_UNREACHABLE:
            continue
        dist[start] = 0
        q.append(int(start))
    while q:
        u = q.popleft()
        du = dist[u] + 1
        for v in adjacency[u]:
            if dist[v] != _DIST_UNREACHABLE:
                continue
            dist[v] = du
            q.append(v)
    return dist


def _validate_edge_index(edge_index: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not torch.is_tensor(edge_index):
        raise TypeError("edge_index must be a torch.Tensor.")
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), 0
    edge_index = edge_index.to(device="cpu", dtype=torch.long)
    if edge_index.dim() != 2 or int(edge_index.size(0)) != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
        )
    return edge_index, int(edge_index.size(1))


def _build_digraph_overwrite(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[dict[tuple[int, int], int], list[list[int]], list[list[int]]]:
    """Collapse multi-edges by keeping the last seen edge id for each (u, v)."""
    pair_to_edge: dict[tuple[int, int], int] = {}
    out_sets: list[set[int]] = [set() for _ in range(num_nodes)]
    in_sets: list[set[int]] = [set() for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for e_id, (u_raw, v_raw) in enumerate(zip(src, dst)):
        u = int(u_raw)
        v = int(v_raw)
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        pair_to_edge[(u, v)] = int(e_id)
        out_sets[u].add(v)
        in_sets[v].add(u)
    adjacency = [sorted(nbrs) for nbrs in out_sets]
    rev_adjacency = [sorted(nbrs) for nbrs in in_sets]
    return pair_to_edge, adjacency, rev_adjacency


def _build_forward_adjacency_with_edges(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[list[list[tuple[int, int]]], list[list[int]]]:
    outgoing: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    incoming_sets: list[set[int]] = [set() for _ in range(num_nodes)]
    if int(edge_index.numel()) == 0:
        return outgoing, [sorted(values) for values in incoming_sets]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for edge_id, (u_raw, v_raw) in enumerate(zip(src, dst)):
        u = int(u_raw)
        v = int(v_raw)
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        outgoing[u].append((int(edge_id), v))
        incoming_sets[v].add(u)
    for neighbors in outgoing:
        neighbors.sort(key=lambda item: (item[0], item[1]))
    return outgoing, [sorted(values) for values in incoming_sets]


def _resolve_forward_shortest_path_to_target(
    *,
    outgoing: list[list[tuple[int, int]]],
    anchor_node: int,
    target_node: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if anchor_node == target_node:
        return ((int(anchor_node),), ())
    parent_node: dict[int, int | None] = {int(anchor_node): None}
    parent_edge: dict[int, int] = {}
    queue: deque[int] = deque([int(anchor_node)])
    while queue:
        current = int(queue.popleft())
        if current == int(target_node):
            break
        for edge_id, neighbor in outgoing[current]:
            neighbor = int(neighbor)
            if neighbor in parent_node:
                continue
            parent_node[neighbor] = current
            parent_edge[neighbor] = int(edge_id)
            queue.append(neighbor)
    if int(target_node) not in parent_node:
        return None
    path_nodes = [int(target_node)]
    path_edge_ids: list[int] = []
    current = int(target_node)
    while parent_node[current] is not None:
        path_edge_ids.append(int(parent_edge[current]))
        current = int(parent_node[current])
        path_nodes.append(current)
    path_nodes.reverse()
    path_edge_ids.reverse()
    return tuple(path_nodes), tuple(path_edge_ids)


def _dedup_preserve_order(values: list[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _precompute_dist_maps(
    *,
    adjacency: list[list[int]],
    rev_adjacency: list[list[int]],
    anchor_nodes: list[int],
    a_nodes: list[int],
) -> tuple[
    dict[int, list[int]],
    dict[int, list[int]],
    dict[int, list[int]],
    dict[int, list[int]],
]:
    dist_from_anchor = {anchor: _bfs_dist(adjacency, anchor) for anchor in anchor_nodes}
    dist_to_anchor = {
        anchor: _bfs_dist(rev_adjacency, anchor) for anchor in anchor_nodes
    }
    dist_from_a = {a: _bfs_dist(adjacency, a) for a in a_nodes}
    dist_to_a = {a: _bfs_dist(rev_adjacency, a) for a in a_nodes}
    return dist_from_anchor, dist_to_anchor, dist_from_a, dist_to_a


def _resolve_min_direction(
    *, forward_len: int, backward_len: int
) -> tuple[bool, bool, int]:
    """Return (use_forward, use_backward, min_len)."""
    f_ok = forward_len != _DIST_UNREACHABLE
    b_ok = backward_len != _DIST_UNREACHABLE
    if not (f_ok or b_ok):
        return False, False, _DIST_UNREACHABLE
    if f_ok and b_ok:
        min_len = min(int(forward_len), int(backward_len))
        return int(forward_len) == min_len, int(backward_len) == min_len, int(min_len)
    if f_ok:
        return True, False, int(forward_len)
    return False, True, int(backward_len)


def _collect_edges_on_shortest_paths(
    *,
    pair_to_edge: dict[tuple[int, int], int],
    dist_from_src: list[int],
    dist_to_dst: list[int],
    target_len: int,
) -> set[int]:
    if target_len <= 0:
        return set()
    out: set[int] = set()
    for (u, v), e_id in pair_to_edge.items():
        du = dist_from_src[u]
        dv = dist_to_dst[v]
        if du == _DIST_UNREACHABLE or dv == _DIST_UNREACHABLE:
            continue
        if int(du) + 1 + int(dv) == int(target_len):
            out.add(int(e_id))
    return out


def _label_edges_for_pair(
    *,
    pair_to_edge: dict[tuple[int, int], int],
    dist_from_anchor: dict[int, list[int]],
    dist_to_anchor: dict[int, list[int]],
    dist_from_a: dict[int, list[int]],
    dist_to_a: dict[int, list[int]],
    anchor: int,
    a: int,
) -> tuple[set[int], Optional[int]]:
    dist_anchor_to_nodes = dist_from_anchor[anchor]
    dist_nodes_to_anchor = dist_to_anchor[anchor]
    da = dist_from_a[a]
    dta = dist_to_a[a]
    use_f, use_b, min_len = _resolve_min_direction(
        forward_len=int(dist_anchor_to_nodes[a]), backward_len=int(da[anchor])
    )
    if min_len == _DIST_UNREACHABLE:
        return set(), None
    edges: set[int] = set()
    if use_f:
        edges |= _collect_edges_on_shortest_paths(
            pair_to_edge=pair_to_edge,
            dist_from_src=dist_anchor_to_nodes,
            dist_to_dst=dta,
            target_len=int(dist_anchor_to_nodes[a]),
        )
    if use_b:
        edges |= _collect_edges_on_shortest_paths(
            pair_to_edge=pair_to_edge,
            dist_from_src=da,
            dist_to_dst=dist_nodes_to_anchor,
            target_len=int(da[anchor]),
        )
    return edges, int(min_len)


def _label_edges_for_pairs(
    *,
    pair_to_edge: dict[tuple[int, int], int],
    dist_from_anchor: dict[int, list[int]],
    dist_to_anchor: dict[int, list[int]],
    dist_from_a: dict[int, list[int]],
    dist_to_a: dict[int, list[int]],
    anchor_nodes: list[int],
    a_nodes: list[int],
) -> tuple[set[int], Optional[int]]:
    pos_edge_ids: set[int] = set()
    max_len: Optional[int] = None
    for anchor in anchor_nodes:
        for a in a_nodes:
            edges, min_len = _label_edges_for_pair(
                pair_to_edge=pair_to_edge,
                dist_from_anchor=dist_from_anchor,
                dist_to_anchor=dist_to_anchor,
                dist_from_a=dist_from_a,
                dist_to_a=dist_to_a,
                anchor=anchor,
                a=a,
            )
            if min_len is None:
                continue
            pos_edge_ids |= edges
            if max_len is None:
                max_len = int(min_len)
            else:
                max_len = max(int(max_len), int(min_len))
    return pos_edge_ids, max_len


def compute_shortest_path_labels(
    *,
    edge_index: torch.Tensor,
    anchor_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
    num_nodes: int,
) -> ShortestPathLabels:
    """Compute strict SubgraphRAG-style shortest-path labels.

    Key properties (to match the original implementation):
    - Collapse multi-edges by treating the graph as a DiGraph (u->v keeps last edge id).
    - For each (anchor, answer), consider directed shortest paths anchor->answer and
      answer->anchor; if both exist,
      keep only the direction(s) with smaller length.
    - Mark an edge as positive iff it lies on at least one kept shortest path.
    """
    edge_index, num_edges = _validate_edge_index(edge_index)
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return ShortestPathLabels(
            num_edges=num_edges,
            positive_edge_ids=torch.empty((0,), dtype=torch.long),
            max_path_length=None,
        )

    anchor_nodes = _unique_valid_indices(
        torch.as_tensor(anchor_local_indices), num_nodes=num_nodes
    )
    a_nodes = _unique_valid_indices(
        torch.as_tensor(a_local_indices), num_nodes=num_nodes
    )
    if not anchor_nodes or not a_nodes:
        return ShortestPathLabels(
            num_edges=num_edges,
            positive_edge_ids=torch.empty((0,), dtype=torch.long),
            max_path_length=None,
        )

    pair_to_edge, adjacency, rev_adjacency = _build_digraph_overwrite(
        edge_index, num_nodes=num_nodes
    )
    dist_from_anchor, dist_to_anchor, dist_from_a, dist_to_a = _precompute_dist_maps(
        adjacency=adjacency,
        rev_adjacency=rev_adjacency,
        anchor_nodes=anchor_nodes,
        a_nodes=a_nodes,
    )
    pos_edge_ids, max_len = _label_edges_for_pairs(
        pair_to_edge=pair_to_edge,
        dist_from_anchor=dist_from_anchor,
        dist_to_anchor=dist_to_anchor,
        dist_from_a=dist_from_a,
        dist_to_a=dist_to_a,
        anchor_nodes=anchor_nodes,
        a_nodes=a_nodes,
    )
    positive = torch.as_tensor(sorted(pos_edge_ids), dtype=torch.long)
    return ShortestPathLabels(
        num_edges=num_edges, positive_edge_ids=positive, max_path_length=max_len
    )


def compute_forward_answer_distances(
    *,
    edge_index: torch.Tensor,
    a_local_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return forward distance-to-answer for every node, or -1 if unreachable."""

    edge_index, _ = _validate_edge_index(edge_index)
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return torch.empty((0,), dtype=torch.long)
    answer_nodes = _unique_valid_indices(
        torch.as_tensor(a_local_indices), num_nodes=num_nodes
    )
    if not answer_nodes:
        return torch.full((num_nodes,), fill_value=_DIST_UNREACHABLE, dtype=torch.long)
    _, _, rev_adjacency = _build_digraph_overwrite(edge_index, num_nodes=num_nodes)
    distances = _multi_source_bfs_dist(rev_adjacency, answer_nodes)
    return torch.as_tensor(distances, dtype=torch.long)


def compute_forward_shortest_path_edge_mask(
    *,
    edge_index: torch.Tensor,
    anchor_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Mark executable forward edges that lie on a shortest anchor->answer path."""

    edge_index, num_edges = _validate_edge_index(edge_index)
    num_nodes = int(num_nodes)
    if num_nodes <= 0 or num_edges == 0:
        return torch.zeros((num_edges,), dtype=torch.bool)
    anchor_nodes = _unique_valid_indices(
        torch.as_tensor(anchor_local_indices), num_nodes=num_nodes
    )
    if not anchor_nodes:
        return torch.zeros((num_edges,), dtype=torch.bool)
    answer_dist = compute_forward_answer_distances(
        edge_index=edge_index,
        a_local_indices=a_local_indices,
        num_nodes=num_nodes,
    ).tolist()
    reachable_starts = [
        int(start)
        for start in anchor_nodes
        if answer_dist[int(start)] != _DIST_UNREACHABLE
    ]
    if not reachable_starts:
        return torch.zeros((num_edges,), dtype=torch.bool)
    best_hop = min(int(answer_dist[start]) for start in reachable_starts)
    best_starts = [
        start for start in reachable_starts if int(answer_dist[start]) == best_hop
    ]
    _, adjacency, _ = _build_digraph_overwrite(edge_index, num_nodes=num_nodes)
    dist_from_best_starts = _multi_source_bfs_dist(adjacency, best_starts)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    keep = torch.zeros((num_edges,), dtype=torch.bool)
    for edge_id, (u_raw, v_raw) in enumerate(zip(src, dst)):
        u = int(u_raw)
        v = int(v_raw)
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        du = dist_from_best_starts[u]
        dv = answer_dist[v]
        if du == _DIST_UNREACHABLE or dv == _DIST_UNREACHABLE:
            continue
        if int(du) + 1 + int(dv) == int(best_hop):
            keep[edge_id] = True
    return keep


def resolve_forward_shortest_path_trajectory(
    *,
    edge_index: torch.Tensor,
    anchor_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
    num_nodes: int,
) -> ForwardShortestPathTrajectory | None:
    """Return a deterministic executable shortest path for forward replay guidance."""

    edge_index, _ = _validate_edge_index(edge_index)
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return None
    anchor_nodes = _unique_valid_indices(
        torch.as_tensor(anchor_local_indices), num_nodes=num_nodes
    )
    if not anchor_nodes:
        return None
    answer_dist_tensor = compute_forward_answer_distances(
        edge_index=edge_index,
        a_local_indices=a_local_indices,
        num_nodes=num_nodes,
    )
    answer_dist = answer_dist_tensor.tolist()
    best_start: int | None = None
    best_hop: int | None = None
    for start in anchor_nodes:
        hop = int(answer_dist[start])
        if hop == _DIST_UNREACHABLE:
            continue
        if (
            best_hop is None
            or hop < best_hop
            or (hop == best_hop and (best_start is None or start < best_start))
        ):
            best_start = int(start)
            best_hop = int(hop)
    if best_start is None or best_hop is None:
        return None
    outgoing, _ = _build_forward_adjacency_with_edges(edge_index, num_nodes=num_nodes)
    current = int(best_start)
    remaining = int(best_hop)
    path_nodes = [int(best_start)]
    path_edge_ids: list[int] = []
    while remaining > 0:
        candidates = [
            (edge_id, dst)
            for edge_id, dst in outgoing[current]
            if 0 <= dst < num_nodes and int(answer_dist[dst]) == int(remaining) - 1
        ]
        if not candidates:
            return None
        edge_id, dst = min(candidates, key=lambda item: (item[0], item[1]))
        path_edge_ids.append(int(edge_id))
        path_nodes.append(int(dst))
        current = int(dst)
        remaining -= 1
    return ForwardShortestPathTrajectory(
        anchor_node=int(best_start),
        path_nodes=tuple(path_nodes),
        path_edge_ids=tuple(path_edge_ids),
        hop_length=int(best_hop),
    )


def resolve_forward_multi_anchor_union_trajectory(
    *,
    edge_index: torch.Tensor,
    anchor_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
    num_nodes: int,
) -> ForwardMultiAnchorUnionTrajectory | None:
    """Return a deterministic executable union of shortest anchor-to-answer paths."""

    edge_index, _ = _validate_edge_index(edge_index)
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return None
    anchor_nodes = _unique_valid_indices(
        torch.as_tensor(anchor_local_indices), num_nodes=num_nodes
    )
    a_nodes = _unique_valid_indices(
        torch.as_tensor(a_local_indices), num_nodes=num_nodes
    )
    if not anchor_nodes or not a_nodes:
        return None
    outgoing, _ = _build_forward_adjacency_with_edges(edge_index, num_nodes=num_nodes)
    best_candidate: (
        tuple[
            tuple[int, int, int],
            ForwardMultiAnchorUnionTrajectory,
        ]
        | None
    ) = None
    for answer_node in a_nodes:
        anchor_path_nodes: list[tuple[int, ...]] = []
        anchor_path_edge_ids: list[tuple[int, ...]] = []
        total_hop_length = 0
        feasible = True
        for anchor_node in anchor_nodes:
            resolved = _resolve_forward_shortest_path_to_target(
                outgoing=outgoing,
                anchor_node=int(anchor_node),
                target_node=int(answer_node),
            )
            if resolved is None:
                feasible = False
                break
            path_nodes, path_edge_ids = resolved
            anchor_path_nodes.append(path_nodes)
            anchor_path_edge_ids.append(path_edge_ids)
            total_hop_length += int(len(path_edge_ids))
        if not feasible:
            continue
        concatenated_edges: list[int] = []
        for path_edge_ids in anchor_path_edge_ids:
            concatenated_edges.extend(int(edge_id) for edge_id in path_edge_ids)
        ordered_edge_ids = _dedup_preserve_order(concatenated_edges)
        union_edge_ids = tuple(
            sorted(set(int(edge_id) for edge_id in ordered_edge_ids))
        )
        trajectory = ForwardMultiAnchorUnionTrajectory(
            answer_node=int(answer_node),
            anchor_nodes=tuple(int(node_id) for node_id in anchor_nodes),
            anchor_path_nodes=tuple(anchor_path_nodes),
            anchor_path_edge_ids=tuple(anchor_path_edge_ids),
            ordered_edge_ids=tuple(int(edge_id) for edge_id in ordered_edge_ids),
            union_edge_ids=union_edge_ids,
            total_hop_length=int(total_hop_length),
        )
        ranking_key = (
            int(len(trajectory.union_edge_ids)),
            int(trajectory.total_hop_length),
            int(trajectory.answer_node),
        )
        if best_candidate is None or ranking_key < best_candidate[0]:
            best_candidate = (ranking_key, trajectory)
    if best_candidate is None:
        return None
    return best_candidate[1]


# ---------------------------------------------------------------------------
# Label store (disk cache)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeLabelEntry:
    num_edges: int
    positive_edge_ids: torch.Tensor  # Long[P]
    max_path_length: Optional[int]


class EdgeLabelStore:
    """Read-only label store keyed by `sample_id`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"EdgeLabelStore file not found: {self.path}")
        payload = torch.load(self.path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(
                f"EdgeLabelStore expects a dict payload, got {type(payload)!r}"
            )
        self._meta = payload.get("meta", {})
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise TypeError(
                "EdgeLabelStore payload must contain a dict at key 'entries'."
            )
        self._entries: Dict[str, Dict[str, Any]] = {
            str(k): v for k, v in entries.items()
        }

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta) if isinstance(self._meta, dict) else {}

    def get(self, sample_id: str) -> EdgeLabelEntry:
        raw = self._entries.get(str(sample_id))
        if raw is None:
            raise KeyError(f"Label missing for sample_id={sample_id!r} in {self.path}")
        num_edges = int(raw.get("num_edges", 0))
        pos = raw.get("positive_edge_ids")
        if pos is None:
            pos_ids = torch.empty((0,), dtype=torch.long)
        else:
            pos_ids = torch.as_tensor(pos, dtype=torch.long, device="cpu").view(-1)
        max_len = raw.get("max_path_length")
        max_path_length = None if max_len is None else int(max_len)
        return EdgeLabelEntry(
            num_edges=num_edges,
            positive_edge_ids=pos_ids,
            max_path_length=max_path_length,
        )


__all__ = [
    "ForwardMultiAnchorUnionTrajectory",
    "ForwardShortestPathTrajectory",
    "ShortestPathLabels",
    "compute_forward_answer_distances",
    "compute_forward_shortest_path_edge_mask",
    "resolve_forward_shortest_path_trajectory",
    "resolve_forward_multi_anchor_union_trajectory",
    "compute_shortest_path_labels",
    "EdgeLabelEntry",
    "EdgeLabelStore",
]
