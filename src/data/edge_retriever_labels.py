from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_ZERO = 0
_ONE = 1
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


def _unique_valid_indices(raw: torch.Tensor, *, num_nodes: int) -> list[int]:
    if raw.numel() == _ZERO:
        return []
    vals = raw.view(-1).detach().to(dtype=torch.long, device="cpu").tolist()
    return sorted({int(v) for v in vals if _ZERO <= int(v) < int(num_nodes)})


def _bfs_dist(adjacency: list[list[int]], start: int) -> list[int]:
    num_nodes = len(adjacency)
    dist = [_DIST_UNREACHABLE] * num_nodes
    if not (_ZERO <= start < num_nodes):
        return dist
    dist[start] = _ZERO
    q: deque[int] = deque([int(start)])
    while q:
        u = q.popleft()
        du = dist[u] + _ONE
        for v in adjacency[u]:
            if dist[v] != _DIST_UNREACHABLE:
                continue
            dist[v] = du
            q.append(v)
    return dist


def _validate_edge_index(edge_index: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not torch.is_tensor(edge_index):
        raise TypeError("edge_index must be a torch.Tensor.")
    if edge_index.numel() == _ZERO:
        return torch.empty((2, _ZERO), dtype=torch.long), _ZERO
    edge_index = edge_index.to(device="cpu", dtype=torch.long)
    if edge_index.dim() != 2 or int(edge_index.size(0)) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
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
    src = edge_index[_ZERO].tolist()
    dst = edge_index[_ONE].tolist()
    for e_id, (u_raw, v_raw) in enumerate(zip(src, dst)):
        u = int(u_raw)
        v = int(v_raw)
        if u < _ZERO or v < _ZERO or u >= num_nodes or v >= num_nodes:
            continue
        pair_to_edge[(u, v)] = int(e_id)
        out_sets[u].add(v)
        in_sets[v].add(u)
    adjacency = [sorted(nbrs) for nbrs in out_sets]
    rev_adjacency = [sorted(nbrs) for nbrs in in_sets]
    return pair_to_edge, adjacency, rev_adjacency


def _precompute_dist_maps(
    *,
    adjacency: list[list[int]],
    rev_adjacency: list[list[int]],
    q_nodes: list[int],
    a_nodes: list[int],
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    dist_from_q = {q: _bfs_dist(adjacency, q) for q in q_nodes}
    dist_to_q = {q: _bfs_dist(rev_adjacency, q) for q in q_nodes}
    dist_from_a = {a: _bfs_dist(adjacency, a) for a in a_nodes}
    dist_to_a = {a: _bfs_dist(rev_adjacency, a) for a in a_nodes}
    return dist_from_q, dist_to_q, dist_from_a, dist_to_a


def _resolve_min_direction(*, forward_len: int, backward_len: int) -> tuple[bool, bool, int]:
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
    if target_len <= _ZERO:
        return set()
    out: set[int] = set()
    for (u, v), e_id in pair_to_edge.items():
        du = dist_from_src[u]
        dv = dist_to_dst[v]
        if du == _DIST_UNREACHABLE or dv == _DIST_UNREACHABLE:
            continue
        if int(du) + _ONE + int(dv) == int(target_len):
            out.add(int(e_id))
    return out


def _label_edges_for_pair(
    *,
    pair_to_edge: dict[tuple[int, int], int],
    dist_from_q: dict[int, list[int]],
    dist_to_q: dict[int, list[int]],
    dist_from_a: dict[int, list[int]],
    dist_to_a: dict[int, list[int]],
    q: int,
    a: int,
) -> tuple[set[int], Optional[int]]:
    dq = dist_from_q[q]
    dtq = dist_to_q[q]
    da = dist_from_a[a]
    dta = dist_to_a[a]
    use_f, use_b, min_len = _resolve_min_direction(forward_len=int(dq[a]), backward_len=int(da[q]))
    if min_len == _DIST_UNREACHABLE:
        return set(), None
    edges: set[int] = set()
    if use_f:
        edges |= _collect_edges_on_shortest_paths(
            pair_to_edge=pair_to_edge,
            dist_from_src=dq,
            dist_to_dst=dta,
            target_len=int(dq[a]),
        )
    if use_b:
        edges |= _collect_edges_on_shortest_paths(
            pair_to_edge=pair_to_edge,
            dist_from_src=da,
            dist_to_dst=dtq,
            target_len=int(da[q]),
        )
    return edges, int(min_len)


def _label_edges_for_pairs(
    *,
    pair_to_edge: dict[tuple[int, int], int],
    dist_from_q: dict[int, list[int]],
    dist_to_q: dict[int, list[int]],
    dist_from_a: dict[int, list[int]],
    dist_to_a: dict[int, list[int]],
    q_nodes: list[int],
    a_nodes: list[int],
) -> tuple[set[int], Optional[int]]:
    pos_edge_ids: set[int] = set()
    max_len: Optional[int] = None
    for q in q_nodes:
        for a in a_nodes:
            edges, min_len = _label_edges_for_pair(
                pair_to_edge=pair_to_edge,
                dist_from_q=dist_from_q,
                dist_to_q=dist_to_q,
                dist_from_a=dist_from_a,
                dist_to_a=dist_to_a,
                q=q,
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
    q_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
    num_nodes: int,
) -> ShortestPathLabels:
    """Compute strict SubgraphRAG-style shortest-path labels.

    Key properties (to match the original implementation):
    - Collapse multi-edges by treating the graph as a DiGraph (u->v keeps last edge id).
    - For each (q, a), consider directed shortest paths q->a and a->q; if both exist,
      keep only the direction(s) with smaller length.
    - Mark an edge as positive iff it lies on at least one kept shortest path.
    """
    edge_index, num_edges = _validate_edge_index(edge_index)
    num_nodes = int(num_nodes)
    if num_nodes <= _ZERO:
        return ShortestPathLabels(
            num_edges=num_edges,
            positive_edge_ids=torch.empty((_ZERO,), dtype=torch.long),
            max_path_length=None,
        )

    q_nodes = _unique_valid_indices(torch.as_tensor(q_local_indices), num_nodes=num_nodes)
    a_nodes = _unique_valid_indices(torch.as_tensor(a_local_indices), num_nodes=num_nodes)
    if not q_nodes or not a_nodes:
        return ShortestPathLabels(
            num_edges=num_edges,
            positive_edge_ids=torch.empty((_ZERO,), dtype=torch.long),
            max_path_length=None,
        )

    pair_to_edge, adjacency, rev_adjacency = _build_digraph_overwrite(edge_index, num_nodes=num_nodes)
    dist_from_q, dist_to_q, dist_from_a, dist_to_a = _precompute_dist_maps(
        adjacency=adjacency,
        rev_adjacency=rev_adjacency,
        q_nodes=q_nodes,
        a_nodes=a_nodes,
    )
    pos_edge_ids, max_len = _label_edges_for_pairs(
        pair_to_edge=pair_to_edge,
        dist_from_q=dist_from_q,
        dist_to_q=dist_to_q,
        dist_from_a=dist_from_a,
        dist_to_a=dist_to_a,
        q_nodes=q_nodes,
        a_nodes=a_nodes,
    )
    positive = torch.as_tensor(sorted(pos_edge_ids), dtype=torch.long)
    return ShortestPathLabels(num_edges=num_edges, positive_edge_ids=positive, max_path_length=max_len)


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
            raise TypeError(f"EdgeLabelStore expects a dict payload, got {type(payload)!r}")
        self._meta = payload.get("meta", {})
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise TypeError("EdgeLabelStore payload must contain a dict at key 'entries'.")
        self._entries: Dict[str, Dict[str, Any]] = {str(k): v for k, v in entries.items()}

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta) if isinstance(self._meta, dict) else {}

    def get(self, sample_id: str) -> EdgeLabelEntry:
        raw = self._entries.get(str(sample_id))
        if raw is None:
            raise KeyError(f"Label missing for sample_id={sample_id!r} in {self.path}")
        num_edges = int(raw.get("num_edges", _ZERO))
        pos = raw.get("positive_edge_ids")
        if pos is None:
            pos_ids = torch.empty((_ZERO,), dtype=torch.long)
        else:
            pos_ids = torch.as_tensor(pos, dtype=torch.long, device="cpu").view(-1)
        max_len = raw.get("max_path_length")
        max_path_length = None if max_len is None else int(max_len)
        return EdgeLabelEntry(num_edges=num_edges, positive_edge_ids=pos_ids, max_path_length=max_path_length)


__all__ = [
    "ShortestPathLabels",
    "compute_shortest_path_labels",
    "EdgeLabelEntry",
    "EdgeLabelStore",
]

