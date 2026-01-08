from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from src.data.schema.constants import (
    _DIST_UNREACHABLE,
    _PATH_MODE_QA_DIRECTED,
    _PATH_MODE_UNDIRECTED,
    _PATH_MODES,
)


def _validate_path_mode(path_mode: str) -> str:
    mode = str(path_mode)
    if mode not in _PATH_MODES:
        raise ValueError(f"Unsupported path_mode: {mode}. Expected one of {_PATH_MODES}.")
    return mode


def _shortest_path_single(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    sources: Sequence[int],
    targets: Sequence[int],
) -> Tuple[List[int], List[int]]:
    if not sources or not targets or num_nodes <= 0:
        return [], []

    from collections import deque

    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for idx, (u_raw, v_raw) in enumerate(zip(edge_src, edge_dst)):
        u = int(u_raw)
        v = int(v_raw)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adjacency[u].append((v, idx))
            if u != v:
                adjacency[v].append((u, idx))

    for nbrs in adjacency:
        nbrs.sort(key=lambda item: (item[0], item[1]))

    sources_unique = sorted({int(s) for s in sources if 0 <= int(s) < num_nodes})
    targets_unique = sorted({int(t) for t in targets if 0 <= int(t) < num_nodes})
    if not sources_unique or not targets_unique:
        return [], []

    dist = [-1] * num_nodes
    parent = [-1] * num_nodes
    parent_edge = [-1] * num_nodes
    q: deque[int] = deque()
    for s in sources_unique:
        dist[s] = 0
        q.append(s)

    while q:
        cur = q.popleft()
        next_dist = dist[cur] + 1
        for nb, e_idx in adjacency[cur]:
            if dist[nb] != -1:
                continue
            dist[nb] = next_dist
            parent[nb] = cur
            parent_edge[nb] = int(e_idx)
            q.append(nb)

    best_target = None
    best_dist = None
    for tgt in targets_unique:
        if dist[tgt] < 0:
            continue
        if best_dist is None or dist[tgt] < best_dist or (dist[tgt] == best_dist and tgt < best_target):
            best_target = tgt
            best_dist = dist[tgt]

    if best_target is None:
        return [], []

    nodes_rev: List[int] = [int(best_target)]
    edges_rev: List[int] = []
    cur = int(best_target)
    sources_set = set(sources_unique)
    while cur not in sources_set:
        prev = int(parent[cur])
        edge = int(parent_edge[cur])
        if prev < 0 or edge < 0:
            return [], []
        edges_rev.append(edge)
        nodes_rev.append(prev)
        cur = prev

    edges = list(reversed(edges_rev))
    nodes = list(reversed(nodes_rev))
    if not edges:
        return [], nodes
    return edges, nodes


def _build_undirected_adjacency(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for u_raw, v_raw in zip(edge_src, edge_dst):
        u = int(u_raw)
        v = int(v_raw)
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        adjacency[u].append(v)
        if u != v:
            adjacency[v].append(u)
    for nbrs in adjacency:
        nbrs.sort()
    return adjacency


def _build_directed_adjacency(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for u_raw, v_raw in zip(edge_src, edge_dst):
        u = int(u_raw)
        v = int(v_raw)
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        adjacency[u].append(v)
    for nbrs in adjacency:
        nbrs.sort()
    return adjacency


def _bfs_dist(num_nodes: int, adjacency: Sequence[Sequence[int]], sources: Sequence[int]) -> List[int]:
    dist = [_DIST_UNREACHABLE] * num_nodes
    if num_nodes <= 0:
        return dist
    from collections import deque

    q: deque[int] = deque()
    for s_raw in sources:
        s = int(s_raw)
        if 0 <= s < num_nodes and dist[s] < 0:
            dist[s] = 0
            q.append(s)

    while q:
        u = q.popleft()
        du = dist[u] + 1
        for v in adjacency[u]:
            if dist[v] >= 0:
                continue
            dist[v] = du
            q.append(v)
    return dist


def shortest_path_edge_indices_undirected(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    seeds: Sequence[int],
    answers: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Deterministic single shortest path between seeds and answers (undirected traversal)."""
    return _shortest_path_single(num_nodes, edge_src, edge_dst, seeds, answers)


def has_connectivity(
    graph: Sequence[Tuple[str, str, str]],
    seeds: Sequence[str],
    answers: Sequence[str],
    *,
    path_mode: str = _PATH_MODE_UNDIRECTED,
) -> bool:
    """Check existence of path seed->answer using local indexing."""
    if not graph or not seeds or not answers:
        return False
    node_index: Dict[str, int] = {}
    edge_src: List[int] = []
    edge_dst: List[int] = []

    def local_idx(node: str) -> int:
        if node not in node_index:
            node_index[node] = len(node_index)
        return node_index[node]

    for h, _, t in graph:
        edge_src.append(local_idx(h))
        edge_dst.append(local_idx(t))

    seed_ids = [node_index[s] for s in seeds if s in node_index]
    answer_ids = [node_index[a] for a in answers if a in node_index]
    if not seed_ids or not answer_ids:
        return False
    mode = _validate_path_mode(path_mode)
    if mode == _PATH_MODE_QA_DIRECTED:
        adjacency = _build_directed_adjacency(len(node_index), edge_src, edge_dst)
    else:
        adjacency = _build_undirected_adjacency(len(node_index), edge_src, edge_dst)
    dist = _bfs_dist(len(node_index), adjacency, seed_ids)
    return any(dist[a] >= 0 for a in answer_ids)
