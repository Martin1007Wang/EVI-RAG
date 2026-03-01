from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class OracleTrajectory:
    start_local: int
    edge_local_ids: tuple[int, ...]
    target_local: int | None = None
    shortest_gap: int = 0
    revisit_count: int = 0


def _build_adjacency(
    *,
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for local_eid, (head_raw, tail_raw) in enumerate(zip(edge_src, edge_dst)):
        head = int(head_raw)
        tail = int(tail_raw)
        if head < 0 or head >= num_nodes or tail < 0 or tail >= num_nodes:
            continue
        adjacency[head].append((tail, local_eid))
    return adjacency


def _build_reverse_neighbors(
    *,
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
) -> list[list[int]]:
    reverse_neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for head_raw, tail_raw in zip(edge_src, edge_dst):
        head = int(head_raw)
        tail = int(tail_raw)
        if head < 0 or head >= num_nodes or tail < 0 or tail >= num_nodes:
            continue
        reverse_neighbors[tail].append(head)
    return reverse_neighbors


def _bfs_distance_to_target(
    *,
    num_nodes: int,
    reverse_neighbors: list[list[int]],
    target_local: int,
) -> list[int]:
    if target_local < 0 or target_local >= num_nodes:
        return [-1 for _ in range(num_nodes)]
    dist = [-1 for _ in range(num_nodes)]
    dist[target_local] = 0
    queue: deque[int] = deque([target_local])
    while len(queue) > 0:
        node = queue.popleft()
        next_dist = dist[node] + 1
        for prev in reverse_neighbors[node]:
            if dist[prev] != -1:
                continue
            dist[prev] = next_dist
            queue.append(prev)
    return dist


def _enumerate_shortest_paths(
    *,
    adjacency: list[list[tuple[int, int]]],
    num_nodes: int,
    start_local: int,
    target_local: int,
    limit: int,
) -> list[tuple[int, ...]]:
    if limit <= 0:
        return []
    if start_local < 0 or start_local >= num_nodes:
        return []
    if target_local < 0 or target_local >= num_nodes:
        return []
    dist = [-1 for _ in range(num_nodes)]
    dist[start_local] = 0
    queue: deque[int] = deque([start_local])
    while len(queue) > 0:
        node = queue.popleft()
        for nxt, _ in adjacency[node]:
            if dist[nxt] != -1:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    if dist[target_local] < 0:
        return []

    paths: list[tuple[int, ...]] = []
    current: list[int] = []

    def dfs(node: int) -> None:
        if len(paths) >= limit:
            return
        if node == target_local:
            if len(current) > 0:
                paths.append(tuple(current))
            return
        for nxt, eid in adjacency[node]:
            if dist[nxt] != dist[node] + 1:
                continue
            current.append(eid)
            dfs(nxt)
            current.pop()
            if len(paths) >= limit:
                return

    dfs(start_local)
    return paths


def _enumerate_dfs_paths(
    *,
    adjacency: list[list[tuple[int, int]]],
    num_nodes: int,
    start_local: int,
    target_local: int,
    limit: int,
    max_depth: int,
    allow_cycles: bool,
    max_node_visits: int,
    dist_to_target: list[int] | None = None,
) -> list[tuple[int, ...]]:
    if limit <= 0 or max_depth <= 0:
        return []
    if start_local < 0 or start_local >= num_nodes:
        return []
    if target_local < 0 or target_local >= num_nodes:
        return []
    if max_node_visits <= 0:
        raise ValueError("max_node_visits must be a positive integer.")

    paths: list[tuple[int, ...]] = []
    current: list[int] = []
    visits = [0 for _ in range(num_nodes)]
    visits[start_local] = 1

    def dfs(node: int) -> None:
        if len(paths) >= limit:
            return
        if dist_to_target is not None:
            steps_to_target = dist_to_target[node]
            if steps_to_target < 0:
                return
            # Prune branches that cannot reach target within remaining depth budget.
            if len(current) + steps_to_target > max_depth:
                return
        if node == target_local and len(current) > 0:
            paths.append(tuple(current))
            if len(paths) >= limit:
                return
        if len(current) >= max_depth:
            return
        for nxt, eid in adjacency[node]:
            if not allow_cycles and visits[nxt] > 0:
                continue
            if allow_cycles and visits[nxt] >= max_node_visits:
                continue
            visits[nxt] += 1
            current.append(eid)
            dfs(nxt)
            current.pop()
            visits[nxt] -= 1
            if len(paths) >= limit:
                return

    dfs(start_local)
    return paths


def enumerate_oracle_trajectories(
    *,
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    start_nodes: Sequence[int],
    target_nodes: Sequence[int],
    max_paths_per_pair: int,
    max_paths_per_graph: int,
    max_shortest_paths_per_pair: int,
    max_dfs_paths_per_pair: int,
    max_depth: int,
    allow_cycles: bool,
    max_node_visits: int,
) -> list[OracleTrajectory]:
    if num_nodes <= 0:
        return []
    if max_paths_per_pair <= 0:
        return []
    if max_paths_per_graph <= 0:
        return []
    if max_depth <= 0:
        return []

    adjacency = _build_adjacency(num_nodes=num_nodes, edge_src=edge_src, edge_dst=edge_dst)
    reverse_neighbors = _build_reverse_neighbors(num_nodes=num_nodes, edge_src=edge_src, edge_dst=edge_dst)
    dist_to_target_cache: dict[int, list[int]] = {}
    trajectories: list[OracleTrajectory] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    edge_dst_lookup = [int(node) for node in edge_dst]
    for start_local_raw in start_nodes:
        start_local = int(start_local_raw)
        if start_local < 0 or start_local >= num_nodes:
            continue
        for target_local_raw in target_nodes:
            target_local = int(target_local_raw)
            if target_local < 0 or target_local >= num_nodes:
                continue
            dist_to_target = dist_to_target_cache.get(target_local)
            if dist_to_target is None:
                dist_to_target = _bfs_distance_to_target(
                    num_nodes=num_nodes,
                    reverse_neighbors=reverse_neighbors,
                    target_local=target_local,
                )
                dist_to_target_cache[target_local] = dist_to_target
            if dist_to_target[start_local] < 0:
                continue
            path_bank: list[tuple[int, ...]] = []
            if max_shortest_paths_per_pair > 0:
                path_bank.extend(
                    _enumerate_shortest_paths(
                        adjacency=adjacency,
                        num_nodes=num_nodes,
                        start_local=start_local,
                        target_local=target_local,
                        limit=max_shortest_paths_per_pair,
                    )
                )
            if max_dfs_paths_per_pair > 0:
                path_bank.extend(
                    _enumerate_dfs_paths(
                        adjacency=adjacency,
                        num_nodes=num_nodes,
                        start_local=start_local,
                        target_local=target_local,
                        limit=max_dfs_paths_per_pair,
                        max_depth=max_depth,
                        allow_cycles=allow_cycles,
                        max_node_visits=max_node_visits,
                        dist_to_target=dist_to_target,
                    )
                )
            pair_count = 0
            for edge_path in path_bank:
                if len(edge_path) == 0:
                    continue
                key = (start_local, edge_path)
                if key in seen:
                    continue
                seen.add(key)
                shortest_len = int(dist_to_target[start_local])
                shortest_gap = max(len(edge_path) - shortest_len, 0)
                current = start_local
                visit_counts: dict[int, int] = {start_local: 1}
                revisit_count = 0
                valid_path = True
                for edge_id in edge_path:
                    if edge_id < 0 or edge_id >= len(edge_dst_lookup):
                        valid_path = False
                        break
                    next_node = edge_dst_lookup[edge_id]
                    current = next_node
                    next_visits = visit_counts.get(next_node, 0) + 1
                    visit_counts[next_node] = next_visits
                    if next_visits > 1:
                        revisit_count += 1
                if not valid_path:
                    continue
                trajectories.append(
                    OracleTrajectory(
                        start_local=start_local,
                        edge_local_ids=edge_path,
                        target_local=target_local,
                        shortest_gap=shortest_gap,
                        revisit_count=revisit_count,
                    )
                )
                pair_count += 1
                if pair_count >= max_paths_per_pair or len(trajectories) >= max_paths_per_graph:
                    break
            if len(trajectories) >= max_paths_per_graph:
                break
        if len(trajectories) >= max_paths_per_graph:
            break
    return trajectories


def pack_oracle_trajectories(trajectories: Sequence[OracleTrajectory]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts: list[int] = []
    path_lengths: list[int] = []
    edge_ids: list[int] = []
    for trajectory in trajectories:
        edge_path = tuple(int(eid) for eid in trajectory.edge_local_ids)
        if len(edge_path) == 0:
            continue
        starts.append(int(trajectory.start_local))
        path_lengths.append(len(edge_path))
        edge_ids.extend(edge_path)
    return (
        torch.as_tensor(starts, dtype=torch.long),
        torch.as_tensor(path_lengths, dtype=torch.long),
        torch.as_tensor(edge_ids, dtype=torch.long),
    )


__all__ = ["OracleTrajectory", "enumerate_oracle_trajectories", "pack_oracle_trajectories"]
