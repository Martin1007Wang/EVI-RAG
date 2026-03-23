from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


_DIST_UNREACHABLE = -1


@dataclass(frozen=True)
class ShortestPathRewardOracle:
    sample_id: str
    num_nodes: int
    num_edges: int
    dist_to_answer: tuple[int, ...]
    oracle_transitions: tuple[tuple[tuple[int, int], ...], ...]

    def distance_to_answer(self, start_node: int) -> int:
        if start_node < 0 or start_node >= int(self.num_nodes):
            return _DIST_UNREACHABLE
        return int(self.dist_to_answer[start_node])


def _unique_valid_indices(raw: torch.Tensor, *, num_nodes: int) -> list[int]:
    if raw.numel() == 0:
        return []
    values = raw.view(-1).detach().to(device="cpu", dtype=torch.long).tolist()
    return sorted({int(value) for value in values if 0 <= int(value) < int(num_nodes)})


def _build_adjacency(
    *,
    edge_index: torch.Tensor,
    edge_relations: torch.Tensor,
    num_nodes: int,
) -> tuple[list[list[tuple[int, int]]], list[list[int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    reverse_adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    if int(edge_index.numel()) == 0:
        return adjacency, reverse_adjacency
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    edge_relations_cpu = edge_relations.to(device="cpu", dtype=torch.long)
    src_nodes = edge_index_cpu[0].tolist()
    dst_nodes = edge_index_cpu[1].tolist()
    relation_ids = edge_relations_cpu.tolist()
    for src_raw, dst_raw, relation_raw in zip(src_nodes, dst_nodes, relation_ids):
        src = int(src_raw)
        dst = int(dst_raw)
        relation_id = int(relation_raw)
        if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            continue
        adjacency[src].append((relation_id, dst))
        reverse_adjacency[dst].append(src)
    return adjacency, reverse_adjacency


def _multi_source_reverse_bfs(
    *,
    reverse_adjacency: list[list[int]],
    answer_nodes: list[int],
) -> list[int]:
    num_nodes = len(reverse_adjacency)
    dist = [_DIST_UNREACHABLE] * num_nodes
    queue: deque[int] = deque()
    for answer_node in answer_nodes:
        if not (0 <= int(answer_node) < num_nodes):
            continue
        if dist[int(answer_node)] != _DIST_UNREACHABLE:
            continue
        dist[int(answer_node)] = 0
        queue.append(int(answer_node))
    while queue:
        node = queue.popleft()
        next_distance = dist[node] + 1
        for parent in reverse_adjacency[node]:
            if dist[parent] != _DIST_UNREACHABLE:
                continue
            dist[parent] = next_distance
            queue.append(parent)
    return dist


def build_shortest_path_reward_oracle(
    *,
    sample_id: str,
    edge_index: torch.Tensor,
    edge_relations: torch.Tensor,
    answer_local_indices: torch.Tensor,
    num_nodes: int,
) -> ShortestPathRewardOracle:
    num_nodes = int(num_nodes)
    answer_nodes = _unique_valid_indices(answer_local_indices, num_nodes=num_nodes)
    adjacency, reverse_adjacency = _build_adjacency(
        edge_index=edge_index,
        edge_relations=edge_relations,
        num_nodes=num_nodes,
    )
    dist_to_answer = _multi_source_reverse_bfs(
        reverse_adjacency=reverse_adjacency,
        answer_nodes=answer_nodes,
    )
    oracle_transitions: list[tuple[tuple[int, int], ...]] = []
    for node, outgoing_edges in enumerate(adjacency):
        node_distance = int(dist_to_answer[node])
        if node_distance <= 0:
            oracle_transitions.append(tuple())
            continue
        transitions = {
            (int(relation_id), int(dst_node))
            for relation_id, dst_node in outgoing_edges
            if int(dst_node) < num_nodes
            and int(dist_to_answer[int(dst_node)]) == node_distance - 1
        }
        oracle_transitions.append(tuple(sorted(transitions)))
    return ShortestPathRewardOracle(
        sample_id=str(sample_id),
        num_nodes=num_nodes,
        num_edges=int(edge_index.size(1)) if edge_index.dim() == 2 else 0,
        dist_to_answer=tuple(int(value) for value in dist_to_answer),
        oracle_transitions=tuple(oracle_transitions),
    )


def compute_shortest_path_prefix_alignment(
    *,
    oracle: ShortestPathRewardOracle,
    start_node: int,
    relation_ids: list[int],
) -> float:
    alignments = compute_shortest_path_alignment_trace(
        oracle=oracle,
        start_node=start_node,
        relation_ids=relation_ids,
    )
    if not alignments:
        shortest_distance = oracle.distance_to_answer(int(start_node))
        if shortest_distance == 0:
            return 1.0
        return 0.0
    return float(alignments[-1])


def compute_shortest_path_alignment_trace(
    *,
    oracle: ShortestPathRewardOracle,
    start_node: int,
    relation_ids: list[int],
) -> list[float]:
    shortest_distance = oracle.distance_to_answer(int(start_node))
    if shortest_distance == _DIST_UNREACHABLE:
        return [0.0 for _ in relation_ids]
    if shortest_distance == 0:
        return [0.0 for _ in relation_ids]
    frontier = {int(start_node)}
    matched_prefix_len = 0
    alignments: list[float] = []
    blocked = False
    for relation_id in relation_ids:
        if blocked or matched_prefix_len >= shortest_distance:
            alignments.append(float(matched_prefix_len) / float(shortest_distance))
            continue
        next_frontier: set[int] = set()
        for node in frontier:
            for oracle_relation_id, oracle_dst in oracle.oracle_transitions[node]:
                if int(oracle_relation_id) == int(relation_id):
                    next_frontier.add(int(oracle_dst))
        if not next_frontier:
            blocked = True
            alignments.append(float(matched_prefix_len) / float(shortest_distance))
            continue
        matched_prefix_len += 1
        frontier = next_frontier
        alignments.append(float(matched_prefix_len) / float(shortest_distance))
    return alignments


__all__ = [
    "ShortestPathRewardOracle",
    "compute_shortest_path_alignment_trace",
    "build_shortest_path_reward_oracle",
    "compute_shortest_path_prefix_alignment",
]
