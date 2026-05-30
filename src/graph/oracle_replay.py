from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from src.graph.paths import unreachable_distance

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ReplayProgram:
    candidate_edge_ids: Tensor
    candidate_ptr: Tensor
    candidate_target_positions: Tensor
    candidate_target_ptr: Tensor
    edge_to_candidate_ids: Tensor
    edge_to_candidate_ptr: Tensor
    path_truncated: bool = False


@dataclass(frozen=True, slots=True)
class PathCandidate:
    edge_ids: tuple[int, ...]
    edge_bits: int
    covered_target_bits: int


@dataclass(frozen=True, slots=True)
class OracleTerminalSet:
    edge_masks: tuple[int, ...]
    covered_count: int
    used_edges: int


def build_replay_program(
    *,
    edge_index: Tensor,
    anchor_node_ids: Tensor,
    reachable_target_node_ids: Tensor,
    num_nodes: int,
    max_paths_per_target: int = 64,
) -> ReplayProgram:
    targets = [int(x) for x in reachable_target_node_ids.view(-1).tolist()]
    anchors = [int(x) for x in anchor_node_ids.view(-1).tolist()]
    if not targets or not anchors or int(num_nodes) <= 0:
        return ReplayProgram(
            candidate_edge_ids=torch.empty(0, dtype=torch.long),
            candidate_ptr=torch.zeros(1, dtype=torch.long),
            candidate_target_positions=torch.empty(0, dtype=torch.long),
            candidate_target_ptr=torch.zeros(1, dtype=torch.long),
            edge_to_candidate_ids=torch.empty(0, dtype=torch.long),
            edge_to_candidate_ptr=torch.zeros(int(edge_index.size(1)) + 1, dtype=torch.long),
            path_truncated=False,
        )

    distances = torch.tensor(
        [_bfs_reverse(edge_index=edge_index, start=target, num_nodes=int(num_nodes)) for target in targets],
        dtype=torch.long,
    )
    outgoing = outgoing_edges_by_src(edge_index=edge_index, num_nodes=int(num_nodes))
    target_pos_by_node = {node: pos for pos, node in enumerate(targets)}

    candidate_edges: list[tuple[int, ...]] = []
    candidate_target_positions: list[tuple[int, ...]] = []
    truncated = False
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    for target_pos in range(len(targets)):
        target_candidates = 0
        anchor_order = sorted(
            (
                int(distances[target_pos, anchor].item()),
                int(anchor),
            )
            for anchor in anchors
            if 0 <= int(anchor) < int(num_nodes)
            and int(distances[target_pos, anchor].item()) != unreachable_distance
        )
        for distance, anchor in anchor_order:
            if distance == 0:
                continue
            remaining = int(max_paths_per_target) - int(target_candidates)
            if remaining <= 0:
                truncated = True
                break
            paths, hit_limit = enumerate_shortest_paths(
                outgoing=outgoing,
                distances=distances[target_pos],
                anchor=int(anchor),
                limit=int(remaining),
            )
            truncated = truncated or hit_limit
            target_candidates += len(paths)
            for edge_tuple, node_tuple in paths:
                covered_target_ids = tuple(
                    sorted(
                        {
                            int(node)
                            for node in node_tuple
                            if int(node) in target_pos_by_node
                        }
                    )
                )
                covered_target_pos = tuple(
                    sorted(int(target_pos_by_node[int(node_id)]) for node_id in covered_target_ids)
                )
                key = (tuple(edge_tuple), covered_target_ids)
                if key in seen:
                    continue
                seen.add(key)
                candidate_edges.append(tuple(int(edge_id) for edge_id in edge_tuple))
                candidate_target_positions.append(covered_target_pos)

    flat_candidate_edge_ids: list[int] = []
    candidate_ptr = [0]
    flat_candidate_target_positions: list[int] = []
    candidate_target_ptr = [0]
    edge_to_candidate_lists: list[list[int]] = [[] for _ in range(int(edge_index.size(1)))]
    for candidate_id, edge_tuple in enumerate(candidate_edges):
        flat_candidate_edge_ids.extend(edge_tuple)
        candidate_ptr.append(len(flat_candidate_edge_ids))
        flat_candidate_target_positions.extend(candidate_target_positions[candidate_id])
        candidate_target_ptr.append(len(flat_candidate_target_positions))
        for edge_id in edge_tuple:
            edge_to_candidate_lists[int(edge_id)].append(int(candidate_id))

    edge_to_candidate_ids: list[int] = []
    edge_to_candidate_ptr = [0]
    for candidate_ids in edge_to_candidate_lists:
        edge_to_candidate_ids.extend(candidate_ids)
        edge_to_candidate_ptr.append(len(edge_to_candidate_ids))

    return ReplayProgram(
        candidate_edge_ids=torch.tensor(flat_candidate_edge_ids, dtype=torch.long).contiguous()
        if flat_candidate_edge_ids
        else torch.empty(0, dtype=torch.long),
        candidate_ptr=torch.tensor(candidate_ptr, dtype=torch.long).contiguous(),
        candidate_target_positions=torch.tensor(flat_candidate_target_positions, dtype=torch.long).contiguous()
        if flat_candidate_target_positions
        else torch.empty(0, dtype=torch.long),
        candidate_target_ptr=torch.tensor(candidate_target_ptr, dtype=torch.long).contiguous(),
        edge_to_candidate_ids=torch.tensor(edge_to_candidate_ids, dtype=torch.long).contiguous()
        if edge_to_candidate_ids
        else torch.empty(0, dtype=torch.long),
        edge_to_candidate_ptr=torch.tensor(edge_to_candidate_ptr, dtype=torch.long).contiguous(),
        path_truncated=bool(truncated),
    )


def build_path_candidates_from_local_labels(
    *,
    candidate_edge_ids: Tensor,
    candidate_edge_candidate_ids: Tensor,
    candidate_target_node_ids: Tensor,
    candidate_target_candidate_ids: Tensor,
    reachable_target_node_ids: Tensor,
) -> tuple[list[PathCandidate], int]:
    target_nodes = [int(x) for x in reachable_target_node_ids.view(-1).tolist()]
    target_pos_by_node = {node: pos for pos, node in enumerate(target_nodes)}
    initial_target_bits = 0

    edge_groups = _group_ids_by_candidate(
        values=candidate_edge_ids,
        candidate_ids=candidate_edge_candidate_ids,
    )
    target_groups = _group_ids_by_candidate(
        values=candidate_target_node_ids,
        candidate_ids=candidate_target_candidate_ids,
    )
    num_candidates = max(len(edge_groups), len(target_groups))
    out: list[PathCandidate] = []
    for candidate_id in range(num_candidates):
        edge_tuple = tuple(edge_groups[candidate_id]) if candidate_id < len(edge_groups) else tuple()
        target_ids = tuple(target_groups[candidate_id]) if candidate_id < len(target_groups) else tuple()
        if not edge_tuple:
            continue
        covered_bits = 0
        for node_id in target_ids:
            pos = target_pos_by_node.get(int(node_id))
            if pos is not None:
                covered_bits |= 1 << pos
        out.append(
            PathCandidate(
                edge_ids=edge_tuple,
                edge_bits=edge_bits(edge_tuple),
                covered_target_bits=covered_bits,
            )
        )

    for node_id in target_nodes:
        if int(node_id) in target_pos_by_node:
            pos = target_pos_by_node[int(node_id)]
            del pos
    return out, initial_target_bits


def optimal_terminal_edge_masks(
    *,
    candidates: list[PathCandidate],
    initial_target_bits: int,
    target_count: int,
    budget: int,
    max_dp_states: int = 200_000,
) -> OracleTerminalSet:
    states: set[tuple[int, int]] = {(0, int(initial_target_bits))}
    for candidate in candidates:
        next_states = set(states)
        for edge_mask, covered_bits in states:
            new_edge_mask = edge_mask | int(candidate.edge_bits)
            if new_edge_mask.bit_count() > int(budget):
                continue
            new_covered = covered_bits | int(candidate.covered_target_bits)
            next_states.add((new_edge_mask, new_covered))
        if len(next_states) > int(max_dp_states):
            break
        states = next_states

    best_cover = -1
    best_used = 0
    best_masks: list[int] = []
    for edge_mask, covered_bits in states:
        edge_count = edge_mask.bit_count()
        if edge_count > int(budget):
            continue
        cover_count = int(covered_bits.bit_count())
        if cover_count > best_cover or (cover_count == best_cover and edge_count < best_used):
            best_cover = cover_count
            best_used = edge_count
            best_masks = [int(edge_mask)]
        elif cover_count == best_cover and edge_count == best_used:
            best_masks.append(int(edge_mask))

    if best_cover < 0:
        best_cover = int(initial_target_bits).bit_count()
        best_used = 0
        best_masks = [0]

    return OracleTerminalSet(
        edge_masks=tuple(sorted(set(best_masks))),
        covered_count=min(int(best_cover), int(target_count)),
        used_edges=int(best_used),
    )


def enumerate_shortest_paths(
    *,
    outgoing: list[list[tuple[int, int]]],
    distances: Tensor,
    anchor: int,
    limit: int,
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], bool]:
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    hit_limit = False

    def dfs(node: int, edge_path: tuple[int, ...], node_path: tuple[int, ...]) -> None:
        nonlocal hit_limit
        if len(out) >= int(limit):
            hit_limit = True
            return
        distance = int(distances[node].item())
        if distance == 0:
            out.append((edge_path, node_path))
            return
        if distance == unreachable_distance:
            return
        for edge_id, dst in outgoing[node]:
            dst_distance = int(distances[dst].item())
            if dst_distance != distance - 1:
                continue
            dfs(
                int(dst),
                (*edge_path, int(edge_id)),
                (*node_path, int(dst)),
            )
            if hit_limit:
                return

    dfs(int(anchor), (), (int(anchor),))
    return out, hit_limit


def outgoing_edges_by_src(*, edge_index: Tensor, num_nodes: int) -> list[list[tuple[int, int]]]:
    outgoing: list[list[tuple[int, int]]] = [[] for _ in range(int(num_nodes))]
    edge_index = edge_index.to(dtype=torch.long, device="cpu").contiguous()
    for edge_id in range(int(edge_index.size(1))):
        src = int(edge_index[0, edge_id].item())
        dst = int(edge_index[1, edge_id].item())
        outgoing[src].append((int(edge_id), int(dst)))
    return outgoing


def edge_bits(edge_ids: Iterable[int]) -> int:
    bits = 0
    for edge_id in edge_ids:
        bits |= 1 << int(edge_id)
    return bits


def _group_ids_by_candidate(*, values: Tensor, candidate_ids: Tensor) -> list[list[int]]:
    if int(values.numel()) == 0:
        return []
    max_id = int(candidate_ids.max().item()) + 1
    out: list[list[int]] = [[] for _ in range(max_id)]
    for value, candidate_id in zip(values.tolist(), candidate_ids.tolist(), strict=True):
        out[int(candidate_id)].append(int(value))
    return out


def _bfs_reverse(*, edge_index: Tensor, start: int, num_nodes: int) -> list[int]:
    reverse: list[list[int]] = [[] for _ in range(int(num_nodes))]
    edge_index = edge_index.to(dtype=torch.long, device="cpu").contiguous()
    for edge_id in range(int(edge_index.size(1))):
        src = int(edge_index[0, edge_id].item())
        dst = int(edge_index[1, edge_id].item())
        reverse[dst].append(int(src))

    dist = [unreachable_distance] * int(num_nodes)
    dist[int(start)] = 0
    queue = [int(start)]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        next_dist = dist[node] + 1
        for prev in reverse[node]:
            if dist[prev] != unreachable_distance:
                continue
            dist[prev] = next_dist
            queue.append(int(prev))
    return dist


__all__ = [
    "ReplayProgram",
    "OracleTerminalSet",
    "PathCandidate",
    "build_replay_program",
    "build_path_candidates_from_local_labels",
    "edge_bits",
    "optimal_terminal_edge_masks",
]
