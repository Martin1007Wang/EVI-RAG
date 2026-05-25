from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch

unreachable_distance = -1
node_target_unreachable_distance = 1_000_000_000
_replay_reward_epsilon = 1.0e-6
_replay_edge_cost = 0.05
_max_replay_state_candidates = 50_000


@dataclass(frozen=True, slots=True)
class PathLabels:
    reachable_target_node_ids: torch.Tensor
    anchor_node_forward_distances_flat: torch.Tensor
    anchor_node_backward_distances_flat: torch.Tensor
    node_target_distance: torch.Tensor
    node_target_distances_flat: torch.Tensor
    node_target_shortest_path_count_flat: torch.Tensor
    node_target_shortest_path_edge_mask_flat: torch.Tensor
    node_target_shortest_path_edge_count_flat: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReplayPathCandidates:
    edge_ids: torch.Tensor
    lengths: torch.Tensor


@dataclass(frozen=True, slots=True)
class TargetPathLabels:
    target_node_ids: torch.Tensor
    node_target_distances_flat: torch.Tensor
    node_target_shortest_path_count_flat: torch.Tensor
    node_target_shortest_path_edge_mask_flat: torch.Tensor
    node_target_shortest_path_edge_count_flat: torch.Tensor


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
    anchor = _anchor_path_labels(
        graph=graph,
        anchors=anchors,
        num_nodes=num_nodes,
        anchor_distances=anchor_distances,
    )

    num_targets = int(target.target_node_ids.numel())
    return PathLabels(
        reachable_target_node_ids=target.target_node_ids,
        anchor_node_forward_distances_flat=anchor.anchor_node_forward_distances_flat,
        anchor_node_backward_distances_flat=anchor.anchor_node_backward_distances_flat,
        node_target_distance=_nearest_target_distance(
            node_target_distances_flat=target.node_target_distances_flat,
            num_targets=num_targets,
            num_nodes=num_nodes,
        ),
        node_target_distances_flat=target.node_target_distances_flat,
        node_target_shortest_path_count_flat=target.node_target_shortest_path_count_flat,
        node_target_shortest_path_edge_mask_flat=target.node_target_shortest_path_edge_mask_flat,
        node_target_shortest_path_edge_count_flat=target.node_target_shortest_path_edge_count_flat,
    )


def compute_replay_path_candidates(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    reachable_target_node_ids: torch.Tensor,
    node_target_distances_flat: torch.Tensor,
    node_target_shortest_path_edge_count_flat: torch.Tensor,
    num_nodes: int,
    max_trajectories: int,
    max_length: int,
) -> ReplayPathCandidates:
    max_trajectories = int(max_trajectories)
    max_length = int(max_length)
    if max_trajectories <= 0 or max_length < 0:
        return _empty_replay_candidates()

    graph = _graph_from_edge_index(edge_index=edge_index, num_nodes=num_nodes)
    anchors = _valid_unique_nodes(anchor_node_ids, num_nodes=num_nodes)
    targets = _valid_unique_nodes(reachable_target_node_ids, num_nodes=num_nodes)
    if not anchors or not targets:
        return _empty_replay_candidates()

    _ = node_target_distances_flat.to(dtype=torch.long, device="cpu").view(
        len(targets), int(num_nodes)
    )
    edge_counts = node_target_shortest_path_edge_count_flat.to(
        dtype=torch.float32, device="cpu"
    ).view(len(targets), graph.num_edges)

    return _pack_replay_candidates(
        _evidence_subgraph_sequences(
            graph=graph,
            anchors=anchors,
            targets=targets,
            target_edge_mask=edge_counts.gt(0),
            max_sequences=max_trajectories,
            max_length=max_length,
        )
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
    suffix_counts = _suffix_count_matrix(
        adjacency=graph.adjacency,
        targets=reachable_targets,
        target_distances=target_distances,
    )
    edge_mask, edge_counts = _shortest_path_edge_stats(
        graph=graph,
        anchor_distances=anchor_distances,
        targets=reachable_targets,
        target_distances=target_distances,
        suffix_counts=suffix_counts,
    )

    return TargetPathLabels(
        target_node_ids=torch.tensor(reachable_targets, dtype=torch.long),
        node_target_distances_flat=target_distances.reshape(-1).contiguous(),
        node_target_shortest_path_count_flat=suffix_counts.reshape(-1).contiguous(),
        node_target_shortest_path_edge_mask_flat=edge_mask.reshape(-1).contiguous(),
        node_target_shortest_path_edge_count_flat=edge_counts.reshape(-1).contiguous(),
    )


def _evidence_subgraph_sequences(
    *,
    graph: _Graph,
    anchors: Sequence[int],
    targets: Sequence[int],
    target_edge_mask: torch.Tensor,
    max_sequences: int,
    max_length: int,
) -> list[tuple[int, ...]]:
    if max_sequences <= 0 or max_length <= 0:
        return []

    target_set = frozenset(int(target) for target in targets)
    admissible_edge_mask = target_edge_mask.any(dim=0).to(
        dtype=torch.bool,
        device="cpu",
    )
    if not bool(admissible_edge_mask.any()):
        return []

    edge_target_coverage = target_edge_mask.to(dtype=torch.long, device="cpu").sum(
        dim=0
    )
    active0 = frozenset(int(anchor) for anchor in anchors)
    root = _ReplayCandidateState(
        key=(),
        sequence=(),
        active_nodes=active0,
        score=_replay_candidate_score(
            active_nodes=active0,
            targets=target_set,
            edge_count=0,
        ),
        recall=_replay_candidate_recall(active_nodes=active0, targets=target_set),
    )
    states: dict[tuple[int, ...], _ReplayCandidateState] = {root.key: root}
    current_layer = [root]

    for _ in range(int(max_length)):
        next_layer: list[_ReplayCandidateState] = []
        for parent in current_layer:
            for edge_id in _candidate_frontier_edges(
                graph=graph,
                active_nodes=parent.active_nodes,
                selected_edges=parent.key,
                admissible_edge_mask=admissible_edge_mask,
                edge_target_coverage=edge_target_coverage,
            ):
                child_key = tuple(sorted((*parent.key, int(edge_id))))
                if child_key in states:
                    continue
                child_active = frozenset(
                    (
                        *parent.active_nodes,
                        graph.src[int(edge_id)],
                        graph.dst[int(edge_id)],
                    )
                )
                child = _ReplayCandidateState(
                    key=child_key,
                    sequence=(*parent.sequence, int(edge_id)),
                    active_nodes=child_active,
                    score=_replay_candidate_score(
                        active_nodes=child_active,
                        targets=target_set,
                        edge_count=len(child_key),
                    ),
                    recall=_replay_candidate_recall(
                        active_nodes=child_active,
                        targets=target_set,
                    ),
                )
                states[child_key] = child
                next_layer.append(child)
                if len(states) >= _max_replay_state_candidates:
                    return _top_replay_sequences(
                        states.values(),
                        max_sequences=max_sequences,
                    )
        current_layer = next_layer

    return _top_replay_sequences(states.values(), max_sequences=max_sequences)


@dataclass(frozen=True, slots=True)
class _ReplayCandidateState:
    key: tuple[int, ...]
    sequence: tuple[int, ...]
    active_nodes: frozenset[int]
    score: float
    recall: float


def _candidate_frontier_edges(
    *,
    graph: _Graph,
    active_nodes: frozenset[int],
    selected_edges: tuple[int, ...],
    admissible_edge_mask: torch.Tensor,
    edge_target_coverage: torch.Tensor,
) -> list[int]:
    selected = set(selected_edges)
    candidates: list[tuple[int, int]] = []
    for edge_id, src in enumerate(graph.src):
        if edge_id in selected:
            continue
        if int(src) not in active_nodes:
            continue
        if not bool(admissible_edge_mask[edge_id].item()):
            continue
        candidates.append((-int(edge_target_coverage[edge_id].item()), int(edge_id)))
    candidates.sort()
    return [edge_id for _, edge_id in candidates]


def _top_replay_sequences(
    states: Sequence[_ReplayCandidateState],
    *,
    max_sequences: int,
) -> list[tuple[int, ...]]:
    candidates = [state for state in states if state.sequence]
    candidates.sort(
        key=lambda state: (
            -state.score,
            -state.recall,
            len(state.sequence),
            state.sequence,
        )
    )
    return [state.sequence for state in candidates[: int(max_sequences)]]


def _replay_candidate_score(
    *,
    active_nodes: frozenset[int],
    targets: frozenset[int],
    edge_count: int,
) -> float:
    recall = _replay_candidate_recall(active_nodes=active_nodes, targets=targets)
    return math.log(_replay_reward_epsilon + recall) - _replay_edge_cost * float(
        edge_count
    )


def _replay_candidate_recall(
    *,
    active_nodes: frozenset[int],
    targets: frozenset[int],
) -> float:
    if not targets:
        return 0.0
    return float(len(active_nodes & targets)) / float(len(targets))


def _pack_replay_candidates(sequences: Sequence[tuple[int, ...]]) -> ReplayPathCandidates:
    if not sequences:
        return _empty_replay_candidates()
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    if int(lengths.sum().item()) == 0:
        edge_ids = torch.empty((0,), dtype=torch.long)
    else:
        edge_ids = torch.tensor(
            [edge_id for sequence in sequences for edge_id in sequence],
            dtype=torch.long,
        )
    return ReplayPathCandidates(edge_ids=edge_ids.contiguous(), lengths=lengths.contiguous())


def _empty_replay_candidates() -> ReplayPathCandidates:
    return ReplayPathCandidates(
        edge_ids=torch.empty((0,), dtype=torch.long),
        lengths=torch.empty((0,), dtype=torch.long),
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


def _suffix_count_matrix(
    *,
    adjacency: list[list[int]],
    targets: Sequence[int],
    target_distances: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            _shortest_suffix_counts(
                adjacency=adjacency,
                target=target,
                distance_to_target=target_distances[target_idx].tolist(),
            )
            for target_idx, target in enumerate(targets)
        ],
        dim=0,
    )


def _shortest_suffix_counts(
    *,
    adjacency: list[list[int]],
    target: int,
    distance_to_target: Sequence[int],
) -> torch.Tensor:
    counts = torch.zeros(len(adjacency), dtype=torch.float32)
    if distance_to_target[target] != 0:
        return counts

    counts[target] = 1.0
    buckets = _distance_buckets(distance_to_target)
    for distance in range(1, len(buckets)):
        for u in buckets[distance]:
            counts[u] = sum(
                float(counts[v])
                for v in adjacency[u]
                if distance_to_target[v] == distance - 1
            )
    return counts


def _prefix_count_matrix(
    *,
    adjacency: list[list[int]],
    anchor_distances: Sequence[Sequence[int]],
) -> torch.Tensor:
    if not anchor_distances:
        return torch.empty((0, len(adjacency)), dtype=torch.float32)
    return torch.stack(
        [
            _shortest_prefix_counts(
                adjacency=adjacency,
                anchor_to_node_distance=distances,
            )
            for distances in anchor_distances
        ],
        dim=0,
    )


def _shortest_prefix_counts(
    *,
    adjacency: list[list[int]],
    anchor_to_node_distance: Sequence[int],
) -> torch.Tensor:
    counts = torch.zeros(len(adjacency), dtype=torch.float32)
    buckets = _distance_buckets(anchor_to_node_distance)
    if not buckets:
        return counts

    for node in buckets[0]:
        counts[node] = 1.0
    for distance in range(len(buckets) - 1):
        for u in buckets[distance]:
            prefix_count = float(counts[u])
            if prefix_count <= 0.0:
                continue
            for v in adjacency[u]:
                if anchor_to_node_distance[v] == distance + 1:
                    counts[v] += prefix_count
    return counts


def _shortest_path_edge_stats(
    *,
    graph: _Graph,
    anchor_distances: Sequence[Sequence[int]],
    targets: Sequence[int],
    target_distances: torch.Tensor,
    suffix_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_targets = len(targets)
    num_edges = graph.num_edges
    mask = torch.zeros((num_targets, num_edges), dtype=torch.bool)
    counts = torch.zeros((num_targets, num_edges), dtype=torch.float32)
    if num_targets == 0 or num_edges == 0 or not anchor_distances:
        return mask, counts

    anchor_matrix = torch.tensor(anchor_distances, dtype=torch.long)
    prefix_counts = _prefix_count_matrix(
        adjacency=graph.adjacency,
        anchor_distances=anchor_distances,
    )

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
        active_prefix_counts = prefix_counts[active_anchors]
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
            prefix = active_prefix_counts[:, src].to(dtype=torch.float32)
            suffix = suffix_counts[target_idx, dst].view(1, -1)
            counts[target_idx, edge_chunk] = (prefix * suffix * on_path).sum(dim=0)

    return mask, counts.masked_fill(~mask, 0.0)


def _distance_buckets(distances: Sequence[int]) -> list[list[int]]:
    buckets: list[list[int]] = []
    for node, distance in enumerate(distances):
        if distance < 0:
            continue
        while len(buckets) <= distance:
            buckets.append([])
        buckets[distance].append(node)
    return buckets


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
        node_target_shortest_path_count_flat=torch.empty((0,), dtype=torch.float32),
        node_target_shortest_path_edge_mask_flat=torch.empty((0,), dtype=torch.bool),
        node_target_shortest_path_edge_count_flat=torch.empty(
            (0,), dtype=torch.float32
        ),
    )


def _empty_anchor_labels() -> AnchorPathLabels:
    return AnchorPathLabels(
        anchor_node_forward_distances_flat=torch.empty((0,), dtype=torch.long),
        anchor_node_backward_distances_flat=torch.empty((0,), dtype=torch.long),
    )


__all__ = [
    "AnchorPathLabels",
    "PathLabels",
    "ReplayPathCandidates",
    "TargetPathLabels",
    "compute_anchor_path_labels",
    "compute_path_labels",
    "compute_replay_path_candidates",
    "compute_target_path_labels",
    "node_target_unreachable_distance",
    "unreachable_distance",
]
