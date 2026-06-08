from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib

import torch

from src.graph.paths import unreachable_distance

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ShortestPathDag:
    pair_anchor_node_ids: Tensor
    pair_target_node_ids: Tensor
    pair_distance: Tensor
    pair_edge_ids: Tensor
    pair_edge_depth: Tensor
    pair_edge_ptr: Tensor


@dataclass(frozen=True, slots=True)
class ReplayBank:
    edge_ids: Tensor
    edge_count: Tensor
    priority: Tensor


@dataclass(frozen=True, slots=True)
class _Candidate:
    edges: frozenset[int]
    targets: frozenset[int]
    pairs: frozenset[int]


@dataclass(frozen=True, slots=True)
class _CandidatePath:
    edges: frozenset[int]
    targets: frozenset[int]


def build_shortest_path_dag(
    *,
    edge_index: Tensor,
    anchor_node_ids: Tensor,
    reachable_target_node_ids: Tensor,
    num_nodes: int,
) -> ShortestPathDag:
    edge_index = edge_index.to(dtype=torch.long, device="cpu").contiguous()
    anchors = _unique_valid(anchor_node_ids, num_nodes=num_nodes)
    targets = _unique_valid(reachable_target_node_ids, num_nodes=num_nodes)
    outgoing = _adjacency(edge_index=edge_index, num_nodes=num_nodes, reverse=False)
    incoming = _adjacency(edge_index=edge_index, num_nodes=num_nodes, reverse=True)

    pair_anchors: list[int] = []
    pair_targets: list[int] = []
    pair_distances: list[int] = []
    flat_edge_ids: list[int] = []
    flat_edge_depths: list[int] = []
    edge_ptr = [0]

    forward_by_anchor = {anchor: _bfs(outgoing, anchor) for anchor in anchors}
    backward_by_target = {target: _bfs(incoming, target) for target in targets}
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for anchor in anchors:
        forward = forward_by_anchor[anchor]
        for target in targets:
            distance = int(forward[target])
            if distance == unreachable_distance:
                continue
            backward = backward_by_target[target]
            dag_edges = [
                (int(forward[u]), edge_id)
                for edge_id, (u, v) in enumerate(zip(src, dst, strict=True))
                if int(forward[u]) != unreachable_distance
                and int(backward[v]) != unreachable_distance
                and int(forward[u]) + 1 + int(backward[v]) == distance
            ]
            dag_edges.sort()
            pair_anchors.append(anchor)
            pair_targets.append(target)
            pair_distances.append(distance)
            flat_edge_depths.extend(depth for depth, _ in dag_edges)
            flat_edge_ids.extend(edge_id for _, edge_id in dag_edges)
            edge_ptr.append(len(flat_edge_ids))

    return ShortestPathDag(
        pair_anchor_node_ids=_long(pair_anchors),
        pair_target_node_ids=_long(pair_targets),
        pair_distance=_long(pair_distances),
        pair_edge_ids=_long(flat_edge_ids),
        pair_edge_depth=_long(flat_edge_depths),
        pair_edge_ptr=_long(edge_ptr),
    )


def build_replay_bank(
    *,
    edge_index: Tensor,
    anchor_node_ids: Tensor,
    reachable_target_node_ids: Tensor,
    num_nodes: int,
    sample_id: str,
    max_edges: int | None = None,
    max_budget: int | None = None,
    round_variants: int,
    trajectories_per_graph: int,
    beam_width: int,
    path_variants_per_pair: int,
    max_expansions_per_state: int,
    seed: int,
) -> ReplayBank:
    if max_edges is None:
        if max_budget is None:
            raise ValueError("max_edges is required.")
        max_edges = int(max_budget)
    elif max_budget is not None and int(max_budget) != int(max_edges):
        raise ValueError("max_budget and max_edges must agree when both are provided.")
    max_edges = int(max_edges)
    round_variants = int(round_variants)
    trajectories_per_graph = int(trajectories_per_graph)
    for name, value in (
        ("max_edges", max_edges),
        ("round_variants", round_variants),
        ("trajectories_per_graph", trajectories_per_graph),
        ("beam_width", beam_width),
        ("path_variants_per_pair", path_variants_per_pair),
        ("max_expansions_per_state", max_expansions_per_state),
    ):
        if int(value) <= 0 and name != "max_edges":
            raise ValueError(f"{name} must be positive.")
        if name == "max_edges" and int(value) < 0:
            raise ValueError("max_edges must be nonnegative.")

    edge_index = edge_index.to(dtype=torch.long, device="cpu").contiguous()
    anchors = set(_unique_valid(anchor_node_ids, num_nodes=num_nodes))
    targets = set(_unique_valid(reachable_target_node_ids, num_nodes=num_nodes))
    dag = build_shortest_path_dag(
        edge_index=edge_index,
        anchor_node_ids=anchor_node_ids,
        reachable_target_node_ids=reachable_target_node_ids,
        num_nodes=num_nodes,
    )
    edge_ids = torch.full(
        (round_variants, trajectories_per_graph, max_edges),
        -1,
        dtype=torch.long,
    )
    edge_count = torch.full(
        (round_variants, trajectories_per_graph),
        -1,
        dtype=torch.long,
    )
    priority = torch.full(
        (round_variants, trajectories_per_graph),
        float("-inf"),
        dtype=torch.float32,
    )
    paths_by_variant = [
        [
            [
                _CandidatePath(
                    edges=frozenset(path),
                    targets=frozenset(_covered_targets(edges=frozenset(path), anchors=anchors, targets=targets, edge_index=edge_index)),
                )
                for path in _pair_path_variants(
                    dag=dag,
                    edge_index=edge_index,
                    pair_id=pair_id,
                    limit=int(path_variants_per_pair),
                    token=_token(seed, sample_id, variant, pair_id),
                )
            ]
            for pair_id in range(int(dag.pair_distance.numel()))
        ]
        for variant in range(round_variants)
    ]
    for variant in range(round_variants):
        candidates = _plan_candidates(
            dag=dag,
            anchors=anchors,
            targets=targets,
            sample_id=sample_id,
            replay_round=variant,
            budget=max_edges,
            trajectories_per_graph=trajectories_per_graph,
            beam_width=int(beam_width),
            max_expansions_per_state=int(max_expansions_per_state),
            seed=int(seed),
            paths_by_pair=paths_by_variant[variant],
        )
        selected = _select_submodular_set(
            candidates=candidates,
            anchors=anchors,
            targets=targets,
            limit=trajectories_per_graph,
            token=_token(seed, sample_id, variant),
        )
        for slot, (candidate, score) in enumerate(selected):
            ordered = _frontier_legal_order(
                edge_ids=candidate.edges,
                anchors=anchors,
                edge_index=edge_index,
                token=_token(seed, sample_id, variant, slot),
            )
            if ordered is None:
                continue
            _validate_replay_trajectory(
                ordered=ordered,
                anchors=anchors,
                edge_index=edge_index,
                sample_id=sample_id,
                replay_round=variant,
                slot=slot,
            )
            if ordered:
                edge_ids[variant, slot, : len(ordered)] = torch.tensor(ordered)
            edge_count[variant, slot] = len(ordered)
            priority[variant, slot] = float(score)
    return ReplayBank(edge_ids=edge_ids.contiguous(), edge_count=edge_count.contiguous(), priority=priority.contiguous())


def _plan_candidates(
    *,
    dag: ShortestPathDag,
    anchors: set[int],
    targets: set[int],
    sample_id: str,
    replay_round: int,
    budget: int,
    trajectories_per_graph: int,
    beam_width: int,
    max_expansions_per_state: int,
    seed: int,
    paths_by_pair: list[list[_CandidatePath]],
) -> list[_Candidate]:
    initial_targets = targets & anchors
    root = _Candidate(edges=frozenset(), targets=frozenset(initial_targets), pairs=frozenset())
    beam = [root]
    pool: dict[frozenset[int], _Candidate] = {root.edges: root} if initial_targets else {}
    for _ in range(max(int(budget), 1)):
        expanded = list(beam)
        for state in beam:
            additions: list[_Candidate] = []
            for pair_id in range(int(dag.pair_distance.numel())):
                if pair_id in state.pairs:
                    continue
                for path in paths_by_pair[pair_id]:
                    edges = state.edges | path.edges
                    if len(edges) > budget:
                        continue
                    covered = state.targets | path.targets
                    if covered == state.targets:
                        continue
                    additions.append(_Candidate(edges=edges, targets=frozenset(covered), pairs=state.pairs | {pair_id}))
            additions.sort(key=lambda item: _quality_key(item, token=_token(seed, sample_id, replay_round, len(state.edges))))
            expanded.extend(additions[:max_expansions_per_state])
        dedup = {item.edges: item for item in expanded}
        beam = sorted(dedup.values(), key=lambda item: _quality_key(item, token=_token(seed, sample_id, replay_round, len(item.edges))))[:beam_width]
        for item in beam:
            if item.targets - initial_targets:
                pool[item.edges] = item
    return list(pool.values())


def _pair_path_variants(*, dag: ShortestPathDag, edge_index: Tensor, pair_id: int, limit: int, token: int) -> list[tuple[int, ...]]:
    start = int(dag.pair_edge_ptr[pair_id].item())
    end = int(dag.pair_edge_ptr[pair_id + 1].item())
    target = int(dag.pair_target_node_ids[pair_id].item())
    by_src: dict[int, list[int]] = {}
    for edge_id in dag.pair_edge_ids[start:end].tolist():
        by_src.setdefault(int(edge_index[0, edge_id].item()), []).append(int(edge_id))
    out: list[tuple[int, ...]] = []

    def visit(node: int, path: tuple[int, ...]) -> None:
        if len(out) >= limit:
            return
        if node == target:
            out.append(path)
            return
        for edge_id in sorted(by_src.get(node, ()), key=lambda value: _stable_rank(token, value)):
            visit(int(edge_index[1, edge_id].item()), (*path, edge_id))

    visit(int(dag.pair_anchor_node_ids[pair_id].item()), tuple())
    return out


def _diverse_top_k(candidates: list[_Candidate], *, limit: int, token: int) -> list[_Candidate]:
    selected: list[_Candidate] = []
    remaining = list(candidates)
    while remaining and len(selected) < limit:
        def key(item: _Candidate) -> tuple[int, int, int, int, int]:
            overlap = max((len(item.edges & prior.edges) for prior in selected), default=0)
            pair_overlap = max((len(item.pairs & prior.pairs) for prior in selected), default=0)
            return (-len(item.targets), len(item.edges), overlap, pair_overlap, _stable_rank(token, *sorted(item.edges)))
        remaining.sort(key=key)
        selected.append(remaining.pop(0))
    return selected


def _select_submodular_set(
    *,
    candidates: list[_Candidate],
    anchors: set[int],
    targets: set[int],
    limit: int,
    token: int,
) -> list[tuple[_Candidate, float]]:
    remaining = list(candidates)
    selected: list[_Candidate] = []
    selected_with_gain: list[tuple[_Candidate, float]] = []
    if limit <= 0:
        return selected_with_gain
    quality_weight = 1.0
    diversity_weight = 0.75
    length_weight = 0.25
    best_similarity = [0.0] * len(remaining)
    covered_lengths: set[int] = set()
    while remaining and len(selected) < limit:
        best_idx = -1
        best_gain = float("-inf")
        for idx, item in enumerate(remaining):
            gain = quality_weight * _quality_score(item=item, anchors=anchors, targets=targets)
            gain += diversity_weight * _diversity_gain(candidate=item, universe=remaining, current_best=best_similarity)
            gain += length_weight * (0.0 if len(item.edges) in covered_lengths else 1.0)
            gain += 1.0e-6 / float(_stable_rank(token, *sorted(item.edges)) + 1)
            if gain > best_gain:
                best_idx = idx
                best_gain = gain
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_with_gain.append((chosen, float(best_gain)))
        covered_lengths.add(len(chosen.edges))
        for idx, candidate in enumerate(remaining):
            best_similarity[idx] = max(
                best_similarity[idx],
                _edge_jaccard(candidate.edges, chosen.edges),
            )
    return selected_with_gain


def _quality_score(*, item: _Candidate, anchors: set[int], targets: set[int]) -> float:
    answer_count = float(len(item.targets))
    target_count = float(max(len(targets), 1))
    recall = answer_count / target_count
    coverage = torch.log(torch.tensor(answer_count * torch.exp(torch.tensor(2.0)).item() + 1.0)).item()
    return 6.0 * recall + coverage - 0.15 * float(len(item.edges))


def _diversity_gain(*, candidate: _Candidate, universe: list[_Candidate], current_best: list[float]) -> float:
    gain = 0.0
    for idx, item in enumerate(universe):
        sim = _edge_jaccard(candidate.edges, item.edges)
        gain += max(0.0, sim - current_best[idx])
    return gain


def _edge_jaccard(lhs: frozenset[int], rhs: frozenset[int]) -> float:
    if not lhs and not rhs:
        return 1.0
    union = lhs | rhs
    if not union:
        return 0.0
    return float(len(lhs & rhs)) / float(len(union))


def _quality_key(item: _Candidate, *, token: int) -> tuple[int, int, int]:
    return (-len(item.targets), len(item.edges), _stable_rank(token, *sorted(item.edges)))


def _frontier_legal_order(*, edge_ids: frozenset[int], anchors: set[int], edge_index: Tensor, token: int) -> list[int] | None:
    active = set(anchors)
    remaining = set(edge_ids)
    ordered: list[int] = []
    while remaining:
        legal = [edge_id for edge_id in remaining if int(edge_index[0, edge_id].item()) in active]
        if not legal:
            return None
        edge_id = min(legal, key=lambda value: _stable_rank(token, value))
        ordered.append(edge_id)
        active.add(int(edge_index[1, edge_id].item()))
        remaining.remove(edge_id)
    return ordered


def _validate_replay_trajectory(
    *,
    ordered: list[int],
    anchors: set[int],
    edge_index: Tensor,
    sample_id: str,
    replay_round: int,
    slot: int,
) -> None:
    active = set(anchors)
    for step, edge_id in enumerate(ordered):
        src = int(edge_index[0, edge_id].item())
        dst = int(edge_index[1, edge_id].item())
        if src not in active:
            raise ValueError(
                "Replay trajectory is not frontier-legal: "
                f"sample_id={sample_id}, replay_round={replay_round}, slot={slot}, "
                f"step={step}, edge_id={edge_id}, src={src}, active={sorted(active)}."
            )
        active.add(dst)


def _covered_targets(*, edges: frozenset[int], anchors: set[int], targets: set[int], edge_index: Tensor) -> set[int]:
    nodes = set(anchors)
    for edge_id in edges:
        nodes.add(int(edge_index[0, edge_id].item()))
        nodes.add(int(edge_index[1, edge_id].item()))
    return nodes & targets


def _token(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()[:8], "big")


def _stable_rank(token: int, *values: int) -> int:
    return _token(token, *values)


def _unique_valid(node_ids: Tensor, *, num_nodes: int) -> list[int]:
    return sorted({int(node) for node in node_ids.view(-1).tolist() if 0 <= int(node) < int(num_nodes)})


def _adjacency(*, edge_index: Tensor, num_nodes: int, reverse: bool) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(int(num_nodes))]
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True):
        u, v = (int(dst), int(src)) if reverse else (int(src), int(dst))
        adjacency[u].append(v)
    return adjacency


def _bfs(adjacency: list[list[int]], start: int) -> list[int]:
    distances = [unreachable_distance] * len(adjacency)
    distances[int(start)] = 0
    queue: deque[int] = deque([int(start)])
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1
        for neighbor in adjacency[node]:
            if distances[neighbor] != unreachable_distance:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def _long(values: list[int]) -> Tensor:
    return torch.tensor(values, dtype=torch.long).contiguous()


__all__ = ["ReplayBank", "ShortestPathDag", "build_replay_bank", "build_shortest_path_dag"]
