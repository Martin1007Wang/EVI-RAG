from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.analyze_budget_recall_oracle import analyze_sample


def sample(
    *,
    edge_index: list[tuple[int, int]],
    anchors: list[int],
    targets: list[int],
) -> SimpleNamespace:
    num_nodes = max([0, *anchors, *targets, *(x for edge in edge_index for x in edge)]) + 1
    edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    distances = []
    adjacency_reverse: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edge_index:
        adjacency_reverse[dst].append(src)
    for target in targets:
        dist = [-1] * num_nodes
        queue = [target]
        dist[target] = 0
        for node in queue:
            for prev in adjacency_reverse[node]:
                if dist[prev] == -1:
                    dist[prev] = dist[node] + 1
                    queue.append(prev)
        distances.extend(dist)
    return SimpleNamespace(
        sample_id="toy",
        edge_index=edge_tensor,
        anchor_node_ids=torch.tensor(anchors, dtype=torch.long),
        reachable_target_node_ids=torch.tensor(targets, dtype=torch.long),
        node_target_distances_flat=torch.tensor(distances, dtype=torch.long),
        num_nodes=num_nodes,
    )


def run(sample_obj, budgets: list[int]):
    return analyze_sample(
        sample=sample_obj,
        budgets=budgets,
        max_paths_per_target=64,
        max_dp_states=10000,
    )


def test_disjoint_two_hop_targets_need_six_edges_for_full_cover() -> None:
    graph = sample(
        edge_index=[
            (0, 1),
            (1, 2),
            (0, 3),
            (3, 4),
            (0, 5),
            (5, 6),
        ],
        anchors=[0],
        targets=[2, 4, 6],
    )

    result = run(graph, [2, 3, 4, 6])

    assert result.recall_by_budget[2] == 1 / 3
    assert result.recall_by_budget[3] == 1 / 3
    assert result.recall_by_budget[4] == 2 / 3
    assert result.recall_by_budget[6] == 1.0
    assert result.b_hit == 2.0
    assert result.b_cover100 == 6.0


def test_shared_prefix_counts_shared_edge_once() -> None:
    graph = sample(
        edge_index=[
            (0, 1),
            (1, 2),
            (1, 3),
        ],
        anchors=[0],
        targets=[2, 3],
    )

    result = run(graph, [2, 3, 4])

    assert result.recall_by_budget[2] == 0.5
    assert result.recall_by_budget[3] == 1.0
    assert result.b_cover100 == 3.0


def test_anchor_target_has_positive_recall_at_budget_zero() -> None:
    graph = sample(
        edge_index=[
            (0, 1),
            (1, 2),
        ],
        anchors=[0],
        targets=[0, 2],
    )

    result = run(graph, [0, 1, 2])

    assert result.recall_by_budget[0] == 0.5
    assert result.b_hit == 0.0
    assert result.recall_by_budget[2] == 1.0


def test_intermediate_target_on_path_is_covered() -> None:
    graph = sample(
        edge_index=[
            (0, 1),
            (1, 2),
        ],
        anchors=[0],
        targets=[1, 2],
    )

    result = run(graph, [1, 2])

    assert result.recall_by_budget[1] == 0.5
    assert result.recall_by_budget[2] == 1.0
