from __future__ import annotations

import torch

from src.graph.paths import compute_path_labels


def test_compute_path_labels_reports_reachable_target_max_distance() -> None:
    edge_index = torch.tensor([[0, 0, 1, 2, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
    labels = compute_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([3, 4], dtype=torch.long),
        num_nodes=5,
    )

    assert torch.equal(labels.reachable_target_node_ids, torch.tensor([3, 4], dtype=torch.long))
    assert torch.equal(labels.node_target_distance, torch.tensor([1, 1, 1, 0, 0], dtype=torch.long))
    assert torch.equal(labels.edge_on_shortest_path, torch.tensor([True, False, True, False, True], dtype=torch.bool))
    assert labels.reachable_target_max_distance == 2


def test_compute_path_labels_marks_only_edges_on_anchor_answer_shortest_paths() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 0, 3, 4],
            [1, 2, 3, 4, 2],
        ],
        dtype=torch.long,
    )
    labels = compute_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([2], dtype=torch.long),
        num_nodes=5,
    )

    assert torch.equal(labels.edge_on_shortest_path, torch.tensor([True, True, False, False, False], dtype=torch.bool))
