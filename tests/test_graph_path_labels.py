from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if "torch_scatter" not in sys.modules:
    torch_scatter_stub = types.ModuleType("torch_scatter")

    def _scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out_shape = (size,) + tuple(src.shape[1:])
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] += src[row]
        return out

    def _scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.full((size,), -float("inf"), dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

    def _scatter_logsumexp(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.full((size,), -torch.inf, dtype=src.dtype, device=src.device)
        for dest in range(size):
            mask = index == dest
            if bool(mask.any()):
                out[dest] = torch.logsumexp(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.graph.paths import (
    compute_path_labels,
    compute_target_path_labels,
    node_target_unreachable_distance,
)


def test_target_path_labels_count_suffixes_and_mark_shortest_edges() -> None:
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [1, 2, 3, 3],
        ],
        dtype=torch.long,
    )

    labels = compute_target_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([3], dtype=torch.long),
        num_nodes=4,
    )

    assert torch.equal(labels.target_node_ids, torch.tensor([3], dtype=torch.long))
    assert torch.equal(
        labels.target_node_distances_flat.view(1, 4),
        torch.tensor([[2, 1, 1, 0]], dtype=torch.long),
    )
    assert torch.equal(
        labels.target_shortest_path_count_flat.view(1, 4),
        torch.tensor([[2.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
    )
    assert torch.equal(
        labels.target_shortest_path_edge_mask_flat.view(1, 4),
        torch.tensor([[True, True, True, True]], dtype=torch.bool),
    )


def test_target_path_labels_keep_target_order_with_multiple_anchors() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3],
            [1, 4, 4, 3, 5],
        ],
        dtype=torch.long,
    )

    labels = compute_target_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0, 2], dtype=torch.long),
        target_node_ids=torch.tensor([5, 4], dtype=torch.long),
        num_nodes=6,
    )

    assert torch.equal(labels.target_node_ids, torch.tensor([5, 4], dtype=torch.long))
    assert torch.equal(
        labels.target_shortest_path_edge_mask_flat.view(2, 5),
        torch.tensor(
            [
                [False, False, False, True, True],
                [True, True, True, False, False],
            ],
            dtype=torch.bool,
        ),
    )


def test_path_labels_return_empty_target_labels_when_targets_are_unreachable() -> None:
    edge_index = torch.tensor(
        [
            [0],
            [1],
        ],
        dtype=torch.long,
    )

    labels = compute_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([3], dtype=torch.long),
        num_nodes=4,
    )

    assert labels.reachable_target_node_ids.numel() == 0
    assert labels.target_node_distances_flat.numel() == 0
    assert labels.target_shortest_path_count_flat.numel() == 0
    assert labels.target_shortest_path_edge_mask_flat.numel() == 0
    assert torch.equal(
        labels.node_target_distance,
        torch.full((4,), node_target_unreachable_distance, dtype=torch.long),
    )
    assert torch.equal(
        labels.anchor_node_forward_distances_flat,
        torch.tensor([0, 1, -1, -1], dtype=torch.long),
    )
