from __future__ import annotations

from pathlib import Path
import sys
import types

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
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
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
            if bool(mask.any().item()):
                out[dest] = torch.logsumexp(src[mask], dim=0)
        return out

    def _scatter_softmax(
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
        out = torch.zeros_like(src)
        for dest in range(size):
            mask = index == dest
            if bool(mask.any().item()):
                out[mask] = torch.softmax(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    torch_scatter_stub.scatter_softmax = _scatter_softmax
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.graph.ops import compute_uniform_nonroot_backward_removals


def test_backward_removals_exclude_edges_that_break_reachability() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    active_edges = torch.tensor([True, True, True])
    root_active_edges = torch.tensor([True, False, False])
    anchor_mask = torch.tensor([True, True, False, False])
    edge_batch = torch.zeros(3, dtype=torch.long)

    removable_mask, removable_counts = compute_uniform_nonroot_backward_removals(
        active_edges=active_edges,
        edge_index=edge_index,
        is_anchor_mask=anchor_mask,
        edge_batch=edge_batch,
        num_graphs=1,
        root_active_edges=root_active_edges,
    )

    assert torch.equal(removable_mask, torch.tensor([False, False, True]))
    assert torch.equal(removable_counts, torch.tensor([1]))


def test_backward_removal_count_matches_number_of_valid_parents() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    edge_batch = torch.zeros(3, dtype=torch.long)
    anchor_mask = torch.tensor([True, True, False, False])
    root_active_edges = torch.tensor([True, False, False])
    active_edges = torch.tensor([True, True, True])

    removable_mask, removable_counts = compute_uniform_nonroot_backward_removals(
        active_edges=active_edges,
        edge_index=edge_index,
        is_anchor_mask=anchor_mask,
        edge_batch=edge_batch,
        num_graphs=1,
        root_active_edges=root_active_edges,
    )

    assert torch.equal(removable_mask, torch.tensor([False, True, True]))
    assert torch.equal(removable_counts, torch.tensor([2]))
