from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
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
            if bool(mask.any().item()):
                out[dest] = torch.logsumexp(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.weaver.state import State


def _build_state(*, active_edges: torch.Tensor, expand_budget: int = 3) -> State:
    return State(
        root_edges=torch.tensor([True, False, False, True, False], dtype=torch.bool),
        active_nodes=torch.tensor(
            [True, True, True, True, False, False], dtype=torch.bool
        ),
        active_edges=active_edges,
        expand_budget=expand_budget,
    )


def test_state_derives_per_graph_budget_from_non_root_edges() -> None:
    state = _build_state(active_edges=torch.tensor([True, True, False, True, False]))
    edge_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    counts = state.per_graph_selected_nonroot_edge_count(
        edge_batch=edge_batch,
        num_graphs=2,
    )
    remaining_budget = state.remaining_budget_per_graph(
        edge_batch=edge_batch,
        num_graphs=2,
    )
    expand_ratio = state.expand_ratio_per_graph(
        edge_batch=edge_batch,
        num_graphs=2,
    )

    assert torch.equal(counts, torch.tensor([1, 0], dtype=torch.long))
    assert torch.equal(remaining_budget, torch.tensor([2, 3], dtype=torch.long))
    assert torch.allclose(
        expand_ratio, torch.tensor([1.0 / 3.0, 0.0], dtype=torch.float32)
    )
    assert not state.is_root_state


def test_synchronous_rollout_depth_checks_unfinished_graphs_only() -> None:
    state = _build_state(active_edges=torch.tensor([True, True, False, True, True]))
    edge_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    assert (
        state.synchronous_rollout_depth(
            edge_batch=edge_batch,
            num_graphs=2,
            active_graphs=torch.tensor([True, True]),
        )
        == 1
    )

    mismatched = _build_state(
        active_edges=torch.tensor([True, True, False, True, False])
    )
    with pytest.raises(RuntimeError, match="Synchronous rollout depth"):
        mismatched.synchronous_rollout_depth(
            edge_batch=edge_batch,
            num_graphs=2,
            active_graphs=torch.tensor([True, True]),
        )

    assert (
        mismatched.synchronous_rollout_depth(
            edge_batch=edge_batch,
            num_graphs=2,
            active_graphs=torch.tensor([True, False]),
        )
        == 1
    )


def test_state_detach_clones_root_active_edges() -> None:
    state = _build_state(active_edges=torch.tensor([True, False, False, True, False]))

    detached = state.detach()
    detached.root_edges[0] = False

    assert bool(state.root_edges[0])
    assert not bool(detached.root_edges[0])


def test_create_initial_rejects_out_of_range_anchor_ids_by_default() -> None:
    batch = types.SimpleNamespace(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        anchor_node_ids=torch.tensor([0, 3], dtype=torch.long),
        num_nodes_total=3,
    )

    with pytest.raises(ValueError, match="anchor_node_ids"):
        State.create_initial(batch, expand_budget=1)
