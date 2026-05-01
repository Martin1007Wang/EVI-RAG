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
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        if dim == 0:
            out_shape = (size,) + tuple(src.shape[1:])
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            for row, dest in enumerate(index.tolist()):
                out[dest] += src[row]
            return out
        if dim == 1:
            out_shape = (src.shape[0], size) + tuple(src.shape[2:])
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            for row in range(src.shape[0]):
                for col, dest in enumerate(index[row].tolist()):
                    out[row, dest] += src[row, col]
            return out
        raise NotImplementedError("test stub only supports dim=0 or dim=1")

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

from src.data.collate import RetrievalCollator
from src.data.schema import RetrievalData
from src.weaver.reward import RewardModel


def _batch(
    *,
    edge_index: torch.Tensor,
    anchor_node_ids: torch.Tensor,
    target_node_ids: torch.Tensor,
    reachable_target_node_ids: torch.Tensor,
) -> object:
    data = RetrievalData(
        num_nodes=3,
        edge_index=edge_index,
        question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
        anchor_node_ids=anchor_node_ids,
        target_node_ids=target_node_ids,
        reachable_target_node_ids=reachable_target_node_ids,
    )
    return RetrievalCollator()([data])


def test_reward_requires_anchor_connected_answer_support() -> None:
    batch = _batch(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([1], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
    )
    active_nodes = torch.tensor([True, True, False])
    active_edges = torch.tensor([False])

    reward = RewardModel(edge_cost=0.0).evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=active_nodes,
        active_edges=active_edges,
    )

    assert reward.supported_answer_recall.item() == pytest.approx(0.0)
    assert reward.utility.item() == pytest.approx(0.0)
    assert reward.answer_recall.item() == pytest.approx(1.0)


def test_reward_formula_uses_supported_recall_and_nonroot_edge_prior() -> None:
    batch = _batch(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([1], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
    )
    active_nodes = torch.tensor([True, True, False])
    active_edges = torch.tensor([True])

    reward = RewardModel(edge_cost=0.2, utility_epsilon=1.0e-4).evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=active_nodes,
        active_edges=active_edges,
    )

    expected = torch.log(torch.tensor(1.0 + 1.0e-4)) - 0.2
    assert reward.supported_answer_recall.item() == pytest.approx(1.0)
    assert reward.expanded_edge_count.item() == pytest.approx(1.0)
    assert reward.complexity_penalty.item() == pytest.approx(0.2)
    assert reward.log_reward.item() == pytest.approx(expected.item())


def test_reward_does_not_charge_anchor_induced_root_edges() -> None:
    batch = _batch(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        anchor_node_ids=torch.tensor([0, 1], dtype=torch.long),
        target_node_ids=torch.tensor([2], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([2], dtype=torch.long),
    )
    active_nodes = torch.tensor([True, True, True])
    active_edges = torch.tensor([True, True])

    reward = RewardModel(edge_cost=0.5).evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=active_nodes,
        active_edges=active_edges,
    )

    assert reward.supported_answer_recall.item() == pytest.approx(1.0)
    assert reward.expanded_edge_count.item() == pytest.approx(1.0)
    assert reward.complexity_penalty.item() == pytest.approx(0.5)


def test_reward_uses_empty_reachable_targets_without_all_answer_fallback() -> None:
    batch = _batch(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([1], dtype=torch.long),
        reachable_target_node_ids=torch.empty(0, dtype=torch.long),
    )
    active_nodes = torch.tensor([True, True, False])
    active_edges = torch.tensor([True])

    reward = RewardModel(edge_cost=0.0, utility_epsilon=1.0e-4).evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=active_nodes,
        active_edges=active_edges,
    )

    assert reward.reward_answer_count.item() == pytest.approx(0.0)
    assert reward.supported_answer_count.item() == pytest.approx(0.0)
    assert reward.utility.item() == pytest.approx(0.0)
    expected_base = torch.log(torch.tensor(1.0e-4)).item()
    assert reward.base_log_reward.item() == pytest.approx(expected_base)
    assert not hasattr(reward, "minimal_edge_count")
