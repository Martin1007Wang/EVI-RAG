from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch_scatter")

from src.data.schema import RetrievalBatch
from src.models.policy import PolicyStepOutput
from src.models.rollout import RolloutEngine
from src.utils.graph_utils import compute_valid_backward_removals


def _compute_removals(
    *,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    active_edges: torch.Tensor,
    root_active_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_nodes = is_anchor_mask.clone()
    if bool(active_edges.any().item()):
        src = edge_index[0][active_edges]
        dst = edge_index[1][active_edges]
        active_nodes[src] = True
        active_nodes[dst] = True

    num_nodes = int(is_anchor_mask.numel())
    return compute_valid_backward_removals(
        active_nodes=active_nodes,
        active_edges=active_edges,
        root_active_edges=root_active_edges,
        edge_index=edge_index,
        is_anchor_mask=is_anchor_mask,
        node_batch=torch.zeros(num_nodes, dtype=torch.long),
        edge_batch=torch.zeros(edge_index.size(1), dtype=torch.long),
        num_graphs=1,
    )


def test_valid_backward_removals_reject_anchor_disconnect_in_chain() -> None:
    removable_mask, removable_counts = _compute_removals(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        is_anchor_mask=torch.tensor([True, False, False]),
        active_edges=torch.tensor([True, True]),
        root_active_edges=torch.tensor([False, False]),
    )

    assert removable_mask.tolist() == [False, True]
    assert removable_counts.tolist() == [1]


def test_valid_backward_removals_allow_cycle_edge_deletions() -> None:
    removable_mask, removable_counts = _compute_removals(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        is_anchor_mask=torch.tensor([True, False, False]),
        active_edges=torch.tensor([True, True, True]),
        root_active_edges=torch.tensor([False, False, False]),
    )

    assert removable_mask.tolist() == [True, True, True]
    assert removable_counts.tolist() == [3]


def test_valid_backward_removals_allow_multi_anchor_split() -> None:
    removable_mask, removable_counts = _compute_removals(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        is_anchor_mask=torch.tensor([True, False, True]),
        active_edges=torch.tensor([True, True]),
        root_active_edges=torch.tensor([False, False]),
    )

    assert removable_mask.tolist() == [True, True]
    assert removable_counts.tolist() == [2]


class _DeterministicPolicy:
    def __call__(
        self,
        batch: RetrievalBatch,
        state: object,
    ) -> PolicyStepOutput:
        del state
        num_edges = int(batch.edge_index.size(1))
        num_graphs = int(batch.ptr.numel() - 1)
        return PolicyStepOutput(
            action_logits={
                "type_logits": torch.tensor(
                    [[20.0, -20.0]], device=batch.node_tokens.device
                ).repeat(num_graphs, 1),
                "expand_edge_logits": torch.zeros(
                    num_edges, device=batch.node_tokens.device
                ),
            },
            question_h=torch.zeros((num_graphs, 1), device=batch.node_tokens.device),
            subgraph_h=torch.zeros((num_graphs, 1), device=batch.node_tokens.device),
        )

    def root_log_z(
        self,
        question_h: torch.Tensor,
        root_subgraph_h: torch.Tensor,
    ) -> torch.Tensor:
        del root_subgraph_h
        return torch.zeros(1, device=question_h.device)


class _ZeroReward:
    def __call__(
        self,
        *,
        base_graph: RetrievalBatch,
        active_nodes: torch.Tensor,
        active_edges: torch.Tensor,
        root_active_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del active_nodes, active_edges, root_active_edges
        return torch.zeros(base_graph.num_graphs, device=base_graph.node_tokens.device)


def _build_chain_batch() -> RetrievalBatch:
    batch = RetrievalBatch()
    batch.node_tokens = torch.zeros((3, 1))
    batch.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    batch.batch = torch.tensor([0, 0, 0], dtype=torch.long)
    batch.edge_batch = torch.tensor([0, 0], dtype=torch.long)
    batch.ptr = torch.tensor([0, 3], dtype=torch.long)
    batch.is_anchor_mask = torch.tensor([True, False, False])
    batch.num_nodes = 3
    return batch


def _build_edgeless_batch() -> RetrievalBatch:
    batch = RetrievalBatch()
    batch.node_tokens = torch.zeros((1, 1))
    batch.edge_index = torch.empty((2, 0), dtype=torch.long)
    batch.batch = torch.tensor([0], dtype=torch.long)
    batch.edge_batch = torch.empty((0,), dtype=torch.long)
    batch.ptr = torch.tensor([0, 1], dtype=torch.long)
    batch.is_anchor_mask = torch.tensor([True])
    batch.num_nodes = 1
    return batch


def _build_mixed_batch() -> RetrievalBatch:
    batch = RetrievalBatch()
    batch.node_tokens = torch.zeros((4, 1))
    batch.edge_index = torch.tensor([[1, 2], [2, 3]], dtype=torch.long)
    batch.batch = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    batch.edge_batch = torch.tensor([1, 1], dtype=torch.long)
    batch.ptr = torch.tensor([0, 1, 4], dtype=torch.long)
    batch.is_anchor_mask = torch.tensor([True, True, False, False])
    batch.num_nodes = 4
    return batch


def test_rollout_backward_log_prob_uses_only_valid_parents() -> None:
    rollout = RolloutEngine(max_steps=2)._run_exploration_once(
        policy=_DeterministicPolicy(),
        base_graph=_build_chain_batch(),
        reward_model=_ZeroReward(),
        temperature=1.0,
        collect_terminal_state=False,
    )

    assert rollout.trajectory_log_pb.tolist() == pytest.approx([0.0])


def test_rollout_forces_legal_stop_after_expand_budget() -> None:
    rollout = RolloutEngine(max_steps=2)._run_exploration_once(
        policy=_DeterministicPolicy(),
        base_graph=_build_chain_batch(),
        reward_model=_ZeroReward(),
        temperature=1.0,
        collect_terminal_state=True,
    )

    assert rollout.termination_action_steps.tolist() == [3]
    assert rollout.trajectory_log_pf.tolist() == pytest.approx([0.0])
    assert rollout.trajectory_log_pb.tolist() == pytest.approx([0.0])
    assert rollout.terminal_active_edges is not None
    assert rollout.terminal_active_edges.tolist() == [True, True]


def test_rollout_zero_expand_budget_stops_at_root() -> None:
    rollout = RolloutEngine(max_steps=0)._run_exploration_once(
        policy=_DeterministicPolicy(),
        base_graph=_build_chain_batch(),
        reward_model=_ZeroReward(),
        temperature=1.0,
        collect_terminal_state=True,
    )

    assert rollout.termination_action_steps.tolist() == [1]
    assert rollout.trajectory_log_pf.tolist() == pytest.approx([0.0])
    assert rollout.terminal_active_edges is not None
    assert rollout.terminal_active_edges.tolist() == [False, False]


def test_rollout_mask_value_is_safe_for_half_precision() -> None:
    mask_value = RolloutEngine(1)._masked_logit_value(
        torch.empty(1, dtype=torch.float16)
    )

    assert mask_value == torch.finfo(torch.float16).min


def test_rollout_edgeless_graph_forces_stop_without_expand_sampling() -> None:
    rollout = RolloutEngine(max_steps=1)._run_exploration_once(
        policy=_DeterministicPolicy(),
        base_graph=_build_edgeless_batch(),
        reward_model=_ZeroReward(),
        temperature=1.0,
        collect_terminal_state=True,
    )

    assert rollout.termination_action_steps.tolist() == [1]
    assert rollout.trajectory_log_pf.tolist() == pytest.approx([0.0])
    assert rollout.terminal_active_edges is not None
    assert rollout.terminal_active_edges.numel() == 0


def test_rollout_partial_expand_batch_keeps_log_prob_alignment() -> None:
    rollout = RolloutEngine(max_steps=1)._run_exploration_once(
        policy=_DeterministicPolicy(),
        base_graph=_build_mixed_batch(),
        reward_model=_ZeroReward(),
        temperature=1.0,
        collect_terminal_state=True,
    )

    assert rollout.termination_action_steps.tolist() == [1, 2]
    assert rollout.trajectory_log_pf.tolist() == pytest.approx([0.0, 0.0])
    assert rollout.terminal_active_edges is not None
    assert rollout.terminal_active_edges.tolist() == [True, True]
