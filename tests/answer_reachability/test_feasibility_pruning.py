from __future__ import annotations

import pytest
import torch

from src.metrics.search_backends import FlowFrontierBackend
from src.models.gflownet import BaseSearchPolicy
from src.models.gflownet import apply_forward_constraints
from src.models.gflownet import SearchState
from src.models.configs import SearchEvalConfig

from .conftest import make_batch_from_graph, make_policy


def _make_policy(*, max_steps: int) -> BaseSearchPolicy:
    return make_policy(max_steps=max_steps)


def test_answer_start_without_future_support_is_absorbing_success() -> None:
    batch = make_batch_from_graph(
        num_nodes=1,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0], dtype=torch.long),
        answer_entity_ids=torch.tensor([100], dtype=torch.long),
        node_entity_ids=torch.tensor([100], dtype=torch.long),
        sample_id="answer-start-dead-end",
    )
    policy = _make_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    backend = FlowFrontierBackend(
        max_steps=2,
        eval_cfg=SearchEvalConfig(
            support_search_method="flow_frontier",
            flow_prune_epsilon=0.0,
        ),
    )

    distribution = policy.compute_start_distribution(prepared_batch)
    result = backend.evaluate_graph(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
        metrics_profile="rank_only",
        include_answer_support=False,
    )

    assert torch.isfinite(distribution.log_probs).all()
    assert result.gold_answer_mass == pytest.approx(1.0)


def test_cycle_only_start_becomes_stop_only_after_revisit_is_masked() -> None:
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="cycle-only-start",
    )
    policy = _make_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0,),
        max_steps=2,
        device=batch.node_ptr.device,
    )

    distribution = policy.compute_forward_distribution(prepared_batch, state)
    constrained = apply_forward_constraints(
        distribution,
        state=state,
        max_steps=2,
    )
    move_log_probs, _, has_values = policy.compute_move_log_probs(constrained)
    stop_mask = constrained.is_stop_action

    assert stop_mask is not None
    assert bool(has_values.item()) is True
    assert torch.isfinite(move_log_probs[stop_mask]).all()
    assert not torch.isfinite(move_log_probs[~stop_mask]).any()


def test_forward_distribution_keeps_edges_into_future_failures() -> None:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="future-dead-end",
    )
    policy = _make_policy(max_steps=3)
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
    )

    distribution = policy.compute_forward_distribution(prepared_batch, state)
    constrained = apply_forward_constraints(
        distribution,
        state=state,
        max_steps=3,
    )
    move_log_probs, _, _ = policy.compute_move_log_probs(constrained)

    assert 1 in {int(node) for node in constrained.target_nodes.tolist()}
    assert 2 in {int(node) for node in constrained.target_nodes.tolist()}
    assert bool(torch.isfinite(move_log_probs).all().item())
