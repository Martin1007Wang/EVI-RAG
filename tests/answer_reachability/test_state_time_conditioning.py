from __future__ import annotations

import pytest
import torch

from src.models.gflownet import SearchState

from .conftest import make_batch_from_graph, make_policy, make_toy_batch


def test_same_node_changes_with_time_budget() -> None:
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="time-conditioned-cycle",
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)

    state_t0 = SearchState.initialize(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        max_steps=2,
    )
    state_t2 = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0, 1),
        max_steps=2,
        device=batch.node_ptr.device,
    )

    score_t0 = policy.compute_log_state_scores(prepared_batch, state_t0)
    score_t2 = policy.compute_log_state_scores(prepared_batch, state_t2)
    dist_t0 = policy.compute_forward_distribution(prepared_batch, state_t0)
    dist_t2 = policy.compute_forward_distribution(prepared_batch, state_t2)

    assert torch.equal(state_t0.current_nodes, state_t2.current_nodes)
    assert not torch.allclose(score_t0, score_t2)
    assert not torch.allclose(dist_t0.edge_logits, dist_t2.edge_logits)


def test_same_node_and_time_depends_on_prefix_history() -> None:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([3], dtype=torch.long),
        answer_entity_ids=torch.tensor([103], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="merged-prefix",
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)

    state_path_a = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0, 2),
        max_steps=2,
        device=batch.node_ptr.device,
    )
    state_path_b = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(1, 3),
        max_steps=2,
        device=batch.node_ptr.device,
    )

    score_a = policy.compute_log_state_scores(prepared_batch, state_path_a)
    score_b = policy.compute_log_state_scores(prepared_batch, state_path_b)
    dist_a = policy.compute_backward_distribution(prepared_batch, state_path_a)
    dist_b = policy.compute_backward_distribution(prepared_batch, state_path_b)

    assert not torch.allclose(score_a, score_b)
    assert not torch.allclose(dist_a.edge_logits, dist_b.edge_logits)


def test_start_log_flows_match_state_scores_at_time_zero() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    start_dist = policy.compute_start_distribution(prepared_batch)
    candidate_nodes_abs = start_dist.candidate_nodes_abs

    state_t0 = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=candidate_nodes_abs.view(1, -1),
        done_mask=torch.zeros((1, int(candidate_nodes_abs.numel())), dtype=torch.bool),
        num_steps=torch.zeros((1, int(candidate_nodes_abs.numel())), dtype=torch.long),
    )

    start_log_flows = policy.compute_start_log_flows(
        prepared_batch=prepared_batch,
        candidate_nodes_abs=candidate_nodes_abs,
    )
    state_scores = policy.compute_log_state_scores(prepared_batch, state_t0).view(-1)

    assert torch.allclose(start_log_flows, state_scores)


def test_non_root_state_without_path_history_is_rejected() -> None:
    batch = make_toy_batch()
    policy = make_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    invalid_state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[1]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.ones((1, 1), dtype=torch.long),
    )

    with pytest.raises(ValueError, match="exact path_token_ids"):
        policy.compute_log_state_scores(prepared_batch, invalid_state)
