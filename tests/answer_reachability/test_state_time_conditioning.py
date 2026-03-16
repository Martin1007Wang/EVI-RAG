from __future__ import annotations

import torch

from src.models.policy.state import SearchState

from .conftest import make_policy, make_toy_batch


def test_same_node_changes_with_time_budget() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    with torch.no_grad():
        policy.step_embedding.weight.zero_()
        policy.remaining_embedding.weight.zero_()
        policy.step_embedding.weight[0, 0] = -3.0
        policy.step_embedding.weight[1, 0] = 3.0
        policy.remaining_embedding.weight[2, 1] = 2.0
        policy.remaining_embedding.weight[1, 1] = -2.0
    prepared_batch = policy.prepare_batch(batch)

    state_t0 = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )
    state_t1 = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[1]], dtype=torch.long),
    )

    score_t0 = policy.compute_log_state_scores(prepared_batch, state_t0)
    score_t1 = policy.compute_log_state_scores(prepared_batch, state_t1)
    dist_t0 = policy.compute_forward_distribution(prepared_batch, state_t0)
    dist_t1 = policy.compute_forward_distribution(prepared_batch, state_t1)

    assert not torch.allclose(score_t0, score_t1)
    assert not torch.allclose(dist_t0.edge_logits, dist_t1.edge_logits)


def test_same_node_and_time_ignore_prefix_history() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)

    state_path_a = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[2]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[1]], dtype=torch.long),
    )
    state_path_b = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[2]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[1]], dtype=torch.long),
    )

    score_a = policy.compute_log_state_scores(prepared_batch, state_path_a)
    score_b = policy.compute_log_state_scores(prepared_batch, state_path_b)

    assert torch.allclose(score_a, score_b)
