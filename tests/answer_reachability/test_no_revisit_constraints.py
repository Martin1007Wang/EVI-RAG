from __future__ import annotations

import torch

from src.models.gflownet import apply_forward_constraints
from src.models.gflownet import SearchState

from .conftest import make_batch_from_graph, make_policy


def test_forward_constraints_only_mask_horizon() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[1]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.ones((1, 1), dtype=torch.long),
    )
    distribution = policy.compute_forward_distribution(prepared_batch, state)
    distribution = apply_forward_constraints(
        distribution,
        state=state,
        max_steps=2,
    )
    move_log_probs, _, _ = policy.compute_move_log_probs(distribution)
    revisit_mask = distribution.target_nodes == 0
    fresh_mask = distribution.target_nodes == 2
    assert bool(fresh_mask.any().item())
    assert bool(revisit_mask.any().item())
    assert torch.isfinite(move_log_probs[revisit_mask]).all()
    assert torch.isfinite(move_log_probs[fresh_mask]).all()


def test_forward_constraints_mask_all_moves_at_horizon() -> None:
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101], dtype=torch.long),
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.full((1, 1), 2, dtype=torch.long),
    )

    distribution = policy.compute_forward_distribution(prepared_batch, state)
    distribution = apply_forward_constraints(
        distribution,
        state=state,
        max_steps=2,
    )
    move_log_probs, _, has_values = policy.compute_move_log_probs(distribution)

    assert bool(has_values.item()) is False
    assert not torch.isfinite(move_log_probs).any()
