from __future__ import annotations

import torch

from src.models.policy.state import SearchState
from src.models.policy.transition import compute_constrained_forward_step

from .conftest import make_batch_from_graph, make_policy, make_toy_batch


def test_compute_constrained_forward_step_normalizes_per_agent() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0, 1]], dtype=torch.long),
        done_mask=torch.zeros((1, 2), dtype=torch.bool),
        num_steps=torch.zeros((1, 2), dtype=torch.long),
    )

    step = compute_constrained_forward_step(
        policy=policy,
        prepared_batch=prepared_batch,
        state=state,
        max_steps=2,
    )

    total_agents = int(step.distribution.out_degrees.numel())
    for agent_idx in range(total_agents):
        if not bool(step.has_values[agent_idx].item()):
            continue
        mask = step.distribution.edge_agent_batch == agent_idx
        move_mass = float(step.move_probs[mask].sum().item())
        assert abs(move_mass - 1.0) < 1.0e-5


def test_compute_constrained_forward_step_masks_horizon_agents() -> None:
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

    step = compute_constrained_forward_step(
        policy=policy,
        prepared_batch=prepared_batch,
        state=state,
        max_steps=2,
    )

    assert bool(step.has_values.item()) is False
    assert not torch.isfinite(step.move_log_probs).any()
