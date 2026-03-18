from __future__ import annotations

import torch

from src.models.gflownet import apply_forward_constraints
from src.models.gflownet import SearchState

from .conftest import make_policy, make_toy_batch


def test_policy_normalization() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)

    start_dist = policy.compute_start_distribution(prepared_batch)
    assert torch.allclose(
        start_dist.log_probs.exp().sum(), torch.tensor(1.0), atol=1.0e-5
    )

    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0, 1]], dtype=torch.long),
        done_mask=torch.zeros((1, 2), dtype=torch.bool),
        num_steps=torch.zeros((1, 2), dtype=torch.long),
    )
    distribution = policy.compute_forward_distribution(prepared_batch, state)
    distribution = apply_forward_constraints(
        distribution,
        state=state,
        max_steps=2,
    )
    move_log_probs, _, has_values = policy.compute_move_log_probs(distribution)
    total_agents = int(distribution.out_degrees.numel())
    for agent_idx in range(total_agents):
        if not bool(has_values[agent_idx].item()):
            continue
        mask = distribution.edge_agent_batch == agent_idx
        move_mass = float(move_log_probs[mask].exp().sum().item())
        assert abs(move_mass - 1.0) < 1.0e-5
