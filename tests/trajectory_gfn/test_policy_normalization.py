from __future__ import annotations

import torch

from src.models.trajectory_gfn.reward import TrajectoryReward
from src.models.trajectory_gfn.state import TrajectoryState
from src.models.trajectory_gfn.transition import apply_forward_constraints

from .conftest import make_policy, make_toy_batch


def test_policy_normalization() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    context = policy.encode(batch)

    start_dist = policy.compute_start_distribution(context)
    assert torch.allclose(
        start_dist.log_probs.exp().sum(), torch.tensor(1.0), atol=1.0e-5
    )

    state = TrajectoryState(
        step_t=0,
        current_node=torch.tensor([[0, 1]], dtype=torch.long),
        done_mask=torch.zeros((1, 2), dtype=torch.bool),
        num_moves=torch.zeros((1, 2), dtype=torch.long),
    )
    target_mask = TrajectoryReward(epsilon=1.0e-3).build_target_mask(batch)
    distribution = policy.compute_forward_distribution(context, state)
    distribution = apply_forward_constraints(
        distribution,
        state=state,
        node_is_target=target_mask,
        min_stop_steps=1,
        max_steps=2,
    )
    move_log_probs, stop_log_probs, _ = policy.compute_forward_log_probs(distribution)
    total_agents = int(distribution.out_degrees.numel())
    for agent_idx in range(total_agents):
        move_mass = 0.0
        mask = distribution.edge_agent_batch == agent_idx
        if int(mask.sum().item()) > 0:
            move_mass = float(move_log_probs[mask].exp().sum().item())
        total_mass = move_mass + float(stop_log_probs[agent_idx].exp().item())
        assert abs(total_mass - 1.0) < 1.0e-5

    next_state = TrajectoryState(
        step_t=1,
        current_node=torch.tensor([[1, 2]], dtype=torch.long),
        done_mask=torch.zeros((1, 2), dtype=torch.bool),
        num_moves=torch.ones((1, 2), dtype=torch.long),
        path_nodes=torch.tensor([[[0, 1, -1], [0, 2, -1]]], dtype=torch.long),
        path_edge_ids=torch.tensor([[[0, -1], [1, -1]]], dtype=torch.long),
    )
    backward = policy.compute_backward_distribution(context, next_state)
    for agent_idx in range(2):
        mask = backward.edge_agent_batch == agent_idx
        total_mass = float(backward.parent_log_probs[mask].exp().sum().item())
        assert abs(total_mass - 1.0) < 1.0e-5
