from __future__ import annotations

import torch

from src.models.trajectory_gfn.state import TrajectoryState

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
    context = policy.encode(batch)

    state_t0 = TrajectoryState(
        step_t=0,
        current_node=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_moves=torch.tensor([[0]], dtype=torch.long),
    )
    state_t1 = TrajectoryState(
        step_t=1,
        current_node=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_moves=torch.tensor([[1]], dtype=torch.long),
        path_nodes=torch.tensor([[[0, 0, -1]]], dtype=torch.long),
        path_edge_ids=torch.tensor([[[0, -1]]], dtype=torch.long),
    )

    log_flow_t0 = policy.compute_log_flow(context, state_t0)
    log_flow_t1 = policy.compute_log_flow(context, state_t1)
    dist_t0 = policy.compute_forward_distribution(context, state_t0)
    dist_t1 = policy.compute_forward_distribution(context, state_t1)

    assert not torch.allclose(log_flow_t0, log_flow_t1)
    assert not torch.allclose(dist_t0.edge_logits, dist_t1.edge_logits)
