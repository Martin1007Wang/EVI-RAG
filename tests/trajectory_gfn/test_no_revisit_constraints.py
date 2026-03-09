from __future__ import annotations

import torch

from src.models.trajectory_gfn.reward import TrajectoryReward
from src.models.trajectory_gfn.state import TrajectoryState
from src.models.trajectory_gfn.transition import apply_forward_constraints

from .conftest import make_batch_from_graph, make_policy


def test_forward_constraints_mask_revisits() -> None:
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
    context = policy.encode(batch)
    state = TrajectoryState(
        step_t=1,
        current_node=torch.tensor([[1]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_moves=torch.ones((1, 1), dtype=torch.long),
        path_nodes=torch.tensor([[[0, 1, -1]]], dtype=torch.long),
        path_edge_ids=torch.tensor([[[0, -1]]], dtype=torch.long),
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
    move_log_probs, _, _ = policy.compute_forward_log_probs(distribution)
    revisit_mask = distribution.target_nodes == 0
    fresh_mask = distribution.target_nodes == 2
    assert bool(revisit_mask.any().item())
    assert bool(fresh_mask.any().item())
    assert not torch.isfinite(move_log_probs[revisit_mask]).any()
    assert torch.isfinite(move_log_probs[fresh_mask]).all()
