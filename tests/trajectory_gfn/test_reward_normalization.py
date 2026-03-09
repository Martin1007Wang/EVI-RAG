from __future__ import annotations

import torch

from src.models.trajectory_gfn.reward import TrajectoryReward

from .conftest import make_batch_from_graph


def test_graph_normalized_wrong_stop_reward() -> None:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([3], dtype=torch.long),
        answer_entity_ids=torch.tensor([103], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
    )
    reward = TrajectoryReward(
        epsilon=1.0e-3,
        wrong_stop_reward_mode="graph_normalized",
    )
    stop_nodes = torch.tensor([[0, 2, 3]], dtype=torch.long).expand(2, 3)
    _, rewards, _ = reward.compute(batch=batch, stop_nodes=stop_nodes)
    assert torch.allclose(rewards[:, :2], torch.full((2, 2), 1.0e-3 / 3.0))
    assert torch.allclose(rewards[:, 2], torch.ones((2,)))
