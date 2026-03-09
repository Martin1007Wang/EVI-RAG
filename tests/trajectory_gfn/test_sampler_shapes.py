from __future__ import annotations

import torch

from src.models.configs.trajectory_gfn import HorizonConfig, TrajectoryTrainingConfig
from src.models.trajectory_gfn.reward import TrajectoryReward
from src.models.trajectory_gfn.sampler import ForwardRolloutSampler

from .conftest import make_policy, make_toy_batch


def test_sampler_emits_consistent_shapes() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    context = policy.encode(batch)
    sampler = ForwardRolloutSampler(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        training_cfg=TrajectoryTrainingConfig(rollout_batch_size=2),
        reward=TrajectoryReward(epsilon=1.0e-3),
    )
    sample_batch = sampler.sample(batch=batch, policy=policy, context=context)
    assert sample_batch.start_nodes.shape == (1, 2)
    assert sample_batch.log_pf_steps.shape == (1, 2, 3)
    assert sample_batch.log_pb_steps.shape == (1, 2, 3)
    assert sample_batch.chosen_edge_ids_steps.shape == (1, 2, 3)
    assert sample_batch.stop_nodes.shape == (1, 2)
    stop_counts = (sample_batch.active_steps & sample_batch.is_stop_steps).sum(dim=-1)
    assert torch.equal(stop_counts, torch.ones_like(stop_counts))
