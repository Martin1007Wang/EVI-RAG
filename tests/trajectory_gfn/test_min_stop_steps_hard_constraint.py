from __future__ import annotations

import pytest

from src.models.configs.trajectory_gfn import (
    HorizonConfig,
    TrajectoryInferenceConfig,
    TrajectoryTrainingConfig,
)
from src.models.trajectory_gfn.analyzer import AnswerMassAnalyzer
from src.models.trajectory_gfn.reward import TrajectoryReward
from src.models.trajectory_gfn.sampler import ForwardRolloutSampler
from src.models.trajectory_gfn.search import MassAdaptiveTrajectorySearch

from .conftest import make_dead_end_batch, make_policy


def test_dead_end_before_min_stop_steps_is_invalid() -> None:
    batch = make_dead_end_batch()
    policy = make_policy()
    context = policy.encode(batch)

    sampler = ForwardRolloutSampler(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        training_cfg=TrajectoryTrainingConfig(rollout_batch_size=1),
        reward=TrajectoryReward(epsilon=1.0e-3),
    )
    analyzer = AnswerMassAnalyzer(max_steps=2, min_stop_steps=1)
    search = MassAdaptiveTrajectorySearch(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        inference_cfg=TrajectoryInferenceConfig(
            answer_mass_threshold=0.9,
            support_mass_threshold=0.9,
            max_expansions=32,
            max_frontier_size=32,
        ),
        analyzer=analyzer,
    )

    with pytest.raises(ValueError, match="valid start candidate|empty forward support"):
        sampler.sample(batch=batch, policy=policy, context=context)
    with pytest.raises(ValueError, match="valid start candidate|empty forward support"):
        analyzer.analyze(batch=batch, policy=policy, context=context)
    with pytest.raises(ValueError, match="valid start candidate|empty forward support"):
        search.generate_window(batch=batch, policy=policy, context=context)
