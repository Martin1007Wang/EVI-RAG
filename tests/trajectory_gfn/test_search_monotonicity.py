from __future__ import annotations

from src.models.configs.trajectory_gfn import HorizonConfig, TrajectoryInferenceConfig
from src.models.trajectory_gfn.analyzer import AnswerMassAnalyzer
from src.models.trajectory_gfn.search import MassAdaptiveTrajectorySearch

from .conftest import make_policy, make_toy_batch


def test_search_monotonicity() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    context = policy.encode(batch)
    analyzer = AnswerMassAnalyzer(max_steps=2, min_stop_steps=1)
    search_lo = MassAdaptiveTrajectorySearch(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        inference_cfg=TrajectoryInferenceConfig(
            answer_mass_threshold=0.5,
            support_mass_threshold=1.0,
            max_expansions=128,
            max_frontier_size=128,
        ),
        analyzer=analyzer,
    )
    search_hi = MassAdaptiveTrajectorySearch(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        inference_cfg=TrajectoryInferenceConfig(
            answer_mass_threshold=0.9,
            support_mass_threshold=1.0,
            max_expansions=128,
            max_frontier_size=128,
        ),
        analyzer=analyzer,
    )
    result_lo = search_lo.generate_window(batch=batch, policy=policy, context=context)
    result_hi = search_hi.generate_window(batch=batch, policy=policy, context=context)
    assert result_hi.window_size >= result_lo.window_size
    assert result_hi.covered_mass >= result_lo.covered_mass
    assert result_hi.missed_gold_mass <= result_lo.missed_gold_mass + 1.0e-6
