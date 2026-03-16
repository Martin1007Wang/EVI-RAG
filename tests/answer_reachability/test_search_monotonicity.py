from __future__ import annotations

from src.metrics.answer_reachability import ExactReachabilityAnalyzer
from src.metrics.answer_reachability.search import ReachabilityGuidedSearch
from src.models.configs import AnswerReachabilityInferenceConfig, HorizonConfig

from .conftest import make_policy, make_toy_batch


def test_search_monotonicity() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    analyzer = ExactReachabilityAnalyzer(max_steps=2)
    search_lo = ReachabilityGuidedSearch(
        horizon_cfg=HorizonConfig(max_steps=2),
        inference_cfg=AnswerReachabilityInferenceConfig(
            answer_mass_threshold=0.5,
            support_mass_threshold=1.0,
            max_expansions=128,
            max_frontier_size=128,
        ),
        analyzer=analyzer,
    )
    search_hi = ReachabilityGuidedSearch(
        horizon_cfg=HorizonConfig(max_steps=2),
        inference_cfg=AnswerReachabilityInferenceConfig(
            answer_mass_threshold=0.9,
            support_mass_threshold=1.0,
            max_expansions=128,
            max_frontier_size=128,
        ),
        analyzer=analyzer,
    )
    result_lo = search_lo.generate_window(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
    )
    result_hi = search_hi.generate_window(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
    )
    assert result_hi.window_size >= result_lo.window_size
    assert result_hi.covered_mass >= result_lo.covered_mass
    assert result_hi.missed_gold_mass <= result_lo.missed_gold_mass + 1.0e-6
