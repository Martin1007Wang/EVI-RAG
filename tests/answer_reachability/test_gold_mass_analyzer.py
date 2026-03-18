from __future__ import annotations

from src.metrics.answer_reachability import ExactReachabilityAnalyzer

from .conftest import make_policy, make_toy_batch


def test_gold_mass_analyzer_conserves_terminal_mass() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    analyzer = ExactReachabilityAnalyzer(max_steps=2)
    analysis = analyzer.analyze(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
    )
    assert abs(float(analysis.terminal_mass.sum().item()) - 1.0) < 1.0e-5
    assert (
        abs(analysis.gold_total_mass - float(analysis.terminal_mass[2].item())) < 1.0e-6
    )
    assert analysis.retrieval_answer_probs is not None
    assert abs(float(analysis.retrieval_answer_probs.sum().item()) - 1.0) < 1.0e-5
