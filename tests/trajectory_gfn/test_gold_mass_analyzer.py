from __future__ import annotations

from src.models.trajectory_gfn.analyzer import AnswerMassAnalyzer

from .conftest import make_policy, make_toy_batch


def test_gold_mass_analyzer_conserves_terminal_mass() -> None:
    batch = make_toy_batch()
    policy = make_policy()
    context = policy.encode(batch)
    analyzer = AnswerMassAnalyzer(max_steps=2, min_stop_steps=1)
    analysis = analyzer.analyze(batch=batch, policy=policy, context=context)
    assert abs(float(analysis.terminal_mass.sum().item()) - 1.0) < 1.0e-5
    assert (
        abs(analysis.gold_total_mass - float(analysis.terminal_mass[2].item())) < 1.0e-6
    )
