from __future__ import annotations

import torch

from src.graph_runtime import TrajectoryBatch
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


def test_gold_mass_analyzer_matches_single_graph_results_on_disconnected_batch() -> (
    None
):
    first_batch = make_toy_batch()
    second_batch = make_toy_batch()
    batch = TrajectoryBatch.concatenate([first_batch, second_batch])
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    analyzer = ExactReachabilityAnalyzer(max_steps=2)

    batched_result = analyzer.compute_dynamic_program(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
    )
    first_result = analyzer.compute_dynamic_program(
        batch=first_batch,
        policy=policy,
        prepared_batch=policy.prepare_batch(first_batch),
    )
    second_result = analyzer.compute_dynamic_program(
        batch=second_batch,
        policy=policy,
        prepared_batch=policy.prepare_batch(second_batch),
    )

    assert torch.allclose(
        batched_result.log_gold_mass_by_graph,
        torch.stack((first_result.log_gold_mass, second_result.log_gold_mass)),
        atol=1.0e-6,
    )
    assert torch.allclose(
        batched_result.log_terminal_mass[: first_batch.num_nodes_total],
        first_result.log_terminal_mass,
        atol=1.0e-6,
    )
    assert torch.allclose(
        batched_result.log_terminal_mass[first_batch.num_nodes_total :],
        second_result.log_terminal_mass,
        atol=1.0e-6,
    )
