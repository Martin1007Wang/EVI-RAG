from __future__ import annotations

from src.metrics.answer_metrics import (
    AnswerPosteriorRecord,
    EdgeRecord,
    SupportWindowResult,
    TrajectoryRecord,
    aggregate_rank_metrics,
    compute_support_metrics,
)


def test_elastic_metrics_compute_expected_fields() -> None:
    result = SupportWindowResult(
        sample_id="toy",
        dataset_scope="sub",
        mass_threshold=0.9,
        window_size=2,
        covered_mass=0.9,
        residual_mass=0.1,
        gold_answer_mass=0.7,
        covered_gold_answer_mass=0.6,
        missed_gold_answer_mass=0.1,
        unique_answer_count=2,
        unique_path_count=2,
        gold_answer_entity_ids=[102],
        start_entity_ids=[100],
        trajectories=[
            TrajectoryRecord(
                sample_id="toy",
                path_rank=1,
                log_prob=-0.5,
                prob=0.6,
                cumulative_mass=0.6,
                terminal_entity_id=102,
                is_gold=True,
                edges=[
                    EdgeRecord(
                        edge_id=0, src_entity_id=100, relation_id=1, dst_entity_id=102
                    )
                ],
            ),
            TrajectoryRecord(
                sample_id="toy",
                path_rank=2,
                log_prob=-1.2,
                prob=0.3,
                cumulative_mass=0.9,
                terminal_entity_id=101,
                is_gold=False,
                edges=[
                    EdgeRecord(
                        edge_id=1, src_entity_id=100, relation_id=2, dst_entity_id=101
                    )
                ],
            ),
        ],
        answer_posterior=[
            AnswerPosteriorRecord(
                answer_entity_id=102,
                prob=0.6,
                cumulative_mass=0.6,
                is_gold=True,
                is_selected=True,
                support_mass=0.6,
                support_conditioned_mass=1.0,
                support_path_count=1,
            ),
            AnswerPosteriorRecord(
                answer_entity_id=101,
                prob=0.3,
                cumulative_mass=0.9,
                is_gold=False,
                is_selected=False,
            ),
        ],
    )
    metrics = compute_support_metrics([result])
    assert metrics["support/hit"] == 1.0
    assert metrics["support/path_mass"] == 0.9
    assert metrics["support/missed_gold_answer_mass"] == 0.1
    assert metrics["support/diversity"] == 1.0
    assert metrics["search/coverage_rate"] == 0.0
    assert "elastic_hit" not in metrics

    rank_metrics = aggregate_rank_metrics(results=[result], answer_top_ks=(1, 2))
    assert rank_metrics["answer/hit@1"] == 1.0
    assert rank_metrics["answer/recall@2"] == 1.0
    assert "pass@1" not in rank_metrics
