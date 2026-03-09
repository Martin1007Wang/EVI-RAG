from __future__ import annotations

from src.models.trajectory_gfn.metrics import compute_elastic_metrics
from src.models.trajectory_gfn.schema import (
    EdgeRecord,
    ElasticEvalBatch,
    ElasticWindowResult,
    TrajectoryRecord,
)


def test_elastic_metrics_compute_expected_fields() -> None:
    result = ElasticWindowResult(
        sample_id="toy",
        dataset_scope="sub",
        mass_threshold=0.9,
        window_size=2,
        covered_mass=0.9,
        tail_rollout_mass=0.1,
        gold_total_mass=0.7,
        covered_gold_mass=0.6,
        missed_gold_mass=0.1,
        unique_answer_count=2,
        unique_path_count=2,
        gold_answer_entity_ids=[102],
        start_entity_ids=[100],
        trajectories=[
            TrajectoryRecord(
                sample_id="toy",
                rollout_rank=1,
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
                rollout_rank=2,
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
    )
    metrics = compute_elastic_metrics(
        ElasticEvalBatch(dataset_scope="sub", mass_threshold=0.9, results=[result])
    )
    assert metrics["elastic_hit"] == 1.0
    assert metrics["elastic_mass"] == 0.9
    assert metrics["missed_gold_mass"] == 0.1
    assert "path_entropy@90" in metrics
    assert "support_path_diversity@90" in metrics
