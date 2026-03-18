from __future__ import annotations

import pytest
import torch

from src.metrics.answer_reachability.exact_analysis import ExactReachabilityAnalysis
from src.metrics.answer_reachability.posterior import (
    build_rank_only_result,
    compute_rank_metrics,
)
from src.models.configs import (
    BackboneConfig,
    SearchEvalConfig,
    GFlowNetTrainingConfig,
    HeuristicConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
    StateScoreHeadConfig,
)
from src.models.gflownet_module import GFlowNetModule
from src.metrics.answer_reachability.runtime import (
    SearchMetricRuntimeFactory,
)

from .conftest import make_batch_from_graph


def _make_module() -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(),
        heuristic_cfg=HeuristicConfig(beta=0.0),
        policy_cfg=PolicyConfig(
            backbone=BackboneConfig(
                embedding_dim=8,
                hidden_dim=8,
                gnn_layers=1,
                gnn_dropout=0.0,
                use_adapter=True,
                adapter_dim=4,
                adapter_dropout=0.0,
            ),
            state_score_head=StateScoreHeadConfig(
                hidden_dim=8,
                num_layers=2,
                dropout=0.0,
            ),
        ),
        eval_cfg=SearchEvalConfig(
            metrics_profile="rank_only",
            answer_mass_threshold=0.9,
            support_mass_threshold=0.9,
            answer_top_ks=(1, 5),
            max_expansions=32,
            max_frontier_size=32,
        ),
        optimizer_cfg=OptimizerConfig(),
        scheduler_cfg=SchedulerConfig(),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )


def test_rank_only_validation_skips_support_search() -> None:
    torch.manual_seed(7)
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="rank-only",
    )
    module = _make_module()

    def _fail_search(**_: object) -> None:
        raise AssertionError("support search should be skipped in rank-only validation")

    module.search.generate_window = _fail_search  # type: ignore[assignment]

    support_metrics, window_results, model_metrics, rank_metrics = (
        module._evaluate_batch(
            batch=batch,
        )
    )

    assert support_metrics == {}
    assert model_metrics == {}
    assert len(window_results) == 1
    assert window_results[0].stop_reason == "rank_only_exact"
    assert window_results[0].answer_posterior
    assert window_results[0].trajectories == []
    assert window_results[0].support_mass_reference == "skipped"
    assert "answer/gold_mass" in rank_metrics
    assert "answer/hit@1" in rank_metrics


def test_rank_only_predict_skips_support_search_and_support_metrics() -> None:
    torch.manual_seed(11)
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="rank-only-predict",
    )
    module = _make_module()

    def _fail_search(**_: object) -> None:
        raise AssertionError("support search should be skipped in rank-only predict")

    module.search.generate_window = _fail_search  # type: ignore[assignment]

    module.on_predict_epoch_start()
    outputs = module.predict_step(batch, batch_idx=0)
    module.on_predict_batch_end(outputs, batch, batch_idx=0)
    module.on_predict_epoch_end()

    assert len(outputs) == 1
    assert outputs[0].stop_reason == "rank_only_exact"
    assert outputs[0].trajectories == []
    assert outputs[0].support_mass_reference == "skipped"
    assert "answer/hit@1" in module.predict_metrics
    assert "window/adaptive/hit" not in module.predict_metrics


def test_rank_only_metrics_follow_retrieval_ranking_semantics() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="rank-only-retrieval",
    )
    analysis = ExactReachabilityAnalysis(
        terminal_mass=torch.tensor([0.0, 0.0, 0.4], dtype=torch.float32),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        answer_probs=torch.tensor([0.4], dtype=torch.float32),
        gold_total_mass=0.4,
        retrieval_answer_entity_ids=torch.tensor([101, 102], dtype=torch.long),
        retrieval_answer_probs=torch.tensor([0.6, 0.4], dtype=torch.float32),
    )

    result = build_rank_only_result(
        batch=batch,
        analysis=analysis,
        inference_mode="exact",
        answer_mass_threshold=0.9,
        support_mass_threshold=0.9,
        probe_count=0,
        remaining_mass_upper=0.0,
        stop_reason="rank_only_exact",
        coverage_certified=True,
        answer_mass_reference="exact",
        answer_mass_reference_total=1.0,
    )
    metrics = compute_rank_metrics(
        answer_records=result.answer_posterior, answer_top_ks=(1, 2)
    )

    assert [record.answer_entity_id for record in result.answer_posterior] == [101, 102]
    assert metrics["answer/gold_mass"] == pytest.approx(0.4)
    assert metrics["answer/hit@1"] == 0.0
    assert metrics["answer/recall@1"] == 0.0
    assert metrics["answer/precision@1"] == 0.0
    assert metrics["answer/f1@1"] == 0.0
    assert metrics["answer/hit@2"] == 1.0
    assert metrics["answer/recall@2"] == 1.0
    assert metrics["answer/precision@2"] == 0.5
    assert metrics["answer/f1@2"] == 2.0 / 3.0
