from __future__ import annotations

import pytest
import torch

from src.graph import TrajectoryBatch
from src.metrics.answer_metrics import (
    ReachabilityAnalysis,
    ReachabilityRanking,
    SearchDiagnostics,
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
from src.metrics.runtime_factory import GraphTaskRuntimeFactory

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
        metric_runtime_factory=GraphTaskRuntimeFactory(),
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
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="rank-only",
    )
    module = _make_module()

    rank_metrics, window_results, model_metrics, diagnostics = module._evaluate_batch(
        batch=batch,
    )

    assert diagnostics["invalid_start_count"] == 0.0
    assert diagnostics["invalid_start_rate"] == 0.0
    assert model_metrics == {}
    assert len(window_results) == 1
    assert window_results[0].stop_reason == "flow_frontier_exhausted"
    assert window_results[0].answer_posterior
    assert window_results[0].trajectories == []
    assert window_results[0].support_mass_reference == "skipped"
    assert window_results[0].answer_mass_reference == "flow_frontier"
    assert "answer/gold_answer_mass" in rank_metrics
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
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="rank-only-predict",
    )
    module = _make_module()

    module.on_predict_epoch_start()
    outputs = module.predict_step(batch, batch_idx=0)
    module.on_predict_batch_end(outputs, batch, batch_idx=0)
    module.on_predict_epoch_end()

    assert len(outputs) == 1
    assert outputs[0].stop_reason == "flow_frontier_exhausted"
    assert outputs[0].trajectories == []
    assert outputs[0].support_mass_reference == "skipped"
    assert outputs[0].answer_mass_reference == "flow_frontier"
    assert "answer/hit@1" in module.predict_metrics
    assert "support/hit" not in module.predict_metrics


def test_rank_only_validation_batches_disconnected_graphs(monkeypatch) -> None:
    torch.manual_seed(17)
    batch_one = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="rank-only-batch-a",
    )
    batch_two = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([202], dtype=torch.long),
        node_entity_ids=torch.tensor([200, 201, 202], dtype=torch.long),
        sample_id="rank-only-batch-b",
    )
    batch = TrajectoryBatch.concatenate([batch_one, batch_two])
    module = _make_module()
    prepare_call_count = 0
    original_prepare_batch = module.policy.prepare_batch

    def _count_prepare(current_batch):  # type: ignore[no-untyped-def]
        nonlocal prepare_call_count
        prepare_call_count += 1
        return original_prepare_batch(current_batch)

    monkeypatch.setattr(module.policy, "prepare_batch", _count_prepare)

    rank_metrics, window_results, model_metrics, diagnostics = module._evaluate_batch(
        batch=batch
    )

    assert diagnostics["invalid_start_count"] == 0.0
    assert model_metrics == {}
    assert len(window_results) == 2
    assert [result.sample_id for result in window_results] == [
        "rank-only-batch-a",
        "rank-only-batch-b",
    ]
    assert prepare_call_count == 1
    assert "answer/gold_answer_mass" in rank_metrics


def test_rank_only_metrics_follow_retrieval_ranking_semantics() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="rank-only-retrieval",
    )
    analysis = ReachabilityAnalysis(
        terminal_mass=torch.tensor([0.0, 0.0, 0.4], dtype=torch.float32),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        answer_probs=torch.tensor([0.4], dtype=torch.float32),
        gold_answer_mass=0.4,
    )
    ranking = ReachabilityRanking(
        answer_entity_ids=torch.tensor([101, 102], dtype=torch.long),
        answer_probs=torch.tensor([0.6, 0.4], dtype=torch.float32),
    )

    result = build_rank_only_result(
        batch=batch,
        analysis=analysis,
        ranking=ranking,
        diagnostics=SearchDiagnostics(
            inference_mode="monte_carlo",
            probe_count=0,
            remaining_mass_upper=0.0,
            stop_reason="rank_only_monte_carlo",
            coverage_certified=False,
        ),
        answer_mass_threshold=0.9,
        support_mass_threshold=0.9,
        answer_mass_reference="monte_carlo",
        answer_mass_reference_total=1.0,
    )
    metrics = compute_rank_metrics(
        answer_records=result.answer_posterior, answer_top_ks=(1, 2)
    )

    assert [record.answer_entity_id for record in result.answer_posterior] == [101, 102]
    assert metrics["answer/gold_answer_mass"] == pytest.approx(0.4)
    assert metrics["answer/hit@1"] == 0.0
    assert metrics["answer/recall@1"] == 0.0
    assert metrics["answer/hit@2"] == 1.0
    assert metrics["answer/recall@2"] == 1.0
