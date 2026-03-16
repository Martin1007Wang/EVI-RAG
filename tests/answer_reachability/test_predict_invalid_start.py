from __future__ import annotations

import torch

from src.models.configs import (
    BackboneConfig,
    AnswerReachabilityInferenceConfig,
    GFlowNetTrainingConfig,
    GraphLogZHeadConfig,
    HeuristicConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
    StartHeadConfig,
    StateScoreHeadConfig,
)
from src.models.gflownet_module import GFlowNetModule
from src.metrics.answer_reachability.runtime import (
    AnswerReachabilityMetricRuntimeFactory,
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
            start_head=StartHeadConfig(hidden_dim=16, dropout=0.0),
            graph_log_z_head=GraphLogZHeadConfig(hidden_dim=16, dropout=0.0),
        ),
        inference_cfg=AnswerReachabilityInferenceConfig(
            answer_mass_threshold=0.9,
            support_mass_threshold=0.9,
            answer_top_ks=(1, 5),
            max_expansions=32,
            max_frontier_size=32,
        ),
        optimizer_cfg=OptimizerConfig(),
        scheduler_cfg=SchedulerConfig(),
        metric_runtime_factory=AnswerReachabilityMetricRuntimeFactory(),
    )


def test_predict_step_emits_placeholder_for_missing_start_candidates() -> None:
    torch.manual_seed(31)
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.empty((0,), dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_global_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="missing-start-candidates",
    )
    module = _make_module()

    module.on_predict_epoch_start()
    outputs = module.predict_step(batch, batch_idx=0)
    module.on_predict_batch_end(outputs, batch, batch_idx=0)
    module.on_predict_epoch_end()

    assert len(outputs) == 1
    result = outputs[0]
    assert result.sample_id == "missing-start-candidates"
    assert result.stop_reason == "invalid_start_candidates"
    assert result.inference_mode == "exact"
    assert result.window_size == 0
    assert result.covered_mass == 0.0
    assert result.residual_mass == 1.0
    assert result.start_entity_ids == []
    assert result.answer_posterior == []
    assert module.predict_metrics["invalid_start_count"] == 1
    assert module.predict_metrics["invalid_start_rate"] == 1.0
    assert module.predict_metrics["meta/num_samples"] == 1.0
    assert module.predict_labels[0].sample_id == "missing-start-candidates"
    assert module.predict_labels[0].start_entity_ids == []


def test_evaluate_batch_emits_placeholder_for_missing_start_candidates() -> None:
    torch.manual_seed(37)
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.empty((0,), dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_global_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="missing-start-candidates-eval",
    )
    module = _make_module()

    _, window_results, model_metrics, rank_metrics = module._evaluate_batch(batch=batch)

    assert len(window_results) == 1
    assert window_results[0].stop_reason == "invalid_start_candidates"
    assert window_results[0].answer_posterior == []
    assert model_metrics == {}
    assert rank_metrics["answer/gold_mass"] == 0.0
