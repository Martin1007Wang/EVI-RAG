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
            eval_profile="rank_only",
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
