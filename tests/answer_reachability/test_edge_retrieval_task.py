from __future__ import annotations

import torch

from src.models.configs import (
    SearchEvalConfig,
    GFlowNetTrainingConfig,
    HeuristicConfig,
    HorizonConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from src.models.gflownet_module import GFlowNetModule
from src.metrics.answer_reachability import compute_edge_retrieval_labels
from src.metrics.answer_reachability.runtime import (
    SearchMetricRuntimeFactory,
)

from .conftest import make_policy_config, make_toy_batch


def _make_edge_retrieval_module() -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            sampling_temperature=1.0,
        ),
        heuristic_cfg=HeuristicConfig(kind="topology", beta=0.5),
        policy_cfg=make_policy_config(),
        eval_cfg=SearchEvalConfig(
            task="edge_retrieval",
            metrics_profile="rank_only",
            support_search_method="monte_carlo",
            edge_top_ks=(1, 2, 3),
            edge_emit_top_k=3,
        ),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )


def test_compute_edge_retrieval_labels_marks_direct_shortest_path() -> None:
    labels = compute_edge_retrieval_labels(batch=make_toy_batch())

    assert labels.num_edges == 3
    assert labels.max_path_length == 1
    assert labels.positive_edge_ids.tolist() == [1]


def test_edge_retrieval_validation_step_smoke() -> None:
    module = _make_edge_retrieval_module()
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    module.validation_step(make_toy_batch(), batch_idx=0)


def test_edge_retrieval_predict_metrics_smoke() -> None:
    module = _make_edge_retrieval_module()
    batch = make_toy_batch()

    outputs = module.predict_step(batch, batch_idx=0)
    module.on_predict_batch_end(outputs, batch, batch_idx=0)
    module.on_predict_epoch_end()

    metrics = module.get_predict_metrics()
    assert "edge/hit@1" in metrics
    assert torch.isfinite(torch.tensor(metrics["edge/hit@1"], dtype=torch.float32))
