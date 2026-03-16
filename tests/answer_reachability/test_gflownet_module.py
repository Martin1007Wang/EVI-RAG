from __future__ import annotations

import pytest
import torch

from src.models.configs import (
    AnswerReachabilityInferenceConfig,
    BackboneConfig,
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
from src.models.policy.heuristic_utils import (
    compute_embedding_log_heuristic,
    compute_topology_log_heuristic,
)
from src.graph_runtime import build_graph_batch
from src.models.gflownet_module import GFlowNetModule
from src.metrics.answer_reachability.runtime import (
    AnswerReachabilityMetricRuntimeFactory,
)

from .conftest import make_toy_batch


def _make_policy_config() -> PolicyConfig:
    return PolicyConfig(
        backbone=BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            gnn_layers=1,
            gnn_dropout=0.0,
            use_adapter=True,
            adapter_dim=4,
            adapter_dropout=0.0,
        ),
        state_score_head=StateScoreHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
        start_head=StartHeadConfig(hidden_dim=16, dropout=0.0),
        graph_log_z_head=GraphLogZHeadConfig(hidden_dim=16, dropout=0.0),
    )


def _make_module(h_kind: str) -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
            sampling_temperature=1.0,
        ),
        heuristic_cfg=HeuristicConfig(
            kind=h_kind,
            beta=0.5,
            topology_restart_prob=0.3,
            topology_num_iters=6,
            topology_eps=1.0e-8,
            embedding_temperature=0.7,
            critic_hidden_dim=16,
            critic_dropout=0.0,
            critic_loss_weight=0.0,
            critic_target_floor=1.0e-3,
        ),
        policy_cfg=_make_policy_config(),
        inference_cfg=AnswerReachabilityInferenceConfig(eval_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=AnswerReachabilityMetricRuntimeFactory(),
    )


def test_compute_topology_log_heuristic_prefers_question_seed() -> None:
    batch = make_toy_batch()
    topology, observation = build_graph_batch(batch)

    log_heuristic = compute_topology_log_heuristic(
        topology=topology,
        observation=observation,
        restart_prob=0.3,
        num_iters=6,
        eps=1.0e-8,
    )

    assert tuple(log_heuristic.shape) == (batch.num_nodes_total,)
    assert torch.isfinite(log_heuristic).all()
    assert int(torch.argmax(log_heuristic).item()) == 0


def test_compute_embedding_log_heuristic_tracks_cosine_similarity() -> None:
    batch = make_toy_batch()
    topology, _ = build_graph_batch(batch)

    node_tokens = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    question_tokens = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    log_heuristic = compute_embedding_log_heuristic(
        topology=topology,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
        temperature=0.5,
    )

    assert log_heuristic[0] > log_heuristic[2] > log_heuristic[1]


def test_gflownet_module_uses_heuristic_config() -> None:
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(),
        heuristic_cfg=HeuristicConfig(beta=0.0),
        policy_cfg=_make_policy_config(),
        inference_cfg=AnswerReachabilityInferenceConfig(eval_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=AnswerReachabilityMetricRuntimeFactory(),
    )

    assert module.cfg.heuristic_cfg.beta == 0.0
    assert module.policy.heuristic_cfg.beta == 0.0


def test_gflownet_module_exposes_canonical_inference_aliases() -> None:
    module = _make_module("topology")

    assert module.metrics_profile == "rank_only"
    assert module.task_view == "answer_reachability"


def test_predict_epoch_start_resets_prediction_state() -> None:
    module = _make_module("topology")
    module.predict_results = ["stale"]
    module.predict_labels = ["label"]
    module.predict_metrics = {"answer/hit@1": 0.5}

    module.on_predict_epoch_start()

    assert module.predict_results == []
    assert module.predict_labels == []
    assert module.predict_metrics == {}


def test_predict_epoch_end_summarizes_prediction_metrics() -> None:
    module = _make_module("topology")
    module.predict_results = ["result"]
    module.metric_runtime.summarize_predict_epoch = lambda **kwargs: {  # type: ignore[method-assign]
        "answer/hit@1": 0.25
    }

    module.on_predict_epoch_end()

    assert module.get_predict_metrics() == {"answer/hit@1": 0.25}


def test_write_prediction_artifacts_uses_prediction_state(tmp_path) -> None:
    module = _make_module("topology")
    module.predict_results = ["result"]
    module.predict_labels = ["label"]
    captured: dict[str, object] = {}

    def _write_prediction_artifacts(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"prompt_path": tmp_path / "test.jsonl"}

    module.metric_runtime.write_prediction_artifacts = _write_prediction_artifacts  # type: ignore[method-assign]

    paths = module.write_prediction_artifacts(
        output_dir=tmp_path,
        split="test",
        artifact_name="eval_answer_reachability",
    )

    assert captured["results"] == ["result"]
    assert captured["labels"] == ["label"]
    assert paths == {"prompt_path": tmp_path / "test.jsonl"}


@pytest.mark.parametrize("h_kind", ["topology", "embedding", "learned"])
def test_gflownet_training_step_smoke(h_kind: str) -> None:
    torch.manual_seed(7)
    module = _make_module(h_kind)
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    loss = module.training_step(make_toy_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gflownet_validation_step_smoke() -> None:
    module = _make_module("topology")
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    module.validation_step(make_toy_batch(), batch_idx=0)
