from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

import src.models.gflownet.policy as gflownet_policy_impl
import src.models.gflownet_module as gflownet_module_impl
from src.models.configs import (
    ActionPriorConfig,
    ActionPriorScheduleConfig,
    AnswerQuotientConfig,
    BackboneConfig,
    GFlowNetTrainingConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    PotentialRewardConfig,
    SamplingTemperatureScheduleConfig,
    SchedulerConfig,
    SearchEvalConfig,
    StateScoreHeadConfig,
    SuccessReplayConfig,
    TransitionHeadConfig,
)
from src.models.gflownet import (
    AnswerReachabilityTrajectorySupervisor,
    ForwardTrajectoryGFNSampler,
    SearchState,
    SubTrajectoryBalanceLoss,
    SubTrajectoryBalanceLossOutput,
    TrajectoryGFNSampleBatch,
    TrainingScheduleContext,
    compute_embedding_log_heuristic,
    compute_topology_log_heuristic,
)
from src.models.gflownet.answer_supervision import compute_gold_entity_ranking_loss
from src.graph import TrajectoryBatch, build_graph_batch
from src.models.gflownet_module import (
    GFlowNetModule,
    PredictionArtifactWriteConfig,
)
from src.metrics.runtime_factory import GraphTaskRuntimeFactory
from src.utils.fit_schedule import ResolvedPassFitSchedule

from .conftest import make_batch_from_graph, make_toy_batch


def _make_policy_config(*, transition_enabled: bool = False) -> PolicyConfig:
    return PolicyConfig(
        backbone=BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            use_adapter=True,
            adapter_dim=4,
            adapter_dropout=0.0,
        ),
        state_score_head=StateScoreHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
        transition_head=TransitionHeadConfig(
            enabled=transition_enabled,
            hidden_dim=8,
            num_layers=2,
            dropout=0.0,
        ),
    )


def _make_action_prior_config(
    prior_kind: str,
    *,
    root_beta: float,
    edge_beta: float,
    **overrides: Any,
) -> ActionPriorConfig:
    prior_overrides: dict[str, Any] = {
        "root_beta": root_beta,
        "edge_beta": edge_beta,
    }
    if prior_kind == "none":
        prior_overrides.update(
            node_topology_weight=0.0,
            node_embedding_weight=0.0,
        )
    elif prior_kind == "topology":
        prior_overrides.update(node_embedding_weight=0.0)
    elif prior_kind == "embedding":
        prior_overrides.update(node_topology_weight=0.0)
    elif prior_kind != "hybrid":
        raise ValueError(f"Unsupported test prior kind: {prior_kind!r}.")
    prior_overrides.update(overrides)
    return ActionPriorConfig(
        topology_restart_prob=0.3,
        topology_num_iters=6,
        topology_eps=1.0e-8,
        embedding_temperature=0.7,
        **prior_overrides,
    )


def _make_module(prior_kind: str) -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=3,
            sampling_temperature=1.0,
        ),
        action_prior_cfg=_make_action_prior_config(
            prior_kind,
            root_beta=0.5,
            edge_beta=0.5,
        ),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )


def _make_module_with_training_cfg(
    prior_kind: str,
    *,
    beta: float = 0.5,
    training_cfg: GFlowNetTrainingConfig | None = None,
) -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=(
            training_cfg
            if training_cfg is not None
            else GFlowNetTrainingConfig(
                rollouts_per_graph=3,
                sampling_temperature=1.0,
            )
        ),
        action_prior_cfg=_make_action_prior_config(
            prior_kind,
            root_beta=beta,
            edge_beta=beta,
        ),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )


def _make_manual_sample_batch(
    *,
    batch: TrajectoryBatch,
    rollouts_per_graph: int,
    success_mask: torch.Tensor,
    start_entropy: float,
    start_entropy_normalized: float,
) -> TrajectoryGFNSampleBatch:
    max_actions = 3
    start_nodes = torch.zeros(
        (batch.num_graphs, rollouts_per_graph),
        dtype=torch.long,
        device=batch.node_ptr.device,
    )
    trace_nodes = torch.zeros(
        (batch.num_graphs, rollouts_per_graph, max_actions),
        dtype=torch.long,
        device=batch.node_ptr.device,
    )
    trace_edge_ids = torch.full_like(trace_nodes, fill_value=-1)
    trace_num_steps = torch.zeros_like(trace_nodes)
    trace_mask = torch.zeros_like(trace_nodes, dtype=torch.bool)
    return TrajectoryGFNSampleBatch(
        graph_log_z=torch.zeros((batch.num_graphs,), dtype=torch.float32),
        start_nodes=start_nodes,
        start_log_probs=torch.zeros_like(start_nodes, dtype=torch.float32),
        start_state_log_f=torch.zeros_like(start_nodes, dtype=torch.float32),
        log_pf_steps=torch.zeros_like(trace_nodes, dtype=torch.float32),
        log_pb_steps=torch.zeros_like(trace_nodes, dtype=torch.float32),
        state_log_f_steps=torch.zeros_like(trace_nodes, dtype=torch.float32),
        next_state_log_f_steps=torch.zeros_like(trace_nodes, dtype=torch.float32),
        move_mask=torch.zeros_like(trace_nodes, dtype=torch.bool),
        trace_nodes=trace_nodes,
        trace_edge_ids=trace_edge_ids,
        trace_num_steps=trace_num_steps,
        trace_mask=trace_mask,
        terminal_nodes=start_nodes.clone(),
        terminal_num_steps=torch.zeros_like(start_nodes),
        terminal_state_log_f=torch.zeros_like(start_nodes, dtype=torch.float32),
        terminal_rewards=torch.zeros_like(start_nodes, dtype=torch.float32),
        terminal_log_rewards=torch.zeros_like(start_nodes, dtype=torch.float32),
        success_mask=success_mask,
        start_entropy=torch.full(
            (batch.num_graphs,),
            fill_value=float(start_entropy),
            dtype=torch.float32,
            device=batch.node_ptr.device,
        ),
        start_entropy_normalized=torch.full(
            (batch.num_graphs,),
            fill_value=float(start_entropy_normalized),
            dtype=torch.float32,
            device=batch.node_ptr.device,
        ),
    )


def _make_loss_output(loss_value: float) -> SubTrajectoryBalanceLossOutput:
    scalar = torch.tensor(float(loss_value), dtype=torch.float32)
    zero = torch.tensor(0.0, dtype=torch.float32)
    return SubTrajectoryBalanceLossOutput(
        loss=scalar,
        subtb_loss=scalar,
        residual_abs=zero,
        residual_variance=zero,
        root_abs=zero,
        success_rate=zero,
        log_z_mean=zero,
        log_z_variance=zero,
        root_component_loss=zero,
        pairwise_component_loss=zero,
        terminal_component_loss=zero,
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


def test_compute_topology_log_heuristic_preserves_seed_mass_for_no_edge_graph() -> None:
    batch_one = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="seeded-edge-graph",
    )
    batch_two = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([1], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([201], dtype=torch.long),
        node_entity_ids=torch.tensor([200, 201], dtype=torch.long),
        sample_id="seeded-no-edge-graph",
    )
    batch = TrajectoryBatch.concatenate([batch_one, batch_two], validate=False)
    topology, observation = build_graph_batch(batch)

    log_heuristic = compute_topology_log_heuristic(
        topology=topology,
        observation=observation,
        restart_prob=0.3,
        num_iters=6,
        eps=1.0e-8,
    )

    second_graph_scores = log_heuristic[2:4]
    assert second_graph_scores[1].item() == pytest.approx(0.0, abs=1.0e-6)
    assert second_graph_scores[0].item() < -10.0


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


def test_gflownet_module_uses_action_prior_config() -> None:
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(),
        action_prior_cfg=ActionPriorConfig(),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )

    assert module.cfg.action_prior_cfg.root_beta == pytest.approx(0.0)
    assert module.cfg.action_prior_cfg.edge_beta == pytest.approx(0.0)
    assert module.policy.action_prior_cfg.root_beta == pytest.approx(0.0)
    assert module.policy.action_prior_cfg.edge_beta == pytest.approx(0.0)


def test_gflownet_module_exposes_eval_settings() -> None:
    module = _make_module("topology")

    assert module.report_profile == "rank_only"
    assert module.evaluation_task == "answer_search"


def test_predict_epoch_start_resets_prediction_state() -> None:
    module = _make_module("topology")
    module.replace_prediction_state(
        results=cast(Any, ["stale"]),
        labels=cast(Any, ["label"]),
        metrics={"answer/hit@1": 0.5},
    )

    module.on_predict_epoch_start()

    assert module.predict_results == []
    assert module.predict_labels == []
    assert module.predict_metrics == {}


def test_predict_state_accessors_return_copies() -> None:
    module = _make_module("topology")
    module.replace_prediction_state(
        results=cast(Any, ["result"]),
        labels=cast(Any, ["label"]),
        metrics={"answer/hit@1": 0.5},
    )

    results = module.predict_results
    labels = module.predict_labels
    metrics = module.predict_metrics
    results.append(cast(Any, "mutated"))
    labels.append(cast(Any, "mutated"))
    metrics["mutated"] = 1.0

    assert module.predict_results == ["result"]
    assert module.predict_labels == ["label"]
    assert module.predict_metrics == {"answer/hit@1": 0.5}


def test_predict_epoch_end_summarizes_prediction_metrics() -> None:
    module = _make_module("topology")
    module.replace_prediction_state(results=cast(Any, ["result"]))
    module.metric_runtime.summarize_predict_epoch = lambda **kwargs: {  # type: ignore[method-assign]
        "answer/hit@1": 0.25
    }

    module.on_predict_epoch_end()

    assert module.get_predict_metrics() == {"answer/hit@1": 0.25}


def test_predict_epoch_end_prefers_online_accumulator_over_jsonl_reread() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    module.on_predict_epoch_start()
    outputs = module.predict_step(batch, batch_idx=0)
    module.on_predict_batch_end(outputs, batch, batch_idx=0)

    def _unexpected_jsonl_summary(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise AssertionError("expected online predict metric accumulation")

    module.metric_runtime.summarize_predict_epoch_from_jsonl = _unexpected_jsonl_summary  # type: ignore[attr-defined, method-assign]

    module.on_predict_epoch_end()

    assert module.get_predict_metrics()["answer/gold_answer_mass"] >= 0.0


def test_on_predict_batch_end_ignores_none_outputs() -> None:
    module = _make_module("topology")
    module.on_predict_epoch_start()

    module.on_predict_batch_end(None, make_toy_batch(), batch_idx=0)

    assert module.predict_results == []
    assert module.predict_labels == []


def test_write_prediction_artifacts_uses_prediction_state(tmp_path) -> None:
    module = _make_module("topology")
    module.replace_prediction_state(
        results=cast(Any, ["result"]),
        labels=cast(Any, ["label"]),
    )
    captured: dict[str, object] = {}

    def _write_prediction_artifacts(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"prompt_path": tmp_path / "test.jsonl"}

    module.metric_runtime.write_prediction_artifacts = _write_prediction_artifacts  # type: ignore[method-assign]

    paths = module.write_prediction_artifacts(
        output_dir=tmp_path,
        split="test",
        artifact_name="rankflow",
    )

    assert captured["results"] == ["result"]
    assert captured["labels"] == ["label"]
    assert paths == {"prompt_path": tmp_path / "test.jsonl"}


def test_write_prediction_artifacts_accepts_write_config(tmp_path) -> None:
    module = _make_module("topology")
    module.replace_prediction_state(
        results=cast(Any, ["result"]),
        labels=cast(Any, ["label"]),
    )
    captured: dict[str, object] = {}

    def _write_prediction_artifacts(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"prompt_path": tmp_path / "test.jsonl"}

    module.metric_runtime.write_prediction_artifacts = _write_prediction_artifacts  # type: ignore[method-assign]

    paths = module.write_prediction_artifacts(
        write_config=PredictionArtifactWriteConfig(
            output_dir=tmp_path,
            split="test",
            artifact_name="rankflow",
        )
    )

    assert captured["results"] == ["result"]
    assert captured["labels"] == ["label"]
    assert paths == {"prompt_path": tmp_path / "test.jsonl"}


def test_write_prediction_artifacts_prefers_jsonl_cache_when_predict_batches_recorded(
    tmp_path,
) -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    module.on_predict_epoch_start()
    outputs = module.predict_step(batch, batch_idx=0)
    module.on_predict_batch_end(outputs, batch, batch_idx=0)
    captured: dict[str, object] = {}

    def _write_prediction_artifacts_from_jsonl(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"prompt_path": tmp_path / "streamed.jsonl"}

    def _unexpected_write_prediction_artifacts(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise AssertionError("expected jsonl-backed artifact writing path")

    module.metric_runtime.write_prediction_artifacts_from_jsonl = (  # type: ignore[attr-defined, method-assign]
        _write_prediction_artifacts_from_jsonl
    )
    module.metric_runtime.write_prediction_artifacts = (
        _unexpected_write_prediction_artifacts  # type: ignore[method-assign]
    )

    paths = module.write_prediction_artifacts(
        output_dir=tmp_path,
        split="test",
        artifact_name="rankflow",
    )

    assert paths == {"prompt_path": tmp_path / "streamed.jsonl"}
    assert str(captured["results_path"]).endswith("results.jsonl")
    assert str(captured["labels_path"]).endswith("labels.jsonl")


def test_log_metric_bundle_syncs_only_epoch_metrics() -> None:
    module = _make_module("topology")
    logged_calls: list[tuple[str, bool]] = []

    def _capture_log(name: str, value: torch.Tensor, **kwargs: object) -> None:
        del value
        logged_calls.append((name, bool(kwargs["sync_dist"])))

    module.log = _capture_log  # type: ignore[method-assign]

    module._log_metric_bundle(
        metrics={"loss": torch.tensor(1.0)},
        prefix="train",
        batch_size=2,
        on_step=True,
        on_epoch=False,
    )
    module._log_metric_bundle(
        metrics={"hit@1": torch.tensor(0.5)},
        prefix="val/webqsp-sub",
        batch_size=2,
        on_step=False,
        on_epoch=True,
    )

    assert logged_calls[0] == ("train/loss", False)
    assert logged_calls[1] == ("val/webqsp-sub/hit@1", True)


def test_cfg_to_dict_rejects_dataclass_type_objects() -> None:
    with pytest.raises(TypeError, match="Expected dataclass or dict config"):
        GFlowNetModule._cfg_to_dict(GFlowNetTrainingConfig)


def test_training_schedule_context_override_is_used() -> None:
    module = _make_module("topology")
    module.set_training_schedule_context(
        TrainingScheduleContext(estimated_stepping_batches=7, trainer_max_steps=11)
    )

    schedule_context = module._trainer_schedule_context()

    assert schedule_context.estimated_stepping_batches == 7
    assert schedule_context.trainer_max_steps == 11


def test_transfer_batch_to_device_rejects_unexpected_batch_types() -> None:
    module = _make_module("topology")

    with pytest.raises(TypeError, match="expects TrajectoryBatch inputs"):
        module.transfer_batch_to_device(
            batch={"bad": "batch"}, device=torch.device("cpu"), dataloader_idx=0
        )


def test_log_invalid_start_tracks_count(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _make_module("topology")
    logged_events: list[dict[str, object]] = []

    monkeypatch.setattr(
        gflownet_module_impl,
        "log_event",
        lambda *args, **kwargs: logged_events.append(kwargs),
    )

    module._log_invalid_start(make_toy_batch())
    module._log_invalid_start(make_toy_batch())

    assert module._invalid_start_count == 2
    assert logged_events[-1]["invalid_start_count"] == 2


def test_gflownet_training_step_logs_log_z_statistics() -> None:
    module = _make_module("topology")
    captured_metrics: dict[str, object] = {}

    def _capture_metric_bundle(*, metrics: dict[str, object], **kwargs: object) -> None:
        del kwargs
        captured_metrics.update(metrics)

    module._log_metric_bundle = _capture_metric_bundle  # type: ignore[method-assign]

    loss = module.training_step(make_toy_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert "log_z_mean" in captured_metrics
    assert "log_z_variance" in captured_metrics
    assert "log_z_num_nodes_corr" in captured_metrics
    assert "log_z_num_edges_corr" in captured_metrics
    assert "log_z_start_candidates_corr" in captured_metrics
    assert captured_metrics["sampling_temperature"] == pytest.approx(1.0)
    assert "sampling_temperature_multiplier" not in captured_metrics


def test_compute_gold_entity_ranking_loss_aggregates_alias_mass() -> None:
    entity_scores = torch.log(
        torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0], dtype=torch.float32)
    )

    loss, gold_mass, entity_count = compute_gold_entity_ranking_loss(
        graph_ids=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        entity_ids=torch.tensor([10, 11, 11, 20, 21], dtype=torch.long),
        entity_scores=entity_scores,
        answer_entity_ids=torch.tensor([11, 20], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    expected_graph_zero = 5.0 / 6.0
    expected_graph_one = 1.0 / 5.0
    expected_loss = (
        -torch.log(torch.tensor(expected_graph_zero))
        - torch.log(torch.tensor(expected_graph_one))
    ) / 2.0

    assert loss.item() == pytest.approx(float(expected_loss.item()))
    assert gold_mass.item() == pytest.approx(
        (expected_graph_zero + expected_graph_one) / 2.0
    )
    assert entity_count.item() == pytest.approx(4.0)


def test_module_direct_gold_entity_ranking_loss_scores_unique_entities() -> None:
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            answer_quotient=AnswerQuotientConfig(
                enabled=True,
                direct_entity_ranking_weight=0.25,
            )
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 101], dtype=torch.long),
        sample_id="direct-entity-ranking",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    module.policy.base_policy._compute_log_state_scores_from_flat_features = (  # type: ignore[method-assign]
        lambda **kwargs: torch.log(
            torch.tensor([1.0, 2.0, 3.0], device=kwargs["graph_ids"].device)
        )
    )

    loss, gold_mass, entity_count = module._compute_direct_gold_entity_ranking_loss(
        batch=batch,
        prepared_batch=prepared_batch,
    )

    assert loss.item() == pytest.approx(
        float((-torch.log(torch.tensor(5.0 / 6.0))).item())
    )
    assert gold_mass.item() == pytest.approx(5.0 / 6.0)
    assert entity_count.item() == pytest.approx(2.0)


def test_gflownet_training_step_logs_direct_entity_ranking_metrics() -> None:
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            answer_quotient=AnswerQuotientConfig(
                enabled=True,
                direct_entity_ranking_weight=0.25,
            )
        ),
    )
    captured_metrics: dict[str, object] = {}

    def _capture_metric_bundle(*, metrics: dict[str, object], **kwargs: object) -> None:
        del kwargs
        captured_metrics.update(metrics)

    module._log_metric_bundle = _capture_metric_bundle  # type: ignore[method-assign]

    loss = module.training_step(make_toy_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert "answer_quotient_direct_entity_ranking_loss" in captured_metrics
    assert "answer_quotient_direct_gold_entity_mass" in captured_metrics
    assert "answer_quotient_direct_entity_count" in captured_metrics
    assert "replay_answer_quotient_direct_entity_ranking_loss" in captured_metrics
    assert captured_metrics[
        "answer_quotient_direct_entity_ranking_weight"
    ] == pytest.approx(0.25)


def test_start_distribution_defines_virtual_source_log_z() -> None:
    module = _make_module("topology")
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [2, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="multi-start-log-z",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    start_dist = module.policy.compute_start_distribution(prepared_batch)
    (
        sampled_nodes,
        sampled_log_probs,
        sampled_log_flows,
    ) = module.policy.sample_start_nodes(
        start_dist,
        num_rollouts=4,
        deterministic=True,
    )

    assert torch.allclose(
        module.policy.compute_graph_log_z(prepared_batch),
        start_dist.graph_log_z,
        atol=1.0e-6,
    )
    graph_mask = start_dist.candidate_graph_ids == 0
    assert torch.allclose(
        torch.logsumexp(start_dist.log_probs[graph_mask], dim=0),
        torch.tensor(0.0),
        atol=1.0e-6,
    )
    assert sampled_nodes.shape == (1, 4)
    candidate_nodes = start_dist.candidate_nodes_abs[graph_mask]
    candidate_log_probs = start_dist.log_probs[graph_mask]
    candidate_log_flows = start_dist.log_flows[graph_mask]
    for rollout_idx in range(int(sampled_nodes.size(1))):
        sampled_node = sampled_nodes[0, rollout_idx]
        candidate_idx = torch.nonzero(
            candidate_nodes == sampled_node, as_tuple=False
        ).view(-1)
        assert int(candidate_idx.numel()) == 1
        selected_idx = int(candidate_idx.item())
        assert sampled_log_probs[0, rollout_idx].item() == pytest.approx(
            float(candidate_log_probs[selected_idx].item())
        )
        assert sampled_log_flows[0, rollout_idx].item() == pytest.approx(
            float(candidate_log_flows[selected_idx].item())
        )


def test_sampler_preserves_selected_start_flow_gradients() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)

    original_forward = module.policy.compute_forward_distribution

    def _prefer_graph_moves(prepared_batch_arg, state, **kwargs):  # noqa: ANN001
        distribution = original_forward(prepared_batch_arg, state, **kwargs)
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_stop_action is not None:
            logits[distribution.is_stop_action.to(dtype=torch.bool)] = -1.0e9
        return replace(distribution, edge_logits=logits)

    module.policy.compute_forward_distribution = _prefer_graph_moves  # type: ignore[method-assign]

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.start_state_log_f.requires_grad
    assert sample_batch.start_state_log_f.grad_fn is not None
    assert sample_batch.start_log_probs.requires_grad
    assert sample_batch.start_log_probs.grad_fn is not None


def test_root_flow_features_include_start_pool_and_size_scalars() -> None:
    module = _make_module("none")
    shared_question = torch.zeros((1, 8), dtype=torch.float32)
    shared_context = torch.zeros((1, 2, 8), dtype=torch.float32)
    batch_one = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="root-z-one",
        question_emb=shared_question,
        question_ctx=shared_context,
    )
    batch_two = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([202], dtype=torch.long),
        node_entity_ids=torch.tensor([200, 201, 202], dtype=torch.long),
        sample_id="root-z-two",
        question_emb=shared_question.clone(),
        question_ctx=shared_context.clone(),
    )
    batch = TrajectoryBatch.concatenate([batch_one, batch_two], validate=False)
    prepared_batch = module.policy.prepare_batch(batch)
    node_tokens = torch.zeros_like(prepared_batch.node_tokens)
    node_tokens[0, 0] = 1.0
    node_tokens[1, 0] = 3.0
    node_tokens[2, 0] = 4.0
    node_tokens[3, 0] = 5.0
    node_tokens[4, 0] = 6.0
    prepared_batch = replace(
        prepared_batch,
        question_tokens=torch.zeros_like(prepared_batch.question_tokens),
        node_tokens=node_tokens,
    )
    root_features = module.policy.base_policy._build_root_flow_features(prepared_batch)

    assert root_features.shape == (2, 28)
    assert root_features[0, 8].item() == pytest.approx(2.0)
    assert root_features[1, 8].item() == pytest.approx(5.0)
    assert root_features[0, 16].item() == pytest.approx(1.0)
    assert root_features[1, 16].item() == pytest.approx(4.5)
    assert root_features[0, 24].item() == pytest.approx(
        float(torch.log1p(torch.tensor(2.0)).item())
    )
    assert root_features[1, 24].item() == pytest.approx(
        float(torch.log1p(torch.tensor(3.0)).item())
    )
    assert root_features[0, 25].item() == pytest.approx(
        float(torch.log1p(torch.tensor(1.0)).item())
    )
    assert root_features[1, 25].item() == pytest.approx(
        float(torch.log1p(torch.tensor(2.0)).item())
    )
    assert root_features[0, 26].item() == pytest.approx(
        float(torch.log1p(torch.tensor(1.0)).item())
    )
    assert root_features[1, 26].item() == pytest.approx(
        float(torch.log1p(torch.tensor(2.0)).item())
    )


def test_sampler_can_force_stop_on_terminal_targets_before_proposal_expansion() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=3,
            sampling_temperature=1.0,
            force_stop_on_answer_hit=True,
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0], dtype=torch.long),
        answer_entity_ids=torch.tensor([100], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="stop-before-expand",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    assert module.sampler is not None
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    call_count = 0
    original_proposal = module.policy.compute_proposal_forward_distribution
    terminal_target_mask = sampler.trajectory_supervisor.build_terminal_target_mask(
        batch=batch
    )

    def _wrapped_proposal_distribution(prepared_batch_arg, state):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        active_nodes = state.current_nodes[~state.done_mask].reshape(-1)
        if int(active_nodes.numel()) > 0:
            assert not bool(
                terminal_target_mask.index_select(0, active_nodes).any().item()
            )
        return original_proposal(prepared_batch_arg, state)

    module.policy.compute_proposal_forward_distribution = (  # type: ignore[method-assign]
        _wrapped_proposal_distribution
    )

    sample_batch = sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.trace_stop_mask is not None
    assert sample_batch.termination_action_steps is not None
    assert call_count == 0
    assert bool(sample_batch.trace_stop_mask[0, 0, 0].item()) is True
    assert int(sample_batch.trace_edge_ids[0, 0, 0].item()) == -1
    assert int(sample_batch.termination_action_steps[0, 0].item()) == 1
    assert int(sample_batch.terminal_num_steps[0, 0].item()) == 0
    assert int(sample_batch.terminal_nodes[0, 0].item()) == 0
    assert bool(sample_batch.success_mask[0, 0].item()) is True


def test_sampler_does_not_force_stop_on_terminal_targets_by_default() -> None:
    module = _make_module("topology")
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0], dtype=torch.long),
        answer_entity_ids=torch.tensor([100], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="no-forced-stop-default",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    original_forward = module.policy.compute_forward_distribution

    def _prefer_graph_moves(prepared_batch_arg, state, **kwargs):  # noqa: ANN001
        distribution = original_forward(prepared_batch_arg, state, **kwargs)
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_stop_action is not None:
            logits[distribution.is_stop_action.to(dtype=torch.bool)] = -1.0e9
        return replace(distribution, edge_logits=logits)

    module.policy.compute_forward_distribution = _prefer_graph_moves  # type: ignore[method-assign]

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.trace_stop_mask is not None
    assert sample_batch.termination_action_steps is not None
    assert bool(sample_batch.trace_stop_mask[0, 0, 0].item()) is False
    assert int(sample_batch.trace_edge_ids[0, 0, 0].item()) == 0
    assert int(sample_batch.termination_action_steps[0, 0].item()) == 2
    assert int(sample_batch.terminal_num_steps[0, 0].item()) == 1
    assert int(sample_batch.terminal_nodes[0, 0].item()) == 1
    assert bool(sample_batch.success_mask[0, 0].item()) is False
    assert bool(sample_batch.move_mask[0, 0, 0].item()) is True
    assert bool(sample_batch.move_mask[0, 0, 1].item()) is False


def test_sampler_forces_stop_on_alias_answer_entities() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            force_stop_on_answer_hit=True,
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[3], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([3], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 102], dtype=torch.long),
        sample_id="alias-answer-force-stop",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    original_forward = module.policy.compute_forward_distribution

    def _prefer_graph_moves(prepared_batch_arg, state, **kwargs):  # noqa: ANN001
        distribution = original_forward(prepared_batch_arg, state, **kwargs)
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_stop_action is not None:
            logits[distribution.is_stop_action.to(dtype=torch.bool)] = -1.0e9
        return replace(distribution, edge_logits=logits)

    module.policy.compute_forward_distribution = _prefer_graph_moves  # type: ignore[method-assign]

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.trace_stop_mask is not None
    assert sample_batch.termination_action_steps is not None
    assert bool(sample_batch.trace_stop_mask[0, 0, 0].item()) is True
    assert int(sample_batch.trace_edge_ids[0, 0, 0].item()) == -1
    assert int(sample_batch.termination_action_steps[0, 0].item()) == 1
    assert int(sample_batch.terminal_num_steps[0, 0].item()) == 0
    assert int(sample_batch.terminal_nodes[0, 0].item()) == 3
    assert bool(sample_batch.success_mask[0, 0].item()) is True
    assert bool(sample_batch.move_mask[0, 0, 0].item()) is False


def test_sampler_uses_deterministic_terminal_backward_log_prob() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([100], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 100], dtype=torch.long),
        sample_id="stop-action-submit",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.trace_stop_mask is not None
    assert sample_batch.terminal_entity_ids is not None
    assert sample_batch.terminal_backward_log_probs is not None
    assert bool(sample_batch.trace_stop_mask[0, 0, 0].item()) is True
    assert sample_batch.terminal_entity_ids[0, 0].item() == 100
    assert sample_batch.terminal_backward_log_probs[0, 0].item() == pytest.approx(0.0)
    assert sample_batch.log_pb_steps[0, 0, 0].item() == pytest.approx(0.0)


def test_sampler_keeps_terminal_reward_pure_without_length_discount() -> None:
    module = _make_module_with_training_cfg(
        "none",
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            step_log_penalty=float(torch.log(torch.tensor(0.5)).item()),
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="length-penalty",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    original_forward = module.policy.compute_forward_distribution

    def _prefer_graph_moves(prepared_batch_arg, state, **kwargs):  # noqa: ANN001
        distribution = original_forward(prepared_batch_arg, state, **kwargs)
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_stop_action is not None:
            logits[distribution.is_stop_action.to(dtype=torch.bool)] = -1.0e9
        return replace(distribution, edge_logits=logits)

    module.policy.compute_forward_distribution = _prefer_graph_moves  # type: ignore[method-assign]

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.terminal_num_steps[0, 0].item() == 1
    assert sample_batch.terminal_rewards[0, 0].item() == pytest.approx(1.0)
    assert sample_batch.terminal_log_rewards[0, 0].item() == pytest.approx(0.0)
    assert sample_batch.log_reward_steps is not None
    assert sample_batch.log_reward_steps[0, 0, 0].item() == pytest.approx(
        float(torch.log(torch.tensor(0.5)).item())
    )


def test_sampler_samples_root_actions_with_proposal_helpers() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)

    assert module.sampler is not None
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    start_grad_enabled: list[bool] = []
    proposal_start_calls: list[bool] = []
    proposal_edge_calls: list[bool] = []
    proposal_start_scales: list[float] = []
    proposal_edge_scales: list[float] = []
    original_start = module.policy.compute_root_action_distribution
    original_proposal_start = module.policy.compute_proposal_start_distribution
    original_proposal_logits = module.policy.compute_proposal_edge_logits

    def _wrapped_start(prepared_batch_arg):  # noqa: ANN001
        start_grad_enabled.append(torch.is_grad_enabled())
        return original_start(prepared_batch_arg)

    def _wrapped_proposal_start(prepared_batch_arg, **kwargs):  # noqa: ANN001
        proposal_start_calls.append(torch.is_grad_enabled())
        proposal_start_scales.append(float(kwargs.get("action_prior_scale", 1.0)))
        return original_proposal_start(prepared_batch_arg, **kwargs)

    def _wrapped_proposal_logits(prepared_batch_arg, state, distribution, **kwargs):  # noqa: ANN001
        proposal_edge_calls.append(torch.is_grad_enabled())
        proposal_edge_scales.append(float(kwargs.get("action_prior_scale", 1.0)))
        return original_proposal_logits(
            prepared_batch_arg, state, distribution, **kwargs
        )

    module.policy.compute_root_action_distribution = _wrapped_start  # type: ignore[method-assign]
    module.policy.compute_proposal_start_distribution = _wrapped_proposal_start  # type: ignore[method-assign]
    module.policy.compute_proposal_edge_logits = _wrapped_proposal_logits  # type: ignore[method-assign]

    sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert start_grad_enabled
    assert all(flag is True for flag in start_grad_enabled)
    assert proposal_start_calls == [False]
    assert proposal_start_scales == [1.0]
    assert proposal_edge_calls
    assert all(flag is False for flag in proposal_edge_calls)
    assert all(scale == pytest.approx(1.0) for scale in proposal_edge_scales)


def test_sampler_passes_sampling_temperature_to_root_sampling() -> None:
    module = _make_module("topology")
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="root-temp-pass-through",
    )
    prepared_batch = module.policy.prepare_batch(batch)
    recorded_temperatures: list[float] = []
    original_sample_start_nodes = module.policy.sample_start_nodes

    def _wrapped_sample_start_nodes(
        distribution, *, num_rollouts, deterministic=False, temperature=1.0
    ):  # noqa: ANN001
        recorded_temperatures.append(float(temperature))
        return original_sample_start_nodes(
            distribution,
            num_rollouts=num_rollouts,
            deterministic=deterministic,
            temperature=temperature,
        )

    module.policy.sample_start_nodes = _wrapped_sample_start_nodes  # type: ignore[method-assign]

    assert module.sampler is not None
    module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=2,
        temperature=0.5,
    )

    assert recorded_temperatures == [0.5]


def test_sampler_reports_tempered_root_entropy() -> None:
    module = _make_module("topology")
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="tempered-root-entropy",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    proposal_distribution = gflownet_policy_impl.RootActionDistribution(
        candidate_nodes_abs=torch.tensor([0, 1], dtype=torch.long),
        candidate_graph_ids=torch.tensor([0, 0], dtype=torch.long),
        log_flows=torch.tensor([0.0, 0.0], dtype=torch.float32),
        log_probs=torch.log(torch.tensor([0.8, 0.2], dtype=torch.float32)),
        graph_log_z=torch.tensor([0.0], dtype=torch.float32),
    )

    module.policy.compute_proposal_start_distribution = (  # type: ignore[method-assign]
        lambda prepared_batch_arg, **kwargs: proposal_distribution
    )

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=2,
        temperature=2.0,
    )

    expected_probs = torch.tensor([0.8**0.5, 0.2**0.5], dtype=torch.float32)
    expected_probs = expected_probs / expected_probs.sum()
    expected_entropy = -torch.sum(expected_probs * torch.log(expected_probs))
    expected_normalized = expected_entropy / torch.log(torch.tensor(2.0))

    assert sample_batch.proposal_start_entropy is not None
    assert sample_batch.proposal_start_entropy_normalized is not None
    assert sample_batch.proposal_start_entropy[0].item() == pytest.approx(
        float(expected_entropy.item())
    )
    assert sample_batch.proposal_start_entropy_normalized[0].item() == pytest.approx(
        float(expected_normalized.item())
    )


def test_target_policy_ignores_proposal_action_prior_beta() -> None:
    torch.manual_seed(5)
    target_only = _make_module_with_training_cfg("topology", beta=0.0)
    torch.manual_seed(5)
    proposal_guided = _make_module_with_training_cfg("topology", beta=0.5)
    batch = make_toy_batch()

    prepared_target = target_only.policy.prepare_batch(batch)
    prepared_guided = proposal_guided.policy.prepare_batch(batch)

    start_target = target_only.policy.compute_start_distribution(prepared_target)
    start_guided = proposal_guided.policy.compute_start_distribution(prepared_guided)
    assert torch.allclose(start_target.log_probs, start_guided.log_probs, atol=1.0e-6)

    state = SearchState(
        topology=prepared_target.topology,
        observation=prepared_target.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )
    target_distribution = target_only.policy.compute_forward_distribution(
        prepared_target,
        state,
    )
    guided_target_distribution = proposal_guided.policy.compute_forward_distribution(
        prepared_guided,
        state,
    )
    proposal_distribution = (
        proposal_guided.policy.compute_proposal_forward_distribution(
            prepared_guided,
            state,
        )
    )

    assert torch.allclose(
        target_distribution.edge_logits,
        guided_target_distribution.edge_logits,
        atol=1.0e-6,
    )
    assert not torch.allclose(
        target_distribution.edge_logits,
        proposal_distribution.edge_logits,
    )


def test_zero_action_prior_disables_proposal_bias() -> None:
    module = _make_module_with_training_cfg("none", beta=0.0)
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)

    start_target = module.policy.compute_start_distribution(prepared_batch)
    start_proposal = module.policy.compute_proposal_start_distribution(prepared_batch)

    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )
    target_distribution = module.policy.compute_forward_distribution(
        prepared_batch,
        state,
    )
    proposal_distribution = module.policy.compute_proposal_forward_distribution(
        prepared_batch,
        state,
    )

    assert torch.allclose(start_target.log_probs, start_proposal.log_probs, atol=1.0e-6)
    assert torch.allclose(
        target_distribution.edge_logits,
        proposal_distribution.edge_logits,
        atol=1.0e-6,
    )


def test_relation_action_prior_biases_proposal_edge_sampling() -> None:
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=3, sampling_temperature=1.0
        ),
        action_prior_cfg=_make_action_prior_config(
            "none",
            root_beta=0.0,
            edge_beta=1.0,
            relation_embedding_weight=1.0,
            target_node_weight=0.0,
            progress_weight=0.0,
        ),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    hidden_dim = int(prepared_batch.question_tokens.size(-1))
    forced_question = torch.zeros_like(prepared_batch.question_tokens)
    forced_question[0, 0] = 1.0
    forced_relations = torch.zeros_like(prepared_batch.relation_tokens)
    forced_relations[0, 0] = 1.0
    forced_relations[1, 1] = 1.0
    prepared_batch = replace(
        prepared_batch,
        question_tokens=forced_question,
        relation_tokens=forced_relations,
    )
    assert hidden_dim >= 2

    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )
    target_distribution = module.policy.compute_forward_distribution(
        prepared_batch, state
    )
    proposal_distribution = module.policy.compute_proposal_forward_distribution(
        prepared_batch,
        state,
    )
    assert proposal_distribution.is_stop_action is not None
    move_mask = ~proposal_distribution.is_stop_action
    move_logits = proposal_distribution.edge_logits[move_mask]
    move_edge_ids = proposal_distribution.edge_ids[move_mask]
    relation_ids = prepared_batch.topology.edge_type.index_select(0, move_edge_ids)
    relation0_logit = move_logits[relation_ids == 0].max().item()
    relation1_logit = move_logits[relation_ids == 1].max().item()

    assert not torch.allclose(
        target_distribution.edge_logits,
        proposal_distribution.edge_logits,
    )
    assert relation0_logit > relation1_logit


def test_control_state_intent_prior_biases_proposal_edge_sampling() -> None:
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=3, sampling_temperature=1.0
        ),
        action_prior_cfg=_make_action_prior_config(
            "none",
            root_beta=0.0,
            edge_beta=1.0,
            relation_embedding_weight=0.0,
            target_node_weight=0.0,
            progress_weight=0.0,
            intent_alignment_weight=1.0,
            intent_relation_weight=0.0,
            intent_target_weight=1.0,
        ),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    hidden_dim = int(prepared_batch.question_tokens.size(-1))
    forced_nodes = torch.zeros_like(prepared_batch.node_tokens)
    forced_nodes[1, 0] = 1.0
    forced_nodes[2, 1] = 1.0
    prepared_batch = replace(prepared_batch, node_tokens=forced_nodes)
    assert hidden_dim >= 2

    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
        control_state=torch.tensor(
            [[[1.0] + [0.0] * (hidden_dim - 1)]], dtype=torch.float32
        ),
    )
    target_distribution = module.policy.compute_forward_distribution(
        prepared_batch, state
    )
    proposal_distribution = module.policy.compute_proposal_forward_distribution(
        prepared_batch,
        state,
    )
    assert proposal_distribution.is_stop_action is not None
    move_mask = ~proposal_distribution.is_stop_action
    move_logits = proposal_distribution.edge_logits[move_mask]
    move_target_nodes = proposal_distribution.target_nodes[move_mask]
    target1_logit = move_logits[move_target_nodes == 1].max().item()
    target2_logit = move_logits[move_target_nodes == 2].max().item()

    assert not torch.allclose(
        target_distribution.edge_logits,
        proposal_distribution.edge_logits,
    )
    assert target1_logit > target2_logit


def test_shortest_path_action_prior_biases_proposal_toward_bridge_edge() -> None:
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=3),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
        ),
        action_prior_cfg=_make_action_prior_config(
            "none",
            root_beta=0.0,
            edge_beta=1.0,
            relation_embedding_weight=0.0,
            target_node_weight=0.0,
            progress_weight=0.0,
            intent_alignment_weight=0.0,
            shortest_path_edge_weight=1.0,
            answer_distance_weight=0.0,
        ),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 1], [1, 3, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 0, 0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="shortest-path-prior",
    )
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )

    proposal_distribution = module.policy.compute_proposal_forward_distribution(
        prepared_batch,
        state,
    )

    assert proposal_distribution.is_stop_action is not None
    move_mask = ~proposal_distribution.is_stop_action
    move_edge_ids = proposal_distribution.edge_ids[move_mask]
    move_logits = proposal_distribution.edge_logits[move_mask]
    shortest_logit = move_logits[move_edge_ids == 0].item()
    distractor_logit = move_logits[move_edge_ids == 1].item()
    assert shortest_logit > distractor_logit


def test_gflownet_training_step_logs_core_local_flow_metrics() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        beta=0.5,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=3,
            sampling_temperature=1.0,
        ),
    )
    captured_metrics: dict[str, object] = {}

    def _capture_metric_bundle(*, metrics: dict[str, object], **kwargs: object) -> None:
        del kwargs
        captured_metrics.update(metrics)

    module._log_metric_bundle = _capture_metric_bundle  # type: ignore[method-assign]

    loss = module.training_step(make_toy_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert "subtb_loss" in captured_metrics
    assert "subtb_root_loss" in captured_metrics
    assert "subtb_pairwise_loss" in captured_metrics
    assert "subtb_terminal_loss" in captured_metrics
    assert "subtb_residual" in captured_metrics
    assert "subtb_residual_variance_per_batch" in captured_metrics
    assert "subtb_root" in captured_metrics
    assert "unique_success_paths_per_100_rollouts" in captured_metrics
    assert "start_node_entropy" in captured_metrics
    assert "proposal_start_target_kl" in captured_metrics
    assert "active_forward_states" in captured_metrics
    assert "unique_forward_states" in captured_metrics
    assert "raw_graph_candidates" in captured_metrics
    assert "scored_graph_candidates" in captured_metrics
    assert "proposal_action_prior_scale" in captured_metrics
    assert "proposal_root_beta" in captured_metrics
    assert "proposal_edge_beta" in captured_metrics
    assert "proposal_stop_beta" in captured_metrics
    assert "proposal_intent_alignment_weight" in captured_metrics
    assert "proposal_intent_alignment_strength" in captured_metrics
    assert "success_replay_mix_alpha" in captured_metrics
    assert "coverage_replay_mix_alpha" in captured_metrics
    assert "success_replay_buffer_size" in captured_metrics
    assert "success_replay_ready" in captured_metrics
    assert "success_replay_added" in captured_metrics
    assert "success_replay_sampled" in captured_metrics
    assert "online_subtb_loss" in captured_metrics
    assert "replay_subtb_loss" in captured_metrics
    assert module.cfg.training_cfg.step_log_penalty is not None
    assert captured_metrics["step_log_penalty"] == pytest.approx(
        float(module.cfg.training_cfg.step_log_penalty)
    )
    assert captured_metrics["terminal_failure_log_reward"] == pytest.approx(
        float(module.cfg.training_cfg.terminal_failure_log_reward)
    )
    assert captured_metrics["proposal_action_prior_scale"] == pytest.approx(1.0)
    assert captured_metrics["proposal_root_beta"] == pytest.approx(
        float(module.cfg.action_prior_cfg.root_beta or 0.0)
    )
    assert captured_metrics["proposal_edge_beta"] == pytest.approx(
        float(module.cfg.action_prior_cfg.edge_beta or 0.0)
    )
    assert captured_metrics["proposal_stop_beta"] == pytest.approx(
        float(module.cfg.action_prior_cfg.stop_beta)
    )
    assert captured_metrics["proposal_intent_alignment_weight"] == pytest.approx(
        float(module.cfg.action_prior_cfg.intent_alignment_weight)
    )
    assert captured_metrics["proposal_intent_alignment_strength"] == pytest.approx(
        float(module.cfg.action_prior_cfg.edge_beta or 0.0)
        * float(module.cfg.action_prior_cfg.intent_alignment_weight)
    )
    assert captured_metrics["success_replay_mix_alpha"] == pytest.approx(0.0)
    assert captured_metrics["coverage_replay_mix_alpha"] == pytest.approx(0.0)
    assert captured_metrics["success_replay_buffer_size"] == pytest.approx(0.0)
    assert captured_metrics["success_replay_ready"] == pytest.approx(0.0)
    assert captured_metrics["success_replay_added"] == pytest.approx(0.0)
    assert captured_metrics["success_replay_sampled"] == pytest.approx(0.0)
    assert captured_metrics["replay_subtb_loss"] == pytest.approx(0.0)
    assert not any(str(key).startswith("exact_aux") for key in captured_metrics)


def test_gflownet_training_step_mixes_success_replay_loss_when_buffer_ready() -> None:
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            force_stop_on_answer_hit=True,
            success_replay=SuccessReplayConfig(
                mix_alpha=0.25,
                min_buffer_size=1,
                capacity=8,
                replay_trajectories_per_step=1,
            ),
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=1,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0], dtype=torch.long),
        answer_entity_ids=torch.tensor([100], dtype=torch.long),
        node_entity_ids=torch.tensor([100], dtype=torch.long),
        sample_id="replay-ready-training-step",
    )
    captured_metrics: dict[str, object] = {}
    loss_outputs = [_make_loss_output(2.0), _make_loss_output(6.0)]

    def _capture_metric_bundle(*, metrics: dict[str, object], **kwargs: object) -> None:
        del kwargs
        captured_metrics.update(metrics)

    def _fake_loss_compute(
        *args: object, **kwargs: object
    ) -> SubTrajectoryBalanceLossOutput:
        del args, kwargs
        return loss_outputs.pop(0)

    module._log_metric_bundle = _capture_metric_bundle  # type: ignore[method-assign]
    module.loss_fn.compute = _fake_loss_compute  # type: ignore[method-assign]

    loss = module.training_step(batch, batch_idx=0)

    assert loss.item() == pytest.approx(3.0)
    assert captured_metrics["subtb_loss"] == pytest.approx(3.0)
    assert captured_metrics["online_subtb_loss"] == pytest.approx(2.0)
    assert captured_metrics["replay_subtb_loss"] == pytest.approx(6.0)
    assert captured_metrics["success_replay_mix_alpha"] == pytest.approx(0.25)
    assert captured_metrics["coverage_replay_mix_alpha"] == pytest.approx(0.25)
    assert captured_metrics["success_replay_buffer_size"] == pytest.approx(1.0)
    assert captured_metrics["success_replay_ready"] == pytest.approx(1.0)
    assert captured_metrics["success_replay_added"] == pytest.approx(1.0)
    assert captured_metrics["success_replay_sampled"] == pytest.approx(1.0)


def test_training_rollout_metrics_report_forward_search_observability() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    sample_batch = replace(
        _make_manual_sample_batch(
            batch=batch,
            rollouts_per_graph=2,
            success_mask=torch.tensor([[True, False]], dtype=torch.bool),
            start_entropy=0.3,
            start_entropy_normalized=0.6,
        ),
        proposal_start_target_kl=torch.tensor([0.12], dtype=torch.float32),
        total_active_agent_count=10,
        total_unique_active_state_count=4,
        total_raw_graph_candidate_count=18,
        total_scored_graph_candidate_count=7,
    )

    metrics = module._compute_training_rollout_metrics(
        batch=batch, sample_batch=sample_batch
    )

    assert metrics.active_forward_states == pytest.approx(10.0)
    assert metrics.unique_forward_states == pytest.approx(4.0)
    assert metrics.forward_state_dedup_keep_ratio == pytest.approx(0.4)
    assert metrics.proposal_start_target_kl.item() == pytest.approx(0.12)
    assert metrics.raw_graph_candidates == pytest.approx(18.0)
    assert metrics.scored_graph_candidates == pytest.approx(7.0)
    assert metrics.raw_graph_candidates_per_unique_state == pytest.approx(4.5)
    assert metrics.scored_graph_candidates_per_unique_state == pytest.approx(1.75)


def test_sampler_skips_move_backward_reconstruction() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    assert module.sampler is not None

    def _unexpected_backward_distribution(
        *args: object, **kwargs: object
    ) -> torch.Tensor:
        del args, kwargs
        raise AssertionError(
            "training sampler should not reconstruct move backward logits"
        )

    module.policy.compute_backward_distribution = _unexpected_backward_distribution  # type: ignore[method-assign]

    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=2,
        temperature=1.0,
    )

    assert sample_batch.trace_stop_mask is not None
    assert torch.isfinite(sample_batch.log_pf_steps).all()
    assert torch.equal(
        sample_batch.log_pb_steps[~sample_batch.trace_stop_mask],
        torch.zeros_like(sample_batch.log_pb_steps[~sample_batch.trace_stop_mask]),
    )


def test_gflownet_training_step_raises_on_nonfinite_loss() -> None:
    module = _make_module("topology")
    module.loss_fn.compute = lambda *args, **kwargs: SubTrajectoryBalanceLossOutput(  # type: ignore[method-assign]
        loss=torch.tensor(float("nan")),
        subtb_loss=torch.tensor(float("nan")),
        residual_abs=torch.tensor(0.0),
        residual_variance=torch.tensor(0.0),
        root_abs=torch.tensor(0.0),
        success_rate=torch.tensor(0.0),
        log_z_mean=torch.tensor(0.0),
        log_z_variance=torch.tensor(0.0),
    )

    with pytest.raises(RuntimeError, match="Non-finite training loss detected"):
        module.training_step(make_toy_batch(), batch_idx=0)


def test_gflownet_training_step_logs_effective_pass_when_schedule_is_set() -> None:
    module = _make_module("topology")
    module.set_fit_schedule(
        ResolvedPassFitSchedule(
            max_passes=12.0,
            val_every_passes=2.0,
            early_stopping_patience_passes=6.0,
            train_size=12,
            per_device_batch_size=6,
            data_parallel_size=1,
            global_batch_size=6,
            accumulate_grad_batches=1,
            examples_per_optimizer_step=6,
            train_batches_per_pass=2.0,
            optimizer_steps_per_pass=2.0,
            max_steps=24,
            val_check_interval_batches=4,
            early_stopping_patience_checks=3,
        )
    )
    captured_metrics: dict[str, object] = {}

    def _capture_metric_bundle(*, metrics: dict[str, object], **kwargs: object) -> None:
        del kwargs
        captured_metrics.update(metrics)

    module._log_metric_bundle = _capture_metric_bundle  # type: ignore[method-assign]

    module.training_step(make_toy_batch(), batch_idx=0)

    assert captured_metrics["effective_pass"] == pytest.approx(0.5)


def test_gflownet_sampling_temperature_schedule_anneals() -> None:
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            sampling_temperature=2.0,
            sampling_temperature_schedule=SamplingTemperatureScheduleConfig(
                type="linear",
                final_temperature=0.5,
                total_steps=4,
            ),
        ),
        action_prior_cfg=ActionPriorConfig(),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )

    assert module._resolve_sampling_temperature(global_step=0) == pytest.approx(2.0)
    assert module._resolve_sampling_temperature(global_step=3) == pytest.approx(0.5)
    assert module._resolve_sampling_temperature(global_step=1) == pytest.approx(1.5)


def test_gflownet_action_prior_schedule_anneals() -> None:
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            action_prior_schedule=ActionPriorScheduleConfig(
                type="linear",
                final_scale=0.25,
                total_steps=4,
            )
        ),
        action_prior_cfg=_make_action_prior_config(
            "topology",
            root_beta=0.6,
            edge_beta=0.6,
        ),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )

    assert module._resolve_action_prior_scale(global_step=0) == pytest.approx(1.0)
    assert module._resolve_action_prior_scale(global_step=3) == pytest.approx(0.25)
    assert module._resolve_action_prior_scale(global_step=1) == pytest.approx(0.75)


def test_sampler_applies_constant_step_log_penalty_to_each_move() -> None:
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=2,
            sampling_temperature=1.0,
            step_log_penalty=float(torch.log(torch.tensor(0.5)).item()),
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([7, 8, 9], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="constant-step-penalty",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    original_forward = module.policy.compute_forward_distribution

    def _prefer_graph_moves(prepared_batch_arg, state, **kwargs):  # noqa: ANN001
        distribution = original_forward(prepared_batch_arg, state, **kwargs)
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_stop_action is not None:
            logits[distribution.is_stop_action.to(dtype=torch.bool)] = -1.0e9
        return replace(distribution, edge_logits=logits)

    module.policy.compute_forward_distribution = _prefer_graph_moves  # type: ignore[method-assign]

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.log_reward_steps is not None
    step_penalty = float(torch.log(torch.tensor(0.5)).item())
    move_penalties = sample_batch.log_reward_steps[sample_batch.move_mask]
    assert int(move_penalties.numel()) >= 1
    assert torch.allclose(move_penalties, torch.full_like(move_penalties, step_penalty))
    assert torch.equal(
        sample_batch.log_reward_steps[~sample_batch.move_mask],
        torch.zeros_like(sample_batch.log_reward_steps[~sample_batch.move_mask]),
    )


def test_root_distribution_applies_answer_distance_potential_to_start_actions() -> None:
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            step_log_penalty=0.0,
            potential_reward=PotentialRewardConfig(
                answer_distance_weight=0.5,
            ),
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([7, 8], dtype=torch.long),
        q_local_indices=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="root-potential-reward",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    module.policy.base_policy.compute_start_log_flows = (  # type: ignore[method-assign]
        lambda **kwargs: torch.zeros(
            (int(kwargs["candidate_nodes_abs"].numel()),),
            device=kwargs["candidate_nodes_abs"].device,
            dtype=torch.float32,
        )
    )

    distribution = module.policy.compute_root_action_distribution(prepared_batch)
    rewards = distribution.start_log_rewards

    assert rewards is not None
    reward_by_node = {
        int(node.item()): float(reward.item())
        for node, reward in zip(distribution.candidate_nodes_abs, rewards)
    }
    assert reward_by_node[0] == pytest.approx(-1.0)
    assert reward_by_node[1] == pytest.approx(-0.5)
    log_prob_by_node = {
        int(node.item()): float(log_prob.item())
        for node, log_prob in zip(
            distribution.candidate_nodes_abs, distribution.log_probs
        )
    }
    assert log_prob_by_node[1] > log_prob_by_node[0]


def test_forward_distribution_applies_answer_distance_potential_to_move_logits() -> (
    None
):
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            step_log_penalty=0.0,
            potential_reward=PotentialRewardConfig(
                answer_distance_weight=0.5,
            ),
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([7, 8], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="move-potential-reward",
    )
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
    )

    module.policy.base_policy.state_score_head.forward = (  # type: ignore[method-assign]
        lambda node_features, question_features: torch.zeros(
            (int(node_features.size(0)),),
            device=node_features.device,
            dtype=torch.float32,
        )
    )

    distribution = module.policy.compute_forward_distribution(prepared_batch, state)
    stop_mask = distribution.is_stop_action
    assert stop_mask is not None
    move_logits = distribution.edge_logits[~stop_mask].to(dtype=torch.float32)

    assert move_logits.tolist() == pytest.approx([0.5])


def test_potential_rewards_telescope_to_zero_on_successful_answer_path() -> None:
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            step_log_penalty=0.0,
            potential_reward=PotentialRewardConfig(
                answer_distance_weight=0.5,
            ),
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([7, 8], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="potential-telescoping",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    assert isinstance(module.sampler, ForwardTrajectoryGFNSampler)
    sample_batch = module.sampler.rebuild_sample_batch(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        planned_edge_ids=torch.tensor([[[0, 1, -1]]], dtype=torch.long),
        planned_stop_mask=torch.tensor([[[False, False, True]]], dtype=torch.bool),
        path_lengths=torch.tensor([[2]], dtype=torch.long),
        termination_action_steps=torch.tensor([[3]], dtype=torch.long),
        trace_nodes=torch.tensor([[[0, 1, 2]]], dtype=torch.long),
        trace_edge_ids=torch.tensor([[[0, 1, -1]]], dtype=torch.long),
        trace_num_steps=torch.tensor([[[0, 1, 2]]], dtype=torch.long),
        trace_mask=torch.tensor([[[True, True, True]]], dtype=torch.bool),
        trace_stop_mask=torch.tensor([[[False, False, True]]], dtype=torch.bool),
    )

    assert sample_batch.start_log_rewards is not None
    assert sample_batch.log_reward_steps is not None
    assert sample_batch.start_log_rewards[0, 0].item() == pytest.approx(-1.0)
    assert sample_batch.log_reward_steps[0, 0, 0].item() == pytest.approx(0.5)
    assert sample_batch.log_reward_steps[0, 0, 1].item() == pytest.approx(0.5)
    assert sample_batch.log_reward_steps[0, 0, 2].item() == pytest.approx(0.0)
    total_shaping = float(sample_batch.start_log_rewards[0, 0].item()) + float(
        sample_batch.log_reward_steps[0, 0].sum().item()
    )
    assert total_shaping == pytest.approx(0.0)


def test_sampler_applies_answer_stop_bonus_on_gold_stop() -> None:
    bonus = 0.7
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            answer_stop_log_reward_bonus=bonus,
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=1,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0], dtype=torch.long),
        answer_entity_ids=torch.tensor([100], dtype=torch.long),
        node_entity_ids=torch.tensor([100], dtype=torch.long),
        sample_id="answer-stop-bonus",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.log_reward_steps is not None
    assert sample_batch.trace_stop_mask is not None
    assert sample_batch.terminal_log_rewards[0, 0].item() == pytest.approx(0.0)
    assert bool(sample_batch.trace_stop_mask[0, 0, 0].item()) is True
    assert sample_batch.log_reward_steps[0, 0, 0].item() == pytest.approx(bonus)
    assert torch.equal(
        sample_batch.log_reward_steps[~sample_batch.trace_stop_mask],
        torch.zeros_like(sample_batch.log_reward_steps[~sample_batch.trace_stop_mask]),
    )


def test_terminal_supervisor_builds_answer_sink_targets() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor(
        non_gold_terminal_log_reward=-3.0,
        gold_reward_mode="shared",
    )
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1, 2], dtype=torch.long),
        answer_entity_ids=torch.tensor([101, 102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="answer-sink-targets",
    )

    transitions = supervisor.resolve_terminal_transitions(
        batch=batch,
        terminal_nodes=torch.tensor([[1, 2, 3]], dtype=torch.long),
    )

    assert torch.equal(
        transitions.terminal_is_gold,
        torch.tensor([[True, True, False]], dtype=torch.bool),
    )
    assert torch.equal(
        transitions.terminal_sink_ids,
        torch.tensor([[0, 1, 2]], dtype=torch.long),
    )
    expected_gold_log_reward = -float(torch.log(torch.tensor(2.0)).item())
    assert torch.allclose(
        transitions.terminal_sink_log_rewards,
        torch.tensor(
            [[expected_gold_log_reward, expected_gold_log_reward, -3.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        transitions.gold_answer_counts, torch.tensor([2], dtype=torch.long)
    )


def test_answer_quotient_loss_deduplicates_identical_terminal_paths() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        answer_quotient_config=AnswerQuotientConfig(
            enabled=True,
            weight=1.0,
            replace_terminal_loss=True,
        )
    )
    sample_batch = TrajectoryGFNSampleBatch(
        graph_log_z=torch.zeros((1,), dtype=torch.float32),
        start_nodes=torch.tensor([[0, 0, 1]], dtype=torch.long),
        start_log_probs=torch.tensor([[-0.7, -0.7, -1.2]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[-0.7, -0.7, -1.2]], dtype=torch.float32),
        log_pf_steps=torch.zeros((1, 3, 1), dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 3, 1), dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 3, 1), dtype=torch.float32),
        move_mask=torch.zeros((1, 3, 1), dtype=torch.bool),
        trace_nodes=torch.tensor([[[0], [0], [1]]], dtype=torch.long),
        trace_edge_ids=torch.full((1, 3, 1), fill_value=-1, dtype=torch.long),
        trace_num_steps=torch.zeros((1, 3, 1), dtype=torch.long),
        trace_mask=torch.ones((1, 3, 1), dtype=torch.bool),
        trace_stop_mask=torch.ones((1, 3, 1), dtype=torch.bool),
        terminal_nodes=torch.tensor([[0, 0, 1]], dtype=torch.long),
        terminal_entity_ids=torch.tensor([[101, 101, 102]], dtype=torch.long),
        terminal_is_gold=torch.tensor([[True, True, True]], dtype=torch.bool),
        terminal_sink_ids=torch.tensor([[0, 0, 1]], dtype=torch.long),
        terminal_sink_log_rewards=torch.tensor(
            [[-0.7, -0.7, -1.2]], dtype=torch.float32
        ),
        gold_answer_counts=torch.tensor([2], dtype=torch.long),
        terminal_num_steps=torch.zeros((1, 3), dtype=torch.long),
        termination_action_steps=torch.ones((1, 3), dtype=torch.long),
        terminal_rewards=torch.ones((1, 3), dtype=torch.float32),
        terminal_log_rewards=torch.zeros((1, 3), dtype=torch.float32),
        success_mask=torch.ones((1, 3), dtype=torch.bool),
    )

    output = loss_fn.compute(sample_batch)

    assert output.loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert output.answer_quotient_component_loss.item() == pytest.approx(
        0.0, abs=1.0e-6
    )
    assert output.answer_quotient_residual_abs.item() == pytest.approx(0.0, abs=1.0e-6)
    assert output.answer_quotient_observed_sink_count.item() == pytest.approx(2.0)


def test_forward_distribution_allocates_stop_mass_within_answer_sink() -> None:
    module = _make_module_with_training_cfg(
        "none",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            answer_quotient=AnswerQuotientConfig(
                enabled=True,
                allocate_stop_mass=True,
            ),
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1, 2], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 101], dtype=torch.long),
        sample_id="allocated-stop-mass",
    )
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[1, 2]], dtype=torch.long),
        done_mask=torch.zeros((1, 2), dtype=torch.bool),
        num_steps=torch.zeros((1, 2), dtype=torch.long),
    )

    module.policy.base_policy._compute_stop_allocation_scores = (  # type: ignore[method-assign]
        lambda state_features: torch.zeros(
            (int(state_features.size(0)),),
            device=state_features.device,
            dtype=torch.float32,
        )
    )

    distribution = module.policy.compute_forward_distribution(prepared_batch, state)
    stop_mask = distribution.is_stop_action
    assert stop_mask is not None
    expected_log_mass = -float(torch.log(torch.tensor(2.0)).item())
    assert torch.allclose(
        distribution.edge_logits[stop_mask].to(dtype=torch.float32),
        torch.full((2,), fill_value=expected_log_mass, dtype=torch.float32),
    )


def test_sampler_emits_deterministic_backward_log_probs_for_path_state() -> None:
    module = _make_module("topology")
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [2, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="uniform-backward",
    )
    prepared_batch = module.policy.prepare_batch(batch)
    assert module.sampler is not None

    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )

    assert sample_batch.log_pb_steps[0, 0, 0].item() == pytest.approx(0.0)


def test_forward_distribution_is_derived_from_child_flows_and_terminal_rewards() -> (
    None
):
    step_log_penalty = float(torch.log(torch.tensor(0.5)).item())
    terminal_failure_log_reward = -2.0
    module = _make_module_with_training_cfg(
        "topology",
        beta=0.0,
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            step_log_penalty=step_log_penalty,
            terminal_failure_log_reward=terminal_failure_log_reward,
        ),
    )
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )

    module.policy.base_policy.state_score_head.forward = (  # type: ignore[method-assign]
        lambda node_features, question_features: torch.arange(
            int(node_features.size(0)),
            device=node_features.device,
            dtype=torch.float32,
        )
        + 10.0
    )

    distribution = module.policy.compute_forward_distribution(prepared_batch, state)
    submit_mask = distribution.is_stop_action
    assert submit_mask is not None
    graph_mask = ~submit_mask
    child_states = [
        SearchState.from_edge_path(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            start_node=0,
            edge_ids=(int(edge_id),),
            max_steps=int(module.cfg.horizon_cfg.max_steps),
            device=batch.node_ptr.device,
        )
        for edge_id in distribution.edge_ids[graph_mask].tolist()
    ]
    child_state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.cat([path.current_nodes for path in child_states], dim=0),
        done_mask=torch.zeros(
            (int(distribution.target_nodes[graph_mask].numel()), 1), dtype=torch.bool
        ),
        num_steps=torch.cat([path.num_steps for path in child_states], dim=0),
        path_token_ids=torch.cat([path.path_token_ids for path in child_states], dim=0),
    )
    expected_child_log_flows = module.policy.compute_log_state_scores(
        prepared_batch,
        child_state,
    ).view(-1)

    assert torch.allclose(
        distribution.edge_logits[graph_mask].to(dtype=torch.float32),
        expected_child_log_flows + step_log_penalty,
    )
    assert torch.allclose(
        distribution.edge_logits[submit_mask].to(dtype=torch.float32),
        torch.tensor([terminal_failure_log_reward], dtype=torch.float32),
    )


def test_transition_head_only_changes_proposal_logits() -> None:
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            step_log_penalty=0.0,
        ),
        action_prior_cfg=_make_action_prior_config(
            "none",
            root_beta=0.0,
            edge_beta=0.0,
        ),
        policy_cfg=_make_policy_config(transition_enabled=True),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )

    module.policy.base_policy.state_score_head.forward = (  # type: ignore[method-assign]
        lambda node_features, question_features: torch.full(
            (int(node_features.size(0)),),
            fill_value=3.0,
            device=node_features.device,
            dtype=torch.float32,
        )
    )
    assert module.policy.base_policy.transition_proposal_head is not None
    module.policy.base_policy.transition_proposal_head.forward = (  # type: ignore[method-assign]
        lambda current_state_features,
        candidate_state_features,
        relation_features: torch.arange(
            int(current_state_features.size(0)),
            device=current_state_features.device,
            dtype=torch.float32,
        )
    )

    target_distribution = module.policy.compute_forward_distribution(
        prepared_batch, state
    )
    proposal_distribution = module.policy.compute_proposal_forward_distribution(
        prepared_batch,
        state,
        action_prior_scale=0.0,
    )
    assert target_distribution.is_stop_action is not None
    assert proposal_distribution.is_stop_action is not None
    target_move_logits = target_distribution.edge_logits[
        ~target_distribution.is_stop_action
    ]
    proposal_move_logits = proposal_distribution.edge_logits[
        ~proposal_distribution.is_stop_action
    ]
    assert target_move_logits.tolist() == pytest.approx([3.0, 3.0])
    assert proposal_move_logits.tolist() == pytest.approx([3.0, 4.0])


def test_base_policy_no_longer_registers_stop_relation_buffer() -> None:
    module = _make_module("topology")

    parameter_names = dict(module.policy.base_policy.named_parameters())
    buffer_names = dict(module.policy.base_policy.named_buffers())

    assert "stop_action_relation_feature" not in parameter_names
    assert "stop_action_relation_feature" not in buffer_names


def test_forward_distribution_matches_under_aggressive_chunking() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState.initialize(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_nodes=torch.tensor([[0, 0, 0]], dtype=torch.long),
        max_steps=2,
    )

    baseline = module.policy.compute_forward_distribution(prepared_batch, state)
    original_chunk = gflownet_policy_impl._CANDIDATE_SCORING_CHUNK_SIZE
    gflownet_policy_impl._CANDIDATE_SCORING_CHUNK_SIZE = 1
    try:
        chunked = module.policy.compute_forward_distribution(prepared_batch, state)
    finally:
        gflownet_policy_impl._CANDIDATE_SCORING_CHUNK_SIZE = original_chunk

    assert torch.equal(chunked.edge_agent_batch, baseline.edge_agent_batch)
    assert torch.equal(chunked.edge_ids, baseline.edge_ids)
    assert torch.equal(chunked.target_nodes, baseline.target_nodes)
    assert torch.equal(chunked.out_degrees, baseline.out_degrees)
    assert torch.equal(chunked.is_stop_action, baseline.is_stop_action)
    assert torch.allclose(
        chunked.edge_logits.to(dtype=torch.float32),
        baseline.edge_logits.to(dtype=torch.float32),
        atol=1.0e-6,
    )


@pytest.mark.parametrize("h_kind", ["topology", "embedding"])
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


def test_validation_step_does_not_use_proposal_policy_hooks() -> None:
    module = _make_module("topology")

    def _unexpected_proposal(*args, **kwargs):  # noqa: ANN001
        del args, kwargs
        raise AssertionError("validation must stay on the target policy")

    module.policy.compute_proposal_start_distribution = _unexpected_proposal  # type: ignore[method-assign]
    module.policy.compute_proposal_edge_logits = _unexpected_proposal  # type: ignore[method-assign]

    module.validation_step(make_toy_batch(), batch_idx=0)
