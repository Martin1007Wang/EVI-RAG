from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch

import src.models.gflownet_module as gflownet_module_impl
from src.models.configs import (
    ExactAnswerObjectiveConfig,
    SearchEvalConfig,
    BackboneConfig,
    GFlowNetTrainingConfig,
    HeuristicConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SamplingTemperatureScheduleConfig,
    SchedulerConfig,
    StateScoreHeadConfig,
    SuccessfulTrajectoryReplayConfig,
)
from src.models.gflownet import (
    SearchState,
    TrainingScheduleContext,
    compute_embedding_log_heuristic,
    compute_topology_log_heuristic,
)
from src.graph_runtime import TrajectoryBatch, build_graph_batch
from src.metrics.answer_reachability.exact_analysis import ExactDynamicProgramResult
from src.models.gflownet_module import (
    AuxiliaryLossResult,
    GFlowNetModule,
    PredictionArtifactWriteConfig,
)
from src.metrics.answer_reachability.runtime import (
    SearchMetricRuntimeFactory,
)
from src.utils.fit_schedule import ResolvedPassFitSchedule

from .conftest import make_batch_from_graph, make_toy_batch


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
            learned_hidden_dim=16,
            learned_dropout=0.0,
        ),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(metrics_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )


def _make_module_with_training_cfg(
    h_kind: str,
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
                rollout_batch_size=3,
                reward_epsilon=1.0e-3,
                failure_reward_mode="graph_normalized",
                sampling_temperature=1.0,
            )
        ),
        heuristic_cfg=HeuristicConfig(
            kind=h_kind,
            beta=beta,
            topology_restart_prob=0.3,
            topology_num_iters=6,
            topology_eps=1.0e-8,
            embedding_temperature=0.7,
            learned_hidden_dim=16,
            learned_dropout=0.0,
        ),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(metrics_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
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
        eval_cfg=SearchEvalConfig(metrics_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )

    assert module.cfg.heuristic_cfg.beta == 0.0
    assert module.policy.heuristic_cfg.beta == 0.0


def test_gflownet_module_exposes_eval_settings() -> None:
    module = _make_module("topology")

    assert module.metrics_profile == "rank_only"
    assert module.evaluation_task == "answer_ranking"


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
    assert captured_metrics["sampling_temperature"] == pytest.approx(1.0)


def test_start_distribution_defines_virtual_source_log_z() -> None:
    module = _make_module("topology")
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [2, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="multi-start-log-z",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    start_dist = module.policy.compute_start_distribution(prepared_batch)
    sampled_nodes, sampled_log_probs, sampled_log_flows = (
        module.policy.sample_start_nodes(
            start_dist,
            num_rollouts=4,
            deterministic=True,
        )
    )

    assert torch.allclose(
        start_dist.graph_log_z,
        torch.tensor([torch.logsumexp(start_dist.log_flows, dim=0).item()]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        module.policy.compute_graph_log_z(prepared_batch),
        start_dist.graph_log_z,
        atol=1.0e-6,
    )
    assert sampled_nodes.shape == (1, 4)
    assert torch.allclose(
        sampled_log_flows - sampled_log_probs,
        start_dist.graph_log_z.unsqueeze(1).expand_as(sampled_log_probs),
        atol=1.0e-6,
    )


def test_target_policy_ignores_behavior_heuristic_beta() -> None:
    torch.manual_seed(5)
    target_only = _make_module_with_training_cfg("topology", beta=0.0)
    torch.manual_seed(5)
    behavior_guided = _make_module_with_training_cfg("topology", beta=0.5)
    batch = make_toy_batch()

    prepared_target = target_only.policy.prepare_batch(batch)
    prepared_guided = behavior_guided.policy.prepare_batch(batch)

    start_target = target_only.policy.compute_start_distribution(prepared_target)
    start_guided = behavior_guided.policy.compute_start_distribution(prepared_guided)
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
    guided_target_distribution = behavior_guided.policy.compute_forward_distribution(
        prepared_guided,
        state,
    )
    behavior_distribution = (
        behavior_guided.policy.compute_behavior_forward_distribution(
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
        behavior_distribution.edge_logits,
    )


def test_gflownet_training_step_logs_exact_auxiliary_metrics() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        beta=0.5,
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
            sampling_temperature=1.0,
            exact_aux=ExactAnswerObjectiveConfig(
                enabled=True,
                success_weight=0.1,
                coverage_weight=0.2,
                warmup_passes=0.0,
                interval_steps=1,
                max_graphs_per_batch=1,
            ),
        ),
    )
    captured_metrics: dict[str, object] = {}

    def _capture_metric_bundle(*, metrics: dict[str, object], **kwargs: object) -> None:
        del kwargs
        captured_metrics.update(metrics)

    module._log_metric_bundle = _capture_metric_bundle  # type: ignore[method-assign]

    loss = module.training_step(make_toy_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert "exact_aux_loss" in captured_metrics
    assert "exact_aux_success_loss" in captured_metrics
    assert "exact_aux_coverage_loss" in captured_metrics


def test_aggregate_entity_log_mass_handles_duplicate_entity_ids() -> None:
    aggregated = GFlowNetModule._aggregate_entity_log_mass(
        node_entity_ids=torch.tensor([4, 5, 4, 7], dtype=torch.long),
        log_node_mass=torch.log(
            torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        ),
        entity_ids=torch.tensor([7, 4, 6, 4], dtype=torch.long),
    )

    assert torch.allclose(
        aggregated[torch.isfinite(aggregated)].exp(),
        torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32),
        atol=1.0e-6,
    )
    assert bool(torch.isneginf(aggregated[2]).item()) is True


def test_exact_auxiliary_loss_uses_log_space_mass() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            exact_aux=ExactAnswerObjectiveConfig(
                enabled=True,
                success_weight=1.0,
                coverage_weight=0.0,
                warmup_passes=0.0,
                interval_steps=1,
                max_graphs_per_batch=1,
            )
        ),
    )
    batch = make_toy_batch()

    module.training_exact_analyzer.compute_dynamic_program = (  # type: ignore[method-assign]
        lambda **kwargs: ExactDynamicProgramResult(
            log_terminal_mass=torch.tensor(
                [float("-inf"), float("-inf"), -120.0], dtype=torch.float32
            ),
            log_retrieval_terminal_mass=torch.tensor(
                [float("-inf"), float("-inf"), -120.0], dtype=torch.float32
            ),
            log_success_by_step=torch.full((3, 3), fill_value=float("-inf")),
            log_gold_mass=torch.tensor(-120.0, dtype=torch.float32),
            log_gold_mass_by_graph=torch.tensor([-120.0], dtype=torch.float32),
            log_edge_success_mass=torch.full(
                (int(batch.edge_index.size(1)),), fill_value=float("-inf")
            ),
        )
    )

    result = module._compute_exact_auxiliary_loss(batch=batch)

    assert result is not None
    assert float(result.metrics["exact_aux_success_loss"].item()) == pytest.approx(
        120.0, rel=1.0e-5
    )


def test_success_replay_rollout_resolution_rejects_invalid_ratio() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            success_replay=SuccessfulTrajectoryReplayConfig(
                enabled=True,
                ratio=0.25,
                warmup_passes=0.0,
                min_buffer_size=1,
                max_buffer_size=8,
                max_trajectories_per_sample=2,
            )
        ),
    )
    object.__setattr__(module.cfg.training_cfg.success_replay, "ratio", 1.1)

    with pytest.raises(ValueError, match="training.success_replay.ratio"):
        module._resolve_success_replay_rollouts_per_graph()


def test_success_replay_rollout_resolution_matches_target_fraction() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            success_replay=SuccessfulTrajectoryReplayConfig(
                enabled=True,
                ratio=0.25,
                warmup_passes=0.0,
                min_buffer_size=1,
                max_buffer_size=8,
                max_trajectories_per_sample=2,
            ),
        ),
    )

    assert module._resolve_success_replay_rollouts_per_graph() == 1


def test_exact_auxiliary_loss_prepares_shared_batch_once() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            exact_aux=ExactAnswerObjectiveConfig(
                enabled=True,
                success_weight=1.0,
                coverage_weight=0.0,
                warmup_passes=0.0,
                interval_steps=1,
                max_graphs_per_batch=2,
            )
        ),
    )
    batch = TrajectoryBatch.concatenate([make_toy_batch(), make_toy_batch()])
    prepare_calls = 0
    dp_calls = 0
    original_prepare_batch = module.policy.prepare_batch
    original_compute_dynamic_program = (
        module.training_exact_analyzer.compute_dynamic_program
    )

    def _count_prepare(batch_arg: TrajectoryBatch):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare_batch(batch_arg)

    def _count_dynamic_program(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal dp_calls
        dp_calls += 1
        return original_compute_dynamic_program(**kwargs)

    module.policy.prepare_batch = _count_prepare  # type: ignore[method-assign]
    module.training_exact_analyzer.compute_dynamic_program = _count_dynamic_program  # type: ignore[method-assign]

    result = module._compute_exact_auxiliary_loss(batch=batch)

    assert result is not None
    assert prepare_calls == 1
    assert dp_calls == 1


def test_gflownet_training_step_raises_on_nonfinite_loss() -> None:
    module = _make_module("topology")
    module._compute_exact_auxiliary_loss = lambda **kwargs: AuxiliaryLossResult(  # type: ignore[method-assign]
        loss=torch.tensor(float("nan")),
        metrics={},
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
        heuristic_cfg=HeuristicConfig(beta=0.0),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(metrics_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )

    assert module._resolve_sampling_temperature(global_step=0) == pytest.approx(2.0)
    assert module._resolve_sampling_temperature(global_step=3) == pytest.approx(0.5)
    assert module._resolve_sampling_temperature(global_step=1) == pytest.approx(1.5)


def test_sampler_emits_uniform_backward_log_probs_for_multi_parent_state() -> None:
    module = _make_module("topology")
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [2, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="uniform-backward",
    )
    prepared_batch = module.policy.prepare_batch(batch)
    assert module.sampler is not None

    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollout_batch_size=1,
        temperature=1.0,
    )

    assert sample_batch.log_pb_steps[0, 0, 0].item() == pytest.approx(-math.log(2.0))


def test_forward_distribution_uses_effective_child_flows_plus_backward() -> None:
    module = _make_module("topology")
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [2, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="forward-effective-flow",
    )
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )

    distribution = module.policy.compute_forward_distribution(prepared_batch, state)
    child_state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=distribution.target_nodes.view(-1, 1),
        done_mask=torch.zeros(
            (int(distribution.target_nodes.numel()), 1), dtype=torch.bool
        ),
        num_steps=torch.ones(
            (int(distribution.target_nodes.numel()), 1), dtype=torch.long
        ),
    )
    expected_child_log_flows = module.policy.compute_log_state_scores(
        prepared_batch,
        child_state,
    ).view(-1)
    expected_log_pb = torch.full_like(expected_child_log_flows, -math.log(2.0))

    assert torch.allclose(
        distribution.edge_logits.to(dtype=torch.float32),
        expected_child_log_flows + expected_log_pb,
        atol=1.0e-6,
    )


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
