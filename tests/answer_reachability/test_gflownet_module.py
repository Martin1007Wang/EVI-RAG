from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

import src.models.gflownet.policy as gflownet_policy_impl
import src.models.gflownet_module as gflownet_module_impl
from src.models.configs import (
    AdaptiveSamplingConfig,
    AnswerRewardConfig,
    CandidateShortlistConfig,
    SearchEvalConfig,
    BackboneConfig,
    GFlowNetTrainingConfig,
    GuidanceLossConfig,
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
    ForwardTrajectoryGFNSampler,
    SearchState,
    SubTrajectoryBalanceLossOutput,
    TrajectoryGFNSampleBatch,
    TrainingScheduleContext,
    compute_embedding_log_heuristic,
    compute_topology_log_heuristic,
)
from src.graph_runtime import TrajectoryBatch, build_graph_batch
from src.models.gflownet_module import (
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


def _make_manual_sample_batch(
    *,
    batch: TrajectoryBatch,
    rollout_batch_size: int,
    success_mask: torch.Tensor,
    start_entropy: float,
    start_entropy_normalized: float,
) -> TrajectoryGFNSampleBatch:
    max_actions = 3
    start_nodes = torch.zeros(
        (batch.num_graphs, rollout_batch_size),
        dtype=torch.long,
        device=batch.node_ptr.device,
    )
    trace_nodes = torch.zeros(
        (batch.num_graphs, rollout_batch_size, max_actions),
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
        behavior_start_entropy=torch.full(
            (batch.num_graphs,),
            fill_value=float(start_entropy),
            dtype=torch.float32,
            device=batch.node_ptr.device,
        ),
        behavior_start_entropy_normalized=torch.full(
            (batch.num_graphs,),
            fill_value=float(start_entropy_normalized),
            dtype=torch.float32,
            device=batch.node_ptr.device,
        ),
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


def test_sampler_preserves_selected_start_flow_gradients() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)

    def _prefer_graph_moves(*args: object, **kwargs: object) -> torch.Tensor:
        distribution = kwargs.get("distribution")
        if distribution is None and len(args) >= 3:
            distribution = args[2]
        distribution = cast(Any, distribution)
        assert distribution is not None
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_submit is not None:
            logits[distribution.is_submit.to(dtype=torch.bool)] = -1.0e9
        return logits

    module.policy.compute_behavior_edge_logits = _prefer_graph_moves  # type: ignore[method-assign]

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollout_batch_size=1,
        temperature=1.0,
    )

    assert sample_batch.start_state_log_f.requires_grad
    assert sample_batch.start_state_log_f.grad_fn is not None
    assert sample_batch.start_log_probs.requires_grad
    assert sample_batch.start_log_probs.grad_fn is not None


def test_sampler_forces_submit_on_terminal_targets_before_behavior_expansion() -> None:
    module = _make_module("learned")
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0], dtype=torch.long),
        answer_entity_ids=torch.tensor([100], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="submit-before-expand",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    assert module.sampler is not None
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    call_count = 0
    original_behavior = module.policy.compute_behavior_forward_distribution
    terminal_target_mask = sampler.trajectory_supervisor.build_terminal_target_mask(
        batch=batch
    )

    def _wrapped_behavior_distribution(prepared_batch_arg, state):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        active_nodes = state.current_nodes[~state.done_mask].reshape(-1)
        if int(active_nodes.numel()) > 0:
            assert not bool(
                terminal_target_mask.index_select(0, active_nodes).any().item()
            )
        return original_behavior(prepared_batch_arg, state)

    module.policy.compute_behavior_forward_distribution = (  # type: ignore[method-assign]
        _wrapped_behavior_distribution
    )

    sample_batch = sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollout_batch_size=1,
        temperature=1.0,
    )

    assert sample_batch.trace_submit_mask is not None
    assert sample_batch.terminal_action_counts is not None
    assert call_count == 0
    assert bool(sample_batch.trace_submit_mask[0, 0, 0].item()) is True
    assert int(sample_batch.trace_edge_ids[0, 0, 0].item()) == -1
    assert int(sample_batch.terminal_action_counts[0, 0].item()) == 1
    assert int(sample_batch.terminal_num_steps[0, 0].item()) == 0
    assert int(sample_batch.terminal_nodes[0, 0].item()) == 0
    assert bool(sample_batch.success_mask[0, 0].item()) is True


def test_sampler_entity_sink_uses_deterministic_terminal_backward_log_prob() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=1,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
            answer_reward=AnswerRewardConfig(
                mode="entity_sink",
                beta=1.0,
                terminal_reward_scale="none",
                terminal_backward_mode="uniform_entity_alias",
            ),
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
        node_global_ids=torch.tensor([100, 100], dtype=torch.long),
        sample_id="entity-sink-submit",
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

    assert sample_batch.trace_submit_mask is not None
    assert sample_batch.terminal_entity_ids is not None
    assert sample_batch.terminal_backward_log_probs is not None
    assert bool(sample_batch.trace_submit_mask[0, 0, 0].item()) is True
    assert sample_batch.terminal_entity_ids[0, 0].item() == 100
    assert sample_batch.terminal_backward_log_probs[0, 0].item() == pytest.approx(0.0)
    assert sample_batch.log_pb_steps[0, 0, 0].item() == pytest.approx(0.0)


def test_sampler_keeps_success_terminal_reward_free_of_length_penalty() -> None:
    alpha = 0.3
    module = _make_module_with_training_cfg(
        "none",
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=1,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
            answer_reward=AnswerRewardConfig(
                mode="entity_sink",
                beta=1.0,
                length_penalty_alpha=alpha,
                terminal_reward_scale="none",
                terminal_backward_mode="uniform_entity_alias",
            ),
            sampling_temperature=1.0,
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="length-penalty",
    )
    prepared_batch = module.policy.prepare_batch(batch)

    def _prefer_graph_moves(*args: object, **kwargs: object) -> torch.Tensor:
        distribution = kwargs.get("distribution")
        if distribution is None and len(args) >= 3:
            distribution = args[2]
        distribution = cast(Any, distribution)
        assert distribution is not None
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_submit is not None:
            logits[distribution.is_submit.to(dtype=torch.bool)] = -1.0e9
        return logits

    module.policy.compute_behavior_edge_logits = _prefer_graph_moves  # type: ignore[method-assign]

    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollout_batch_size=1,
        temperature=1.0,
    )

    base_reward = 1.0e-3 + torch.exp(torch.tensor(1.0)).item()
    assert sample_batch.terminal_num_steps[0, 0].item() == 1
    assert sample_batch.terminal_rewards[0, 0].item() == pytest.approx(base_reward)
    assert sample_batch.terminal_log_rewards[0, 0].item() == pytest.approx(
        torch.log(torch.tensor(base_reward)).item()
    )


def test_sampler_disables_autograd_for_behavior_policy_queries() -> None:
    module = _make_module("learned")
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)

    assert module.sampler is not None
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    start_grad_enabled: list[bool] = []
    behavior_grad_enabled: list[bool] = []
    original_start = module.policy.compute_behavior_start_distribution
    original_behavior_logits = module.policy.compute_behavior_edge_logits

    def _wrapped_start(prepared_batch_arg):  # noqa: ANN001
        start_grad_enabled.append(torch.is_grad_enabled())
        return original_start(prepared_batch_arg)

    def _wrapped_behavior_logits(prepared_batch_arg, state, distribution):  # noqa: ANN001
        behavior_grad_enabled.append(torch.is_grad_enabled())
        return original_behavior_logits(prepared_batch_arg, state, distribution)

    module.policy.compute_behavior_start_distribution = _wrapped_start  # type: ignore[method-assign]
    module.policy.compute_behavior_edge_logits = _wrapped_behavior_logits  # type: ignore[method-assign]

    sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollout_batch_size=1,
        temperature=1.0,
    )

    assert start_grad_enabled and all(flag is False for flag in start_grad_enabled)
    assert behavior_grad_enabled and all(
        flag is False for flag in behavior_grad_enabled
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


def test_none_heuristic_disables_behavior_guidance() -> None:
    module = _make_module_with_training_cfg("none", beta=5.0)
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)

    start_target = module.policy.compute_start_distribution(prepared_batch)
    start_behavior = module.policy.compute_behavior_start_distribution(prepared_batch)

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
    behavior_distribution = module.policy.compute_behavior_forward_distribution(
        prepared_batch,
        state,
    )

    assert torch.allclose(start_target.log_probs, start_behavior.log_probs, atol=1.0e-6)
    assert torch.allclose(
        target_distribution.edge_logits,
        behavior_distribution.edge_logits,
        atol=1.0e-6,
    )


def test_learned_behavior_proposal_uses_cached_local_bias() -> None:
    module = _make_module("learned")
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    assert prepared_batch.heuristic_cache.step_node_log_heuristic is not None

    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )
    target_distribution = module.policy.base_policy.compute_forward_distribution(
        prepared_batch,
        state,
    )

    def _boom(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("behavior proposal should use cached local bias")

    module.policy.search_heuristic.compute_state_logits = _boom  # type: ignore[method-assign]

    behavior_logits = module.policy.compute_behavior_edge_logits(
        prepared_batch,
        state,
        target_distribution,
    )
    start_distribution = module.policy.compute_behavior_start_distribution(
        prepared_batch
    )

    assert behavior_logits.shape == target_distribution.edge_logits.shape
    assert torch.isfinite(behavior_logits).all()
    assert torch.isfinite(start_distribution.log_probs).all()


def test_gflownet_training_step_logs_core_local_flow_metrics() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        beta=0.5,
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
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
    assert "subtb_residual" in captured_metrics
    assert "subtb_residual_variance_per_batch" in captured_metrics
    assert "subtb_root" in captured_metrics
    assert "unique_success_paths_per_100_rollouts" in captured_metrics
    assert "start_node_entropy" in captured_metrics
    assert "active_forward_states" in captured_metrics
    assert "unique_forward_states" in captured_metrics
    assert "raw_graph_candidates" in captured_metrics
    assert "scored_graph_candidates" in captured_metrics
    assert "candidate_shortlist_keep_ratio" in captured_metrics
    assert not any(str(key).startswith("exact_aux") for key in captured_metrics)


def test_training_rollout_metrics_report_forward_search_observability() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    sample_batch = replace(
        _make_manual_sample_batch(
            batch=batch,
            rollout_batch_size=2,
            success_mask=torch.tensor([[True, False]], dtype=torch.bool),
            start_entropy=0.3,
            start_entropy_normalized=0.6,
        ),
        total_active_agent_count=10,
        total_unique_active_state_count=4,
        total_raw_graph_candidate_count=18,
        total_scored_graph_candidate_count=7,
        total_shortlist_active_state_count=3,
    )

    metrics = module._compute_training_rollout_metrics(
        batch=batch, sample_batch=sample_batch
    )

    assert metrics.active_forward_states == pytest.approx(10.0)
    assert metrics.unique_forward_states == pytest.approx(4.0)
    assert metrics.forward_state_dedup_keep_ratio == pytest.approx(0.4)
    assert metrics.raw_graph_candidates == pytest.approx(18.0)
    assert metrics.scored_graph_candidates == pytest.approx(7.0)
    assert metrics.raw_graph_candidates_per_unique_state == pytest.approx(4.5)
    assert metrics.scored_graph_candidates_per_unique_state == pytest.approx(1.75)
    assert metrics.shortlist_active_states == pytest.approx(3.0)
    assert metrics.candidate_shortlist_activation_rate == pytest.approx(0.75)
    assert metrics.candidate_shortlist_keep_ratio == pytest.approx(7.0 / 18.0)


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
        rollout_batch_size=2,
        temperature=1.0,
    )

    assert sample_batch.trace_submit_mask is not None
    assert torch.isfinite(sample_batch.log_pf_steps).all()
    assert torch.equal(
        sample_batch.log_pb_steps[~sample_batch.trace_submit_mask],
        torch.zeros_like(sample_batch.log_pb_steps[~sample_batch.trace_submit_mask]),
    )


def test_guidance_loss_uses_learned_behavior_head_targets() -> None:
    module = _make_module_with_training_cfg(
        "learned",
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=2,
            answer_reward=AnswerRewardConfig(mode="binary_ranking", beta=1.0),
            guidance=GuidanceLossConfig(loss_weight=0.2, detach_features=True),
        ),
    )
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    assert module.sampler is not None
    sample_batch = module.sampler.sample(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollout_batch_size=2,
        temperature=1.0,
    )

    guidance_result = module._compute_guidance_loss(
        prepared_batch=prepared_batch,
        sample_batch=sample_batch,
    )

    assert guidance_result is not None
    assert torch.isfinite(guidance_result.loss)
    assert guidance_result.active_states > 0
    assert 0.0 <= guidance_result.prediction_mean.item() <= 1.0
    assert 0.0 <= guidance_result.target_mean.item() <= 1.0


def test_training_step_logs_guidance_metrics_when_enabled() -> None:
    torch.manual_seed(5)
    module = _make_module_with_training_cfg(
        "learned",
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=2,
            answer_reward=AnswerRewardConfig(mode="binary_ranking", beta=1.0),
            guidance=GuidanceLossConfig(loss_weight=0.1, detach_features=True),
        ),
    )
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="train-rank-aux",
    )
    captured_metrics: dict[str, object] = {}

    def _capture_metric_bundle(*, metrics: dict[str, object], **kwargs: object) -> None:
        del kwargs
        captured_metrics.update(metrics)

    module._log_metric_bundle = _capture_metric_bundle  # type: ignore[method-assign]

    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert "guidance_loss" in captured_metrics
    assert "actor_loss" in captured_metrics
    assert "rank_aux_loss" not in captured_metrics


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


def test_success_replay_rollout_resolution_caps_dynamic_budget() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=64,
            success_replay=SuccessfulTrajectoryReplayConfig(
                enabled=True,
                ratio=0.25,
                warmup_passes=0.0,
                min_buffer_size=1,
                max_buffer_size=32,
                max_trajectories_per_sample=8,
                max_rollouts_per_graph=8,
            ),
        ),
    )

    assert module._resolve_success_replay_rollouts_per_graph() == 8


def test_training_rollout_metrics_track_new_success_paths_once_per_sample() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=1,
            adaptive_sampling=AdaptiveSamplingConfig(
                enabled=True,
                min_rollout_batch_size=1,
                max_rollout_batch_size=4,
                warmup_steps=0,
            ),
        ),
    )
    batch = make_toy_batch()
    sample_batch = replace(
        _make_manual_sample_batch(
            batch=batch,
            rollout_batch_size=1,
            success_mask=torch.ones((1, 1), dtype=torch.bool),
            start_entropy=0.0,
            start_entropy_normalized=0.0,
        ),
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        terminal_nodes=torch.tensor([[2]], dtype=torch.long),
        terminal_num_steps=torch.tensor([[1]], dtype=torch.long),
        trace_edge_ids=torch.tensor([[[1, -1, -1]]], dtype=torch.long),
    )

    first_metrics = module._compute_training_rollout_metrics(
        batch=batch,
        sample_batch=sample_batch,
    )
    second_metrics = module._compute_training_rollout_metrics(
        batch=batch,
        sample_batch=sample_batch,
    )

    assert first_metrics.new_success_paths == 1
    assert first_metrics.unique_success_paths_per_100_rollouts == pytest.approx(100.0)
    assert second_metrics.new_success_paths == 0
    assert second_metrics.unique_success_paths_per_100_rollouts == pytest.approx(0.0)


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


def test_adaptive_sampling_controller_increases_budget_after_sparse_rollouts() -> None:
    module = _make_module_with_training_cfg(
        "topology",
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=2,
            sampling_temperature=1.0,
            adaptive_sampling=AdaptiveSamplingConfig(
                enabled=True,
                min_rollout_batch_size=2,
                max_rollout_batch_size=6,
                warmup_steps=0,
                rollout_growth_factor=2.0,
                rollout_shrink_factor=0.5,
                temperature_multiplier_up=1.5,
                low_success_rate_threshold=0.2,
                high_success_rate_threshold=0.8,
                low_unique_success_paths_per_100_rollouts=1.0,
                high_unique_success_paths_per_100_rollouts=20.0,
                low_start_entropy_normalized=0.3,
                high_start_entropy_normalized=0.9,
                low_subtb_residual_variance=0.1,
                high_subtb_residual_variance=0.3,
            ),
        ),
    )
    batch = make_toy_batch()
    captured_rollout_calls: list[tuple[int, float]] = []

    def _fake_sample(
        *,
        batch: TrajectoryBatch,
        policy: object,
        prepared_batch: object,
        rollout_batch_size: int,
        temperature: float,
    ) -> TrajectoryGFNSampleBatch:
        del policy, prepared_batch
        captured_rollout_calls.append((rollout_batch_size, temperature))
        return _make_manual_sample_batch(
            batch=batch,
            rollout_batch_size=rollout_batch_size,
            success_mask=torch.zeros(
                (batch.num_graphs, rollout_batch_size),
                dtype=torch.bool,
                device=batch.node_ptr.device,
            ),
            start_entropy=0.1,
            start_entropy_normalized=0.1,
        )

    module.sampler.sample = _fake_sample  # type: ignore[method-assign]
    module.loss_fn.compute = lambda *args, **kwargs: SubTrajectoryBalanceLossOutput(  # type: ignore[method-assign]
        loss=torch.tensor(1.0),
        subtb_loss=torch.tensor(1.0),
        residual_abs=torch.tensor(0.5),
        residual_variance=torch.tensor(0.8),
        root_abs=torch.tensor(0.5),
        success_rate=torch.tensor(0.0),
        log_z_mean=torch.tensor(0.0),
        log_z_variance=torch.tensor(0.0),
    )

    module.training_step(batch, batch_idx=0)
    module.training_step(batch, batch_idx=1)

    assert captured_rollout_calls[0][0] == 2
    assert captured_rollout_calls[1][0] == 4
    assert captured_rollout_calls[1][1] == pytest.approx(1.5)


def test_sampler_emits_deterministic_backward_log_probs_for_path_state() -> None:
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

    assert sample_batch.log_pb_steps[0, 0, 0].item() == pytest.approx(0.0)


def test_forward_distribution_is_decoupled_from_state_flow_head() -> None:
    module = _make_module("topology")
    batch = make_toy_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )

    module.policy.base_policy.forward_policy_head.forward = (  # type: ignore[method-assign]
        lambda current_state_features,
        candidate_state_features,
        relation_features: torch.zeros(
            (int(current_state_features.size(0)),),
            device=current_state_features.device,
            dtype=torch.float32,
        )
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
    submit_mask = distribution.is_submit
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
        distribution.edge_logits[graph_mask].to(dtype=torch.float32), torch.zeros(2)
    )
    assert not torch.allclose(
        distribution.edge_logits[graph_mask].to(dtype=torch.float32),
        expected_child_log_flows,
    )


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
    original_forward_chunk = gflownet_policy_impl._FORWARD_EDGE_CHUNK_SIZE
    original_transition_chunk = gflownet_policy_impl._TRANSITION_LOGIT_CHUNK_SIZE
    gflownet_policy_impl._FORWARD_EDGE_CHUNK_SIZE = 1
    gflownet_policy_impl._TRANSITION_LOGIT_CHUNK_SIZE = 1
    try:
        chunked = module.policy.compute_forward_distribution(prepared_batch, state)
    finally:
        gflownet_policy_impl._FORWARD_EDGE_CHUNK_SIZE = original_forward_chunk
        gflownet_policy_impl._TRANSITION_LOGIT_CHUNK_SIZE = original_transition_chunk

    assert torch.equal(chunked.edge_agent_batch, baseline.edge_agent_batch)
    assert torch.equal(chunked.edge_ids, baseline.edge_ids)
    assert torch.equal(chunked.target_nodes, baseline.target_nodes)
    assert torch.equal(chunked.out_degrees, baseline.out_degrees)
    assert torch.equal(chunked.is_submit, baseline.is_submit)
    assert torch.allclose(
        chunked.edge_logits.to(dtype=torch.float32),
        baseline.edge_logits.to(dtype=torch.float32),
        atol=1.0e-6,
    )


def test_forward_distribution_shortlists_high_degree_candidates() -> None:
    policy_cfg = PolicyConfig(
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
        candidate_shortlist=CandidateShortlistConfig(
            enabled=True,
            topk=1,
            degree_threshold=1,
            heuristic_weight=0.0,
        ),
    )
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
            sampling_temperature=1.0,
        ),
        heuristic_cfg=HeuristicConfig(kind="topology", beta=0.0),
        policy_cfg=policy_cfg,
        eval_cfg=SearchEvalConfig(metrics_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
    )
    prepared_batch = module.policy.prepare_batch(batch)
    prepared_batch = replace(
        prepared_batch,
        question_tokens=torch.tensor(
            [[1.0] + [0.0] * 7],
            dtype=prepared_batch.question_tokens.dtype,
            device=prepared_batch.question_tokens.device,
        ),
    )
    state = SearchState.initialize(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        max_steps=2,
    )

    module.policy.base_policy._build_flat_state_features = (  # type: ignore[method-assign]
        lambda prepared_batch,
        flat_nodes,
        flat_num_steps,
        flat_done_mask,
        flat_control_states: torch.zeros(
            (int(flat_nodes.numel()), 8),
            device=flat_nodes.device,
            dtype=prepared_batch.node_tokens.dtype,
        )
    )

    def _mock_local_features(
        prepared_batch,
        *,
        flat_nodes,
        flat_num_steps,
        flat_done_mask,
    ):
        del prepared_batch, flat_num_steps, flat_done_mask
        features = torch.zeros(
            (int(flat_nodes.numel()), 8),
            device=flat_nodes.device,
            dtype=torch.float32,
        )
        features[:, 0] = torch.tensor(
            [
                {1: 1.0, 2: 4.0, 3: 2.0}.get(int(node.item()), 0.0)
                for node in flat_nodes
            ],
            device=flat_nodes.device,
            dtype=torch.float32,
        )
        return features

    module.policy.base_policy.build_local_state_features = _mock_local_features  # type: ignore[method-assign]
    module.policy.base_policy.forward_policy_head.forward = (  # type: ignore[method-assign]
        lambda current_state_features,
        candidate_state_features,
        relation_features: torch.zeros(
            (int(current_state_features.size(0)),),
            device=current_state_features.device,
            dtype=torch.float32,
        )
    )

    distribution = module.policy.compute_forward_distribution(prepared_batch, state)
    assert distribution.is_submit is not None
    graph_mask = ~distribution.is_submit.to(dtype=torch.bool)

    assert torch.equal(distribution.edge_ids[graph_mask], torch.tensor([1]))
    assert torch.equal(distribution.target_nodes[graph_mask], torch.tensor([2]))
    assert torch.equal(distribution.out_degrees.view(-1), torch.tensor([2]))


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
