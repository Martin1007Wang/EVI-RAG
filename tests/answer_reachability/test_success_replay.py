from __future__ import annotations

import pytest
import torch
from typing import cast

from src.graph_runtime import TrajectoryBatch
from src.metrics.answer_reachability.runtime import SearchMetricRuntimeFactory
from src.models.configs import (
    BackboneConfig,
    CandidateShortlistConfig,
    GFlowNetTrainingConfig,
    HeuristicConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SearchEvalConfig,
    StateScoreHeadConfig,
    SuccessfulTrajectoryReplayConfig,
)
from src.models.configs.training import SchedulerConfig
from src.models.gflownet import TrajectoryGFNSampleBatch
from src.models.gflownet import SearchState
from src.models.gflownet import ForwardTrajectoryGFNSampler
from src.models.gflownet.replay import (
    SuccessfulTrajectoryRecord,
    SuccessfulTrajectoryReplayBuffer,
    build_replay_sample_batch,
)
from src.models.gflownet_module import GFlowNetModule

from .conftest import make_batch_from_graph, make_toy_batch


def _make_policy_config(
    *, candidate_shortlist: CandidateShortlistConfig | None = None
) -> PolicyConfig:
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
        candidate_shortlist=candidate_shortlist or CandidateShortlistConfig(),
    )


def _make_module() -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
            sampling_temperature=1.0,
            success_replay=SuccessfulTrajectoryReplayConfig(
                enabled=True,
                ratio=0.25,
                warmup_passes=0.0,
                min_buffer_size=1,
                max_buffer_size=16,
                max_trajectories_per_sample=4,
            ),
        ),
        heuristic_cfg=HeuristicConfig(kind="topology", beta=0.5),
        policy_cfg=_make_policy_config(),
        eval_cfg=SearchEvalConfig(metrics_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )


def _make_success_sample_batch(batch: TrajectoryBatch) -> TrajectoryGFNSampleBatch:
    return TrajectoryGFNSampleBatch(
        graph_log_z=torch.zeros((1,), dtype=torch.float32),
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        start_log_probs=torch.zeros((1, 1), dtype=torch.float32),
        start_state_log_f=torch.zeros((1, 1), dtype=torch.float32),
        log_pf_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        move_mask=torch.tensor([[[True, False]]], dtype=torch.bool),
        trace_nodes=torch.tensor([[[0, 2]]], dtype=torch.long),
        trace_edge_ids=torch.tensor([[[1, -1]]], dtype=torch.long),
        trace_num_steps=torch.tensor([[[0, 1]]], dtype=torch.long),
        trace_mask=torch.tensor([[[True, False]]], dtype=torch.bool),
        terminal_nodes=torch.tensor([[2]], dtype=torch.long),
        terminal_num_steps=torch.tensor([[1]], dtype=torch.long),
        terminal_state_log_f=torch.zeros((1, 1), dtype=torch.float32),
        terminal_rewards=torch.ones((1, 1), dtype=torch.float32),
        terminal_log_rewards=torch.zeros((1, 1), dtype=torch.float32),
        success_mask=torch.ones((1, 1), dtype=torch.bool),
    )


def test_trajectory_batch_concatenate_preserves_graph_boundaries() -> None:
    first = make_toy_batch()
    second = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([201], dtype=torch.long),
        node_global_ids=torch.tensor([200, 201], dtype=torch.long),
        sample_id="second-sample",
    )

    combined = TrajectoryBatch.concatenate([first, second])

    assert combined.num_graphs == 2
    assert combined.sample_ids == ["toy-sample", "second-sample"]
    assert combined.select_graph(0).sample_ids == ["toy-sample"]
    assert combined.select_graph(1).sample_ids == ["second-sample"]


def test_successful_trajectory_replay_buffer_extracts_local_paths() -> None:
    batch = make_toy_batch()
    sample_batch = _make_success_sample_batch(batch)
    buffer = SuccessfulTrajectoryReplayBuffer(
        max_buffer_size=8,
        max_trajectories_per_sample=4,
    )

    added = buffer.add_successes(batch=batch, sample_batch=sample_batch)
    plan = buffer.plan_for_batch(batch=batch, replay_rollouts_per_graph=1)

    assert added == 1
    assert plan is not None
    assert plan.graph_indices == (0,)
    assert plan.records_by_graph[0][0] == SuccessfulTrajectoryRecord(
        sample_id="toy-sample",
        start_local_node=0,
        local_edge_ids=(1,),
    )


def test_build_replay_sample_batch_replays_successful_path() -> None:
    batch = make_toy_batch()
    module = _make_module()
    assert module.sampler is not None
    prepared_batch = module.policy.prepare_batch(batch)

    replay_sample_batch = build_replay_sample_batch(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        trajectory_supervisor=cast(
            ForwardTrajectoryGFNSampler, module.sampler
        ).trajectory_supervisor,
        replay_records=(
            (
                SuccessfulTrajectoryRecord(
                    sample_id="toy-sample",
                    start_local_node=0,
                    local_edge_ids=(1,),
                ),
            ),
        ),
        max_steps=2,
    )
    loss = module.loss_fn.compute(replay_sample_batch)

    assert bool(replay_sample_batch.success_mask.item()) is True
    assert int(replay_sample_batch.terminal_num_steps.item()) == 1
    assert int(replay_sample_batch.terminal_nodes.item()) == 2
    assert int(replay_sample_batch.trace_edge_ids[0, 0, 0].item()) == 1
    assert torch.isfinite(loss.loss)
    assert loss.success_rate == pytest.approx(1.0)


def test_build_replay_sample_batch_skips_move_backward_reconstruction() -> None:
    batch = make_toy_batch()
    module = _make_module()
    assert module.sampler is not None
    prepared_batch = module.policy.prepare_batch(batch)

    def _unexpected_backward_distribution(
        *args: object, **kwargs: object
    ) -> torch.Tensor:
        del args, kwargs
        raise AssertionError(
            "replay builder should not reconstruct move backward logits"
        )

    module.policy.compute_backward_distribution = _unexpected_backward_distribution  # type: ignore[method-assign]

    replay_sample_batch = build_replay_sample_batch(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        trajectory_supervisor=cast(
            ForwardTrajectoryGFNSampler, module.sampler
        ).trajectory_supervisor,
        replay_records=((SuccessfulTrajectoryRecord("toy-sample", 0, (1,)),),),
        max_steps=2,
    )

    assert replay_sample_batch.trace_stop_mask is not None
    assert torch.isfinite(replay_sample_batch.log_pf_steps).all()
    assert torch.equal(
        replay_sample_batch.log_pb_steps[~replay_sample_batch.trace_stop_mask],
        torch.zeros_like(
            replay_sample_batch.log_pb_steps[~replay_sample_batch.trace_stop_mask]
        ),
    )


def test_build_replay_sample_batch_force_keeps_recorded_edges_in_shortlist() -> None:
    batch = make_toy_batch()
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
            sampling_temperature=1.0,
            success_replay=SuccessfulTrajectoryReplayConfig(
                enabled=True,
                ratio=0.25,
                warmup_passes=0.0,
                min_buffer_size=1,
                max_buffer_size=16,
                max_trajectories_per_sample=4,
            ),
        ),
        heuristic_cfg=HeuristicConfig(kind="none", beta=0.0),
        policy_cfg=_make_policy_config(
            candidate_shortlist=CandidateShortlistConfig(
                enabled=True,
                topk=1,
                degree_threshold=1,
                heuristic_weight=0.0,
            )
        ),
        eval_cfg=SearchEvalConfig(metrics_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )
    prepared_batch = module.policy.prepare_batch(batch)
    assert module.sampler is not None

    def _force_shortlist_away_from_answer(
        *args: object, **kwargs: object
    ) -> torch.Tensor:
        del args, kwargs
        return torch.tensor([10.0, -10.0], dtype=torch.float32)

    module.policy.base_policy._compute_shortlist_scores = (  # type: ignore[method-assign]
        _force_shortlist_away_from_answer
    )

    state = SearchState.initialize(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        max_steps=module.policy.base_policy.max_steps,
    )
    shortlisted_distribution = module.policy.compute_forward_distribution(
        prepared_batch,
        state,
    )
    assert 1 not in shortlisted_distribution.edge_ids.tolist()

    forced_distribution = module.policy.compute_forward_distribution(
        prepared_batch,
        state,
        required_edge_ids=torch.tensor([1], dtype=torch.long),
    )
    assert forced_distribution.is_stop_action is not None
    forced_graph_edges = forced_distribution.edge_ids[
        ~forced_distribution.is_stop_action.to(dtype=torch.bool)
    ]
    assert torch.equal(torch.sort(forced_graph_edges).values, torch.tensor([0, 1]))
    assert torch.equal(forced_distribution.out_degrees.view(-1), torch.tensor([3]))

    replay_sample_batch = build_replay_sample_batch(
        batch=batch,
        policy=module.policy,
        prepared_batch=prepared_batch,
        trajectory_supervisor=cast(
            ForwardTrajectoryGFNSampler, module.sampler
        ).trajectory_supervisor,
        replay_records=((SuccessfulTrajectoryRecord("toy-sample", 0, (1,)),),),
        max_steps=2,
    )

    assert int(replay_sample_batch.trace_edge_ids[0, 0, 0].item()) == 1
    assert bool(replay_sample_batch.success_mask.item()) is True


def test_forward_distribution_rejects_missing_required_replay_edge() -> None:
    batch = make_toy_batch()
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollout_batch_size=3,
            reward_epsilon=1.0e-3,
            failure_reward_mode="graph_normalized",
            sampling_temperature=1.0,
        ),
        heuristic_cfg=HeuristicConfig(kind="none", beta=0.0),
        policy_cfg=_make_policy_config(
            candidate_shortlist=CandidateShortlistConfig(
                enabled=True,
                topk=1,
                degree_threshold=1,
                heuristic_weight=0.0,
            )
        ),
        eval_cfg=SearchEvalConfig(metrics_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=SearchMetricRuntimeFactory(),
    )
    prepared_batch = module.policy.prepare_batch(batch)
    state = SearchState.initialize(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_nodes=torch.tensor([[1]], dtype=torch.long),
        max_steps=module.policy.base_policy.max_steps,
    )

    with pytest.raises(ValueError, match="required replay edge"):
        module.policy.compute_forward_distribution(
            prepared_batch,
            state,
            required_edge_ids=torch.tensor([0], dtype=torch.long),
        )
