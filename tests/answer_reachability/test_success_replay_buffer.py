from __future__ import annotations

from dataclasses import replace
import math
from typing import cast

import pytest
import torch

from src.graph import TrajectoryBatch
from src.models.configs import (
    ActionPriorConfig,
    GFlowNetTrainingConfig,
    HorizonConfig,
    OptimizerConfig,
    SchedulerConfig,
    SearchEvalConfig,
    SuccessReplayConfig,
)
from src.models.gflownet import (
    ForwardTrajectoryGFNSampler,
    PreparedGFlowNetBatch,
    SuccessReplayBuffer,
    TrajectoryGFNSampleBatch,
)
from src.models.gflownet_module import GFlowNetModule
from src.metrics.runtime_factory import GraphTaskRuntimeFactory

from .conftest import make_batch_from_graph, make_policy_config


def _make_replay_module() -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            force_stop_on_answer_hit=True,
            step_log_penalty=float(math.log(0.5)),
        ),
        action_prior_cfg=ActionPriorConfig(
            node_topology_weight=0.0,
            node_embedding_weight=0.0,
        ),
        policy_cfg=make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )


def _make_success_batch() -> TrajectoryBatch:
    return make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="success-replay-roundtrip",
    )


def _make_two_hop_batch() -> TrajectoryBatch:
    return make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="success-replay-guidance",
    )


def _force_graph_moves(module: GFlowNetModule) -> None:
    original_forward = module.policy.compute_forward_distribution

    def _prefer_graph_moves(prepared_batch, state, **kwargs):  # noqa: ANN001
        distribution = original_forward(prepared_batch, state, **kwargs)
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_stop_action is not None:
            logits[distribution.is_stop_action.to(dtype=torch.bool)] = -1.0e9
        return replace(distribution, edge_logits=logits)

    module.policy.compute_forward_distribution = _prefer_graph_moves  # type: ignore[method-assign]


def _sample_success_rollout(
    module: GFlowNetModule,
) -> tuple[TrajectoryBatch, PreparedGFlowNetBatch, TrajectoryGFNSampleBatch]:
    batch = _make_success_batch()
    _force_graph_moves(module)
    prepared_batch = module.policy.prepare_batch(batch)
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    sample_batch = sampler.sample(
        batch=batch.without_raw_features(),
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )
    return batch, prepared_batch, sample_batch


def test_success_replay_buffer_deduplicates_identical_successes() -> None:
    module = _make_replay_module()
    batch, _, sample_batch = _sample_success_rollout(module)
    buffer = SuccessReplayBuffer(
        config=SuccessReplayConfig(
            mix_alpha=0.5,
            min_buffer_size=1,
            capacity=4,
            replay_trajectories_per_step=1,
            deduplicate=True,
        )
    )

    first_added = buffer.add_successes(batch=batch, sample_batch=sample_batch)
    second_added = buffer.add_successes(batch=batch, sample_batch=sample_batch)

    assert first_added == 1
    assert second_added == 0
    assert len(buffer) == 1
    assert buffer.ready is True


def test_success_replay_buffer_round_trips_teacher_forced_success() -> None:
    module = _make_replay_module()
    batch, _, sample_batch = _sample_success_rollout(module)
    buffer = SuccessReplayBuffer(
        config=SuccessReplayConfig(
            mix_alpha=0.5,
            min_buffer_size=1,
            capacity=4,
            replay_trajectories_per_step=1,
        )
    )
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)

    added = buffer.add_successes(batch=batch, sample_batch=sample_batch)
    replay_batch = buffer.sample_replay_batch(
        device=torch.device("cpu"),
        replay_trajectories_per_step=1,
    )

    assert added == 1
    assert replay_batch is not None

    replay_prepared_batch = module.policy.prepare_batch(replay_batch.batch)
    rebuilt_batch = sampler.rebuild_sample_batch(
        batch=replay_batch.batch,
        policy=module.policy,
        prepared_batch=replay_prepared_batch,
        start_nodes=replay_batch.start_nodes,
        planned_edge_ids=replay_batch.planned_edge_ids,
        planned_stop_mask=replay_batch.planned_stop_mask,
        path_lengths=replay_batch.path_lengths,
        termination_action_steps=replay_batch.termination_action_steps,
        trace_nodes=replay_batch.trace_nodes,
        trace_edge_ids=replay_batch.trace_edge_ids,
        trace_num_steps=replay_batch.trace_num_steps,
        trace_mask=replay_batch.trace_mask,
        trace_stop_mask=replay_batch.trace_stop_mask,
    )

    assert bool(sample_batch.success_mask[0, 0].item()) is True
    assert bool(rebuilt_batch.success_mask[0, 0].item()) is True
    assert rebuilt_batch.log_reward_steps is not None
    assert torch.equal(rebuilt_batch.move_mask, sample_batch.move_mask[:, :1])
    assert torch.equal(rebuilt_batch.trace_stop_mask, replay_batch.trace_stop_mask)
    assert rebuilt_batch.termination_action_steps is not None
    assert rebuilt_batch.termination_action_steps[0, 0].item() == pytest.approx(2)
    step_penalty = float(math.log(0.5))
    assert rebuilt_batch.log_reward_steps[
        rebuilt_batch.move_mask
    ].item() == pytest.approx(step_penalty)


def test_success_replay_buffer_reuses_single_graph_payload_for_duplicate_records() -> (
    None
):
    module = _make_replay_module()
    batch, _, sample_batch = _sample_success_rollout(module)
    buffer = SuccessReplayBuffer(
        config=SuccessReplayConfig(
            mix_alpha=0.5,
            min_buffer_size=1,
            capacity=4,
            replay_trajectories_per_step=1,
            deduplicate=False,
        )
    )

    first_added = buffer.add_successes(batch=batch, sample_batch=sample_batch)
    second_added = buffer.add_successes(batch=batch, sample_batch=sample_batch)

    assert first_added == 1
    assert second_added == 1
    assert len(buffer) == 2
    assert len(buffer._graph_payloads) == 1


def test_success_replay_roundtrip_preserves_answer_stop_bonus() -> None:
    bonus = 0.7
    module = GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            force_stop_on_answer_hit=True,
            answer_stop_log_reward_bonus=bonus,
        ),
        action_prior_cfg=ActionPriorConfig(
            node_topology_weight=0.0,
            node_embedding_weight=0.0,
        ),
        policy_cfg=make_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )
    batch = make_batch_from_graph(
        num_nodes=1,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0], dtype=torch.long),
        answer_entity_ids=torch.tensor([100], dtype=torch.long),
        node_entity_ids=torch.tensor([100], dtype=torch.long),
        sample_id="replay-answer-stop-bonus",
    )
    prepared_batch = module.policy.prepare_batch(batch)
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    sample_batch = sampler.sample(
        batch=batch.without_raw_features(),
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )
    buffer = SuccessReplayBuffer(
        config=SuccessReplayConfig(
            mix_alpha=0.5,
            min_buffer_size=1,
            capacity=4,
            replay_trajectories_per_step=1,
        )
    )

    added = buffer.add_successes(batch=batch, sample_batch=sample_batch)
    replay_batch = buffer.sample_replay_batch(
        device=torch.device("cpu"),
        replay_trajectories_per_step=1,
    )

    assert added == 1
    assert replay_batch is not None
    replay_prepared_batch = module.policy.prepare_batch(replay_batch.batch)
    rebuilt_batch = sampler.rebuild_sample_batch(
        batch=replay_batch.batch,
        policy=module.policy,
        prepared_batch=replay_prepared_batch,
        start_nodes=replay_batch.start_nodes,
        planned_edge_ids=replay_batch.planned_edge_ids,
        planned_stop_mask=replay_batch.planned_stop_mask,
        path_lengths=replay_batch.path_lengths,
        termination_action_steps=replay_batch.termination_action_steps,
        trace_nodes=replay_batch.trace_nodes,
        trace_edge_ids=replay_batch.trace_edge_ids,
        trace_num_steps=replay_batch.trace_num_steps,
        trace_mask=replay_batch.trace_mask,
        trace_stop_mask=replay_batch.trace_stop_mask,
    )

    assert sample_batch.log_reward_steps is not None
    assert rebuilt_batch.log_reward_steps is not None
    assert sample_batch.log_reward_steps[0, 0, 0].item() == pytest.approx(bonus)
    assert rebuilt_batch.log_reward_steps[0, 0, 0].item() == pytest.approx(bonus)


def test_success_replay_can_seed_shortest_path_guidance_without_online_success() -> (
    None
):
    module = _make_replay_module()
    batch = _make_two_hop_batch()
    prepared_batch = module.policy.prepare_batch(batch)
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    sample_batch = sampler.sample(
        batch=batch.without_raw_features(),
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )
    failed_sample_batch = replace(
        sample_batch,
        success_mask=torch.zeros_like(sample_batch.success_mask, dtype=torch.bool),
    )
    buffer = SuccessReplayBuffer(
        config=SuccessReplayConfig(
            mix_alpha=0.5,
            min_buffer_size=1,
            capacity=4,
            replay_trajectories_per_step=1,
            add_shortest_path_guidance=True,
        )
    )

    added = buffer.add_successes(batch=batch, sample_batch=failed_sample_batch)
    replay_batch = buffer.sample_replay_batch(
        device=torch.device("cpu"),
        replay_trajectories_per_step=1,
    )

    assert added == 1
    assert replay_batch is not None
    assert replay_batch.start_nodes[0, 0].item() == 0
    assert replay_batch.path_lengths[0, 0].item() == 2
    assert replay_batch.termination_action_steps[0, 0].item() == 3
    assert replay_batch.planned_edge_ids[0, 0, :2].tolist() == [0, 1]
    assert replay_batch.trace_stop_mask[0, 0, 2].item() is True
