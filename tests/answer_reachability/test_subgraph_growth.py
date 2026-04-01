from __future__ import annotations

import math

import pytest
import torch

from src.graph import TrajectoryBatch
from src.metrics.runtime_factory import GraphTaskRuntimeFactory
from src.metrics.subgraph_answer_search_runtime import SubgraphAnswerSearchRuntime
from src.models.gflownet.actor import (
    HierarchicalStateActionDistribution,
    SubgraphActionDistribution,
)
from src.models.gflownet import SubgraphSuccessReplayBuffer
from src.models.gflownet.losses import SubgraphSubTrajectoryBalanceLoss
from src.models.gflownet.policy import SUBGRAPH_STATE_MODE, SubgraphPolicy
from src.models.gflownet.prepared_batch import SubgraphPreparedBatch
from src.models.gflownet.sampler import SubgraphSampler, SubgraphTrajectorySampleBatch
from src.models.gflownet.state import (
    SubgraphAction,
    SubgraphState,
    forward_valid_removable_edge_ids,
)
from src.models.gflownet_module import GFlowNetModule

from .conftest import make_batch_from_graph


def _make_subgraph_policy_config() -> dict[str, object]:
    return {
        "state_mode": SUBGRAPH_STATE_MODE,
        "backbone": {
            "embedding_dim": 8,
            "hidden_dim": 8,
            "use_adapter": True,
            "adapter_dim": 4,
            "adapter_dropout": 0.0,
        },
        "flow_head": {
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "conditioning": "concat",
        },
        "state_encoder": {
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
        },
        "actor": {
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
        },
    }


def _make_training_cfg(**overrides: object) -> dict[str, object]:
    training_cfg: dict[str, object] = {
        "rollouts_per_graph": 2,
        "sampling_temperature": 1.0,
        "sampling_temperature_schedule": {"type": "constant", "hold_steps": 0},
        "action_pruning": {"per_node_top_k": 0, "per_state_top_k": 0},
        "answer_reward": {
            "gold_answer_bonus": 2.0,
            "wrong_answer_penalty": 2.0,
            "failure_penalty": 4.0,
            "size_penalty": 0.1,
            "redundancy_penalty": 0.25,
            "component_penalty": 0.5,
        },
        "subtb": {
            "lambda_weight": 1.0,
            "topology_weight_alpha": 0.0,
        },
        "auxiliary": {
            "proposal": {
                "enabled": False,
                "prior": {
                    "oracle_answer_distance_weight": 0.0,
                    "prior_question_similarity_weight": 0.0,
                    "prior_component_merge_weight": 0.0,
                    "stop_hit_bias": 0.0,
                },
                "schedule": {"type": "constant", "hold_steps": 0},
            },
            "replay": {
                "enabled": False,
                "mix_alpha": 0.0,
                "buffer": {
                    "capacity": 1024,
                    "min_buffer_size": 64,
                    "replay_trajectories_per_step": None,
                    "deduplicate": True,
                },
                "guidance": {
                    "add_shortest_path_guidance": False,
                    "expand_imitation_weight": 0.0,
                    "expand_imitation_from_anchor_bonus": 0.0,
                    "expand_imitation_answer_finish_bonus": 0.0,
                    "mask_stop_loss": True,
                },
                "schedule": {"type": "constant", "hold_steps": 0},
            },
        },
    }
    training_cfg.update(overrides)
    return training_cfg


def _make_eval_cfg(**overrides: object) -> dict[str, object]:
    eval_cfg: dict[str, object] = {
        "report_profile": "full",
        "task": "answer_ranking",
        "support_mass_threshold": 0.9,
        "support_path_overlap_penalty": 0.25,
        "answer_top_ks": (1, 5, 10),
        "edge_top_ks": (1, 5, 10, 25, 50),
        "edge_emit_top_k": 25,
        "monte_carlo": {
            "rollouts": 32,
            "batch_rollouts": 16,
            "temperature": 1.0,
            "confidence": 0.95,
            "early_stop": {
                "enabled": True,
                "min_rollouts": 512,
                "stability_top_k": 1,
            },
            "action_pruning": {"per_node_top_k": 100, "per_state_top_k": 256},
        },
    }
    eval_cfg.update(overrides)
    return eval_cfg


def _make_bridge_batch(*, sample_id: str = "bridge-subgraph") -> TrajectoryBatch:
    return make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        anchor_local_indices=torch.tensor([0, 2], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id=sample_id,
    )


def _make_multi_answer_bridge_batch() -> TrajectoryBatch:
    return make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 2, 0, 2], [1, 1, 3, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        anchor_local_indices=torch.tensor([0, 2], dtype=torch.long),
        a_local_indices=torch.tensor([1, 3], dtype=torch.long),
        answer_entity_ids=torch.tensor([101, 103], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="multi-answer-bridge",
    )


def _make_two_hop_multi_anchor_batch() -> TrajectoryBatch:
    return make_batch_from_graph(
        num_nodes=5,
        edge_index=torch.tensor([[0, 3, 2, 4], [3, 1, 4, 1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        anchor_local_indices=torch.tensor([0, 2], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 103, 104], dtype=torch.long),
        sample_id="two-hop-multi-anchor",
    )


def _make_subgraph_policy(*, max_steps: int = 2) -> SubgraphPolicy:
    policy_cfg = _make_subgraph_policy_config()
    training_cfg = _make_training_cfg()
    return SubgraphPolicy(
        state_mode=str(policy_cfg["state_mode"]),
        backbone=dict(policy_cfg["backbone"]),
        flow_head=dict(policy_cfg["flow_head"]),
        state_encoder=dict(policy_cfg["state_encoder"]),
        actor=dict(policy_cfg["actor"]),
        answer_reward=dict(training_cfg["answer_reward"]),
        proposal_prior=dict(training_cfg["auxiliary"]["proposal"]["prior"]),
        max_steps=max_steps,
    )


def _build_state_from_edges(edge_ids: tuple[int, ...]) -> SubgraphState:
    state = SubgraphState()
    for edge_id in edge_ids:
        state = state.with_edge(int(edge_id))
    return state


def _make_terminal_only_distribution(
    *,
    flat_state_index: int,
    current_component_count: int,
) -> HierarchicalStateActionDistribution:
    empty_long = torch.empty((0,), dtype=torch.long)
    empty_float = torch.empty((0,), dtype=torch.float32)
    return HierarchicalStateActionDistribution(
        flat_state_index=int(flat_state_index),
        stop_logit=torch.tensor(0.0, dtype=torch.float32),
        continue_logit=torch.tensor(float("-inf"), dtype=torch.float32),
        stop_choice_logits=torch.tensor([0.0], dtype=torch.float32),
        stop_choice_answer_entity_ids=torch.tensor([-1], dtype=torch.long),
        stop_choice_support_node_counts=torch.tensor([0], dtype=torch.long),
        node_choice_graph_node_ids=empty_long,
        node_choice_logits=empty_float,
        relation_choice_relation_ids=empty_long,
        relation_choice_logits=empty_float,
        relation_choice_node_choice_indices=empty_long,
        edge_choice_edge_ids=empty_long,
        edge_choice_source_graph_nodes=empty_long,
        edge_choice_relation_ids=empty_long,
        edge_choice_target_graph_nodes=empty_long,
        edge_choice_logits=empty_float,
        edge_choice_next_component_counts=empty_long,
        edge_choice_question_similarity=empty_float,
        edge_choice_semantic_overlap=empty_float,
        edge_choice_action_new_bit_gain=empty_long,
        edge_choice_candidate_commit_counts=empty_long,
        edge_choice_target_answer_distance=empty_float,
        edge_choice_relation_choice_indices=empty_long,
        node_relation_ptr=torch.tensor([0], dtype=torch.long),
        relation_edge_ptr=torch.tensor([0], dtype=torch.long),
        current_component_count=int(current_component_count),
        current_commit_candidate_count=0,
        current_oracle_distance=0.0,
    )


def _fake_sample_batch(
    *,
    num_graphs: int,
    num_rollouts: int,
    max_actions: int,
    max_edges: int,
    sample_ids: tuple[str, ...],
    question_ids: tuple[str, ...],
    termination_action_steps: torch.Tensor,
    terminal_commit_candidate_counts: torch.Tensor,
    terminal_gold_answer_counts: torch.Tensor,
    terminal_hit_mask: torch.Tensor,
    terminal_component_counts: torch.Tensor,
    chosen_answer_entity_ids: torch.Tensor,
    terminal_edge_ids: tuple[tuple[int, ...], ...],
    terminal_node_ids: tuple[tuple[int, ...], ...],
    terminal_reachability_bits: tuple[dict[int, int], ...],
) -> SubgraphTrajectorySampleBatch:
    return SubgraphTrajectorySampleBatch(
        state_log_flows=torch.zeros(
            (num_graphs, num_rollouts, max_actions), dtype=torch.float32
        ),
        log_pf_actions=torch.zeros(
            (num_graphs, num_rollouts, max_actions), dtype=torch.float32
        ),
        log_pb_actions=torch.zeros(
            (num_graphs, num_rollouts, max_actions), dtype=torch.float32
        ),
        log_reward_actions=torch.zeros(
            (num_graphs, num_rollouts, max_actions), dtype=torch.float32
        ),
        action_mask=torch.zeros(
            (num_graphs, num_rollouts, max_actions), dtype=torch.bool
        ),
        termination_action_steps=termination_action_steps,
        chosen_edge_ids=torch.full(
            (num_graphs, num_rollouts, max_edges), -1, dtype=torch.long
        ),
        stop_actions=torch.zeros(
            (num_graphs, num_rollouts, max_actions), dtype=torch.bool
        ),
        terminal_commit_candidate_counts=terminal_commit_candidate_counts,
        terminal_gold_answer_counts=terminal_gold_answer_counts,
        terminal_hit_mask=terminal_hit_mask,
        terminal_component_counts=terminal_component_counts,
        chosen_answer_entity_ids=chosen_answer_entity_ids,
        terminal_edge_ids=terminal_edge_ids,
        terminal_node_ids=terminal_node_ids,
        terminal_reachability_bits=terminal_reachability_bits,
        sample_ids=sample_ids,
        question_ids=question_ids,
        num_graphs=num_graphs,
        num_rollouts=num_rollouts,
    )


def test_initial_subgraph_state_starts_from_all_anchors() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=SubgraphState(),
    )

    assert analysis.selected_node_ids == (0, 2)
    assert analysis.reachability_bits == {0: 1, 2: 2}
    assert analysis.anchor_component_count == 2


def test_subgraph_transition_semantically_merges_answer_entity() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=_build_state_from_edges((0, 1)),
    )

    assert analysis.entity_reachability_bits[101] == 3
    assert analysis.anchor_component_count == 1


def test_answer_commit_reward_prefers_gold_answer_over_failure() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=_build_state_from_edges((0, 1)),
    )

    failure_reward, failure_commit_count, failure_gold_count, failure_hit = (
        policy.compute_stop_log_reward(
            prepared_batch=prepared_batch,
            graph_idx=0,
            analysis=analysis,
            answer_entity_id=None,
        )
    )
    gold_reward, gold_commit_count, gold_gold_count, gold_hit = (
        policy.compute_stop_log_reward(
            prepared_batch=prepared_batch,
            graph_idx=0,
            analysis=analysis,
            answer_entity_id=101,
        )
    )

    assert failure_hit is False
    assert failure_commit_count == 1
    assert failure_gold_count == 1
    assert gold_hit is True
    assert gold_commit_count == 1
    assert gold_gold_count == 1
    assert gold_reward > failure_reward


def test_invalid_answer_commit_is_rejected_before_answer_ready() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=_build_state_from_edges((0,)),
    )

    with pytest.raises(ValueError, match="admissible answer entity"):
        policy.compute_stop_log_reward(
            prepared_batch=prepared_batch,
            graph_idx=0,
            analysis=analysis,
            answer_entity_id=101,
        )


def test_backward_policy_rejects_non_forward_valid_parent_states() -> None:
    batch = _make_two_hop_multi_anchor_batch()
    policy = _make_subgraph_policy(max_steps=4)
    prepared_batch = policy.prepare_batch(batch)
    terminal_state = _build_state_from_edges((0, 1, 2, 3))

    removable = forward_valid_removable_edge_ids(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=terminal_state,
    )

    assert removable == (1, 3)
    backward_log_prob = policy.compute_backward_log_prob(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=terminal_state,
    )
    assert backward_log_prob == pytest.approx(-math.log(2.0))


def test_teacher_guidance_uses_multi_anchor_union_path() -> None:
    batch = _make_two_hop_multi_anchor_batch()
    policy = _make_subgraph_policy(max_steps=4)
    prepared_batch = policy.prepare_batch(batch)

    assert prepared_batch.graph_teacher_action_edge_ids == ((0, 1, 2, 3),)
    assert prepared_batch.graph_teacher_edge_count == (4,)


def test_teacher_forced_union_guidance_commits_to_gold_answer() -> None:
    batch = _make_two_hop_multi_anchor_batch()
    policy = _make_subgraph_policy(max_steps=4)
    prepared_batch = policy.prepare_batch(batch)
    sampler = SubgraphSampler(max_steps=4)

    sample_batch = sampler.teacher_force(
        policy=policy,
        prepared_batch=prepared_batch,
        edge_sequences=((0, 1, 2, 3),),
    )

    assert sample_batch.terminal_hit_mask[0, 0].item() is True
    assert sample_batch.terminal_commit_candidate_counts[0, 0].item() == 1
    assert sample_batch.terminal_gold_answer_counts[0, 0].item() == 1
    assert sample_batch.chosen_edge_ids[0, 0].tolist() == [0, 1, 2, 3]
    assert sample_batch.chosen_answer_entity_ids[0, 0].item() == 101


def test_guidance_replay_reindexes_edge_ids_across_records() -> None:
    batch = TrajectoryBatch.concatenate(
        [
            _make_bridge_batch(sample_id="bridge-guidance-a"),
            _make_bridge_batch(sample_id="bridge-guidance-b"),
        ]
    )
    training_cfg = _make_training_cfg(
        auxiliary={
            "replay": {
                "enabled": True,
                "mix_alpha": 0.2,
                "buffer": {
                    "capacity": 16,
                    "min_buffer_size": 64,
                    "replay_trajectories_per_step": 2,
                    "deduplicate": True,
                },
                "guidance": {"add_shortest_path_guidance": True},
            }
        }
    )
    module = GFlowNetModule(
        horizon_cfg={"max_steps": 2},
        training_cfg=training_cfg,
        policy_cfg=_make_subgraph_policy_config(),
        eval_cfg=_make_eval_cfg(report_profile="rank_only"),
        optimizer_cfg={"type": "adamw", "lr": 1.0e-4, "weight_decay": 0.0},
        scheduler_cfg={"type": "cosine", "interval": "step", "t_max": 8},
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )

    replay_payload = module._build_replay_batch(
        batch=batch,
        prepared_batch=module.policy.prepare_batch(batch),
    )

    assert replay_payload is not None
    replay_batch, replay_sequences, replay_metadata = replay_payload
    assert replay_batch.num_graphs == 2
    assert replay_sequences == ((0, 1), (2, 3))
    assert replay_metadata["guidance_records"] == pytest.approx(2.0)


def test_success_replay_buffer_keeps_only_hit_rollouts() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    sampler = SubgraphSampler(max_steps=2)
    success_sample = sampler.teacher_force(
        policy=policy,
        prepared_batch=prepared_batch,
        edge_sequences=((0, 1),),
    )
    failure_sample = _fake_sample_batch(
        num_graphs=1,
        num_rollouts=1,
        max_actions=3,
        max_edges=2,
        sample_ids=("bridge-subgraph",),
        question_ids=("bridge-subgraph",),
        termination_action_steps=torch.tensor([[3]], dtype=torch.long),
        terminal_commit_candidate_counts=torch.tensor([[0]], dtype=torch.long),
        terminal_gold_answer_counts=torch.tensor([[0]], dtype=torch.long),
        terminal_hit_mask=torch.tensor([[False]], dtype=torch.bool),
        terminal_component_counts=torch.tensor([[2]], dtype=torch.long),
        chosen_answer_entity_ids=torch.tensor([[-1]], dtype=torch.long),
        terminal_edge_ids=((0,),),
        terminal_node_ids=((0, 2),),
        terminal_reachability_bits=({0: 1, 2: 2},),
    )
    replay_buffer = SubgraphSuccessReplayBuffer(capacity=4, deduplicate=True)

    assert (
        replay_buffer.add_successful_trajectories(
            batch=batch,
            sample_batch=success_sample,
        )
        == 1
    )
    assert (
        replay_buffer.add_successful_trajectories(
            batch=batch,
            sample_batch=failure_sample,
        )
        == 0
    )
    assert len(replay_buffer) == 1


def test_subtb_loss_is_finite_on_sampled_batch() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    sampler = SubgraphSampler(max_steps=2)
    sample_batch = sampler.sample(
        policy=policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=2,
        temperature=1.0,
        proposal_bias_scale=0.0,
        action_pruning={"per_node_top_k": 0, "per_state_top_k": 0},
    )
    loss_output = SubgraphSubTrajectoryBalanceLoss().compute(sample_batch)

    assert torch.isfinite(loss_output.loss).item()
    assert torch.isfinite(loss_output.subtb_loss).item()


def test_subtb_loss_reports_true_residual_variance() -> None:
    sample_batch = SubgraphTrajectorySampleBatch(
        state_log_flows=torch.tensor([[[1.0, 0.5]]], dtype=torch.float32),
        log_pf_actions=torch.tensor([[[0.4, 0.0]]], dtype=torch.float32),
        log_pb_actions=torch.zeros((1, 1, 2), dtype=torch.float32),
        log_reward_actions=torch.zeros((1, 1, 2), dtype=torch.float32),
        action_mask=torch.tensor([[[True, True]]], dtype=torch.bool),
        termination_action_steps=torch.tensor([[2]], dtype=torch.long),
        chosen_edge_ids=torch.zeros((1, 1, 2), dtype=torch.long),
        stop_actions=torch.zeros((1, 1, 2), dtype=torch.bool),
        terminal_commit_candidate_counts=torch.tensor([[1]], dtype=torch.long),
        terminal_gold_answer_counts=torch.tensor([[1]], dtype=torch.long),
        terminal_hit_mask=torch.tensor([[True]]),
        terminal_component_counts=torch.tensor([[1]], dtype=torch.long),
        terminal_edge_ids=((0, 1),),
        terminal_node_ids=((0, 1),),
        terminal_reachability_bits=({},),
        sample_ids=("sample-1",),
        question_ids=("question-1",),
        num_graphs=1,
        num_rollouts=1,
    )

    loss_output = SubgraphSubTrajectoryBalanceLoss().compute(sample_batch)

    assert loss_output.subtb_loss == pytest.approx((0.81 + 1.96 + 0.25) / 3.0)
    assert loss_output.residual_variance == pytest.approx(
        ((0.81 + 1.96 + 0.25) / 3.0) - (((0.9 + 1.4 + 0.5) / 3.0) ** 2)
    )


def test_sampler_deduplicates_identical_active_states_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    sampler = SubgraphSampler(max_steps=2)
    build_calls: list[int] = []

    def _fake_build_action_distribution_from_state_features(**kwargs: object):
        rollout_batch = kwargs["rollout_batch"]
        analyses = kwargs["analyses"]
        state_features = kwargs["state_features"]
        build_calls.append(int(len(rollout_batch.states)))
        return SubgraphActionDistribution(
            flat_state_indices=torch.arange(
                len(rollout_batch.states), dtype=torch.long
            ),
            state_features=state_features,
            state_distributions=tuple(
                _make_terminal_only_distribution(
                    flat_state_index=flat_state_index,
                    current_component_count=int(analysis.anchor_component_count),
                )
                for flat_state_index, analysis in enumerate(analyses)
            ),
        )

    monkeypatch.setattr(
        policy,
        "build_action_distribution_from_state_features",
        _fake_build_action_distribution_from_state_features,
    )

    sample_batch = sampler.sample(
        policy=policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=4,
        temperature=1.0,
        proposal_bias_scale=0.0,
        action_pruning={"per_node_top_k": 0, "per_state_top_k": 0},
    )

    assert build_calls == [1]
    assert sample_batch.termination_action_steps[0].tolist() == [1, 1, 1, 1]


def test_runtime_counts_committed_answers_instead_of_all_ready_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_multi_answer_bridge_batch()
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=_make_eval_cfg(
            report_profile="rank_only",
            monte_carlo={"rollouts": 4, "batch_rollouts": 4, "confidence": 0.95},
        ),
        policy=_make_subgraph_policy(max_steps=4),
        sampler=SubgraphSampler(max_steps=4),
    )

    def _fake_sample(**kwargs: object) -> SubgraphTrajectorySampleBatch:
        del kwargs
        return _fake_sample_batch(
            num_graphs=1,
            num_rollouts=4,
            max_actions=5,
            max_edges=4,
            sample_ids=("multi-answer-bridge",),
            question_ids=("multi-answer-bridge",),
            termination_action_steps=torch.tensor([[5, 5, 5, 5]], dtype=torch.long),
            terminal_commit_candidate_counts=torch.tensor(
                [[2, 2, 0, 0]], dtype=torch.long
            ),
            terminal_gold_answer_counts=torch.tensor([[2, 2, 0, 0]], dtype=torch.long),
            terminal_hit_mask=torch.tensor(
                [[True, True, False, False]], dtype=torch.bool
            ),
            terminal_component_counts=torch.tensor([[1, 1, 1, 2]], dtype=torch.long),
            chosen_answer_entity_ids=torch.tensor(
                [[101, 101, 103, -1]], dtype=torch.long
            ),
            terminal_edge_ids=((0, 1, 2, 3), (0, 1, 2, 3), (0, 1), (0,)),
            terminal_node_ids=((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2), (0, 2)),
            terminal_reachability_bits=(
                {0: 1, 1: 3, 2: 2, 3: 3},
                {0: 1, 1: 3, 2: 2, 3: 3},
                {0: 1, 1: 3, 2: 2},
                {0: 1, 2: 2},
            ),
        )

    monkeypatch.setattr(runtime.sampler, "sample", _fake_sample)

    result = runtime._predict_single_graph(batch=batch, include_answer_support=True)

    assert result["predicted_answer_entity_ids"] == [101, 103]
    assert result["answer_log_masses"] == pytest.approx([math.log(0.5), math.log(0.25)])
    assert result["answering_rollout_count"] == 3
    assert result["hit_rollout_count"] == 2
    assert result["top_subgraph_answer_entity_id"] == 101


def test_runtime_stops_early_when_top_answer_is_statistically_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_bridge_batch()
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=_make_eval_cfg(
            report_profile="rank_only",
            monte_carlo={
                "rollouts": 20,
                "batch_rollouts": 4,
                "temperature": 0.7,
                "confidence": 0.95,
                "early_stop": {
                    "enabled": True,
                    "min_rollouts": 12,
                    "stability_top_k": 1,
                },
                "action_pruning": {"per_node_top_k": 5, "per_state_top_k": 7},
            },
        ),
        policy=_make_subgraph_policy(max_steps=2),
        sampler=SubgraphSampler(max_steps=2),
    )
    seen_calls: list[dict[str, object]] = []

    def _fake_sample(**kwargs: object) -> SubgraphTrajectorySampleBatch:
        seen_calls.append(dict(kwargs))
        return _fake_sample_batch(
            num_graphs=1,
            num_rollouts=4,
            max_actions=3,
            max_edges=2,
            sample_ids=("bridge-subgraph",),
            question_ids=("bridge-subgraph",),
            termination_action_steps=torch.tensor([[3, 3, 3, 3]], dtype=torch.long),
            terminal_commit_candidate_counts=torch.tensor(
                [[1, 1, 1, 1]], dtype=torch.long
            ),
            terminal_gold_answer_counts=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            terminal_hit_mask=torch.tensor(
                [[True, True, True, True]], dtype=torch.bool
            ),
            terminal_component_counts=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            chosen_answer_entity_ids=torch.tensor(
                [[101, 101, 101, 101]], dtype=torch.long
            ),
            terminal_edge_ids=((0, 1), (0, 1), (0, 1), (0, 1)),
            terminal_node_ids=((0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)),
            terminal_reachability_bits=(
                {0: 1, 1: 3, 2: 2},
                {0: 1, 1: 3, 2: 2},
                {0: 1, 1: 3, 2: 2},
                {0: 1, 1: 3, 2: 2},
            ),
        )

    monkeypatch.setattr(runtime.sampler, "sample", _fake_sample)

    result = runtime._predict_single_graph(batch=batch, include_answer_support=False)

    assert len(seen_calls) == 3
    assert seen_calls[0]["temperature"] == pytest.approx(0.7)
    assert seen_calls[0]["action_pruning"] == {
        "per_node_top_k": 5,
        "per_state_top_k": 7,
    }
    assert result["rollout_count"] == 12
    assert result["stopped_early"] is True
    assert result["predicted_answer_entity_ids"] == [101]
    assert result["answer_log_masses"] == pytest.approx([0.0])


def test_runtime_batches_graphs_and_shrinks_active_eval_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = TrajectoryBatch.concatenate(
        [_make_bridge_batch(sample_id="g0"), _make_bridge_batch(sample_id="g1")]
    )
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=_make_eval_cfg(
            report_profile="rank_only",
            monte_carlo={
                "rollouts": 8,
                "batch_rollouts": 4,
                "temperature": 1.0,
                "confidence": 0.95,
                "early_stop": {
                    "enabled": True,
                    "min_rollouts": 4,
                    "stability_top_k": 1,
                },
                "action_pruning": {"per_node_top_k": 5, "per_state_top_k": 7},
            },
        ),
        policy=_make_subgraph_policy(max_steps=2),
        sampler=SubgraphSampler(max_steps=2),
    )
    seen_num_graphs: list[int] = []
    prepare_batch_calls = 0

    original_prepare_batch = runtime.policy.prepare_batch

    def _counted_prepare_batch(
        *args: object, **kwargs: object
    ) -> SubgraphPreparedBatch:
        nonlocal prepare_batch_calls
        prepare_batch_calls += 1
        return original_prepare_batch(*args, **kwargs)

    def _unexpected_select_graphs(
        self: TrajectoryBatch, *args: object, **kwargs: object
    ) -> TrajectoryBatch:
        del self, args, kwargs
        raise AssertionError(
            "runtime should reuse the prepared batch instead of selecting graphs"
        )

    def _fake_sample(**kwargs: object) -> SubgraphTrajectorySampleBatch:
        prepared_batch = kwargs["prepared_batch"]
        num_graphs = int(prepared_batch.num_graphs)
        seen_num_graphs.append(num_graphs)
        if num_graphs == 2:
            return _fake_sample_batch(
                num_graphs=2,
                num_rollouts=4,
                max_actions=3,
                max_edges=2,
                sample_ids=("g0", "g1"),
                question_ids=("g0", "g1"),
                termination_action_steps=torch.tensor(
                    [[3, 3, 3, 3], [3, 3, 3, 3]], dtype=torch.long
                ),
                terminal_commit_candidate_counts=torch.tensor(
                    [[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.long
                ),
                terminal_gold_answer_counts=torch.tensor(
                    [[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.long
                ),
                terminal_hit_mask=torch.tensor(
                    [[True, True, True, True], [False, False, False, False]],
                    dtype=torch.bool,
                ),
                terminal_component_counts=torch.tensor(
                    [[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.long
                ),
                chosen_answer_entity_ids=torch.tensor(
                    [[101, 101, 101, 101], [-1, -1, -1, -1]], dtype=torch.long
                ),
                terminal_edge_ids=(
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (2,),
                    (2,),
                    (2,),
                    (2,),
                ),
                terminal_node_ids=(
                    (0, 1, 2),
                    (0, 1, 2),
                    (0, 1, 2),
                    (0, 1, 2),
                    (3, 5),
                    (3, 5),
                    (3, 5),
                    (3, 5),
                ),
                terminal_reachability_bits=(
                    {0: 1, 1: 3, 2: 2},
                    {0: 1, 1: 3, 2: 2},
                    {0: 1, 1: 3, 2: 2},
                    {0: 1, 1: 3, 2: 2},
                    {3: 1, 5: 2},
                    {3: 1, 5: 2},
                    {3: 1, 5: 2},
                    {3: 1, 5: 2},
                ),
            )
        return _fake_sample_batch(
            num_graphs=1,
            num_rollouts=4,
            max_actions=3,
            max_edges=2,
            sample_ids=("g1",),
            question_ids=("g1",),
            termination_action_steps=torch.tensor([[3, 3, 3, 3]], dtype=torch.long),
            terminal_commit_candidate_counts=torch.tensor(
                [[0, 0, 0, 0]], dtype=torch.long
            ),
            terminal_gold_answer_counts=torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
            terminal_hit_mask=torch.tensor(
                [[False, False, False, False]], dtype=torch.bool
            ),
            terminal_component_counts=torch.tensor([[2, 2, 2, 2]], dtype=torch.long),
            chosen_answer_entity_ids=torch.tensor([[-1, -1, -1, -1]], dtype=torch.long),
            terminal_edge_ids=((2,), (2,), (2,), (2,)),
            terminal_node_ids=((3, 5), (3, 5), (3, 5), (3, 5)),
            terminal_reachability_bits=(
                {3: 1, 5: 2},
                {3: 1, 5: 2},
                {3: 1, 5: 2},
                {3: 1, 5: 2},
            ),
        )

    def _fake_stability_margin(**kwargs: object) -> float | None:
        return 1.0 if kwargs["answer_vote_counts"] else None

    monkeypatch.setattr(runtime.policy, "prepare_batch", _counted_prepare_batch)
    monkeypatch.setattr(TrajectoryBatch, "select_graphs", _unexpected_select_graphs)
    monkeypatch.setattr(runtime.sampler, "sample", _fake_sample)
    monkeypatch.setattr(
        "src.metrics.subgraph_answer_search_runtime._topk_stability_margin",
        _fake_stability_margin,
    )

    results = runtime._predict_batch_results(batch=batch, include_answer_support=False)

    assert seen_num_graphs == [2, 1]
    assert prepare_batch_calls == 1
    assert [result["sample_id"] for result in results] == ["g0", "g1"]
    assert results[0]["rollout_count"] == 4
    assert results[0]["stopped_early"] is True
    assert results[1]["rollout_count"] == 8
    assert results[1]["stopped_early"] is False


def test_runtime_applies_support_mass_threshold_and_overlap_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_multi_answer_bridge_batch()
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=_make_eval_cfg(
            report_profile="full",
            support_mass_threshold=0.6,
            support_path_overlap_penalty=1.0,
            edge_emit_top_k=3,
            monte_carlo={
                "rollouts": 4,
                "batch_rollouts": 4,
                "temperature": 1.0,
                "confidence": 0.95,
                "early_stop": {
                    "enabled": False,
                    "min_rollouts": 4,
                    "stability_top_k": 1,
                },
                "action_pruning": {"per_node_top_k": 100, "per_state_top_k": 256},
            },
        ),
        policy=_make_subgraph_policy(max_steps=4),
        sampler=SubgraphSampler(max_steps=4),
    )

    def _fake_sample(**kwargs: object) -> SubgraphTrajectorySampleBatch:
        del kwargs
        return _fake_sample_batch(
            num_graphs=1,
            num_rollouts=4,
            max_actions=5,
            max_edges=4,
            sample_ids=("multi-answer-bridge",),
            question_ids=("multi-answer-bridge",),
            termination_action_steps=torch.tensor([[5, 5, 5, 5]], dtype=torch.long),
            terminal_commit_candidate_counts=torch.tensor(
                [[2, 0, 0, 1]], dtype=torch.long
            ),
            terminal_gold_answer_counts=torch.tensor([[2, 0, 0, 1]], dtype=torch.long),
            terminal_hit_mask=torch.tensor(
                [[True, False, False, True]], dtype=torch.bool
            ),
            terminal_component_counts=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            chosen_answer_entity_ids=torch.tensor(
                [[101, 101, -1, 103]], dtype=torch.long
            ),
            terminal_edge_ids=((0, 1), (0, 1), (0, 1, 2), (3,)),
            terminal_node_ids=((0, 1, 2), (0, 1, 2), (0, 1, 2, 3), (0, 2, 3)),
            terminal_reachability_bits=(
                {0: 1, 1: 3, 2: 2},
                {0: 1, 1: 3, 2: 2},
                {0: 1, 1: 3, 2: 2, 3: 3},
                {0: 1, 2: 2, 3: 3},
            ),
        )

    monkeypatch.setattr(runtime.sampler, "sample", _fake_sample)

    result = runtime._predict_single_graph(batch=batch, include_answer_support=True)
    metrics = runtime.summarize_predict_epoch(
        predict_results=[result],
        report_profile="full",
    )

    assert [entry["edge_ids"] for entry in result["terminal_subgraphs"]] == [
        [0, 1],
        [3],
    ]
    assert [
        entry["chosen_answer_entity_id"] for entry in result["terminal_subgraphs"]
    ] == [
        101,
        103,
    ]
    assert result["support_probabilities"] == pytest.approx([0.5, 0.25])
    assert result["support_probability_mass"] == pytest.approx(0.75)
    assert metrics["support/mass@1"] == pytest.approx(0.5)
    assert metrics["answer_commit/rollout_count"] == pytest.approx(4.0)
