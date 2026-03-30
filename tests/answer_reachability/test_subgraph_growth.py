from __future__ import annotations

import math

import pytest
import torch

from src.graph import TrajectoryBatch
from src.metrics import subgraph_answer_search_runtime as subgraph_runtime_module
from src.metrics.runtime_factory import GraphTaskRuntimeFactory
from src.metrics.subgraph_answer_search_runtime import SubgraphAnswerSearchRuntime
from src.models.gflownet import SubgraphSuccessReplayBuffer
from src.models.gflownet.losses import SubgraphSubTrajectoryBalanceLoss
from src.models.gflownet.policy import SUBGRAPH_STATE_MODE, SubgraphPolicy
from src.models.gflownet.sampler import SubgraphSampler
from src.models.gflownet.search import (
    SubgraphBeamSearchResult,
    SubgraphTerminalSubgraph,
    beam_search_subgraphs,
)
from src.models.gflownet.state import SubgraphAction, SubgraphState
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
        "force_stop_on_answer_hit": False,
        "terminal_failure_log_reward": -3.0,
        "step_log_penalty": 0.0,
        "answer_stop_log_reward_bonus": 0.0,
        "sampling_temperature_schedule": {"type": "constant", "hold_steps": 0},
        "proposal_bias_schedule": {"type": "constant", "hold_steps": 0},
        "success_replay": {
            "mix_alpha": 0.0,
            "capacity": 1024,
            "min_buffer_size": 64,
            "replay_trajectories_per_step": None,
            "deduplicate": True,
            "add_shortest_path_guidance": False,
            "expand_imitation_weight": 0.0,
            "expand_imitation_from_anchor_bonus": 0.0,
            "expand_imitation_answer_finish_bonus": 0.0,
            "mask_stop_loss": True,
        },
        "replay_mix_schedule": {"type": "constant", "hold_steps": 0},
        "answer_quotient": {
            "enabled": False,
            "weight": 0.0,
            "direct_entity_ranking_weight": 0.0,
            "replace_terminal_loss": False,
            "gold_reward_mode": "shared",
            "allocate_stop_mass": False,
        },
        "potential_reward": {
            "answer_distance_weight": 0.0,
            "unreachable_distance": None,
        },
        "subgraph_reward": {
            "c_step": 0.1,
            "lambda_conn": 0.5,
            "beta_answer_bits": 0.0,
            "beta_answer_full": 0.0,
            "beta_hit": 2.0,
            "beta_cnt": 0.25,
            "beta_early": 1.0,
            "min_stop_edges": 1,
        },
        "subgraph_proposal": {
            "oracle_answer_distance_weight": 0.0,
            "prior_question_similarity_weight": 0.0,
            "prior_component_merge_weight": 0.0,
            "stop_hit_bias": 0.0,
        },
        "subtb": {
            "lambda_weight": 1.0,
            "normalize": True,
            "root_loss_weight": 1.0,
            "pairwise_loss_weight": 1.0,
            "terminal_loss_weight": 1.0,
        },
    }
    training_cfg.update(overrides)
    return training_cfg


def _make_eval_cfg(**overrides: object) -> dict[str, object]:
    eval_cfg: dict[str, object] = {
        "report_profile": "full",
        "task": "answer_ranking",
        "answer_mass_threshold": 0.9,
        "support_mass_threshold": 0.9,
        "support_path_overlap_penalty": 0.25,
        "answer_top_ks": (1, 5, 10),
        "edge_top_ks": (1, 5, 10, 25, 50),
        "edge_emit_top_k": 25,
        "answer_posterior_backend": "flow_frontier",
        "flow_frontier": {
            "prune_epsilon": 1.0e-3,
            "max_expansions": 20000,
            "max_frontier_size": 4096,
        },
        "monte_carlo": {"rollouts": 4096, "confidence": 0.95},
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


def _make_subgraph_policy(*, max_steps: int = 2) -> SubgraphPolicy:
    policy_cfg = _make_subgraph_policy_config()
    training_cfg = _make_training_cfg(rollouts_per_graph=2, sampling_temperature=1.0)
    return SubgraphPolicy(
        state_mode=str(policy_cfg["state_mode"]),
        backbone=dict(policy_cfg["backbone"]),
        flow_head=dict(policy_cfg["flow_head"]),
        state_encoder=dict(policy_cfg["state_encoder"]),
        actor=dict(policy_cfg["actor"]),
        subgraph_reward=dict(training_cfg["subgraph_reward"]),
        subgraph_proposal=dict(training_cfg["subgraph_proposal"]),
        max_steps=max_steps,
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


def test_initial_subgraph_state_starts_from_all_anchors() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    state = SubgraphState()
    analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=state,
    )

    assert state.edge_ids == ()
    assert analysis.selected_node_ids == (0, 2)
    assert analysis.reachability_bits == {0: 1, 2: 2}
    assert analysis.anchor_component_count == 2


def test_subgraph_transition_allows_bridge_to_existing_node() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    rollout_batch = policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=1,
    )
    first_state = policy.transition(
        rollout_batch=rollout_batch,
        chosen_actions=(SubgraphAction.add_edge(0),),
    )
    first_analysis = policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=first_state,
    )
    distribution = policy.compute_action_distribution(
        prepared_batch=prepared_batch,
        rollout_batch=first_state,
        analyses=first_analysis,
    )

    matching = torch.nonzero(
        (distribution.edge_ids == 1) & (~distribution.is_stop_action), as_tuple=False
    ).view(-1)
    assert matching.numel() == 1
    action_pos = int(matching.item())
    assert int(distribution.target_nodes[action_pos].item()) == 1
    assert int(distribution.current_component_counts[action_pos].item()) == 2
    assert int(distribution.next_component_counts[action_pos].item()) == 1

    second_state = policy.transition(
        rollout_batch=first_state,
        chosen_actions=(SubgraphAction.add_edge(1),),
    )
    second_analysis = policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=second_state,
    )
    assert second_state.states[0].edge_ids == (0, 1)
    assert second_analysis[0].selected_node_ids == (0, 1, 2)
    assert second_analysis[0].anchor_component_count == 1


def test_stop_reward_requires_full_anchor_coverage() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    rollout_batch = policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=1,
    )
    partial_state = policy.transition(
        rollout_batch=rollout_batch,
        chosen_actions=(SubgraphAction.add_edge(0),),
    )
    partial_analysis = policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=partial_state,
    )
    partial_reward, partial_count, partial_hit = policy.compute_stop_log_reward(
        prepared_batch=prepared_batch,
        graph_idx=0,
        analysis=partial_analysis[0],
    )
    assert partial_hit is False
    assert partial_count == 0
    assert partial_reward == pytest.approx(-float(policy.reward_model.beta_early))

    full_state = policy.transition(
        rollout_batch=partial_state,
        chosen_actions=(SubgraphAction.add_edge(1),),
    )
    full_analysis = policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=full_state,
    )
    full_reward, full_count, full_hit = policy.compute_stop_log_reward(
        prepared_batch=prepared_batch,
        graph_idx=0,
        analysis=full_analysis[0],
    )
    assert full_hit is True
    assert full_count == 1
    assert full_reward == pytest.approx(
        float(policy.reward_model.beta_hit)
        + float(policy.reward_model.beta_cnt) * math.log1p(1.0)
    )


def test_expand_reward_can_shape_answer_bit_progress_before_full_hit() -> None:
    batch = _make_bridge_batch()
    training_cfg = _make_training_cfg(
        subgraph_reward={
            "c_step": 0.1,
            "lambda_conn": 0.5,
            "beta_answer_bits": 0.2,
            "beta_answer_full": 0.5,
            "beta_hit": 2.0,
            "beta_cnt": 0.25,
            "beta_early": 1.0,
            "min_stop_edges": 1,
        }
    )
    policy_cfg = _make_subgraph_policy_config()
    policy = SubgraphPolicy(
        state_mode=str(policy_cfg["state_mode"]),
        backbone=dict(policy_cfg["backbone"]),
        flow_head=dict(policy_cfg["flow_head"]),
        state_encoder=dict(policy_cfg["state_encoder"]),
        actor=dict(policy_cfg["actor"]),
        subgraph_reward=dict(training_cfg["subgraph_reward"]),
        subgraph_proposal=dict(training_cfg["subgraph_proposal"]),
        max_steps=2,
    )
    prepared_batch = policy.prepare_batch(batch)
    current_analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=SubgraphState(),
    )
    next_analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=SubgraphState(edge_ids=(0,)),
    )

    reward = policy.compute_expand_log_reward(
        current_analysis=current_analysis,
        next_analysis=next_analysis,
        prepared_batch=prepared_batch,
        graph_idx=0,
    )

    assert reward == pytest.approx(0.1)


def test_teacher_guidance_uses_multi_anchor_union_path() -> None:
    batch = _make_two_hop_multi_anchor_batch()
    policy = _make_subgraph_policy(max_steps=4)
    prepared_batch = policy.prepare_batch(batch)

    assert prepared_batch.graph_teacher_action_edge_ids == ((0, 1, 2, 3),)
    assert prepared_batch.graph_teacher_edge_count == (4,)


def test_stop_reward_does_not_penalize_over_budget_multi_anchor_samples() -> None:
    batch = _make_two_hop_multi_anchor_batch()
    policy = _make_subgraph_policy(max_steps=3)
    prepared_batch = policy.prepare_batch(batch)
    analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=SubgraphState(edge_ids=(0, 1, 2)),
    )

    reward, answer_count, hit = policy.compute_stop_log_reward(
        prepared_batch=prepared_batch,
        graph_idx=0,
        analysis=analysis,
    )

    assert hit is False
    assert answer_count == 0
    assert reward == pytest.approx(0.0)


def test_teacher_forced_union_guidance_can_hit_multi_anchor_answer() -> None:
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
    assert sample_batch.terminal_answer_counts[0, 0].item() == 1
    assert sample_batch.chosen_edge_ids[0, 0].tolist() == [0, 1, 2, 3]


def test_guidance_replay_reindexes_edge_ids_across_records() -> None:
    batch = TrajectoryBatch.concatenate(
        [
            _make_bridge_batch(sample_id="bridge-guidance-a"),
            _make_bridge_batch(sample_id="bridge-guidance-b"),
        ]
    )
    training_cfg = _make_training_cfg(
        success_replay={
            "mix_alpha": 0.2,
            "capacity": 16,
            "min_buffer_size": 64,
            "replay_trajectories_per_step": 2,
            "deduplicate": True,
            "add_shortest_path_guidance": True,
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
    assert replay_sequences == ((0, 1), (2, 3))
    assert replay_metadata["guidance_records"] == pytest.approx(2.0)

    sample_batch = module._require_subgraph_sampler().teacher_force(
        policy=module.policy,
        prepared_batch=module.policy.prepare_batch(replay_batch),
        edge_sequences=replay_sequences,
    )

    assert sample_batch.terminal_hit_mask[:, 0].tolist() == [True, True]


def test_success_replay_buffer_stores_local_edge_ids() -> None:
    batch = TrajectoryBatch.concatenate(
        [
            _make_bridge_batch(sample_id="bridge-buffer-a"),
            _make_bridge_batch(sample_id="bridge-buffer-b"),
        ]
    )
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    sampler = SubgraphSampler(max_steps=2)
    sample_batch = sampler.teacher_force(
        policy=policy,
        prepared_batch=prepared_batch,
        edge_sequences=((0, 1), (2, 3)),
    )
    replay_buffer = SubgraphSuccessReplayBuffer(capacity=4, deduplicate=True)

    added = replay_buffer.add_successful_trajectories(
        batch=batch,
        sample_batch=sample_batch,
    )
    records = replay_buffer.sample(max_records=2)
    records.sort(key=lambda record: str(record.trajectory_batch.sample_ids[0]))

    assert added == 2
    assert [record.edge_ids for record in records] == [(0, 1), (0, 1)]


def test_beam_search_keeps_only_explicit_stop_terminals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    eval_cfg = _make_eval_cfg(
        report_profile="rank_only",
        flow_frontier={"max_frontier_size": 1, "max_expansions": 1},
    )

    def _prefer_expand_over_stop(distribution: object) -> torch.Tensor:
        if not hasattr(distribution, "is_stop_action"):
            raise AssertionError("unexpected distribution type")
        stop_mask = distribution.is_stop_action  # type: ignore[attr-defined]
        return torch.where(
            stop_mask,
            torch.full_like(stop_mask, -20.0, dtype=torch.float32),
            torch.zeros_like(stop_mask, dtype=torch.float32),
        )

    monkeypatch.setattr(policy, "compute_target_log_probs", _prefer_expand_over_stop)

    search_result = beam_search_subgraphs(
        policy=policy,
        eval_cfg=eval_cfg,
        prepared_batch=prepared_batch,
    )

    assert search_result.terminal_subgraphs == ()
    assert search_result.frontier_state_count == 1
    assert search_result.frontier_answering_state_count == 0


def test_subgraph_runtime_splits_terminal_mass_across_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_multi_answer_bridge_batch()
    policy = _make_subgraph_policy(max_steps=4)
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=_make_eval_cfg(report_profile="rank_only"),
        policy=policy,
        sampler=SubgraphSampler(max_steps=4),
    )

    def _fake_beam_search(**kwargs: object) -> SubgraphBeamSearchResult:
        prepared_batch = kwargs["prepared_batch"]
        if prepared_batch is None:
            raise AssertionError("prepared_batch is required")
        return SubgraphBeamSearchResult(
            terminal_subgraphs=(
                SubgraphTerminalSubgraph(
                    edge_ids=(0, 1, 2, 3),
                    log_mass=math.log(0.8),
                    selected_node_ids=(0, 1, 2, 3),
                    reachability_bits={0: 1, 1: 3, 2: 2, 3: 3},
                    answer_count=2,
                ),
            ),
            frontier_state_count=3,
            frontier_answering_state_count=1,
        )

    monkeypatch.setattr(
        subgraph_runtime_module, "beam_search_subgraphs", _fake_beam_search
    )

    result = runtime._predict_single_graph(batch=batch, include_answer_support=True)

    assert result["predicted_answer_entity_ids"] == [101, 103]
    assert result["answer_log_masses"] == pytest.approx(
        [math.log(0.8) - math.log(2.0), math.log(0.8) - math.log(2.0)]
    )
    assert result["frontier_state_count"] == 3
    assert result["frontier_answering_state_count"] == 1
    assert result["terminal_subgraphs"][0]["per_answer_log_mass"] == pytest.approx(
        math.log(0.8) - math.log(2.0)
    )


def test_gflownet_module_subgraph_mode_uses_subgraph_runtime() -> None:
    batch = _make_bridge_batch()
    module = GFlowNetModule(
        horizon_cfg={"max_steps": 2},
        training_cfg=_make_training_cfg(
            rollouts_per_graph=2,
            sampling_temperature=1.0,
        ),
        policy_cfg=_make_subgraph_policy_config(),
        eval_cfg=_make_eval_cfg(report_profile="rank_only"),
        optimizer_cfg={"type": "adamw", "lr": 1.0e-4, "weight_decay": 0.0},
        scheduler_cfg={"type": "cosine", "interval": "step", "t_max": 8},
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )

    assert isinstance(module.policy, SubgraphPolicy)
    assert isinstance(module.sampler, SubgraphSampler)
    assert isinstance(module.loss_fn, SubgraphSubTrajectoryBalanceLoss)

    loss = module.training_step(batch, 0)
    outputs = module._evaluate_batch_output(batch=batch)

    assert torch.isfinite(loss).item()
    assert "answer/hit@1" in outputs.primary_metrics
    assert len(outputs.results) == 1
    assert outputs.results[0]["sample_id"] == "bridge-subgraph"


def test_gflownet_module_supports_guidance_replay_training() -> None:
    batch = _make_bridge_batch()
    training_cfg = _make_training_cfg(
        success_replay={
            "mix_alpha": 0.2,
            "capacity": 16,
            "min_buffer_size": 64,
            "replay_trajectories_per_step": 1,
            "deduplicate": True,
            "add_shortest_path_guidance": True,
            "expand_imitation_weight": 1.0,
            "expand_imitation_from_anchor_bonus": 2.0,
            "expand_imitation_answer_finish_bonus": 4.0,
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

    loss = module.training_step(batch, 0)
    metrics = module.pop_latest_train_metrics()

    assert torch.isfinite(loss).item()
    assert metrics is not None
    assert metrics["train/replay_expand_imitation_loss"] > 0.0
    assert metrics["train/replay_expand_imitation_weight"] == pytest.approx(1.0)
    assert metrics["train/replay_expand_imitation_from_anchor_steps"] > 0.0
    assert metrics["train/replay_expand_imitation_answer_finish_steps"] > 0.0
    assert metrics["train/replay_expand_imitation_mean_weight"] > 1.0
    assert metrics["train/replay_mask_stop_loss"] == pytest.approx(1.0)
    assert metrics["train/replay_stop_loss_masked_count"] > 0.0
