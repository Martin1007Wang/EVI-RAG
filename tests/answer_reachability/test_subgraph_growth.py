from __future__ import annotations

import math

import pytest
import torch

from src.graph import TrajectoryBatch
from src.metrics import subgraph_answer_search_runtime as subgraph_runtime_module
from src.metrics.runtime_factory import GraphTaskRuntimeFactory
from src.metrics.subgraph_answer_search_runtime import SubgraphAnswerSearchRuntime
from src.models.configs import (
    ActionPriorConfig,
    BackboneConfig,
    FlowFrontierEvalConfig,
    GFlowNetTrainingConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
    SearchEvalConfig,
    StateScoreHeadConfig,
)
from src.models.configs.policy import (
    SUBGRAPH_STATE_MODE,
    SubgraphActionHeadConfig,
    SubgraphStateEncoderConfig,
)
from src.models.gflownet.subgraph.losses import SubgraphSubTrajectoryBalanceLoss
from src.models.gflownet.subgraph.policy import SubgraphPolicy
from src.models.gflownet.subgraph.sampler import SubgraphSampler
from src.models.gflownet.subgraph.search import (
    SubgraphBeamSearchResult,
    SubgraphTerminalSubgraph,
    beam_search_subgraphs,
)
from src.models.gflownet.subgraph.state import SubgraphAction, SubgraphState
from src.models.gflownet_module import GFlowNetModule

from .conftest import make_batch_from_graph


def _make_subgraph_policy_config() -> PolicyConfig:
    return PolicyConfig(
        state_mode=SUBGRAPH_STATE_MODE,
        backbone=BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            use_adapter=True,
            adapter_dim=4,
            adapter_dropout=0.0,
        ),
        state_score_head=StateScoreHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
        subgraph_state_encoder=SubgraphStateEncoderConfig(
            hidden_dim=8,
            num_layers=2,
            dropout=0.0,
        ),
        subgraph_action_head=SubgraphActionHeadConfig(
            hidden_dim=8,
            num_layers=2,
            dropout=0.0,
        ),
    )


def _make_bridge_batch() -> TrajectoryBatch:
    return make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0, 2], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="bridge-subgraph",
    )


def _make_subgraph_policy(*, max_steps: int = 2) -> SubgraphPolicy:
    return SubgraphPolicy(
        policy_cfg=_make_subgraph_policy_config(),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=2, sampling_temperature=1.0
        ),
        max_steps=max_steps,
    )


def _make_multi_answer_bridge_batch() -> TrajectoryBatch:
    return make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 2, 0, 2], [1, 1, 3, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        q_local_indices=torch.tensor([0, 2], dtype=torch.long),
        a_local_indices=torch.tensor([1, 3], dtype=torch.long),
        answer_entity_ids=torch.tensor([101, 103], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="multi-answer-bridge",
    )


def test_initial_subgraph_state_starts_from_all_anchors() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    state = SubgraphState()
    analysis = policy.env.analyze_state(
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
    first_state = policy.env.transition(
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

    second_state = policy.env.transition(
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
    partial_state = policy.env.transition(
        rollout_batch=rollout_batch,
        chosen_actions=(SubgraphAction.add_edge(0),),
    )
    partial_analysis = policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=partial_state,
    )
    partial_reward, partial_count, partial_hit = policy.env.compute_stop_log_reward(
        prepared_batch=prepared_batch,
        graph_idx=0,
        analysis=partial_analysis[0],
    )
    assert partial_hit is False
    assert partial_count == 0
    assert partial_reward == pytest.approx(
        -float(policy.training_cfg.subgraph_reward.beta_early)
    )

    full_state = policy.env.transition(
        rollout_batch=partial_state,
        chosen_actions=(SubgraphAction.add_edge(1),),
    )
    full_analysis = policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=full_state,
    )
    full_reward, full_count, full_hit = policy.env.compute_stop_log_reward(
        prepared_batch=prepared_batch,
        graph_idx=0,
        analysis=full_analysis[0],
    )
    assert full_hit is True
    assert full_count == 1
    assert full_reward == pytest.approx(
        float(policy.training_cfg.subgraph_reward.beta_hit)
        + float(policy.training_cfg.subgraph_reward.beta_cnt) * math.log1p(1.0)
    )


def test_beam_search_keeps_only_explicit_stop_terminals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    eval_cfg = SearchEvalConfig(
        report_profile="rank_only",
        flow_frontier=FlowFrontierEvalConfig(max_frontier_size=1, max_expansions=1),
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
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
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
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=2, sampling_temperature=1.0
        ),
        action_prior_cfg=ActionPriorConfig(),
        policy_cfg=_make_subgraph_policy_config(),
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
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
