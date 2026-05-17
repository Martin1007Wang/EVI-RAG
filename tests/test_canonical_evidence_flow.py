from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace
from torch import nn

from src.data.collate import RetrievalCollator
from src.data.dataset import _build_retrieval_data
from src.training.config import EvalRuntimeConfig, OptimizationRuntimeConfig, OptimizerRuntimeConfig
from src.data.schema.fields import SampleFields
from src.weaver.context import FlowContext
from src.weaver.loss import ProbabilityDBLoss
from src.weaver.module import WeaverModule
from src.weaver.nn.edge_flow_scorer import EdgeActionScorer
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.nn.frontier_encoder import FrontierEncoder
from src.weaver.nn.state_encoder import StateEncoder
from src.weaver.policy import Policy, PolicyOutput
from src.weaver.reward import EvidenceLogReward
from src.weaver.rollout.engine import RolloutContext, RolloutEngine
from src.weaver.rollout.replay import ReplayBatch
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.runner import RolloutChunk
from src.weaver.rollout.sampling import SampledAction
from src.weaver.rollout.trace import RolloutTrace
from src.weaver.state import Frontier, FrontierBuilder, State, assert_anchor_connected_state
from src.weaver.transitions import TransitionBatch


def test_policy_outputs_normalized_probabilities() -> None:
    batch = _batch()
    features = _features()
    context = FlowContext.from_batch(batch)
    state = State.initial(batch, budget=2)
    policy = _policy()

    out = policy(
        context=context,
        state=state,
        features=features,
        frontier_builder=FrontierBuilder.from_batch(batch),
    )

    assert torch.allclose(out.stop_log_prob, torch.nn.functional.logsigmoid(out.stop_logit))
    assert torch.allclose(out.continue_log_prob, torch.nn.functional.logsigmoid(-out.stop_logit))
    rows = out.frontier.row_ids
    transition_mass = torch.zeros_like(out.stop_log_prob).scatter_add_(
        0,
        rows,
        out.transition_log_prob.exp(),
    )
    assert torch.allclose(out.stop_log_prob.exp() + transition_mass, torch.ones_like(transition_mass), atol=1e-5)


def test_policy_frontier_scoring_chunk_size_preserves_outputs() -> None:
    batch = _batch()
    features = _features()
    context = FlowContext.from_batch(batch)
    state = State.initial(batch, budget=2)
    policy = _policy()

    policy.frontier_score_chunk_size = 1024
    full = policy(
        context=context,
        state=state,
        features=features,
        frontier_builder=FrontierBuilder.from_batch(batch),
    )
    policy.frontier_score_chunk_size = 1
    chunked = policy(
        context=context,
        state=state,
        features=features,
        frontier_builder=FrontierBuilder.from_batch(batch),
    )

    assert torch.allclose(chunked.edge_logits, full.edge_logits)
    assert torch.allclose(chunked.edge_log_prob, full.edge_log_prob)
    assert torch.allclose(chunked.stop_log_prob, full.stop_log_prob)
    assert torch.allclose(chunked.transition_log_prob, full.transition_log_prob)


def test_rollout_context_rejects_attached_features() -> None:
    batch = _batch()
    features = _features()
    features.edge_h.requires_grad_()
    flow_context = FlowContext.from_batch(batch)

    with pytest.raises(ValueError, match="features must be detached"):
        RolloutContext(
            flow_context=flow_context,
            features=features,
            frontier_builder=FrontierBuilder.from_flow_context(flow_context),
        )


def test_terminal_write_uses_policy_stop_log_prob() -> None:
    trace = RolloutTrace(R=3, T=1, device=torch.device("cpu"))
    active_rows = torch.tensor([0, 1, 2], dtype=torch.long)
    trace.write_state(t=0, rows=active_rows)
    alive = torch.ones(3, dtype=torch.bool)

    policy_out = PolicyOutput(
        frontier=Frontier(
            row_ids=torch.empty(0, dtype=torch.long),
            edge_ids=torch.empty(0, dtype=torch.long),
            edge_direction=torch.empty(0, dtype=torch.long),
        ),
        stop_logit=torch.zeros(3),
        stop_log_prob=torch.tensor([1.5, 2.5, 3.5]),
        continue_log_prob=torch.zeros(3),
        edge_logits=torch.empty(0),
        edge_log_prob=torch.empty(0),
        transition_log_prob=torch.empty(0),
    )
    action = SampledAction(
        stop_rows=torch.tensor([2, 0], dtype=torch.long),
        stop_logprob=torch.zeros(2),
        forced_stop=torch.tensor([False, True]),
        expand_rows=torch.empty(0, dtype=torch.long),
        expand_edge_ids=torch.empty(0, dtype=torch.long),
        expand_logprob=torch.empty(0),
    )

    RolloutEngine._write_terminal_rows(
        t=0,
        active_rows=active_rows,
        policy_out=policy_out,
        action=action,
        trace=trace,
        alive=alive,
    )

    assert torch.allclose(trace.terminal_stop_log_prob, torch.tensor([1.5, 0.0, 3.5]))
    assert trace.forced_stop_mask[:, 0].tolist() == [True, False, False]
    assert alive.tolist() == [False, True, False]


def test_probability_db_loss_matches_manual_residual() -> None:
    loss = ProbabilityDBLoss()
    output = loss(
        parent_log_reward=torch.tensor([1.0, 0.5]),
        child_log_reward=torch.tensor([1.2, 0.3]),
        log_backward_prob=torch.tensor([-0.7, -0.2]),
        parent_stop_log_prob=torch.tensor([-0.4, -0.6]),
        parent_continue_log_prob=torch.tensor([-0.2, -0.3]),
        parent_edge_log_prob=torch.tensor([-0.1, -0.1]),
        child_stop_log_prob=torch.tensor([-0.1, -0.5]),
    )
    residual = torch.tensor(
        [
            1.2 - 0.7 - 0.4 + 0.2 + 0.1 - 1.0 + 0.1,
            0.3 - 0.2 - 0.6 + 0.3 + 0.1 - 0.5 + 0.5,
        ]
    )
    assert torch.allclose(output.per_unit_loss, residual.square())
    assert torch.allclose(output.loss, residual.square().mean())


def test_reward_uses_answer_support_and_edge_penalty() -> None:
    batch = _batch()
    reward = EvidenceLogReward(alpha=4.0, lambda_=0.5, eta=2.0)
    context = reward.prepare_context(batch, expand_budget=2)
    state = State.initial(batch, budget=2)

    no_answer = reward(state=state, context=context)
    assert torch.allclose(no_answer.log_reward, torch.tensor([-2.0]))
    assert torch.allclose(no_answer.answer_recall, torch.tensor([0.0]))
    assert no_answer.no_answer.tolist() == [True]

    child = state.clone()
    child.apply_edges_(
        edge_index=batch.edge_index,
        rows=torch.tensor([0]),
        edge_ids=torch.tensor([2]),
    )
    hit = reward(state=child, context=context)
    assert torch.allclose(hit.answer_recall, torch.tensor([1.0]))
    assert hit.no_answer.tolist() == [False]
    assert torch.allclose(hit.log_reward, torch.tensor([3.5]))


def test_weaver_module_direct_db_path_requires_transition_actions_in_frontier() -> None:
    batch = _batch()
    features = _features()
    flow_context = FlowContext.from_batch(batch)
    frontier_builder = FrontierBuilder.from_batch(batch)
    reward = EvidenceLogReward()
    reward_context = reward.prepare_context(batch, expand_budget=2)
    policy = _policy()

    parent = State.initial(batch, budget=2)
    child = parent.clone()
    child.apply_edges_(
        edge_index=batch.edge_index,
        rows=torch.tensor([0]),
        edge_ids=torch.tensor([2]),
    )
    transitions = TransitionBatch(
        parent_state=parent,
        child_state=child,
        action_edge_ids=torch.tensor([2], dtype=torch.long),
        log_backward_prob=torch.tensor([0.0]),
    )
    module = _module(policy=policy, reward_model=reward)
    output = module._forward_transitions(
        transitions=transitions,
        features=features,
        rollout_context=RolloutContext(
            flow_context=flow_context,
            features=features.detach_to(device=features.edge_h.device) if hasattr(features, "detach_to") else _detached_rollout_context_features(features),
            frontier_builder=frontier_builder,
        ),
        reward_context=reward_context,
    )
    assert output.loss.requires_grad
    assert output.num_states == 1


def test_invalid_disconnected_selected_edges_are_rejected() -> None:
    batch = _batch()
    state = State.initial(batch, budget=2)
    state.edge_mask[0, 1] = True
    state.rebuild_node_mask_(edge_index=FrontierBuilder.from_batch(batch).edge_index)

    with pytest.raises(AssertionError, match="recursive frontier expansion"):
        assert_anchor_connected_state(
            state=state,
            edge_index=FrontierBuilder.from_batch(batch).edge_index,
        )


def _policy() -> Policy:
    hidden_dim = 4
    return Policy(
        state_encoder=StateEncoder(hidden_dim=hidden_dim, max_budget=2),
        frontier_encoder=FrontierEncoder(hidden_dim=hidden_dim),
        edge_scorer=EdgeActionScorer(hidden_dim=hidden_dim, semantic_prior_scale=1.0),
    )


def _module(
    *,
    policy: Policy,
    reward_model: EvidenceLogReward,
) -> WeaverModule:
    return WeaverModule(
        feature_encoder=_UnusedFeatureEncoder(),
        policy=policy,
        reward_model=reward_model,
        runner=SimpleNamespace(progress_fn=None),
        optimization=OptimizationRuntimeConfig(
            optimizer=OptimizerRuntimeConfig(
                type="adamw",
                lr=1.0e-4,
                weight_decay=0.0,
                betas=(0.9, 0.999),
                no_decay_on_bias_and_norm=True,
            ),
            scheduler=None,
        ),
        evaluation=EvalRuntimeConfig(
            best_of_k_values=(1, 2),
            utility_k=2,
            utility_lambda=0.02,
            exclude_anchors_from_retrieved=True,
            use_reachable_targets=True,
        ),
    )


def _detached_rollout_context_features(features: FeatureBank) -> FeatureBank:
    return FeatureBank(
        node_h=features.node_h.detach(),
        edge_h=features.edge_h.detach(),
        query_h=features.query_h.detach(),
        node_is_non_text=features.node_is_non_text.detach(),
        node_sem_h=features.node_sem_h.detach(),
        rel_sem_h=features.rel_sem_h.detach(),
        query_sem_h=features.query_sem_h.detach(),
        rel_h=features.rel_h.detach(),
    )


class _UnusedFeatureEncoder(nn.Module):
    def forward(self, batch) -> FeatureBank:
        del batch
        raise RuntimeError("This stub should not be called in the direct transition test.")


def _features() -> FeatureBank:
    return FeatureBank(
        node_h=torch.eye(3, 4, dtype=torch.float32),
        edge_h=torch.eye(3, 4, dtype=torch.float32),
        query_h=torch.ones((1, 4), dtype=torch.float32),
        node_is_non_text=torch.zeros(3, dtype=torch.bool),
        node_sem_h=torch.eye(3, dtype=torch.float32),
        rel_sem_h=torch.eye(3, dtype=torch.float32),
        query_sem_h=torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
        rel_h=torch.eye(3, 4, dtype=torch.float32),
    )


def _batch():
    data = _build_retrieval_data(
        raw={
            SampleFields.EDGE_INDEX: torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long),
            SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.NUM_NODES: torch.tensor(3, dtype=torch.long),
            SampleFields.NUM_EDGES: torch.tensor(3, dtype=torch.long),
            SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
            SampleFields.TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 1], dtype=torch.long),
            SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, -1], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCE: torch.tensor([1, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([1, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.ones(3, dtype=torch.float32),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([2], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.ones(1, dtype=torch.float32),
        },
        sample_id="canonical",
        question_emb=torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])
