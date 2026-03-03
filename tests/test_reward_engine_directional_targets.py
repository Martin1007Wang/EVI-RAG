from __future__ import annotations

import pytest
import torch

from src.models.configs.environment import StopConfig
from src.models.environment import CsrAdjacency, GraphEnvContext
from src.models.reward.reward_engine import DualFlowRewardEngine


def _make_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_ids = torch.arange(2, dtype=torch.long)
    adj_t_fwd = CsrAdjacency(
        crow=torch.tensor([0, 1, 2, 2], dtype=torch.long),
        col=torch.tensor([1, 2], dtype=torch.long),
        edge_ids=edge_ids,
        size=(3, 3),
    )
    adj_t_bwd = CsrAdjacency(
        crow=torch.tensor([0, 0, 1, 2], dtype=torch.long),
        col=torch.tensor([0, 1], dtype=torch.long),
        edge_ids=edge_ids,
        size=(3, 3),
    )
    return GraphEnvContext(
        num_graphs=1,
        num_nodes_total=3,
        node_ptr=torch.tensor([0, 3], dtype=torch.long),
        edge_index=edge_index,
        edge_relations=torch.tensor([0, 0], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 0], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        node_batch=torch.tensor([0, 0, 0], dtype=torch.long),
        adj_t_fwd=adj_t_fwd,
        adj_t_bwd=adj_t_bwd,
        node_embeddings=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # q node
                [1.0, 0.0, 0.0, 0.0],  # non-answer node
                [0.0, 1.0, 0.0, 0.0],  # answer node
            ],
            dtype=torch.float32,
        ),
        node_tokens=torch.zeros((3, 4), dtype=torch.float32),
        relation_tokens=torch.zeros((1, 4), dtype=torch.float32),
        question_emb=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([2], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_global_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        dummy_mask=torch.tensor([False]),
        sample_ids=["sample_0"],
    )


def test_reward_engine_directional_target_masks_and_rewards() -> None:
    context = _make_context()
    engine = DualFlowRewardEngine(stop_cfg=StopConfig())
    stop_nodes = torch.tensor([[2, 0]], dtype=torch.long)

    forward_hits = engine.compute_hit_mask(stop_nodes, context)
    backward_hits = engine.compute_hit_mask(
        stop_nodes,
        context,
        target_local_indices=context.q_local_indices,
        target_ptr=context.q_ptr,
        target_field_name="q_local_indices",
    )
    assert torch.equal(forward_hits, torch.tensor([[True, False]]))
    assert torch.equal(backward_hits, torch.tensor([[False, True]]))

    rewards_forward, _ = engine.compute_rewards(
        stop_nodes_abs=stop_nodes,
        context=context,
        reward_beta=1.0,
    )
    rewards_backward, _ = engine.compute_rewards(
        stop_nodes_abs=stop_nodes,
        context=context,
        reward_beta=1.0,
        target_local_indices=context.q_local_indices,
        target_ptr=context.q_ptr,
        target_field_name="q_local_indices",
    )
    assert float(rewards_forward[0, 0].item()) > float(rewards_forward[0, 1].item())
    assert float(rewards_backward[0, 1].item()) > float(rewards_backward[0, 0].item())


def test_reward_engine_miss_nodes_always_receive_epsilon() -> None:
    context = _make_context()
    engine = DualFlowRewardEngine(
        stop_cfg=StopConfig(
            reward_base=1.0,
            reward_epsilon=1.0e-6,
            semantic_miss_scale=0.5,
            degree_penalty_alpha=0.0,
        )
    )
    stop_nodes = torch.tensor([[2, 1, 0]], dtype=torch.long)

    rewards_forward, _ = engine.compute_rewards(
        stop_nodes_abs=stop_nodes,
        context=context,
        reward_beta=1.0,
    )
    assert float(rewards_forward[0, 0].item()) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert float(rewards_forward[0, 1].item()) == pytest.approx(
        1.0e-6, rel=1e-6, abs=1e-6
    )
    assert float(rewards_forward[0, 2].item()) == pytest.approx(
        1.0e-6, rel=1e-6, abs=1e-6
    )

    rewards_backward, _ = engine.compute_rewards(
        stop_nodes_abs=stop_nodes,
        context=context,
        reward_beta=1.0,
        target_local_indices=context.q_local_indices,
        target_ptr=context.q_ptr,
        target_field_name="q_local_indices",
    )
    assert float(rewards_backward[0, 0].item()) == pytest.approx(
        1.0e-6, rel=1e-6, abs=1e-6
    )
    assert float(rewards_backward[0, 1].item()) == pytest.approx(
        1.0e-6, rel=1e-6, abs=1e-6
    )
    assert float(rewards_backward[0, 2].item()) == pytest.approx(
        1.0, rel=1e-6, abs=1e-6
    )


def test_reward_engine_terminal_done_mask_forces_timeout_to_epsilon() -> None:
    context = _make_context()
    engine = DualFlowRewardEngine(
        stop_cfg=StopConfig(
            reward_base=1.0, reward_epsilon=1.0e-6, semantic_miss_scale=0.5
        )
    )
    stop_nodes = torch.tensor([[2, 2]], dtype=torch.long)
    terminal_done_mask = torch.tensor([[True, False]], dtype=torch.bool)
    rewards, _ = engine.compute_rewards(
        stop_nodes_abs=stop_nodes,
        context=context,
        reward_beta=1.0,
        terminal_done_mask=terminal_done_mask,
    )
    hits = engine.compute_hit_mask(
        stop_nodes_abs=stop_nodes,
        context=context,
        terminal_done_mask=terminal_done_mask,
    )
    assert float(rewards[0, 0].item()) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert float(rewards[0, 1].item()) == pytest.approx(1.0e-6, rel=1e-6, abs=1e-6)
    assert bool(hits[0, 0].item()) is True
    assert bool(hits[0, 1].item()) is False
