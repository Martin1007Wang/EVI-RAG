from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.objectives.latent_prefix import (
    joint_prefix_score,
    latent_prefix_targets,
    recompute_trajectory_log_prob,
)
from src.weaver.policy.latent_prefix import EdgeOnlyPolicyOutput
from src.weaver.rollout.latent_prefix import LatentPrefixRolloutEngine


class SingleEdgePolicy:
    def __call__(self, *, features, state, context, frontier):
        del features, context
        logits = torch.zeros(frontier.edge_ids.numel(), dtype=torch.float32, device=state.device)
        if frontier.edge_ids.numel() == 0:
            log_prob = logits
        else:
            log_prob = torch.zeros_like(logits)
        return EdgeOnlyPolicyOutput(
            frontier_row_ids=frontier.row_ids,
            frontier_edge_ids=frontier.edge_ids,
            edge_logits=logits,
            edge_log_prob=log_prob,
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )


def tiny_context() -> GraphContext:
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False]),
        adjacency=build_directed_adjacency_index(edge_index=edge_index, num_nodes=2),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )


def test_fixed_horizon_rollout_emits_all_prefixes_and_dead_end_padding() -> None:
    context = tiny_context()
    rollout = LatentPrefixRolloutEngine(expand_budget=3).sample_rollouts(
        policy=SingleEdgePolicy(),
        context=context,
        features=SimpleNamespace(),
        rollouts_per_graph=1,
    )

    assert rollout.prefixes.num_items == 4
    assert rollout.num_trajectories == 1
    assert rollout.expansions.num_items == 1
    assert rollout.dead_end.tolist() == [True]
    assert rollout.trajectory_log_prob.tolist() == pytest.approx([0.0])
    assert rollout.prefixes.prefix_step.tolist() == [0, 1, 2, 3]
    assert rollout.prefixes.state.selected_edge_count.tolist() == [0, 1, 1, 1]


def test_edge_only_output_has_no_stop_action_and_normalized_edges() -> None:
    out = EdgeOnlyPolicyOutput(
        frontier_row_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        frontier_edge_ids=torch.tensor([2, 3, 4], dtype=torch.long),
        edge_logits=torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32),
        edge_log_prob=torch.tensor([-0.6931472, -0.6931472, 0.0], dtype=torch.float32),
        num_rows=2,
        num_edges=8,
    )

    assert out.has_frontier().tolist() == [True, True]
    assert out.gather_log_prob(
        row_ids=torch.tensor([0, 1]),
        edge_ids=torch.tensor([2, 4]),
    ).tolist() == pytest.approx([-0.6931472, 0.0])
    with pytest.raises(IndexError):
        out.gather_log_prob(row_ids=torch.tensor([0]), edge_ids=torch.tensor([-1]))


def test_latent_prefix_targets_are_temperature_scaled_segment_distributions() -> None:
    log_reward = torch.tensor([0.0, 2.0, -1.0, 1.0], dtype=torch.float32)
    prefix_step = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    trajectory_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    targets = latent_prefix_targets(
        prefix_log_reward=log_reward,
        prefix_step=prefix_step,
        prefix_trajectory_ids=trajectory_ids,
        num_trajectories=2,
        temperature=0.5,
        length_prior_gamma=0.1,
    )

    expected0 = 0.5 * torch.logsumexp(torch.tensor([0.0, 1.9]) / 0.5, dim=0)
    expected1 = 0.5 * torch.logsumexp(torch.tensor([-1.0, 0.9]) / 0.5, dim=0)
    assert torch.allclose(targets.trajectory_log_reward, torch.stack([expected0, expected1]))
    assert targets.prefix_target_prob[trajectory_ids.eq(0)].sum().item() == pytest.approx(1.0)
    assert targets.prefix_target_prob[trajectory_ids.eq(1)].sum().item() == pytest.approx(1.0)


def test_joint_prefix_score_prevents_selector_only_cross_trajectory_error() -> None:
    trajectory_log_prob = torch.tensor([-20.0, 0.0], dtype=torch.float32)
    prefix_trajectory_ids = torch.tensor([0, 1], dtype=torch.long)
    selector_log_prob = torch.tensor([0.0, -0.7], dtype=torch.float32)

    selector_only_best = int(torch.argmax(selector_log_prob).item())
    joint_best = int(
        torch.argmax(
            joint_prefix_score(
                trajectory_log_prob=trajectory_log_prob,
                selector_log_prob=selector_log_prob,
                prefix_trajectory_ids=prefix_trajectory_ids,
            )
        ).item()
    )

    assert selector_only_best == 0
    assert joint_best == 1


def test_recompute_trajectory_log_prob_sums_only_real_expansions() -> None:
    context = tiny_context()
    rollout = LatentPrefixRolloutEngine(expand_budget=3).sample_rollouts(
        policy=SingleEdgePolicy(),
        context=context,
        features=SimpleNamespace(),
        rollouts_per_graph=1,
    )
    log_prob = torch.tensor([-0.25], dtype=torch.float32)

    assert torch.allclose(
        recompute_trajectory_log_prob(
            expansion_log_prob=log_prob,
            expansions=rollout.expansions,
            num_trajectories=1,
        ),
        torch.tensor([-0.25]),
    )
