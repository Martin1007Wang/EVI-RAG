from __future__ import annotations

from types import SimpleNamespace

import torch

from src.eval.rollout import auc_score
from src.eval.rollout import evaluate_rollout_samples
from src.eval.rollout import per_graph_auc
from src.eval.rollout import per_graph_spearman
from src.weaver.context import GraphContext
from src.weaver.rollout.result import RolloutResult
from src.weaver.state import State


class TinyBatch:
    def __init__(self) -> None:
        self.edge_index = torch.tensor(
            [
                [0, 0, 0, 4, 4],
                [1, 2, 3, 5, 6],
            ],
            dtype=torch.long,
        )
        self.batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        self.edge_batch = self.batch.index_select(0, self.edge_index[0])
        self.anchor_node_ids = torch.tensor([0, 4], dtype=torch.long)
        self.target_node_ids = torch.tensor([1, 2, 5], dtype=torch.long)
        self.reachable_target_node_ids = self.target_node_ids
        self.node_target_shortest_path_edge_mask_flat = torch.empty(0, dtype=torch.bool)
        self.num_graphs = 2

    @property
    def num_nodes_total(self) -> int:
        return int(self.batch.numel())

    @property
    def num_edges_total(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def num_graphs_total(self) -> int:
        return int(self.num_graphs)


class FakeReward:
    def __call__(self, *, state, graph_context, target_context):
        del graph_context
        answer_count = (state.active_node_mask & target_context.target_mask.view(1, -1)).sum(dim=1).float()
        edge_count = state.selected_edge_mask.sum(dim=1).float()
        return SimpleNamespace(log_reward=answer_count - 0.1 * edge_count)


class FakePolicy:
    def __call__(self, *, features, state, context, frontier):
        del features, context, frontier
        score = torch.zeros(state.num_rows, dtype=torch.float32, device=state.device)
        score += state.selected_edge_mask[:, 0].float() * 5.0
        score += state.selected_edge_mask[:, 1].float() * 4.0
        score += state.selected_edge_mask[:, 3].float() * 5.0
        score -= state.selected_edge_mask[:, 2].float() * 5.0
        score -= state.selected_edge_mask[:, 4].float() * 5.0
        return SimpleNamespace(
            state_log_flow=score,
            terminal_log_flow=score,
        )


def rollout(edge_rows: list[list[int]], logprob_rows: list[list[float]]) -> RolloutResult:
    selected = torch.tensor(edge_rows, dtype=torch.long)
    logprob = torch.tensor(logprob_rows, dtype=torch.float32)
    terminal_step = torch.tensor([len(row) - 1 for row in edge_rows], dtype=torch.long)
    batch = TinyBatch()
    context = GraphContext.from_batch(batch)
    edge_mask = torch.zeros((len(edge_rows), batch.num_edges_total), dtype=torch.bool)
    for row_id, row in enumerate(edge_rows):
        for edge_id in row:
            if edge_id >= 0:
                edge_mask[row_id, int(edge_id)] = True
    terminal_state = State.from_selected_edges(
        graph=context,
        graph_ids=torch.arange(len(edge_rows), dtype=torch.long),
        selected_edge_mask=edge_mask,
        expand_budget=2,
    )
    return RolloutResult(
        source_graph_id=torch.arange(len(edge_rows), dtype=torch.long),
        selected_edge_ids=selected,
        policy_action_log_prob=logprob,
        behavior_action_log_prob=logprob,
        terminal_step=terminal_step,
        stop_reason=torch.full((len(edge_rows),), RolloutResult.POLICY_STOP, dtype=torch.long),
        expand_budget=2,
        terminal_state=terminal_state,
    )


def build_eval_inputs():
    batch = TinyBatch()
    context = GraphContext.from_batch(batch)
    target = SimpleNamespace(
        target_mask=torch.tensor([False, True, True, False, False, True, False]),
        target_count_by_graph=torch.tensor([2, 1], dtype=torch.long),
    )
    rollouts = [
        rollout(
            edge_rows=[[2, -1, -1], [4, -1, -1]],
            logprob_rows=[[-0.01, -0.01, 0.0], [-0.01, -0.01, 0.0]],
        ),
        rollout(
            edge_rows=[[0, -1, -1], [3, -1, -1]],
            logprob_rows=[[-5.0, -5.0, 0.0], [-5.0, -5.0, 0.0]],
        ),
        rollout(
            edge_rows=[[1, -1, -1], [3, -1, -1]],
            logprob_rows=[[-6.0, -6.0, 0.0], [-6.0, -6.0, 0.0]],
        ),
    ]
    return batch, context, target, rollouts


def test_rollout_metric_contract_and_selector_semantics() -> None:
    batch, context, target, rollouts = build_eval_inputs()
    metrics = evaluate_rollout_samples(
        rollout_samples=rollouts,
        batch=batch,
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
        k_windows=[1, 2, 3],
        enable_calibration_metrics=True,
        enable_terminal_diagnostics=True,
        context=context,
        features=SimpleNamespace(),
        reward_model=FakeReward(),
        target_context=target,
        policy=FakePolicy(),
    )

    assert "candidate_reward_best@2/recall" in metrics
    assert "selector_reward@2/recall" not in metrics
    assert "model_best@2/recall" not in metrics
    assert "sample@1/recall" not in metrics
    assert not any(key.startswith("expected/") for key in metrics)

    assert metrics["selector_traj_prob@2/recall"] == 0.0
    assert metrics["selector_terminal_flow@2/recall"] == 0.75
    assert metrics["selector_terminal_flow@2/f1"] > 0.0
    assert metrics["candidate_union@3/recall"] > metrics["candidate_oracle_best@3/recall"]
    assert metrics["candidate_union@3/edges"] == 2.5
    assert "candidate_union@3/precision" not in metrics
    assert "selector_terminal_flow@2/edges" not in metrics
    assert not any(key.startswith("diversity@") for key in metrics)


def test_same_terminal_graph_counts_once_but_trajectory_scores_can_differ() -> None:
    batch = TinyBatch()
    context = GraphContext.from_batch(batch)
    target = SimpleNamespace(
        target_mask=torch.tensor([False, True, True, False, False, True, False]),
        target_count_by_graph=torch.tensor([2, 1], dtype=torch.long),
    )
    rollouts = [
        rollout([[0, 1, -1], [3, -1, -1]], [[-0.1, -0.2, -0.3], [-0.1, -0.2, 0.0]]),
        rollout([[1, 0, -1], [3, -1, -1]], [[-2.0, -2.0, -2.0], [-2.0, -2.0, 0.0]]),
    ]
    metrics = evaluate_rollout_samples(
        rollout_samples=rollouts,
        batch=batch,
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
        k_windows=[2],
        enable_calibration_metrics=False,
        enable_terminal_diagnostics=False,
        context=context,
        features=SimpleNamespace(),
        reward_model=FakeReward(),
        target_context=target,
        policy=FakePolicy(),
    )

    assert metrics["selector_traj_prob@2/recall"] == metrics["selector_terminal_flow@2/recall"]
    assert not any(key.startswith("diversity@") for key in metrics)


def test_calibration_is_per_graph_and_masks_undefined_auc() -> None:
    scores = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    rewards = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
        ]
    )
    valid = torch.tensor([True, True])

    spearman, valid_rate = per_graph_spearman(scores, rewards, valid)
    assert spearman == 0.0
    assert valid_rate == 1.0

    labels = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, False],
        ]
    )
    auc, auc_valid_rate = per_graph_auc(scores, labels, valid)
    assert auc == auc_score([1.0, 2.0, 3.0], [True, False, False])
    assert auc_valid_rate == 0.5

    all_positive_auc, all_positive_rate = per_graph_auc(scores[:, :1], torch.ones((3, 1), dtype=torch.bool), valid[:1])
    assert all_positive_auc == 0.5
    assert all_positive_rate == 0.0
