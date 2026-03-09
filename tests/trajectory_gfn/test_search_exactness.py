from __future__ import annotations

import math

import pytest
import torch

from src.models.configs.trajectory_gfn import HorizonConfig, TrajectoryInferenceConfig
from src.models.trajectory_gfn.analyzer import AnswerMassAnalysis
from src.models.trajectory_gfn.policy import (
    ForwardActionDistribution,
    StartDistribution,
    TrajectoryPolicy,
)
from src.models.trajectory_gfn.search import MassAdaptiveTrajectorySearch

from .conftest import make_batch_from_graph


class _ConstantAnalyzer:
    def __init__(self) -> None:
        self.terminal_mass = torch.tensor([0.0, 0.06, 0.4, 0.54], dtype=torch.float32)

    def analyze(self, *, batch, policy, context) -> AnswerMassAnalysis:  # noqa: ANN001
        del policy, context
        return AnswerMassAnalysis(
            terminal_mass=self.terminal_mass,
            answer_entity_ids=torch.tensor([101, 102, 103], dtype=torch.long),
            answer_probs=torch.tensor([0.06, 0.4, 0.54], dtype=torch.float32),
            gold_total_mass=0.54,
        )


class _ManualPolicy:
    def __init__(self) -> None:
        self.start_nodes = torch.tensor([0], dtype=torch.long)
        self.start_log_probs = torch.tensor([0.0], dtype=torch.float32)
        self.transitions = {
            (0, 0): {
                "stop": 0.0,
                "moves": [(0, 1, 0.6), (1, 2, 0.4)],
            },
            (1, 1): {
                "stop": 0.1,
                "moves": [(2, 3, 0.9)],
            },
            (2, 1): {
                "stop": 1.0,
                "moves": [],
            },
            (3, 2): {
                "stop": 1.0,
                "moves": [],
            },
        }

    def compute_start_distribution(self, context) -> StartDistribution:  # noqa: ANN001
        del context
        return StartDistribution(
            candidate_nodes_abs=self.start_nodes,
            candidate_graph_ids=torch.zeros((1,), dtype=torch.long),
            log_probs=self.start_log_probs,
        )

    def compute_forward_distribution(self, context, state) -> ForwardActionDistribution:  # noqa: ANN001
        del context
        current_node = int(state.current_node.view(-1)[0].item())
        num_moves = int(state.num_moves.view(-1)[0].item())
        spec = self.transitions[(current_node, num_moves)]
        moves = spec["moves"]
        if moves:
            edge_ids = torch.tensor(
                [edge_id for edge_id, _, _ in moves], dtype=torch.long
            )
            target_nodes = torch.tensor([dst for _, dst, _ in moves], dtype=torch.long)
            edge_logits = torch.tensor(
                [math.log(prob) for _, _, prob in moves], dtype=torch.float32
            )
            edge_agent_batch = torch.zeros((len(moves),), dtype=torch.long)
        else:
            edge_ids = torch.empty((0,), dtype=torch.long)
            target_nodes = torch.empty((0,), dtype=torch.long)
            edge_logits = torch.empty((0,), dtype=torch.float32)
            edge_agent_batch = torch.empty((0,), dtype=torch.long)
        stop_prob = float(spec["stop"])
        stop_logit = float("-inf") if stop_prob <= 0.0 else math.log(stop_prob)
        return ForwardActionDistribution(
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            stop_logits=torch.tensor([[stop_logit]], dtype=torch.float32),
            edge_ids=edge_ids,
            target_nodes=target_nodes,
            out_degrees=torch.tensor([[len(moves)]], dtype=torch.long),
            state_log_flows=torch.zeros((1, 1), dtype=torch.float32),
            invalid_rows=torch.zeros((1, 1), dtype=torch.bool),
        )

    @staticmethod
    def compute_forward_log_probs(distribution: ForwardActionDistribution):
        return TrajectoryPolicy.compute_forward_log_probs(distribution)


def _make_search_fixture() -> tuple[object, _ManualPolicy, _ConstantAnalyzer]:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([3], dtype=torch.long),
        answer_entity_ids=torch.tensor([103], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="manual-search",
    )
    policy = _ManualPolicy()
    analyzer = _ConstantAnalyzer()
    return batch, policy, analyzer


def test_search_exact_top_order() -> None:
    batch, policy, analyzer = _make_search_fixture()
    search = MassAdaptiveTrajectorySearch(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        inference_cfg=TrajectoryInferenceConfig(
            answer_mass_threshold=1.0,
            support_mass_threshold=1.0,
            max_expansions=32,
            max_frontier_size=32,
        ),
        analyzer=analyzer,
    )
    result = search.generate_window(batch=batch, policy=policy, context=None)
    probs = [round(traj.prob, 6) for traj in result.trajectories]
    assert probs == [0.54, 0.4, 0.06]


def test_window_is_minimal_probability_prefix() -> None:
    batch, policy, analyzer = _make_search_fixture()
    search = MassAdaptiveTrajectorySearch(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        inference_cfg=TrajectoryInferenceConfig(
            answer_mass_threshold=0.55,
            support_mass_threshold=1.0,
            max_expansions=32,
            max_frontier_size=32,
        ),
        analyzer=analyzer,
    )
    result = search.generate_window(batch=batch, policy=policy, context=None)
    assert result.window_size == 2
    assert result.trajectories[-1].cumulative_mass >= 0.55
    assert result.trajectories[-2].cumulative_mass < 0.55
    assert abs(result.covered_mass - 0.94) < 1.0e-6


def test_search_raises_on_truncation_guard() -> None:
    batch, policy, analyzer = _make_search_fixture()
    search = MassAdaptiveTrajectorySearch(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        inference_cfg=TrajectoryInferenceConfig(
            answer_mass_threshold=0.9,
            support_mass_threshold=1.0,
            max_expansions=32,
            max_frontier_size=1,
        ),
        analyzer=analyzer,
    )
    with pytest.raises(RuntimeError, match="max_frontier_size"):
        search.generate_window(batch=batch, policy=policy, context=None)
