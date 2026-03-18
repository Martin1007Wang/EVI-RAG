from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from src.graph_runtime import build_graph_batch
from src.graph_runtime.batch import TrajectoryBatch
from src.metrics.answer_reachability import ExactReachabilityAnalysis
from src.metrics.answer_reachability.support_search import ExactSupportSearch
from src.models.configs import SearchEvalConfig, HorizonConfig
from src.models.gflownet import BaseSearchPolicy
from src.models.gflownet import (
    ForwardActionDistribution,
    PreparedSearchBatch,
    StartDistribution,
)

from .conftest import make_batch_from_graph


class _ConstantAnalyzer:
    def __init__(self) -> None:
        self.terminal_mass = torch.tensor([0.0, 0.0, 0.0, 0.6], dtype=torch.float32)

    def analyze(self, *, batch, policy, prepared_batch) -> ExactReachabilityAnalysis:  # noqa: ANN001
        del policy, prepared_batch
        return ExactReachabilityAnalysis(
            terminal_mass=self.terminal_mass,
            answer_entity_ids=torch.tensor([103], dtype=torch.long),
            answer_probs=torch.tensor([0.6], dtype=torch.float32),
            gold_total_mass=0.6,
        )


class _ManualPolicy:
    def __init__(self) -> None:
        self.start_nodes = torch.tensor([0], dtype=torch.long)
        self.start_log_flows = torch.tensor([0.0], dtype=torch.float32)
        self.start_log_probs = torch.tensor([0.0], dtype=torch.float32)
        self.graph_log_z = torch.tensor([0.0], dtype=torch.float32)
        self.transitions = {
            (0, 0): [(0, 1, 0.6), (1, 2, 0.4)],
            (1, 1): [(2, 3, 1.0)],
            (2, 1): [],
            (3, 2): [],
        }

    def compute_start_distribution(self, prepared_batch) -> StartDistribution:  # noqa: ANN001
        del prepared_batch
        return StartDistribution(
            candidate_nodes_abs=self.start_nodes,
            candidate_graph_ids=torch.zeros((1,), dtype=torch.long),
            log_flows=self.start_log_flows,
            log_probs=self.start_log_probs,
            graph_log_z=self.graph_log_z,
        )

    def compute_forward_distribution(
        self, prepared_batch, state
    ) -> ForwardActionDistribution:  # noqa: ANN001
        del prepared_batch
        current_node = int(state.current_nodes.view(-1)[0].item())
        num_moves = int(state.num_steps.view(-1)[0].item())
        moves = self.transitions[(current_node, num_moves)]
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
        return ForwardActionDistribution(
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            edge_ids=edge_ids,
            target_nodes=target_nodes,
            out_degrees=torch.tensor([[len(moves)]], dtype=torch.long),
        )

    @staticmethod
    def compute_move_log_probs(distribution: ForwardActionDistribution):
        return BaseSearchPolicy.compute_move_log_probs(distribution)

    def prepare_batch(self, batch: Any) -> PreparedSearchBatch:
        del batch
        raise NotImplementedError

    def compute_log_state_scores(self, prepared_batch: Any, state: Any) -> torch.Tensor:
        del prepared_batch, state
        raise NotImplementedError


def _make_search_fixture() -> tuple[
    TrajectoryBatch, PreparedSearchBatch, _ManualPolicy, _ConstantAnalyzer
]:
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
    topology, observation = build_graph_batch(batch)
    prepared_batch = PreparedSearchBatch(
        topology=topology,
        observation=observation,
        node_tokens=torch.empty((0, 0), dtype=torch.float32),
        relation_tokens=torch.empty((0, 0), dtype=torch.float32),
        question_tokens=torch.empty((0, 0), dtype=torch.float32),
    )
    policy = _ManualPolicy()
    analyzer = _ConstantAnalyzer()
    return batch, prepared_batch, policy, analyzer


def test_search_exact_top_order() -> None:
    batch, prepared_batch, policy, analyzer = _make_search_fixture()
    search = ExactSupportSearch(
        horizon_cfg=HorizonConfig(max_steps=2),
        eval_cfg=SearchEvalConfig(
            answer_mass_threshold=1.0,
            support_mass_threshold=1.0,
            max_expansions=32,
            max_frontier_size=32,
        ),
        analyzer=analyzer,
    )
    result = search.generate_window(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
    )
    probs = [round(traj.prob, 6) for traj in result.trajectories]
    assert probs == [0.6]


def test_window_is_minimal_probability_prefix() -> None:
    batch, prepared_batch, policy, analyzer = _make_search_fixture()
    search = ExactSupportSearch(
        horizon_cfg=HorizonConfig(max_steps=2),
        eval_cfg=SearchEvalConfig(
            answer_mass_threshold=0.55,
            support_mass_threshold=1.0,
            max_expansions=32,
            max_frontier_size=32,
        ),
        analyzer=analyzer,
    )
    result = search.generate_window(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
    )
    assert result.window_size == 1
    assert result.trajectories[-1].cumulative_mass >= 0.6
    assert abs(result.covered_mass - 0.6) < 1.0e-6


def test_search_raises_on_truncation_guard() -> None:
    batch, prepared_batch, policy, analyzer = _make_search_fixture()
    search = ExactSupportSearch(
        horizon_cfg=HorizonConfig(max_steps=2),
        eval_cfg=SearchEvalConfig(
            answer_mass_threshold=0.9,
            support_mass_threshold=1.0,
            max_expansions=32,
            max_frontier_size=1,
        ),
        analyzer=analyzer,
    )
    with pytest.raises(RuntimeError, match="max_frontier_size"):
        search.generate_window(
            batch=batch,
            policy=policy,
            prepared_batch=prepared_batch,
        )


def test_search_returns_partial_window_when_non_strict() -> None:
    batch, prepared_batch, policy, analyzer = _make_search_fixture()
    search = ExactSupportSearch(
        horizon_cfg=HorizonConfig(max_steps=2),
        eval_cfg=SearchEvalConfig(
            answer_mass_threshold=0.9,
            support_mass_threshold=1.0,
            max_expansions=32,
            max_frontier_size=1,
            strict_search=False,
        ),
        analyzer=analyzer,
    )

    result = search.generate_window(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
    )

    assert result.inference_mode == "exact"
    assert result.coverage_certified is False
    assert result.answer_mass_reference == "exact"
    assert result.support_mass_reference == "partial_exact"
    assert result.stop_reason == "exact_frontier_truncated"
