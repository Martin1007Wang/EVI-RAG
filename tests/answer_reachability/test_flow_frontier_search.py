from __future__ import annotations

import math

import pytest
import torch

from src.graph_runtime import build_graph_batch
from src.graph_runtime.batch import TrajectoryBatch
from src.metrics.search_backends import (
    FlowFrontierBackend,
    run_flow_frontier_search,
)
from src.models.configs import HorizonConfig, SearchEvalConfig
from src.models.gflownet import (
    BaseSearchPolicy,
    ForwardActionDistribution,
    PreparedSearchBatch,
    StartDistribution,
)

from .conftest import make_batch_from_graph


class _ManualFlowFrontierPolicy:
    def __init__(self) -> None:
        self.start_nodes = torch.tensor([0], dtype=torch.long)
        self.start_log_flows = torch.tensor([0.0], dtype=torch.float32)
        self.start_log_probs = torch.tensor([0.0], dtype=torch.float32)
        self.graph_log_z = torch.tensor([0.0], dtype=torch.float32)
        self.transitions = {
            (0, 0): [(0, 1, 0.6, False), (1, 2, 0.4, False)],
            (1, 1): [(2, 3, 1.0, False)],
            (2, 1): [],
            (3, 2): [(-1, 3, 1.0, True)],
        }
        self.log_state_flows = {
            (0, 0): 0.0,
            (1, 1): math.log(0.6),
            (2, 1): math.log(0.4),
            (3, 2): math.log(0.6),
        }

    def prepare_batch(self, batch: TrajectoryBatch) -> PreparedSearchBatch:
        del batch
        raise NotImplementedError

    def build_start_control_states(
        self,
        prepared_batch: PreparedSearchBatch,
        start_nodes: torch.Tensor,
    ) -> torch.Tensor:
        del prepared_batch
        return torch.zeros((*start_nodes.shape, 1), dtype=torch.float32)

    def compute_next_control_states(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        control_states: torch.Tensor,
        next_nodes: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        del prepared_batch, next_nodes, relation_ids
        return torch.zeros_like(control_states)

    def compute_start_distribution(
        self, prepared_batch: PreparedSearchBatch
    ) -> StartDistribution:
        del prepared_batch
        return StartDistribution(
            candidate_nodes_abs=self.start_nodes,
            candidate_graph_ids=torch.zeros((1,), dtype=torch.long),
            log_flows=self.start_log_flows,
            log_probs=self.start_log_probs,
            graph_log_z=self.graph_log_z,
            action_logits=self.start_log_probs,
        )

    def compute_root_action_distribution(
        self, prepared_batch: PreparedSearchBatch
    ) -> StartDistribution:
        return self.compute_start_distribution(prepared_batch)

    def compute_forward_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
        state,
        *,
        required_edge_ids: torch.Tensor | None = None,
    ) -> ForwardActionDistribution:
        del prepared_batch, required_edge_ids
        edge_logits: list[float] = []
        edge_agent_batch: list[int] = []
        edge_ids: list[int] = []
        target_nodes: list[int] = []
        is_submit: list[bool] = []
        out_degrees: list[int] = []
        flat_nodes = state.current_nodes.view(-1)
        flat_steps = state.num_steps.view(-1)
        for agent_idx, (current_node, num_moves) in enumerate(
            zip(flat_nodes.tolist(), flat_steps.tolist())
        ):
            moves = self.transitions[(int(current_node), int(num_moves))]
            out_degrees.append(len(moves))
            for edge_id, dst, prob, submit in moves:
                edge_logits.append(math.log(prob))
                edge_agent_batch.append(agent_idx)
                edge_ids.append(edge_id)
                target_nodes.append(dst)
                is_submit.append(submit)
        if not edge_ids:
            return ForwardActionDistribution(
                edge_logits=torch.empty((0,), dtype=torch.float32),
                edge_agent_batch=torch.empty((0,), dtype=torch.long),
                edge_ids=torch.empty((0,), dtype=torch.long),
                target_nodes=torch.empty((0,), dtype=torch.long),
                out_degrees=torch.zeros_like(state.current_nodes, dtype=torch.long),
                is_stop_action=torch.empty((0,), dtype=torch.bool),
            )
        return ForwardActionDistribution(
            edge_logits=torch.tensor(edge_logits, dtype=torch.float32),
            edge_agent_batch=torch.tensor(edge_agent_batch, dtype=torch.long),
            edge_ids=torch.tensor(edge_ids, dtype=torch.long),
            target_nodes=torch.tensor(target_nodes, dtype=torch.long),
            out_degrees=torch.tensor(out_degrees, dtype=torch.long).view_as(
                state.current_nodes
            ),
            is_stop_action=torch.tensor(is_submit, dtype=torch.bool),
        )

    def compute_log_state_scores(
        self,
        prepared_batch: PreparedSearchBatch,
        state,
    ) -> torch.Tensor:
        del prepared_batch
        scores = [
            self.log_state_flows[(int(node), int(step))]
            for node, step in zip(
                state.current_nodes.view(-1).tolist(),
                state.num_steps.view(-1).tolist(),
            )
        ]
        return torch.tensor(scores, dtype=torch.float32).view_as(state.current_nodes)

    @staticmethod
    def compute_move_log_probs(distribution: ForwardActionDistribution):
        return BaseSearchPolicy.compute_move_log_probs(distribution)


def _make_manual_fixture() -> tuple[
    TrajectoryBatch, PreparedSearchBatch, _ManualFlowFrontierPolicy
]:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([3], dtype=torch.long),
        answer_entity_ids=torch.tensor([103], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="manual-flow-frontier",
    )
    topology, observation = build_graph_batch(batch)
    prepared_batch = PreparedSearchBatch(
        topology=topology,
        observation=observation,
        node_tokens=torch.empty((0, 0), dtype=torch.float32),
        relation_tokens=torch.empty((0, 0), dtype=torch.float32),
        question_tokens=torch.empty((0, 0), dtype=torch.float32),
        question_context_tokens=torch.empty((0, 0, 0), dtype=torch.float32),
        question_context_mask=torch.empty((0, 0), dtype=torch.bool),
    )
    return batch, prepared_batch, _ManualFlowFrontierPolicy()


def test_flow_frontier_support_search_returns_exact_window() -> None:
    batch, prepared_batch, policy = _make_manual_fixture()
    search = FlowFrontierBackend(
        max_steps=2,
        eval_cfg=SearchEvalConfig(
            support_search_method="flow_frontier",
            flow_prune_epsilon=0.0,
            answer_mass_threshold=0.55,
            support_mass_threshold=1.0,
            max_expansions=32,
            max_frontier_size=32,
        ),
    )

    result = search.evaluate_graph(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
        metrics_profile="full",
        include_answer_support=True,
    )

    assert result.inference_mode == "flow_frontier"
    assert result.answer_mass_reference == "flow_frontier"
    assert result.support_mass_reference == "flow_frontier"
    assert result.stop_reason == "flow_frontier_exhausted"
    assert result.coverage_certified is True
    assert result.window_size == 1
    assert [edge.edge_id for edge in result.trajectories[0].edges] == [0, 2]
    assert result.covered_mass == pytest.approx(0.6)
    assert result.gold_answer_mass == pytest.approx(0.6)
    assert result.remaining_mass_upper == pytest.approx(0.4)
    assert [record.answer_entity_id for record in result.answer_posterior] == [103, 102]
    assert result.answer_posterior[0].prob == pytest.approx(0.6)
    assert result.answer_posterior[1].prob == pytest.approx(0.4)


def test_flow_frontier_prunes_low_flow_branch_with_global_mass_bound() -> None:
    batch, prepared_batch, policy = _make_manual_fixture()
    start_distribution = policy.compute_start_distribution(prepared_batch)

    summary = run_flow_frontier_search(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
        max_steps=2,
        eval_cfg=SearchEvalConfig(
            support_search_method="flow_frontier",
            flow_prune_epsilon=0.5,
            max_expansions=32,
            max_frontier_size=32,
        ),
        start_distribution=start_distribution,
    )

    assert summary.stop_reason == "flow_frontier_exhausted"
    assert summary.coverage_certified is True
    assert summary.analysis.gold_answer_mass == pytest.approx(0.6)
    assert summary.remaining_mass_upper == pytest.approx(0.4)
    assert [path.edge_ids for path in summary.discovered_paths] == [(0, 2)]
