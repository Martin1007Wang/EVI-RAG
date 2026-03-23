from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from src.graph_runtime import build_graph_batch
from src.graph_runtime.batch import TrajectoryBatch
from src.metrics.answer_metrics import SupportWindowResult
from src.metrics.search_backends import (
    MonteCarloBackend,
    MonteCarloRolloutSummary,
    build_monte_carlo_edge_support_analysis,
)
from src.metrics.runtime_factory import GraphTaskRuntimeFactory
from src.models.configs import (
    BackboneConfig,
    GFlowNetTrainingConfig,
    HeuristicConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
    SearchEvalConfig,
    StateScoreHeadConfig,
)
from src.models.gflownet import (
    BaseSearchPolicy,
    ForwardActionDistribution,
    PreparedSearchBatch,
    StartDistribution,
)
from src.models.gflownet_module import GFlowNetModule

from .conftest import make_batch_from_graph


class _ManualMonteCarloPolicy:
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

    def compute_start_distribution(self, prepared_batch) -> StartDistribution:  # noqa: ANN001
        del prepared_batch
        return StartDistribution(
            candidate_nodes_abs=self.start_nodes,
            candidate_graph_ids=torch.zeros((1,), dtype=torch.long),
            log_flows=self.start_log_flows,
            log_probs=self.start_log_probs,
            graph_log_z=self.graph_log_z,
            action_logits=self.start_log_probs,
        )

    def compute_root_action_distribution(self, prepared_batch) -> StartDistribution:  # noqa: ANN001
        return self.compute_start_distribution(prepared_batch)

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

    def compute_forward_distribution(
        self,
        prepared_batch,
        state,
        *,
        required_edge_ids: torch.Tensor | None = None,
    ) -> ForwardActionDistribution:  # noqa: ANN001
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

    @staticmethod
    def compute_move_log_probs(distribution: ForwardActionDistribution):
        return BaseSearchPolicy.compute_move_log_probs(distribution)

    @staticmethod
    def sample_start_nodes(
        distribution: StartDistribution,
        *,
        num_rollouts: int,
        deterministic: bool = False,
    ):
        return BaseSearchPolicy.sample_start_nodes(
            distribution,
            num_rollouts=num_rollouts,
            deterministic=deterministic,
        )

    def prepare_batch(self, batch: TrajectoryBatch) -> PreparedSearchBatch:
        del batch
        raise NotImplementedError

    def compute_log_state_scores(self, prepared_batch, state):  # noqa: ANN001
        del prepared_batch, state
        raise NotImplementedError


def _make_manual_fixture() -> tuple[
    TrajectoryBatch, PreparedSearchBatch, _ManualMonteCarloPolicy
]:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([3], dtype=torch.long),
        answer_entity_ids=torch.tensor([103], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
        sample_id="manual-monte-carlo",
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
    return batch, prepared_batch, _ManualMonteCarloPolicy()


def _make_rank_only_module() -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(),
        heuristic_cfg=HeuristicConfig(beta=0.0),
        policy_cfg=PolicyConfig(
            backbone=BackboneConfig(
                embedding_dim=8,
                hidden_dim=8,
                gnn_layers=1,
                gnn_dropout=0.0,
                use_adapter=True,
                adapter_dim=4,
                adapter_dropout=0.0,
            ),
            state_score_head=StateScoreHeadConfig(
                hidden_dim=8,
                num_layers=2,
                dropout=0.0,
            ),
        ),
        eval_cfg=SearchEvalConfig(
            metrics_profile="rank_only",
            support_search_method="monte_carlo",
            monte_carlo_rollouts=2048,
            monte_carlo_confidence=0.95,
            answer_mass_threshold=0.9,
            support_mass_threshold=0.9,
            answer_top_ks=(1, 5),
        ),
        optimizer_cfg=OptimizerConfig(),
        scheduler_cfg=SchedulerConfig(),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )


def _make_full_module() -> GFlowNetModule:
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(),
        heuristic_cfg=HeuristicConfig(beta=0.0),
        policy_cfg=PolicyConfig(
            backbone=BackboneConfig(
                embedding_dim=8,
                hidden_dim=8,
                gnn_layers=1,
                gnn_dropout=0.0,
                use_adapter=True,
                adapter_dim=4,
                adapter_dropout=0.0,
            ),
            state_score_head=StateScoreHeadConfig(
                hidden_dim=8,
                num_layers=2,
                dropout=0.0,
            ),
        ),
        eval_cfg=SearchEvalConfig(
            metrics_profile="full",
            support_search_method="monte_carlo",
            monte_carlo_rollouts=64,
            monte_carlo_confidence=0.95,
            answer_mass_threshold=0.9,
            support_mass_threshold=0.9,
            answer_top_ks=(1, 5),
            window_top_ks=(1, 5),
            max_expansions=32,
            max_frontier_size=32,
        ),
        optimizer_cfg=OptimizerConfig(),
        scheduler_cfg=SchedulerConfig(),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )


def test_monte_carlo_support_search_reports_confidence_intervals() -> None:
    torch.manual_seed(0)
    batch, prepared_batch, policy = _make_manual_fixture()
    search = MonteCarloBackend(
        max_steps=2,
        eval_cfg=SearchEvalConfig(
            support_search_method="monte_carlo",
            monte_carlo_rollouts=4096,
            monte_carlo_confidence=0.95,
            answer_mass_threshold=0.55,
            support_mass_threshold=1.0,
        ),
    )

    result = search.evaluate_graph(
        batch=batch,
        policy=policy,
        prepared_batch=prepared_batch,
        metrics_profile="full",
        include_answer_support=True,
    )

    assert result.inference_mode == "monte_carlo"
    assert result.answer_mass_reference == "monte_carlo"
    assert result.support_mass_reference == "monte_carlo"
    assert result.coverage_certified is False
    assert result.window_size == 1
    assert result.trajectories[0].edges[0].edge_id == 0
    assert result.trajectories[0].edges[1].edge_id == 2
    assert result.covered_mass == pytest.approx(0.6, abs=0.05)
    assert result.covered_mass_ci_low is not None
    assert result.covered_mass_ci_high is not None
    assert result.gold_answer_mass_ci_low is not None
    assert result.gold_answer_mass_ci_high is not None
    assert (
        float(result.covered_mass_ci_low)
        <= result.covered_mass
        <= float(result.covered_mass_ci_high)
    )
    assert (
        float(result.gold_answer_mass_ci_low)
        <= 0.6
        <= float(result.gold_answer_mass_ci_high)
    )
    assert result.remaining_mass_upper == pytest.approx(
        max(1.0 - float(result.covered_mass_ci_low), 0.0)
    )


def test_monte_carlo_edge_support_tracks_successful_edge_presence() -> None:
    batch, _, _ = _make_manual_fixture()
    rollout_summary = MonteCarloRolloutSummary(
        start_nodes=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        terminal_nodes=torch.tensor([3, 3, 2, 2], dtype=torch.long),
        trace_edge_ids=torch.tensor(
            [[0, 2], [0, 2], [1, -1], [1, -1]], dtype=torch.long
        ),
        terminal_num_steps=torch.tensor([2, 2, 1, 1], dtype=torch.long),
        total_rollouts=4,
    )

    edge_support = build_monte_carlo_edge_support_analysis(
        batch=batch,
        rollout_summary=rollout_summary,
    )

    assert edge_support.success_rollout_mass == pytest.approx(0.5)
    assert edge_support.edge_success_mass[0].item() == pytest.approx(0.5)
    assert edge_support.edge_success_mass[2].item() == pytest.approx(0.5)
    assert edge_support.edge_success_mass[1].item() == pytest.approx(0.0)
    assert edge_support.edge_conditional_success_prob[0].item() == pytest.approx(1.0)
    assert edge_support.edge_conditional_success_prob[2].item() == pytest.approx(1.0)


def test_rank_only_monte_carlo_uses_interval_aware_answer_posterior() -> None:
    torch.manual_seed(7)
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="rank-only-monte-carlo",
    )
    module = _make_rank_only_module()

    rank_metrics, window_results, model_metrics, diagnostics = module._evaluate_batch(
        batch=batch
    )

    del rank_metrics, model_metrics, diagnostics
    assert len(window_results) == 1
    result = cast(SupportWindowResult, window_results[0])
    assert result.inference_mode == "monte_carlo"
    assert result.answer_mass_reference == "monte_carlo"
    assert result.stop_reason == "rank_only_monte_carlo"
    assert result.gold_answer_mass_ci_low is not None
    assert result.gold_answer_mass_ci_high is not None
    assert result.answer_posterior
    assert result.answer_posterior[0].prob_ci_low >= 0.0
    assert (
        result.answer_posterior[0].prob_ci_high
        >= result.answer_posterior[0].prob_ci_low
    )


def test_full_monte_carlo_validation_batches_disconnected_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(19)
    batch = TrajectoryBatch.concatenate(
        [
            make_batch_from_graph(
                num_nodes=3,
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
                q_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([2], dtype=torch.long),
                answer_entity_ids=torch.tensor([102], dtype=torch.long),
                node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
                sample_id="full-batch-a",
            ),
            make_batch_from_graph(
                num_nodes=3,
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
                q_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([2], dtype=torch.long),
                answer_entity_ids=torch.tensor([202], dtype=torch.long),
                node_global_ids=torch.tensor([200, 201, 202], dtype=torch.long),
                sample_id="full-batch-b",
            ),
        ]
    )
    module = _make_full_module()
    prepare_call_count = 0
    original_prepare_batch = module.policy.prepare_batch

    def _count_prepare(current_batch):  # type: ignore[no-untyped-def]
        nonlocal prepare_call_count
        prepare_call_count += 1
        return original_prepare_batch(current_batch)

    monkeypatch.setattr(module.policy, "prepare_batch", _count_prepare)

    rank_metrics, window_results, model_metrics, diagnostics = module._evaluate_batch(
        batch=batch
    )

    assert prepare_call_count == 1
    assert model_metrics == {}
    assert rank_metrics
    assert len(window_results) == 2
    assert [result.sample_id for result in window_results] == [
        "full-batch-a",
        "full-batch-b",
    ]
    assert "answer/gold_answer_mass" in rank_metrics
    assert "support/path_mass" in diagnostics
