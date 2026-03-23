from __future__ import annotations

import pytest
import torch

from src.metrics.answer_metrics import (
    DiscoveredTrajectory,
    ReachabilityAnalysis,
    SearchDiagnostics,
    build_window_result,
    compute_support_metrics,
)

from .conftest import make_batch_from_graph


def test_support_window_penalizes_overlapping_paths() -> None:
    batch = make_batch_from_graph(
        num_nodes=6,
        edge_index=torch.tensor(
            [[0, 1, 1, 0, 3, 5], [1, 2, 5, 3, 2, 2]], dtype=torch.long
        ),
        edge_rel_global=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.long),
        sample_id="support-diversity",
    )
    analysis = ReachabilityAnalysis(
        terminal_mass=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        answer_probs=torch.tensor([1.0], dtype=torch.float32),
        gold_answer_mass=1.0,
    )
    discovered_paths = [
        DiscoveredTrajectory(
            start_node=0,
            terminal_node=2,
            answer_entity_id=102,
            edge_ids=(0, 1),
            log_prob=float(torch.log(torch.tensor(0.46)).item()),
            is_gold=True,
        ),
        DiscoveredTrajectory(
            start_node=0,
            terminal_node=2,
            answer_entity_id=102,
            edge_ids=(0, 2, 5),
            log_prob=float(torch.log(torch.tensor(0.42)).item()),
            is_gold=True,
        ),
        DiscoveredTrajectory(
            start_node=0,
            terminal_node=2,
            answer_entity_id=102,
            edge_ids=(3, 4),
            log_prob=float(torch.log(torch.tensor(0.38)).item()),
            is_gold=True,
        ),
    ]

    result = build_window_result(
        batch=batch,
        analysis=analysis,
        diagnostics=SearchDiagnostics(
            inference_mode="monte_carlo",
            probe_count=3,
            remaining_mass_upper=0.0,
            stop_reason="support_mass_reached",
            coverage_certified=False,
        ),
        discovered_paths=discovered_paths,
        answer_mass_threshold=1.0,
        support_mass_threshold=0.8,
        support_path_overlap_penalty=1.0,
        answer_mass_reference="monte_carlo",
        support_mass_reference="monte_carlo",
    )

    emitted_edge_ids = [
        tuple(edge.edge_id for edge in traj.edges) for traj in result.trajectories
    ]
    assert emitted_edge_ids == [(0, 1), (3, 4)]
    assert result.covered_mass == pytest.approx(0.84)
    assert result.answer_posterior[0].support_path_count == 2

    metrics = compute_support_metrics([result])
    assert metrics["support/diversity"] == pytest.approx(1.0)
