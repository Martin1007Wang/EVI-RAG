from __future__ import annotations

import pytest
import torch

from src.models.gflownet.sampler import _compute_start_distribution_entropy


def test_single_candidate_graph_reports_full_normalized_entropy() -> None:
    log_probs = torch.tensor([0.0], dtype=torch.float32)
    candidate_graph_ids = torch.tensor([0], dtype=torch.long)

    entropy, normalized = _compute_start_distribution_entropy(
        log_probs=log_probs,
        candidate_graph_ids=candidate_graph_ids,
        num_graphs=1,
    )

    assert entropy.tolist() == [0.0]
    # The current metric treats singleton graphs as maximally normalized because
    # there is no non-trivial choice to normalize against.
    assert normalized.tolist() == [1.0]


def test_mean_normalized_entropy_is_misleading_when_most_graphs_are_singletons() -> (
    None
):
    singleton_count = 49
    singleton_log_probs = torch.zeros((singleton_count,), dtype=torch.float32)
    multi_log_probs = torch.tensor([0.0, float("-inf")], dtype=torch.float32)
    log_probs = torch.cat((singleton_log_probs, multi_log_probs), dim=0)
    candidate_graph_ids = torch.cat(
        (
            torch.arange(singleton_count, dtype=torch.long),
            torch.full((2,), fill_value=singleton_count, dtype=torch.long),
        ),
        dim=0,
    )

    _, normalized = _compute_start_distribution_entropy(
        log_probs=log_probs,
        candidate_graph_ids=candidate_graph_ids,
        num_graphs=singleton_count + 1,
    )

    # Only the last graph has a real start-choice decision and it is perfectly
    # concentrated, yet the batch mean still appears near 1 because singleton
    # graphs contribute 1.0 by construction.
    assert normalized[-1].item() == 0.0
    assert normalized.mean().item() == pytest.approx(0.98)
