from __future__ import annotations

import pytest
import torch

from src.models.gflownet import InvalidStartCandidatesError

from .conftest import make_batch_from_graph, make_policy


def test_start_distribution_keeps_dead_end_question_entities_in_support() -> None:
    torch.manual_seed(23)
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[1], [2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="mixed-start-support",
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)

    distribution = policy.compute_start_distribution(prepared_batch)

    assert torch.equal(distribution.candidate_nodes_abs, torch.tensor([0, 1]))
    assert torch.isfinite(distribution.log_probs).all()


def test_start_distribution_rejects_all_nonfinite_start_logits() -> None:
    torch.manual_seed(29)
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="nonfinite-start-logits",
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)

    def _nan_state_scores(
        node_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        del question_features
        return torch.full(
            (node_features.size(0),), float("nan"), device=node_features.device
        )

    policy.state_score_head.forward = _nan_state_scores  # type: ignore[method-assign]

    with pytest.raises(InvalidStartCandidatesError, match="finite start candidate"):
        policy.compute_start_distribution(prepared_batch)
