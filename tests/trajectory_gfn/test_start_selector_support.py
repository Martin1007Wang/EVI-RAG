from __future__ import annotations

import torch

from .conftest import make_batch_from_graph, make_policy


def test_start_distribution_masks_dead_end_question_entities() -> None:
    torch.manual_seed(23)
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[1], [2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="mixed-start-support",
    )
    policy = make_policy()
    context = policy.encode(batch)

    distribution = policy.compute_start_distribution(context)

    assert torch.equal(distribution.candidate_nodes_abs, torch.tensor([0, 1]))
    assert torch.isneginf(distribution.log_probs[0])
    sampled_nodes, _ = policy.sample_start_nodes(
        distribution,
        num_rollouts=16,
        deterministic=False,
    )
    assert torch.equal(sampled_nodes.unique(), torch.tensor([1]))
