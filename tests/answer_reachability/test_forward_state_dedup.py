from __future__ import annotations

import torch

from src.graph_runtime import TrajectoryBatch
from src.models.gflownet import SearchState

from .conftest import make_batch_from_graph, make_policy


def test_forward_distribution_deduplicates_identical_states_with_graph_aligned_scores() -> (
    None
):
    policy = make_policy(max_steps=2)
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    edge_rel_global = torch.tensor([0, 1], dtype=torch.long)
    batch = TrajectoryBatch.concatenate(
        [
            make_batch_from_graph(
                num_nodes=3,
                edge_index=edge_index,
                edge_rel_global=edge_rel_global,
                q_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([2], dtype=torch.long),
                answer_entity_ids=torch.tensor([102], dtype=torch.long),
                node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
                question_emb=torch.tensor([[10.0] + [0.0] * 7], dtype=torch.float32),
                question_ctx=torch.zeros((1, 2, 8), dtype=torch.float32),
                sample_id="graph-a",
            ),
            make_batch_from_graph(
                num_nodes=3,
                edge_index=edge_index,
                edge_rel_global=edge_rel_global,
                q_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([2], dtype=torch.long),
                answer_entity_ids=torch.tensor([202], dtype=torch.long),
                node_global_ids=torch.tensor([200, 201, 202], dtype=torch.long),
                question_emb=torch.tensor([[20.0] + [0.0] * 7], dtype=torch.float32),
                question_ctx=torch.zeros((1, 2, 8), dtype=torch.float32),
                sample_id="graph-b",
            ),
        ]
    )
    prepared_batch = policy.prepare_batch(batch)
    scored_batch_sizes: list[int] = []

    def _mock_forward(
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        del candidate_state_features, relation_features
        scored_batch_sizes.append(int(current_state_features.size(0)))
        return current_state_features[:, 0].to(dtype=torch.float32)

    policy.forward_policy_head.forward = _mock_forward  # type: ignore[method-assign]

    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0, 0], [3, 3]], dtype=torch.long),
        done_mask=torch.tensor([[True, True], [False, False]], dtype=torch.bool),
        num_steps=torch.zeros((2, 2), dtype=torch.long),
    )

    distribution = policy.compute_forward_distribution(prepared_batch, state)

    assert sum(scored_batch_sizes) == 3
    assert torch.equal(distribution.edge_agent_batch, torch.tensor([2, 2, 2, 3, 3, 3]))
    assert torch.allclose(
        distribution.edge_logits[:3].to(dtype=torch.float32),
        distribution.edge_logits[3:].to(dtype=torch.float32),
    )
    assert torch.equal(distribution.out_degrees.view(-1), torch.tensor([0, 0, 3, 3]))


def test_forward_distribution_dedup_keeps_distinct_path_histories_separate() -> None:
    batch = make_batch_from_graph(
        num_nodes=5,
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 3, 4]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([4], dtype=torch.long),
        answer_entity_ids=torch.tensor([104], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103, 104], dtype=torch.long),
        sample_id="path-sensitive-forward",
    )
    policy = make_policy(max_steps=3)
    prepared_batch = policy.prepare_batch(batch)
    scored_batch_sizes: list[int] = []

    def _mock_forward(
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        del candidate_state_features, relation_features
        scored_batch_sizes.append(int(current_state_features.size(0)))
        return torch.zeros(
            (int(current_state_features.size(0)),),
            device=current_state_features.device,
            dtype=torch.float32,
        )

    policy.forward_policy_head.forward = _mock_forward  # type: ignore[method-assign]

    state_path_a = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0, 2),
        max_steps=3,
        device=batch.node_ptr.device,
    )
    state_path_b = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(1, 3),
        max_steps=3,
        device=batch.node_ptr.device,
    )
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.cat(
            (
                state_path_a.current_nodes,
                state_path_a.current_nodes,
                state_path_b.current_nodes,
            ),
            dim=1,
        ),
        done_mask=torch.zeros((1, 3), dtype=torch.bool),
        num_steps=torch.cat(
            (state_path_a.num_steps, state_path_a.num_steps, state_path_b.num_steps),
            dim=1,
        ),
        path_token_ids=torch.cat(
            (
                state_path_a.path_token_ids,
                state_path_a.path_token_ids,
                state_path_b.path_token_ids,
            ),
            dim=1,
        ),
    )

    distribution = policy.compute_forward_distribution(prepared_batch, state)

    assert sum(scored_batch_sizes) == 4
    assert torch.equal(distribution.edge_agent_batch, torch.tensor([0, 0, 1, 1, 2, 2]))
    assert torch.equal(distribution.out_degrees.view(-1), torch.tensor([2, 2, 2]))
