from __future__ import annotations

import torch

from src.models.gflownet import ForwardActionDistribution
from src.models.gflownet import SearchState
from src.models.gflownet import apply_forward_constraints
from src.models.gflownet.repetition import build_entity_revisit_mask
from src.models.gflownet.repetition import build_entity_revisit_mask_from_flat_state

from .conftest import make_batch_from_graph, make_policy


def test_search_observation_keeps_node_entity_ids_for_no_repeat_constraints() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
    )
    policy = make_policy()

    prepared_batch = policy.prepare_batch(batch)

    assert torch.equal(
        prepared_batch.observation.node_entity_ids, batch.node_entity_ids
    )


def test_path_token_ids_interleave_absolute_nodes_and_relations() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([7, 8], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
    )
    policy = make_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0, 1),
        max_steps=2,
        device=batch.node_ptr.device,
    )
    flat_path = state.flatten_path_token_ids(max_steps=2)[0]
    rel0 = int(prepared_batch.topology.edge_type[0].item())
    rel1 = int(prepared_batch.topology.edge_type[1].item())

    assert flat_path.tolist() == [0, rel0, 1, rel1, 2, 0]


def test_flat_no_repeat_helper_matches_state_wrapper() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
    )
    policy = make_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0,),
        max_steps=2,
        device=batch.node_ptr.device,
    )
    candidate_target_nodes = torch.tensor([0, 2], dtype=torch.long)
    candidate_edge_agent_batch = torch.tensor([0, 0], dtype=torch.long)

    wrapped_mask = build_entity_revisit_mask(
        state=state,
        candidate_target_abs_nodes=candidate_target_nodes,
        candidate_agent_indices=candidate_edge_agent_batch,
        max_steps=2,
    )
    flat_mask = build_entity_revisit_mask_from_flat_state(
        flat_current_abs_nodes=state.flatten_current_nodes(),
        flat_num_steps=state.flatten_num_steps(),
        flat_path_token_ids=state.flatten_path_token_ids(max_steps=2),
        node_entity_ids_by_abs_node=prepared_batch.observation.node_entity_ids,
        num_nodes=int(prepared_batch.topology.num_nodes),
        candidate_target_abs_nodes=candidate_target_nodes,
        candidate_agent_indices=candidate_edge_agent_batch,
    )

    assert torch.equal(wrapped_mask, flat_mask)
    assert torch.equal(flat_mask, torch.tensor([True, False], dtype=torch.bool))


def test_root_state_no_repeat_only_blocks_current_entity() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
    )
    policy = make_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
    )

    repeat_mask = build_entity_revisit_mask(
        state=state,
        candidate_target_abs_nodes=torch.tensor([0, 1, 2], dtype=torch.long),
        candidate_agent_indices=torch.tensor([0, 0, 0], dtype=torch.long),
        max_steps=2,
    )

    assert torch.equal(
        repeat_mask, torch.tensor([True, False, False], dtype=torch.bool)
    )


def test_forward_distribution_filters_full_revisit_before_scoring() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0,),
        max_steps=2,
        device=batch.node_ptr.device,
    )

    distribution = policy.compute_forward_distribution(prepared_batch, state)

    stop_mask = (
        distribution.is_stop_action.to(dtype=torch.bool)
        if distribution.is_stop_action is not None
        else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
    )
    graph_targets = distribution.target_nodes[~stop_mask]

    assert 0 not in {int(node) for node in graph_targets.tolist()}
    assert 2 in {int(node) for node in graph_targets.tolist()}


def test_apply_forward_constraints_masks_revisit_on_manual_distribution() -> None:
    batch = make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0,),
        max_steps=2,
        device=batch.node_ptr.device,
    )
    distribution = ForwardActionDistribution(
        edge_logits=torch.tensor([0.0, -1.0, -0.5], dtype=torch.float32),
        edge_agent_batch=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_ids=torch.tensor([1, 2, -1], dtype=torch.long),
        target_nodes=torch.tensor([0, 2, 1], dtype=torch.long),
        out_degrees=torch.tensor([[3]], dtype=torch.long),
        is_stop_action=torch.tensor([False, False, True], dtype=torch.bool),
    )

    constrained = apply_forward_constraints(
        distribution,
        state=state,
        max_steps=2,
    )
    move_log_probs, _, _ = policy.compute_move_log_probs(constrained)
    revisit_mask = constrained.target_nodes == 0
    fresh_mask = constrained.target_nodes == 2
    stop_mask = constrained.is_stop_action

    assert stop_mask is not None
    assert bool(revisit_mask.any().item())
    assert bool(fresh_mask.any().item())
    assert not torch.isfinite(move_log_probs[revisit_mask]).any()
    assert torch.isfinite(move_log_probs[fresh_mask]).all()
    assert torch.isfinite(move_log_probs[stop_mask]).all()


def test_forward_distribution_masks_repeated_entity_across_duplicate_nodes() -> None:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 2], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([3], dtype=torch.long),
        answer_entity_ids=torch.tensor([103], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 100, 103], dtype=torch.long),
    )
    policy = make_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0,),
        max_steps=2,
        device=batch.node_ptr.device,
    )

    distribution = policy.compute_forward_distribution(prepared_batch, state)

    stop_mask = (
        distribution.is_stop_action.to(dtype=torch.bool)
        if distribution.is_stop_action is not None
        else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
    )
    graph_targets = distribution.target_nodes[~stop_mask]

    assert 2 not in {int(node) for node in graph_targets.tolist()}
    assert 3 in {int(node) for node in graph_targets.tolist()}


def test_forward_constraints_mask_all_moves_at_horizon() -> None:
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
    )
    policy = make_policy()
    prepared_batch = policy.prepare_batch(batch)
    state = SearchState.from_edge_path(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_node=0,
        edge_ids=(0, 1),
        max_steps=2,
        device=batch.node_ptr.device,
    )

    distribution = policy.compute_forward_distribution(prepared_batch, state)
    distribution = apply_forward_constraints(
        distribution,
        state=state,
        max_steps=2,
    )
    move_log_probs, _, has_values = policy.compute_move_log_probs(distribution)
    submit_mask = distribution.is_stop_action

    assert submit_mask is not None
    assert bool(has_values.item()) is True
    assert bool(submit_mask.any().item()) is True
    assert torch.isfinite(move_log_probs[submit_mask]).all()
    assert not torch.isfinite(move_log_probs[~submit_mask]).any()
