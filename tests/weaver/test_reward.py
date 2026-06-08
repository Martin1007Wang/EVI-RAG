from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext, TargetContext
from src.weaver.reward import EvidenceStateScorer
from src.weaver.state import StateBatch


def _graph_context() -> GraphContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_ptr=torch.tensor([0, 2], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False]),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 1, 2, 2], dtype=torch.long),
            edge_ids_by_src=torch.tensor([0, 1], dtype=torch.long),
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )


def _target_context() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, False, True]),
        reachable_target_node_ids=torch.tensor([2], dtype=torch.long),
        reachable_target_node_ids_ptr=torch.tensor([0, 1], dtype=torch.long),
        target_count_by_graph=torch.tensor([1], dtype=torch.long),
        node_target_distance=torch.tensor([2, 1, 0], dtype=torch.long),
        edge_on_shortest_path=torch.tensor([True, True], dtype=torch.bool),
        target_max_distance_by_graph=torch.tensor([2], dtype=torch.long),
        anchor_target_count_by_graph=torch.tensor([0], dtype=torch.long),
    )


def _anchor_target_context() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([True, False, False]),
        reachable_target_node_ids=torch.tensor([0], dtype=torch.long),
        reachable_target_node_ids_ptr=torch.tensor([0, 1], dtype=torch.long),
        target_count_by_graph=torch.tensor([1], dtype=torch.long),
        node_target_distance=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_on_shortest_path=torch.tensor([False, False], dtype=torch.bool),
        target_max_distance_by_graph=torch.tensor([0], dtype=torch.long),
        anchor_target_count_by_graph=torch.tensor([1], dtype=torch.long),
    )


def test_evidence_state_score_uses_distance_coverage_potential_and_terminal_recall() -> None:
    graph_context = _graph_context()
    target_context = _target_context()
    state = StateBatch(
        graph_ids=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_ids=torch.tensor([[-1, -1], [0, -1], [0, 1]], dtype=torch.long),
        edge_count=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    output = EvidenceStateScorer(budget=2)(
        state=state,
        target_context=target_context,
        graph_context=graph_context,
    )

    assert torch.equal(output.answer_count, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(output.target_count, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(output.state_potential, torch.tensor([0.0, 1.25, 1.5]))
    assert torch.allclose(output.terminal_quality, torch.tensor([-9.2104406, -9.4604406, -0.5]))
    assert torch.allclose(output.log_reward, torch.tensor([0.0, -26.881321, 0.0]))
    assert torch.allclose(output.remaining_log_reward, torch.tensor([0.0, -28.131321, -1.5]))
    assert torch.allclose(output.raw_log_reward, torch.tensor([0.0, -28.381321, -1.5]))
    assert torch.equal(output.terminal_valid_mask, torch.tensor([False, True, True]))
    assert "reward/residual_mean" in output.metrics


def test_evidence_state_score_excludes_anchor_target_hits() -> None:
    graph_context = _graph_context()
    target_context = _anchor_target_context()
    state = StateBatch(
        graph_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor([[-1, -1], [0, -1]], dtype=torch.long),
        edge_count=torch.tensor([0, 1], dtype=torch.long),
    )

    output = EvidenceStateScorer(budget=2)(
        state=state,
        target_context=target_context,
        graph_context=graph_context,
    )

    assert torch.equal(output.state_potential, torch.tensor([0.0, 0.0]))
    assert torch.equal(output.log_reward, torch.tensor([0.0, 0.0]))
    assert torch.equal(output.terminal_valid_mask, torch.tensor([False, False]))
