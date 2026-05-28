from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch

from src.data.collate import RetrievalCollator
from src.data.dataset import RetrievalDataset
from src.data.artifacts import MaterializationArtifact, SplitArtifacts
from src.data.tensor_table import TensorTable
from src.data.schema import RetrievalData
from src.graph.paths import compute_path_labels
from src.weaver.context import GraphContext, TargetContext
from src.weaver.rollout.replay import WeakTransitionSource, initial_replay_state_batch


def test_weak_replay_edges_are_batched_as_local_edge_ids() -> None:
    sample = RetrievalData(
        edge_index=_edge_tensor([(0, 1), (1, 2)]),
        num_nodes=3,
        node_entity_catalog_ids=torch.arange(3),
        edge_relation_catalog_ids=torch.arange(2),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([2]),
        reachable_target_node_ids=torch.tensor([2]),
        node_target_distance=torch.tensor([2, 1, 0]),
        weak_replay_edge_ids=torch.tensor([0, 1]),
        weak_replay_edge_weight=torch.tensor([1.0, 1.0]),
        witness_path_edge_ids=torch.tensor([0, 1]),
        witness_path_edge_path_ids=torch.tensor([0, 0]),
        witness_path_target_node_ids=torch.tensor([2]),
    )
    sample.sample_id = "toy"

    batch = RetrievalCollator()([sample])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)

    assert target.shortest_path_edge_mask.tolist() == [True, True]
    assert target.shortest_path_edge_weight.tolist() == [1.0, 1.0]
    assert target.witness_path_edge_ids.tolist() == [0, 1]
    assert target.witness_path_edge_path_ids.tolist() == [0, 0]
    assert target.witness_path_target_node_ids.tolist() == [2]


def test_weak_transition_source_collects_positive_frontier_transitions() -> None:
    edge_index = _edge_tensor([(0, 1), (1, 2), (0, 3)])
    labels = compute_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([2]),
        num_nodes=4,
    )
    data = RetrievalData(
        edge_index=edge_index,
        num_nodes=4,
        node_entity_catalog_ids=torch.arange(4),
        edge_relation_catalog_ids=torch.arange(3),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([2]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        node_target_distance=labels.node_target_distance,
        weak_replay_edge_ids=torch.tensor([0, 1]),
        weak_replay_edge_weight=torch.tensor([1.0, 1.0]),
        witness_path_edge_ids=labels.witness_path_edge_ids,
        witness_path_edge_path_ids=labels.witness_path_edge_path_ids,
        witness_path_target_node_ids=labels.witness_path_target_node_ids,
    )
    data.sample_id = "toy"

    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    weak = WeakTransitionSource(
        max_depth=2,
        mode="positive_frontier",
        max_transitions_per_graph=64,
        max_states_per_graph=4,
        max_positive_edges_per_state=8,
    ).collect(
        graph_context=graph,
        target_context=target,
        initial_state=initial_replay_state_batch(
            graph_context=graph,
            target_context=target,
            budget=2,
        ),
    )

    assert weak.nonterminal is not None
    assert weak.nonterminal.edge_ids.tolist() == [0, 1]
    assert weak.nonterminal.parent_state.edge_count.tolist() == [0, 1]
    assert weak.nonterminal.child_state is not None
    assert weak.nonterminal.child_state.edge_count.tolist() == [1, 2]
    assert weak.stats.prefix_count == 2
    assert weak.stats.positive_transition_count == 2


def test_positive_frontier_replay_caps_to_one_edge_per_state() -> None:
    edge_index = _edge_tensor([(0, 1), (0, 2), (1, 3), (2, 3)])
    labels = compute_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([3]),
        num_nodes=4,
    )
    data = RetrievalData(
        edge_index=edge_index,
        num_nodes=4,
        node_entity_catalog_ids=torch.arange(4),
        edge_relation_catalog_ids=torch.arange(4),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([3]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        node_target_distance=labels.node_target_distance,
        weak_replay_edge_ids=torch.tensor([0, 1, 2, 3]),
        weak_replay_edge_weight=torch.tensor([1.0, 2.0, 1.0, 1.0]),
        witness_path_edge_ids=labels.witness_path_edge_ids,
        witness_path_edge_path_ids=labels.witness_path_edge_path_ids,
        witness_path_target_node_ids=labels.witness_path_target_node_ids,
    )
    data.sample_id = "toy-positive-sampled"

    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    weak = WeakTransitionSource(
        max_depth=1,
        mode="positive_frontier",
        max_transitions_per_graph=1,
        max_states_per_graph=4,
        max_positive_edges_per_state=1,
    ).collect(
        graph_context=graph,
        target_context=target,
        initial_state=initial_replay_state_batch(
            graph_context=graph,
            target_context=target,
            budget=2,
        ),
    )

    assert weak.nonterminal is not None
    assert weak.nonterminal.edge_ids.numel() == 1
    assert weak.stats.positive_transition_count == 1


def test_positive_frontier_replay_caps_states_per_graph() -> None:
    edge_index = _edge_tensor([(0, 1), (0, 2), (1, 3), (2, 3)])
    labels = compute_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([3]),
        num_nodes=4,
    )
    data = RetrievalData(
        edge_index=edge_index,
        num_nodes=4,
        node_entity_catalog_ids=torch.arange(4),
        edge_relation_catalog_ids=torch.arange(4),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([3]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        node_target_distance=labels.node_target_distance,
        weak_replay_edge_ids=torch.tensor([0, 1, 2, 3]),
        weak_replay_edge_weight=torch.tensor([1.0, 2.0, 1.0, 1.0]),
        witness_path_edge_ids=labels.witness_path_edge_ids,
        witness_path_edge_path_ids=labels.witness_path_edge_path_ids,
        witness_path_target_node_ids=labels.witness_path_target_node_ids,
    )
    data.sample_id = "toy-state-cap"

    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    weak = WeakTransitionSource(
        max_depth=2,
        mode="positive_frontier",
        max_transitions_per_graph=16,
        max_states_per_graph=1,
        max_positive_edges_per_state=2,
    ).collect(
        graph_context=graph,
        target_context=target,
        initial_state=initial_replay_state_batch(
            graph_context=graph,
            target_context=target,
            budget=2,
        ),
    )

    assert weak.nonterminal is not None
    assert weak.nonterminal.child_state is not None
    assert weak.nonterminal.edge_ids.tolist() == [0, 1, 0, 3]
    assert weak.nonterminal.parent_state.edge_count.tolist() == [0, 1]
    assert weak.nonterminal.parent_state_ids.tolist() == [0, 0, 1, 1]
    assert weak.nonterminal.child_state.edge_count.tolist() == [1, 1, 2, 2]


def test_retrieval_dataset_allows_non_witness_path_materialization_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        question_path = root / "questions.f32"
        question_path.write_bytes(b"\x00" * 16)
        train_lmdb = root / "train.lmdb"
        train_lmdb.mkdir()
        table = TensorTable(
            path=question_path,
            shape=(1, 4),
        )
        materialization = MaterializationArtifact(
            generation_id="gen",
            manifest_path=root / "materialization_manifest.json",
            materialization_dir=root / "materialization",
            catalog=root / "catalog.pt",
            entity_text_semantic_table=table,
            relation_semantic_table=table,
            question_texts=None,
            splits={
                "train": SplitArtifacts(
                    lmdb=train_lmdb,
                    question_embeddings=table,
                    num_samples=1,
                )
            },
            provenance={"preprocess": {"weak_replay_labels": {"kind": "shortest_path_edge_set_v1"}}},
        )

        dataset = RetrievalDataset(materialization=materialization, split="train")
        assert dataset.require_witness_paths is False


def _edge_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
