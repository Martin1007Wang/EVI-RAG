from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch

from src.data.artifacts import MaterializationArtifact, SplitArtifacts
from src.data.collate import RetrievalCollator
from src.data.dataset import RetrievalDataset
from src.data.schema import ReplayProgramSample, RetrievalData
from src.data.tensor_table import TensorTable
from src.graph.oracle_replay import build_replay_program
from src.graph.paths import compute_path_labels
from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.rollout.replay import ReplaySource, initial_replay_state_batch
from src.weaver.state import ExpansionBatch, StateBatch


def test_replay_program_edges_are_batched_as_local_edge_ids() -> None:
    replay = build_replay_program(
        edge_index=_edge_tensor([(0, 1), (1, 2)]),
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=torch.tensor([2]),
        num_nodes=3,
    )
    sample = _data(
        edge_index=_edge_tensor([(0, 1), (1, 2)]),
        num_nodes=3,
        target_node_ids=torch.tensor([2]),
        reachable_target_node_ids=torch.tensor([2]),
        node_target_distance=torch.tensor([2, 1, 0]),
        replay=replay,
    )

    batch = RetrievalCollator()([sample])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    replay_context = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target)

    assert replay_context.candidate_edge_ids.tolist() == [0, 1]
    assert replay_context.candidate_ptr.tolist() == [0, 2]
    assert replay_context.candidate_target_positions.tolist() == [0]
    assert replay_context.candidate_target_ptr.tolist() == [0, 1]


def test_program_frontier_replay_prefers_shared_trunk_edge() -> None:
    edge_index = _edge_tensor([(0, 1), (1, 2), (1, 3)])
    labels = compute_path_labels(edge_index=edge_index, anchor_node_ids=torch.tensor([0]), target_node_ids=torch.tensor([2, 3]), num_nodes=4)
    replay = build_replay_program(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        num_nodes=4,
    )
    data = _data(
        edge_index=edge_index,
        num_nodes=4,
        target_node_ids=torch.tensor([2, 3]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        node_target_distance=labels.node_target_distance,
        replay=replay,
    )

    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph, validate=True)
    replay_context = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target, validate=True)
    replay_out = ReplaySource(
        max_transitions_per_graph=8,
        top_k_per_state=1,
    ).collect(
        graph_context=graph,
        target_context=target,
        replay_context=replay_context,
        initial_state=initial_replay_state_batch(
            graph_context=graph,
            target_context=target,
            budget=2,
        ),
    )

    assert replay_out.nonterminal is not None
    assert replay_out.nonterminal.edge_ids.tolist() == [0]
    assert replay_out.nonterminal.parent_state.edge_count.tolist() == [0]
    assert replay_out.nonterminal.materialize_child_state().edge_count.tolist() == [1]
    assert replay_out.stats.prefix_count == 1
    assert replay_out.stats.positive_transition_count == 1
    assert replay_out.stats.prefix_with_positive_rate == pytest.approx(1.0)


def test_program_frontier_replay_advances_to_remaining_target_after_cover() -> None:
    edge_index = _edge_tensor([(0, 1), (1, 2), (1, 3)])
    labels = compute_path_labels(edge_index=edge_index, anchor_node_ids=torch.tensor([0]), target_node_ids=torch.tensor([2, 3]), num_nodes=4)
    replay = build_replay_program(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        num_nodes=4,
    )
    data = _data(
        edge_index=edge_index,
        num_nodes=4,
        target_node_ids=torch.tensor([2, 3]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        node_target_distance=labels.node_target_distance,
        replay=replay,
    )

    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph, validate=True)
    replay_context = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target, validate=True)
    state = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=3,
        graph_context=graph,
    ).branch(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        graph_context=graph,
    ).branch(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([1], dtype=torch.long),
        ),
        graph_context=graph,
    )
    replay_out = ReplaySource(
        max_transitions_per_graph=8,
        top_k_per_state=2,
    ).collect(
        graph_context=graph,
        target_context=target,
        replay_context=replay_context,
        initial_state=state,
    )

    assert replay_out.nonterminal is not None
    assert replay_out.nonterminal.edge_ids.tolist() == [2]


def test_replay_collect_is_one_step_only_over_prefix_batch() -> None:
    edge_index = _edge_tensor([(0, 1), (1, 2), (1, 3)])
    labels = compute_path_labels(edge_index=edge_index, anchor_node_ids=torch.tensor([0]), target_node_ids=torch.tensor([2, 3]), num_nodes=4)
    replay = build_replay_program(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        num_nodes=4,
    )
    data = _data(
        edge_index=edge_index,
        num_nodes=4,
        target_node_ids=torch.tensor([2, 3]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        node_target_distance=labels.node_target_distance,
        replay=replay,
    )
    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph, validate=True)
    replay_context = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target, validate=True)

    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    depth_one = root.branch(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        graph_context=graph,
    )
    prefix_batch = StateBatch.from_selected_edges(
        graph_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor([[0, -1], [-1, -1]], dtype=torch.long),
        edge_count=torch.tensor([1, 0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )

    replay_out = ReplaySource(
        max_transitions_per_graph=8,
        top_k_per_state=2,
    ).collect(
        graph_context=graph,
        target_context=target,
        replay_context=replay_context,
        initial_state=prefix_batch,
    )

    assert replay_out.nonterminal is not None
    assert replay_out.nonterminal.parent_state.edge_count.tolist() == [1, 0]
    assert replay_out.nonterminal.parent_state_ids.tolist() == [0, 0, 1]
    assert replay_out.nonterminal.edge_ids.tolist()[2] == 0
    assert sorted(replay_out.nonterminal.edge_ids.tolist()[:2]) == [1, 2]
    assert replay_out.nonterminal.materialize_child_state().edge_count.tolist() == [2, 2, 1]
    assert depth_one.edge_count.tolist() == [1]


def test_replay_collect_counts_prefixes_without_positive_edges() -> None:
    edge_index = _edge_tensor([(0, 1), (1, 2)])
    replay = build_replay_program(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=torch.tensor([2]),
        num_nodes=3,
    )
    data = _data(
        edge_index=edge_index,
        num_nodes=3,
        target_node_ids=torch.tensor([2]),
        reachable_target_node_ids=torch.tensor([2]),
        node_target_distance=torch.tensor([2, 1, 0]),
        replay=replay,
    )
    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph, validate=True)
    replay_context = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target, validate=True)

    recoverable = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    dead_end = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    ).branch(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        graph_context=graph,
    ).branch(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([1], dtype=torch.long),
        ),
        graph_context=graph,
    )
    prefix_batch = StateBatch.from_selected_edges(
        graph_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor([[-1, -1], [0, 1]], dtype=torch.long),
        edge_count=torch.tensor([0, 2], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )

    replay_out = ReplaySource(
        max_transitions_per_graph=8,
        top_k_per_state=2,
    ).collect(
        graph_context=graph,
        target_context=target,
        replay_context=replay_context,
        initial_state=prefix_batch,
    )

    assert replay_out.nonterminal is not None
    assert replay_out.nonterminal.parent_state_ids.tolist() == [0]
    assert replay_out.nonterminal.parent_state.edge_count.tolist() == [0, 2]
    assert replay_out.nonterminal.edge_ids.tolist() == [0]
    assert dead_end.edge_count.tolist() == [2]
    assert recoverable.edge_count.tolist() == [0]
    assert replay_out.stats.prefix_count == 2
    assert replay_out.stats.positive_transition_count == 1
    assert replay_out.stats.prefix_with_positive_rate == pytest.approx(0.5)


def test_retrieval_dataset_requires_replay_program_materialization() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        question_path = root / "questions.f32"
        question_path.write_bytes(b"\x00" * 16)
        train_lmdb = root / "train.lmdb"
        train_lmdb.mkdir()
        table = TensorTable(path=question_path, shape=(1, 4))
        materialization = MaterializationArtifact(
            generation_id="gen",
            manifest_path=root / "materialization_manifest.json",
            materialization_dir=root / "materialization",
            catalog=root / "catalog.pt",
            entity_text_semantic_table=table,
            relation_semantic_table=table,
            question_texts=None,
            splits={"train": SplitArtifacts(lmdb=train_lmdb, question_embeddings=table, num_samples=1)},
            provenance={"preprocess": {"replay_program": {"kind": "legacy"}}},
        )

        with pytest.raises(ValueError, match="replay_program_v3"):
            RetrievalDataset(materialization=materialization, split="train")


def test_retrieval_dataset_accepts_replay_program_materialization() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        question_path = root / "questions.f32"
        question_path.write_bytes(b"\x00" * 16)
        train_lmdb = root / "train.lmdb"
        train_lmdb.mkdir()
        table = TensorTable(path=question_path, shape=(1, 4))
        materialization = MaterializationArtifact(
            generation_id="gen",
            manifest_path=root / "materialization_manifest.json",
            materialization_dir=root / "materialization",
            catalog=root / "catalog.pt",
            entity_text_semantic_table=table,
            relation_semantic_table=table,
            question_texts=None,
            splits={"train": SplitArtifacts(lmdb=train_lmdb, question_embeddings=table, num_samples=1)},
            provenance={"preprocess": {"replay_program": {"kind": "replay_program_v3"}}},
        )

        dataset = RetrievalDataset(materialization=materialization, split="train")
        assert dataset.split == "train"


def test_replay_program_supports_more_than_64_reachable_targets() -> None:
    num_targets = 65
    edges = [(0, target_id) for target_id in range(1, num_targets + 1)]
    replay = build_replay_program(
        edge_index=_edge_tensor(edges),
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=torch.arange(1, num_targets + 1, dtype=torch.long),
        num_nodes=num_targets + 1,
    )

    assert replay.candidate_ptr.numel() == num_targets + 1
    assert replay.candidate_target_ptr.numel() == num_targets + 1
    assert replay.candidate_target_positions.numel() == num_targets
    assert replay.candidate_target_positions.tolist() == list(range(num_targets))


def test_replay_program_batch_offsets_multi_graph_inputs() -> None:
    replay_a = build_replay_program(
        edge_index=_edge_tensor([(0, 1), (1, 2)]),
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=torch.tensor([2]),
        num_nodes=3,
    )
    replay_b = build_replay_program(
        edge_index=_edge_tensor([(0, 1), (1, 2), (0, 2)]),
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=torch.tensor([2]),
        num_nodes=3,
    )
    sample_a = _data(
        edge_index=_edge_tensor([(0, 1), (1, 2)]),
        num_nodes=3,
        target_node_ids=torch.tensor([2]),
        reachable_target_node_ids=torch.tensor([2]),
        node_target_distance=torch.tensor([2, 1, 0]),
        replay=replay_a,
        sample_id="a",
    )
    sample_b = _data(
        edge_index=_edge_tensor([(0, 1), (1, 2), (0, 2)]),
        num_nodes=3,
        target_node_ids=torch.tensor([2]),
        reachable_target_node_ids=torch.tensor([2]),
        node_target_distance=torch.tensor([1, 1, 0]),
        replay=replay_b,
        sample_id="b",
    )

    batch = RetrievalCollator()([sample_a, sample_b])

    expected_edges = replay_a.candidate_edge_ids.tolist() + [int(v) + int(sample_a.num_edges) for v in replay_b.candidate_edge_ids.tolist()]
    assert batch.replay_program.candidate_edge_ids.tolist() == expected_edges
    assert batch.replay_program.candidate_ptr.tolist() == [0, int(replay_a.candidate_edge_ids.numel()), int(replay_a.candidate_edge_ids.numel() + replay_b.candidate_edge_ids.numel())]
    expected_edge_to_candidate_ids = replay_a.edge_to_candidate_ids.tolist() + [int(v) + (int(replay_a.candidate_ptr.numel()) - 1) for v in replay_b.edge_to_candidate_ids.tolist()]
    assert batch.replay_program.edge_to_candidate_ids.tolist() == expected_edge_to_candidate_ids
    assert batch.replay_program.candidate_graph_ptr.tolist() == [
        0,
        int(replay_a.candidate_ptr.numel()) - 1,
        (int(replay_a.candidate_ptr.numel()) - 1) + (int(replay_b.candidate_ptr.numel()) - 1),
    ]
    assert batch.replay_program.path_truncated_by_graph.tolist() == [0, 0]


def _edge_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _data(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    target_node_ids: torch.Tensor,
    reachable_target_node_ids: torch.Tensor,
    node_target_distance: torch.Tensor,
    replay,
    sample_id: str = "toy",
) -> RetrievalData:
    num_edges = int(edge_index.size(1))
    return RetrievalData(
        sample_id=sample_id,
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_entity_catalog_ids=torch.arange(num_nodes),
        edge_relation_catalog_ids=torch.arange(num_edges),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=target_node_ids,
        reachable_target_node_ids=reachable_target_node_ids,
        node_target_distance=node_target_distance,
        replay_program=ReplayProgramSample(
            candidate_edge_ids_local=replay.candidate_edge_ids,
            candidate_ptr=replay.candidate_ptr,
            candidate_target_positions=replay.candidate_target_positions,
            candidate_target_ptr=replay.candidate_target_ptr,
            edge_to_candidate_ids_local=replay.edge_to_candidate_ids,
            edge_to_candidate_ptr=replay.edge_to_candidate_ptr,
            path_truncated=torch.as_tensor(int(replay.path_truncated), dtype=torch.long),
        ),
    )
