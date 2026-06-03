from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch

from src.data.artifacts import MaterializationArtifact, SplitArtifacts
from src.data.collate import RetrievalCollator
from src.data.dataset import RetrievalDataset
from src.data.schema import ReplayBankBatch, ReplayBankSample, RetrievalData
from src.data.tensor_table import write_table
from src.graph.oracle_replay import build_replay_bank, build_shortest_path_dag
from src.graph.paths import compute_path_labels
from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.rollout.replay import ReplaySource
from src.weaver.rollout.trajectory import EXTERNAL_TERMINAL
from src.weaver.rollout.runner import _train_rollout_metrics


def test_shortest_path_dag_preserves_all_pairs_and_shared_prefix() -> None:
    edges = _edge_tensor([(0, 2), (1, 2), (2, 3), (2, 4)])
    dag = build_shortest_path_dag(
        edge_index=edges,
        anchor_node_ids=torch.tensor([0, 1]),
        reachable_target_node_ids=torch.tensor([3, 4]),
        num_nodes=5,
    )
    assert dag.pair_anchor_node_ids.tolist() == [0, 0, 1, 1]
    assert dag.pair_target_node_ids.tolist() == [3, 4, 3, 4]
    assert dag.pair_distance.tolist() == [2, 2, 2, 2]
    assert dag.pair_edge_ptr.tolist() == [0, 2, 4, 6, 8]


def test_layered_shortest_path_dag_storage_is_linear_in_edges() -> None:
    edges = _edge_tensor([(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5)])
    dag = build_shortest_path_dag(
        edge_index=edges,
        anchor_node_ids=torch.tensor([0]),
        reachable_target_node_ids=torch.tensor([5]),
        num_nodes=6,
    )
    assert dag.pair_distance.tolist() == [3]
    assert dag.pair_edge_ids.numel() == edges.size(1)


def test_replay_bank_supports_multiple_budgets_and_covers_shared_targets() -> None:
    sample = _data(edges=[(0, 1), (1, 2), (1, 3)], anchors=[0], targets=[2, 3])
    graph, target, replay = _contexts(sample)
    source = ReplaySource()
    budget_two = source.sample_trajectories(graph_context=graph, target_context=target, replay_context=replay, budget=2)
    budget_three = source.sample_trajectories(graph_context=graph, target_context=target, replay_context=replay, budget=3)
    assert budget_two.num_trajectories > 0
    assert all(count <= 2 for count in budget_two.edge_count.tolist())
    assert any(set(row[:count].tolist()) == {0, 1, 2} for row, count in zip(budget_three.edge_ids, budget_three.edge_count, strict=True))
    assert budget_three.stop_reason.unique().tolist() == [EXTERNAL_TERMINAL]


def test_replay_bank_is_stable_per_round_and_rotates_equal_paths() -> None:
    sample = _data(edges=[(0, 1), (0, 2), (1, 3), (2, 3)], anchors=[0], targets=[3])
    graph, target, replay = _contexts(sample)
    source = ReplaySource()
    first = source.sample_trajectories(graph_context=graph, target_context=target, replay_context=replay, budget=2, replay_round=0)
    repeated = source.sample_trajectories(graph_context=graph, target_context=target, replay_context=replay, budget=2, replay_round=0)
    assert torch.equal(first.edge_ids, repeated.edge_ids)
    assert {tuple(sorted(row[:count].tolist())) for row, count in zip(first.edge_ids, first.edge_count, strict=True)} == {(0, 2), (1, 3)}


def test_replay_bank_emits_zero_edge_terminal_for_anchor_answer() -> None:
    sample = _data(edges=[(0, 1)], anchors=[0], targets=[0])
    graph, target, replay = _contexts(sample)
    trajectories = ReplaySource().sample_trajectories(graph_context=graph, target_context=target, replay_context=replay, budget=0)
    assert trajectories.num_trajectories == 1
    assert trajectories.edge_count.tolist() == [0]
    assert trajectories.stop_reason.tolist() == [EXTERNAL_TERMINAL]


def test_replay_source_anneals_trajectory_count_by_global_step() -> None:
    sample = _data(edges=[(0, 1), (1, 2), (1, 3)], anchors=[0], targets=[2, 3])
    graph, target, replay = _contexts(sample)
    source = ReplaySource(anneal_steps=20)
    raw_count = source.raw_trajectory_count(replay_context=replay, budget=2)

    start = source.sample_trajectories(
        graph_context=graph,
        target_context=target,
        replay_context=replay,
        budget=2,
        global_step=0,
    )
    halfway = source.sample_trajectories(
        graph_context=graph,
        target_context=target,
        replay_context=replay,
        budget=2,
        global_step=10,
    )
    finished = source.sample_trajectories(
        graph_context=graph,
        target_context=target,
        replay_context=replay,
        budget=2,
        global_step=20,
    )

    assert raw_count > 0
    assert start.num_trajectories == raw_count
    assert halfway.num_trajectories == int(0.5 * raw_count)
    assert finished.num_trajectories == 0
    assert source.replay_weight(global_step=0) == pytest.approx(1.0)
    assert source.replay_weight(global_step=10) == pytest.approx(0.5)
    assert source.replay_weight(global_step=20) == pytest.approx(0.0)


def test_train_rollout_metrics_include_replay_schedule_fields() -> None:
    sample = _data(edges=[(0, 1), (1, 2), (1, 3)], anchors=[0], targets=[2, 3])
    graph, target, replay = _contexts(sample)
    source = ReplaySource(anneal_steps=20)
    raw_count = source.raw_trajectory_count(replay_context=replay, budget=2)
    trajectories = source.sample_trajectories(
        graph_context=graph,
        target_context=target,
        replay_context=replay,
        budget=2,
        global_step=10,
    )

    metrics = _train_rollout_metrics(
        trajectories=trajectories,
        device=graph.device,
        context=graph,
        target_context=target,
        replay_weight=torch.tensor(0.5, device=graph.device),
        replay_raw_count=raw_count,
    )

    assert float(metrics["replay_weight"]) == pytest.approx(0.5)
    assert float(metrics["replay_raw_trajectory_count"]) == pytest.approx(float(raw_count))
    assert float(metrics["replay_kept_trajectory_count"]) == pytest.approx(float(int(0.5 * raw_count)))
    assert float(metrics["replay/trajectory_count"]) == pytest.approx(float(int(0.5 * raw_count)))


def test_replay_bank_batch_to_moves_all_tensors() -> None:
    bank = ReplayBankBatch(
        edge_ids=torch.tensor([[[[[0, -1]]]]], dtype=torch.long),
        edge_count=torch.tensor([[[[1]]]], dtype=torch.long),
    )
    moved = bank.to(torch.device("cpu"))
    assert isinstance(moved, ReplayBankBatch)
    assert moved is not bank
    assert moved.edge_ids.device.type == "cpu"
    assert moved.edge_count.device.type == "cpu"
    assert torch.equal(moved.edge_ids, bank.edge_ids)
    assert torch.equal(moved.edge_count, bank.edge_count)


def test_replay_context_aligns_bank_tensors_to_graph_device() -> None:
    sample = _data(edges=[(0, 1), (1, 2)], anchors=[0], targets=[2])
    batch = RetrievalCollator()([sample])
    batch.replay_bank = ReplayBankBatch(
        edge_ids=batch.replay_bank.edge_ids.clone(),
        edge_count=batch.replay_bank.edge_count.clone(),
    )
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph, validate=True)
    replay = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target, validate=True)
    assert replay.edge_ids.device == graph.device
    assert replay.edge_count.device == graph.device


def test_replay_source_outputs_trajectories_on_graph_device() -> None:
    sample = _data(edges=[(0, 1), (1, 2)], anchors=[0], targets=[2])
    graph, target, replay = _contexts(sample)
    replay = ReplayContext(
        edge_ids=replay.edge_ids.to(device=torch.device("cpu")),
        edge_count=replay.edge_count.to(device=torch.device("cpu")),
    )
    trajectories = ReplaySource().sample_trajectories(
        graph_context=graph,
        target_context=target,
        replay_context=replay,
        budget=2,
    )
    assert trajectories.graph_ids.device == graph.device
    assert trajectories.edge_ids.device == graph.device
    assert trajectories.edge_count.device == graph.device
    assert trajectories.stop_logp.device == graph.device


def test_retrieval_dataset_requires_replay_bank_materialization() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact = _artifact(Path(tmp), kind="terminal_trajectory_v1")
        with pytest.raises(ValueError, match="replay_bank_v1"):
            RetrievalDataset(materialization=artifact, split="train")


def test_retrieval_dataset_accepts_replay_bank_materialization() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dataset = RetrievalDataset(materialization=_artifact(Path(tmp), kind="replay_bank_v1"), split="train")
        assert dataset.num_samples == 1


def _contexts(sample: RetrievalData):
    batch = RetrievalCollator()([sample])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph, validate=True)
    replay = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target, validate=True)
    return graph, target, replay


def _data(*, edges: list[tuple[int, int]], anchors: list[int], targets: list[int]) -> RetrievalData:
    edge_index = _edge_tensor(edges)
    num_nodes = max([*anchors, *targets, *(node for edge in edges for node in edge)]) + 1
    labels = compute_path_labels(edge_index=edge_index, anchor_node_ids=torch.tensor(anchors), target_node_ids=torch.tensor(targets), num_nodes=num_nodes)
    bank = build_replay_bank(
        edge_index=edge_index, anchor_node_ids=torch.tensor(anchors), reachable_target_node_ids=labels.reachable_target_node_ids,
        num_nodes=num_nodes, sample_id="sample", max_budget=3, round_variants=2, trajectories_per_graph=4,
        beam_width=32, path_variants_per_pair=2, max_expansions_per_state=32, seed=42,
    )
    return RetrievalData(
        sample_id="sample", edge_index=edge_index, node_entity_catalog_ids=torch.arange(num_nodes),
        edge_relation_catalog_ids=torch.zeros(len(edges), dtype=torch.long), num_nodes=num_nodes, num_edges=len(edges),
        question_emb=torch.ones(4), anchor_node_ids=torch.tensor(anchors), target_node_ids=torch.tensor(targets),
        reachable_target_node_ids=labels.reachable_target_node_ids, node_target_distance=labels.node_target_distance,
        replay_bank=ReplayBankSample(edge_ids_local=bank.edge_ids, edge_count=bank.edge_count),
    )


def _edge_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _artifact(root: Path, *, kind: str) -> MaterializationArtifact:
    (root / "train").mkdir()
    return MaterializationArtifact(
        generation_id="test", manifest_path=root / "manifest.json", materialization_dir=root, catalog=root / "catalog.json",
        entity_text_semantic_table=write_table(root / "entity.pt", torch.zeros((1, 1))),
        entity_relation_neighborhood_semantic_table=write_table(root / "entity_relation_neighborhood.pt", torch.zeros((0, 1))),
        relation_semantic_table=write_table(root / "relation.pt", torch.zeros((1, 1))), question_texts=None,
        splits={"train": SplitArtifacts(lmdb=root / "train", question_embeddings=write_table(root / "questions.pt", torch.zeros((1, 4))), num_samples=1)},
        provenance={"preprocess": {"replay": {"kind": kind}}},
    )
