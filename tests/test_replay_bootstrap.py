from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import RetrievalCollator
from src.models.replay import OnlineReplayBuffer, TrajectoryTrace, residual_priority
from src.models.teacher_guidance import TeacherGuidance
from src.models.training_schedule import CurriculumSchedule
from src.utils.path_utils import compute_bounded_suffix_count


class _DummyEmbeddingStore:
    def __init__(self, *, embedding_dim: int) -> None:
        self._entity_embeddings = torch.arange(10 * embedding_dim, dtype=torch.float32).view(
            10, embedding_dim
        )
        self._relation_embeddings = torch.arange(
            10 * embedding_dim, dtype=torch.float32
        ).view(10, embedding_dim)

    def get_entity_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self._entity_embeddings.index_select(0, ids.long())

    def get_relation_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self._relation_embeddings.index_select(0, ids.long())


class _DummyDataResource:
    def __init__(self, *, embedding_dim: int) -> None:
        self.embedding_store = _DummyEmbeddingStore(embedding_dim=embedding_dim)
        self.entity_embedding_map = torch.arange(10, dtype=torch.long)
        self.cvt_mask = torch.zeros(10, dtype=torch.bool)


def _build_teacher_batch() -> Data:
    graph = Data(
        sample_id="sample-a",
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
        edge_relation_ids_global=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        node_entity_ids_global=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        question_emb=torch.ones(8, dtype=torch.float32),
        is_anchor_mask=torch.tensor([True, False, False, False]),
        is_target_mask=torch.tensor([False, False, False, True]),
        anchor_signed_distance=torch.tensor([0, 1, 1, 2], dtype=torch.long),
        answer_entity_ids_global=torch.tensor([3], dtype=torch.long),
        positive_edge_mask=torch.tensor([True, True, True, True]),
        node_to_target_distance=torch.tensor([2, 1, 1, 0], dtype=torch.long),
        shortest_suffix_count=torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        bounded_suffix_count=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        max_path_length=torch.tensor(2, dtype=torch.long),
    )
    data_resource = _DummyDataResource(embedding_dim=8)
    return RetrievalCollator(data_resource)([graph])


def test_teacher_guidance_marks_all_shortest_frontier_edges_valid() -> None:
    batch = _build_teacher_batch()
    state = type("StateLike", (), {})()
    state.active_nodes = batch.is_anchor_mask.clone()
    state.active_edges = torch.zeros(batch.edge_index.size(1), dtype=torch.bool)

    from src.models.policy import CandidateEdges

    candidates = CandidateEdges(
        edge_ids=torch.tensor([0, 1], dtype=torch.long),
        expand_logits=torch.tensor([0.1, 0.2], dtype=torch.float32),
        batch_index=torch.tensor([0, 0], dtype=torch.long),
    )
    guidance = TeacherGuidance(
        mode="bounded_path",
        score_exponent=0.5,
        undirected=False,
        fallback_to_policy=True,
    )

    valid_mask, scores = guidance.candidate_scores(
        base_graph=batch,
        state=state,
        candidates=candidates,
        remaining_expand_budget=1,
    )

    assert valid_mask.tolist() == [True, True]
    assert scores.tolist() == pytest.approx([1.0, 1.0])


def test_compute_bounded_suffix_count_respects_budget_axis() -> None:
    adjacency = [[1, 2], [3], [3], []]
    counts = compute_bounded_suffix_count(
        adjacency=adjacency,
        is_target_mask=torch.tensor([False, False, False, True]),
        budget_max_steps=2,
    )

    assert counts.shape == (3, 4)
    assert counts[0].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert counts[1].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert counts[2].tolist() == [2.0, 1.0, 1.0, 1.0]


def test_curriculum_schedule_decays_to_zero() -> None:
    schedule = CurriculumSchedule(
        warmup_steps=10,
        decay_steps=20,
        initial_teacher_prob=1.0,
        final_teacher_prob=0.0,
    )

    assert schedule.phase(0) == "warmup"
    assert schedule.teacher_force_prob(5) == pytest.approx(1.0)
    assert schedule.phase(15) == "mix"
    assert schedule.teacher_force_prob(20) == pytest.approx(0.5)
    assert schedule.phase(30) == "online"
    assert schedule.teacher_force_prob(30) == pytest.approx(0.0)


def test_online_replay_buffer_uses_round_robin_replacement() -> None:
    buffer = OnlineReplayBuffer(capacity=2)
    traces = [
        TrajectoryTrace(
            sample_id=f"sample-{idx}",
            edge_trace_local=(0,),
            traj_len=2,
            terminal_log_reward=1.0,
            priority=float(idx + 1),
            insert_step=idx,
        )
        for idx in range(3)
    ]

    buffer.add_many(traces)

    sampled = buffer.sample(8, importance_sampling_exponent=0.4)
    sampled_ids = {trace.sample_id for trace in sampled.traces}
    assert len(buffer) == 2
    assert sampled.indices.shape == (8,)
    assert sampled.importance_weights.shape == (8,)
    assert sampled_ids <= {"sample-1", "sample-2"}


def test_residual_priority_prefers_larger_tb_error() -> None:
    low = residual_priority(0.1, epsilon=1.0e-6, exponent=0.6)
    high = residual_priority(2.0, epsilon=1.0e-6, exponent=0.6)
    assert float(high.item()) > float(low.item())


def test_online_replay_buffer_updates_priorities_in_place() -> None:
    buffer = OnlineReplayBuffer(capacity=2)
    trace = TrajectoryTrace(
        sample_id="sample-a",
        edge_trace_local=(0,),
        traj_len=2,
        terminal_log_reward=1.0,
        priority=0.5,
        insert_step=0,
    )
    buffer.add_many([trace])
    buffer.update_priorities(torch.tensor([0]), torch.tensor([1.5]))

    replay_sample = buffer.sample(1)
    assert replay_sample.traces[0].priority == pytest.approx(1.5)
