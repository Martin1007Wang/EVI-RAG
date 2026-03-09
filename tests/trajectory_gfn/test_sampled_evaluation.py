from __future__ import annotations

import math

import pytest
import torch

from src.models.trajectory_gfn.analyzer import AnswerMassAnalysis
from src.models.trajectory_gfn.inference import AdaptivePosteriorInference
from src.models.trajectory_gfn.sampler import TrajectorySampleBatch

from .conftest import make_batch_from_graph


class _FakeSampler:
    def __init__(self, sample_batches: list[TrajectorySampleBatch]) -> None:
        self.sample_batches = list(sample_batches)
        self.calls = 0

    def sample(self, **kwargs) -> TrajectorySampleBatch:  # noqa: ANN003
        del kwargs
        sample_batch = self.sample_batches[self.calls]
        self.calls += 1
        return sample_batch


def _make_sample_batch(
    *, stop_node: int, edge_id: int, prob: float
) -> TrajectorySampleBatch:
    return TrajectorySampleBatch(
        graph_log_z=torch.zeros((1,), dtype=torch.float32),
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        start_log_probs=torch.zeros((1, 1), dtype=torch.float32),
        start_state_log_f=torch.zeros((1, 1), dtype=torch.float32),
        log_pf_steps=torch.tensor([[[math.log(prob), 0.0]]], dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        chosen_edge_ids_steps=torch.tensor([[[edge_id, -1]]], dtype=torch.long),
        active_steps=torch.tensor([[[True, True]]]),
        is_stop_steps=torch.tensor([[[False, True]]]),
        stop_nodes=torch.tensor([[stop_node]], dtype=torch.long),
        hit_mask=torch.tensor([[stop_node in {1, 2}]]),
        terminal_rewards=torch.ones((1, 1), dtype=torch.float32),
        terminal_log_rewards=torch.zeros((1, 1), dtype=torch.float32),
    )


def test_sampled_inference_stops_after_covering_selected_answer_support() -> None:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([10, 11, 12], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1, 2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 102, 102, 101], dtype=torch.long),
        sample_id="sampled-eval",
    )
    analysis = AnswerMassAnalysis(
        terminal_mass=torch.tensor([0.0, 0.3, 0.3, 0.4], dtype=torch.float32),
        answer_entity_ids=torch.tensor([101, 102], dtype=torch.long),
        answer_probs=torch.tensor([0.4, 0.6], dtype=torch.float32),
        gold_total_mass=0.6,
    )
    inference = AdaptivePosteriorInference(
        answer_mass_threshold=0.6,
        support_mass_threshold=0.9,
        rollout_chunk_size=1,
        max_rollouts=4,
        answer_top_ks=(1, 2),
    )
    sampler = _FakeSampler(
        [
            _make_sample_batch(stop_node=1, edge_id=0, prob=0.3),
            _make_sample_batch(stop_node=2, edge_id=1, prob=0.3),
        ]
    )

    result = inference.infer_sampled_graph(
        batch=batch,
        policy=None,
        context=None,
        sampler=sampler,
        analysis=analysis,
    )

    assert sampler.calls == 2
    assert result.selected_answer_ids == [102]
    assert result.stop_reason == "support_mass_reached"
    assert result.probe_count == 2
    assert result.window_size == 2
    assert result.covered_mass == pytest.approx(0.6)
    assert result.trajectories[0].prob == pytest.approx(0.3)
    assert result.trajectories[1].prob == pytest.approx(0.3)
    selected_answer = next(
        answer for answer in result.answer_posterior if answer.answer_entity_id == 102
    )
    assert selected_answer.is_selected is True
    assert selected_answer.support_conditioned_mass == pytest.approx(1.0)
