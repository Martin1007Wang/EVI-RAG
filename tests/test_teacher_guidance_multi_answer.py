from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import torch
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if "torch_scatter" not in sys.modules:
    torch_scatter_stub = types.ModuleType("torch_scatter")

    def _scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out_shape = (size,) + tuple(src.shape[1:])
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] += src[row]
        return out

    def _scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

    def _scatter_min(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        argmin = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmin[dest] == -1 or src[row] < out[dest]:
                out[dest] = src[row]
                argmin[dest] = row
        return out, argmin

    def _scatter_softmax(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.zeros_like(src)
        for dest in range(size):
            mask = index == dest
            if bool(mask.any().item()):
                out[mask] = torch.softmax(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_min = _scatter_min
    torch_scatter_stub.scatter_softmax = _scatter_softmax
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.data.schema import RetrievalData
from src.eval.hit_graph_reward import evaluate_hit_graph_reward
from src.models.guidance import TeacherGuidance
from src.models.policy import CandidateEdges
from src.models.reward import RewardModel
from src.models.rollout.sampling import ActionSampler
from src.models.state import State
from src.utils.path_utils import compute_shortest_path_teacher_targets


def _build_multi_answer_batch() -> types.SimpleNamespace:
    edge_index = torch.tensor(
        [[0, 1, 1, 3], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    is_anchor_mask = torch.tensor([True, False, False, False, False], dtype=torch.bool)
    train_target_mask = torch.tensor([False, False, True, False, True], dtype=torch.bool)
    teacher_targets = compute_shortest_path_teacher_targets(
        edge_index=edge_index,
        is_anchor_mask=is_anchor_mask,
        target_mask=train_target_mask,
        num_nodes=5,
    )
    shortest_path_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    shortest_path_edge_mask[teacher_targets.positive_edge_ids] = True
    return types.SimpleNamespace(
        num_graphs=1,
        num_nodes=5,
        edge_index=edge_index,
        ptr=torch.tensor([0, 5], dtype=torch.long),
        edge_ptr=torch.tensor([0, 4], dtype=torch.long),
        batch=torch.zeros(5, dtype=torch.long),
        edge_batch=torch.zeros(4, dtype=torch.long),
        is_anchor_mask=is_anchor_mask,
        train_target_mask=train_target_mask,
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        relation_tokens=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=torch.float32,
        ),
        train_target_node_ids=teacher_targets.target_node_ids,
        shortest_path_edge_mask=shortest_path_edge_mask,
        node_to_target_distance=teacher_targets.node_to_target_distance,
        shortest_path_count=teacher_targets.shortest_path_count,
        target_node_distance_flat=teacher_targets.target_node_distance_flat,
        target_shortest_path_count_flat=teacher_targets.target_shortest_path_count_flat,
        target_shortest_path_edge_mask_flat=teacher_targets.target_shortest_path_edge_mask_flat,
    )


def _build_multi_answer_data() -> RetrievalData:
    batch = _build_multi_answer_batch()
    return RetrievalData(
        num_nodes=batch.num_nodes,
        edge_index=batch.edge_index,
        is_anchor_mask=batch.is_anchor_mask,
        train_target_mask=batch.train_target_mask,
        question_emb=batch.question_emb.reshape(-1),
        train_target_node_ids=batch.train_target_node_ids,
        target_node_distance_flat=batch.target_node_distance_flat,
        target_shortest_path_count_flat=batch.target_shortest_path_count_flat,
        target_shortest_path_edge_mask_flat=batch.target_shortest_path_edge_mask_flat,
    )


def test_compute_shortest_path_teacher_targets_keeps_per_target_labels() -> None:
    batch = _build_multi_answer_batch()
    dist = batch.target_node_distance_flat.view(2, 5)
    edge_mask = batch.target_shortest_path_edge_mask_flat.view(2, 4)

    assert torch.equal(
        batch.train_target_node_ids, torch.tensor([2, 4], dtype=torch.long)
    )
    assert torch.equal(
        dist,
        torch.tensor(
            [[2, 1, 0, -1, -1], [3, 2, -1, 1, 0]],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        edge_mask,
        torch.tensor(
            [[True, True, False, False], [True, False, True, True]],
            dtype=torch.bool,
        ),
    )


def test_batching_offsets_train_target_node_ids_per_graph() -> None:
    data_list = [_build_multi_answer_data(), _build_multi_answer_data()]
    batch = Batch.from_data_list(data_list)

    assert torch.equal(
        batch.train_target_node_ids,
        torch.tensor([2, 4, 7, 9], dtype=torch.long),
    )


def test_teacher_guidance_does_not_stop_after_first_hit_if_more_gold_is_reachable() -> (
    None
):
    batch = _build_multi_answer_batch()
    guidance = TeacherGuidance(score_exponent=1.0)
    state = State.create_initial(batch, expand_budget=4)
    state.apply_expansion(
        chosen_edges=torch.tensor([0, 1], dtype=torch.long),
        src=batch.edge_index[0],
        dst=batch.edge_index[1],
    )
    candidates = CandidateEdges(
        edge_ids=torch.tensor([2], dtype=torch.long),
        expand_logits=torch.zeros(1, dtype=torch.float32),
        batch_index=torch.zeros(1, dtype=torch.long),
    )

    should_stop = guidance.graph_should_stop(
        retrieval_batch=batch,
        state=state,
        candidates=candidates,
        remaining_expand_budget=1,
        num_graphs=1,
    )
    valid_mask, scores = guidance.candidate_scores(
        retrieval_batch=batch,
        state=state,
        candidates=candidates,
        remaining_expand_budget=1,
    )

    assert torch.equal(should_stop, torch.tensor([False]))
    assert torch.equal(valid_mask, torch.tensor([True]))
    assert torch.allclose(scores, torch.tensor([1.0], dtype=torch.float32))


def test_teacher_guidance_stops_when_remaining_budget_cannot_add_new_gold() -> None:
    batch = _build_multi_answer_batch()
    guidance = TeacherGuidance(score_exponent=1.0)
    state = State.create_initial(batch, expand_budget=4)
    state.apply_expansion(
        chosen_edges=torch.tensor([0, 1], dtype=torch.long),
        src=batch.edge_index[0],
        dst=batch.edge_index[1],
    )
    candidates = CandidateEdges(
        edge_ids=torch.tensor([2], dtype=torch.long),
        expand_logits=torch.zeros(1, dtype=torch.float32),
        batch_index=torch.zeros(1, dtype=torch.long),
    )

    should_stop = guidance.graph_should_stop(
        retrieval_batch=batch,
        state=state,
        candidates=candidates,
        remaining_expand_budget=0,
        num_graphs=1,
    )

    assert torch.equal(should_stop, torch.tensor([True]))


def test_action_sampler_teacher_forces_expand_when_new_gold_is_still_reachable() -> (
    None
):
    batch = _build_multi_answer_batch()
    guidance = TeacherGuidance(score_exponent=1.0)
    sampler = ActionSampler(
        teacher_guidance=guidance,
        teacher_force_prob=1.0,
        edge_ptr=batch.edge_ptr,
        batch_size=1,
        device=torch.device("cpu"),
        expand_budget=4,
    )
    state = State.create_initial(batch, expand_budget=4)
    state.apply_expansion(
        chosen_edges=torch.tensor([0, 1], dtype=torch.long),
        src=batch.edge_index[0],
        dst=batch.edge_index[1],
    )
    candidates = CandidateEdges(
        edge_ids=torch.tensor([2], dtype=torch.long),
        expand_logits=torch.zeros(1, dtype=torch.float32),
        batch_index=torch.zeros(1, dtype=torch.long),
    )
    behavior_logits = torch.tensor([[-3.0, 3.0]], dtype=torch.float32)
    target_logits = behavior_logits.clone()

    action_type, _ = sampler.sample_action_types(
        behavior_logits=behavior_logits,
        target_logits=target_logits,
        step_mask=torch.tensor([True]),
        num_expands=2,
        retrieval_batch=batch,
        state=state,
        candidates=candidates,
    )

    assert torch.equal(action_type, torch.tensor([0], dtype=torch.long))


def test_reward_model_prefers_full_answer_coverage_over_partial_hit() -> None:
    batch = _build_multi_answer_batch()
    reward_model = RewardModel(log_r_min=-5.0, zero_f1_edge_bonus_scale=0.0)

    partial_reward = reward_model(
        retrieval_batch=batch,
        active_nodes=torch.tensor([True, True, True, False, False], dtype=torch.bool),
        active_edges=torch.tensor([True, True, False, False], dtype=torch.bool),
    )
    full_reward = reward_model(
        retrieval_batch=batch,
        active_nodes=torch.tensor([True, True, True, True, True], dtype=torch.bool),
        active_edges=torch.tensor([True, True, True, True], dtype=torch.bool),
    )

    assert float(full_reward.item()) > float(partial_reward.item())
    assert math.isclose(float(partial_reward.item()), math.log(0.5), rel_tol=1e-6)
    assert math.isclose(float(full_reward.item()), math.log(2.0 / 3.0), rel_tol=1e-6)


def test_eval_hit_graph_reward_oracle_continues_past_first_answer() -> None:
    batch = _build_multi_answer_batch()
    reward_model = RewardModel(log_r_min=-5.0, zero_f1_edge_bonus_scale=0.0)

    result = evaluate_hit_graph_reward(
        batch,
        reward_model=reward_model,
        teacher_guidance=TeacherGuidance(score_exponent=1.0),
        expand_budget=4,
    )

    assert result.status == "ok"
    assert result.recall == 1.0
    assert result.added_edges == 4
    assert result.log_reward is not None
    assert math.isclose(result.log_reward, math.log(2.0 / 3.0), rel_tol=1e-6)
