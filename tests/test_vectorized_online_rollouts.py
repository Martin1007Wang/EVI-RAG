from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

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
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        if dim == 0:
            out_shape = (size,) + tuple(src.shape[1:])
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            for row, dest in enumerate(index.tolist()):
                out[dest] += src[row]
            return out
        if dim == 1:
            out_shape = (src.shape[0], size) + tuple(src.shape[2:])
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            for row in range(src.shape[0]):
                for col, dest in enumerate(index[row].tolist()):
                    out[row, dest] += src[row, col]
            return out
        raise NotImplementedError("test stub only supports dim=0 or dim=1")

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
        out = torch.full((size,), -float("inf"), dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

    def _scatter_logsumexp(
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
        out = torch.full((size,), -torch.inf, dtype=src.dtype, device=src.device)
        for dest in range(size):
            mask = index == dest
            if bool(mask.any().item()):
                out[dest] = torch.logsumexp(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.data.collate import RetrievalCollator
from src.data.schema import RetrievalData, repeat_retrieval_batch
from src.weaver.policy import CandidateEdges, PolicyStepOutput
from src.weaver.proposal import MinimalSufficiencyTeacher
from src.weaver.reward import RewardModel
from src.weaver.rollout.engine import RolloutEngine, mixed_top_k_candidates_by_graph
from src.weaver.rollout.sampling import option_action_log_probs, option_action_probs
from src.weaver.state import State


def _sample(question_scale: float) -> RetrievalData:
    return RetrievalData(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        node_entity_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        question_emb=torch.tensor([question_scale, 0.0], dtype=torch.float32),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([1], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
        anchor_node_forward_distances_flat=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor([0, 1], dtype=torch.long),
        node_target_distance=torch.tensor([1, 0], dtype=torch.long),
        target_node_distances_flat=torch.tensor([1, 0], dtype=torch.long),
        target_shortest_path_count_flat=torch.tensor([1.0, 1.0], dtype=torch.float32),
        target_shortest_path_edge_mask_flat=torch.tensor([True], dtype=torch.bool),
        non_text_node_mask=torch.tensor([False, True], dtype=torch.bool),
    )


def _batch():
    return RetrievalCollator()([_sample(1.0), _sample(2.0)])


def _three_node_batch():
    return RetrievalCollator()(
        [
            RetrievalData(
                num_nodes=3,
                edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
                node_entity_catalog_ids=torch.tensor([0, 1, 2], dtype=torch.long),
                edge_relation_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
                question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
                anchor_node_ids=torch.tensor([0], dtype=torch.long),
                target_node_ids=torch.tensor([1], dtype=torch.long),
                reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
                non_text_node_mask=torch.tensor([False, True, True], dtype=torch.bool),
            )
        ]
    )


def test_option_action_probs_separate_expand_mass_from_edge_count() -> None:
    candidates = CandidateEdges(
        edge_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        expand_logits=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
    )

    stop_prob, continue_prob, edge_prob = option_action_probs(
        stop_logits=torch.tensor([0.0, 0.0], dtype=torch.float32),
        expand_logits=torch.tensor([0.0, 0.0], dtype=torch.float32),
        candidates=candidates,
        can_expand=torch.tensor([True, False]),
        batch_size=2,
    )

    expected_edge_graph0 = torch.softmax(torch.tensor([1.0, 2.0]), dim=0)

    assert torch.allclose(stop_prob[0], torch.tensor(0.5))
    assert torch.allclose(continue_prob[0], torch.tensor(0.5))
    assert torch.allclose(edge_prob[:2], expected_edge_graph0)

    assert torch.allclose(stop_prob[1], torch.tensor(1.0))
    assert torch.allclose(continue_prob[1], torch.tensor(0.0))
    assert torch.allclose(edge_prob[2], torch.tensor(0.0))


def test_minimal_sufficiency_teacher_scores_marginal_reward_gain() -> None:
    batch = _three_node_batch()
    state = State.create_initial(batch, expand_budget=2)
    candidates = CandidateEdges(
        edge_ids=torch.tensor([0, 1], dtype=torch.long),
        expand_logits=torch.zeros(2, dtype=torch.float32),
        batch_index=torch.zeros(2, dtype=torch.long),
    )

    teacher = MinimalSufficiencyTeacher(gain_margin=0.02)
    context = teacher.build_context(
        retrieval_batch=batch,
        state=state,
        candidates=candidates,
        expand_budget_before_action=torch.tensor([2], dtype=torch.long),
        num_graphs=1,
        reward_model=RewardModel(),
    )

    decision = context.teacher_decision
    assert decision is not None
    assert not bool(decision.should_stop[0])
    assert int(decision.best_expand_edge[0].item()) == 0
    assert decision.expand_gain[0] > decision.expand_gain[1]
    assert bool(context.has_valid_expand[0])


def test_minimal_sufficiency_teacher_stops_when_expansion_lowers_reward() -> None:
    batch = _three_node_batch()
    state = State.create_initial(batch, expand_budget=2)
    state.apply_expansion(
        chosen_edges=torch.tensor([0], dtype=torch.long),
        edge_index=batch.edge_index,
    )
    candidates = CandidateEdges(
        edge_ids=torch.tensor([1], dtype=torch.long),
        expand_logits=torch.zeros(1, dtype=torch.float32),
        batch_index=torch.zeros(1, dtype=torch.long),
    )

    teacher = MinimalSufficiencyTeacher(gain_margin=0.02)
    context = teacher.build_context(
        retrieval_batch=batch,
        state=state,
        candidates=candidates,
        expand_budget_before_action=torch.tensor([1], dtype=torch.long),
        num_graphs=1,
        reward_model=RewardModel(),
    )

    decision = context.teacher_decision
    assert decision is not None
    assert bool(decision.should_stop[0])
    assert bool(context.stop_decision.should_stop[0])
    assert decision.best_expand_gain[0] <= 0.02


def test_option_action_log_probs_backward_with_forced_stop_graph() -> None:
    stop_logits = torch.tensor([0.2, -0.3], dtype=torch.float32, requires_grad=True)
    expand_logits = torch.tensor([0.7, 0.4], dtype=torch.float32, requires_grad=True)
    candidates = CandidateEdges(
        edge_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        expand_logits=torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float32,
            requires_grad=True,
        ),
        batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
    )

    type_logp, edge_logp = option_action_log_probs(
        stop_logits=stop_logits,
        expand_logits=expand_logits,
        candidates=candidates,
        can_expand=torch.tensor([True, False]),
        batch_size=2,
    )

    loss = type_logp[0, 0] + type_logp[0, 1] + edge_logp[:2].sum()
    loss.backward()

    assert stop_logits.grad is not None
    assert expand_logits.grad is not None
    assert candidates.expand_logits.grad is not None
    assert torch.allclose(type_logp[1], torch.tensor([0.0, -float("inf")]))


def test_mixed_top_k_candidates_combines_prior_final_and_random_sources() -> None:
    candidates = CandidateEdges(
        edge_ids=torch.tensor([10, 11, 12, 20], dtype=torch.long),
        expand_logits=torch.tensor([0.1, 4.0, 0.2, 9.0], dtype=torch.float32),
        batch_index=torch.tensor([0, 0, 0, 1], dtype=torch.long),
    )

    teacher_candidates, original_pos = mixed_top_k_candidates_by_graph(
        candidates=candidates,
        graph_mask=torch.tensor([True, False]),
        topk_prior=1,
        topk_final=1,
        random_k=1,
        num_graphs=2,
        prior_logits=torch.tensor([5.0, 1.0, 0.0, 7.0], dtype=torch.float32),
    )

    assert set(original_pos.tolist()) == {0, 1, 2}
    assert set(teacher_candidates.edge_ids.tolist()) == {10, 11, 12}
    assert torch.equal(teacher_candidates.batch_index, torch.zeros(3, dtype=torch.long))


def test_repeat_retrieval_batch_offsets_graph_structure_and_node_ids() -> None:
    batch = _batch()
    repeated = repeat_retrieval_batch(batch, 3)

    assert repeated.num_graphs == 6
    assert repeated.num_nodes_total == 12

    assert torch.equal(
        repeated.edge_index,
        torch.tensor(
            [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        repeated.ptr,
        torch.tensor([0, 2, 4, 6, 8, 10, 12], dtype=torch.long),
    )
    assert torch.equal(repeated.node_ptr, repeated.ptr)
    assert torch.equal(
        repeated.edge_ptr,
        torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long),
    )
    assert torch.equal(
        repeated.batch,
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.long),
    )
    assert torch.equal(repeated.edge_batch, torch.arange(6, dtype=torch.long))
    assert torch.equal(
        repeated.anchor_node_ids,
        torch.tensor([0, 2, 4, 6, 8, 10], dtype=torch.long),
    )
    assert torch.equal(
        repeated.reachable_target_node_ids,
        torch.tensor([1, 3, 5, 7, 9, 11], dtype=torch.long),
    )
    assert torch.equal(
        repeated.question_emb,
        torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        repeated.target_node_distances_flat,
        batch.target_node_distances_flat.repeat(3),
    )
    assert torch.equal(
        repeated.node_target_distance,
        batch.node_target_distance.repeat(3),
    )


class _FakeOnlinePolicy:
    def prepare_rollout_context(self, batch):
        del batch
        return object()

    def __call__(self, batch, state: State, rollout_context=None):
        del rollout_context

        device = batch.edge_index.device
        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        active_edges = state.active_edges.to(device=device, dtype=torch.bool)

        src, dst = edge_index
        candidate_mask = (
            active_nodes.index_select(0, src) | active_nodes.index_select(0, dst)
        ) & ~active_edges
        candidate_edge_ids = torch.nonzero(candidate_mask, as_tuple=False).view(-1)

        stop_logits = torch.full(
            (batch.num_graphs,),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        option_expand_logits = torch.zeros(
            batch.num_graphs,
            dtype=torch.float32,
            device=device,
        )
        remaining_budget = state.remaining_budget_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=int(batch.num_graphs),
        )
        expandable = remaining_budget.gt(0)
        stop_logits[~expandable] = 0.0

        state_log_flow = torch.zeros(
            batch.num_graphs, dtype=torch.float32, device=device
        )
        root_log_z = state_log_flow.clone() if state.is_root_state else None

        return PolicyStepOutput(
            stop_logits=stop_logits,
            expand_logits=option_expand_logits,
            edge_logits=torch.zeros(
                candidate_edge_ids.numel(),
                dtype=torch.float32,
                device=device,
            ),
            candidate_batch_ids=edge_batch.index_select(0, candidate_edge_ids),
            candidate_edge_ids=candidate_edge_ids,
            state_log_flow=state_log_flow,
            root_log_z=root_log_z,
        )


def test_run_online_vectorized_splits_rollouts_back_to_original_batch() -> None:
    batch = _batch()
    engine = RolloutEngine(expand_budget=1)

    rollouts = engine.run_online_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(edge_cost=0.0),
        num_rollouts=3,
        temperature=1.0,
    )

    assert len(rollouts) == 3
    for rollout in rollouts:
        assert rollout.stats.root_log_z.shape == (2,)
        assert rollout.stats.trajectory_length.shape == (2,)
        assert rollout.stats.terminal_log_reward.shape == (2,)
        assert rollout.traces.state_log_flows.shape == (2, 2)
        assert rollout.traces.log_pf.shape == (2, 2)
        assert rollout.traces.log_pb.shape == (2, 2)
        assert rollout.traces.action_type.shape == (2, 2)
        assert rollout.traces.continue_mask.shape == (2, 2)
        assert rollout.traces.stop_mask.shape == (2, 2)
        assert rollout.traces.stop_now_log_reward.shape == (2, 2)
        assert rollout.traces.stop_now_answer_f1.shape == (2, 2)
        assert rollout.traces.stop_now_valid_mask.shape == (2, 2)
        assert rollout.traces.stop_log_pf.shape == (2, 2)
        assert rollout.traces.stop_tb_valid_mask.shape == (2, 2)
        assert rollout.traces.target_stop_prob.shape == (2, 2)
        assert rollout.traces.target_continue_prob.shape == (2, 2)
        assert rollout.traces.policy_action_valid_mask.shape == (2, 2)
        assert rollout.traces.proposal_intervention_mask.shape == (2, 2)
        assert rollout.traces.selected_edge_ids.shape == (2, 2)
        assert torch.equal(
            rollout.stats.trajectory_length, torch.tensor([2, 2], dtype=torch.long)
        )
        assert torch.equal(
            rollout.traces.selected_edge_ids[:, 0],
            torch.tensor([0, 1], dtype=torch.long),
        )
        assert int(
            rollout.traces.selected_edge_ids.max().item()
        ) < batch.edge_index.size(1)
        assert bool(rollout.traces.stop_now_valid_mask[:, 0].all())
        assert bool(rollout.traces.stop_now_valid_mask[:, 1].all())
        assert bool(rollout.traces.stop_tb_valid_mask[:, 0].all())
        assert not bool(rollout.traces.stop_tb_valid_mask[:, 1].any())
        assert torch.allclose(
            rollout.traces.target_stop_prob[:, 0],
            torch.zeros(2, dtype=torch.float32),
        )
        assert torch.isneginf(rollout.traces.stop_log_pf[:, 0]).all()
        assert torch.allclose(
            rollout.traces.target_continue_prob[:, 0],
            torch.ones(2, dtype=torch.float32),
        )
        assert torch.allclose(
            rollout.traces.target_stop_prob[:, 1],
            torch.ones(2, dtype=torch.float32),
        )
        assert torch.allclose(
            rollout.traces.stop_log_pf[:, 1],
            torch.zeros(2, dtype=torch.float32),
        )
        assert torch.allclose(
            rollout.traces.target_continue_prob[:, 1],
            torch.zeros(2, dtype=torch.float32),
        )
        assert bool(rollout.traces.policy_action_valid_mask[:, 0].all())
        assert not bool(rollout.traces.policy_action_valid_mask[:, 1].any())
        assert torch.equal(
            rollout.stats.proposal_intervention_count,
            torch.zeros(2, dtype=torch.float32),
        )
        assert torch.equal(
            rollout.stats.edge_action_entropy,
            torch.zeros(2, dtype=torch.float32),
        )
        assert torch.equal(
            rollout.stats.edge_action_entropy_valid_mask,
            torch.ones(2, dtype=torch.float32),
        )
        assert rollout.stats.terminal_answer_f1.shape == (2,)
