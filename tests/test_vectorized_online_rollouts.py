from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest
import torch

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
from src.data.schema import RetrievalData
from src.graph.segments import scatter_log_softmax, segment_topk_positions
from src.weaver.nn.frontier_pointer import FrontierPointerDiagnostics
from src.weaver.nn.feature_encoder import node_incidence
from src.weaver.policy import (
    Policy,
    PolicyOutput,
    hazard_policy_log_probs,
)
from src.weaver.reward import RewardModel
from src.weaver.rollout.executor import FusedStepExecutor
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.rollout.sampling import (
    ActionSample,
    CONTINUE_ACTION,
    STOP_ACTION,
    action_log_probs,
    action_probs,
    sample_action_for_generation,
    stop_continue_log_probs,
)
from src.weaver.state import RolloutState, State


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
        target_shortest_path_edge_count_flat=torch.tensor([1.0], dtype=torch.float32),
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


def _branching_batch():
    return RetrievalCollator()(
        [
            RetrievalData(
                num_nodes=4,
                edge_index=torch.tensor(
                    [
                        [0, 1, 1],
                        [1, 2, 3],
                    ],
                    dtype=torch.long,
                ),
                node_entity_catalog_ids=torch.arange(4, dtype=torch.long),
                edge_relation_catalog_ids=torch.arange(3, dtype=torch.long),
                question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
                anchor_node_ids=torch.tensor([0, 1], dtype=torch.long),
                target_node_ids=torch.tensor([2], dtype=torch.long),
                reachable_target_node_ids=torch.tensor([2], dtype=torch.long),
                non_text_node_mask=torch.tensor(
                    [False, True, True, True],
                    dtype=torch.bool,
                ),
            ),
            _sample(2.0),
        ]
    )


def test_action_probs_use_full_action_softmax() -> None:
    edge_logits = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    frontier_batch_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    stop_prob, continue_prob, edge_prob = action_probs(
        stop_logits=torch.tensor([0.0, 0.0], dtype=torch.float32),
        edge_logits=edge_logits,
        frontier_batch_ids=frontier_batch_ids,
        can_expand=torch.tensor([True, False]),
        batch_size=2,
    )

    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)

    assert torch.allclose(stop_prob[0], expected[0])
    assert torch.allclose(continue_prob[0], expected[1:].sum())
    assert torch.allclose(edge_prob[:2], expected[1:])

    assert torch.allclose(stop_prob[1], torch.tensor(1.0))
    assert torch.allclose(continue_prob[1], torch.tensor(0.0))
    assert torch.allclose(edge_prob[2], torch.tensor(0.0))


def test_policy_log_probs_normalize_learned_actions() -> None:
    stop_logits = torch.tensor([0.0, 1.0], dtype=torch.float32)
    edge_logits = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    frontier_batch_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    log_p_stop, log_p_continue, edge_cond_logprob, edge_expand_logprob = (
        hazard_policy_log_probs(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            frontier_batch_ids=frontier_batch_ids,
            num_graphs=2,
        )
    )

    local_action = torch.softmax(torch.tensor([0.0, 2.0, 3.0]), dim=0)
    assert torch.allclose(log_p_stop[0].exp(), local_action[0])
    assert torch.allclose(log_p_continue[0].exp(), local_action[1:].sum())
    assert torch.allclose(edge_expand_logprob[:2].exp(), local_action[1:])
    assert torch.allclose(edge_cond_logprob[:2].exp().sum(), torch.tensor(1.0))
    assert torch.allclose(
        log_p_stop.exp()
        + torch.stack(
            [
                edge_expand_logprob[:2].exp().sum(),
                edge_expand_logprob[2].exp(),
            ]
        ),
        torch.ones(2),
    )


def test_action_log_probs_masks_forced_stop_rows() -> None:
    stop_logp, edge_logp = action_log_probs(
        stop_logits=torch.tensor([0.0, 0.0], dtype=torch.float32),
        edge_logits=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        frontier_batch_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        can_expand=torch.tensor([True, False]),
        batch_size=2,
    )

    assert torch.allclose(stop_logp[1], torch.tensor(0.0))
    assert torch.isneginf(edge_logp[2])
    assert torch.allclose(
        stop_logp[0].exp() + edge_logp[:2].exp().sum(),
        torch.tensor(1.0),
    )


def test_action_log_probs_backward_with_forced_stop_graph() -> None:
    stop_logits = torch.tensor([0.2, -0.3], dtype=torch.float32, requires_grad=True)
    edge_logits = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    frontier_batch_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    stop_logp, edge_logp = action_log_probs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        frontier_batch_ids=frontier_batch_ids,
        can_expand=torch.tensor([True, False]),
        batch_size=2,
    )

    loss = stop_logp[0] + edge_logp[:2].sum()
    loss.backward()

    assert stop_logits.grad is not None
    assert edge_logits.grad is not None
    assert torch.allclose(stop_logp[1], torch.tensor(0.0))
    assert torch.isneginf(edge_logp[2])


def test_stop_continue_log_probs_exposes_continue_hazard() -> None:
    option_logp = stop_continue_log_probs(
        stop_logits=torch.tensor([0.5, -0.5], dtype=torch.float32),
        can_expand=torch.tensor([True, False]),
        batch_size=2,
    )
    _, edge_logp = action_log_probs(
        stop_logits=torch.tensor([0.5, -0.5], dtype=torch.float32),
        edge_logits=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        frontier_batch_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        can_expand=torch.tensor([True, False]),
        batch_size=2,
    )

    assert torch.allclose(
        option_logp[0, 0],
        torch.nn.functional.logsigmoid(torch.tensor(0.5)),
    )
    assert torch.allclose(
        option_logp[0, 1],
        torch.nn.functional.logsigmoid(torch.tensor(-0.5)),
    )
    assert torch.allclose(option_logp[1, 0], torch.tensor(0.0))
    assert torch.isneginf(option_logp[1, 1])
    assert torch.isneginf(edge_logp[2])


def test_scatter_log_softmax_matches_per_segment_reference() -> None:
    logits = torch.tensor(
        [1.0, -2.0, 0.5, 3.0, -1.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    segment_ids = torch.tensor([2, 0, 2, 0, 2], dtype=torch.long)

    log_probs = scatter_log_softmax(logits, segment_ids, num_segments=4)
    expected = torch.empty_like(logits)
    for segment_id in (0, 2):
        mask = segment_ids.eq(segment_id)
        expected[mask] = logits[mask] - torch.logsumexp(logits[mask], dim=0)

    assert torch.allclose(log_probs, expected)

    log_probs.sum().backward()
    assert logits.grad is not None


def test_segment_topk_positions_ignores_empty_segment_argmax_sentinel() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    segment_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    positions = segment_topk_positions(
        values=values,
        segment_ids=segment_ids,
        num_segments=4,
        k=2,
    )

    assert torch.equal(positions, torch.tensor([1, 2, 0], dtype=torch.long))


def test_sample_action_for_generation_uses_stop_then_expand_hazard() -> None:
    torch.manual_seed(4)

    sample = sample_action_for_generation(
        stop_logits=torch.tensor([-1000.0, 1000.0, 0.5], dtype=torch.float32),
        edge_logits=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        frontier_edge_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        frontier_batch_ids=torch.tensor([0, 0, 2], dtype=torch.long),
        active=torch.tensor([True, True, False]),
        can_expand=torch.tensor([True, False, False]),
        temperature=1.0,
        batch_size=3,
    )

    _, target_edge_logp = action_log_probs(
        stop_logits=torch.tensor([-1000.0, 1000.0, 0.5], dtype=torch.float32),
        edge_logits=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        frontier_batch_ids=torch.tensor([0, 0, 2], dtype=torch.long),
        can_expand=torch.tensor([True, False, False]),
        batch_size=3,
    )

    assert sample.action_type[0] == CONTINUE_ACTION
    chosen_edge = int(sample.chosen_edges[0].item())
    assert chosen_edge in {10, 11}
    chosen_pos = 0 if chosen_edge == 10 else 1
    assert torch.allclose(sample.target_log_prob[0], target_edge_logp[chosen_pos])

    assert sample.action_type[1] == STOP_ACTION
    assert sample.chosen_edges[1] == -1
    assert torch.allclose(sample.target_log_prob[1], torch.tensor(0.0))
    assert sample.action_type[2] == STOP_ACTION
    assert sample.chosen_edges[2] == -1


def test_sample_action_for_generation_samples_multiple_expand_segments() -> None:
    torch.manual_seed(0)

    sample = sample_action_for_generation(
        stop_logits=torch.full((3,), -1000.0, dtype=torch.float32),
        edge_logits=torch.tensor([0.0, 2.0, 1.0, 3.0, 4.0], dtype=torch.float32),
        frontier_edge_ids=torch.tensor([10, 11, 20, 21, 30], dtype=torch.long),
        frontier_batch_ids=torch.tensor([0, 0, 1, 1, 2], dtype=torch.long),
        active=torch.tensor([True, True, False]),
        can_expand=torch.tensor([True, True, False]),
        temperature=1.0,
        batch_size=3,
    )

    _, target_edge_logp = action_log_probs(
        stop_logits=torch.full((3,), -1000.0, dtype=torch.float32),
        edge_logits=torch.tensor([0.0, 2.0, 1.0, 3.0, 4.0], dtype=torch.float32),
        frontier_batch_ids=torch.tensor([0, 0, 1, 1, 2], dtype=torch.long),
        can_expand=torch.tensor([True, True, False]),
        batch_size=3,
    )

    assert torch.equal(
        sample.action_type,
        torch.tensor([CONTINUE_ACTION, CONTINUE_ACTION, STOP_ACTION]),
    )
    assert int(sample.chosen_edges[0].item()) in {10, 11}
    assert int(sample.chosen_edges[1].item()) in {20, 21}
    assert int(sample.chosen_edges[2].item()) == -1

    edge_to_pos = {10: 0, 11: 1, 20: 2, 21: 3}
    for graph_id in (0, 1):
        chosen = int(sample.chosen_edges[graph_id].item())
        assert torch.allclose(
            sample.target_log_prob[graph_id],
            target_edge_logp[edge_to_pos[chosen]],
        )
    assert torch.allclose(sample.target_log_prob[2], torch.tensor(0.0))


def test_executor_backward_log_pb_matches_canonical_local_parent_count() -> None:
    batch = _branching_batch()
    state = RolloutState.create_initial(
        batch,
        expand_budget=2,
        rollout_to_graph=torch.tensor([0, 0, 1], dtype=torch.long),
    )
    state.apply_expansion(
        rollout_ids=torch.tensor([0, 1], dtype=torch.long),
        chosen_edges=torch.tensor([1, 1], dtype=torch.long),
        edge_index=batch.edge_index,
    )
    chosen_edges = torch.tensor([2, 2, 3], dtype=torch.long)
    rollout_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    state.apply_expansion(
        rollout_ids=rollout_ids,
        chosen_edges=chosen_edges,
        edge_index=batch.edge_index,
    )

    executor = FusedStepExecutor(
        retrieval_batch=batch,
        reward_model=RewardModel(),
    )

    log_pb = executor._uniform_log_pb_after_continue(
        state=state,
        rollout_ids=rollout_ids,
        selected_edge_ids=chosen_edges,
    )

    assert torch.allclose(
        log_pb,
        torch.tensor(
            [
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_rollout_state_uses_sparse_traces_until_materialized() -> None:
    batch = _branching_batch()
    state = RolloutState.create_initial(
        batch,
        expand_budget=2,
        rollout_to_graph=torch.tensor([0, 0, 1], dtype=torch.long),
    )

    assert state._active_nodes is None
    assert state._active_edges is None
    assert state._root_edges is None
    assert state._anchor_nodes is None

    state.apply_expansion(
        rollout_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        chosen_edges=torch.tensor([1, 2, 3], dtype=torch.long),
        edge_index=batch.edge_index,
    )
    detached = state.detach()

    assert detached._active_nodes is None
    assert detached._active_edges is None
    assert detached._root_edges is None
    assert detached._anchor_nodes is None

    active_edges = detached.materialize_active_edges()
    active_nodes = detached.materialize_active_nodes(edge_index=batch.edge_index)
    assert torch.equal(
        active_edges,
        torch.tensor(
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, False, False, True],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        active_nodes,
        torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, False, True, False, False],
                [False, False, False, False, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    assert detached._active_nodes is None
    assert detached._active_edges is None


def test_engine_eager_stop_reward_uses_sparse_rollout_state() -> None:
    batch = _branching_batch()
    engine = RolloutEngine(expand_budget=1)

    captured: list[RolloutState] = []

    class _CaptureReward(RewardModel):
        def evaluate_terminal_state(self, **kwargs):
            state = kwargs.get("state")
            if isinstance(state, RolloutState):
                captured.append(state)
                assert kwargs.get("active_nodes") is None
                assert kwargs.get("active_edges") is None
                assert kwargs.get("diagnostics") == "basic"
            return super().evaluate_terminal_state(**kwargs)

    engine.run_vectorized(
        policy=_GradFakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=_CaptureReward(),
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )

    assert captured
    assert all(state._active_nodes is None for state in captured)
    assert all(state._active_edges is None for state in captured)


def test_engine_backward_survives_state_mutation_across_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _branching_batch()
    engine = RolloutEngine(expand_budget=2)
    live_policy = _StateIndexedGradPolicy()

    def _choose_first_frontier_edge(**kwargs):
        stop_logits = kwargs["stop_logits"]
        frontier_edge_ids = kwargs["frontier_edge_ids"]
        frontier_batch_ids = kwargs["frontier_batch_ids"]
        active = kwargs["active"]
        can_expand = kwargs["can_expand"]
        batch_size = int(kwargs["batch_size"])
        device = stop_logits.device

        action_type = torch.full(
            (batch_size,),
            STOP_ACTION,
            dtype=torch.long,
            device=device,
        )
        chosen_edges = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=device,
        )
        target_log_prob = torch.zeros(batch_size, dtype=torch.float32, device=device)
        for row in range(batch_size):
            if bool(active[row]) and bool(can_expand[row]):
                pos = frontier_batch_ids.eq(row).nonzero(as_tuple=False).flatten()[0]
                action_type[row] = CONTINUE_ACTION
                chosen_edges[row] = frontier_edge_ids[pos]
                target_log_prob[row] = stop_logits[row]
        return ActionSample(
            action_type=action_type,
            chosen_edges=chosen_edges,
            target_log_prob=target_log_prob,
        )

    monkeypatch.setattr(
        "src.weaver.rollout.engine.sample_action_for_generation",
        _choose_first_frontier_edge,
    )

    rollout = engine.run_vectorized(
        policy=live_policy,
        retrieval_batch=batch,
        reward_model=RewardModel(),
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )[0]

    loss = rollout.traces.state_log_flow.sum() + rollout.traces.log_pf.sum()
    loss.backward()

    assert live_policy.weight.grad is not None
    assert torch.isfinite(live_policy.weight.grad)


class _FakeOnlinePolicy:
    def prepare_rollout_context(self, batch):
        del batch
        return object()

    def __call__(self, batch, state: State, rollout_context=None, **kwargs):
        return_edge_diagnostics = bool(kwargs.get("return_edge_diagnostics", False))
        del rollout_context

        device = batch.edge_index.device
        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        active_edges = state.active_edges.to(device=device, dtype=torch.bool)

        src, dst = edge_index
        if active_nodes.ndim == 1:
            num_policy_graphs = int(batch.num_graphs)
            frontier_mask = (
                active_nodes.index_select(0, src) | active_nodes.index_select(0, dst)
            ) & ~active_edges
            frontier_edge_ids = torch.nonzero(frontier_mask, as_tuple=False).view(-1)
            frontier_batch_ids = edge_batch.index_select(0, frontier_edge_ids)
        else:
            num_policy_graphs = int(state.num_rollouts)
            rollout_to_graph = state.rollout_to_graph.to(
                device=device,
                dtype=torch.long,
            )
            belongs = edge_batch.view(1, -1).eq(rollout_to_graph.view(-1, 1))
            frontier_mask = (
                (active_nodes.index_select(1, src) | active_nodes.index_select(1, dst))
                & ~active_edges
                & belongs
            )
            frontier_batch_ids, frontier_edge_ids = frontier_mask.nonzero(
                as_tuple=True
            )

        stop_logits = torch.full(
            (num_policy_graphs,),
            -1000.0,
            dtype=torch.float32,
            device=device,
        )
        remaining_budget = state.remaining_budget_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=num_policy_graphs,
        )
        expandable = remaining_budget.gt(0)
        stop_logits[~expandable] = 0.0

        edge_logits = torch.zeros(
            frontier_edge_ids.numel(),
            dtype=torch.float32,
            device=device,
        )
        edge_diagnostics = None
        if return_edge_diagnostics:
            empty_edge_values = torch.zeros_like(edge_logits)
            edge_diagnostics = FrontierPointerDiagnostics(
                frontier_h=torch.zeros((edge_logits.numel(), 2), dtype=torch.float32),
                pointer_context=torch.zeros((num_policy_graphs, 2), dtype=torch.float32),
                query_relation_score=empty_edge_values,
                query_new_node_score=empty_edge_values,
                semantic_score=empty_edge_values,
                new_text_mask=torch.zeros_like(edge_logits, dtype=torch.bool),
                final_logits=edge_logits,
            )
        (
            log_p_stop,
            log_p_continue,
            edge_cond_logprob,
            edge_expand_logprob,
        ) = hazard_policy_log_probs(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            frontier_batch_ids=frontier_batch_ids,
            num_graphs=num_policy_graphs,
        )

        return PolicyOutput(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            state_log_flow=torch.zeros_like(stop_logits),
            log_p_stop=log_p_stop,
            log_p_continue=log_p_continue,
            edge_cond_logprob=edge_cond_logprob,
            edge_expand_logprob=edge_expand_logprob,
            frontier_batch_ids=frontier_batch_ids,
            frontier_edge_ids=frontier_edge_ids,
            edge_policy_diagnostics=edge_diagnostics,
        )


class _CountingFakeOnlinePolicy(_FakeOnlinePolicy):
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.prepare_num_graphs: list[int] = []

    def prepare_rollout_context(self, batch):
        self.prepare_calls += 1
        self.prepare_num_graphs.append(int(batch.num_graphs))
        return object()


class _SeenRowsPolicy(_FakeOnlinePolicy):
    def __init__(self) -> None:
        self.seen_rollout_to_graph: list[torch.Tensor] = []

    def __call__(self, batch, state: State, rollout_context=None, **kwargs):
        if isinstance(state, RolloutState):
            self.seen_rollout_to_graph.append(state.rollout_to_graph.detach().cpu())
        return super().__call__(
            batch,
            state,
            rollout_context=rollout_context,
            **kwargs,
        )


class _ConstantStopTargetPolicy(_FakeOnlinePolicy):
    def __init__(self, *, stop_logit: float) -> None:
        self.stop_logit = float(stop_logit)
        self.prepare_calls = 0
        self.prepared_contexts: list[object] = []
        self.seen_contexts: list[object] = []

    def prepare_rollout_context(self, batch):
        self.prepare_calls += 1
        context = object()
        self.prepared_contexts.append(context)
        return context

    def __call__(self, batch, state: State, rollout_context=None, **kwargs):
        self.seen_contexts.append(rollout_context)
        output = super().__call__(
            batch,
            state,
            rollout_context=rollout_context,
            **kwargs,
        )
        stop_logits = torch.full_like(output.stop_logits, self.stop_logit)
        (
            log_p_stop,
            log_p_continue,
            edge_cond_logprob,
            edge_expand_logprob,
        ) = hazard_policy_log_probs(
            stop_logits=stop_logits,
            edge_logits=output.edge_logits,
            frontier_batch_ids=output.frontier_batch_ids,
            num_graphs=int(stop_logits.numel()),
        )
        return PolicyOutput(
            stop_logits=stop_logits,
            edge_logits=output.edge_logits,
            state_log_flow=output.state_log_flow,
            log_p_stop=log_p_stop,
            log_p_continue=log_p_continue,
            edge_cond_logprob=edge_cond_logprob,
            edge_expand_logprob=edge_expand_logprob,
            frontier_batch_ids=output.frontier_batch_ids,
            frontier_edge_ids=output.frontier_edge_ids,
            edge_policy_diagnostics=output.edge_policy_diagnostics,
        )


class _GradFakeOnlinePolicy(_FakeOnlinePolicy):
    def __call__(self, batch, state: State, rollout_context=None, **kwargs):
        output = super().__call__(
            batch,
            state,
            rollout_context=rollout_context,
            **kwargs,
        )
        stop_logits = output.stop_logits.detach().clone().requires_grad_()
        edge_logits = output.edge_logits.detach().clone().requires_grad_()
        (
            log_p_stop,
            log_p_continue,
            edge_cond_logprob,
            edge_expand_logprob,
        ) = hazard_policy_log_probs(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            frontier_batch_ids=output.frontier_batch_ids,
            num_graphs=int(stop_logits.numel()),
        )
        return PolicyOutput(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            state_log_flow=output.state_log_flow.detach().clone().requires_grad_(),
            log_p_stop=log_p_stop,
            log_p_continue=log_p_continue,
            edge_cond_logprob=edge_cond_logprob,
            edge_expand_logprob=edge_expand_logprob,
            frontier_batch_ids=output.frontier_batch_ids,
            frontier_edge_ids=output.frontier_edge_ids,
            edge_policy_diagnostics=output.edge_policy_diagnostics,
        )


class _StateIndexedGradPolicy(_FakeOnlinePolicy):
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def __call__(self, batch, state: State, rollout_context=None, **kwargs):
        output = super().__call__(
            batch,
            state,
            rollout_context=rollout_context,
            **kwargs,
        )
        rollout_ids = torch.arange(
            int(state.num_rollouts),
            dtype=torch.long,
            device=batch.edge_index.device,
        )
        _, lengths = state.expanded_edge_trace_for_rollouts_tensor(rollout_ids)
        state_log_flow = lengths.to(dtype=torch.float32) * self.weight
        stop_logits = state_log_flow - 1000.0
        edge_logits = torch.zeros_like(output.edge_logits) + self.weight * 0.0
        (
            log_p_stop,
            log_p_continue,
            edge_cond_logprob,
            edge_expand_logprob,
        ) = hazard_policy_log_probs(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            frontier_batch_ids=output.frontier_batch_ids,
            num_graphs=int(stop_logits.numel()),
        )
        return PolicyOutput(
            stop_logits=stop_logits,
            edge_logits=edge_logits,
            state_log_flow=state_log_flow,
            log_p_stop=log_p_stop,
            log_p_continue=log_p_continue,
            edge_cond_logprob=edge_cond_logprob,
            edge_expand_logprob=edge_expand_logprob,
            frontier_batch_ids=output.frontier_batch_ids,
            frontier_edge_ids=output.frontier_edge_ids,
            edge_policy_diagnostics=output.edge_policy_diagnostics,
        )


class _CountingRewardModel(RewardModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.evaluate_calls = 0
        self.evaluated_num_rollouts: list[int] = []

    def evaluate_terminal_state(self, **kwargs):
        self.evaluate_calls += 1
        rollout_ids = kwargs.get("rollout_ids")
        if isinstance(rollout_ids, torch.Tensor):
            self.evaluated_num_rollouts.append(int(rollout_ids.numel()))
            return super().evaluate_terminal_state(**kwargs)
        state = kwargs.get("state")
        if isinstance(state, RolloutState):
            self.evaluated_num_rollouts.append(state.num_rollouts)
        return super().evaluate_terminal_state(**kwargs)


def test_run_online_vectorized_splits_rollouts_back_to_original_batch() -> None:
    batch = _batch()
    engine = RolloutEngine(expand_budget=1)

    rollouts = engine.run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(),
        num_rollouts=3,
        temperature=1.0,
    )

    assert len(rollouts) == 3
    for rollout in rollouts:
        assert rollout.stats.trajectory_length.shape == (2,)
        assert rollout.stats.terminal_log_reward.shape == (2,)
        assert rollout.traces.log_pf.shape == (2, 2)
        assert rollout.traces.log_pb.shape == (2, 2)
        assert rollout.traces.db_valid_mask.shape == (2, 2)
        assert rollout.traces.db_log_pf_expand.shape == (2, 2)
        assert rollout.traces.action_type.shape == (2, 2)
        assert rollout.traces.continue_mask.shape == (2, 2)
        assert rollout.traces.stop_mask.shape == (2, 2)
        assert rollout.traces.stop_now_log_reward is None
        assert rollout.traces.stop_now_answer_f1 is None
        assert rollout.traces.stop_now_valid_mask is None
        assert rollout.traces.target_stop_prob.shape == (2, 2)
        assert rollout.traces.target_continue_prob.shape == (2, 2)
        assert rollout.traces.policy_action_valid_mask.shape == (2, 2)
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
        assert bool(rollout.traces.policy_action_valid_mask[:, 0].all())
        assert not bool(rollout.traces.policy_action_valid_mask[:, 1].any())
        assert bool(rollout.traces.policy_action_valid_mask[:, 0].all())
        assert not bool(rollout.traces.policy_action_valid_mask[:, 1].any())
        assert torch.equal(
            rollout.stats.edge_action_entropy,
            torch.zeros(2, dtype=torch.float32),
        )
        assert torch.equal(
            rollout.stats.edge_action_count,
            torch.zeros(2, dtype=torch.float32),
        )
        assert rollout.stats.terminal_answer_f1.shape == (2,)


def test_subtb_records_terminal_state_flow_when_expand_reaches_budget() -> None:
    batch = _batch()

    rollout = RolloutEngine(expand_budget=1).run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(),
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )[0]

    assert bool(rollout.traces.continue_mask[:, 0].all())
    assert bool(rollout.traces.stop_mask[:, 1].all())
    assert rollout.traces.state_log_flow.shape == rollout.traces.log_pf.shape
    assert torch.allclose(
        rollout.traces.state_log_flow[:, :2],
        torch.zeros(2, dtype=torch.float32),
    )


def test_subtb_records_terminal_state_flow_when_frontier_is_empty() -> None:
    batch = _batch()

    rollout = RolloutEngine(expand_budget=2).run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(),
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )[0]

    assert bool(rollout.traces.continue_mask[:, 0].all())
    assert bool(rollout.traces.stop_mask.any())
    assert rollout.traces.state_log_flow.shape == rollout.traces.log_pf.shape
    assert torch.allclose(
        rollout.traces.state_log_flow[
            rollout.traces.stop_mask
        ],
        torch.zeros_like(rollout.traces.state_log_flow[rollout.traces.stop_mask]),
    )


def test_subtb_log_pf_expand_stores_full_expand_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _branching_batch()
    captured: dict[str, torch.Tensor] = {}

    class _NondegenerateHazardPolicy(_FakeOnlinePolicy):
        def __call__(self, batch, state: State, rollout_context=None, **kwargs):
            output = super().__call__(
                batch,
                state,
                rollout_context=rollout_context,
                **kwargs,
            )
            stop_logits = torch.zeros_like(output.stop_logits)
            edge_logits = torch.arange(
                output.edge_logits.numel(),
                dtype=output.edge_logits.dtype,
                device=output.edge_logits.device,
            )
            (
                log_p_stop,
                log_p_continue,
                edge_cond_logprob,
                edge_expand_logprob,
            ) = hazard_policy_log_probs(
                stop_logits=stop_logits,
                edge_logits=edge_logits,
                frontier_batch_ids=output.frontier_batch_ids,
                num_graphs=int(stop_logits.numel()),
            )
            return PolicyOutput(
                stop_logits=stop_logits,
                edge_logits=edge_logits,
                state_log_flow=output.state_log_flow,
                log_p_stop=log_p_stop,
                log_p_continue=log_p_continue,
                edge_cond_logprob=edge_cond_logprob,
                edge_expand_logprob=edge_expand_logprob,
                frontier_batch_ids=output.frontier_batch_ids,
                frontier_edge_ids=output.frontier_edge_ids,
                edge_policy_diagnostics=output.edge_policy_diagnostics,
            )

    def forced_first_row_continue(**kwargs) -> ActionSample:
        step_out = PolicyOutput(
            stop_logits=kwargs["stop_logits"],
            edge_logits=kwargs["edge_logits"],
            state_log_flow=torch.zeros_like(kwargs["stop_logits"]),
            log_p_stop=torch.nn.functional.logsigmoid(kwargs["stop_logits"]),
            log_p_continue=torch.nn.functional.logsigmoid(-kwargs["stop_logits"]),
            edge_cond_logprob=torch.empty(0),
            edge_expand_logprob=torch.empty(0),
            frontier_edge_ids=kwargs["frontier_edge_ids"],
            frontier_batch_ids=kwargs["frontier_batch_ids"],
        )
        (
            _,
            _,
            edge_cond_logprob,
            edge_expand_logprob,
        ) = hazard_policy_log_probs(
            stop_logits=step_out.stop_logits,
            edge_logits=step_out.edge_logits,
            frontier_batch_ids=step_out.frontier_batch_ids,
            num_graphs=int(kwargs["batch_size"]),
        )

        active = kwargs["active"].to(dtype=torch.bool)
        can_expand = kwargs["can_expand"].to(dtype=torch.bool)
        batch_size = int(kwargs["batch_size"])
        action_type = torch.full(
            (batch_size,),
            STOP_ACTION,
            dtype=torch.long,
            device=step_out.stop_logits.device,
        )
        chosen_edges = torch.full_like(action_type, -1)
        target_log_prob = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=step_out.stop_logits.device,
        )
        expandable = active & can_expand
        target_log_prob[expandable] = step_out.log_p_stop[expandable]

        row0_positions = step_out.frontier_batch_ids.eq(0).nonzero(
            as_tuple=False
        ).view(-1)
        if bool(expandable[0]) and row0_positions.numel() > 0:
            selected_pos = row0_positions[0]
            action_type[0] = CONTINUE_ACTION
            chosen_edges[0] = step_out.frontier_edge_ids[selected_pos]
            target_log_prob[0] = edge_expand_logprob[selected_pos]
            captured["full"] = edge_expand_logprob[selected_pos].detach()
            captured["conditional"] = edge_cond_logprob[selected_pos].detach()

        return ActionSample(
            action_type=action_type,
            chosen_edges=chosen_edges,
            target_log_prob=target_log_prob,
        )

    monkeypatch.setattr(
        "src.weaver.rollout.engine.sample_action_for_generation",
        forced_first_row_continue,
    )

    rollout = RolloutEngine(expand_budget=1).run_vectorized(
        policy=_NondegenerateHazardPolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(),
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )[0]

    assert bool(rollout.traces.continue_mask[0, 0])
    stored = rollout.traces.log_pf[0, 0]
    assert torch.allclose(stored, captured["full"], atol=1e-6)
    assert not torch.allclose(stored, captured["conditional"], atol=1e-6)


def test_subtb_expand_trace_keeps_online_gradient() -> None:
    batch = _branching_batch()

    rollout = RolloutEngine(expand_budget=2).run_vectorized(
        policy=_GradFakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(),
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )[0]

    assert bool(rollout.traces.continue_mask.any())
    assert rollout.traces.log_pf.requires_grad
    assert rollout.traces.state_log_flow.requires_grad


def test_subtb_rollout_rejects_target_policy_argument() -> None:
    batch = _branching_batch()
    target_policy = _ConstantStopTargetPolicy(stop_logit=2.0)

    with pytest.raises(TypeError, match="target_policy"):
        RolloutEngine(expand_budget=2).run_vectorized(
            policy=_GradFakeOnlinePolicy(),
            target_policy=target_policy,
            retrieval_batch=batch,
            reward_model=RewardModel(),
            num_rollouts=1,
            temperature=1.0,
            collect_policy_diagnostics=False,
        )

    assert target_policy.prepare_calls == 0
    assert target_policy.seen_contexts == []


def test_fused_static_batch_rollouts_reuse_context_and_split_logical_rollouts() -> None:
    batch = _batch()
    fused_policy = _CountingFakeOnlinePolicy()

    fused_rollouts = RolloutEngine(expand_budget=1).run_vectorized(
        policy=fused_policy,
        retrieval_batch=batch,
        reward_model=RewardModel(),
        num_rollouts=3,
        temperature=1.0,
    )

    assert fused_policy.prepare_calls == 1
    assert fused_policy.prepare_num_graphs == [2]
    assert len(fused_rollouts) == 3
    for fused in fused_rollouts:
        assert torch.equal(
            fused.stats.trajectory_length,
            torch.tensor([2, 2], dtype=torch.long),
        )
        assert torch.allclose(
            fused.stats.terminal_log_reward,
            torch.log(torch.full((2,), 0.01 + 1.0, dtype=torch.float32)) - 0.1,
            atol=1.0e-7,
        )
        assert torch.equal(
            fused.traces.action_type,
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        )
        assert torch.equal(
            fused.traces.selected_edge_ids[:, 0],
            torch.tensor([0, 1], dtype=torch.long),
        )
        assert int(fused.traces.selected_edge_ids.max().item()) < batch.edge_index.size(
            1
        )
        assert bool(fused.traces.stop_mask[:, 1].all())


def test_rollout_engine_policy_skips_rows_that_already_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _branching_batch()
    policy = _SeenRowsPolicy()

    def stop_second_row_after_first_row_continues(**kwargs) -> ActionSample:
        stop_logits = kwargs["stop_logits"]
        frontier_edge_ids = kwargs["frontier_edge_ids"]
        frontier_batch_ids = kwargs["frontier_batch_ids"]
        active = kwargs["active"].to(dtype=torch.bool)
        can_expand = kwargs["can_expand"].to(dtype=torch.bool)
        batch_size = int(kwargs["batch_size"])
        device = stop_logits.device

        action_type = torch.full(
            (batch_size,),
            STOP_ACTION,
            dtype=torch.long,
            device=device,
        )
        chosen_edges = torch.full_like(action_type, -1)
        target_log_prob = torch.zeros(batch_size, dtype=torch.float32, device=device)

        if bool(active[0]) and bool(can_expand[0]):
            pos = frontier_batch_ids.eq(0).nonzero(as_tuple=False).flatten()[0]
            action_type[0] = CONTINUE_ACTION
            chosen_edges[0] = frontier_edge_ids[pos]

        return ActionSample(
            action_type=action_type,
            chosen_edges=chosen_edges,
            target_log_prob=target_log_prob,
        )

    monkeypatch.setattr(
        "src.weaver.rollout.engine.sample_action_for_generation",
        stop_second_row_after_first_row_continues,
    )

    RolloutEngine(expand_budget=1).run_vectorized(
        policy=policy,
        retrieval_batch=batch,
        reward_model=RewardModel(),
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )

    assert len(policy.seen_rollout_to_graph) == 2
    assert torch.equal(policy.seen_rollout_to_graph[0], torch.tensor([0, 1]))
    assert torch.equal(policy.seen_rollout_to_graph[1], torch.tensor([0]))


def test_rollout_engine_stop_reward_evaluates_only_stopping_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _branching_batch()
    reward_model = _CountingRewardModel()

    def stop_second_row_after_first_row_continues(**kwargs) -> ActionSample:
        stop_logits = kwargs["stop_logits"]
        frontier_edge_ids = kwargs["frontier_edge_ids"]
        frontier_batch_ids = kwargs["frontier_batch_ids"]
        active = kwargs["active"].to(dtype=torch.bool)
        can_expand = kwargs["can_expand"].to(dtype=torch.bool)
        batch_size = int(kwargs["batch_size"])
        device = stop_logits.device

        action_type = torch.full(
            (batch_size,),
            STOP_ACTION,
            dtype=torch.long,
            device=device,
        )
        chosen_edges = torch.full_like(action_type, -1)
        target_log_prob = torch.zeros(batch_size, dtype=torch.float32, device=device)

        if bool(active[0]) and bool(can_expand[0]):
            pos = frontier_batch_ids.eq(0).nonzero(as_tuple=False).flatten()[0]
            action_type[0] = CONTINUE_ACTION
            chosen_edges[0] = frontier_edge_ids[pos]

        return ActionSample(
            action_type=action_type,
            chosen_edges=chosen_edges,
            target_log_prob=target_log_prob,
        )

    monkeypatch.setattr(
        "src.weaver.rollout.engine.sample_action_for_generation",
        stop_second_row_after_first_row_continues,
    )

    RolloutEngine(expand_budget=1).run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=reward_model,
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )

    assert reward_model.evaluated_num_rollouts == [1, 1]


def test_policy_forward_uses_rollout_ids_and_static_query_ids_for_fused_state() -> None:
    batch = _batch()
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.eye(2, dtype=torch.float32),
            "entity_embedding_map": torch.tensor([0, 1], dtype=torch.long),
            "relation_embeddings": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "dde": {"enabled": False},
        },
    )
    rollout_to_graph = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    state = RolloutState.create_initial(
        batch,
        expand_budget=1,
        rollout_to_graph=rollout_to_graph,
    )
    context = policy.prepare_rollout_context(batch)

    output = policy(
        batch,
        state,
        rollout_context=context,
    )

    assert context.fb.query_h.shape == (2, 2)
    assert torch.equal(
        output.frontier_batch_ids,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    assert torch.equal(
        output.frontier_edge_ids,
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )
    assert output.edge_logits.shape == (4,)


def test_policy_continuation_bias_shifts_log_c() -> None:
    batch = _batch()
    policy = Policy(
        hidden_dim=2,
        continuation_logit_bias_init=0.0,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.eye(2, dtype=torch.float32),
            "entity_embedding_map": torch.tensor([0, 1], dtype=torch.long),
            "relation_embeddings": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "dde": {"enabled": False},
        },
    )
    state = State.create_initial(batch, expand_budget=1)
    context = policy.prepare_rollout_context(batch)

    with torch.no_grad():
        policy.continuation_logit_bias.fill_(0.0)
    base = policy(batch, state, rollout_context=context)

    with torch.no_grad():
        policy.continuation_logit_bias.fill_(-2.0)
    calibrated = policy(batch, state, rollout_context=context)

    assert base.log_c_continue is not None
    assert calibrated.log_c_continue is not None
    assert torch.allclose(calibrated.edge_logits, base.edge_logits - 2.0)
    assert torch.allclose(calibrated.log_c_continue, base.log_c_continue - 2.0)
    assert bool((calibrated.log_p_stop > base.log_p_stop).all())


def test_policy_outputs_are_invariant_to_expanded_edge_trace_order() -> None:
    batch = _branching_batch()
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.eye(6, 2, dtype=torch.float32),
            "entity_embedding_map": torch.arange(6, dtype=torch.long),
            "relation_embeddings": torch.eye(5, 2, dtype=torch.float32),
            "dde": {"enabled": False},
        },
    )
    context = policy.prepare_rollout_context(batch)
    node_incident_edge_ids, node_incident_ptr = node_incidence(
        edge_index=batch.edge_index,
        num_nodes=int(batch.num_nodes_total),
    )
    context = dataclasses.replace(
        context,
        fb=dataclasses.replace(
            context.fb,
            node_incident_edge_ids=node_incident_edge_ids,
            node_incident_ptr=node_incident_ptr,
        ),
    )
    state_a = RolloutState.create_initial(
        batch,
        expand_budget=3,
        rollout_to_graph=torch.tensor([0], dtype=torch.long),
    )
    state_b = RolloutState.create_initial(
        batch,
        expand_budget=3,
        rollout_to_graph=torch.tensor([0], dtype=torch.long),
    )
    edge_index = batch.edge_index.to(dtype=torch.long)
    state_a.apply_expansion(
        rollout_ids=torch.tensor([0], dtype=torch.long),
        chosen_edges=torch.tensor([0], dtype=torch.long),
        edge_index=edge_index,
    )
    state_a.apply_expansion(
        rollout_ids=torch.tensor([0], dtype=torch.long),
        chosen_edges=torch.tensor([1], dtype=torch.long),
        edge_index=edge_index,
    )
    state_b.apply_expansion(
        rollout_ids=torch.tensor([0], dtype=torch.long),
        chosen_edges=torch.tensor([1], dtype=torch.long),
        edge_index=edge_index,
    )
    state_b.apply_expansion(
        rollout_ids=torch.tensor([0], dtype=torch.long),
        chosen_edges=torch.tensor([0], dtype=torch.long),
        edge_index=edge_index,
    )

    out_a = policy(batch, state_a, rollout_context=context)
    out_b = policy(batch, state_b, rollout_context=context)

    assert torch.equal(out_a.frontier_edge_ids, out_b.frontier_edge_ids)
    assert torch.equal(out_a.frontier_batch_ids, out_b.frontier_batch_ids)
    assert torch.allclose(out_a.state_log_flow, out_b.state_log_flow)
    assert torch.allclose(out_a.stop_logits, out_b.stop_logits)
    assert torch.allclose(out_a.edge_logits, out_b.edge_logits)


def test_direct_policy_uses_learned_stop_without_explicit_reward() -> None:
    batch = _three_node_batch()
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.eye(3, 2, dtype=torch.float32),
            "entity_embedding_map": torch.tensor([0, 1, 2], dtype=torch.long),
            "relation_embeddings": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            ),
            "dde": {"enabled": False},
        },
    )
    state = State.create_initial(batch, expand_budget=2)
    reward = RewardModel().evaluate_terminal_state(
        retrieval_batch=batch,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        state=state,
    )

    output = policy(
        batch,
        state,
        rollout_context=policy.prepare_rollout_context(batch),
    )

    del reward
    assert output.state_log_flow.shape == (1,)
    assert not hasattr(output, "log_terminal_mass")
    assert output.edge_logits.shape == (2,)
    assert output.frontier_edge_ids.shape == (2,)


def test_rollout_engine_skips_stop_now_traces_unless_requested() -> (
    None
):
    batch = _batch()
    engine = RolloutEngine(expand_budget=1)
    reward_model = _CountingRewardModel()

    rollout = engine.run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=reward_model,
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
    )[0]

    assert reward_model.evaluate_calls == 1
    assert rollout.traces.stop_now_valid_mask is None
    assert bool(rollout.traces.policy_action_valid_mask[:, 0].all())

def test_rollout_engine_writes_stop_now_traces_when_requested() -> None:
    batch = _three_node_batch()
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.eye(3, 2, dtype=torch.float32),
            "entity_embedding_map": torch.tensor([0, 1, 2], dtype=torch.long),
            "relation_embeddings": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            ),
            "dde": {"enabled": False},
        },
    )

    reward_model = _CountingRewardModel()
    rollout = RolloutEngine(expand_budget=1).run_vectorized(
        policy=policy,
        retrieval_batch=batch,
        reward_model=reward_model,
        num_rollouts=1,
        temperature=1.0,
        collect_policy_diagnostics=False,
        store_stop_now_reward=True,
    )[0]

    assert reward_model.evaluate_calls >= 1
    assert rollout.traces.stop_now_log_reward is not None
    assert rollout.traces.stop_now_answer_f1 is not None
    assert rollout.traces.stop_now_valid_mask is not None
