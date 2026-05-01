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
from src.weaver.policy import Policy, PolicyOutput
from src.weaver.reward import RewardModel
from src.weaver.rollout.engine import RewardMode, RolloutEngine
from src.weaver.rollout.stop_advantage import (
    StopAdvantageAuxiliary,
    StopAdvantageConfig,
)
from src.weaver.rollout.sampling import option_action_log_probs, option_action_probs
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
    edge_logits = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    candidate_batch_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    stop_prob, continue_prob, edge_prob = option_action_probs(
        stop_logits=torch.tensor([0.0, 0.0], dtype=torch.float32),
        expand_logits=torch.tensor([0.0, 0.0], dtype=torch.float32),
        edge_logits=edge_logits,
        candidate_batch_ids=candidate_batch_ids,
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


def test_option_action_log_probs_backward_with_forced_stop_graph() -> None:
    stop_logits = torch.tensor([0.2, -0.3], dtype=torch.float32, requires_grad=True)
    expand_logits = torch.tensor([0.7, 0.4], dtype=torch.float32, requires_grad=True)
    edge_logits = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    candidate_batch_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    type_logp, edge_logp = option_action_log_probs(
        stop_logits=stop_logits,
        expand_logits=expand_logits,
        edge_logits=edge_logits,
        candidate_batch_ids=candidate_batch_ids,
        can_expand=torch.tensor([True, False]),
        batch_size=2,
    )

    loss = type_logp[0, 0] + type_logp[0, 1] + edge_logp[:2].sum()
    loss.backward()

    assert stop_logits.grad is not None
    assert expand_logits.grad is not None
    assert edge_logits.grad is not None
    assert torch.allclose(type_logp[1], torch.tensor([0.0, -float("inf")]))


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

    def __call__(self, batch, state: State, rollout_context=None, **kwargs):
        del kwargs
        del rollout_context

        device = batch.edge_index.device
        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        active_edges = state.active_edges.to(device=device, dtype=torch.bool)

        src, dst = edge_index
        if active_nodes.ndim == 1:
            num_policy_graphs = int(batch.num_graphs)
            candidate_mask = (
                active_nodes.index_select(0, src) | active_nodes.index_select(0, dst)
            ) & ~active_edges
            candidate_edge_ids = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
            candidate_batch_ids = edge_batch.index_select(0, candidate_edge_ids)
        else:
            num_policy_graphs = int(state.num_rollouts)
            rollout_to_graph = state.rollout_to_graph.to(
                device=device,
                dtype=torch.long,
            )
            belongs = edge_batch.view(1, -1).eq(rollout_to_graph.view(-1, 1))
            candidate_mask = (
                (active_nodes.index_select(1, src) | active_nodes.index_select(1, dst))
                & ~active_edges
                & belongs
            )
            candidate_batch_ids, candidate_edge_ids = candidate_mask.nonzero(
                as_tuple=True
            )

        stop_logits = torch.full(
            (num_policy_graphs,),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        option_expand_logits = torch.zeros(
            num_policy_graphs,
            dtype=torch.float32,
            device=device,
        )
        remaining_budget = state.remaining_budget_per_graph(
            edge_batch=batch.edge_batch,
            num_graphs=num_policy_graphs,
        )
        expandable = remaining_budget.gt(0)
        stop_logits[~expandable] = 0.0

        state_log_flow = torch.zeros(
            num_policy_graphs, dtype=torch.float32, device=device
        )

        return PolicyOutput(
            stop_logits=stop_logits,
            expand_logits=option_expand_logits,
            edge_logits=torch.zeros(
                candidate_edge_ids.numel(),
                dtype=torch.float32,
                device=device,
            ),
            candidate_batch_ids=candidate_batch_ids,
            candidate_edge_ids=candidate_edge_ids,
            state_log_flow=state_log_flow,
        )


class _CountingFakeOnlinePolicy(_FakeOnlinePolicy):
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.prepare_num_graphs: list[int] = []

    def prepare_rollout_context(self, batch):
        self.prepare_calls += 1
        self.prepare_num_graphs.append(int(batch.num_graphs))
        return object()


class _CountingRewardModel(RewardModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.evaluate_calls = 0

    def evaluate_terminal_state(self, **kwargs):
        self.evaluate_calls += 1
        return super().evaluate_terminal_state(**kwargs)


def test_run_online_vectorized_splits_rollouts_back_to_original_batch() -> None:
    batch = _batch()
    engine = RolloutEngine(expand_budget=1)

    rollouts = engine.run_vectorized(
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
        assert rollout.traces.selected_edge_ids.shape == (2, 2)
        assert rollout.traces.stop_adv_loss is not None
        assert rollout.traces.stop_adv_loss.shape == (2, 2)
        assert rollout.traces.stop_adv_valid_mask is not None
        assert not bool(rollout.traces.stop_adv_valid_mask.any())
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
            rollout.stats.edge_action_entropy,
            torch.zeros(2, dtype=torch.float32),
        )
        assert torch.equal(
            rollout.stats.edge_action_count,
            torch.ones(2, dtype=torch.float32),
        )
        assert rollout.stats.terminal_answer_f1.shape == (2,)


def test_static_batch_rollouts_reuse_context_and_match_physical_repeat() -> None:
    batch = _batch()
    static_policy = _CountingFakeOnlinePolicy()

    static_rollouts = RolloutEngine(expand_budget=1).run_vectorized(
        policy=static_policy,
        retrieval_batch=batch,
        reward_model=RewardModel(edge_cost=0.0),
        num_rollouts=3,
        temperature=1.0,
        use_static_batch_rollouts=True,
    )
    physical_rollouts = RolloutEngine(expand_budget=1).run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(edge_cost=0.0),
        num_rollouts=3,
        temperature=1.0,
    )

    assert static_policy.prepare_calls == 1
    assert static_policy.prepare_num_graphs == [2]
    assert len(static_rollouts) == len(physical_rollouts) == 3
    for static, physical in zip(static_rollouts, physical_rollouts):
        assert torch.equal(
            static.stats.trajectory_length,
            physical.stats.trajectory_length,
        )
        assert torch.allclose(
            static.stats.terminal_log_reward,
            physical.stats.terminal_log_reward,
        )
        assert torch.equal(
            static.traces.action_type,
            physical.traces.action_type,
        )
        assert torch.equal(
            static.traces.selected_edge_ids,
            physical.traces.selected_edge_ids,
        )
        assert torch.equal(
            static.traces.stop_mask,
            physical.traces.stop_mask,
        )


def test_fused_static_batch_rollouts_reuse_context_and_match_physical_repeat() -> None:
    batch = _batch()
    fused_policy = _CountingFakeOnlinePolicy()

    fused_rollouts = RolloutEngine(expand_budget=1).run_vectorized(
        policy=fused_policy,
        retrieval_batch=batch,
        reward_model=RewardModel(edge_cost=0.0),
        num_rollouts=3,
        temperature=1.0,
        use_fused_static_batch_rollouts=True,
    )
    physical_rollouts = RolloutEngine(expand_budget=1).run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(edge_cost=0.0),
        num_rollouts=3,
        temperature=1.0,
    )

    assert fused_policy.prepare_calls == 1
    assert fused_policy.prepare_num_graphs == [2]
    assert len(fused_rollouts) == len(physical_rollouts) == 3
    for fused, physical in zip(fused_rollouts, physical_rollouts):
        assert torch.equal(
            fused.stats.trajectory_length,
            physical.stats.trajectory_length,
        )
        assert torch.allclose(
            fused.stats.terminal_log_reward,
            physical.stats.terminal_log_reward,
        )
        assert torch.equal(
            fused.traces.action_type,
            physical.traces.action_type,
        )
        assert torch.equal(
            fused.traces.selected_edge_ids,
            physical.traces.selected_edge_ids,
        )
        assert torch.equal(
            fused.traces.stop_mask,
            physical.traces.stop_mask,
        )


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
        state_readout_cfg={
            "use_path_memory": False,
            "use_frontier_summary": False,
        },
    )
    rollout_to_graph = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    state = RolloutState.create_initial(
        batch,
        expand_budget=1,
        rollout_to_graph=rollout_to_graph,
    )
    context = policy.prepare_rollout_context(batch)

    output = policy(batch, state, rollout_context=context)

    assert context.query_h.shape == (2, 2)
    assert output.state_log_flow.shape == (4,)
    assert torch.equal(
        output.candidate_batch_ids,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )
    assert torch.equal(
        output.candidate_edge_ids,
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )
    assert output.edge_logits.shape == (4,)


def test_rollout_engine_lazy_reward_only_evaluates_terminal_stops() -> None:
    batch = _batch()
    engine = RolloutEngine(expand_budget=1)
    reward_model = _CountingRewardModel(edge_cost=0.0)

    rollout = engine.run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=reward_model,
        num_rollouts=1,
        temperature=1.0,
        collect_stop_counterfactual=False,
        collect_policy_diagnostics=True,
        reward_mode=RewardMode.LAZY_TERMINAL,
    )[0]

    assert reward_model.evaluate_calls == 1
    assert not bool(rollout.traces.stop_now_valid_mask.any())
    assert torch.equal(
        rollout.stats.trajectory_length,
        torch.tensor([2, 2], dtype=torch.long),
    )


def test_rollout_engine_eager_reward_writes_stop_tb_traces_without_diagnostics() -> (
    None
):
    batch = _batch()
    engine = RolloutEngine(expand_budget=1)
    reward_model = _CountingRewardModel(edge_cost=0.0)

    rollout = engine.run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=reward_model,
        num_rollouts=1,
        temperature=1.0,
        collect_stop_counterfactual=False,
        collect_policy_diagnostics=False,
        reward_mode=RewardMode.EAGER_STOP_NOW,
    )[0]

    assert reward_model.evaluate_calls == 2
    assert bool(rollout.traces.stop_now_valid_mask[:, 0].all())
    assert bool(rollout.traces.stop_now_valid_mask[:, 1].all())
    assert bool(rollout.traces.stop_tb_valid_mask[:, 0].all())
    assert not bool(rollout.traces.stop_tb_valid_mask[:, 1].any())
    assert bool(rollout.traces.policy_action_valid_mask[:, 0].all())


def test_stop_advantage_auxiliary_writes_option_boundary_targets() -> None:
    batch = _three_node_batch()
    engine = RolloutEngine(expand_budget=1)
    auxiliary = StopAdvantageAuxiliary(
        StopAdvantageConfig(
            enabled=True,
            topk_by_semantic=2,
            topk_by_final=2,
            random_k=0,
            continue_pool_temperature=0.5,
            label_temperature=0.5,
        )
    )

    rollouts = engine.run_vectorized(
        policy=_FakeOnlinePolicy(),
        retrieval_batch=batch,
        reward_model=RewardModel(edge_cost=0.0),
        num_rollouts=1,
        temperature=1.0,
        auxiliary=auxiliary,
    )

    traces = rollouts[0].traces
    assert traces.stop_adv_valid_mask is not None
    assert bool(traces.stop_adv_valid_mask[0, 0])
    assert not bool(traces.stop_adv_valid_mask[0, 1])
    assert traces.stop_adv_target is not None
    assert traces.stop_adv_target[0, 0].item() < 0.01
    assert traces.stop_adv_loss is not None
    assert torch.isfinite(traces.stop_adv_loss[0, 0])
    assert traces.stop_adv_loss[0, 0].item() >= 0.0
