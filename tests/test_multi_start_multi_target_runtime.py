from __future__ import annotations

import pytest
import torch

from src.models.components.beam_decoder import BeamDecoderEngine
from src.models.components.offline_forced_eval import OfflineForcedEvalEngine
from src.models.components.online_rollout import OnlineRolloutEngine
from src.models.components.sampler import RolloutSampler
from src.models.configs.search import RolloutConfig
from src.models.environment.contracts import CsrAdjacency, DynamicAgentState, GraphEnvContext


def _extract_policy_edges(
    *,
    env_context: GraphEnvContext,
    current_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    crow = env_context.adj_t_fwd.crow_indices()
    col = env_context.adj_t_fwd.col_indices()
    vals = env_context.adj_t_fwd.values()
    start_ptr = crow.index_select(0, current_nodes)
    end_ptr = crow.index_select(0, current_nodes + 1)
    out_degrees = end_ptr - start_ptr
    total_edges = int(out_degrees.sum().item())
    if total_edges <= 0:
        empty_long = torch.empty((0,), device=current_nodes.device, dtype=torch.long)
        return out_degrees, empty_long, empty_long, empty_long
    base = start_ptr.repeat_interleave(out_degrees)
    seg_start = out_degrees.cumsum(0) - out_degrees
    offsets = torch.arange(total_edges, device=current_nodes.device, dtype=torch.long)
    increments = offsets - seg_start.repeat_interleave(out_degrees)
    gather_idx = base + increments
    edge_ids = vals.index_select(0, gather_idx)
    target_nodes = col.index_select(0, gather_idx)
    edge_agent_batch = torch.arange(current_nodes.numel(), device=current_nodes.device, dtype=torch.long).repeat_interleave(
        out_degrees
    )
    return out_degrees, edge_ids, target_nodes, edge_agent_batch


class _PathPolicy:
    def __init__(self) -> None:
        self.training = False
        self.memory_tracker = torch.nn.GRUCell(input_size=8, hidden_size=4)

    def encode_context(self, env_context: GraphEnvContext) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return env_context.node_embeddings, env_context.relation_tokens, env_context.question_emb

    def compute_state_flow(
        self,
        *,
        agent_state: DynamicAgentState,
        question_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
    ) -> torch.Tensor:
        del question_tokens, node_tokens
        return torch.zeros_like(agent_state.current_nodes, dtype=torch.float32)

    def compute_action_scores(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del node_tokens, question_tokens, relation_tokens
        num_graphs, num_agents = agent_state.current_nodes.shape
        current = agent_state.current_nodes.view(-1)
        out_degrees, edge_ids, target_nodes, edge_agent_batch = _extract_policy_edges(
            env_context=env_context,
            current_nodes=current,
        )
        edge_logits = torch.full((edge_ids.numel(),), 6.0, device=current.device, dtype=torch.float32)
        stop_logits = torch.full((num_graphs, num_agents), -6.0, device=current.device, dtype=torch.float32)
        return {
            "edge_logits": edge_logits,
            "edge_agent_batch": edge_agent_batch,
            "stop_logits": stop_logits,
            "edge_ids": edge_ids,
            "target_nodes": target_nodes,
            "out_degrees": out_degrees.view(num_graphs, num_agents),
        }

    def evolve_state(
        self,
        *,
        agent_state: DynamicAgentState,
        chosen_target_nodes: torch.Tensor,
        chosen_edge_relations: torch.Tensor,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        is_stop: torch.Tensor,
    ) -> DynamicAgentState:
        del chosen_edge_relations, node_tokens, relation_tokens
        num_graphs, num_agents = agent_state.current_nodes.shape
        next_nodes = torch.where(
            is_stop.view(num_graphs, num_agents),
            agent_state.current_nodes,
            chosen_target_nodes.view(num_graphs, num_agents),
        )
        return DynamicAgentState(
            step_t=agent_state.step_t + 1,
            current_nodes=next_nodes,
            hidden_states=agent_state.hidden_states,
            visited_mask=agent_state.visited_mask,
            cumulative_rewards=agent_state.cumulative_rewards,
            done_mask=agent_state.done_mask | is_stop.view(num_graphs, num_agents),
        )


class _StaticLogitPolicy(_PathPolicy):
    def __init__(
        self,
        *,
        default_edge_logit: float,
        stop_logit: float,
        edge_logit_overrides: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        self.default_edge_logit = float(default_edge_logit)
        self.stop_logit = float(stop_logit)
        self.edge_logit_overrides = edge_logit_overrides or {}

    def compute_action_scores(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del node_tokens, question_tokens, relation_tokens
        num_graphs, num_agents = agent_state.current_nodes.shape
        current = agent_state.current_nodes.view(-1)
        out_degrees, edge_ids, target_nodes, edge_agent_batch = _extract_policy_edges(
            env_context=env_context,
            current_nodes=current,
        )
        edge_logits = torch.full(
            (edge_ids.numel(),),
            self.default_edge_logit,
            device=current.device,
            dtype=torch.float32,
        )
        for edge_id, value in self.edge_logit_overrides.items():
            edge_logits = torch.where(
                edge_ids == int(edge_id),
                torch.full_like(edge_logits, float(value)),
                edge_logits,
            )
        stop_logits = torch.full(
            (num_graphs, num_agents),
            self.stop_logit,
            device=current.device,
            dtype=torch.float32,
        )
        return {
            "edge_logits": edge_logits,
            "edge_agent_batch": edge_agent_batch,
            "stop_logits": stop_logits,
            "edge_ids": edge_ids,
            "target_nodes": target_nodes,
            "out_degrees": out_degrees.view(num_graphs, num_agents),
        }


class _NonFiniteLogitPolicy(_PathPolicy):
    def compute_action_scores(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del node_tokens, question_tokens, relation_tokens
        num_graphs, num_agents = agent_state.current_nodes.shape
        current = agent_state.current_nodes.view(-1)
        out_degrees, edge_ids, target_nodes, edge_agent_batch = _extract_policy_edges(
            env_context=env_context,
            current_nodes=current,
        )
        edge_logits = torch.full(
            (edge_ids.numel(),),
            float("nan"),
            device=current.device,
            dtype=torch.float32,
        )
        stop_logits = torch.full(
            (num_graphs, num_agents),
            float("nan"),
            device=current.device,
            dtype=torch.float32,
        )
        return {
            "edge_logits": edge_logits,
            "edge_agent_batch": edge_agent_batch,
            "stop_logits": stop_logits,
            "edge_ids": edge_ids,
            "target_nodes": target_nodes,
            "out_degrees": out_degrees.view(num_graphs, num_agents),
        }


def _build_csr(*, row: torch.Tensor, col: torch.Tensor, edge_ids: torch.Tensor, num_nodes: int) -> CsrAdjacency:
    if int(row.numel()) == 0:
        return CsrAdjacency(
            crow=torch.zeros((num_nodes + 1,), dtype=torch.long),
            col=col,
            edge_ids=edge_ids,
            size=(num_nodes, num_nodes),
        )
    order = torch.argsort(row)
    row_sorted = row.index_select(0, order)
    col_sorted = col.index_select(0, order)
    edge_sorted = edge_ids.index_select(0, order)
    crow = torch.searchsorted(
        row_sorted,
        torch.arange(num_nodes + 1, dtype=torch.long, device=row.device),
        right=False,
    )
    return CsrAdjacency(
        crow=crow,
        col=col_sorted,
        edge_ids=edge_sorted,
        size=(num_nodes, num_nodes),
    )


def _build_context(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    q_local_indices: torch.Tensor,
    q_ptr: torch.Tensor,
    a_local_indices: torch.Tensor,
    a_ptr: torch.Tensor,
    start_local_indices: torch.Tensor | None = None,
    node_global_ids: torch.Tensor | None = None,
) -> GraphEnvContext:
    edge_ids = torch.arange(edge_index.size(1), dtype=torch.long)
    row_fwd = edge_index[0]
    col_fwd = edge_index[1]
    row_bwd = edge_index[1]
    col_bwd = edge_index[0]
    adj_t_fwd = _build_csr(row=row_fwd, col=col_fwd, edge_ids=edge_ids, num_nodes=num_nodes)
    adj_t_bwd = _build_csr(row=row_bwd, col=col_bwd, edge_ids=edge_ids, num_nodes=num_nodes)
    if node_global_ids is None:
        node_global_ids = torch.arange(num_nodes, dtype=torch.long)
    if node_global_ids.dim() != 1 or int(node_global_ids.numel()) != num_nodes:
        raise ValueError(
            "node_global_ids must be 1D with num_nodes entries in _build_context: "
            f"shape={tuple(node_global_ids.shape)} num_nodes={num_nodes}."
        )
    return GraphEnvContext(
        num_graphs=1,
        num_nodes_total=num_nodes,
        node_ptr=torch.tensor([0, num_nodes], dtype=torch.long),
        edge_index=edge_index,
        edge_relations=torch.zeros((edge_index.size(1),), dtype=torch.long),
        edge_rel_global=torch.zeros((edge_index.size(1),), dtype=torch.long),
        edge_batch=torch.zeros((edge_index.size(1),), dtype=torch.long),
        node_batch=torch.zeros((num_nodes,), dtype=torch.long),
        adj_t_fwd=adj_t_fwd,
        adj_t_bwd=adj_t_bwd,
        node_embeddings=torch.zeros((num_nodes, 4), dtype=torch.float32),
        node_tokens=torch.zeros((num_nodes, 4), dtype=torch.float32),
        relation_tokens=torch.zeros((1, 4), dtype=torch.float32),
        question_emb=torch.zeros((1, 4), dtype=torch.float32),
        q_local_indices=q_local_indices,
        a_local_indices=a_local_indices,
        q_ptr=q_ptr,
        a_ptr=a_ptr,
        answer_entity_ids=a_local_indices.clone(),
        answer_ptr=a_ptr.clone(),
        node_global_ids=node_global_ids,
        dummy_mask=torch.tensor([False]),
        sample_ids=["sample_0"],
        start_local_indices=start_local_indices,
    )


def _make_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return _build_context(
        num_nodes=3,
        edge_index=edge_index,
        q_local_indices=torch.tensor([0], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
    )


def _make_multi_start_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0, 2], [3, 3]], dtype=torch.long)
    return _build_context(
        num_nodes=4,
        edge_index=edge_index,
        q_local_indices=torch.tensor([0, 2], dtype=torch.long),
        q_ptr=torch.tensor([0, 2], dtype=torch.long),
        a_local_indices=torch.tensor([3], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
    )


def _make_super_start_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return _build_context(
        num_nodes=3,
        edge_index=edge_index,
        q_local_indices=torch.tensor([1], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        start_local_indices=torch.tensor([0], dtype=torch.long),
    )


def _make_virtual_start_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return _build_context(
        num_nodes=3,
        edge_index=edge_index,
        q_local_indices=torch.tensor([1], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        start_local_indices=torch.tensor([0], dtype=torch.long),
        node_global_ids=torch.tensor([-1, 101, 102], dtype=torch.long),
    )


def _make_zero_hop_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return _build_context(
        num_nodes=2,
        edge_index=edge_index,
        q_local_indices=torch.tensor([0], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([0], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
    )


def _make_parallel_edge_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0, 0, 0], [1, 1, 2]], dtype=torch.long)
    return _build_context(
        num_nodes=3,
        edge_index=edge_index,
        q_local_indices=torch.tensor([0], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
    )


def _make_sampler(*, stop_min_steps: int = 0, num_rollouts: int = 1) -> RolloutSampler:
    cfg = RolloutConfig(
        num_rollouts=num_rollouts,
        max_steps=4,
        stop_min_steps=stop_min_steps,
        sampling_temperature=1.0,
        sampling_mode="greedy",
        eval_sampling_temperature=0.5,
        eval_sample_without_replacement=True,
        backward_prior_mode="uniform",
    )
    return RolloutSampler(cfg)


def test_rollout_sampler_exposes_split_engines() -> None:
    sampler = _make_sampler()
    assert isinstance(sampler.online_engine, OnlineRolloutEngine)
    assert isinstance(sampler.offline_engine, OfflineForcedEvalEngine)
    assert isinstance(sampler.beam_engine, BeamDecoderEngine)


def test_online_rollout_smoke_path() -> None:
    sampler = _make_sampler()
    rollout = sampler.sample_forward(_make_context(), _PathPolicy(), deterministic=True, collect_traces=True)
    assert rollout.log_pf_steps is not None
    assert rollout.log_pb_steps is not None
    assert rollout.log_f_steps is not None
    assert torch.equal(rollout.stop_nodes, torch.tensor([[2]], dtype=torch.long))
    assert torch.equal(rollout.num_moves, torch.tensor([[2]], dtype=torch.long))


def test_offline_forced_eval_smoke_path() -> None:
    sampler = _make_sampler()
    rollout = sampler.evaluate_forced_paths(
        _make_context(),
        _PathPolicy(),
        start_local_indices=torch.tensor([[0]], dtype=torch.long),
        forced_edge_ids=torch.tensor([[[0, 1, -1, -1]]], dtype=torch.long),
        path_lengths=torch.tensor([[2]], dtype=torch.long),
        collect_traces=True,
        use_visited_mask=False,
    )
    assert rollout.valid_mask is not None
    assert bool(rollout.valid_mask.all().item())
    assert torch.equal(rollout.stop_nodes, torch.tensor([[2]], dtype=torch.long))
    assert torch.equal(rollout.num_moves, torch.tensor([[2]], dtype=torch.long))


def test_offline_forced_eval_masks_nonfinite_logits_rows() -> None:
    sampler = _make_sampler()
    rollout = sampler.evaluate_forced_paths(
        _make_context(),
        _NonFiniteLogitPolicy(),
        start_local_indices=torch.tensor([[0]], dtype=torch.long),
        forced_edge_ids=torch.tensor([[[0, -1, -1, -1]]], dtype=torch.long),
        path_lengths=torch.tensor([[1]], dtype=torch.long),
        collect_traces=True,
        use_visited_mask=False,
    )
    assert rollout.valid_mask is not None
    assert not bool(rollout.valid_mask.any().item())
    assert torch.isfinite(rollout.log_pf_sum).all()
    assert rollout.log_f_steps is not None
    assert torch.isfinite(rollout.log_f_steps).all()


def test_beam_decode_smoke_path() -> None:
    sampler = _make_sampler()
    rollout = sampler.beam_search_forward(
        _make_context(),
        _PathPolicy(),
        beam_size=1,
        max_steps=4,
        require_done=True,
        diverse_penalty=0.0,
    )
    assert torch.equal(rollout.stop_nodes, torch.tensor([[2]], dtype=torch.long))
    assert torch.equal(rollout.num_moves, torch.tensor([[2]], dtype=torch.long))


def test_beam_init_uses_multi_start_round_robin() -> None:
    context = _make_multi_start_context()
    agent_state = BeamDecoderEngine._init_agent_state(
        env_context=context,
        num_agents=4,
        deterministic=True,
    )
    assert torch.equal(agent_state.current_nodes, torch.tensor([[0, 2, 0, 2]], dtype=torch.long))


def test_beam_init_prefers_explicit_start_override() -> None:
    context = _make_super_start_context()
    agent_state = BeamDecoderEngine._init_agent_state(
        env_context=context,
        num_agents=4,
        deterministic=True,
    )
    assert torch.equal(agent_state.current_nodes, torch.tensor([[0, 0, 0, 0]], dtype=torch.long))


def test_online_and_beam_start_from_explicit_start_override() -> None:
    sampler = _make_sampler(stop_min_steps=0, num_rollouts=1)
    policy = _PathPolicy()
    context = _make_super_start_context()

    online_rollout = sampler.sample_forward(context, policy, deterministic=True, collect_traces=False)
    assert torch.equal(online_rollout.stop_nodes, torch.tensor([[2]], dtype=torch.long))
    assert torch.equal(online_rollout.num_moves, torch.tensor([[2]], dtype=torch.long))

    beam_rollout = sampler.beam_search_forward(
        context,
        policy,
        beam_size=1,
        max_steps=4,
        require_done=True,
        diverse_penalty=0.0,
    )
    assert torch.equal(beam_rollout.stop_nodes, torch.tensor([[2]], dtype=torch.long))
    assert torch.equal(beam_rollout.num_moves, torch.tensor([[2]], dtype=torch.long))


def test_zero_hop_respects_stop_min_steps_without_answer_override() -> None:
    sampler = _make_sampler(stop_min_steps=1, num_rollouts=1)
    policy = _StaticLogitPolicy(default_edge_logit=0.0, stop_logit=10.0)
    context = _make_zero_hop_context()
    beam_rollout = sampler.beam_search_forward(
        context,
        policy,
        beam_size=1,
        max_steps=2,
        require_done=True,
        diverse_penalty=0.0,
    )
    assert torch.equal(beam_rollout.stop_nodes, torch.tensor([[1]], dtype=torch.long))
    assert torch.equal(beam_rollout.num_moves, torch.tensor([[1]], dtype=torch.long))

    online_rollout = sampler.sample_forward(context, policy, deterministic=True, collect_traces=False)
    assert torch.equal(online_rollout.stop_nodes, torch.tensor([[1]], dtype=torch.long))
    assert torch.equal(online_rollout.num_moves, torch.tensor([[1]], dtype=torch.long))


def test_stop_min_steps_ignores_virtual_start_hop() -> None:
    sampler = _make_sampler(stop_min_steps=1, num_rollouts=1)
    policy = _StaticLogitPolicy(default_edge_logit=0.0, stop_logit=10.0)
    context = _make_virtual_start_context()

    online_rollout = sampler.sample_forward(context, policy, deterministic=True, collect_traces=False)
    assert torch.equal(online_rollout.stop_nodes, torch.tensor([[2]], dtype=torch.long))
    assert torch.equal(online_rollout.num_moves, torch.tensor([[2]], dtype=torch.long))

    beam_rollout = sampler.beam_search_forward(
        context,
        policy,
        beam_size=1,
        max_steps=3,
        require_done=True,
        diverse_penalty=0.0,
    )
    assert torch.equal(beam_rollout.stop_nodes, torch.tensor([[2]], dtype=torch.long))
    assert torch.equal(beam_rollout.num_moves, torch.tensor([[2]], dtype=torch.long))


def test_online_rollout_rejects_oracle_force_stop_for_leakage_compliance() -> None:
    sampler = RolloutSampler(
        RolloutConfig(
            num_rollouts=1,
            max_steps=4,
            stop_min_steps=1,
            train_oracle_force_stop=True,
            sampling_temperature=1.0,
            sampling_mode="greedy",
            eval_sampling_temperature=0.5,
            eval_sample_without_replacement=True,
            backward_prior_mode="uniform",
        )
    )
    policy = _StaticLogitPolicy(default_edge_logit=0.0, stop_logit=0.0)
    with pytest.raises(ValueError, match="forbidden to prevent label leakage"):
        _ = sampler.sample_forward(_make_context(), policy, deterministic=True, collect_traces=False)


def test_beam_dedup_keeps_distinct_targets_for_same_parent_trace() -> None:
    sampler = _make_sampler(stop_min_steps=0, num_rollouts=2)
    policy = _StaticLogitPolicy(
        default_edge_logit=-20.0,
        stop_logit=-20.0,
        edge_logit_overrides={0: 10.0, 1: 10.0, 2: 9.0},
    )
    rollout = sampler.beam_search_forward(
        _make_parallel_edge_context(),
        policy,
        beam_size=2,
        max_steps=1,
        require_done=False,
        diverse_penalty=0.0,
    )
    assert torch.equal(rollout.stop_nodes.sort(dim=1).values, torch.tensor([[1, 2]], dtype=torch.long))


def test_online_rollout_log_f_steps_tracks_policy_partition() -> None:
    sampler = _make_sampler(stop_min_steps=0, num_rollouts=1)
    policy = _StaticLogitPolicy(default_edge_logit=2.0, stop_logit=0.0)
    rollout = sampler.sample_forward(_make_zero_hop_context(), policy, deterministic=True, collect_traces=True)
    assert rollout.log_f_steps is not None
    expected = torch.logsumexp(torch.tensor([2.0, 0.0], dtype=torch.float32), dim=0)
    assert torch.allclose(rollout.log_f_steps[0, 0, 0], expected, atol=1.0e-6)
