from __future__ import annotations

import types

import torch

from src.models.components.high_energy_replay import HighEnergyReplayBuffer
from src.models.components.sampler import RolloutSampler
from src.models.environment.contracts import CsrAdjacency, DynamicAgentState, GraphEnvContext


class _PathPolicy:
    def __init__(self) -> None:
        self.training = True

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
        crow = env_context.adj_t_fwd.crow_indices()
        col = env_context.adj_t_fwd.col_indices()
        vals = env_context.adj_t_fwd.values()
        start_ptr = crow.index_select(0, current)
        end_ptr = crow.index_select(0, current + 1)
        out_degrees = end_ptr - start_ptr
        total_edges = int(out_degrees.sum().item())
        if total_edges > 0:
            base = start_ptr.repeat_interleave(out_degrees)
            seg_start = out_degrees.cumsum(0) - out_degrees
            offsets = torch.arange(total_edges, device=current.device, dtype=torch.long)
            increments = offsets - seg_start.repeat_interleave(out_degrees)
            gather_idx = base + increments
            edge_ids = vals.index_select(0, gather_idx)
            target_nodes = col.index_select(0, gather_idx)
            edge_agent_batch = torch.arange(current.numel(), device=current.device, dtype=torch.long).repeat_interleave(
                out_degrees
            )
            edge_logits = torch.full((total_edges,), 6.0, device=current.device, dtype=torch.float32)
        else:
            edge_ids = torch.empty((0,), device=current.device, dtype=torch.long)
            target_nodes = torch.empty((0,), device=current.device, dtype=torch.long)
            edge_agent_batch = torch.empty((0,), device=current.device, dtype=torch.long)
            edge_logits = torch.empty((0,), device=current.device, dtype=torch.float32)
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


def _make_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_ids = torch.arange(2, dtype=torch.long)
    adj_t_fwd = CsrAdjacency(
        crow=torch.tensor([0, 1, 2, 2], dtype=torch.long),
        col=torch.tensor([1, 2], dtype=torch.long),
        edge_ids=edge_ids,
        size=(3, 3),
    )
    adj_t_bwd = CsrAdjacency(
        crow=torch.tensor([0, 0, 1, 2], dtype=torch.long),
        col=torch.tensor([0, 1], dtype=torch.long),
        edge_ids=edge_ids,
        size=(3, 3),
    )
    return GraphEnvContext(
        num_graphs=1,
        num_nodes_total=3,
        node_ptr=torch.tensor([0, 3], dtype=torch.long),
        edge_index=edge_index,
        edge_relations=torch.tensor([0, 0], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 0], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        node_batch=torch.tensor([0, 0, 0], dtype=torch.long),
        adj_t_fwd=adj_t_fwd,
        adj_t_bwd=adj_t_bwd,
        node_embeddings=torch.zeros((3, 4), dtype=torch.float32),
        node_tokens=torch.zeros((3, 4), dtype=torch.float32),
        relation_tokens=torch.zeros((1, 4), dtype=torch.float32),
        question_emb=torch.zeros((1, 4), dtype=torch.float32),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([2], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_global_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        dummy_mask=torch.tensor([False]),
        sample_ids=["sample_0"],
    )


def _make_super_source_context() -> GraphEnvContext:
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 0]], dtype=torch.long)
    edge_ids = torch.arange(3, dtype=torch.long)
    adj_t_fwd = CsrAdjacency(
        crow=torch.tensor([0, 1, 2, 2, 3], dtype=torch.long),
        col=torch.tensor([1, 2, 0], dtype=torch.long),
        edge_ids=edge_ids,
        size=(4, 4),
    )
    adj_t_bwd = CsrAdjacency(
        crow=torch.tensor([0, 1, 2, 3, 3], dtype=torch.long),
        col=torch.tensor([3, 0, 1], dtype=torch.long),
        edge_ids=edge_ids,
        size=(4, 4),
    )
    return GraphEnvContext(
        num_graphs=1,
        num_nodes_total=4,
        node_ptr=torch.tensor([0, 4], dtype=torch.long),
        edge_index=edge_index,
        edge_relations=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 0], dtype=torch.long),
        node_batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        adj_t_fwd=adj_t_fwd,
        adj_t_bwd=adj_t_bwd,
        node_embeddings=torch.zeros((4, 4), dtype=torch.float32),
        node_tokens=torch.zeros((4, 4), dtype=torch.float32),
        relation_tokens=torch.zeros((1, 4), dtype=torch.float32),
        question_emb=torch.zeros((1, 4), dtype=torch.float32),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([2], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_global_ids=torch.tensor([10, 11, 12, -1], dtype=torch.long),
        dummy_mask=torch.tensor([False]),
        sample_ids=["sample_super"],
        start_local_indices=torch.tensor([3], dtype=torch.long),
    )


def test_high_energy_replay_buffer_sampling_and_forced_eval() -> None:
    context = _make_context()
    replay_cfg = types.SimpleNamespace(
        enabled=True,
        max_paths_per_pair=8,
        max_paths_per_graph=32,
        max_shortest_paths_per_pair=4,
        max_dfs_paths_per_pair=4,
        max_depth=4,
        allow_cycles=True,
        max_node_visits=2,
    )
    replay = HighEnergyReplayBuffer(replay_cfg)
    batch = replay.build_and_sample(
        context=context,
        num_rollouts=2,
        max_steps=3,
        alpha=1.0,
        stop_min_steps=1,
        device=torch.device("cpu"),
    )
    assert bool(batch.graph_has_oracle[0].item())
    assert bool(batch.use_offline_mask.all().item())
    assert bool((batch.path_lengths > 0).all().item())

    sampler = RolloutSampler(
        types.SimpleNamespace(
            num_rollouts=2,
            max_steps=3,
            stop_min_steps=1,
            sampling_temperature=1.0,
            sampling_mode="gumbel",
            eval_sampling_temperature=0.5,
            eval_sample_without_replacement=True,
        )
    )
    rollout = sampler.evaluate_forced_paths(
        context,
        _PathPolicy(),
        start_local_indices=batch.start_local_indices,
        forced_edge_ids=batch.edge_ids,
        path_lengths=batch.path_lengths,
        collect_traces=True,
        use_visited_mask=False,
    )
    assert rollout.log_pf_steps is not None
    assert rollout.log_pb_steps is not None
    assert rollout.log_f_steps is not None
    assert torch.equal(rollout.stop_nodes, torch.tensor([[2, 2]], dtype=torch.long))
    assert torch.equal(rollout.num_moves, torch.tensor([[2, 2]], dtype=torch.long))


def test_high_energy_replay_buffer_uses_precomputed_oracle_when_online_disabled() -> None:
    context = _make_context()
    context = GraphEnvContext(
        **{
            **context.__dict__,
            "replay_start_local": torch.tensor([0], dtype=torch.long),
            "replay_path_lengths": torch.tensor([2], dtype=torch.long),
            "replay_edge_local_ids": torch.tensor([0, 1], dtype=torch.long),
            "replay_path_ptr": torch.tensor([0, 1], dtype=torch.long),
            "replay_edge_ptr": torch.tensor([0, 2], dtype=torch.long),
        }
    )
    replay_cfg = types.SimpleNamespace(
        enabled=True,
        max_paths_per_pair=8,
        max_paths_per_graph=32,
        max_shortest_paths_per_pair=4,
        max_dfs_paths_per_pair=4,
        max_depth=0,
        allow_cycles=True,
        max_node_visits=2,
    )
    replay = HighEnergyReplayBuffer(replay_cfg)
    batch = replay.build_and_sample(
        context=context,
        num_rollouts=2,
        max_steps=3,
        alpha=1.0,
        stop_min_steps=1,
        device=torch.device("cpu"),
    )
    assert bool(batch.graph_has_oracle[0].item())
    assert bool(batch.use_offline_mask.all().item())
    assert torch.equal(batch.path_lengths, torch.tensor([[2, 2]], dtype=torch.long))


def test_high_energy_replay_buffer_promotes_precomputed_q_paths_with_super_source() -> None:
    context = _make_super_source_context()
    context = GraphEnvContext(
        **{
            **context.__dict__,
            "replay_start_local": torch.tensor([0], dtype=torch.long),
            "replay_path_lengths": torch.tensor([2], dtype=torch.long),
            "replay_edge_local_ids": torch.tensor([0, 1], dtype=torch.long),
            "replay_path_ptr": torch.tensor([0, 1], dtype=torch.long),
            "replay_edge_ptr": torch.tensor([0, 2], dtype=torch.long),
        }
    )
    replay_cfg = types.SimpleNamespace(
        enabled=True,
        max_paths_per_pair=8,
        max_paths_per_graph=32,
        max_shortest_paths_per_pair=4,
        max_dfs_paths_per_pair=4,
        max_depth=0,
        allow_cycles=True,
        max_node_visits=2,
    )
    replay = HighEnergyReplayBuffer(replay_cfg)
    batch = replay.build_and_sample(
        context=context,
        num_rollouts=1,
        max_steps=4,
        alpha=1.0,
        stop_min_steps=1,
        device=torch.device("cpu"),
    )
    assert bool(batch.use_offline_mask[0, 0].item())
    assert int(batch.start_local_indices[0, 0].item()) == 3
    assert int(batch.path_lengths[0, 0].item()) == 3
    assert torch.equal(batch.edge_ids[0, 0, :3], torch.tensor([2, 0, 1], dtype=torch.long))
