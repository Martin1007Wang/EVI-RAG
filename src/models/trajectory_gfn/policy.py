from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.backbone import EmbeddingBackbone
from src.models.configs.policy import PolicyConfig
from src.models.policy.action import (
    gather_actions_from_csr_lock_free as gather_actions_from_csr_lock_free_helper,
)
from src.models.policy.modules import (
    EdgeScoreModule,
    NodeFlowHead,
    PolicyProjectionModule,
    QuestionContextModule,
    StartLogitHead,
    StopDeltaHead,
)
from src.models.policy.path import build_path_token_embeddings, encode_path_history
from src.utils.segment_ops import segment_logsumexp_1d

from .encoder import TrajectoryEncoder, TrajectoryPolicyContext
from .state import TrajectoryState


class InvalidStartCandidatesError(ValueError):
    def __init__(self, *, min_stop_steps: int, invalid_samples: list[str]) -> None:
        self.min_stop_steps = int(min_stop_steps)
        self.invalid_samples = tuple(str(sample_id) for sample_id in invalid_samples)
        super().__init__(
            "Each graph must expose at least one valid start candidate under "
            f"min_stop_steps={self.min_stop_steps}. "
            f"invalid_samples={list(self.invalid_samples)}"
        )


def _has_simple_path_of_length(*, adj_t, start_node: int, min_stop_steps: int) -> bool:
    if min_stop_steps <= 0:
        return True
    crow = adj_t.crow_indices()
    col = adj_t.col_indices()

    def _dfs(node: int, remaining: int, visited: set[int]) -> bool:
        if remaining <= 0:
            return True
        begin = int(crow[node].item())
        end = int(crow[node + 1].item())
        for offset in range(begin, end):
            child = int(col[offset].item())
            if child in visited:
                continue
            if _dfs(child, remaining - 1, visited | {child}):
                return True
        return False

    return _dfs(int(start_node), int(min_stop_steps), {int(start_node)})


def _compute_valid_start_candidates(
    *,
    context: TrajectoryPolicyContext,
    candidate_nodes_abs: torch.Tensor,
    min_stop_steps: int,
) -> torch.Tensor:
    if min_stop_steps <= 0:
        return torch.ones_like(candidate_nodes_abs, dtype=torch.bool)
    valid = []
    adj_t = context.env_context.adj_t_fwd
    for node in candidate_nodes_abs.tolist():
        valid.append(
            _has_simple_path_of_length(
                adj_t=adj_t,
                start_node=int(node),
                min_stop_steps=min_stop_steps,
            )
        )
    return torch.tensor(valid, device=candidate_nodes_abs.device, dtype=torch.bool)


@dataclass(frozen=True)
class StartDistribution:
    candidate_nodes_abs: torch.Tensor
    candidate_graph_ids: torch.Tensor
    log_probs: torch.Tensor


@dataclass(frozen=True)
class ForwardActionDistribution:
    edge_logits: torch.Tensor
    edge_agent_batch: torch.Tensor
    stop_logits: torch.Tensor
    edge_ids: torch.Tensor
    target_nodes: torch.Tensor
    out_degrees: torch.Tensor
    state_log_flows: torch.Tensor
    invalid_rows: torch.Tensor

    def to_policy_output(self) -> dict[str, torch.Tensor]:
        return {
            "edge_logits": self.edge_logits,
            "edge_agent_batch": self.edge_agent_batch,
            "stop_logits": self.stop_logits,
            "edge_ids": self.edge_ids,
            "target_nodes": self.target_nodes,
            "out_degrees": self.out_degrees,
            "state_log_flows": self.state_log_flows,
        }


@dataclass(frozen=True)
class BackwardActionDistribution:
    parent_log_probs: torch.Tensor
    edge_agent_batch: torch.Tensor
    parent_edge_ids: torch.Tensor
    parent_nodes: torch.Tensor


class TrajectoryPolicy(nn.Module):
    def __init__(
        self,
        config: PolicyConfig,
        *,
        max_steps: int,
        min_stop_steps: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_steps = int(max_steps)
        self.min_stop_steps = int(min_stop_steps)
        if self.min_stop_steps < 0:
            raise ValueError("min_stop_steps must be >= 0.")
        if self.min_stop_steps > self.max_steps:
            raise ValueError("min_stop_steps cannot exceed max_steps.")
        graph_hidden_dim = int(config.backbone.hidden_dim)
        policy_dim = int(config.backbone.embedding_dim)
        dropout = float(config.flow_head.dropout)
        lexical_rank = int(config.flow_head.relation_low_rank)
        backbone = EmbeddingBackbone(config.backbone)
        question_modules = QuestionContextModule(
            policy_dim=policy_dim,
            graph_hidden_dim=graph_hidden_dim,
            embedding_dim=int(config.backbone.embedding_dim),
            dropout=dropout,
            lexical_rank=lexical_rank,
        )
        node_flow_head = NodeFlowHead(
            node_dim=graph_hidden_dim,
            question_dim=graph_hidden_dim,
            hidden_dim=config.priority_head.hidden_dim,
            num_layers=config.priority_head.num_layers,
            dropout=config.priority_head.dropout,
        )
        self.projections = PolicyProjectionModule(
            graph_hidden_dim=graph_hidden_dim,
            policy_dim=policy_dim,
        )
        self.edge_scorer = EdgeScoreModule(
            policy_dim=policy_dim,
            hidden_dim=int(config.flow_head.hidden_dim),
            dropout=dropout,
            lexical_rank=lexical_rank,
            doob_h_alpha=float(config.doob_h_alpha),
        )
        self.stop_delta_head = StopDeltaHead(
            policy_dim=policy_dim,
            stop_bias_init=float(config.stop_bias_init),
            stop_delta_scale=float(config.stop_delta_scale),
            stop_delta_temperature=float(config.stop_delta_temperature),
        )
        self.start_head = StartLogitHead(
            policy_dim=graph_hidden_dim,
            hidden_dim=int(config.flow_head.hidden_dim),
            dropout=dropout,
        )
        self.path_pos_embedding = nn.Embedding(self.max_steps * 2 + 1, graph_hidden_dim)
        self.path_self_attention = nn.MultiheadAttention(
            embed_dim=graph_hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.path_self_attention_norm = nn.LayerNorm(graph_hidden_dim)
        self.step_embedding = nn.Embedding(self.max_steps + 1, graph_hidden_dim)
        self.remaining_embedding = nn.Embedding(self.max_steps + 1, graph_hidden_dim)
        self.state_feature_norm = nn.LayerNorm(graph_hidden_dim)
        from .heads import GraphLogZHead

        self.encoder = TrajectoryEncoder(
            backbone=backbone,
            question_modules=question_modules,
            node_flow_head=node_flow_head,
            graph_log_z_head=GraphLogZHead(
                hidden_dim=graph_hidden_dim,
                dropout=dropout,
            ),
            doob_h_node_temperature=float(config.doob_h_node_temperature),
        )

    def encode(self, batch) -> TrajectoryPolicyContext:
        return self.encoder.encode(batch)

    def compute_start_distribution(
        self,
        context: TrajectoryPolicyContext,
    ) -> StartDistribution:
        env_context = context.env_context
        q_counts = (env_context.q_ptr[1:] - env_context.q_ptr[:-1]).clamp(min=0)
        if bool((q_counts <= 0).any().item()):
            raise ValueError(
                "q_local_indices contains empty graphs; cannot build start distribution."
            )
        q_offsets = env_context.node_ptr[:-1].repeat_interleave(q_counts)
        candidate_nodes_abs = env_context.q_local_indices + q_offsets
        candidate_graph_ids = torch.arange(
            int(env_context.num_graphs),
            device=candidate_nodes_abs.device,
            dtype=torch.long,
        ).repeat_interleave(q_counts)
        node_features = context.node_tokens.index_select(0, candidate_nodes_abs)
        question_features = context.question_tokens.index_select(0, candidate_graph_ids)
        logits = self.start_head(
            node_features=node_features,
            question_features=question_features,
        ).to(dtype=torch.float32)
        valid_candidates = _compute_valid_start_candidates(
            context=context,
            candidate_nodes_abs=candidate_nodes_abs,
            min_stop_steps=self.min_stop_steps,
        )
        logits = logits.masked_fill(~valid_candidates, float("-inf"))
        lse, has_values = segment_logsumexp_1d(
            values=logits,
            segment_ids=candidate_graph_ids,
            num_segments=int(env_context.num_graphs),
            dtype=torch.float32,
            ignore_non_finite=True,
            empty_value=float("-inf"),
        )
        if not bool(has_values.all().item()):
            invalid_graphs = (
                torch.nonzero(~has_values, as_tuple=False).view(-1).tolist()
            )
            invalid_samples = [env_context.sample_ids[idx] for idx in invalid_graphs]
            raise InvalidStartCandidatesError(
                min_stop_steps=self.min_stop_steps,
                invalid_samples=invalid_samples,
            )
        return StartDistribution(
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            log_probs=logits - lse.index_select(0, candidate_graph_ids),
        )

    @staticmethod
    def sample_start_nodes(
        distribution: StartDistribution,
        *,
        num_rollouts: int,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs = int(distribution.candidate_graph_ids.max().item()) + 1
        selected_nodes: list[torch.Tensor] = []
        selected_log_probs: list[torch.Tensor] = []
        for graph_idx in range(num_graphs):
            mask = distribution.candidate_graph_ids == graph_idx
            graph_nodes = distribution.candidate_nodes_abs[mask]
            graph_log_probs = distribution.log_probs[mask]
            if int(graph_nodes.numel()) == 0:
                raise ValueError("Each graph must expose at least one start candidate.")
            if deterministic:
                order = torch.argsort(graph_log_probs, descending=True)
                ranked_nodes = graph_nodes.index_select(0, order)
                ranked_log_probs = graph_log_probs.index_select(0, order)
                if int(ranked_nodes.numel()) < num_rollouts:
                    repeat_idx = torch.remainder(
                        torch.arange(num_rollouts, device=ranked_nodes.device),
                        int(ranked_nodes.numel()),
                    )
                    ranked_nodes = ranked_nodes.index_select(0, repeat_idx)
                    ranked_log_probs = ranked_log_probs.index_select(0, repeat_idx)
                else:
                    ranked_nodes = ranked_nodes[:num_rollouts]
                    ranked_log_probs = ranked_log_probs[:num_rollouts]
            else:
                probs = torch.softmax(graph_log_probs, dim=0)
                sampled_idx = torch.multinomial(
                    probs, num_samples=num_rollouts, replacement=True
                )
                ranked_nodes = graph_nodes.index_select(0, sampled_idx)
                ranked_log_probs = graph_log_probs.index_select(0, sampled_idx)
            selected_nodes.append(ranked_nodes)
            selected_log_probs.append(ranked_log_probs)
        return torch.stack(selected_nodes, dim=0), torch.stack(
            selected_log_probs, dim=0
        )

    def compute_log_flow(
        self,
        context: TrajectoryPolicyContext,
        state: TrajectoryState,
    ) -> torch.Tensor:
        state_features = self.build_state_features(context, state)
        graph_ids = self._agent_graph_ids(
            num_graphs=int(state.current_node.size(0)),
            num_rollouts=int(state.current_node.size(1)),
            device=state.current_node.device,
        )
        log_flows = self._compute_log_flow_from_flat_features(
            context=context,
            flat_state_features=state_features.reshape(
                -1, int(state_features.size(-1))
            ),
            graph_ids=graph_ids,
        ).view_as(state.current_node)
        return torch.where(state.done_mask, torch.zeros_like(log_flows), log_flows)

    def _agent_graph_ids(
        self, *, num_graphs: int, num_rollouts: int, device: torch.device
    ) -> torch.Tensor:
        return torch.arange(
            num_graphs, device=device, dtype=torch.long
        ).repeat_interleave(num_rollouts)

    def _build_path_tokens_from_prefix(
        self,
        context: TrajectoryPolicyContext,
        *,
        flat_path_nodes: torch.Tensor,
        flat_path_edge_ids: torch.Tensor,
        flat_num_moves: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_agents = int(flat_path_nodes.size(0))
        token_len = self.max_steps * 2 + 1
        path_token_ids = torch.zeros(
            (total_agents, token_len),
            device=flat_path_nodes.device,
            dtype=torch.long,
        )
        path_token_types = torch.zeros_like(path_token_ids, dtype=torch.bool)
        node_width = int(flat_path_nodes.size(1))
        path_token_ids[:, : node_width * 2 : 2] = flat_path_nodes.clamp(min=0)
        if int(flat_path_edge_ids.numel()) > 0:
            safe_edge_ids = flat_path_edge_ids.clamp(min=0)
            relation_ids = context.env_context.edge_relations.index_select(
                0, safe_edge_ids.reshape(-1)
            ).view(total_agents, -1)
            edge_width = int(flat_path_edge_ids.size(1))
            path_token_ids[:, 1 : 1 + edge_width * 2 : 2] = relation_ids
            path_token_types[:, 1 : 1 + edge_width * 2 : 2] = flat_path_edge_ids >= 0
        path_lengths = flat_num_moves.to(dtype=torch.long) * 2 + 1
        return path_token_ids, path_token_types, path_lengths

    def _encode_path_history_from_prefix(
        self,
        context: TrajectoryPolicyContext,
        *,
        flat_path_nodes: torch.Tensor,
        flat_path_edge_ids: torch.Tensor,
        flat_num_moves: torch.Tensor,
    ) -> torch.Tensor:
        path_ids, path_types, path_lengths = self._build_path_tokens_from_prefix(
            context,
            flat_path_nodes=flat_path_nodes,
            flat_path_edge_ids=flat_path_edge_ids,
            flat_num_moves=flat_num_moves,
        )
        path_tokens = build_path_token_embeddings(
            path_token_ids=path_ids,
            path_token_types=path_types,
            node_tokens=context.node_tokens,
            relation_tokens=context.relation_tokens,
            pos_encoder=self.path_pos_embedding,
        )
        return encode_path_history(
            path_tokens=path_tokens,
            path_lengths=path_lengths,
            path_self_attention=self.path_self_attention,
            path_self_attention_norm=self.path_self_attention_norm,
        )

    def _build_flat_state_features_from_prefix(
        self,
        context: TrajectoryPolicyContext,
        *,
        flat_nodes: torch.Tensor,
        flat_num_moves: torch.Tensor,
        flat_done_mask: torch.Tensor,
        flat_path_nodes: torch.Tensor,
        flat_path_edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes_total = int(context.env_context.num_nodes_total)
        safe_nodes = flat_nodes.clamp(min=0, max=max(num_nodes_total - 1, 0))
        node_features = context.node_tokens.index_select(0, safe_nodes)
        path_features = self._encode_path_history_from_prefix(
            context,
            flat_path_nodes=flat_path_nodes,
            flat_path_edge_ids=flat_path_edge_ids,
            flat_num_moves=flat_num_moves,
        )
        step_ids = flat_num_moves.clamp(min=0, max=self.max_steps)
        remaining_ids = (self.max_steps - flat_num_moves).clamp(
            min=0, max=self.max_steps
        )
        step_features = self.step_embedding(step_ids).to(dtype=node_features.dtype)
        remaining_features = self.remaining_embedding(remaining_ids).to(
            dtype=node_features.dtype
        )
        state_features = self.state_feature_norm(
            node_features + path_features + step_features + remaining_features
        )
        return torch.where(
            flat_done_mask.unsqueeze(-1),
            torch.zeros_like(state_features),
            state_features,
        )

    def build_state_features(
        self,
        context: TrajectoryPolicyContext,
        state: TrajectoryState,
    ) -> torch.Tensor:
        batch_size, num_rollouts = state.current_node.shape
        flat_features = self._build_flat_state_features_from_prefix(
            context,
            flat_nodes=state.flatten_current(),
            flat_num_moves=state.flatten_num_moves(),
            flat_done_mask=state.flatten_done(),
            flat_path_nodes=state.flatten_path_nodes(),
            flat_path_edge_ids=state.flatten_path_edge_ids(),
        )
        return flat_features.view(batch_size, num_rollouts, -1)

    def _compute_stop_scores(
        self, *, agent_potential: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        scores = self.stop_delta_head.stop_head(
            agent_potential.to(dtype=torch.float32)
        ).squeeze(-1)
        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        return scores.to(dtype=dtype)

    def _compute_log_flow_from_flat_features(
        self,
        *,
        context: TrajectoryPolicyContext,
        flat_state_features: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        question_features = context.question_tokens.index_select(0, graph_ids)
        log_flows = self.encoder.policy_encoder.node_flow_head(
            flat_state_features,
            question_features,
        )
        log_flows = log_flows.to(dtype=torch.float32)
        log_flows = log_flows / self.encoder.policy_encoder.doob_h_node_temperature
        return torch.where(
            torch.isfinite(log_flows), log_flows, torch.zeros_like(log_flows)
        )

    def compute_forward_distribution(
        self,
        context: TrajectoryPolicyContext,
        state: TrajectoryState,
    ) -> ForwardActionDistribution:
        env_context = context.env_context
        batch_size, num_rollouts = state.current_node.shape
        total_agents = batch_size * num_rollouts
        flat_current = state.flatten_current()
        active_flat = ~state.flatten_done()
        flat_num_moves = state.flatten_num_moves()
        agent_graph_ids = self._agent_graph_ids(
            num_graphs=batch_size,
            num_rollouts=num_rollouts,
            device=flat_current.device,
        )
        state_features = self._build_flat_state_features_from_prefix(
            context,
            flat_nodes=flat_current,
            flat_num_moves=flat_num_moves,
            flat_done_mask=state.flatten_done(),
            flat_path_nodes=state.flatten_path_nodes(),
            flat_path_edge_ids=state.flatten_path_edge_ids(),
        )
        agent_history = state_features
        cache = context.action_cache
        agent_potential, _, lexical_question_tokens, agent_question_padding_mask = (
            self.encoder.policy_encoder.compute_agent_potentials(
                env_context=env_context,
                question_tokens=context.question_tokens,
                agent_history=agent_history,
                num_agents=num_rollouts,
                question_context_tokens=cache["question_context_tokens"],
                question_padding_mask=cache["question_padding_mask"],
                lexical_question_tokens=cache["lexical_question_tokens"],
            )
        )
        state_log_flows = self._compute_log_flow_from_flat_features(
            context=context,
            flat_state_features=state_features,
            graph_ids=agent_graph_ids,
        ).view(batch_size, num_rollouts)
        stop_logits = self._compute_stop_scores(
            agent_potential=agent_potential,
            dtype=context.node_tokens.dtype,
        ).view(batch_size, num_rollouts)
        active_nodes = torch.where(
            active_flat, flat_current, torch.zeros_like(flat_current)
        )
        edge_ids, target_nodes, out_degrees = gather_actions_from_csr_lock_free_helper(
            adj_t=env_context.adj_t_fwd,
            active_nodes=active_nodes,
        )
        edge_agent_batch = torch.empty(
            (0,), device=flat_current.device, dtype=torch.long
        )
        edge_logits = torch.empty(
            (0,), device=flat_current.device, dtype=context.node_tokens.dtype
        )
        if int(edge_ids.numel()) > 0:
            all_agent_rows = torch.arange(
                total_agents, device=flat_current.device, dtype=torch.long
            )
            edge_agent_batch_full = all_agent_rows.repeat_interleave(out_degrees)
            edge_active_mask = active_flat.index_select(0, edge_agent_batch_full)
            edge_ids = edge_ids[edge_active_mask]
            target_nodes = target_nodes[edge_active_mask]
            edge_agent_batch = edge_agent_batch_full[edge_active_mask]
            filtered_out_degrees = torch.zeros_like(out_degrees)
            if int(edge_agent_batch.numel()) > 0:
                filtered_out_degrees.scatter_add_(
                    0,
                    edge_agent_batch,
                    torch.ones_like(edge_agent_batch, dtype=torch.long),
                )
            out_degrees = filtered_out_degrees
            if int(edge_ids.numel()) > 0:
                edge_logits = torch.full(
                    (int(edge_ids.numel()),),
                    fill_value=float("-inf"),
                    device=flat_current.device,
                    dtype=context.node_tokens.dtype,
                )
                expandable = (
                    flat_num_moves.index_select(0, edge_agent_batch) < self.max_steps
                )
                if bool(expandable.any().item()):
                    child_path_nodes, child_path_edge_ids, edge_next_num_moves = (
                        state.build_child_prefix_tensors(
                            edge_agent_batch=edge_agent_batch[expandable],
                            target_nodes=target_nodes[expandable],
                            chosen_edge_ids=edge_ids[expandable],
                        )
                    )
                    edge_next_state_features = (
                        self._build_flat_state_features_from_prefix(
                            context,
                            flat_nodes=target_nodes[expandable],
                            flat_num_moves=edge_next_num_moves,
                            flat_done_mask=torch.zeros_like(
                                edge_next_num_moves, dtype=torch.bool
                            ),
                            flat_path_nodes=child_path_nodes,
                            flat_path_edge_ids=child_path_edge_ids,
                        )
                    )
                    edge_graph_ids = agent_graph_ids.index_select(
                        0, edge_agent_batch[expandable]
                    )
                    edge_next_log_f = self._compute_log_flow_from_flat_features(
                        context=context,
                        flat_state_features=edge_next_state_features,
                        graph_ids=edge_graph_ids,
                    )
                    scored_edge_logits, _, _ = self.edge_scorer.compute_edge_logits(
                        env_context=env_context,
                        node_tokens=context.node_tokens,
                        relation_tokens=context.relation_tokens,
                        edge_next_log_f=edge_next_log_f,
                        edge_agent_batch=edge_agent_batch[expandable],
                        target_nodes=target_nodes[expandable],
                        edge_relations=env_context.edge_relations.index_select(
                            0, edge_ids[expandable].clamp(min=0)
                        ),
                        current_nodes=flat_current,
                        total_agents=total_agents,
                        agent_potential=agent_potential,
                        lexical_question_tokens=lexical_question_tokens,
                        agent_question_padding_mask=agent_question_padding_mask,
                        relation_to_policy=self.projections.relation_to_policy,
                        node_to_policy=self.projections.node_to_policy,
                    )
                    edge_logits[expandable] = scored_edge_logits
        return ForwardActionDistribution(
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            stop_logits=stop_logits,
            edge_ids=edge_ids,
            target_nodes=target_nodes,
            out_degrees=out_degrees.view(batch_size, num_rollouts),
            state_log_flows=state_log_flows,
            invalid_rows=torch.zeros(
                (batch_size, num_rollouts),
                device=flat_current.device,
                dtype=torch.bool,
            ),
        )

    def compute_backward_distribution(
        self,
        context: TrajectoryPolicyContext,
        state: TrajectoryState,
    ) -> BackwardActionDistribution:
        del context
        batch_size, num_rollouts = state.current_node.shape
        total_agents = batch_size * num_rollouts
        active_flat = ~state.flatten_done()
        flat_num_moves = state.flatten_num_moves()
        edge_agent_batch = torch.arange(
            total_agents, device=state.current_node.device, dtype=torch.long
        )
        edge_agent_batch = edge_agent_batch[active_flat & (flat_num_moves > 0)]
        if int(edge_agent_batch.numel()) == 0:
            empty = torch.empty(
                (0,), device=state.current_node.device, dtype=torch.long
            )
            return BackwardActionDistribution(
                parent_log_probs=torch.empty(
                    (0,), device=state.current_node.device, dtype=torch.float32
                ),
                edge_agent_batch=empty,
                parent_edge_ids=empty,
                parent_nodes=empty,
            )
        parent_nodes = state.flatten_previous_nodes().index_select(0, edge_agent_batch)
        parent_edge_ids = state.flatten_incoming_edge_ids().index_select(
            0, edge_agent_batch
        )
        return BackwardActionDistribution(
            parent_log_probs=torch.zeros(
                (int(edge_agent_batch.numel()),),
                device=state.current_node.device,
                dtype=torch.float32,
            ),
            edge_agent_batch=edge_agent_batch,
            parent_edge_ids=parent_edge_ids,
            parent_nodes=parent_nodes,
        )

    @staticmethod
    def select_backward_log_probs(
        distribution: BackwardActionDistribution,
        *,
        chosen_edge_ids: torch.Tensor,
        active_move: torch.Tensor,
    ) -> torch.Tensor:
        total_agents = int(chosen_edge_ids.numel())
        out = torch.zeros(
            (total_agents,), device=chosen_edge_ids.device, dtype=torch.float32
        )
        if int(distribution.parent_edge_ids.numel()) == 0:
            if bool(active_move.any().item()):
                raise ValueError("Backward distribution is empty for active move rows.")
            return out
        chosen_rows = torch.arange(
            total_agents, device=chosen_edge_ids.device, dtype=torch.long
        )
        active_rows = chosen_rows[active_move]
        for row in active_rows.tolist():
            row_mask = distribution.edge_agent_batch == row
            row_edges = distribution.parent_edge_ids[row_mask]
            row_log_probs = distribution.parent_log_probs[row_mask]
            match = row_edges == int(chosen_edge_ids[row].item())
            if not bool(match.any().item()):
                raise ValueError(
                    f"Chosen edge is missing from backward distribution for row {row}."
                )
            out[row] = row_log_probs[match][0]
        return out

    @staticmethod
    def compute_forward_log_probs(
        distribution: ForwardActionDistribution,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool(distribution.invalid_rows.view(-1).any().item()):
            raise ValueError(
                "Forward action distribution contains invalid support rows."
            )
        total_agents = int(distribution.out_degrees.numel())
        edge_lse = torch.full(
            (total_agents,),
            fill_value=float("-inf"),
            device=distribution.stop_logits.device,
            dtype=torch.float32,
        )
        if int(distribution.edge_logits.numel()) > 0:
            move_lse, has_values = segment_logsumexp_1d(
                values=distribution.edge_logits.to(dtype=torch.float32),
                segment_ids=distribution.edge_agent_batch,
                num_segments=total_agents,
                dtype=torch.float32,
                ignore_non_finite=True,
                empty_value=float("-inf"),
            )
            edge_lse = torch.where(has_values, move_lse, edge_lse)
        stop_logits = distribution.stop_logits.view(-1).to(dtype=torch.float32)
        partition = torch.logaddexp(edge_lse, stop_logits)
        if bool((~torch.isfinite(partition)).any().item()):
            raise ValueError(
                "Forward action distribution contains rows without finite mass."
            )
        move_log_probs = distribution.edge_logits.to(dtype=torch.float32)
        if int(move_log_probs.numel()) > 0:
            move_log_probs = move_log_probs - partition.index_select(
                0, distribution.edge_agent_batch
            )
        stop_log_probs = stop_logits - partition
        return move_log_probs, stop_log_probs, partition
