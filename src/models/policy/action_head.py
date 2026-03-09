from __future__ import annotations

import torch

from src.models.environment import DynamicAgentState, GraphEnvContext
from src.utils.segment_ops import segment_logsumexp_1d
from .action import (
    apply_visited_mask_to_edge_logits as apply_visited_mask_to_edge_logits_helper,
    build_empty_output as build_empty_output_helper,
    compute_stop_logits as compute_stop_logits_helper,
    gather_actions_from_csr_lock_free as gather_actions_from_csr_lock_free_helper,
)
from .encoder import PolicyEncoder
from .modules import EdgeScoreModule, PolicyProjectionModule, StopDeltaHead
from .path import validate_node_markov_state as validate_node_markov_state_helper


class ForwardActionHead:
    def __init__(
        self,
        *,
        encoder: PolicyEncoder,
        projections: PolicyProjectionModule,
        edge_scorer: EdgeScoreModule,
        stop_delta_head: StopDeltaHead,
    ) -> None:
        self.encoder = encoder
        self.projections = projections
        self.edge_scorer = edge_scorer
        self.stop_delta_head = stop_delta_head

    def _compute_stop_logits(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        stop_delta: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        edge_logits: torch.Tensor | None = None,
        edge_agent_batch: torch.Tensor | None = None,
        total_agents: int | None = None,
        super_node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_stop_logits_helper(
            env_context=env_context,
            agent_state=agent_state,
            stop_delta=stop_delta,
            device=device,
            dtype=dtype,
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            total_agents=total_agents,
            super_node_mask=super_node_mask,
        )

    @staticmethod
    def _gather_actions_from_csr_lock_free(
        *,
        adj_t,
        active_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return gather_actions_from_csr_lock_free_helper(
            adj_t=adj_t,
            active_nodes=active_nodes,
        )

    @staticmethod
    def _build_empty_output(
        *,
        B: int,
        num_agents: int,
        device: torch.device,
        dtype: torch.dtype,
        stop_logits: torch.Tensor,
        state_log_flows: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return build_empty_output_helper(
            B=B,
            num_agents=num_agents,
            device=device,
            dtype=dtype,
            stop_logits=stop_logits,
            state_log_flows=state_log_flows,
        )

    def compute_action_scores(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        action_cache: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        validate_node_markov_state_helper(agent_state=agent_state)
        B, num_agents = agent_state.current_nodes.shape
        total_agents = B * num_agents
        flat_curr_nodes = agent_state.current_nodes.view(-1)
        flat_active_mask = ~agent_state.done_mask.view(-1)

        if action_cache is None:
            resolved_cache = self.encoder.build_action_cache(
                env_context=env_context,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
            )
        else:
            resolved_cache = action_cache
        node_log_f = resolved_cache.get("node_log_f")
        if node_log_f is None:
            node_log_f = self.encoder.compute_node_log_f(
                env_context=env_context,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
            )
        question_context_tokens = resolved_cache.get("question_context_tokens")
        if question_context_tokens is None:
            raise ValueError("action_cache must provide `question_context_tokens`.")
        question_padding_mask = resolved_cache.get("question_padding_mask")
        if question_padding_mask is None:
            raise ValueError(
                "action_cache must provide `question_padding_mask` for token-level question interaction."
            )
        lexical_question_tokens = resolved_cache.get("lexical_question_tokens")
        super_node_mask = resolved_cache.get("super_node_mask")
        edge_disallowed_forward = resolved_cache.get("edge_disallowed_forward")
        edge_disallowed_backward = resolved_cache.get("edge_disallowed_backward")

        safe_curr_nodes = flat_curr_nodes.clamp(
            min=0, max=max(int(env_context.num_nodes_total) - 1, 0)
        )
        agent_history = node_tokens.index_select(0, safe_curr_nodes)
        invalid_nodes = flat_curr_nodes < 0
        if bool(invalid_nodes.any().item()):
            agent_history = torch.where(
                invalid_nodes.unsqueeze(-1),
                torch.zeros_like(agent_history),
                agent_history,
            )
        agent_potential, _, lexical_question_tokens, agent_question_padding_mask = (
            self.encoder.compute_agent_potentials(
                env_context=env_context,
                question_tokens=question_tokens,
                agent_history=agent_history,
                num_agents=num_agents,
                question_context_tokens=question_context_tokens,
                question_padding_mask=question_padding_mask,
                lexical_question_tokens=lexical_question_tokens,
            )
        )
        state_log_flows = self.encoder.gather_state_log_flows(
            node_log_f=node_log_f,
            current_nodes=flat_curr_nodes,
            num_nodes_total=int(env_context.num_nodes_total),
            dtype=node_tokens.dtype,
        )
        stop_delta = self.stop_delta_head(
            agent_potential=agent_potential, dtype=node_tokens.dtype
        )

        active_nodes = torch.where(
            flat_active_mask, flat_curr_nodes, torch.zeros_like(flat_curr_nodes)
        )
        if agent_state.flow_direction == "forward":
            adj_t = env_context.adj_t_fwd
        elif agent_state.flow_direction == "backward":
            adj_t = env_context.adj_t_bwd
        else:
            raise ValueError(
                f"Unsupported flow_direction in policy: {agent_state.flow_direction!r}."
            )
        edge_ids, target_nodes, out_degrees = self._gather_actions_from_csr_lock_free(
            adj_t=adj_t,
            active_nodes=active_nodes,
        )
        if int(edge_ids.numel()) == 0:
            stop_logits = self._compute_stop_logits(
                env_context=env_context,
                agent_state=agent_state,
                stop_delta=stop_delta,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                total_agents=total_agents,
                super_node_mask=super_node_mask,
            )
            return self._build_empty_output(
                B=B,
                num_agents=num_agents,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                stop_logits=stop_logits.view(B, num_agents),
                state_log_flows=state_log_flows.view(B, num_agents),
            )

        all_agent_rows = torch.arange(
            total_agents, device=target_nodes.device, dtype=torch.long
        )
        edge_agent_batch_full = all_agent_rows.repeat_interleave(out_degrees)
        edge_active_mask = flat_active_mask.index_select(0, edge_agent_batch_full)
        if not bool(edge_active_mask.all().item()):
            edge_ids = edge_ids[edge_active_mask]
            target_nodes = target_nodes[edge_active_mask]
            edge_agent_batch_full = edge_agent_batch_full[edge_active_mask]
        out_degrees_active = torch.zeros(
            (total_agents,), dtype=torch.long, device=node_tokens.device
        )
        if int(edge_agent_batch_full.numel()) > 0:
            out_degrees_active.scatter_add_(
                0,
                edge_agent_batch_full,
                torch.ones_like(
                    edge_agent_batch_full,
                    dtype=torch.long,
                    device=node_tokens.device,
                ),
            )
        out_degrees = out_degrees_active
        if int(edge_ids.numel()) == 0:
            stop_logits = self._compute_stop_logits(
                env_context=env_context,
                agent_state=agent_state,
                stop_delta=stop_delta,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                total_agents=total_agents,
                super_node_mask=super_node_mask,
            )
            return self._build_empty_output(
                B=B,
                num_agents=num_agents,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                stop_logits=stop_logits.view(B, num_agents),
                state_log_flows=state_log_flows.view(B, num_agents),
            )

        edge_agent_batch = edge_agent_batch_full
        out_degrees_filtered = out_degrees

        edge_logits, edge_agent_batch, edge_meta = self.edge_scorer.compute_edge_logits(
            env_context=env_context,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            node_log_f=node_log_f,
            edge_agent_batch=edge_agent_batch,
            target_nodes=target_nodes,
            edge_relations=env_context.edge_relations.index_select(
                0, edge_ids.clamp(min=0)
            ),
            current_nodes=flat_curr_nodes,
            total_agents=total_agents,
            agent_potential=agent_potential,
            lexical_question_tokens=lexical_question_tokens,
            agent_question_padding_mask=agent_question_padding_mask,
            relation_to_policy=self.projections.relation_to_policy,
            node_to_policy=self.projections.node_to_policy,
        )
        edge_logits = apply_visited_mask_to_edge_logits_helper(
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            target_nodes=target_nodes,
            visited_mask=agent_state.visited_mask,
        )
        if edge_disallowed_forward is not None and edge_disallowed_backward is not None:
            edge_disallowed = (
                edge_disallowed_forward
                if agent_state.flow_direction == "forward"
                else edge_disallowed_backward
            )
            disallowed_edges = edge_disallowed.index_select(0, edge_ids.clamp(min=0))
            neg_inf = torch.tensor(
                float("-inf"), device=edge_logits.device, dtype=edge_logits.dtype
            )
            edge_logits = edge_logits.masked_fill(disallowed_edges, neg_inf)

        stop_logits = self._compute_stop_logits(
            env_context=env_context,
            agent_state=agent_state,
            stop_delta=stop_delta,
            device=node_tokens.device,
            dtype=node_tokens.dtype,
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            total_agents=total_agents,
            super_node_mask=super_node_mask,
        )

        return {
            "edge_logits": edge_logits,
            "edge_agent_batch": edge_agent_batch,
            "stop_logits": stop_logits.view(B, num_agents),
            "edge_ids": edge_ids,
            "target_nodes": target_nodes,
            "relation_group_keys": edge_meta.get("relation_group_keys"),
            "relation_log_prob_group": edge_meta.get("relation_log_prob_group"),
            "edge_group_index": edge_meta.get("edge_group_index"),
            "num_relations": edge_meta.get("num_relations"),
            "out_degrees": out_degrees_filtered.view(B, num_agents),
            "state_log_flows": state_log_flows.view(B, num_agents),
        }


__all__ = ["ForwardActionHead"]
