from __future__ import annotations

import torch

from src.models.environment import DynamicAgentState, GraphEnvContext
from src.utils.segment_ops import segment_logsumexp_1d
from .action import (
    compute_parent_log_probs as compute_parent_log_probs_helper,
    compute_stop_logits as compute_stop_logits_helper,
    gather_parent_edges_from_csr as gather_parent_edges_from_csr_helper,
    select_parent_log_prob as select_parent_log_prob_helper,
)
from .encoder import PolicyEncoder
from .modules import EdgeScoreModule, PolicyProjectionModule, StopDeltaHead
from .path import validate_node_markov_state as validate_node_markov_state_helper


class BackwardLogProbHead:
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

    def compute_backward_log_prob(
        self,
        *,
        env_context: GraphEnvContext,
        agent_state: DynamicAgentState,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        prev_nodes: torch.Tensor,
        chosen_edge_ids: torch.Tensor,
        active_flat: torch.Tensor,
        is_stop_flat: torch.Tensor,
        stop_guard_active: torch.Tensor | None = None,
        action_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_node_markov_state_helper(agent_state=agent_state)
        B, num_agents = agent_state.current_nodes.shape
        total_agents = B * num_agents
        if int(prev_nodes.numel()) != total_agents:
            raise ValueError(
                "prev_nodes length mismatch with agent batch in backward log prob: "
                f"prev_nodes={int(prev_nodes.numel())}, total_agents={total_agents}."
            )
        if int(chosen_edge_ids.numel()) != total_agents:
            raise ValueError(
                "chosen_edge_ids length mismatch with agent batch in backward log prob: "
                f"chosen_edge_ids={int(chosen_edge_ids.numel())}, total_agents={total_agents}."
            )
        if int(active_flat.numel()) != total_agents:
            raise ValueError(
                "active_flat length mismatch with agent batch in backward log prob: "
                f"active_flat={int(active_flat.numel())}, total_agents={total_agents}."
            )
        if int(is_stop_flat.numel()) != total_agents:
            raise ValueError(
                "is_stop_flat length mismatch with agent batch in backward log prob: "
                f"is_stop_flat={int(is_stop_flat.numel())}, total_agents={total_agents}."
            )
        if action_cache is None:
            resolved_cache = self.encoder.build_action_cache(
                env_context=env_context,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
            )
        else:
            resolved_cache = action_cache
        question_context_tokens = resolved_cache.get("question_context_tokens")
        if question_context_tokens is None:
            raise ValueError(
                "action_cache must provide `question_context_tokens` for backward log prob."
            )
        question_padding_mask = resolved_cache.get("question_padding_mask")
        if question_padding_mask is None:
            raise ValueError(
                "action_cache must provide `question_padding_mask` for backward log prob."
            )
        lexical_question_tokens = resolved_cache.get("lexical_question_tokens")
        edge_disallowed_backward = resolved_cache.get("edge_disallowed_backward")

        flat_curr_nodes = agent_state.current_nodes.view(-1)
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

        active_move = active_flat & (~is_stop_flat)
        active_stop = active_flat & is_stop_flat
        log_pb = torch.zeros(
            (total_agents,), device=node_tokens.device, dtype=node_tokens.dtype
        )
        invalid_rows = torch.zeros(
            (total_agents,), device=node_tokens.device, dtype=torch.bool
        )
        if not bool(active_flat.any().item()):
            return log_pb, invalid_rows
        num_nodes_total = int(env_context.num_nodes_total)
        if num_nodes_total <= 0:
            raise ValueError("num_nodes_total must be > 0 for backward log prob.")
        num_edges_total = int(env_context.edge_index.size(1))
        if num_edges_total <= 0:
            return log_pb, invalid_rows

        if not bool(active_move.any().item()):
            stop_delta = self.stop_delta_head(
                agent_potential=agent_potential, dtype=node_tokens.dtype
            )
            stop_logits = self._compute_stop_logits(
                env_context=env_context,
                agent_state=agent_state,
                stop_delta=stop_delta,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                total_agents=total_agents,
                super_node_mask=None,
            ).view(-1)
            if stop_guard_active is not None:
                stop_guard = stop_guard_active.to(
                    device=node_tokens.device, dtype=torch.bool
                ).view(-1)
                if int(stop_guard.numel()) != total_agents:
                    raise ValueError(
                        "stop_guard_active size mismatch with agent batch in backward log prob."
                    )
                stop_logits = stop_logits.masked_fill(
                    stop_guard,
                    torch.tensor(
                        float("-inf"),
                        device=stop_logits.device,
                        dtype=stop_logits.dtype,
                    ),
                )
                invalid_rows = invalid_rows | (stop_guard & active_stop)
            stop_log_prob = torch.where(
                torch.isfinite(stop_logits), stop_logits, torch.zeros_like(stop_logits)
            )
            if bool(active_stop.any().item()):
                log_pb = torch.where(active_stop, stop_log_prob, log_pb)
            return log_pb, invalid_rows

        chosen_parents = prev_nodes.to(device=node_tokens.device, dtype=torch.long)
        if bool((chosen_parents < 0).any().item()) or bool(
            (chosen_parents >= int(env_context.num_nodes_total)).any().item()
        ):
            raise ValueError("prev_nodes out of range for backward log prob.")
        chosen_edge_ids = chosen_edge_ids.to(
            device=node_tokens.device, dtype=torch.long
        )
        invalid_chosen = (chosen_edge_ids < 0) | (chosen_edge_ids >= num_edges_total)
        if bool((invalid_chosen & active_move).any().item()):
            raise ValueError("chosen_edge_ids out of range for backward log prob.")
        active_move_for_parents = active_move
        if not bool(active_move_for_parents.any().item()):
            return log_pb, invalid_rows

        parent_edge_ids, parent_nodes, edge_agent_batch, empty_parent_rows = (
            gather_parent_edges_from_csr_helper(
                adj_t=env_context.adj_t_bwd,
                current_nodes=flat_curr_nodes,
                active_move=active_move_for_parents,
                num_nodes_total=num_nodes_total,
                exclude_parents=None,
            )
        )
        invalid_rows = invalid_rows | (empty_parent_rows & active_move)
        if int(parent_edge_ids.numel()) == 0:
            stop_delta = self.stop_delta_head(
                agent_potential=agent_potential, dtype=node_tokens.dtype
            )
            stop_logits = self._compute_stop_logits(
                env_context=env_context,
                agent_state=agent_state,
                stop_delta=stop_delta,
                device=node_tokens.device,
                dtype=node_tokens.dtype,
                total_agents=total_agents,
                super_node_mask=None,
            ).view(-1)
            if stop_guard_active is not None:
                stop_guard = stop_guard_active.to(
                    device=node_tokens.device, dtype=torch.bool
                ).view(-1)
                if int(stop_guard.numel()) != total_agents:
                    raise ValueError(
                        "stop_guard_active size mismatch with agent batch in backward log prob."
                    )
                stop_logits = stop_logits.masked_fill(
                    stop_guard,
                    torch.tensor(
                        float("-inf"),
                        device=stop_logits.device,
                        dtype=stop_logits.dtype,
                    ),
                )
                invalid_rows = invalid_rows | (stop_guard & active_stop)
            stop_log_prob = torch.where(
                torch.isfinite(stop_logits), stop_logits, torch.zeros_like(stop_logits)
            )
            if bool(active_stop.any().item()):
                log_pb = torch.where(active_stop, stop_log_prob, log_pb)
            return log_pb, invalid_rows

        edge_relations = env_context.edge_relations.index_select(0, parent_edge_ids)

        edge_logits, edge_group_index, group_keys, edge_keys = (
            compute_parent_log_probs_helper(
                agent_potential=agent_potential,
                node_tokens=node_tokens,
                relation_tokens=relation_tokens,
                node_to_policy=self.projections.node_to_policy,
                relation_to_policy=self.projections.relation_to_policy,
                edge_action_encoder=self.edge_scorer.edge_action_encoder,
                edge_action_norm=self.edge_scorer.edge_action_norm,
                relation_group_head=self.edge_scorer.relation_group_head,
                relation_lexical_proj=self.edge_scorer.relation_lexical_proj,
                lexical_bias_log_scale=self.edge_scorer.lexical_bias_log_scale,
                parent_nodes=parent_nodes,
                parent_edge_ids=parent_edge_ids,
                edge_relations=edge_relations,
                edge_agent_batch=edge_agent_batch,
                lexical_question_tokens=lexical_question_tokens,
                agent_question_padding_mask=agent_question_padding_mask,
                total_agents=total_agents,
                num_edges_total=num_edges_total,
            )
        )
        if edge_disallowed_backward is not None and int(edge_logits.numel()) > 0:
            disallowed_edges = edge_disallowed_backward.index_select(
                0, parent_edge_ids.clamp(min=0)
            )
            neg_inf = torch.tensor(
                float("-inf"), device=edge_logits.device, dtype=edge_logits.dtype
            )
            edge_logits = edge_logits.masked_fill(disallowed_edges, neg_inf)
        edge_lse, _ = segment_logsumexp_1d(
            values=edge_logits.to(dtype=torch.float32),
            segment_ids=edge_agent_batch.to(dtype=torch.long),
            num_segments=total_agents,
            dtype=torch.float32,
            ignore_non_finite=True,
            empty_value=float("-inf"),
        )
        stop_delta = self.stop_delta_head(
            agent_potential=agent_potential, dtype=node_tokens.dtype
        )
        stop_logits = self._compute_stop_logits(
            env_context=env_context,
            agent_state=agent_state,
            stop_delta=stop_delta,
            device=node_tokens.device,
            dtype=node_tokens.dtype,
            edge_logits=edge_logits,
            edge_agent_batch=edge_agent_batch,
            total_agents=total_agents,
            super_node_mask=None,
        ).view(-1)
        if stop_guard_active is not None:
            stop_guard = stop_guard_active.to(
                device=node_tokens.device, dtype=torch.bool
            ).view(-1)
            if int(stop_guard.numel()) != total_agents:
                raise ValueError(
                    "stop_guard_active size mismatch with agent batch in backward log prob."
                )
            stop_logits = stop_logits.masked_fill(
                stop_guard,
                torch.tensor(
                    float("-inf"),
                    device=stop_logits.device,
                    dtype=stop_logits.dtype,
                ),
            )
            invalid_rows = invalid_rows | (stop_guard & active_stop)
        num_relations = int(relation_tokens.size(0))
        if num_relations <= 0:
            raise ValueError("relation_tokens must contain at least one relation.")
        num_groups = int(group_keys.numel())
        if num_groups == 0:
            invalid_rows = invalid_rows | active_move_for_parents
            return log_pb, invalid_rows

        group_log_mass, _ = segment_logsumexp_1d(
            values=edge_logits.to(dtype=torch.float32),
            segment_ids=edge_group_index.to(dtype=torch.long),
            num_segments=num_groups,
            dtype=torch.float32,
            ignore_non_finite=True,
            empty_value=float("-inf"),
        )
        group_agent_ids = torch.div(
            group_keys.to(dtype=torch.long), num_relations, rounding_mode="floor"
        )
        relation_lse, relation_has_finite = segment_logsumexp_1d(
            values=group_log_mass,
            segment_ids=group_agent_ids,
            num_segments=total_agents,
            dtype=torch.float32,
            ignore_non_finite=True,
            empty_value=float("-inf"),
        )
        invalid_rel = ~relation_has_finite
        invalid_rows = invalid_rows | (invalid_rel & active_move)
        log_partition = torch.logaddexp(edge_lse, stop_logits.to(dtype=torch.float32))
        edge_log_prob_raw = edge_logits.to(
            dtype=torch.float32
        ) - log_partition.index_select(0, edge_agent_batch)
        neg_inf = torch.tensor(
            float("-inf"),
            device=edge_log_prob_raw.device,
            dtype=edge_log_prob_raw.dtype,
        )
        # Keep impossible edges at -inf (do not 0-fill); if such an edge is chosen,
        # it must be treated as invalid backward support.
        edge_log_prob = torch.where(
            torch.isfinite(edge_log_prob_raw), edge_log_prob_raw, neg_inf
        )

        sorted_keys, order = torch.sort(edge_keys)
        sorted_log_prob = edge_log_prob.index_select(0, order)
        active_rows = torch.where(active_move)[0]
        chosen_log_prob, valid_rows, missing_rows = select_parent_log_prob_helper(
            parent_log_prob=sorted_log_prob,
            unique_keys=sorted_keys,
            chosen_edge_ids=chosen_edge_ids,
            active_rows=active_rows,
            num_edges_total=num_edges_total,
            total_agents=total_agents,
        )
        invalid_rows = invalid_rows | missing_rows
        if int(valid_rows.numel()) > 0:
            chosen_finite = torch.isfinite(chosen_log_prob)
            if not bool(chosen_finite.all().item()):
                invalid_rows.index_fill_(0, valid_rows[~chosen_finite], True)
            if bool(chosen_finite.any().item()):
                good_rows = valid_rows[chosen_finite]
                good_log_prob = chosen_log_prob[chosen_finite]
                log_pb.index_copy_(
                    0, good_rows, good_log_prob.to(dtype=node_tokens.dtype)
                )
        if bool(active_stop.any().item()):
            stop_log_prob = stop_logits.to(dtype=torch.float32) - log_partition
            stop_log_prob = torch.where(
                torch.isfinite(stop_log_prob), stop_log_prob, neg_inf
            )
            stop_finite = torch.isfinite(stop_log_prob)
            invalid_rows = invalid_rows | (active_stop & (~stop_finite))
            stop_log_prob = torch.where(
                stop_finite, stop_log_prob, torch.zeros_like(stop_log_prob)
            )
            log_pb = torch.where(
                active_stop, stop_log_prob.to(dtype=node_tokens.dtype), log_pb
            )
        return log_pb, invalid_rows


__all__ = ["BackwardLogProbHead"]
