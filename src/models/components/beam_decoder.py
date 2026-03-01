from __future__ import annotations

import torch

from src.models.configs.search import RolloutConfig
from src.models.environment import DynamicAgentState, GraphEnvContext

from .policy import DualFlowPolicy
from .rollout_types import STOP_REASON_ACTION, STOP_REASON_DEAD_END, STOP_REASON_MAX_STEPS, RolloutResult

EncodedPolicyContext = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_TRACE_HASH_SEED = 1469598103934665603
_TRACE_HASH_PRIME = 1099511628211
_TRACE_HASH_STEP_OFFSET = 1099511628211
_CANDIDATE_EXPANSION_FACTOR = 4


class BeamDecoderEngine:
    """Deterministic beam decoder used by eval/predict paths."""

    def __init__(self, *, config: RolloutConfig) -> None:
        self.config = config

    @staticmethod
    def _expand_grouped_start_nodes(
        *,
        q_local_indices: torch.Tensor,
        q_ptr: torch.Tensor,
        num_agents: int,
        deterministic: bool,
    ) -> torch.Tensor:
        num_graphs = int(q_ptr.numel()) - 1
        counts = (q_ptr[1:] - q_ptr[:-1]).clamp(min=0)
        if int(counts.numel()) != num_graphs:
            raise ValueError("q_ptr shape mismatch in grouped start expansion.")
        if bool((counts <= 0).any().item()):
            raise ValueError("q_local_indices has empty groups; fail-fast by contract.")
        base = q_ptr[:-1].unsqueeze(1)
        slots = torch.arange(num_agents, device=q_local_indices.device, dtype=torch.long).unsqueeze(0)
        slots = slots.expand(num_graphs, -1)
        if deterministic:
            offsets = torch.remainder(slots, counts.unsqueeze(1))
        else:
            random_shift = torch.floor(
                torch.rand((num_graphs, 1), device=q_local_indices.device) * counts.unsqueeze(1).to(dtype=torch.float32)
            ).to(dtype=torch.long)
            offsets = torch.remainder(slots + random_shift, counts.unsqueeze(1))
        gather_idx = (base + offsets).reshape(-1)
        return q_local_indices.index_select(0, gather_idx).view(num_graphs, num_agents)

    @staticmethod
    def _init_agent_state(
        *,
        env_context: GraphEnvContext,
        num_agents: int,
        deterministic: bool,
    ) -> DynamicAgentState:
        device = env_context.node_embeddings.device
        num_graphs = int(env_context.num_graphs)
        start_local = BeamDecoderEngine._expand_grouped_start_nodes(
            q_local_indices=env_context.q_local_indices,
            q_ptr=env_context.q_ptr,
            num_agents=num_agents,
            deterministic=deterministic,
        )
        start_nodes_absolute = start_local + env_context.node_ptr[:-1].unsqueeze(1)
        current_nodes = start_nodes_absolute.clone()
        hidden_states = env_context.question_emb.unsqueeze(1).expand(num_graphs, num_agents, -1).clone()
        visited_mask = torch.zeros(
            (num_graphs * num_agents, env_context.num_nodes_total),
            dtype=torch.bool,
            device=device,
        )
        row_ids = torch.arange(num_graphs * num_agents, device=device, dtype=torch.long)
        col_ids = current_nodes.view(-1)
        visited_mask[row_ids, col_ids] = True
        return DynamicAgentState(
            step_t=0,
            current_nodes=current_nodes,
            hidden_states=hidden_states,
            visited_mask=visited_mask,
            cumulative_rewards=torch.zeros((num_graphs, num_agents), device=device),
            done_mask=torch.zeros((num_graphs, num_agents), dtype=torch.bool, device=device),
        )

    @staticmethod
    def _seed_trace_hash(current_nodes: torch.Tensor) -> torch.Tensor:
        seed = torch.full_like(current_nodes, fill_value=_TRACE_HASH_SEED, dtype=torch.long)
        return torch.bitwise_xor(seed, current_nodes.to(dtype=torch.long) + 1)

    @staticmethod
    def _transition_trace_hash(
        parent_trace_hash: torch.Tensor,
        edge_ids: torch.Tensor,
        is_stop: torch.Tensor,
    ) -> torch.Tensor:
        stop_token = torch.ones_like(edge_ids, dtype=torch.long)
        move_token = edge_ids.to(dtype=torch.long).clamp(min=0) + 2
        token = torch.where(is_stop, stop_token, move_token)
        mixed = torch.bitwise_xor(parent_trace_hash.to(dtype=torch.long), token + _TRACE_HASH_STEP_OFFSET)
        return mixed * _TRACE_HASH_PRIME

    @staticmethod
    def _select_indices_with_dedup(
        *,
        flat_rank_scores: torch.Tensor,
        flat_targets: torch.Tensor,
        flat_parent_trace_hash: torch.Tensor,
        beam_size: int,
    ) -> torch.Tensor:
        num_graphs, candidate_count = flat_rank_scores.shape
        device = flat_rank_scores.device
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=flat_rank_scores.dtype)
        selected = torch.zeros((num_graphs, beam_size), dtype=torch.long, device=device)
        for graph_idx in range(num_graphs):
            scores = flat_rank_scores[graph_idx]
            finite_mask = torch.isfinite(scores)
            if not bool(finite_mask.any().item()):
                continue
            valid_idx = torch.where(finite_mask)[0]
            valid_scores = scores.index_select(0, valid_idx)
            dedup_pairs = torch.stack(
                (
                    flat_targets[graph_idx].index_select(0, valid_idx),
                    flat_parent_trace_hash[graph_idx].index_select(0, valid_idx),
                ),
                dim=1,
            )
            _, inverse = torch.unique(dedup_pairs, dim=0, return_inverse=True, sorted=False)
            num_groups = int(inverse.max().item()) + 1 if int(inverse.numel()) > 0 else 0
            group_best_score = torch.full((num_groups,), neg_inf, device=device, dtype=scores.dtype)
            group_best_score.scatter_reduce_(0, inverse, valid_scores, reduce="amax", include_self=True)

            candidate_positions = torch.arange(valid_idx.numel(), device=device, dtype=torch.long)
            invalid_positions = torch.full_like(candidate_positions, fill_value=int(valid_idx.numel()))
            is_group_best = valid_scores == group_best_score.index_select(0, inverse)
            best_candidate_pos = torch.where(is_group_best, candidate_positions, invalid_positions)
            group_first_best_pos = torch.full(
                (num_groups,),
                fill_value=int(valid_idx.numel()),
                device=device,
                dtype=torch.long,
            )
            group_first_best_pos.scatter_reduce_(0, inverse, best_candidate_pos, reduce="amin", include_self=True)
            keep_pos = group_first_best_pos[group_first_best_pos < int(valid_idx.numel())]
            if int(keep_pos.numel()) == 0:
                continue

            dedup_idx = valid_idx.index_select(0, keep_pos)
            dedup_scores = scores.index_select(0, dedup_idx)
            top_dedup_k = min(beam_size, int(dedup_idx.numel()))
            top_dedup_order = torch.topk(dedup_scores, k=top_dedup_k, dim=0).indices
            graph_selected = dedup_idx.index_select(0, top_dedup_order)
            selected_count = int(graph_selected.numel())

            if selected_count < beam_size:
                full_order = torch.topk(scores, k=candidate_count, dim=0).indices
                seen = torch.zeros((candidate_count,), dtype=torch.bool, device=device)
                seen[graph_selected] = True
                fill_candidates = full_order[~seen.index_select(0, full_order)]
                fill_needed = beam_size - selected_count
                if fill_needed > 0 and int(fill_candidates.numel()) > 0:
                    graph_selected = torch.cat((graph_selected, fill_candidates[:fill_needed]), dim=0)
                    selected_count = int(graph_selected.numel())

            if selected_count == 0:
                continue
            if selected_count < beam_size:
                graph_selected = torch.cat((graph_selected, graph_selected[:1].expand(beam_size - selected_count)), dim=0)
            selected[graph_idx] = graph_selected[:beam_size]
        return selected

    def beam_search_forward(
        self,
        env_context: GraphEnvContext,
        policy: DualFlowPolicy,
        *,
        beam_size: int,
        max_steps: int,
        require_done: bool,
        diverse_penalty: float = 0.0,
        encoded_context: EncodedPolicyContext | None = None,
    ) -> RolloutResult:
        beam_size = int(beam_size)
        max_steps = int(max_steps)
        if beam_size <= 0:
            raise ValueError("eval.beam_size must be a positive integer.")
        if max_steps <= 0:
            raise ValueError("eval.max_steps must be a positive integer.")
        stop_min_steps = int(self.config.stop_min_steps)
        if stop_min_steps < 0:
            raise ValueError("sampling.stop_min_steps must be >= 0.")
        diverse_penalty = max(float(diverse_penalty), 0.0)

        num_graphs = int(env_context.num_graphs)
        device = env_context.node_embeddings.device
        agent_state = self._init_agent_state(env_context=env_context, num_agents=beam_size, deterministic=True)
        current_nodes = agent_state.current_nodes
        hidden_states = agent_state.hidden_states
        visited_mask = agent_state.visited_mask
        done_mask = agent_state.done_mask
        trace_hash = self._seed_trace_hash(current_nodes)
        log_pf_sum = torch.zeros((num_graphs, beam_size), device=device)
        num_moves = torch.zeros((num_graphs, beam_size), dtype=torch.long, device=device)
        num_steps = torch.zeros((num_graphs, beam_size), dtype=torch.long, device=device)
        stop_reason = torch.zeros((num_graphs, beam_size), dtype=torch.long, device=device)

        with torch.no_grad():
            if encoded_context is None:
                node_tokens, relation_tokens, question_tokens = policy.encode_context(env_context)
            else:
                node_tokens, relation_tokens, question_tokens = encoded_context
            build_cache_fn = getattr(policy, "build_action_cache", None)
            action_cache: dict[str, torch.Tensor | None] | None = None
            if callable(build_cache_fn):
                action_cache = build_cache_fn(
                    env_context=env_context,
                    node_tokens=node_tokens,
                    question_tokens=question_tokens,
                )
            for step in range(max_steps):
                if done_mask.all():
                    break
                eval_state = DynamicAgentState(
                    step_t=step,
                    current_nodes=current_nodes,
                    hidden_states=hidden_states,
                    visited_mask=visited_mask,
                    cumulative_rewards=torch.zeros((num_graphs, beam_size), device=device),
                    done_mask=done_mask,
                )
                if action_cache is None:
                    policy_out = policy.compute_action_scores(
                        env_context=env_context,
                        agent_state=eval_state,
                        node_tokens=node_tokens,
                        question_tokens=question_tokens,
                        relation_tokens=relation_tokens,
                    )
                else:
                    policy_out = policy.compute_action_scores(
                        env_context=env_context,
                        agent_state=eval_state,
                        node_tokens=node_tokens,
                        question_tokens=question_tokens,
                        relation_tokens=relation_tokens,
                        action_cache=action_cache,
                    )
                out_degrees_flat = policy_out["out_degrees"].view(-1)
                edge_logits = policy_out["edge_logits"]
                edge_ids = policy_out["edge_ids"]
                target_nodes = policy_out["target_nodes"]
                stop_logits = policy_out["stop_logits"].view(-1)
                offsets = out_degrees_flat.cumsum(0) - out_degrees_flat
                total_agents = num_graphs * beam_size
                if int(out_degrees_flat.numel()) != total_agents:
                    raise ValueError("beam_search_forward: out_degrees shape mismatch with num_graphs * beam_size.")

                done_flat = done_mask.view(-1)
                parent_nodes_flat = current_nodes.view(-1)
                parent_log_pf_flat = log_pf_sum.view(-1)
                parent_stop_reason_flat = stop_reason.view(-1)
                parent_trace_flat = trace_hash.view(-1)

                max_deg = int(out_degrees_flat.max().item()) if int(out_degrees_flat.numel()) > 0 else 0
                neg_inf = torch.tensor(float("-inf"), device=device, dtype=log_pf_sum.dtype)
                candidate_width = min(max(beam_size * _CANDIDATE_EXPANSION_FACTOR, beam_size), max_deg + 1)

                edge_logits_dense = torch.full((total_agents, max_deg), neg_inf, device=device, dtype=log_pf_sum.dtype)
                if max_deg > 0:
                    edge_cols = torch.arange(max_deg, device=device, dtype=torch.long)
                    edge_valid = edge_cols.unsqueeze(0) < out_degrees_flat.unsqueeze(1)
                    edge_logits_dense[edge_valid] = edge_logits.to(dtype=log_pf_sum.dtype)
                else:
                    edge_valid = torch.empty((total_agents, 0), dtype=torch.bool, device=device)

                finite_move_exists = (torch.isfinite(edge_logits_dense) & edge_valid).any(dim=1)
                allow_stop = torch.ones((total_agents,), dtype=torch.bool, device=device)
                if step < stop_min_steps:
                    allow_stop = ~((out_degrees_flat > 0) & finite_move_exists)
                stop_valid = allow_stop & (~done_flat)

                candidate_logits = torch.cat([edge_logits_dense, stop_logits.to(dtype=log_pf_sum.dtype).unsqueeze(1)], dim=1)
                action_valid = torch.cat([edge_valid, stop_valid.unsqueeze(1)], dim=1)
                candidate_logits = candidate_logits.masked_fill(~action_valid, neg_inf)

                has_finite_candidate = torch.isfinite(candidate_logits).any(dim=1)
                active_flat = ~done_flat
                regular_flat = active_flat & has_finite_candidate
                dead_end_flat = active_flat & (~has_finite_candidate)

                cand_true_scores = torch.full(
                    (total_agents, candidate_width),
                    neg_inf,
                    device=device,
                    dtype=log_pf_sum.dtype,
                )
                cand_rank_scores = torch.full_like(cand_true_scores, neg_inf)
                cand_is_stop = torch.ones((total_agents, candidate_width), dtype=torch.bool, device=device)
                cand_targets = parent_nodes_flat.unsqueeze(1).expand(-1, candidate_width).clone()
                cand_edges = torch.full((total_agents, candidate_width), -1, dtype=torch.long, device=device)
                cand_reasons = torch.zeros((total_agents, candidate_width), dtype=torch.long, device=device)
                cand_parent_trace = parent_trace_flat.unsqueeze(1).expand(-1, candidate_width).clone()

                done_rows = torch.where(done_flat)[0]
                if int(done_rows.numel()) > 0:
                    done_scores = parent_log_pf_flat.index_select(0, done_rows)
                    cand_true_scores[done_rows, 0] = done_scores
                    cand_rank_scores[done_rows, 0] = done_scores
                    cand_reasons[done_rows, 0] = parent_stop_reason_flat.index_select(0, done_rows)

                dead_rows = torch.where(dead_end_flat)[0]
                if int(dead_rows.numel()) > 0:
                    dead_scores = parent_log_pf_flat.index_select(0, dead_rows)
                    cand_true_scores[dead_rows, 0] = dead_scores
                    cand_rank_scores[dead_rows, 0] = dead_scores
                    cand_reasons[dead_rows, 0] = torch.full(
                        (dead_rows.numel(),), STOP_REASON_DEAD_END, dtype=torch.long, device=device
                    )

                regular_rows = torch.where(regular_flat)[0]
                if int(regular_rows.numel()) > 0:
                    regular_logits = candidate_logits.index_select(0, regular_rows)
                    regular_action_count = action_valid.index_select(0, regular_rows).sum(dim=1).to(dtype=torch.long)
                    topk_width = min(candidate_width, int(regular_logits.size(1)))

                    regular_log_probs = torch.log_softmax(regular_logits, dim=1)
                    topk_log_prob, topk_action = torch.topk(regular_log_probs, k=topk_width, dim=1)

                    rank_idx = torch.arange(topk_width, device=device, dtype=torch.long)
                    local_k = regular_action_count.clamp(max=candidate_width)
                    valid_rank = rank_idx.unsqueeze(0) < local_k.unsqueeze(1)

                    regular_parent_log_pf = parent_log_pf_flat.index_select(0, regular_rows).unsqueeze(1)
                    regular_true_scores = regular_parent_log_pf + topk_log_prob
                    rank_penalty = diverse_penalty * rank_idx.to(dtype=regular_true_scores.dtype).unsqueeze(0)
                    regular_rank_scores = regular_true_scores - rank_penalty

                    regular_true_scores = torch.where(
                        valid_rank, regular_true_scores, torch.full_like(regular_true_scores, neg_inf)
                    )
                    regular_rank_scores = torch.where(
                        valid_rank, regular_rank_scores, torch.full_like(regular_rank_scores, neg_inf)
                    )

                    regular_parent_nodes = parent_nodes_flat.index_select(0, regular_rows).unsqueeze(1)
                    if int(edge_ids.numel()) == 0:
                        selected_targets = regular_parent_nodes.expand(-1, topk_width).clone()
                        selected_edges = torch.full_like(topk_action, -1)
                    else:
                        regular_offsets = offsets.index_select(0, regular_rows).unsqueeze(1)
                        flat_edge_idx = regular_offsets + topk_action
                        safe_edge_idx = flat_edge_idx.clamp(min=0, max=int(edge_ids.numel()) - 1)
                        selected_targets = target_nodes.index_select(0, safe_edge_idx.reshape(-1)).view_as(topk_action)
                        selected_edges = edge_ids.index_select(0, safe_edge_idx.reshape(-1)).view_as(topk_action)

                    stop_col = max_deg
                    regular_is_stop = topk_action == stop_col
                    selected_targets = torch.where(regular_is_stop, regular_parent_nodes, selected_targets)
                    selected_edges = torch.where(regular_is_stop, torch.full_like(selected_edges, -1), selected_edges)

                    regular_move_reason = torch.zeros_like(selected_edges, dtype=torch.long)
                    regular_move_finite = finite_move_exists.index_select(0, regular_rows).unsqueeze(1)
                    stop_reason_code = torch.where(
                        regular_move_finite,
                        torch.full_like(regular_move_reason, STOP_REASON_ACTION),
                        torch.full_like(regular_move_reason, STOP_REASON_DEAD_END),
                    )
                    regular_reasons = torch.where(regular_is_stop, stop_reason_code, regular_move_reason)

                    write_cols = torch.arange(topk_width, device=device, dtype=torch.long).unsqueeze(0)
                    write_cols = write_cols.expand(int(regular_rows.numel()), -1)
                    write_rows = regular_rows.unsqueeze(1).expand_as(write_cols)

                    cand_true_scores[write_rows, write_cols] = regular_true_scores
                    cand_rank_scores[write_rows, write_cols] = regular_rank_scores
                    cand_is_stop[write_rows, write_cols] = regular_is_stop
                    cand_targets[write_rows, write_cols] = selected_targets
                    cand_edges[write_rows, write_cols] = selected_edges
                    cand_reasons[write_rows, write_cols] = regular_reasons

                flat_rank_scores = cand_rank_scores.view(num_graphs, beam_size * candidate_width)

                flat_true_scores = cand_true_scores.view(num_graphs, beam_size * candidate_width)
                flat_is_stop = cand_is_stop.view(num_graphs, beam_size * candidate_width)
                flat_targets = cand_targets.view(num_graphs, beam_size * candidate_width)
                flat_edges = cand_edges.view(num_graphs, beam_size * candidate_width)
                flat_reasons = cand_reasons.view(num_graphs, beam_size * candidate_width)
                flat_parent_trace = cand_parent_trace.view(num_graphs, beam_size * candidate_width)
                select_indices = self._select_indices_with_dedup(
                    flat_rank_scores=flat_rank_scores,
                    flat_targets=flat_targets,
                    flat_parent_trace_hash=flat_parent_trace,
                    beam_size=beam_size,
                )

                sel_true_scores = flat_true_scores.gather(1, select_indices)
                sel_is_stop = flat_is_stop.gather(1, select_indices)
                sel_targets = flat_targets.gather(1, select_indices)
                sel_edges = flat_edges.gather(1, select_indices)
                sel_reasons = flat_reasons.gather(1, select_indices)
                sel_parent_trace = flat_parent_trace.gather(1, select_indices)

                sel_parents = torch.div(select_indices, candidate_width, rounding_mode="floor")
                graph_offsets = torch.arange(num_graphs, device=device, dtype=torch.long).unsqueeze(1) * beam_size
                parent_rows = (sel_parents + graph_offsets).reshape(-1)

                selected_visited = visited_mask.index_select(0, parent_rows).clone()
                move_mask_flat = (~sel_is_stop).reshape(-1)
                if bool(move_mask_flat.any().item()):
                    row_ids = torch.where(move_mask_flat)[0]
                    col_ids = sel_targets.reshape(-1).index_select(0, row_ids)
                    selected_visited[row_ids, col_ids] = True

                sel_parent_nodes = current_nodes.gather(1, sel_parents)
                next_current = torch.where(sel_is_stop, sel_parent_nodes, sel_targets)

                hidden_dim = int(hidden_states.size(-1))
                sel_parent_hidden = hidden_states.view(-1, hidden_dim).index_select(0, parent_rows)
                next_hidden = sel_parent_hidden.view(num_graphs, beam_size, hidden_dim).clone()
                if bool(move_mask_flat.any().item()):
                    flat_next_hidden = next_hidden.view(-1, hidden_dim)
                    move_targets = sel_targets.reshape(-1)[move_mask_flat]
                    move_edges = sel_edges.reshape(-1)[move_mask_flat].clamp(min=0)
                    move_rel = env_context.edge_relations.index_select(0, move_edges)
                    move_node_emb = node_tokens.index_select(0, move_targets)
                    move_rel_emb = relation_tokens.index_select(0, move_rel)
                    gru_input = torch.cat([move_node_emb, move_rel_emb], dim=-1)
                    move_rows = torch.where(move_mask_flat)[0]
                    move_hidden = flat_next_hidden.index_select(0, move_rows)
                    flat_next_hidden[move_rows] = policy.memory_tracker(gru_input, move_hidden)
                    next_hidden = flat_next_hidden.view(num_graphs, beam_size, hidden_dim)

                sel_parent_done = done_mask.gather(1, sel_parents)
                sel_parent_reason = stop_reason.gather(1, sel_parents)
                next_done = sel_parent_done | sel_is_stop
                next_stop_reason = torch.where(sel_parent_done, sel_parent_reason, sel_reasons)
                transitioned_trace = self._transition_trace_hash(sel_parent_trace, sel_edges, sel_is_stop)
                next_trace_hash = torch.where(sel_is_stop, sel_parent_trace, transitioned_trace)
                next_trace_hash = torch.where(sel_parent_done, sel_parent_trace, next_trace_hash)

                sel_parent_moves = num_moves.gather(1, sel_parents)
                move_increment = ((~sel_is_stop) & (~sel_parent_done)).to(dtype=torch.long)
                next_num_moves = sel_parent_moves + move_increment
                sel_parent_steps = num_steps.gather(1, sel_parents)
                step_increment = (~sel_parent_done).to(dtype=torch.long)
                next_num_steps = sel_parent_steps + step_increment

                current_nodes = next_current
                hidden_states = next_hidden
                done_mask = next_done
                log_pf_sum = sel_true_scores
                num_moves = next_num_moves
                num_steps = next_num_steps
                stop_reason = next_stop_reason
                visited_mask = selected_visited
                trace_hash = next_trace_hash

        unfinished = ~done_mask
        if bool(unfinished.any().item()):
            stop_reason = torch.where(
                unfinished,
                torch.full_like(stop_reason, STOP_REASON_MAX_STEPS),
                stop_reason,
            )
        stop_nodes = current_nodes
        if require_done:
            stop_nodes = torch.where(done_mask, stop_nodes, torch.full_like(stop_nodes, -1))
        return RolloutResult(
            log_pf_sum=log_pf_sum,
            stop_nodes=stop_nodes,
            num_moves=num_moves,
            num_steps=num_steps,
            stop_reason=stop_reason,
            actions=None,
            log_pf_steps=None,
            log_pb_steps=None,
            log_f_steps=None,
            policy_metrics=None,
        )


__all__ = ["BeamDecoderEngine"]
