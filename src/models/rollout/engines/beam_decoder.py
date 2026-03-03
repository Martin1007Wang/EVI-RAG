from __future__ import annotations

import torch

from src.models.configs.search import RolloutConfig
from src.models.environment import DynamicAgentState, FlowDirection, GraphEnvContext

from src.models.policy import DualFlowPolicy
from ..state_init import (
    compute_effective_max_steps,
    compute_max_allowed_moves,
    compute_min_required_moves,
    initialize_agent_state,
)
from ..types import (
    STOP_REASON_ACTION,
    STOP_REASON_DEAD_END,
    STOP_REASON_MAX_STEPS_REACHED,
    RolloutResult,
)

EncodedPolicyContext = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_TRACE_HASH_SEED = 1469598103934665603
_TRACE_HASH_PRIME = 1099511628211
_TRACE_HASH_STEP_OFFSET = 1099511628211
_DEFAULT_CANDIDATE_EXPANSION_FACTOR = 8


class BeamDecoderEngine:
    """Deterministic beam decoder used by eval/predict paths."""

    def __init__(self, *, config: RolloutConfig) -> None:
        self.config = config

    @staticmethod
    def _seed_trace_hash(current_nodes: torch.Tensor) -> torch.Tensor:
        seed = torch.full_like(
            current_nodes, fill_value=_TRACE_HASH_SEED, dtype=torch.long
        )
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
        mixed = torch.bitwise_xor(
            parent_trace_hash.to(dtype=torch.long), token + _TRACE_HASH_STEP_OFFSET
        )
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
        total_candidates = int(num_graphs * candidate_count)
        graph_ids = torch.arange(
            num_graphs, device=device, dtype=torch.long
        ).repeat_interleave(candidate_count)
        candidate_ids = torch.arange(
            candidate_count, device=device, dtype=torch.long
        ).repeat(num_graphs)
        flat_scores = flat_rank_scores.reshape(-1)
        flat_targets = flat_targets.reshape(-1)
        flat_parent_trace_hash = flat_parent_trace_hash.reshape(-1)
        graph_has_finite = BeamDecoderEngine._graph_has_finite_scores(
            scores=flat_scores,
            graph_ids=graph_ids,
            num_graphs=num_graphs,
        )
        dedup_idx = BeamDecoderEngine._dedup_candidate_indices(
            scores=flat_scores,
            targets=flat_targets,
            trace_hash=flat_parent_trace_hash,
            graph_ids=graph_ids,
        )
        dedup_selected, dedup_mask = BeamDecoderEngine._topk_per_graph(
            scores=flat_scores.index_select(0, dedup_idx),
            graph_ids=graph_ids.index_select(0, dedup_idx),
            candidate_ids=candidate_ids.index_select(0, dedup_idx),
            num_graphs=num_graphs,
            beam_size=beam_size,
        )
        return BeamDecoderEngine._fill_selected_candidates(
            flat_rank_scores=flat_rank_scores,
            dedup_selected=dedup_selected,
            dedup_mask=dedup_mask,
            graph_has_finite=graph_has_finite,
            beam_size=beam_size,
            candidate_count=candidate_count,
        )

    @staticmethod
    def _graph_has_finite_scores(
        *,
        scores: torch.Tensor,
        graph_ids: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        finite = torch.isfinite(scores).to(dtype=torch.int32)
        has_finite = torch.zeros((num_graphs,), device=scores.device, dtype=torch.int32)
        has_finite.scatter_reduce_(
            0, graph_ids, finite, reduce="amax", include_self=True
        )
        return has_finite > 0

    @staticmethod
    def _dedup_candidate_indices(
        *,
        scores: torch.Tensor,
        targets: torch.Tensor,
        trace_hash: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        device = scores.device
        total_candidates = int(scores.numel())
        if total_candidates == 0:
            return scores.new_zeros((0,), dtype=torch.long)
        valid_mask = torch.isfinite(scores)
        if not bool(valid_mask.any().item()):
            return scores.new_zeros((0,), dtype=torch.long)
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=scores.dtype)
        safe_scores = torch.where(valid_mask, scores, neg_inf)
        key = torch.stack((graph_ids, targets, trace_hash), dim=1)
        _, inverse = torch.unique(key, dim=0, return_inverse=True, sorted=False)
        num_groups = int(inverse.max().item()) + 1
        group_best = torch.full(
            (num_groups,), neg_inf, device=device, dtype=scores.dtype
        )
        group_best.scatter_reduce_(
            0, inverse, safe_scores, reduce="amax", include_self=True
        )
        candidate_idx = torch.arange(total_candidates, device=device, dtype=torch.long)
        is_best = valid_mask & (safe_scores == group_best.index_select(0, inverse))
        sentinel = total_candidates
        best_choice = torch.where(
            is_best,
            candidate_idx,
            torch.full_like(candidate_idx, sentinel),
        )
        best_pos = torch.full(
            (num_groups,), fill_value=sentinel, device=device, dtype=torch.long
        )
        best_pos.scatter_reduce_(
            0, inverse, best_choice, reduce="amin", include_self=True
        )
        return best_pos[best_pos < sentinel]

    @staticmethod
    def _topk_per_graph(
        *,
        scores: torch.Tensor,
        graph_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        num_graphs: int,
        beam_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = scores.device
        selected = torch.zeros((num_graphs, beam_size), dtype=torch.long, device=device)
        filled = torch.zeros((num_graphs, beam_size), dtype=torch.bool, device=device)
        if int(scores.numel()) == 0:
            return selected, filled
        score_order = torch.argsort(scores, descending=True, stable=True)
        graph_sorted = graph_ids.index_select(0, score_order)
        graph_order = torch.argsort(graph_sorted, stable=True)
        sorted_idx = score_order.index_select(0, graph_order)
        sorted_graph = graph_ids.index_select(0, sorted_idx)
        positions = torch.arange(
            int(sorted_idx.numel()), device=device, dtype=torch.long
        )
        group_start = torch.full(
            (num_graphs,),
            fill_value=int(sorted_idx.numel()),
            device=device,
            dtype=torch.long,
        )
        group_start.scatter_reduce_(
            0, sorted_graph, positions, reduce="amin", include_self=True
        )
        within = positions - group_start.index_select(0, sorted_graph)
        keep = within < beam_size
        if not bool(keep.any().item()):
            return selected, filled
        keep_idx = sorted_idx[keep]
        keep_graph = sorted_graph[keep]
        keep_pos = within[keep]
        keep_candidates = candidate_ids.index_select(0, keep_idx)
        selected[keep_graph, keep_pos] = keep_candidates
        filled[keep_graph, keep_pos] = True
        return selected, filled

    @staticmethod
    def _fill_selected_candidates(
        *,
        flat_rank_scores: torch.Tensor,
        dedup_selected: torch.Tensor,
        dedup_mask: torch.Tensor,
        graph_has_finite: torch.Tensor,
        beam_size: int,
        candidate_count: int,
    ) -> torch.Tensor:
        num_graphs = int(flat_rank_scores.size(0))
        device = flat_rank_scores.device
        full_order = torch.topk(flat_rank_scores, k=candidate_count, dim=1).indices
        seen = torch.zeros(
            (num_graphs, candidate_count), dtype=torch.bool, device=device
        )
        seen_rows, seen_cols = torch.where(dedup_mask)
        if int(seen_rows.numel()) > 0:
            seen[seen_rows, dedup_selected[seen_rows, seen_cols]] = True
        available = ~seen.gather(1, full_order)
        available_rank = available.cumsum(dim=1) - 1
        dedup_counts = dedup_mask.sum(dim=1)
        fill_needed = (beam_size - dedup_counts).clamp(min=0)
        take_mask = available & (available_rank < fill_needed.unsqueeze(1))
        fill_matrix = torch.full(
            (num_graphs, candidate_count),
            fill_value=-1,
            device=device,
            dtype=torch.long,
        )
        take_rows, take_cols = torch.where(take_mask)
        if int(take_rows.numel()) > 0:
            fill_matrix[take_rows, available_rank[take_rows, take_cols]] = full_order[
                take_rows, take_cols
            ]
        unfilled = ~dedup_mask
        unfilled_rank = unfilled.cumsum(dim=1) - 1
        max_rank = max(candidate_count - 1, 0)
        safe_unfilled_rank = unfilled_rank.clamp(min=0, max=max_rank)
        fill_from = fill_matrix.gather(1, safe_unfilled_rank)
        valid_fill_rank = unfilled_rank < candidate_count
        fill_from = torch.where(
            valid_fill_rank, fill_from, fill_from.new_full(fill_from.shape, -1)
        )
        selected = torch.where(unfilled, fill_from, dedup_selected)
        if candidate_count == 0:
            return torch.zeros_like(selected)
        first_selected = selected[:, :1].expand(-1, beam_size)
        needs_pad = selected < 0
        selected = torch.where(
            needs_pad & graph_has_finite.unsqueeze(1), first_selected, selected
        )
        return torch.where(
            graph_has_finite.unsqueeze(1),
            selected,
            selected.new_zeros(selected.shape),
        )

    def beam_search_forward(
        self,
        env_context: GraphEnvContext,
        policy: DualFlowPolicy,
        *,
        flow_direction: FlowDirection = "forward",
        beam_size: int,
        max_steps: int,
        require_done: bool,
        diverse_penalty: float = 0.0,
        candidate_expansion_factor: int = _DEFAULT_CANDIDATE_EXPANSION_FACTOR,
        encoded_context: EncodedPolicyContext | None = None,
    ) -> RolloutResult:
        if flow_direction not in {"forward", "backward"}:
            raise ValueError(f"Unsupported flow_direction: {flow_direction!r}.")
        beam_size = int(beam_size)
        max_steps = int(max_steps)
        if beam_size <= 0:
            raise ValueError("eval.beam_size must be a positive integer.")
        if max_steps <= 0:
            raise ValueError("eval.max_steps must be a positive integer.")
        candidate_expansion_factor = int(candidate_expansion_factor)
        if candidate_expansion_factor <= 0:
            raise ValueError(
                "eval.candidate_expansion_factor must be a positive integer."
            )
        stop_min_steps = int(self.config.stop_min_steps)
        if stop_min_steps < 0:
            raise ValueError("sampling.stop_min_steps must be >= 0.")
        diverse_penalty = max(float(diverse_penalty), 0.0)

        num_graphs = int(env_context.num_graphs)
        device = env_context.node_embeddings.device
        agent_state = initialize_agent_state(
            env_context=env_context,
            num_agents=beam_size,
            deterministic=True,
            flow_direction=flow_direction,
            backward_without_super_error=(
                "Backward beam decoding requires super-source layout to start from backward super."
            ),
        )
        current_nodes = agent_state.current_nodes
        hidden_states = agent_state.hidden_states
        visited_mask = agent_state.visited_mask
        done_mask = agent_state.done_mask
        path_token_ids = agent_state.path_token_ids
        path_token_types = agent_state.path_token_types
        path_lengths = agent_state.path_lengths
        if path_token_ids is None or path_token_types is None or path_lengths is None:
            raise ValueError(
                "Beam decoder initialization must provide path token state."
            )
        min_required_moves = compute_min_required_moves(
            env_context=env_context,
            start_nodes_abs=current_nodes,
            base_stop_min_steps=stop_min_steps,
            flatten=False,
        )
        max_allowed_moves = compute_max_allowed_moves(
            env_context=env_context,
            start_nodes_abs=current_nodes,
            base_max_steps=max_steps,
            flatten=False,
        )
        if bool((min_required_moves > max_allowed_moves).any().item()):
            raise ValueError(
                "Invalid stop/move budget: stop_min_steps exceeds max_steps for beam decoding."
            )
        effective_max_steps = compute_effective_max_steps(
            env_context=env_context,
            start_nodes_abs=current_nodes,
            base_max_steps=max_steps,
        )
        trace_hash = self._seed_trace_hash(current_nodes)
        log_pf_sum = torch.zeros((num_graphs, beam_size), device=device)
        num_moves = agent_state.num_moves
        num_steps = torch.zeros(
            (num_graphs, beam_size), dtype=torch.long, device=device
        )
        stop_reason = torch.zeros(
            (num_graphs, beam_size), dtype=torch.long, device=device
        )

        with torch.no_grad():
            if encoded_context is None:
                node_tokens, relation_tokens, question_tokens = policy.encode_context(
                    env_context
                )
            else:
                node_tokens, relation_tokens, question_tokens = encoded_context
            if hidden_states.dtype != node_tokens.dtype:
                hidden_states = hidden_states.to(dtype=node_tokens.dtype)
            build_cache_fn = getattr(policy, "build_action_cache", None)
            action_cache: dict[str, torch.Tensor] | None = None
            if callable(build_cache_fn):
                action_cache = build_cache_fn(
                    env_context=env_context,
                    node_tokens=node_tokens,
                    question_tokens=question_tokens,
                )
            for step in range(effective_max_steps):
                if done_mask.all():
                    break
                eval_state = DynamicAgentState(
                    step_t=step,
                    current_nodes=current_nodes,
                    flow_direction=flow_direction,
                    hidden_states=hidden_states,
                    visited_mask=visited_mask,
                    cumulative_rewards=torch.zeros(
                        (num_graphs, beam_size), device=device
                    ),
                    done_mask=done_mask,
                    num_moves=num_moves,
                    path_token_ids=path_token_ids,
                    path_token_types=path_token_types,
                    path_lengths=path_lengths,
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
                    raise ValueError(
                        "beam_search_forward: out_degrees shape mismatch with num_graphs * beam_size."
                    )

                done_flat = done_mask.view(-1)
                parent_nodes_flat = current_nodes.view(-1)
                parent_log_pf_flat = log_pf_sum.view(-1)
                parent_stop_reason_flat = stop_reason.view(-1)
                parent_trace_flat = trace_hash.view(-1)

                max_deg = (
                    int(out_degrees_flat.max().item())
                    if int(out_degrees_flat.numel()) > 0
                    else 0
                )
                neg_inf = torch.tensor(
                    float("-inf"), device=device, dtype=log_pf_sum.dtype
                )
                candidate_width = min(
                    max(beam_size * candidate_expansion_factor, beam_size), max_deg + 1
                )

                edge_logits_dense = torch.full(
                    (total_agents, max_deg),
                    neg_inf,
                    device=device,
                    dtype=log_pf_sum.dtype,
                )
                if max_deg > 0:
                    edge_cols = torch.arange(max_deg, device=device, dtype=torch.long)
                    edge_valid = edge_cols.unsqueeze(0) < out_degrees_flat.unsqueeze(1)
                    edge_logits_dense[edge_valid] = edge_logits.to(
                        dtype=log_pf_sum.dtype
                    )
                else:
                    edge_valid = torch.empty(
                        (total_agents, 0), dtype=torch.bool, device=device
                    )

                num_moves_flat = num_moves.view(-1)
                min_required_flat = min_required_moves.view(-1)
                max_allowed_flat = max_allowed_moves.view(-1)
                over_max_moves = num_moves_flat >= max_allowed_flat
                if bool(over_max_moves.any().item()) and max_deg > 0:
                    edge_valid = edge_valid & (~over_max_moves.unsqueeze(1))
                    edge_logits_dense = edge_logits_dense.masked_fill(
                        ~edge_valid, neg_inf
                    )
                finite_move_exists = (
                    torch.isfinite(edge_logits_dense) & edge_valid
                ).any(dim=1)
                need_more_moves = num_moves_flat < min_required_flat
                allow_stop = ~(
                    (out_degrees_flat > 0) & finite_move_exists & need_more_moves
                )
                stop_valid = allow_stop & (~done_flat)

                candidate_logits = torch.cat(
                    [
                        edge_logits_dense,
                        stop_logits.to(dtype=log_pf_sum.dtype).unsqueeze(1),
                    ],
                    dim=1,
                )
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
                cand_is_stop = torch.ones(
                    (total_agents, candidate_width), dtype=torch.bool, device=device
                )
                cand_targets = (
                    parent_nodes_flat.unsqueeze(1).expand(-1, candidate_width).clone()
                )
                cand_edges = torch.full(
                    (total_agents, candidate_width), -1, dtype=torch.long, device=device
                )
                cand_reasons = torch.zeros(
                    (total_agents, candidate_width), dtype=torch.long, device=device
                )
                cand_parent_trace = (
                    parent_trace_flat.unsqueeze(1).expand(-1, candidate_width).clone()
                )

                done_rows = torch.where(done_flat)[0]
                if int(done_rows.numel()) > 0:
                    done_scores = parent_log_pf_flat.index_select(0, done_rows)
                    cand_true_scores[done_rows, 0] = done_scores
                    cand_rank_scores[done_rows, 0] = done_scores
                    cand_reasons[done_rows, 0] = parent_stop_reason_flat.index_select(
                        0, done_rows
                    )

                dead_rows = torch.where(dead_end_flat)[0]
                if int(dead_rows.numel()) > 0:
                    dead_scores = parent_log_pf_flat.index_select(0, dead_rows)
                    cand_true_scores[dead_rows, 0] = dead_scores
                    cand_rank_scores[dead_rows, 0] = dead_scores
                    cand_reasons[dead_rows, 0] = torch.full(
                        (dead_rows.numel(),),
                        STOP_REASON_DEAD_END,
                        dtype=torch.long,
                        device=device,
                    )

                regular_rows = torch.where(regular_flat)[0]
                if int(regular_rows.numel()) > 0:
                    regular_logits = candidate_logits.index_select(0, regular_rows)
                    regular_action_count = (
                        action_valid.index_select(0, regular_rows)
                        .sum(dim=1)
                        .to(dtype=torch.long)
                    )
                    topk_width = min(candidate_width, int(regular_logits.size(1)))

                    regular_log_probs = torch.log_softmax(regular_logits, dim=1)
                    topk_log_prob, topk_action = torch.topk(
                        regular_log_probs, k=topk_width, dim=1
                    )

                    rank_idx = torch.arange(topk_width, device=device, dtype=torch.long)
                    local_k = regular_action_count.clamp(max=candidate_width)
                    valid_rank = rank_idx.unsqueeze(0) < local_k.unsqueeze(1)

                    regular_parent_log_pf = parent_log_pf_flat.index_select(
                        0, regular_rows
                    ).unsqueeze(1)
                    regular_true_scores = regular_parent_log_pf + topk_log_prob
                    rank_penalty = diverse_penalty * rank_idx.to(
                        dtype=regular_true_scores.dtype
                    ).unsqueeze(0)
                    regular_rank_scores = regular_true_scores - rank_penalty

                    regular_true_scores = torch.where(
                        valid_rank,
                        regular_true_scores,
                        torch.full_like(regular_true_scores, neg_inf),
                    )
                    regular_rank_scores = torch.where(
                        valid_rank,
                        regular_rank_scores,
                        torch.full_like(regular_rank_scores, neg_inf),
                    )

                    regular_parent_nodes = parent_nodes_flat.index_select(
                        0, regular_rows
                    ).unsqueeze(1)
                    if int(edge_ids.numel()) == 0:
                        selected_targets = regular_parent_nodes.expand(
                            -1, topk_width
                        ).clone()
                        selected_edges = torch.full_like(topk_action, -1)
                    else:
                        regular_offsets = offsets.index_select(
                            0, regular_rows
                        ).unsqueeze(1)
                        flat_edge_idx = regular_offsets + topk_action
                        safe_edge_idx = flat_edge_idx.clamp(
                            min=0, max=int(edge_ids.numel()) - 1
                        )
                        selected_targets = target_nodes.index_select(
                            0, safe_edge_idx.reshape(-1)
                        ).view_as(topk_action)
                        selected_edges = edge_ids.index_select(
                            0, safe_edge_idx.reshape(-1)
                        ).view_as(topk_action)

                    stop_col = max_deg
                    regular_is_stop = topk_action == stop_col
                    selected_targets = torch.where(
                        regular_is_stop, regular_parent_nodes, selected_targets
                    )
                    selected_edges = torch.where(
                        regular_is_stop,
                        torch.full_like(selected_edges, -1),
                        selected_edges,
                    )

                    regular_move_reason = torch.zeros_like(
                        selected_edges, dtype=torch.long
                    )
                    regular_move_finite = finite_move_exists.index_select(
                        0, regular_rows
                    ).unsqueeze(1)
                    regular_over_max = over_max_moves.index_select(
                        0, regular_rows
                    ).unsqueeze(1)
                    regular_stop_action = regular_move_finite | regular_over_max
                    stop_reason_code = torch.where(
                        regular_stop_action,
                        torch.full_like(regular_move_reason, STOP_REASON_ACTION),
                        torch.full_like(regular_move_reason, STOP_REASON_DEAD_END),
                    )
                    regular_reasons = torch.where(
                        regular_is_stop, stop_reason_code, regular_move_reason
                    )

                    write_cols = torch.arange(
                        topk_width, device=device, dtype=torch.long
                    ).unsqueeze(0)
                    write_cols = write_cols.expand(int(regular_rows.numel()), -1)
                    write_rows = regular_rows.unsqueeze(1).expand_as(write_cols)

                    cand_true_scores[write_rows, write_cols] = regular_true_scores
                    cand_rank_scores[write_rows, write_cols] = regular_rank_scores
                    cand_is_stop[write_rows, write_cols] = regular_is_stop
                    cand_targets[write_rows, write_cols] = selected_targets
                    cand_edges[write_rows, write_cols] = selected_edges
                    cand_reasons[write_rows, write_cols] = regular_reasons

                flat_rank_scores = cand_rank_scores.view(
                    num_graphs, beam_size * candidate_width
                )

                flat_true_scores = cand_true_scores.view(
                    num_graphs, beam_size * candidate_width
                )
                flat_is_stop = cand_is_stop.view(
                    num_graphs, beam_size * candidate_width
                )
                flat_targets = cand_targets.view(
                    num_graphs, beam_size * candidate_width
                )
                flat_edges = cand_edges.view(num_graphs, beam_size * candidate_width)
                flat_reasons = cand_reasons.view(
                    num_graphs, beam_size * candidate_width
                )
                flat_parent_trace = cand_parent_trace.view(
                    num_graphs, beam_size * candidate_width
                )
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

                sel_parents = torch.div(
                    select_indices, candidate_width, rounding_mode="floor"
                )
                graph_offsets = (
                    torch.arange(num_graphs, device=device, dtype=torch.long).unsqueeze(
                        1
                    )
                    * beam_size
                )
                parent_rows = (sel_parents + graph_offsets).reshape(-1)

                hidden_dim = int(hidden_states.size(-1))
                token_width = int(path_token_ids.size(-1))
                sel_parent_nodes = current_nodes.gather(1, sel_parents)
                sel_parent_hidden = hidden_states.view(-1, hidden_dim).index_select(
                    0, parent_rows
                )
                sel_parent_path_ids = (
                    path_token_ids.view(-1, token_width)
                    .index_select(0, parent_rows)
                    .view(num_graphs, beam_size, token_width)
                )
                sel_parent_path_types = (
                    path_token_types.view(-1, token_width)
                    .index_select(0, parent_rows)
                    .view(num_graphs, beam_size, token_width)
                )
                sel_parent_path_lengths = (
                    path_lengths.view(-1)
                    .index_select(0, parent_rows)
                    .view(num_graphs, beam_size)
                )
                selected_visited = visited_mask.index_select(0, parent_rows)
                sel_parent_done = done_mask.gather(1, sel_parents)
                sel_parent_reason = stop_reason.gather(1, sel_parents)
                sel_parent_min_required = min_required_moves.gather(1, sel_parents)
                sel_parent_moves = num_moves.gather(1, sel_parents)
                sel_parent_max_allowed = max_allowed_moves.gather(1, sel_parents)

                parent_state = DynamicAgentState(
                    step_t=step,
                    current_nodes=sel_parent_nodes,
                    flow_direction=flow_direction,
                    hidden_states=sel_parent_hidden.view(
                        num_graphs, beam_size, hidden_dim
                    ),
                    visited_mask=selected_visited,
                    cumulative_rewards=torch.zeros(
                        (num_graphs, beam_size), device=device
                    ),
                    done_mask=sel_parent_done,
                    num_moves=sel_parent_moves,
                    path_token_ids=sel_parent_path_ids,
                    path_token_types=sel_parent_path_types,
                    path_lengths=sel_parent_path_lengths,
                )
                chosen_edge_flat = sel_edges.reshape(-1).clamp(min=0)
                if int(env_context.edge_relations.numel()) == 0:
                    chosen_rel = torch.zeros_like(chosen_edge_flat)
                else:
                    chosen_rel = env_context.edge_relations.index_select(
                        0, chosen_edge_flat
                    )
                next_state = policy.evolve_state(
                    agent_state=parent_state,
                    chosen_target_nodes=sel_targets.reshape(-1),
                    chosen_edge_relations=chosen_rel,
                    node_tokens=node_tokens,
                    relation_tokens=relation_tokens,
                    is_stop=sel_is_stop.reshape(-1),
                )

                next_done = next_state.done_mask
                next_stop_reason = torch.where(
                    sel_parent_done, sel_parent_reason, sel_reasons
                )
                transitioned_trace = self._transition_trace_hash(
                    sel_parent_trace, sel_edges, sel_is_stop
                )
                next_trace_hash = torch.where(
                    sel_is_stop, sel_parent_trace, transitioned_trace
                )
                next_trace_hash = torch.where(
                    sel_parent_done, sel_parent_trace, next_trace_hash
                )

                sel_parent_steps = num_steps.gather(1, sel_parents)
                step_increment = (~sel_parent_done).to(dtype=torch.long)
                next_num_steps = sel_parent_steps + step_increment

                current_nodes = next_state.current_nodes
                hidden_states = next_state.hidden_states
                done_mask = next_done
                log_pf_sum = sel_true_scores
                num_moves = next_state.num_moves
                num_steps = next_num_steps
                stop_reason = next_stop_reason
                min_required_moves = sel_parent_min_required
                max_allowed_moves = sel_parent_max_allowed
                visited_mask = next_state.visited_mask
                if (
                    next_state.path_token_ids is None
                    or next_state.path_token_types is None
                    or next_state.path_lengths is None
                ):
                    path_token_ids = next_state.current_nodes.unsqueeze(-1).clone()
                    path_token_types = torch.zeros_like(
                        path_token_ids, dtype=torch.bool
                    )
                    path_lengths = torch.ones_like(
                        next_state.current_nodes, dtype=torch.long
                    )
                else:
                    path_token_ids = next_state.path_token_ids
                    path_token_types = next_state.path_token_types
                    path_lengths = next_state.path_lengths
                trace_hash = next_trace_hash

        unfinished = ~done_mask
        if bool(unfinished.any().item()):
            stop_reason = torch.where(
                unfinished,
                torch.full_like(stop_reason, STOP_REASON_MAX_STEPS_REACHED),
                stop_reason,
            )
        stop_nodes = current_nodes
        if require_done:
            stop_nodes = torch.where(
                done_mask, stop_nodes, torch.full_like(stop_nodes, -1)
            )
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
