from __future__ import annotations

import torch

from src.models.configs.search import RolloutConfig
from src.models.environment import DynamicAgentState, GraphEnvContext

from .backward_prior import StructuralBackwardPrior
from .policy import DualFlowPolicy
from .rollout_types import STOP_REASON_ACTION, STOP_REASON_DEAD_END, STOP_REASON_MAX_STEPS, RolloutResult


class OfflineForcedEvalEngine:
    """Forced-path evaluator used by offline replay mixing."""

    def __init__(self, *, config: RolloutConfig, backward_prior: StructuralBackwardPrior) -> None:
        self.config = config
        self.backward_prior = backward_prior

    @staticmethod
    def _compute_has_finite_edges(
        *,
        edge_logits: torch.Tensor,
        out_degrees: torch.Tensor,
    ) -> torch.Tensor:
        num_agents_total = int(out_degrees.numel())
        if num_agents_total == 0:
            return out_degrees.new_zeros((0,), dtype=torch.bool)
        if edge_logits.numel() == 0:
            return out_degrees.new_zeros((num_agents_total,), dtype=torch.bool)
        agent_ids = torch.arange(num_agents_total, device=out_degrees.device, dtype=torch.long)
        edge_agent_ids = agent_ids.repeat_interleave(out_degrees)
        finite_edges = torch.isfinite(edge_logits).to(dtype=torch.int32)
        has_finite = torch.zeros((num_agents_total,), device=out_degrees.device, dtype=torch.int32)
        has_finite.scatter_reduce_(0, edge_agent_ids, finite_edges, reduce="amax", include_self=True)
        return has_finite > 0

    def _mask_stop_logits_for_min_steps(
        self,
        *,
        policy_out: dict[str, torch.Tensor],
        active_flat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out_degrees_flat = policy_out["out_degrees"].view(-1)
        has_finite_edges = self._compute_has_finite_edges(
            edge_logits=policy_out["edge_logits"],
            out_degrees=out_degrees_flat,
        )
        ban_stop = active_flat & (out_degrees_flat > 0) & has_finite_edges
        if not bool(ban_stop.any().item()):
            return policy_out
        stop_logits_flat = policy_out["stop_logits"].view(-1)
        masked_stop = stop_logits_flat.masked_fill(
            ban_stop,
            torch.tensor(float("-inf"), device=stop_logits_flat.device, dtype=stop_logits_flat.dtype),
        )
        patched = dict(policy_out)
        patched["stop_logits"] = masked_stop.view_as(policy_out["stop_logits"])
        return patched

    def _compute_log_pb(
        self,
        *,
        env_context: GraphEnvContext,
        source_nodes: torch.Tensor,
        chosen_target_nodes: torch.Tensor,
        active_flat: torch.Tensor,
        is_stop_flat: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.backward_prior.log_prob(
            env_context=env_context,
            source_nodes=source_nodes,
            chosen_target_nodes=chosen_target_nodes,
            active_flat=active_flat,
            is_stop_flat=is_stop_flat,
            dtype=dtype,
        )

    def evaluate_forced_paths(
        self,
        env_context: GraphEnvContext,
        policy: DualFlowPolicy,
        *,
        start_local_indices: torch.Tensor,
        forced_edge_ids: torch.Tensor,
        path_lengths: torch.Tensor,
        collect_traces: bool = True,
        use_visited_mask: bool = False,
    ) -> RolloutResult:
        num_graphs = int(env_context.num_graphs)
        if start_local_indices.dim() != 2:
            raise ValueError(f"start_local_indices must be 2D [B, K], got {tuple(start_local_indices.shape)}")
        if forced_edge_ids.dim() != 3:
            raise ValueError(f"forced_edge_ids must be 3D [B, K, T], got {tuple(forced_edge_ids.shape)}")
        if path_lengths.dim() != 2:
            raise ValueError(f"path_lengths must be 2D [B, K], got {tuple(path_lengths.shape)}")
        if int(start_local_indices.size(0)) != num_graphs:
            raise ValueError("start_local_indices batch size mismatch in evaluate_forced_paths.")
        if int(forced_edge_ids.size(0)) != num_graphs:
            raise ValueError("forced_edge_ids batch size mismatch in evaluate_forced_paths.")
        if int(path_lengths.size(0)) != num_graphs:
            raise ValueError("path_lengths batch size mismatch in evaluate_forced_paths.")
        if int(forced_edge_ids.size(1)) != int(start_local_indices.size(1)):
            raise ValueError("forced_edge_ids rollout axis mismatch with start_local_indices.")
        if int(path_lengths.size(1)) != int(start_local_indices.size(1)):
            raise ValueError("path_lengths rollout axis mismatch with start_local_indices.")

        num_agents = int(start_local_indices.size(1))
        max_steps = int(forced_edge_ids.size(-1))
        stop_min_steps = int(self.config.stop_min_steps)
        if stop_min_steps < 0:
            raise ValueError("sampling.stop_min_steps must be >= 0.")

        device = env_context.node_embeddings.device
        start_local = start_local_indices.to(device=device, dtype=torch.long)
        node_offsets = env_context.node_ptr[:-1].to(device=device).view(num_graphs, 1)
        current_nodes = start_local + node_offsets
        hidden_states = env_context.question_emb.unsqueeze(1).expand(num_graphs, num_agents, -1).clone()
        visited_mask = torch.zeros(
            (num_graphs * num_agents, env_context.num_nodes_total),
            dtype=torch.bool,
            device=device,
        )
        if use_visited_mask:
            row_ids = torch.arange(num_graphs * num_agents, device=device, dtype=torch.long)
            col_ids = current_nodes.view(-1).clamp(min=0)
            visited_mask[row_ids, col_ids] = True
        agent_state = DynamicAgentState(
            step_t=0,
            current_nodes=current_nodes,
            hidden_states=hidden_states,
            visited_mask=visited_mask,
            cumulative_rewards=torch.zeros((num_graphs, num_agents), device=device),
            done_mask=torch.zeros((num_graphs, num_agents), dtype=torch.bool, device=device),
        )

        log_pf_sum = torch.zeros((num_graphs, num_agents), device=device)
        num_moves = torch.zeros((num_graphs, num_agents), dtype=torch.long, device=device)
        num_steps = torch.zeros((num_graphs, num_agents), dtype=torch.long, device=device)
        stop_reason = torch.zeros((num_graphs, num_agents), dtype=torch.long, device=device)
        rollout_valid_mask = torch.ones((num_graphs, num_agents), dtype=torch.bool, device=device)
        log_pf_steps = torch.zeros((num_graphs, num_agents, max_steps), device=device) if collect_traces else None
        log_pb_steps = torch.zeros((num_graphs, num_agents, max_steps), device=device) if collect_traces else None
        log_f_steps = torch.zeros((num_graphs, num_agents, max_steps), device=device) if collect_traces else None

        forced_edge_ids = forced_edge_ids.to(device=device, dtype=torch.long)
        path_lengths = path_lengths.to(device=device, dtype=torch.long).clamp(min=0, max=max_steps)

        node_tokens, relation_tokens, question_tokens = policy.encode_context(env_context)
        for step in range(max_steps):
            if agent_state.done_mask.all():
                break
            active_mask = (~agent_state.done_mask) & rollout_valid_mask
            active_flat = active_mask.view(-1)

            policy_out = policy.compute_action_scores(
                env_context=env_context,
                agent_state=agent_state,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
                relation_tokens=relation_tokens,
            )
            if step < stop_min_steps:
                policy_out = self._mask_stop_logits_for_min_steps(
                    policy_out=policy_out,
                    active_flat=active_flat,
                )

            out_degrees_flat = policy_out["out_degrees"].view(-1)
            edge_logits = policy_out["edge_logits"]
            edge_ids = policy_out["edge_ids"]
            stop_logits = policy_out["stop_logits"].view(-1, 1)
            total_agents = int(out_degrees_flat.numel())

            max_deg = int(out_degrees_flat.max().item()) if total_agents > 0 else 0
            neg_inf = torch.tensor(float("-inf"), device=device, dtype=edge_logits.dtype)
            edge_logits_dense = torch.full((total_agents, max_deg), neg_inf, device=device, dtype=edge_logits.dtype)
            edge_ids_dense = torch.full((total_agents, max_deg), -1, device=device, dtype=torch.long)
            if max_deg > 0:
                edge_cols = torch.arange(max_deg, device=device, dtype=torch.long)
                edge_valid = edge_cols.unsqueeze(0) < out_degrees_flat.unsqueeze(1)
                edge_logits_dense[edge_valid] = edge_logits
                edge_ids_dense[edge_valid] = edge_ids
            else:
                edge_valid = torch.empty((total_agents, 0), dtype=torch.bool, device=device)

            final_logits = torch.cat([edge_logits_dense, stop_logits.to(dtype=edge_logits_dense.dtype)], dim=1)
            if collect_traces:
                if log_f_steps is None:
                    raise RuntimeError("collect_traces=True requires log_f_steps tensor in evaluate_forced_paths.")
                log_f = torch.logsumexp(final_logits, dim=1).view(num_graphs, num_agents)
                log_f_steps[:, :, step] = torch.where(active_mask, log_f, torch.zeros_like(log_f))
            stop_action_idx = max_deg

            forced_len_flat = path_lengths.view(-1)
            forced_move_flat = active_flat & (forced_len_flat > step)
            force_stop_flat = active_flat & (~forced_move_flat)
            chosen_edge_flat = forced_edge_ids[:, :, step].reshape(-1)
            invalid_flat = torch.zeros((total_agents,), dtype=torch.bool, device=device)

            action_idx = torch.full((total_agents,), fill_value=stop_action_idx, dtype=torch.long, device=device)
            move_rows = torch.where(forced_move_flat)[0]
            if int(move_rows.numel()) > 0:
                move_edges = chosen_edge_flat.index_select(0, move_rows)
                row_edge_ids = edge_ids_dense.index_select(0, move_rows)
                row_edge_valid = edge_valid.index_select(0, move_rows)
                edge_match = (row_edge_ids == move_edges.unsqueeze(1)) & row_edge_valid
                non_negative = move_edges >= 0
                valid_match = non_negative & edge_match.any(dim=1)
                valid_rows = move_rows[valid_match]
                if int(valid_rows.numel()) > 0:
                    valid_edge_match = edge_match.index_select(0, torch.where(valid_match)[0])
                    action_idx_move = valid_edge_match.to(dtype=torch.long).argmax(dim=1)
                    action_idx.index_copy_(0, valid_rows, action_idx_move)
                invalid_rows = move_rows[~valid_match]
                if int(invalid_rows.numel()) > 0:
                    invalid_flat[invalid_rows] = True
                    rollout_valid_mask.view(-1)[invalid_rows] = False

            active_effective_flat = active_flat & (~invalid_flat)

            true_dist = torch.distributions.Categorical(logits=final_logits)
            log_prob_flat = true_dist.log_prob(action_idx)
            log_prob_flat = torch.where(active_effective_flat, log_prob_flat, torch.zeros_like(log_prob_flat))
            log_prob = log_prob_flat.view(num_graphs, num_agents)
            log_pf_sum = torch.where(active_mask, log_pf_sum + log_prob, log_pf_sum)
            active_effective = active_effective_flat.view(num_graphs, num_agents)
            num_steps = torch.where(active_effective, num_steps + 1, num_steps)

            is_stop_flat = (~active_effective_flat) | (action_idx == stop_action_idx) | invalid_flat
            is_stop = is_stop_flat.view(num_graphs, num_agents)
            move_mask = active_effective & (~is_stop)
            num_moves = torch.where(move_mask, num_moves + 1, num_moves)
            if log_pf_steps is not None:
                log_pf_steps[:, :, step] = log_prob

            current_flat = agent_state.current_nodes.view(-1)
            valid_forced_move_flat = forced_move_flat & (~invalid_flat)
            chosen_edge_ids_flat = torch.where(
                valid_forced_move_flat,
                chosen_edge_flat,
                torch.full_like(chosen_edge_flat, -1),
            )
            safe_edge_ids = chosen_edge_ids_flat.clamp(min=0)
            edge_targets = env_context.edge_index[1].index_select(0, safe_edge_ids)
            chosen_target_nodes = torch.where(valid_forced_move_flat, edge_targets, current_flat)

            if log_pb_steps is not None:
                log_pb = self._compute_log_pb(
                    env_context=env_context,
                    source_nodes=current_flat,
                    chosen_target_nodes=chosen_target_nodes,
                    active_flat=active_effective_flat,
                    is_stop_flat=is_stop_flat,
                    dtype=log_prob.dtype,
                )
                log_pb_steps[:, :, step] = log_pb.view(num_graphs, num_agents)

            chosen_edge_rel = env_context.edge_relations.index_select(0, safe_edge_ids)
            agent_state = policy.evolve_state(
                agent_state=agent_state,
                chosen_target_nodes=chosen_target_nodes,
                chosen_edge_relations=chosen_edge_rel,
                node_tokens=node_tokens,
                relation_tokens=relation_tokens,
                is_stop=is_stop_flat,
            )
            if not use_visited_mask:
                agent_state.visited_mask.zero_()

            stop_reason_flat = stop_reason.view(-1)
            natural_stop_flat = force_stop_flat & (~invalid_flat)
            stop_reason_flat = torch.where(
                natural_stop_flat,
                torch.full_like(stop_reason_flat, STOP_REASON_ACTION),
                stop_reason_flat,
            )
            stop_reason_flat = torch.where(
                invalid_flat,
                torch.full_like(stop_reason_flat, STOP_REASON_DEAD_END),
                stop_reason_flat,
            )
            stop_reason = stop_reason_flat.view(num_graphs, num_agents)

        unfinished = ~agent_state.done_mask
        if bool(unfinished.any().item()):
            stop_reason = torch.where(
                unfinished,
                torch.full_like(stop_reason, STOP_REASON_MAX_STEPS),
                stop_reason,
            )
        return RolloutResult(
            log_pf_sum=log_pf_sum,
            stop_nodes=agent_state.current_nodes,
            num_moves=num_moves,
            num_steps=num_steps,
            stop_reason=stop_reason,
            actions=None,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            log_f_steps=log_f_steps,
            valid_mask=rollout_valid_mask,
            policy_metrics=None,
        )


__all__ = ["OfflineForcedEvalEngine"]
