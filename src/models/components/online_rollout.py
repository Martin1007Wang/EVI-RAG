from __future__ import annotations

import torch
from torch import nn

from src.models.configs.search import RolloutConfig
from src.models.environment import DynamicAgentState, GraphEnvContext

from .backward_prior import StructuralBackwardPrior
from .policy import DualFlowPolicy
from .rollout_types import STOP_REASON_ACTION, STOP_REASON_DEAD_END, STOP_REASON_MAX_STEPS, RolloutResult

EncodedPolicyContext = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class OnlineRolloutEngine:
    """On-policy rollout engine with strict label-free decision semantics."""

    def __init__(
        self,
        *,
        config: RolloutConfig,
        action_sampler: nn.Module,
        backward_prior: StructuralBackwardPrior,
    ) -> None:
        self.config = config
        self.action_sampler = action_sampler
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
        start_local = OnlineRolloutEngine._expand_grouped_start_nodes(
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

    def sample_forward(
        self,
        env_context: GraphEnvContext,
        policy: DualFlowPolicy,
        *,
        deterministic: bool = False,
        temperature: float | None = None,
        encoded_context: EncodedPolicyContext | None = None,
        collect_traces: bool = True,
    ) -> RolloutResult:
        num_graphs = int(env_context.num_graphs)
        num_agents = max(int(self.config.num_rollouts), 1)
        stop_min_steps = int(self.config.stop_min_steps)
        if stop_min_steps < 0:
            raise ValueError("sampling.stop_min_steps must be >= 0.")
        if bool(self.config.train_oracle_force_stop):
            raise ValueError("sampling.train_oracle_force_stop is forbidden to prevent label leakage.")

        device = env_context.node_embeddings.device
        agent_state = self._init_agent_state(
            env_context=env_context,
            num_agents=num_agents,
            deterministic=deterministic,
        )
        agent_graph_ids = torch.arange(num_graphs, device=device, dtype=torch.long).repeat_interleave(num_agents)

        log_pf_sum = torch.zeros((num_graphs, num_agents), device=device)
        num_moves = torch.zeros((num_graphs, num_agents), dtype=torch.long, device=device)
        num_steps = torch.zeros((num_graphs, num_agents), dtype=torch.long, device=device)
        stop_reason = torch.zeros((num_graphs, num_agents), dtype=torch.long, device=device)
        rollout_valid_mask = torch.ones((num_graphs, num_agents), dtype=torch.bool, device=device)
        max_steps = int(self.config.max_steps)
        log_pf_steps = torch.zeros((num_graphs, num_agents, max_steps), device=device) if collect_traces else None
        log_pb_steps = torch.zeros((num_graphs, num_agents, max_steps), device=device) if collect_traces else None
        log_f_steps = torch.zeros((num_graphs, num_agents, max_steps), device=device) if collect_traces else None

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
            if agent_state.done_mask.all():
                break
            active_mask = ~agent_state.done_mask
            active_flat = active_mask.view(-1)

            if action_cache is None:
                policy_out = policy.compute_action_scores(
                    env_context=env_context,
                    agent_state=agent_state,
                    node_tokens=node_tokens,
                    question_tokens=question_tokens,
                    relation_tokens=relation_tokens,
                )
            else:
                policy_out = policy.compute_action_scores(
                    env_context=env_context,
                    agent_state=agent_state,
                    node_tokens=node_tokens,
                    question_tokens=question_tokens,
                    relation_tokens=relation_tokens,
                    action_cache=action_cache,
                )
            out_degrees_flat = policy_out["out_degrees"].view(-1)
            if step < stop_min_steps:
                policy_out = self._mask_stop_logits_for_min_steps(
                    policy_out=policy_out,
                    active_flat=active_flat,
                )

            action_info = self.action_sampler(
                policy_out,
                is_training=policy.training,
                deterministic=deterministic,
                sampling_mode=str(self.config.sampling_mode),
                sampling_temperature=float(self.config.sampling_temperature),
                eval_sampling_temperature=float(self.config.eval_sampling_temperature),
                eval_sample_without_replacement=bool(self.config.eval_sample_without_replacement),
                agent_graph_ids=agent_graph_ids,
                source_nodes=agent_state.current_nodes.view(-1),
                active_mask=active_flat,
                num_nodes_total=env_context.num_nodes_total,
                temperature=temperature,
            )
            if collect_traces:
                if log_f_steps is None:
                    raise RuntimeError("collect_traces=True requires log_f_steps tensor.")
                log_f = action_info["log_partition"].view(num_graphs, num_agents)
                log_f_steps[:, :, step] = torch.where(active_mask, log_f, torch.zeros_like(log_f))
            num_steps = torch.where(active_mask, num_steps + 1, num_steps)

            sampled_is_stop = action_info["is_stop"].view(-1)
            is_stop_flat = sampled_is_stop | (~active_flat)
            is_stop = is_stop_flat.view(num_graphs, num_agents)
            newly_stopped = active_flat & sampled_is_stop
            dead_end_stop = newly_stopped & (out_degrees_flat == 0)
            normal_stop = newly_stopped & (out_degrees_flat > 0)

            stop_reason_flat = stop_reason.view(-1)
            stop_reason_flat = torch.where(
                normal_stop,
                torch.full_like(stop_reason_flat, STOP_REASON_ACTION),
                stop_reason_flat,
            )
            stop_reason_flat = torch.where(
                dead_end_stop,
                torch.full_like(stop_reason_flat, STOP_REASON_DEAD_END),
                stop_reason_flat,
            )
            stop_reason = stop_reason_flat.view(num_graphs, num_agents)

            log_prob_flat = action_info["log_prob"].view(-1)
            log_prob_flat = torch.where(active_flat, log_prob_flat, torch.zeros_like(log_prob_flat))
            log_prob = log_prob_flat.view(num_graphs, num_agents)
            log_pf_sum = torch.where(active_mask, log_pf_sum + log_prob, log_pf_sum)
            move_mask = active_mask & (~is_stop)
            num_moves = torch.where(move_mask, num_moves + 1, num_moves)
            if log_pf_steps is not None:
                log_pf_steps[:, :, step] = log_prob

            chosen_target_nodes = action_info["chosen_target_nodes"].view(-1)
            current_flat = agent_state.current_nodes.view(-1)
            chosen_target_nodes = torch.where(active_flat & (~is_stop_flat), chosen_target_nodes, current_flat)
            chosen_edge_ids = action_info["chosen_edge_ids"].view(-1)
            chosen_edge_ids = torch.where(
                active_flat & (~is_stop_flat),
                chosen_edge_ids,
                torch.full_like(chosen_edge_ids, -1),
            )
            if log_pb_steps is not None:
                log_pb = self._compute_log_pb(
                    env_context=env_context,
                    source_nodes=current_flat,
                    chosen_target_nodes=chosen_target_nodes,
                    active_flat=active_flat,
                    is_stop_flat=is_stop_flat,
                    dtype=log_prob.dtype,
                )
                log_pb_steps[:, :, step] = log_pb.view(num_graphs, num_agents)
            safe_edge_ids = chosen_edge_ids.clamp(min=0)
            chosen_edge_relations = env_context.edge_relations[safe_edge_ids]
            agent_state = policy.evolve_state(
                agent_state=agent_state,
                chosen_target_nodes=chosen_target_nodes,
                chosen_edge_relations=chosen_edge_relations,
                node_tokens=node_tokens,
                relation_tokens=relation_tokens,
                is_stop=is_stop_flat,
            )

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


__all__ = ["OnlineRolloutEngine"]
