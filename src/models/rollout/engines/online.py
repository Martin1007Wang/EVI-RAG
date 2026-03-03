from __future__ import annotations

import torch
from torch import nn

from src.models.configs.search import RolloutConfig
from src.models.environment import FlowDirection, GraphEnvContext

from src.models.policy import DualFlowPolicy
from ..state_init import (
    compute_effective_max_steps,
    compute_max_allowed_moves,
    compute_min_required_moves,
    initialize_agent_state,
)
from src.utils.segment_ops import (
    compute_has_finite_edges,
    mask_stop_logits_for_min_steps,
)
from ..types import (
    STOP_REASON_ACTION,
    STOP_REASON_DEAD_END,
    STOP_REASON_MAX_STEPS_REACHED,
    RolloutResult,
)

EncodedPolicyContext = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class OnlineRolloutEngine:
    """On-policy rollout engine with strict label-free decision semantics."""

    def __init__(
        self,
        *,
        config: RolloutConfig,
        action_sampler: nn.Module,
    ) -> None:
        self.config = config
        self.action_sampler = action_sampler

    @staticmethod
    def _mask_move_logits_for_max_moves(
        *,
        policy_out: dict[str, torch.Tensor],
        active_flat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out_degrees_flat = policy_out["out_degrees"].view(-1)
        edge_logits = policy_out["edge_logits"]
        if int(edge_logits.numel()) == 0 or int(out_degrees_flat.numel()) == 0:
            return policy_out
        agent_ids = torch.arange(
            int(out_degrees_flat.numel()),
            device=out_degrees_flat.device,
            dtype=torch.long,
        )
        edge_agent_ids = agent_ids.repeat_interleave(out_degrees_flat)
        if int(edge_agent_ids.numel()) != int(edge_logits.numel()):
            raise ValueError(
                "edge_logits size mismatch with ragged out_degrees in max-moves masking: "
                f"edge_logits={int(edge_logits.numel())}, expanded_agents={int(edge_agent_ids.numel())}."
            )
        force_stop_edge_mask = active_flat.index_select(0, edge_agent_ids)
        if not bool(force_stop_edge_mask.any().item()):
            return policy_out
        masked_edge_logits = edge_logits.masked_fill(
            force_stop_edge_mask,
            torch.tensor(
                float("-inf"), device=edge_logits.device, dtype=edge_logits.dtype
            ),
        )
        patched = dict(policy_out)
        patched["edge_logits"] = masked_edge_logits
        return patched

    def _compute_log_pb(
        self,
        *,
        active_flat: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros_like(active_flat, dtype=dtype)

    def sample_forward(
        self,
        env_context: GraphEnvContext,
        policy: DualFlowPolicy,
        *,
        flow_direction: FlowDirection = "forward",
        deterministic: bool = False,
        temperature: float | None = None,
        encoded_context: EncodedPolicyContext | None = None,
        collect_traces: bool = True,
    ) -> RolloutResult:
        if flow_direction not in {"forward", "backward"}:
            raise ValueError(f"Unsupported flow_direction: {flow_direction!r}.")
        num_graphs = int(env_context.num_graphs)
        num_agents = max(int(self.config.num_rollouts), 1)
        stop_min_steps = int(self.config.stop_min_steps)
        if stop_min_steps < 0:
            raise ValueError("sampling.stop_min_steps must be >= 0.")
        if bool(self.config.train_oracle_force_stop):
            raise ValueError(
                "sampling.train_oracle_force_stop is forbidden to prevent label leakage."
            )

        device = env_context.node_embeddings.device
        agent_state = initialize_agent_state(
            env_context=env_context,
            num_agents=num_agents,
            deterministic=deterministic,
            flow_direction=flow_direction,
            backward_without_super_error=(
                "Backward rollout requires super-source layout so the policy can choose among a_local_indices "
                "from the backward super node."
            ),
        )
        min_required_moves_flat = compute_min_required_moves(
            env_context=env_context,
            start_nodes_abs=agent_state.current_nodes,
            base_stop_min_steps=stop_min_steps,
            flatten=True,
        )
        max_allowed_moves_flat = compute_max_allowed_moves(
            env_context=env_context,
            start_nodes_abs=agent_state.current_nodes,
            base_max_steps=int(self.config.max_steps),
            flatten=True,
        )
        if bool((min_required_moves_flat > max_allowed_moves_flat).any().item()):
            raise ValueError(
                "Invalid stop/move budget: stop_min_steps exceeds max_steps after virtual-start offset adjustment."
            )
        agent_graph_ids = torch.arange(
            num_graphs, device=device, dtype=torch.long
        ).repeat_interleave(num_agents)

        log_pf_sum = torch.zeros((num_graphs, num_agents), device=device)
        num_steps = torch.zeros(
            (num_graphs, num_agents), dtype=torch.long, device=device
        )
        stop_reason = torch.zeros(
            (num_graphs, num_agents), dtype=torch.long, device=device
        )
        rollout_valid_mask = torch.ones(
            (num_graphs, num_agents), dtype=torch.bool, device=device
        )
        max_steps = compute_effective_max_steps(
            env_context=env_context,
            start_nodes_abs=agent_state.current_nodes,
            base_max_steps=int(self.config.max_steps),
        )
        log_pf_steps = (
            torch.zeros((num_graphs, num_agents, max_steps), device=device)
            if collect_traces
            else None
        )
        log_pb_steps = (
            torch.zeros((num_graphs, num_agents, max_steps), device=device)
            if collect_traces
            else None
        )
        log_f_steps = (
            torch.zeros((num_graphs, num_agents, max_steps), device=device)
            if collect_traces
            else None
        )
        stop_logprob_steps = (
            torch.zeros((num_graphs, num_agents, max_steps), device=device)
            if collect_traces
            else None
        )
        state_nodes_steps = (
            torch.full(
                (num_graphs, num_agents, max_steps),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            )
            if collect_traces
            else None
        )
        continue_valid_steps = (
            torch.zeros(
                (num_graphs, num_agents, max_steps), device=device, dtype=torch.bool
            )
            if collect_traces
            else None
        )
        stop_valid_steps = (
            torch.zeros(
                (num_graphs, num_agents, max_steps), device=device, dtype=torch.bool
            )
            if collect_traces
            else None
        )

        if encoded_context is None:
            node_tokens, relation_tokens, question_tokens = policy.encode_context(
                env_context
            )
        else:
            node_tokens, relation_tokens, question_tokens = encoded_context
        build_cache_fn = getattr(policy, "build_action_cache", None)
        action_cache: dict[str, torch.Tensor] | None = None
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
            num_moves_flat = agent_state.num_moves.view(-1)
            need_more_moves = num_moves_flat < min_required_moves_flat
            stop_guard_active = active_flat & need_more_moves
            if bool(stop_guard_active.any().item()):
                policy_out = mask_stop_logits_for_min_steps(
                    policy_out=policy_out,
                    active_flat=stop_guard_active,
                )
            over_max_moves = num_moves_flat >= max_allowed_moves_flat
            max_guard_active = active_flat & over_max_moves
            if bool(max_guard_active.any().item()):
                policy_out = self._mask_move_logits_for_max_moves(
                    policy_out=policy_out,
                    active_flat=max_guard_active,
                )

            action_info = self.action_sampler(
                policy_out,
                is_training=policy.training,
                deterministic=deterministic,
                sampling_mode=str(self.config.sampling_mode),
                sampling_temperature=float(self.config.sampling_temperature),
                eval_sampling_temperature=float(self.config.eval_sampling_temperature),
                eval_sample_without_replacement=bool(
                    self.config.eval_sample_without_replacement
                ),
                agent_graph_ids=agent_graph_ids,
                source_nodes=agent_state.current_nodes.view(-1),
                active_mask=active_flat,
                num_nodes_total=env_context.num_nodes_total,
                temperature=temperature,
            )
            if collect_traces:
                if log_f_steps is None:
                    raise RuntimeError(
                        "collect_traces=True requires log_f_steps tensor."
                    )
                if (
                    stop_logprob_steps is None
                    or state_nodes_steps is None
                    or continue_valid_steps is None
                    or stop_valid_steps is None
                ):
                    raise RuntimeError(
                        "collect_traces=True requires stop/node validity trace tensors."
                    )
                state_log_flows = policy_out.get("state_log_flows")
                if state_log_flows is None:
                    raise ValueError(
                        "policy_output must provide `state_log_flows` for independent logF estimation."
                    )
                if tuple(state_log_flows.shape) != (num_graphs, num_agents):
                    raise ValueError(
                        "state_log_flows shape mismatch with rollout batch: "
                        f"state_log_flows={tuple(state_log_flows.shape)}, expected={(num_graphs, num_agents)}."
                    )
                log_f = state_log_flows.to(device=device, dtype=log_f_steps.dtype)
                log_f_steps[:, :, step] = torch.where(
                    active_mask, log_f, torch.zeros_like(log_f)
                )
                state_nodes_steps[:, :, step] = torch.where(
                    active_mask,
                    agent_state.current_nodes,
                    torch.full_like(agent_state.current_nodes, -1),
                )
                has_finite_edges = compute_has_finite_edges(
                    edge_logits=policy_out["edge_logits"],
                    out_degrees=policy_out["out_degrees"].view(-1),
                )
                continue_valid_flat = (
                    active_flat
                    & (policy_out["out_degrees"].view(-1) > 0)
                    & has_finite_edges
                )
                stop_valid_flat = active_flat & torch.isfinite(
                    policy_out["stop_logits"].view(-1)
                )
                continue_valid_steps[:, :, step] = continue_valid_flat.view(
                    num_graphs, num_agents
                )
                stop_valid_steps[:, :, step] = stop_valid_flat.view(
                    num_graphs, num_agents
                )
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
            log_prob_flat = torch.where(
                active_flat, log_prob_flat, torch.zeros_like(log_prob_flat)
            )
            log_prob = log_prob_flat.view(num_graphs, num_agents)
            log_pf_sum = torch.where(active_mask, log_pf_sum + log_prob, log_pf_sum)
            if log_pf_steps is not None:
                log_pf_steps[:, :, step] = log_prob
            if stop_logprob_steps is not None:
                stop_log_prob = action_info["stop_log_prob"].view(
                    num_graphs, num_agents
                )
                stop_logprob_steps[:, :, step] = torch.where(
                    active_mask, stop_log_prob, torch.zeros_like(stop_log_prob)
                )

            chosen_target_nodes = action_info["chosen_target_nodes"].view(-1)
            current_flat = agent_state.current_nodes.view(-1)
            chosen_target_nodes = torch.where(
                active_flat & (~is_stop_flat), chosen_target_nodes, current_flat
            )
            chosen_edge_ids = action_info["chosen_edge_ids"].view(-1)
            chosen_edge_ids = torch.where(
                active_flat & (~is_stop_flat),
                chosen_edge_ids,
                torch.full_like(chosen_edge_ids, -1),
            )
            if log_pb_steps is not None:
                log_pb = self._compute_log_pb(
                    active_flat=active_flat,
                    dtype=log_prob.dtype,
                )
                log_pb_steps[:, :, step] = log_pb.view(num_graphs, num_agents)
            safe_edge_ids = chosen_edge_ids.clamp(min=0)
            if int(env_context.edge_relations.numel()) == 0:
                chosen_edge_relations = torch.zeros_like(safe_edge_ids)
            else:
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
                torch.full_like(stop_reason, STOP_REASON_MAX_STEPS_REACHED),
                stop_reason,
            )
        return RolloutResult(
            log_pf_sum=log_pf_sum,
            stop_nodes=agent_state.current_nodes,
            num_moves=agent_state.num_moves,
            num_steps=num_steps,
            stop_reason=stop_reason,
            actions=None,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            log_f_steps=log_f_steps,
            stop_logprob_steps=stop_logprob_steps,
            state_nodes_steps=state_nodes_steps,
            continue_valid_steps=continue_valid_steps,
            stop_valid_steps=stop_valid_steps,
            valid_mask=rollout_valid_mask,
            policy_metrics=None,
        )


__all__ = ["OnlineRolloutEngine"]
