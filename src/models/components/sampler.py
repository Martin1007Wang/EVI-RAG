# src/models/components/sampler.py
"""
[系统实体] 并行 Rollout 采样门面
职责：
1. 统一暴露 sample_forward / evaluate_forced_paths / beam_search_forward 三类接口。
2. 将具体执行委托给独立引擎，避免单文件过度膨胀。
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.configs.search import RolloutConfig
from src.models.environment.contracts import GraphEnvContext

from .backward_prior import StructuralBackwardPrior
from .beam_decoder import BeamDecoderEngine
from .offline_forced_eval import OfflineForcedEvalEngine
from .online_rollout import OnlineRolloutEngine
from .policy import DualFlowPolicy
from .rollout_types import RolloutResult

EncodedPolicyContext = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ActionSampler(nn.Module):
    """将 ragged 边 logits 打包为逐 agent 的分类分布并采样动作。"""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _sanitize_logits(
        *,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D [num_rows, num_actions], got {tuple(logits.shape)}")
        if int(logits.size(1)) == 0:
            raise ValueError("logits must contain at least one action column.")
        has_finite = torch.isfinite(logits).any(dim=1)
        has_nan = torch.isnan(logits).any(dim=1)
        has_pos_inf = torch.isposinf(logits).any(dim=1)
        invalid_rows = has_nan | has_pos_inf | (~has_finite)
        if not bool(invalid_rows.any().item()):
            return logits
        safe_logits = logits.clone()
        neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
        stop_index = int(logits.size(1)) - 1
        safe_logits[invalid_rows] = neg_inf
        safe_logits[invalid_rows, stop_index] = 0.0
        return safe_logits

    def forward(
        self,
        policy_output: dict[str, torch.Tensor],
        *,
        is_training: bool,
        deterministic: bool,
        sampling_mode: str,
        sampling_temperature: float,
        eval_sampling_temperature: float,
        eval_sample_without_replacement: bool,
        agent_graph_ids: torch.Tensor | None = None,
        source_nodes: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        num_nodes_total: int = 0,
        temperature: float | None = None,
    ) -> dict[str, torch.Tensor]:
        edge_logits = policy_output["edge_logits"]
        out_degrees = policy_output["out_degrees"].view(-1)
        stop_logits = policy_output["stop_logits"].view(-1, 1)
        edge_ids = policy_output["edge_ids"]
        target_nodes = policy_output["target_nodes"]

        total_agents = out_degrees.size(0)
        device = edge_logits.device
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=edge_logits.dtype)
        max_deg = int(out_degrees.max().item()) if total_agents > 0 else 0

        padded_logits = torch.full((total_agents, max_deg), neg_inf, device=device, dtype=edge_logits.dtype)
        if max_deg > 0:
            mask = torch.arange(max_deg, device=device).unsqueeze(0) < out_degrees.unsqueeze(1)
            padded_logits[mask] = edge_logits

        final_logits = torch.cat([padded_logits, stop_logits], dim=-1)
        final_logits = self._sanitize_logits(logits=final_logits)
        log_partition = torch.logsumexp(final_logits, dim=-1)

        if temperature is not None:
            actual_temp = temperature
        else:
            actual_temp = eval_sampling_temperature if deterministic else sampling_temperature

        actual_temp = float(max(actual_temp, 1e-4))
        if is_training and abs(actual_temp - 1.0) > 1e-6:
            raise ValueError(
                "On-policy SubTB requires sampling_temperature == 1.0 during training "
                f"(got {actual_temp})."
            )

        mode = str(sampling_mode).strip().lower()
        if actual_temp <= 1e-3 or mode == "greedy":
            action_idx = final_logits.argmax(dim=-1)
        else:
            scaled_logits = final_logits / actual_temp
            dist = torch.distributions.Categorical(logits=scaled_logits, validate_args=False)
            if not is_training and eval_sample_without_replacement:
                action_idx = self._sample_eval_actions_without_replacement(
                    final_logits=scaled_logits,
                    active_mask=active_mask,
                    agent_graph_ids=agent_graph_ids,
                    source_nodes=source_nodes,
                    num_nodes_total=num_nodes_total,
                    enable_grouped_without_replacement=True,
                )
            else:
                action_idx = dist.sample()

        if edge_ids.numel() == 0:
            return {
                "is_stop": torch.ones_like(action_idx, dtype=torch.bool),
                "chosen_edge_ids": torch.full_like(action_idx, -1),
                "chosen_target_nodes": torch.zeros_like(action_idx),
                "log_prob": torch.zeros_like(action_idx, dtype=edge_logits.dtype),
                "log_partition": log_partition,
            }

        is_stop = action_idx == max_deg
        base_offsets = out_degrees.cumsum(0) - out_degrees
        flat_chosen_idx = base_offsets + action_idx
        safe_flat_idx = flat_chosen_idx.clamp(max=max(0, edge_ids.numel() - 1))

        chosen_edge_ids = edge_ids[safe_flat_idx]
        chosen_target_nodes = target_nodes[safe_flat_idx]
        chosen_edge_ids = torch.where(is_stop, torch.full_like(chosen_edge_ids, -1), chosen_edge_ids)
        chosen_target_nodes = torch.where(is_stop, torch.zeros_like(chosen_target_nodes), chosen_target_nodes)

        true_dist = torch.distributions.Categorical(logits=final_logits, validate_args=False)
        return {
            "is_stop": is_stop,
            "chosen_edge_ids": chosen_edge_ids,
            "chosen_target_nodes": chosen_target_nodes,
            "log_prob": true_dist.log_prob(action_idx),
            "log_partition": log_partition,
        }

    def _sample_eval_actions_without_replacement(
        self,
        *,
        final_logits: torch.Tensor,
        active_mask: torch.Tensor | None,
        agent_graph_ids: torch.Tensor | None,
        source_nodes: torch.Tensor | None,
        num_nodes_total: int,
        enable_grouped_without_replacement: bool,
    ) -> torch.Tensor:
        num_rows, num_actions = final_logits.shape
        stop_index = num_actions - 1
        action_idx = torch.full(
            (num_rows,),
            fill_value=stop_index,
            device=final_logits.device,
            dtype=torch.long,
        )
        if num_rows == 0:
            return action_idx
        if active_mask is None:
            active_mask = torch.ones((num_rows,), dtype=torch.bool, device=final_logits.device)
        else:
            active_mask = active_mask.to(device=final_logits.device, dtype=torch.bool).view(-1)
        active_rows = torch.where(active_mask)[0]
        if int(active_rows.numel()) == 0:
            return action_idx

        if not enable_grouped_without_replacement or agent_graph_ids is None or source_nodes is None:
            sampled = torch.distributions.Categorical(
                logits=final_logits.index_select(0, active_rows),
                validate_args=False,
            ).sample()
            action_idx[active_rows] = sampled
            return action_idx

        keys = agent_graph_ids.to(device=final_logits.device, dtype=torch.long) * max(int(num_nodes_total), 1) + source_nodes.to(
            device=final_logits.device,
            dtype=torch.long,
        ).clamp(min=0)
        active_keys = keys.index_select(0, active_rows)
        unique_keys = torch.unique(active_keys, sorted=False)
        for key in unique_keys:
            rows = active_rows[active_keys == key]
            if int(rows.numel()) == 0:
                continue
            group_logits = final_logits.index_select(0, rows).mean(dim=0)
            valid_actions = torch.where(torch.isfinite(group_logits))[0]
            if int(valid_actions.numel()) == 0:
                continue
            valid_logits = group_logits.index_select(0, valid_actions)
            sampled_count = min(int(rows.numel()), int(valid_actions.numel()))
            valid_probs = torch.softmax(valid_logits, dim=0)
            sampled_local = torch.multinomial(valid_probs, sampled_count, replacement=False)
            sampled_actions = valid_actions.index_select(0, sampled_local)
            action_idx[rows[:sampled_count]] = sampled_actions
            if sampled_count < int(rows.numel()):
                fallback = valid_actions[valid_logits.argmax()]
                action_idx[rows[sampled_count:]] = fallback
        return action_idx


class RolloutSampler:
    """Facade over three rollout engines: online, offline-forced, and beam decode."""

    def __init__(self, config: RolloutConfig) -> None:
        self.config = config
        self.action_sampler = ActionSampler()
        backward_prior_mode = str(config.backward_prior_mode)
        self.backward_prior = StructuralBackwardPrior(mode=backward_prior_mode)
        self.online_engine = OnlineRolloutEngine(
            config=config,
            action_sampler=self.action_sampler,
            backward_prior=self.backward_prior,
        )
        self.offline_engine = OfflineForcedEvalEngine(
            config=config,
            backward_prior=self.backward_prior,
        )
        self.beam_engine = BeamDecoderEngine(config=config)

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
        return self.online_engine.sample_forward(
            env_context,
            policy,
            deterministic=deterministic,
            temperature=temperature,
            encoded_context=encoded_context,
            collect_traces=collect_traces,
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
        encoded_context: EncodedPolicyContext | None = None,
    ) -> RolloutResult:
        return self.offline_engine.evaluate_forced_paths(
            env_context,
            policy,
            start_local_indices=start_local_indices,
            forced_edge_ids=forced_edge_ids,
            path_lengths=path_lengths,
            collect_traces=collect_traces,
            use_visited_mask=use_visited_mask,
            encoded_context=encoded_context,
        )

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
        return self.beam_engine.beam_search_forward(
            env_context,
            policy,
            beam_size=beam_size,
            max_steps=max_steps,
            require_done=require_done,
            diverse_penalty=diverse_penalty,
            encoded_context=encoded_context,
        )


__all__ = ["ActionSampler", "RolloutResult", "RolloutSampler"]
