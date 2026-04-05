from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch_scatter import scatter_max, scatter_sum

from src.data.schema import RetrievalBatch
from .policy import Policy


@dataclass(frozen=True)
class RolloutBatch:
    """GFlowNet 采样轨迹张量组 (B: num_graphs, T: max_steps)"""

    action_mask: torch.Tensor
    termination_action_steps: torch.Tensor
    state_log_flows: torch.Tensor
    log_pf_actions: torch.Tensor
    log_pb_actions: torch.Tensor
    log_reward_actions: torch.Tensor
    terminal_active_nodes: Optional[torch.Tensor] = None
    terminal_active_edges: Optional[torch.Tensor] = None
    terminal_sinks: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            **{
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )


class RolloutEngine:
    """增量式子图 MDP 轨迹引擎"""

    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    @staticmethod
    def _segmented_gumbel_sample(
        logits: torch.Tensor, batch_idx: torch.Tensor, num_segments: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """利用 Gumbel-Max trick 在变长分段中进行高效并行采样"""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-10)))
        noisy_logits = logits + gumbel_noise

        _, sampled_idx = scatter_max(
            noisy_logits, batch_idx, dim=0, dim_size=num_segments
        )

        max_logits, _ = scatter_max(logits, batch_idx, dim=0, dim_size=num_segments)
        exp_logits = torch.exp(logits - max_logits[batch_idx])
        sum_exp = scatter_sum(exp_logits, batch_idx, dim=0, dim_size=num_segments)
        log_sum_exp = max_logits + torch.log(sum_exp.clamp(min=1e-10))

        log_prob = logits[sampled_idx] - log_sum_exp
        return sampled_idx, log_prob

    def run_exploration(
        self,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
    ) -> RolloutBatch:
        """
        单次探索：每个图采样一次
        """
        return self._run_single_exploration(policy, base_graph, reward_model)

    def run_multiple_exploration(
        self,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        num_rollouts: int = 1,
        temperature: float = 1.0,
    ) -> list[RolloutBatch]:
        """
        多次探索：每个图采样多次

        Args:
            num_rollouts: 每个图的采样次数
            temperature: 采样温度（目前未使用，保留接口）

        Returns:
            采样结果列表，每个元素是一个 RolloutBatch
        """
        return [
            self._run_single_exploration(policy, base_graph, reward_model)
            for _ in range(num_rollouts)
        ]

    def _run_single_exploration(
        self,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
    ) -> RolloutBatch:
        B = int(base_graph.ptr.numel()) - 1
        device = base_graph.node_tokens.device

        active_nodes = base_graph.is_anchor_mask.clone()
        active_edges = torch.zeros(
            base_graph.edge_index.size(1), dtype=torch.bool, device=device
        )

        is_terminated = torch.zeros(B, dtype=torch.bool, device=device)
        termination_steps = torch.full(
            (B,), self.max_steps, dtype=torch.long, device=device
        )

        records_mask, records_log_flow = [], []
        records_log_pf, records_log_pb, records_log_reward = [], [], []

        src, dst = base_graph.edge_index[0], base_graph.edge_index[1]
        edge_batch_idx = base_graph.edge_batch
        sampled_sinks = torch.full((B,), -1, dtype=torch.long, device=device)

        for t in range(self.max_steps):
            if is_terminated.all():
                break

            log_flows, action_logits = policy(base_graph, active_nodes, active_edges)

            # --- MDP 物理掩码约束 ---
            valid_edges_mask = active_nodes[src] & ~active_edges
            valid_sinks_mask = active_nodes & ~base_graph.is_anchor_mask

            has_valid_edges = (
                scatter_max(valid_edges_mask.int(), edge_batch_idx, dim=0, dim_size=B)[
                    0
                ]
                > 0
            )
            has_valid_sinks = (
                scatter_max(
                    valid_sinks_mask.int(), base_graph.batch, dim=0, dim_size=B
                )[0]
                > 0
            )

            # 边界安全防护：如果图陷入死胡同 (无路可走且无法沉汇)，强制终止
            dead_end = ~has_valid_edges & ~has_valid_sinks
            if dead_end.any():
                is_terminated |= dead_end
                termination_steps[dead_end & (termination_steps == self.max_steps)] = t
                if is_terminated.all():
                    break

            # --- 宏观决策：Expand vs Sink ---
            type_logits = action_logits["type_logits"]
            type_logits[~has_valid_edges | is_terminated, 0] = -1e9
            type_logits[~has_valid_sinks | is_terminated, 1] = -1e9

            type_dist = torch.distributions.Categorical(logits=type_logits)
            action_type = type_dist.sample()
            type_log_prob = type_dist.log_prob(action_type)

            step_log_pf = torch.zeros(B, device=device)
            step_log_pb = torch.zeros(B, device=device)
            step_log_reward = torch.zeros(B, device=device)
            step_mask = ~is_terminated.clone()

            # --- 拓扑扩张 (Expand: action_type == 0) ---
            expand_mask = (action_type == 0) & step_mask
            if expand_mask.any():
                joint_edge_logits = (
                    action_logits["expand_u_logits"][src]
                    + action_logits["expand_edge_rel_logits"]
                    + action_logits["expand_v_logits"][dst]
                )

                valid_edge_candidates = valid_edges_mask & expand_mask[edge_batch_idx]
                joint_edge_logits[~valid_edge_candidates] = -1e9

                sampled_edges, edge_log_prob = self._segmented_gumbel_sample(
                    logits=joint_edge_logits, batch_idx=edge_batch_idx, num_segments=B
                )

                active_edges[sampled_edges[expand_mask]] = True
                active_nodes[dst[sampled_edges[expand_mask]]] = True

                step_log_pf[expand_mask] = (
                    type_log_prob[expand_mask] + edge_log_prob[expand_mask]
                )
                current_e_counts = scatter_sum(
                    active_edges.int(), edge_batch_idx, dim=0, dim_size=B
                )
                step_log_pb[expand_mask] = -torch.log(
                    current_e_counts[expand_mask].float().clamp(min=1.0)
                )

            # --- 终止汇聚 (Sink: action_type == 1) ---
            sink_mask = (action_type == 1) & step_mask
            if sink_mask.any():
                sink_y_logits = action_logits["sink_logits"]
                valid_sink_candidates = valid_sinks_mask & sink_mask[base_graph.batch]
                sink_y_logits[~valid_sink_candidates] = -1e9

                sampled_sink_candidates, sink_log_prob = self._segmented_gumbel_sample(
                    logits=sink_y_logits, batch_idx=base_graph.batch, num_segments=B
                )
                sampled_sinks[sink_mask] = sampled_sink_candidates[sink_mask]

                is_terminated[sink_mask] = True
                termination_steps[sink_mask] = t

                step_log_pf[sink_mask] = (
                    type_log_prob[sink_mask] + sink_log_prob[sink_mask]
                )
                step_log_pb[sink_mask] = 0.0

                reward_tensor = reward_model(
                    base_graph=base_graph,
                    sampled_sinks=sampled_sinks,
                    active_nodes=active_nodes,
                    active_edges=active_edges,
                )
                step_log_reward[sink_mask] = reward_tensor[sink_mask]

            # --- 轨迹记录 ---
            records_mask.append(step_mask)
            records_log_flow.append(log_flows)
            records_log_pf.append(step_log_pf)
            records_log_pb.append(step_log_pb)
            records_log_reward.append(step_log_reward)

        return RolloutBatch(
            action_mask=torch.stack(records_mask, dim=1),
            termination_action_steps=termination_steps,
            state_log_flows=torch.stack(records_log_flow, dim=1),
            log_pf_actions=torch.stack(records_log_pf, dim=1),
            log_pb_actions=torch.stack(records_log_pb, dim=1),
            log_reward_actions=torch.stack(records_log_reward, dim=1),
            terminal_active_nodes=active_nodes.clone(),
            terminal_active_edges=active_edges.clone(),
            terminal_sinks=sampled_sinks.clone(),
        )


__all__ = ["RolloutBatch", "RolloutEngine"]
