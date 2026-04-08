from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch_scatter import scatter_max, scatter_sum

from src.data.schema import RetrievalBatch
from src.utils.graph_utils import compute_valid_backward_removals
from src.utils.reward_utils import build_anchor_induced_edge_mask
from .policy import Policy
from .subgraph_state import SubgraphState


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _safe_masked_logit_value(logits: torch.Tensor) -> float:
    """返回当前 dtype 下最小的有限浮点数，用于 logit masking。"""
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating-point tensors.")
    return torch.finfo(logits.dtype).min


def _scatter_log_softmax(
    logits: torch.Tensor,
    batch_idx: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    数值稳定的分段 log-softmax。

    修复点：
    - 使用 scatter_max 做 max-shift，避免 exp 溢出。
    - sum_exp 用 clamp(min=eps) 而非固定 1e-10，
      eps 随 dtype 自适应，fp16 下不会直接下溢为 0。
    - 返回每个元素的 log P，而非仅采样位置的值。
    """
    eps = torch.finfo(logits.dtype).tiny  # fp16: ~6e-5，fp32: ~1.2e-38

    max_logits, _ = scatter_max(
        logits.detach(), batch_idx, dim=0, dim_size=num_segments
    )
    # out-of-place：不修改原始 logits 计算图
    shifted = logits - max_logits[batch_idx]
    exp_shifted = torch.exp(shifted)
    sum_exp = scatter_sum(exp_shifted, batch_idx, dim=0, dim_size=num_segments)
    log_z = max_logits + torch.log(sum_exp.clamp(min=eps))

    return logits - log_z[batch_idx]  # (E,) 每条边的 log P


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutBatch:
    """GFlowNet 采样轨迹张量组 (B: num_graphs)。"""

    root_log_z: torch.Tensor
    termination_action_steps: torch.Tensor
    trajectory_log_pf: torch.Tensor
    trajectory_log_pb: torch.Tensor
    terminal_log_rewards: torch.Tensor
    root_active_edges: Optional[torch.Tensor] = None
    terminal_active_nodes: Optional[torch.Tensor] = None
    terminal_active_edges: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            **{
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self.__dict__.items()
            }
        )


@dataclass
class RolloutState:
    """Mutable rollout state advanced step-by-step by the environment."""

    root_active_edges: torch.Tensor
    active_nodes: torch.Tensor
    active_edges: torch.Tensor

    @classmethod
    def initialize(cls, base_graph: RetrievalBatch) -> "RolloutState":
        root_active_nodes = base_graph.is_anchor_mask.clone()
        root_active_edges = build_anchor_induced_edge_mask(
            base_graph.edge_index, root_active_nodes
        )
        return cls(
            root_active_edges=root_active_edges,
            active_nodes=root_active_nodes.clone(),
            active_edges=root_active_edges.clone(),
        )

    def snapshot(self) -> SubgraphState:
        return SubgraphState.from_tensors(self.active_nodes, self.active_edges)

    def apply_expansion(
        self,
        *,
        chosen_edges: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> None:
        if chosen_edges.numel() == 0:
            return

        chosen_src = src[chosen_edges]
        chosen_dst = dst[chosen_edges]
        chosen_src_active = self.active_nodes[chosen_src]
        chosen_dst_active = self.active_nodes[chosen_dst]

        activate_dst = chosen_src_active & ~chosen_dst_active
        activate_src = ~chosen_src_active & chosen_dst_active

        self.active_edges[chosen_edges] = True
        if activate_dst.any():
            self.active_nodes[chosen_dst[activate_dst]] = True
        if activate_src.any():
            self.active_nodes[chosen_src[activate_src]] = True


# ---------------------------------------------------------------------------
# 采样引擎
# ---------------------------------------------------------------------------


class RolloutEngine:
    """增量式子图 MDP 轨迹引擎。"""

    def __init__(self, max_steps: int) -> None:
        if max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps}.")
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # 核心采样：分段 Gumbel-max + log P_F（T=1）
    # ------------------------------------------------------------------

    @staticmethod
    def _segmented_gumbel_sample(
        logits: torch.Tensor,
        batch_idx: torch.Tensor,
        num_segments: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        分段 Gumbel-Max 采样，返回 (sampled_local_idx, log_prob_under_behavior_policy)。

        当前版本严格 on-policy：采样分布与记录到 TB 的 log P_F 使用同一组
        temperature-scaled logits。
        """
        eps = torch.finfo(logits.dtype).tiny  # 修复：随 dtype 自适应
        behavior_logits = logits / temperature

        # ── 行为策略：带温度的 Gumbel-max（detach，不参与反向传播）──
        sample_logits = behavior_logits.detach()
        u = torch.rand_like(sample_logits).clamp(min=eps)  # 修复：eps 随 dtype
        gumbel_noise = -torch.log(-torch.log(u))
        noisy_logits = sample_logits + gumbel_noise
        _, sampled_idx = scatter_max(
            noisy_logits, batch_idx, dim=0, dim_size=num_segments
        )

        # ── 目标策略：与采样一致的 on-policy log P（挂梯度）──
        log_probs = _scatter_log_softmax(behavior_logits, batch_idx, num_segments)
        log_prob_sampled = log_probs[sampled_idx]  # (num_segments,)

        return sampled_idx, log_prob_sampled

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def run_exploration(
        self,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        num_rollouts: int = 1,
        temperature: float = 1.0,
        collect_terminal_state: bool = True,
        terminal_state_device: torch.device | str | None = None,
    ) -> list[RolloutBatch]:
        """统一探索接口：始终返回 rollout 列表。"""
        if num_rollouts < 1:
            raise ValueError(f"num_rollouts must be >= 1, got {num_rollouts}.")
        return [
            self._run_exploration_once(
                policy=policy,
                base_graph=base_graph,
                reward_model=reward_model,
                temperature=temperature,
                collect_terminal_state=collect_terminal_state,
                terminal_state_device=terminal_state_device,
            )
            for _ in range(num_rollouts)
        ]

    @staticmethod
    def _maybe_store_tensor(
        tensor: torch.Tensor,
        *,
        collect: bool,
        device: torch.device | str | None,
    ) -> torch.Tensor | None:
        if not collect:
            return None
        snapshot = tensor.detach().clone()
        if device is not None:
            snapshot = snapshot.to(device)
        return snapshot

    # ------------------------------------------------------------------
    # 单次轨迹展开
    # ------------------------------------------------------------------

    def _run_exploration_once(
        self,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        temperature: float = 1.0,
        collect_terminal_state: bool = True,
        terminal_state_device: torch.device | str | None = None,
    ) -> RolloutBatch:
        B = int(base_graph.ptr.numel()) - 1
        device = base_graph.node_tokens.device
        src, dst = base_graph.edge_index[0], base_graph.edge_index[1]
        edge_batch_idx = base_graph.edge_batch

        # ── 初始子图：锚点诱导子图 ──
        rollout_state = RolloutState.initialize(base_graph)

        is_terminated = torch.zeros(B, dtype=torch.bool, device=device)
        termination_steps = torch.zeros(B, dtype=torch.long, device=device)
        terminal_log_rewards = torch.zeros(B, device=device)
        trajectory_log_pf = torch.zeros(B, device=device)
        trajectory_log_pb = torch.zeros(B, device=device)

        # ── Step 0：共享前向（计算 root log Z）──
        root_step_output = policy(base_graph, rollout_state.snapshot())
        root_log_z = policy.root_log_z(
            question_h=root_step_output.question_h,
            root_subgraph_h=root_step_output.subgraph_h,
        )

        # type_mask_tensor 在循环外预分配，避免每步重复创建小 tensor
        # shape (1, 2)，broadcastable 到 (B, 2)
        expand_col_mask = torch.tensor([[True, False]], device=device)  # mask Expand
        stop_col_mask = torch.tensor([[False, True]], device=device)  # mask Stop

        for t in range(self.max_steps + 1):
            active_graphs = ~is_terminated
            if not active_graphs.any():
                break

            step_output = (
                root_step_output
                if t == 0
                else policy(base_graph, rollout_state.snapshot())
            )
            action_logits = step_output.action_logits

            # ── MDP 物理约束：有效扩展边 ──
            src_active = rollout_state.active_nodes[src]
            dst_active = rollout_state.active_nodes[dst]
            valid_edges_mask = (src_active | dst_active) & ~rollout_state.active_edges

            has_valid_edges = scatter_sum(
                valid_edges_mask.int(), edge_batch_idx, dim=0, dim_size=B
            ).bool()

            step_mask = active_graphs

            # ── 宏观决策：Expand(0) vs Stop(1) ──
            raw_type_logits = action_logits["type_logits"]  # (B, 2)
            expand_forbidden = ~has_valid_edges | ~step_mask | (t >= self.max_steps)
            stop_forbidden = ~step_mask

            type_logits = raw_type_logits.masked_fill(
                expand_forbidden.unsqueeze(1) & expand_col_mask,
                _safe_masked_logit_value(raw_type_logits),
            )
            type_logits = type_logits.masked_fill(
                stop_forbidden.unsqueeze(1) & stop_col_mask,
                _safe_masked_logit_value(raw_type_logits),
            )
            behavior_type_logits = type_logits / temperature

            # 行为策略采样（带温度），并用同一分布记录 log P_F
            action_type = torch.distributions.Categorical(
                logits=behavior_type_logits.detach()
            ).sample()

            type_log_prob = torch.distributions.Categorical(
                logits=behavior_type_logits
            ).log_prob(action_type)

            # 修复：改为直接索引赋值，语义清晰且避免 masked_scatter 的
            # 隐式展平顺序依赖（形状不匹配时 masked_scatter 静默错位，
            # 而索引赋值会直接报错）
            step_log_pf = torch.zeros(B, device=device)
            step_log_pb = torch.zeros(B, device=device)

            # ── Expand 分支 ──
            expand_mask = (action_type == 0) & step_mask
            if expand_mask.any():
                raw_edge_logits = action_logits["expand_edge_logits"]  # (E,)

                valid_edge_candidates = valid_edges_mask & expand_mask[edge_batch_idx]

                candidate_counts = scatter_sum(
                    valid_edge_candidates.int(), edge_batch_idx, dim=0, dim_size=B
                )

                # 防御性检查：Expand 被采样但无候选边（理论上不应出现）
                invalid_expand = expand_mask & candidate_counts.eq(0)
                if invalid_expand.any():
                    bad = torch.nonzero(invalid_expand, as_tuple=False).view(-1)
                    raise RuntimeError(
                        "Expand sampled for graphs without valid candidate edges: "
                        f"{bad.tolist()}."
                    )

                expand_graph_ids = torch.nonzero(expand_mask, as_tuple=False).view(-1)
                candidate_edge_ids = torch.nonzero(
                    valid_edge_candidates, as_tuple=False
                ).view(-1)

                # 修复：expand_graph_remap 改用 scatter 方式构造，
                # 避免 torch.full(B) + arange 在 B 较大时的冗余内存分配
                expand_graph_remap = torch.empty(B, dtype=torch.long, device=device)
                expand_graph_remap[expand_graph_ids] = torch.arange(
                    expand_graph_ids.numel(), dtype=torch.long, device=device
                )
                candidate_batch_idx = expand_graph_remap[
                    edge_batch_idx[candidate_edge_ids]
                ]

                sampled_local_edges, edge_log_prob = self._segmented_gumbel_sample(
                    logits=raw_edge_logits[candidate_edge_ids],
                    batch_idx=candidate_batch_idx,
                    num_segments=expand_graph_ids.numel(),
                    temperature=temperature,
                )
                chosen_edges = candidate_edge_ids[sampled_local_edges]

                # 状态转移
                rollout_state.apply_expansion(
                    chosen_edges=chosen_edges, src=src, dst=dst
                )

                # log P_F = log P(Expand) + log P(chosen edge)
                # 修复：直接索引赋值，替代 masked_scatter
                step_log_pf[expand_mask] = type_log_prob[expand_mask] + edge_log_prob

                # log P_B = −log |可合法反向移除的边数|
                _, removable_counts = compute_valid_backward_removals(
                    active_nodes=rollout_state.active_nodes,
                    active_edges=rollout_state.active_edges,
                    root_active_edges=rollout_state.root_active_edges,
                    edge_index=base_graph.edge_index,
                    is_anchor_mask=base_graph.is_anchor_mask,
                    node_batch=base_graph.batch,
                    edge_batch=edge_batch_idx,
                    num_graphs=B,
                )

                # 修复：用 assert 代替 clamp 静默掩盖 removable_counts=0 的 bug
                rc_expand = removable_counts[expand_mask]
                assert (rc_expand >= 1).all(), (
                    "removable_counts must be >= 1 after expansion; "
                    "got zeros — check compute_valid_backward_removals."
                )
                step_log_pb[expand_mask] = -torch.log(rc_expand.float())

            # ── Stop 分支 ──
            stop_mask = (action_type == 1) & step_mask
            if stop_mask.any():
                is_terminated[stop_mask] = True
                termination_steps[stop_mask] = t + 1

                # log P_F(Stop) = log P(type=1)，log P_B(Stop) = 0（约定）
                step_log_pf[stop_mask] = type_log_prob[stop_mask]
                # step_log_pb[stop_mask] 已初始化为 0，无需赋值

                reward_tensor = reward_model(
                    base_graph=base_graph,
                    active_nodes=rollout_state.active_nodes,
                    active_edges=rollout_state.active_edges,
                    root_active_edges=rollout_state.root_active_edges,
                )
                terminal_log_rewards[stop_mask] = reward_tensor[stop_mask]

            trajectory_log_pf = trajectory_log_pf + step_log_pf
            trajectory_log_pb = trajectory_log_pb + step_log_pb

        # ── 防御性检查：强制 Stop 后不应再有未终止图 ──
        unfinished = ~is_terminated
        if unfinished.any():
            raise RuntimeError(
                "Rollout ended with unfinished graphs after the forced-Stop horizon. "
                "Check Stop action masking in RolloutEngine."
            )

        return RolloutBatch(
            root_log_z=root_log_z,
            termination_action_steps=termination_steps,
            trajectory_log_pf=trajectory_log_pf,
            trajectory_log_pb=trajectory_log_pb,
            terminal_log_rewards=terminal_log_rewards,
            root_active_edges=self._maybe_store_tensor(
                rollout_state.root_active_edges,
                collect=collect_terminal_state,
                device=terminal_state_device,
            ),
            terminal_active_nodes=self._maybe_store_tensor(
                rollout_state.active_nodes,
                collect=collect_terminal_state,
                device=terminal_state_device,
            ),
            terminal_active_edges=self._maybe_store_tensor(
                rollout_state.active_edges,
                collect=collect_terminal_state,
                device=terminal_state_device,
            ),
        )


__all__ = ["RolloutBatch", "RolloutEngine", "RolloutState"]
