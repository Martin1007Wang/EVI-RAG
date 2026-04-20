from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch

from src.data.schema import RetrievalBatch
from src.utils.reward_utils import build_anchor_induced_edge_mask


class Phase(Enum):
    ACTIVE   = auto()  # ∘ — 构造态，可继续扩张
    TERMINAL = auto()  # ★ — 终止态，轨迹已结束


@dataclass
class State:
    """MDP 状态 s_t = ⟨G_t, φ⟩ 的完整表示。

    与数学定义的对应关系
    ─────────────────────
    active_nodes / active_edges  ↔  G_t 的点边集
    root_active_edges            ↔  G_0 的锚点诱导边（不变量，全程不改变）
    phase                        ↔  φ ∈ {ACTIVE, TERMINAL}

    关于 root_active_edges
    ──────────────────────
    root_active_edges 记录 s_0 时锚点诱导子图的初始边集合，是整条轨迹的
    不变量。它之所以挂在 State 上（而非由调用方按需重算），是因为：

    1. log P_B 的计算需要区分"s_0 就有的边"和"expand 进来的边"——
       只有后者才是合法的 backward remove 动作。
    2. edge_state_ids 的构造需要 root_active_edges 作为 mask。
    3. 每步从 base_graph.edge_index + is_anchor_mask 重算代价等价，
       但存储一次语义更清晰，也避免重复传递 base_graph。

    如果将来 RolloutEngine 统一管理 base_graph 上下文，
    root_active_edges 可以从 State 移出。

    可变性约定
    ──────────
    State 是可变对象（mutable dataclass）。

    - RolloutEngine 直接写 rollout_step（state.rollout_step = t），
      这是有意设计：rollout 步数是 engine 的上下文，不属于 MDP 语义。
    - max_steps 不属于 State。horizon 判断由 RolloutEngine 负责，
      State 只描述"当前世界是什么"，不决定"规则允许什么"。
    - active_nodes / active_edges 只通过 apply_expansion() / apply_stop()
      修改，外部不应直接写这两个字段。
    - as_policy_input() 返回 active_nodes / active_edges 的 detached clone，
      与内部张量不共享版本计数器。
    - root_active_edges 是不变量，as_policy_input() 传引用，不 clone。
    """

    # ── 不变量 ──────────────────────────────────────────────────────
    root_active_edges: torch.Tensor   # [E] bool，G_0 的锚点诱导边

    # ── 可变量：当前子图的点边集 ─────────────────────────────────────
    active_nodes: torch.Tensor        # [N] bool
    active_edges: torch.Tensor        # [E] bool

    # ── 相位 ─────────────────────────────────────────────────────────
    phase: Phase = field(default=Phase.ACTIVE)

    # ── rollout 上下文（由 RolloutEngine 管理，不参与状态转移语义）──
    rollout_step: int = field(default=0)

    # ------------------------------------------------------------------
    # 工厂方法：初始状态 s_0 = ⟨G_0, ∘⟩
    # ------------------------------------------------------------------

    @classmethod
    def create_initial(
        cls,
        base_graph: RetrievalBatch,
    ) -> State:
        """构造初始构造态 s_0 = ⟨G_0, ∘⟩。

        G_0 = 锚点诱导子图：节点为锚点集，边为两端均为锚点的全部边。

        注：max_steps 不再由 State 持有，由 RolloutEngine 管理。
        调用方不需要也不应该在此处传入 max_steps。
        """
        root_active_nodes = base_graph.is_anchor_mask.clone()
        root_active_edges = build_anchor_induced_edge_mask(
            base_graph.edge_index, root_active_nodes
        )
        return cls(
            root_active_edges=root_active_edges,
            active_nodes=root_active_nodes.clone(),
            active_edges=root_active_edges.clone(),
            phase=Phase.ACTIVE,
            rollout_step=0,
        )

    # ------------------------------------------------------------------
    # Policy 视图：detached clone，与内部张量不共享版本计数器
    # ------------------------------------------------------------------

    def as_policy_input(self) -> State:
        """返回供 policy forward 使用的状态视图。

        active_nodes / active_edges 是 detached clone：
        RolloutEngine 在 policy forward 期间继续修改 State 不会影响此视图。

        root_active_edges 是不变量，传引用即可（无需 clone）。
        """
        return State(
            root_active_edges=self.root_active_edges,
            active_nodes=self.active_nodes.detach().clone(),
            active_edges=self.active_edges.detach().clone(),
            phase=self.phase,
            rollout_step=int(self.rollout_step),
        )

    # ------------------------------------------------------------------
    # 状态转移：扩张转移 ⟨G_t, ∘⟩ → ⟨G_{t+1}, ∘⟩
    # ------------------------------------------------------------------

    def apply_expansion(
        self,
        *,
        chosen_edges: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> None:
        """执行加边动作 a_t，在 S∘ 内部游走。

        V_{t+1} = V_t ∪ {u, v}（由边集端点自动诱导）
        E_{t+1} = E_t ∪ {(u, r, v)}

        端点无条件激活——候选边约束（至少一端已激活）在 policy 侧保证，
        State 层直接激活更简洁且防御性更强。
        """
        if self.phase is not Phase.ACTIVE:
            raise RuntimeError(
                "apply_expansion called on a terminal state. "
                "State must be in Phase.ACTIVE."
            )
        if chosen_edges.numel() == 0:
            return

        self.active_edges[chosen_edges] = True
        self.active_nodes[src[chosen_edges]] = True
        self.active_nodes[dst[chosen_edges]] = True

    # ------------------------------------------------------------------
    # 状态转移：终止转移 ⟨G_t, ∘⟩ → ⟨G_t, ★⟩
    # ------------------------------------------------------------------

    def apply_stop(self) -> None:
        """执行 Stop 动作：∘ → ★。

        物理子图 G_t 保持不变，只更新相位。
        """
        if self.phase is not Phase.ACTIVE:
            raise RuntimeError(
                "apply_stop called on an already-terminal state."
            )
        self.phase = Phase.TERMINAL

    # ------------------------------------------------------------------
    # 属性查询
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """s_t ∈ S★"""
        return self.phase is Phase.TERMINAL

    @property
    def device(self) -> torch.device:
        return self.active_nodes.device

    @property
    def num_active_nodes(self) -> int:
        return int(self.active_nodes.sum().item())

    @property
    def num_active_edges(self) -> int:
        return int(self.active_edges.sum().item())

    # ------------------------------------------------------------------
    # 调试输出：避免 dataclass 默认打印完整 tensor
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"State("
            f"phase={self.phase.name}, "
            f"step={self.rollout_step}, "
            f"nodes={self.num_active_nodes}/{self.active_nodes.numel()}, "
            f"edges={self.num_active_edges}/{self.active_edges.numel()}, "
            f"device={self.device}"
            f")"
        )


__all__ = ["Phase", "State"]