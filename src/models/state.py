from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch

from src.data.schema import RetrievalBatch
from src.utils.reward_utils import build_anchor_induced_edge_mask


class Phase(Enum):
    ACTIVE = auto()
    TERMINAL = auto()


@dataclass
class State:
    """Subgraph MDP state with an expand budget and dynamic subgraph."""
    
    # ── 无默认值字段 (必须前置) ──
    root_active_edges: torch.Tensor  # G_0 锚点诱导边 (不变量)
    active_nodes: torch.Tensor       # V_t (动态集)
    active_edges: torch.Tensor       # E_t (动态集)

    # ── 有默认值字段 ──
    expand_budget: int = 0           # 最大扩张次数
    phase: Phase = Phase.ACTIVE      # phi (相位)
    num_expands: int = 0             # 当前已经 expand 了几次

    @classmethod
    def create_initial(
        cls, base_graph: RetrievalBatch, *, expand_budget: int = 0
    ) -> State:
        root_active_nodes = base_graph.is_anchor_mask.clone()
        root_active_edges = build_anchor_induced_edge_mask(base_graph.edge_index, root_active_nodes)
        
        return cls(
            root_active_edges=root_active_edges,
            active_nodes=root_active_nodes.clone(),
            active_edges=root_active_edges.clone(),
            expand_budget=expand_budget,
        )

    def as_policy_input(self) -> State:
        """返回 detached 视图，阻断在位修改梯度。"""
        return State(
            root_active_edges=self.root_active_edges,
            active_nodes=self.active_nodes.detach().clone(),
            active_edges=self.active_edges.detach().clone(),
            expand_budget=self.expand_budget,
            phase=self.phase,
            num_expands=self.num_expands,
        )

    def apply_expansion(self, *, chosen_edges: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        """V_{t+1} = V_t U {u, v}, E_{t+1} = E_t U {e}"""
        if self.phase is not Phase.ACTIVE:
            raise RuntimeError("Cannot expand a terminal state.")
        if chosen_edges.numel() == 0:
            return

        self.active_edges[chosen_edges] = True
        self.active_nodes[src[chosen_edges]] = True
        self.active_nodes[dst[chosen_edges]] = True

    def apply_stop(self) -> None:
        if self.phase is not Phase.ACTIVE:
            raise RuntimeError("State is already terminal.")
        self.phase = Phase.TERMINAL

    @property
    def remaining_budget(self) -> int:
        return max(0, self.expand_budget - self.num_expands)

    @property
    def expand_ratio(self) -> float:
        if self.expand_budget <= 0:
            return 0.0
        return min(1.0, self.num_expands / self.expand_budget)

    @property
    def is_terminal(self) -> bool:
        return self.phase is Phase.TERMINAL

    @property
    def device(self) -> torch.device:
        return self.active_nodes.device


__all__ = ["Phase", "State"]
