# src/models/environment/contracts.py
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CsrAdjacency:
    """Lightweight CSR adjacency wrapper that preserves multi-edges."""

    crow: torch.Tensor
    col: torch.Tensor
    edge_ids: torch.Tensor
    size: tuple[int, int]

    def crow_indices(self) -> torch.Tensor:
        return self.crow

    def col_indices(self) -> torch.Tensor:
        return self.col

    def values(self) -> torch.Tensor:
        return self.edge_ids


@dataclass(frozen=True)
class GraphEnvContext:
    """
    [环境基座] 静态图上下文，整个 Episode 生命周期内绝对不可变。
    由 Builder 在 DataLoader 输出后单次构建。
    """

    # 宏观物理统计
    num_graphs: int
    num_nodes_total: int
    # 基础图结构 (PyG Batch 语义)
    node_ptr: torch.Tensor  # [B+1]
    edge_index: torch.Tensor  # [2, E]
    edge_relations: torch.Tensor  # [E]
    edge_rel_global: torch.Tensor  # [E]
    edge_batch: torch.Tensor  # [E]
    node_batch: torch.Tensor  # [N]
    # 现代稀疏张量拓扑 (核心优化点：替代手写的 8 个张量)
    adj_t_fwd: CsrAdjacency
    adj_t_bwd: CsrAdjacency
    # 神经网络连续表征
    node_embeddings: torch.Tensor  # [N, d]
    node_tokens: torch.Tensor  # [N, d]
    relation_tokens: torch.Tensor  # [num_relations, d]
    question_emb: torch.Tensor  # [B, d]
    # 任务信标 (绝对坐标)
    q_local_indices: torch.Tensor  # [B]
    a_local_indices: torch.Tensor  # [num_answers]
    q_ptr: torch.Tensor  # [B+1]
    a_ptr: torch.Tensor  # [B+1]
    answer_entity_ids: torch.Tensor  # [num_answers]
    answer_ptr: torch.Tensor  # [B+1]
    # 辅助与元数据
    node_global_ids: torch.Tensor  # [N]
    dummy_mask: torch.Tensor  # [B]
    sample_ids: list[str]  # [B]
    heuristic_log_v: torch.Tensor | None = None  # [N], optional frozen guidance
    start_local_indices: torch.Tensor | None = None  # [B], optional explicit per-graph start override
    replay_start_local: torch.Tensor | None = None  # [sum paths], optional replay oracle starts (local node ids)
    replay_path_lengths: torch.Tensor | None = None  # [sum paths], optional replay oracle path lengths
    replay_edge_local_ids: torch.Tensor | None = None  # [sum edges], optional replay oracle local edge ids
    replay_path_ptr: torch.Tensor | None = None  # [B+1], optional graph ptr for replay_path_lengths/starts
    replay_edge_ptr: torch.Tensor | None = None  # [B+1], optional graph ptr for replay_edge_local_ids


@dataclass
class DynamicAgentState:
    """
    [推演状态] 智能体在时间步 t 的动态游走状态。
    """

    step_t: int
    # 智能体绝对物理坐标 (考虑了 node_ptr 的偏移)
    current_nodes: torch.Tensor  # [B, num_agents_per_graph]
    # 当前持有的隐状态 (例如 LSTM/GRU 隐藏层，或历史路径向量)
    # 初始时等同于 question_emb 的扩展
    hidden_states: torch.Tensor  # [B, num_agents_per_graph, d]
    # 访问历史与遮罩，防止死循环 (True 表示已访问)
    visited_mask: torch.Tensor  # [N_total] or [B*num_agents_per_graph, N_total]
    # 轨迹记录与结算
    cumulative_rewards: torch.Tensor  # [B, num_agents_per_graph]
    done_mask: torch.Tensor  # [B, num_agents_per_graph], 命中或超步数标记


__all__ = [
    "GraphEnvContext",
    "DynamicAgentState",
]
