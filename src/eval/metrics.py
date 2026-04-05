from __future__ import annotations

from typing import Any
import torch
from torch_scatter import scatter_sum
import math

from src.data.schema import RetrievalBatch

# 导入修复后的 RolloutBatch
from src.models.rollout import RolloutBatch


def build_union_context_graph(
    rollouts: list[RolloutBatch], batch: RetrievalBatch
) -> dict[str, torch.Tensor]:
    """
    拓扑聚合器：将 N 次独立采样的局部子图合并为全局联合交付图。

    Args:
        rollouts: N 次采样的结果列表，必须包含 terminal_active_nodes/edges。
        batch: PyG 原始图谱批次。

    Returns:
        dict 包含合并后的布尔掩码与沉没节点矩阵。
    """
    device = batch.node_tokens.device
    num_nodes = batch.node_tokens.size(0)
    num_edges = batch.edge_index.size(1)
    B = int(batch.ptr.numel()) - 1
    N = len(rollouts)

    # 1. 初始化全零的联合掩码
    union_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    union_edges = torch.zeros(num_edges, dtype=torch.bool, device=device)

    # 记录每次采样的终点 (用于计算多样性和命中率)
    all_sinks = torch.zeros((B, N), dtype=torch.long, device=device)

    # 2. 累加/并集操作 (Bitwise OR)
    for i, rollout in enumerate(rollouts):
        if rollout.terminal_active_nodes is not None:
            union_nodes |= rollout.terminal_active_nodes
        if rollout.terminal_active_edges is not None:
            union_edges |= rollout.terminal_active_edges
        if rollout.terminal_sinks is not None:
            all_sinks[:, i] = rollout.terminal_sinks

    return {
        "union_nodes": union_nodes,
        "union_edges": union_edges,
        "all_sinks": all_sinks,  # shape: [B, N]
    }


def compute_union_coverage_metrics(
    union_graph: dict[str, torch.Tensor], batch: RetrievalBatch
) -> dict[str, float]:
    """
    覆盖率计算：评估合并后的子图是否包含了大模型所需的金标准线索。
    """
    union_nodes = union_graph["union_nodes"]
    all_sinks = union_graph["all_sinks"]  # [B, N]

    # 假设 Schema 中 is_target_mask 标识了知识图谱中的金答案
    # 如果你的变量名不同，请在此处修改 (例如 target_nodes_mask)
    gold_mask = batch.is_target_mask

    B = int(batch.ptr.numel()) - 1

    # --- 1. Union Answer Recall (核心指标：答案是否在图中？) ---
    # 计算每个图中被纳入的正确答案数量
    hit_nodes = union_nodes & gold_mask
    hits_per_graph = scatter_sum(hit_nodes.float(), batch.batch, dim=0, dim_size=B)
    gold_per_graph = scatter_sum(gold_mask.float(), batch.batch, dim=0, dim_size=B)

    # 避免除以 0 (如果某个 batch 没有答案)
    recall_per_graph = hits_per_graph / gold_per_graph.clamp(min=1.0)
    mean_union_recall = recall_per_graph.mean().item()

    # --- 2. Sink Hit Rate (GFlowNet 是否至少一次收敛到了正确答案？) ---
    # 构建一个 [B] 形状的掩码，表示该图的 N 个 sink 中是否至少有一个是 target
    sink_hit_per_graph = torch.zeros(B, dtype=torch.bool, device=union_nodes.device)
    for b in range(B):
        # 取出第 b 个图的所有 sink 节点 ID
        sinks_for_b = all_sinks[b]
        sinks_for_b = sinks_for_b[sinks_for_b >= 0]
        if sinks_for_b.numel() == 0:
            continue
        # 检查这些节点是否在 gold_mask 中为 True
        if gold_mask[sinks_for_b].any():
            sink_hit_per_graph[b] = True

    mean_sink_hit_rate = sink_hit_per_graph.float().mean().item()

    return {
        "union_answer_recall": mean_union_recall,
        "sink_hit_rate": mean_sink_hit_rate,
    }


def compute_context_efficiency(
    union_graph: dict[str, torch.Tensor], batch: RetrievalBatch
) -> dict[str, float]:
    """
    上下文效率测算：评估冗余度。约束子图规模，防止 LLM Context 爆炸。
    """
    union_nodes = union_graph["union_nodes"]
    union_edges = union_graph["union_edges"]
    B = int(batch.ptr.numel()) - 1

    # 统计每个图的节点和边数量
    nodes_per_graph = scatter_sum(union_nodes.float(), batch.batch, dim=0, dim_size=B)

    # 对于边，我们需要用到 edge_batch_idx
    src = batch.edge_index[0]
    edge_batch_idx = batch.batch[src]
    edges_per_graph = scatter_sum(
        union_edges.float(), edge_batch_idx, dim=0, dim_size=B
    )

    return {
        "num_nodes": nodes_per_graph.mean().item(),
        "num_edges": edges_per_graph.mean().item(),
    }


def compute_exploration_diversity(
    union_graph: dict[str, torch.Tensor], batch: RetrievalBatch
) -> dict[str, float]:
    """
    探索多样性：量化 GFlowNet 在多次 Rollout 中探索不同模式的能力。
    """
    all_sinks = union_graph["all_sinks"]  # [B, N]
    B, N = all_sinks.shape

    if N == 0:
        return {"unique_sink_ratio": 0.0, "sink_entropy": 0.0}

    unique_ratios = []
    entropies = []

    for b in range(B):
        sinks = all_sinks[b]

        # 统计独立终点数量
        unique_sinks, counts = torch.unique(sinks, return_counts=True)
        unique_ratio = len(unique_sinks) / N
        unique_ratios.append(unique_ratio)

        # 计算香农熵 (Shannon Entropy) -> 衡量分布的均匀性
        probs = counts.float() / N
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        # 归一化熵 (可选): entropy / math.log(N)
        entropies.append(entropy)

    return {
        "unique_sink_ratio": sum(unique_ratios) / B,
        "sink_entropy": sum(entropies) / B,
    }


__all__ = [
    "build_union_context_graph",
    "compute_union_coverage_metrics",
    "compute_context_efficiency",
    "compute_exploration_diversity",
]
