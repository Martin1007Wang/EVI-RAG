# src/models/environment/builder.py
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from src.models.configs.environment import EnvironmentConfig
from .contracts import GraphEnvContext, DynamicAgentState, CsrAdjacency


def validate_ranges(
    *,
    edge_index: torch.Tensor,
    edge_relations: torch.Tensor,
    edge_rel_global: torch.Tensor | None = None,
    num_nodes_total: int,
) -> int:
    """验证数据范围，建议在训练成熟后通过 flag 关闭以提升吞吐量"""
    if edge_index.numel() > 0:
        torch._assert((edge_index >= 0).all(), "edge_index contains negative values.")
        torch._assert((edge_index < num_nodes_total).all(), "edge_index out of range for num_nodes_total.")
        torch._assert((edge_relations >= 0).all(), "edge_relations contains negative values.")
        if edge_rel_global is not None and edge_rel_global.numel() > 0:
            torch._assert((edge_rel_global >= 0).all(), "edge_rel_global contains negative values.")
    return int(edge_relations.max().item()) + 1 if edge_relations.numel() > 0 else 0


class GraphEnvironmentBuilder:
    """
    [系统实体] 动静分离环境构建器
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config

    @staticmethod
    def _build_csr_with_edge_ids(
        edge_index: torch.Tensor,
        edge_ids: torch.Tensor,
        *,
        num_nodes_total: int,
    ) -> CsrAdjacency:
        if edge_index.numel() == 0:
            crow = torch.zeros((num_nodes_total + 1,), device=edge_index.device, dtype=torch.long)
            col = edge_index.new_empty((0,), dtype=torch.long)
            vals = edge_ids.new_empty((0,), dtype=torch.long)
            return CsrAdjacency(crow=crow, col=col, edge_ids=vals, size=(num_nodes_total, num_nodes_total))
        heads = edge_index[0]
        heads_sorted, order = torch.sort(heads)
        tails_sorted = edge_index[1].index_select(0, order)
        vals_sorted = edge_ids.index_select(0, order)
        row_ids = torch.arange(num_nodes_total + 1, device=edge_index.device, dtype=heads_sorted.dtype)
        crow = torch.searchsorted(heads_sorted, row_ids, right=False).to(dtype=torch.long)
        if int(crow[-1].item()) != int(tails_sorted.numel()):
            raise ValueError("CSR invariant violated: crow[-1] != col length.")
        if int(tails_sorted.numel()) != int(vals_sorted.numel()):
            raise ValueError("CSR invariant violated: col length != values length.")
        return CsrAdjacency(crow=crow, col=tails_sorted, edge_ids=vals_sorted, size=(num_nodes_total, num_nodes_total))

    def build_context(self, batch: Mapping[str, Any]) -> GraphEnvContext:
        num_graphs = int(batch["num_graphs"])
        num_nodes_total = int(batch["num_nodes_total"])
        device = batch["edge_index"].device
        edge_index = batch["edge_index"]
        edge_relations = batch["edge_relations"]
        edge_rel_global = batch["edge_rel_global"]
        # 校验范围
        validate_ranges(
            edge_index=edge_index,
            edge_relations=edge_relations,
            edge_rel_global=edge_rel_global,
            num_nodes_total=num_nodes_total,
        )
        num_edges = edge_index.size(1)
        # 为每条边生成全局唯一的自增 ID [0, 1, ..., E-1]
        edge_ids = torch.arange(num_edges, dtype=torch.long, device=device)
        # 1. 前向拓扑 (Out-degree CSR)
        # 注意：必须保留多重边，因此不能使用会 coalesce 的 to_torch_csr_tensor。
        adj_t_fwd = self._build_csr_with_edge_ids(
            edge_index=edge_index,
            edge_ids=edge_ids,
            num_nodes_total=num_nodes_total,
        )
        # 2. 后向拓扑 (In-degree CSR)
        # [核心修正]：利用高级索引 [1, 0] 直接交换 COO 的行。
        # 这是一个零内存拷贝的视图(View)操作，速度达到内存带宽极限。
        edge_index_bwd = edge_index[[1, 0], :]
        adj_t_bwd = self._build_csr_with_edge_ids(
            edge_index=edge_index_bwd,
            edge_ids=edge_ids,
            num_nodes_total=num_nodes_total,
        )
        return GraphEnvContext(
            num_graphs=num_graphs,
            num_nodes_total=num_nodes_total,
            node_ptr=batch["node_ptr"],
            edge_index=edge_index,
            edge_relations=edge_relations,
            edge_rel_global=edge_rel_global,
            edge_batch=batch["edge_batch"],
            node_batch=batch["node_batch"],
            adj_t_fwd=adj_t_fwd,
            adj_t_bwd=adj_t_bwd,
            node_embeddings=batch["node_embeddings"],
            node_tokens=batch["node_tokens"],
            relation_tokens=batch["relation_tokens"],
            question_emb=batch["question_emb"],
            question_ctx=batch.get("question_ctx"),
            question_ctx_mask=batch.get("question_ctx_mask"),
            q_local_indices=batch["q_local_indices"],
            a_local_indices=batch["a_local_indices"],
            q_ptr=batch["q_ptr"],
            a_ptr=batch["a_ptr"],
            answer_entity_ids=batch["answer_entity_ids"],
            answer_ptr=batch["answer_ptr"],
            node_global_ids=batch["node_global_ids"],
            dummy_mask=batch.get("dummy_mask", torch.zeros((num_graphs,), dtype=torch.bool, device=device)),
            sample_ids=batch.get("sample_ids", [f"sample_{i}" for i in range(num_graphs)]),
            heuristic_log_v=batch.get("heuristic_log_v"),
            start_local_indices=batch.get("start_local_indices"),
            backward_start_local_indices=batch.get("backward_start_local_indices"),
        )

    def initialize_state(self, context: GraphEnvContext, num_agents: int = 1) -> DynamicAgentState:
        """
        基于静态上下文，生成时间步 t=0 的初始游走状态
        """
        device = context.node_ptr.device  # 修正了先前的 devices 拼写错误
        B = context.num_graphs
        if context.start_local_indices is not None:
            if context.start_local_indices.dim() != 1 or int(context.start_local_indices.numel()) != B:
                raise ValueError(
                    "start_local_indices must be [B] in initialize_state when provided: "
                    f"got shape={tuple(context.start_local_indices.shape)} B={B}."
                )
            start_local = context.start_local_indices.to(device=device, dtype=torch.long)
        else:
            q_counts = (context.q_ptr[1:] - context.q_ptr[:-1]).clamp(min=0)
            if bool((q_counts <= 0).any().item()):
                raise ValueError("q_local_indices has empty groups in initialize_state.")
            start_local = context.q_local_indices.index_select(0, context.q_ptr[:-1].to(device=device, dtype=torch.long))
        # 计算绝对起始坐标：local_index + node_ptr
        start_nodes_absolute = start_local + context.node_ptr[:-1]
        # 扩展为多智能体视角 [B, num_agents]
        current_nodes = start_nodes_absolute.unsqueeze(1).expand(B, num_agents).clone()
        # 初始化隐藏状态为问题向量 [B, num_agents, d]
        hidden_states = context.question_emb.unsqueeze(1).expand(B, num_agents, -1).clone()
        path_token_ids = current_nodes.unsqueeze(-1).clone()
        path_token_types = torch.zeros_like(path_token_ids, dtype=torch.bool)
        path_lengths = torch.ones((B, num_agents), dtype=torch.long, device=device)
        # 初始化访问遮罩
        visited_mask = torch.zeros((context.num_nodes_total,), dtype=torch.bool, device=device)
        visited_mask.scatter_(0, start_nodes_absolute, True)
        return DynamicAgentState(
            step_t=0,
            current_nodes=current_nodes,
            hidden_states=hidden_states,
            visited_mask=visited_mask,
            cumulative_rewards=torch.zeros((B, num_agents), dtype=torch.float, device=device),
            done_mask=torch.zeros((B, num_agents), dtype=torch.bool, device=device),
            path_token_ids=path_token_ids,
            path_token_types=path_token_types,
            path_lengths=path_lengths,
        )


__all__ = [
    "GraphEnvironmentBuilder",
]
