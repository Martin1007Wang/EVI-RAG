# src/models/components/gnn.py
"""
[系统实体] 关系图神经网络层
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class RelationalGNNLayer(nn.Module):
    """
    关系图神经网络层 (PNA 架构)
    修复了异构图关系映射，移除了数值地雷。
    """

    def __init__(self, *, hidden_dim: int, dropout: float, avg_d: float = 3.0) -> None:
        """
        Args:
            hidden_dim: 隐层维度
            dropout: 丢弃率
            avg_d: 训练集的全局平均度数 (必须是静态常量，默认设为经验值 3.0)
        """
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        # 预计算全局平均对数度数 delta = log(avg_d + 1)
        self.delta = torch.log(torch.tensor(avg_d + 1.0)).clamp(min=1.0e-6).item()
        self.msg_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        # 4 种聚合 (Mean, Max, Min, Std) x 3 种缩放 (Id, Amp, Att) = 12 组特征
        self.agg_proj = nn.Linear(self.hidden_dim * 3 * 4, self.hidden_dim)
        self.update_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        _init_linear(self.msg_proj)
        _init_linear(self.agg_proj)
        _init_linear(self.update_proj)

    def _safe_pna_stats(
        self,
        *,
        messages: torch.Tensor,
        tails: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算数值安全的 PNA 聚合统计量"""
        hidden_dim = messages.size(-1)
        device = messages.device
        dtype = messages.dtype
        # 1. 基础度数统计
        ones = torch.ones_like(tails, dtype=dtype)
        deg = torch.zeros((num_nodes,), device=device, dtype=dtype)
        deg.scatter_add_(0, tails, ones)
        has_in = deg > float(0)
        deg_safe = deg.clamp(min=1).unsqueeze(-1)
        # 2. Mean (替代了会随图规模爆炸的 Sum)
        sums = torch.zeros((num_nodes, hidden_dim), device=device, dtype=dtype)
        sums.scatter_add_(0, tails.unsqueeze(-1).expand(-1, hidden_dim), messages)
        mean = sums / deg_safe
        # 3. 数值安全的 Std (避免 catastrophic cancellation)
        # 计算 (x_i - mean_j)^2，然后再聚合求均值开根号
        mean_gathered = mean.index_select(0, tails)
        # In mixed precision, square can be promoted to fp32 while var_sum keeps fp16/bf16.
        # Force dtype alignment for scatter_add_ to avoid runtime dtype mismatch.
        diff_sq = (messages - mean_gathered).square().to(dtype=dtype)
        var_sum = torch.zeros((num_nodes, hidden_dim), device=device, dtype=dtype)
        var_sum.scatter_add_(0, tails.unsqueeze(-1).expand(-1, hidden_dim), diff_sq)
        var = var_sum / deg_safe
        std = var.clamp(min=1.0e-6).sqrt()
        # 4. Max & Min (使用 PyTorch 原生的 reduce)
        finfo = torch.finfo(dtype)
        tail_index = tails.unsqueeze(-1).expand(-1, hidden_dim)
        max_vals = torch.full((num_nodes, hidden_dim), finfo.min, device=device, dtype=dtype)
        max_vals.scatter_reduce_(0, tail_index, messages, reduce="amax", include_self=False)
        min_vals = torch.full((num_nodes, hidden_dim), finfo.max, device=device, dtype=dtype)
        min_vals.scatter_reduce_(0, tail_index, messages, reduce="amin", include_self=False)
        # 清理没有入边的节点特征 (避免产生无限大或无效的极端值)
        mask = has_in.unsqueeze(-1)
        max_vals = torch.where(mask, max_vals, torch.zeros_like(max_vals))
        min_vals = torch.where(mask, min_vals, torch.zeros_like(min_vals))
        std = torch.where(mask, std, torch.zeros_like(std))
        mean = torch.where(mask, mean, torch.zeros_like(mean))
        # [num_nodes, hidden_dim * 4]
        stats = torch.cat((mean, max_vals, min_vals, std), dim=-1)
        return stats, deg, has_in

    def _pna_aggregate(
        self,
        *,
        messages: torch.Tensor,
        tails: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """执行 PNA 的 Scaled Aggregation"""
        # stats: [N, D*4]
        stats, deg, has_in = self._safe_pna_stats(messages=messages, tails=tails, num_nodes=num_nodes)
        # 使用静态常量 self.delta 计算尺度
        log_deg = torch.log(deg + float(1)).clamp(min=float(1.0e-6))
        scale_identity = torch.ones_like(log_deg)
        scale_amplify = log_deg / self.delta
        scale_attenuate = self.delta / log_deg
        stats_dim = int(self.hidden_dim * 4)
        # 避免显式构造 [N, 3, 4D] / [N, 12D] 大中间张量：
        # Linear(concat(s_i * stats)) == Σ_i s_i * Linear_i(stats)
        w_id, w_amp, w_att = self.agg_proj.weight.split(stats_dim, dim=1)
        proj_id = F.linear(stats, w_id, bias=None)
        proj_amp = F.linear(stats, w_amp, bias=None)
        proj_att = F.linear(stats, w_att, bias=None)
        output = (
            proj_id * scale_identity.unsqueeze(-1)
            + proj_amp * scale_amplify.unsqueeze(-1)
            + proj_att * scale_attenuate.unsqueeze(-1)
        )
        if self.agg_proj.bias is not None:
            output = output + self.agg_proj.bias
        output = torch.where(has_in.unsqueeze(-1), output, torch.zeros_like(output))
        return output

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,  # <-- 修复契约：物理链路必须打通
        num_nodes: int,
    ) -> torch.Tensor:
        """
        前向传播：异构关系驱动的消息传递
        """
        if node_tokens.numel() == 0 or edge_index.numel() == 0:
            return node_tokens
        head = edge_index[0]
        tail = edge_index[1]
        # relation_tokens 尺寸为 [num_relations, d]，提取后成为 [E, d]
        edge_rel_features = relation_tokens.index_select(0, edge_relations)
        # 构建异构消息 M = W_msg( h_head + h_relation )
        head_features = node_tokens.index_select(0, head)
        msg = self.msg_proj(head_features + edge_rel_features)
        # 聚合消息
        agg = self._pna_aggregate(messages=msg, tails=tail, num_nodes=num_nodes)
        # 更新节点状态
        update_in = torch.cat((node_tokens, agg), dim=-1)
        update = self.update_proj(update_in)
        # 残差连接与归一化
        out = node_tokens + self.drop(self.act(update))
        return self.norm(out)


__all__ = ["RelationalGNNLayer"]
