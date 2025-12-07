from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add, scatter_log_softmax, scatter_sum


class GFlowNetEstimator(nn.Module):
    """负责 logZ / logPB 估计。支持均匀 / 学习式反向策略。"""

    def __init__(
        self,
        *,
        hidden_dim: int,
        log_pb_mode: str = "learned",
        learn_pb: bool = True,
        pb_entropy_coef: float = 0.0,
        pb_l2_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.log_pb_mode = str(log_pb_mode).lower()
        self.learn_pb = bool(learn_pb)
        self.pb_entropy_coef = float(pb_entropy_coef)
        self.pb_l2_reg = float(pb_l2_reg)
        # Warning: edge_tokens MUST encode source-node信息，否则无法区分同关系的不同父节点。
        self.log_z_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        last_linear = self.log_z_head[-1]
        if hasattr(last_linear, "bias") and last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 0.0)
        # Concatenate start/question then project back to hidden_dim.
        self.ctx_projector = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.backward_head: Optional[nn.Module] = None
        if self.log_pb_mode == "learned":
            # Edge + question + tail node tokens (source already在 edge_tokens 内部)
            input_dim = self.hidden_dim * 3
            self.backward_head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, 1),
            )
            # Zero-init 末层，令初始分布为均匀
            nn.init.constant_(self.backward_head[-1].weight, 0.0)
            nn.init.constant_(self.backward_head[-1].bias, 0.0)
        self._eps = 1e-12

    def build_context(self, start_summary: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([start_summary, question_tokens], dim=-1)
        return self.ctx_projector(combined)

    def aggregate_start(
        self,
        *,
        node_tokens: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_node_ptr: torch.Tensor,
        num_graphs: int,
        question_tokens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if start_node_ptr.numel() != num_graphs + 1:
            raise ValueError("start_node_ptr length mismatch; expected one offset per graph.")
        counts = start_node_ptr[1:] - start_node_ptr[:-1]
        if counts.numel() != num_graphs:
            raise ValueError("start_node_ptr length mismatch; expected one count per graph.")
        if (counts <= 0).any():
            missing = torch.nonzero(counts <= 0, as_tuple=False).view(-1).cpu().tolist()
            raise ValueError(f"start_node_locals must be non-empty per graph; missing at indices {missing}")

        start_tokens = node_tokens[start_node_locals]
        batch_index = torch.repeat_interleave(torch.arange(num_graphs, device=device), counts)
        query_tokens = question_tokens[batch_index]

        scores = (start_tokens * query_tokens).sum(dim=-1, keepdim=True) / (self.hidden_dim ** 0.5)
        alpha = pyg_softmax(scores, batch_index)
        weighted = start_tokens * alpha

        summary = torch.zeros(num_graphs, self.hidden_dim, device=device)
        summary = summary.index_add(0, batch_index, weighted)
        return summary

    def log_z(self, context: torch.Tensor) -> torch.Tensor:
        return self.log_z_head(context).squeeze(-1)

    def log_pb(
        self,
        *,
        edge_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        selected_mask: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute backward log-probabilities and auxiliary losses."""
        num_graphs = int(question_tokens.size(0))
        target_nodes = edge_index[1]

        if self.log_pb_mode == "tree":
            raise ValueError("log_pb_mode 'tree' is invalid for non-tree graphs; use 'uniform' or 'learned'.")

        # --- Uniform 模式：log(1/deg) ---
        if self.log_pb_mode == "uniform" or self.backward_head is None:
            ones = torch.ones_like(target_nodes, dtype=torch.float)
            degree = scatter_add(ones, target_nodes, dim=0)
            log_pb_edges = -torch.log(degree[target_nodes] + self._eps)
            log_pb_per_graph = scatter_add(
                log_pb_edges * selected_mask.float(), edge_batch, dim=0, dim_size=num_graphs
            )
            return log_pb_per_graph, {}

        # --- Learned 模式：均匀先验 + 残差 ---
        tail_tokens = node_tokens[target_nodes]
        edge_ctx = torch.cat([edge_tokens, question_tokens[edge_batch], tail_tokens], dim=-1)
        logits = self.backward_head(edge_ctx).squeeze(-1)
        device = logits.device
        log_probs = scatter_log_softmax(logits, target_nodes, dim=0)

        selected_f = selected_mask.float()
        log_pb_per_graph = scatter_add(log_probs * selected_f, edge_batch, dim=0, dim_size=num_graphs)

        aux_losses: Dict[str, torch.Tensor] = {}
        if self.learn_pb:
            if selected_mask.any():
                nll_per_edge = -log_probs * selected_f
                nll_per_graph = scatter_add(nll_per_edge, edge_batch, dim=0, dim_size=num_graphs)
                has_edge = scatter_add(selected_f, edge_batch, dim=0, dim_size=num_graphs) > 0
                aux_losses["pb_nll"] = nll_per_graph[has_edge].mean()
            else:
                aux_losses["pb_nll"] = logits.sum() * 0.0

            if self.pb_entropy_coef > 0.0:
                probs = log_probs.exp()
                entropy_edge = -probs * log_probs
                entropy_edge = torch.nan_to_num(entropy_edge, nan=0.0, posinf=0.0, neginf=0.0)
                node_entropy = scatter_sum(entropy_edge, target_nodes, dim=0)
                unique_targets = torch.unique(target_nodes)
                if unique_targets.numel() > 0:
                    mean_entropy = node_entropy[unique_targets].mean()
                    aux_losses["pb_entropy"] = -self.pb_entropy_coef * mean_entropy
                    aux_losses["pb_avg_entropy"] = mean_entropy.detach()
                else:
                    zero = torch.zeros((), device=device, dtype=logits.dtype)
                    aux_losses["pb_entropy"] = zero
                    aux_losses["pb_avg_entropy"] = zero

            if self.pb_l2_reg > 0.0:
                l2 = torch.zeros((), device=logits.device, dtype=logits.dtype)
                for param in self.backward_head.parameters():
                    l2 = l2 + param.pow(2).sum()
                aux_losses["pb_l2"] = self.pb_l2_reg * l2

        return log_pb_per_graph, aux_losses


__all__ = ["GFlowNetEstimator"]
