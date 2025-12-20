from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TrajectoryStateEncoder(nn.Module):
    """Order-aware state encoder for Tree-state GFlowNet (trajectory-based)."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        dropout: float = 0.0,
        use_start_state: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_start_state = bool(use_start_state)
        self.input_dropout = nn.Dropout(float(dropout))
        self.gru_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        *,
        actions_seq: torch.Tensor,          # [B, T]
        edge_tokens: torch.Tensor,          # [E_total, H]
        stop_indices: torch.Tensor,         # [B]
        question_tokens: torch.Tensor,      # [B, H]
        node_tokens: Optional[torch.Tensor] = None,  # [N_total, H]
        start_node_locals: Optional[torch.Tensor] = None,  # [S_total]
        start_ptr: Optional[torch.Tensor] = None,  # [B+1]
    ) -> torch.Tensor:
        if actions_seq.dim() != 2:
            raise ValueError(f"actions_seq must be [B,T], got shape={tuple(actions_seq.shape)}")
        if edge_tokens.dim() != 2:
            raise ValueError(f"edge_tokens must be [E,H], got shape={tuple(edge_tokens.shape)}")
        num_graphs, num_steps = actions_seq.shape
        if question_tokens.shape != (num_graphs, self.hidden_dim):
            raise ValueError(
                f"question_tokens must be [B,H]=({num_graphs},{self.hidden_dim}), got {tuple(question_tokens.shape)}"
            )
        if stop_indices.shape != (num_graphs,):
            raise ValueError(f"stop_indices must be [B], got shape={tuple(stop_indices.shape)}")
        if edge_tokens.size(-1) != self.hidden_dim:
            raise ValueError(f"edge_tokens hidden_dim mismatch: {edge_tokens.size(-1)} vs {self.hidden_dim}")

        device = edge_tokens.device
        dtype = edge_tokens.dtype
        actions_seq = actions_seq.to(device=device)
        stop_indices = stop_indices.to(device=device)
        h = question_tokens.to(device=device, dtype=dtype)

        if self.use_start_state:
            if node_tokens is None or start_node_locals is None or start_ptr is None:
                raise ValueError("use_start_state=True requires node_tokens, start_node_locals, and start_ptr.")
            start_state = self._compute_start_state(
                node_tokens=node_tokens,
                start_node_locals=start_node_locals,
                start_ptr=start_ptr,
                num_graphs=num_graphs,
                device=device,
                dtype=dtype,
            )
            h = h + start_state

        h = self.state_norm(h)
        state_emb_seq = torch.zeros(num_graphs, num_steps, self.hidden_dim, device=device, dtype=dtype)
        done = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        max_edge = int(edge_tokens.size(0))

        for step in range(num_steps):
            state_emb_seq[:, step] = self.state_norm(h)
            actions = actions_seq[:, step]
            is_stop = (actions == stop_indices) | (actions < 0)
            if bool(((actions >= max_edge) & (actions != stop_indices)).any().item()):
                raise ValueError("actions_seq contains out-of-range edge indices.")
            active = (~done) & (~is_stop)
            if active.any():
                edge_idx = actions.clamp(min=0, max=max_edge - 1)
                edge_in = self.input_dropout(edge_tokens[edge_idx])
                h_next = self.gru_cell(edge_in, h)
                h = torch.where(active.unsqueeze(-1), h_next, h)
            done = done | is_stop

        return state_emb_seq

    def _compute_start_state(
        self,
        *,
        node_tokens: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if node_tokens.dim() != 2:
            raise ValueError(f"node_tokens must be [N,H], got shape={tuple(node_tokens.shape)}")
        if start_ptr.numel() != num_graphs + 1:
            raise ValueError("start_ptr length mismatch with num_graphs.")
        counts = start_ptr.to(device=device).view(-1)[1:] - start_ptr.to(device=device).view(-1)[:-1]
        if counts.numel() != num_graphs:
            raise ValueError("start_ptr length mismatch; expected one count per graph.")
        if (counts < 0).any():
            raise ValueError("start_ptr must be non-decreasing.")

        start_nodes = start_node_locals.to(device=device, dtype=torch.long).view(-1)
        start_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), counts)
        start_sum = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=dtype)
        start_sum.index_add_(0, start_batch, node_tokens.to(device=device, dtype=dtype)[start_nodes])
        denom = counts.clamp(min=1).to(dtype=dtype).unsqueeze(-1)
        return start_sum / denom


__all__ = ["TrajectoryStateEncoder"]
