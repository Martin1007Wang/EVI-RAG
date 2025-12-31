from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.models.components.graph import DDE


@dataclass(frozen=True)
class StateEncoderCache:
    question_tokens: torch.Tensor   # [B, H]
    node_tokens: torch.Tensor       # [N_total, H]
    node_batch: torch.Tensor        # [N_total]
    node_struct_tokens: torch.Tensor  # [N_total, H] or empty


class StateEncoder(nn.Module):
    """State = mean(active nodes) + question + step embedding + action history."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_steps: int,
        use_state_dde: bool = False,
        state_dde_cfg: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        self.use_state_dde = bool(use_state_dde)
        self._state_dde: Optional[DDE] = None
        self._state_dde_proj: Optional[nn.Linear] = None
        self._state_dde_num_topics = 0
        self._state_dde_dim = 0
        if self.use_state_dde:
            if state_dde_cfg is None:
                raise ValueError("state_dde_cfg must be provided when use_state_dde is True.")
            num_topics = int(state_dde_cfg.get("num_topics", 0))
            num_rounds = int(state_dde_cfg.get("num_rounds", 0))
            num_rev = int(state_dde_cfg.get("num_reverse_rounds", 0))
            if num_topics <= 1:
                raise ValueError(f"state_dde_cfg.num_topics must be >= 2 when use_state_dde=1, got {num_topics}")
            struct_dim = int(num_topics) * (1 + int(num_rounds) + int(num_rev))
            self._state_dde = DDE(num_rounds=num_rounds, num_reverse_rounds=num_rev)
            self._state_dde_proj = nn.Linear(struct_dim, self.hidden_dim)
            self._state_dde_num_topics = int(num_topics)
            self._state_dde_dim = int(struct_dim)

        self.step_embeddings = nn.Embedding(self.max_steps + 1, self.hidden_dim)
        nn.init.constant_(self.step_embeddings.weight, 0.0)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def precompute(
        self,
        *,
        node_ptr: torch.Tensor,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        start_node_locals: Optional[torch.Tensor] = None,
        reverse_edge_index: Optional[torch.Tensor] = None,
        node_struct_raw: Optional[torch.Tensor] = None,
        **_: torch.Tensor,
    ) -> StateEncoderCache:
        if question_tokens.dim() != 2 or question_tokens.size(-1) != self.hidden_dim:
            raise ValueError("question_tokens must be [B,H] with hidden_dim H.")
        num_graphs = int(question_tokens.size(0))
        if node_ptr.numel() != num_graphs + 1:
            raise ValueError("node_ptr length mismatch with question_tokens batch size.")
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        node_batch = torch.repeat_interleave(
            torch.arange(num_graphs, device=node_ptr.device),
            node_counts,
        )
        if node_batch.numel() != node_tokens.size(0):
            raise ValueError("node_batch length mismatch with node_tokens in encoder precompute.")
        node_struct_tokens = torch.empty(0, self.hidden_dim, device=node_tokens.device, dtype=node_tokens.dtype)
        if self.use_state_dde:
            if node_struct_raw is not None and node_struct_raw.numel() > 0:
                if self._state_dde_proj is None:
                    raise RuntimeError("state_dde projection is not initialized.")
                node_struct_raw = node_struct_raw.to(device=node_tokens.device, dtype=node_tokens.dtype)
                if node_struct_raw.dim() != 2 or node_struct_raw.size(0) != node_tokens.size(0):
                    raise ValueError("node_struct_raw must be [N_total, D] aligned with node_tokens.")
                if node_struct_raw.size(1) != self._state_dde_dim:
                    raise ValueError(
                        f"node_struct_raw dim {node_struct_raw.size(1)} != expected {self._state_dde_dim}"
                    )
                node_struct_tokens = self._state_dde_proj(node_struct_raw)
            else:
                if edge_index is None:
                    raise ValueError("use_state_dde=True but edge_index is missing in StateEncoder.precompute.")
                if start_node_locals is None:
                    raise ValueError("use_state_dde=True but start_node_locals is missing in StateEncoder.precompute.")
                node_struct_tokens = self._build_state_dde_tokens(
                    edge_index=edge_index,
                    start_node_locals=start_node_locals,
                    num_nodes=node_tokens.size(0),
                    device=node_tokens.device,
                    dtype=node_tokens.dtype,
                    reverse_edge_index=reverse_edge_index,
                )
        return StateEncoderCache(
            question_tokens=question_tokens,
            node_tokens=node_tokens,
            node_batch=node_batch,
            node_struct_tokens=node_struct_tokens,
        )

    def encode_state(self, *, cache: StateEncoderCache, state: "GraphState") -> torch.Tensor:
        question_tokens = cache.question_tokens
        num_graphs = int(question_tokens.size(0))
        device = question_tokens.device
        dtype = question_tokens.dtype

        node_tokens = cache.node_tokens.to(device=device)
        node_batch = cache.node_batch.to(device=device)
        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        if active_nodes.numel() != node_tokens.size(0):
            raise ValueError("active_nodes length mismatch with node_tokens in encoder.")

        struct_tokens = cache.node_struct_tokens.to(device=device) if cache.node_struct_tokens.numel() > 0 else None
        active_mean, struct_mean = self._compute_active_means(
            node_tokens=node_tokens,
            struct_tokens=struct_tokens,
            node_batch=node_batch,
            active_nodes=active_nodes,
            num_graphs=num_graphs,
        )

        action_context = self._update_action_context(
            state=state,
            num_graphs=num_graphs,
            device=device,
            dtype=node_tokens.dtype,
        )

        step_counts = state.step_counts.to(device=device, dtype=torch.long).clamp(min=0, max=self.max_steps)
        remaining = (self.max_steps - step_counts).clamp(min=0, max=self.max_steps)
        step_emb = self.step_embeddings(remaining)

        state_tokens = (
            active_mean.to(dtype=dtype)
            + question_tokens
            + step_emb.to(dtype=dtype)
            + action_context.to(dtype=dtype)
        )
        if struct_mean is not None:
            state_tokens = state_tokens + struct_mean.to(dtype=dtype)
        return self.norm(state_tokens)

    def _compute_active_means(
        self,
        *,
        node_tokens: torch.Tensor,
        struct_tokens: Optional[torch.Tensor],
        node_batch: torch.Tensor,
        active_nodes: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not bool(active_nodes.any().item()):
            active_mean = torch.zeros(num_graphs, self.hidden_dim, device=node_tokens.device, dtype=node_tokens.dtype)
            struct_mean = None
            if struct_tokens is not None:
                struct_mean = torch.zeros_like(active_mean)
            return active_mean, struct_mean

        active_batch = node_batch[active_nodes]
        active_count = torch.bincount(active_batch, minlength=num_graphs).clamp(min=1).to(device=node_tokens.device)

        active_sum = torch.zeros(num_graphs, self.hidden_dim, device=node_tokens.device, dtype=node_tokens.dtype)
        active_sum.index_add_(0, active_batch, node_tokens[active_nodes])
        active_mean = active_sum / active_count.unsqueeze(-1).to(dtype=active_sum.dtype)

        struct_mean = None
        if struct_tokens is not None:
            if struct_tokens.size(0) != node_tokens.size(0):
                raise ValueError("struct_tokens length mismatch with node_tokens in encoder.")
            struct_sum = torch.zeros(num_graphs, self.hidden_dim, device=node_tokens.device, dtype=struct_tokens.dtype)
            struct_sum.index_add_(0, active_batch, struct_tokens[active_nodes])
            struct_mean = struct_sum / active_count.unsqueeze(-1).to(dtype=struct_sum.dtype)
        return active_mean, struct_mean

    def _build_state_dde_tokens(
        self,
        *,
        edge_index: torch.Tensor,
        start_node_locals: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        reverse_edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._state_dde is None or self._state_dde_proj is None:
            raise RuntimeError("state_dde modules are not initialized.")
        edge_index = edge_index.to(device=device)
        start_node_locals = start_node_locals.to(device=device, dtype=torch.long).view(-1)
        if start_node_locals.numel() > 0:
            if (start_node_locals < 0).any() or (start_node_locals >= num_nodes).any():
                raise ValueError("start_node_locals out of range for state_dde feature construction.")

        topic_entity_mask = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if start_node_locals.numel() > 0:
            topic_entity_mask[start_node_locals] = 1
        topic_one_hot = F.one_hot(topic_entity_mask, num_classes=int(self._state_dde_num_topics)).to(dtype=dtype)

        feats = [topic_one_hot]
        if self._state_dde is not None:
            rev = reverse_edge_index.to(device=device) if reverse_edge_index is not None else None
            feats.extend(self._state_dde(topic_one_hot, edge_index, rev))
        stacked = torch.stack(feats, dim=-1)
        node_struct = stacked.reshape(num_nodes, -1)
        if node_struct.size(-1) != self._state_dde_dim:
            raise ValueError(
                f"state_dde feature dim {node_struct.size(-1)} != expected {self._state_dde_dim}"
            )
        return self._state_dde_proj(node_struct)

    def _update_action_context(
        self,
        *,
        state: "GraphState",
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        action_hidden = getattr(state, "action_hidden", None)
        if action_hidden is None or not torch.is_tensor(action_hidden) or action_hidden.numel() == 0:
            return torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=dtype)
        action_hidden = action_hidden.to(device=device, dtype=dtype)
        if action_hidden.shape != (num_graphs, self.hidden_dim):
            return torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=dtype)
        return action_hidden


__all__ = ["StateEncoder", "StateEncoderCache"]
