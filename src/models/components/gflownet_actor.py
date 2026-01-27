from __future__ import annotations

import torch
from torch import nn

from .gflownet_layers import LogZPredictor
from .qc_bia_network import QCBiANetwork

_ZERO = 0
_ONE = 1
_TWO = 2
_DEFAULT_EDGE_INTER_DIM = 256
_DEFAULT_EDGE_DROPOUT = 0.1


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= _ZERO:
            raise ValueError("dim must be > 0.")
        half_dim = self.dim // _TWO
        inv_freq = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / max(half_dim, _ONE))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._has_odd = bool(self.dim % _TWO)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        steps = steps.to(device=self.inv_freq.device, dtype=torch.float32).view(-1, _ONE)
        freqs = steps * self.inv_freq.view(_ONE, -1)
        emb = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=-1)
        if self._has_odd:
            emb = torch.nn.functional.pad(emb, (_ZERO, _ONE))
        return emb


class GFlowNetActor(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        context_dim: int,
        edge_inter_dim: int = _DEFAULT_EDGE_INTER_DIM,
        edge_dropout: float = _DEFAULT_EDGE_DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.time_encoder = SinusoidalPositionalEncoding(self.hidden_dim)
        self.log_z_predictor = LogZPredictor(hidden_dim=self.hidden_dim, context_dim=self.context_dim)
        self.rel_logit_predictor = LogZPredictor(hidden_dim=self.hidden_dim, context_dim=self.context_dim)
        self.edge_policy = QCBiANetwork(
            d_plm=self.hidden_dim,
            d_kg=self.hidden_dim,
            d_inter=edge_inter_dim,
            dropout=edge_dropout,
        )

    def set_log_z_bias(self, bias: float) -> None:
        self.log_z_predictor.set_output_bias(bias)

    def _time_for_index(self, steps: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        steps = steps.to(device=index.device, dtype=torch.long).view(-1)
        if steps.numel() <= int(index.max().detach().tolist()):
            raise ValueError("steps length must cover max index.")
        time_emb = self.time_encoder(steps)
        return time_emb.index_select(0, index)

    def log_z(
        self,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        node_batch: torch.Tensor,
        steps: torch.Tensor,
        node_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_ids is None:
            node_tokens_sel = node_tokens
            node_batch_sel = node_batch
        else:
            node_ids = node_ids.to(device=node_tokens.device, dtype=torch.long).view(-1)
            node_tokens_sel = node_tokens.index_select(0, node_ids)
            node_batch_sel = node_batch.index_select(0, node_ids)
        time_emb = self._time_for_index(steps, node_batch_sel)
        node_tokens_sel = node_tokens_sel + time_emb
        return self.log_z_predictor(
            node_tokens=node_tokens_sel,
            question_tokens=question_tokens,
            node_batch=node_batch_sel,
        )

    def log_rel(
        self,
        *,
        edge_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        edge_batch = edge_batch.to(device=edge_tokens.device, dtype=torch.long).view(-1)
        time_emb = self._time_for_index(steps, edge_batch)
        edge_tokens = edge_tokens + time_emb
        return self.rel_logit_predictor(
            node_tokens=edge_tokens,
            question_tokens=question_tokens,
            node_batch=edge_batch,
        )

    def edge_logits(
        self,
        *,
        head_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        tail_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        edge_batch = edge_batch.to(device=head_tokens.device, dtype=torch.long).view(-1)
        time_emb = self._time_for_index(steps, edge_batch)
        head_tokens = head_tokens + time_emb
        question_tokens = question_tokens.index_select(0, edge_batch)
        return self.edge_policy(
            question_tokens,
            head_tokens,
            relation_tokens,
            tail_tokens,
            None,
        )


__all__ = ["GFlowNetActor", "SinusoidalPositionalEncoding"]
