from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_FOUR = 4
_LOGZ_OUTPUT_DIM = 1
_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_GNN_LAYERS = 2
_DEFAULT_GNN_DROPOUT = 0.0
_DEFAULT_ADAPTER_ENABLED = False
_DEFAULT_ADAPTER_DIM_DIVISOR = 4
_DEFAULT_ADAPTER_DROPOUT = 0.1
_PNA_EPS = 1.0e-6
_PNA_SCALERS = _THREE
_PNA_AGGREGATORS = _FOUR
_PNA_FEATURE_MULT = _PNA_SCALERS * _PNA_AGGREGATORS


def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


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


class RelationalGNNLayer(nn.Module):
    def __init__(self, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= _ZERO:
            raise ValueError("hidden_dim must be > 0.")
        self.dropout = float(dropout)
        if self.dropout < float(_ZERO):
            raise ValueError("dropout must be >= 0.")
        self.msg_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.agg_proj = nn.Linear(self.hidden_dim * _PNA_FEATURE_MULT, self.hidden_dim)
        self.update_proj = nn.Linear(self.hidden_dim * _TWO, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        _init_linear(self.msg_proj)
        _init_linear(self.agg_proj)
        _init_linear(self.update_proj)

    def _pna_stats(
        self,
        *,
        messages: torch.Tensor,
        tails: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_dim = int(messages.size(-1))
        sums = torch.zeros((num_nodes, hidden_dim), device=messages.device, dtype=messages.dtype)
        sums.index_add_(0, tails, messages)
        sums_sq = torch.zeros((num_nodes, hidden_dim), device=messages.device, dtype=messages.dtype)
        sums_sq.index_add_(0, tails, messages.square())
        deg = torch.zeros((num_nodes,), device=messages.device, dtype=messages.dtype)
        ones = torch.ones_like(tails, dtype=messages.dtype)
        deg.index_add_(0, tails, ones)
        deg_safe = deg.clamp(min=_ONE).unsqueeze(-1)
        mean = sums / deg_safe
        var = sums_sq / deg_safe - mean.square()
        std = var.clamp(min=_PNA_EPS).sqrt()
        tail_index = tails.view(-1, _ONE).expand(-1, hidden_dim)
        finfo = torch.finfo(messages.dtype)
        max_vals = torch.full((num_nodes, hidden_dim), finfo.min, device=messages.device, dtype=messages.dtype)
        max_vals.scatter_reduce_(0, tail_index, messages, reduce="amax", include_self=True)
        min_vals = torch.full((num_nodes, hidden_dim), finfo.max, device=messages.device, dtype=messages.dtype)
        min_vals.scatter_reduce_(0, tail_index, messages, reduce="amin", include_self=True)
        has_in = deg > float(_ZERO)
        mask = has_in.unsqueeze(-1)
        max_vals = torch.where(mask, max_vals, torch.zeros_like(max_vals))
        min_vals = torch.where(mask, min_vals, torch.zeros_like(min_vals))
        std = torch.where(mask, std, torch.zeros_like(std))
        stats = torch.cat((sums, max_vals, min_vals, std), dim=-1)
        return stats, deg, has_in

    @staticmethod
    def _pna_scales(*, degree: torch.Tensor, has_in: torch.Tensor) -> torch.Tensor:
        deg = degree.to(dtype=torch.float32)
        has_in_f = has_in.to(device=degree.device, dtype=torch.float32)
        log_deg = torch.log(deg + float(_ONE))
        denom = has_in_f.sum().clamp(min=float(_ONE))
        avg_log_deg = (log_deg.mul(has_in_f).sum() / denom).clamp(min=float(_PNA_EPS))
        log_deg_safe = log_deg.clamp(min=float(_PNA_EPS))
        scale_identity = torch.ones_like(log_deg)
        scale_amplify = log_deg / avg_log_deg
        scale_attenuate = avg_log_deg / log_deg_safe
        return torch.stack((scale_identity, scale_amplify, scale_attenuate), dim=-1)

    def _pna_aggregate(
        self,
        *,
        messages: torch.Tensor,
        tails: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        stats, deg, has_in = self._pna_stats(messages=messages, tails=tails, num_nodes=num_nodes)
        scales = self._pna_scales(degree=deg, has_in=has_in).to(device=messages.device, dtype=messages.dtype)
        scaled = stats.unsqueeze(_ONE) * scales.unsqueeze(-1)
        features = scaled.reshape(num_nodes, -1)
        features = torch.where(has_in.unsqueeze(-1), features, torch.zeros_like(features))
        return self.agg_proj(features)

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if node_tokens.numel() == _ZERO:
            return node_tokens
        num_nodes = int(num_nodes)
        if num_nodes <= _ZERO:
            return node_tokens
        if node_tokens.size(0) != num_nodes:
            raise ValueError("num_nodes must match node_tokens length.")
        if edge_index.numel() == _ZERO:
            return node_tokens
        if relation_tokens.size(0) != edge_index.size(1):
            raise ValueError("relation_tokens must align with edge_index.")
        head = edge_index[_ZERO].to(dtype=torch.long)
        tail = edge_index[_ONE].to(dtype=torch.long)
        msg = node_tokens.index_select(0, head) + relation_tokens
        msg = self.msg_proj(msg)
        agg = self._pna_aggregate(messages=msg, tails=tail, num_nodes=num_nodes)
        update_in = torch.cat((node_tokens, agg), dim=-1)
        update = self.update_proj(update_in)
        out = node_tokens + self.drop(self.act(update))
        return self.norm(out)


class EmbeddingAdapter(nn.Module):
    def __init__(self, *, emb_dim: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.adapter_dim = int(adapter_dim)
        if self.emb_dim <= _ZERO or self.adapter_dim <= _ZERO:
            raise ValueError("emb_dim and adapter_dim must be > 0.")
        self.dropout = float(dropout)
        if self.dropout < float(_ZERO):
            raise ValueError("dropout must be >= 0.")
        self.norm = nn.LayerNorm(self.emb_dim)
        self.down = nn.Linear(self.emb_dim, self.adapter_dim)
        self.up = nn.Linear(self.adapter_dim, self.emb_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        _init_linear(self.down)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == _ZERO:
            return embeddings
        normalized = self.norm(embeddings)
        delta = self.drop(self.up(self.act(self.down(normalized))))
        out = embeddings + delta
        return out

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        _init_linear(self.down)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def sanitize_parameters_(self) -> bool:
        nonfinite = False
        for param in self.parameters():
            if not torch.isfinite(param).all():
                nonfinite = True
                break
        if nonfinite:
            with torch.no_grad():
                self.reset_parameters()
        return nonfinite


class EmbeddingBackbone(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        finetune: bool = _DEFAULT_BACKBONE_FINETUNE,
        gnn_layers: int = _DEFAULT_GNN_LAYERS,
        gnn_dropout: float = _DEFAULT_GNN_DROPOUT,
        adapter_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.finetune = bool(finetune)
        self.gnn_layers_count = int(gnn_layers)
        if self.gnn_layers_count < _ZERO:
            raise ValueError("gnn_layers must be >= 0.")
        self.gnn_dropout = float(gnn_dropout)
        if self.gnn_dropout < float(_ZERO):
            raise ValueError("gnn_dropout must be >= 0.")

        self.node_adapter, self.rel_adapter = self._init_adapter(adapter_cfg)
        self.node_norm = nn.LayerNorm(self.emb_dim)
        self.rel_norm = nn.LayerNorm(self.emb_dim)
        self.node_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.rel_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.q_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        _init_linear(self.node_proj)
        _init_linear(self.rel_proj)
        _init_linear(self.q_proj)
        self.gnn_layers = nn.ModuleList(
            [
                RelationalGNNLayer(hidden_dim=self.hidden_dim, dropout=self.gnn_dropout)
                for _ in range(self.gnn_layers_count)
            ]
        )
        if not self.finetune:
            for module in (self.node_norm, self.rel_norm, self.node_proj, self.rel_proj, self.q_proj):
                for param in module.parameters():
                    param.requires_grad = False

    def _init_adapter(
        self,
        adapter_cfg: Optional[Mapping[str, Any]],
    ) -> tuple[Optional[EmbeddingAdapter], Optional[EmbeddingAdapter]]:
        cfg = adapter_cfg or {}
        extra = set(cfg.keys()) - {"enabled", "adapter_dim", "dropout", "dim_divisor"}
        if extra:
            raise ValueError(f"Unsupported adapter_cfg keys: {sorted(extra)}")
        enabled = bool(cfg.get("enabled", _DEFAULT_ADAPTER_ENABLED))
        if not enabled:
            return None, None
        dim_divisor = int(cfg.get("dim_divisor", _DEFAULT_ADAPTER_DIM_DIVISOR))
        if dim_divisor <= _ZERO:
            raise ValueError("adapter_cfg.dim_divisor must be > 0.")
        adapter_dim = cfg.get("adapter_dim", None)
        if adapter_dim is None:
            adapter_dim = max(_ONE, self.emb_dim // dim_divisor)
        adapter_dim = int(adapter_dim)
        if adapter_dim <= _ZERO:
            raise ValueError("adapter_cfg.adapter_dim must be > 0.")
        dropout = float(cfg.get("dropout", _DEFAULT_ADAPTER_DROPOUT))
        if dropout < float(_ZERO):
            raise ValueError("adapter_cfg.dropout must be >= 0.")
        return (
            EmbeddingAdapter(emb_dim=self.emb_dim, adapter_dim=adapter_dim, dropout=dropout),
            EmbeddingAdapter(emb_dim=self.emb_dim, adapter_dim=adapter_dim, dropout=dropout),
        )

    def forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        question_emb = batch.question_emb.to(device=device, non_blocking=True)
        node_embeddings = batch.node_embeddings.to(device=device, non_blocking=True)
        edge_embeddings = batch.edge_embeddings.to(device=device, non_blocking=True)
        question_tokens = self.project_question_embeddings(question_emb)
        node_tokens = self.project_node_embeddings(node_embeddings)
        relation_tokens = self.project_relation_embeddings(edge_embeddings)
        return node_tokens, relation_tokens, question_tokens

    def project_node_embeddings(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if self.node_adapter is not None:
            node_embeddings = self.node_adapter(node_embeddings)
        node_normed = self.node_norm(node_embeddings)
        out = self.node_proj(node_normed)
        return out

    def project_relation_embeddings(self, relation_embeddings: torch.Tensor) -> torch.Tensor:
        if self.rel_adapter is not None:
            relation_embeddings = self.rel_adapter(relation_embeddings)
        rel_normed = self.rel_norm(relation_embeddings)
        return self.rel_proj(rel_normed)

    def project_question_embeddings(self, question_emb: torch.Tensor) -> torch.Tensor:
        return self.q_proj(question_emb)

    def encode_graph(
        self,
        *,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if self.gnn_layers_count == _ZERO:
            return node_tokens
        out = node_tokens
        for layer in self.gnn_layers:
            out = layer(
                node_tokens=out,
                relation_tokens=relation_tokens,
                edge_index=edge_index,
                num_nodes=num_nodes,
            )
        return out


class CvtNodeInitializer(nn.Module):
    """Zero-shot CVT initialization via neighbor + relation averaging."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _aggregate_incoming_mean(
        *,
        relation_embeddings: torch.Tensor,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heads = edge_index[_ZERO]
        tails = edge_index[_ONE]
        msg = node_embeddings.index_select(0, heads) + relation_embeddings
        sums = torch.zeros((num_nodes, msg.size(-1)), device=msg.device, dtype=msg.dtype)
        sums.index_add_(0, tails, msg)
        counts = torch.zeros((num_nodes,), device=msg.device, dtype=msg.dtype)
        ones = torch.ones_like(tails, dtype=msg.dtype)
        counts.index_add_(0, tails, ones)
        return sums, counts

    def forward(
        self,
        *,
        node_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        node_is_cvt: torch.Tensor,
    ) -> torch.Tensor:
        cvt_mask = node_is_cvt.to(dtype=torch.bool, device=node_embeddings.device)
        if not bool(cvt_mask.any().detach().tolist()):
            return node_embeddings
        num_nodes = int(node_embeddings.size(0))
        sums, counts = self._aggregate_incoming_mean(
            relation_embeddings=relation_embeddings,
            node_embeddings=node_embeddings,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        has_in = counts > float(_ZERO)
        missing = cvt_mask & (~has_in.to(dtype=torch.bool, device=cvt_mask.device))
        if bool(missing.any().detach().tolist()):
            raise ValueError("CVT nodes missing incoming edges; cannot compute head+relation mean.")
        mean = sums / counts.unsqueeze(-1)
        return torch.where(cvt_mask.unsqueeze(-1), mean, node_embeddings)


class LogZPredictor(nn.Module):
    def __init__(self, *, hidden_dim: int, context_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        input_dim = self.hidden_dim + self.context_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, _LOGZ_OUTPUT_DIM),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_linear(layer)

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if question_tokens.dim() == _THREE and question_tokens.size(1) == _ONE:
            question_tokens = question_tokens.squeeze(1)
        if question_tokens.dim() != _TWO:
            raise ValueError("question_tokens must be [num_graphs, hidden_dim].")
        node_batch = node_batch.to(device=node_tokens.device, dtype=torch.long).view(-1)
        context = question_tokens.index_select(0, node_batch)
        fused = torch.cat((node_tokens, context), dim=-1)
        return self.net(fused).squeeze(-1)

    def set_output_bias(self, bias: float) -> None:
        last_linear = None
        for layer in reversed(self.net):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break
        if last_linear is None or last_linear.bias is None:
            raise RuntimeError("LogZPredictor missing output bias for initialization.")
        with torch.no_grad():
            last_linear.bias.fill_(float(bias))


__all__ = [
    "EmbeddingBackbone",
    "CvtNodeInitializer",
    "RelationalGNNLayer",
    "LogZPredictor",
    "SinusoidalPositionalEncoding",
]
