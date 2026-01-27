from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_LOGZ_OUTPUT_DIM = 1
_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_GNN_LAYERS = 2
_DEFAULT_GNN_DROPOUT = 0.0
_DEFAULT_ADAPTER_ENABLED = False
_DEFAULT_ADAPTER_DIM_DIVISOR = 4
_DEFAULT_ADAPTER_DROPOUT = 0.1


def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


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
        self.update_proj = nn.Linear(self.hidden_dim * _TWO, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        _init_linear(self.msg_proj)
        _init_linear(self.update_proj)

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
        agg = torch.zeros((num_nodes, self.hidden_dim), device=node_tokens.device, dtype=node_tokens.dtype)
        agg.index_add_(0, tail, msg)
        deg = torch.zeros((num_nodes,), device=node_tokens.device, dtype=node_tokens.dtype)
        ones = torch.ones_like(tail, dtype=node_tokens.dtype)
        deg.index_add_(0, tail, ones)
        agg = agg / deg.clamp(min=_ONE).unsqueeze(-1)
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
        return embeddings + delta


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
        return self.node_proj(node_normed)

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
]
