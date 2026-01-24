from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_FLOW_OUTPUT_DIM = 1
_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_AGENT_DROPOUT = 0.0
_CONTEXT_FUSION_DIM_MULT = _TWO


def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class EntrySelector(nn.Module):
    """Bilinear entry selector for start/target sampling."""

    def __init__(
        self,
        *,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        _init_linear(self.query_proj)
        _init_linear(self.key_proj)

    def score(
        self,
        *,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_batch: torch.LongTensor,
    ) -> torch.Tensor:
        query_proj = self.query_proj(query_tokens)
        key_proj = self.key_proj(candidate_tokens)
        query_sel = query_proj.index_select(_ZERO, candidate_batch.to(device=query_proj.device))
        return torch.einsum("nh,nh->n", query_sel, key_proj)


class EmbeddingBackbone(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        finetune: bool = _DEFAULT_BACKBONE_FINETUNE,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.finetune = bool(finetune)

        self.node_norm = nn.LayerNorm(self.emb_dim)
        self.rel_norm = nn.LayerNorm(self.emb_dim)
        self.node_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.rel_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.q_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        _init_linear(self.node_proj)
        _init_linear(self.rel_proj)
        _init_linear(self.q_proj)
        if not self.finetune:
            for param in self.parameters():
                param.requires_grad = False

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
        node_normed = self.node_norm(node_embeddings)
        return self.node_proj(node_normed)

    def project_relation_embeddings(self, relation_embeddings: torch.Tensor) -> torch.Tensor:
        rel_normed = self.rel_norm(relation_embeddings)
        return self.rel_proj(rel_normed)

    def project_question_embeddings(self, question_emb: torch.Tensor) -> torch.Tensor:
        return self.q_proj(question_emb)


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
        counts_safe = counts.clamp(min=float(_ONE))
        mean = sums / counts_safe.unsqueeze(-1)
        has_in = counts > float(_ZERO)
        use_mask = cvt_mask & has_in.to(dtype=torch.bool, device=cvt_mask.device)
        return torch.where(use_mask.unsqueeze(-1), mean, node_embeddings)


class TrajectoryAgent(nn.Module):
    """RNN-based trajectory agent with query-style action scoring."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        dropout: float = _DEFAULT_AGENT_DROPOUT,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.action_dim = self.hidden_dim * int(_TWO)
        self.rnn = nn.GRU(
            input_size=self.token_dim * int(_TWO),
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.query_proj = nn.Linear(self.hidden_dim, self.action_dim, bias=False)
        self.rel_key_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.node_key_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.q_key_proj = nn.Linear(self.token_dim, self.action_dim, bias=False)
        self.context_adapter_question = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim, bias=False),
        )
        self.context_adapter_node = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim, bias=False),
        )
        self.context_fusion = nn.Sequential(
            nn.LayerNorm(self.token_dim * int(_CONTEXT_FUSION_DIM_MULT)),
            nn.Linear(self.token_dim * int(_CONTEXT_FUSION_DIM_MULT), self.token_dim, bias=False),
            nn.GELU(),
        )
        self.context_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.context_out_norm = nn.LayerNorm(self.hidden_dim)
        _init_linear(self.query_proj)
        _init_linear(self.rel_key_proj)
        _init_linear(self.node_key_proj)
        _init_linear(self.q_key_proj)
        for adapter in (self.context_adapter_question, self.context_adapter_node):
            for layer in adapter:
                if isinstance(layer, nn.Linear):
                    _init_linear(layer)
        for layer in self.context_fusion:
            if isinstance(layer, nn.Linear):
                _init_linear(layer)
        _init_linear(self.context_proj)
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_dim, self.action_dim),
            nn.GELU(),
            nn.Linear(self.action_dim, self.action_dim),
        )
        for layer in self.action_mlp:
            if isinstance(layer, nn.Linear):
                _init_linear(layer)
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()

    def initialize_state(self, *, question_tokens: torch.Tensor, node_tokens: torch.Tensor) -> torch.Tensor:
        if question_tokens.shape != node_tokens.shape:
            raise ValueError("question_tokens/node_tokens shape mismatch for TrajectoryAgent.")
        adapted_q = self.context_adapter_question(question_tokens)
        adapted_n = self.context_adapter_node(node_tokens)
        fused = torch.cat((adapted_q, adapted_n), dim=-1)
        fused = self.context_fusion(fused)
        hidden = self.context_proj(fused)
        hidden = self.dropout_layer(self.context_out_norm(hidden))
        return hidden

    def step(
        self,
        *,
        hidden: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
    ) -> torch.Tensor:
        step_input = torch.cat((relation_tokens, node_tokens), dim=-1).unsqueeze(_ONE)
        out, h_next = self.rnn(step_input, hidden.unsqueeze(_ZERO))
        h_next = h_next.squeeze(_ZERO)
        h_next = self.dropout_layer(h_next)
        if h_next.dtype != hidden.dtype:
            h_next = h_next.to(dtype=hidden.dtype)
        return h_next

    def precompute_action_keys(
        self,
        *,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = edge_index.to(device=relation_tokens.device, dtype=torch.long)
        num_edges = int(edge_index.size(1))
        tail_nodes = edge_index[_ONE]
        tail_tokens = node_tokens.index_select(0, tail_nodes)
        q_tokens = question_tokens.to(device=relation_tokens.device, dtype=relation_tokens.dtype)
        if q_tokens.dim() == 3 and q_tokens.size(1) == _ONE:
            q_tokens = q_tokens.squeeze(1)
        if q_tokens.dim() != 2:
            raise ValueError("question_tokens must be [B, H] or [B, 1, H] for action keys.")
        edge_batch = edge_batch.to(device=relation_tokens.device, dtype=torch.long).view(-1)
        if edge_batch.numel() != num_edges:
            raise ValueError("edge_batch length mismatch with edge_index for action keys.")
        q_tokens = q_tokens.index_select(0, edge_batch)
        rel_key = self.dropout_layer(self.rel_key_proj(relation_tokens))
        node_key = self.dropout_layer(self.node_key_proj(tail_tokens))
        q_gate = torch.sigmoid(self.q_key_proj(q_tokens))
        action_key = torch.cat((rel_key, node_key), dim=-1)
        action_key = action_key * q_gate
        return self.action_mlp(action_key)

    def score_cached(
        self,
        *,
        hidden: torch.Tensor,
        action_keys: torch.Tensor,
        edge_batch: torch.Tensor,
        valid_edges_mask: Optional[torch.Tensor] = None,
        edge_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_batch = edge_batch.to(device=hidden.device, dtype=torch.long).view(-1)
        if edge_batch.numel() == _ZERO:
            return torch.zeros((0,), device=hidden.device, dtype=hidden.dtype)
        if edge_ids is not None:
            edge_ids = edge_ids.to(device=hidden.device, dtype=torch.long).view(-1)
        if valid_edges_mask is not None:
            valid_edges_mask = valid_edges_mask.to(device=hidden.device, dtype=torch.bool).view(-1)
        query = self.dropout_layer(self.query_proj(hidden))
        if valid_edges_mask is None:
            q_expanded = query.index_select(0, edge_batch)
            if edge_ids is None:
                keys = action_keys
            else:
                keys = action_keys.index_select(0, edge_ids)
            return (keys * q_expanded).sum(dim=-1)
        q_expanded = query.index_select(0, edge_batch)
        if edge_ids is None:
            keys = action_keys
        else:
            keys = action_keys.index_select(0, edge_ids)
        scores = (keys * q_expanded).sum(dim=-1)
        neg_inf = torch.finfo(hidden.dtype).min
        return torch.where(valid_edges_mask, scores, torch.full_like(scores, neg_inf))

    def encode_state_sequence(
        self,
        *,
        hidden: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((relation_tokens, node_tokens), dim=-1)
        if action_mask is not None:
            mask = action_mask.to(device=inputs.device, dtype=inputs.dtype).unsqueeze(-1)
            inputs = inputs * mask
        output, _ = self.rnn(inputs, hidden.unsqueeze(_ZERO))
        if output.size(1) <= _ZERO:
            empty = torch.zeros(
                (hidden.size(0), _ZERO, hidden.size(1)),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            return empty, empty
        if output.size(1) == _ONE:
            state_seq = hidden.unsqueeze(_ONE)
        else:
            state_seq = torch.cat((hidden.unsqueeze(_ONE), output[:, :-_ONE, :]), dim=_ONE)
        state_seq = self.dropout_layer(state_seq)
        output = self.dropout_layer(output)
        return state_seq, output


class FlowPredictor(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        input_dim = self.hidden_dim + self.hidden_dim + self.feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, _FLOW_OUTPUT_DIM),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_linear(layer)

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        graph_features: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        q_tokens = question_tokens.index_select(0, node_batch)
        g_tokens = graph_features.index_select(0, node_batch)
        context = torch.cat((q_tokens, node_tokens, g_tokens), dim=-1)
        return self.net(context).squeeze(-1)

    def set_output_bias(self, bias: float) -> None:
        last_linear = None
        for layer in reversed(self.net):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break
        if last_linear is None or last_linear.bias is None:
            raise RuntimeError("FlowPredictor missing output bias for initialization.")
        with torch.no_grad():
            last_linear.bias.fill_(float(bias))


__all__ = [
    "EntrySelector",
    "EmbeddingBackbone",
    "CvtNodeInitializer",
    "FlowPredictor",
]
