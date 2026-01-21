from __future__ import annotations

from typing import Any, Optional

import math

import torch
from torch import nn
from torch_scatter.composite import scatter_softmax

_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_FLOW_OUTPUT_DIM = 1
_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_AGENT_DROPOUT = 0.0
_CONTEXT_QUESTION = "question"
_CONTEXT_START_NODE = "start_node"
_CONTEXT_MODES = {_CONTEXT_QUESTION, _CONTEXT_START_NODE}
_DEFAULT_CONTEXT_MODE = _CONTEXT_QUESTION


def _init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _init_vector(param: nn.Parameter, *, dim: int) -> None:
    std = float(_ONE) / math.sqrt(float(dim))
    nn.init.normal_(param, mean=float(_ZERO), std=std)


def _maybe_force_fp32(tensor: torch.Tensor, *, force_fp32: bool) -> torch.Tensor:
    if force_fp32 and tensor.dtype != torch.float32:
        return tensor.float()
    return tensor


class EntrySelector(nn.Module):
    """Bilinear entry selector for start/target sampling."""

    def __init__(
        self,
        *,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if hidden_dim <= _ZERO:
            raise ValueError("hidden_dim must be positive for EntrySelector.")
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
        if query_tokens.dim() != _TWO or candidate_tokens.dim() != _TWO:
            raise ValueError("query_tokens and candidate_tokens must be [*, H] for EntrySelector.")
        if candidate_batch.dim() != _ONE:
            raise ValueError("candidate_batch must be [N] for EntrySelector.")
        if candidate_tokens.size(_ZERO) != candidate_batch.numel():
            raise ValueError("candidate_tokens length mismatch with candidate_batch.")
        if query_tokens.size(_ONE) != candidate_tokens.size(_ONE):
            raise ValueError("query_tokens dim mismatch with candidate_tokens.")
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
        force_fp32: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.finetune = bool(finetune)
        self.force_fp32 = bool(force_fp32)

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
        question_emb = _maybe_force_fp32(question_emb, force_fp32=self.force_fp32)
        node_embeddings = _maybe_force_fp32(node_embeddings, force_fp32=self.force_fp32)
        edge_embeddings = _maybe_force_fp32(edge_embeddings, force_fp32=self.force_fp32)
        question_tokens = self.q_proj(question_emb)
        node_tokens = self.node_proj(self.node_norm(node_embeddings))
        relation_tokens = self.rel_proj(self.rel_norm(edge_embeddings))
        return node_tokens, relation_tokens, question_tokens


class CvtNodeInitializer(nn.Module):
    """Shared CVT embedding updated by incoming relation context."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        force_fp32: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.force_fp32 = bool(force_fp32)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive for NonTextNodeInitializer.")
        self.shared_cvt = nn.Parameter(torch.zeros(self.hidden_dim))
        self.attn_vector = nn.Parameter(torch.zeros(self.hidden_dim))
        self.msg_proj = nn.Linear(self.hidden_dim * int(_TWO), self.hidden_dim, bias=False)
        _init_linear(self.msg_proj)
        _init_vector(self.shared_cvt, dim=self.hidden_dim)
        _init_vector(self.attn_vector, dim=self.hidden_dim)

    def _aggregate_incoming(
        self,
        *,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        cvt_mask: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        heads = edge_index[_ZERO]
        tails = edge_index[_ONE]
        cvt_tail_mask = cvt_mask.index_select(0, tails)
        if not bool(cvt_tail_mask.any().detach().tolist()):
            return torch.zeros((num_nodes, relation_tokens.size(-1)), device=relation_tokens.device, dtype=relation_tokens.dtype)
        slot_indices = cvt_tail_mask.nonzero(as_tuple=False).view(-1)
        slot_rel = relation_tokens.index_select(0, slot_indices)
        slot_heads = heads.index_select(0, slot_indices)
        slot_nbr = node_tokens.index_select(0, slot_heads)
        slot_nodes = tails.index_select(0, slot_indices)
        msg = self.msg_proj(torch.cat((slot_rel, slot_nbr), dim=-1))
        logits = (msg * self.attn_vector.to(device=msg.device, dtype=msg.dtype)).sum(dim=-1)
        attn_weights = scatter_softmax(logits, slot_nodes, dim=0).to(dtype=msg.dtype)
        weighted = msg * attn_weights.unsqueeze(-1)
        agg = torch.zeros((num_nodes, self.hidden_dim), device=msg.device, dtype=msg.dtype)
        agg.index_add_(0, slot_nodes, weighted)
        return agg

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        node_is_cvt: torch.Tensor,
    ) -> torch.Tensor:
        node_tokens = _maybe_force_fp32(node_tokens, force_fp32=self.force_fp32)
        relation_tokens = _maybe_force_fp32(relation_tokens, force_fp32=self.force_fp32)
        if node_tokens.dim() != 2:
            raise ValueError("node_tokens must be [N, H] for NonTextNodeInitializer.")
        if node_is_cvt.dim() != 1:
            raise ValueError("node_is_cvt must be 1D for NonTextNodeInitializer.")
        if node_is_cvt.numel() != node_tokens.size(0):
            raise ValueError("node_is_cvt length mismatch with node_tokens.")
        cvt_mask = node_is_cvt.to(dtype=torch.bool, device=node_tokens.device)
        if not bool(cvt_mask.any().detach().tolist()):
            return node_tokens
        num_nodes = int(node_tokens.size(0))
        rel_ctx = self._aggregate_incoming(
            relation_tokens=relation_tokens,
            node_tokens=node_tokens,
            edge_index=edge_index,
            cvt_mask=cvt_mask,
            num_nodes=num_nodes,
        )
        shared = self.shared_cvt.to(device=node_tokens.device, dtype=node_tokens.dtype)
        cvt_tokens = rel_ctx + shared
        return torch.where(cvt_mask.unsqueeze(-1), cvt_tokens, node_tokens)


class TrajectoryAgent(nn.Module):
    """RNN-based trajectory agent with query-style action scoring."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        dropout: float = _DEFAULT_AGENT_DROPOUT,
        force_fp32: bool = False,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.force_fp32 = bool(force_fp32)
        if self.token_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("token_dim/hidden_dim must be positive for TrajectoryAgent.")
        self.action_dim = self.hidden_dim * int(_TWO)
        self.rnn = nn.GRU(
            input_size=self.token_dim * int(_TWO),
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.query_proj = nn.Linear(self.hidden_dim, self.action_dim, bias=False)
        self.rel_key_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.node_key_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.context_adapter_question = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim, bias=False),
        )
        self.context_adapter_node = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim, bias=False),
        )
        self.context_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.context_out_norm = nn.LayerNorm(self.hidden_dim)
        _init_linear(self.query_proj)
        _init_linear(self.rel_key_proj)
        _init_linear(self.node_key_proj)
        for adapter in (self.context_adapter_question, self.context_adapter_node):
            for layer in adapter:
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

    def initialize_state(self, condition_tokens: torch.Tensor, *, context_mode: Optional[str] = None) -> torch.Tensor:
        condition_tokens = _maybe_force_fp32(condition_tokens, force_fp32=self.force_fp32)
        if condition_tokens.dim() != _TWO:
            raise ValueError("condition_tokens must be [B, D] for TrajectoryAgent.")
        mode = str(context_mode or _DEFAULT_CONTEXT_MODE).strip().lower()
        if mode not in _CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {sorted(_CONTEXT_MODES)}, got {mode!r}.")
        if mode == _CONTEXT_QUESTION:
            adapted = self.context_adapter_question(condition_tokens)
        else:
            adapted = self.context_adapter_node(condition_tokens)
        hidden = self.context_proj(adapted)
        hidden = self.dropout_layer(self.context_out_norm(hidden))
        return hidden

    def step(
        self,
        *,
        hidden: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
    ) -> torch.Tensor:
        hidden = _maybe_force_fp32(hidden, force_fp32=self.force_fp32)
        relation_tokens = _maybe_force_fp32(relation_tokens, force_fp32=self.force_fp32)
        node_tokens = _maybe_force_fp32(node_tokens, force_fp32=self.force_fp32)
        if hidden.dim() != _TWO or relation_tokens.dim() != _TWO or node_tokens.dim() != _TWO:
            raise ValueError("hidden/relation_tokens/node_tokens must be [B, D] for TrajectoryAgent.step.")
        if relation_tokens.size(0) != hidden.size(0) or node_tokens.size(0) != hidden.size(0):
            raise ValueError("TrajectoryAgent.step input batch mismatch.")
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
    ) -> torch.Tensor:
        relation_tokens = _maybe_force_fp32(relation_tokens, force_fp32=self.force_fp32)
        node_tokens = _maybe_force_fp32(node_tokens, force_fp32=self.force_fp32)
        if (
            relation_tokens.dim() != _TWO
            or node_tokens.dim() != _TWO
            or edge_index.dim() != _TWO
            or edge_index.size(0) != _TWO
        ):
            raise ValueError(
                "relation_tokens/node_tokens must be [*, H] and edge_index must be [2, E] for precompute_action_keys."
            )
        edge_index = edge_index.to(device=relation_tokens.device, dtype=torch.long)
        num_edges = int(edge_index.size(1))
        if relation_tokens.size(0) != num_edges:
            raise ValueError("edge_index length mismatch with relation_tokens.")
        tail_nodes = edge_index[_ONE]
        tail_tokens = node_tokens.index_select(0, tail_nodes)
        rel_key = self.dropout_layer(self.rel_key_proj(relation_tokens))
        node_key = self.dropout_layer(self.node_key_proj(tail_tokens))
        action_key = torch.cat((rel_key, node_key), dim=-1)
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
        hidden = _maybe_force_fp32(hidden, force_fp32=self.force_fp32)
        if hidden.dim() != _TWO or action_keys.dim() != _TWO:
            raise ValueError("hidden/action_keys must be [*, H] for TrajectoryAgent.score_cached.")
        edge_batch = edge_batch.to(device=hidden.device, dtype=torch.long).view(-1)
        if edge_batch.numel() == _ZERO:
            return torch.zeros((0,), device=hidden.device, dtype=hidden.dtype)
        if edge_ids is not None:
            edge_ids = edge_ids.to(device=hidden.device, dtype=torch.long).view(-1)
            if edge_ids.numel() != edge_batch.numel():
                raise ValueError("edge_ids length mismatch with edge_batch.")
        if valid_edges_mask is not None:
            valid_edges_mask = valid_edges_mask.to(device=hidden.device, dtype=torch.bool).view(-1)
            if valid_edges_mask.numel() != edge_batch.numel():
                raise ValueError("valid_edges_mask length mismatch with edge_batch.")
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
        hidden = _maybe_force_fp32(hidden, force_fp32=self.force_fp32)
        relation_tokens = _maybe_force_fp32(relation_tokens, force_fp32=self.force_fp32)
        node_tokens = _maybe_force_fp32(node_tokens, force_fp32=self.force_fp32)
        if hidden.dim() != _TWO:
            raise ValueError("hidden must be [B, D] for TrajectoryAgent.encode_state_sequence.")
        if relation_tokens.dim() != _THREE or node_tokens.dim() != _THREE:
            raise ValueError("relation_tokens/node_tokens must be [B, T, D] for TrajectoryAgent.encode_state_sequence.")
        if relation_tokens.shape != node_tokens.shape:
            raise ValueError("relation_tokens/node_tokens shape mismatch in TrajectoryAgent.encode_state_sequence.")
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
        if node_tokens.dim() != 2 or question_tokens.dim() != 2:
            raise ValueError("node_tokens and question_tokens must be [*, H] for FlowPredictor.")
        if graph_features.dim() != 2:
            raise ValueError("graph_features must be [B, F] for FlowPredictor.")
        if node_batch.dim() != 1:
            raise ValueError("node_batch must be [N] for FlowPredictor.")
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
