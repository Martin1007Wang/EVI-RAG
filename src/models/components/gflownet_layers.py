from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch_scatter.composite import scatter_softmax

_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_FLOW_OUTPUT_DIM = 1
_EMBED_INIT_STD_POWER = 0.25
_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_AGENT_DROPOUT = 0.0
_SCORE_BLOCK_SIZE = 8192


def _init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _init_embedding_linear(layer: nn.Linear, *, out_dim: int) -> None:
    std = float(out_dim) ** (-_EMBED_INIT_STD_POWER)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


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
        _init_embedding_linear(self.node_proj, out_dim=self.hidden_dim)
        _init_linear(self.rel_proj)
        _init_linear(self.q_proj)
        if not self.finetune:
            for param in self.node_proj.parameters():
                param.requires_grad = False
            for param in self.rel_proj.parameters():
                param.requires_grad = False
            for param in self.q_proj.parameters():
                param.requires_grad = False

    def forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        question_emb = batch.question_emb.to(device=device, non_blocking=True)
        node_embeddings = batch.node_embeddings.to(device=device, non_blocking=True)
        edge_embeddings = batch.edge_embeddings.to(device=device, non_blocking=True)
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
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive for NonTextNodeInitializer.")
        self.shared_cvt = nn.Parameter(torch.zeros(self.hidden_dim))
        self.attn_vector = nn.Parameter(torch.zeros(self.hidden_dim))
        self.msg_proj = nn.Linear(self.hidden_dim * int(_TWO), self.hidden_dim, bias=False)
        _init_linear(self.msg_proj)

    def _aggregate_incoming(
        self,
        *,
        relation_tokens: torch.Tensor,
        nbr_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        cvt_mask: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        tails = edge_index[_ONE]
        cvt_tail_mask = cvt_mask.index_select(0, tails)
        if not bool(cvt_tail_mask.any().detach().tolist()):
            return torch.zeros((num_nodes, relation_tokens.size(-1)), device=relation_tokens.device, dtype=relation_tokens.dtype)
        slot_indices = cvt_tail_mask.nonzero(as_tuple=False).view(-1)
        slot_rel = relation_tokens.index_select(0, slot_indices)
        slot_nbr = nbr_tokens.index_select(0, slot_indices)
        slot_nodes = tails.index_select(0, slot_indices)
        msg = self.msg_proj(torch.cat((slot_rel, slot_nbr), dim=-1))
        logits = (msg * self.attn_vector.to(device=msg.device, dtype=msg.dtype)).sum(dim=-1)
        attn_weights = scatter_softmax(logits, slot_nodes, dim=0)
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
            nbr_tokens=node_tokens,
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
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        if self.token_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("token_dim/hidden_dim must be positive for TrajectoryAgent.")
        self.action_dim = self.hidden_dim * int(_THREE)
        self.rnn = nn.GRU(
            input_size=self.token_dim * int(_TWO),
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.query_proj = nn.Linear(self.hidden_dim, self.action_dim, bias=False)
        self.rel_key_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.node_key_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.head_key_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.context_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.init_proj = nn.Linear(self.hidden_dim * int(_TWO), self.hidden_dim, bias=False)
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        self.start_norm = nn.LayerNorm(self.hidden_dim)
        _init_linear(self.query_proj)
        _init_linear(self.rel_key_proj)
        _init_linear(self.node_key_proj)
        _init_linear(self.head_key_proj)
        _init_linear(self.context_proj)
        _init_linear(self.init_proj)
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_dim, self.action_dim),
            nn.GELU(),
            nn.Linear(self.action_dim, self.action_dim),
        )
        for layer in self.action_mlp:
            if isinstance(layer, nn.Linear):
                _init_linear(layer)
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()

    def initialize_state(self, context_tokens: torch.Tensor, start_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context_tokens.dim() != _TWO:
            raise ValueError("context_tokens must be [B, D] for TrajectoryAgent.")
        context_vec = self.context_proj(context_tokens)
        if start_nodes is not None:
            start_nodes = start_nodes.to(device=context_tokens.device, dtype=context_tokens.dtype)
            start_vec = self.start_norm(start_nodes)
        else:
            start_vec = torch.zeros_like(context_vec)
        fused = torch.cat((self.context_norm(context_vec), start_vec), dim=-1)
        hidden = self.dropout_layer(self.init_proj(fused))
        return hidden

    def step(
        self,
        *,
        hidden: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.dim() != _TWO or relation_tokens.dim() != _TWO or node_tokens.dim() != _TWO:
            raise ValueError("hidden/relation_tokens/node_tokens must be [B, D] for TrajectoryAgent.step.")
        if relation_tokens.size(0) != hidden.size(0) or node_tokens.size(0) != hidden.size(0):
            raise ValueError("TrajectoryAgent.step input batch mismatch.")
        step_input = torch.cat((relation_tokens, node_tokens), dim=-1).unsqueeze(_ONE)
        out, h_next = self.rnn(step_input, hidden.unsqueeze(_ZERO))
        h_next = h_next.squeeze(_ZERO)
        h_next = self.dropout_layer(h_next)
        return h_next

    def score(
        self,
        *,
        hidden: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        head_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hidden.dim() != _TWO
            or relation_tokens.dim() != _TWO
            or node_tokens.dim() != _TWO
            or head_tokens.dim() != _TWO
            or edge_ptr.dim() != _ONE
        ):
            raise ValueError(
                "hidden/relation_tokens/node_tokens/head_tokens must be [*, H] and edge_ptr must be 1D for TrajectoryAgent.score."
            )
        edge_batch = edge_batch.to(device=hidden.device, dtype=torch.long).view(-1)
        if relation_tokens.size(0) != edge_batch.numel() or node_tokens.size(0) != edge_batch.numel() or head_tokens.size(0) != edge_batch.numel():
            raise ValueError("edge_batch length mismatch with relation_tokens/node_tokens/head_tokens.")
        edge_ptr = edge_ptr.to(device=hidden.device, dtype=torch.long).view(-1)
        query = self.dropout_layer(self.query_proj(hidden))
        scores = torch.empty((relation_tokens.size(0),), device=relation_tokens.device, dtype=relation_tokens.dtype)
        num_edges = int(relation_tokens.size(0))
        block = int(_SCORE_BLOCK_SIZE)
        for start in range(0, num_edges, block):
            end = min(num_edges, start + block)
            rel_key_blk = self.dropout_layer(self.rel_key_proj(relation_tokens[start:end]))
            node_key_blk = self.dropout_layer(self.node_key_proj(node_tokens[start:end]))
            head_key_blk = self.dropout_layer(self.head_key_proj(head_tokens[start:end]))
            action_key_blk = torch.cat((rel_key_blk, node_key_blk, head_key_blk), dim=-1)
            action_key_blk = self.action_mlp(action_key_blk)
            q_blk = query.index_select(0, edge_batch[start:end])
            scores[start:end] = (action_key_blk * q_blk).sum(dim=-1)
        return scores

    def encode_state_sequence(
        self,
        *,
        hidden: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return state_seq, output


class StartSelector(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        if self.token_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("token_dim/hidden_dim must be positive for StartSelector.")
        self.node_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        self.q_proj = nn.Linear(self.token_dim, self.hidden_dim, bias=False)
        _init_linear(self.node_proj)
        _init_linear(self.q_proj)
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if node_tokens.dim() != 2 or question_tokens.dim() != 2:
            raise ValueError("node_tokens/question_tokens must be [N, H] for StartSelector.")
        if node_tokens.size(0) != question_tokens.size(0):
            raise ValueError("node_tokens/question_tokens length mismatch in StartSelector.")
        node_ctx = self.dropout_layer(self.node_proj(node_tokens))
        q_ctx = self.dropout_layer(self.q_proj(question_tokens))
        scores = (node_ctx * q_ctx).sum(dim=-1)
        return scores


class SinkSelector(StartSelector):
    pass


class FlowPredictor(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        input_dim = self.hidden_dim + self.hidden_dim + self.feature_dim + self.state_dim
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
        state_vec: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if node_tokens.dim() != 2 or question_tokens.dim() != 2:
            raise ValueError("node_tokens and question_tokens must be [*, H] for FlowPredictor.")
        if graph_features.dim() != 2:
            raise ValueError("graph_features must be [B, F] for FlowPredictor.")
        if state_vec.dim() != 2:
            raise ValueError("state_vec must be [*, S] for FlowPredictor.")
        if node_batch.dim() != 1:
            raise ValueError("node_batch must be [N] for FlowPredictor.")
        if state_vec.size(0) != node_tokens.size(0):
            raise ValueError("state_vec length mismatch with node_tokens.")
        q_tokens = question_tokens.index_select(0, node_batch)
        g_tokens = graph_features.index_select(0, node_batch)
        context = torch.cat((q_tokens, node_tokens, g_tokens, state_vec), dim=-1)
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
    "EmbeddingBackbone",
    "CvtNodeInitializer",
    "StartSelector",
    "SinkSelector",
    "FlowPredictor",
]
