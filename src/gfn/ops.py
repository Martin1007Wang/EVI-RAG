from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from src.utils.graph import compute_edge_batch, compute_undirected_degree, directed_bfs_distances

_GUMBEL_EPS = 1e-10
_ZERO = 0
_ONE = 1
_TWO = 2
_NEG_ONE = -1
_PTR_MIN_LEN = 2
_CORRIDOR_ENABLED_KEY = "enabled"
_CORRIDOR_DISTANCE_BIAS_ALPHA_KEY = "distance_bias_alpha"
_DEFAULT_CORRIDOR_ENABLED = False
_DEFAULT_CORRIDOR_DISTANCE_BIAS_ALPHA = 0.0


@dataclass(frozen=True)
class CorridorSpec:
    enabled: bool
    distance_bias_alpha: float


def _resolve_corridor_spec(cfg: Optional[Mapping[str, Any]]) -> CorridorSpec:
    if cfg is None:
        return CorridorSpec(
            enabled=_DEFAULT_CORRIDOR_ENABLED,
            distance_bias_alpha=_DEFAULT_CORRIDOR_DISTANCE_BIAS_ALPHA,
        )
    if isinstance(cfg, bool):
        return CorridorSpec(enabled=bool(cfg), distance_bias_alpha=_DEFAULT_CORRIDOR_DISTANCE_BIAS_ALPHA)
    if not isinstance(cfg, Mapping):
        raise TypeError("corridor_cfg must be a mapping or bool.")
    enabled = bool(cfg.get(_CORRIDOR_ENABLED_KEY, _DEFAULT_CORRIDOR_ENABLED))
    alpha_raw = cfg.get(_CORRIDOR_DISTANCE_BIAS_ALPHA_KEY, _DEFAULT_CORRIDOR_DISTANCE_BIAS_ALPHA)
    if isinstance(alpha_raw, bool):
        raise TypeError("corridor_cfg.distance_bias_alpha must be a float, got bool.")
    alpha = float(alpha_raw)
    if alpha < float(_ZERO):
        raise ValueError("corridor_cfg.distance_bias_alpha must be >= 0.")
    return CorridorSpec(enabled=enabled, distance_bias_alpha=alpha)


def neg_inf_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


def segment_logsumexp_1d(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    if logits.dim() != 1 or segment_ids.dim() != 1:
        raise ValueError("logits and segment_ids must be 1D tensors.")
    if logits.numel() != segment_ids.numel():
        raise ValueError("logits and segment_ids must have the same length.")
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")
    if logits.numel() == 0:
        return torch.full((num_segments,), neg_inf_value(logits), device=logits.device, dtype=logits.dtype)

    device = logits.device
    calc_dtype = logits.dtype
    if calc_dtype in (torch.float16, torch.bfloat16):
        calc_dtype = torch.float32
    logits = logits.to(dtype=calc_dtype)
    neg_inf = torch.finfo(calc_dtype).min
    max_per = torch.full((num_segments,), neg_inf, device=device, dtype=calc_dtype)
    max_per.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    shifted = logits - max_per[segment_ids]
    exp = torch.exp(shifted)
    sum_per = torch.zeros((num_segments,), device=device, dtype=calc_dtype)
    sum_per.index_add_(0, segment_ids, exp)
    eps = torch.finfo(calc_dtype).eps
    return torch.log(sum_per.clamp(min=eps)) + max_per


def compute_policy_log_probs(
    *,
    edge_logits: torch.Tensor,
    stop_logits: Optional[torch.Tensor],
    edge_batch: torch.Tensor,
    valid_edges: torch.Tensor,
    num_graphs: int,
    temperature: float,
    allow_stop: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    if num_graphs <= 0:
        device = edge_logits.device
        zeros = torch.zeros(0, device=device)
        has_edge = torch.zeros(0, device=device, dtype=torch.bool)
        if stop_logits is None:
            return zeros, None, zeros, has_edge
        return zeros, zeros, zeros, has_edge
    if edge_logits.dtype != torch.float32:
        edge_logits = edge_logits.to(dtype=torch.float32)
    if stop_logits is not None and stop_logits.dtype != torch.float32:
        stop_logits = stop_logits.to(dtype=torch.float32)

    stop_scaled = None
    if stop_logits is not None:
        stop_scaled = stop_logits / float(temperature)
    if edge_logits.numel() == 0:
        log_denom = edge_logits.new_full((num_graphs,), neg_inf_value(edge_logits))
        has_edge = torch.zeros(num_graphs, device=edge_logits.device, dtype=torch.bool)
        if stop_scaled is None:
            return edge_logits, None, log_denom, has_edge
        log_denom = stop_scaled
        return edge_logits, stop_scaled - log_denom, log_denom, has_edge

    edge_scaled = edge_logits / float(temperature)
    valid_edges = valid_edges.to(device=edge_scaled.device, dtype=torch.bool)
    logsumexp_edges = segment_logsumexp_1d(edge_scaled[valid_edges], edge_batch[valid_edges], num_graphs)
    has_edge = logsumexp_edges > neg_inf_value(logsumexp_edges)
    if stop_scaled is None:
        if allow_stop is not None:
            raise ValueError("allow_stop requires stop_logits.")
        log_denom = logsumexp_edges
        log_prob_edge = edge_scaled - log_denom[edge_batch]
        log_prob_edge = torch.where(
            valid_edges,
            log_prob_edge,
            torch.full_like(log_prob_edge, neg_inf_value(log_prob_edge)),
        )
        return log_prob_edge, None, log_denom, has_edge
    if allow_stop is None:
        allow_stop_mask = torch.ones(num_graphs, device=stop_logits.device, dtype=torch.bool)
    else:
        allow_stop_mask = allow_stop.to(device=stop_logits.device, dtype=torch.bool).view(-1)
        if allow_stop_mask.numel() != num_graphs:
            raise ValueError("allow_stop length mismatch with batch size.")
    allow_stop_mask = allow_stop_mask | (~has_edge)
    log_denom = torch.where(
        allow_stop_mask,
        torch.logaddexp(logsumexp_edges, stop_scaled),
        logsumexp_edges,
    )
    log_prob_edge = edge_scaled - log_denom[edge_batch]
    log_prob_edge = torch.where(
        valid_edges,
        log_prob_edge,
        torch.full_like(log_prob_edge, neg_inf_value(log_prob_edge)),
    )
    log_prob_stop = stop_scaled - log_denom
    log_prob_stop = torch.where(
        allow_stop_mask,
        log_prob_stop,
        torch.full_like(log_prob_stop, neg_inf_value(log_prob_stop)),
    )
    return log_prob_edge, log_prob_stop, log_denom, has_edge


def _build_relation_mask(
    *,
    edge_batch: torch.Tensor,
    edge_relations: torch.Tensor,
    valid_edges: torch.Tensor,
    num_graphs: int,
    num_relations: int,
) -> torch.Tensor:
    if num_graphs <= _ZERO or num_relations <= _ZERO:
        return torch.zeros((num_graphs, num_relations), device=edge_batch.device, dtype=torch.bool)
    rel_mask_flat = torch.zeros(
        num_graphs * num_relations,
        device=edge_batch.device,
        dtype=torch.bool,
    )
    if valid_edges.any():
        rel_index = edge_batch[valid_edges] * num_relations + edge_relations[valid_edges]
        rel_mask_flat[rel_index] = True
    return rel_mask_flat.view(num_graphs, num_relations)


def _prepare_factorized_inputs(
    *,
    relation_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    edge_batch: torch.Tensor,
    edge_relations: torch.Tensor,
    valid_edges: torch.Tensor,
    stop_logits: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if relation_logits.dim() != _TWO:
        raise ValueError("relation_logits must be [B, R] for factorized policy.")
    if relation_logits.dtype != torch.float32:
        relation_logits = relation_logits.to(dtype=torch.float32)
    if edge_logits.dtype != torch.float32:
        edge_logits = edge_logits.to(dtype=torch.float32)
    if stop_logits is not None and stop_logits.dtype != torch.float32:
        stop_logits = stop_logits.to(dtype=torch.float32)
    edge_batch = edge_batch.to(device=relation_logits.device, dtype=torch.long).view(-1)
    edge_relations = edge_relations.to(device=relation_logits.device, dtype=torch.long).view(-1)
    valid_edges = valid_edges.to(device=relation_logits.device, dtype=torch.bool).view(-1)
    return relation_logits, edge_logits, stop_logits, edge_batch, edge_relations, valid_edges


def _resolve_empty_factorized(
    *,
    relation_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    stop_logits: Optional[torch.Tensor],
    num_graphs: int,
    temperature: float,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]]:
    if num_graphs <= _ZERO:
        device = edge_logits.device
        zeros = torch.zeros(0, device=device)
        has_edge = torch.zeros(0, device=device, dtype=torch.bool)
        if stop_logits is None:
            return zeros, None, zeros, has_edge
        return zeros, zeros, zeros, has_edge
    if edge_logits.numel() != _ZERO:
        return None
    log_denom = relation_logits.new_full((num_graphs,), neg_inf_value(relation_logits))
    has_edge = torch.zeros(num_graphs, device=relation_logits.device, dtype=torch.bool)
    if stop_logits is None:
        return edge_logits, None, log_denom, has_edge
    log_denom = stop_logits / float(temperature)
    log_prob_stop = log_denom - log_denom
    return edge_logits, log_prob_stop, log_denom, has_edge


def _compute_relation_log_probs(
    *,
    relation_logits: torch.Tensor,
    relation_mask: torch.Tensor,
    stop_logits: Optional[torch.Tensor],
    temperature: float,
    allow_stop: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    rel_scaled = relation_logits / float(temperature)
    neg_inf = neg_inf_value(rel_scaled)
    rel_mask = relation_mask.to(device=rel_scaled.device, dtype=torch.bool)
    rel_scaled = rel_scaled.masked_fill(~rel_mask, neg_inf)
    logsumexp_rel = torch.logsumexp(rel_scaled, dim=1)
    has_relation = logsumexp_rel > neg_inf
    if stop_logits is None:
        log_denom = logsumexp_rel
        log_prob_rel = rel_scaled - log_denom.unsqueeze(1)
        log_prob_rel = torch.where(rel_mask, log_prob_rel, torch.full_like(log_prob_rel, neg_inf))
        return log_prob_rel, None, log_denom, has_relation
    stop_scaled = stop_logits / float(temperature)
    if allow_stop is None:
        allow_stop_mask = torch.ones_like(stop_scaled, dtype=torch.bool)
    else:
        allow_stop_mask = allow_stop.to(device=stop_scaled.device, dtype=torch.bool).view(-1)
        if allow_stop_mask.numel() != stop_scaled.numel():
            raise ValueError("allow_stop length mismatch with batch size.")
    allow_stop_mask = allow_stop_mask | (~has_relation)
    log_denom = torch.where(
        allow_stop_mask,
        torch.logaddexp(logsumexp_rel, stop_scaled),
        logsumexp_rel,
    )
    log_prob_rel = rel_scaled - log_denom.unsqueeze(1)
    log_prob_rel = torch.where(rel_mask, log_prob_rel, torch.full_like(log_prob_rel, neg_inf))
    log_prob_stop = stop_scaled - log_denom
    log_prob_stop = torch.where(
        allow_stop_mask,
        log_prob_stop,
        torch.full_like(log_prob_stop, neg_inf),
    )
    return log_prob_rel, log_prob_stop, log_denom, has_relation


def _compute_edge_log_probs_given_relation(
    *,
    edge_logits: torch.Tensor,
    edge_batch: torch.Tensor,
    edge_relations: torch.Tensor,
    valid_edges: torch.Tensor,
    num_graphs: int,
    num_relations: int,
    temperature: float,
) -> torch.Tensor:
    if edge_logits.numel() == _ZERO:
        return edge_logits
    edge_scaled = edge_logits / float(temperature)
    valid_edges = valid_edges.to(device=edge_scaled.device, dtype=torch.bool)
    seg_ids = edge_batch * num_relations + edge_relations
    seg_ids = seg_ids.to(device=edge_scaled.device, dtype=torch.long)
    logsumexp_seg = segment_logsumexp_1d(
        edge_scaled[valid_edges],
        seg_ids[valid_edges],
        num_graphs * num_relations,
    )
    log_prob = edge_scaled - logsumexp_seg.index_select(0, seg_ids)
    neg_inf = neg_inf_value(edge_scaled)
    return torch.where(valid_edges, log_prob, torch.full_like(log_prob, neg_inf))


def _compute_factorized_nonempty(
    *,
    relation_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    edge_relations: torch.Tensor,
    edge_batch: torch.Tensor,
    valid_edges: torch.Tensor,
    num_graphs: int,
    temperature: float,
    stop_logits: Optional[torch.Tensor],
    allow_stop: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    num_relations = int(relation_logits.size(1))
    out_of_range = (edge_relations < _ZERO) | (edge_relations >= num_relations)
    if torch.any(out_of_range):
        raise ValueError("edge_relations contains ids outside relation_logits range.")
    relation_mask = _build_relation_mask(
        edge_batch=edge_batch,
        edge_relations=edge_relations,
        valid_edges=valid_edges,
        num_graphs=num_graphs,
        num_relations=num_relations,
    )
    log_prob_rel, log_prob_stop, log_denom, has_relation = _compute_relation_log_probs(
        relation_logits=relation_logits,
        relation_mask=relation_mask,
        stop_logits=stop_logits,
        temperature=temperature,
        allow_stop=allow_stop,
    )
    log_prob_edge_given_rel = _compute_edge_log_probs_given_relation(
        edge_logits=edge_logits,
        edge_batch=edge_batch,
        edge_relations=edge_relations,
        valid_edges=valid_edges,
        num_graphs=num_graphs,
        num_relations=num_relations,
        temperature=temperature,
    )
    rel_for_edge = log_prob_rel.index_select(0, edge_batch).gather(
        1,
        edge_relations.view(-1, 1),
    ).view(-1)
    neg_inf = neg_inf_value(log_prob_edge_given_rel)
    log_prob_edge = rel_for_edge + log_prob_edge_given_rel
    log_prob_edge = torch.where(valid_edges, log_prob_edge, torch.full_like(log_prob_edge, neg_inf))
    return log_prob_edge, log_prob_stop, log_denom, has_relation


def compute_factorized_log_probs(
    *,
    relation_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    edge_relations: torch.Tensor,
    edge_batch: torch.Tensor,
    valid_edges: torch.Tensor,
    num_graphs: int,
    temperature: float,
    stop_logits: Optional[torch.Tensor] = None,
    allow_stop: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    relation_logits, edge_logits, stop_logits, edge_batch, edge_relations, valid_edges = (
        _prepare_factorized_inputs(
            relation_logits=relation_logits,
            edge_logits=edge_logits,
            edge_batch=edge_batch,
            edge_relations=edge_relations,
            valid_edges=valid_edges,
            stop_logits=stop_logits,
        )
    )
    empty = _resolve_empty_factorized(
        relation_logits=relation_logits,
        edge_logits=edge_logits,
        stop_logits=stop_logits,
        num_graphs=num_graphs,
        temperature=temperature,
    )
    if empty is not None:
        return empty
    return _compute_factorized_nonempty(
        relation_logits=relation_logits,
        edge_logits=edge_logits,
        edge_relations=edge_relations,
        edge_batch=edge_batch,
        valid_edges=valid_edges,
        num_graphs=num_graphs,
        temperature=temperature,
        stop_logits=stop_logits,
        allow_stop=allow_stop,
    )

def gumbel_noise_like(tensor: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(tensor)
    return -torch.log(-torch.log(u.clamp(min=_GUMBEL_EPS, max=1.0 - _GUMBEL_EPS)))


def pool_nodes_mean_by_ptr(
    *,
    node_tokens: torch.Tensor,
    node_locals: torch.Tensor,
    ptr: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if node_tokens.dim() != 2:
        raise ValueError("node_tokens must be [N, H] for pooling.")
    node_locals = node_locals.to(device=node_tokens.device, dtype=torch.long).view(-1)
    dim = int(node_tokens.size(-1))
    pooled = torch.zeros((num_graphs, dim), device=node_tokens.device, dtype=node_tokens.dtype)
    if node_locals.numel() == 0:
        return pooled
    positions = torch.arange(node_locals.numel(), device=node_tokens.device, dtype=ptr.dtype)
    batch_ids = torch.bucketize(positions, ptr[1:], right=True)
    pooled.index_add_(0, batch_ids, node_tokens[node_locals])
    counts = (ptr[1:] - ptr[:-1]).clamp(min=_ONE).to(dtype=node_tokens.dtype)
    return pooled / counts.unsqueeze(-1)


@dataclass(frozen=True)
class CommonBatchTensors:
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    question_emb: torch.Tensor
    node_embeddings: torch.Tensor
    node_embedding_ids: torch.Tensor
    edge_embeddings: torch.Tensor


@dataclass(frozen=True)
class RolloutBatchTensors:
    q_local_indices: torch.Tensor
    a_local_indices: torch.Tensor
    node_min_dists: torch.Tensor
    start_ptr: torch.Tensor
    answer_ptr: torch.Tensor
    node_in_degree: torch.Tensor | None


@dataclass(frozen=True)
class CommonBatchStats:
    num_graphs: int
    num_nodes_total: int
    num_edges: int
    node_counts: torch.Tensor
    node_ptr: torch.Tensor


class GFlowNetInputValidator:
    def __init__(self, *, validate_edge_batch: bool = True, validate_rollout_batch: bool = True) -> None:
        self.validate_edge_batch_enabled = bool(validate_edge_batch)
        self.validate_rollout_batch_enabled = bool(validate_rollout_batch)

    def validate_edge_batch(self, batch: Any, *, device: torch.device) -> None:
        if not self.validate_edge_batch_enabled:
            return
        common = self._collect_common_tensors(batch, device)
        stats = self._validate_common(common)
        self._validate_embeddings(common, stats)
        self._validate_node_embedding_ids(common.node_embedding_ids, stats.num_nodes_total)

    def validate_rollout_batch(self, batch: Any, *, device: torch.device, is_training: bool) -> None:
        if not self.validate_rollout_batch_enabled:
            return
        common = self._collect_common_tensors(batch, device)
        stats = self._validate_common(common)
        self._validate_embeddings(common, stats)
        self._validate_node_embedding_ids(common.node_embedding_ids, stats.num_nodes_total)
        rollout = self._collect_rollout_tensors(batch, device=device)
        self._validate_rollout_anchors(rollout, stats, is_training=is_training)
        self._validate_node_min_dists(rollout.node_min_dists, stats.num_nodes_total)
        self._validate_node_in_degree(rollout.node_in_degree, stats.num_nodes_total)

    @staticmethod
    def _require_tensor_attr(batch: Any, name: str, *, device: torch.device) -> torch.Tensor:
        value = getattr(batch, name, None)
        if not torch.is_tensor(value):
            raise ValueError(f"Batch missing tensor field: {name}.")
        return value.to(device=device, non_blocking=True)

    @staticmethod
    def _require_slice_ptr(batch: Any, name: str, *, device: torch.device) -> torch.Tensor:
        slice_dict = getattr(batch, "_slice_dict", None)
        if not isinstance(slice_dict, dict):
            raise ValueError("Batch missing _slice_dict required for packed pointers.")
        ptr = slice_dict.get(name)
        if ptr is None:
            raise ValueError(f"Batch missing _slice_dict entry: {name}.")
        return torch.as_tensor(ptr, dtype=torch.long, device=device).view(_NEG_ONE)

    def _collect_common_tensors(self, batch: Any, device: torch.device) -> CommonBatchTensors:
        return CommonBatchTensors(
            node_ptr=self._require_tensor_attr(batch, "ptr", device=device),
            edge_index=self._require_tensor_attr(batch, "edge_index", device=device),
            edge_attr=self._require_tensor_attr(batch, "edge_attr", device=device),
            question_emb=self._require_tensor_attr(batch, "question_emb", device=device),
            node_embeddings=self._require_tensor_attr(batch, "node_embeddings", device=device),
            node_embedding_ids=self._require_tensor_attr(batch, "node_embedding_ids", device=device),
            edge_embeddings=self._require_tensor_attr(batch, "edge_embeddings", device=device),
        )

    def _collect_rollout_tensors(self, batch: Any, *, device: torch.device) -> RolloutBatchTensors:
        node_in_degree = getattr(batch, "node_in_degree", None)
        if torch.is_tensor(node_in_degree):
            node_in_degree = node_in_degree.to(device=device, non_blocking=True)
        return RolloutBatchTensors(
            q_local_indices=self._require_tensor_attr(batch, "q_local_indices", device=device),
            a_local_indices=self._require_tensor_attr(batch, "a_local_indices", device=device),
            node_min_dists=self._require_tensor_attr(batch, "node_min_dists", device=device),
            start_ptr=self._require_slice_ptr(batch, "q_local_indices", device=device),
            answer_ptr=self._require_slice_ptr(batch, "a_local_indices", device=device),
            node_in_degree=node_in_degree if torch.is_tensor(node_in_degree) else None,
        )

    def _validate_common(self, tensors: CommonBatchTensors) -> CommonBatchStats:
        self._validate_integer_dtype(tensors.node_ptr, "ptr")
        num_graphs, num_nodes_total, node_counts = self._validate_node_ptr(tensors.node_ptr)
        self._validate_integer_dtype(tensors.edge_index, "edge_index")
        num_edges = self._validate_edge_index(tensors.edge_index, tensors.node_ptr, num_graphs)
        self._validate_integer_dtype(tensors.edge_attr, "edge_attr")
        self._validate_edge_attr(tensors.edge_attr, num_edges)
        return CommonBatchStats(
            num_graphs=num_graphs,
            num_nodes_total=num_nodes_total,
            num_edges=num_edges,
            node_counts=node_counts,
            node_ptr=tensors.node_ptr,
        )

    @staticmethod
    def _validate_integer_dtype(tensor: torch.Tensor, name: str) -> None:
        if tensor.dtype != torch.long:
            raise ValueError(f"{name} must be torch.long, got {tensor.dtype}.")

    @staticmethod
    def _validate_floating_dtype(tensor: torch.Tensor, name: str) -> None:
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must be floating point, got {tensor.dtype}.")

    @staticmethod
    def _validate_node_ptr(node_ptr: torch.Tensor) -> tuple[int, int, torch.Tensor]:
        if node_ptr.dim() != 1:
            raise ValueError(f"ptr must be 1D, got shape={tuple(node_ptr.shape)}.")
        if node_ptr.numel() < _PTR_MIN_LEN:
            raise ValueError("ptr must have at least two offsets.")
        if int(node_ptr[0].item()) != _ZERO:
            raise ValueError("ptr must start at 0.")
        if bool((node_ptr[1:] < node_ptr[:-1]).any().item()):
            raise ValueError("ptr must be non-decreasing.")
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= _ZERO:
            raise ValueError("ptr must encode at least one graph.")
        num_nodes_total = int(node_ptr[-1].item())
        if num_nodes_total <= _ZERO:
            raise ValueError("ptr must encode positive node counts.")
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=_ZERO)
        return num_graphs, num_nodes_total, node_counts

    def _validate_edge_index(
        self,
        edge_index: torch.Tensor,
        node_ptr: torch.Tensor,
        num_graphs: int,
    ) -> int:
        if edge_index.dim() != _TWO or edge_index.size(0) != _TWO:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
        num_edges = int(edge_index.size(1))
        if edge_index.numel() == _ZERO:
            return num_edges
        if not self.validate_edge_batch_enabled:
            return num_edges
        compute_edge_batch(
            edge_index,
            node_ptr=node_ptr,
            num_graphs=num_graphs,
            device=edge_index.device,
            validate=True,
        )
        return num_edges

    @staticmethod
    def _validate_edge_attr(edge_attr: torch.Tensor, num_edges: int) -> None:
        if edge_attr.dim() != 1:
            raise ValueError(f"edge_attr must be 1D, got shape={tuple(edge_attr.shape)}.")
        if edge_attr.numel() != num_edges:
            raise ValueError(f"edge_attr length {edge_attr.numel()} != num_edges {num_edges}.")
        if edge_attr.numel() > _ZERO and bool((edge_attr < _ZERO).any().item()):
            raise ValueError("edge_attr contains negative relation ids (stop edges are disallowed).")

    def _validate_embeddings(self, common: CommonBatchTensors, stats: CommonBatchStats) -> None:
        self._validate_floating_dtype(common.question_emb, "question_emb")
        self._validate_floating_dtype(common.node_embeddings, "node_embeddings")
        self._validate_floating_dtype(common.edge_embeddings, "edge_embeddings")
        if common.question_emb.dim() != _TWO:
            raise ValueError(f"question_emb must be [B, D], got shape={tuple(common.question_emb.shape)}.")
        if common.node_embeddings.dim() != _TWO:
            raise ValueError(f"node_embeddings must be [N, D], got shape={tuple(common.node_embeddings.shape)}.")
        if common.edge_embeddings.dim() != _TWO:
            raise ValueError(f"edge_embeddings must be [E, D], got shape={tuple(common.edge_embeddings.shape)}.")
        if common.question_emb.size(0) != stats.num_graphs:
            raise ValueError("question_emb batch size mismatch with ptr.")
        if common.node_embeddings.size(0) != stats.num_nodes_total:
            raise ValueError("node_embeddings length mismatch with ptr.")
        if common.edge_embeddings.size(0) != stats.num_edges:
            raise ValueError("edge_embeddings length mismatch with edge_index.")
        self._ensure_finite(common.question_emb, "question_emb")
        self._ensure_finite(common.node_embeddings, "node_embeddings")
        self._ensure_finite(common.edge_embeddings, "edge_embeddings")

    def _validate_node_embedding_ids(self, node_embedding_ids: torch.Tensor, num_nodes_total: int) -> None:
        self._validate_integer_dtype(node_embedding_ids, "node_embedding_ids")
        if node_embedding_ids.dim() != 1:
            raise ValueError(f"node_embedding_ids must be 1D, got shape={tuple(node_embedding_ids.shape)}.")
        if node_embedding_ids.numel() != num_nodes_total:
            raise ValueError("node_embedding_ids length mismatch with ptr.")

    def _validate_rollout_anchors(
        self,
        rollout: RolloutBatchTensors,
        stats: CommonBatchStats,
        *,
        is_training: bool,
    ) -> None:
        q_local = rollout.q_local_indices.view(_NEG_ONE)
        a_local = rollout.a_local_indices.view(_NEG_ONE)
        self._validate_integer_dtype(q_local, "q_local_indices")
        self._validate_integer_dtype(a_local, "a_local_indices")
        self._validate_integer_dtype(rollout.start_ptr, "q_local_indices_ptr")
        self._validate_integer_dtype(rollout.answer_ptr, "a_local_indices_ptr")
        self._validate_ptr(rollout.start_ptr, stats.num_graphs, q_local.numel(), "q_local_indices")
        self._validate_ptr(rollout.answer_ptr, stats.num_graphs, a_local.numel(), "a_local_indices")
        start_counts = rollout.start_ptr[1:] - rollout.start_ptr[:-1]
        missing_start = start_counts == _ZERO
        if bool(missing_start.any().item()):
            raise ValueError("Start nodes missing from batch; sub dataset should exclude these samples.")
        answer_counts = rollout.answer_ptr[1:] - rollout.answer_ptr[:-1]
        missing_answer = answer_counts == _ZERO
        if bool(missing_answer.any().item()):
            raise ValueError("Answer nodes missing in batch; filter unreachable samples before GFlowNet.")
        self._validate_local_indices(q_local, rollout.start_ptr, stats.node_ptr, "q_local_indices")
        self._validate_local_indices(a_local, rollout.answer_ptr, stats.node_ptr, "a_local_indices")

    @staticmethod
    def _validate_ptr(ptr: torch.Tensor, num_graphs: int, total: int, name: str) -> None:
        if ptr.dim() != 1:
            raise ValueError(f"{name}_ptr must be 1D, got shape={tuple(ptr.shape)}.")
        if ptr.numel() < _PTR_MIN_LEN:
            raise ValueError(f"{name}_ptr must have at least two offsets.")
        if ptr.numel() != num_graphs + 1:
            raise ValueError(f"{name}_ptr length mismatch with num_graphs.")
        if int(ptr[0].item()) != _ZERO:
            raise ValueError(f"{name}_ptr must start at 0.")
        if bool((ptr[1:] < ptr[:-1]).any().item()):
            raise ValueError(f"{name}_ptr must be non-decreasing.")
        if int(ptr[-1].item()) != total:
            raise ValueError(f"{name}_ptr must end at {total}, got {int(ptr[-1].item())}.")

    @staticmethod
    def _validate_local_indices(
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        name: str,
    ) -> None:
        if local_indices.numel() == _ZERO:
            return
        device = node_ptr.device
        local_indices = local_indices.to(device=device, non_blocking=True)
        ptr = ptr.to(device=device, non_blocking=True)
        node_ptr = node_ptr.to(device=device, non_blocking=True)
        positions = torch.arange(local_indices.numel(), device=device, dtype=ptr.dtype)
        graph_ids = torch.bucketize(positions, ptr[1:], right=True)
        if bool((local_indices < _ZERO).any().item()):
            raise ValueError(f"{name} contains negative indices.")
        node_start = node_ptr[graph_ids]
        node_end = node_ptr[graph_ids + _ONE]
        in_range = (local_indices >= node_start) & (local_indices < node_end)
        if bool((~in_range).any().item()):
            raise ValueError(f"{name} indices fall outside per-graph node ranges.")

    @staticmethod
    def _validate_node_min_dists(node_min_dists: torch.Tensor, num_nodes_total: int) -> None:
        if node_min_dists.dim() != 1:
            raise ValueError(f"node_min_dists must be 1D, got shape={tuple(node_min_dists.shape)}.")
        if node_min_dists.numel() != num_nodes_total:
            raise ValueError("node_min_dists length mismatch with ptr.")

    @staticmethod
    def _validate_node_in_degree(node_in_degree: torch.Tensor | None, num_nodes_total: int) -> None:
        if node_in_degree is None:
            return
        if node_in_degree.dim() != 1:
            raise ValueError(f"node_in_degree must be 1D, got shape={tuple(node_in_degree.shape)}.")
        if node_in_degree.numel() != num_nodes_total:
            raise ValueError("node_in_degree length mismatch with ptr.")

    @staticmethod
    def _ensure_finite(tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).sum().item()
            raise ValueError(f"{name} contains {bad} non-finite values.")


@dataclass(frozen=True)
class EdgeTokenInputs:
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor


@dataclass(frozen=True)
class RolloutInputs:
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    node_in_degree: torch.Tensor
    node_min_dists: torch.Tensor
    node_q_min_dists: torch.Tensor
    node_embedding_ids: torch.Tensor
    start_node_locals: torch.Tensor
    answer_node_locals: torch.Tensor
    start_ptr: torch.Tensor
    answer_ptr: torch.Tensor
    dummy_mask: torch.Tensor


class GFlowNetBatchProcessor:
    """Prepare rollout inputs and graph caches for EB-GFN batches."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        require_precomputed_edge_batch: bool = True,
        require_precomputed_node_in_degree: bool = True,
        corridor_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.backbone = backbone
        self.require_precomputed_edge_batch = bool(require_precomputed_edge_batch)
        self.require_precomputed_node_in_degree = bool(require_precomputed_node_in_degree)
        self._corridor_spec = _resolve_corridor_spec(corridor_cfg)

    @staticmethod
    def _get_node_ptr(batch: Any, device: torch.device) -> torch.Tensor:
        return batch.ptr.to(device=device, non_blocking=True)

    @staticmethod
    def _get_start_answer_ptrs(
        batch: Any,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        override_nodes = getattr(batch, "start_override_locals", None)
        override_ptr = getattr(batch, "start_override_ptr", None)
        if torch.is_tensor(override_nodes) or torch.is_tensor(override_ptr):
            if not torch.is_tensor(override_nodes) or not torch.is_tensor(override_ptr):
                raise ValueError("start_override_locals/start_override_ptr must be provided together.")
            start_node_locals = override_nodes.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
            start_ptr = override_ptr.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        else:
            start_node_locals = batch.q_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
            slice_dict = batch._slice_dict
            start_ptr = torch.as_tensor(slice_dict["q_local_indices"], dtype=torch.long, device=device)
        answer_node_locals = batch.a_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        slice_dict = batch._slice_dict
        answer_ptr = torch.as_tensor(slice_dict["a_local_indices"], dtype=torch.long, device=device)
        return start_node_locals, answer_node_locals, start_ptr, answer_ptr

    @staticmethod
    def _get_q_local_indices(
        batch: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return batch.q_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)

    @staticmethod
    def build_dummy_mask(
        *,
        answer_ptr: torch.Tensor,
    ) -> torch.Tensor:
        answer_counts = answer_ptr[1:] - answer_ptr[:-1]
        missing = answer_counts == _ZERO
        if bool(missing.any().item()):
            raise ValueError("Answer nodes missing in batch; filter missing answers before GFlowNet.")
        return missing.to(dtype=torch.bool)

    def _resolve_edge_batch_and_ptr(
        self,
        batch: Any,
        *,
        edge_index: torch.Tensor,
        node_ptr: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_edges = int(edge_index.size(1))
        edge_batch = getattr(batch, "edge_batch", None)
        edge_ptr = getattr(batch, "edge_ptr", None)
        if edge_batch is None and edge_ptr is None:
            if self.require_precomputed_edge_batch:
                raise ValueError("Batch missing edge_batch/edge_ptr; enable precompute_edge_batch.")
            edge_batch, edge_ptr = compute_edge_batch(
                edge_index,
                node_ptr=node_ptr,
                num_graphs=num_graphs,
                device=edge_index.device,
                validate=False,
            )
            return (
                edge_batch.to(device=edge_index.device, dtype=torch.long).view(-1),
                edge_ptr.to(device=edge_index.device, dtype=torch.long).view(-1),
            )
        if not (torch.is_tensor(edge_batch) and torch.is_tensor(edge_ptr)):
            raise ValueError("edge_batch and edge_ptr must be provided together.")
        edge_batch = edge_batch.to(device=edge_index.device, dtype=torch.long, non_blocking=True).view(-1)
        edge_ptr = edge_ptr.to(device=edge_index.device, dtype=torch.long, non_blocking=True).view(-1)
        if edge_batch.numel() != num_edges:
            raise ValueError("edge_batch length mismatch with edge_index.")
        if edge_ptr.numel() != num_graphs + _ONE:
            raise ValueError("edge_ptr length mismatch with num_graphs.")
        return edge_batch, edge_ptr

    @staticmethod
    def _get_precomputed_node_in_degree(
        batch: Any,
        *,
        device: torch.device,
        num_nodes_total: int,
    ) -> torch.Tensor | None:
        node_in_degree = getattr(batch, "node_in_degree", None)
        if not torch.is_tensor(node_in_degree):
            return None
        node_in_degree = node_in_degree.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if node_in_degree.numel() != num_nodes_total:
            raise ValueError("node_in_degree length mismatch with ptr.")
        return node_in_degree

    def encode_batch_tokens(
        self,
        batch: Any,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_tokens, relation_tokens, question_tokens = self.backbone(batch)
        return (
            node_tokens.to(device=device, dtype=torch.float32),
            relation_tokens.to(device=device, dtype=torch.float32),
            question_tokens.to(device=device, dtype=torch.float32),
        )

    def prepare_edge_token_inputs(
        self,
        batch: Any,
        device: torch.device,
    ) -> EdgeTokenInputs:
        node_ptr = self._get_node_ptr(batch, device)
        edge_index = batch.edge_index.to(device=device, non_blocking=True)
        edge_relations = batch.edge_attr.to(device=device, non_blocking=True)
        edge_batch_full, edge_ptr = self._resolve_edge_batch_and_ptr(
            batch,
            edge_index=edge_index,
            node_ptr=node_ptr,
            num_graphs=int(node_ptr.numel() - 1),
        )
        node_tokens, relation_tokens, question_tokens = self.encode_batch_tokens(batch, device=device)
        return EdgeTokenInputs(
            edge_batch=edge_batch_full,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            question_tokens=question_tokens,
        )

    @staticmethod
    def _get_node_min_dists(
        batch: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return batch.node_min_dists.to(device=device, dtype=torch.long, non_blocking=True)

    @staticmethod
    def _compute_node_q_min_dists(
        edge_index: torch.Tensor,
        *,
        num_nodes_total: int,
        q_node_locals: torch.Tensor,
    ) -> torch.Tensor:
        if num_nodes_total <= _ZERO:
            return torch.zeros((_ZERO,), device=edge_index.device, dtype=torch.long)
        return directed_bfs_distances(
            edge_index,
            num_nodes=num_nodes_total,
            start_nodes=q_node_locals,
        )

    @staticmethod
    def _get_node_embedding_ids(
        batch: Any,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return batch.node_embedding_ids.to(device=device, dtype=torch.long, non_blocking=True)

    @staticmethod
    def _build_rollout_inputs(
        *,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        node_in_degree: torch.Tensor,
        node_min_dists: torch.Tensor,
        node_q_min_dists: torch.Tensor,
        node_embedding_ids: torch.Tensor,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        answer_ptr: torch.Tensor,
        dummy_mask: torch.Tensor,
    ) -> RolloutInputs:
        return RolloutInputs(
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            question_tokens=question_tokens,
            node_in_degree=node_in_degree,
            node_min_dists=node_min_dists,
            node_q_min_dists=node_q_min_dists,
            node_embedding_ids=node_embedding_ids,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            start_ptr=start_ptr,
            answer_ptr=answer_ptr,
            dummy_mask=dummy_mask,
        )

    def prepare_full_rollout_inputs(
        self,
        batch: Any,
        device: torch.device,
    ) -> RolloutInputs:
        token_inputs = self.prepare_edge_token_inputs(batch, device)
        node_ptr = token_inputs.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        edge_index_base = token_inputs.edge_index
        node_in_degree = self._get_precomputed_node_in_degree(
            batch,
            device=device,
            num_nodes_total=num_nodes_total,
        )
        if node_in_degree is None:
            if self.require_precomputed_node_in_degree:
                raise ValueError("Batch missing node_in_degree; enable precompute_node_in_degree.")
            node_in_degree = compute_undirected_degree(
                edge_index_base,
                num_nodes=num_nodes_total,
            ).to(device=device, dtype=torch.long)
        node_min_dists = self._get_node_min_dists(batch, device=device)
        node_embedding_ids = self._get_node_embedding_ids(batch, device=device)
        q_node_locals = self._get_q_local_indices(batch, device=device)
        node_q_min_dists = self._compute_node_q_min_dists(
            edge_index_base,
            num_nodes_total=num_nodes_total,
            q_node_locals=q_node_locals,
        )

        start_node_locals, answer_node_locals, start_ptr, answer_ptr = self._get_start_answer_ptrs(
            batch, device=device
        )
        dummy_mask = self.build_dummy_mask(answer_ptr=answer_ptr)
        edge_ptr = token_inputs.edge_ptr
        return self._build_rollout_inputs(
            edge_batch=token_inputs.edge_batch,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            edge_index=token_inputs.edge_index,
            edge_relations=token_inputs.edge_relations,
            node_tokens=token_inputs.node_tokens,
            relation_tokens=token_inputs.relation_tokens,
            question_tokens=token_inputs.question_tokens,
            node_in_degree=node_in_degree,
            node_min_dists=node_min_dists,
            node_q_min_dists=node_q_min_dists,
            node_embedding_ids=node_embedding_ids,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            start_ptr=start_ptr,
            answer_ptr=answer_ptr,
            dummy_mask=dummy_mask,
        )

    def prepare_rollout_inputs(
        self,
        batch: Any,
        device: torch.device,
    ) -> RolloutInputs:
        return self.prepare_full_rollout_inputs(batch, device)

    @staticmethod
    def compute_node_batch(node_ptr: torch.Tensor, num_graphs: int, device: torch.device) -> torch.Tensor:
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        return torch.repeat_interleave(torch.arange(num_graphs, device=device), node_counts)

    @staticmethod
    def compute_node_flags(
        num_nodes_total: int,
        start_node_locals: torch.Tensor,
        answer_node_locals: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_is_start = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        if start_node_locals.numel() > 0:
            node_is_start[start_node_locals] = True
        node_is_answer = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        if answer_node_locals.numel() > 0:
            node_is_answer[answer_node_locals] = True
        return node_is_start, node_is_answer

    @staticmethod
    def _compute_edge_corridor_mask(
        *,
        edge_index: torch.Tensor,
        node_q_min_dists: torch.Tensor,
        node_min_dists: torch.Tensor,
    ) -> torch.Tensor:
        heads = edge_index[0]
        tails = edge_index[1]
        from_start = node_q_min_dists.index_select(0, heads) >= _ZERO
        to_answer = node_min_dists.index_select(0, tails) >= _ZERO
        return from_start & to_answer

    @staticmethod
    def _compute_edge_distance_bias(
        *,
        edge_index: torch.Tensor,
        node_q_min_dists: torch.Tensor,
        node_min_dists: torch.Tensor,
        corridor_mask: torch.Tensor,
        alpha: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        heads = edge_index[0]
        tails = edge_index[1]
        dist_start = node_q_min_dists.index_select(0, heads).to(dtype=dtype)
        dist_answer = node_min_dists.index_select(0, tails).to(dtype=dtype)
        bias = (dist_start + dist_answer) * (-float(alpha))
        return torch.where(corridor_mask, bias, torch.zeros_like(bias))

    def build_graph_cache(self, inputs: RolloutInputs, *, device: torch.device) -> Dict[str, torch.Tensor]:
        node_ptr = inputs.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        node_batch = self.compute_node_batch(node_ptr, num_graphs, device)
        node_is_start, node_is_answer = self.compute_node_flags(
            num_nodes_total,
            inputs.start_node_locals,
            inputs.answer_node_locals,
            device,
        )
        edge_corridor_mask = None
        edge_distance_bias = None
        corridor_mask = None
        if inputs.edge_index.numel() > 0 and (
            self._corridor_spec.enabled or self._corridor_spec.distance_bias_alpha > float(_ZERO)
        ):
            corridor_mask = self._compute_edge_corridor_mask(
                edge_index=inputs.edge_index,
                node_q_min_dists=inputs.node_q_min_dists,
                node_min_dists=inputs.node_min_dists,
            )
            if self._corridor_spec.enabled:
                edge_corridor_mask = corridor_mask
            if self._corridor_spec.distance_bias_alpha > float(_ZERO):
                edge_distance_bias = self._compute_edge_distance_bias(
                    edge_index=inputs.edge_index,
                    node_q_min_dists=inputs.node_q_min_dists,
                    node_min_dists=inputs.node_min_dists,
                    corridor_mask=corridor_mask,
                    alpha=self._corridor_spec.distance_bias_alpha,
                    dtype=inputs.node_tokens.dtype,
                )
        graph_cache: Dict[str, torch.Tensor] = {
            "edge_index": inputs.edge_index,
            "edge_batch": inputs.edge_batch,
            "edge_relations": inputs.edge_relations,
            "node_ptr": node_ptr,
            "edge_ptr": inputs.edge_ptr,
            "node_tokens": inputs.node_tokens,
            "relation_tokens": inputs.relation_tokens,
            "question_tokens": inputs.question_tokens,
            "node_embedding_ids": inputs.node_embedding_ids,
            "node_batch": node_batch,
            "node_is_start": node_is_start,
            "node_is_answer": node_is_answer,
            "node_in_degree": inputs.node_in_degree,
            "node_min_dists": inputs.node_min_dists,
            "node_q_min_dists": inputs.node_q_min_dists,
            "start_node_locals": inputs.start_node_locals,
            "start_ptr": inputs.start_ptr,
            "answer_node_locals": inputs.answer_node_locals,
            "answer_ptr": inputs.answer_ptr,
            "dummy_mask": inputs.dummy_mask,
        }
        if edge_corridor_mask is not None:
            graph_cache["edge_corridor_mask"] = edge_corridor_mask
        if edge_distance_bias is not None:
            graph_cache["edge_distance_bias"] = edge_distance_bias
        return graph_cache


__all__ = [
    "compute_factorized_log_probs",
    "compute_policy_log_probs",
    "gumbel_noise_like",
    "neg_inf_value",
    "pool_nodes_mean_by_ptr",
    "segment_logsumexp_1d",
    "GFlowNetInputValidator",
    "EdgeTokenInputs",
    "GFlowNetBatchProcessor",
    "RolloutInputs",
]
