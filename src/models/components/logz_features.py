from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from src.gfn.ops import segment_logsumexp_1d
from src.utils.graph import directed_bfs_distances

_ZERO = 0
_ONE = 1
_GRAPH_STATS_DIM = 3
_MIN_HOP_STEPS = 1
_MIN_TEMPERATURE = 1.0e-6
_RELEVANCE_STATS_DIM = 3


@dataclass(frozen=True)
class LogZFeatureSpec:
    use_graph_stats: bool
    graph_stats_log1p: bool
    use_hop_counts: bool
    hop_steps: int
    use_relevance: bool
    use_relevance_stats: bool
    relevance_temperature: float
    stats_dim: int


def resolve_logz_spec(
    logz_cfg: Mapping[str, object] | None,
    *,
    hidden_dim: int,
    max_steps: int,
) -> LogZFeatureSpec:
    if logz_cfg is None:
        raise ValueError("logz_cfg must be provided for LogZ features.")
    use_graph_stats = bool(logz_cfg.get("use_graph_stats", True))
    graph_stats_log1p = bool(logz_cfg.get("graph_stats_log1p", True))
    use_hop_counts = bool(logz_cfg.get("use_hop_counts", True))
    hop_steps_cfg = logz_cfg.get("hop_steps")
    hop_steps = int(max_steps) if hop_steps_cfg is None else int(hop_steps_cfg)
    if use_hop_counts and hop_steps < _MIN_HOP_STEPS:
        raise ValueError("logz_cfg.hop_steps must be >= 1 when hop counts are enabled.")
    if use_hop_counts and hop_steps > int(max_steps):
        raise ValueError("logz_cfg.hop_steps must be <= max_steps.")
    use_relevance = bool(logz_cfg.get("use_relevance", True))
    use_relevance_stats = bool(logz_cfg.get("use_relevance_stats", False))
    relevance_temperature = float(_ONE)
    if use_relevance or use_relevance_stats:
        temp_cfg = logz_cfg.get("relevance_temperature")
        if temp_cfg is None:
            raise ValueError("logz_cfg.relevance_temperature must be set when relevance pooling is enabled.")
        relevance_temperature = float(temp_cfg)
        if relevance_temperature < _MIN_TEMPERATURE:
            raise ValueError("logz_cfg.relevance_temperature must be positive.")
    stats_dim = _ZERO
    if use_graph_stats:
        stats_dim += _GRAPH_STATS_DIM
    if use_hop_counts:
        stats_dim += hop_steps
    if use_relevance:
        stats_dim += int(hidden_dim)
    if use_relevance_stats:
        stats_dim += _RELEVANCE_STATS_DIM
    if stats_dim <= _ZERO:
        raise ValueError("LogZ feature spec resolved to empty feature set.")
    return LogZFeatureSpec(
        use_graph_stats=use_graph_stats,
        graph_stats_log1p=graph_stats_log1p,
        use_hop_counts=use_hop_counts,
        hop_steps=hop_steps,
        use_relevance=use_relevance,
        use_relevance_stats=use_relevance_stats,
        relevance_temperature=relevance_temperature,
        stats_dim=stats_dim,
    )


def _node_batch_from_ptr(node_ptr: torch.Tensor, num_graphs: int) -> torch.Tensor:
    node_counts = (node_ptr[_ONE:] - node_ptr[:-_ONE]).clamp(min=_ZERO)
    graph_ids = torch.arange(num_graphs, device=node_ptr.device)
    return torch.repeat_interleave(graph_ids, node_counts)


def _compute_graph_stats(
    *,
    node_ptr: torch.Tensor,
    edge_ptr: torch.Tensor,
    start_ptr: torch.Tensor,
    log1p: bool,
) -> torch.Tensor:
    node_counts = (node_ptr[_ONE:] - node_ptr[:-_ONE]).to(dtype=torch.float32)
    edge_counts = (edge_ptr[_ONE:] - edge_ptr[:-_ONE]).to(dtype=torch.float32)
    start_counts = (start_ptr[_ONE:] - start_ptr[:-_ONE]).to(dtype=torch.float32)
    stats = torch.stack((node_counts, edge_counts, start_counts), dim=1)
    if log1p:
        stats = torch.log1p(stats)
    return stats


def _compute_hop_counts(
    *,
    edge_index: torch.Tensor,
    node_ptr: torch.Tensor,
    start_node_locals: torch.Tensor,
    hop_steps: int,
) -> torch.Tensor:
    num_graphs = int(node_ptr.numel() - _ONE)
    if num_graphs <= _ZERO or hop_steps <= _ZERO:
        return torch.zeros((num_graphs, _ZERO), device=edge_index.device, dtype=torch.float32)
    num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else _ZERO
    if num_nodes_total <= _ZERO:
        return torch.zeros((num_graphs, hop_steps), device=edge_index.device, dtype=torch.float32)
    if start_node_locals.numel() == _ZERO:
        return torch.zeros((num_graphs, hop_steps), device=edge_index.device, dtype=torch.float32)
    dist = directed_bfs_distances(edge_index, num_nodes=num_nodes_total, start_nodes=start_node_locals)
    if dist.numel() == _ZERO:
        return torch.zeros((num_graphs, hop_steps), device=edge_index.device, dtype=torch.float32)
    node_batch = _node_batch_from_ptr(node_ptr, num_graphs)
    hop_ids = torch.arange(_ONE, hop_steps + _ONE, device=dist.device, dtype=dist.dtype)
    dist = dist.view(-1, _ONE)
    valid = dist >= _ZERO
    hop_mask = valid & (dist <= hop_ids)
    hop_counts = torch.zeros((num_graphs, hop_steps), device=dist.device, dtype=torch.float32)
    if hop_mask.numel() == _ZERO:
        return hop_counts
    hop_counts.index_add_(0, node_batch, hop_mask.to(dtype=torch.float32))
    return hop_counts


def _compute_relevance_scores(
    *,
    node_tokens: torch.Tensor,
    question_tokens: torch.Tensor,
    node_batch: torch.Tensor,
    num_graphs: int,
    temperature: float,
) -> torch.Tensor:
    if node_tokens.numel() == _ZERO or num_graphs <= _ZERO:
        return torch.zeros((node_batch.numel(),), device=node_tokens.device, dtype=node_tokens.dtype)
    token_dim = int(node_tokens.size(-1))
    q_tokens = question_tokens.index_select(0, node_batch)
    scale = math.sqrt(float(token_dim)) * float(temperature)
    return (node_tokens * q_tokens).sum(dim=-1) / scale


def _compute_relevance_pool(
    *,
    node_tokens: torch.Tensor,
    scores: torch.Tensor,
    node_batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if node_tokens.numel() == _ZERO or num_graphs <= _ZERO:
        dim = int(node_tokens.size(-1)) if node_tokens.dim() == 2 else _ZERO
        return torch.zeros((num_graphs, dim), device=node_tokens.device, dtype=node_tokens.dtype)
    token_dim = int(node_tokens.size(-1))
    log_denom = segment_logsumexp_1d(scores, node_batch, num_graphs)
    weights = torch.exp(scores - log_denom[node_batch])
    pooled = torch.zeros((num_graphs, token_dim), device=node_tokens.device, dtype=node_tokens.dtype)
    pooled.index_add_(0, node_batch, node_tokens * weights.unsqueeze(-1))
    return pooled


def _segment_max_1d(values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    max_vals = torch.full((num_segments,), neg_inf, device=values.device, dtype=values.dtype)
    max_vals.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    return max_vals


def _compute_relevance_score_stats(
    *,
    scores: torch.Tensor,
    node_batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if scores.numel() == _ZERO or num_graphs <= _ZERO:
        return torch.zeros((num_graphs, _RELEVANCE_STATS_DIM), device=scores.device, dtype=scores.dtype)
    counts = torch.bincount(node_batch, minlength=num_graphs).to(dtype=scores.dtype)
    count_safe = counts.clamp(min=_ONE)
    sum_scores = torch.zeros((num_graphs,), device=scores.device, dtype=scores.dtype)
    sum_scores.index_add_(0, node_batch, scores)
    mean_scores = sum_scores / count_safe
    sum_sq = torch.zeros((num_graphs,), device=scores.device, dtype=scores.dtype)
    sum_sq.index_add_(0, node_batch, scores * scores)
    var_scores = sum_sq / count_safe - mean_scores * mean_scores
    var_scores = torch.clamp(var_scores, min=float(_ZERO))
    std_scores = torch.sqrt(var_scores)
    max_scores = _segment_max_1d(scores, node_batch, num_graphs)
    max_scores = torch.where(counts > _ZERO, max_scores, torch.zeros_like(max_scores))
    return torch.stack((mean_scores, max_scores, std_scores), dim=1)


def build_logz_features(
    *,
    node_ptr: torch.Tensor,
    edge_ptr: torch.Tensor,
    start_ptr: torch.Tensor,
    edge_index: torch.Tensor,
    start_node_locals: torch.Tensor,
    node_tokens: torch.Tensor,
    question_tokens: torch.Tensor,
    spec: LogZFeatureSpec,
) -> torch.Tensor:
    num_graphs = int(node_ptr.numel() - _ONE)
    if num_graphs <= _ZERO:
        return torch.zeros((num_graphs, spec.stats_dim), device=node_ptr.device, dtype=torch.float32)
    features: list[torch.Tensor] = []
    if spec.use_graph_stats:
        features.append(
            _compute_graph_stats(
                node_ptr=node_ptr,
                edge_ptr=edge_ptr,
                start_ptr=start_ptr,
                log1p=spec.graph_stats_log1p,
            )
        )
    if spec.use_hop_counts:
        features.append(
            _compute_hop_counts(
                edge_index=edge_index,
                node_ptr=node_ptr,
                start_node_locals=start_node_locals,
                hop_steps=spec.hop_steps,
            )
        )
    if spec.use_relevance or spec.use_relevance_stats:
        node_batch = _node_batch_from_ptr(node_ptr, num_graphs)
        scores = _compute_relevance_scores(
            node_tokens=node_tokens,
            question_tokens=question_tokens,
            node_batch=node_batch,
            num_graphs=num_graphs,
            temperature=spec.relevance_temperature,
        )
        if spec.use_relevance:
            features.append(
                _compute_relevance_pool(
                    node_tokens=node_tokens,
                    scores=scores,
                    node_batch=node_batch,
                    num_graphs=num_graphs,
                )
            )
        if spec.use_relevance_stats:
            features.append(
                _compute_relevance_score_stats(
                    scores=scores,
                    node_batch=node_batch,
                    num_graphs=num_graphs,
                )
            )
    return torch.cat(features, dim=-1)


__all__ = [
    "LogZFeatureSpec",
    "resolve_logz_spec",
    "build_logz_features",
]
