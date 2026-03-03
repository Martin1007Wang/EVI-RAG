from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from src.metrics.base import normalize_k_values
from src.metrics.dual_flow.config import CompositeScoreConfig

DEFAULT_STOP_MARGIN = 0.0
STOP_MARGIN_P50 = 0.5
STOP_MARGIN_P90 = 0.9


def _normalize_k_pairs(
    k_values: Sequence[int], num_rollouts: int
) -> List[Tuple[int, int]]:
    ks = normalize_k_values(k_values)
    if not ks:
        return []
    if num_rollouts <= 0:
        return [(int(k), 0) for k in ks]
    pairs = []
    for k in ks:
        k_int = int(k)
        k_clamped = min(max(k_int, 1), num_rollouts)
        pairs.append((k_int, k_clamped))
    return pairs


def _select_prefix_rows(
    values: torch.Tensor, k_pairs: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    if not k_pairs:
        return values.new_zeros((0, values.size(1)))
    indices = torch.as_tensor(
        [k_clamped - 1 for _, k_clamped in k_pairs],
        device=values.device,
        dtype=torch.long,
    )
    return values.index_select(0, indices)


def _prefix_metric_map(
    *,
    values: torch.Tensor,
    k_pairs: Sequence[Tuple[int, int]],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if not k_pairs:
        return {}
    selected = _select_prefix_rows(values, k_pairs)
    mapped: Dict[str, torch.Tensor] = {}
    for idx, (k_int, _) in enumerate(k_pairs):
        mapped[f"{prefix}@{k_int}"] = selected[idx]
    return mapped


def _reduce_rollout_stack(values: torch.Tensor, best_of: bool) -> torch.Tensor:
    if not best_of:
        return values.mean(dim=0)
    return values.max(dim=0).values


def _reduce_rollout_tensor(
    value: torch.Tensor,
    *,
    num_rollouts: int,
    num_graphs: int,
    best_of: bool,
) -> torch.Tensor:
    total = num_rollouts * num_graphs
    if value.numel() == total:
        stack = value.reshape(num_rollouts, num_graphs).float()
        return _reduce_rollout_stack(stack, best_of)
    if value.numel() == num_rollouts and value.dim() <= 1:
        stack = value.reshape(num_rollouts).float()
        return _reduce_rollout_stack(stack, best_of)
    return value


def reduce_rollout_metrics(
    metrics: Dict[str, torch.Tensor],
    *,
    num_rollouts: int,
    num_graphs: int,
    best_of: bool = False,
) -> Dict[str, torch.Tensor]:
    if not metrics:
        return {}
    reduced: Dict[str, torch.Tensor] = {}
    for key, value in metrics.items():
        if not torch.is_tensor(value):
            reduced[key] = value
            continue
        reduced[key] = _reduce_rollout_tensor(
            value,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            best_of=best_of,
        )
    return reduced


def stack_rollout_metrics(
    metrics_list: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {key: torch.stack([m[key] for m in metrics_list], dim=0) for key in keys}


def finalize_rollout_metrics(
    loss_list: List[torch.Tensor],
    metrics_list: List[Dict[str, torch.Tensor]],
    *,
    num_rollouts: int,
    num_graphs: int,
    best_of: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not loss_list:
        return torch.zeros((), dtype=torch.float32), {}
    if num_rollouts > 1:
        loss = torch.stack(loss_list, dim=0).mean()
        stacked = stack_rollout_metrics(metrics_list)
        metrics = reduce_rollout_metrics(
            stacked,
            num_rollouts=num_rollouts,
            num_graphs=num_graphs,
            best_of=best_of,
        )
        return loss, metrics
    return loss_list[0], metrics_list[0]


def compute_terminal_hits(
    *,
    stop_node_locals: torch.Tensor,
    node_ptr: torch.Tensor,
    node_is_target: torch.Tensor,
) -> torch.Tensor:
    num_graphs = int(node_ptr.numel() - 1)
    if num_graphs <= 0:
        return torch.zeros((0,), device=node_is_target.device, dtype=torch.bool)
    stop_node_locals = stop_node_locals.to(device=node_ptr.device, dtype=torch.long)
    node_ptr = node_ptr.to(device=node_is_target.device, dtype=torch.long)
    node_offsets = node_ptr[:-1]
    valid = stop_node_locals >= 0
    stop_globals = node_offsets + stop_node_locals.clamp(min=0)
    node_is_target = node_is_target.to(device=node_ptr.device, dtype=torch.bool)
    hits = node_is_target.index_select(0, stop_globals.clamp(min=0))
    return valid & hits


def compute_terminal_hit_prefixes(
    *,
    terminal_hits: torch.Tensor,
    k_values: Sequence[int],
) -> Dict[str, torch.Tensor]:
    num_rollouts = int(terminal_hits.size(0))
    k_pairs = _normalize_k_pairs(k_values, num_rollouts)
    if not k_pairs:
        return {}
    hit_cum = terminal_hits.to(dtype=torch.bool).cumsum(dim=0) > 0
    return _prefix_metric_map(
        values=hit_cum.to(dtype=torch.float32),
        k_pairs=k_pairs,
        prefix="terminal_hit",
    )


def compute_composite_score(
    *,
    metrics: Dict[str, torch.Tensor],
    k_values: Sequence[int],
    composite_cfg: CompositeScoreConfig,
) -> Dict[str, torch.Tensor]:
    if not composite_cfg.enabled:
        return {}
    ks = normalize_k_values(k_values)
    if not ks:
        return {}
    pass_prob = metrics.get("pass@1")
    if pass_prob is None:
        return {}
    if not torch.is_tensor(pass_prob):
        pass_prob = torch.as_tensor(pass_prob)
    pass_prob = pass_prob.to(dtype=torch.float32)
    ones = torch.ones_like(pass_prob)
    weight_context = float(composite_cfg.weight_context_hit)
    weight_terminal = float(composite_cfg.weight_terminal_hit)
    weight_pass_best = float(composite_cfg.weight_pass_best)
    composite: Dict[str, torch.Tensor] = {}
    for k_int in ks:
        context = metrics.get(f"context_hit@{k_int}")
        terminal = metrics.get(f"terminal_hit@{k_int}")
        if context is None or terminal is None:
            continue
        if not torch.is_tensor(context):
            context = torch.as_tensor(context, device=pass_prob.device)
        if not torch.is_tensor(terminal):
            terminal = torch.as_tensor(terminal, device=pass_prob.device)
        context = context.to(dtype=torch.float32, device=pass_prob.device)
        terminal = terminal.to(dtype=torch.float32, device=pass_prob.device)
        pass_best = ones - torch.pow(ones - pass_prob, int(k_int))
        composite[f"composite_score@{k_int}"] = (
            (weight_context * context)
            + (weight_terminal * terminal)
            + (weight_pass_best * pass_best)
        )
    return composite


def _reshape_rollout_metric(
    values: torch.Tensor,
    *,
    num_rollouts: int,
    num_graphs: int,
) -> torch.Tensor:
    if values.dim() == 2 and values.shape == (num_rollouts, num_graphs):
        return values
    expected = num_rollouts * num_graphs
    return values.reshape(num_rollouts, num_graphs)


def compute_reward_gap(
    *,
    log_reward: torch.Tensor,
    pass_hits: torch.Tensor,
    num_rollouts: int,
    num_graphs: int,
) -> torch.Tensor:
    log_reward = _reshape_rollout_metric(
        log_reward, num_rollouts=num_rollouts, num_graphs=num_graphs
    ).float()
    pass_hits = _reshape_rollout_metric(
        pass_hits, num_rollouts=num_rollouts, num_graphs=num_graphs
    ).to(dtype=torch.bool)
    finite = torch.isfinite(log_reward)
    hit_mask = pass_hits & finite
    miss_mask = (~pass_hits) & finite
    hit_sum = (log_reward * hit_mask.to(dtype=log_reward.dtype)).sum(dim=0)
    miss_sum = (log_reward * miss_mask.to(dtype=log_reward.dtype)).sum(dim=0)
    hit_count = hit_mask.sum(dim=0).clamp(min=1)
    miss_count = miss_mask.sum(dim=0).clamp(min=1)
    hit_mean = hit_sum / hit_count
    miss_mean = miss_sum / miss_count
    has_both = (hit_mask.sum(dim=0) > 0) & (miss_mask.sum(dim=0) > 0)
    gap = torch.where(has_both, hit_mean - miss_mean, torch.zeros_like(hit_mean))
    return gap.to(dtype=torch.float32)


def _stop_margin_metrics_for_step(
    *,
    has_edge_seq: torch.Tensor,
    stop_margin_seq: torch.Tensor,
    step_idx: int,
) -> Dict[str, torch.Tensor]:
    num_steps = int(has_edge_seq.size(1))
    if step_idx < 0 or step_idx >= num_steps:
        return {}
    has_edge = has_edge_seq[:, step_idx].to(dtype=torch.bool)
    stop_margin = stop_margin_seq[:, step_idx]
    rate = has_edge.to(dtype=stop_margin.dtype).mean()
    valid = stop_margin[has_edge]
    if valid.numel() > 0:
        mean = valid.mean()
        p50 = valid.quantile(STOP_MARGIN_P50)
        p90 = valid.quantile(STOP_MARGIN_P90)
    else:
        mean = stop_margin.new_tensor(DEFAULT_STOP_MARGIN)
        p50 = stop_margin.new_tensor(DEFAULT_STOP_MARGIN)
        p90 = stop_margin.new_tensor(DEFAULT_STOP_MARGIN)
    return {
        f"has_edge@{step_idx}": rate,
        f"stop_margin_mean@{step_idx}": mean,
        f"stop_margin_p50@{step_idx}": p50,
        f"stop_margin_p90@{step_idx}": p90,
    }


def compute_diag_metrics(rollout: Any) -> Dict[str, torch.Tensor]:
    has_edge_seq = getattr(rollout, "has_edge_seq", None)
    stop_margin_seq = getattr(rollout, "stop_margin_seq", None)
    if not torch.is_tensor(has_edge_seq) or not torch.is_tensor(stop_margin_seq):
        return {}
    if has_edge_seq.dim() != 2 or stop_margin_seq.dim() != 2:
        return {}
    if has_edge_seq.shape != stop_margin_seq.shape:
        return {}
    metrics: Dict[str, torch.Tensor] = {}
    metrics.update(
        _stop_margin_metrics_for_step(
            has_edge_seq=has_edge_seq,
            stop_margin_seq=stop_margin_seq,
            step_idx=0,
        )
    )
    metrics.update(
        _stop_margin_metrics_for_step(
            has_edge_seq=has_edge_seq,
            stop_margin_seq=stop_margin_seq,
            step_idx=1,
        )
    )
    return metrics


def build_potential_metrics(
    *,
    reach_success: torch.Tensor,
    num_moves: torch.Tensor,
    reward_out: Any,
    log_reward: torch.Tensor,
    phi_start: torch.Tensor,
    phi_target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    reward_metrics = reward_out.as_dict()
    log_reward_metric = reward_metrics.pop("log_reward")
    if torch.is_tensor(log_reward_metric):
        log_reward_metric = log_reward_metric.detach()
    reward_metrics.pop("reward", None)
    answer_hit = reward_metrics.pop("answer_hit", None)
    success = reward_metrics.pop("success", None)
    if answer_hit is None:
        answer_hit = success
    answer_tensor = (
        answer_hit if isinstance(answer_hit, torch.Tensor) else reach_success
    )
    return {
        "log_reward": log_reward_metric,
        "phi": phi_start.detach(),
        "phi_target": phi_target.detach(),
        "pass@1": answer_tensor.detach(),
        "length_mean": num_moves.detach(),
        **{k: v.detach() for k, v in reward_metrics.items()},
    }


__all__ = [
    "reduce_rollout_metrics",
    "stack_rollout_metrics",
    "finalize_rollout_metrics",
    "compute_terminal_hits",
    "compute_terminal_hit_prefixes",
    "compute_composite_score",
    "compute_reward_gap",
    "compute_diag_metrics",
    "build_potential_metrics",
]
