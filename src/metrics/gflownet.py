from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from src.metrics.common import normalize_k_values

_ZERO = 0
_ONE = 1
_TWO = 2
_NEG_ONE = -1
_FLOAT_ZERO = 0.0
_FLOAT_ONE = 1.0
_DEFAULT_K = 10
_DEFAULT_K_VALUES = (_DEFAULT_K,)
_DEFAULT_STOP_MARGIN = 0.0
_STOP_MARGIN_P50 = 0.5
_STOP_MARGIN_P90 = 0.9
_DEFAULT_COMPOSITE_ENABLED = False
_DEFAULT_COMPOSITE_WEIGHT_CONTEXT_HIT = 0.6
_DEFAULT_COMPOSITE_WEIGHT_TERMINAL_HIT = 0.3
_DEFAULT_COMPOSITE_WEIGHT_PASS_BEST = 0.1


@dataclass(frozen=True)
class CompositeScoreConfig:
    enabled: bool = _DEFAULT_COMPOSITE_ENABLED
    weight_context_hit: float = _DEFAULT_COMPOSITE_WEIGHT_CONTEXT_HIT
    weight_terminal_hit: float = _DEFAULT_COMPOSITE_WEIGHT_TERMINAL_HIT
    weight_pass_best: float = _DEFAULT_COMPOSITE_WEIGHT_PASS_BEST

    @property
    def weight_sum(self) -> float:
        return float(self.weight_context_hit + self.weight_terminal_hit + self.weight_pass_best)


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a float, got bool.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TypeError(f"{name} must be a float, got empty string.")
        try:
            return float(text)
        except ValueError as exc:
            raise TypeError(f"{name} must be a float, got {value!r}.") from exc
    raise TypeError(f"{name} must be a float, got {type(value).__name__}.")


def _require_non_negative_float(value: Any, name: str) -> float:
    parsed = _require_float(value, name)
    if parsed < _FLOAT_ZERO:
        raise ValueError(f"{name} must be >= 0, got {parsed}.")
    return parsed


def resolve_composite_score_cfg(raw_cfg: Optional[Any]) -> CompositeScoreConfig:
    if isinstance(raw_cfg, CompositeScoreConfig):
        return raw_cfg
    if raw_cfg is None:
        return CompositeScoreConfig()
    if not isinstance(raw_cfg, Mapping):
        raise TypeError(f"composite_score_cfg must be a mapping or None, got {type(raw_cfg).__name__}.")
    enabled = bool(raw_cfg.get("enabled", _DEFAULT_COMPOSITE_ENABLED))
    weight_context = _require_non_negative_float(
        raw_cfg.get("weight_context_hit", _DEFAULT_COMPOSITE_WEIGHT_CONTEXT_HIT),
        "composite_score_cfg.weight_context_hit",
    )
    weight_terminal = _require_non_negative_float(
        raw_cfg.get("weight_terminal_hit", _DEFAULT_COMPOSITE_WEIGHT_TERMINAL_HIT),
        "composite_score_cfg.weight_terminal_hit",
    )
    weight_pass_best = _require_non_negative_float(
        raw_cfg.get("weight_pass_best", _DEFAULT_COMPOSITE_WEIGHT_PASS_BEST),
        "composite_score_cfg.weight_pass_best",
    )
    cfg = CompositeScoreConfig(
        enabled=enabled,
        weight_context_hit=weight_context,
        weight_terminal_hit=weight_terminal,
        weight_pass_best=weight_pass_best,
    )
    if cfg.enabled and cfg.weight_sum <= _FLOAT_ZERO:
        raise ValueError("composite_score_cfg weights must sum to a positive value.")
    return cfg


def _normalize_k_pairs(k_values: Sequence[int], num_rollouts: int) -> List[Tuple[int, int]]:
    ks = normalize_k_values(k_values)
    if not ks:
        return []
    if num_rollouts <= _ZERO:
        return [(int(k), _ZERO) for k in ks]
    pairs = []
    for k in ks:
        k_int = int(k)
        k_clamped = min(max(k_int, _ONE), num_rollouts)
        pairs.append((k_int, k_clamped))
    return pairs


def _select_prefix_rows(values: torch.Tensor, k_pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if not k_pairs:
        return values.new_zeros((_ZERO, values.size(1)))
    indices = torch.as_tensor([k_clamped - _ONE for _, k_clamped in k_pairs], device=values.device, dtype=torch.long)
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
    if value.numel() == num_rollouts and value.dim() <= _ONE:
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


def stack_rollout_metrics(metrics_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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
        raise RuntimeError("No rollouts recorded for rollout aggregation.")
    if num_rollouts > _ONE:
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
    num_graphs = int(node_ptr.numel() - _ONE)
    if num_graphs <= _ZERO:
        return torch.zeros((_ZERO,), device=node_is_target.device, dtype=torch.bool)
    stop_node_locals = stop_node_locals.to(device=node_ptr.device, dtype=torch.long)
    if stop_node_locals.numel() != num_graphs:
        raise ValueError("stop_node_locals length mismatch for terminal hit computation.")
    node_ptr = node_ptr.to(device=node_is_target.device, dtype=torch.long)
    node_offsets = node_ptr[:-_ONE]
    valid = stop_node_locals >= _ZERO
    stop_globals = node_offsets + stop_node_locals.clamp(min=_ZERO)
    if stop_globals.numel() > _ZERO and int(stop_globals.max().detach().tolist()) >= int(node_is_target.numel()):
        raise ValueError("stop_node_locals index out of range for terminal hit computation.")
    node_is_target = node_is_target.to(device=node_ptr.device, dtype=torch.bool)
    hits = node_is_target.index_select(_ZERO, stop_globals.clamp(min=_ZERO))
    return valid & hits


def compute_terminal_hit_prefixes(
    *,
    terminal_hits: torch.Tensor,
    k_values: Sequence[int],
) -> Dict[str, torch.Tensor]:
    if terminal_hits.dim() != _TWO:
        raise ValueError("terminal_hits must be [R, B] for prefix metrics.")
    num_rollouts = int(terminal_hits.size(0))
    k_pairs = _normalize_k_pairs(k_values, num_rollouts)
    if not k_pairs:
        return {}
    hit_cum = terminal_hits.to(dtype=torch.bool).cumsum(dim=0) > _ZERO
    return _prefix_metric_map(values=hit_cum.to(dtype=torch.float32), k_pairs=k_pairs, prefix="terminal_hit")


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
    if values.dim() == _TWO and values.shape == (num_rollouts, num_graphs):
        return values
    expected = num_rollouts * num_graphs
    if values.numel() != expected:
        raise ValueError("rollout metric length mismatch for reward gap.")
    return values.reshape(num_rollouts, num_graphs)


def compute_reward_gap(
    *,
    log_reward: torch.Tensor,
    pass_hits: torch.Tensor,
    num_rollouts: int,
    num_graphs: int,
) -> torch.Tensor:
    log_reward = _reshape_rollout_metric(log_reward, num_rollouts=num_rollouts, num_graphs=num_graphs).float()
    pass_hits = _reshape_rollout_metric(pass_hits, num_rollouts=num_rollouts, num_graphs=num_graphs).to(dtype=torch.bool)
    finite = torch.isfinite(log_reward)
    hit_mask = pass_hits & finite
    miss_mask = (~pass_hits) & finite
    hit_sum = (log_reward * hit_mask.to(dtype=log_reward.dtype)).sum(dim=0)
    miss_sum = (log_reward * miss_mask.to(dtype=log_reward.dtype)).sum(dim=0)
    hit_count = hit_mask.sum(dim=0).clamp(min=_ONE)
    miss_count = miss_mask.sum(dim=0).clamp(min=_ONE)
    hit_mean = hit_sum / hit_count
    miss_mean = miss_sum / miss_count
    has_both = (hit_mask.sum(dim=0) > _ZERO) & (miss_mask.sum(dim=0) > _ZERO)
    gap = torch.where(has_both, hit_mean - miss_mean, torch.zeros_like(hit_mean))
    return gap.to(dtype=torch.float32)


def _stop_margin_metrics_for_step(
    *,
    has_edge_seq: torch.Tensor,
    stop_margin_seq: torch.Tensor,
    step_idx: int,
) -> Dict[str, torch.Tensor]:
    num_steps = int(has_edge_seq.size(_ONE))
    if step_idx < _ZERO or step_idx >= num_steps:
        return {}
    has_edge = has_edge_seq[:, step_idx].to(dtype=torch.bool)
    stop_margin = stop_margin_seq[:, step_idx]
    rate = has_edge.to(dtype=stop_margin.dtype).mean()
    valid = stop_margin[has_edge]
    if valid.numel() > _ZERO:
        mean = valid.mean()
        p50 = valid.quantile(_STOP_MARGIN_P50)
        p90 = valid.quantile(_STOP_MARGIN_P90)
    else:
        mean = stop_margin.new_tensor(_DEFAULT_STOP_MARGIN)
        p50 = stop_margin.new_tensor(_DEFAULT_STOP_MARGIN)
        p90 = stop_margin.new_tensor(_DEFAULT_STOP_MARGIN)
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
    if has_edge_seq.dim() != _TWO or stop_margin_seq.dim() != _TWO:
        return {}
    if has_edge_seq.shape != stop_margin_seq.shape:
        return {}
    metrics: Dict[str, torch.Tensor] = {}
    metrics.update(
        _stop_margin_metrics_for_step(
            has_edge_seq=has_edge_seq,
            stop_margin_seq=stop_margin_seq,
            step_idx=_ZERO,
        )
    )
    metrics.update(
        _stop_margin_metrics_for_step(
            has_edge_seq=has_edge_seq,
            stop_margin_seq=stop_margin_seq,
            step_idx=_ONE,
        )
    )
    return metrics


def build_flow_metrics(
    *,
    reach_success: torch.Tensor,
    num_moves: torch.Tensor,
    reward_out: Any,
    log_reward: torch.Tensor,
    log_f_start: torch.Tensor,
    log_f_target: torch.Tensor,
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
    answer_tensor = answer_hit if isinstance(answer_hit, torch.Tensor) else reach_success
    metrics: Dict[str, torch.Tensor] = {
        "log_reward": log_reward_metric,
        "log_f": log_f_start.detach(),
        "log_f_target": log_f_target.detach(),
        "pass@1": answer_tensor.detach(),
        "length_mean": num_moves.detach(),
        **{k: v.detach() for k, v in reward_metrics.items()},
    }
    return metrics


@dataclass
class GFlowNetEvalState:
    num_samples: int = _ZERO
    num_rollouts: int = _ZERO
    answer_samples: int = _ZERO
    answer_rollouts: int = _ZERO
    pass_hits: int = _ZERO
    length_sum: int = _ZERO
    length_count: int = _ZERO
    diversity_ratio_sum: float = _FLOAT_ZERO
    diversity_count: int = _ZERO
    terminal_hit_counts: Dict[int, int] = field(default_factory=dict)
    context_recall_sum: Dict[int, float] = field(default_factory=dict)
    context_precision_sum: Dict[int, float] = field(default_factory=dict)
    context_f1_sum: Dict[int, float] = field(default_factory=dict)
    context_hit_counts: Dict[int, int] = field(default_factory=dict)
    composite_score_sum: Dict[int, float] = field(default_factory=dict)


def _as_int_set(values: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for val in values:
        try:
            out.add(int(val))
        except (TypeError, ValueError):
            continue
    return out


def _sort_rollouts(rollouts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rollouts,
        key=lambda r: int(r.get("rollout_index", _ZERO) or _ZERO),
    )


def _rollout_nodes(rollout: Dict[str, Any]) -> set[int]:
    edges = rollout.get("edges") or []
    nodes: set[int] = set()
    for edge in edges:
        for key in ("src_entity_id", "dst_entity_id", "head_entity_id", "tail_entity_id"):
            val = edge.get(key)
            if val is None:
                continue
            try:
                nodes.add(int(val))
            except (TypeError, ValueError):
                continue
    stop_node = rollout.get("stop_node_entity_id")
    if stop_node is not None:
        try:
            nodes.add(int(stop_node))
        except (TypeError, ValueError):
            pass
    return nodes


def _rollout_signature(edges: Iterable[Dict[str, Any]]) -> Tuple[Tuple[int, int, int], ...]:
    signature: List[Tuple[int, int, int]] = []
    for edge in edges:
        src = edge.get("src_entity_id")
        if src is None:
            src = edge.get("head_entity_id")
        dst = edge.get("dst_entity_id")
        if dst is None:
            dst = edge.get("tail_entity_id")
        rel = edge.get("relation_id")
        if src is not None and dst is not None:
            src_val = int(src)
            dst_val = int(dst)
            if dst_val < src_val:
                src_val, dst_val = dst_val, src_val
        else:
            src_val = _NEG_ONE
            dst_val = _NEG_ONE
        signature.append(
            (
                src_val,
                int(rel) if rel is not None else _NEG_ONE,
                dst_val,
            )
        )
    return tuple(signature)


def _context_stats(
    *,
    context_nodes: set[int],
    answer_set: set[int],
    start_set: set[int],
) -> Tuple[float, float, float, float]:
    if not answer_set:
        return _FLOAT_ZERO, _FLOAT_ZERO, _FLOAT_ZERO, _FLOAT_ZERO
    recall = float(len(context_nodes & answer_set)) / float(len(answer_set))
    context_nonstart = context_nodes - start_set
    answer_nonstart = answer_set - start_set
    if context_nonstart:
        precision = float(len(context_nonstart & answer_nonstart)) / float(len(context_nonstart))
    else:
        precision = _FLOAT_ZERO
    denom = precision + recall
    f1 = (_TWO * precision * recall / denom) if denom > _FLOAT_ZERO else _FLOAT_ZERO
    hit = _ONE if recall > _FLOAT_ZERO else _ZERO
    return recall, precision, f1, float(hit)


def _rollout_lengths_and_signatures(
    rollouts_sorted: List[Dict[str, Any]],
) -> Tuple[List[int], set[Tuple[Tuple[int, int, int], ...]]]:
    lengths = [len(r.get("edges") or []) for r in rollouts_sorted]
    signatures = {_rollout_signature(r.get("edges") or []) for r in rollouts_sorted}
    return lengths, signatures


def _rollout_hit_stats(
    rollouts_sorted: List[Dict[str, Any]],
    answer_set: set[int],
) -> Tuple[List[bool], List[bool], List[set[int]]]:
    pass_hits = [bool(r.get("reach_success", False)) for r in rollouts_sorted]
    terminal_hits = [
        bool(r.get("stop_node_entity_id") in answer_set) if r.get("stop_node_entity_id") is not None else False
        for r in rollouts_sorted
    ]
    rollout_nodes = [_rollout_nodes(r) for r in rollouts_sorted]
    return pass_hits, terminal_hits, rollout_nodes


def _prefix_rollout_stats(
    *,
    rollout_nodes: List[set[int]],
    terminal_hits: List[bool],
    start_set: set[int],
) -> Tuple[List[set[int]], List[bool]]:
    prefix_nodes: List[set[int]] = []
    running = set(start_set)
    for nodes in rollout_nodes:
        running.update(nodes)
        prefix_nodes.append(set(running))
    prefix_terminal: List[bool] = []
    running_hit = False
    for hit in terminal_hits:
        running_hit = running_hit or hit
        prefix_terminal.append(running_hit)
    return prefix_nodes, prefix_terminal


def _update_prefix_state(
    *,
    state: GFlowNetEvalState,
    k_pairs: List[Tuple[int, int]],
    prefix_nodes: List[set[int]],
    prefix_terminal: List[bool],
    answer_set: set[int],
    start_set: set[int],
    pass_rate: float,
    composite_cfg: CompositeScoreConfig,
) -> None:
    for k_int, k_clamped in k_pairs:
        if k_clamped <= _ZERO:
            continue
        idx = k_clamped - _ONE
        context_nodes = prefix_nodes[idx]
        recall, precision, f1, hit = _context_stats(
            context_nodes=context_nodes,
            answer_set=answer_set,
            start_set=start_set,
        )
        if prefix_terminal[idx]:
            state.terminal_hit_counts[k_int] += _ONE
        state.context_recall_sum[k_int] += recall
        state.context_precision_sum[k_int] += precision
        state.context_f1_sum[k_int] += f1
        if hit > _FLOAT_ZERO:
            state.context_hit_counts[k_int] += _ONE
        if composite_cfg.enabled:
            pass_best = _FLOAT_ONE - (_FLOAT_ONE - pass_rate) ** float(k_int)
            score = (
                (composite_cfg.weight_context_hit * hit)
                + (composite_cfg.weight_terminal_hit * float(prefix_terminal[idx]))
                + (composite_cfg.weight_pass_best * pass_best)
            )
            state.composite_score_sum[k_int] = state.composite_score_sum.get(k_int, _FLOAT_ZERO) + score


class GFlowNetEvalAccumulator:
    def __init__(
        self,
        *,
        k_values: Optional[Sequence[int]] = None,
        composite_score_cfg: Optional[Any] = None,
    ) -> None:
        self.k_values = normalize_k_values(k_values, default=_DEFAULT_K_VALUES)
        self._composite_cfg = resolve_composite_score_cfg(composite_score_cfg)
        self._state = GFlowNetEvalState(
            terminal_hit_counts={int(k): _ZERO for k in self.k_values},
            context_recall_sum={int(k): _FLOAT_ZERO for k in self.k_values},
            context_precision_sum={int(k): _FLOAT_ZERO for k in self.k_values},
            context_f1_sum={int(k): _FLOAT_ZERO for k in self.k_values},
            context_hit_counts={int(k): _ZERO for k in self.k_values},
            composite_score_sum=(
                {int(k): _FLOAT_ZERO for k in self.k_values} if self._composite_cfg.enabled else {}
            ),
        )

    def update_from_records(self, records: List[Dict[str, Any]]) -> None:
        for record in records:
            self._update_from_record(record)

    def finalize(self) -> Dict[str, float]:
        if self._state.num_samples <= _ZERO:
            return {}
        denom_rollouts = max(self._state.answer_rollouts, _ONE)
        denom_length = max(self._state.length_count, _ONE)
        denom_diversity = max(self._state.diversity_count, _ONE)
        denom_answers = max(self._state.answer_samples, _ONE)
        metrics: Dict[str, float] = {
            "num_samples": float(self._state.num_samples),
            "num_rollouts": float(self._state.num_rollouts),
            "answer_eval_samples": float(self._state.answer_samples),
            "answer_eval_rollouts": float(self._state.answer_rollouts),
            "pass@1": float(self._state.pass_hits) / float(denom_rollouts),
            "length_mean": float(self._state.length_sum) / float(denom_length),
            "path_diversity": float(self._state.diversity_ratio_sum) / float(denom_diversity),
        }
        for k in self.k_values:
            k_int = int(k)
            terminal_hits = self._state.terminal_hit_counts.get(k_int, _ZERO)
            metrics[f"terminal_hit@{k_int}"] = float(terminal_hits) / float(denom_answers)
            metrics[f"context_recall@{k_int}"] = float(self._state.context_recall_sum.get(k_int, _FLOAT_ZERO)) / float(denom_answers)
            metrics[f"context_precision@{k_int}"] = float(self._state.context_precision_sum.get(k_int, _FLOAT_ZERO)) / float(denom_answers)
            metrics[f"context_f1@{k_int}"] = float(self._state.context_f1_sum.get(k_int, _FLOAT_ZERO)) / float(denom_answers)
            metrics[f"context_hit@{k_int}"] = float(self._state.context_hit_counts.get(k_int, _ZERO)) / float(denom_answers)
            if self._composite_cfg.enabled:
                composite_sum = self._state.composite_score_sum.get(k_int, _FLOAT_ZERO)
                metrics[f"composite_score@{k_int}"] = float(composite_sum) / float(denom_answers)
        return metrics

    def _update_from_record(self, record: Dict[str, Any]) -> None:
        rollouts = record.get("rollouts") or []
        if not isinstance(rollouts, list) or not rollouts:
            return
        rollouts_sorted = _sort_rollouts(rollouts)
        num_rollouts = len(rollouts_sorted)
        self._state.num_samples += _ONE
        self._state.num_rollouts += num_rollouts
        lengths, signatures = _rollout_lengths_and_signatures(rollouts_sorted)
        self._state.length_sum += sum(lengths)
        self._state.length_count += num_rollouts
        unique_count = len(signatures)
        denom_rollouts = max(num_rollouts, _ONE)
        self._state.diversity_ratio_sum += float(unique_count) / float(denom_rollouts)
        self._state.diversity_count += _ONE

        answer_set = _as_int_set(record.get("answer_entity_ids") or [])
        if not answer_set:
            return
        start_set = _as_int_set(record.get("start_entity_ids") or [])
        pass_hits, terminal_hits, rollout_nodes = _rollout_hit_stats(rollouts_sorted, answer_set)

        self._state.answer_samples += _ONE
        self._state.answer_rollouts += num_rollouts
        self._state.pass_hits += sum(pass_hits)
        denom_rollouts = float(max(num_rollouts, _ONE))
        pass_rate = float(sum(pass_hits)) / denom_rollouts

        k_pairs = _normalize_k_pairs(self.k_values, num_rollouts)
        if not k_pairs:
            return
        prefix_nodes, prefix_terminal = _prefix_rollout_stats(
            rollout_nodes=rollout_nodes,
            terminal_hits=terminal_hits,
            start_set=start_set,
        )
        _update_prefix_state(
            state=self._state,
            k_pairs=k_pairs,
            prefix_nodes=prefix_nodes,
            prefix_terminal=prefix_terminal,
            answer_set=answer_set,
            start_set=start_set,
            pass_rate=pass_rate,
            composite_cfg=self._composite_cfg,
        )


__all__ = [
    "CompositeScoreConfig",
    "resolve_composite_score_cfg",
    "reduce_rollout_metrics",
    "stack_rollout_metrics",
    "finalize_rollout_metrics",
    "compute_terminal_hits",
    "compute_terminal_hit_prefixes",
    "compute_composite_score",
    "compute_reward_gap",
    "compute_diag_metrics",
    "build_flow_metrics",
    "GFlowNetEvalAccumulator",
]
