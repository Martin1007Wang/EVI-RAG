from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torchmetrics import Metric

from src.metrics.base import normalize_k_values
from src.metrics.dual_flow.config import resolve_composite_score_cfg

DEFAULT_K = 10
DEFAULT_K_VALUES = (DEFAULT_K,)


def _as_int_set(values: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for val in values:
        try:
            out.add(int(val))
        except (TypeError, ValueError):
            continue
    return out


def _sort_rollouts(rollouts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rollouts, key=lambda r: int(r.get("rollout_index", 0) or 0))


def _rollout_nodes(rollout: Dict[str, Any]) -> set[int]:
    edges = rollout.get("edges") or []
    nodes: set[int] = set()
    for edge in edges:
        for key in (
            "src_entity_id",
            "dst_entity_id",
            "head_entity_id",
            "tail_entity_id",
        ):
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


def _rollout_signature(
    edges: Iterable[Dict[str, Any]],
) -> Tuple[Tuple[int, int, int], ...]:
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
            src_val = -1
            dst_val = -1
        signature.append(
            (
                src_val,
                int(rel) if rel is not None else -1,
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
        return 0.0, 0.0, 0.0, 0.0
    recall = float(len(context_nodes & answer_set)) / float(len(answer_set))
    context_nonstart = context_nodes - start_set
    answer_nonstart = answer_set - start_set
    if context_nonstart:
        precision = float(len(context_nonstart & answer_nonstart)) / float(
            len(context_nonstart)
        )
    else:
        precision = 0.0
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0
    hit = 1 if recall > 0.0 else 0
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
    rollout_nodes = [_rollout_nodes(r) for r in rollouts_sorted]
    pass_hits = [bool(nodes & answer_set) for nodes in rollout_nodes]
    terminal_hits = [
        bool(r.get("stop_node_entity_id") in answer_set)
        if r.get("stop_node_entity_id") is not None
        else False
        for r in rollouts_sorted
    ]
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


class DualFlowRolloutMetrics(Metric):
    full_state_update = False

    def __init__(
        self,
        *,
        k_values: Optional[Sequence[int]] = None,
        composite_score_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__(dist_sync_on_step=False)
        self.k_values = normalize_k_values(k_values, default=DEFAULT_K_VALUES)
        self._composite_cfg = resolve_composite_score_cfg(composite_score_cfg)
        k_len = len(self.k_values)

        self.add_state(
            "num_samples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_rollouts",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "answer_samples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "answer_rollouts",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "pass_hits",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "length_sum",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "length_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "diversity_ratio_sum",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "diversity_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "terminal_hit_counts",
            default=torch.zeros(k_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "context_recall_sum",
            default=torch.zeros(k_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "context_precision_sum",
            default=torch.zeros(k_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "context_f1_sum",
            default=torch.zeros(k_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "context_hit_counts",
            default=torch.zeros(k_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "composite_score_sum",
            default=torch.zeros(k_len, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, records: List[Dict[str, Any]]) -> None:  # type: ignore[override]
        if not records:
            return
        for record in records:
            self._update_from_record(record)

    def compute(self) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        if int(self.num_samples.item()) <= 0:
            return {}

        num_samples = self.num_samples.to(dtype=torch.float32)
        num_rollouts = self.num_rollouts.to(dtype=torch.float32)
        answer_samples = self.answer_samples.to(dtype=torch.float32)
        answer_rollouts = self.answer_rollouts.to(dtype=torch.float32)
        pass_hits = self.pass_hits.to(dtype=torch.float32)
        length_sum = self.length_sum.to(dtype=torch.float32)
        length_count = self.length_count.to(dtype=torch.float32)
        diversity_ratio_sum = self.diversity_ratio_sum.to(dtype=torch.float32)
        diversity_count = self.diversity_count.to(dtype=torch.float32)

        denom_rollouts = answer_rollouts.clamp(min=1.0)
        denom_length = length_count.clamp(min=1.0)
        denom_diversity = diversity_count.clamp(min=1.0)
        denom_answers = answer_samples.clamp(min=1.0)

        metrics: Dict[str, torch.Tensor] = {
            "num_samples": num_samples,
            "num_rollouts": num_rollouts,
            "answer_eval_samples": answer_samples,
            "answer_eval_rollouts": answer_rollouts,
            "pass@1": pass_hits / denom_rollouts,
            "length_mean": length_sum / denom_length,
            "path_diversity": diversity_ratio_sum / denom_diversity,
        }

        for idx, k in enumerate(self.k_values):
            k_int = int(k)
            terminal_hits = self.terminal_hit_counts[idx] / denom_answers
            context_recall = self.context_recall_sum[idx] / denom_answers
            context_precision = self.context_precision_sum[idx] / denom_answers
            context_f1 = self.context_f1_sum[idx] / denom_answers
            context_hit = self.context_hit_counts[idx] / denom_answers
            metrics[f"terminal_hit@{k_int}"] = terminal_hits
            metrics[f"context_recall@{k_int}"] = context_recall
            metrics[f"context_precision@{k_int}"] = context_precision
            metrics[f"context_f1@{k_int}"] = context_f1
            metrics[f"context_hit@{k_int}"] = context_hit
            metrics[f"hit@{k_int}"] = terminal_hits
            metrics[f"recall@{k_int}"] = context_recall
            metrics[f"precision@{k_int}"] = context_precision
            metrics[f"f1@{k_int}"] = context_f1
            if self._composite_cfg.enabled:
                metrics[f"composite_score@{k_int}"] = (
                    self.composite_score_sum[idx] / denom_answers
                )

        return metrics

    def _update_from_record(self, record: Dict[str, Any]) -> None:
        rollouts = record.get("rollouts") or []
        if not isinstance(rollouts, list) or not rollouts:
            return
        rollouts_sorted = _sort_rollouts(rollouts)
        num_rollouts = len(rollouts_sorted)
        self.num_samples += 1
        self.num_rollouts += num_rollouts

        lengths, signatures = _rollout_lengths_and_signatures(rollouts_sorted)
        self.length_sum += sum(lengths)
        self.length_count += num_rollouts
        unique_count = len(signatures)
        denom_rollouts = max(num_rollouts, 1)
        self.diversity_ratio_sum += float(unique_count) / float(denom_rollouts)
        self.diversity_count += 1

        answer_set = _as_int_set(record.get("answer_entity_ids") or [])
        if not answer_set:
            return
        start_set = _as_int_set(record.get("start_entity_ids") or [])
        pass_hits, terminal_hits, rollout_nodes = _rollout_hit_stats(
            rollouts_sorted, answer_set
        )

        self.answer_samples += 1
        self.answer_rollouts += num_rollouts
        self.pass_hits += sum(pass_hits)
        pass_rate = float(sum(pass_hits)) / float(denom_rollouts)

        if not self.k_values:
            return
        prefix_nodes, prefix_terminal = _prefix_rollout_stats(
            rollout_nodes=rollout_nodes,
            terminal_hits=terminal_hits,
            start_set=start_set,
        )
        for idx, k in enumerate(self.k_values):
            k_int = int(k)
            k_clamped = min(max(k_int, 1), num_rollouts) if num_rollouts > 0 else 0
            if k_clamped <= 0:
                continue
            prefix_idx = k_clamped - 1
            context_nodes = prefix_nodes[prefix_idx]
            recall, precision, f1, hit = _context_stats(
                context_nodes=context_nodes,
                answer_set=answer_set,
                start_set=start_set,
            )
            if prefix_terminal[prefix_idx]:
                self.terminal_hit_counts[idx] += 1.0
            self.context_recall_sum[idx] += recall
            self.context_precision_sum[idx] += precision
            self.context_f1_sum[idx] += f1
            if hit > 0.0:
                self.context_hit_counts[idx] += 1.0
            if self._composite_cfg.enabled:
                pass_best = 1.0 - (1.0 - pass_rate) ** float(k_int)
                score = (
                    (self._composite_cfg.weight_context_hit * hit)
                    + (
                        self._composite_cfg.weight_terminal_hit
                        * float(prefix_terminal[prefix_idx])
                    )
                    + (self._composite_cfg.weight_pass_best * pass_best)
                )
                self.composite_score_sum[idx] += score


__all__ = ["DualFlowRolloutMetrics"]
