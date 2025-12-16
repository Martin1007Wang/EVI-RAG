from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


def _normalize(text: str) -> str:
    text = text.lower()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _extract_predictions(raw: str) -> List[str]:
    if raw is None:
        return []
    lines = [line.strip() for line in str(raw).split("\n") if line.strip()]
    ans_lines = [ln for ln in lines if ln.lower().startswith("ans:")]
    if ans_lines:
        return ans_lines
    return lines[:1] if lines else []


def _match(pred: str, answer: str) -> bool:
    return _normalize(pred) == _normalize(answer) or _normalize(answer) in _normalize(pred)


def _score_answers(preds: List[str], gold_answers: List[str]) -> Dict[str, float]:
    if not gold_answers:
        return {"hit@1": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Hit@1
    if preds:
        hit = 1.0 if any(_match(preds[0], ans) for ans in gold_answers) else 0.0
    else:
        hit = 0.0

    matched = 0
    remaining_preds = preds.copy()
    for gold in gold_answers:
        for pred in remaining_preds:
            if _match(pred, gold):
                matched += 1
                remaining_preds.remove(pred)
                break
    precision = matched / max(len(preds), 1)
    recall = matched / len(gold_answers)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"hit@1": float(hit), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _as_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values if v is not None]
    try:
        return [int(values)]
    except (TypeError, ValueError):
        return []


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


@dataclass
class _SemanticAccumulator:
    """Accumulates semantic dissipation stats with interface awareness."""

    total_samples: int = 0
    with_gt: int = 0
    set_hit_sum: float = 0.0
    vis_hit_sum: float = 0.0
    hit_score_sum: float = 0.0
    hit_count: int = 0
    miss_score_sum: float = 0.0
    miss_count: int = 0
    evidence_tokens_sum: float = 0.0
    evidence_tokens_count: int = 0
    prompt_tokens_sum: float = 0.0
    prompt_tokens_count: int = 0
    token_budget_sum: float = 0.0
    token_budget_count: int = 0
    k_visible_sum: float = 0.0
    k_visible_count: int = 0
    truncated_count: int = 0

    def update(
        self,
        *,
        score: Optional[float],
        hit_set: Optional[bool],
        hit_vis: Optional[bool],
        evidence_tokens: Optional[int],
        prompt_tokens: Optional[int],
        token_budget: Optional[int],
        k_visible: Optional[int],
        evidence_truncated: bool,
    ) -> None:
        self.total_samples += 1
        if evidence_tokens is not None:
            self.evidence_tokens_sum += int(evidence_tokens)
            self.evidence_tokens_count += 1
        if prompt_tokens is not None:
            self.prompt_tokens_sum += int(prompt_tokens)
            self.prompt_tokens_count += 1
        if token_budget is not None:
            self.token_budget_sum += int(token_budget)
            self.token_budget_count += 1
        if k_visible is not None:
            self.k_visible_sum += int(k_visible)
            self.k_visible_count += 1
        if evidence_truncated:
            self.truncated_count += 1

        if hit_set is None or hit_vis is None or score is None:
            return
        self.with_gt += 1
        self.set_hit_sum += float(hit_set)
        self.vis_hit_sum += float(hit_vis)
        if hit_vis:
            self.hit_score_sum += float(score)
            self.hit_count += 1
        else:
            self.miss_score_sum += float(score)
            self.miss_count += 1

    def finalize(self, prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            f"{prefix}/total": float(self.total_samples),
            f"{prefix}/with_gt": float(self.with_gt),
        }
        if self.with_gt > 0:
            s_ret_set = self.set_hit_sum / float(self.with_gt)
            s_ret_vis = self.vis_hit_sum / float(self.with_gt)
            acc_hit = self.hit_score_sum / float(self.hit_count or 1)
            acc_miss = self.miss_score_sum / float(self.miss_count or 1)
            metrics[f"{prefix}/s_ret_set"] = s_ret_set
            metrics[f"{prefix}/s_ret_vis"] = s_ret_vis
            metrics[f"{prefix}/acc_hit"] = acc_hit
            metrics[f"{prefix}/acc_miss"] = acc_miss
            metrics[f"{prefix}/d_rate"] = 1.0 - acc_hit
            metrics[f"{prefix}/d_mass"] = s_ret_vis * (1.0 - acc_hit)
            metrics[f"{prefix}/l_leak"] = (1.0 - s_ret_vis) * acc_miss
            metrics[f"{prefix}/l_iface"] = s_ret_set - s_ret_vis
        else:
            metrics[f"{prefix}/s_ret_set"] = 0.0
            metrics[f"{prefix}/s_ret_vis"] = 0.0
            metrics[f"{prefix}/acc_hit"] = 0.0
            metrics[f"{prefix}/acc_miss"] = 0.0
            metrics[f"{prefix}/d_rate"] = 0.0
            metrics[f"{prefix}/d_mass"] = 0.0
            metrics[f"{prefix}/l_leak"] = 0.0
            metrics[f"{prefix}/l_iface"] = 0.0

        if self.prompt_tokens_count > 0:
            metrics[f"{prefix}/avg_prompt_tokens"] = self.prompt_tokens_sum / float(self.prompt_tokens_count)
        if self.evidence_tokens_count > 0:
            metrics[f"{prefix}/avg_evidence_tokens"] = self.evidence_tokens_sum / float(self.evidence_tokens_count)
        if self.token_budget_count > 0:
            metrics[f"{prefix}/avg_token_budget"] = self.token_budget_sum / float(self.token_budget_count)
            metrics[f"{prefix}/truncation_rate"] = self.truncated_count / float(self.token_budget_count)
        if self.k_visible_count > 0:
            metrics[f"{prefix}/avg_k_visible"] = self.k_visible_sum / float(self.k_visible_count)
        return metrics


def evaluate_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute answer metrics + semantic dissipation metrics (see docs/Semantic Dissipation.md)."""

    hits: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    total = 0

    semantic_global = _SemanticAccumulator()
    semantic_by_window: Dict[int, _SemanticAccumulator] = {}
    base_by_window: Dict[int, Dict[str, List[float]]] = {}

    for item in predictions:
        gold_answers: List[str] = item.get("answers") or []
        pred_raw: str = item.get("prediction") or ""
        preds = _extract_predictions(pred_raw)
        score = _score_answers(preds, gold_answers) if gold_answers else None

        if score is not None:
            total += 1
            hits.append(score["hit@1"])
            precisions.append(score["precision"])
            recalls.append(score["recall"])
            f1s.append(score["f1"])

        gt_edges = _as_int_list(item.get("gt_path_edge_local_ids") or item.get("gt_path_edges"))
        retrieved_edges = _as_int_list(item.get("retrieved_edge_ids"))
        visible_edges = _as_int_list(item.get("visible_edge_ids")) or retrieved_edges

        hit_set = None
        hit_vis = None
        if gt_edges:
            gt_set = set(gt_edges)
            hit_set = gt_set.issubset(set(retrieved_edges)) if retrieved_edges else False
            hit_vis = gt_set.issubset(set(visible_edges)) if visible_edges else False

        evidence_tokens = item.get("evidence_token_count")
        prompt_tokens = item.get("prompt_token_count")
        token_budget = item.get("token_budget")
        k_visible = len(visible_edges) if visible_edges else item.get("k_effective")
        truncated = bool(item.get("evidence_truncated", False))
        semantic_global.update(
            score=score["f1"] if score is not None else None,
            hit_set=hit_set,
            hit_vis=hit_vis,
            evidence_tokens=evidence_tokens if evidence_tokens is not None else None,
            prompt_tokens=prompt_tokens if prompt_tokens is not None else None,
            token_budget=int(token_budget) if token_budget is not None else None,
            k_visible=int(k_visible) if k_visible is not None else None,
            evidence_truncated=truncated,
        )

        window_k_raw = item.get("window_k")
        window_k: Optional[int] = None
        try:
            window_k = int(window_k_raw) if window_k_raw is not None else None
        except (TypeError, ValueError):
            window_k = None

        if window_k is not None:
            base_stats = base_by_window.setdefault(window_k, {"hits": [], "precisions": [], "recalls": [], "f1s": [], "total": 0})
            if score is not None:
                base_stats["hits"].append(score["hit@1"])
                base_stats["precisions"].append(score["precision"])
                base_stats["recalls"].append(score["recall"])
                base_stats["f1s"].append(score["f1"])
                base_stats["total"] += 1

            sem_acc = semantic_by_window.setdefault(window_k, _SemanticAccumulator())
            sem_acc.update(
                score=score["f1"] if score is not None else None,
                hit_set=hit_set,
                hit_vis=hit_vis,
                evidence_tokens=evidence_tokens if evidence_tokens is not None else None,
                prompt_tokens=prompt_tokens if prompt_tokens is not None else None,
                token_budget=int(token_budget) if token_budget is not None else None,
                k_visible=int(k_visible) if k_visible is not None else None,
                evidence_truncated=truncated,
            )

    metrics: Dict[str, float] = {
        "results/hit@1": _mean(hits),
        "results/macro_precision": _mean(precisions),
        "results/macro_recall": _mean(recalls),
        "results/macro_f1": _mean(f1s),
        "results/total": float(total),
    }

    metrics.update(semantic_global.finalize(prefix="semantic"))

    for k_int, stats in sorted(base_by_window.items()):
        metrics[f"results/window_{k_int}/hit@1"] = _mean(stats["hits"])
        metrics[f"results/window_{k_int}/macro_precision"] = _mean(stats["precisions"])
        metrics[f"results/window_{k_int}/macro_recall"] = _mean(stats["recalls"])
        metrics[f"results/window_{k_int}/macro_f1"] = _mean(stats["f1s"])
        metrics[f"results/window_{k_int}/total"] = float(stats["total"])

    for k_int, sem in sorted(semantic_by_window.items()):
        metrics.update(sem.finalize(prefix=f"semantic/window_{k_int}"))

    return metrics


__all__ = ["evaluate_predictions"]
