from __future__ import annotations

import json
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


class PredictionParseError(ValueError):
    pass


def _extract_predictions(raw: Any) -> List[str]:
    if raw is None:
        raise PredictionParseError("prediction is None")
    text = str(raw).strip()
    if not text:
        raise PredictionParseError("prediction is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PredictionParseError(
            f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(parsed, dict):
        raise PredictionParseError(f"JSON root must be an object with 'answers', got {type(parsed).__name__}")
    if "answers" not in parsed:
        raise PredictionParseError("missing required key 'answers'")
    answers = parsed["answers"]
    if not isinstance(answers, list):
        raise PredictionParseError(f"'answers' must be a list, got {type(answers).__name__}")

    normalized: List[str] = []
    for idx, ans in enumerate(answers):
        if not isinstance(ans, str):
            raise PredictionParseError(f"'answers[{idx}]' must be a string, got {type(ans).__name__}")
        text_ans = ans.strip()
        if not text_ans:
            raise PredictionParseError(f"'answers[{idx}]' is empty")
        normalized.append(text_ans)
    return normalized


def _match(pred: str, answer: str) -> bool:
    return _normalize(pred) == _normalize(answer) or _normalize(answer) in _normalize(pred)


def _score_match(preds: List[str], gold_answers: List[str]) -> Dict[str, float]:
    if not gold_answers:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
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
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _dedupe_answers(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for val in values:
        norm = _normalize(val)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(val)
    return out


def _score_answers(preds: List[str], gold_answers: List[str]) -> Dict[str, float]:
    if not gold_answers:
        return {
            "hit@1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "set_precision": 0.0,
            "set_recall": 0.0,
            "set_f1": 0.0,
            "set_exact": 0.0,
        }

    if preds:
        hit = 1.0 if any(_match(preds[0], ans) for ans in gold_answers) else 0.0
    else:
        hit = 0.0

    list_scores = _score_match(preds, gold_answers)
    pred_set = _dedupe_answers(preds)
    gold_set = _dedupe_answers(gold_answers)
    set_scores = _score_match(pred_set, gold_set)

    pred_norm = {_normalize(p) for p in preds if _normalize(p)}
    gold_norm = {_normalize(g) for g in gold_answers if _normalize(g)}
    set_exact = 1.0 if pred_norm == gold_norm else 0.0

    return {
        "hit@1": float(hit),
        "precision": list_scores["precision"],
        "recall": list_scores["recall"],
        "f1": list_scores["f1"],
        "set_precision": set_scores["precision"],
        "set_recall": set_scores["recall"],
        "set_f1": set_scores["f1"],
        "set_exact": float(set_exact),
    }


def _as_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values if v is not None]
    try:
        return [int(values)]
    except (TypeError, ValueError):
        return []


def _require_bool(value: Any, *, name: str, sample_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be bool/0/1 for id={sample_id}, got {value!r}")


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
    set_precisions: List[float] = []
    set_recalls: List[float] = []
    set_f1s: List[float] = []
    set_exacts: List[float] = []
    total = 0

    semantic_global = _SemanticAccumulator()
    semantic_by_window: Dict[int, _SemanticAccumulator] = {}
    base_by_window: Dict[int, Dict[str, List[float]]] = {}

    for item in predictions:
        sample_id = str(item.get("id", "unknown"))
        gold_raw = item.get("answers")
        if gold_raw is None:
            raise ValueError(f"missing gold answers for id={sample_id}")
        if not isinstance(gold_raw, list):
            raise ValueError(f"gold answers must be a list for id={sample_id}, got {type(gold_raw).__name__}")
        if not gold_raw:
            raise ValueError(f"gold answers list is empty for id={sample_id}")
        gold_answers: List[str] = []
        for idx, ans in enumerate(gold_raw):
            if not isinstance(ans, str):
                raise ValueError(f"gold answers[{idx}] must be string for id={sample_id}, got {type(ans).__name__}")
            text_ans = ans.strip()
            if not text_ans:
                raise ValueError(f"gold answers[{idx}] is empty for id={sample_id}")
            gold_answers.append(text_ans)

        pred_raw = item.get("prediction")
        try:
            preds = _extract_predictions(pred_raw)
        except PredictionParseError as exc:
            raise ValueError(f"prediction parse failed for id={sample_id}: {exc}") from exc
        score = _score_answers(preds, gold_answers)

        if score is not None:
            total += 1
            hits.append(score["hit@1"])
            precisions.append(score["precision"])
            recalls.append(score["recall"])
            f1s.append(score["f1"])
            set_precisions.append(score["set_precision"])
            set_recalls.append(score["set_recall"])
            set_f1s.append(score["set_f1"])
            set_exacts.append(score["set_exact"])

        if "hit_set" not in item or "hit_vis" not in item:
            raise ValueError(f"missing hit_set/hit_vis for id={sample_id}")
        hit_set = _require_bool(item.get("hit_set"), name="hit_set", sample_id=sample_id)
        hit_vis = _require_bool(item.get("hit_vis"), name="hit_vis", sample_id=sample_id)

        if "visible_edge_ids" not in item:
            raise ValueError(f"missing visible_edge_ids for id={sample_id}")
        visible_edges = _as_int_list(item.get("visible_edge_ids"))

        evidence_tokens = item.get("evidence_token_count")
        prompt_tokens = item.get("prompt_token_count")
        token_budget = item.get("token_budget")
        k_visible = len(visible_edges)
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
            base_stats = base_by_window.setdefault(
                window_k,
                {
                    "hits": [],
                    "precisions": [],
                    "recalls": [],
                    "f1s": [],
                    "set_precisions": [],
                    "set_recalls": [],
                    "set_f1s": [],
                    "set_exacts": [],
                    "total": 0,
                },
            )
            if score is not None:
                base_stats["hits"].append(score["hit@1"])
                base_stats["precisions"].append(score["precision"])
                base_stats["recalls"].append(score["recall"])
                base_stats["f1s"].append(score["f1"])
                base_stats["set_precisions"].append(score["set_precision"])
                base_stats["set_recalls"].append(score["set_recall"])
                base_stats["set_f1s"].append(score["set_f1"])
                base_stats["set_exacts"].append(score["set_exact"])
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
        "results/answer_set_precision": _mean(set_precisions),
        "results/answer_set_recall": _mean(set_recalls),
        "results/answer_set_f1": _mean(set_f1s),
        "results/answer_set_exact": _mean(set_exacts),
        "results/total": float(total),
    }

    metrics.update(semantic_global.finalize(prefix="semantic"))

    for k_int, stats in sorted(base_by_window.items()):
        metrics[f"results/window_{k_int}/hit@1"] = _mean(stats["hits"])
        metrics[f"results/window_{k_int}/macro_precision"] = _mean(stats["precisions"])
        metrics[f"results/window_{k_int}/macro_recall"] = _mean(stats["recalls"])
        metrics[f"results/window_{k_int}/macro_f1"] = _mean(stats["f1s"])
        metrics[f"results/window_{k_int}/answer_set_precision"] = _mean(stats["set_precisions"])
        metrics[f"results/window_{k_int}/answer_set_recall"] = _mean(stats["set_recalls"])
        metrics[f"results/window_{k_int}/answer_set_f1"] = _mean(stats["set_f1s"])
        metrics[f"results/window_{k_int}/answer_set_exact"] = _mean(stats["set_exacts"])
        metrics[f"results/window_{k_int}/total"] = float(stats["total"])

    for k_int, sem in sorted(semantic_by_window.items()):
        metrics.update(sem.finalize(prefix=f"semantic/window_{k_int}"))

    return metrics


__all__ = ["evaluate_predictions"]
