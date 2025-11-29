from __future__ import annotations

import re
import string
from typing import Dict, Iterable, List, Tuple


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
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    ans_lines = [ln for ln in lines if ln.lower().startswith("ans:")]
    if ans_lines:
        return ans_lines
    return lines[:1] if lines else []


def _match(pred: str, answer: str) -> bool:
    return _normalize(pred) == _normalize(answer) or _normalize(answer) in _normalize(pred)


def evaluate_predictions(
    predictions: List[Dict],
) -> Dict[str, float]:
    """Compute simple hit@1 / macro precision / recall / f1 against string answers."""
    hits = []
    precisions = []
    recalls = []
    f1s = []
    total = 0

    for item in predictions:
        gold_answers: List[str] = item.get("answers") or []
        pred_raw: str = item.get("prediction") or ""
        preds = _extract_predictions(pred_raw)
        if not gold_answers:
            continue
        total += 1

        # Hit@1
        if preds:
            hit = 1.0 if any(_match(preds[0], ans) for ans in gold_answers) else 0.0
        else:
            hit = 0.0
        hits.append(hit)

        # Precision / recall
        matched = 0
        remaining_preds = preds.copy()
        remaining_gold = gold_answers.copy()
        for g in gold_answers:
            for p in remaining_preds:
                if _match(p, g):
                    matched += 1
                    remaining_preds.remove(p)
                    break
        precision = matched / max(len(preds), 1)
        recall = matched / len(gold_answers)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    def _mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "results/hit@1": _mean(hits),
        "results/macro_precision": _mean(precisions),
        "results/macro_recall": _mean(recalls),
        "results/macro_f1": _mean(f1s),
        "results/total": float(total),
    }


__all__ = ["evaluate_predictions"]
