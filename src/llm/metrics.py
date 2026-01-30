from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set

from src.metrics.dual_flow import DualFlowEvalAccumulator

_ZERO = 0
_ONE = 1
_NEG_INF = float("-inf")

_FIELD_SAMPLE_ID = "sample_id"
_FIELD_SCORE = "score"
_FIELD_ROLLOUT_INDEX = "rollout_index"


def write_llm_metrics(
    *,
    input_path: Path,
    output_path: Path,
    output_dir: Path,
    split: str,
    provider: str,
    top_k: int,
    answer_key: str,
    answer_separator: str,
    metrics_filename_template: Optional[str] = None,
) -> Path:
    template = str(metrics_filename_template or "{split}_k{k}_{provider}.metrics.json")
    metrics_path = output_dir / template.format(split=split, k=int(top_k), provider=provider)
    metrics = compute_llm_metrics(
        input_path=input_path,
        output_path=output_path,
        split=split,
        provider=provider,
        top_k=top_k,
        answer_key=answer_key,
        answer_separator=answer_separator,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def compute_llm_metrics(
    *,
    input_path: Path,
    output_path: Path,
    split: str,
    provider: str,
    top_k: int,
    answer_key: str,
    answer_separator: str,
) -> Dict[str, Any]:
    pred_map = _load_predictions(output_path, answer_key=answer_key)
    metrics: Dict[str, Any] = {
        "split": split,
        "provider": provider,
        "top_k": int(top_k),
        "input": str(input_path),
        "output": str(output_path),
        "llm/num_predictions": float(len(pred_map)),
    }
    if not pred_map:
        return metrics

    text_acc = _F1Accumulator()
    ent_acc = _F1Accumulator()
    retrieval_acc = DualFlowEvalAccumulator(k_values=[int(top_k)])
    for record in _iter_jsonl(input_path):
        sample_id = str(record.get(_FIELD_SAMPLE_ID) or "")
        if not sample_id or sample_id not in pred_map:
            continue
        pred_text = pred_map.get(sample_id, "")
        pred_text_set = _parse_answer_set(pred_text, answer_separator=answer_separator)

        gold_texts = _resolve_gold_answer_texts(record, answer_separator=answer_separator)
        gold_text_set = _parse_answer_set_from_list(gold_texts)
        if gold_text_set:
            text_acc.update(pred=pred_text_set, gold=gold_text_set)

        gold_ent_set = _parse_gold_entity_ids(record)
        if gold_ent_set:
            pred_ent_set = _parse_pred_entity_ids(record, pred_text_set=pred_text_set, top_k=top_k)
            ent_acc.update(pred=pred_ent_set, gold=gold_ent_set)

        retrieval_acc.update_from_records([record])

    metrics.update(_finalize_f1(text_acc, prefix="llm/text"))
    metrics.update(_finalize_f1(ent_acc, prefix="llm/entity"))
    for key, value in retrieval_acc.finalize().items():
        metrics[f"retrieval/{key}"] = float(value)
    return metrics


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_predictions(path: Path, *, answer_key: str) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for record in _iter_jsonl(path):
        sample_id = record.get(_FIELD_SAMPLE_ID)
        if not sample_id:
            continue
        out[str(sample_id)] = str(record.get(answer_key) or "")
    return out


def _resolve_gold_answer_texts(record: Dict[str, Any], *, answer_separator: str) -> List[str]:
    raw = record.get("answer_texts")
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    text = str(record.get("answer_text") or "").strip()
    if not text:
        return []
    return list(_parse_answer_set(text, answer_separator=answer_separator))


def _normalize_answer_token(text: str) -> str:
    token = str(text or "").strip()
    if not token:
        return ""
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        token = token[1:-1].strip()
    token = " ".join(token.split())
    return token.casefold()


def _parse_answer_set(text: str, *, answer_separator: str) -> Set[str]:
    raw = str(text or "").strip()
    if not raw:
        return set()
    sep = str(answer_separator)
    if sep and sep in raw:
        parts = raw.split(sep)
    elif "|" in sep and "|" in raw:
        parts = raw.split("|")
    elif "\n" in raw:
        parts = raw.splitlines()
    else:
        parts = [raw]
    return {t for t in (_normalize_answer_token(p) for p in parts) if t}


def _parse_answer_set_from_list(values: List[str]) -> Set[str]:
    return {t for t in (_normalize_answer_token(v) for v in values) if t}


def _parse_gold_entity_ids(record: Dict[str, Any]) -> Set[int]:
    raw = record.get("answer_entity_ids") or []
    if not isinstance(raw, list):
        return set()
    out: set[int] = set()
    for val in raw:
        try:
            out.add(int(val))
        except (TypeError, ValueError):
            continue
    return out


def _parse_pred_entity_ids(record: Dict[str, Any], *, pred_text_set: Set[str], top_k: int) -> Set[int]:
    label_map = _build_candidate_label_map(record, top_k=top_k)
    out: set[int] = set()
    for token in pred_text_set:
        if not token:
            continue
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            try:
                out.add(int(token))
            except ValueError:
                pass
            continue
        out.update(label_map.get(token, set()))
    return out


def _build_candidate_label_map(record: Dict[str, Any], *, top_k: int) -> Dict[str, Set[int]]:
    rollouts = record.get("rollouts") or []
    if not isinstance(rollouts, list) or not rollouts:
        return {}
    selected = _select_rollouts(rollouts, top_k=top_k)
    out: Dict[str, Set[int]] = {}
    for rollout in selected:
        edges = rollout.get("edges") or []
        if not isinstance(edges, list):
            continue
        for edge in edges:
            _update_label_map(out, edge.get("src_text"), edge.get("src_entity_id"))
            _update_label_map(out, edge.get("dst_text"), edge.get("dst_entity_id"))
            _update_label_map(out, edge.get("head_text"), edge.get("head_entity_id"))
            _update_label_map(out, edge.get("tail_text"), edge.get("tail_entity_id"))
    return out


def _update_label_map(out: Dict[str, Set[int]], text: Any, entity_id: Any) -> None:
    token = _normalize_answer_token(str(text or ""))
    if not token:
        return
    try:
        ent = int(entity_id)
    except (TypeError, ValueError):
        return
    out.setdefault(token, set()).add(ent)


def _select_rollouts(rollouts: Sequence[Dict[str, Any]], *, top_k: int) -> List[Dict[str, Any]]:
    sorted_rollouts = sorted(
        rollouts,
        key=lambda r: (float(r.get(_FIELD_SCORE, _NEG_INF)), int(r.get(_FIELD_ROLLOUT_INDEX, _ZERO))),
        reverse=True,
    )
    return sorted_rollouts[: int(top_k)]


@dataclass
class _F1Accumulator:
    samples: int = 0
    hit_samples: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sum_f1: float = 0.0

    def update(self, *, pred: Set[Any], gold: Set[Any]) -> None:
        self.samples += 1
        overlap = pred & gold
        tp = len(overlap)
        fp = len(pred - gold)
        fn = len(gold - pred)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        if tp > 0:
            self.hit_samples += 1
        precision = float(tp) / float(len(pred)) if pred else 0.0
        recall = float(tp) / float(len(gold)) if gold else 0.0
        denom = precision + recall
        f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0
        self.sum_f1 += float(f1)


def _finalize_f1(acc: _F1Accumulator, *, prefix: str) -> Dict[str, float]:
    if acc.samples <= 0:
        return {f"{prefix}/samples": 0.0, f"{prefix}/hit": 0.0, f"{prefix}/micro_f1": 0.0, f"{prefix}/macro_f1": 0.0}
    tp = float(acc.tp)
    fp = float(acc.fp)
    fn = float(acc.fn)
    denom = (2.0 * tp) + fp + fn
    micro_f1 = (2.0 * tp / denom) if denom > 0.0 else 0.0
    macro_f1 = float(acc.sum_f1) / float(max(acc.samples, 1))
    hit_rate = float(acc.hit_samples) / float(max(acc.samples, 1))
    return {
        f"{prefix}/samples": float(acc.samples),
        f"{prefix}/hit": float(hit_rate),
        f"{prefix}/micro_f1": float(micro_f1),
        f"{prefix}/macro_f1": float(macro_f1),
    }


__all__ = ["compute_llm_metrics", "write_llm_metrics"]
