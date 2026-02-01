from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

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
    pred_map, output_meta = _load_predictions_and_meta(output_path, answer_key=answer_key)
    metrics: Dict[str, Any] = {
        "split": split,
        "provider": provider,
        "top_k": int(top_k),
        "input": str(input_path),
        "output": str(output_path),
        "llm/num_predictions": float(len(pred_map)),
    }
    metrics.update(output_meta)
    if not pred_map:
        return metrics

    sub_filter_ids = _maybe_load_sub_filter_ids(input_path)
    if sub_filter_ids is not None:
        metrics["sub_filter/path"] = str(_resolve_sub_filter_path(input_path))
        metrics["sub_filter/size"] = float(len(sub_filter_ids))

    text_acc = _F1Accumulator()
    ent_acc = _F1Accumulator()
    retrieval_acc = DualFlowEvalAccumulator(k_values=[int(top_k)])

    # Subset A (SubgraphRAG-style): answer entity appears anywhere in the retrieved context (a_entity_in_graph).
    text_acc_ingraph = _F1Accumulator()
    ent_acc_ingraph = _F1Accumulator()
    retrieval_acc_ingraph = DualFlowEvalAccumulator(k_values=[int(top_k)])
    ingraph_eval_samples = 0

    text_acc_sub = _F1Accumulator()
    ent_acc_sub = _F1Accumulator()
    retrieval_acc_sub = DualFlowEvalAccumulator(k_values=[int(top_k)])
    sub_eval_samples = 0
    full_eval_samples = 0

    for record in _iter_jsonl(input_path):
        sample_id = str(record.get(_FIELD_SAMPLE_ID) or "")
        if not sample_id or sample_id not in pred_map:
            continue
        full_eval_samples += 1
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

        ingraph = bool(gold_ent_set) and _answer_in_retrieved_context(record, answer_set=gold_ent_set, top_k=top_k)
        if ingraph:
            ingraph_eval_samples += 1
            if gold_text_set:
                text_acc_ingraph.update(pred=pred_text_set, gold=gold_text_set)
            if gold_ent_set:
                ent_acc_ingraph.update(pred=pred_ent_set, gold=gold_ent_set)
            retrieval_acc_ingraph.update_from_records([record])

        in_sub = sub_filter_ids is not None and sample_id in sub_filter_ids
        if in_sub:
            sub_eval_samples += 1
            if gold_text_set:
                text_acc_sub.update(pred=pred_text_set, gold=gold_text_set)
            if gold_ent_set:
                ent_acc_sub.update(pred=pred_ent_set, gold=gold_ent_set)
            retrieval_acc_sub.update_from_records([record])

    metrics.update(_finalize_f1(text_acc, prefix="llm/text"))
    metrics.update(_finalize_f1(ent_acc, prefix="llm/entity"))
    for key, value in retrieval_acc.finalize().items():
        metrics[f"retrieval/{key}"] = float(value)

    if ingraph_eval_samples > 0:
        metrics["ingraph/eval_samples"] = float(ingraph_eval_samples)
        metrics["ingraph/eval_ratio"] = float(ingraph_eval_samples) / float(max(full_eval_samples, 1))
        metrics.update(_finalize_f1(text_acc_ingraph, prefix="llm/ingraph/text"))
        metrics.update(_finalize_f1(ent_acc_ingraph, prefix="llm/ingraph/entity"))
        for key, value in retrieval_acc_ingraph.finalize().items():
            metrics[f"retrieval/ingraph/{key}"] = float(value)

    if sub_filter_ids is not None:
        metrics["sub_filter/eval_samples"] = float(sub_eval_samples)
        metrics["sub_filter/eval_ratio"] = float(sub_eval_samples) / float(max(full_eval_samples, 1))
        # Backward-compatible prefixes (historical).
        metrics.update(_finalize_f1(text_acc_sub, prefix="llm/sub/text"))
        metrics.update(_finalize_f1(ent_acc_sub, prefix="llm/sub/entity"))
        # Clearer alias: this "sub" is the dataset-level sub_filter, not the a_entity_in_graph subset.
        metrics.update(_finalize_f1(text_acc_sub, prefix="llm/sub_filter/text"))
        metrics.update(_finalize_f1(ent_acc_sub, prefix="llm/sub_filter/entity"))
        for key, value in retrieval_acc_sub.finalize().items():
            metrics[f"retrieval/sub/{key}"] = float(value)
            metrics[f"retrieval/sub_filter/{key}"] = float(value)
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


def _load_predictions_and_meta(path: Path, *, answer_key: str) -> tuple[Dict[str, str], Dict[str, float]]:
    if not path.exists():
        return {}, {}
    out: Dict[str, str] = {}
    num = 0
    empty = 0
    schema_total = 0
    schema_valid = 0
    retries_sum = 0
    forced_non_empty = 0
    abstain = 0
    for record in _iter_jsonl(path):
        sample_id = record.get(_FIELD_SAMPLE_ID)
        if not sample_id:
            continue
        num += 1
        answer = str(record.get(answer_key) or "")
        out[str(sample_id)] = answer
        if not answer.strip():
            empty += 1
        schema_flag = record.get("schema_valid")
        if isinstance(schema_flag, bool):
            schema_total += 1
            if schema_flag:
                schema_valid += 1
        retry = record.get("schema_retries")
        if retry is not None:
            try:
                retries_sum += int(retry)
            except Exception:
                pass
        if bool(record.get("forced_non_empty_answer", False)):
            forced_non_empty += 1
        abstain_reason = str(record.get("abstain_reason") or "").strip()
        if abstain_reason:
            abstain += 1

    meta: Dict[str, float] = {}
    denom = float(max(num, 1))
    meta["llm/empty_rate"] = float(empty) / denom
    meta["llm/forced_non_empty_rate"] = float(forced_non_empty) / denom
    meta["llm/abstain_rate"] = float(abstain) / denom
    if schema_total > 0:
        meta["llm/schema_invalid_rate"] = float(schema_total - schema_valid) / float(schema_total)
        meta["llm/schema_retries_mean"] = float(retries_sum) / float(schema_total)
    return out, meta


def _resolve_sub_filter_path(input_path: Path) -> Path:
    parts = input_path.resolve().parts
    # Expected layout:
    #   .../retrieval_dataset/<dataset_family>/artifacts/<dataset_name>/eval_dual_flow/<split>.jsonl
    family = None
    for idx, part in enumerate(parts):
        if part == "retrieval_dataset" and idx + 1 < len(parts):
            family = parts[idx + 1]
            break
    if not family:
        raise ValueError(f"Cannot infer dataset_family from input_path: {input_path}")
    base = Path(*parts[: parts.index("retrieval_dataset") + 2])  # .../retrieval_dataset/<family>
    return base / "normalized" / "sub_filter.json"


def _maybe_load_sub_filter_ids(input_path: Path) -> Optional[Set[str]]:
    try:
        path = _resolve_sub_filter_path(input_path)
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    sample_ids = payload.get("sample_ids")
    if not isinstance(sample_ids, list):
        return None
    return {str(x) for x in sample_ids if str(x).strip()}


def _answer_in_retrieved_context(record: Dict[str, Any], *, answer_set: Set[int], top_k: int) -> bool:
    if not answer_set:
        return False
    rollouts = record.get("rollouts") or []
    if not isinstance(rollouts, list) or not rollouts:
        return False
    selected = _select_rollouts(rollouts, top_k=int(top_k))
    context_nodes: set[int] = set()
    for rollout in selected:
        edges = rollout.get("edges") or []
        if isinstance(edges, list):
            for edge in edges:
                for key in ("src_entity_id", "dst_entity_id", "head_entity_id", "tail_entity_id"):
                    val = edge.get(key)
                    if val is None:
                        continue
                    try:
                        context_nodes.add(int(val))
                    except (TypeError, ValueError):
                        continue
        stop_node = rollout.get("stop_node_entity_id")
        if stop_node is not None:
            try:
                context_nodes.add(int(stop_node))
            except (TypeError, ValueError):
                pass
    return bool(context_nodes & answer_set)


def _resolve_gold_answer_texts(record: Dict[str, Any], *, answer_separator: str) -> List[str]:
    raw = record.get("answer_texts")
    if isinstance(raw, list):
        answers = [str(x) for x in raw if str(x).strip()]
        if len(answers) == 1:
            parsed = _maybe_parse_bracketed_answers(answers[0])
            if parsed:
                return parsed
        return answers
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
    # Robustness: some prompts annotate candidates like `Entity (support: k, evidence: ...)`.
    # If the model mistakenly copies the parenthetical metadata, strip it for matching.
    for marker in (" (support:", " (evidence:"):
        if marker in token:
            token = token.split(marker, 1)[0].rstrip()
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
        # Numeric answers (e.g., zip codes, years) are often entity *labels* rather than entity IDs.
        # Prefer matching against the candidate label map first.
        if token in label_map:
            out.update(label_map.get(token, set()))
            continue
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            try:
                out.add(int(token))
            except ValueError:
                pass
            continue
        out.update(label_map.get(token, set()))
    return out


_BRACKETED_ARRAY_RE = re.compile(r"^(\s*\[)(.*)(\]\s*)$", flags=re.DOTALL)
_QUOTED_TOKEN_RE = re.compile(r"(?:'([^']*)'|\"([^\"]*)\")", flags=re.DOTALL)


def _maybe_parse_bracketed_answers(text: str) -> List[str]:
    """Parse a list-like string into answer tokens.

    Supports JSON lists, python literal lists, and numpy-style array repr like:
      ['A' 'B' 'C']
    """

    raw = str(text or "").strip()
    if not raw:
        return []
    match = _BRACKETED_ARRAY_RE.match(raw)
    if match is None:
        return []

    # Fall back to extracting quoted substrings (numpy repr).
    tokens: List[str] = []
    for m in _QUOTED_TOKEN_RE.finditer(raw):
        token = m.group(1) if m.group(1) is not None else m.group(2)
        token = str(token or "").replace("\\'", "'").replace('\\"', '"').strip()
        if token:
            tokens.append(token)
    if tokens:
        return tokens

    # Try strict parsers last. (ast.literal_eval can incorrectly concatenate adjacent string literals
    # like ['A' 'B'] -> ['AB'], so we only use it when quoting extraction fails.)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(x) for x in parsed if str(x).strip()]

    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(x) for x in parsed if str(x).strip()]
    return []


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
