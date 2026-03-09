from __future__ import annotations

import ast
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

_ZERO = 0
_ONE = 1
_TWO = 2.0
_PERCENT = 100.0
_HAL_BAD_OUT_GRAPH_PENALTY = -1.5
_HAL_BAD_IN_GRAPH_PENALTY = -1.0
_HAL_GOOD_CORRECT_REWARD = 1.0
_HAL_GOOD_WRONG_PENALTY = -1.0
_HAL_BAD_NO_ANS_REWARD = 1.0
_HAL_SCORE_SHIFT = 1.5
_HAL_SCORE_SCALE = 1.0 + _HAL_SCORE_SHIFT
_NO_ANS_MARKERS = ("ans: not available", "ans: no information available")
_MAX_ERROR_IDS_TO_SHOW = 5

_FIELD_SAMPLE_ID = "sample_id"


@dataclass(frozen=True)
class _EvalSample:
    sample_id: str
    question: str
    answers: List[str]
    pred_text: str
    pred_lines: List[str]
    double_check: bool
    a_entity_in_graph: Optional[bool]
    trajectories: Optional[List[Dict[str, Any]]]


@dataclass(frozen=True)
class _EvalResult:
    hit_at_1: float
    hit: float
    f1: float
    precision: float
    recall: float
    exact_match: bool
    totally_wrong: bool
    matched: int
    num_pred: int
    num_answer: int
    no_ans: bool


@dataclass
class _EvalAccumulator:
    samples: int = 0
    hit_at_1_sum: float = float(_ZERO)
    hit_sum: float = float(_ZERO)
    f1_sum: float = float(_ZERO)
    precision_sum: float = float(_ZERO)
    recall_sum: float = float(_ZERO)
    exact_match_count: int = 0
    totally_wrong_count: int = 0
    total_pred: int = 0
    total_answer: int = 0
    total_match: int = 0
    no_ans_count: int = 0
    hal_score_sum: float = float(_ZERO)
    hal_stats: Optional[Dict[str, int]] = None


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
    input_labels_path: Optional[Path] = None,
) -> tuple[Path, Dict[str, Any]]:
    template = str(metrics_filename_template or "{split}_k{k}_{provider}.metrics.json")
    metrics_path = output_dir / template.format(
        split=split, k=int(top_k), provider=provider
    )
    metrics = compute_llm_metrics(
        input_path=input_path,
        input_labels_path=input_labels_path,
        output_path=output_path,
        split=split,
        provider=provider,
        top_k=top_k,
        answer_key=answer_key,
        answer_separator=answer_separator,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path, metrics


def compute_llm_metrics(
    *,
    input_path: Path,
    output_path: Path,
    split: str,
    provider: str,
    top_k: int,
    answer_key: str,
    answer_separator: str,
    input_labels_path: Optional[Path] = None,
) -> Dict[str, Any]:
    pred_map = _load_predictions(output_path, answer_key=answer_key)
    label_map = _load_label_records(input_labels_path)
    metrics: Dict[str, Any] = {
        "split": split,
        "provider": provider,
        "top_k": int(top_k),
        "input": str(input_path),
        "input_labels": str(input_labels_path) if input_labels_path is not None else "",
        "output": str(output_path),
        "llm/num_predictions": float(len(pred_map)),
        "llm/run/top_k": int(top_k),
    }
    if not pred_map:
        return metrics

    full_acc = _EvalAccumulator(hal_stats=_init_hal_stats())
    sub_acc = _EvalAccumulator(hal_stats=_init_hal_stats())
    input_sample_ids: Set[str] = set()
    trajectory_count_sum = 0
    trajectory_count_min: Optional[int] = None
    trajectory_count_max = 0
    for record in _iter_jsonl(input_path):
        sample_id = str(record.get(_FIELD_SAMPLE_ID) or "")
        if not sample_id:
            raise ValueError(f"Input JSONL record missing sample_id in {input_path}.")
        if sample_id in input_sample_ids:
            raise ValueError(
                f"Duplicate sample_id in input JSONL {input_path}: {sample_id!r}"
            )
        input_sample_ids.add(sample_id)
        if sample_id not in pred_map:
            continue
        merged = _merge_with_label_record(record, label_map)
        sample = _build_eval_sample(merged, pred_map, answer_separator=answer_separator)
        if sample is None:
            raise ValueError(
                "Missing or invalid gold labels for predicted sample_id="
                f"{sample_id!r}. Provide a valid labels sidecar via llm.input_labels_path."
            )
        result = _evaluate_sample(sample)
        trajectory_count = len(sample.trajectories or [])
        trajectory_count_sum += trajectory_count
        trajectory_count_max = max(trajectory_count_max, trajectory_count)
        if trajectory_count_min is None:
            trajectory_count_min = trajectory_count
        else:
            trajectory_count_min = min(trajectory_count_min, trajectory_count)
        _accumulate(full_acc, sample, result, include_hal=True)
        if sample.a_entity_in_graph is True:
            _accumulate(sub_acc, sample, result, include_hal=True)
    unknown_pred_ids = sorted(set(pred_map.keys()) - input_sample_ids)
    if unknown_pred_ids:
        preview = unknown_pred_ids[:_MAX_ERROR_IDS_TO_SHOW]
        raise ValueError(
            f"Predictions contain sample_id values not found in input JSONL {input_path}: "
            f"{preview} (showing up to {_MAX_ERROR_IDS_TO_SHOW})."
        )
    metrics.update(_finalize_scope(full_acc, prefix="llm/subgraphrag/full"))
    metrics.update(_finalize_scope(sub_acc, prefix="llm/subgraphrag/sub"))
    if full_acc.samples > _ZERO:
        metrics["llm/input/trajectory_count_mean"] = float(
            trajectory_count_sum
        ) / float(full_acc.samples)
        metrics["llm/input/trajectory_count_min"] = int(trajectory_count_min or 0)
        metrics["llm/input/trajectory_count_max"] = int(trajectory_count_max)
    if full_acc.hal_stats is not None:
        metrics.update(
            _format_hal_stats(full_acc.hal_stats, prefix="llm/subgraphrag/full/stats")
        )
    if sub_acc.hal_stats is not None:
        metrics.update(
            _format_hal_stats(sub_acc.hal_stats, prefix="llm/subgraphrag/sub/stats")
        )
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
        sample_id = str(record.get(_FIELD_SAMPLE_ID) or "")
        if not sample_id:
            raise ValueError(f"Prediction record missing sample_id in {path}.")
        if sample_id in out:
            raise ValueError(
                f"Duplicate sample_id in prediction JSONL {path}: {sample_id!r}"
            )
        out[sample_id] = str(record.get(answer_key) or "")
    return out


def _load_label_records(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for record in _iter_jsonl(path):
        sample_id = str(record.get(_FIELD_SAMPLE_ID) or "")
        if not sample_id:
            raise ValueError(f"Label record missing sample_id in {path}.")
        if sample_id in out:
            raise ValueError(
                f"Duplicate sample_id in label JSONL {path}: {sample_id!r}"
            )
        out[sample_id] = dict(record)
    return out


def _merge_with_label_record(
    record: Dict[str, Any], labels: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    sample_id = str(record.get(_FIELD_SAMPLE_ID) or "")
    if not sample_id:
        return record
    label_record = labels.get(sample_id)
    if label_record is None:
        return record
    merged = dict(record)
    for key, value in label_record.items():
        if key == _FIELD_SAMPLE_ID:
            continue
        if key == "trajectories":
            continue
        merged[key] = value
    return merged


def _build_eval_sample(
    record: Dict[str, Any],
    pred_map: Dict[str, str],
    *,
    answer_separator: str,
) -> Optional[_EvalSample]:
    sample_id = str(record.get(_FIELD_SAMPLE_ID) or "")
    if not sample_id:
        return None
    if sample_id not in pred_map:
        return None
    question = str(record.get("question_text") or record.get("question") or "")
    answers = _resolve_gold_answer_texts(record, answer_separator=answer_separator)
    if not answers:
        return None
    answers = _remove_duplicates_preserve_order(answers)
    answers = sorted(answers, key=len, reverse=True)
    answers = _subgraphrag_year_fix(answers, question)
    pred_text = pred_map.get(sample_id, "") or ""
    pred_lines = _subgraphrag_get_pred_lines(pred_text)
    if not pred_lines:
        pred_lines = _subgraphrag_get_pred_lines_from_answer(
            pred_text, answer_separator=answer_separator
        )
    double_check = _subgraphrag_is_double_check(question)
    a_entity_in_graph = _resolve_a_entity_in_graph(record)
    trajectories = record.get("trajectories")
    trajectories = trajectories if isinstance(trajectories, list) else None
    return _EvalSample(
        sample_id=sample_id,
        question=question,
        answers=answers,
        pred_text=str(pred_text or ""),
        pred_lines=pred_lines,
        double_check=double_check,
        a_entity_in_graph=a_entity_in_graph,
        trajectories=trajectories,
    )


def _resolve_a_entity_in_graph(record: Dict[str, Any]) -> Optional[bool]:
    value = record.get("a_entity_in_graph")
    if isinstance(value, bool):
        return value
    return None


def _evaluate_sample(sample: _EvalSample) -> _EvalResult:
    hit_at_1 = float(
        _subgraphrag_hit_at_1(sample.pred_lines, sample.answers, sample.double_check)
    )
    hit = float(
        _subgraphrag_eval_hit(sample.pred_text, sample.answers, sample.double_check)
    )
    matched, num_pred, num_answer = _subgraphrag_match_count(
        sample.pred_lines, sample.answers, sample.double_check
    )
    precision = float(matched) / float(num_pred) if num_pred > _ZERO else float(_ZERO)
    recall = float(matched) / float(num_answer) if num_answer > _ZERO else float(_ZERO)
    f1 = _safe_f1(precision, recall)
    no_ans = _subgraphrag_no_answer(sample.pred_text, sample.pred_lines)
    return _EvalResult(
        hit_at_1=hit_at_1,
        hit=hit,
        f1=f1,
        precision=precision,
        recall=recall,
        exact_match=f1 == float(_ONE),
        totally_wrong=recall == float(_ZERO),
        matched=matched,
        num_pred=num_pred,
        num_answer=num_answer,
        no_ans=no_ans,
    )


def _accumulate(
    acc: _EvalAccumulator,
    sample: _EvalSample,
    result: _EvalResult,
    *,
    include_hal: bool,
) -> None:
    acc.samples += _ONE
    acc.hit_at_1_sum += result.hit_at_1
    acc.hit_sum += result.hit
    acc.f1_sum += result.f1
    acc.precision_sum += result.precision
    acc.recall_sum += result.recall
    acc.total_pred += int(result.num_pred)
    acc.total_answer += int(result.num_answer)
    acc.total_match += int(result.matched)
    if result.exact_match:
        acc.exact_match_count += _ONE
    if result.totally_wrong:
        acc.totally_wrong_count += _ONE
    if result.no_ans:
        acc.no_ans_count += _ONE
    if include_hal:
        if acc.hal_stats is None:
            acc.hal_stats = _init_hal_stats()
        entities = _extract_retrieved_entities(sample.trajectories)
        hal_score, stats = _subgraphrag_hal_score(
            predictions=sample.pred_lines,
            answers=sample.answers,
            double_check=sample.double_check,
            good_sample=bool(sample.a_entity_in_graph),
            no_ans=result.no_ans,
            subgraph_entities=entities,
            stats=acc.hal_stats,
        )
        acc.hal_score_sum += float(hal_score)
        acc.hal_stats = stats


def _finalize_scope(acc: _EvalAccumulator, *, prefix: str) -> Dict[str, Any]:
    if acc.samples <= _ZERO:
        return {
            f"{prefix}/hit@1": float(_ZERO),
            f"{prefix}/hit": float(_ZERO),
            f"{prefix}/macro_f1": float(_ZERO),
            f"{prefix}/macro_precision": float(_ZERO),
            f"{prefix}/macro_recall": float(_ZERO),
            f"{prefix}/exact_match": float(_ZERO),
            f"{prefix}/totally_wrong": float(_ZERO),
            f"{prefix}/micro_f1": float(_ZERO),
            f"{prefix}/micro_precision": float(_ZERO),
            f"{prefix}/micro_recall": float(_ZERO),
            f"{prefix}/total_cnt": 0,
            f"{prefix}/no_ans_cnt": 0,
            f"{prefix}/no_ans_ratio": float(_ZERO),
            f"{prefix}/hal_score": float(_ZERO),
        }
    denom = float(acc.samples)
    micro_precision = (
        float(acc.total_match) / float(acc.total_pred)
        if acc.total_pred > _ZERO
        else float(_ZERO)
    )
    micro_recall = (
        float(acc.total_match) / float(acc.total_answer)
        if acc.total_answer > _ZERO
        else float(_ZERO)
    )
    micro_f1 = _safe_f1(micro_precision, micro_recall)
    if acc.hal_stats is None:
        hal_scaled = float(_ZERO)
    else:
        hal_avg = float(acc.hal_score_sum) / denom
        hal_scaled = ((hal_avg + _HAL_SCORE_SHIFT) / _HAL_SCORE_SCALE) * _PERCENT
    return {
        f"{prefix}/hit@1": (acc.hit_at_1_sum * _PERCENT) / denom,
        f"{prefix}/hit": (acc.hit_sum * _PERCENT) / denom,
        f"{prefix}/macro_f1": (acc.f1_sum * _PERCENT) / denom,
        f"{prefix}/macro_precision": (acc.precision_sum * _PERCENT) / denom,
        f"{prefix}/macro_recall": (acc.recall_sum * _PERCENT) / denom,
        f"{prefix}/exact_match": (float(acc.exact_match_count) * _PERCENT) / denom,
        f"{prefix}/totally_wrong": (float(acc.totally_wrong_count) * _PERCENT) / denom,
        f"{prefix}/micro_f1": micro_f1,
        f"{prefix}/micro_precision": micro_precision,
        f"{prefix}/micro_recall": micro_recall,
        f"{prefix}/total_cnt": int(acc.samples),
        f"{prefix}/no_ans_cnt": int(acc.no_ans_count),
        f"{prefix}/no_ans_ratio": float(acc.no_ans_count) / denom,
        f"{prefix}/hal_score": float(hal_scaled),
    }


def _format_hal_stats(stats: Dict[str, int], *, prefix: str) -> Dict[str, Any]:
    return {f"{prefix}/{key}": float(value) for key, value in stats.items()}


def _resolve_gold_answer_texts(
    record: Dict[str, Any], *, answer_separator: str
) -> List[str]:
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
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
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


_SUBGRAPHRAG_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_SUBGRAPHRAG_PAD_RE = re.compile(r"\b(<pad>)\b", flags=re.IGNORECASE)
_SUBGRAPHRAG_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_SUBGRAPHRAG_DOUBLE_CHECK_KEYWORDS = (
    "when",
    "what year",
    "which year",
    "where",
    "sport",
    "what countr",
    "language",
    "nba finals",
    "world series",
)


def _subgraphrag_normalize(text: str) -> str:
    s = str(text or "").lower()
    s = s.translate(_SUBGRAPHRAG_PUNCT_TABLE)
    s = _SUBGRAPHRAG_ARTICLES_RE.sub(" ", s)
    s = _SUBGRAPHRAG_PAD_RE.sub(" ", s)
    return " ".join(s.split())


def _subgraphrag_match(s1: str, s2: str) -> bool:
    left = _subgraphrag_normalize(s1)
    right = _subgraphrag_normalize(s2)
    return bool(right) and right in left


def _subgraphrag_get_pred_lines(prediction: str) -> List[str]:
    raw = str(prediction or "")
    candidates = [p for p in raw.split("\n") if "ans:" in p and "none" not in p.lower()]
    if candidates:
        lowered = [p.lower() for p in candidates]
        candidates = [
            p
            for p, lo in zip(candidates, lowered)
            if all(marker not in lo for marker in _NO_ANS_MARKERS)
        ]
    return _remove_duplicates_preserve_order(candidates)


def _subgraphrag_get_pred_lines_from_answer(
    answer: str, *, answer_separator: str
) -> List[str]:
    tokens = _split_prediction_tokens(answer, answer_separator=answer_separator)
    return [f"ans: {token}" for token in tokens]


def _split_prediction_tokens(prediction: str, *, answer_separator: str) -> List[str]:
    raw = str(prediction or "").strip()
    if not raw:
        return []
    sep = str(answer_separator or "")
    if sep and sep in raw:
        parts = raw.split(sep)
    elif "\n" in raw:
        parts = raw.splitlines()
    else:
        parts = [raw]
    tokens: List[str] = []
    for part in parts:
        token = str(part or "").strip()
        if not token:
            continue
        lower = token.lower()
        idx = lower.find("ans:")
        if idx >= _ZERO:
            token = token[idx + len("ans:") :].strip()
        if not token or _is_no_ans_token(token):
            continue
        tokens.append(token)
    return _remove_duplicates_preserve_order(tokens)


def _subgraphrag_year_fix(gold: List[str], question: str) -> List[str]:
    q = str(question or "").lower()
    if "when" not in q and "what year" not in q:
        return gold
    out: List[str] = []
    for answer in gold:
        raw = str(answer or "").strip()
        if "-" in raw:
            head = raw.split("-", 1)[0]
            if head.isdigit():
                raw = head
        if raw:
            out.append(raw)
    return out


def _subgraphrag_is_double_check(question: str) -> bool:
    q = str(question or "").lower()
    return any(keyword in q for keyword in _SUBGRAPHRAG_DOUBLE_CHECK_KEYWORDS)


def _safe_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom == float(_ZERO):
        return float(_ZERO)
    return (_TWO * precision * recall) / denom


def _extract_ans_tokens(prediction: str) -> List[str]:
    tokens: List[str] = []
    for line in str(prediction or "").splitlines():
        lower = line.lower()
        idx = lower.find("ans:")
        if idx < _ZERO:
            continue
        token = line[idx + len("ans:") :].strip()
        if token:
            tokens.append(token)
    return tokens


def _is_no_ans_token(token: str) -> bool:
    lowered = str(token or "").strip().lower()
    if not lowered:
        return True
    if lowered in {"none", "n/a", "na"}:
        return True
    return any(marker in f"ans: {lowered}" for marker in _NO_ANS_MARKERS)


def _subgraphrag_no_answer(prediction: str, pred_lines: Sequence[str]) -> bool:
    if pred_lines:
        return False
    tokens = _extract_ans_tokens(prediction)
    if not tokens:
        return True
    return all(_is_no_ans_token(token) for token in tokens)


def _subgraphrag_hit_at_1(
    prediction: Sequence[str], answer: List[str], double_check: bool
) -> int:
    if not prediction:
        return 0
    top = str(prediction[0])
    for a in answer:
        if _subgraphrag_match(top, a):
            return 1
        if double_check and _subgraphrag_match(a, top.split("ans:")[-1].strip()):
            return 1
    return 0


def _subgraphrag_match_count(
    prediction: Sequence[str],
    answer: List[str],
    double_check: bool,
) -> Tuple[int, int, int]:
    prediction_sorted = sorted([str(p) for p in prediction], key=len, reverse=True)
    matched = 0
    for a in answer:
        for pred in list(prediction_sorted):
            if _subgraphrag_pred_matches(pred, a, double_check):
                matched += 1
                prediction_sorted.remove(pred)
                break
    return matched, len(prediction), len(answer)


def _subgraphrag_pred_matches(pred: str, answer: str, double_check: bool) -> bool:
    if _subgraphrag_match(pred, answer):
        return True
    if not double_check:
        return False
    tail = pred.split("ans:")[-1].strip()
    return _subgraphrag_match(answer, tail) or _subgraphrag_match(answer, pred)


def _init_hal_stats() -> Dict[str, int]:
    return {
        "g_no_ans": 0,
        "g_c": 0,
        "g_w": 0,
        "b_no_ans": 0,
        "b_in_graph": 0,
        "b_out_graph_c": 0,
        "b_out_graph_w": 0,
        "total_ans": 0,
        "total_g_samples": 0,
        "total_b_samples": 0,
        "total_samples": 0,
        "total_g_ans": 0,
        "total_b_ans": 0,
        "g_c_out_graph": 0,
        "g_w_out_graph": 0,
        "g_c_in_graph": 0,
        "g_w_in_graph": 0,
    }


def _extract_retrieved_entities(
    trajectories: Optional[List[Dict[str, Any]]],
) -> List[str]:
    if not trajectories:
        return []
    entities: List[str] = []
    for trajectory in trajectories:
        edges = trajectory.get("edges")
        if not isinstance(edges, list):
            continue
        for edge in edges:
            src = _resolve_edge_value(
                edge, ("src_text", "head_text", "src_entity_id", "head_entity_id")
            )
            dst = _resolve_edge_value(
                edge, ("dst_text", "tail_text", "dst_entity_id", "tail_entity_id")
            )
            if src:
                entities.append(str(src))
            if dst:
                entities.append(str(dst))
    return _remove_duplicates_preserve_order(entities)


def _resolve_edge_value(edge: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = edge.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _pred_in_entities(pred: str, subgraph_entities: Sequence[str]) -> bool:
    pred_lower = str(pred or "").lower()
    tail = pred_lower.split("ans:")[-1].strip()
    for ent in subgraph_entities:
        ent_lower = str(ent or "").lower()
        if tail and tail in ent_lower:
            return True
        if ent_lower and ent_lower in pred_lower:
            return True
    return False


def _subgraphrag_hal_score(
    *,
    predictions: Sequence[str],
    answers: List[str],
    double_check: bool,
    good_sample: bool,
    no_ans: bool,
    subgraph_entities: Sequence[str],
    stats: Dict[str, int],
) -> Tuple[float, Dict[str, int]]:
    stats["total_samples"] += _ONE
    if good_sample:
        return _subgraphrag_hal_score_good(
            predictions=predictions,
            answers=answers,
            double_check=double_check,
            no_ans=no_ans,
            subgraph_entities=subgraph_entities,
            stats=stats,
        )
    return _subgraphrag_hal_score_bad(
        predictions=predictions,
        answers=answers,
        double_check=double_check,
        no_ans=no_ans,
        subgraph_entities=subgraph_entities,
        stats=stats,
    )


def _subgraphrag_hal_score_good(
    *,
    predictions: Sequence[str],
    answers: List[str],
    double_check: bool,
    no_ans: bool,
    subgraph_entities: Sequence[str],
    stats: Dict[str, int],
) -> Tuple[float, Dict[str, int]]:
    stats["total_g_samples"] += _ONE
    if no_ans:
        stats["g_no_ans"] += _ONE
        return float(_ZERO), stats
    answer_pool = list(answers)
    score = float(_ZERO)
    for pred in predictions:
        stats["total_ans"] += _ONE
        stats["total_g_ans"] += _ONE
        matched = False
        for ans in list(answer_pool):
            if _subgraphrag_pred_matches(str(pred), ans, double_check):
                score += _HAL_GOOD_CORRECT_REWARD
                stats["g_c"] += _ONE
                matched = True
                answer_pool.remove(ans)
                if _pred_in_entities(str(pred), subgraph_entities):
                    stats["g_c_in_graph"] += _ONE
                else:
                    stats["g_c_out_graph"] += _ONE
                break
        if not matched:
            score += _HAL_GOOD_WRONG_PENALTY
            stats["g_w"] += _ONE
            if _pred_in_entities(str(pred), subgraph_entities):
                stats["g_w_in_graph"] += _ONE
            else:
                stats["g_w_out_graph"] += _ONE
    denom = float(len(predictions)) if predictions else float(_ONE)
    return score / denom, stats


def _subgraphrag_hal_score_bad(
    *,
    predictions: Sequence[str],
    answers: List[str],
    double_check: bool,
    no_ans: bool,
    subgraph_entities: Sequence[str],
    stats: Dict[str, int],
) -> Tuple[float, Dict[str, int]]:
    stats["total_b_samples"] += _ONE
    if no_ans:
        stats["b_no_ans"] += _ONE
        return _HAL_BAD_NO_ANS_REWARD, stats
    answer_pool = list(answers)
    score = float(_ZERO)
    for pred in predictions:
        stats["total_ans"] += _ONE
        stats["total_b_ans"] += _ONE
        if _pred_in_entities(str(pred), subgraph_entities):
            score += _HAL_BAD_IN_GRAPH_PENALTY
            stats["b_in_graph"] += _ONE
            continue
        score += _HAL_BAD_OUT_GRAPH_PENALTY
        matched = False
        for ans in list(answer_pool):
            if _subgraphrag_pred_matches(str(pred), ans, double_check):
                stats["b_out_graph_c"] += _ONE
                matched = True
                answer_pool.remove(ans)
                break
        if not matched:
            stats["b_out_graph_w"] += _ONE
    denom = float(len(predictions)) if predictions else float(_ONE)
    return score / denom, stats


def _subgraphrag_eval_hit(
    prediction: str, answer: List[str], double_check: bool
) -> int:
    """SubgraphRAG's Hit metric (see SubgraphRAG/reason/metrics/evaluate_results.py)."""

    pred_text = str(prediction or "")
    for a in answer:
        if "ans:" in pred_text:
            all_pred = _subgraphrag_get_pred_lines(pred_text)
            for each_pred in all_pred:
                if _subgraphrag_match(each_pred, a):
                    return 1
                if double_check and _subgraphrag_match(
                    a, each_pred.split("ans:")[-1].strip()
                ):
                    return 1
        else:
            if _subgraphrag_match(pred_text, a):
                return 1
            if double_check:
                for each_pred in pred_text.split("\n"):
                    if _subgraphrag_match(a, each_pred):
                        return 1
    return 0


def _remove_duplicates_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = ["compute_llm_metrics", "write_llm_metrics"]
