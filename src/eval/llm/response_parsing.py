from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from src.eval.llm.prompting import (
    PromptSpec,
    _FIELD_ABSTAIN_REASON,
    _FIELD_BEST_GUESS,
    _FIELD_EVIDENCE_TRAJECTORY_IDS,
    _FIELD_JUSTIFICATION,
    _PROMPT_MODE_SUBGRAPHRAG_ICL_DC,
    _remove_duplicates_preserve_order,
)

_ZERO = 0
_ONE = 1
_DEFAULT_ANSWER_KEY = "answer"


@dataclass(frozen=True)
class SchemaSpec:
    enabled: bool
    max_retries: int
    allow_coerce: bool
    max_retry_chars: int
    retry_message: str
    schema: Optional[Dict[str, Any]]
    schema_json: str
    validator: Optional[Any]


@dataclass(frozen=True)
class _ParsedResponse:
    answer: str
    payload: Optional[Dict[str, Any]]
    schema_valid: bool


class _ResponseBackend(Protocol):
    def generate(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]: ...


def _needs_subgraphrag_dc_retry(response: str) -> bool:
    raw = str(response or "").strip().lower()
    if not raw:
        return True
    if "ans:" not in raw:
        return True
    if "ans: not available" in raw:
        return True
    if "ans: no information available" in raw:
        return True
    return False


def _parse_response(response: str, prompt: PromptSpec) -> str:
    raw = (response or "").strip()
    if not raw:
        return ""
    if prompt.mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC:
        return _parse_subgraphrag_answer(raw, prompt)
    payload = _parse_json_payload(raw)
    if payload is not None:
        return _extract_answer_from_payload(payload, prompt, raw)

    fragment = _parse_json_fragment(raw)
    if fragment is not None:
        return _extract_answer_from_payload(fragment, prompt, raw)

    return raw


def _parse_subgraphrag_answer(response: str, prompt: PromptSpec) -> str:
    lines = _subgraphrag_get_pred_lines(str(response or ""))
    if not lines:
        return ""
    answers: List[str] = []
    for line in lines:
        lower = line.lower()
        idx = lower.find("ans:")
        token = line[idx + len("ans:") :] if idx >= _ZERO else line
        token = token.strip()
        if token:
            answers.append(token)
    return prompt.answer_separator.join(answers)


def _subgraphrag_get_pred_lines(prediction: str) -> List[str]:
    raw = str(prediction or "")
    candidates = [p for p in raw.split("\n") if "ans:" in p and "none" not in p.lower()]
    if candidates:
        candidates = [
            p
            for p in candidates
            if "ans: not available" not in p.lower()
            and "ans: no information available" not in p.lower()
        ]
    return _remove_duplicates_preserve_order(candidates)


def _parse_and_validate_response(
    response: str,
    prompt: PromptSpec,
    schema_spec: SchemaSpec,
) -> _ParsedResponse:
    raw = (response or "").strip()
    if not raw:
        return _ParsedResponse(
            answer="", payload=None, schema_valid=not schema_spec.enabled
        )
    payload = _parse_json_payload(raw)
    if payload is None:
        payload = _parse_json_fragment(raw)
    if not schema_spec.enabled:
        answer = _parse_response(raw, prompt)
        return _ParsedResponse(
            answer=answer,
            payload=payload if isinstance(payload, dict) else None,
            schema_valid=True,
        )
    if not isinstance(payload, dict):
        return _ParsedResponse(answer="", payload=None, schema_valid=False)
    normalized = _normalize_payload(
        payload, prompt, allow_coerce=schema_spec.allow_coerce
    )
    if normalized is None:
        return _ParsedResponse(answer="", payload=None, schema_valid=False)
    if not _validate_payload_schema(normalized, schema_spec):
        return _ParsedResponse(answer="", payload=normalized, schema_valid=False)
    answer = _extract_answer_from_payload(normalized, prompt, raw)
    return _ParsedResponse(answer=answer, payload=normalized, schema_valid=True)


def _normalize_payload(
    payload: Dict[str, Any], prompt: PromptSpec, *, allow_coerce: bool
) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    normalized: Dict[str, Any] = dict(payload)
    if prompt.answer_key not in normalized and _DEFAULT_ANSWER_KEY in normalized:
        normalized[prompt.answer_key] = normalized.get(_DEFAULT_ANSWER_KEY)
    if allow_coerce:
        normalized = _coerce_payload(normalized)
    if not prompt.allow_empty_answer:
        answer = str(normalized.get(prompt.answer_key) or "").strip()
        if not answer:
            best_guess = str(normalized.get(_FIELD_BEST_GUESS) or "").strip()
            if best_guess:
                normalized[prompt.answer_key] = best_guess
    return normalized


def _coerce_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    coerced: Dict[str, Any] = dict(payload)
    if _FIELD_EVIDENCE_TRAJECTORY_IDS in coerced:
        value = coerced.get(_FIELD_EVIDENCE_TRAJECTORY_IDS)
        if isinstance(value, list):
            cleaned: List[int] = []
            for item in value:
                try:
                    cleaned.append(int(item))
                except Exception:
                    continue
            coerced[_FIELD_EVIDENCE_TRAJECTORY_IDS] = cleaned
        else:
            try:
                coerced[_FIELD_EVIDENCE_TRAJECTORY_IDS] = [int(value)]
            except Exception:
                coerced[_FIELD_EVIDENCE_TRAJECTORY_IDS] = value
    return coerced


def _validate_payload_schema(payload: Dict[str, Any], schema_spec: SchemaSpec) -> bool:
    if schema_spec.validator is None:
        return True
    return bool(schema_spec.validator.is_valid(payload))


def _build_schema_retry_message(schema_spec: SchemaSpec, raw_response: str) -> str:
    snippet = raw_response or ""
    if (
        schema_spec.max_retry_chars > _ZERO
        and len(snippet) > schema_spec.max_retry_chars
    ):
        snippet = snippet[: schema_spec.max_retry_chars] + "..."
    try:
        return schema_spec.retry_message.format(
            schema=schema_spec.schema_json, response=snippet
        )
    except Exception:
        return schema_spec.retry_message


def _retry_schema_batch(
    *,
    backend: _ResponseBackend,
    batch_items: Sequence[Any],
    responses: List[str],
    parsed_list: List[_ParsedResponse],
    prompt_spec: PromptSpec,
    schema_spec: SchemaSpec,
) -> Tuple[List[str], List[_ParsedResponse], List[int]]:
    if not schema_spec.enabled or schema_spec.max_retries <= _ZERO:
        return responses, parsed_list, [0 for _ in parsed_list]
    retries = [0 for _ in parsed_list]
    current_messages = [list(item.messages) for item in batch_items]
    for _ in range(schema_spec.max_retries):
        invalid_idx = [
            idx for idx, parsed in enumerate(parsed_list) if not parsed.schema_valid
        ]
        if not invalid_idx:
            break
        retry_messages: List[List[Dict[str, str]]] = []
        for idx in invalid_idx:
            retry_message = _build_schema_retry_message(schema_spec, responses[idx])
            current_messages[idx] = current_messages[idx] + [
                {"role": "user", "content": retry_message}
            ]
            retry_messages.append(current_messages[idx])
        retry_outputs = backend.generate(retry_messages)
        for idx, output in zip(invalid_idx, retry_outputs):
            responses[idx] = (output or "").strip()
            parsed_list[idx] = _parse_and_validate_response(
                responses[idx], prompt_spec, schema_spec
            )
            retries[idx] += _ONE
    return responses, parsed_list, retries


def _parse_json_payload(text: str) -> Optional[Any]:
    sentinel = object()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = sentinel

    if payload is not sentinel:
        return payload

    extracted = _extract_json_object(text)
    if extracted is not None:
        return extracted

    return None


def _parse_json_fragment(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if "{" in stripped or "}" in stripped:
        return None
    if ":" not in stripped:
        return None
    candidate = "{" + stripped.rstrip(",") + "}"
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < _ZERO or end < _ZERO or end <= start:
        return None
    snippet = text[start : end + _ONE]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_answer_from_payload(payload: Any, prompt: PromptSpec, raw: str) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        answer_value = payload.get(prompt.answer_key)
        if answer_value is None and prompt.answer_key != _DEFAULT_ANSWER_KEY:
            answer_value = payload.get(_DEFAULT_ANSWER_KEY)
        if answer_value is None:
            return raw
        return _normalize_answer(answer_value, prompt.answer_separator)

    if isinstance(payload, str):
        nested = _parse_json_fragment(payload)
        if nested is not None:
            return _extract_answer_from_payload(nested, prompt, raw)
        nested_obj = _extract_json_object(payload)
        if nested_obj is not None:
            return _extract_answer_from_payload(nested_obj, prompt, raw)

    return _normalize_answer(payload, prompt.answer_separator)


def _normalize_answer(answer: Any, separator: str) -> str:
    if answer is None:
        return ""
    if isinstance(answer, list):
        return separator.join(str(item).strip() for item in answer if str(item).strip())
    return str(answer).strip()


def _extract_structured_model_fields(raw_response: str) -> Dict[str, Any]:
    payload = _parse_json_payload((raw_response or "").strip())
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Any] = {}
    if _FIELD_EVIDENCE_TRAJECTORY_IDS in payload:
        ids = payload.get(_FIELD_EVIDENCE_TRAJECTORY_IDS)
        if isinstance(ids, list):
            cleaned: List[int] = []
            for item in ids:
                try:
                    cleaned.append(int(item))
                except Exception:
                    continue
            out[_FIELD_EVIDENCE_TRAJECTORY_IDS] = cleaned
        else:
            try:
                out[_FIELD_EVIDENCE_TRAJECTORY_IDS] = [int(ids)]
            except Exception:
                out[_FIELD_EVIDENCE_TRAJECTORY_IDS] = []
    for key in (_FIELD_ABSTAIN_REASON, _FIELD_BEST_GUESS, _FIELD_JUSTIFICATION):
        if key in payload:
            out[key] = str(payload.get(key) or "").strip()
    return out
