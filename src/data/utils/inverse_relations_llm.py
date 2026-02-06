from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class InverseRelationLLMError(RuntimeError):
    pass


_UNCERTAIN_PATTERNS_DEFAULT = (
    "not sure",
    "i'm not sure",
    "unsure",
    "unclear",
    "unknown",
    "cannot determine",
    "can't determine",
    "do not know",
    "don't know",
    "unable to",
    "无法",
    "不清楚",
    "不知道",
    "无法确定",
    "无法判断",
    "难以判断",
    "不确定",
)
_UNCERTAIN_FIELDS = ("forward_label", "forward_text", "inverse", "inverse_text")
_UNCERTAIN_EXAMPLE_LIMIT = 5
_INVERSE_RELATIONS_LIST_KEY = "inverse_relations"

_DEFAULT_SYSTEM = (
    "You are a knowledge graph relation expert. "
    "For each Freebase-style relation id, produce a short human-readable forward predicate "
    "and its inverse predicate, plus one-sentence explanations. "
    "Use the relation id verbatim in the `forward` field. "
    "Return strict JSON only."
)
_JSON_EXAMPLE = (
    "["
    "{"
    "\"forward\":\"/people/person/place_of_birth\","
    "\"forward_label\":\"place of birth\","
    "\"forward_text\":\"X was born in Y.\","
    "\"inverse\":\"people born here\","
    "\"inverse_text\":\"Y is the birthplace of X.\""
    "}"
    "]"
)


def _chunk(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield list(values[i : i + size])


def _extract_json_payload(text: str) -> Any:
    if not text:
        raise ValueError("empty LLM output")
    s = text.strip()
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    start = s.find("[")
    end = s.rfind("]")
    if start >= 0 and end > start:
        return json.loads(s[start : end + 1])
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start : end + 1])
    raise ValueError("no JSON object found in LLM output")


def _format_examples(examples: Sequence[str]) -> str:
    if not examples:
        return ""
    lines = [f"- {ex}" for ex in examples if str(ex).strip()]
    if not lines:
        return ""
    return "Examples (head -> tail):\n" + "\n".join(lines) + "\n"


def _build_user_prompt(
    relations: Sequence[str],
    *,
    attempt: int,
    examples_by_relation: Optional[Mapping[str, Sequence[str]]],
) -> str:
    header = (
        "Return a JSON array. Each element must contain fields: "
        "`forward` (relation id), `forward_label` (short predicate), "
        "`forward_text` (one-sentence explanation), `inverse` (short predicate), "
        "`inverse_text` (one-sentence explanation).\n"
        "Use the same order as the provided relations and do not include any extra text.\n"
        "Do not echo the input list.\n"
    )
    if attempt > 0:
        header = (
            "Your previous response was invalid. Return ONLY a JSON array with no extra text.\n"
            f"Example:\n{_JSON_EXAMPLE}\n"
        )
    lines: List[str] = []
    for rel in relations:
        lines.append(f"- {rel}")
        if examples_by_relation is None:
            continue
        examples = examples_by_relation.get(rel)
        if examples:
            lines.append("  Examples (head -> tail):")
            lines.extend(f"    - {ex}" for ex in examples if str(ex).strip())
    return header + "Relations:\n" + "\n".join(lines)


def _build_single_relation_prompt(
    relation: str,
    *,
    attempt: int,
    examples: Optional[Sequence[str]],
) -> str:
    header = (
        "Return a JSON object with fields: "
        "`forward` (relation id), `forward_label` (short predicate), "
        "`forward_text` (one-sentence explanation), `inverse` (short predicate), "
        "`inverse_text` (one-sentence explanation).\n"
        "Do not include any extra text.\n"
    )
    if attempt > 0:
        header = (
            "Your previous response was invalid. Return ONLY a JSON object with no extra text.\n"
            f"Example:\n{_JSON_EXAMPLE}\n"
        )
    examples_block = _format_examples(list(examples or []))
    return header + f"Relation:\n- {relation}\n" + examples_block


def _extract_first_entry(payload_obj: object) -> Dict[str, Any]:
    if isinstance(payload_obj, dict) and _INVERSE_RELATIONS_LIST_KEY in payload_obj:
        payload_obj = payload_obj[_INVERSE_RELATIONS_LIST_KEY]
    if isinstance(payload_obj, dict):
        return payload_obj
    if isinstance(payload_obj, list):
        for item in payload_obj:
            if isinstance(item, dict):
                return item
    raise InverseRelationLLMError("LLM output does not contain a JSON object entry")

def _normalize_entry(entry: Mapping[str, Any], relation_id: str) -> Dict[str, str]:
    forward = str(entry.get("forward") or relation_id).strip()
    forward_label = str(entry.get("forward_label") or "").strip()
    forward_text = str(entry.get("forward_text") or "").strip()
    inverse = str(entry.get("inverse") or "").strip()
    inverse_text = str(entry.get("inverse_text") or "").strip()
    if not forward_label:
        forward_label = forward
    if not inverse:
        raise ValueError(f"missing inverse for relation {relation_id!r}")
    return {
        "forward": forward,
        "forward_label": forward_label,
        "forward_text": forward_text,
        "inverse": inverse,
        "inverse_text": inverse_text,
    }


def _resolve_uncertain_patterns(llm_cfg: Mapping[str, Any]) -> List[str]:
    raw = llm_cfg.get("uncertain_patterns")
    if raw is None:
        return list(_UNCERTAIN_PATTERNS_DEFAULT)
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw if str(item).strip()]
    return [str(raw)]


def _is_uncertain(text: str, patterns: Sequence[str]) -> bool:
    if not text or not str(text).strip():
        return True
    lowered = str(text).lower()
    for pattern in patterns:
        if pattern and str(pattern).lower() in lowered:
            return True
    return False


def _validate_entries(
    entries: Sequence[Dict[str, str]],
    *,
    patterns: Sequence[str],
) -> None:
    errors: List[str] = []
    for entry in entries:
        rel = entry.get("forward", "")
        for field in _UNCERTAIN_FIELDS:
            if _is_uncertain(entry.get(field, ""), patterns):
                errors.append(f"{rel}:{field}")
                break
    if errors:
        preview = ", ".join(errors[:_UNCERTAIN_EXAMPLE_LIMIT])
        raise InverseRelationLLMError(f"LLM produced uncertain/empty fields for {len(errors)} entries: {preview}")


def generate_inverse_relations_llm(
    *,
    relations: Sequence[str],
    llm_cfg: Mapping[str, Any],
    examples_by_relation: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Dict[str, str]]:
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("vllm is required for inverse_relations auto-generation.") from exc

    model = llm_cfg.get("model")
    if not model:
        raise ValueError("inverse_relations.llm.model must be set for LLM auto-generation.")
    batch_size = int(llm_cfg.get("batch_size", 16))
    temperature = float(llm_cfg.get("temperature", 0.1))
    top_p = float(llm_cfg.get("top_p", 0.9))
    max_tokens = int(llm_cfg.get("max_tokens", 256))
    max_retries = int(llm_cfg.get("max_retries", 4))
    max_model_len = llm_cfg.get("max_model_len")
    tensor_parallel_size = int(llm_cfg.get("tensor_parallel_size", 1))
    dtype = llm_cfg.get("dtype")
    trust_remote_code = bool(llm_cfg.get("trust_remote_code", False))

    llm_kwargs = {
        "model": str(model),
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = int(max_model_len)
    if dtype not in (None, "", "null", "None"):
        llm_kwargs["dtype"] = dtype
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    system = str(llm_cfg.get("system_prompt") or _DEFAULT_SYSTEM).strip()
    use_chat = bool(llm_cfg.get("use_chat", True))
    per_relation_prompt = bool(llm_cfg.get("per_relation_prompt", True))
    disallow_uncertain = bool(llm_cfg.get("disallow_uncertain", True))
    patterns = _resolve_uncertain_patterns(llm_cfg)
    raw_example_limit = llm_cfg.get("example_limit", 3)
    example_limit = 3 if raw_example_limit is None else int(raw_example_limit)
    if example_limit < 0:
        example_limit = 0
    if examples_by_relation is not None and example_limit == 0:
        examples_by_relation = None
    if examples_by_relation is not None and example_limit > 0:
        examples_by_relation = {
            rel: list(examples)[:example_limit]
            for rel, examples in examples_by_relation.items()
            if examples
        }
    show_progress = bool(llm_cfg.get("show_progress", True))
    try:
        from tqdm import tqdm as _tqdm  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        _tqdm = None
    results: Dict[str, Dict[str, str]] = {}

    def _generate_payload(relations_chunk: Sequence[str]) -> List[Dict[str, Any]]:
        if not per_relation_prompt:
            payload_obj = None
            last_text = ""
            last_error: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                user_prompt = _build_user_prompt(
                    relations_chunk,
                    attempt=attempt,
                    examples_by_relation=examples_by_relation,
                )
                if use_chat and hasattr(llm, "chat"):
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ]
                    outputs = llm.chat([messages], sampling_params=sampling, use_tqdm=False)
                    if not outputs:
                        last_error = InverseRelationLLMError("empty LLM output")
                        continue
                    last_text = outputs[0].outputs[0].text if outputs[0].outputs else ""
                else:
                    prompt = f"{system}\n{user_prompt}"
                    outputs = llm.generate([prompt], sampling)
                    if not outputs:
                        last_error = InverseRelationLLMError("empty LLM output")
                        continue
                    last_text = outputs[0].outputs[0].text if outputs[0].outputs else ""
                try:
                    payload_obj = _extract_json_payload(last_text)
                except Exception as exc:
                    last_error = exc
                    continue
                break
            if payload_obj is None:
                snippet = (last_text or "").strip().replace("\n", " ")
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                raise InverseRelationLLMError(f"LLM output was not valid JSON: {snippet}") from last_error
            if isinstance(payload_obj, dict) and _INVERSE_RELATIONS_LIST_KEY in payload_obj:
                payload_obj = payload_obj[_INVERSE_RELATIONS_LIST_KEY]
            if isinstance(payload_obj, dict):
                payload_obj = [payload_obj]
            if not isinstance(payload_obj, list):
                raise InverseRelationLLMError("LLM output is not a JSON array")
            return [item for item in payload_obj if isinstance(item, dict)]

        def _examples_for(rel: str) -> Optional[Sequence[str]]:
            if examples_by_relation is None:
                return None
            return examples_by_relation.get(rel)

        messages_batch = []
        for rel in relations_chunk:
            prompt = _build_single_relation_prompt(rel, attempt=0, examples=_examples_for(rel))
            messages_batch.append(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
        if use_chat and hasattr(llm, "chat"):
            outputs = llm.chat(messages_batch, sampling_params=sampling, use_tqdm=False)
        else:
            prompts = [
                f"{system}\n{_build_single_relation_prompt(rel, attempt=0, examples=_examples_for(rel))}"
                for rel in relations_chunk
            ]
            outputs = llm.generate(prompts, sampling)
        payloads: List[Dict[str, Any]] = []
        for rel, out in zip(relations_chunk, outputs):
            text = out.outputs[0].text if out.outputs else ""
            try:
                payload_obj = _extract_json_payload(text)
                payloads.append(_extract_first_entry(payload_obj))
            except Exception:
                last_text = ""
                last_error: Optional[Exception] = None
                for attempt in range(1, max_retries + 1):
                    retry_prompt = _build_single_relation_prompt(
                        rel,
                        attempt=attempt,
                        examples=_examples_for(rel),
                    )
                    if use_chat and hasattr(llm, "chat"):
                        retry_outputs = llm.chat(
                            [[{"role": "system", "content": system}, {"role": "user", "content": retry_prompt}]],
                            sampling_params=sampling,
                            use_tqdm=False,
                        )
                        last_text = (
                            retry_outputs[0].outputs[0].text
                            if retry_outputs and retry_outputs[0].outputs
                            else ""
                        )
                    else:
                        retry_outputs = llm.generate([f"{system}\n{retry_prompt}"], sampling)
                        last_text = (
                            retry_outputs[0].outputs[0].text
                            if retry_outputs and retry_outputs[0].outputs
                            else ""
                        )
                    try:
                        payload_obj = _extract_json_payload(last_text)
                        payloads.append(_extract_first_entry(payload_obj))
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    snippet = (last_text or "").strip().replace("\n", " ")
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    raise InverseRelationLLMError(f"LLM output was not valid JSON: {snippet}") from last_error
        return payloads

    total_batches = int(math.ceil(len(relations) / float(batch_size))) if batch_size > 0 else 0
    chunk_iter = enumerate(_chunk(list(relations), batch_size))
    if show_progress and _tqdm is not None:
        chunk_iter = _tqdm(chunk_iter, total=total_batches, desc="inverse_relations", unit="batch")
    for batch_idx, chunk_relations in chunk_iter:
        if show_progress and _tqdm is None:
            print(f"[inverse_relations] batch {batch_idx + 1}/{total_batches}")
        try:
            payload = _generate_payload(chunk_relations)
        except InverseRelationLLMError:
            if len(chunk_relations) <= 1:
                raise
            payload = []
            for rel in chunk_relations:
                payload.extend(_generate_payload([rel]))
        if per_relation_prompt and len(payload) != len(chunk_relations):
            if len(chunk_relations) <= 1:
                raise InverseRelationLLMError("LLM output size mismatch for per-relation prompt.")
            payload = []
            for rel in chunk_relations:
                payload.extend(_generate_payload([rel]))
        if per_relation_prompt:
            for rel, item in zip(chunk_relations, payload):
                clean = dict(item)
                clean.pop("forward", None)
                entry = _normalize_entry(clean, str(rel))
                if disallow_uncertain:
                    _validate_entries([entry], patterns=patterns)
                results[str(rel)] = entry
        else:
            for item in payload:
                forward = str(item.get("forward") or "").strip()
                if not forward:
                    continue
                entry = _normalize_entry(item, forward)
                if disallow_uncertain:
                    _validate_entries([entry], patterns=patterns)
                results[forward] = entry

    missing = [rel for rel in relations if rel not in results]
    if missing:
        preview = ", ".join(missing[:5])
        raise InverseRelationLLMError(f"LLM failed to generate inverse labels for {len(missing)} relations: {preview}")

    entries = [results[rel] for rel in relations]
    if disallow_uncertain:
        _validate_entries(entries, patterns=patterns)
    return entries


__all__ = ["generate_inverse_relations_llm", "InverseRelationLLMError"]
