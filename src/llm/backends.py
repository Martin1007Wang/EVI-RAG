from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ZERO = 0
_ONE = 1

_DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"
_DEFAULT_VLLM_PRETRIM_TO_BUDGET = True
_DEFAULT_VLLM_BUDGET_MARGIN = 0

_VLLM_PROMPT_TOO_LONG_MARKERS = (
    "longer than the maximum model length",
    "maximum model length",
)


class _LLMBackend:
    def __init__(self, generate_fn):
        self._generate_fn = generate_fn

    def generate(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        return self._generate_fn(messages_batch)


def _build_backend(provider: str, provider_cfg: Any, llm_cfg: Any) -> _LLMBackend:
    del llm_cfg
    name = provider.lower()
    if name == "vllm":
        return _LLMBackend(_build_vllm_generate(provider_cfg))
    if name == "openai":
        return _LLMBackend(_build_openai_generate(provider_cfg))
    raise ValueError(f"Unsupported provider: {provider}")


def _is_vllm_prompt_too_long_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if not text:
        return False
    if "prompt" not in text:
        return False
    return any(marker in text for marker in _VLLM_PROMPT_TOO_LONG_MARKERS)


def _infer_vllm_max_model_len(llm: Any) -> Optional[int]:
    engine = getattr(llm, "llm_engine", None)
    cfg = getattr(engine, "model_config", None) if engine is not None else None
    value = getattr(cfg, "max_model_len", None) if cfg is not None else None
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _count_vllm_chat_tokens(
    tokenizer: Any, messages: List[Dict[str, str]]
) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    except Exception:
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return None
        try:
            tokens = tokenizer.encode(rendered)
        except Exception:
            return None
    if hasattr(tokens, "shape"):
        try:
            return int(tokens.shape[-1])
        except Exception:
            return None
    try:
        return int(len(tokens))
    except Exception:
        return None


def _find_last_user_message(messages: Sequence[Dict[str, str]]) -> Optional[int]:
    for idx in range(len(messages) - 1, -1, -1):
        role = str(messages[idx].get("role") or "").strip().lower()
        if role == "user":
            return idx
    return None


def _parse_subgraphrag_user_content(content: str) -> Optional[Tuple[List[str], str]]:
    lines = [str(line) for line in str(content or "").splitlines()]
    try:
        triplets_idx = lines.index("Triplets:")
    except ValueError:
        return None
    try:
        question_idx = lines.index("Question:")
    except ValueError:
        return None
    if question_idx <= triplets_idx:
        return None
    raw_triplets = lines[triplets_idx + 1 : question_idx]
    triplets: List[str] = []
    for line in raw_triplets:
        stripped = str(line or "").strip()
        if stripped == "(none)":
            continue
        if not stripped:
            # Preserve a single blank-line separator between retrieval elements.
            if triplets and triplets[-1] != "":
                triplets.append("")
            continue
        triplets.append(stripped)
    while triplets and triplets[-1] == "":
        triplets.pop()
    question = lines[question_idx + 1].strip() if question_idx + 1 < len(lines) else ""
    return triplets, question


def _format_subgraphrag_user_content(triplets: Sequence[str], question: str) -> str:
    triplet_lines = list(triplets) if triplets else ["(none)"]
    lines = [
        "Triplets:",
        *triplet_lines,
        "",
        "",
        "Question:",
        str(question or "").strip(),
    ]
    return "\n".join(lines)


def _replace_message_content(
    messages: Sequence[Dict[str, str]], idx: int, content: str
) -> List[Dict[str, str]]:
    out = [dict(m) for m in messages]
    out[idx] = dict(out[idx])
    out[idx]["content"] = content
    return out


def _trim_subgraphrag_messages_to_budget(
    messages: Sequence[Dict[str, str]],
    *,
    user_idx: int,
    triplets: List[str],
    question: str,
    tokenizer: Any,
    budget: int,
) -> List[Dict[str, str]]:
    if budget <= _ZERO:
        return [dict(m) for m in messages]

    def _split_groups(lines: Sequence[str]) -> List[List[str]]:
        groups: List[List[str]] = []
        current: List[str] = []
        for line in lines:
            if not str(line).strip():
                if current:
                    groups.append(current)
                    current = []
                continue
            current.append(str(line))
        if current:
            groups.append(current)
        return groups

    def _flatten_groups(groups: Sequence[Sequence[str]]) -> List[str]:
        flattened: List[str] = []
        for idx, group in enumerate(groups):
            flattened.extend([str(x) for x in group if str(x).strip()])
            if idx < len(groups) - 1:
                flattened.append("")
        return flattened

    groups = _split_groups(triplets)
    lo = 0
    hi = len(groups)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        content = _format_subgraphrag_user_content(
            _flatten_groups(groups[:mid]), question
        )
        candidate = _replace_message_content(messages, user_idx, content)
        tokens = _count_vllm_chat_tokens(tokenizer, candidate)
        if tokens is not None and tokens <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    final_content = _format_subgraphrag_user_content(
        _flatten_groups(groups[:best]), question
    )
    return _replace_message_content(messages, user_idx, final_content)


def _trim_last_user_suffix_to_budget(
    messages: Sequence[Dict[str, str]],
    *,
    user_idx: int,
    tokenizer: Any,
    budget: int,
) -> List[Dict[str, str]]:
    if budget <= _ZERO:
        return [dict(m) for m in messages]
    original = str(messages[user_idx].get("content") or "")
    if not original:
        return [dict(m) for m in messages]
    lo = 1
    hi = len(original)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        suffix = original[-mid:]
        candidate = _replace_message_content(messages, user_idx, suffix)
        tokens = _count_vllm_chat_tokens(tokenizer, candidate)
        if tokens is not None and tokens <= budget:
            best = suffix
            lo = mid + 1
        else:
            hi = mid - 1
    if not best:
        best = original[-1:]
    return _replace_message_content(messages, user_idx, best)


def _trim_messages_to_vllm_budget(
    messages: Sequence[Dict[str, str]],
    *,
    tokenizer: Any,
    budget: int,
) -> List[Dict[str, str]]:
    copied = [dict(m) for m in messages]
    current_tokens = _count_vllm_chat_tokens(tokenizer, copied)
    if current_tokens is None or current_tokens <= budget:
        return copied
    user_idx = _find_last_user_message(copied)
    if user_idx is None:
        return copied
    parsed = _parse_subgraphrag_user_content(str(copied[user_idx].get("content") or ""))
    if parsed is None:
        return _trim_last_user_suffix_to_budget(
            copied, user_idx=user_idx, tokenizer=tokenizer, budget=budget
        )
    triplets, question = parsed
    trimmed = _trim_subgraphrag_messages_to_budget(
        copied,
        user_idx=user_idx,
        triplets=triplets,
        question=question,
        tokenizer=tokenizer,
        budget=budget,
    )
    tokens = _count_vllm_chat_tokens(tokenizer, trimmed)
    if tokens is None or tokens <= budget:
        return trimmed
    return _trim_last_user_suffix_to_budget(
        trimmed, user_idx=user_idx, tokenizer=tokenizer, budget=budget
    )


def _build_vllm_generate(provider_cfg: Any):
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("vllm is required for provider=vllm.") from exc
    model = str(provider_cfg.get("model"))
    if not model:
        raise ValueError("llm.vllm.model must be set.")
    tensor_parallel_size = int(provider_cfg.get("tensor_parallel_size") or _ONE)
    max_model_len = provider_cfg.get("max_model_len")
    max_model_len_int = int(max_model_len) if max_model_len is not None else None
    seed = provider_cfg.get("seed")
    seed_int = int(seed) if seed is not None else None
    if seed_int is None:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len_int,
        )
    else:
        try:
            llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len_int,
                seed=seed_int,
            )
        except TypeError:
            llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len_int,
            )
    if max_model_len_int is None:
        max_model_len_int = _infer_vllm_max_model_len(llm)
    tokenizer = None
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = None
    max_tokens = int(provider_cfg.get("max_tokens") or _ONE)
    pretrim_to_budget = bool(
        provider_cfg.get("pretrim_to_budget", _DEFAULT_VLLM_PRETRIM_TO_BUDGET)
    )
    budget_margin = int(provider_cfg.get("budget_margin", _DEFAULT_VLLM_BUDGET_MARGIN))
    if budget_margin < _ZERO:
        raise ValueError("llm.vllm.budget_margin must be >= 0.")
    sampling_kwargs: Dict[str, Any] = {
        "temperature": float(provider_cfg.get("temperature", _ZERO)),
        "max_tokens": max_tokens,
        "top_p": float(provider_cfg.get("top_p", _ONE)),
    }
    if seed_int is not None:
        sampling_kwargs["seed"] = seed_int
    try:
        sampling_params = SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("seed", None)
        sampling_params = SamplingParams(**sampling_kwargs)

    def _generate(messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        chat_batch = messages_batch
        if (
            pretrim_to_budget
            and tokenizer is not None
            and max_model_len_int is not None
        ):
            budget = int(max_model_len_int) - int(max_tokens) - int(budget_margin)
            if budget > _ZERO:
                chat_batch = [
                    _trim_messages_to_vllm_budget(
                        messages, tokenizer=tokenizer, budget=budget
                    )
                    for messages in messages_batch
                ]
        try:
            outputs = llm.chat(
                chat_batch, sampling_params=sampling_params, use_tqdm=False
            )
        except ValueError as exc:
            if (
                tokenizer is None
                or max_model_len_int is None
                or not _is_vllm_prompt_too_long_error(exc)
            ):
                raise
            budget = int(max_model_len_int) - int(max_tokens) - int(budget_margin)
            trimmed_batch = [
                _trim_messages_to_vllm_budget(
                    messages, tokenizer=tokenizer, budget=budget
                )
                for messages in messages_batch
            ]
            outputs = llm.chat(
                trimmed_batch, sampling_params=sampling_params, use_tqdm=False
            )
        return [out.outputs[0].text if out.outputs else "" for out in outputs]

    return _generate


def _build_openai_generate(provider_cfg: Any):
    model = str(provider_cfg.get("model"))
    if not model:
        raise ValueError("llm.openai.model must be set.")
    temperature = float(provider_cfg.get("temperature", _ZERO))
    max_tokens = int(provider_cfg.get("max_tokens") or _ONE)
    timeout = provider_cfg.get("timeout_sec")
    max_retries = int(provider_cfg.get("max_retries") or _ZERO)
    backoff_seconds = float(provider_cfg.get("backoff_seconds") or _ZERO)
    backoff_base = float(provider_cfg.get("backoff_base") or _ONE)
    base_url = str(provider_cfg.get("base_url") or _DEFAULT_OPENAI_BASE_URL).strip()
    api_key = _resolve_openai_api_key(provider_cfg)
    require_api_key = bool(provider_cfg.get("require_api_key", True))
    if require_api_key and not api_key:
        raise ValueError(
            "Missing OpenAI API key. Set env OPENAI_API_KEY (default) or configure llm.openai.api_key/api_key_env."
        )
    url = _join_url(
        base_url,
        str(
            provider_cfg.get("chat_completions_path")
            or _DEFAULT_OPENAI_CHAT_COMPLETIONS_PATH
        ),
    )
    default_headers = _build_openai_headers(
        api_key, extra_headers=provider_cfg.get("headers")
    )

    def _generate(messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        outputs: List[str] = []
        for messages in messages_batch:
            outputs.append(
                _openai_with_retry(
                    url=url,
                    headers=default_headers,
                    model=str(model),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    max_retries=max_retries,
                    backoff_seconds=backoff_seconds,
                    backoff_base=backoff_base,
                )
            )
        return outputs

    return _generate


def _openai_with_retry(
    *,
    url: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: Optional[int],
    max_retries: int,
    backoff_seconds: float,
    backoff_base: float,
) -> str:
    attempt = _ZERO
    while True:
        try:
            return _openai_chat_completions_http(
                url=url,
                headers=headers,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except (
            urllib.error.HTTPError
        ) as exc:  # pragma: no cover - depends on external service
            if attempt >= max_retries or not _is_retryable_http_status(exc.code):
                raise _raise_openai_http_error(exc) from exc
            wait = _compute_backoff_seconds(
                backoff_seconds=backoff_seconds,
                backoff_base=backoff_base,
                attempt=attempt,
            )
            wait = _maybe_override_with_retry_after(exc, wait)
            time.sleep(wait)
            attempt += _ONE
        except Exception as exc:  # pragma: no cover - depends on external service
            if attempt >= max_retries:
                raise exc
            wait = _compute_backoff_seconds(
                backoff_seconds=backoff_seconds,
                backoff_base=backoff_base,
                attempt=attempt,
            )
            time.sleep(wait)
            attempt += _ONE


def _resolve_openai_api_key(provider_cfg: Any) -> str:
    raw = provider_cfg.get("api_key")
    if raw:
        return str(raw).strip()
    env_name = str(
        provider_cfg.get("api_key_env") or _DEFAULT_OPENAI_API_KEY_ENV
    ).strip()
    if not env_name:
        return ""
    return str(os.getenv(env_name, "")).strip()


def _build_openai_headers(api_key: str, *, extra_headers: Any) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if isinstance(extra_headers, dict):
        for key, value in extra_headers.items():
            if key and value is not None:
                headers[str(key)] = str(value)
    return headers


def _join_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("base_url must be a non-empty string.")
    base = base.rstrip("/")
    suffix = str(path or "").strip()
    if not suffix:
        raise ValueError("path must be a non-empty string.")
    if not suffix.startswith("/"):
        suffix = "/" + suffix
    return base + suffix


def _compute_backoff_seconds(
    *, backoff_seconds: float, backoff_base: float, attempt: int
) -> float:
    return float(backoff_seconds) * float(backoff_base**attempt)


def _is_retryable_http_status(status_code: int) -> bool:
    return int(status_code) in {429, 500, 502, 503, 504}


def _maybe_override_with_retry_after(exc: urllib.error.HTTPError, wait: float) -> float:
    retry_after = exc.headers.get("Retry-After")
    if not retry_after:
        return wait
    try:
        seconds = float(str(retry_after).strip())
    except ValueError:
        return wait
    return max(wait, seconds)


def _raise_openai_http_error(exc: urllib.error.HTTPError) -> RuntimeError:
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    snippet = body.strip()
    if len(snippet) > 2000:
        snippet = snippet[:2000] + "..."
    return RuntimeError(f"OpenAI API error {exc.code}: {snippet}")


def _openai_chat_completions_http(
    *,
    url: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: Optional[int],
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    timeout_val = float(timeout) if timeout is not None else None
    with urllib.request.urlopen(req, timeout=timeout_val) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return str(content or "")


__all__ = [
    "_LLMBackend",
    "_build_backend",
    "_build_openai_headers",
    "_format_subgraphrag_user_content",
    "_join_url",
    "_parse_subgraphrag_user_content",
    "_trim_messages_to_vllm_budget",
]
