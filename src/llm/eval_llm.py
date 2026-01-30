from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from src.utils.logging_utils import get_logger, log_event

log = get_logger(__name__)

_ZERO = 0
_ONE = 1
_NEG_INF = float("-inf")

_DEFAULT_INPUT_SUBDIR = "eval_dual_flow"
_DEFAULT_OUTPUT_SUBDIR = "eval_llm"
_DEFAULT_FILENAME_TEMPLATE = "{split}_k{k}_{provider}.jsonl"
_DEFAULT_METRICS_FILENAME_TEMPLATE = "{split}_k{k}_{provider}.metrics.json"
_DEFAULT_ANSWER_KEY = "answer"
_DEFAULT_ANSWER_SEPARATOR = " | "
_DEFAULT_ALLOW_EMPTY_PROMPT_ANSWER = True
_DEFAULT_STOP_RELATION = -1
_DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"
_DEFAULT_MAX_CANDIDATES_IN_PROMPT = 30
_DEFAULT_MAX_PROMPT_CHARS = 0
_DEFAULT_MAX_TRAJECTORIES_IN_PROMPT = 0
_DEFAULT_MAX_JUSTIFICATION_WORDS = 25
_DEFAULT_SCHEMA_ENABLED = True
_DEFAULT_SCHEMA_MAX_RETRIES = 2
_DEFAULT_SCHEMA_ALLOW_COERCE = True
_DEFAULT_SCHEMA_MAX_RETRY_CHARS = 2000
_DEFAULT_SCHEMA_RETRY_MESSAGE = (
    "Your previous response did not match the required JSON schema. "
    "Return a corrected JSON object only.\n\n"
    "Schema:\n{schema}\n\n"
    "Previous response:\n{response}"
)

_FIELD_SAMPLE_ID = "sample_id"
_FIELD_QUESTION = "question_text"
_FIELD_ROLLOUTS = "rollouts"
_FIELD_TRAJECTORY = "trajectory_text"
_FIELD_EDGES = "edges"
_FIELD_SCORE = "score"
_FIELD_ROLLOUT_INDEX = "rollout_index"

_FIELD_MESSAGES = "messages"
_FIELD_RAW_RESPONSE = "raw_response"
_FIELD_SELECTED_TRAJECTORIES = "selected_trajectories"
_FIELD_EVIDENCE_TRAJECTORY_IDS = "evidence_trajectory_ids"
_FIELD_ABSTAIN_REASON = "abstain_reason"
_FIELD_BEST_GUESS = "best_guess"
_FIELD_JUSTIFICATION = "justification"
_FIELD_SCHEMA_VALID = "schema_valid"
_FIELD_SCHEMA_RETRIES = "schema_retries"

_LOG_PROGRESS_EVERY = 200


@dataclass(frozen=True)
class PromptSpec:
    system: str
    answer_key: str
    answer_separator: str
    allow_empty_answer: bool
    max_prompt_chars: int
    max_trajectories: int
    max_candidates: int


@dataclass(frozen=True)
class OutputSpec:
    include_question: bool
    include_trajectories: bool
    include_messages: bool
    include_raw_response: bool
    debug_only_on_empty: bool


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
class _LLMRequest:
    sample_id: str
    question: str
    trajectories: List[str]
    messages: List[Dict[str, str]]


@dataclass(frozen=True)
class _ParsedResponse:
    answer: str
    payload: Optional[Dict[str, Any]]
    schema_valid: bool


def run_llm_eval(cfg: Any) -> None:
    llm_cfg = cfg.get("llm")
    if llm_cfg is None:
        raise ValueError("Missing config group: llm (required for eval_llm).")
    dataset_cfg = cfg.get("dataset") or {}
    _validate_dataset_scope(dataset_cfg, llm_cfg.get("allow_sub", False))
    split = str((cfg.get("run") or {}).get("split", "test"))
    providers = _resolve_provider_list(llm_cfg)
    prompt_spec = _resolve_prompt_spec(llm_cfg)
    output_spec = _resolve_output_spec(llm_cfg)
    schema_spec = _resolve_schema_spec(llm_cfg, prompt_spec)
    input_path = _resolve_input_path(dataset_cfg, llm_cfg, split)
    output_dir = _resolve_output_dir(dataset_cfg, llm_cfg, cfg.get("paths"))
    output_dir.mkdir(parents=True, exist_ok=True)
    topk_list = _resolve_topk_list(llm_cfg)
    for provider in providers:
        provider_cfg = llm_cfg.get(provider)
        if provider_cfg is None:
            raise ValueError(f"Missing llm.{provider} config.")
        backend = _build_backend(provider, provider_cfg, llm_cfg)
        for top_k in topk_list:
            _run_provider_topk(
                backend=backend,
                provider=provider,
                provider_cfg=provider_cfg,
                llm_cfg=llm_cfg,
                input_path=input_path,
                output_dir=output_dir,
                split=split,
                prompt_spec=prompt_spec,
                output_spec=output_spec,
                schema_spec=schema_spec,
                top_k=top_k,
            )


def _validate_dataset_scope(dataset_cfg: Any, allow_sub: bool) -> None:
    name = str(dataset_cfg.get("name") or "")
    if not allow_sub and name.endswith("-sub"):
        raise ValueError("eval_llm is configured for full datasets only; set llm.allow_sub=true to override.")


def _resolve_provider_list(llm_cfg: Any) -> List[str]:
    providers = llm_cfg.get("providers")
    if providers:
        if isinstance(providers, (list, tuple)):
            return [str(p) for p in providers]
        return [str(providers)]
    provider = llm_cfg.get("provider")
    if provider:
        return [str(provider)]
    raise ValueError("llm.provider or llm.providers must be set.")


def _resolve_prompt_spec(llm_cfg: Any) -> PromptSpec:
    prompt_cfg = llm_cfg.get("prompt") or {}
    system = str(prompt_cfg.get("system") or "").strip()
    if not system:
        raise ValueError("llm.prompt.system must be a non-empty string.")
    answer_key = str(prompt_cfg.get("answer_key") or _DEFAULT_ANSWER_KEY).strip()
    answer_separator = str(prompt_cfg.get("answer_separator") or _DEFAULT_ANSWER_SEPARATOR)
    allow_empty_answer = bool(prompt_cfg.get("allow_empty", _DEFAULT_ALLOW_EMPTY_PROMPT_ANSWER))
    max_prompt_chars = int(prompt_cfg.get("max_prompt_chars", _DEFAULT_MAX_PROMPT_CHARS))
    max_trajectories = int(prompt_cfg.get("max_trajectories", _DEFAULT_MAX_TRAJECTORIES_IN_PROMPT))
    max_candidates = int(prompt_cfg.get("max_candidates", _DEFAULT_MAX_CANDIDATES_IN_PROMPT))
    if max_prompt_chars < _ZERO:
        raise ValueError("llm.prompt.max_prompt_chars must be >= 0.")
    if max_trajectories < _ZERO:
        raise ValueError("llm.prompt.max_trajectories must be >= 0.")
    if max_candidates < _ZERO:
        raise ValueError("llm.prompt.max_candidates must be >= 0.")
    return PromptSpec(
        system=system,
        answer_key=answer_key,
        answer_separator=answer_separator,
        allow_empty_answer=allow_empty_answer,
        max_prompt_chars=max_prompt_chars,
        max_trajectories=max_trajectories,
        max_candidates=max_candidates,
    )


def _resolve_output_spec(llm_cfg: Any) -> OutputSpec:
    output_cfg = llm_cfg.get("output") or {}
    include_question = bool(output_cfg.get("include_question", False))
    include_trajectories = bool(output_cfg.get("include_trajectories", False))
    include_messages = bool(output_cfg.get("include_messages", False))
    include_raw_response = bool(output_cfg.get("include_raw_response", False))
    debug_only_on_empty = bool(output_cfg.get("debug_only_on_empty", False))
    return OutputSpec(
        include_question=include_question,
        include_trajectories=include_trajectories,
        include_messages=include_messages,
        include_raw_response=include_raw_response,
        debug_only_on_empty=debug_only_on_empty,
    )


def _resolve_schema_spec(llm_cfg: Any, prompt_spec: PromptSpec) -> SchemaSpec:
    schema_cfg = llm_cfg.get("schema") or {}
    enabled = bool(schema_cfg.get("enabled", _DEFAULT_SCHEMA_ENABLED))
    max_retries = int(schema_cfg.get("max_retries", _DEFAULT_SCHEMA_MAX_RETRIES))
    allow_coerce = bool(schema_cfg.get("allow_coerce", _DEFAULT_SCHEMA_ALLOW_COERCE))
    max_retry_chars = int(schema_cfg.get("max_retry_chars", _DEFAULT_SCHEMA_MAX_RETRY_CHARS))
    retry_message = str(schema_cfg.get("retry_message", _DEFAULT_SCHEMA_RETRY_MESSAGE)).strip()
    if max_retries < _ZERO:
        raise ValueError("llm.schema.max_retries must be >= 0.")
    if max_retry_chars < _ZERO:
        raise ValueError("llm.schema.max_retry_chars must be >= 0.")
    if not enabled:
        return SchemaSpec(
            enabled=False,
            max_retries=max_retries,
            allow_coerce=allow_coerce,
            max_retry_chars=max_retry_chars,
            retry_message=retry_message,
            schema=None,
            schema_json="",
            validator=None,
        )
    try:
        import jsonschema
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("jsonschema is required when llm.schema.enabled=true.") from exc
    schema = _build_llm_output_schema(prompt_spec)
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    validator = jsonschema.Draft7Validator(schema)
    return SchemaSpec(
        enabled=True,
        max_retries=max_retries,
        allow_coerce=allow_coerce,
        max_retry_chars=max_retry_chars,
        retry_message=retry_message,
        schema=schema,
        schema_json=schema_json,
        validator=validator,
    )


def _build_llm_output_schema(prompt: PromptSpec) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            prompt.answer_key: {"type": "string"},
            _FIELD_EVIDENCE_TRAJECTORY_IDS: {"type": "array", "items": {"type": "integer"}},
            _FIELD_ABSTAIN_REASON: {"type": "string"},
            _FIELD_BEST_GUESS: {"type": "string"},
            _FIELD_JUSTIFICATION: {"type": "string"},
        },
        "required": [
            prompt.answer_key,
            _FIELD_EVIDENCE_TRAJECTORY_IDS,
            _FIELD_ABSTAIN_REASON,
            _FIELD_BEST_GUESS,
            _FIELD_JUSTIFICATION,
        ],
        "additionalProperties": True,
    }


def _resolve_topk_list(llm_cfg: Any) -> List[int]:
    topk_list = llm_cfg.get("topk_list")
    if not topk_list:
        raise ValueError("llm.topk_list must be provided.")
    is_list = isinstance(topk_list, (list, tuple))
    if not is_list:
        try:
            from omegaconf import OmegaConf
        except ModuleNotFoundError:
            OmegaConf = None
        if OmegaConf is not None and OmegaConf.is_list(topk_list):
            is_list = True
    if not is_list:
        raise ValueError("llm.topk_list must be a list of integers.")
    return [int(k) for k in list(topk_list)]


def _resolve_input_path(dataset_cfg: Any, llm_cfg: Any, split: str) -> Path:
    input_path = llm_cfg.get("input_path")
    if input_path:
        return Path(input_path)
    artifact_dir = Path(str(dataset_cfg.get("artifact_dir")))
    subdir = str(llm_cfg.get("input_subdir") or _DEFAULT_INPUT_SUBDIR)
    return artifact_dir / subdir / f"{split}.jsonl"


def _resolve_output_dir(dataset_cfg: Any, llm_cfg: Any, paths_cfg: Any = None) -> Path:
    output_dir = llm_cfg.get("output_dir")
    if output_dir:
        return Path(output_dir)
    if paths_cfg is not None:
        paths_output = paths_cfg.get("output_dir") if hasattr(paths_cfg, "get") else None
        if paths_output:
            subdir = str(llm_cfg.get("output_subdir") or _DEFAULT_OUTPUT_SUBDIR)
            return Path(str(paths_output)) / subdir
    artifact_dir = Path(str(dataset_cfg.get("artifact_dir")))
    subdir = str(llm_cfg.get("output_subdir") or _DEFAULT_OUTPUT_SUBDIR)
    return artifact_dir / subdir


def _run_provider_topk(
    *,
    backend: "_LLMBackend",
    provider: str,
    provider_cfg: Any,
    llm_cfg: Any,
    input_path: Path,
    output_dir: Path,
    split: str,
    prompt_spec: PromptSpec,
    output_spec: OutputSpec,
    schema_spec: SchemaSpec,
    top_k: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    output_path, seen, batch_size, max_samples, file_mode = _prepare_llm_run(llm_cfg, output_dir, split, top_k, provider)
    _log_llm_start(provider, provider_cfg, split, top_k, output_path)
    processed, written = _run_llm_batches(
        backend=backend,
        input_path=input_path,
        output_path=output_path,
        file_mode=file_mode,
        seen=seen,
        batch_size=batch_size,
        max_samples=max_samples,
        top_k=top_k,
        prompt_spec=prompt_spec,
        output_spec=output_spec,
        schema_spec=schema_spec,
    )
    _log_llm_done(processed, written, top_k, output_path)
    if bool(llm_cfg.get("compute_metrics", True)):
        from src.llm.metrics import write_llm_metrics

        metrics_path = write_llm_metrics(
            input_path=input_path,
            output_path=output_path,
            output_dir=output_dir,
            split=split,
            provider=provider,
            top_k=top_k,
            answer_key=prompt_spec.answer_key,
            answer_separator=prompt_spec.answer_separator,
            metrics_filename_template=str(llm_cfg.get("metrics_filename_template") or _DEFAULT_METRICS_FILENAME_TEMPLATE),
        )
        log_event(log, "llm_eval_metrics_written", provider=provider, split=split, top_k=top_k, path=str(metrics_path))


def _prepare_llm_run(
    llm_cfg: Any,
    output_dir: Path,
    split: str,
    top_k: int,
    provider: str,
) -> Tuple[Path, set[str], int, Optional[int], str]:
    filename_template = str(llm_cfg.get("output_filename_template") or _DEFAULT_FILENAME_TEMPLATE)
    output_path = output_dir / filename_template.format(split=split, k=top_k, provider=provider)
    resume = bool(llm_cfg.get("resume", True))
    seen = _load_seen_ids(output_path) if resume else set()
    batch_size = int(llm_cfg.get("batch_size") or _ONE)
    max_samples = llm_cfg.get("max_samples")
    file_mode = "a" if resume else "w"
    return output_path, seen, batch_size, int(max_samples) if max_samples is not None else None, file_mode


def _log_llm_start(
    provider: str,
    provider_cfg: Any,
    split: str,
    top_k: int,
    output_path: Path,
) -> None:
    log_event(
        log,
        "llm_eval_start",
        provider=provider,
        model=str(provider_cfg.get("model")),
        split=split,
        top_k=top_k,
        output=str(output_path),
    )


def _log_llm_done(processed: int, written: int, top_k: int, output_path: Path) -> None:
    log_event(log, "llm_eval_done", processed=processed, written=written, top_k=top_k, output=str(output_path))


def _run_llm_batches(
    *,
    backend: "_LLMBackend",
    input_path: Path,
    output_path: Path,
    file_mode: str,
    seen: set[str],
    batch_size: int,
    max_samples: Optional[int],
    top_k: int,
    prompt_spec: PromptSpec,
    output_spec: OutputSpec,
    schema_spec: SchemaSpec,
) -> Tuple[int, int]:
    processed = _ZERO
    written = _ZERO
    batch_items: List[_LLMRequest] = []
    with output_path.open(file_mode, encoding="utf-8") as f_out:
        for request in _iter_requests(input_path, seen, top_k, prompt_spec, max_samples):
            processed += _ONE
            batch_items.append(request)
            if len(batch_items) >= batch_size:
                written += _flush_batch(backend, batch_items, f_out, prompt_spec, output_spec, schema_spec)
                batch_items = []
            if processed % _LOG_PROGRESS_EVERY == _ZERO:
                log_event(log, "llm_eval_progress", processed=processed, written=written, top_k=top_k)
        if batch_items:
            written += _flush_batch(backend, batch_items, f_out, prompt_spec, output_spec, schema_spec)
    return processed, written


def _iter_requests(
    input_path: Path,
    seen: set[str],
    top_k: int,
    prompt_spec: PromptSpec,
    max_samples: Optional[int],
) -> Iterator[_LLMRequest]:
    processed = _ZERO
    for record in _iter_jsonl(input_path):
        sample_id = str(record.get(_FIELD_SAMPLE_ID) or "")
        if not sample_id or sample_id in seen:
            continue
        if max_samples is not None and processed >= max_samples:
            break
        question = str(record.get(_FIELD_QUESTION) or "")
        rollouts = record.get(_FIELD_ROLLOUTS) or []
        trajectories = _select_trajectories(rollouts, top_k, max_trajectories=prompt_spec.max_trajectories)
        if prompt_spec.max_prompt_chars > _ZERO:
            trajectories = _trim_trajectories_for_prompt(question, trajectories, prompt_spec)
        messages = _build_messages(question, trajectories, prompt_spec)
        processed += _ONE
        yield _LLMRequest(sample_id=sample_id, question=question, trajectories=list(trajectories), messages=messages)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _select_trajectories(rollouts: Sequence[Dict[str, Any]], top_k: int, *, max_trajectories: int) -> List[str]:
    sorted_rollouts = sorted(
        rollouts,
        key=lambda r: (float(r.get(_FIELD_SCORE, _NEG_INF)), int(r.get(_FIELD_ROLLOUT_INDEX, _ZERO))),
        reverse=True,
    )
    limit = int(top_k)
    if max_trajectories > _ZERO:
        limit = min(limit, int(max_trajectories))
    selected = sorted_rollouts[: limit]
    return [_trajectory_text(r) for r in selected if _trajectory_text(r)]


def _trajectory_text(rollout: Dict[str, Any]) -> str:
    text = rollout.get(_FIELD_TRAJECTORY)
    if isinstance(text, str) and text.strip():
        return text.strip()
    edges = rollout.get(_FIELD_EDGES)
    if not isinstance(edges, list) or not edges:
        return ""
    parts = [_edge_to_text(edge) for edge in edges]
    return " ; ".join([p for p in parts if p])


def _edge_to_text(edge: Dict[str, Any]) -> str:
    src = edge.get("src_text") or edge.get("src_entity_id")
    rel = edge.get("relation_text") or edge.get("relation_id")
    dst = edge.get("dst_text") or edge.get("dst_entity_id")
    if rel == _DEFAULT_STOP_RELATION or str(rel) == str(_DEFAULT_STOP_RELATION):
        rel = "SELF"
    return f"{src} --{rel}--> {dst}"


def _trim_trajectories_for_prompt(
    question: str,
    trajectories: Sequence[str],
    prompt: PromptSpec,
) -> List[str]:
    max_chars = int(prompt.max_prompt_chars)
    if max_chars <= _ZERO:
        return list(trajectories)
    kept: List[str] = []
    for traj in trajectories:
        candidate = kept + [traj]
        user_text = _build_user_text(question, candidate, prompt)
        total_chars = len(prompt.system) + len(user_text)
        if total_chars > max_chars:
            break
        kept = candidate
    return kept


def _build_user_text(question: str, trajectories: Sequence[str], prompt: PromptSpec) -> str:
    lines = []
    for idx, traj in enumerate(trajectories, start=_ONE):
        lines.append(f"{idx}. {traj}")
    traj_block = "\n".join(lines) if lines else "(no trajectories)"
    candidates = _extract_destination_candidates(trajectories, max_candidates=prompt.max_candidates)
    if candidates:
        candidate_block = "\n".join(f"- {candidate}" for candidate in candidates)
    else:
        candidate_block = "(none)"
    answer_schema = (
        "{\n"
        f'  "{prompt.answer_key}": "<string>",\n'
        f'  "{_FIELD_EVIDENCE_TRAJECTORY_IDS}": [<int>, ...],\n'
        f'  "{_FIELD_ABSTAIN_REASON}": "<string>",\n'
        f'  "{_FIELD_BEST_GUESS}": "<string>",\n'
        f'  "{_FIELD_JUSTIFICATION}": "<string>"\n'
        "}"
    )
    answer_example_single = (
        "{"
        f'"{prompt.answer_key}": "Answer A", '
        f'"{_FIELD_EVIDENCE_TRAJECTORY_IDS}": [1], '
        f'"{_FIELD_ABSTAIN_REASON}": "", '
        f'"{_FIELD_BEST_GUESS}": "", '
        f'"{_FIELD_JUSTIFICATION}": "Trajectory 1 supports Answer A."'
        "}"
    )
    answer_example_multi = (
        "{"
        f'"{prompt.answer_key}": "Answer A{prompt.answer_separator}Answer B", '
        f'"{_FIELD_EVIDENCE_TRAJECTORY_IDS}": [1, 2], '
        f'"{_FIELD_ABSTAIN_REASON}": "", '
        f'"{_FIELD_BEST_GUESS}": "", '
        f'"{_FIELD_JUSTIFICATION}": "Trajectories 1 and 2 support both answers."'
        "}"
    )
    answer_example_abstain = (
        "{"
        f'"{prompt.answer_key}": "", '
        f'"{_FIELD_EVIDENCE_TRAJECTORY_IDS}": [], '
        f'"{_FIELD_ABSTAIN_REASON}": "no_supported_candidate", '
        f'"{_FIELD_BEST_GUESS}": "Candidate X", '
        f'"{_FIELD_JUSTIFICATION}": "Candidates are present but none answers the question."'
        "}"
    )
    if prompt.allow_empty_answer:
        empty_clause = (
            f'Only set "{prompt.answer_key}" to an empty string when there is no supported answer. '
            "Prefer selecting the best-supported candidate entity over returning an empty answer when at least one "
            "candidate is plausible."
        )
    else:
        empty_clause = (
            f'Always return a non-empty string for "{prompt.answer_key}". '
            "If insufficient evidence, make the best guess using only the trajectories; "
            "prefer entity names that appear in the trajectories (e.g., destination text after \"-->\")."
        )
    return (
        "Question:\n"
        f"{question}\n\n"
        "Trajectories:\n"
        f"{traj_block}\n\n"
        "Candidate answer entities (trajectory destinations):\n"
        f"{candidate_block}\n\n"
        "Return a single JSON object with the following schema:\n"
        f"{answer_schema}\n\n"
        "Rules:\n"
        f'- The value of "{prompt.answer_key}" must be a string.\n'
        "- Use exact surface forms from the trajectories (or the candidate list).\n"
        f'- If multiple answers, join exactly with "{prompt.answer_separator}" (example below).\n'
        f'- "{_FIELD_EVIDENCE_TRAJECTORY_IDS}" must list 1-based trajectory indices that directly support the answer.\n'
        f'- "{_FIELD_JUSTIFICATION}" must be short (<= {_DEFAULT_MAX_JUSTIFICATION_WORDS} words).\n'
        f'- If you output an empty "{prompt.answer_key}", set "{_FIELD_ABSTAIN_REASON}" to a short reason string and fill "{_FIELD_BEST_GUESS}" with the closest candidate (or empty if none).\n'
        f"- {empty_clause}\n\n"
        "Examples:\n"
        f"{answer_example_single}\n"
        f"{answer_example_multi}\n"
        f"{answer_example_abstain}\n\n"
        "Output JSON only."
    )


def _build_messages(question: str, trajectories: Sequence[str], prompt: PromptSpec) -> List[Dict[str, str]]:
    user_text = _build_user_text(question, trajectories, prompt)
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_text},
    ]


def _extract_destination_candidates(trajectories: Sequence[str], *, max_candidates: int) -> List[str]:
    seen: set[str] = set()
    candidates: List[str] = []
    for traj in trajectories:
        if not isinstance(traj, str) or not traj.strip():
            continue
        arrow = traj.rfind("-->")
        if arrow < _ZERO:
            continue
        candidate = traj[arrow + len("-->") :].strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _flush_batch(
    backend: "_LLMBackend",
    batch_items: List[_LLMRequest],
    f_out,
    prompt_spec: PromptSpec,
    output_spec: OutputSpec,
    schema_spec: SchemaSpec,
) -> int:
    messages_batch = [item.messages for item in batch_items]
    responses = backend.generate(messages_batch)
    responses = [(response or "").strip() for response in responses]
    parsed_list = [_parse_and_validate_response(response, prompt_spec, schema_spec) for response in responses]
    retries = [0 for _ in batch_items]
    if schema_spec.enabled and any(not parsed.schema_valid for parsed in parsed_list):
        responses, parsed_list, retries = _retry_schema_batch(
            backend=backend,
            batch_items=batch_items,
            responses=responses,
            parsed_list=parsed_list,
            prompt_spec=prompt_spec,
            schema_spec=schema_spec,
        )
    written = _ZERO
    for request, raw_response, parsed, retry_count in zip(batch_items, responses, parsed_list, retries):
        if schema_spec.enabled and not parsed.schema_valid:
            log_event(
                log,
                "llm_schema_invalid",
                sample_id=request.sample_id,
                retries=retry_count,
            )
        schema_meta = {
            _FIELD_SCHEMA_VALID: bool(parsed.schema_valid),
            _FIELD_SCHEMA_RETRIES: int(retry_count),
        }
        extra = _build_output_extra(request, parsed.answer, raw_response, output_spec, schema_meta=schema_meta)
        _write_answer(f_out, request.sample_id, parsed.answer, prompt_spec.answer_key, extra=extra)
        written += _ONE
    return written


def _parse_response(response: str, prompt: PromptSpec) -> str:
    raw = (response or "").strip()
    if not raw:
        return ""
    payload = _parse_json_payload(raw)
    if payload is not None:
        return _extract_answer_from_payload(payload, prompt, raw)

    fragment = _parse_json_fragment(raw)
    if fragment is not None:
        return _extract_answer_from_payload(fragment, prompt, raw)

    return raw


def _parse_and_validate_response(
    response: str,
    prompt: PromptSpec,
    schema_spec: SchemaSpec,
) -> _ParsedResponse:
    raw = (response or "").strip()
    if not raw:
        return _ParsedResponse(answer="", payload=None, schema_valid=not schema_spec.enabled)
    payload = _parse_json_payload(raw)
    if payload is None:
        payload = _parse_json_fragment(raw)
    if not schema_spec.enabled:
        answer = _parse_response(raw, prompt)
        return _ParsedResponse(answer=answer, payload=payload if isinstance(payload, dict) else None, schema_valid=True)
    if not isinstance(payload, dict):
        return _ParsedResponse(answer="", payload=None, schema_valid=False)
    normalized = _normalize_payload(payload, prompt, allow_coerce=schema_spec.allow_coerce)
    if normalized is None:
        return _ParsedResponse(answer="", payload=None, schema_valid=False)
    if not _validate_payload_schema(normalized, schema_spec):
        return _ParsedResponse(answer="", payload=normalized, schema_valid=False)
    answer = _extract_answer_from_payload(normalized, prompt, raw)
    return _ParsedResponse(answer=answer, payload=normalized, schema_valid=True)


def _normalize_payload(payload: Dict[str, Any], prompt: PromptSpec, *, allow_coerce: bool) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    normalized: Dict[str, Any] = dict(payload)
    if prompt.answer_key not in normalized and _DEFAULT_ANSWER_KEY in normalized:
        normalized[prompt.answer_key] = normalized.get(_DEFAULT_ANSWER_KEY)
    if allow_coerce:
        normalized = _coerce_payload(normalized)
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
    if schema_spec.max_retry_chars > _ZERO and len(snippet) > schema_spec.max_retry_chars:
        snippet = snippet[: schema_spec.max_retry_chars] + "..."
    try:
        return schema_spec.retry_message.format(schema=schema_spec.schema_json, response=snippet)
    except Exception:
        return schema_spec.retry_message


def _retry_schema_batch(
    *,
    backend: "_LLMBackend",
    batch_items: List[_LLMRequest],
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
        invalid_idx = [idx for idx, parsed in enumerate(parsed_list) if not parsed.schema_valid]
        if not invalid_idx:
            break
        retry_messages: List[List[Dict[str, str]]] = []
        for idx in invalid_idx:
            retry_message = _build_schema_retry_message(schema_spec, responses[idx])
            current_messages[idx] = current_messages[idx] + [{"role": "user", "content": retry_message}]
            retry_messages.append(current_messages[idx])
        retry_outputs = backend.generate(retry_messages)
        for idx, output in zip(invalid_idx, retry_outputs):
            responses[idx] = (output or "").strip()
            parsed_list[idx] = _parse_and_validate_response(responses[idx], prompt_spec, schema_spec)
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


def _build_output_extra(
    request: _LLMRequest,
    answer: str,
    raw_response: str,
    spec: OutputSpec,
    *,
    schema_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if spec.debug_only_on_empty and answer.strip():
        return schema_meta or {}
    extra: Dict[str, Any] = {}
    if spec.include_question:
        extra[_FIELD_QUESTION] = request.question
    if spec.include_trajectories:
        extra[_FIELD_SELECTED_TRAJECTORIES] = request.trajectories
    if spec.include_messages:
        extra[_FIELD_MESSAGES] = request.messages
    if spec.include_raw_response:
        extra[_FIELD_RAW_RESPONSE] = raw_response
    if schema_meta:
        extra.update(schema_meta)
    extra.update(_extract_structured_model_fields(raw_response))
    return extra


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


def _write_answer(f_out, sample_id: str, answer: str, answer_key: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {_FIELD_SAMPLE_ID: sample_id, answer_key: answer}
    if extra:
        payload.update(extra)
    f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for record in _iter_jsonl(path):
        sample_id = record.get(_FIELD_SAMPLE_ID)
        if sample_id:
            seen.add(str(sample_id))
    return seen


class _LLMBackend:
    def __init__(self, generate_fn):
        self._generate_fn = generate_fn

    def generate(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        return self._generate_fn(messages_batch)


def _build_backend(provider: str, provider_cfg: Any, llm_cfg: Any) -> _LLMBackend:
    name = provider.lower()
    if name == "vllm":
        return _LLMBackend(_build_vllm_generate(provider_cfg))
    if name == "openai":
        return _LLMBackend(_build_openai_generate(provider_cfg))
    raise ValueError(f"Unsupported provider: {provider}")


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
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)
    sampling_params = SamplingParams(
        temperature=float(provider_cfg.get("temperature", _ZERO)),
        max_tokens=int(provider_cfg.get("max_tokens") or _ONE),
        top_p=float(provider_cfg.get("top_p", _ONE)),
    )

    def _generate(messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        outputs = llm.chat(messages_batch, sampling_params=sampling_params, use_tqdm=False)
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
    url = _join_url(base_url, str(provider_cfg.get("chat_completions_path") or _DEFAULT_OPENAI_CHAT_COMPLETIONS_PATH))
    default_headers = _build_openai_headers(api_key, extra_headers=provider_cfg.get("headers"))

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
        except urllib.error.HTTPError as exc:  # pragma: no cover - depends on external service
            if attempt >= max_retries or not _is_retryable_http_status(exc.code):
                raise _raise_openai_http_error(exc) from exc
            wait = _compute_backoff_seconds(backoff_seconds=backoff_seconds, backoff_base=backoff_base, attempt=attempt)
            wait = _maybe_override_with_retry_after(exc, wait)
            time.sleep(wait)
            attempt += _ONE
        except Exception as exc:  # pragma: no cover - depends on external service
            if attempt >= max_retries:
                raise exc
            wait = _compute_backoff_seconds(backoff_seconds=backoff_seconds, backoff_base=backoff_base, attempt=attempt)
            time.sleep(wait)
            attempt += _ONE


def _resolve_openai_api_key(provider_cfg: Any) -> str:
    raw = provider_cfg.get("api_key")
    if raw:
        return str(raw).strip()
    env_name = str(provider_cfg.get("api_key_env") or _DEFAULT_OPENAI_API_KEY_ENV).strip()
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


def _compute_backoff_seconds(*, backoff_seconds: float, backoff_base: float, attempt: int) -> float:
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
