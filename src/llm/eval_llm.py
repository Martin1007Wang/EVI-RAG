from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from src.llm.backends import _LLMBackend, _build_backend
from src.llm.prompting import (
    PromptSpec,
    _CANDIDATE_SOURCE_ENDPOINTS_ONLY,
    _CANDIDATE_SOURCE_TRAJECTORY_NODES,
    _DEFAULT_CANDIDATE_SOURCE,
    _DEFAULT_CONSTRAIN_TO_CANDIDATES,
    _DEFAULT_FALLBACK_ANSWER,
    _FIELD_ABSTAIN_REASON,
    _FIELD_BEST_GUESS,
    _FIELD_EVIDENCE_TRAJECTORY_IDS,
    _FIELD_JUSTIFICATION,
    _FIELD_QUESTION,
    _FIELD_TRAJECTORIES,
    _PROMPT_MODE_JSON_SCHEMA,
    _PROMPT_MODE_SUBGRAPHRAG_ICL_DC,
    _build_messages,
    _enforce_candidate_answers,
    _extract_destination_candidates,
    _select_trajectories,
    _trim_context_for_prompt,
)
from src.llm.response_parsing import (
    SchemaSpec,
    _extract_structured_model_fields,
    _needs_subgraphrag_dc_retry,
    _parse_and_validate_response,
    _retry_schema_batch,
)
from src.llm.metrics import write_llm_metrics_artifacts
from src.utils.logging_utils import get_logger, log_event

log = get_logger(__name__)

_ZERO = 0
_ONE = 1
_NEG_INF = float("-inf")

_DEFAULT_INPUT_SUBDIR = "rankflow"
_DEFAULT_OUTPUT_SUBDIR = "eval_llm"
_DEFAULT_INPUT_LABELS_SUFFIX = ".labels.jsonl"
_DEFAULT_FILENAME_TEMPLATE = "{split}_k{k}_{provider}.jsonl"
_DEFAULT_METRICS_FILENAME_TEMPLATE = "{split}_k{k}_{provider}.metrics.json"
_DEFAULT_ANSWER_KEY = "answer"
_DEFAULT_ANSWER_SEPARATOR = " | "
_DEFAULT_ALLOW_EMPTY_PROMPT_ANSWER = True
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
    "Do not refuse. Return a corrected JSON object only.\n\n"
    "Schema:\n{schema}\n\n"
    "Previous response:\n{response}"
)

_FIELD_SAMPLE_ID = "sample_id"

_FIELD_MESSAGES = "messages"
_FIELD_RAW_RESPONSE = "raw_response"
_FIELD_SELECTED_TRAJECTORIES = "selected_trajectories"
_FIELD_SCHEMA_VALID = "schema_valid"
_FIELD_SCHEMA_RETRIES = "schema_retries"
_FIELD_DC_RETRIES = "dc_retries"

_LOG_PROGRESS_EVERY = 200


@dataclass(frozen=True)
class OutputSpec:
    include_question: bool
    include_trajectories: bool
    include_messages: bool
    include_raw_response: bool
    debug_only_on_empty: bool


@dataclass(frozen=True)
class _LLMRequest:
    sample_id: str
    question: str
    trajectories: List[str]
    messages: List[Dict[str, str]]


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
    compute_metrics = bool(llm_cfg.get("compute_metrics", True))
    input_labels_path = _resolve_input_labels_path(
        input_path=input_path,
        llm_cfg=llm_cfg,
        require_labels=compute_metrics,
    )
    output_dir = _resolve_output_dir(dataset_cfg, llm_cfg, cfg.get("paths"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_dir = _resolve_metrics_log_dir(llm_cfg, cfg.get("paths"))
    topk_list = _resolve_topk_list(llm_cfg)
    _validate_topk_against_prompt_limits(topk_list=topk_list, prompt_spec=prompt_spec)
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
                dataset_cfg=dataset_cfg,
                input_path=input_path,
                input_labels_path=input_labels_path,
                output_dir=output_dir,
                metrics_log_dir=metrics_log_dir,
                split=split,
                prompt_spec=prompt_spec,
                output_spec=output_spec,
                schema_spec=schema_spec,
                top_k=top_k,
            )


def _validate_dataset_scope(dataset_cfg: Any, allow_sub: bool) -> None:
    name = str(dataset_cfg.get("name") or "")
    if not allow_sub and name.endswith("-sub"):
        raise ValueError(
            "eval_llm is configured for full datasets only; set llm.allow_sub=true to override."
        )


def _resolve_provider_list(llm_cfg: Any) -> List[str]:
    providers = llm_cfg.get("providers")
    if providers:
        if isinstance(providers, (list, tuple)):
            return [str(p) for p in providers]
        return [str(providers)]
    raise ValueError("llm.providers must be set.")


def _resolve_prompt_spec(llm_cfg: Any) -> PromptSpec:
    prompt_cfg = llm_cfg.get("prompt") or {}
    mode = str(prompt_cfg.get("mode") or _PROMPT_MODE_JSON_SCHEMA).strip()
    if mode not in {_PROMPT_MODE_JSON_SCHEMA, _PROMPT_MODE_SUBGRAPHRAG_ICL_DC}:
        raise ValueError(f"Unsupported llm.prompt.mode: {mode}")
    system = str(prompt_cfg.get("system") or "").strip()
    if not system and mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC:
        from src.llm.subgraphrag_prompts import DEFAULT_SUBGRAPHRAG_ICL_SYSTEM

        system = DEFAULT_SUBGRAPHRAG_ICL_SYSTEM
    if not system:
        raise ValueError("llm.prompt.system must be a non-empty string.")
    answer_key = str(prompt_cfg.get("answer_key") or _DEFAULT_ANSWER_KEY).strip()
    answer_separator = str(
        prompt_cfg.get("answer_separator") or _DEFAULT_ANSWER_SEPARATOR
    )
    allow_empty_answer = bool(
        prompt_cfg.get("allow_empty", _DEFAULT_ALLOW_EMPTY_PROMPT_ANSWER)
    )
    constrain_default = (
        _DEFAULT_CONSTRAIN_TO_CANDIDATES
        if mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC
        else False
    )
    constrain_to_candidates = bool(
        prompt_cfg.get("constrain_to_candidates", constrain_default)
    )
    candidate_source = (
        str(prompt_cfg.get("candidate_source", _DEFAULT_CANDIDATE_SOURCE))
        .strip()
        .lower()
    )
    if candidate_source not in {
        _CANDIDATE_SOURCE_ENDPOINTS_ONLY,
        _CANDIDATE_SOURCE_TRAJECTORY_NODES,
    }:
        raise ValueError(
            "llm.prompt.candidate_source must be one of "
            f"{{{_CANDIDATE_SOURCE_ENDPOINTS_ONLY!r}, {_CANDIDATE_SOURCE_TRAJECTORY_NODES!r}}}."
        )
    max_prompt_chars = int(
        prompt_cfg.get("max_prompt_chars", _DEFAULT_MAX_PROMPT_CHARS)
    )
    max_trajectories = int(
        prompt_cfg.get("max_trajectories", _DEFAULT_MAX_TRAJECTORIES_IN_PROMPT)
    )
    max_candidates = int(
        prompt_cfg.get("max_candidates", _DEFAULT_MAX_CANDIDATES_IN_PROMPT)
    )
    if max_prompt_chars < _ZERO:
        raise ValueError("llm.prompt.max_prompt_chars must be >= 0.")
    if max_trajectories < _ZERO:
        raise ValueError("llm.prompt.max_trajectories must be >= 0.")
    if max_candidates < _ZERO:
        raise ValueError("llm.prompt.max_candidates must be >= 0.")

    icl_user_prompt = str(prompt_cfg.get("icl_user_prompt") or "").strip()
    icl_assistant_prompt = str(prompt_cfg.get("icl_assistant_prompt") or "").strip()
    cot_prompt = str(prompt_cfg.get("cot_prompt") or "").strip()
    if mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC:
        from src.llm.subgraphrag_prompts import (
            DEFAULT_SUBGRAPHRAG_ICL_ASSISTANT,
            DEFAULT_SUBGRAPHRAG_ICL_COT,
            DEFAULT_SUBGRAPHRAG_ICL_USER,
        )

        if not icl_user_prompt:
            icl_user_prompt = DEFAULT_SUBGRAPHRAG_ICL_USER
        if not icl_assistant_prompt:
            icl_assistant_prompt = DEFAULT_SUBGRAPHRAG_ICL_ASSISTANT
        if not cot_prompt:
            cot_prompt = DEFAULT_SUBGRAPHRAG_ICL_COT
    return PromptSpec(
        mode=mode,
        system=system,
        answer_key=answer_key,
        answer_separator=answer_separator,
        allow_empty_answer=allow_empty_answer,
        constrain_to_candidates=constrain_to_candidates,
        candidate_source=candidate_source,
        max_prompt_chars=max_prompt_chars,
        max_trajectories=max_trajectories,
        max_candidates=max_candidates,
        icl_user_prompt=icl_user_prompt,
        icl_assistant_prompt=icl_assistant_prompt,
        cot_prompt=cot_prompt,
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
    max_retry_chars = int(
        schema_cfg.get("max_retry_chars", _DEFAULT_SCHEMA_MAX_RETRY_CHARS)
    )
    retry_message = str(
        schema_cfg.get("retry_message", _DEFAULT_SCHEMA_RETRY_MESSAGE)
    ).strip()
    if max_retries < _ZERO:
        raise ValueError("llm.schema.max_retries must be >= 0.")
    if max_retry_chars < _ZERO:
        raise ValueError("llm.schema.max_retry_chars must be >= 0.")
    if prompt_spec.mode != _PROMPT_MODE_JSON_SCHEMA:
        if enabled:
            log_event(
                log, "llm_schema_disabled_for_prompt_mode", prompt_mode=prompt_spec.mode
            )
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
        raise ModuleNotFoundError(
            "jsonschema is required when llm.schema.enabled=true."
        ) from exc
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
    answer_schema: Dict[str, Any] = {"type": "string"}
    if not prompt.allow_empty_answer:
        answer_schema["minLength"] = _ONE
    return {
        "type": "object",
        "properties": {
            prompt.answer_key: answer_schema,
            _FIELD_EVIDENCE_TRAJECTORY_IDS: {
                "type": "array",
                "items": {"type": "integer"},
            },
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


def _validate_topk_against_prompt_limits(
    *, topk_list: Sequence[int], prompt_spec: PromptSpec
) -> None:
    max_trajectories = int(prompt_spec.max_trajectories)
    if max_trajectories <= _ZERO:
        return
    if not topk_list:
        return
    topk_max = max(int(k) for k in topk_list)
    if topk_max > max_trajectories:
        raise ValueError(
            "llm.topk_list must be <= llm.prompt.max_trajectories to avoid implicit clipping "
            f"(max topk={topk_max}, max_trajectories={max_trajectories})."
        )


def _resolve_input_path(dataset_cfg: Any, llm_cfg: Any, split: str) -> Path:
    input_path = llm_cfg.get("input_path")
    if input_path:
        return Path(input_path)
    artifact_dir = Path(str(dataset_cfg.get("artifact_dir")))
    subdir = str(llm_cfg.get("input_subdir") or _DEFAULT_INPUT_SUBDIR)
    return artifact_dir / subdir / f"{split}.jsonl"


def _resolve_input_labels_path(
    *, input_path: Path, llm_cfg: Any, require_labels: bool = False
) -> Optional[Path]:
    explicit = llm_cfg.get("input_labels_path")
    if explicit:
        path = Path(str(explicit))
        if not path.exists():
            raise FileNotFoundError(f"Input labels JSONL not found: {path}")
        return path
    stem = input_path.stem
    candidate = input_path.with_name(f"{stem}{_DEFAULT_INPUT_LABELS_SUFFIX}")
    if candidate.exists():
        return candidate
    if require_labels:
        raise FileNotFoundError(
            "Input labels JSONL not found for metrics. "
            f"Expected: {candidate}. "
            "Set llm.input_labels_path explicitly or generate the sidecar labels file."
        )
    return None


def _resolve_output_dir(dataset_cfg: Any, llm_cfg: Any, paths_cfg: Any = None) -> Path:
    output_dir = llm_cfg.get("output_dir")
    if output_dir:
        return Path(output_dir)
    if paths_cfg is not None:
        paths_output = (
            paths_cfg.get("output_dir") if hasattr(paths_cfg, "get") else None
        )
        if paths_output:
            subdir = str(llm_cfg.get("output_subdir") or _DEFAULT_OUTPUT_SUBDIR)
            return Path(str(paths_output)) / subdir
    artifact_dir = Path(str(dataset_cfg.get("artifact_dir")))
    subdir = str(llm_cfg.get("output_subdir") or _DEFAULT_OUTPUT_SUBDIR)
    return artifact_dir / subdir


def _resolve_metrics_log_dir(llm_cfg: Any, paths_cfg: Any = None) -> Optional[Path]:
    metrics_log_dir = llm_cfg.get("metrics_log_dir")
    if metrics_log_dir in (None, ""):
        return None
    if metrics_log_dir:
        return Path(str(metrics_log_dir))
    if paths_cfg is None:
        return None
    paths_output = paths_cfg.get("output_dir") if hasattr(paths_cfg, "get") else None
    if not paths_output:
        return None
    return Path(str(paths_output)) / "metrics"


def _run_provider_topk(
    *,
    backend: "_LLMBackend",
    provider: str,
    provider_cfg: Any,
    llm_cfg: Any,
    dataset_cfg: Any,
    input_path: Path,
    input_labels_path: Optional[Path],
    output_dir: Path,
    metrics_log_dir: Optional[Path],
    split: str,
    prompt_spec: PromptSpec,
    output_spec: OutputSpec,
    schema_spec: SchemaSpec,
    top_k: int,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    output_path, seen, batch_size, max_samples, file_mode = _prepare_llm_run(
        llm_cfg, output_dir, split, top_k, provider
    )
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
        dataset_name = (
            str(dataset_cfg.get("name") or "") if hasattr(dataset_cfg, "get") else ""
        )
        dataset_scope = (
            str(dataset_cfg.get("dataset_scope") or "")
            if hasattr(dataset_cfg, "get")
            else ""
        )
        metrics_path, metrics = write_llm_metrics_artifacts(
            input_path=input_path,
            input_labels_path=input_labels_path,
            output_path=output_path,
            output_dir=output_dir,
            split=split,
            provider=provider,
            top_k=top_k,
            answer_key=prompt_spec.answer_key,
            answer_separator=prompt_spec.answer_separator,
            metrics_filename_template=str(
                llm_cfg.get("metrics_filename_template")
                or _DEFAULT_METRICS_FILENAME_TEMPLATE
            ),
            metrics_log_dir=metrics_log_dir,
            metrics_jsonl_name=str(llm_cfg.get("metrics_jsonl_name") or "llm.jsonl"),
            dataset_name=dataset_name,
            dataset_scope=dataset_scope,
        )
        log_event(
            log,
            "llm_eval_metrics_written",
            provider=provider,
            split=split,
            top_k=top_k,
            path=str(metrics_path),
        )


def _prepare_llm_run(
    llm_cfg: Any,
    output_dir: Path,
    split: str,
    top_k: int,
    provider: str,
) -> Tuple[Path, set[str], int, Optional[int], str]:
    filename_template = str(
        llm_cfg.get("output_filename_template") or _DEFAULT_FILENAME_TEMPLATE
    )
    output_path = output_dir / filename_template.format(
        split=split, k=top_k, provider=provider
    )
    resume = bool(llm_cfg.get("resume", True))
    seen = _load_seen_ids(output_path) if resume else set()
    batch_size = int(llm_cfg.get("batch_size") or _ONE)
    max_samples = llm_cfg.get("max_samples")
    file_mode = "a" if resume else "w"
    return (
        output_path,
        seen,
        batch_size,
        int(max_samples) if max_samples is not None else None,
        file_mode,
    )


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
    log_event(
        log,
        "llm_eval_done",
        processed=processed,
        written=written,
        top_k=top_k,
        output=str(output_path),
    )


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
        for request in _iter_requests(
            input_path, seen, top_k, prompt_spec, max_samples
        ):
            processed += _ONE
            batch_items.append(request)
            if len(batch_items) >= batch_size:
                written += _flush_batch(
                    backend, batch_items, f_out, prompt_spec, output_spec, schema_spec
                )
                batch_items = []
            if processed % _LOG_PROGRESS_EVERY == _ZERO:
                log_event(
                    log,
                    "llm_eval_progress",
                    processed=processed,
                    written=written,
                    top_k=top_k,
                )
        if batch_items:
            written += _flush_batch(
                backend, batch_items, f_out, prompt_spec, output_spec, schema_spec
            )
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
        trajectory_records = record.get(_FIELD_TRAJECTORIES) or []
        trajectories = _select_trajectories(
            trajectory_records,
            top_k,
            max_trajectories=prompt_spec.max_trajectories,
            include_score=(prompt_spec.mode == _PROMPT_MODE_JSON_SCHEMA),
        )
        if prompt_spec.max_prompt_chars > _ZERO:
            trajectories = _trim_context_for_prompt(question, trajectories, prompt_spec)
        messages = _build_messages(question, trajectories, prompt_spec)
        processed += _ONE
        yield _LLMRequest(
            sample_id=sample_id,
            question=question,
            trajectories=list(trajectories),
            messages=messages,
        )


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _flush_batch(
    backend: "_LLMBackend",
    batch_items: List[_LLMRequest],
    f_out,
    prompt_spec: PromptSpec,
    output_spec: OutputSpec,
    schema_spec: SchemaSpec,
) -> int:
    if prompt_spec.mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC:
        return _flush_batch_subgraphrag(
            backend=backend,
            batch_items=batch_items,
            f_out=f_out,
            prompt_spec=prompt_spec,
            output_spec=output_spec,
            schema_spec=schema_spec,
        )
    messages_batch = [item.messages for item in batch_items]
    responses = backend.generate(messages_batch)
    responses = [(response or "").strip() for response in responses]
    parsed_list = [
        _parse_and_validate_response(response, prompt_spec, schema_spec)
        for response in responses
    ]
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
    for request, raw_response, parsed, retry_count in zip(
        batch_items, responses, parsed_list, retries
    ):
        if schema_spec.enabled and not parsed.schema_valid:
            log_event(
                log,
                "llm_schema_invalid",
                sample_id=request.sample_id,
                retries=retry_count,
            )
        answer_raw = (parsed.answer or "").strip()
        answer_final = answer_raw
        forced = False
        if not prompt_spec.allow_empty_answer and not answer_final:
            # Force a non-empty answer for downstream metric computation.
            forced = True
            if isinstance(parsed.payload, dict):
                best_guess = str(parsed.payload.get(_FIELD_BEST_GUESS) or "").strip()
                if best_guess:
                    answer_final = best_guess
            if not answer_final:
                answer_final = _DEFAULT_FALLBACK_ANSWER
            log_event(
                log,
                "llm_forced_non_empty_answer",
                sample_id=request.sample_id,
                answer=answer_final,
                schema_valid=bool(parsed.schema_valid),
                retries=retry_count,
            )
        schema_meta = {
            _FIELD_SCHEMA_VALID: bool(parsed.schema_valid),
            _FIELD_SCHEMA_RETRIES: int(retry_count),
        }
        if forced:
            schema_meta["forced_non_empty_answer"] = True
        extra = _build_output_extra(
            request, answer_raw, raw_response, output_spec, schema_meta=schema_meta
        )
        _write_answer(
            f_out, request.sample_id, answer_final, prompt_spec.answer_key, extra=extra
        )
        written += _ONE
    return written


def _flush_batch_subgraphrag(
    *,
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
    dc_retries = [0 for _ in batch_items]
    retry_idx = [
        idx for idx, resp in enumerate(responses) if _needs_subgraphrag_dc_retry(resp)
    ]
    if retry_idx and prompt_spec.cot_prompt.strip():
        retry_messages = [
            batch_items[idx].messages
            + [{"role": "user", "content": prompt_spec.cot_prompt}]
            for idx in retry_idx
        ]
        retry_outputs = backend.generate(retry_messages)
        for idx, output in zip(retry_idx, retry_outputs):
            responses[idx] = (output or "").strip()
            dc_retries[idx] = _ONE

    parsed_list = [
        _parse_and_validate_response(response, prompt_spec, schema_spec)
        for response in responses
    ]
    written = _ZERO
    for request, raw_response, parsed, retries in zip(
        batch_items, responses, parsed_list, dc_retries
    ):
        answer_raw = (parsed.answer or "").strip()
        answer_final = answer_raw
        candidate_pool = _extract_destination_candidates(
            request.trajectories,
            max_candidates=prompt_spec.max_candidates,
            question=request.question,
            candidate_source=prompt_spec.candidate_source,
        )
        constrained = False
        if prompt_spec.constrain_to_candidates:
            answer_final, constrained = _enforce_candidate_answers(
                answer_raw=answer_final,
                candidates=candidate_pool,
                answer_separator=prompt_spec.answer_separator,
                allow_empty=prompt_spec.allow_empty_answer,
            )
        if not prompt_spec.allow_empty_answer and not answer_final:
            constrained = True
            answer_final = _DEFAULT_FALLBACK_ANSWER
        schema_meta: Dict[str, Any] = {_FIELD_DC_RETRIES: retries}
        if constrained:
            schema_meta["candidate_constrained_answer"] = True
        extra = _build_output_extra(
            request, answer_final, raw_response, output_spec, schema_meta=schema_meta
        )
        _write_answer(
            f_out, request.sample_id, answer_final, prompt_spec.answer_key, extra=extra
        )
        written += _ONE
    return written


def _build_output_extra(
    request: _LLMRequest,
    answer: str,
    raw_response: str,
    spec: OutputSpec,
    *,
    schema_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Always keep lightweight structured fields for auditing/analysis.
    extra: Dict[str, Any] = {}
    if schema_meta:
        extra.update(schema_meta)
    extra.update(_extract_structured_model_fields(raw_response))
    if spec.debug_only_on_empty and answer.strip():
        return extra
    if spec.include_question:
        extra[_FIELD_QUESTION] = request.question
    if spec.include_trajectories:
        extra[_FIELD_SELECTED_TRAJECTORIES] = request.trajectories
    if spec.include_messages:
        extra[_FIELD_MESSAGES] = request.messages
    if spec.include_raw_response:
        extra[_FIELD_RAW_RESPONSE] = raw_response
    return extra


def _write_answer(
    f_out,
    sample_id: str,
    answer: str,
    answer_key: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
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
