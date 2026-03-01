from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from src.utils.logging_utils import get_logger, log_event

log = get_logger(__name__)

_ZERO = 0
_ONE = 1
_NEG_INF = float("-inf")

_DEFAULT_INPUT_SUBDIR = "eval_dual_flow"
_DEFAULT_OUTPUT_SUBDIR = "eval_llm"
_DEFAULT_INPUT_LABELS_SUFFIX = ".labels.jsonl"
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
_DEFAULT_MAX_EVIDENCE_IDS_PER_CANDIDATE = 8
_DEFAULT_MAX_PROMPT_NODE_CHARS = 120
_DEFAULT_MAX_PROMPT_REL_CHARS = 80
_DEFAULT_MAX_PROMPT_LAST_DST_CHARS = 240
_DEFAULT_MAX_PROMPT_CANDIDATE_CHARS = 240
_DEFAULT_FALLBACK_ANSWER = "unknown"
_DEFAULT_CANDIDATE_FUZZY_MATCH_THRESHOLD = 0.85
_DEFAULT_SUPER_SOURCE_ENTITY_ID = -1
_DEFAULT_CONSTRAIN_TO_CANDIDATES = True
_DEFAULT_CANDIDATE_SOURCE = "stop_only"
_CANDIDATE_SOURCE_STOP_ONLY = "stop_only"
_CANDIDATE_SOURCE_TRAJECTORY_NODES = "trajectory_nodes"
_DEFAULT_VLLM_PRETRIM_TO_BUDGET = True
_DEFAULT_VLLM_BUDGET_MARGIN = 0
_DEFAULT_SCHEMA_RETRY_MESSAGE = (
    "Your previous response did not match the required JSON schema. "
    "Do not refuse. Return a corrected JSON object only.\n\n"
    "Schema:\n{schema}\n\n"
    "Previous response:\n{response}"
)

_FIELD_SAMPLE_ID = "sample_id"
_FIELD_QUESTION = "question_text"
_FIELD_QUESTION_ALT = "question"
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
_FIELD_DC_RETRIES = "dc_retries"

_LOG_PROGRESS_EVERY = 200

_FREEBASE_ID_RE = re.compile(r"^[mg]\.[0-9a-z_]+$", flags=re.IGNORECASE)
_NUMERIC_CANDIDATE_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
_YEAR_VALUE_RE = re.compile(r"^\d{4}$")
_TRAILING_PARENS_RE = re.compile(r"\s*\([^)]*\)\s*$")

_NUMERIC_QUESTION_HINTS = (
    "when",
    "what year",
    "which year",
    "how many",
    "how much",
    "number of",
    "amount of",
    "percentage",
    "percent",
    "ratio",
    "population",
    "age",
    "score",
)

_YEAR_QUESTION_HINTS = ("when", "what year", "which year")

_PROMPT_MODE_JSON_SCHEMA = "json_schema"
_PROMPT_MODE_SUBGRAPHRAG_ICL_DC = "subgraphrag_icl_dc"


@dataclass(frozen=True)
class PromptSpec:
    mode: str
    system: str
    answer_key: str
    answer_separator: str
    allow_empty_answer: bool
    constrain_to_candidates: bool
    candidate_source: str
    max_prompt_chars: int
    max_trajectories: int
    max_candidates: int
    icl_user_prompt: str
    icl_assistant_prompt: str
    cot_prompt: str


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
    compute_metrics = bool(llm_cfg.get("compute_metrics", True))
    input_labels_path = _resolve_input_labels_path(
        input_path=input_path,
        llm_cfg=llm_cfg,
        require_labels=compute_metrics,
    )
    output_dir = _resolve_output_dir(dataset_cfg, llm_cfg, cfg.get("paths"))
    output_dir.mkdir(parents=True, exist_ok=True)
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
                input_path=input_path,
                input_labels_path=input_labels_path,
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
    answer_separator = str(prompt_cfg.get("answer_separator") or _DEFAULT_ANSWER_SEPARATOR)
    allow_empty_answer = bool(prompt_cfg.get("allow_empty", _DEFAULT_ALLOW_EMPTY_PROMPT_ANSWER))
    constrain_default = _DEFAULT_CONSTRAIN_TO_CANDIDATES if mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC else False
    constrain_to_candidates = bool(prompt_cfg.get("constrain_to_candidates", constrain_default))
    candidate_source = str(prompt_cfg.get("candidate_source", _DEFAULT_CANDIDATE_SOURCE)).strip().lower()
    if candidate_source not in {_CANDIDATE_SOURCE_STOP_ONLY, _CANDIDATE_SOURCE_TRAJECTORY_NODES}:
        raise ValueError(
            "llm.prompt.candidate_source must be one of "
            f"{{{_CANDIDATE_SOURCE_STOP_ONLY!r}, {_CANDIDATE_SOURCE_TRAJECTORY_NODES!r}}}."
        )
    max_prompt_chars = int(prompt_cfg.get("max_prompt_chars", _DEFAULT_MAX_PROMPT_CHARS))
    max_trajectories = int(prompt_cfg.get("max_trajectories", _DEFAULT_MAX_TRAJECTORIES_IN_PROMPT))
    max_candidates = int(prompt_cfg.get("max_candidates", _DEFAULT_MAX_CANDIDATES_IN_PROMPT))
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
    max_retry_chars = int(schema_cfg.get("max_retry_chars", _DEFAULT_SCHEMA_MAX_RETRY_CHARS))
    retry_message = str(schema_cfg.get("retry_message", _DEFAULT_SCHEMA_RETRY_MESSAGE)).strip()
    if max_retries < _ZERO:
        raise ValueError("llm.schema.max_retries must be >= 0.")
    if max_retry_chars < _ZERO:
        raise ValueError("llm.schema.max_retry_chars must be >= 0.")
    if prompt_spec.mode != _PROMPT_MODE_JSON_SCHEMA:
        if enabled:
            log_event(log, "llm_schema_disabled_for_prompt_mode", prompt_mode=prompt_spec.mode)
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
    answer_schema: Dict[str, Any] = {"type": "string"}
    if not prompt.allow_empty_answer:
        answer_schema["minLength"] = _ONE
    return {
        "type": "object",
        "properties": {
            prompt.answer_key: answer_schema,
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


def _validate_topk_against_prompt_limits(*, topk_list: Sequence[int], prompt_spec: PromptSpec) -> None:
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


def _resolve_input_labels_path(*, input_path: Path, llm_cfg: Any, require_labels: bool = False) -> Optional[Path]:
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
    input_labels_path: Optional[Path],
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
            input_labels_path=input_labels_path,
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
        question = str(record.get(_FIELD_QUESTION) or record.get(_FIELD_QUESTION_ALT) or "")
        rollouts = record.get(_FIELD_ROLLOUTS) or []
        trajectories = _select_trajectories(
            rollouts,
            top_k,
            max_trajectories=prompt_spec.max_trajectories,
            include_score=(prompt_spec.mode == _PROMPT_MODE_JSON_SCHEMA),
        )
        if prompt_spec.max_prompt_chars > _ZERO:
            trajectories = _trim_context_for_prompt(question, trajectories, prompt_spec)
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


def _select_trajectories(
    rollouts: Sequence[Dict[str, Any]],
    top_k: int,
    *,
    max_trajectories: int,
    include_score: bool,
) -> List[str]:
    sorted_rollouts = sorted(
        rollouts,
        key=lambda r: (float(r.get(_FIELD_SCORE, _NEG_INF)), int(r.get(_FIELD_ROLLOUT_INDEX, _ZERO))),
        reverse=True,
    )
    limit = int(top_k)
    if max_trajectories > _ZERO:
        limit = min(limit, int(max_trajectories))
    selected = sorted_rollouts[: limit]
    out: List[str] = []
    for rollout in selected:
        traj = _trajectory_text(rollout)
        if not traj:
            continue
        if include_score:
            # Expose rollout score to the model: higher-scoring trajectories are generally more trustworthy.
            score = rollout.get(_FIELD_SCORE)
            if score is None:
                out.append(traj)
                continue
            try:
                score_val = float(score)
            except Exception:
                out.append(traj)
                continue
            out.append(f"[score={score_val:.6g}] {traj}")
        else:
            out.append(traj)
    return out


def _trajectory_text(rollout: Dict[str, Any]) -> str:
    text = rollout.get(_FIELD_TRAJECTORY)
    if isinstance(text, str) and text.strip():
        return text.strip()
    edges = rollout.get(_FIELD_EDGES)
    if not isinstance(edges, list) or not edges:
        stop_node = rollout.get("stop_node_entity_id")
        # Some rollouts terminate immediately (no edges). We still surface the terminal node so the LLM can
        # pick a non-empty answer (numeric entity id) and metrics can score it.
        if stop_node is None:
            return ""
        try:
            stop_int = int(stop_node)
        except Exception:
            return ""
        if stop_int < _ZERO:
            return ""
        return f"(no_edge) --STOP--> {stop_int}"
    filtered_edges = [edge for edge in edges if not _is_super_source_edge(edge)]
    parts = [_edge_to_text(edge) for edge in filtered_edges]
    parts = [p for p in parts if p]
    if parts:
        return " ; ".join(parts)
    stop_node = rollout.get("stop_node_entity_id")
    if stop_node is None:
        return ""
    try:
        stop_int = int(stop_node)
    except Exception:
        return ""
    if stop_int < _ZERO:
        return ""
    return f"(no_edge) --STOP--> {stop_int}"


def _edge_to_text(edge: Dict[str, Any]) -> str:
    src = edge.get("src_text") or edge.get("src_entity_id")
    rel = edge.get("relation_text") or edge.get("relation_id")
    dst = edge.get("dst_text") or edge.get("dst_entity_id")
    if rel == _DEFAULT_STOP_RELATION or str(rel) == str(_DEFAULT_STOP_RELATION):
        rel = "SELF"
    return f"{src} --{rel}--> {dst}"


def _trim_context_for_prompt(question: str, trajectories: Sequence[str], prompt: PromptSpec) -> List[str]:
    if prompt.mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC:
        return _trim_trajectories_for_subgraphrag_prompt(question, trajectories, prompt)
    return _trim_trajectories_for_prompt(question, trajectories, prompt)


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
            # Skip a single oversized trajectory instead of dropping all remaining ones.
            continue
        kept = candidate
    return kept


def _trim_trajectories_for_subgraphrag_prompt(
    question: str,
    trajectories: Sequence[str],
    prompt: PromptSpec,
) -> List[str]:
    max_chars = int(prompt.max_prompt_chars)
    if max_chars <= _ZERO:
        return list(trajectories)
    kept: List[str] = []
    base_chars = len(prompt.system) + len(prompt.icl_user_prompt) + len(prompt.icl_assistant_prompt) + len(prompt.cot_prompt)
    for traj in trajectories:
        candidate = kept + [traj]
        user_text = _build_subgraphrag_user_text(question, candidate, prompt)
        total_chars = int(base_chars) + len(user_text)
        if total_chars > max_chars:
            continue
        kept = candidate
    return kept


def _build_user_text(question: str, trajectories: Sequence[str], prompt: PromptSpec) -> str:
    lines = []
    for idx, traj in enumerate(trajectories, start=_ONE):
        lines.append(f"{idx}. {_sanitize_trajectory_for_prompt(str(traj))}")
    traj_block = "\n".join(lines) if lines else "(no trajectories)"
    candidates = _extract_destination_candidates_with_evidence(
        trajectories,
        max_candidates=prompt.max_candidates,
        max_ids_per_candidate=_DEFAULT_MAX_EVIDENCE_IDS_PER_CANDIDATE,
        question=question,
        candidate_source=prompt.candidate_source,
    )
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
    if prompt.allow_empty_answer:
        answer_example_abstain = (
            "{"
            f'"{prompt.answer_key}": "", '
            f'"{_FIELD_EVIDENCE_TRAJECTORY_IDS}": [], '
            f'"{_FIELD_ABSTAIN_REASON}": "no_supported_candidate", '
            f'"{_FIELD_BEST_GUESS}": "Candidate X", '
            f'"{_FIELD_JUSTIFICATION}": "Candidates are present but none answers the question."'
            "}"
        )
        empty_clause = (
            f'Only set "{prompt.answer_key}" to an empty string when there is no supported answer. '
            "Prefer selecting the best-supported candidate entity over returning an empty answer when at least one "
            "candidate is plausible."
        )
        abstain_rule = (
            f'- If you output an empty "{prompt.answer_key}", set "{_FIELD_ABSTAIN_REASON}" to a short reason string '
            f'and fill "{_FIELD_BEST_GUESS}" with the closest candidate (or empty if none).\n'
        )
        examples = f"{answer_example_single}\n{answer_example_multi}\n{answer_example_abstain}\n\n"
    else:
        answer_example_uncertain = (
            "{"
            f'"{prompt.answer_key}": "Candidate X", '
            f'"{_FIELD_EVIDENCE_TRAJECTORY_IDS}": [3], '
            f'"{_FIELD_ABSTAIN_REASON}": "insufficient_evidence", '
            f'"{_FIELD_BEST_GUESS}": "Candidate X", '
            f'"{_FIELD_JUSTIFICATION}": "Best-supported candidate from trajectory 3."'
            "}"
        )
        empty_clause = (
            f'Always return a non-empty string for "{prompt.answer_key}". '
            "If insufficient evidence, set it to your best guess from the candidate list; "
            f'if there are no candidates, output "{_DEFAULT_FALLBACK_ANSWER}".'
        )
        abstain_rule = (
            f'- If uncertain, keep "{prompt.answer_key}" non-empty (set it to "{_FIELD_BEST_GUESS}") and set '
            f'"{_FIELD_ABSTAIN_REASON}" to a short reason string.\n'
        )
        examples = f"{answer_example_single}\n{answer_example_multi}\n{answer_example_uncertain}\n\n"
    return (
        "Question:\n"
        f"{question}\n\n"
        "Trajectories:\n"
        f"{traj_block}\n\n"
        "Candidate answer entities (trajectory-derived; each line shows support count and evidence indices):\n"
        f"{candidate_block}\n\n"
        "Return a single JSON object with the following schema:\n"
        f"{answer_schema}\n\n"
        "Rules:\n"
        f'- The value of "{prompt.answer_key}" must be a string.\n'
        "- Use exact surface forms from the trajectories (or the candidate list).\n"
        '- If selecting from the candidate list, output only the entity string before " (support:" (exclude the parentheses).\n'
        "- Trajectories are prefixed with a numeric score; higher is generally more reliable.\n"
        f'- If multiple answers, join exactly with "{prompt.answer_separator}" (example below).\n'
        f'- "{_FIELD_EVIDENCE_TRAJECTORY_IDS}" must list 1-based trajectory indices that directly support the answer.\n'
        f'- "{_FIELD_JUSTIFICATION}" must be short (<= {_DEFAULT_MAX_JUSTIFICATION_WORDS} words).\n'
        f"{abstain_rule}"
        f"- {empty_clause}\n\n"
        "Examples:\n"
        f"{examples}"
        "Output JSON only."
    )


def _build_messages(question: str, trajectories: Sequence[str], prompt: PromptSpec) -> List[Dict[str, str]]:
    if prompt.mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC:
        return _build_subgraphrag_messages(question, trajectories, prompt)
    return _build_json_messages(question, trajectories, prompt)


def _build_json_messages(question: str, trajectories: Sequence[str], prompt: PromptSpec) -> List[Dict[str, str]]:
    user_text = _build_user_text(question, trajectories, prompt)
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user_text}]


def _build_subgraphrag_messages(question: str, trajectories: Sequence[str], prompt: PromptSpec) -> List[Dict[str, str]]:
    user_text = _build_subgraphrag_user_text(question, trajectories, prompt)
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.icl_user_prompt},
        {"role": "assistant", "content": prompt.icl_assistant_prompt},
        {"role": "user", "content": user_text},
    ]


def _build_subgraphrag_user_text(question: str, trajectories: Sequence[str], prompt: PromptSpec) -> str:
    triplet_lines = _extract_subgraphrag_triplet_lines_from_trajectories(trajectories)
    candidate_lines = _extract_destination_candidates_with_evidence(
        trajectories,
        max_candidates=prompt.max_candidates,
        max_ids_per_candidate=_DEFAULT_MAX_EVIDENCE_IDS_PER_CANDIDATE,
        question=question,
        candidate_source=prompt.candidate_source,
    )
    lines = ["Triplets:"]
    if triplet_lines:
        lines.extend(triplet_lines)
    else:
        lines.append("(none)")
    lines.extend(["", "Candidate answers (must choose from this list when non-empty):"])
    if candidate_lines:
        lines.extend(f"- {candidate}" for candidate in candidate_lines)
    else:
        lines.append("(none)")
    lines.extend(
        [
            "",
            "Output format:",
            '- Output one or more lines starting with "ans:".',
            '- If candidate answers are listed, each "ans:" value must exactly match one candidate entity string (before " (support:").',
            f'- If no candidate is supported, output "ans: {_DEFAULT_FALLBACK_ANSWER}".',
            "",
            "Question:",
            str(question or "").strip(),
        ]
    )
    return "\n".join(lines)


def _extract_subgraphrag_triplet_lines_from_trajectories(trajectories: Sequence[str]) -> List[str]:
    """Extract SubgraphRAG-style `(h,r,t)` lines from retrieval elements.

    Important: retrieval elements are *paths* for DualFlow and *triples* for edge retriever.
    We therefore group triplets by trajectory and insert a blank line between elements so that
    later token-budget trimming can drop whole retrieval elements without cutting inside a path.
    """

    out: List[str] = []
    for traj in trajectories:
        cleaned = _strip_score_prefix(str(traj or ""))
        sanitized = _sanitize_trajectory_for_prompt(cleaned)
        segments = [s.strip() for s in sanitized.split(" ; ") if s.strip()]
        group_lines: List[str] = []
        for seg in segments:
            parsed = _try_parse_edge_segment(seg)
            if parsed is None:
                continue
            src, rel, dst = parsed
            if str(rel).strip().upper() in {"SELF", "STOP"}:
                continue
            if str(src).strip() == "(no_edge)":
                continue
            # Hide virtual super-source edges from LLM prompts.
            if _is_super_source_node_text(src):
                continue
            group_lines.append(f"({src},{rel},{dst})")
        if group_lines:
            out.extend(group_lines)
            out.append("")
    while out and out[-1] == "":
        out.pop()
    return out


def _strip_score_prefix(text: str) -> str:
    raw = str(text or "").lstrip()
    if not raw.startswith("[score="):
        return raw
    end = raw.find("] ")
    if end < _ZERO:
        return raw
    return raw[end + len("] ") :].lstrip()


def _extract_destination_candidates(
    trajectories: Sequence[str],
    *,
    max_candidates: int,
    question: str = "",
    candidate_source: str = _DEFAULT_CANDIDATE_SOURCE,
) -> List[str]:
    seen: set[str] = set()
    candidates: List[str] = []
    for traj in trajectories:
        for candidate in _extract_trajectory_candidates(str(traj or ""), candidate_source=candidate_source):
            if not candidate or candidate in seen or not _is_prompt_candidate_ok(candidate, question=question):
                continue
            seen.add(candidate)
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _extract_destination_candidates_with_evidence(
    trajectories: Sequence[str],
    *,
    max_candidates: int,
    max_ids_per_candidate: int,
    question: str = "",
    candidate_source: str = _DEFAULT_CANDIDATE_SOURCE,
) -> List[str]:
    """Return `Candidate (evidence: i, j, ...)` lines for prompting.

    This is intentionally prompt-oriented (string output) to keep the call site simple.
    """

    if max_candidates <= _ZERO:
        return []
    if max_ids_per_candidate <= _ZERO:
        max_ids_per_candidate = _ONE

    # Preserve first-seen order (rollouts are already score-sorted) and track support counts.
    candidate_ids: Dict[str, List[int]] = {}
    candidate_support: Dict[str, int] = {}
    for idx, traj in enumerate(trajectories, start=_ONE):
        local_seen: set[str] = set()
        for candidate in _extract_trajectory_candidates(str(traj or ""), candidate_source=candidate_source):
            if not candidate or candidate in local_seen or not _is_prompt_candidate_ok(candidate, question=question):
                continue
            local_seen.add(candidate)
            ids = candidate_ids.get(candidate)
            if ids is not None:
                candidate_support[candidate] = candidate_support.get(candidate, _ZERO) + _ONE
                if len(ids) < max_ids_per_candidate:
                    ids.append(int(idx))
                continue
            if len(candidate_ids) >= max_candidates:
                continue
            candidate_ids[candidate] = [int(idx)]
            candidate_support[candidate] = candidate_support.get(candidate, _ZERO) + _ONE

    out: List[str] = []
    for candidate, ids in candidate_ids.items():
        support = int(candidate_support.get(candidate, len(ids)))
        out.append(f"{candidate} (support: {support}, evidence: {', '.join(str(i) for i in ids)})")
    return out


def _extract_trajectory_candidates(trajectory: str, *, candidate_source: str) -> List[str]:
    raw = str(trajectory or "").strip()
    if not raw:
        return []
    source = str(candidate_source or _DEFAULT_CANDIDATE_SOURCE).strip().lower()
    if source == _CANDIDATE_SOURCE_STOP_ONLY:
        candidate = _extract_trajectory_stop_candidate(raw)
        return [candidate] if candidate else []
    if source != _CANDIDATE_SOURCE_TRAJECTORY_NODES:
        raise ValueError(
            "Unsupported candidate_source "
            f"{source!r}. Expected one of {{{_CANDIDATE_SOURCE_STOP_ONLY!r}, {_CANDIDATE_SOURCE_TRAJECTORY_NODES!r}}}."
        )
    return _extract_trajectory_node_candidates(raw)


def _extract_trajectory_stop_candidate(trajectory: str) -> str:
    arrow = trajectory.rfind("-->")
    if arrow < _ZERO:
        return ""
    return trajectory[arrow + len("-->") :].strip()


def _extract_trajectory_node_candidates(trajectory: str) -> List[str]:
    cleaned = _sanitize_trajectory_for_prompt(_strip_score_prefix(trajectory))
    segments = [s.strip() for s in cleaned.split(" ; ") if s.strip()]
    nodes: List[str] = []
    for seg in segments:
        parsed = _try_parse_edge_segment(seg)
        if parsed is None:
            continue
        src, rel, dst = parsed
        if str(rel).strip().upper() in {"SELF", "STOP"}:
            continue
        if str(src).strip() == "(no_edge)":
            continue
        if _is_super_source_node_text(src):
            continue
        src_clean = _normalize_prompt_text(src)
        dst_clean = _normalize_prompt_text(dst)
        if src_clean:
            nodes.append(src_clean)
        if dst_clean:
            nodes.append(dst_clean)
    if nodes:
        return _remove_duplicates_preserve_order(nodes)
    fallback = _extract_trajectory_stop_candidate(trajectory)
    if not fallback:
        return []
    return [fallback]


def _is_prompt_candidate_ok(candidate: str, *, question: str = "") -> bool:
    text = _normalize_prompt_text(str(candidate))
    if not text:
        return False
    if len(text) > _DEFAULT_MAX_PROMPT_CANDIDATE_CHARS:
        return False
    # Avoid emitting opaque KB IDs as "answers" in forced fallback paths.
    if _FREEBASE_ID_RE.match(text.strip()) is not None:
        return False
    if _is_numeric_candidate(text):
        if not _question_allows_numeric_answer(question):
            return False
        if _question_prefers_year_answer(question) and _YEAR_VALUE_RE.match(text.replace(",", "").strip()) is None:
            return False
    return True


def _sanitize_trajectory_for_prompt(traj: str) -> str:
    raw = _normalize_prompt_text(str(traj))
    if not raw:
        return ""
    segments = [s.strip() for s in raw.split(" ; ") if s.strip()]
    if not segments:
        return raw
    out: List[str] = []
    for i, seg in enumerate(segments):
        is_last = i == (len(segments) - 1)
        parsed = _try_parse_edge_segment(seg)
        if parsed is None:
            out.append(_truncate_text(seg, _DEFAULT_MAX_PROMPT_NODE_CHARS))
            continue
        src, rel, dst = parsed
        # Drop virtual super-source edges before prompting.
        if _is_super_source_node_text(src):
            continue
        src = _truncate_text(src, _DEFAULT_MAX_PROMPT_NODE_CHARS)
        rel = _truncate_text(rel, _DEFAULT_MAX_PROMPT_REL_CHARS)
        if is_last:
            dst = _truncate_text(dst, _DEFAULT_MAX_PROMPT_LAST_DST_CHARS)
        else:
            dst = _truncate_text(dst, _DEFAULT_MAX_PROMPT_NODE_CHARS)
        out.append(f"{src} --{rel}--> {dst}")
    return " ; ".join(out)


def _try_parse_edge_segment(seg: str) -> Optional[Tuple[str, str, str]]:
    arrow = seg.rfind("-->")
    if arrow < _ZERO:
        return None
    left = seg[:arrow].rstrip()
    dst = seg[arrow + len("-->") :].strip()
    sep = left.rfind(" --")
    if sep < _ZERO:
        return None
    src = left[:sep].strip()
    rel = left[sep + len(" --") :].strip()
    if not src or not rel or not dst:
        return None
    return src, rel, dst


def _is_super_source_node_text(node_text: str) -> bool:
    text = str(node_text or "").strip()
    if not text:
        return False
    if text == str(_DEFAULT_SUPER_SOURCE_ENTITY_ID):
        return True
    return text.lower() in {"super_source", "__super_source__"}


def _is_super_source_edge(edge: Dict[str, Any]) -> bool:
    for key in ("src_entity_id", "head_entity_id"):
        value = edge.get(key)
        try:
            if int(value) == _DEFAULT_SUPER_SOURCE_ENTITY_ID:
                return True
        except Exception:
            continue
    src_text = str(edge.get("src_text") or edge.get("head_text") or "").strip()
    if not src_text:
        return False
    return _is_super_source_node_text(src_text)


def _normalize_prompt_text(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").replace("\r", " ").split())


def _is_numeric_candidate(text: str) -> bool:
    cleaned = _normalize_prompt_text(text).replace(",", "").strip()
    if not cleaned:
        return False
    return _NUMERIC_CANDIDATE_RE.match(cleaned) is not None


def _question_allows_numeric_answer(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    if not lowered:
        return False
    return any(hint in lowered for hint in _NUMERIC_QUESTION_HINTS)


def _question_prefers_year_answer(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    if not lowered:
        return False
    return any(hint in lowered for hint in _YEAR_QUESTION_HINTS)


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = _normalize_prompt_text(text)
    if max_chars <= _ZERO or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


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
        extra = _build_output_extra(request, answer_raw, raw_response, output_spec, schema_meta=schema_meta)
        _write_answer(f_out, request.sample_id, answer_final, prompt_spec.answer_key, extra=extra)
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
    retry_idx = [idx for idx, resp in enumerate(responses) if _needs_subgraphrag_dc_retry(resp)]
    if retry_idx and prompt_spec.cot_prompt.strip():
        retry_messages = [
            batch_items[idx].messages + [{"role": "user", "content": prompt_spec.cot_prompt}] for idx in retry_idx
        ]
        retry_outputs = backend.generate(retry_messages)
        for idx, output in zip(retry_idx, retry_outputs):
            responses[idx] = (output or "").strip()
            dc_retries[idx] = _ONE

    parsed_list = [_parse_and_validate_response(response, prompt_spec, schema_spec) for response in responses]
    written = _ZERO
    for request, raw_response, parsed, retries in zip(batch_items, responses, parsed_list, dc_retries):
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
        extra = _build_output_extra(request, answer_final, raw_response, output_spec, schema_meta=schema_meta)
        _write_answer(f_out, request.sample_id, answer_final, prompt_spec.answer_key, extra=extra)
        written += _ONE
    return written


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


def _split_answer_tokens(answer_raw: str, *, answer_separator: str) -> List[str]:
    raw = str(answer_raw or "").strip()
    if not raw:
        return []
    separator = str(answer_separator or "")
    if separator and separator in raw:
        parts = raw.split(separator)
    elif "\n" in raw:
        parts = raw.splitlines()
    else:
        parts = [raw]
    out: List[str] = []
    for part in parts:
        token = str(part or "").strip()
        if not token:
            continue
        if token.lower().startswith("ans:"):
            token = token[len("ans:") :].strip()
        if token:
            out.append(token)
    return _remove_duplicates_preserve_order(out)


def _normalize_candidate_key(text: str) -> str:
    token = _normalize_prompt_text(str(text))
    if not token:
        return ""
    lowered = token.casefold()
    for marker in (" (support:", " (evidence:"):
        idx = lowered.find(marker)
        if idx >= _ZERO:
            token = token[:idx].rstrip()
            lowered = token.casefold()
    token = _TRAILING_PARENS_RE.sub("", token).strip()
    token = token.strip("\"'`").strip()
    return _normalize_prompt_text(token).casefold()


def _approximate_candidate_match(token_key: str, normalized_candidates: Sequence[Tuple[str, str]]) -> Optional[str]:
    if not token_key:
        return None
    best_score = float(_NEG_INF)
    best_value: Optional[str] = None
    for cand_key, raw_candidate in normalized_candidates:
        score = _candidate_match_score(token_key, cand_key)
        if score > best_score:
            best_score = score
            best_value = raw_candidate
    if best_value is None:
        return None
    if best_score < float(_DEFAULT_CANDIDATE_FUZZY_MATCH_THRESHOLD):
        return None
    return best_value


def _candidate_match_score(token_key: str, candidate_key: str) -> float:
    if not token_key or not candidate_key:
        return float(_NEG_INF)
    if token_key == candidate_key:
        return 1.0
    if token_key in candidate_key or candidate_key in token_key:
        shorter = min(len(token_key), len(candidate_key))
        longer = max(len(token_key), len(candidate_key))
        if longer <= _ZERO:
            return float(_NEG_INF)
        return float(shorter) / float(longer)
    token_words = {w for w in token_key.split() if w}
    candidate_words = {w for w in candidate_key.split() if w}
    if token_words and candidate_words:
        union = token_words | candidate_words
        if union:
            overlap = float(len(token_words & candidate_words)) / float(len(union))
        else:
            overlap = float(_NEG_INF)
    else:
        overlap = float(_NEG_INF)
    ratio = SequenceMatcher(None, token_key, candidate_key).ratio()
    return max(overlap, ratio)


def _enforce_candidate_answers(
    *,
    answer_raw: str,
    candidates: Sequence[str],
    answer_separator: str,
    allow_empty: bool,
) -> Tuple[str, bool]:
    tokens = _split_answer_tokens(answer_raw, answer_separator=answer_separator)
    if not tokens:
        if allow_empty:
            return "", False
        return _DEFAULT_FALLBACK_ANSWER, True
    if not candidates:
        return answer_separator.join(tokens), False
    cand_map: Dict[str, str] = {}
    normalized_candidates: List[Tuple[str, str]] = []
    for candidate in candidates:
        key = _normalize_candidate_key(candidate)
        if not key or key in cand_map:
            continue
        cand_map[key] = str(candidate)
        normalized_candidates.append((key, str(candidate)))
    kept: List[str] = []
    for token in tokens:
        key = _normalize_candidate_key(token)
        matched = cand_map.get(key)
        if matched is None:
            matched = _approximate_candidate_match(key, normalized_candidates)
        if matched is None:
            continue
        kept.append(matched)
    kept = _remove_duplicates_preserve_order(kept)
    if kept:
        constrained = _normalize_candidate_key(answer_separator.join(tokens)) != _normalize_candidate_key(
            answer_separator.join(kept)
        )
        return answer_separator.join(kept), constrained
    if allow_empty:
        return "", True
    return _DEFAULT_FALLBACK_ANSWER, True


def _subgraphrag_get_pred_lines(prediction: str) -> List[str]:
    raw = str(prediction or "")
    candidates = [p for p in raw.split("\n") if "ans:" in p and "none" not in p.lower()]
    if candidates:
        candidates = [
            p
            for p in candidates
            if "ans: not available" not in p.lower() and "ans: no information available" not in p.lower()
        ]
    return _remove_duplicates_preserve_order(candidates)


def _remove_duplicates_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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
    # If empty answers are disallowed, salvage "answer" from best_guess to reduce retries.
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


_VLLM_PROMPT_TOO_LONG_MARKERS = ("longer than the maximum model length", "maximum model length")


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


def _count_vllm_chat_tokens(tokenizer: Any, messages: List[Dict[str, str]]) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    lines = ["Triplets:", *triplet_lines, "", "", "Question:", str(question or "").strip()]
    return "\n".join(lines)


def _replace_message_content(messages: Sequence[Dict[str, str]], idx: int, content: str) -> List[Dict[str, str]]:
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
        content = _format_subgraphrag_user_content(_flatten_groups(groups[:mid]), question)
        candidate = _replace_message_content(messages, user_idx, content)
        tokens = _count_vllm_chat_tokens(tokenizer, candidate)
        if tokens is not None and tokens <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    final_content = _format_subgraphrag_user_content(_flatten_groups(groups[:best]), question)
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
        return _trim_last_user_suffix_to_budget(copied, user_idx=user_idx, tokenizer=tokenizer, budget=budget)
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
    return _trim_last_user_suffix_to_budget(trimmed, user_idx=user_idx, tokenizer=tokenizer, budget=budget)


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
        llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len_int)
    else:
        try:
            llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len_int,
                seed=seed_int,
            )
        except TypeError:
            llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len_int)
    if max_model_len_int is None:
        max_model_len_int = _infer_vllm_max_model_len(llm)
    tokenizer = None
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = None
    max_tokens = int(provider_cfg.get("max_tokens") or _ONE)
    pretrim_to_budget = bool(provider_cfg.get("pretrim_to_budget", _DEFAULT_VLLM_PRETRIM_TO_BUDGET))
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
        if pretrim_to_budget and tokenizer is not None and max_model_len_int is not None:
            budget = int(max_model_len_int) - int(max_tokens) - int(budget_margin)
            if budget > _ZERO:
                chat_batch = [
                    _trim_messages_to_vllm_budget(messages, tokenizer=tokenizer, budget=budget)
                    for messages in messages_batch
                ]
        try:
            outputs = llm.chat(chat_batch, sampling_params=sampling_params, use_tqdm=False)
        except ValueError as exc:
            if tokenizer is None or max_model_len_int is None or not _is_vllm_prompt_too_long_error(exc):
                raise
            budget = int(max_model_len_int) - int(max_tokens) - int(budget_margin)
            trimmed_batch = [
                _trim_messages_to_vllm_budget(messages, tokenizer=tokenizer, budget=budget) for messages in messages_batch
            ]
            outputs = llm.chat(trimmed_batch, sampling_params=sampling_params, use_tqdm=False)
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
