from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ZERO = 0
_ONE = 1
_NEG_INF = float("-inf")

_DEFAULT_MAX_JUSTIFICATION_WORDS = 25
_DEFAULT_MAX_EVIDENCE_IDS_PER_CANDIDATE = 8
_DEFAULT_MAX_PROMPT_NODE_CHARS = 120
_DEFAULT_MAX_PROMPT_REL_CHARS = 80
_DEFAULT_MAX_PROMPT_LAST_DST_CHARS = 240
_DEFAULT_MAX_PROMPT_CANDIDATE_CHARS = 240
_DEFAULT_FALLBACK_ANSWER = "unknown"
_DEFAULT_CANDIDATE_FUZZY_MATCH_THRESHOLD = 0.85
_DEFAULT_SUPER_SOURCE_ENTITY_ID = -1
_DEFAULT_CONSTRAIN_TO_CANDIDATES = True
_DEFAULT_CANDIDATE_SOURCE = "endpoints_only"
_CANDIDATE_SOURCE_ENDPOINTS_ONLY = "endpoints_only"
_CANDIDATE_SOURCE_TRAJECTORY_NODES = "trajectory_nodes"

_FIELD_QUESTION = "question"
_FIELD_TRAJECTORIES = "trajectories"
_FIELD_TRAJECTORY = "trajectory_text"
_FIELD_EDGES = "edges"
_FIELD_PROB = "prob"
_FIELD_PATH_RANK = "path_rank"
_FIELD_SINGLETON_TERMINAL_ANSWER_SET_ENTITY_ID = (
    "singleton_terminal_answer_set_entity_id"
)
_FIELD_SINGLETON_TERMINAL_ANSWER_SET_ENTITY_TEXT = (
    "singleton_terminal_answer_set_entity_text"
)
_FIELD_EVIDENCE_TRAJECTORY_IDS = "evidence_trajectory_ids"
_FIELD_ABSTAIN_REASON = "abstain_reason"
_FIELD_BEST_GUESS = "best_guess"
_FIELD_JUSTIFICATION = "justification"

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


def _select_trajectories(
    trajectories: Sequence[Dict[str, Any]],
    top_k: int,
    *,
    max_trajectories: int,
    include_score: bool,
) -> List[str]:
    sorted_trajectories = sorted(
        trajectories,
        key=lambda r: (
            float(r.get(_FIELD_PROB, _NEG_INF)),
            -int(r.get(_FIELD_PATH_RANK, _ZERO)),
        ),
        reverse=True,
    )
    limit = int(top_k)
    if max_trajectories > _ZERO:
        limit = min(limit, int(max_trajectories))
    selected = sorted_trajectories[:limit]
    out: List[str] = []
    for trajectory in selected:
        traj = _trajectory_text(trajectory)
        if not traj:
            continue
        if include_score:
            # Expose trajectory probability to the model: higher-mass trajectories are generally more trustworthy.
            prob = trajectory.get(_FIELD_PROB)
            if prob is None:
                out.append(traj)
                continue
            try:
                prob_value = float(prob)
            except Exception:
                out.append(traj)
                continue
            out.append(f"[prob={prob_value:.6g}] {traj}")
        else:
            out.append(traj)
    return out


def _trajectory_text(trajectory: Dict[str, Any]) -> str:
    text = trajectory.get(_FIELD_TRAJECTORY)
    if isinstance(text, str) and text.strip():
        return text.strip()
    edges = trajectory.get(_FIELD_EDGES)
    if not isinstance(edges, list) or not edges:
        stop_node = trajectory.get(_FIELD_SINGLETON_TERMINAL_ANSWER_SET_ENTITY_TEXT)
        if stop_node is None:
            stop_node = trajectory.get(_FIELD_SINGLETON_TERMINAL_ANSWER_SET_ENTITY_ID)
        # Some trajectories terminate immediately (no edges). We still surface the terminal node so the LLM can
        # pick a non-empty answer (numeric entity id) and metrics can score it.
        if stop_node is None:
            return ""
        stop_text = str(stop_node).strip()
        if not stop_text:
            return ""
        return f"(start_only) {stop_text}"
    filtered_edges = [edge for edge in edges if not _is_super_source_edge(edge)]
    parts = [_edge_to_text(edge) for edge in filtered_edges]
    parts = [p for p in parts if p]
    if parts:
        return " ; ".join(parts)
    stop_node = trajectory.get(_FIELD_SINGLETON_TERMINAL_ANSWER_SET_ENTITY_TEXT)
    if stop_node is None:
        stop_node = trajectory.get(_FIELD_SINGLETON_TERMINAL_ANSWER_SET_ENTITY_ID)
    if stop_node is None:
        return ""
    stop_text = str(stop_node).strip()
    if not stop_text:
        return ""
    return f"(start_only) {stop_text}"


def _edge_to_text(edge: Dict[str, Any]) -> str:
    src = edge.get("src_text") or edge.get("src_entity_id")
    rel = edge.get("relation_text") or edge.get("relation_id")
    dst = edge.get("dst_text") or edge.get("dst_entity_id")
    return f"{src} --{rel}--> {dst}"


def _trim_context_for_prompt(
    question: str, trajectories: Sequence[str], prompt: PromptSpec
) -> List[str]:
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
    base_chars = (
        len(prompt.system)
        + len(prompt.icl_user_prompt)
        + len(prompt.icl_assistant_prompt)
        + len(prompt.cot_prompt)
    )
    for traj in trajectories:
        candidate = kept + [traj]
        user_text = _build_subgraphrag_user_text(question, candidate, prompt)
        total_chars = int(base_chars) + len(user_text)
        if total_chars > max_chars:
            continue
        kept = candidate
    return kept


def _build_user_text(
    question: str, trajectories: Sequence[str], prompt: PromptSpec
) -> str:
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
        "- Trajectories are prefixed with their probability mass; higher is generally more reliable.\n"
        f'- If multiple answers, join exactly with "{prompt.answer_separator}" (example below).\n'
        f'- "{_FIELD_EVIDENCE_TRAJECTORY_IDS}" must list 1-based trajectory indices that directly support the answer.\n'
        f'- "{_FIELD_JUSTIFICATION}" must be short (<= {_DEFAULT_MAX_JUSTIFICATION_WORDS} words).\n'
        f"{abstain_rule}"
        f"- {empty_clause}\n\n"
        "Examples:\n"
        f"{examples}"
        "Output JSON only."
    )


def _build_messages(
    question: str, trajectories: Sequence[str], prompt: PromptSpec
) -> List[Dict[str, str]]:
    if prompt.mode == _PROMPT_MODE_SUBGRAPHRAG_ICL_DC:
        return _build_subgraphrag_messages(question, trajectories, prompt)
    return _build_json_messages(question, trajectories, prompt)


def _build_json_messages(
    question: str, trajectories: Sequence[str], prompt: PromptSpec
) -> List[Dict[str, str]]:
    user_text = _build_user_text(question, trajectories, prompt)
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_text},
    ]


def _build_subgraphrag_messages(
    question: str, trajectories: Sequence[str], prompt: PromptSpec
) -> List[Dict[str, str]]:
    user_text = _build_subgraphrag_user_text(question, trajectories, prompt)
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.icl_user_prompt},
        {"role": "assistant", "content": prompt.icl_assistant_prompt},
        {"role": "user", "content": user_text},
    ]


def _build_subgraphrag_user_text(
    question: str, trajectories: Sequence[str], prompt: PromptSpec
) -> str:
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


def _extract_subgraphrag_triplet_lines_from_trajectories(
    trajectories: Sequence[str],
) -> List[str]:
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
    if not raw.startswith("[prob="):
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
        for candidate in _extract_trajectory_candidates(
            str(traj or ""), candidate_source=candidate_source
        ):
            if (
                not candidate
                or candidate in seen
                or not _is_prompt_candidate_ok(candidate, question=question)
            ):
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
    if max_candidates <= _ZERO:
        return []
    if max_ids_per_candidate <= _ZERO:
        max_ids_per_candidate = _ONE

    candidate_ids: Dict[str, List[int]] = {}
    candidate_support: Dict[str, int] = {}
    for idx, traj in enumerate(trajectories, start=_ONE):
        local_seen: set[str] = set()
        for candidate in _extract_trajectory_candidates(
            str(traj or ""), candidate_source=candidate_source
        ):
            if (
                not candidate
                or candidate in local_seen
                or not _is_prompt_candidate_ok(candidate, question=question)
            ):
                continue
            local_seen.add(candidate)
            ids = candidate_ids.get(candidate)
            if ids is not None:
                candidate_support[candidate] = (
                    candidate_support.get(candidate, _ZERO) + _ONE
                )
                if len(ids) < max_ids_per_candidate:
                    ids.append(int(idx))
                continue
            if len(candidate_ids) >= max_candidates:
                continue
            candidate_ids[candidate] = [int(idx)]
            candidate_support[candidate] = (
                candidate_support.get(candidate, _ZERO) + _ONE
            )

    out: List[str] = []
    for candidate, ids in candidate_ids.items():
        support = int(candidate_support.get(candidate, len(ids)))
        out.append(
            f"{candidate} (support: {support}, evidence: {', '.join(str(i) for i in ids)})"
        )
    return out


def _extract_trajectory_candidates(
    trajectory: str, *, candidate_source: str
) -> List[str]:
    raw = str(trajectory or "").strip()
    if not raw:
        return []
    source = str(candidate_source or _DEFAULT_CANDIDATE_SOURCE).strip().lower()
    if source == _CANDIDATE_SOURCE_ENDPOINTS_ONLY:
        candidate = _extract_trajectory_endpoint_candidate(raw)
        return [candidate] if candidate else []
    if source != _CANDIDATE_SOURCE_TRAJECTORY_NODES:
        raise ValueError(
            "Unsupported candidate_source "
            f"{source!r}. Expected one of {{{_CANDIDATE_SOURCE_ENDPOINTS_ONLY!r}, {_CANDIDATE_SOURCE_TRAJECTORY_NODES!r}}}."
        )
    return _extract_trajectory_node_candidates(raw)


def _extract_trajectory_endpoint_candidate(trajectory: str) -> str:
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


def _extract_trajectory_stop_candidate(trajectory: str) -> str:
    cleaned = _sanitize_trajectory_for_prompt(_strip_score_prefix(trajectory))
    if not cleaned:
        return ""
    if cleaned.startswith("(start_only)"):
        return _normalize_prompt_text(cleaned[len("(start_only)") :])
    candidate = _extract_trajectory_endpoint_candidate(cleaned)
    if candidate:
        return _normalize_prompt_text(candidate)
    return _normalize_prompt_text(cleaned)


def _is_prompt_candidate_ok(candidate: str, *, question: str = "") -> bool:
    text = _normalize_prompt_text(str(candidate))
    if not text:
        return False
    if len(text) > _DEFAULT_MAX_PROMPT_CANDIDATE_CHARS:
        return False
    if _FREEBASE_ID_RE.match(text.strip()) is not None:
        return False
    if _is_numeric_candidate(text):
        if not _question_allows_numeric_answer(question):
            return False
        if (
            _question_prefers_year_answer(question)
            and _YEAR_VALUE_RE.match(text.replace(",", "").strip()) is None
        ):
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
    return text.lower() in {
        "super_source",
        "__super_source__",
        "question_super",
        "answer_super",
        "__question_super__",
        "__answer_super__",
    }


def _is_super_source_edge(edge: Dict[str, Any]) -> bool:
    for key in (
        "src_entity_id",
        "head_entity_id",
        "dst_entity_id",
        "tail_entity_id",
    ):
        value = edge.get(key)
        try:
            if int(value) == _DEFAULT_SUPER_SOURCE_ENTITY_ID:
                return True
        except Exception:
            continue
    for key in ("src_text", "head_text", "dst_text", "tail_text"):
        text = str(edge.get(key) or "").strip()
        if text and _is_super_source_node_text(text):
            return True
    return False


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


def _approximate_candidate_match(
    token_key: str, normalized_candidates: Sequence[Tuple[str, str]]
) -> Optional[str]:
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
        constrained = _normalize_candidate_key(
            answer_separator.join(tokens)
        ) != _normalize_candidate_key(answer_separator.join(kept))
        return answer_separator.join(kept), constrained
    if allow_empty:
        return "", True
    return _DEFAULT_FALLBACK_ANSWER, True


def _remove_duplicates_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
