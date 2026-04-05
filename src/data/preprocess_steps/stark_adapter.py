from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch

from .sample_types import RawSample

_PRIME_DATASET = "prime"
_SOURCE_SPLIT_BY_NORMALIZED_SPLIT = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_WHITESPACE_RE = re.compile(r"\s+")
_FS_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TEXT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "these",
        "this",
        "to",
        "what",
        "which",
        "who",
        "with",
    }
)
_DEFAULT_MAX_CANDIDATES = 12
_DEFAULT_MAX_ENTITIES = 4
_DEFAULT_NUM_HOPS = 2
_DEFAULT_MAX_NODES = 64
_DEFAULT_MAX_EDGES = 256
_DEFAULT_MAX_ALIAS_VALUES = 8
_DEFAULT_SUMMARY_CHARS = 220
_DEFAULT_VLLM_MAX_TOKENS = 192
_DEFAULT_VLLM_TENSOR_PARALLEL = 1


def _int_or_default(value: object, default: int) -> int:
    if value in (None, ""):
        return int(default)
    return int(value)


def _float_or_default(value: object, default: float) -> float:
    if value in (None, ""):
        return float(default)
    return float(value)


@dataclass(frozen=True)
class PrimeNodeRecord:
    node_id: int
    entity_id: str
    name: str
    node_type: str
    source: str
    summary: str
    normalized_name: str
    normalized_aliases: tuple[str, ...]
    search_tokens: frozenset[str]


@dataclass(frozen=True)
class PrimeCandidate:
    node_id: int
    score: float
    record: PrimeNodeRecord


_ADAPTER_CACHE: Dict[str, "StarkPrimeAdapter"] = {}


def iter_stark_samples(
    *,
    dataset: str,
    kb: str,
    splits: Sequence[str],
    stark_cfg: Mapping[str, object],
) -> Iterable[RawSample]:
    adapter = _get_cached_prime_adapter(dataset=dataset, kb=kb, stark_cfg=stark_cfg)
    return adapter.iter_samples(splits)


def _get_cached_prime_adapter(
    *, dataset: str, kb: str, stark_cfg: Mapping[str, object]
) -> "StarkPrimeAdapter":
    cache_key = json.dumps(
        {
            "dataset": dataset,
            "kb": kb,
            "stark_cfg": _mapping_to_plain_dict(stark_cfg),
        },
        sort_keys=True,
        default=str,
    )
    adapter = _ADAPTER_CACHE.get(cache_key)
    if adapter is not None:
        return adapter
    adapter = StarkPrimeAdapter(dataset=dataset, kb=kb, stark_cfg=stark_cfg)
    _ADAPTER_CACHE[cache_key] = adapter
    return adapter


def _mapping_to_plain_dict(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _mapping_to_plain_dict(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mapping_to_plain_dict(item) for item in value]
    return value


def _normalize_text(text: object) -> str:
    value = _WHITESPACE_RE.sub(" ", str(text or "")).strip()
    return value


def _tokenize(text: object) -> List[str]:
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(_normalize_text(text).lower()):
        token = match.group(0).strip("._/-")
        if len(token) < 2:
            continue
        if token in _TEXT_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _sanitize_fs_component(value: str) -> str:
    cleaned = _FS_SAFE_RE.sub("_", str(value or "")).strip("._")
    return cleaned or "sample"


def _extract_json_dict(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return {}
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _stable_signature_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _build_sample_cache_signature(
    *,
    dataset: str,
    kb: str,
    indirected: bool,
    linker_cfg: Mapping[str, object],
    local_graph_cfg: Mapping[str, object],
) -> str:
    payload = {
        "dataset": str(dataset),
        "kb": str(kb),
        "indirected": bool(indirected),
        "linker": _mapping_to_plain_dict(linker_cfg),
        "local_graph": _mapping_to_plain_dict(local_graph_cfg),
    }
    return _stable_signature_json(payload)


def _require_cached_payload_list(
    payload: Mapping[str, object],
    *,
    field_name: str,
    legacy_field_name: str,
    cache_path: Path,
) -> list[object]:
    if field_name in payload:
        value = payload.get(field_name)
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []
    if legacy_field_name in payload:
        raise ValueError(
            f"Legacy STaRK cache payload at {cache_path} uses {legacy_field_name}; "
            f"delete cached samples and rebuild to emit {field_name}."
        )
    return []


def _flatten_string_values(value: object, *, limit: int) -> List[str]:
    if limit <= 0:
        return []
    queue: deque[object] = deque([value])
    out: List[str] = []
    seen: set[str] = set()
    while queue and len(out) < limit:
        current = queue.popleft()
        if current is None:
            continue
        if isinstance(current, str):
            text = _normalize_text(current)
            if not text or text.lower() == "no meta data":
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
            continue
        if isinstance(current, Mapping):
            for key, item in current.items():
                if str(key).startswith("_"):
                    continue
                queue.append(item)
            continue
        if isinstance(current, (list, tuple, set)):
            for item in current:
                queue.append(item)
            continue
        text = _normalize_text(current)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _coerce_int_list(values: object) -> List[int]:
    if values is None:
        return []
    if torch.is_tensor(values):
        flat = values.detach().view(-1).tolist()
        return [int(item) for item in flat]
    if isinstance(values, (list, tuple, set)):
        return [int(item) for item in values]
    return [int(values)]


class StarkPrimeAdapter:
    def __init__(
        self,
        *,
        dataset: str,
        kb: str,
        stark_cfg: Mapping[str, object],
    ) -> None:
        dataset_name = str(stark_cfg.get("dataset") or _PRIME_DATASET).strip().lower()
        if dataset_name != _PRIME_DATASET:
            raise ValueError(
                f"Only STaRK prime is supported today, got stark.dataset={dataset_name!r}."
            )
        self.dataset = str(dataset)
        self.kb = str(kb)
        self.dataset_name = dataset_name
        self.root = _optional_path(stark_cfg.get("root"))
        self.cache_dir = _optional_path(stark_cfg.get("cache_dir"))
        self.download_processed = bool(stark_cfg.get("download_processed", True))
        self.indirected = bool(stark_cfg.get("indirected", False))
        self.linker_cfg = dict(_as_mapping(stark_cfg.get("linker")))
        self.local_graph_cfg = dict(_as_mapping(stark_cfg.get("local_graph")))
        self._sample_cache_signature = _build_sample_cache_signature(
            dataset=self.dataset,
            kb=self.kb,
            indirected=self.indirected,
            linker_cfg=self.linker_cfg,
            local_graph_cfg=self.local_graph_cfg,
        )
        self._sample_cache: Dict[tuple[str, str], RawSample] = {}
        self._vllm_generate = None
        self._qa_dataset, self._skb = _load_prime_resources(
            root=self.root,
            download_processed=self.download_processed,
            indirected=self.indirected,
        )
        self._split_indices = _normalize_split_indices(self._qa_dataset.get_idx_split())
        self._node_records = self._build_node_records()
        self._token_to_node_ids: Dict[str, List[int]] = defaultdict(list)
        self._token_idf: Dict[str, float] = {}
        self._adjacency_out: Dict[int, List[tuple[int, int, int]]] = defaultdict(list)
        self._adjacency_in: Dict[int, List[tuple[int, int, int]]] = defaultdict(list)
        self._build_token_index()
        self._build_adjacency()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_samples(self, splits: Sequence[str]) -> Iterator[RawSample]:
        for split in splits:
            normalized_split = _normalize_requested_split(split)
            source_split = _SOURCE_SPLIT_BY_NORMALIZED_SPLIT[normalized_split]
            split_indices = self._split_indices.get(source_split)
            if split_indices is None:
                available = sorted(self._split_indices)
                raise ValueError(
                    f"STaRK split {source_split!r} is unavailable. Available splits: {available}."
                )
            for dataset_index in split_indices:
                query, question_id, answer_ids, _meta = self._qa_dataset[
                    int(dataset_index)
                ]
                yield self._load_or_build_sample(
                    split=normalized_split,
                    question_id=str(question_id),
                    question=str(query),
                    answer_ids=_coerce_int_list(answer_ids),
                )

    def _build_node_records(self) -> Dict[int, PrimeNodeRecord]:
        records: Dict[int, PrimeNodeRecord] = {}
        for node_id in range(int(self._skb.num_nodes())):
            node = self._skb[node_id]
            node_info = dict(self._skb.node_info[int(node_id)])
            node_type = (
                _normalize_text(getattr(node, "type", node_info.get("type"))) or "node"
            )
            name = _normalize_text(getattr(node, "name", node_info.get("name")))
            if not name:
                name = f"node_{node_id}"
            source = _normalize_text(getattr(node, "source", node_info.get("source")))
            aliases = self._extract_aliases(node_info)
            summary = self._build_summary(
                node_info=node_info, node_type=node_type, source=source
            )
            search_tokens = frozenset(
                _tokenize(name) + _tokenize(" ".join(aliases)) + _tokenize(node_type)
            )
            entity_id = self._entity_id_for_node(
                node_id=node_id,
                node_name=name,
                node_type=node_type,
            )
            records[node_id] = PrimeNodeRecord(
                node_id=node_id,
                entity_id=entity_id,
                name=name,
                node_type=node_type,
                source=source,
                summary=summary,
                normalized_name=name.lower(),
                normalized_aliases=tuple(alias.lower() for alias in aliases),
                search_tokens=search_tokens,
            )
        return records

    def _build_token_index(self) -> None:
        num_records = max(len(self._node_records), 1)
        posting_counts: Dict[str, int] = defaultdict(int)
        for record in self._node_records.values():
            for token in sorted(record.search_tokens):
                self._token_to_node_ids[token].append(record.node_id)
                posting_counts[token] += 1
        for token, count in posting_counts.items():
            self._token_idf[token] = math.log((1.0 + num_records) / (1.0 + count)) + 1.0

    def _build_adjacency(self) -> None:
        edge_index = torch.as_tensor(self._skb.edge_index, dtype=torch.long)
        edge_types = torch.as_tensor(self._skb.edge_types, dtype=torch.long).view(-1)
        if edge_index.numel() == 0:
            return
        for edge_id in range(int(edge_index.shape[1])):
            src = int(edge_index[0, edge_id].item())
            dst = int(edge_index[1, edge_id].item())
            rel_id = int(edge_types[edge_id].item())
            self._adjacency_out[src].append((edge_id, dst, rel_id))
            self._adjacency_in[dst].append((edge_id, src, rel_id))

    def _extract_aliases(self, node_info: Mapping[str, object]) -> List[str]:
        aliases: List[str] = []
        details = node_info.get("details")
        if isinstance(details, Mapping):
            for key in (
                "alias",
                "aliases",
                "other_names",
                "synonym",
                "synonyms",
                "gene_synonym",
            ):
                aliases.extend(
                    _flatten_string_values(
                        details.get(key), limit=_DEFAULT_MAX_ALIAS_VALUES - len(aliases)
                    )
                )
                if len(aliases) >= _DEFAULT_MAX_ALIAS_VALUES:
                    break
        deduped: List[str] = []
        seen: set[str] = set()
        for alias in aliases:
            lowered = alias.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(alias)
        return deduped[:_DEFAULT_MAX_ALIAS_VALUES]

    def _build_summary(
        self,
        *,
        node_info: Mapping[str, object],
        node_type: str,
        source: str,
    ) -> str:
        summary_bits = [f"type={node_type}"]
        if source:
            summary_bits.append(f"source={source}")
        details = node_info.get("details")
        if isinstance(details, Mapping):
            detail_lines: List[str] = []
            for key, value in details.items():
                if str(key).startswith("_"):
                    continue
                values = _flatten_string_values(value, limit=2)
                if not values:
                    continue
                joined = "; ".join(values)
                detail_lines.append(f"{key}={joined}")
                if len(detail_lines) >= 3:
                    break
            if detail_lines:
                summary_bits.append("details=" + " | ".join(detail_lines))
        summary = " ; ".join(summary_bits)
        return summary[:_DEFAULT_SUMMARY_CHARS].rstrip()

    def _entity_id_for_node(
        self, *, node_id: int, node_name: str, node_type: str
    ) -> str:
        return f"prime[{_normalize_text(node_type)}] {_normalize_text(node_name)} <id={int(node_id)}>"

    def _load_or_build_sample(
        self,
        *,
        split: str,
        question_id: str,
        question: str,
        answer_ids: Sequence[int],
    ) -> RawSample:
        cache_key = (split, question_id)
        cached = self._sample_cache.get(cache_key)
        if cached is not None:
            return cached
        cache_path = self._sample_cache_path(split=split, question_id=question_id)
        if cache_path is not None and cache_path.exists():
            payload = self._load_cached_payload(cache_path)
            if self._payload_cache_signature_matches(payload):
                sample = self._sample_from_payload(payload, cache_path=cache_path)
                self._sample_cache[cache_key] = sample
                return sample
        sample, metadata = self._build_sample(
            split=split,
            question_id=question_id,
            question=question,
            answer_ids=answer_ids,
        )
        if cache_path is not None:
            payload = self._sample_to_payload(sample)
            payload["metadata"] = metadata
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        self._sample_cache[cache_key] = sample
        return sample

    def _sample_cache_path(self, *, split: str, question_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        filename = f"{_sanitize_fs_component(question_id)}.json"
        return self.cache_dir / "samples" / split / filename

    def _payload_cache_signature_matches(self, payload: Mapping[str, object]) -> bool:
        return str(payload.get("cache_signature") or "") == self._sample_cache_signature

    def _load_cached_payload(self, path: Path) -> Dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Cached STaRK sample at {path} must be a JSON object.")
        return payload

    def _sample_from_payload(
        self, payload: Mapping[str, object], *, cache_path: Path
    ) -> RawSample:
        question_entities = _require_cached_payload_list(
            payload,
            field_name="question_entities",
            legacy_field_name="q_entity",
            cache_path=cache_path,
        )
        answer_entities = _require_cached_payload_list(
            payload,
            field_name="answer_entities",
            legacy_field_name="a_entity",
            cache_path=cache_path,
        )
        graph = [
            (str(edge[0]), str(edge[1]), str(edge[2]))
            for edge in payload.get("graph", [])
            if isinstance(edge, (list, tuple)) and len(edge) >= 3
        ]
        return RawSample(
            dataset=str(payload.get("dataset") or self.dataset),
            split=str(payload.get("split") or "train"),
            question_id=str(payload.get("question_id") or ""),
            kb=str(payload.get("kb") or self.kb),
            question=str(payload.get("question") or ""),
            graph=graph,
            question_entities=[str(item) for item in question_entities],
            answer_entities=[str(item) for item in answer_entities],
            answer_texts=[str(item) for item in payload.get("answer_texts", [])],
        )

    def _load_cached_sample(self, path: Path) -> RawSample:
        payload = self._load_cached_payload(path)
        return self._sample_from_payload(payload, cache_path=path)

    def _sample_to_payload(self, sample: RawSample) -> Dict[str, object]:
        return {
            "dataset": sample.dataset,
            "split": sample.split,
            "question_id": sample.question_id,
            "kb": sample.kb,
            "question": sample.question,
            "graph": [list(edge) for edge in sample.graph],
            "question_entities": list(sample.question_entities),
            "answer_entities": list(sample.answer_entities),
            "answer_texts": list(sample.answer_texts),
            "cache_signature": self._sample_cache_signature,
        }

    def _build_sample(
        self,
        *,
        split: str,
        question_id: str,
        question: str,
        answer_ids: Sequence[int],
    ) -> tuple[RawSample, Dict[str, object]]:
        candidates = self._retrieve_candidates(question)
        anchor_node_ids, linker_metadata = self._select_anchor_node_ids(
            question=question,
            candidates=candidates,
        )
        graph = self._extract_local_graph(anchor_node_ids)
        question_entities = [
            self._node_records[node_id].entity_id for node_id in anchor_node_ids
        ]
        answer_entities: List[str] = []
        answer_texts: List[str] = []
        for node_id in answer_ids:
            record = self._node_records.get(int(node_id))
            if record is None:
                continue
            answer_entities.append(record.entity_id)
            answer_texts.append(record.name)
        metadata = {
            "candidate_node_ids": [candidate.node_id for candidate in candidates],
            "anchor_node_ids": list(anchor_node_ids),
            "linker": linker_metadata,
        }
        sample = RawSample(
            dataset=self.dataset,
            split=split,
            question_id=str(question_id),
            kb=self.kb,
            question=str(question),
            graph=graph,
            question_entities=question_entities,
            answer_entities=answer_entities,
            answer_texts=answer_texts,
        )
        return sample, metadata

    def _retrieve_candidates(self, question: str) -> List[PrimeCandidate]:
        max_candidates = _int_or_default(
            self.linker_cfg.get("max_candidates"), _DEFAULT_MAX_CANDIDATES
        )
        query_tokens = set(_tokenize(question))
        scores: Dict[int, float] = defaultdict(float)
        for token in sorted(query_tokens):
            for node_id in self._token_to_node_ids.get(token, []):
                scores[node_id] += self._token_idf.get(token, 1.0)
        normalized_question = _normalize_text(question).lower()
        if not scores and normalized_question:
            for record in self._node_records.values():
                if len(record.normalized_name) < 4:
                    continue
                if record.normalized_name in normalized_question:
                    scores[record.node_id] += 8.0
        candidates: List[PrimeCandidate] = []
        for node_id, score in scores.items():
            record = self._node_records[node_id]
            overlap = len(query_tokens & set(record.search_tokens))
            score += 0.5 * float(overlap)
            if record.normalized_name and record.normalized_name in normalized_question:
                score += 6.0
            if any(
                alias and len(alias) >= 3 and alias in normalized_question
                for alias in record.normalized_aliases
            ):
                score += 4.0
            candidates.append(
                PrimeCandidate(node_id=node_id, score=score, record=record)
            )
        candidates.sort(
            key=lambda item: (-item.score, len(item.record.name), item.node_id)
        )
        return candidates[:max_candidates]

    def _select_anchor_node_ids(
        self,
        *,
        question: str,
        candidates: Sequence[PrimeCandidate],
    ) -> tuple[List[int], Dict[str, object]]:
        backend = str(self.linker_cfg.get("backend") or "keyword").strip().lower()
        if backend == "vllm" and candidates:
            selected_ids, raw_response = self._select_with_vllm(
                question=question,
                candidates=candidates,
            )
            selected_ids = self._dedup_node_ids(selected_ids)
            if selected_ids:
                return selected_ids, {
                    "backend": backend,
                    "raw_response": raw_response,
                }
        heuristic_ids = self._select_with_heuristics(
            question=question, candidates=candidates
        )
        return heuristic_ids, {"backend": backend, "raw_response": None}

    def _select_with_heuristics(
        self,
        *,
        question: str,
        candidates: Sequence[PrimeCandidate],
    ) -> List[int]:
        max_entities = _int_or_default(
            self.linker_cfg.get("max_entities"), _DEFAULT_MAX_ENTITIES
        )
        normalized_question = _normalize_text(question).lower()
        query_tokens = set(_tokenize(question))
        selected: List[int] = []
        for candidate in candidates:
            record = candidate.record
            overlap = len(query_tokens & set(record.search_tokens))
            alias_match = any(
                alias and len(alias) >= 3 and alias in normalized_question
                for alias in record.normalized_aliases
            )
            if record.normalized_name and record.normalized_name in normalized_question:
                selected.append(candidate.node_id)
                continue
            if overlap >= 2 or alias_match:
                selected.append(candidate.node_id)
        if not selected and candidates:
            selected.append(candidates[0].node_id)
        return self._dedup_node_ids(selected)[:max_entities]

    def _select_with_vllm(
        self,
        *,
        question: str,
        candidates: Sequence[PrimeCandidate],
    ) -> tuple[List[int], str]:
        generate = self._get_vllm_generate()
        messages = [
            {
                "role": "system",
                "content": (
                    "You link biomedical questions to PrimeKG entities. "
                    "Choose only from the provided candidate ids. "
                    "Return JSON only with the shape "
                    '{"selected_candidate_ids": [<int>, ...], "mentions": [<string>, ...]}. '
                    "Pick entities that should be anchor nodes for graph retrieval."
                ),
            },
            {
                "role": "user",
                "content": self._build_vllm_user_prompt(
                    question=question, candidates=candidates
                ),
            },
        ]
        outputs = generate([messages])
        raw_response = str(outputs[0] if outputs else "")
        payload = _extract_json_dict(raw_response)
        candidate_ids = {candidate.node_id for candidate in candidates}
        selected = []
        for item in payload.get("selected_candidate_ids", []):
            try:
                node_id = int(item)
            except Exception:
                continue
            if node_id in candidate_ids:
                selected.append(node_id)
        max_entities = _int_or_default(
            self.linker_cfg.get("max_entities"), _DEFAULT_MAX_ENTITIES
        )
        return self._dedup_node_ids(selected)[:max_entities], raw_response

    def _build_vllm_user_prompt(
        self,
        *,
        question: str,
        candidates: Sequence[PrimeCandidate],
    ) -> str:
        candidate_lines = []
        for candidate in candidates:
            record = candidate.record
            candidate_lines.append(
                "- id={id} | type={type} | name={name} | summary={summary}".format(
                    id=candidate.node_id,
                    type=record.node_type,
                    name=record.name,
                    summary=record.summary,
                )
            )
        max_entities = _int_or_default(
            self.linker_cfg.get("max_entities"), _DEFAULT_MAX_ENTITIES
        )
        lines = [
            "Question:",
            str(question).strip(),
            "",
            f"Select up to {max_entities} candidate ids.",
            "Candidates:",
            *candidate_lines,
        ]
        return "\n".join(lines)

    def _get_vllm_generate(self):
        if self._vllm_generate is not None:
            return self._vllm_generate
        model = str(self.linker_cfg.get("model") or "").strip()
        if not model:
            raise ValueError(
                "stark.linker.model must be set when stark.linker.backend=vllm. "
                "Set STARK_VLLM_MODEL or override stark.linker.model on the CLI."
            )
        from src.llm.backends import _build_vllm_generate

        provider_cfg = {
            "model": model,
            "tensor_parallel_size": _int_or_default(
                self.linker_cfg.get("tensor_parallel_size"),
                _DEFAULT_VLLM_TENSOR_PARALLEL,
            ),
            "max_model_len": self.linker_cfg.get("max_model_len"),
            "max_tokens": _int_or_default(
                self.linker_cfg.get("max_tokens"), _DEFAULT_VLLM_MAX_TOKENS
            ),
            "temperature": _float_or_default(self.linker_cfg.get("temperature"), 0.0),
            "top_p": _float_or_default(self.linker_cfg.get("top_p"), 1.0),
            "seed": self.linker_cfg.get("seed"),
            "pretrim_to_budget": bool(self.linker_cfg.get("pretrim_to_budget", True)),
            "budget_margin": _int_or_default(self.linker_cfg.get("budget_margin"), 0),
        }
        self._vllm_generate = _build_vllm_generate(provider_cfg)
        return self._vllm_generate

    def _extract_local_graph(
        self, anchor_node_ids: Sequence[int]
    ) -> List[tuple[str, str, str]]:
        if not anchor_node_ids:
            return []
        num_hops = _int_or_default(
            self.local_graph_cfg.get("num_hops"), _DEFAULT_NUM_HOPS
        )
        max_nodes = _int_or_default(
            self.local_graph_cfg.get("max_nodes"), _DEFAULT_MAX_NODES
        )
        max_edges = _int_or_default(
            self.local_graph_cfg.get("max_edges"), _DEFAULT_MAX_EDGES
        )
        include_inverse_edges = bool(
            self.local_graph_cfg.get("include_inverse_edges", False)
        )
        direction = str(self.local_graph_cfg.get("direction") or "both").strip().lower()
        if max_nodes < len(anchor_node_ids):
            raise ValueError(
                f"local_graph.max_nodes={max_nodes} is smaller than anchor count={len(anchor_node_ids)}."
            )
        visited_order: List[int] = []
        visited_set: set[int] = set()
        frontier = deque((int(node_id), 0) for node_id in anchor_node_ids)
        while frontier and len(visited_order) < max_nodes:
            node_id, depth = frontier.popleft()
            if node_id in visited_set:
                continue
            visited_set.add(node_id)
            visited_order.append(node_id)
            if depth >= num_hops:
                continue
            neighbors = []
            if direction in {"out", "both"}:
                neighbors.extend(
                    dst for _edge_id, dst, _rel_id in self._adjacency_out[node_id]
                )
            if direction in {"in", "both"}:
                neighbors.extend(
                    src for _edge_id, src, _rel_id in self._adjacency_in[node_id]
                )
            for neighbor in neighbors:
                if neighbor in visited_set:
                    continue
                frontier.append((neighbor, depth + 1))
        node_set = set(visited_order)
        graph: List[tuple[str, str, str]] = []
        emitted_edges: set[tuple[str, str, str]] = set()
        for src in visited_order:
            for _edge_id, dst, rel_id in self._adjacency_out[src]:
                if dst not in node_set:
                    continue
                relation = self._relation_label(rel_id)
                edge = (
                    self._node_records[src].entity_id,
                    relation,
                    self._node_records[dst].entity_id,
                )
                if edge in emitted_edges:
                    continue
                emitted_edges.add(edge)
                graph.append(edge)
                if len(graph) >= max_edges:
                    return graph
                if include_inverse_edges:
                    inverse_edge = (
                        self._node_records[dst].entity_id,
                        f"inverse::{relation}",
                        self._node_records[src].entity_id,
                    )
                    if inverse_edge not in emitted_edges:
                        emitted_edges.add(inverse_edge)
                        graph.append(inverse_edge)
                        if len(graph) >= max_edges:
                            return graph
        return graph

    def _relation_label(self, relation_id: int) -> str:
        return str(self._skb.get_edge_type_by_id(int(relation_id)))

    def _dedup_node_ids(self, node_ids: Iterable[int]) -> List[int]:
        seen: set[int] = set()
        deduped: List[int] = []
        for node_id in node_ids:
            value = int(node_id)
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped


def _load_prime_resources(
    *, root: Optional[Path], download_processed: bool, indirected: bool
):
    try:
        from stark_qa import load_qa, load_skb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stark-qa is required for dataset_source=stark. Install `stark-qa` first."
        ) from exc
    root_str = str(root) if root is not None else None
    qa_dataset = load_qa(_PRIME_DATASET, root=root_str)
    skb = load_skb(
        _PRIME_DATASET,
        root=root_str,
        download_processed=download_processed,
        indirected=indirected,
    )
    return qa_dataset, skb


def _normalize_split_indices(
    split_indices: Mapping[str, object],
) -> Dict[str, List[int]]:
    normalized: Dict[str, List[int]] = {}
    for split, indices in split_indices.items():
        source_split = str(split).strip().lower()
        normalized[source_split] = _coerce_int_list(indices)
    return normalized


def _normalize_requested_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in _SOURCE_SPLIT_BY_NORMALIZED_SPLIT:
        allowed = sorted(_SOURCE_SPLIT_BY_NORMALIZED_SPLIT)
        raise ValueError(
            f"Unsupported split {split!r} for STaRK source. Expected one of {allowed}."
        )
    return normalized


def _optional_path(value: object) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser().resolve()


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


__all__ = ["StarkPrimeAdapter", "iter_stark_samples"]
