import json
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset

from .components import (
    DISTANCE_CACHE_PT,
    DistancePTStore,
    EmbeddingStore,
    GlobalEmbeddingStore,
    GraphStore,
    SharedDataResources,
)
from src.utils.logging_utils import get_logger, log_event

logger = get_logger(__name__)
_QUESTIONS_PARQUET_FILENAME = "questions.parquet"
_PARQUET_SCAN_BATCH_SIZE = 65536
_MIN_SPLIT_SAMPLE_COUNT = 1
_ALLOWED_SPLITS = ("train", "validation", "test")
_SAMPLE_IDS_SOURCE_LMDB = "lmdb"
_SAMPLE_IDS_SOURCE_PARQUET = "parquet"
_ALLOWED_SAMPLE_ID_SOURCES = {_SAMPLE_IDS_SOURCE_LMDB, _SAMPLE_IDS_SOURCE_PARQUET}
_LMDB_SHARD_TOKEN = ".shard"
_LMDB_SHARD_GLOB_TEMPLATE = "{split}.shard*.lmdb"
_DUPLICATE_SAMPLE_ID_PREVIEW = 5
_CACHE_DISABLED = 0
_CACHE_MAX_ENTRIES_DEFAULT = 0
_DISTANCE_PRELOAD_BATCH_SIZE = 256
_FILTER_MISSING_START_BATCH_SIZE = 256
_FILTER_MISSING_ANSWER_BATCH_SIZE = 256
_SEQUENTIAL_SAMPLE_IDS_DEFAULT = False
_ZERO = 0
_DEPRECATED_RAW_KEYS = ("labels", "answer_subgraph", "topic_pe", "node_topic_dist")
_DEPRECATED_RAW_PREFIXES = ("pair_",)
_REMOVED_DATASET_KEYS = (
    "edge_dropout_p",
    "edge_dropout_min_edges",
    "edge_dropout_strategy",
    "edge_dropout_qa_bias",
    "edge_dropout_degree_bias",
    "edge_dropout_degree_power",
)


class GRetrievalData(GraphData):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> int:
        if key in {"q_local_indices", "a_local_indices"}:
            return int(self.num_nodes)
        if key in {"node_type_counts", "node_type_ids"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class GRetrievalDataset(Dataset):

    def __init__(
        self,
        *,
        split_path: Path,
        vocabulary_path: Path,
        embeddings_dir: Path,
        dataset_name: str = "unknown",
        split_name: str = "train",
        resources: Optional[SharedDataResources] = None,
        sample_limit: Optional[int] = None,
        sample_filter_path: Optional[Union[Path, Sequence[Path]]] = None,
        random_seed: Optional[int] = None,
        validate_on_init: bool = False,
        validate_on_get: bool = True,
        sample_ids: Optional[Sequence[str]] = None,
        cache_node_min_dists: bool = False,
        cache_max_entries: int = _CACHE_MAX_ENTRIES_DEFAULT,
        sequential_sample_ids: bool = _SEQUENTIAL_SAMPLE_IDS_DEFAULT,
        distance_cache_format: str = DISTANCE_CACHE_PT,
        distance_cache_path: Optional[Path] = None,
        lmdb_readahead: bool = False,
        preload_node_min_dists: bool = False,
        filter_missing_start: bool = False,
        filter_missing_answer: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = _assert_allowed_split_name(split_name)
        self.split_path = Path(split_path)
        self._split_paths = _resolve_lmdb_split_paths(self.split_path)
        self._num_shards = len(self._split_paths)
        self._vocabulary_path = Path(vocabulary_path)
        self._embeddings_dir = Path(embeddings_dir)
        self._shared_resources = resources
        self._graph_store: Optional[GraphStore] = None
        self._global_embeddings: Optional[GlobalEmbeddingStore] = None
        self._sample_stores: Optional[List[EmbeddingStore]] = None
        self._sample_ids_source = _SAMPLE_IDS_SOURCE_LMDB
        self._validate_on_get = bool(validate_on_get)
        self._preload_node_min_dists = bool(preload_node_min_dists)
        self._cache_node_min_dists = bool(cache_node_min_dists) or self._preload_node_min_dists
        self._cache_max_entries = int(cache_max_entries)
        self._sequential_sample_ids = bool(sequential_sample_ids)
        self._distance_cache_format = str(distance_cache_format).strip().lower()
        if self._distance_cache_format != DISTANCE_CACHE_PT:
            raise ValueError(f"distance_cache_format must be '{DISTANCE_CACHE_PT}', got {self._distance_cache_format!r}.")
        self._distance_cache_path = Path(distance_cache_path) if distance_cache_path is not None else None
        self._lmdb_readahead = bool(lmdb_readahead)
        self._distance_store: Optional[object] = None
        self._distance_store_checked = False
        self._node_min_dists_cache: Optional[Dict[str, torch.Tensor]] = None
        if self._cache_node_min_dists:
            if not self._preload_node_min_dists and self._cache_max_entries <= _CACHE_DISABLED:
                raise ValueError("cache_max_entries must be positive when cache_node_min_dists is enabled.")
            if self._preload_node_min_dists:
                self._node_min_dists_cache = {}
            else:
                self._node_min_dists_cache = OrderedDict()

        self._assert_split_path_exists()
        self._init_distance_store()
        self._init_resources(vocabulary_path, embeddings_dir)
        self._init_sample_ids(sample_ids)
        self._assert_sample_ids_match_lmdb()
        self._apply_filters(
            sample_filter_path=sample_filter_path,
            sample_limit=sample_limit,
            random_seed=random_seed,
            validate_on_init=validate_on_init,
            filter_missing_start=bool(filter_missing_start),
            filter_missing_answer=bool(filter_missing_answer),
        )
        self._maybe_sort_sample_ids()
        self._maybe_preload_node_min_dists()

        log_event(
            logger,
            "g_retrieval_dataset_init",
            split=self.split,
            samples=len(self.sample_ids),
        )

    def _get_cached_node_min_dists(self, sample_id: str) -> Optional[torch.Tensor]:
        if self._node_min_dists_cache is None:
            return None
        cached = self._node_min_dists_cache.get(sample_id)
        if cached is None:
            return None
        if not self._preload_node_min_dists and isinstance(self._node_min_dists_cache, OrderedDict):
            self._node_min_dists_cache.move_to_end(sample_id)
        return cached

    def _store_cached_node_min_dists(self, sample_id: str, node_min_dists: torch.Tensor) -> None:
        if self._node_min_dists_cache is None:
            return
        self._node_min_dists_cache[sample_id] = node_min_dists.detach()
        if self._preload_node_min_dists:
            return
        if isinstance(self._node_min_dists_cache, OrderedDict):
            self._node_min_dists_cache.move_to_end(sample_id)
            if len(self._node_min_dists_cache) > self._cache_max_entries:
                self._node_min_dists_cache.popitem(last=False)

    def _load_distance_cache(self, sample_id: str, num_nodes: int) -> torch.Tensor:
        if self._distance_store is None:
            self._init_distance_store()
        if self._distance_store is None:
            raise RuntimeError("Distance cache not initialized; configure distances_dir and rebuild distances.")
        return self._distance_store.load(sample_id, num_nodes)

    def len(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> GRetrievalData:
        if torch.is_tensor(idx):
            if idx.dim() != 0:
                raise TypeError("GRetrievalDataset only supports integer indexing; batching belongs in DataLoader.")
            idx = int(idx.item())
        if not isinstance(idx, int):
            raise TypeError("GRetrievalDataset only supports integer indexing; batching belongs in DataLoader.")
        data = self.get(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get(self, idx: int) -> GRetrievalData:
        """Load single sample from LMDB with strict PyG validation."""
        sample_id = self.sample_ids[idx]
        raw = self._load_raw_sample(sample_id)
        return self._build_data(raw, sample_id, idx)

    def _build_data(self, raw: Dict[str, Any], sample_id: str, idx: int) -> GRetrievalData:
        if self._validate_on_get:
            _validate_raw_sample(
                raw,
                sample_id,
            )

        edge_index = raw["edge_index"]
        edge_attr = raw["edge_attr"]
        num_nodes = _coerce_num_nodes(raw.get("num_nodes"), sample_id)
        node_global_ids = raw["node_global_ids"]
        node_embedding_ids = raw["node_embedding_ids"]
        node_type_counts = raw["node_type_counts"]
        node_type_ids = raw["node_type_ids"]

        question_emb = raw["question_emb"]

        answer_ids = raw["answer_entity_ids"]

        q_local_indices = raw["q_local_indices"]
        a_local_indices = raw["a_local_indices"]
        node_min_dists = self._get_cached_node_min_dists(sample_id)
        if node_min_dists is None:
            node_min_dists = self._load_distance_cache(sample_id, num_nodes)
            self._store_cached_node_min_dists(sample_id, node_min_dists)
        data_kwargs: Dict[str, Any] = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "edge_attr": edge_attr,  # Global relation IDs
            "node_global_ids": node_global_ids,
            "node_embedding_ids": node_embedding_ids,
            "node_type_counts": node_type_counts,
            "node_type_ids": node_type_ids,
            "question_emb": question_emb,
            "q_local_indices": q_local_indices,
            "a_local_indices": a_local_indices,
            "answer_entity_ids": answer_ids,
            "node_min_dists": node_min_dists,
            "sample_id": sample_id,
            "idx": idx,
        }
        question_text = raw.get("question")
        if question_text is not None:
            data_kwargs["question"] = question_text
        data = GRetrievalData(**data_kwargs)
        return data

    def close(self) -> None:
        if self._sample_stores is not None:
            for store in self._sample_stores:
                store.close()
            self._sample_stores = None
        if self._distance_store is not None:
            self._distance_store.close()
            self._distance_store = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()

    def _get_sample_store(self, shard_id: int) -> EmbeddingStore:
        """Lazy store accessor used inside worker processes."""
        if self._sample_stores is None:
            self._sample_stores = [
                EmbeddingStore(path, readahead=self._lmdb_readahead) for path in self._split_paths
            ]
        return self._sample_stores[shard_id]

    def _select_shard_id(self, sample_id: str) -> int:
        return _assign_lmdb_shard(sample_id, self._num_shards)

    def _group_sample_ids_by_shard(self, sample_ids: Sequence[str]) -> Dict[int, List[tuple[int, str]]]:
        shard_groups: Dict[int, List[tuple[int, str]]] = {}
        for idx, sample_id in enumerate(sample_ids):
            shard_id = self._select_shard_id(sample_id)
            shard_groups.setdefault(shard_id, []).append((idx, sample_id))
        return shard_groups

    def _load_raw_sample(self, sample_id: str) -> Dict[str, Any]:
        shard_id = self._select_shard_id(sample_id)
        return self._get_sample_store(shard_id).load_sample(sample_id)

    def _load_raw_samples(self, sample_ids: Sequence[str]) -> List[Dict[str, Any]]:
        sample_ids = list(sample_ids)
        if not sample_ids:
            return []
        if self._num_shards == 1:
            return self._get_sample_store(0).load_samples(sample_ids)
        shard_groups = self._group_sample_ids_by_shard(sample_ids)
        out: List[Optional[Dict[str, Any]]] = [None] * len(sample_ids)
        for shard_id, entries in shard_groups.items():
            shard_ids = [sid for _, sid in entries]
            shard_raws = self._get_sample_store(shard_id).load_samples(shard_ids)
            for (idx, _), raw in zip(entries, shard_raws):
                out[idx] = raw
        if any(raw is None for raw in out):
            raise RuntimeError("Sharded LMDB load returned missing entries.")
        return [raw for raw in out if raw is not None]

    def _assert_split_path_exists(self) -> None:
        missing_core = [path for path in self._split_paths if not path.exists()]
        if missing_core:
            raise FileNotFoundError(f"Split LMDB not found at {missing_core}")

    def _init_distance_store(self) -> None:
        if self._distance_store_checked:
            return
        self._distance_store_checked = True
        if self._distance_cache_path is None:
            raise ValueError("distance_cache_path must be configured; distances are required.")
        if self._distance_cache_format != DISTANCE_CACHE_PT:
            raise ValueError(f"distance_cache_format must be '{DISTANCE_CACHE_PT}'.")
        if not self._distance_cache_path.exists():
            raise FileNotFoundError(f"Distance PT cache not found at {self._distance_cache_path}")
        try:
            self._distance_store = DistancePTStore(self._distance_cache_path)
            log_event(
                logger,
                "distance_cache_enabled",
                format=self._distance_cache_format,
                path=str(self._distance_cache_path),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open distance PT cache at {self._distance_cache_path}: {exc}"
            ) from exc

    def _init_resources(self, vocabulary_path: Path, embeddings_dir: Path) -> None:
        self._vocabulary_path = Path(vocabulary_path)
        self._embeddings_dir = Path(embeddings_dir)
        self._global_embeddings = None

    def _init_sample_ids(self, sample_ids: Optional[Sequence[str]]) -> None:
        self._sample_stores = None
        if sample_ids is None:
            self.sample_ids = _load_sample_ids_from_lmdb(self._split_paths, readahead=self._lmdb_readahead)
            _assert_unique_sample_ids(self.sample_ids)
            self._sample_ids_source = _SAMPLE_IDS_SOURCE_LMDB
        else:
            self.sample_ids = [str(sid) for sid in sample_ids]
            self._sample_ids_source = _SAMPLE_IDS_SOURCE_PARQUET

    def _assert_sample_ids_match_lmdb(self) -> None:
        if self._sample_ids_source != _SAMPLE_IDS_SOURCE_PARQUET:
            return
        expected = len(self.sample_ids)
        if expected < _MIN_SPLIT_SAMPLE_COUNT:
            raise ValueError(f"Split {self.split} has no samples in questions.parquet.")
        actual = _load_lmdb_entry_total(self._split_paths)
        if actual != expected:
            raise ValueError(
                "Split size mismatch between questions.parquet and LMDB: "
                f"split={self.split} parquet={expected} lmdb={actual}."
            )

    def _apply_filters(
        self,
        *,
        sample_filter_path: Optional[Union[Path, Sequence[Path]]],
        sample_limit: Optional[int],
        random_seed: Optional[int],
        validate_on_init: bool,
        filter_missing_start: bool,
        filter_missing_answer: bool,
    ) -> None:
        if sample_filter_path:
            for path in self._coerce_filter_paths(sample_filter_path):
                self._apply_sample_filter(path)
        if filter_missing_start:
            self._filter_missing_start(batch_size=_FILTER_MISSING_START_BATCH_SIZE)
        if filter_missing_answer:
            self._filter_missing_answer(batch_size=_FILTER_MISSING_ANSWER_BATCH_SIZE)
        if sample_limit:
            self._apply_sample_limit(sample_limit, random_seed)
        if validate_on_init:
            self._assert_all_samples_valid()

    def _maybe_sort_sample_ids(self) -> None:
        if not self._sequential_sample_ids:
            return
        if not self.sample_ids:
            return
        self.sample_ids = sorted(self.sample_ids)

    def _maybe_preload_node_min_dists(self) -> None:
        if not self._preload_node_min_dists:
            return
        if self._node_min_dists_cache is None:
            self._node_min_dists_cache = {}
        total = len(self.sample_ids)
        if total <= _ZERO:
            return
        if self._cache_max_entries < total:
            self._cache_max_entries = total
        log_event(
            logger,
            "distance_cache_preload_start",
            split=self.split,
            samples=total,
        )
        batch_size = _DISTANCE_PRELOAD_BATCH_SIZE
        for start in range(_ZERO, total, batch_size):
            batch_ids = self.sample_ids[start:start + batch_size]
            raws = self._load_raw_samples(batch_ids)
            for raw, sample_id in zip(raws, batch_ids):
                num_nodes = int(raw["num_nodes"])
                node_min_dists = self._load_distance_cache(sample_id, num_nodes)
                self._store_cached_node_min_dists(sample_id, node_min_dists)
        log_event(
            logger,
            "distance_cache_preload_done",
            split=self.split,
            entries=len(self._node_min_dists_cache),
        )

    @property
    def graph_store(self) -> GraphStore:
        if self._graph_store is None:
            if self._shared_resources is not None:
                self._graph_store = self._shared_resources.graph_store
            else:
                self._graph_store = GraphStore(vocabulary_path=str(self._vocabulary_path))
        return self._graph_store

    @property
    def global_embeddings(self) -> GlobalEmbeddingStore:
        if self._shared_resources is not None:
            return self._shared_resources.global_embeddings
        if self._global_embeddings is None:
            self._global_embeddings = GlobalEmbeddingStore(self._embeddings_dir, self._vocabulary_path)
        return self._global_embeddings

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_graph_store"] = None
        state["_global_embeddings"] = None
        state["_sample_stores"] = None
        state["_distance_store"] = None
        state["_distance_store_checked"] = False
        return state

    # ------------------------------------------------------------------ #
    # Filtering Utilities
    # ------------------------------------------------------------------ #
    def _apply_sample_limit(self, limit: int, seed: Optional[int]) -> None:
        if limit <= 0 or len(self.sample_ids) <= limit:
            return

        generator = torch.Generator()
        if seed is not None:
            # Deterministic subset per split (avoid Python's randomized hash()).
            split_hash = zlib.crc32(str(self.split).encode("utf-8")) & 0x7FFFFFFF
            generator.manual_seed(int(seed) + int(split_hash))

        perm = torch.randperm(len(self.sample_ids), generator=generator).tolist()
        self.sample_ids = [self.sample_ids[i] for i in perm[:limit]]
        log_event(
            logger,
            "sample_limit_applied",
            split=self.split,
            limit=limit,
            samples=len(self.sample_ids),
        )

    def _assert_all_samples_valid(self) -> None:
        """
        Eagerly validate every sample in the split and fail fast on malformed graphs.
        This prevents silent sample drops that skew data distribution.
        """
        if not self.sample_ids:
            return
        for sid in self.sample_ids:
            raw = self._load_raw_sample(sid)
            _validate_raw_sample(
                raw,
                sid,
            )

    def _apply_sample_filter(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Filter path {path} not found.")

        keep_ids = self._load_filter_ids(path)
        before = len(self.sample_ids)
        self.sample_ids = [sid for sid in self.sample_ids if sid in keep_ids]
        log_event(
            logger,
            "sample_filter_applied",
            split=self.split,
            before=before,
            after=len(self.sample_ids),
            filter=path.name,
        )

    def _filter_missing_start(self, *, batch_size: int) -> None:
        if not self.sample_ids:
            return
        before = len(self.sample_ids)
        kept: List[str] = []
        dropped = _ZERO
        for start in range(_ZERO, before, batch_size):
            batch_ids = self.sample_ids[start : start + batch_size]
            raws = self._load_raw_samples(batch_ids)
            for sample_id, raw in zip(batch_ids, raws):
                q_local = raw.get("q_local_indices")
                if q_local is None:
                    raise ValueError(f"Sample {sample_id} missing q_local_indices; cannot filter missing starts.")
                if torch.as_tensor(q_local).numel() > _ZERO:
                    kept.append(sample_id)
                else:
                    dropped += 1
        self.sample_ids = kept
        log_event(
            logger,
            "filter_missing_start",
            split=self.split,
            before=before,
            after=len(self.sample_ids),
            dropped=dropped,
        )

    def _filter_missing_answer(self, *, batch_size: int) -> None:
        if not self.sample_ids:
            return
        before = len(self.sample_ids)
        kept: List[str] = []
        dropped = _ZERO
        for start in range(_ZERO, before, batch_size):
            batch_ids = self.sample_ids[start : start + batch_size]
            raws = self._load_raw_samples(batch_ids)
            for sample_id, raw in zip(batch_ids, raws):
                a_local = raw.get("a_local_indices")
                if a_local is None:
                    raise ValueError(f"Sample {sample_id} missing a_local_indices; cannot filter missing answers.")
                if torch.as_tensor(a_local).numel() > _ZERO:
                    kept.append(sample_id)
                else:
                    dropped += 1
        self.sample_ids = kept
        log_event(
            logger,
            "filter_missing_answer",
            split=self.split,
            before=before,
            after=len(self.sample_ids),
            dropped=dropped,
        )

    @staticmethod
    def _coerce_filter_paths(path_or_paths: Union[Path, Sequence[Path]]) -> Sequence[Path]:
        if isinstance(path_or_paths, (list, tuple, set)):
            return [Path(path) for path in path_or_paths]
        return [Path(path_or_paths)]

    @staticmethod
    def _load_filter_ids(path: Path) -> Set[str]:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return set()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Filter file must be JSON list/dict: {path}") from exc
        if isinstance(data, list):
            return set(map(str, data))
        if isinstance(data, dict):
            if "sample_ids" not in data:
                raise ValueError(f"Filter JSON dict missing 'sample_ids': {path}")
            return set(map(str, data["sample_ids"]))
        raise ValueError(f"Filter JSON must be list or dict: {path}")


# ------------------------------------------------------------------ #
# Factory Function (The Adapter)
# ------------------------------------------------------------------ #


def create_g_retrieval_dataset(
    cfg: Dict[str, Any],
    split_name: str,
    *,
    resources: Optional[SharedDataResources] = None,
) -> GRetrievalDataset:
    """Factory adapting dataset_cfg into GRetrievalDataset."""
    _assert_no_removed_dataset_keys(cfg)
    split_name = _assert_allowed_split_name(split_name)
    emb_root = Path(cfg["paths"]["embeddings"])
    split_path = emb_root / f"{split_name}.lmdb"
    sample_limit = _resolve_sample_limit(cfg, split_name)
    filter_missing_start = _resolve_split_bool(cfg, split_name, "filter_missing_start", False)
    filter_missing_answer = _resolve_split_bool(cfg, split_name, "filter_missing_answer", False)
    sample_ids = _load_sample_ids_from_parquet(cfg, split_name)
    cache_node_min_dists = _resolve_split_bool(cfg, split_name, "cache_node_min_dists", False)
    cache_max_entries = _resolve_split_int(cfg, split_name, "cache_max_entries", _CACHE_MAX_ENTRIES_DEFAULT)
    sequential_sample_ids = _resolve_split_bool(
        cfg, split_name, "sequential_sample_ids", _SEQUENTIAL_SAMPLE_IDS_DEFAULT
    )
    distance_cache_format = _resolve_distance_cache_format(cfg)
    distance_cache_path = _resolve_distance_cache_path(cfg, split_name, distance_cache_format)
    lmdb_readahead = _resolve_split_bool(cfg, split_name, "lmdb_readahead", False)
    preload_node_min_dists = _resolve_split_bool(cfg, split_name, "preload_node_min_dists", False)
    if sample_ids is None:
        log_event(
            logger,
            "sample_ids_loaded",
            split=split_name,
            source=_SAMPLE_IDS_SOURCE_LMDB,
        )
    else:
        log_event(
            logger,
            "sample_ids_loaded",
            split=split_name,
            source=_SAMPLE_IDS_SOURCE_PARQUET,
        )

    filter_paths = _resolve_filter_paths(cfg, split_name)

    return GRetrievalDataset(
        split_path=split_path,
        vocabulary_path=Path(cfg["paths"]["vocabulary"]),
        embeddings_dir=emb_root,
        dataset_name=cfg.get("name", "unknown"),
        split_name=split_name,
        resources=resources,
        sample_limit=sample_limit,
        sample_filter_path=filter_paths if filter_paths else None,
        random_seed=cfg.get("random_seed"),
        validate_on_init=bool(cfg.get("validate_on_init", False)),
        validate_on_get=bool(cfg.get("validate_on_get", True)),
        sample_ids=sample_ids,
        cache_node_min_dists=cache_node_min_dists,
        cache_max_entries=cache_max_entries,
        sequential_sample_ids=sequential_sample_ids,
        distance_cache_format=distance_cache_format,
        distance_cache_path=distance_cache_path,
        lmdb_readahead=lmdb_readahead,
        preload_node_min_dists=preload_node_min_dists,
        filter_missing_start=filter_missing_start,
        filter_missing_answer=filter_missing_answer,
    )


def _resolve_sample_limit(cfg: Dict[str, Any], split_name: str) -> Optional[int]:
    sample_limit = cfg.get("sample_limit")
    if not sample_limit:
        return None
    if isinstance(sample_limit, dict):
        return int(sample_limit.get(split_name, 0))
    return int(sample_limit)


def _assert_no_removed_dataset_keys(cfg: Dict[str, Any]) -> None:
    removed = [key for key in _REMOVED_DATASET_KEYS if key in cfg]
    if removed:
        raise ValueError(f"Removed dataset config keys detected: {removed}. Edge dropout is no longer supported.")


def _resolve_split_bool(cfg: Dict[str, Any], split_name: str, key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, dict):
        value = value.get(split_name, default)
    return bool(value)


def _resolve_split_int(cfg: Dict[str, Any], split_name: str, key: str, default: int) -> int:
    value = cfg.get(key, default)
    if isinstance(value, dict):
        value = value.get(split_name, default)
    return int(value)


def _resolve_distance_cache_format(cfg: Dict[str, Any]) -> str:
    if "distance_cache_format" not in cfg:
        raise ValueError("distance_cache_format must be set explicitly; no defaults are allowed.")
    fmt = cfg.get("distance_cache_format")
    fmt = str(fmt).strip().lower()
    if fmt != DISTANCE_CACHE_PT:
        raise ValueError(f"distance_cache_format must be '{DISTANCE_CACHE_PT}', got {fmt!r}.")
    return fmt


def _resolve_distance_cache_path(cfg: Dict[str, Any], split_name: str, cache_format: str) -> Path:
    distances_dir = cfg.get("distances_dir")
    if not distances_dir:
        raise ValueError("distances_dir must be set explicitly; no defaults are allowed.")
    if cache_format != DISTANCE_CACHE_PT:
        raise ValueError(f"distance_cache_format must be '{DISTANCE_CACHE_PT}'.")
    return Path(distances_dir) / f"{split_name}.pt"


def _resolve_filter_paths(cfg: Dict[str, Any], split_name: str) -> list[Path]:
    filter_paths: list[Path] = []
    base_filter = cfg.get("sample_filter_path")
    if base_filter:
        filter_paths.extend(GRetrievalDataset._coerce_filter_paths(base_filter))
    split_filter = cfg.get("sample_filter_paths")
    if isinstance(split_filter, dict):
        split_filter = split_filter.get(split_name)
    if split_filter:
        filter_paths.extend(GRetrievalDataset._coerce_filter_paths(split_filter))
    return filter_paths


def _resolve_sample_ids_source(cfg: Dict[str, Any]) -> str:
    if "sample_ids_source" not in cfg:
        raise ValueError("sample_ids_source must be set explicitly; no defaults are allowed.")
    source = cfg.get("sample_ids_source")
    source = str(source).strip().lower()
    if source not in _ALLOWED_SAMPLE_ID_SOURCES:
        raise ValueError(f"sample_ids_source must be one of {_ALLOWED_SAMPLE_ID_SOURCES}, got {source!r}.")
    return source


def _load_sample_ids_from_parquet(cfg: Dict[str, Any], split_name: str) -> Optional[List[str]]:
    split_name = _assert_allowed_split_name(split_name)
    source = _resolve_sample_ids_source(cfg)
    if source == _SAMPLE_IDS_SOURCE_LMDB:
        return None
    parquet_dir = cfg.get("parquet_dir") or cfg.get("out_dir")
    if not parquet_dir:
        raise ValueError("sample_ids_source=parquet requires parquet_dir or out_dir to be set.")
    parquet_path = Path(parquet_dir) / _QUESTIONS_PARQUET_FILENAME
    if not parquet_path.exists():
        raise FileNotFoundError(f"questions.parquet not found at {parquet_path}")
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required when sample_ids_source=parquet.") from exc
    dataset = ds.dataset(str(parquet_path), format="parquet")
    if "graph_id" not in dataset.schema.names or "split" not in dataset.schema.names:
        raise ValueError("questions.parquet missing graph_id/split; cannot load sample ids.")
    scanner = dataset.scanner(
        columns=["graph_id"],
        filter=ds.field("split") == split_name,
        batch_size=_PARQUET_SCAN_BATCH_SIZE,
    )
    graph_ids: List[str] = []
    for batch in scanner.to_batches():
        graph_ids.extend(batch.column(0).to_pylist())
    if len(graph_ids) < _MIN_SPLIT_SAMPLE_COUNT:
        raise ValueError(f"questions.parquet has no samples for split={split_name}.")
    return [str(gid) for gid in graph_ids]


def _assert_allowed_split_name(split_name: str) -> str:
    split = str(split_name)
    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split name: {split}. Expected one of {_ALLOWED_SPLITS}.")
    return split


def _resolve_lmdb_split_paths(split_path: Path) -> List[Path]:
    if split_path.exists():
        shard_paths = _find_lmdb_shard_paths(split_path)
        if shard_paths:
            raise ValueError(f"Both sharded and unsharded LMDBs found under {split_path.parent}.")
        return [split_path]
    shard_paths = _find_lmdb_shard_paths(split_path)
    if not shard_paths:
        raise FileNotFoundError(f"Split LMDB not found at {split_path}")
    shard_map = _index_lmdb_shard_paths(shard_paths, split_path.stem)
    shard_ids = sorted(shard_map)
    if not shard_ids:
        raise FileNotFoundError(f"No LMDB shards found for {split_path.stem}.")
    expected = list(range(shard_ids[-1] + 1))
    if shard_ids != expected:
        raise ValueError(f"LMDB shards must be contiguous from 0; found {shard_ids}.")
    return [shard_map[shard_id] for shard_id in expected]


def _find_lmdb_shard_paths(split_path: Path) -> List[Path]:
    pattern = _LMDB_SHARD_GLOB_TEMPLATE.format(split=split_path.stem)
    return sorted(split_path.parent.glob(pattern))


def _index_lmdb_shard_paths(shard_paths: Sequence[Path], split_prefix: str) -> Dict[int, Path]:
    shard_map: Dict[int, Path] = {}
    for path in shard_paths:
        shard_id = _parse_lmdb_shard_id(path, split_prefix)
        if shard_id in shard_map:
            raise ValueError(f"Duplicate LMDB shard id {shard_id} for {split_prefix}.")
        shard_map[shard_id] = path
    return shard_map


def _parse_lmdb_shard_id(path: Path, split_prefix: str) -> int:
    stem = path.stem
    token = f"{split_prefix}{_LMDB_SHARD_TOKEN}"
    if not stem.startswith(token):
        raise ValueError(f"Unexpected LMDB shard name: {path.name}")
    shard_text = stem[len(token):]
    if not shard_text.isdigit():
        raise ValueError(f"Invalid shard id in LMDB shard {path.name}")
    return int(shard_text)


def _assign_lmdb_shard(sample_id: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    sample_key = sample_id.encode("utf-8")
    return int(zlib.crc32(sample_key) % num_shards)


def _load_sample_ids_from_lmdb(paths: Sequence[Path], *, readahead: bool = False) -> List[str]:
    sample_ids: List[str] = []
    for path in paths:
        store = EmbeddingStore(path, readahead=readahead)
        try:
            sample_ids.extend(store.get_sample_ids())
        finally:
            store.close()
    return sample_ids


def _assert_unique_sample_ids(sample_ids: Sequence[str]) -> None:
    if not sample_ids:
        return
    seen = set()
    duplicates = set()
    for sid in sample_ids:
        if sid in seen:
            duplicates.add(sid)
        else:
            seen.add(sid)
    if duplicates:
        preview = sorted(duplicates)[:_DUPLICATE_SAMPLE_ID_PREVIEW]
        raise ValueError(f"Duplicate sample ids found across LMDB shards: {preview}")


def _load_lmdb_entry_count(path: Path) -> int:
    store = EmbeddingStore(path)
    try:
        return store.get_entry_count()
    finally:
        store.close()


def _load_lmdb_entry_counts(paths: Sequence[Path]) -> List[int]:
    return [_load_lmdb_entry_count(path) for path in paths]


def _load_lmdb_entry_total(paths: Sequence[Path]) -> int:
    return sum(_load_lmdb_entry_counts(paths))


def _validate_local_indices(local_idx: torch.Tensor, num_nodes: int, field: str, sample_id: str) -> None:
    if local_idx.numel() == 0:
        return
    if local_idx.dim() != 1:
        raise ValueError(f"{field} for {sample_id} must be 1D.")
    if local_idx.min().item() < 0 or local_idx.max().item() >= num_nodes:
        raise ValueError(f"{field} out of range for {sample_id}: num_nodes={num_nodes}, values={local_idx.tolist()}")


def _expect_tensor(
    raw: Dict[str, Any],
    key: str,
    *,
    sample_id: str,
    dtype: Optional[torch.dtype] = None,
    dim: Optional[int] = None,
) -> torch.Tensor:
    if key not in raw:
        raise KeyError(f"Sample {sample_id} missing key: {key}")
    value = raw[key]
    if not torch.is_tensor(value):
        raise TypeError(f"{key} for {sample_id} must be a torch.Tensor, got {type(value)!r}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{key} for {sample_id} must be dtype={dtype}, got {value.dtype}")
    if dim is not None and value.dim() != dim:
        raise ValueError(f"{key} for {sample_id} must be {dim}D, got shape {tuple(value.shape)}")
    return value


def _coerce_num_nodes(value: Any, sample_id: str) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"num_nodes for {sample_id} must be a scalar tensor, got shape={tuple(value.shape)}")
        return int(value.view(-1)[0].item())
    return int(value)


def _validate_answer_ids(answer_ids: torch.Tensor, sample_id: str) -> None:
    if answer_ids.dim() != 1:
        raise ValueError(f"answer_entity_ids for {sample_id} must be 1D.")


def _validate_raw_sample(
    raw: Dict[str, Any],
    sample_id: str,
) -> None:
    """Shared raw-schema validation to guarantee PyG Data integrity."""
    _assert_no_deprecated_fields(raw, sample_id)
    num_nodes, num_edges = _validate_core_sample(raw, sample_id)


def _validate_core_sample(raw: Dict[str, Any], sample_id: str) -> tuple[int, int]:
    required_keys = [
        "edge_index",
        "edge_attr",
        "num_nodes",
        "node_global_ids",
        "node_embedding_ids",
        "node_type_counts",
        "node_type_ids",
        "question_emb",
        "q_local_indices",
        "a_local_indices",
        "answer_entity_ids",
    ]
    _require_keys(raw, sample_id, required_keys)

    num_nodes, num_edges = _validate_edge_index(raw, sample_id)
    _validate_node_ids(raw, sample_id, num_nodes)
    _validate_node_type_fields(raw, sample_id, num_nodes)
    _validate_question_emb(raw, sample_id)
    _validate_edge_attr(raw, sample_id, num_edges)
    _validate_anchor_indices(raw, sample_id, num_nodes)
    _validate_answer_fields(raw, sample_id)
    return num_nodes, num_edges


def _require_keys(raw: Dict[str, Any], sample_id: str, keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise KeyError(f"Sample {sample_id} missing keys: {missing}")


def _assert_no_deprecated_fields(raw: Dict[str, Any], sample_id: str) -> None:
    raw_keys = set(raw.keys())
    deprecated = [key for key in _DEPRECATED_RAW_KEYS if key in raw_keys]
    prefixed = [key for key in raw_keys if key.startswith(_DEPRECATED_RAW_PREFIXES)]
    if deprecated or prefixed:
        blocked = sorted(deprecated + prefixed)
        raise ValueError(f"Sample {sample_id} contains deprecated fields: {blocked}")


def _validate_edge_index(raw: Dict[str, Any], sample_id: str) -> tuple[int, int]:
    edge_index = _expect_tensor(raw, "edge_index", sample_id=sample_id, dtype=torch.long, dim=2)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E] for {sample_id}, got {tuple(edge_index.shape)}")
    num_edges = int(edge_index.size(1))
    if num_edges <= 0:
        raise ValueError(f"edge_index empty for {sample_id}")

    num_nodes = _coerce_num_nodes(raw.get("num_nodes"), sample_id)
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive for {sample_id}, got {num_nodes}")
    if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
        raise ValueError(
            f"edge_index out of range for {sample_id}: "
            f"min={edge_index.min().item()} max={edge_index.max().item()} num_nodes={num_nodes}"
        )
    return num_nodes, num_edges


def _validate_node_ids(raw: Dict[str, Any], sample_id: str, num_nodes: int) -> None:
    node_global_ids = _expect_tensor(raw, "node_global_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    if node_global_ids.numel() != num_nodes:
        raise ValueError(f"node_global_ids length {node_global_ids.numel()} != num_nodes {num_nodes} for {sample_id}")
    if torch.unique(node_global_ids).numel() != num_nodes:
        raise ValueError(f"node_global_ids must be unique per sample: {sample_id}")

    node_embedding_ids = _expect_tensor(raw, "node_embedding_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    if node_embedding_ids.numel() != num_nodes:
        raise ValueError(f"node_embedding_ids length {node_embedding_ids.numel()} != num_nodes {num_nodes} for {sample_id}")


def _validate_node_type_fields(raw: Dict[str, Any], sample_id: str, num_nodes: int) -> None:
    node_type_counts = _expect_tensor(raw, "node_type_counts", sample_id=sample_id, dtype=torch.long, dim=1)
    if node_type_counts.numel() != num_nodes:
        raise ValueError(
            f"node_type_counts length {node_type_counts.numel()} != num_nodes {num_nodes} for {sample_id}"
        )
    if node_type_counts.numel() > 0 and int(node_type_counts.min().item()) < _ZERO:
        raise ValueError(f"node_type_counts contains negatives for {sample_id}")
    node_type_ids = _expect_tensor(raw, "node_type_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    expected_ids = int(node_type_counts.sum().item()) if node_type_counts.numel() > 0 else _ZERO
    if node_type_ids.numel() != expected_ids:
        raise ValueError(
            f"node_type_ids length {node_type_ids.numel()} != expected {expected_ids} for {sample_id}"
        )


def _validate_question_emb(raw: Dict[str, Any], sample_id: str) -> None:
    question_emb = _expect_tensor(raw, "question_emb", sample_id=sample_id, dim=2)
    if not torch.is_floating_point(question_emb):
        raise ValueError(f"question_emb for {sample_id} must be floating point, got dtype={question_emb.dtype}")


def _validate_edge_attr(raw: Dict[str, Any], sample_id: str, num_edges: int) -> None:
    edge_attr = _expect_tensor(raw, "edge_attr", sample_id=sample_id, dtype=torch.long, dim=1)
    if edge_attr.numel() != num_edges:
        raise ValueError(f"edge_attr length {edge_attr.numel()} != num_edges {num_edges} for {sample_id}")


def _validate_anchor_indices(raw: Dict[str, Any], sample_id: str, num_nodes: int) -> None:
    q_local_indices = _expect_tensor(raw, "q_local_indices", sample_id=sample_id, dtype=torch.long, dim=1)
    a_local_indices = _expect_tensor(raw, "a_local_indices", sample_id=sample_id, dtype=torch.long, dim=1)
    _validate_local_indices(q_local_indices, num_nodes, "q_local_indices", sample_id)
    _validate_local_indices(a_local_indices, num_nodes, "a_local_indices", sample_id)


def _validate_answer_fields(raw: Dict[str, Any], sample_id: str) -> None:
    answer_ids = _expect_tensor(raw, "answer_entity_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    answer_ids = answer_ids.view(-1)
    _validate_answer_ids(answer_ids, sample_id)
    if "answer_entity_ids_len" in raw:
        answer_len = _expect_tensor(raw, "answer_entity_ids_len", sample_id=sample_id, dtype=torch.long, dim=1)
        answer_len_tensor = answer_len.view(-1)
        if answer_len_tensor.numel() != 1 or int(answer_len_tensor.item()) != answer_ids.numel():
            raise ValueError(
                f"answer_entity_ids_len mismatch for {sample_id}: declared {answer_len_tensor.tolist()} vs actual {answer_ids.numel()}"
            )
