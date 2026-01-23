import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset

try:  # pragma: no cover - optional dependency guard
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    DictConfig = ()  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

from .components import EmbeddingStore, GlobalEmbeddingStore, SharedDataResources
from .components.shared_resources import _load_entity_embedding_map
from .io.lmdb_utils import _apply_filter_intersection, _assign_lmdb_shard, _resolve_core_lmdb_paths
from src.data.schema.constants import _FILTER_MISSING_ANSWER_FILENAME, _FILTER_MISSING_START_FILENAME
from src.utils.logging_utils import get_logger, log_event

logger = get_logger(__name__)
_ALLOWED_SPLITS = ("train", "validation", "test")
_SEQUENTIAL_SAMPLE_IDS_DEFAULT = False
_SPLIT_HASH_MASK = 0x7FFFFFFF
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
        return super().__inc__(key, value, *args, **kwargs)


class GRetrievalDataset(Dataset):

    def __init__(
        self,
        *,
        entity_vocab_path: Path,
        embeddings_dir: Path,
        split_name: str = "train",
        resources: Optional[SharedDataResources] = None,
        sample_limit: Optional[int] = None,
        sample_filter_path: Optional[Union[Path, Sequence[Path]]] = None,
        random_seed: Optional[int] = None,
        lmdb_readahead: bool = False,
    ):
        super().__init__()
        self.split = str(split_name)
        self._entity_vocab_path = Path(entity_vocab_path)
        self._embeddings_dir = Path(embeddings_dir)
        self._split_paths = _resolve_core_lmdb_paths(self._embeddings_dir, self.split)
        self._num_shards = len(self._split_paths)
        self._shared_resources = resources
        self._global_embeddings: Optional[GlobalEmbeddingStore] = None
        self._sample_stores: Optional[List[EmbeddingStore]] = None
        self._entity_embedding_map: Optional[torch.Tensor] = None
        self._lmdb_readahead = bool(lmdb_readahead)

        self._init_sample_ids()
        self._apply_filters(
            sample_filter_path=sample_filter_path,
            sample_limit=sample_limit,
            random_seed=random_seed,
        )

        log_event(
            logger,
            "g_retrieval_dataset_init",
            split=self.split,
            samples=len(self.sample_ids),
        )

    def len(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> GRetrievalData:
        if torch.is_tensor(idx):
            if idx.dim() != 0:
                raise TypeError("GRetrievalDataset only supports integer indexing; batching belongs in DataLoader.")
            idx = int(idx.detach().tolist())
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
        return self._build_data(raw, sample_id)

    def _build_data(self, raw: Dict[str, Any], sample_id: str) -> GRetrievalData:
        edge_index = raw["edge_index"]
        edge_attr = raw["edge_attr"]
        num_nodes = _coerce_num_nodes(raw.get("num_nodes"), sample_id)
        node_global_ids = raw["node_global_ids"]
        node_embedding_ids = raw["node_embedding_ids"]
        question_emb = raw["question_emb"]

        answer_ids = raw["answer_entity_ids"]

        q_local_indices = raw["q_local_indices"]
        a_local_indices = raw["a_local_indices"]
        retrieval_flag = raw.get("retrieval_failure", False)
        if torch.is_tensor(retrieval_flag):
            retrieval_failure = bool(retrieval_flag.detach().tolist())
        else:
            retrieval_failure = bool(retrieval_flag)
        data_kwargs: Dict[str, Any] = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "edge_attr": edge_attr,  # Global relation IDs
            "node_global_ids": node_global_ids,
            "node_embedding_ids": node_embedding_ids,
            "question_emb": question_emb,
            "q_local_indices": q_local_indices,
            "a_local_indices": a_local_indices,
            "answer_entity_ids": answer_ids,
            "sample_id": sample_id,
            "retrieval_failure": retrieval_failure,
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

    def _init_sample_ids(self) -> None:
        self._sample_stores = None
        self.sample_ids = _load_sample_ids_from_lmdb(self._split_paths, readahead=self._lmdb_readahead)

    def _apply_filters(
        self,
        *,
        sample_filter_path: Optional[Union[Path, Sequence[Path]]],
        sample_limit: Optional[int],
        random_seed: Optional[int],
    ) -> None:
        if sample_filter_path:
            filter_paths = self._coerce_filter_paths(sample_filter_path)
            before = len(self.sample_ids)
            self.sample_ids = _apply_filter_intersection(self.sample_ids, filter_paths)
            log_event(
                logger,
                "sample_filter_applied",
                split=self.split,
                before=before,
                after=len(self.sample_ids),
                filters=[path.name for path in filter_paths],
            )
        if sample_limit:
            self._apply_sample_limit(sample_limit, random_seed)

    @property
    def global_embeddings(self) -> GlobalEmbeddingStore:
        if self._shared_resources is not None:
            return self._shared_resources.global_embeddings
        if self._global_embeddings is None:
            self._global_embeddings = GlobalEmbeddingStore(
                self._embeddings_dir,
                entity_vocab_path=self._entity_vocab_path,
            )
        return self._global_embeddings

    @property
    def entity_embedding_map(self) -> torch.Tensor:
        if self._shared_resources is not None:
            return self._shared_resources.entity_embedding_map
        if self._entity_embedding_map is None:
            self._entity_embedding_map = _load_entity_embedding_map(self._entity_vocab_path)
        return self._entity_embedding_map

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_global_embeddings"] = None
        state["_sample_stores"] = None
        state["_entity_embedding_map"] = None
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
            split_hash = zlib.crc32(str(self.split).encode("utf-8")) & _SPLIT_HASH_MASK
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

    @staticmethod
    def _coerce_filter_paths(path_or_paths: Union[Path, Sequence[Path]]) -> Sequence[Path]:
        if isinstance(path_or_paths, (list, tuple, set)):
            return [Path(path) for path in path_or_paths]
        return [Path(path_or_paths)]


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
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if not isinstance(cfg, dict):
        raise TypeError(f"dataset_cfg must be a mapping, got {type(cfg)!r}")
    _assert_no_removed_dataset_keys(cfg)
    split_name = _assert_allowed_split_name(split_name)
    emb_root = Path(cfg["paths"]["embeddings"])
    sample_limit = _resolve_sample_limit(cfg, split_name)
    filter_missing_start = _resolve_split_bool(cfg, split_name, "filter_missing_start", True)
    filter_missing_answer = _resolve_split_bool(cfg, split_name, "filter_missing_answer", True)
    sequential_sample_ids = _resolve_split_bool(
        cfg, split_name, "sequential_sample_ids", _SEQUENTIAL_SAMPLE_IDS_DEFAULT
    )
    lmdb_readahead = _resolve_split_bool(cfg, split_name, "lmdb_readahead", False)

    filter_paths = _resolve_filter_paths(cfg, split_name)
    missing_filters = _resolve_missing_filter_paths(
        cfg,
        split_name=split_name,
        filter_missing_start=filter_missing_start,
        filter_missing_answer=filter_missing_answer,
    )
    if missing_filters:
        filter_paths.extend(missing_filters)

    dataset = GRetrievalDataset(
        entity_vocab_path=Path(cfg["paths"]["entity_vocab"]),
        embeddings_dir=emb_root,
        split_name=split_name,
        resources=resources,
        sample_limit=sample_limit,
        sample_filter_path=filter_paths if filter_paths else None,
        random_seed=cfg.get("random_seed"),
        lmdb_readahead=lmdb_readahead,
    )
    if sequential_sample_ids:
        dataset.sample_ids = sorted(dataset.sample_ids)
    return dataset


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


def _resolve_missing_filter_paths(
    cfg: Dict[str, Any],
    *,
    split_name: str,
    filter_missing_start: bool,
    filter_missing_answer: bool,
) -> list[Path]:
    if not (filter_missing_start or filter_missing_answer):
        return []
    paths = cfg.get("paths")
    if not isinstance(paths, dict) or not paths.get("processed"):
        raise ValueError("dataset_cfg.paths.processed is required when filter_missing_* is enabled.")
    processed_dir = Path(paths["processed"])
    filter_paths: list[Path] = []
    if filter_missing_start:
        filter_paths.append(processed_dir / _FILTER_MISSING_START_FILENAME)
    if filter_missing_answer:
        filter_paths.append(processed_dir / _FILTER_MISSING_ANSWER_FILENAME)
    missing = [str(path) for path in filter_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing filter files for split={split_name}: {missing}. "
            "Rebuild LMDB to emit filter_missing_* artifacts or disable filter_missing_*."
        )
    return filter_paths


def _assert_allowed_split_name(split_name: str) -> str:
    split = str(split_name)
    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split name: {split}. Expected one of {_ALLOWED_SPLITS}.")
    return split


def _load_sample_ids_from_lmdb(paths: Sequence[Path], *, readahead: bool = False) -> List[str]:
    sample_ids: List[str] = []
    for path in paths:
        store = EmbeddingStore(path, readahead=readahead)
        try:
            sample_ids.extend(store.get_sample_ids())
        finally:
            store.close()
    return sample_ids


def _coerce_num_nodes(value: Any, sample_id: str) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        return int(value.view(-1)[0].detach().tolist())
    return int(value)
