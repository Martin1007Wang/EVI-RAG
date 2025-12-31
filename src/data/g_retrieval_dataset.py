import json
import logging
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset

from .components import (
    EmbeddingStore,
    GlobalEmbeddingStore,
    GraphStore,
    SharedDataResources,
)

logger = logging.getLogger(__name__)
_QUESTIONS_PARQUET_FILENAME = "questions.parquet"
_PARQUET_SCAN_BATCH_SIZE = 65536
_MIN_SPLIT_SAMPLE_COUNT = 1
_ALLOWED_SPLITS = ("train", "validation", "test")
_SAMPLE_IDS_SOURCE_LMDB = "lmdb"
_SAMPLE_IDS_SOURCE_PARQUET = "parquet"
_AUX_LMDB_SUFFIX_DEFAULT = ".aux.lmdb"
_AUX_MERGE_ALLOWLIST = {"sample_id", "idx"}


class GRetrievalData(GraphData):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> int:
        if key in {"q_local_indices", "a_local_indices", "pair_start_node_locals", "pair_answer_node_locals"}:
            return int(self.num_nodes)
        if key in {"pair_edge_local_ids"}:
            return int(self.edge_index.size(1))
        if key in {"pair_edge_counts", "pair_shortest_lengths"}:
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
        drop_zero_positive: bool = False,
        random_seed: Optional[int] = None,
        validate_on_init: bool = False,
        validate_on_get: bool = True,
        sample_ids: Optional[Sequence[str]] = None,
        include_aux_fields: bool = False,
        aux_lmdb_suffix: str = _AUX_LMDB_SUFFIX_DEFAULT,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = _assert_allowed_split_name(split_name)
        self.split_path = Path(split_path)
        self._include_aux_fields = bool(include_aux_fields)
        aux_suffix = _AUX_LMDB_SUFFIX_DEFAULT if aux_lmdb_suffix is None else aux_lmdb_suffix
        self._aux_lmdb_suffix = str(aux_suffix)
        self._aux_split_path = _resolve_aux_split_path(self.split_path, self._aux_lmdb_suffix)
        self._vocabulary_path = Path(vocabulary_path)
        self._embeddings_dir = Path(embeddings_dir)
        self._shared_resources = resources
        self._graph_store: Optional[GraphStore] = None
        self._global_embeddings: Optional[GlobalEmbeddingStore] = None
        self._aux_store: Optional[EmbeddingStore] = None
        self._sample_ids_source = _SAMPLE_IDS_SOURCE_LMDB
        self._validate_on_get = bool(validate_on_get)

        self._assert_split_path_exists()
        self._init_resources(vocabulary_path, embeddings_dir)
        self._init_sample_ids(sample_ids)
        self._assert_sample_ids_match_lmdb()
        self._apply_filters(
            sample_filter_path=sample_filter_path,
            drop_zero_positive=drop_zero_positive,
            sample_limit=sample_limit,
            random_seed=random_seed,
            validate_on_init=validate_on_init,
        )

        logger.info(f"GRetrievalDataset[{self.split}] initialized: {len(self.sample_ids)} samples.")

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> GRetrievalData:
        """Load single sample from LMDB with strict PyG validation."""
        sample_id = self.sample_ids[idx]
        raw = self._load_raw_sample(sample_id)
        return self._build_data(raw, sample_id, idx)

    def __getitems__(self, indices) -> list[GRetrievalData]:
        if isinstance(indices, int):
            return [self.get(indices)]
        if isinstance(indices, slice):
            indices = list(range(*indices.indices(len(self.sample_ids))))
        indices = list(indices)
        if not indices:
            return []
        sample_ids = [self.sample_ids[idx] for idx in indices]
        raws = self._load_raw_samples(sample_ids)
        return [self._build_data(raw, sample_id, idx) for raw, sample_id, idx in zip(raws, sample_ids, indices)]

    def _build_data(self, raw: Dict[str, Any], sample_id: str, idx: int) -> GRetrievalData:
        if self._validate_on_get:
            _validate_raw_sample(raw, sample_id, require_aux_fields=self._include_aux_fields)

        edge_index = raw["edge_index"]
        num_nodes = int(raw["num_nodes"])
        node_global_ids = raw["node_global_ids"]
        node_embedding_ids = raw["node_embedding_ids"]

        question_emb = raw["question_emb"]

        answer_ids = raw["answer_entity_ids"]
        answer_ids_len = raw["answer_entity_ids_len"]

        q_local_indices = raw["q_local_indices"]
        a_local_indices = raw["a_local_indices"]
        topic_one_hot = raw["topic_one_hot"]

        data_kwargs: Dict[str, Any] = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "edge_attr": raw["edge_attr"],  # Global relation IDs
            "labels": raw["labels"],
            "node_global_ids": node_global_ids,
            "node_embedding_ids": node_embedding_ids,
            "question_emb": question_emb,
            "topic_one_hot": topic_one_hot,
            "q_local_indices": q_local_indices,
            "a_local_indices": a_local_indices,
            "answer_entity_ids": answer_ids,
            "answer_entity_ids_len": answer_ids_len,
            "sample_id": sample_id,
            "idx": idx,
        }
        question_text = raw.get("question")
        if question_text is not None:
            data_kwargs["question"] = question_text
        pair_fields = _extract_pair_fields(raw, sample_id)
        if pair_fields:
            data_kwargs.update(pair_fields)
        data = GRetrievalData(**data_kwargs)
        return data

    def close(self) -> None:
        if getattr(self, "sample_store", None) is not None:
            self.sample_store.close()
            self.sample_store = None
        if self._aux_store is not None:
            self._aux_store.close()
            self._aux_store = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()

    def _get_sample_store(self) -> EmbeddingStore:
        """Lazy store accessor used inside worker processes."""
        if self.sample_store is None:
            self.sample_store = EmbeddingStore(self.split_path)
        return self.sample_store

    def _get_aux_store(self) -> EmbeddingStore:
        if not self._include_aux_fields:
            raise RuntimeError("Aux LMDB disabled; set include_aux_fields=True to enable.")
        if self._aux_store is None:
            self._aux_store = EmbeddingStore(self._aux_split_path)
        return self._aux_store

    def _load_raw_sample(self, sample_id: str) -> Dict[str, Any]:
        raw = self._get_sample_store().load_sample(sample_id)
        if not self._include_aux_fields:
            return raw
        aux_raw = self._get_aux_store().load_sample(sample_id)
        return _merge_aux_samples(raw, aux_raw, sample_id)

    def _load_raw_samples(self, sample_ids: Sequence[str]) -> List[Dict[str, Any]]:
        sample_ids = list(sample_ids)
        raws = self._get_sample_store().load_samples(sample_ids)
        if not self._include_aux_fields:
            return raws
        aux_raws = self._get_aux_store().load_samples(sample_ids)
        return [_merge_aux_samples(raw, aux_raw, sid) for raw, aux_raw, sid in zip(raws, aux_raws, sample_ids)]

    def _assert_split_path_exists(self) -> None:
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split LMDB not found at {self.split_path}")
        if self._include_aux_fields and not self._aux_split_path.exists():
            raise FileNotFoundError(f"Aux LMDB not found at {self._aux_split_path}")

    def _init_resources(self, vocabulary_path: Path, embeddings_dir: Path) -> None:
        self._vocabulary_path = Path(vocabulary_path)
        self._embeddings_dir = Path(embeddings_dir)
        self._global_embeddings = None

    def _init_sample_ids(self, sample_ids: Optional[Sequence[str]]) -> None:
        self.sample_store: Optional[EmbeddingStore] = None
        if sample_ids is None:
            temp_store = EmbeddingStore(self.split_path)
            try:
                self.sample_ids = temp_store.get_sample_ids()
            finally:
                temp_store.close()
                del temp_store
            self._sample_ids_source = _SAMPLE_IDS_SOURCE_LMDB
        else:
            self.sample_ids = [str(sid) for sid in sample_ids]
            self._sample_ids_source = _SAMPLE_IDS_SOURCE_PARQUET

    def _assert_sample_ids_match_lmdb(self) -> None:
        if self._sample_ids_source != _SAMPLE_IDS_SOURCE_PARQUET:
            if self._include_aux_fields:
                core_count = _load_lmdb_entry_count(self.split_path)
                aux_count = _load_lmdb_entry_count(self._aux_split_path)
                if core_count != aux_count:
                    raise ValueError(
                        f"Split size mismatch between core and aux LMDB: split={self.split} core={core_count} aux={aux_count}."
                    )
            return
        expected = len(self.sample_ids)
        if expected < _MIN_SPLIT_SAMPLE_COUNT:
            raise ValueError(f"Split {self.split} has no samples in questions.parquet.")
        actual = _load_lmdb_entry_count(self.split_path)
        if actual != expected:
            raise ValueError(
                "Split size mismatch between questions.parquet and LMDB: "
                f"split={self.split} parquet={expected} lmdb={actual}."
            )
        if self._include_aux_fields:
            aux_count = _load_lmdb_entry_count(self._aux_split_path)
            if aux_count != expected:
                raise ValueError(
                    "Split size mismatch between questions.parquet and aux LMDB: "
                    f"split={self.split} parquet={expected} aux={aux_count}."
                )

    def _apply_filters(
        self,
        *,
        sample_filter_path: Optional[Union[Path, Sequence[Path]]],
        drop_zero_positive: bool,
        sample_limit: Optional[int],
        random_seed: Optional[int],
        validate_on_init: bool,
    ) -> None:
        if sample_filter_path:
            for path in self._coerce_filter_paths(sample_filter_path):
                self._apply_sample_filter(path)
        if drop_zero_positive:
            self._apply_zero_positive_filter()
        if sample_limit:
            self._apply_sample_limit(sample_limit, random_seed)
        if validate_on_init:
            self._assert_all_samples_valid()

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
        state["_aux_store"] = None
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
        logger.info(f"Subsampled {self.split} to {len(self.sample_ids)} samples.")

    def _assert_all_samples_valid(self) -> None:
        """
        Eagerly validate every sample in the split and fail fast on malformed graphs.
        This prevents silent sample drops that skew data distribution.
        """
        if not self.sample_ids:
            return

        temp_store = EmbeddingStore(self.split_path)
        aux_store = EmbeddingStore(self._aux_split_path) if self._include_aux_fields else None
        try:
            for sid in self.sample_ids:
                raw = temp_store.load_sample(sid)
                if aux_store is not None:
                    aux_raw = aux_store.load_sample(sid)
                    raw = _merge_aux_samples(raw, aux_raw, sid)
                _validate_raw_sample(raw, sid, require_aux_fields=self._include_aux_fields)
        finally:
            temp_store.close()
            if aux_store is not None:
                aux_store.close()

    def _apply_sample_filter(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Filter path {path} not found.")

        keep_ids = self._load_filter_ids(path)
        before = len(self.sample_ids)
        self.sample_ids = [sid for sid in self.sample_ids if sid in keep_ids]
        logger.info(f"Filtered {self.split}: {before} -> {len(self.sample_ids)} using {path.name}")

    def _apply_zero_positive_filter(self) -> None:
        if not self.sample_ids:
            return
        before = len(self.sample_ids)
        keep_ids = []
        temp_store = EmbeddingStore(self.split_path)
        try:
            for sid in self.sample_ids:
                raw = temp_store.load_sample(sid)
                labels = raw.get("labels")
                if not torch.is_tensor(labels):
                    raise TypeError(f"labels for {sid} must be a torch.Tensor, got {type(labels)!r}")
                if bool((labels > 0).any().item()):
                    keep_ids.append(sid)
        finally:
            temp_store.close()
        self.sample_ids = keep_ids
        logger.info(f"Filtered {self.split}: {before} -> {len(self.sample_ids)} using nonzero positive labels.")

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
    split_name = _assert_allowed_split_name(split_name)
    emb_root = Path(cfg["paths"]["embeddings"])
    split_path = emb_root / f"{split_name}.lmdb"
    sample_limit = _resolve_sample_limit(cfg, split_name)
    sample_ids = _load_sample_ids_from_parquet(cfg, split_name)
    include_aux_fields = _resolve_include_aux_fields(cfg, split_name)
    aux_lmdb_suffix = cfg.get("aux_lmdb_suffix") or _AUX_LMDB_SUFFIX_DEFAULT
    if sample_ids is None:
        logger.info("Loaded retrieval LMDB; structural encoding is controlled by the retriever config.")
    else:
        logger.info("Loaded sample ids from normalized parquet for split=%s.", split_name)

    filter_paths, drop_zero_positive = _resolve_filter_paths(cfg, split_name)

    return GRetrievalDataset(
        split_path=split_path,
        vocabulary_path=Path(cfg["paths"]["vocabulary"]),
        embeddings_dir=emb_root,
        dataset_name=cfg.get("name", "unknown"),
        split_name=split_name,
        resources=resources,
        sample_limit=sample_limit,
        sample_filter_path=filter_paths if filter_paths else None,
        drop_zero_positive=drop_zero_positive,
        random_seed=cfg.get("random_seed"),
        validate_on_init=bool(cfg.get("validate_on_init", False)),
        validate_on_get=bool(cfg.get("validate_on_get", True)),
        sample_ids=sample_ids,
        include_aux_fields=include_aux_fields,
        aux_lmdb_suffix=aux_lmdb_suffix,
    )


def _resolve_sample_limit(cfg: Dict[str, Any], split_name: str) -> Optional[int]:
    sample_limit = cfg.get("sample_limit")
    if not sample_limit:
        return None
    if isinstance(sample_limit, dict):
        return int(sample_limit.get(split_name, 0))
    return int(sample_limit)


def _resolve_include_aux_fields(cfg: Dict[str, Any], split_name: str) -> bool:
    include_aux = cfg.get("include_aux_fields")
    if isinstance(include_aux, dict):
        return bool(include_aux.get(split_name, False))
    if include_aux is None:
        return False
    return bool(include_aux)


def _resolve_filter_paths(cfg: Dict[str, Any], split_name: str) -> tuple[list[Path], bool]:
    drop_zero_positive = bool(cfg.get("drop_zero_positive", False)) and split_name == "train"
    filter_paths: list[Path] = []
    base_filter = cfg.get("sample_filter_path")
    if base_filter:
        filter_paths.extend(GRetrievalDataset._coerce_filter_paths(base_filter))
    nonzero_filter = cfg.get("nonzero_positive_filter_path")
    if drop_zero_positive and nonzero_filter:
        filter_paths.extend(GRetrievalDataset._coerce_filter_paths(nonzero_filter))
        drop_zero_positive = False
    return filter_paths, drop_zero_positive


def _load_sample_ids_from_parquet(cfg: Dict[str, Any], split_name: str) -> Optional[List[str]]:
    split_name = _assert_allowed_split_name(split_name)
    parquet_dir = cfg.get("parquet_dir") or cfg.get("out_dir")
    if not parquet_dir:
        return None
    parquet_path = Path(parquet_dir) / _QUESTIONS_PARQUET_FILENAME
    if not parquet_path.exists():
        return None
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError:
        logger.warning("pyarrow is not available; falling back to LMDB key scan for sample ids.")
        return None
    dataset = ds.dataset(str(parquet_path), format="parquet")
    if "graph_id" not in dataset.schema.names or "split" not in dataset.schema.names:
        logger.warning("questions.parquet missing graph_id/split; falling back to LMDB key scan.")
        return None
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


def _load_lmdb_entry_count(path: Path) -> int:
    store = EmbeddingStore(path)
    try:
        return store.get_entry_count()
    finally:
        store.close()


def _resolve_aux_split_path(split_path: Path, aux_lmdb_suffix: str) -> Path:
    suffix = str(aux_lmdb_suffix)
    if not suffix:
        raise ValueError("aux_lmdb_suffix must be non-empty.")
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return split_path.with_name(f"{split_path.stem}{suffix}")


def _merge_aux_samples(core: Dict[str, Any], aux: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    if not aux:
        return core
    merged = dict(core)
    for key, value in aux.items():
        if key not in merged:
            merged[key] = value
            continue
        if key in _AUX_MERGE_ALLOWLIST and _values_equal(merged[key], value):
            continue
        raise ValueError(f"Aux LMDB key overlap for {sample_id}: {key}")
    return merged


def _values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    try:
        return left == right
    except Exception:
        return False


def _extract_pair_fields(raw: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    pair_keys = (
        "pair_start_node_locals",
        "pair_answer_node_locals",
        "pair_edge_local_ids",
        "pair_edge_counts",
    )
    present = [key in raw for key in pair_keys]
    if not any(present):
        return {}
    if not all(present):
        missing = [key for key, has_key in zip(pair_keys, present) if not has_key]
        raise KeyError(f"Sample {sample_id} missing keys: {missing}")
    data = {key: raw[key] for key in pair_keys}
    pair_shortest_lengths = raw.get("pair_shortest_lengths")
    if pair_shortest_lengths is not None:
        data["pair_shortest_lengths"] = pair_shortest_lengths
    return data


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


def _validate_topic_one_hot(topic_one_hot: torch.Tensor, num_nodes: int, sample_id: str) -> None:
    if topic_one_hot.dim() == 1:
        topic_one_hot = topic_one_hot.unsqueeze(-1)
    if topic_one_hot.dim() != 2:
        raise ValueError(f"topic_one_hot must be 2D, got shape {tuple(topic_one_hot.shape)} for {sample_id}")
    if topic_one_hot.size(0) != num_nodes:
        raise ValueError(f"topic_one_hot first dim {topic_one_hot.size(0)} != num_nodes {num_nodes} for {sample_id}")


def _validate_answer_ids(answer_ids: torch.Tensor, sample_id: str) -> None:
    if answer_ids.dim() != 1:
        raise ValueError(f"answer_entity_ids for {sample_id} must be 1D.")


def _validate_raw_sample(raw: Dict[str, Any], sample_id: str, *, require_aux_fields: bool = True) -> None:
    """Shared raw-schema validation to guarantee PyG Data integrity."""
    num_nodes, num_edges = _validate_core_sample(raw, sample_id)
    _validate_aux_fields(raw, sample_id, num_nodes=num_nodes, num_edges=num_edges, require=require_aux_fields)


def _validate_core_sample(raw: Dict[str, Any], sample_id: str) -> tuple[int, int]:
    required_keys = [
        "edge_index",
        "edge_attr",
        "labels",
        "num_nodes",
        "node_global_ids",
        "node_embedding_ids",
        "question_emb",
        "topic_one_hot",
        "q_local_indices",
        "a_local_indices",
        "answer_entity_ids",
        "answer_entity_ids_len",
    ]
    _require_keys(raw, sample_id, required_keys)

    num_nodes, num_edges = _validate_edge_index(raw, sample_id)
    _validate_node_ids(raw, sample_id, num_nodes)
    _validate_question_emb(raw, sample_id)
    _validate_edge_attr_and_labels(raw, sample_id, num_edges)
    _validate_topic_and_anchors(raw, sample_id, num_nodes)
    _validate_answer_fields(raw, sample_id)
    _validate_deprecated_fields(raw, sample_id)
    return num_nodes, num_edges


def _require_keys(raw: Dict[str, Any], sample_id: str, keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise KeyError(f"Sample {sample_id} missing keys: {missing}")


def _validate_edge_index(raw: Dict[str, Any], sample_id: str) -> tuple[int, int]:
    edge_index = _expect_tensor(raw, "edge_index", sample_id=sample_id, dtype=torch.long, dim=2)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E] for {sample_id}, got {tuple(edge_index.shape)}")
    num_edges = int(edge_index.size(1))
    if num_edges <= 0:
        raise ValueError(f"edge_index empty for {sample_id}")

    num_nodes = int(raw.get("num_nodes", 0) or 0)
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


def _validate_question_emb(raw: Dict[str, Any], sample_id: str) -> None:
    question_emb = _expect_tensor(raw, "question_emb", sample_id=sample_id, dim=2)
    if not torch.is_floating_point(question_emb):
        raise ValueError(f"question_emb for {sample_id} must be floating point, got dtype={question_emb.dtype}")


def _validate_edge_attr_and_labels(raw: Dict[str, Any], sample_id: str, num_edges: int) -> None:
    edge_attr = _expect_tensor(raw, "edge_attr", sample_id=sample_id, dtype=torch.long, dim=1)
    labels = _expect_tensor(raw, "labels", sample_id=sample_id, dim=1)
    if not torch.is_floating_point(labels):
        raise ValueError(f"labels for {sample_id} must be floating point, got dtype={labels.dtype}")
    if edge_attr.numel() != num_edges:
        raise ValueError(f"edge_attr length {edge_attr.numel()} != num_edges {num_edges} for {sample_id}")
    if labels.numel() != num_edges:
        raise ValueError(f"labels length {labels.numel()} != num_edges {num_edges} for {sample_id}")
    if labels.numel() > 0:
        if labels.min().item() < 0.0 or labels.max().item() > 1.0:
            raise ValueError(
                f"labels out of range for {sample_id}: min={float(labels.min().item())} max={float(labels.max().item())}"
            )


def _validate_topic_and_anchors(raw: Dict[str, Any], sample_id: str, num_nodes: int) -> None:
    topic_one_hot = _expect_tensor(raw, "topic_one_hot", sample_id=sample_id)
    if not torch.is_floating_point(topic_one_hot):
        raise ValueError(f"topic_one_hot for {sample_id} must be floating point, got dtype={topic_one_hot.dtype}")
    _validate_topic_one_hot(topic_one_hot, num_nodes, sample_id)

    q_local_indices = _expect_tensor(raw, "q_local_indices", sample_id=sample_id, dtype=torch.long, dim=1)
    a_local_indices = _expect_tensor(raw, "a_local_indices", sample_id=sample_id, dtype=torch.long, dim=1)
    _validate_local_indices(q_local_indices, num_nodes, "q_local_indices", sample_id)
    _validate_local_indices(a_local_indices, num_nodes, "a_local_indices", sample_id)


def _validate_answer_fields(raw: Dict[str, Any], sample_id: str) -> None:
    answer_ids = _expect_tensor(raw, "answer_entity_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    answer_ids = answer_ids.view(-1)
    _validate_answer_ids(answer_ids, sample_id)
    answer_len = _expect_tensor(raw, "answer_entity_ids_len", sample_id=sample_id, dtype=torch.long, dim=1)
    answer_len_tensor = answer_len.view(-1)
    if answer_len_tensor.numel() != 1 or int(answer_len_tensor.item()) != answer_ids.numel():
        raise ValueError(
            f"answer_entity_ids_len mismatch for {sample_id}: declared {answer_len_tensor.tolist()} vs actual {answer_ids.numel()}"
        )


def _validate_aux_fields(
    raw: Dict[str, Any],
    sample_id: str,
    *,
    num_nodes: int,
    num_edges: int,
    require: bool,
) -> None:
    _validate_question_field(raw, sample_id, require=require)

    pair_keys = [
        "pair_start_node_locals",
        "pair_answer_node_locals",
        "pair_edge_local_ids",
        "pair_edge_counts",
    ]
    present = [key in raw for key in pair_keys]
    if not any(present):
        if require:
            raise KeyError(f"Sample {sample_id} missing keys: {pair_keys}")
        return
    if not all(present):
        missing = [key for key, has_key in zip(pair_keys, present) if not has_key]
        raise KeyError(f"Sample {sample_id} missing keys: {missing}")

    _validate_pair_fields(raw, sample_id, num_nodes=num_nodes, num_edges=num_edges)


def _validate_question_field(raw: Dict[str, Any], sample_id: str, *, require: bool) -> None:
    if "question" not in raw:
        if require:
            raise KeyError(f"Sample {sample_id} missing key: question")
        return
    if not isinstance(raw.get("question"), str):
        raise TypeError(f"question for {sample_id} must be a string.")


def _validate_pair_fields(raw: Dict[str, Any], sample_id: str, *, num_nodes: int, num_edges: int) -> None:
    pair_start, pair_answer = _validate_pair_nodes(raw, sample_id, num_nodes=num_nodes)
    _validate_pair_edges(raw, sample_id, pair_start=pair_start, num_edges=num_edges)
    _validate_pair_shortest_lengths(raw, sample_id, pair_start=pair_start)


def _validate_pair_nodes(raw: Dict[str, Any], sample_id: str, *, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    pair_start = _expect_tensor(raw, "pair_start_node_locals", sample_id=sample_id, dtype=torch.long, dim=1)
    pair_answer = _expect_tensor(raw, "pair_answer_node_locals", sample_id=sample_id, dtype=torch.long, dim=1)
    if pair_start.numel() != pair_answer.numel():
        raise ValueError(
            f"pair_start_node_locals length {pair_start.numel()} != "
            f"pair_answer_node_locals length {pair_answer.numel()} for {sample_id}"
        )
    _validate_local_indices(pair_start, num_nodes, "pair_start_node_locals", sample_id)
    _validate_local_indices(pair_answer, num_nodes, "pair_answer_node_locals", sample_id)
    return pair_start, pair_answer


def _validate_pair_edges(
    raw: Dict[str, Any],
    sample_id: str,
    *,
    pair_start: torch.Tensor,
    num_edges: int,
) -> torch.Tensor:
    pair_edge_local_ids = _expect_tensor(raw, "pair_edge_local_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    pair_edge_local_ids = pair_edge_local_ids.view(-1)
    if pair_edge_local_ids.numel() > 0:
        if pair_edge_local_ids.min().item() < 0 or pair_edge_local_ids.max().item() >= num_edges:
            raise ValueError(
                f"pair_edge_local_ids out of range for {sample_id}: "
                f"min={int(pair_edge_local_ids.min().item())} max={int(pair_edge_local_ids.max().item())} num_edges={num_edges}"
            )

    pair_edge_counts = _expect_tensor(raw, "pair_edge_counts", sample_id=sample_id, dtype=torch.long, dim=1)
    pair_edge_counts = pair_edge_counts.view(-1)
    if pair_edge_counts.numel() != pair_start.numel():
        raise ValueError(
            f"pair_edge_counts length {pair_edge_counts.numel()} != pair_count ({pair_start.numel()}) for {sample_id}"
        )
    if (pair_edge_counts < 0).any():
        raise ValueError(f"pair_edge_counts contains negative values for {sample_id}")
    total_edges = int(pair_edge_counts.sum().item())
    if total_edges != pair_edge_local_ids.numel():
        raise ValueError(
            f"pair_edge_counts sum {total_edges} != "
            f"pair_edge_local_ids length {pair_edge_local_ids.numel()} for {sample_id}"
        )
    return pair_edge_local_ids


def _validate_pair_shortest_lengths(raw: Dict[str, Any], sample_id: str, *, pair_start: torch.Tensor) -> None:
    pair_shortest_lengths = raw.get("pair_shortest_lengths")
    if pair_shortest_lengths is None:
        return
    if not torch.is_tensor(pair_shortest_lengths):
        raise TypeError(f"pair_shortest_lengths for {sample_id} must be a torch.Tensor.")
    if pair_shortest_lengths.dtype != torch.long:
        raise ValueError(
            f"pair_shortest_lengths for {sample_id} must be dtype=torch.long, got {pair_shortest_lengths.dtype}"
        )
    pair_shortest_lengths = pair_shortest_lengths.view(-1)
    if pair_shortest_lengths.numel() != pair_start.numel():
        raise ValueError(
            f"pair_shortest_lengths length {pair_shortest_lengths.numel()} != pair_count ({pair_start.numel()}) for {sample_id}"
        )
    if (pair_shortest_lengths < 0).any():
        raise ValueError(f"pair_shortest_lengths contains negative values for {sample_id}")


def _validate_deprecated_fields(raw: Dict[str, Any], sample_id: str) -> None:
    if "gt_path_edge_indices" in raw or "gt_path_node_indices" in raw:
        raise ValueError(f"gt_path_* fields are deprecated for {sample_id}; rebuild dataset without them.")
    if "pair_edge_offsets" in raw:
        raise ValueError(f"pair_edge_offsets is deprecated for {sample_id}; rebuild dataset with pair_edge_counts only.")
    if "pair_edge_indices" in raw:
        raise ValueError(f"pair_edge_indices is deprecated for {sample_id}; use pair_edge_local_ids.")
