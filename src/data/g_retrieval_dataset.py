import json
import logging
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Set

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
        sample_filter_path: Optional[Path] = None,
        drop_zero_positive: bool = False,
        random_seed: Optional[int] = None,
        validate_on_init: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split_name
        self.split_path = Path(split_path)

        # 1. Path Validation
        if not split_path.exists():
            raise FileNotFoundError(f"Split LMDB not found at {split_path}")

        # 2. Resource Initialization
        self.resources = resources
        if resources is not None:
            # Shared memory path (Optimization for num_workers > 0)
            self.graph_store = resources.graph_store
            self.global_embeddings = resources.global_embeddings
        else:
            # Standalone mode
            self.graph_store = GraphStore(vocabulary_path=str(vocabulary_path))
            self.global_embeddings = GlobalEmbeddingStore(embeddings_dir, vocabulary_path)

        # 3. Sample Store (Per-process LMDB read handle)
        self.sample_store: Optional[EmbeddingStore] = None
        temp_store = EmbeddingStore(self.split_path)
        try:
            self.sample_ids = temp_store.get_sample_ids()
        finally:
            temp_store.close()
            del temp_store

        # 4. Filtering Logic
        if sample_filter_path:
            self._apply_sample_filter(sample_filter_path)

        if drop_zero_positive:
            self._apply_zero_positive_filter()

        if sample_limit:
            self._apply_sample_limit(sample_limit, random_seed)

        if validate_on_init:
            self._assert_all_samples_valid()

        logger.info(f"GRetrievalDataset[{self.split}] initialized: {len(self.sample_ids)} samples.")

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> GRetrievalData:
        """Load single sample from LMDB with strict PyG validation."""
        sample_id = self.sample_ids[idx]
        raw = self._get_sample_store().load_sample(sample_id)
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
        raws = self._get_sample_store().load_samples(sample_ids)
        return [self._build_data(raw, sample_id, idx) for raw, sample_id, idx in zip(raws, sample_ids, indices)]

    def _build_data(self, raw: Dict[str, Any], sample_id: str, idx: int) -> GRetrievalData:
        _validate_raw_sample(raw, sample_id)

        edge_index = raw["edge_index"]
        num_nodes = int(raw["num_nodes"])
        node_global_ids = raw["node_global_ids"]
        node_embedding_ids = raw["node_embedding_ids"]

        question_emb = raw["question_emb"]

        answer_ids = raw["answer_entity_ids"]
        answer_ids_len = raw["answer_entity_ids_len"]

        q_local_indices = raw["q_local_indices"]
        a_local_indices = raw["a_local_indices"]
        pair_start_node_locals = raw["pair_start_node_locals"]
        pair_answer_node_locals = raw["pair_answer_node_locals"]
        pair_edge_local_ids = raw["pair_edge_local_ids"]
        pair_edge_counts = raw["pair_edge_counts"]
        pair_shortest_lengths = raw.get("pair_shortest_lengths")
        topic_one_hot = raw["topic_one_hot"]

        data_kwargs: Dict[str, Any] = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "edge_attr": raw["edge_attr"],  # Global relation IDs
            "labels": raw["labels"],
            "node_global_ids": node_global_ids,
            "node_embedding_ids": node_embedding_ids,
            "question_emb": question_emb,
            "question": raw["question"],
            "topic_one_hot": topic_one_hot,
            "q_local_indices": q_local_indices,
            "a_local_indices": a_local_indices,
            "pair_start_node_locals": pair_start_node_locals,
            "pair_answer_node_locals": pair_answer_node_locals,
            "pair_edge_local_ids": pair_edge_local_ids,
            "pair_edge_counts": pair_edge_counts,
            "answer_entity_ids": answer_ids,
            "answer_entity_ids_len": answer_ids_len,
            "sample_id": sample_id,
            "idx": idx,
        }
        if pair_shortest_lengths is not None:
            data_kwargs["pair_shortest_lengths"] = pair_shortest_lengths
        data = GRetrievalData(**data_kwargs)
        return data

    def close(self) -> None:
        if getattr(self, "sample_store", None) is not None:
            self.sample_store.close()
            self.sample_store = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()

    def _get_sample_store(self) -> EmbeddingStore:
        """Lazy store accessor used inside worker processes."""
        if self.sample_store is None:
            self.sample_store = EmbeddingStore(self.split_path)
        return self.sample_store

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
        This prevents静默丢样本导致的数据分布漂移。
        """
        if not self.sample_ids:
            return

        temp_store = EmbeddingStore(self.split_path)
        try:
            for sid in self.sample_ids:
                raw = temp_store.load_sample(sid)
                _validate_raw_sample(raw, sid)
        finally:
            temp_store.close()

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
    """
    Factory adapting the injected 'cfg' (dataset_cfg) to the Dataset class.

    Expected cfg structure (from YAML):
    {
       "name": "webqsp",
       "paths": {
           "vocabulary": "...",
           "embeddings": "..."
       },
       "sample_limit": { "train": 1000 } (Optional)
       "random_seed": 42
    }
    """

    # 1. Resolve Split Path
    # Convention: ${embeddings_dir}/${split_name}.lmdb
    # This logic matches what the DataModule expects implicitly
    emb_root = Path(cfg["paths"]["embeddings"])

    split_path = emb_root / f"{split_name}.lmdb"

    # 2. Resolve Filters
    sample_limit = None
    if "sample_limit" in cfg and cfg["sample_limit"]:
        # Support both "sample_limit: 100" and "sample_limit: {train: 100}"
        sl_cfg = cfg["sample_limit"]
        if isinstance(sl_cfg, dict):
            sample_limit = int(sl_cfg.get(split_name, 0))
        else:
            sample_limit = int(sl_cfg)

    logger.info("Loaded retrieval LMDB; structural encoding is controlled by the retriever config.")

    drop_zero_positive = bool(cfg.get("drop_zero_positive", False)) and split_name == "train"

    return GRetrievalDataset(
        split_path=split_path,
        vocabulary_path=Path(cfg["paths"]["vocabulary"]),
        embeddings_dir=emb_root,
        dataset_name=cfg.get("name", "unknown"),
        split_name=split_name,
        resources=resources,
        sample_limit=sample_limit,
        sample_filter_path=Path(cfg.get("sample_filter_path")) if cfg.get("sample_filter_path") else None,
        drop_zero_positive=drop_zero_positive,
        random_seed=cfg.get("random_seed"),
        validate_on_init=bool(cfg.get("validate_on_init", False)),
    )


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


def _validate_raw_sample(raw: Dict[str, Any], sample_id: str) -> None:
    """Shared raw-schema validation to guarantee PyG Data integrity."""
    required_keys = [
        "edge_index",
        "edge_attr",
        "labels",
        "num_nodes",
        "node_global_ids",
        "node_embedding_ids",
        "question_emb",
        "question",
        "topic_one_hot",
        "q_local_indices",
        "a_local_indices",
        "answer_entity_ids",
        "answer_entity_ids_len",
        "pair_start_node_locals",
        "pair_answer_node_locals",
        "pair_edge_local_ids",
        "pair_edge_counts",
    ]
    missing = [k for k in required_keys if k not in raw]
    if missing:
        raise KeyError(f"Sample {sample_id} missing keys: {missing}")

    edge_index = _expect_tensor(raw, "edge_index", sample_id=sample_id, dtype=torch.long, dim=2)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E] for {sample_id}, got {tuple(edge_index.shape)}")
    num_edges = edge_index.size(1)
    if num_edges == 0:
        raise ValueError(f"edge_index empty for {sample_id}")

    num_nodes = int(raw.get("num_nodes", 0) or 0)
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive for {sample_id}, got {num_nodes}")
    if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
        raise ValueError(
            f"edge_index out of range for {sample_id}: min={edge_index.min().item()} max={edge_index.max().item()} num_nodes={num_nodes}"
        )

    node_global_ids = _expect_tensor(raw, "node_global_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    if node_global_ids.numel() != num_nodes:
        raise ValueError(f"node_global_ids length {node_global_ids.numel()} != num_nodes {num_nodes} for {sample_id}")
    if torch.unique(node_global_ids).numel() != num_nodes:
        raise ValueError(f"node_global_ids must be unique per sample: {sample_id}")

    node_embedding_ids = _expect_tensor(raw, "node_embedding_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    if node_embedding_ids.numel() != num_nodes:
        raise ValueError(f"node_embedding_ids length {node_embedding_ids.numel()} != num_nodes {num_nodes} for {sample_id}")

    question_emb = _expect_tensor(raw, "question_emb", sample_id=sample_id, dim=2)
    if not torch.is_floating_point(question_emb):
        raise ValueError(f"question_emb for {sample_id} must be floating point, got dtype={question_emb.dtype}")
    if not isinstance(raw.get("question"), str):
        raise TypeError(f"question for {sample_id} must be a string.")

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
                f"labels out of range for {sample_id}: min={float(labels.min().item())} " f"max={float(labels.max().item())}"
            )

    topic_one_hot = _expect_tensor(raw, "topic_one_hot", sample_id=sample_id)
    if not torch.is_floating_point(topic_one_hot):
        raise ValueError(f"topic_one_hot for {sample_id} must be floating point, got dtype={topic_one_hot.dtype}")
    _validate_topic_one_hot(topic_one_hot, num_nodes, sample_id)

    q_local_indices = _expect_tensor(raw, "q_local_indices", sample_id=sample_id, dtype=torch.long, dim=1)
    a_local_indices = _expect_tensor(raw, "a_local_indices", sample_id=sample_id, dtype=torch.long, dim=1)
    _validate_local_indices(q_local_indices, num_nodes, "q_local_indices", sample_id)
    _validate_local_indices(a_local_indices, num_nodes, "a_local_indices", sample_id)

    answer_ids = _expect_tensor(raw, "answer_entity_ids", sample_id=sample_id, dtype=torch.long, dim=1)
    answer_ids = answer_ids.view(-1)
    _validate_answer_ids(answer_ids, sample_id)
    answer_len = _expect_tensor(raw, "answer_entity_ids_len", sample_id=sample_id, dtype=torch.long, dim=1)
    answer_len_tensor = answer_len.view(-1)
    if answer_len_tensor.numel() != 1 or int(answer_len_tensor.item()) != answer_ids.numel():
        raise ValueError(
            f"answer_entity_ids_len mismatch for {sample_id}: declared {answer_len_tensor.tolist()} vs actual {answer_ids.numel()}"
        )
    if "gt_path_edge_indices" in raw or "gt_path_node_indices" in raw:
        raise ValueError(f"gt_path_* fields are deprecated for {sample_id}; rebuild dataset without them.")
    if "pair_edge_offsets" in raw:
        raise ValueError(f"pair_edge_offsets is deprecated for {sample_id}; rebuild dataset with pair_edge_counts only.")
    if "pair_edge_indices" in raw:
        raise ValueError(f"pair_edge_indices is deprecated for {sample_id}; use pair_edge_local_ids.")

    pair_start = _expect_tensor(raw, "pair_start_node_locals", sample_id=sample_id, dtype=torch.long, dim=1)
    pair_answer = _expect_tensor(raw, "pair_answer_node_locals", sample_id=sample_id, dtype=torch.long, dim=1)
    if pair_start.numel() != pair_answer.numel():
        raise ValueError(
            f"pair_start_node_locals length {pair_start.numel()} != "
            f"pair_answer_node_locals length {pair_answer.numel()} for {sample_id}"
        )
    _validate_local_indices(pair_start, num_nodes, "pair_start_node_locals", sample_id)
    _validate_local_indices(pair_answer, num_nodes, "pair_answer_node_locals", sample_id)

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
            f"pair_edge_counts length {pair_edge_counts.numel()} != pair_count " f"({pair_start.numel()}) for {sample_id}"
        )
    if (pair_edge_counts < 0).any():
        raise ValueError(f"pair_edge_counts contains negative values for {sample_id}")
    total_edges = int(pair_edge_counts.sum().item())
    if total_edges != pair_edge_local_ids.numel():
        raise ValueError(
            f"pair_edge_counts sum {total_edges} != "
            f"pair_edge_local_ids length {pair_edge_local_ids.numel()} for {sample_id}"
        )

    pair_shortest_lengths = raw.get("pair_shortest_lengths")
    if pair_shortest_lengths is not None:
        if not torch.is_tensor(pair_shortest_lengths):
            raise TypeError(f"pair_shortest_lengths for {sample_id} must be a torch.Tensor.")
        if pair_shortest_lengths.dtype != torch.long:
            raise ValueError(
                f"pair_shortest_lengths for {sample_id} must be dtype=torch.long, got {pair_shortest_lengths.dtype}"
            )
        pair_shortest_lengths = pair_shortest_lengths.view(-1)
        if pair_shortest_lengths.numel() != pair_start.numel():
            raise ValueError(
                f"pair_shortest_lengths length {pair_shortest_lengths.numel()} != pair_count "
                f"({pair_start.numel()}) for {sample_id}"
            )
        if (pair_shortest_lengths < 0).any():
            raise ValueError(f"pair_shortest_lengths contains negative values for {sample_id}")
