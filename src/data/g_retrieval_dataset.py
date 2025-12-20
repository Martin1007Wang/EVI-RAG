import json
import logging
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

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
        if key in {"q_local_indices", "a_local_indices"}:
            return int(self.num_nodes)
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
        random_seed: Optional[int] = None,
        gpt_triples_path: Optional[Path] = None,
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
        
        if sample_limit:
            self._apply_sample_limit(sample_limit, random_seed)

        if validate_on_init:
            self._assert_all_samples_valid()

        logger.info(
            f"GRetrievalDataset[{self.split}] initialized: {len(self.sample_ids)} samples."
        )

        # 5. GPT Triples (Lazy Load)
        self.gpt_triples_path = gpt_triples_path
        self.gpt_triples_ids: Dict[str, Any] = {}
        self._gpt_triples_loaded = False

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> GRetrievalData:
        """Load single sample from LMDB with strict PyG validation."""
        sample_id = self.sample_ids[idx]
        raw = self._get_sample_store().load_sample(sample_id)
        _validate_raw_sample(raw, sample_id)

        edge_index = torch.as_tensor(raw["edge_index"], dtype=torch.long)
        num_nodes = int(raw["num_nodes"])
        node_global_ids = torch.as_tensor(raw["node_global_ids"], dtype=torch.long)
        node_embedding_ids = torch.as_tensor(raw["node_embedding_ids"], dtype=torch.long)

        question_emb = torch.as_tensor(raw["question_emb"], dtype=torch.float32)
        if question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)

        answer_ids = raw.get("answer_entity_ids")
        if answer_ids is None:
            answer_ids = torch.empty(0, dtype=torch.long)
        elif not torch.is_tensor(answer_ids):
            answer_ids = torch.tensor(answer_ids, dtype=torch.long)
        answer_ids_len = torch.tensor([answer_ids.numel()], dtype=torch.long)

        q_local_indices = torch.as_tensor(raw.get("q_local_indices", []), dtype=torch.long).view(-1)
        a_local_indices = torch.as_tensor(raw.get("a_local_indices", []), dtype=torch.long).view(-1)

        data = GRetrievalData(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_attr=torch.as_tensor(raw["edge_attr"], dtype=torch.long), # Global relation IDs
            labels=torch.as_tensor(raw["labels"], dtype=torch.float32),
            node_global_ids=node_global_ids,
            node_embedding_ids=node_embedding_ids,
            topic_one_hot=torch.as_tensor(raw["topic_one_hot"], dtype=torch.float32),
            question_emb=question_emb,
            question=raw.get("question", ""),
            q_local_indices=q_local_indices,
            a_local_indices=a_local_indices,
            answer_entity_ids=answer_ids,
            answer_entity_ids_len=answer_ids_len,
            sample_id=sample_id,
            idx=idx,
        )
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
            logger.warning(f"Filter path {path} not found. Ignoring.")
            return
            
        try:
            keep_ids = self._load_filter_ids(path)
            before = len(self.sample_ids)
            self.sample_ids = [sid for sid in self.sample_ids if sid in keep_ids]
            logger.info(f"Filtered {self.split}: {before} -> {len(self.sample_ids)} using {path.name}")
        except Exception as e:
            logger.error(f"Failed to load filter: {e}")

    @staticmethod
    def _load_filter_ids(path: Path) -> Set[str]:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return set()
        
        # Auto-detect format
        if text.startswith("{") or text.startswith("["):
             data = json.loads(text)
             if isinstance(data, list): return set(map(str, data))
             if isinstance(data, dict): return set(map(str, data.get("sample_ids", [])))
        
        # Line-based fallback
        return {line.strip() for line in text.splitlines() if line.strip()}

    # ------------------------------------------------------------------ #
    # GPT Triples Logic
    # ------------------------------------------------------------------ #
    def _ensure_gpt_triples_loaded(self) -> None:
        if self._gpt_triples_loaded:
            return
        
        if not self.gpt_triples_path or not self.gpt_triples_path.exists():
            self._gpt_triples_loaded = True # Mark as done to avoid retrying
            return

        try:
            logger.info(f"Loading GPT triples from {self.gpt_triples_path}...")
            # Use torch.load for .pth files, safer and faster than json for huge dicts
            data = torch.load(self.gpt_triples_path, map_location="cpu")
            if isinstance(data, dict):
                self._map_gpt_triples(data)
            logger.info(f"Loaded GPT triples for {len(self.gpt_triples_ids)} samples.")
        except Exception as e:
            logger.error(f"Failed to load GPT triples: {e}")
        finally:
            self._gpt_triples_loaded = True

    def _map_gpt_triples(self, raw_data: Dict[str, list]) -> None:
        """Convert raw strings to global IDs."""
        e2i = self.graph_store.entity2id
        r2i = self.graph_store.relation2id
        
        for sid, triples in raw_data.items():
            mapped_set = set()
            for t in triples:
                if len(t) != 3: continue
                try:
                    h, r, v = e2i[str(t[0])], r2i[str(t[1])], e2i[str(t[2])]
                    mapped_set.add((h, r, v))
                except KeyError:
                    continue # Skip triples with unknown entities
            
            if mapped_set:
                self.gpt_triples_ids[str(sid)] = mapped_set

    def get_gpt_triples_ids(self, sample_id: str) -> Optional[set]:
        self._ensure_gpt_triples_loaded()
        return self.gpt_triples_ids.get(sample_id)


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
           "embeddings": "...",
           "gpt_triples": "..." (Optional)
       },
       "sample_limit": { "train": 1000 } (Optional)
       "random_seed": 42
    }
    """
    
    # 1. Resolve Split Path
    # Convention: ${embeddings_dir}/${split_name}.lmdb
    # This logic matches what the DataModule expects implicitly
    emb_root = Path(cfg["paths"]["embeddings"])
    
    # Handle aliases like 'validation' -> 'val.lmdb' if needed, 
    # OR assume file names match split names exactly.
    # Let's try explicit match first, then fallback.
    split_path = emb_root / f"{split_name}.lmdb"
    if not split_path.exists() and split_name == "validation":
        split_path = emb_root / "val.lmdb"
    
    # 2. Resolve Filters
    sample_limit = None
    if "sample_limit" in cfg and cfg["sample_limit"]:
        # Support both "sample_limit: 100" and "sample_limit: {train: 100}"
        sl_cfg = cfg["sample_limit"]
        if isinstance(sl_cfg, dict):
            sample_limit = int(sl_cfg.get(split_name, 0))
        else:
            sample_limit = int(sl_cfg)

    # 3. GPT Path
    gpt_path = None
    if "gpt_triples" in cfg["paths"]:
        gpt_path = Path(cfg["paths"]["gpt_triples"])

    return GRetrievalDataset(
        split_path=split_path,
        vocabulary_path=Path(cfg["paths"]["vocabulary"]),
        embeddings_dir=emb_root,
        dataset_name=cfg.get("name", "unknown"),
        split_name=split_name,
        resources=resources,
        sample_limit=sample_limit,
        sample_filter_path=Path(cfg.get("sample_filter_path")) if cfg.get("sample_filter_path") else None,
        random_seed=cfg.get("random_seed"),
        gpt_triples_path=gpt_path,
        validate_on_init=bool(cfg.get("validate_on_init", False)),
    )


def _validate_local_indices(local_idx: torch.Tensor, num_nodes: int, field: str, sample_id: str) -> None:
    if local_idx.numel() == 0:
        return
    if local_idx.dim() != 1:
        raise ValueError(f"{field} for {sample_id} must be 1D.")
    if local_idx.min().item() < 0 or local_idx.max().item() >= num_nodes:
        raise ValueError(f"{field} out of range for {sample_id}: num_nodes={num_nodes}, values={local_idx.tolist()}")


def _ensure_tensor(obj: Any, dtype: torch.dtype, name: str, sample_id: str) -> torch.Tensor:
    try:
        return torch.as_tensor(obj, dtype=dtype)
    except Exception as e:  # pragma: no cover - defensive guardrail
        raise ValueError(f"Failed to convert {name} for {sample_id} to tensor: {e}") from e


def _validate_topic_one_hot(topic_one_hot: torch.Tensor, num_nodes: int, sample_id: str) -> None:
    if topic_one_hot.dim() == 1:
        topic_one_hot = topic_one_hot.unsqueeze(-1)
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
        "topic_one_hot",
    ]
    missing = [k for k in required_keys if k not in raw]
    if missing:
        raise KeyError(f"Sample {sample_id} missing keys: {missing}")

    edge_index = _ensure_tensor(raw["edge_index"], torch.long, "edge_index", sample_id)
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

    node_global_ids = _ensure_tensor(raw["node_global_ids"], torch.long, "node_global_ids", sample_id).view(-1)
    if node_global_ids.numel() != num_nodes:
        raise ValueError(
            f"node_global_ids length {node_global_ids.numel()} != num_nodes {num_nodes} for {sample_id}"
        )
    if torch.unique(node_global_ids).numel() != num_nodes:
        raise ValueError(f"node_global_ids must be unique per sample: {sample_id}")

    node_embedding_ids = _ensure_tensor(raw["node_embedding_ids"], torch.long, "node_embedding_ids", sample_id).view(-1)
    if node_embedding_ids.numel() != num_nodes:
        raise ValueError(
            f"node_embedding_ids length {node_embedding_ids.numel()} != num_nodes {num_nodes} for {sample_id}"
        )

    edge_attr = _ensure_tensor(raw["edge_attr"], torch.long, "edge_attr", sample_id).view(-1)
    labels = _ensure_tensor(raw["labels"], torch.float32, "labels", sample_id).view(-1)
    if edge_attr.numel() != num_edges:
        raise ValueError(f"edge_attr length {edge_attr.numel()} != num_edges {num_edges} for {sample_id}")
    if labels.numel() != num_edges:
        raise ValueError(f"labels length {labels.numel()} != num_edges {num_edges} for {sample_id}")

    topic_one_hot = _ensure_tensor(raw["topic_one_hot"], torch.float32, "topic_one_hot", sample_id)
    _validate_topic_one_hot(topic_one_hot, num_nodes, sample_id)

    q_local_indices = _ensure_tensor(raw.get("q_local_indices", []), torch.long, "q_local_indices", sample_id).view(-1)
    a_local_indices = _ensure_tensor(raw.get("a_local_indices", []), torch.long, "a_local_indices", sample_id).view(-1)
    _validate_local_indices(q_local_indices, num_nodes, "q_local_indices", sample_id)
    _validate_local_indices(a_local_indices, num_nodes, "a_local_indices", sample_id)

    answer_ids = raw.get("answer_entity_ids")
    if answer_ids is None:
        answer_ids = torch.empty(0, dtype=torch.long)
    else:
        answer_ids = _ensure_tensor(answer_ids, torch.long, "answer_entity_ids", sample_id).view(-1)
    _validate_answer_ids(answer_ids, sample_id)
    answer_len = raw.get("answer_entity_ids_len")
    if answer_len is not None:
        answer_len_tensor = _ensure_tensor(answer_len, torch.long, "answer_entity_ids_len", sample_id).view(-1)
        if answer_len_tensor.numel() != 1 or int(answer_len_tensor.item()) != answer_ids.numel():
            raise ValueError(
                f"answer_entity_ids_len mismatch for {sample_id}: declared {answer_len_tensor.tolist()} vs actual {answer_ids.numel()}"
            )
