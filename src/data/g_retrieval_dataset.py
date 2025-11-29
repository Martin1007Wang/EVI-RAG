import json
import logging
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

        self._drop_empty_samples()

        logger.info(
            f"GRetrievalDataset[{self.split}] initialized: {len(self.sample_ids)} samples."
        )

        # 5. GPT Triples (Lazy Load)
        self.gpt_triples_path = gpt_triples_path
        self.gpt_triples_ids: Dict[str, Any] = {}
        self._gpt_triples_loaded = False

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> GraphData:
        """Load single sample from LMDB."""
        sample_id = self.sample_ids[idx]
        raw = self._get_sample_store().load_sample(sample_id)

        # Basic Checks
        edge_index = torch.as_tensor(raw["edge_index"], dtype=torch.long)
        if edge_index.numel() == 0:
            raise RuntimeError(f"Empty graph detected: {sample_id}")
        num_nodes = raw["num_nodes"]

        # Features
        node_global_ids = torch.as_tensor(raw["node_global_ids"], dtype=torch.long)
        node_embedding_ids = raw.get("node_embedding_ids")
        if node_embedding_ids is None:
            raise RuntimeError(f"Missing node_embedding_ids in {sample_id}")
        node_embedding_ids = torch.as_tensor(node_embedding_ids, dtype=torch.long)

        question_emb = torch.as_tensor(raw["question_emb"], dtype=torch.float32)
        if question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)

        # Answers (always 1D long tensor + length tensor)
        answer_ids = raw.get("answer_entity_ids")
        if answer_ids is None:
            answer_ids = torch.empty(0, dtype=torch.long)
        elif not torch.is_tensor(answer_ids):
            answer_ids = torch.tensor(answer_ids, dtype=torch.long)
        answer_ids_len = torch.tensor([answer_ids.numel()], dtype=torch.long)

        q_local_indices = torch.as_tensor(raw.get("q_local_indices", []), dtype=torch.long).view(-1)
        a_local_indices = torch.as_tensor(raw.get("a_local_indices", []), dtype=torch.long).view(-1)

        return GraphData(
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
            # Metadata
            sample_id=sample_id,
            idx=idx,
        )

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
            # Deterministic subset per split
            generator.manual_seed(int(seed) + hash(self.split) % (2**31))
            
        perm = torch.randperm(len(self.sample_ids), generator=generator).tolist()
        self.sample_ids = [self.sample_ids[i] for i in perm[:limit]]
        logger.info(f"Subsampled {self.split} to {len(self.sample_ids)} samples.")

    def _drop_empty_samples(self) -> None:
        """
        Filter out IDs that map to empty graphs in LMDB.
        Note: This requires a full scan, which might be slow for huge datasets.
        Optimization: Trust the preprocessing pipeline instead of checking here?
        Current: Check checks existence of keys, fast enough for LMDB.
        """
        if not self.sample_ids:
            return

        invalid: Set[str] = set()
        temp_store = EmbeddingStore(self.split_path)
        try:
            for sid in self.sample_ids:
                try:
                    raw = temp_store.load_sample(sid)
                except Exception as e:  # pragma: no cover - defensive guardrail
                    logger.warning("Failed to load sample %s: %s. Dropping.", sid, e)
                    invalid.add(sid)
                    continue

                edge_index = raw.get("edge_index")
                num_nodes = int(raw.get("num_nodes", 0) or 0)
                node_embedding_ids = raw.get("node_embedding_ids")

                if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
                    invalid.add(sid)
                    continue
                if num_nodes <= 0:
                    invalid.add(sid)
                    continue
                if node_embedding_ids is None:
                    invalid.add(sid)
                    continue
                if hasattr(node_embedding_ids, "numel") and node_embedding_ids.numel() == 0:
                    invalid.add(sid)
                    continue
                if not hasattr(node_embedding_ids, "numel") and len(node_embedding_ids) == 0:
                    invalid.add(sid)
        finally:
            temp_store.close()

        if invalid:
            before = len(self.sample_ids)
            self.sample_ids = [sid for sid in self.sample_ids if sid not in invalid]
            logger.warning(
                "Pruned %d invalid/empty samples from %s: %d -> %d. Examples: %s",
                len(invalid),
                self.split,
                before,
                len(self.sample_ids),
                list(invalid)[:5],
            )

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
        gpt_triples_path=gpt_path
    )
