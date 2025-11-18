import json
import os
import torch
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Set, Iterable
from torch_geometric.data import Dataset, Data as GraphData
from .components import (
    SharedDataResources,
    GraphStore,
    EmbeddingStore,
    GlobalEmbeddingStore,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetPaths:
    data_root: Path
    vocabulary_path: Path
    embeddings_dir: Path
    samples_path: Path


def resolve_dataset_paths(config: Dict[str, Any], split: str) -> DatasetPaths:
    dataset_name = config["name"]

    data_dir_override = config.get("data_dir")
    if not data_dir_override:
        raise ValueError(
            "dataset_cfg.data_dir must be set in your data config (e.g. configs/data/retrieval.yaml)"
        )
    data_root = Path(data_dir_override).expanduser().resolve()

    vocabulary_path = Path(config.get("vocabulary_path", data_root / "vocabulary" / "vocabulary.lmdb"))
    embeddings_dir = Path(config.get("embeddings_dir", data_root / "embeddings"))
    samples_lmdb_path = Path(config.get("samples_path", embeddings_dir / f"{split}.lmdb"))

    return DatasetPaths(
        data_root=data_root,
        vocabulary_path=vocabulary_path,
        embeddings_dir=embeddings_dir,
        samples_path=samples_lmdb_path,
    )


class RetrievalDataset(Dataset):
    """Efficient retrieval dataset that trusts emb.py output format."""

    def __init__(self, config: Dict[str, Any], *, resources: Optional[SharedDataResources] = None):
        super().__init__()
        self.dataset_name = config["name"]
        self.split = config.get("split", "train")

        self.paths = resolve_dataset_paths(config, self.split)
        data_root = self.paths.data_root
        vocabulary_path = self.paths.vocabulary_path
        samples_lmdb_path = self.paths.samples_path
        embeddings_dir = self.paths.embeddings_dir

        # Initialize storage components
        self.resources = resources
        if resources is not None:
            self.graph_store = resources.graph_store
            self.global_embeddings = resources.global_embeddings
        else:
            self.graph_store = GraphStore(vocabulary_path=str(vocabulary_path))
            self.global_embeddings = GlobalEmbeddingStore(embeddings_dir, vocabulary_path)

        # Core components
        self.sample_store = EmbeddingStore(samples_lmdb_path)

        # Load sample IDs
        self.sample_ids = self.sample_store.get_sample_ids()

        # Debug mode data reduction
        if config.get("debug", False):
            original_size = len(self.sample_ids)
            if self.split == "train":
                self.sample_ids = self.sample_ids[: max(1, len(self.sample_ids) // 10)]
            elif self.split == "test":
                self.sample_ids = self.sample_ids[: max(1, len(self.sample_ids) // 100)]
            logger.info(f"Debug mode: {self.split} reduced from {original_size} to {len(self.sample_ids)}")

        self._apply_sample_filter(config)
        self._apply_sample_limit(config)
        self._drop_empty_samples()

        logger.info(
            f"RetrievalDataset initialized: {len(self.sample_ids)} samples ({self.split}). Note: empty graphs will be skipped during iteration."
        )

        # GPT triples remain lazy-loaded until explicitly requested.
        self.gpt_triples_path = data_root / "gpt_triples.pth"
        self.gpt_triples_ids: Dict[str, Any] = {}
        self._gpt_triples_available = self.gpt_triples_path.exists()
        self._gpt_triples_loaded = False
        if not self._gpt_triples_available:
            logger.info("GPT triples not found at %s; skipping load until required", self.gpt_triples_path)

    def get(self, idx: int) -> GraphData:
        """Load single sample, deferring embedding lookups to the DataLoader."""
        sample_id = self.sample_ids[idx]
        raw = self.sample_store.load_sample(sample_id)

        # Trust emb.py output directly
        edge_index = raw["edge_index"]
        edge_attr = raw["edge_attr"]  # These are global relation IDs
        labels = raw["labels"]
        num_nodes = raw["num_nodes"]

        if edge_index.numel() == 0 or num_nodes == 0:
            raise RuntimeError(f"Empty graph detected after filtering: {sample_id}")

        # Node features - only store IDs, embeddings will be computed in the DataLoader
        node_global_ids = raw["node_global_ids"]

        topic_one_hot = raw["topic_one_hot"]

        # Question embedding - ensure batch dimension
        question_emb = raw["question_emb"]
        if question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)  # [1, D] for batching

        # --- Embedding lookups are now deferred ---

        # Build PyG data object - without x and with edge_attr as global relation IDs
        return GraphData(
            # Core graph structure
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,  # Pass global relation IDs
            labels=labels,
            # Node features and mappings (NO 'x'/'node_embeddings'/'is_text_mask')
            node_global_ids=node_global_ids,
            topic_one_hot=topic_one_hot,
            # Question information
            question_emb=question_emb,
            question=raw.get("question", ""),
            q_local_indices=raw.get("q_local_indices", []),
            a_local_indices=raw.get("a_local_indices", []),
            # Path supervision (optional)
            gt_paths_nodes=raw.get("gt_paths_nodes", []),
            gt_paths_triples=raw.get("gt_paths_triples", []),
            # Metadata
            sample_id=sample_id,
            idx=idx,
        )

    def len(self) -> int:
        """Dataset size"""
        return len(self.sample_ids)

    def close(self) -> None:
        """Close resources"""
        self.sample_store.close()

    def _apply_sample_limit(self, config: Dict[str, Any]) -> None:
        """Optionally subsample a fixed number of samples per split for quick checks."""

        sample_limit_cfg = config.get("sample_limit")
        if sample_limit_cfg is None:
            return

        if isinstance(sample_limit_cfg, dict):
            limit_raw = sample_limit_cfg.get(self.split)
        else:
            limit_raw = sample_limit_cfg

        if limit_raw is None:
            return

        try:
            limit = int(limit_raw)
        except (TypeError, ValueError):
            logger.warning("Invalid sample_limit=%s for split=%s; ignoring", limit_raw, self.split)
            return

        if limit <= 0:
            logger.warning("sample_limit <= 0 for split=%s; ignoring", self.split)
            return

        if len(self.sample_ids) <= limit:
            return

        generator = torch.Generator()
        seed_value = config.get("random_seed")
        if seed_value is not None:
            try:
                generator.manual_seed(int(seed_value) + hash(self.split) % (2**31))
            except Exception:
                generator.manual_seed(int(seed_value))

        perm = torch.randperm(len(self.sample_ids), generator=generator).tolist()
        selected_indices = perm[:limit]
        self.sample_ids = [self.sample_ids[idx] for idx in selected_indices]
        logger.info("Sample limit applied: split=%s, limit=%d, selected=%d", self.split, limit, len(self.sample_ids))

    def _drop_empty_samples(self) -> None:
        """Remove samples that no longer contain any edges or nodes."""
        valid_ids = []
        dropped = 0
        for sid in self.sample_ids:
            raw = self.sample_store.load_sample(sid)
            edge_index = raw.get("edge_index")
            num_nodes = raw.get("num_nodes", 0)
            if edge_index is None or not hasattr(edge_index, "numel"):
                dropped += 1
                continue
            if edge_index.numel() == 0 or num_nodes == 0:
                dropped += 1
                continue
            valid_ids.append(sid)
        if dropped:
            logger.warning("Removed %d empty samples from split=%s", dropped, self.split)
        self.sample_ids = valid_ids

    def _apply_sample_filter(self, config: Dict[str, Any]) -> None:
        """Optionally restrict samples to a curated subset (e.g., cascade hard set)."""
        filter_path = config.get("sample_filter_path") or config.get("cascade_filter_path")
        if not filter_path:
            return
        path = Path(filter_path).expanduser()
        if not path.exists():
            logger.warning("sample_filter_path=%s does not exist; skipping filter.", path)
            return
        try:
            keep_ids = self._load_filter_ids(path)
        except Exception as exc:
            logger.error("Failed to load sample filter from %s: %s", path, exc)
            return
        before = len(self.sample_ids)
        self.sample_ids = [sid for sid in self.sample_ids if sid in keep_ids]
        logger.info(
            "Sample filter applied: split=%s, kept %d/%d samples (path=%s)",
            self.split,
            len(self.sample_ids),
            before,
            path,
        )

    @staticmethod
    def _load_filter_ids(path: Path) -> Set[str]:
        """Load a set of sample IDs from JSON/JSONL/plaintext."""
        text = path.read_text(encoding="utf-8").strip()
        keep: Set[str] = set()
        if not text:
            return keep
        if text[0] in "[{":
            data = json.loads(text)
            ids: Iterable[Any]
            if isinstance(data, dict) and "sample_ids" in data:
                ids = data["sample_ids"]
            elif isinstance(data, list):
                ids = data
            else:
                raise ValueError("Unsupported filter JSON format.")
            for item in ids:
                if item is None:
                    continue
                keep.add(str(item))
            return keep
        # Fallback: newline-delimited entries
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except Exception:
                    keep.add(line)
                    continue
                if isinstance(obj, dict):
                    val = obj.get("sample_id") or obj.get("id")
                    if val is not None:
                        keep.add(str(val))
                else:
                    keep.add(str(obj))
            else:
                keep.add(line)
        return keep

    # --- Extensions for GPT triples integration ---
    def _load_gpt_triples_ids(self) -> None:
        """Load GPT triples and convert to global ID tuples using vocabulary.

        Stores mapping: sample_id -> set((h_id, r_id, t_id), ...)
        """
        import torch

        data = torch.load(self.gpt_triples_path, map_location="cpu")
        if not isinstance(data, dict):
            return
        e2i = self.graph_store.entity2id
        r2i = self.graph_store.relation2id
        mapped: Dict[str, set] = {}
        for sid, triples in data.items():
            id_triples = set()
            for t in triples or []:
                if not isinstance(t, (list, tuple)) or len(t) != 3:
                    continue
                h, r, v = str(t[0]), str(t[1]), str(t[2])
                h_id = e2i.get(h)
                r_id = r2i.get(r)
                v_id = e2i.get(v)
                if h_id is None or r_id is None or v_id is None:
                    continue
                id_triples.add((int(h_id), int(r_id), int(v_id)))
            if id_triples:
                mapped[str(sid)] = id_triples
        self.gpt_triples_ids = mapped

    def _ensure_gpt_triples_loaded(self) -> None:
        if self._gpt_triples_loaded or not self._gpt_triples_available:
            return
        try:
            self._load_gpt_triples_ids()
            logger.info("Loaded GPT triples: %d samples", len(self.gpt_triples_ids))
        except Exception as exc:
            logger.warning("Failed to load GPT triples lazily: %s", exc)
            self.gpt_triples_ids = {}
        finally:
            self._gpt_triples_loaded = True

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Return metadata required by semantic metrics and diagnostics.

        Returns keys (global ID space):
          - sample_id: str
          - h_id_list: List[int]
          - t_id_list: List[int]
          - r_id_list: List[int]
          - a_entity_id_list: List[int]
          - has_gpt: bool, num_gpt_triples: int
        """
        sid = self.sample_ids[idx]
        raw = self.sample_store.load_sample(sid)
        edge_index = raw["edge_index"]  # [2, E]
        edge_attr = raw["edge_attr"]  # [E]
        node_global_ids = raw["node_global_ids"]  # [N]

        # Build per-edge (h_id, r_id, t_id)
        if edge_index.numel() > 0:
            h_local = edge_index[0].tolist()
            t_local = edge_index[1].tolist()
            r_ids = edge_attr.tolist()
            h_ids = [int(node_global_ids[u]) for u in h_local]
            t_ids = [int(node_global_ids[v]) for v in t_local]
        else:
            h_ids, t_ids, r_ids = [], [], []

        # Answer entity global IDs
        a_locals = raw.get("a_local_indices", []) or []
        a_entity_ids = [int(node_global_ids[i]) for i in a_locals if 0 <= i < len(node_global_ids)]

        self._ensure_gpt_triples_loaded()
        gpt_set = self.gpt_triples_ids.get(sid, set())
        return {
            "sample_id": sid,
            "h_id_list": h_ids,
            "t_id_list": t_ids,
            "r_id_list": r_ids,
            "a_entity_id_list": a_entity_ids,
            "has_gpt": bool(gpt_set),
            "num_gpt_triples": len(gpt_set),
        }

    def get_gpt_triples_ids(self, sample_id: str) -> Optional[set]:
        """Get GPT triples as global ID tuples set for a sample (if available)."""
        self._ensure_gpt_triples_loaded()
        return self.gpt_triples_ids.get(sample_id)

    def __getitem__(self, idx: int):
        """Compatibility wrapper for standard PyTorch DataLoader"""
        return self.get(idx)


def create_dataset(
    config: Dict[str, Any],
    split: str,
    *,
    resources: Optional[SharedDataResources] = None,
) -> RetrievalDataset:
    """Helper matching Lightning-Hydra template expectations."""

    dataset_config = dict(config)
    dataset_config["split"] = split
    return RetrievalDataset(dataset_config, resources=resources)
