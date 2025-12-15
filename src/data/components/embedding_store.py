import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import lmdb
import torch

logger = logging.getLogger(__name__)

class GlobalEmbeddingStore:
    """
    In-memory read-only store for Entity/Relation embeddings.
    Designed for fast lookups during training/inference.
    """
    def __init__(self, embeddings_dir: Union[str, Path], vocabulary_path: Union[str, Path]):
        self.embeddings_dir = Path(embeddings_dir)
        # 1. Vocabulary (Optional, only if needed for count checks)
        self.num_total_entities = self._load_vocab_size(Path(vocabulary_path))
        
        # 2. Pre-load Embeddings (CPU Pinned Memory for speed)
        self.entity_embeddings = self._load_tensor(self.embeddings_dir / "entity_embeddings.pt")
        self.relation_embeddings = self._load_tensor(self.embeddings_dir / "relation_embeddings.pt")

        # Dimension note: embedding rows correspond to textual entities only;
        # non-text entities map to embedding_id=0. So rows are expected to be
        # <= vocab size. Warn only if the table is effectively empty.
        rows = self.entity_embeddings.size(0)
        if rows <= 1:
            logger.warning(
                "Entity embedding table has <=1 row (rows=%d). "
                "Non-text fallback uses id=0, but textual entities would be missing.",
                rows,
            )
        else:
            logger.info(
                "Entity embedding rows: %d (vocab entities: %d). "
                "Non-text entities use embedding_id=0; textual entities occupy 1..max_id.",
                rows,
                self.num_total_entities,
            )

    def _load_vocab_size(self, path: Path) -> int:
        if not path.exists():
            logger.warning(f"Vocab LMDB not found at {path}, skipping size check.")
            return 0
        try:
            # Quick open just to check size
            with lmdb.open(str(path), readonly=True, lock=False, max_readers=1) as env:
                with env.begin() as txn:
                    # Faster than unpickling the whole dict if we just need length
                    # But usually we store the dict.
                    data = txn.get(b"entity_to_id")
                    if data:
                        return len(pickle.loads(data))
            return 0
        except Exception as e:
            logger.warning(f"Failed to read vocab size: {e}")
            return 0

    def _load_tensor(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Embedding file missing: {path}")
        logger.info(f"Loading {path}...")
        # map_location='cpu' is crucial to avoid VRAM OOM
        return torch.load(path, map_location="cpu")

    def get_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Fetch embeddings. entity_ids can be on GPU."""
        # Optimized: Indexing CPU tensor with CPU indices, then move to GPU.
        # Moving huge Embedding table to GPU is usually impossible.
        device = entity_ids.device
        cpu_ids = entity_ids.cpu()
        return self.entity_embeddings.index_select(0, cpu_ids).to(device)

    def get_relation_embeddings(self, relation_ids: torch.Tensor) -> torch.Tensor:
        device = relation_ids.device
        cpu_ids = relation_ids.cpu()
        return self.relation_embeddings.index_select(0, cpu_ids).to(device)
    
    @property
    def entity_dim(self) -> int:
        return self.entity_embeddings.size(-1)

    @property
    def relation_dim(self) -> int:
        return self.relation_embeddings.size(-1)


class EmbeddingStore:
    """
    LMDB wrapper for reading graph samples.
    Safe for Multiprocessing DataLoader.
    """
    def __init__(self, lmdb_path: Union[str, Path]):
        self.path = str(lmdb_path)
        self.env: Optional[lmdb.Environment] = None
        
        # Check existence immediately
        if not Path(self.path).exists():
            raise FileNotFoundError(f"LMDB not found: {self.path}")

    def _init_env(self):
        """Lazy initialization per process."""
        if self.env is None:
            # lock=False is critical for concurrent read-only access
            self.env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,
                readahead=False, 
                meminit=False,
                max_readers=256, # Bump this up for high num_workers
            )

    def load_sample(self, sample_id: str) -> Dict:
        self._init_env()
        with self.env.begin(write=False) as txn:
            data = txn.get(sample_id.encode("utf-8"))
            if data is None:
                raise KeyError(f"Sample {sample_id} not found in {self.path}")
            return pickle.loads(data)

    def get_sample_ids(self) -> List[str]:
        """
        Scan all keys. This is slow, usually only done once at init.
        """
        self._init_env()
        keys = []
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(values=False):
                keys.append(key.decode("utf-8"))
        return keys

    def close(self):
        if self.env:
            self.env.close()
            self.env = None
    
    def __getstate__(self):
        """Pickling support for Dataloader: Don't pickle the environment!"""
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
