import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import lmdb
import pickle
import torch

logger = logging.getLogger(__name__)


class GlobalEmbeddingStore:
    """Global entity and relation embedding store.

    New contract: emb.py provides embeddings for ALL entities in
    `embeddings/entity_embeddings.pt`. Downstream no longer needs to split
    between text/non-text entities here. We keep a simple, device-cached
    full matrix and index into it.
    """

    def __init__(self, embeddings_dir: Union[str, Path], vocabulary_path: Union[str, Path]):
        self.embeddings_dir = Path(embeddings_dir)
        self.vocabulary_path = Path(vocabulary_path)

        # Core embedding tensors
        self.entity_embeddings = None
        self.relation_embeddings = None

        # Vocabulary information
        self.vocabulary = None
        self.num_total_entities = 0
        self.embedding_dim = None

        self._load_vocabulary()
        self._load_embeddings()
        # self._build_complete_embeddings()

    def _load_vocabulary(self):
        """Load vocabulary and set total entity count."""
        # vocabulary_path already points to vocabulary.lmdb directory
        vocab_lmdb_path = self.vocabulary_path
        if not vocab_lmdb_path.exists():
            raise FileNotFoundError(f"Vocabulary LMDB not found: {vocab_lmdb_path}")

        try:
            env = lmdb.open(str(vocab_lmdb_path), readonly=True, lock=False)
            with env.begin() as txn:
                # Load entity mappings
                entity_to_id_data = txn.get(b"entity_to_id")
                id_to_entity_data = txn.get(b"id_to_entity")

                if entity_to_id_data is None or id_to_entity_data is None:
                    raise ValueError("Missing entity mappings in vocabulary LMDB")

                entity2id = pickle.loads(entity_to_id_data)
                _ = pickle.loads(id_to_entity_data)  # not used directly here

                self.num_total_entities = len(entity2id)

                logger.info(
                    f"Loaded vocabulary: {self.num_total_entities} total entities"
                )
            env.close()

        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            raise

    def _load_embeddings(self):
        """Load pre-computed embeddings for all entities and relations."""
        entity_path = self.embeddings_dir / "entity_embeddings.pt"
        relation_path = self.embeddings_dir / "relation_embeddings.pt"

        if not entity_path.exists():
            raise FileNotFoundError(f"Entity embeddings not found: {entity_path}")
        if not relation_path.exists():
            raise FileNotFoundError(f"Relation embeddings not found: {relation_path}")

        try:
            # Load full entity embeddings matrix [num_entities, D]
            self.entity_embeddings = torch.load(entity_path, map_location="cpu")
            self.relation_embeddings = torch.load(relation_path, map_location="cpu")

            # Validate dimensions
            if self.entity_embeddings.size(0) != self.num_total_entities:
                logger.warning(
                    f"Entity embedding count mismatch: "
                    f"expected {self.num_total_entities}, got {self.entity_embeddings.size(0)}"
                )

            self.embedding_dim = self.entity_embeddings.size(-1)

            logger.info("Loaded pre-computed embeddings:")
            logger.info(f"  Entities: {self.entity_embeddings.shape}")
            logger.info(f"  Relations: {self.relation_embeddings.shape}")
            logger.info(f"  Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def get_entity_embeddings(self, entity_ids: torch.LongTensor) -> torch.Tensor:
        """Get entity embeddings by global IDs with device-level caching.

        Args:
            entity_ids: Global entity IDs [N]

        Returns:
            Entity embeddings [N, D]
        """
        if self.entity_embeddings is None:
            raise RuntimeError("Entity embeddings not loaded")
        target_device = entity_ids.device
        ids = entity_ids.to(device="cpu", dtype=torch.long)
        selected = self.entity_embeddings.index_select(0, ids)
        return selected.to(target_device)

    def get_relation_embeddings(self, relation_ids: torch.LongTensor) -> torch.Tensor:
        """Get relation embeddings by global IDs - 优化版本使用设备缓存"""
        if self.relation_embeddings is None:
            raise RuntimeError("Relation embeddings not loaded")
        target_device = relation_ids.device
        ids = relation_ids.to(device="cpu", dtype=torch.long)
        selected = self.relation_embeddings.index_select(0, ids)
        return selected.to(target_device)

    def clear_device_cache(self):
        """保持兼容的空实现（单进程情况下无需缓存）。"""
        logger.info("GlobalEmbeddingStore cache cleared (noop)")

    @property
    def entity_dim(self) -> int:
        return self.embedding_dim

    @property
    def relation_dim(self) -> int:
        return self.relation_embeddings.size(-1)



class EmbeddingStore:
    """Sample data store, reads preprocessed sample data from LMDB"""

    def __init__(self, samples_lmdb_path: Union[str, Path], cache_size: int = 1000):
        self.samples_path = Path(samples_lmdb_path)
        self.cache_size = cache_size
        self.env: Optional[lmdb.Environment] = None
        self._env_pid: Optional[int] = None
        self._sample_cache: Dict[str, Dict] = {}

        if not self.samples_path.exists():
            raise FileNotFoundError(f"Sample LMDB not found: {self.samples_path}")

        logger.info(f"EmbeddingStore initialized: {self.samples_path}")

    def _ensure_env(self) -> None:
        """Make sure LMDB env is opened for the current process."""
        current_pid = os.getpid()
        if self.env is not None and self._env_pid == current_pid:
            return

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            finally:
                self.env = None
                self._env_pid = None

        try:
            self.env = lmdb.open(
                str(self.samples_path),
                readonly=True,
                lock=False,
                max_readers=128,
                map_size=1 << 40,
            )
            self._env_pid = current_pid
            logger.info(f"LMDB connection established (pid={current_pid}): {self.samples_path}")
        except Exception as e:
            logger.error(f"Failed to initialize LMDB: {e}")
            raise

    @contextmanager
    def _get_transaction(self):
        """Get LMDB transaction"""
        self._ensure_env()
        txn = self.env.begin()
        try:
            yield txn
        finally:
            txn.__exit__(None, None, None)

    def load_sample(self, sample_id: str) -> Dict:
        """Load sample data with caching"""
        if sample_id in self._sample_cache:
            return self._sample_cache[sample_id]

        with self._get_transaction() as txn:
            key = sample_id.encode("utf-8")
            data = txn.get(key)
            if data is None:
                raise KeyError(f"Sample {sample_id} not found")

            sample = pickle.loads(data)

            # Cache management
            if len(self._sample_cache) < self.cache_size:
                self._sample_cache[sample_id] = sample

            return sample

    def get_sample_ids(self) -> List[str]:
        """Get all sample IDs"""
        with self._get_transaction() as txn:
            return [key.decode("utf-8") for key in txn.cursor().iternext(values=False)]

    def close(self):
        """Close connection"""
        if self.env:
            self.env.close()
            self.env = None
            self._env_pid = None
