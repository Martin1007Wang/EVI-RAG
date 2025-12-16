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
        self._pinned_entity_buffer: Optional[torch.Tensor] = None
        self._pinned_relation_buffer: Optional[torch.Tensor] = None

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

    def clear_device_cache(self) -> None:
        """Release transient pinned buffers used for async H2D transfers."""
        self._pinned_entity_buffer = None
        self._pinned_relation_buffer = None

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

    @staticmethod
    def _ensure_pinned_buffer(
        *,
        buffer: Optional[torch.Tensor],
        num: int,
        dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        needs_alloc = (
            buffer is None
            or buffer.device.type != "cpu"
            or not buffer.is_pinned()
            or buffer.dtype != dtype
            or buffer.dim() != 2
            or int(buffer.size(1)) != dim
            or int(buffer.size(0)) < num
        )
        if needs_alloc:
            return torch.empty((num, dim), dtype=dtype, device="cpu", pin_memory=True)
        return buffer

    def get_entity_embeddings(self, entity_ids: torch.Tensor, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """Fetch entity embeddings.

        The embedding tables live on CPU. `entity_ids` may be on CPU/GPU, while `device`
        controls the desired output device. Avoid moving ids to GPU when unnecessary.
        """
        target_device = device if device is not None else entity_ids.device
        cpu_ids = entity_ids if entity_ids.device.type == "cpu" else entity_ids.detach().to("cpu", non_blocking=True)
        if cpu_ids.dtype != torch.long:
            cpu_ids = cpu_ids.to(dtype=torch.long)
        if cpu_ids.numel() == 0:
            return torch.empty((0, int(self.entity_embeddings.size(1))), dtype=self.entity_embeddings.dtype, device=target_device)
        if target_device.type == "cpu":
            return self.entity_embeddings.index_select(0, cpu_ids)
        if target_device.type == "cuda":
            num = int(cpu_ids.numel())
            dim = int(self.entity_embeddings.size(1))
            self._pinned_entity_buffer = self._ensure_pinned_buffer(
                buffer=self._pinned_entity_buffer,
                num=num,
                dim=dim,
                dtype=self.entity_embeddings.dtype,
            )
            out_cpu = self._pinned_entity_buffer[:num]
            torch.index_select(self.entity_embeddings, 0, cpu_ids, out=out_cpu)
            return out_cpu.to(target_device, non_blocking=True)
        return self.entity_embeddings.index_select(0, cpu_ids).to(target_device)

    def get_relation_embeddings(self, relation_ids: torch.Tensor, *, device: Optional[torch.device] = None) -> torch.Tensor:
        target_device = device if device is not None else relation_ids.device
        cpu_ids = relation_ids if relation_ids.device.type == "cpu" else relation_ids.detach().to("cpu", non_blocking=True)
        if cpu_ids.dtype != torch.long:
            cpu_ids = cpu_ids.to(dtype=torch.long)
        if cpu_ids.numel() == 0:
            return torch.empty((0, int(self.relation_embeddings.size(1))), dtype=self.relation_embeddings.dtype, device=target_device)
        if target_device.type == "cpu":
            return self.relation_embeddings.index_select(0, cpu_ids)
        if target_device.type == "cuda":
            num = int(cpu_ids.numel())
            dim = int(self.relation_embeddings.size(1))
            self._pinned_relation_buffer = self._ensure_pinned_buffer(
                buffer=self._pinned_relation_buffer,
                num=num,
                dim=dim,
                dtype=self.relation_embeddings.dtype,
            )
            out_cpu = self._pinned_relation_buffer[:num]
            torch.index_select(self.relation_embeddings, 0, cpu_ids, out=out_cpu)
            return out_cpu.to(target_device, non_blocking=True)
        return self.relation_embeddings.index_select(0, cpu_ids).to(target_device)
    
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
