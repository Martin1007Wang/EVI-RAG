import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import lmdb
import torch

from src.data.io.lmdb_utils import _deserialize_sample
from src.utils.logging_utils import get_logger, log_event

logger = get_logger(__name__)
_LMDB_STAT_ENTRIES_KEY = "entries"

class GlobalEmbeddingStore:
    """
    In-memory read-only store for Entity/Relation embeddings.
    Designed for fast lookups during training/inference.
    """
    def __init__(
        self,
        embeddings_dir: Union[str, Path],
        vocabulary_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported embeddings_device={self.device}; expected cpu or cuda.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("embeddings_device=cuda requested but CUDA is not available.")
        # 1. Vocabulary (Optional, only if needed for count checks)
        self.num_total_entities = self._load_vocab_size(Path(vocabulary_path))
        
        # 2. Pre-load Embeddings (CPU Pinned Memory for speed)
        self.entity_embeddings = self._load_tensor(self.embeddings_dir / "entity_embeddings.pt")
        self.relation_embeddings = self._load_tensor(self.embeddings_dir / "relation_embeddings.pt")
        self._pinned_entity_buffer: Optional[torch.Tensor] = None
        self._pinned_relation_buffer: Optional[torch.Tensor] = None

        # Dimension note: embedding rows correspond to textual entities only.
        # Non-text entities map to embedding_id=0 and are re-embedded from
        # adjacent relation embeddings in the dataloader. Warn only if empty.
        rows = self.entity_embeddings.size(0)
        if rows <= 1:
            log_event(
                logger,
                "entity_embedding_rows_low",
                level=logging.WARNING,
                rows=rows,
            )
        else:
            log_event(
                logger,
                "entity_embedding_rows",
                rows=rows,
                vocab_entities=self.num_total_entities,
            )

    def clear_device_cache(self) -> None:
        """Release transient pinned buffers used for async H2D transfers."""
        self._pinned_entity_buffer = None
        self._pinned_relation_buffer = None

    def _load_vocab_size(self, path: Path) -> int:
        if not path.exists():
            raise FileNotFoundError(f"Vocab LMDB not found at {path}")
        try:
            # Quick open just to check size
            with lmdb.open(str(path), readonly=True, lock=False, max_readers=1) as env:
                with env.begin() as txn:
                    data = txn.get(b"entity_labels")
                    if data:
                        labels = json.loads(data.decode("utf-8"))
                        return len(labels)
            return 0
        except Exception as e:
            log_event(
                logger,
                "vocab_size_read_failed",
                level=logging.WARNING,
                error=str(e),
            )
            return 0

    def _load_tensor(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Embedding file missing: {path}")
        log_event(
            logger,
            "embedding_load_start",
            path=str(path),
        )
        # map_location='cpu' is crucial to avoid VRAM OOM
        try:
            tensor = torch.load(path, map_location="cpu", mmap=True)
        except TypeError:
            tensor = torch.load(path, map_location="cpu")
        except RuntimeError:
            tensor = torch.load(path, map_location="cpu")
        if self.device.type == "cuda":
            tensor = tensor.to(device=self.device, non_blocking=False)
        return tensor

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
        target_device = torch.device(device) if device is not None else entity_ids.device
        table_device = self.entity_embeddings.device
        if entity_ids.numel() == 0:
            return torch.empty(
                (0, int(self.entity_embeddings.size(1))),
                dtype=self.entity_embeddings.dtype,
                device=target_device,
            )
        if table_device.type == "cuda":
            ids = entity_ids.to(device=table_device, dtype=torch.long, non_blocking=True)
            out = self.entity_embeddings.index_select(0, ids)
            if target_device == table_device:
                return out
            return out.to(target_device, non_blocking=True)
        cpu_ids = entity_ids if entity_ids.device.type == "cpu" else entity_ids.detach().to("cpu", non_blocking=True)
        if cpu_ids.dtype != torch.long:
            cpu_ids = cpu_ids.to(dtype=torch.long)
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
        target_device = torch.device(device) if device is not None else relation_ids.device
        table_device = self.relation_embeddings.device
        if relation_ids.numel() == 0:
            return torch.empty(
                (0, int(self.relation_embeddings.size(1))),
                dtype=self.relation_embeddings.dtype,
                device=target_device,
            )
        if table_device.type == "cuda":
            ids = relation_ids.to(device=table_device, dtype=torch.long, non_blocking=True)
            out = self.relation_embeddings.index_select(0, ids)
            if target_device == table_device:
                return out
            return out.to(target_device, non_blocking=True)
        cpu_ids = relation_ids if relation_ids.device.type == "cpu" else relation_ids.detach().to("cpu", non_blocking=True)
        if cpu_ids.dtype != torch.long:
            cpu_ids = cpu_ids.to(dtype=torch.long)
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
    def __init__(self, lmdb_path: Union[str, Path], *, readahead: bool = False):
        self.path = str(lmdb_path)
        self._readahead = bool(readahead)
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
                readahead=self._readahead,
                meminit=False,
                max_readers=256, # Bump this up for high num_workers
            )

    def load_sample(self, sample_id: str) -> Dict:
        self._init_env()
        with self.env.begin(write=False) as txn:
            data = txn.get(sample_id.encode("utf-8"))
            if data is None:
                raise KeyError(f"Sample {sample_id} not found in {self.path}")
            return _deserialize_sample(data)

    def load_samples(self, sample_ids: List[str]) -> List[Dict]:
        self._init_env()
        if not sample_ids:
            return []
        with self.env.begin(write=False) as txn:
            out: List[Dict] = []
            for sample_id in sample_ids:
                data = txn.get(sample_id.encode("utf-8"))
                if data is None:
                    raise KeyError(f"Sample {sample_id} not found in {self.path}")
                out.append(_deserialize_sample(data))
            return out

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

    def get_entry_count(self) -> int:
        """Fast entry count from LMDB stats (no key scan)."""
        self._init_env()
        stats = self.env.stat()
        if _LMDB_STAT_ENTRIES_KEY not in stats:
            raise RuntimeError("LMDB stats missing entry count.")
        return int(stats[_LMDB_STAT_ENTRIES_KEY])

    def close(self):
        if self.env:
            self.env.close()
            self.env = None
    
    def __getstate__(self):
        """Serialization support for Dataloader: don't serialize the environment."""
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
