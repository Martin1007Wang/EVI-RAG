from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch

from src.utils.logging_utils import get_logger, log_event

logger = get_logger(__name__)
_MIN_ENTITY_ROWS = 1


class GlobalEmbeddingStore:
    """In-memory read-only store for entity/relation embeddings."""

    def __init__(
        self,
        embeddings_dir: Union[str, Path],
        *,
        entity_vocab_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.entity_vocab_path = (
            None if entity_vocab_path is None else Path(entity_vocab_path)
        )
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"Unsupported embeddings_device={self.device}; expected cpu or cuda."
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "embeddings_device=cuda requested but CUDA is not available."
            )
        self.num_total_entities = (
            self._load_entity_vocab_size(self.entity_vocab_path)
            if self.entity_vocab_path is not None
            else 0
        )
        self.entity_embeddings = self._load_tensor(
            self.embeddings_dir / "entity_embeddings.pt"
        )
        self.relation_embeddings = self._load_tensor(
            self.embeddings_dir / "relation_embeddings.pt"
        )
        self._pinned_entity_buffer: Optional[torch.Tensor] = None
        self._pinned_relation_buffer: Optional[torch.Tensor] = None

        rows = self.entity_embeddings.size(0)
        if rows <= _MIN_ENTITY_ROWS:
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
        """Release pinned buffers used for async H2D transfers."""
        self._pinned_entity_buffer = None
        self._pinned_relation_buffer = None

    def _load_entity_vocab_size(self, path: Path) -> int:
        if not path.exists():
            raise FileNotFoundError(f"entity_vocab.parquet not found at {path}")
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyarrow is required to load entity_vocab.parquet."
            ) from exc
        try:
            table = pq.read_table(path, columns=["entity_id"])
        except Exception as exc:
            log_event(
                logger,
                "entity_vocab_read_failed",
                level=logging.WARNING,
                error=str(exc),
            )
            return 0
        if table.num_rows <= 0:
            return 0
        return int(table.num_rows)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Embedding file missing: {path}")
        log_event(
            logger,
            "embedding_load_start",
            path=str(path),
        )
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

    def get_entity_embeddings(
        self,
        entity_ids: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        target_device = (
            torch.device(device) if device is not None else entity_ids.device
        )
        target_dtype = self.entity_embeddings.dtype if dtype is None else dtype
        table_device = self.entity_embeddings.device
        if entity_ids.numel() == 0:
            return torch.empty(
                (0, int(self.entity_embeddings.size(1))),
                dtype=target_dtype,
                device=target_device,
            )
        if table_device.type == "cuda":
            ids = entity_ids.to(
                device=table_device, dtype=torch.long, non_blocking=True
            )
            out = self.entity_embeddings.index_select(0, ids)
            if target_device == table_device:
                if out.dtype != target_dtype:
                    return out.to(dtype=target_dtype)
                return out
            return out.to(device=target_device, dtype=target_dtype, non_blocking=True)
        cpu_ids = (
            entity_ids
            if entity_ids.device.type == "cpu"
            else entity_ids.to("cpu", non_blocking=True)
        )
        if cpu_ids.dtype != torch.long:
            cpu_ids = cpu_ids.to(dtype=torch.long)
        if target_device.type == "cpu":
            out = self.entity_embeddings.index_select(0, cpu_ids)
            if out.dtype != target_dtype:
                return out.to(dtype=target_dtype)
            return out
        if target_device.type == "cuda":
            num = int(cpu_ids.numel())
            dim = int(self.entity_embeddings.size(1))
            self._pinned_entity_buffer = self._ensure_pinned_buffer(
                buffer=self._pinned_entity_buffer,
                num=num,
                dim=dim,
                dtype=self.entity_embeddings.dtype,
            )
            assert self._pinned_entity_buffer is not None
            out_cpu = self._pinned_entity_buffer[:num]
            torch.index_select(self.entity_embeddings, 0, cpu_ids, out=out_cpu)
            return out_cpu.to(
                device=target_device, dtype=target_dtype, non_blocking=True
            )
        out = self.entity_embeddings.index_select(0, cpu_ids)
        return out.to(device=target_device, dtype=target_dtype)

    def get_relation_embeddings(
        self,
        relation_ids: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        target_device = (
            torch.device(device) if device is not None else relation_ids.device
        )
        target_dtype = self.relation_embeddings.dtype if dtype is None else dtype
        table_device = self.relation_embeddings.device
        if relation_ids.numel() == 0:
            return torch.empty(
                (0, int(self.relation_embeddings.size(1))),
                dtype=target_dtype,
                device=target_device,
            )
        if table_device.type == "cuda":
            ids = relation_ids.to(
                device=table_device, dtype=torch.long, non_blocking=True
            )
            out = self.relation_embeddings.index_select(0, ids)
            if target_device == table_device:
                if out.dtype != target_dtype:
                    return out.to(dtype=target_dtype)
                return out
            return out.to(device=target_device, dtype=target_dtype, non_blocking=True)
        cpu_ids = (
            relation_ids
            if relation_ids.device.type == "cpu"
            else relation_ids.to("cpu", non_blocking=True)
        )
        if cpu_ids.dtype != torch.long:
            cpu_ids = cpu_ids.to(dtype=torch.long)
        if target_device.type == "cpu":
            out = self.relation_embeddings.index_select(0, cpu_ids)
            if out.dtype != target_dtype:
                return out.to(dtype=target_dtype)
            return out
        if target_device.type == "cuda":
            num = int(cpu_ids.numel())
            dim = int(self.relation_embeddings.size(1))
            self._pinned_relation_buffer = self._ensure_pinned_buffer(
                buffer=self._pinned_relation_buffer,
                num=num,
                dim=dim,
                dtype=self.relation_embeddings.dtype,
            )
            assert self._pinned_relation_buffer is not None
            out_cpu = self._pinned_relation_buffer[:num]
            torch.index_select(self.relation_embeddings, 0, cpu_ids, out=out_cpu)
            return out_cpu.to(
                device=target_device, dtype=target_dtype, non_blocking=True
            )
        out = self.relation_embeddings.index_select(0, cpu_ids)
        return out.to(device=target_device, dtype=target_dtype)

    @property
    def entity_dim(self) -> int:
        return self.entity_embeddings.size(-1)

    @property
    def relation_dim(self) -> int:
        return self.relation_embeddings.size(-1)


def attach_embeddings_to_batch(
    batch: Any,
    *,
    global_embeddings: GlobalEmbeddingStore,
    embeddings_device: Optional[Union[str, torch.device]] = None,
    feature_dtype: Optional[torch.dtype] = None,
) -> None:
    node_embedding_ids = torch.as_tensor(
        batch.node_embedding_ids, dtype=torch.long, device="cpu"
    )
    relation_ids = torch.as_tensor(batch.edge_attr, dtype=torch.long, device="cpu")
    entity_rows = int(global_embeddings.entity_embeddings.size(0))
    rel_rows = int(global_embeddings.relation_embeddings.size(0))
    if node_embedding_ids.numel() > 0:
        if (
            node_embedding_ids.min().item() < 0
            or node_embedding_ids.max().item() >= entity_rows
        ):
            sample_id = getattr(batch, "sample_id", None)
            raise ValueError(
                f"node_embedding_ids out of range for batch (sample_id={sample_id}); "
                f"valid [0, {entity_rows - 1}]"
            )
    if relation_ids.numel() > 0:
        if relation_ids.min().item() < 0 or relation_ids.max().item() >= rel_rows:
            sample_id = getattr(batch, "sample_id", None)
            raise ValueError(
                f"edge_attr relation ids out of range for batch (sample_id={sample_id}); "
                f"valid [0, {rel_rows - 1}]"
            )
    batch.node_embeddings = global_embeddings.get_entity_embeddings(
        node_embedding_ids,
        device=embeddings_device,
        dtype=feature_dtype,
    )
    relation_ids_unique, edge_rel_local = torch.unique(
        relation_ids, sorted=True, return_inverse=True
    )
    batch.relation_embeddings = global_embeddings.get_relation_embeddings(
        relation_ids_unique,
        device=embeddings_device,
        dtype=feature_dtype,
    )
    batch.edge_rel_local = edge_rel_local
    batch.edge_embeddings = None
