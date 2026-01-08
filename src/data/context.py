from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Tuple

from src.data.schema.constants import _MIN_CHUNK_SIZE
from src.data.schema.types import EmbeddingConfig, EntityVocab, RelationVocab, SplitFilter
from src.data.utils.config import build_embedding_cfg, build_split_filters, _resolve_parquet_chunk_size, _resolve_parquet_num_workers


@dataclass
class StageContext:
    cfg: object
    logger: object
    run_id: str
    base_dir: Optional[Path] = None

    _entity_vocab: Optional[EntityVocab] = None
    _relation_vocab: Optional[RelationVocab] = None

    def __post_init__(self) -> None:
        if self.base_dir is None:
            base = self.cfg.get("out_dir") or self.cfg.get("parquet_dir") or "."
            self.base_dir = self._to_abs_path(base)

    def _to_abs_path(self, value: str | Path) -> Path:
        try:
            import hydra
        except ModuleNotFoundError:  # pragma: no cover
            return Path(str(value)).expanduser().resolve()
        return Path(hydra.utils.to_absolute_path(str(value)))

    def resolve_path(self, value: str | Path) -> Path:
        return self._to_abs_path(value)

    @cached_property
    def raw_root(self) -> Optional[Path]:
        raw_root = self.cfg.get("raw_root")
        if not raw_root:
            return None
        return self._to_abs_path(raw_root)

    @cached_property
    def out_dir(self) -> Path:
        out_dir = self.cfg.get("out_dir")
        if not out_dir:
            raise ValueError("out_dir must be set in config.")
        return self._to_abs_path(out_dir)

    @cached_property
    def parquet_dir(self) -> Path:
        parquet_dir = self.cfg.get("parquet_dir") or self.cfg.get("out_dir")
        if not parquet_dir:
            raise ValueError("parquet_dir or out_dir must be set in config.")
        return self._to_abs_path(parquet_dir)

    @cached_property
    def output_dir(self) -> Path:
        output_dir = self.cfg.get("output_dir")
        if not output_dir:
            raise ValueError("output_dir must be set in config.")
        return self._to_abs_path(output_dir)

    @cached_property
    def embeddings_dir(self) -> Path:
        embeddings_out_dir = self.cfg.get("embeddings_out_dir")
        if embeddings_out_dir:
            return self._to_abs_path(embeddings_out_dir)
        return self.output_dir / "embeddings"

    @cached_property
    def dataset_name(self) -> str:
        return str(self.cfg.get("dataset_name") or self.cfg.get("dataset") or "dataset")

    @cached_property
    def embedding_cfg(self) -> Optional[EmbeddingConfig]:
        return build_embedding_cfg(self.cfg)

    @cached_property
    def split_filters(self) -> Tuple[SplitFilter, SplitFilter, dict[str, SplitFilter]]:
        return build_split_filters(self.cfg)

    @cached_property
    def parquet_chunk_size(self) -> int:
        return _resolve_parquet_chunk_size(self.cfg, fallback=int(self.cfg.get("batch_size", _MIN_CHUNK_SIZE)))

    @cached_property
    def parquet_num_workers(self) -> int:
        return _resolve_parquet_num_workers(self.cfg)

    @property
    def entity_vocab(self) -> EntityVocab:
        if self._entity_vocab is None:
            raise RuntimeError("EntityVocab is not initialized yet.")
        return self._entity_vocab

    @property
    def relation_vocab(self) -> RelationVocab:
        if self._relation_vocab is None:
            raise RuntimeError("RelationVocab is not initialized yet.")
        return self._relation_vocab

    def get_path(self, stage: str, filename: str) -> Path:
        return self.base_dir / stage / filename

    def ensure_stage_dir(self, stage: str) -> Path:
        path = self.base_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path
