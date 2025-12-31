from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lightning import Callback, LightningModule, Trainer

from src.data.components.embedding_store import EmbeddingStore
from src.data.components.g_agent_builder import GAgentBuilder, GAgentSettings


_ZERO = 0
_DEFAULT_DATALOADER_IDX = 0


class GAgentMaterializationCallback(Callback):
    """Bridge Lightning predict outputs into GAgentBuilder materialization."""

    def __init__(
        self,
        settings: GAgentSettings,
        lmdb_path: str | Path | None = None,
        aux_lmdb_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.lmdb_path = Path(lmdb_path) if lmdb_path else None
        self.aux_lmdb_path = Path(aux_lmdb_path) if aux_lmdb_path else None
        self.builder: GAgentBuilder | None = None
        self.embedding_store: EmbeddingStore | None = None
        self.aux_embedding_store: EmbeddingStore | None = None

    def _start(self) -> None:
        """Resource Initialization."""
        if not self.settings.enabled:
            if self.embedding_store:
                self.embedding_store.close()
                self.embedding_store = None
            if self.aux_embedding_store:
                self.aux_embedding_store.close()
                self.aux_embedding_store = None
            self.builder = None
            return
        if self.lmdb_path and self.lmdb_path.exists():
            # rank_zero_only 并不是必须的，因为 EmbeddingStore 是只读且 lazy 的
            # 但为了保险，通常建议加上或确保 EmbeddingStore 支持并发
            self.embedding_store = EmbeddingStore(self.lmdb_path)
        if self.aux_lmdb_path and self.aux_lmdb_path.exists():
            self.aux_embedding_store = EmbeddingStore(self.aux_lmdb_path)

        self.builder = GAgentBuilder(
            self.settings,
            embedding_store=self.embedding_store,
            aux_embedding_store=self.aux_embedding_store,
        )
        self.builder.reset()

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._start()

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._start()

    def _process(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = _DEFAULT_DATALOADER_IDX,
    ) -> None:
        if self.builder is None:
            return

        model_output = outputs
        if isinstance(outputs, dict):
            model_output = SimpleNamespace(**outputs)
        self.builder.process_batch(batch, model_output)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = _DEFAULT_DATALOADER_IDX,
    ) -> None:
        self._process(outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = _DEFAULT_DATALOADER_IDX,
    ) -> None:
        self._process(outputs, batch, batch_idx, dataloader_idx)

    def _end(self, trainer: Trainer) -> None:
        """Cleanup and Save."""
        if self.builder:
            # 仅在主进程保存文件，避免 DDP 写冲突
            if trainer.global_rank == _ZERO:
                self.builder.save(self.settings.output_path)

        if self.embedding_store:
            self.embedding_store.close()
            self.embedding_store = None
        if self.aux_embedding_store:
            self.aux_embedding_store.close()
            self.aux_embedding_store = None

        # Explicitly clear builder to free memory
        self.builder = None

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._end(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._end(trainer)


__all__ = ["GAgentMaterializationCallback"]
