from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from lightning import LightningModule

from .batch import DualFlowBatchMixin
from .beam import DualFlowBeamMixin
from .config import DualFlowConfigMixin
from .constants import (
    _DEFAULT_AVOID_REVISIT,
    _DEFAULT_BACKBONE_FINETUNE,
    _DEFAULT_GNN_DROPOUT,
    _DEFAULT_GNN_LAYERS,
    _DEFAULT_VALIDATE_EDGE_BATCH,
    _ZERO,
)
from .eval import DualFlowEvalMixin
from .init import DualFlowInitMixin
from .metrics import DualFlowMetricsMixin
from .rollout import DualFlowRolloutMixin
from .runtime import DualFlowRuntimeMixin
from .scheduler import DualFlowSchedulerMixin
from .steps import DualFlowStepsMixin


class DualFlowModule(
    DualFlowBatchMixin,
    DualFlowBeamMixin,
    DualFlowConfigMixin,
    DualFlowEvalMixin,
    DualFlowInitMixin,
    DualFlowMetricsMixin,
    DualFlowRolloutMixin,
    DualFlowRuntimeMixin,
    DualFlowSchedulerMixin,
    DualFlowStepsMixin,
    LightningModule,
):
    """Off-policy detailed balance with student rollouts and backward-policy evaluation."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_steps: int,
        emb_dim: int,
        backbone_finetune: bool = _DEFAULT_BACKBONE_FINETUNE,
        gnn_layers: int = _DEFAULT_GNN_LAYERS,
        gnn_dropout: float = _DEFAULT_GNN_DROPOUT,
        cvt_init_cfg: Optional[Mapping[str, Any]] = None,
        embedding_adapter_cfg: Optional[Mapping[str, Any]] = None,
        actor_cfg: Optional[Mapping[str, Any]] = None,
        training_cfg: Mapping[str, Any] = None,
        evaluation_cfg: Mapping[str, Any] = None,
        runtime_cfg: Optional[Mapping[str, Any]] = None,
        optimizer_cfg: Optional[Mapping[str, Any]] = None,
        scheduler_cfg: Optional[Mapping[str, Any]] = None,
        logging_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if training_cfg is None or evaluation_cfg is None:
            raise ValueError("training_cfg and evaluation_cfg are required.")
        self.automatic_optimization = False
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        if self.max_steps <= _ZERO:
            raise ValueError("max_steps must be > 0.")

        self.training_cfg = training_cfg or {}
        self.evaluation_cfg = evaluation_cfg or {}
        self.cvt_init_cfg = cvt_init_cfg or {}
        self.embedding_adapter_cfg = embedding_adapter_cfg or {}
        self.actor_cfg = actor_cfg or {}
        self.runtime_cfg = runtime_cfg or {}
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.logging_cfg = logging_cfg or {}
        self._onecycle_checked = False

        self._validate_edge_batch = bool(self.runtime_cfg.get("validate_edge_batch", _DEFAULT_VALIDATE_EDGE_BATCH))
        self._avoid_revisit = bool(self.runtime_cfg.get("avoid_revisit", _DEFAULT_AVOID_REVISIT))

        self._init_backbone(
            emb_dim=emb_dim,
            finetune=backbone_finetune,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
        )
        self._init_cvt_init()
        self._init_actor()
        self._pb_mode = self._resolve_pb_mode()
        if self._is_static_pb():
            self._freeze_pb_modules()
        self._validate_cfg_contract()
        self._save_serializable_hparams()

        self._cvt_mask = None
        self._relation_inverse_map = None
        self._relation_inverse_mask = None
        self._relation_vocab_size = None


__all__ = ["DualFlowModule"]
