from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import Any, Optional

import torch
from lightning import LightningModule

from src.metrics.common import extract_sample_ids
from src.models.components import (
    CvtNodeInitializer,
    EmbeddingBackbone,
    LogZPredictor,
    QCBiANetwork,
    SinusoidalPositionalEncoding,
)
from src.models.components.gflownet_ops import (
    OutgoingEdges,
    build_edge_head_csr_from_mask,
    build_edge_tail_csr_from_mask,
    gather_outgoing_edges,
    gumbel_noise_like,
    segment_logsumexp_1d,
    segment_max,
)
from src.utils import log_metric, setup_optimizer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_NEG_ONE = -1
_ZERO = 0
_ONE = 1
_TWO = 2
_THREE = 3
_SELF_RELATION_ID = -1
_INVALID_EDGE_ID = -1

_DEFAULT_INVERSE_REL_SUFFIX = "__inv"
_DEFAULT_STRICT_INVERSE = True


_TERMINAL_NONE = 0
_TERMINAL_HIT = 1
_TERMINAL_DEAD_END = 2
_TERMINAL_MAX_STEPS = 3
_TERMINAL_INVALID_START = 4

_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_VALIDATE_EDGE_BATCH = False
_DEFAULT_GNN_LAYERS = 2
_DEFAULT_GNN_DROPOUT = 0.0
_DEFAULT_EDGE_INTER_DIM = 256
_DEFAULT_EDGE_DROPOUT = 0.1

_DEFAULT_METRIC_MODE = "minimal"
_METRIC_MODES = {"minimal", "full"}

_DB_SAMPLING_TEMPERATURE_SCHEDULES = {"constant", "cosine"}
_DEFAULT_TRAIN_ROLLOUTS = 1
_DB_CFG_KEYS = {
    "sampling_temperature",
    "sampling_temperature_start",
    "sampling_temperature_end",
    "sampling_temperature_schedule",
    "dead_end_log_reward",
    "dead_end_weight",
    "pb_mode",
    "pb_edge_dropout",
    "pb_semantic_weight",
    "pb_topo_penalty",
    "pb_cosine_eps",
    "pb_max_hops",
}

_PB_MODE_LEARNED = "learned"
_PB_MODE_TOPO_SEMANTIC = "topo_semantic"
_PB_MODE_UNIFORM = "uniform"
_PB_MODES = {_PB_MODE_LEARNED, _PB_MODE_TOPO_SEMANTIC, _PB_MODE_UNIFORM}

_SCHED_INTERVAL_EPOCH = "epoch"
_SCHED_INTERVAL_STEP = "step"
_SCHED_INTERVALS = {_SCHED_INTERVAL_EPOCH, _SCHED_INTERVAL_STEP}
_SCHED_TYPE_COSINE = "cosine"
_SCHED_TYPE_COSINE_WARM_RESTARTS = "cosine_warm_restarts"
_DEFAULT_SCHED_T_MAX = 10
_DEFAULT_SCHED_T0 = 10
_DEFAULT_SCHED_T_MULT = 1
_DEFAULT_SCHED_ETA_MIN = 0.0


@dataclass(frozen=True)
class _PreparedBatch:
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor
    question_emb_raw: torch.Tensor
    edge_embeddings_raw: torch.Tensor
    node_embeddings: torch.Tensor
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    context_tokens: torch.Tensor
    node_batch: torch.Tensor
    q_local_indices: torch.Tensor
    a_local_indices: torch.Tensor
    q_ptr: torch.Tensor
    a_ptr: torch.Tensor
    dummy_mask: torch.Tensor
    node_global_ids: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor
    sample_ids: list[str]
    start_nodes_fwd: torch.Tensor
    start_tokens_fwd: torch.Tensor
    start_tokens_bwd: torch.Tensor
    edge_ids_by_head_fwd: torch.Tensor
    edge_ptr_by_head_fwd: torch.Tensor
    edge_ids_by_tail_fwd: torch.Tensor
    edge_ptr_by_tail_fwd: torch.Tensor
    edge_ids_by_head_bwd: torch.Tensor
    edge_ptr_by_head_bwd: torch.Tensor
    edge_ids_by_tail_bwd: torch.Tensor
    edge_ptr_by_tail_bwd: torch.Tensor
    edge_inverse_map: torch.Tensor


@dataclass(frozen=True)
class _RolloutResult:
    log_pf_sum: torch.Tensor
    stop_nodes: torch.Tensor
    num_moves: torch.Tensor
    stop_reason: torch.Tensor
    actions: Optional[torch.Tensor]
    log_pf_steps: Optional[torch.Tensor]


class DualFlowModule(LightningModule):
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

        self._validate_edge_batch = bool(self.runtime_cfg.get("validate_edge_batch", _DEFAULT_VALIDATE_EDGE_BATCH))

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

        self._cvt_mask: Optional[torch.Tensor] = None
        self._relation_inverse_map: Optional[torch.Tensor] = None
        self._relation_inverse_mask: Optional[torch.Tensor] = None
        self._relation_vocab_size: Optional[int] = None

    # ------------------------- Init -------------------------

    def _init_backbone(
        self,
        *,
        emb_dim: int,
        finetune: bool,
        gnn_layers: int,
        gnn_dropout: float,
    ) -> None:
        self.backbone_fwd = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            finetune=finetune,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
            adapter_cfg=self.embedding_adapter_cfg,
        )
        self.backbone_bwd = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            finetune=finetune,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
            adapter_cfg=self.embedding_adapter_cfg,
        )

    def _init_cvt_init(self) -> None:
        enabled = bool(self.cvt_init_cfg.get("enabled", True))
        self._cvt_enabled = enabled
        self.cvt_init_fwd = CvtNodeInitializer()
        self.cvt_init_bwd = CvtNodeInitializer()

    def _init_actor(self) -> None:
        actor_cfg = self._resolve_actor_cfg()
        self.policy_fwd = QCBiANetwork(
            d_plm=self.hidden_dim,
            d_kg=self.hidden_dim,
            d_inter=actor_cfg["edge_inter_dim"],
            dropout=actor_cfg["edge_dropout"],
        )
        self.policy_bwd = QCBiANetwork(
            d_plm=self.hidden_dim,
            d_kg=self.hidden_dim,
            d_inter=actor_cfg["edge_inter_dim"],
            dropout=actor_cfg["edge_dropout"],
        )
        self.forward_ctx_proj = self._build_context_mlp(in_dim=self.hidden_dim * _TWO)
        self.backward_ctx_proj = self._build_context_mlp(in_dim=self.hidden_dim * _THREE)
        self.start_selector = self._build_start_selector()
        self.z_time_encoder = SinusoidalPositionalEncoding(self.hidden_dim)
        self.z_predictor = LogZPredictor(hidden_dim=self.hidden_dim, context_dim=self.hidden_dim)

    def _freeze_pb_modules(self) -> None:
        for module in (self.backbone_bwd, self.policy_bwd, self.backward_ctx_proj):
            for param in module.parameters():
                param.requires_grad = False

    def _resolve_pb_mode(self) -> str:
        cfg = self._resolve_db_cfg()
        mode = str(cfg["pb_mode"]).strip().lower()
        if mode not in _PB_MODES:
            raise ValueError(f"db_cfg.pb_mode must be one of {sorted(_PB_MODES)}, got {mode!r}.")
        return mode

    def _build_context_mlp(self, *, in_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _build_start_selector(self) -> torch.nn.Module:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * _TWO, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, _ONE),
        )
        self._zero_init_linear(mlp[_NEG_ONE])
        return mlp

    @staticmethod
    def _zero_init_linear(layer: torch.nn.Linear) -> None:
        torch.nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    def _validate_cfg_contract(self) -> None:
        allowed_training = {
            "accumulate_grad_batches",
            "metrics",
            "db_cfg",
            "num_rollouts",
        }
        extra_training = set(self.training_cfg.keys()) - allowed_training
        if extra_training:
            raise ValueError(f"Unsupported training_cfg keys: {sorted(extra_training)}")
        allowed_eval = {"beam_size", "beam_sizes"}
        extra_eval = set(self.evaluation_cfg.keys()) - allowed_eval
        if extra_eval:
            raise ValueError(f"Unsupported evaluation_cfg keys: {sorted(extra_eval)}")

    def _save_serializable_hparams(self) -> None:
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "backbone_fwd",
                "backbone_bwd",
                "cvt_init_fwd",
                "cvt_init_bwd",
                "policy_fwd",
                "policy_bwd",
                "forward_ctx_proj",
                "backward_ctx_proj",
                "z_time_encoder",
                "z_predictor",
                "training_cfg",
                "evaluation_cfg",
                "cvt_init_cfg",
                "embedding_adapter_cfg",
                "actor_cfg",
                "runtime_cfg",
                "optimizer_cfg",
                "scheduler_cfg",
                "logging_cfg",
            ],
        )

    # ------------------------- Runtime assets -------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        _ = stage
        self._ensure_runtime_initialized()

    def _ensure_runtime_initialized(self) -> None:
        if self._cvt_mask is not None and self._relation_inverse_map is not None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("datamodule is required to initialize CVT assets.")
        resources = getattr(datamodule, "shared_resources", None)
        if resources is None:
            raise RuntimeError("datamodule.shared_resources is required to initialize CVT assets.")
        self._cvt_mask = resources.cvt_mask
        inverse_suffix = self._resolve_inverse_suffix()
        inverse_map, inverse_mask = resources.relation_inverse_assets(suffix=inverse_suffix)
        self._relation_inverse_map = inverse_map
        self._relation_inverse_mask = inverse_mask
        self._relation_vocab_size = int(inverse_map.numel())

    # ------------------------- Optim -------------------------

    def configure_optimizers(self):
        optimizer = setup_optimizer(self, self.optimizer_cfg)
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[dict[str, Any]]:
        sched_type = str(self.scheduler_cfg.get("type", "") or "").strip().lower()
        if not sched_type:
            return None
        interval = str(self.scheduler_cfg.get("interval", _SCHED_INTERVAL_EPOCH) or _SCHED_INTERVAL_EPOCH).strip().lower()
        if interval not in _SCHED_INTERVALS:
            raise ValueError(f"scheduler_cfg.interval must be one of {sorted(_SCHED_INTERVALS)}, got {interval!r}.")
        if sched_type == _SCHED_TYPE_COSINE:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", _DEFAULT_SCHED_T_MAX)),
                eta_min=float(self.scheduler_cfg.get("eta_min", _DEFAULT_SCHED_ETA_MIN)),
            )
        elif sched_type in {"cosine_restart", "cosine_warm_restarts", "cosine_restarts", _SCHED_TYPE_COSINE_WARM_RESTARTS}:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.scheduler_cfg.get("t_0", _DEFAULT_SCHED_T0)),
                T_mult=int(self.scheduler_cfg.get("t_mult", _DEFAULT_SCHED_T_MULT)),
                eta_min=float(self.scheduler_cfg.get("eta_min", _DEFAULT_SCHED_ETA_MIN)),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")
        return {"scheduler": scheduler, "interval": interval}

    def _step_scheduler(self) -> None:
        sched = self.lr_schedulers()
        if sched is None:
            return
        schedulers = sched if isinstance(sched, list) else [sched]
        for scheduler in schedulers:
            self.lr_scheduler_step(scheduler, None)

    def on_train_epoch_end(self) -> None:
        interval = str(self.scheduler_cfg.get("interval", _SCHED_INTERVAL_EPOCH) or _SCHED_INTERVAL_EPOCH).strip().lower()
        if interval == _SCHED_INTERVAL_EPOCH:
            self._step_scheduler()

    def on_train_epoch_start(self) -> None:
        return

    # ------------------------- Lightning hooks -------------------------

    def forward(self, batch: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("DualFlowModule.forward is not supported; use training_step/eval.")

    def training_step(self, batch: Any, batch_idx: int):
        self._ensure_runtime_initialized()
        optimizer = self.optimizers()
        accum = float(self._accumulate_grad_batches())
        if self._should_zero_grad(batch_idx):
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._compute_training_loss(batch)
        if not torch.isfinite(loss).all().item():
            raise ValueError("Non-finite loss detected.")
        metrics = self._select_training_metrics(metrics)
        self.manual_backward(loss / accum)
        if self._should_step_optimizer(batch_idx):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            interval = str(self.scheduler_cfg.get("interval", _SCHED_INTERVAL_EPOCH) or _SCHED_INTERVAL_EPOCH).strip().lower()
            if interval == _SCHED_INTERVAL_STEP:
                self._step_scheduler()
        batch_size = int(getattr(batch, "num_graphs", int(getattr(batch, "ptr").numel() - 1)))
        for name, value in metrics.items():
            log_metric(self, f"train/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
        log_metric(self, "train/loss", loss.detach(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._ensure_runtime_initialized()
        _ = batch_idx
        metrics, batch_size = self._compute_eval_metrics(batch)
        if batch_size <= _ZERO:
            return
        scope = self._resolve_dataset_scope()
        for name, value in metrics.items():
            scoped_name = f"val/{scope}/{name}"
            log_metric(self, scoped_name, value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            if name.startswith(("pass@", "hit@", "recall@", "precision@", "f1@")):
                log_metric(self, f"val/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._ensure_runtime_initialized()
        _ = batch_idx
        metrics, batch_size = self._compute_eval_metrics(batch)
        if batch_size <= _ZERO:
            return
        scope = self._resolve_dataset_scope()
        for name, value in metrics.items():
            scoped_name = f"test/{scope}/{name}"
            log_metric(self, scoped_name, value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            if name.startswith(("pass@", "hit@", "recall@", "precision@", "f1@")):
                log_metric(self, f"test/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)

    # ------------------------- Grad helpers -------------------------

    def _accumulate_grad_batches(self) -> int:
        manual = self.training_cfg.get("accumulate_grad_batches", None)
        if manual is not None:
            return max(int(manual), _ONE)
        if self.trainer is None:
            return _ONE
        return max(int(getattr(self.trainer, "accumulate_grad_batches", _ONE) or _ONE), _ONE)

    def _is_last_train_batch(self, batch_idx: int) -> bool:
        if self.trainer is None:
            return False
        total = getattr(self.trainer, "num_training_batches", None)
        if total is None:
            return False
        return (batch_idx + _ONE) >= int(total)

    def _should_zero_grad(self, batch_idx: int) -> bool:
        accum = self._accumulate_grad_batches()
        if accum <= _ONE:
            return True
        return batch_idx % accum == _ZERO

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        accum = self._accumulate_grad_batches()
        if accum <= _ONE:
            return True
        if self._is_last_train_batch(batch_idx):
            return True
        return (batch_idx + _ONE) % accum == _ZERO

    # ------------------------- Batch prep -------------------------

    @staticmethod
    def _build_dummy_mask(*, answer_ptr: torch.Tensor) -> torch.Tensor:
        answer_counts = answer_ptr[1:] - answer_ptr[:-1]
        return (answer_counts == _ZERO).to(dtype=torch.bool)

    @staticmethod
    def _build_node_batch(*, node_ptr: torch.Tensor, device: torch.device) -> torch.Tensor:
        num_graphs = int(node_ptr.numel() - _ONE)
        if num_graphs <= _ZERO:
            return torch.zeros((_ZERO,), device=device, dtype=torch.long)
        counts = (node_ptr[_ONE:] - node_ptr[:-_ONE]).to(device=device, dtype=torch.long)
        return torch.repeat_interleave(torch.arange(num_graphs, device=device), counts)

    def _resolve_node_is_cvt(self, batch: Any, *, num_nodes_total: int, device: torch.device) -> torch.Tensor:
        if not self._cvt_enabled:
            return torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
        cvt_mask = self._cvt_mask
        if cvt_mask is None:
            return torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
        node_global_ids = getattr(batch, "node_global_ids", None)
        if not torch.is_tensor(node_global_ids):
            raise AttributeError("Batch missing node_global_ids required for CVT initialization.")
        node_global_ids = node_global_ids.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if node_global_ids.numel() != num_nodes_total:
            raise ValueError("node_global_ids length mismatch with ptr.")
        return cvt_mask.to(device=device, dtype=torch.bool).index_select(0, node_global_ids)

    @staticmethod
    def _build_node_mask(num_nodes_total: int, indices: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((num_nodes_total,), device=indices.device, dtype=torch.bool)
        if indices.numel() > 0:
            valid = indices >= _ZERO
            safe = indices[valid].to(dtype=torch.long)
            if safe.numel() > 0:
                mask[safe] = True
        return mask

    @staticmethod
    def _build_anchor_tokens(*, node_tokens: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        node_ids = node_ids.to(device=node_tokens.device, dtype=torch.long).view(-1)
        if node_ids.numel() == _ZERO:
            return torch.zeros((_ZERO, node_tokens.size(-1)), device=node_tokens.device, dtype=node_tokens.dtype)
        safe = node_ids.clamp(min=_ZERO)
        tokens = node_tokens.index_select(0, safe)
        valid = node_ids >= _ZERO
        return torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))

    @staticmethod
    def _pool_answer_tokens(
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        answer_indices: torch.Tensor,
        answer_ptr: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        num_graphs = int(num_graphs)
        if num_graphs <= _ZERO:
            return torch.zeros((_ZERO, node_tokens.size(-1)), device=node_tokens.device, dtype=node_tokens.dtype)
        question_tokens = question_tokens.to(device=node_tokens.device, dtype=node_tokens.dtype)
        answer_ptr = answer_ptr.to(device=node_tokens.device, dtype=torch.long).view(-1)
        if answer_ptr.numel() != num_graphs + _ONE:
            raise ValueError("answer_ptr length mismatch with num_graphs.")
        counts = (answer_ptr[_ONE:] - answer_ptr[:-_ONE]).clamp(min=_ZERO)
        if answer_indices.numel() == _ZERO:
            return torch.zeros((num_graphs, node_tokens.size(-1)), device=node_tokens.device, dtype=node_tokens.dtype)
        answer_indices = answer_indices.to(device=node_tokens.device, dtype=torch.long).view(-1)
        if answer_indices.numel() != int(counts.sum().detach().tolist()):
            raise ValueError("answer_indices length mismatch with answer_ptr.")
        if bool((answer_indices < _ZERO).any().detach().tolist()):
            raise ValueError("answer_indices contain negative values.")
        graph_ids = torch.repeat_interleave(
            torch.arange(num_graphs, device=node_tokens.device, dtype=torch.long),
            counts,
        )
        answer_tokens = node_tokens.index_select(0, answer_indices)
        query_tokens = question_tokens.index_select(0, graph_ids)
        scale = float(node_tokens.size(-1)) ** 0.5
        logits = (answer_tokens * query_tokens).sum(dim=-1) / scale
        log_denom = segment_logsumexp_1d(logits, graph_ids, num_graphs)
        weights = torch.exp(logits - log_denom.index_select(0, graph_ids))
        pooled = torch.zeros((num_graphs, node_tokens.size(-1)), device=node_tokens.device, dtype=node_tokens.dtype)
        pooled.index_add_(0, graph_ids, answer_tokens * weights.unsqueeze(-1))
        has_answer = counts > _ZERO
        return torch.where(has_answer.unsqueeze(-1), pooled, torch.zeros_like(pooled))

    @staticmethod
    def _edge_reorder_perm(
        *,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_relations: torch.Tensor,
        node_ptr: torch.Tensor,
        num_edges_before: int,
    ) -> torch.Tensor:
        if edge_index.numel() == _ZERO:
            return torch.zeros((_ZERO,), device=edge_index.device, dtype=torch.long)
        num_edges = int(edge_index.size(1))
        edge_batch = edge_batch.to(device=edge_index.device, dtype=torch.long).view(-1)
        if edge_batch.numel() != num_edges:
            raise ValueError("edge_batch length must match edge_index.")
        node_ptr = node_ptr.to(device=edge_index.device, dtype=torch.long).view(-1)
        if node_ptr.numel() <= _ONE:
            return torch.arange(num_edges, device=edge_index.device, dtype=torch.long)
        heads = edge_index[_ZERO].to(dtype=torch.long)
        tails = edge_index[_ONE].to(dtype=torch.long)
        is_self = (heads == tails) & (edge_relations == _SELF_RELATION_ID)
        node_offsets = node_ptr.index_select(0, edge_batch)
        local_heads = heads - node_offsets
        if bool((local_heads < _ZERO).any().detach().tolist()):
            raise ValueError("Found edge head outside graph node range.")
        num_nodes_total = int(node_ptr[-1].detach().tolist())
        stride = int(num_edges_before + num_nodes_total + _ONE)
        loop_order = num_edges_before + local_heads
        order_key = torch.where(is_self, loop_order, torch.arange(num_edges, device=edge_index.device, dtype=torch.long))
        sort_key = edge_batch * stride + order_key
        return torch.argsort(sort_key)

    @staticmethod
    def _resolve_context_tokens(context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.dim() == _TWO:
            return context_tokens
        if context_tokens.dim() == _THREE and context_tokens.size(1) == _ONE:
            return context_tokens.squeeze(1)
        raise ValueError("context_tokens must be [num_graphs, hidden_dim].")

    def _build_forward_context(
        self,
        *,
        question_tokens: torch.Tensor,
        start_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if question_tokens.size(0) != start_tokens.size(0):
            raise ValueError("question_tokens and start_tokens must align on batch dimension.")
        fused = torch.cat((question_tokens, start_tokens), dim=-1)
        return self.forward_ctx_proj(fused)

    def _build_backward_context(
        self,
        *,
        question_tokens: torch.Tensor,
        start_tokens: torch.Tensor,
        answer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if question_tokens.size(0) != start_tokens.size(0) or question_tokens.size(0) != answer_tokens.size(0):
            raise ValueError("question_tokens, start_tokens, and answer_tokens must align on batch dimension.")
        fused = torch.cat((question_tokens, start_tokens, answer_tokens), dim=-1)
        return self.backward_ctx_proj(fused)

    @staticmethod
    def _build_step_ids(*, num_graphs: int, step: int, device: torch.device) -> torch.Tensor:
        return torch.full((num_graphs,), int(step), device=device, dtype=torch.long)

    def _compute_log_z_for_nodes(
        self,
        *,
        node_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        node_batch: torch.Tensor,
        steps: torch.Tensor,
        node_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        context_tokens = self._resolve_context_tokens(context_tokens)
        if node_ids is None:
            node_tokens_sel = node_tokens
            node_batch_sel = node_batch
        else:
            node_ids = node_ids.to(device=node_tokens.device, dtype=torch.long).view(-1)
            node_tokens_sel = node_tokens.index_select(0, node_ids)
            node_batch_sel = node_batch.index_select(0, node_ids)
        steps = steps.to(device=node_tokens_sel.device, dtype=torch.long).view(-1)
        if steps.numel() <= int(node_batch_sel.max().detach().tolist()):
            raise ValueError("steps length must cover max node batch index.")
        time_emb = self.z_time_encoder(steps).index_select(0, node_batch_sel)
        node_tokens_sel = node_tokens_sel + time_emb
        return self.z_predictor(
            node_tokens=node_tokens_sel,
            question_tokens=context_tokens,
            node_batch=node_batch_sel,
        )

    def _compute_edge_logits(
        self,
        *,
        policy: QCBiANetwork,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if edge_ids.numel() == _ZERO:
            return torch.zeros((_ZERO,), device=edge_ids.device, dtype=torch.float32)
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        heads = prepared.edge_index[_ZERO].index_select(0, edge_ids)
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        head_tokens = prepared.node_tokens.index_select(0, heads)
        tail_tokens = prepared.node_tokens.index_select(0, tails)
        relation_tokens = prepared.relation_tokens.index_select(0, edge_ids)
        steps = steps.to(device=head_tokens.device, dtype=torch.long).view(-1)
        time_emb = self.z_time_encoder(steps).index_select(0, edge_batch)
        head_tokens = head_tokens + time_emb
        context_tokens = self._resolve_context_tokens(context_tokens)
        context_edge = context_tokens.index_select(0, edge_batch)
        logits = policy(context_edge, head_tokens, relation_tokens, tail_tokens, None)
        if temperature != float(_ONE):
            logits = logits / float(temperature)
        return logits

    @staticmethod
    def _cosine_similarity(x: torch.Tensor, y: torch.Tensor, *, eps: float) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        x_norm = x / x.norm(dim=-1, keepdim=True).clamp(min=eps)
        y_norm = y / y.norm(dim=-1, keepdim=True).clamp(min=eps)
        return (x_norm * y_norm).sum(dim=-1)

    def _compute_distance_to_starts(
        self,
        *,
        prepared: _PreparedBatch,
        max_hops: int,
    ) -> torch.Tensor:
        num_nodes_total = int(prepared.node_ptr[-1].detach().tolist())
        max_hops = int(max_hops)
        if num_nodes_total <= _ZERO:
            return torch.zeros((_ZERO,), device=prepared.edge_index.device, dtype=torch.long)
        distance_inf = max_hops + _ONE
        dist = torch.full((num_nodes_total,), distance_inf, device=prepared.edge_index.device, dtype=torch.long)
        start_nodes = prepared.q_local_indices.to(device=dist.device, dtype=torch.long).view(-1)
        if start_nodes.numel() == _ZERO:
            return dist
        valid = start_nodes >= _ZERO
        if not bool(valid.any().detach().tolist()):
            return dist
        start_nodes = start_nodes[valid]
        dist.index_fill_(0, start_nodes, int(_ZERO))
        frontier = torch.zeros((num_nodes_total,), device=dist.device, dtype=torch.bool)
        frontier.index_fill_(0, start_nodes, True)
        edge_ids = prepared.edge_ids_by_head_fwd
        if edge_ids.numel() == _ZERO or max_hops <= _ZERO:
            return dist
        heads = prepared.edge_index[_ZERO].index_select(0, edge_ids)
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        for step in range(max_hops):
            if not bool(frontier.any().detach().tolist()):
                break
            active = frontier.index_select(0, heads)
            if not bool(active.any().detach().tolist()):
                break
            candidate_tails = tails[active]
            if candidate_tails.numel() == _ZERO:
                break
            unseen = dist.index_select(0, candidate_tails) == distance_inf
            if not bool(unseen.any().detach().tolist()):
                break
            new_nodes = candidate_tails[unseen]
            dist.index_fill_(0, new_nodes, int(step + _ONE))
            frontier = torch.zeros_like(frontier)
            frontier.index_fill_(0, new_nodes, True)
        return dist

    def _compute_pb_logits(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        dist_to_start: Optional[torch.Tensor],
        pb_cfg: dict[str, float | int | str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=edge_ids.device, dtype=torch.long).view(-1)
        mode = str(pb_cfg["mode"])
        if mode == _PB_MODE_UNIFORM:
            logits = torch.zeros((edge_ids.numel(),), device=edge_ids.device, dtype=torch.float32)
            allowed = torch.ones_like(edge_ids, dtype=torch.bool)
            return logits, allowed
        if mode != _PB_MODE_TOPO_SEMANTIC:
            raise ValueError(f"Unsupported static pb mode: {mode!r}.")
        if dist_to_start is None:
            raise ValueError("dist_to_start is required for topo_semantic pb.")
        heads = prepared.edge_index[_ZERO].index_select(0, edge_ids)
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        dist_to_start = dist_to_start.to(device=edge_ids.device, dtype=torch.long)
        dist_heads = dist_to_start.index_select(0, heads)
        dist_tails = dist_to_start.index_select(0, tails)
        allowed = dist_tails < dist_heads
        topo_penalty = float(pb_cfg["topo_penalty"])
        topo_logits = torch.where(
            allowed,
            torch.zeros_like(dist_heads, dtype=torch.float32),
            torch.full_like(dist_heads, topo_penalty, dtype=torch.float32),
        )
        question_emb = self._resolve_context_tokens(prepared.question_emb_raw)
        query = question_emb.index_select(0, edge_batch)
        rel_emb = prepared.edge_embeddings_raw.index_select(0, edge_ids)
        cosine_eps = float(pb_cfg["cosine_eps"])
        sem = self._cosine_similarity(query, rel_emb, eps=cosine_eps)
        semantic_weight = float(pb_cfg["semantic_weight"])
        logits = topo_logits + sem.mul(semantic_weight)
        return logits, allowed

    def _compute_pb_log_prob(
        self,
        *,
        prepared: _PreparedBatch,
        dist_to_start: Optional[torch.Tensor],
        chosen_edge: torch.Tensor,
        parent_nodes: torch.Tensor,
        move_mask: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        pb_cfg: dict[str, float | int | str],
        edge_mask: Optional[torch.Tensor] = None,
        return_no_allowed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for pb log prob.")
        outgoing = gather_outgoing_edges(
            curr_nodes=parent_nodes,
            edge_ids_by_head=edge_ids_by_head,
            edge_ptr_by_head=edge_ptr_by_head,
            active_mask=move_mask,
        )
        if edge_mask is not None:
            outgoing = self._apply_edge_mask_to_outgoing(outgoing, edge_mask=edge_mask, num_graphs=move_mask.numel())
        if outgoing.edge_ids.numel() == _ZERO:
            zeros = torch.zeros_like(move_mask, dtype=torch.float32)
            if return_no_allowed:
                return zeros, move_mask.to(dtype=torch.bool)
            return zeros
        edge_ids = outgoing.edge_ids
        edge_batch = outgoing.edge_batch
        logits, allowed = self._compute_pb_logits(
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            dist_to_start=dist_to_start,
            pb_cfg=pb_cfg,
        )
        num_graphs = move_mask.numel()
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        chosen_edge_safe = chosen_edge.clamp(min=_ZERO)
        chosen_for_edge = chosen_edge_safe.index_select(0, edge_batch)
        match = edge_ids == chosen_for_edge
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(match, logits, torch.full_like(logits, neg_inf))
        chosen_logits, _ = segment_max(masked, edge_batch, num_graphs)
        log_pb_edge = chosen_logits - log_denom
        if bool(allowed.any().detach().tolist()):
            allowed_batch = edge_batch[allowed]
            allowed_counts = torch.bincount(allowed_batch, minlength=num_graphs)
        else:
            allowed_counts = torch.zeros((num_graphs,), device=edge_batch.device, dtype=torch.long)
        no_allowed = allowed_counts == _ZERO
        if bool(no_allowed.any().detach().tolist()):
            topo_penalty = float(pb_cfg["topo_penalty"])
            log_pb_edge = torch.where(no_allowed, torch.full_like(log_pb_edge, topo_penalty), log_pb_edge)
        log_pb_step = torch.where(move_mask, log_pb_edge, torch.zeros_like(log_pb_edge))
        if return_no_allowed:
            return log_pb_step, no_allowed
        return log_pb_step

    def _sample_pb_edges(
        self,
        *,
        prepared: _PreparedBatch,
        dist_to_start: Optional[torch.Tensor],
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        pb_cfg: dict[str, float | int | str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        if edge_ids.numel() == _ZERO:
            zeros = torch.zeros((num_graphs,), device=prepared.edge_index.device, dtype=torch.float32)
            return torch.full((num_graphs,), _NEG_ONE, device=prepared.edge_index.device, dtype=torch.long), zeros, zeros
        logits, allowed = self._compute_pb_logits(
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            dist_to_start=dist_to_start,
            pb_cfg=pb_cfg,
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        log_probs = logits - log_denom.index_select(0, edge_batch)
        scores = log_probs + gumbel_noise_like(log_probs)
        _, argmax = segment_max(scores, edge_batch, num_graphs)
        chosen_edge = edge_ids.index_select(0, argmax)
        log_prob_chosen = log_probs.index_select(0, argmax)
        if bool(allowed.any().detach().tolist()):
            allowed_batch = edge_batch[allowed]
            allowed_counts = torch.bincount(allowed_batch, minlength=num_graphs)
        else:
            allowed_counts = torch.zeros((num_graphs,), device=edge_batch.device, dtype=torch.long)
        has_allowed = allowed_counts > _ZERO
        return chosen_edge, log_prob_chosen, has_allowed

    def _rollout_pb(
        self,
        *,
        prepared: _PreparedBatch,
        dist_to_start: Optional[torch.Tensor],
        graph_mask: torch.Tensor,
        start_nodes: torch.Tensor,
        node_is_target: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        record_actions: bool,
        record_log_pf: bool,
        pb_cfg: dict[str, float | int | str],
        edge_mask: Optional[torch.Tensor] = None,
    ) -> _RolloutResult:
        num_graphs = int(prepared.node_ptr.numel() - _ONE)
        device = prepared.edge_index.device
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for pb rollout.")
        log_pf_sum = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        num_moves = torch.zeros((num_graphs,), device=device, dtype=torch.long)
        curr_nodes = start_nodes.clone()
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        stop_reason = torch.full((num_graphs,), _TERMINAL_NONE, device=device, dtype=torch.long)
        invalid_start = graph_mask & (curr_nodes < _ZERO)
        stop_reason = torch.where(
            invalid_start, torch.full_like(stop_reason, _TERMINAL_INVALID_START), stop_reason
        )
        active = graph_mask & (curr_nodes >= _ZERO)
        stop_nodes = torch.full((num_graphs,), _NEG_ONE, device=device, dtype=torch.long)
        actions = None
        log_pf_steps = None
        if record_actions:
            actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=device, dtype=torch.long)
        if record_log_pf:
            log_pf_steps = torch.zeros((num_graphs, self.max_steps), device=device, dtype=torch.float32)
        for step in range(int(self.max_steps)):
            at_target = node_is_target.index_select(0, curr_nodes.clamp(min=_ZERO)) & active
            stop_nodes = torch.where(at_target, curr_nodes, stop_nodes)
            stop_reason = torch.where(at_target, torch.full_like(stop_reason, _TERMINAL_HIT), stop_reason)
            active = active & ~at_target
            outgoing = gather_outgoing_edges(
                curr_nodes=curr_nodes,
                edge_ids_by_head=edge_ids_by_head,
                edge_ptr_by_head=edge_ptr_by_head,
                active_mask=active,
            )
            if edge_mask is not None:
                outgoing = self._apply_edge_mask_to_outgoing(outgoing, edge_mask=edge_mask, num_graphs=num_graphs)
            move_mask = active & outgoing.has_edge
            if outgoing.edge_ids.numel() > _ZERO:
                chosen_edge, log_pf_step, has_allowed = self._sample_pb_edges(
                    prepared=prepared,
                    dist_to_start=dist_to_start,
                    edge_ids=outgoing.edge_ids,
                    edge_batch=outgoing.edge_batch,
                    num_graphs=num_graphs,
                    pb_cfg=pb_cfg,
                )
                move_mask = move_mask & has_allowed
                chosen_edge = torch.where(move_mask, chosen_edge, torch.full_like(chosen_edge, _NEG_ONE))
                chosen_tail = prepared.edge_index[_ONE].index_select(0, chosen_edge.clamp(min=_ZERO))
                curr_nodes = torch.where(move_mask, chosen_tail, curr_nodes)
                log_pf_step = torch.where(move_mask, log_pf_step, torch.zeros_like(log_pf_step))
                log_pf_sum = log_pf_sum + log_pf_step
                num_moves = num_moves + move_mask.to(dtype=torch.long)
                if record_actions and actions is not None:
                    actions[:, step] = torch.where(move_mask, chosen_edge, actions[:, step])
                if record_log_pf and log_pf_steps is not None:
                    log_pf_steps[:, step] = log_pf_step
            no_edge = active & ~move_mask
            stop_nodes = torch.where(no_edge, curr_nodes, stop_nodes)
            stop_reason = torch.where(no_edge, torch.full_like(stop_reason, _TERMINAL_DEAD_END), stop_reason)
            active = active & move_mask
        stop_nodes = torch.where(
            stop_nodes >= _ZERO,
            stop_nodes,
            torch.where(active, curr_nodes, torch.full_like(curr_nodes, _NEG_ONE)),
        )
        stop_reason = torch.where(active, torch.full_like(stop_reason, _TERMINAL_MAX_STEPS), stop_reason)
        return _RolloutResult(
            log_pf_sum=log_pf_sum,
            stop_nodes=stop_nodes,
            num_moves=num_moves,
            stop_reason=stop_reason,
            actions=actions,
            log_pf_steps=log_pf_steps,
        )

    @staticmethod
    def _compute_log_denom(
        *,
        logits: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        return segment_logsumexp_1d(logits, edge_batch, num_graphs)

    def _select_start_nodes(
        self,
        *,
        question_tokens: torch.Tensor,
        node_tokens_fwd: torch.Tensor,
        node_tokens_bwd: torch.Tensor,
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        allow_empty: bool,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ptr = ptr.to(device=local_indices.device, dtype=torch.long).view(-1)
        local_indices = local_indices.to(device=ptr.device, dtype=torch.long).view(-1)
        counts = (ptr[_ONE:] - ptr[:-_ONE]).clamp(min=_ZERO)
        if not allow_empty and bool((counts <= _ZERO).any().detach().tolist()):
            raise ValueError(f"{name} missing in batch; filter data.")
        num_graphs = int(counts.numel())
        out = torch.full((num_graphs,), _NEG_ONE, device=local_indices.device, dtype=torch.long)
        hidden_dim = int(node_tokens_fwd.size(-1))
        if local_indices.numel() == _ZERO or num_graphs == _ZERO:
            zeros = torch.zeros((num_graphs, hidden_dim), device=node_tokens_fwd.device, dtype=node_tokens_fwd.dtype)
            return out, zeros, zeros
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=local_indices.device), counts)
        question_tokens = self._resolve_context_tokens(question_tokens)
        question_sel = question_tokens.index_select(0, graph_ids)
        node_sel_fwd = node_tokens_fwd.index_select(0, local_indices)
        node_sel_bwd = node_tokens_bwd.index_select(0, local_indices)
        logits = self.start_selector(torch.cat((question_sel, node_sel_fwd), dim=-1)).view(-1)
        log_denom = segment_logsumexp_1d(logits, graph_ids, num_graphs)
        soft_weights = torch.exp(logits - log_denom.index_select(0, graph_ids))
        noise = gumbel_noise_like(torch.zeros_like(logits, dtype=torch.float32))
        scores = logits + noise.to(dtype=logits.dtype)
        _, argmax = segment_max(scores, graph_ids, num_graphs)
        valid = counts > _ZERO
        hard_weights = torch.zeros_like(logits)
        argmax_valid = argmax[valid]
        if argmax_valid.numel() > _ZERO:
            hard_weights.index_put_((argmax_valid,), torch.ones_like(argmax_valid, dtype=logits.dtype))
        # Straight-through: hard selection forward, soft gradients backward.
        weights = hard_weights - soft_weights.detach() + soft_weights
        start_nodes = torch.where(valid, local_indices.index_select(0, argmax), out)
        start_tokens_fwd = torch.zeros((num_graphs, hidden_dim), device=node_sel_fwd.device, dtype=node_sel_fwd.dtype)
        start_tokens_fwd.index_add_(0, graph_ids, node_sel_fwd * weights.unsqueeze(-1))
        start_tokens_bwd = torch.zeros((num_graphs, hidden_dim), device=node_sel_bwd.device, dtype=node_sel_bwd.dtype)
        start_tokens_bwd.index_add_(0, graph_ids, node_sel_bwd * weights.unsqueeze(-1))
        return start_nodes, start_tokens_fwd, start_tokens_bwd

    @staticmethod
    def _sample_nodes_uniform(
        *,
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        allow_empty: bool,
        name: str,
    ) -> torch.Tensor:
        ptr = ptr.to(device=local_indices.device, dtype=torch.long).view(-1)
        local_indices = local_indices.to(device=ptr.device, dtype=torch.long).view(-1)
        counts = (ptr[_ONE:] - ptr[:-_ONE]).clamp(min=_ZERO)
        if not allow_empty and bool((counts <= _ZERO).any().detach().tolist()):
            raise ValueError(f"{name} missing in batch; filter data.")
        num_graphs = int(counts.numel())
        out = torch.full((num_graphs,), _NEG_ONE, device=local_indices.device, dtype=torch.long)
        if local_indices.numel() == _ZERO or num_graphs == _ZERO:
            return out
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=local_indices.device), counts)
        scores = gumbel_noise_like(torch.zeros_like(local_indices, dtype=torch.float32))
        _, argmax = segment_max(scores, graph_ids, num_graphs)
        valid = counts > _ZERO
        out = torch.where(valid, local_indices.index_select(0, argmax), out)
        return out

    def _extract_graph_tensors(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        node_ptr = getattr(batch, "ptr", None)
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)
        if not torch.is_tensor(node_ptr) or not torch.is_tensor(edge_index) or not torch.is_tensor(edge_attr):
            raise AttributeError("Batch missing ptr/edge_index/edge_attr required for DualFlow.")
        node_ptr = node_ptr.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_index = edge_index.to(device=device, dtype=torch.long, non_blocking=True)
        edge_relations = edge_attr.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_batch = getattr(batch, "edge_batch", None)
        edge_ptr = getattr(batch, "edge_ptr", None)
        if edge_batch is None or edge_ptr is None:
            raise AttributeError(
                "Batch missing edge_batch/edge_ptr; enable data.precompute_edge_batch to avoid per-step CPU builds."
            )
        edge_batch = torch.as_tensor(edge_batch, dtype=torch.long, device=device).view(-1)
        edge_ptr = torch.as_tensor(edge_ptr, dtype=torch.long, device=device).view(-1)
        return node_ptr, edge_index, edge_relations, edge_batch, edge_ptr

    def _extract_index_tensors(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        q_local_indices = getattr(batch, "q_local_indices", None)
        a_local_indices = getattr(batch, "a_local_indices", None)
        if not torch.is_tensor(q_local_indices) or not torch.is_tensor(a_local_indices):
            raise AttributeError("Batch missing q_local_indices/a_local_indices required for DualFlow.")
        q_local_indices = q_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        a_local_indices = a_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        slice_dict = getattr(batch, "_slice_dict")
        q_ptr = slice_dict["q_local_indices"].to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        a_ptr = slice_dict["a_local_indices"].to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
        if answer_ptr is None and hasattr(batch, "_slice_dict"):
            answer_ptr = batch._slice_dict.get("answer_entity_ids")
        if answer_ptr is None:
            raise AttributeError("Batch missing answer_entity_ids_ptr required for DualFlow.")
        answer_ptr = torch.as_tensor(answer_ptr, dtype=torch.long, device=device).view(-1)
        return q_local_indices, a_local_indices, q_ptr, a_ptr, answer_ptr

    def _extract_embeddings(
        self,
        batch: Any,
        *,
        edge_index: torch.Tensor,
        node_is_cvt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        question_emb = getattr(batch, "question_emb", None)
        node_embeddings = getattr(batch, "node_embeddings", None)
        edge_embeddings = getattr(batch, "edge_embeddings", None)
        if not torch.is_tensor(question_emb):
            raise AttributeError("Batch missing question_emb required for DualFlow.")
        if not torch.is_tensor(node_embeddings) or not torch.is_tensor(edge_embeddings):
            raise AttributeError("Batch missing node_embeddings/edge_embeddings required for DualFlow.")
        question_emb = question_emb.to(device=device, non_blocking=True)
        node_embeddings = node_embeddings.to(device=device, non_blocking=True)
        edge_embeddings = edge_embeddings.to(device=device, non_blocking=True)
        return question_emb, node_embeddings, edge_embeddings

    def _prepare_batch(self, batch: Any) -> tuple[_PreparedBatch, _PreparedBatch]:
        node_ptr, edge_index, edge_relations, edge_batch, edge_ptr = self._extract_graph_tensors(batch)
        q_local_indices, a_local_indices, q_ptr, a_ptr, answer_ptr = self._extract_index_tensors(batch)
        num_nodes_total = int(node_ptr[-1].detach().tolist()) if node_ptr.numel() > 0 else _ZERO
        dummy_mask = self._build_dummy_mask(answer_ptr=answer_ptr)
        node_batch = self._build_node_batch(node_ptr=node_ptr, device=self.device)
        node_is_cvt = self._resolve_node_is_cvt(batch, num_nodes_total=num_nodes_total, device=self.device)
        question_emb, node_embeddings, edge_embeddings = self._extract_embeddings(
            batch,
            edge_index=edge_index,
            node_is_cvt=node_is_cvt,
        )
        node_embeddings_fwd = node_embeddings
        node_embeddings_bwd = node_embeddings
        if self._cvt_enabled:
            node_embeddings_fwd = self.cvt_init_fwd(
                node_embeddings=node_embeddings_fwd,
                relation_embeddings=edge_embeddings,
                edge_index=edge_index,
                node_is_cvt=node_is_cvt,
            )
            node_embeddings_bwd = self.cvt_init_bwd(
                node_embeddings=node_embeddings_bwd,
                relation_embeddings=edge_embeddings,
                edge_index=edge_index,
                node_is_cvt=node_is_cvt,
            )
        if edge_embeddings.size(0) != edge_index.size(1):
            raise ValueError("edge_embeddings length must match edge_index.")
        heads = edge_index[_ZERO].to(dtype=torch.long)
        tails = edge_index[_ONE].to(dtype=torch.long)
        if bool((heads == tails).any().detach().tolist()):
            raise ValueError("Self-loop edges are not permitted; remove during preprocessing.")
        perm = self._edge_reorder_perm(
            edge_index=edge_index,
            edge_batch=edge_batch,
            edge_relations=edge_relations,
            node_ptr=node_ptr,
            num_edges_before=int(edge_index.size(1)),
        )
        edge_index = edge_index.index_select(1, perm)
        edge_batch = edge_batch.index_select(0, perm)
        edge_relations = edge_relations.index_select(0, perm)
        edge_embeddings = edge_embeddings.index_select(0, perm)
        node_tokens_fwd = self.backbone_fwd.project_node_embeddings(node_embeddings_fwd)
        node_tokens_bwd = self.backbone_bwd.project_node_embeddings(node_embeddings_bwd)
        relation_tokens_fwd = self.backbone_fwd.project_relation_embeddings(edge_embeddings)
        relation_tokens_bwd = self.backbone_bwd.project_relation_embeddings(edge_embeddings)
        node_tokens_fwd = self.backbone_fwd.encode_graph(
            node_tokens=node_tokens_fwd,
            relation_tokens=relation_tokens_fwd,
            edge_index=edge_index,
            num_nodes=num_nodes_total,
        )
        node_tokens_bwd = self.backbone_bwd.encode_graph(
            node_tokens=node_tokens_bwd,
            relation_tokens=relation_tokens_bwd,
            edge_index=edge_index,
            num_nodes=num_nodes_total,
        )
        num_graphs = int(node_ptr.numel() - _ONE)
        question_tokens_fwd_base = self._resolve_context_tokens(
            self.backbone_fwd.project_question_embeddings(question_emb)
        )
        question_tokens_bwd_base = self._resolve_context_tokens(
            self.backbone_bwd.project_question_embeddings(question_emb)
        )
        edge_counts = torch.bincount(edge_batch, minlength=num_graphs).to(device=self.device, dtype=torch.long)
        edge_ptr = torch.zeros((num_graphs + _ONE,), device=self.device, dtype=torch.long)
        edge_ptr[_ONE:] = edge_counts.cumsum(0)
        start_nodes_fwd, start_tokens_fwd, start_tokens_bwd = self._select_start_nodes(
            question_tokens=question_tokens_fwd_base,
            node_tokens_fwd=node_tokens_fwd,
            node_tokens_bwd=node_tokens_bwd,
            local_indices=q_local_indices,
            ptr=q_ptr,
            allow_empty=False,
            name="q_local_indices",
        )
        answer_tokens_bwd = self._pool_answer_tokens(
            node_tokens=node_tokens_bwd,
            question_tokens=question_tokens_bwd_base,
            answer_indices=a_local_indices,
            answer_ptr=a_ptr,
            num_graphs=num_graphs,
        )
        context_tokens_fwd = self._build_forward_context(
            question_tokens=question_tokens_fwd_base,
            start_tokens=start_tokens_fwd,
        )
        context_tokens_bwd = self._build_backward_context(
            question_tokens=question_tokens_bwd_base,
            start_tokens=start_tokens_bwd,
            answer_tokens=answer_tokens_bwd,
        )
        inverse_map = self._relation_inverse_map
        inverse_mask = self._relation_inverse_mask
        if inverse_map is None or inverse_mask is None:
            raise RuntimeError("relation inverse assets are required but not initialized.")
        inverse_map = inverse_map.to(device=edge_relations.device, dtype=torch.long)
        inverse_mask = inverse_mask.to(device=edge_relations.device, dtype=torch.bool)
        edge_is_inverse = self._build_edge_inverse_mask(edge_relations=edge_relations, inverse_mask=inverse_mask)
        self_loop_mask = edge_relations == _SELF_RELATION_ID
        edge_mask_fwd = self._build_edge_direction_mask(
            edge_is_inverse=edge_is_inverse, self_loop_mask=self_loop_mask, forward=True
        )
        edge_mask_bwd = self._build_edge_direction_mask(
            edge_is_inverse=edge_is_inverse, self_loop_mask=self_loop_mask, forward=False
        )
        edge_ids_by_head_fwd, edge_ptr_by_head_fwd = build_edge_head_csr_from_mask(
            edge_index=edge_index,
            edge_mask=edge_mask_fwd,
            num_nodes_total=num_nodes_total,
            device=self.device,
        )
        edge_ids_by_tail_fwd, edge_ptr_by_tail_fwd = build_edge_tail_csr_from_mask(
            edge_index=edge_index,
            edge_mask=edge_mask_fwd,
            num_nodes_total=num_nodes_total,
            device=self.device,
        )
        edge_ids_by_head_bwd, edge_ptr_by_head_bwd = build_edge_head_csr_from_mask(
            edge_index=edge_index,
            edge_mask=edge_mask_bwd,
            num_nodes_total=num_nodes_total,
            device=self.device,
        )
        edge_ids_by_tail_bwd, edge_ptr_by_tail_bwd = build_edge_tail_csr_from_mask(
            edge_index=edge_index,
            edge_mask=edge_mask_bwd,
            num_nodes_total=num_nodes_total,
            device=self.device,
        )
        edge_inverse_map = self._build_edge_inverse_map(
            edge_index=edge_index,
            edge_relations=edge_relations,
            num_nodes_total=num_nodes_total,
            inverse_map=inverse_map,
            num_relations=self._relation_vocab_size,
        )
        strict_inverse = self._resolve_strict_inverse()
        self._validate_edge_inverse_map(
            edge_inverse_map=edge_inverse_map,
            edge_relations=edge_relations,
            strict=strict_inverse,
        )
        sample_ids = extract_sample_ids(batch)
        if len(sample_ids) != num_graphs:
            raise ValueError("sample_id length mismatch with batch graph count.")
        node_global_ids = getattr(batch, "node_global_ids", None)
        if not torch.is_tensor(node_global_ids):
            raise AttributeError("Batch missing node_global_ids required for DualFlow.")
        node_global_ids = node_global_ids.to(device=self.device, dtype=torch.long, non_blocking=True).view(-1)
        answer_entity_ids = getattr(batch, "answer_entity_ids", None)
        if not torch.is_tensor(answer_entity_ids):
            raise AttributeError("Batch missing answer_entity_ids required for DualFlow.")
        answer_entity_ids = answer_entity_ids.to(device=self.device, dtype=torch.long, non_blocking=True).view(-1)
        prepared_fwd = _PreparedBatch(
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            question_emb_raw=question_emb,
            edge_embeddings_raw=edge_embeddings,
            node_embeddings=node_embeddings,
            node_tokens=node_tokens_fwd,
            relation_tokens=relation_tokens_fwd,
            context_tokens=context_tokens_fwd,
            node_batch=node_batch,
            q_local_indices=q_local_indices,
            a_local_indices=a_local_indices,
            q_ptr=q_ptr,
            a_ptr=a_ptr,
            dummy_mask=dummy_mask,
            node_global_ids=node_global_ids,
            answer_entity_ids=answer_entity_ids,
            answer_ptr=answer_ptr,
            sample_ids=sample_ids,
            start_nodes_fwd=start_nodes_fwd,
            start_tokens_fwd=start_tokens_fwd,
            start_tokens_bwd=start_tokens_bwd,
            edge_ids_by_head_fwd=edge_ids_by_head_fwd,
            edge_ptr_by_head_fwd=edge_ptr_by_head_fwd,
            edge_ids_by_tail_fwd=edge_ids_by_tail_fwd,
            edge_ptr_by_tail_fwd=edge_ptr_by_tail_fwd,
            edge_ids_by_head_bwd=edge_ids_by_head_bwd,
            edge_ptr_by_head_bwd=edge_ptr_by_head_bwd,
            edge_ids_by_tail_bwd=edge_ids_by_tail_bwd,
            edge_ptr_by_tail_bwd=edge_ptr_by_tail_bwd,
            edge_inverse_map=edge_inverse_map,
        )
        prepared_bwd = replace(
            prepared_fwd,
            node_tokens=node_tokens_bwd,
            relation_tokens=relation_tokens_bwd,
            context_tokens=context_tokens_bwd,
        )
        return prepared_fwd, prepared_bwd

    @staticmethod
    def _build_edge_inverse_mask(*, edge_relations: torch.Tensor, inverse_mask: torch.Tensor) -> torch.Tensor:
        edge_relations = edge_relations.to(device=inverse_mask.device, dtype=torch.long).view(-1)
        mask = torch.zeros_like(edge_relations, dtype=torch.bool)
        valid = edge_relations >= _ZERO
        if valid.any():
            mask[valid] = inverse_mask.index_select(0, edge_relations[valid]).to(dtype=torch.bool)
        return mask

    @staticmethod
    def _build_edge_direction_mask(
        *,
        edge_is_inverse: torch.Tensor,
        self_loop_mask: torch.Tensor,
        forward: bool,
    ) -> torch.Tensor:
        edge_is_inverse = edge_is_inverse.to(dtype=torch.bool)
        self_loop_mask = self_loop_mask.to(dtype=torch.bool)
        base = ~edge_is_inverse if forward else edge_is_inverse
        return base | self_loop_mask

    @staticmethod
    def _build_edge_inverse_map(
        *,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        num_nodes_total: int,
        inverse_map: torch.Tensor,
        num_relations: Optional[int],
    ) -> torch.Tensor:
        if num_relations is None:
            raise ValueError("num_relations is required to build edge inverse map.")
        num_edges = int(edge_relations.numel())
        if num_edges == _ZERO:
            return torch.zeros((_ZERO,), device=edge_relations.device, dtype=torch.long)
        num_nodes_total = int(num_nodes_total)
        num_relations = int(num_relations)
        if num_nodes_total <= _ZERO or num_relations <= _ZERO:
            return torch.full((num_edges,), _INVALID_EDGE_ID, device=edge_relations.device, dtype=torch.long)
        heads = edge_index[_ZERO].to(dtype=torch.long)
        tails = edge_index[_ONE].to(dtype=torch.long)
        rel = edge_relations.to(dtype=torch.long)
        inverse_edge = torch.full((num_edges,), _INVALID_EDGE_ID, device=edge_relations.device, dtype=torch.long)
        self_loop = (rel == _SELF_RELATION_ID) & (heads == tails)
        if bool(self_loop.any().detach().tolist()):
            self_idx = self_loop.nonzero(as_tuple=False).view(-1)
            inverse_edge[self_idx] = self_idx
        inv_rel = torch.full_like(rel, _INVALID_EDGE_ID)
        valid_rel = rel >= _ZERO
        if valid_rel.any():
            inv_rel[valid_rel] = inverse_map.index_select(0, rel[valid_rel])
        valid_inv = valid_rel & (inv_rel >= _ZERO)
        node_stride = int(num_nodes_total)
        rel_stride = int(num_relations)
        keys = ((heads * node_stride) + tails) * rel_stride + rel
        inv_keys = ((tails * node_stride) + heads) * rel_stride + inv_rel
        sorted_idx = torch.argsort(keys)
        sorted_keys = keys.index_select(0, sorted_idx)
        sorted_count = int(sorted_keys.numel())
        if sorted_count == _ZERO:
            return torch.full((num_edges,), _INVALID_EDGE_ID, device=edge_relations.device, dtype=torch.long)
        pos = torch.searchsorted(sorted_keys, inv_keys)
        pos_safe = pos.clamp(max=sorted_count - _ONE)
        match = (pos < sorted_count) & (sorted_keys.index_select(0, pos_safe) == inv_keys) & valid_inv
        if match.any():
            inverse_edge[match] = sorted_idx.index_select(0, pos_safe[match])
        return inverse_edge

    @staticmethod
    def _validate_edge_inverse_map(
        *,
        edge_inverse_map: torch.Tensor,
        edge_relations: torch.Tensor,
        strict: bool,
    ) -> None:
        if not strict:
            return
        if edge_inverse_map.numel() == _ZERO:
            return
        edge_inverse_map = edge_inverse_map.to(dtype=torch.long)
        edge_relations = edge_relations.to(device=edge_inverse_map.device, dtype=torch.long).view(-1)
        missing = (edge_relations >= _ZERO) & (edge_inverse_map < _ZERO)
        if bool(missing.any().detach().tolist()):
            raise ValueError(f"Missing inverse edges for {int(missing.sum().detach().tolist())} edges.")
        valid = edge_inverse_map >= _ZERO
        if not bool(valid.any().detach().tolist()):
            return
        inv_safe = edge_inverse_map[valid]
        idx = torch.arange(edge_inverse_map.numel(), device=edge_inverse_map.device, dtype=edge_inverse_map.dtype)[valid]
        back = edge_inverse_map.index_select(0, inv_safe)
        if not torch.equal(back, idx):
            raise ValueError("Edge inverse map is not symmetric.")

    # ------------------------- Rollouts -------------------------

    def _resolve_actor_cfg(self) -> dict[str, float | int]:
        raw = self.actor_cfg or {}
        extra = set(raw.keys()) - {"edge_inter_dim", "edge_dropout"}
        if extra:
            raise ValueError(f"Unsupported actor_cfg keys: {sorted(extra)}")
        edge_inter_dim = int(raw.get("edge_inter_dim", _DEFAULT_EDGE_INTER_DIM))
        edge_dropout = float(raw.get("edge_dropout", _DEFAULT_EDGE_DROPOUT))
        if edge_inter_dim <= _ZERO:
            raise ValueError("actor_cfg.edge_inter_dim must be > 0.")
        if edge_dropout < float(_ZERO):
            raise ValueError("actor_cfg.edge_dropout must be >= 0.")
        return {"edge_inter_dim": edge_inter_dim, "edge_dropout": edge_dropout}

    @staticmethod
    def _require_cfg_mapping(raw: Any, name: str) -> Mapping[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} must be a mapping.")
        return raw

    @staticmethod
    def _validate_cfg_keys(raw: Mapping[str, Any], *, required: set[str], name: str) -> None:
        missing = set(required) - set(raw.keys())
        if missing:
            raise ValueError(f"{name} missing keys: {sorted(missing)}")
        extra = set(raw.keys()) - set(required)
        if extra:
            raise ValueError(f"{name} has unsupported keys: {sorted(extra)}")

    @staticmethod
    def _coerce_db_cfg(raw: Mapping[str, Any]) -> dict[str, float | int | str]:
        return {
            "sampling_temperature": float(raw["sampling_temperature"]),
            "sampling_temperature_start": float(raw["sampling_temperature_start"]),
            "sampling_temperature_end": float(raw["sampling_temperature_end"]),
            "sampling_temperature_schedule": str(raw["sampling_temperature_schedule"]).strip().lower(),
            "dead_end_log_reward": float(raw["dead_end_log_reward"]),
            "dead_end_weight": float(raw["dead_end_weight"]),
            "pb_mode": str(raw["pb_mode"]).strip().lower(),
            "pb_edge_dropout": float(raw["pb_edge_dropout"]),
            "pb_semantic_weight": float(raw["pb_semantic_weight"]),
            "pb_topo_penalty": float(raw["pb_topo_penalty"]),
            "pb_cosine_eps": float(raw["pb_cosine_eps"]),
            "pb_max_hops": int(raw["pb_max_hops"]),
        }

    @staticmethod
    def _validate_db_cfg_values(cfg: Mapping[str, float | int | str]) -> None:
        schedule = str(cfg["sampling_temperature_schedule"])
        if schedule not in _DB_SAMPLING_TEMPERATURE_SCHEDULES:
            raise ValueError(
                "db_cfg.sampling_temperature_schedule must be one of "
                f"{sorted(_DB_SAMPLING_TEMPERATURE_SCHEDULES)}, got {schedule!r}."
            )
        if float(cfg["sampling_temperature"]) <= float(_ZERO):
            raise ValueError("db_cfg.sampling_temperature must be > 0.")
        if (
            float(cfg["sampling_temperature_start"]) <= float(_ZERO)
            or float(cfg["sampling_temperature_end"]) <= float(_ZERO)
        ):
            raise ValueError("db_cfg.sampling_temperature_start/end must be > 0.")
        if schedule == "cosine" and float(cfg["sampling_temperature_start"]) < float(cfg["sampling_temperature_end"]):
            raise ValueError("db_cfg.sampling_temperature_start must be >= sampling_temperature_end for cosine.")
        if float(cfg["pb_edge_dropout"]) < float(_ZERO) or float(cfg["pb_edge_dropout"]) >= float(_ONE):
            raise ValueError("db_cfg.pb_edge_dropout must satisfy 0 <= p < 1.")
        if str(cfg["pb_mode"]) not in _PB_MODES:
            raise ValueError(f"db_cfg.pb_mode must be one of {sorted(_PB_MODES)}, got {cfg['pb_mode']!r}.")
        if float(cfg["pb_semantic_weight"]) < float(_ZERO):
            raise ValueError("db_cfg.pb_semantic_weight must be >= 0.")
        if float(cfg["pb_topo_penalty"]) > float(_ZERO):
            raise ValueError("db_cfg.pb_topo_penalty must be <= 0.")
        if float(cfg["pb_cosine_eps"]) <= float(_ZERO):
            raise ValueError("db_cfg.pb_cosine_eps must be > 0.")
        if int(cfg["pb_max_hops"]) < int(_ZERO):
            raise ValueError("db_cfg.pb_max_hops must be >= 0.")
        if float(cfg["dead_end_weight"]) < float(_ZERO):
            raise ValueError("db_cfg.dead_end_weight must be >= 0.")

    def _resolve_db_cfg(self) -> dict[str, float | int | str]:
        raw = self._require_cfg_mapping(self.training_cfg.get("db_cfg"), "training_cfg.db_cfg")
        self._validate_cfg_keys(raw, required=_DB_CFG_KEYS, name="db_cfg")
        cfg = self._coerce_db_cfg(raw)
        self._validate_db_cfg_values(cfg)
        return cfg

    def _resolve_pb_cfg(self) -> dict[str, float | int | str]:
        cfg = self._resolve_db_cfg()
        max_hops = int(cfg["pb_max_hops"])
        if max_hops <= _ZERO:
            max_hops = self.max_steps
        return {
            "mode": str(cfg["pb_mode"]),
            "semantic_weight": float(cfg["pb_semantic_weight"]),
            "topo_penalty": float(cfg["pb_topo_penalty"]),
            "cosine_eps": float(cfg["pb_cosine_eps"]),
            "max_hops": max_hops,
        }

    def _is_static_pb(self) -> bool:
        return self._pb_mode in {_PB_MODE_TOPO_SEMANTIC, _PB_MODE_UNIFORM}

    def _resolve_sampling_temperature(self) -> float:
        cfg = self._resolve_db_cfg()
        schedule = str(cfg["sampling_temperature_schedule"])
        if schedule == "constant":
            return float(cfg["sampling_temperature"])
        start = float(cfg["sampling_temperature_start"])
        end = float(cfg["sampling_temperature_end"])
        progress = self._resolve_training_progress()
        half = float(_ONE) / float(_TWO)
        cosine = half * (float(_ONE) + math.cos(math.pi * progress))
        return end + (start - end) * cosine

    def _resolve_training_progress(self) -> float:
        trainer = self.trainer
        if trainer is None:
            return float(_ZERO)
        max_steps = getattr(trainer, "max_steps", None)
        if max_steps is None or int(max_steps) <= _ZERO or int(max_steps) == _NEG_ONE:
            max_steps = getattr(trainer, "estimated_stepping_batches", None)
        total_steps = int(max_steps) if max_steps is not None else _ZERO
        if total_steps <= _ZERO:
            return float(_ZERO)
        step = float(getattr(trainer, "global_step", self.global_step))
        progress = step / float(total_steps)
        return min(max(progress, float(_ZERO)), float(_ONE))

    def _sample_pb_edge_dropout_mask(self, *, prepared_bwd: _PreparedBatch) -> Optional[torch.Tensor]:
        drop_prob = float(self._resolve_db_cfg()["pb_edge_dropout"])
        if drop_prob <= float(_ZERO):
            return None
        num_edges = int(prepared_bwd.edge_index.size(1))
        if num_edges <= _ZERO:
            return None
        keep = torch.rand((num_edges,), device=prepared_bwd.edge_index.device) >= drop_prob
        return keep

    def _resolve_num_rollouts(self) -> int:
        raw = self.training_cfg.get("num_rollouts", _DEFAULT_TRAIN_ROLLOUTS)
        num_rollouts = int(raw)
        if num_rollouts <= _ZERO:
            raise ValueError("training_cfg.num_rollouts must be > 0.")
        return num_rollouts


    def _resolve_inverse_suffix(self) -> str:
        raw = self.runtime_cfg.get("inverse_relation_suffix") if isinstance(self.runtime_cfg, Mapping) else None
        suffix = str(raw or _DEFAULT_INVERSE_REL_SUFFIX).strip()
        if not suffix:
            raise ValueError("inverse_relation_suffix must be a non-empty string.")
        return suffix

    def _resolve_strict_inverse(self) -> bool:
        if isinstance(self.runtime_cfg, Mapping) and "strict_inverse" in self.runtime_cfg:
            return bool(self.runtime_cfg.get("strict_inverse"))
        return _DEFAULT_STRICT_INVERSE

    def _sample_edges(
        self,
        *,
        policy: QCBiANetwork,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        if edge_ids.numel() == _ZERO:
            zeros = torch.zeros((num_graphs,), device=prepared.edge_index.device, dtype=torch.float32)
            return torch.full((num_graphs,), _NEG_ONE, device=prepared.edge_index.device, dtype=torch.long), zeros, zeros
        logits = self._compute_edge_logits(
            policy=policy,
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            steps=steps + _ONE,
            temperature=temperature,
            context_tokens=context_tokens,
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        log_probs = logits - log_denom.index_select(0, edge_batch)
        scores = log_probs + gumbel_noise_like(log_probs)
        _, argmax = segment_max(scores, edge_batch, num_graphs)
        chosen_edge = edge_ids.index_select(0, argmax)
        log_prob_chosen = log_probs.index_select(0, argmax)
        return chosen_edge, log_prob_chosen, log_denom

    def _compute_forward_log_prob(
        self,
        *,
        policy: QCBiANetwork,
        prepared: _PreparedBatch,
        chosen_edge: torch.Tensor,
        parent_nodes: torch.Tensor,
        move_mask: torch.Tensor,
        steps: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for forward log prob.")
        outgoing = gather_outgoing_edges(
            curr_nodes=parent_nodes,
            edge_ids_by_head=edge_ids_by_head,
            edge_ptr_by_head=edge_ptr_by_head,
            active_mask=move_mask,
        )
        if edge_mask is not None:
            outgoing = self._apply_edge_mask_to_outgoing(outgoing, edge_mask=edge_mask, num_graphs=move_mask.numel())
        if outgoing.edge_ids.numel() == _ZERO:
            return torch.zeros_like(move_mask, dtype=torch.float32)
        edge_ids = outgoing.edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = outgoing.edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        logits = self._compute_edge_logits(
            policy=policy,
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            steps=steps + _ONE,
            temperature=temperature,
            context_tokens=context_tokens,
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=move_mask.numel())
        chosen_edge_safe = chosen_edge.clamp(min=_ZERO)
        chosen_for_edge = chosen_edge_safe.index_select(0, edge_batch)
        match = edge_ids == chosen_for_edge
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(match, logits, torch.full_like(logits, neg_inf))
        chosen_logits, _ = segment_max(masked, edge_batch, move_mask.numel())
        log_pf_edge = chosen_logits - log_denom
        log_pf_step = torch.where(move_mask, log_pf_edge, torch.zeros_like(log_pf_edge))
        return log_pf_step

    def _rollout_policy(
        self,
        *,
        policy: QCBiANetwork,
        prepared: _PreparedBatch,
        graph_mask: torch.Tensor,
        start_nodes: torch.Tensor,
        node_is_target: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        record_actions: bool,
        record_log_pf: bool,
        temperature: float,
        context_tokens: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> _RolloutResult:
        num_graphs = int(prepared.node_ptr.numel() - _ONE)
        device = prepared.edge_index.device
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for rollout policy.")
        log_pf_sum = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        num_moves = torch.zeros((num_graphs,), device=device, dtype=torch.long)
        curr_nodes = start_nodes.clone()
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        stop_reason = torch.full((num_graphs,), _TERMINAL_NONE, device=device, dtype=torch.long)
        invalid_start = graph_mask & (curr_nodes < _ZERO)
        stop_reason = torch.where(
            invalid_start, torch.full_like(stop_reason, _TERMINAL_INVALID_START), stop_reason
        )
        active = graph_mask & (curr_nodes >= _ZERO)
        stop_nodes = torch.full((num_graphs,), _NEG_ONE, device=device, dtype=torch.long)
        actions = None
        log_pf_steps = None
        if record_actions:
            actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=device, dtype=torch.long)
        if record_log_pf:
            log_pf_steps = torch.zeros((num_graphs, self.max_steps), device=device, dtype=torch.float32)
        for step in range(int(self.max_steps)):
            at_target = node_is_target.index_select(0, curr_nodes.clamp(min=_ZERO)) & active
            stop_nodes = torch.where(at_target, curr_nodes, stop_nodes)
            stop_reason = torch.where(at_target, torch.full_like(stop_reason, _TERMINAL_HIT), stop_reason)
            active = active & ~at_target
            outgoing = gather_outgoing_edges(
                curr_nodes=curr_nodes,
                edge_ids_by_head=edge_ids_by_head,
                edge_ptr_by_head=edge_ptr_by_head,
                active_mask=active,
            )
            if edge_mask is not None:
                outgoing = self._apply_edge_mask_to_outgoing(outgoing, edge_mask=edge_mask, num_graphs=num_graphs)
            move_mask = active & outgoing.has_edge
            if outgoing.edge_ids.numel() > _ZERO:
                step_ids = self._build_step_ids(num_graphs=num_graphs, step=step, device=device)
                chosen_edge, log_pf_step, _ = self._sample_edges(
                    policy=policy,
                    prepared=prepared,
                    edge_ids=outgoing.edge_ids,
                    edge_batch=outgoing.edge_batch,
                    num_graphs=num_graphs,
                    steps=step_ids,
                    temperature=temperature,
                    context_tokens=context_tokens,
                )
                chosen_edge = torch.where(outgoing.has_edge, chosen_edge, torch.full_like(chosen_edge, _NEG_ONE))
                chosen_tail = prepared.edge_index[_ONE].index_select(0, chosen_edge.clamp(min=_ZERO))
                curr_nodes = torch.where(move_mask, chosen_tail, curr_nodes)
                log_pf_step = torch.where(move_mask, log_pf_step, torch.zeros_like(log_pf_step))
                log_pf_sum = log_pf_sum + log_pf_step
                num_moves = num_moves + move_mask.to(dtype=torch.long)
                if record_actions and actions is not None:
                    actions[:, step] = torch.where(move_mask, chosen_edge, actions[:, step])
                if record_log_pf and log_pf_steps is not None:
                    log_pf_steps[:, step] = log_pf_step
            no_edge = active & ~outgoing.has_edge
            stop_nodes = torch.where(no_edge, curr_nodes, stop_nodes)
            stop_reason = torch.where(no_edge, torch.full_like(stop_reason, _TERMINAL_DEAD_END), stop_reason)
            active = active & outgoing.has_edge
        stop_nodes = torch.where(
            stop_nodes >= _ZERO,
            stop_nodes,
            torch.where(active, curr_nodes, torch.full_like(curr_nodes, _NEG_ONE)),
        )
        stop_reason = torch.where(active, torch.full_like(stop_reason, _TERMINAL_MAX_STEPS), stop_reason)
        return _RolloutResult(
            log_pf_sum=log_pf_sum,
            stop_nodes=stop_nodes,
            num_moves=num_moves,
            stop_reason=stop_reason,
            actions=actions,
            log_pf_steps=log_pf_steps,
        )

    @staticmethod
    def _apply_edge_mask_to_outgoing(
        outgoing: OutgoingEdges,
        *,
        edge_mask: torch.Tensor,
        num_graphs: int,
    ) -> OutgoingEdges:
        edge_ids = outgoing.edge_ids
        edge_batch = outgoing.edge_batch
        if edge_ids.numel() == _ZERO:
            return outgoing
        edge_mask = edge_mask.to(device=edge_ids.device, dtype=torch.bool).view(-1)
        if edge_mask.numel() == _ZERO:
            return outgoing
        keep = edge_mask.index_select(0, edge_ids)
        edge_ids = edge_ids[keep]
        edge_batch = edge_batch[keep]
        counts = torch.bincount(edge_batch, minlength=num_graphs).to(device=edge_ids.device, dtype=torch.long)
        has_edge = counts > _ZERO
        return OutgoingEdges(edge_ids=edge_ids, edge_batch=edge_batch, edge_counts=counts, has_edge=has_edge)

    def _compute_db_loss(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        prepared_bwd: _PreparedBatch,
        actions: torch.Tensor,
        graph_mask: torch.Tensor,
        traj_lengths: torch.Tensor,
        stop_reason: torch.Tensor,
        node_is_target: torch.Tensor,
        sampling_temperature: float,
        edge_mask_bwd: Optional[torch.Tensor] = None,
        pb_distances: Optional[torch.Tensor] = None,
        pb_cfg: Optional[dict[str, float | int | str]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = prepared_fwd.node_ptr.device
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        num_graphs, max_steps = actions.shape
        if max_steps == _ZERO:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return self._ensure_loss_requires_grad(zero), {"db_loss": zero.detach()}

        db_cfg = self._resolve_db_cfg()
        dead_end_log_reward = float(db_cfg["dead_end_log_reward"])
        dead_end_weight = float(db_cfg["dead_end_weight"])
        edge_mask = actions >= _ZERO
        failure_mask = (stop_reason != _TERMINAL_HIT) & graph_mask
        weight = torch.ones((num_graphs,), device=device, dtype=torch.float32)
        if dead_end_weight != float(_ONE):
            weight = torch.where(failure_mask, weight * dead_end_weight, weight)
        dist_to_start = None
        if pb_distances is not None:
            dist_to_start = pb_distances.to(device=device, dtype=torch.long)

        total = torch.zeros((), device=device, dtype=torch.float32)
        denom = torch.zeros((), device=device, dtype=torch.float32)
        valid_count = torch.zeros((), device=device, dtype=torch.float32)
        move_count = torch.zeros((), device=device, dtype=torch.float32)
        log_pb_sum = torch.zeros((), device=device, dtype=torch.float32)
        log_pb_min = torch.full((), float("inf"), device=device, dtype=torch.float32)
        log_z_u_sum = torch.zeros((), device=device, dtype=torch.float32)
        log_z_v_sum = torch.zeros((), device=device, dtype=torch.float32)
        inv_invalid_count = torch.zeros((), device=device, dtype=torch.float32)
        topo_violation_count = torch.zeros((), device=device, dtype=torch.float32)
        no_allowed_count = torch.zeros((), device=device, dtype=torch.float32)
        for step in range(max_steps):
            edge_ids = actions[:, step]
            move_mask = edge_mask[:, step] & graph_mask
            move_count = move_count + move_mask.to(dtype=torch.float32).sum()
            safe_edges = edge_ids.clamp(min=_ZERO)
            heads = prepared_fwd.edge_index[_ZERO].index_select(0, safe_edges)
            tails = prepared_fwd.edge_index[_ONE].index_select(0, safe_edges)
            step_ids = self._build_step_ids(num_graphs=num_graphs, step=step, device=device)
            next_step_ids = step_ids + _ONE
            log_z_u = self._compute_log_z_for_nodes(
                node_tokens=prepared_fwd.node_tokens,
                context_tokens=prepared_fwd.context_tokens,
                node_batch=prepared_fwd.node_batch,
                steps=step_ids,
                node_ids=heads,
            )
            log_z_v = self._compute_log_z_for_nodes(
                node_tokens=prepared_fwd.node_tokens,
                context_tokens=prepared_fwd.context_tokens,
                node_batch=prepared_fwd.node_batch,
                steps=next_step_ids,
                node_ids=tails,
            )
            log_pf = self._compute_forward_log_prob(
                policy=self.policy_fwd,
                prepared=prepared_fwd,
                chosen_edge=edge_ids,
                parent_nodes=heads,
                move_mask=move_mask,
                steps=step_ids,
                edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
                edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
                temperature=sampling_temperature,
                context_tokens=prepared_fwd.context_tokens,
            )
            inv_edge = prepared_fwd.edge_inverse_map.index_select(0, safe_edges)
            inv_valid = inv_edge >= _ZERO
            inv_edge = torch.where(inv_valid, inv_edge, torch.full_like(inv_edge, _NEG_ONE))
            active_bwd = move_mask & inv_valid
            if self._is_static_pb():
                if pb_cfg is None:
                    pb_cfg = self._resolve_pb_cfg()
                if pb_cfg["mode"] == _PB_MODE_TOPO_SEMANTIC and pb_distances is None:
                    raise ValueError("pb_distances required for topo_semantic pb DB loss.")
                log_pb, no_allowed = self._compute_pb_log_prob(
                    prepared=prepared_fwd,
                    dist_to_start=pb_distances,
                    chosen_edge=inv_edge,
                    parent_nodes=tails,
                    move_mask=active_bwd,
                    edge_ids_by_head=prepared_fwd.edge_ids_by_head_bwd,
                    edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_bwd,
                    pb_cfg=pb_cfg,
                    edge_mask=edge_mask_bwd,
                    return_no_allowed=True,
                )
                no_allowed_count = no_allowed_count + (no_allowed & active_bwd).to(dtype=torch.float32).sum()
            else:
                log_pb = self._compute_forward_log_prob(
                    policy=self.policy_bwd,
                    prepared=prepared_bwd,
                    chosen_edge=inv_edge,
                    parent_nodes=tails,
                    move_mask=active_bwd,
                    steps=next_step_ids,
                    edge_ids_by_head=prepared_bwd.edge_ids_by_head_bwd,
                    edge_ptr_by_head=prepared_bwd.edge_ptr_by_head_bwd,
                    temperature=float(_ONE),
                    context_tokens=prepared_bwd.context_tokens,
                    edge_mask=edge_mask_bwd,
                )
            is_target = node_is_target.index_select(0, tails.clamp(min=_ZERO)) & move_mask
            log_z_v = torch.where(is_target, torch.zeros_like(log_z_v), log_z_v)
            is_terminal = traj_lengths == (step + _ONE)
            dead_end = is_terminal & failure_mask
            log_z_v = torch.where(
                dead_end,
                torch.full_like(log_z_v, dead_end_log_reward),
                log_z_v,
            )
            inv_invalid_count = inv_invalid_count + (move_mask & ~inv_valid).to(dtype=torch.float32).sum()
            valid = move_mask & inv_valid
            valid_f = valid.to(dtype=torch.float32)
            valid_count = valid_count + valid_f.sum()
            log_pb_sum = log_pb_sum + (log_pb * valid_f).sum()
            log_z_u_sum = log_z_u_sum + (log_z_u * valid_f).sum()
            log_z_v_sum = log_z_v_sum + (log_z_v * valid_f).sum()
            pb_for_min = torch.where(valid, log_pb, torch.full_like(log_pb, float("inf")))
            log_pb_min = torch.minimum(log_pb_min, pb_for_min.min())
            if dist_to_start is not None:
                inv_edge_safe = inv_edge.clamp(min=_ZERO)
                inv_heads = prepared_fwd.edge_index[_ZERO].index_select(0, inv_edge_safe)
                inv_tails = prepared_fwd.edge_index[_ONE].index_select(0, inv_edge_safe)
                dist_heads = dist_to_start.index_select(0, inv_heads)
                dist_tails = dist_to_start.index_select(0, inv_tails)
                allowed_inv = dist_tails < dist_heads
                topo_violation = valid & ~allowed_inv
                topo_violation_count = topo_violation_count + topo_violation.to(dtype=torch.float32).sum()
            delta = (log_z_u + log_pf) - (log_z_v + log_pb)
            delta = torch.where(valid, delta, torch.zeros_like(delta))
            step_weight = weight * valid.to(dtype=weight.dtype)
            total = total + (delta.pow(_TWO) * step_weight).sum()
            denom = denom + step_weight.sum()
        if float(denom.item()) <= float(_ZERO):
            zero = torch.zeros((), device=device, dtype=torch.float32)
            metrics = {
                "db_loss": zero.detach(),
                "db_log_pb_mean": zero.detach(),
                "db_log_pb_min": zero.detach(),
                "db_log_z_u_mean": zero.detach(),
                "db_log_z_v_mean": zero.detach(),
                "db_inv_edge_invalid_rate": zero.detach(),
                "db_no_allowed_rate": zero.detach(),
                "db_topo_violation_rate": zero.detach(),
            }
            return self._ensure_loss_requires_grad(zero), metrics
        loss = total / denom
        zero = torch.zeros((), device=device, dtype=torch.float32)
        valid_any = valid_count > _ZERO
        move_any = move_count > _ZERO
        valid_count_safe = torch.where(valid_any, valid_count, torch.ones_like(valid_count))
        move_count_safe = torch.where(move_any, move_count, torch.ones_like(move_count))
        log_pb_mean = torch.where(valid_any, log_pb_sum / valid_count_safe, zero)
        log_z_u_mean = torch.where(valid_any, log_z_u_sum / valid_count_safe, zero)
        log_z_v_mean = torch.where(valid_any, log_z_v_sum / valid_count_safe, zero)
        log_pb_min = torch.where(valid_any, log_pb_min, zero)
        inv_edge_invalid_rate = torch.where(move_any, inv_invalid_count / move_count_safe, zero)
        no_allowed_rate = torch.where(move_any, no_allowed_count / move_count_safe, zero)
        topo_violation_rate = torch.where(valid_any, topo_violation_count / valid_count_safe, zero)
        metrics = {
            "db_loss": loss.detach(),
            "db_log_pb_mean": log_pb_mean.detach(),
            "db_log_pb_min": log_pb_min.detach(),
            "db_log_z_u_mean": log_z_u_mean.detach(),
            "db_log_z_v_mean": log_z_v_mean.detach(),
            "db_inv_edge_invalid_rate": inv_edge_invalid_rate.detach(),
            "db_no_allowed_rate": no_allowed_rate.detach(),
            "db_topo_violation_rate": topo_violation_rate.detach(),
        }
        return self._ensure_loss_requires_grad(loss), metrics

    @staticmethod
    def _build_terminal_metrics(
        *,
        stop_reason: torch.Tensor,
        graph_mask: torch.Tensor,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        stop_reason = stop_reason.to(device=graph_mask.device, dtype=torch.long)
        graph_mask = graph_mask.to(device=stop_reason.device, dtype=torch.bool)
        denom = graph_mask.to(dtype=torch.float32).sum().clamp(min=_ONE)
        hit = ((stop_reason == _TERMINAL_HIT) & graph_mask).to(dtype=torch.float32).sum() / denom
        dead = ((stop_reason == _TERMINAL_DEAD_END) & graph_mask).to(dtype=torch.float32).sum() / denom
        max_steps = ((stop_reason == _TERMINAL_MAX_STEPS) & graph_mask).to(dtype=torch.float32).sum() / denom
        invalid = ((stop_reason == _TERMINAL_INVALID_START) & graph_mask).to(dtype=torch.float32).sum() / denom
        other = ((stop_reason == _TERMINAL_NONE) & graph_mask).to(dtype=torch.float32).sum() / denom
        return {
            f"{prefix}_terminal_hit_rate": hit,
            f"{prefix}_terminal_dead_end_rate": dead,
            f"{prefix}_terminal_max_steps_rate": max_steps,
            f"{prefix}_terminal_invalid_start_rate": invalid,
            f"{prefix}_terminal_other_rate": other,
        }

    # ------------------------- Detailed balance loss -------------------------

    @staticmethod
    def _validate_training_batch(prepared_fwd: _PreparedBatch) -> torch.Tensor:
        num_graphs = int(prepared_fwd.node_ptr.numel() - _ONE)
        if num_graphs <= _ZERO:
            raise ValueError("Empty batch.")
        graph_mask = ~prepared_fwd.dummy_mask
        if not bool(graph_mask.any().detach().tolist()):
            raise ValueError("Training batch contains no valid graphs.")
        return graph_mask

    def _run_training_rollout(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        prepared_bwd: _PreparedBatch,
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        sampling_temperature: float,
        pb_distances: Optional[torch.Tensor] = None,
        pb_cfg: Optional[dict[str, float | int | str]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            rollout_fwd = self._rollout_policy(
                policy=self.policy_fwd,
                prepared=prepared_fwd,
                graph_mask=graph_mask,
                start_nodes=prepared_fwd.start_nodes_fwd,
                node_is_target=node_is_target,
                edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
                edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
                record_actions=True,
                record_log_pf=False,
                temperature=sampling_temperature,
                context_tokens=prepared_fwd.context_tokens,
            )
        if rollout_fwd.actions is None:
            raise RuntimeError("Rollout actions are required for detailed balance training.")
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            prepared_bwd=prepared_bwd,
            actions=rollout_fwd.actions,
            graph_mask=graph_mask,
            traj_lengths=rollout_fwd.num_moves,
            stop_reason=rollout_fwd.stop_reason,
            node_is_target=node_is_target,
            sampling_temperature=sampling_temperature,
            pb_distances=pb_distances,
            pb_cfg=pb_cfg,
        )
        success = (rollout_fwd.stop_reason == _TERMINAL_HIT) & graph_mask
        metrics = {
            **db_metrics,
            "rollout_success_rate": success.to(dtype=torch.float32).mean(),
        }
        metrics.update(
            self._build_terminal_metrics(
                stop_reason=rollout_fwd.stop_reason,
                graph_mask=graph_mask,
                prefix="rollout",
            )
        )
        return db_loss, metrics

    def _run_backward_rollout(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        prepared_bwd: _PreparedBatch,
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        node_is_start: torch.Tensor,
        start_nodes_bwd: torch.Tensor,
        sampling_temperature: float,
        pb_distances: Optional[torch.Tensor] = None,
        pb_cfg: Optional[dict[str, float | int | str]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        edge_mask_bwd = self._sample_pb_edge_dropout_mask(prepared_bwd=prepared_bwd)
        with torch.no_grad():
            if self._is_static_pb():
                if pb_cfg is None:
                    pb_cfg = self._resolve_pb_cfg()
                if pb_cfg["mode"] == _PB_MODE_TOPO_SEMANTIC and pb_distances is None:
                    raise ValueError("pb_distances required for topo_semantic pb rollout.")
                rollout_bwd = self._rollout_pb(
                    prepared=prepared_fwd,
                    dist_to_start=pb_distances,
                    graph_mask=graph_mask,
                    start_nodes=start_nodes_bwd,
                    node_is_target=node_is_start,
                    edge_ids_by_head=prepared_fwd.edge_ids_by_head_bwd,
                    edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_bwd,
                    record_actions=True,
                    record_log_pf=False,
                    pb_cfg=pb_cfg,
                    edge_mask=edge_mask_bwd,
                )
            else:
                rollout_bwd = self._rollout_policy(
                    policy=self.policy_bwd,
                    prepared=prepared_bwd,
                    graph_mask=graph_mask,
                    start_nodes=start_nodes_bwd,
                    node_is_target=node_is_start,
                    edge_ids_by_head=prepared_bwd.edge_ids_by_head_bwd,
                    edge_ptr_by_head=prepared_bwd.edge_ptr_by_head_bwd,
                    record_actions=True,
                    record_log_pf=False,
                    temperature=float(_ONE),
                    context_tokens=prepared_bwd.context_tokens,
                    edge_mask=edge_mask_bwd,
                )
        if rollout_bwd.actions is None:
            raise RuntimeError("Backward rollout actions are required for detailed balance training.")
        actions_fwd = self._map_inverse_actions(
            actions=rollout_bwd.actions,
            edge_inverse_map=prepared_fwd.edge_inverse_map,
        )
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            prepared_bwd=prepared_bwd,
            actions=actions_fwd,
            graph_mask=graph_mask,
            traj_lengths=rollout_bwd.num_moves,
            stop_reason=rollout_bwd.stop_reason,
            node_is_target=node_is_target,
            sampling_temperature=sampling_temperature,
            edge_mask_bwd=edge_mask_bwd,
            pb_distances=pb_distances,
            pb_cfg=pb_cfg,
        )
        success = (rollout_bwd.stop_reason == _TERMINAL_HIT) & graph_mask
        metrics = {
            **db_metrics,
            "rollout_bwd_success_rate": success.to(dtype=torch.float32).mean(),
        }
        metrics.update(
            self._build_terminal_metrics(
                stop_reason=rollout_bwd.stop_reason,
                graph_mask=graph_mask,
                prefix="rollout_bwd",
            )
        )
        return db_loss, metrics

    def _aggregate_training_rollouts(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        prepared_bwd: _PreparedBatch,
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        node_is_start: torch.Tensor,
        start_nodes_bwd: torch.Tensor,
        sampling_temperature: float,
        num_rollouts: int,
        pb_distances: Optional[torch.Tensor] = None,
        pb_cfg: Optional[dict[str, float | int | str]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if num_rollouts <= _ZERO:
            raise ValueError("num_rollouts must be > 0.")
        losses: list[torch.Tensor] = []
        metric_series: dict[str, list[torch.Tensor]] = {}
        for _ in range(num_rollouts):
            db_loss_fwd, metrics_fwd = self._run_training_rollout(
                prepared_fwd=prepared_fwd,
                prepared_bwd=prepared_bwd,
                graph_mask=graph_mask,
                node_is_target=node_is_target,
                sampling_temperature=sampling_temperature,
                pb_distances=pb_distances,
                pb_cfg=pb_cfg,
            )
            db_loss_bwd, metrics_bwd = self._run_backward_rollout(
                prepared_fwd=prepared_fwd,
                prepared_bwd=prepared_bwd,
                graph_mask=graph_mask,
                node_is_target=node_is_target,
                node_is_start=node_is_start,
                start_nodes_bwd=start_nodes_bwd,
                sampling_temperature=sampling_temperature,
                pb_distances=pb_distances,
                pb_cfg=pb_cfg,
            )
            db_loss = (db_loss_fwd + db_loss_bwd) / float(_TWO)
            metrics = self._merge_rollout_metrics(
                metrics_fwd=metrics_fwd,
                metrics_bwd=metrics_bwd,
                db_loss_fwd=db_loss_fwd,
                db_loss_bwd=db_loss_bwd,
                db_loss=db_loss,
            )
            losses.append(db_loss)
            for name, value in metrics.items():
                metric_series.setdefault(name, []).append(value)
        loss = torch.stack(losses).mean()
        averaged = {name: torch.stack(values).mean() for name, values in metric_series.items()}
        averaged["loss_total"] = loss.detach()
        return loss, averaged

    @staticmethod
    def _map_inverse_actions(*, actions: torch.Tensor, edge_inverse_map: torch.Tensor) -> torch.Tensor:
        if actions.numel() == _ZERO:
            return actions
        edge_inverse_map = edge_inverse_map.to(device=actions.device, dtype=torch.long)
        actions = actions.to(device=edge_inverse_map.device, dtype=torch.long)
        safe = actions.clamp(min=_ZERO).view(-1)
        mapped = edge_inverse_map.index_select(0, safe).view_as(actions)
        invalid = (actions >= _ZERO) & (mapped < _ZERO)
        if bool(invalid.any().detach().tolist()):
            raise ValueError("Backward rollout sampled edges without forward inverse.")
        return torch.where(actions >= _ZERO, mapped, actions)

    @staticmethod
    def _merge_rollout_metrics(
        *,
        metrics_fwd: dict[str, torch.Tensor],
        metrics_bwd: dict[str, torch.Tensor],
        db_loss_fwd: torch.Tensor,
        db_loss_bwd: torch.Tensor,
        db_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        merged = dict(metrics_fwd)
        for name, value in metrics_bwd.items():
            if name in merged and name.startswith("db_"):
                merged[name] = (merged[name] + value) / float(_TWO)
            else:
                merged[name] = value
        merged.pop("db_loss", None)
        merged["db_loss_fwd"] = db_loss_fwd.detach()
        merged["db_loss_bwd"] = db_loss_bwd.detach()
        merged["db_loss"] = db_loss.detach()
        return merged

    def _apply_target_roulette(
        self,
        *,
        batch: Any,
        prepared: _PreparedBatch,
        target_nodes: torch.Tensor,
    ) -> _PreparedBatch:
        num_graphs = int(prepared.node_ptr.numel() - _ONE)
        if target_nodes.numel() != num_graphs:
            raise ValueError("target_nodes length mismatch with batch graph count.")
        question_emb = getattr(batch, "question_emb", None)
        if not torch.is_tensor(question_emb):
            raise AttributeError("Batch missing question_emb required for target roulette.")
        question_tokens = self._resolve_context_tokens(
            self.backbone_bwd.project_question_embeddings(
                question_emb.to(device=prepared.node_tokens.device, non_blocking=True)
            )
        )
        target_tokens = self._build_anchor_tokens(node_tokens=prepared.node_tokens, node_ids=target_nodes)
        context_tokens = self._build_backward_context(
            question_tokens=question_tokens,
            start_tokens=prepared.start_tokens_bwd,
            answer_tokens=target_tokens,
        )
        return replace(prepared, context_tokens=context_tokens)

    def _compute_training_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prepared_fwd, prepared_bwd = self._prepare_batch(batch)
        graph_mask = self._validate_training_batch(prepared_fwd)
        num_nodes_total = int(prepared_fwd.node_ptr[-1].detach().tolist())
        start_nodes_bwd = self._sample_nodes_uniform(
            local_indices=prepared_fwd.a_local_indices,
            ptr=prepared_fwd.a_ptr,
            allow_empty=True,
            name="a_local_indices",
        )
        node_is_target = self._build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        node_is_start = self._build_node_mask(num_nodes_total, prepared_fwd.q_local_indices)
        pb_cfg = None
        pb_distances = None
        if self._is_static_pb():
            pb_cfg = self._resolve_pb_cfg()
            if pb_cfg["mode"] == _PB_MODE_TOPO_SEMANTIC:
                with torch.no_grad():
                    pb_distances = self._compute_distance_to_starts(
                        prepared=prepared_fwd,
                        max_hops=int(pb_cfg["max_hops"]),
                    )
        else:
            prepared_bwd = self._apply_target_roulette(
                batch=batch,
                prepared=prepared_bwd,
                target_nodes=start_nodes_bwd,
            )
        sampling_temperature = self._resolve_sampling_temperature()
        num_rollouts = self._resolve_num_rollouts()
        return self._aggregate_training_rollouts(
            prepared_fwd=prepared_fwd,
            prepared_bwd=prepared_bwd,
            graph_mask=graph_mask,
            node_is_target=node_is_target,
            node_is_start=node_is_start,
            start_nodes_bwd=start_nodes_bwd,
            sampling_temperature=sampling_temperature,
            num_rollouts=num_rollouts,
            pb_distances=pb_distances,
            pb_cfg=pb_cfg,
        )

    @staticmethod
    def _ensure_loss_requires_grad(loss: torch.Tensor) -> torch.Tensor:
        if loss.requires_grad:
            return loss
        return loss + torch.zeros((), device=loss.device, dtype=loss.dtype, requires_grad=True)

    # ------------------------- Eval -------------------------

    def _resolve_dataset_scope(self) -> str:
        datamodule = getattr(self.trainer, "datamodule", None)
        cfg = getattr(datamodule, "dataset_cfg", None) if datamodule is not None else None
        scope = None
        if isinstance(cfg, Mapping):
            scope = cfg.get("dataset_scope")
        if not scope:
            return "unknown"
        return str(scope).strip().lower()

    def _resolve_beam_size_value(self) -> int:
        beam_size = int(self.evaluation_cfg.get("beam_size", _ONE))
        if beam_size <= _ZERO:
            raise ValueError("evaluation_cfg.beam_size must be > 0.")
        return beam_size

    def _resolve_beam_sizes(self) -> list[int]:
        raw = self.evaluation_cfg.get("beam_sizes", None)
        if raw is None:
            return [self._resolve_beam_size_value()]
        if isinstance(raw, Mapping):
            raise ValueError("evaluation_cfg.beam_sizes must be a list of positive integers.")
        if isinstance(raw, (list, tuple)):
            values = list(raw)
        elif isinstance(raw, str):
            values = [part for part in (chunk.strip() for chunk in raw.split(",")) if part]
        else:
            try:
                values = list(raw)
            except TypeError:
                values = [raw]
        if not values:
            values = [self._resolve_beam_size_value()]
        parsed: list[int] = []
        for value in values:
            beam_size = int(value)
            if beam_size <= _ZERO:
                raise ValueError("evaluation_cfg.beam_sizes entries must be > 0.")
            parsed.append(beam_size)
        return sorted(set(parsed))

    def _resolve_beam_size(self) -> int:
        beam_sizes = self._resolve_beam_sizes()
        return max(beam_sizes)

    def _beam_search(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
    ) -> list[list[tuple[int, float, list[int]]]]:
        num_graphs = int(prepared.node_ptr.numel() - _ONE)
        if num_graphs <= _ZERO:
            return []
        beam_size = int(beam_size)
        if beam_size <= _ZERO:
            return []
        device = prepared.node_ptr.device
        max_steps = int(self.max_steps)
        neg_inf = torch.finfo(torch.float32).min

        start_nodes = prepared.start_nodes_fwd.to(device=device, dtype=torch.long)
        beam_nodes = torch.full((num_graphs, beam_size), _NEG_ONE, device=device, dtype=torch.long)
        beam_scores = torch.full((num_graphs, beam_size), neg_inf, device=device, dtype=torch.float32)
        beam_paths = torch.full((num_graphs, beam_size, max_steps), _NEG_ONE, device=device, dtype=torch.long)
        beam_lengths = torch.zeros((num_graphs, beam_size), device=device, dtype=torch.long)
        valid_start = start_nodes >= _ZERO
        beam_nodes[:, 0] = start_nodes
        beam_scores[:, 0] = torch.where(valid_start, torch.zeros_like(beam_scores[:, 0]), beam_scores[:, 0])
        start_target = node_is_target.index_select(0, start_nodes.clamp(min=_ZERO))
        beam_done = torch.zeros((num_graphs, beam_size), device=device, dtype=torch.bool)
        beam_done[:, 0] = valid_start & start_target

        num_beams = num_graphs * beam_size
        flat_graph_ids = torch.arange(num_graphs, device=device).repeat_interleave(beam_size)
        flat_beam_ids = torch.arange(beam_size, device=device).repeat(num_graphs)
        beam_context = prepared.context_tokens.index_select(0, flat_graph_ids)

        empty_long = torch.zeros((_ZERO,), device=device, dtype=torch.long)
        empty_bool = torch.zeros((_ZERO,), device=device, dtype=torch.bool)
        empty_float = torch.zeros((_ZERO,), device=device, dtype=torch.float32)

        for step in range(max_steps):
            flat_nodes = beam_nodes.view(-1)
            flat_scores = beam_scores.view(-1)
            flat_done = beam_done.view(-1)
            flat_valid = flat_nodes >= _ZERO
            expand_mask = flat_valid & ~flat_done
            outgoing = gather_outgoing_edges(
                curr_nodes=flat_nodes,
                edge_ids_by_head=prepared.edge_ids_by_head_fwd,
                edge_ptr_by_head=prepared.edge_ptr_by_head_fwd,
                active_mask=expand_mask,
            )

            if outgoing.edge_ids.numel() > _ZERO:
                step_ids = torch.full((num_beams,), step, device=device, dtype=torch.long)
                logits = self._compute_edge_logits(
                    policy=self.policy_fwd,
                    prepared=prepared,
                    edge_ids=outgoing.edge_ids,
                    edge_batch=outgoing.edge_batch,
                    steps=step_ids + _ONE,
                    temperature=float(_ONE),
                    context_tokens=beam_context,
                )
                log_denom = self._compute_log_denom(
                    logits=logits, edge_batch=outgoing.edge_batch, num_graphs=num_beams
                )
                log_probs = logits - log_denom.index_select(0, outgoing.edge_batch)
                cand_scores_edge = flat_scores.index_select(0, outgoing.edge_batch) + log_probs
                cand_nodes_edge = prepared.edge_index[_ONE].index_select(0, outgoing.edge_ids)
                cand_graph_edge = flat_graph_ids.index_select(0, outgoing.edge_batch)
                cand_src_beam_edge = flat_beam_ids.index_select(0, outgoing.edge_batch)
                cand_edge_id_edge = outgoing.edge_ids
                cand_is_edge_edge = torch.ones_like(cand_scores_edge, dtype=torch.bool)
                cand_done_edge = node_is_target.index_select(0, cand_nodes_edge)
            else:
                cand_scores_edge = empty_float
                cand_nodes_edge = empty_long
                cand_graph_edge = empty_long
                cand_src_beam_edge = empty_long
                cand_edge_id_edge = empty_long
                cand_is_edge_edge = empty_bool
                cand_done_edge = empty_bool

            stay_mask = flat_valid & (flat_done | ~outgoing.has_edge)
            cand_scores_stay = flat_scores[stay_mask]
            cand_nodes_stay = flat_nodes[stay_mask]
            cand_graph_stay = flat_graph_ids[stay_mask]
            cand_src_beam_stay = flat_beam_ids[stay_mask]
            cand_edge_id_stay = torch.full_like(cand_nodes_stay, _NEG_ONE)
            cand_is_edge_stay = torch.zeros_like(cand_scores_stay, dtype=torch.bool)
            cand_done_stay = torch.ones_like(cand_scores_stay, dtype=torch.bool)

            cand_scores = torch.cat((cand_scores_edge, cand_scores_stay), dim=0)
            if cand_scores.numel() == _ZERO:
                break
            cand_nodes = torch.cat((cand_nodes_edge, cand_nodes_stay), dim=0)
            cand_graph = torch.cat((cand_graph_edge, cand_graph_stay), dim=0)
            cand_src_beam = torch.cat((cand_src_beam_edge, cand_src_beam_stay), dim=0)
            cand_edge_id = torch.cat((cand_edge_id_edge, cand_edge_id_stay), dim=0)
            cand_is_edge = torch.cat((cand_is_edge_edge, cand_is_edge_stay), dim=0)
            cand_done = torch.cat((cand_done_edge, cand_done_stay), dim=0)

            order = torch.argsort(cand_graph)
            cand_graph = cand_graph.index_select(0, order)
            cand_scores = cand_scores.index_select(0, order)
            cand_nodes = cand_nodes.index_select(0, order)
            cand_src_beam = cand_src_beam.index_select(0, order)
            cand_edge_id = cand_edge_id.index_select(0, order)
            cand_is_edge = cand_is_edge.index_select(0, order)
            cand_done = cand_done.index_select(0, order)

            counts = torch.bincount(cand_graph, minlength=num_graphs)
            offsets = counts.cumsum(0)
            counts_cpu = counts.detach().cpu().tolist()
            offsets_cpu = offsets.detach().cpu().tolist()

            new_nodes = torch.full_like(beam_nodes, _NEG_ONE)
            new_scores = torch.full_like(beam_scores, neg_inf)
            new_paths = torch.full_like(beam_paths, _NEG_ONE)
            new_lengths = torch.zeros_like(beam_lengths)
            new_done = torch.zeros_like(beam_done)

            for graph_idx in range(num_graphs):
                count = counts_cpu[graph_idx]
                if count <= _ZERO:
                    continue
                end = offsets_cpu[graph_idx]
                start = end - count
                scores_g = cand_scores[start:end]
                k = min(beam_size, count)
                top_scores, top_idx = torch.topk(scores_g, k)
                sel_idx = top_idx + start
                sel_nodes = cand_nodes.index_select(0, sel_idx)
                sel_src = cand_src_beam.index_select(0, sel_idx)
                sel_edge_id = cand_edge_id.index_select(0, sel_idx)
                sel_is_edge = cand_is_edge.index_select(0, sel_idx)
                sel_done = cand_done.index_select(0, sel_idx)
                sel_paths = beam_paths[graph_idx].index_select(0, sel_src)
                sel_lengths = beam_lengths[graph_idx].index_select(0, sel_src)
                sel_paths = sel_paths.clone()
                sel_paths[:, step] = torch.where(sel_is_edge, sel_edge_id, sel_paths[:, step])
                sel_lengths = sel_lengths + sel_is_edge.to(dtype=sel_lengths.dtype)
                new_nodes[graph_idx, :k] = sel_nodes
                new_scores[graph_idx, :k] = top_scores
                new_paths[graph_idx, :k] = sel_paths
                new_lengths[graph_idx, :k] = sel_lengths
                new_done[graph_idx, :k] = sel_done

            beam_nodes = new_nodes
            beam_scores = new_scores
            beam_paths = new_paths
            beam_lengths = new_lengths
            beam_done = new_done

        beam_nodes_cpu = beam_nodes.detach().cpu()
        beam_scores_cpu = beam_scores.detach().cpu()
        beam_paths_cpu = beam_paths.detach().cpu()
        beam_lengths_cpu = beam_lengths.detach().cpu()
        beams: list[list[tuple[int, float, list[int]]]] = []
        for graph_idx in range(num_graphs):
            graph_beams: list[tuple[int, float, list[int]]] = []
            for beam_idx in range(beam_size):
                node_id = int(beam_nodes_cpu[graph_idx, beam_idx].item())
                if node_id < _ZERO:
                    continue
                score = float(beam_scores_cpu[graph_idx, beam_idx].item())
                length = int(beam_lengths_cpu[graph_idx, beam_idx].item())
                if length <= _ZERO:
                    path = []
                else:
                    path = beam_paths_cpu[graph_idx, beam_idx, :length].tolist()
                graph_beams.append((node_id, score, path))
            beams.append(graph_beams)
        return beams

    @torch.no_grad()
    def _compute_eval_metrics(self, batch: Any) -> tuple[dict[str, torch.Tensor], int]:
        prepared_fwd, prepared_bwd = self._prepare_batch(batch)
        num_graphs = int(prepared_fwd.node_ptr.numel() - _ONE)
        if num_graphs <= _ZERO:
            return {}, _ZERO
        valid_mask = ~prepared_fwd.dummy_mask
        valid_count = int(valid_mask.sum().detach().tolist())
        if valid_count <= _ZERO:
            return {}, _ZERO
        num_nodes_total = int(prepared_fwd.node_ptr[-1].detach().tolist())
        node_is_target_all = self._build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        pb_cfg = None
        pb_distances = None
        if self._is_static_pb():
            pb_cfg = self._resolve_pb_cfg()
            if pb_cfg["mode"] == _PB_MODE_TOPO_SEMANTIC:
                pb_distances = self._compute_distance_to_starts(
                    prepared=prepared_fwd,
                    max_hops=int(pb_cfg["max_hops"]),
                )
        beam_sizes = self._resolve_beam_sizes()
        max_beam_size = beam_sizes[-1]
        beams = self._beam_search(prepared=prepared_fwd, beam_size=max_beam_size, node_is_target=node_is_target_all)
        pass_hits = {
            beam_size: torch.zeros((num_graphs,), device=self.device, dtype=torch.float32)
            for beam_size in beam_sizes
        }
        hit_hits = {
            beam_size: torch.zeros((num_graphs,), device=self.device, dtype=torch.float32)
            for beam_size in beam_sizes
        }
        recall_scores = {
            beam_size: torch.zeros((num_graphs,), device=self.device, dtype=torch.float32)
            for beam_size in beam_sizes
        }
        precision_scores = {
            beam_size: torch.zeros((num_graphs,), device=self.device, dtype=torch.float32)
            for beam_size in beam_sizes
        }
        f1_scores = {
            beam_size: torch.zeros((num_graphs,), device=self.device, dtype=torch.float32)
            for beam_size in beam_sizes
        }
        length = torch.zeros((num_graphs,), device=self.device, dtype=torch.float32)
        for graph_idx in range(num_graphs):
            if not bool(valid_mask[graph_idx].detach().tolist()):
                continue
            beam = beams[graph_idx]
            if not beam:
                continue
            top_node, _, top_path = beam[0]
            length[graph_idx] = float(len(top_path))
            beam_nodes = [int(beam_node) for beam_node, _, _ in beam]
            hits = [bool(node_is_target_all[beam_node].detach().tolist()) for beam_node in beam_nodes]
            a_start = int(prepared_fwd.a_ptr[graph_idx].detach().tolist())
            a_end = int(prepared_fwd.a_ptr[graph_idx + _ONE].detach().tolist())
            answer_nodes = prepared_fwd.a_local_indices[a_start:a_end].detach().tolist() if a_end > a_start else []
            answer_set = {int(node_id) for node_id in answer_nodes if int(node_id) >= _ZERO}
            for beam_size in beam_sizes:
                topk = min(int(beam_size), len(beam_nodes))
                if topk > _ZERO and any(hits[:topk]):
                    pass_hits[beam_size][graph_idx] = float(_ONE)
                    hit_hits[beam_size][graph_idx] = float(_ONE)
                pred_nodes = {beam_nodes[idx] for idx in range(topk) if beam_nodes[idx] >= _ZERO}
                if answer_set:
                    overlap = pred_nodes & answer_set
                    recall = float(len(overlap)) / float(len(answer_set))
                    precision = float(len(overlap)) / float(len(pred_nodes)) if pred_nodes else float(_ZERO)
                    denom = recall + precision
                    f1 = (float(_TWO) * recall * precision / denom) if denom > float(_ZERO) else float(_ZERO)
                else:
                    recall = float(_ZERO)
                    precision = float(_ZERO)
                    f1 = float(_ZERO)
                recall_scores[beam_size][graph_idx] = recall
                precision_scores[beam_size][graph_idx] = precision
                f1_scores[beam_size][graph_idx] = f1
        metrics = {f"pass@{beam_size}": pass_hits[beam_size] for beam_size in beam_sizes}
        metrics.update({f"hit@{beam_size}": hit_hits[beam_size] for beam_size in beam_sizes})
        metrics.update({f"recall@{beam_size}": recall_scores[beam_size] for beam_size in beam_sizes})
        metrics.update({f"precision@{beam_size}": precision_scores[beam_size] for beam_size in beam_sizes})
        metrics.update({f"f1@{beam_size}": f1_scores[beam_size] for beam_size in beam_sizes})
        metrics["pass@beam"] = pass_hits[max_beam_size]
        metrics["hit@beam"] = hit_hits[max_beam_size]
        metrics["recall@beam"] = recall_scores[max_beam_size]
        metrics["precision@beam"] = precision_scores[max_beam_size]
        metrics["f1@beam"] = f1_scores[max_beam_size]
        metrics["length_mean"] = length
        eval_temperature = float(_ONE)
        rollout_fwd = self._rollout_policy(
            policy=self.policy_fwd,
            prepared=prepared_fwd,
            graph_mask=valid_mask,
            start_nodes=prepared_fwd.start_nodes_fwd,
            node_is_target=node_is_target_all,
            edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
            record_actions=True,
            record_log_pf=False,
            temperature=eval_temperature,
            context_tokens=prepared_fwd.context_tokens,
        )
        fwd_actions = rollout_fwd.actions
        if fwd_actions is None:
            fwd_actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=self.device, dtype=torch.long)
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            prepared_bwd=prepared_bwd,
            actions=fwd_actions,
            graph_mask=valid_mask,
            traj_lengths=rollout_fwd.num_moves,
            stop_reason=rollout_fwd.stop_reason,
            node_is_target=node_is_target_all,
            sampling_temperature=eval_temperature,
            pb_distances=pb_distances,
            pb_cfg=pb_cfg,
        )
        success = (rollout_fwd.stop_reason == _TERMINAL_HIT) & valid_mask
        metrics.update(db_metrics)
        metrics["rollout_success_rate"] = success.to(dtype=torch.float32).mean()
        metrics.update(
            self._build_terminal_metrics(
                stop_reason=rollout_fwd.stop_reason,
                graph_mask=valid_mask,
                prefix="rollout",
            )
        )
        metrics = self._reduce_eval_metrics(metrics, valid_mask=valid_mask)
        return metrics, valid_count

    @staticmethod
    def _reduce_eval_metrics(
        metrics: dict[str, torch.Tensor],
        *,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not metrics:
            return {}
        if valid_mask.numel() <= _ZERO:
            return {}
        valid_mask = valid_mask.to(dtype=torch.bool)
        if not bool(valid_mask.any().detach().tolist()):
            return {}
        reduced: dict[str, torch.Tensor] = {}
        for name, value in metrics.items():
            if not torch.is_tensor(value):
                reduced[name] = value
                continue
            if value.numel() == _ONE:
                reduced[name] = value.reshape(())
                continue
            if value.dim() != _ONE or value.size(0) != valid_mask.numel():
                raise ValueError(f"Eval metric {name} must be [num_graphs]; got {tuple(value.shape)}.")
            selected = value.to(dtype=torch.float32)[valid_mask]
            if selected.numel() == _ZERO:
                continue
            reduced[name] = selected.mean()
        return reduced

    # ------------------------- Predict -------------------------

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._ensure_runtime_initialized()
        _ = batch_idx, dataloader_idx
        prepared_fwd, _ = self._prepare_batch(batch)
        num_graphs = int(prepared_fwd.node_ptr.numel() - _ONE)
        if num_graphs <= _ZERO:
            return []
        valid_mask = ~prepared_fwd.dummy_mask
        num_nodes_total = int(prepared_fwd.node_ptr[-1].detach().tolist())
        node_is_target = self._build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        beam_size = self._resolve_beam_size()
        sample_ids = extract_sample_ids(batch)
        if len(sample_ids) != num_graphs:
            raise ValueError("sample_id length mismatch with batch graph count.")

        beams = self._beam_search(prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target)
        rollouts_per_graph: list[list[dict[str, Any]]] = [[] for _ in range(num_graphs)]
        for graph_idx in range(num_graphs):
            beam = beams[graph_idx]
            for beam_idx, (stop_node, score, path) in enumerate(beam):
                edges_list: list[dict[str, Any]] = []
                for edge_id in path:
                    head = int(prepared_fwd.edge_index[_ZERO, edge_id].detach().tolist())
                    tail = int(prepared_fwd.edge_index[_ONE, edge_id].detach().tolist())
                    rel = int(prepared_fwd.edge_relations[edge_id].detach().tolist())
                    head_ent = int(prepared_fwd.node_global_ids[head].detach().tolist())
                    tail_ent = int(prepared_fwd.node_global_ids[tail].detach().tolist())
                    edges_list.append(
                        {
                            "src_entity_id": head_ent,
                            "dst_entity_id": tail_ent,
                            "head_entity_id": head_ent,
                            "tail_entity_id": tail_ent,
                            "relation_id": rel,
                        }
                    )
                stop_entity = (
                    int(prepared_fwd.node_global_ids[stop_node].detach().tolist()) if stop_node >= _ZERO else None
                )
                success = bool(node_is_target[stop_node].detach().tolist()) if stop_node >= _ZERO else False
                rollouts_per_graph[graph_idx].append(
                    {
                        "rollout_index": beam_idx,
                        "score": float(score),
                        "edges": edges_list,
                        "stop_node_entity_id": stop_entity,
                        "reach_success": success,
                    }
                )

        records: list[dict[str, Any]] = []
        for graph_idx in range(num_graphs):
            node_start = int(prepared_fwd.node_ptr[graph_idx].detach().tolist())
            node_end = int(prepared_fwd.node_ptr[graph_idx + _ONE].detach().tolist())
            q_start = int(prepared_fwd.q_ptr[graph_idx].detach().tolist())
            q_end = int(prepared_fwd.q_ptr[graph_idx + _ONE].detach().tolist())
            start_indices = prepared_fwd.q_local_indices[q_start:q_end].to(dtype=torch.long)
            start_entity_ids: list[int]
            if start_indices.numel() == _ZERO:
                start_entity_ids = []
            else:
                if bool((start_indices < _ZERO).any().detach().tolist()):
                    raise ValueError(f"q_local_indices contain negative values for sample_id={sample_ids[graph_idx]!r}.")
                if bool((start_indices >= num_nodes_total).any().detach().tolist()):
                    raise ValueError(f"q_local_indices out of range for sample_id={sample_ids[graph_idx]!r}.")
                in_graph = (start_indices >= node_start) & (start_indices < node_end)
                if not bool(in_graph.all().detach().tolist()):
                    raise ValueError(f"q_local_indices mismatch node_ptr for sample_id={sample_ids[graph_idx]!r}.")
                start_entity_ids = (
                    prepared_fwd.node_global_ids.index_select(0, start_indices).detach().tolist()
                )
            a_start = int(prepared_fwd.answer_ptr[graph_idx].detach().tolist())
            a_end = int(prepared_fwd.answer_ptr[graph_idx + _ONE].detach().tolist())
            answer_ids = prepared_fwd.answer_entity_ids[a_start:a_end].detach().tolist() if a_end > a_start else []
            record = {
                "sample_id": sample_ids[graph_idx],
                "start_entity_ids": start_entity_ids,
                "answer_entity_ids": answer_ids,
                "rollouts": rollouts_per_graph[graph_idx],
            }
            question_text = getattr(batch, "question", None)
            if isinstance(question_text, (list, tuple)) and graph_idx < len(question_text):
                record["question"] = question_text[graph_idx]
            elif isinstance(question_text, str):
                record["question"] = question_text
            records.append(record)
        return records

    # ------------------------- Metrics -------------------------

    def _resolve_metric_mode(self) -> str:
        cfg = self.training_cfg.get("metrics") or {}
        mode = str(cfg.get("mode", _DEFAULT_METRIC_MODE)).strip().lower()
        if mode not in _METRIC_MODES:
            raise ValueError(f"training_cfg.metrics.mode must be one of {sorted(_METRIC_MODES)}, got {mode!r}.")
        return mode

    def _select_training_metrics(self, metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mode = self._resolve_metric_mode()
        if mode == "full":
            return metrics
        keep = {
            "db_loss",
            "db_loss_fwd",
            "db_loss_bwd",
            "db_log_pb_mean",
            "db_log_pb_min",
            "db_log_z_u_mean",
            "db_log_z_v_mean",
            "db_inv_edge_invalid_rate",
            "db_no_allowed_rate",
            "db_topo_violation_rate",
            "rollout_success_rate",
            "rollout_terminal_hit_rate",
            "rollout_terminal_dead_end_rate",
            "rollout_terminal_max_steps_rate",
            "rollout_terminal_invalid_start_rate",
            "rollout_terminal_other_rate",
            "loss_total",
        }
        return {name: value for name, value in metrics.items() if name in keep}


__all__ = ["DualFlowModule"]
