from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import torch
from lightning import LightningModule
from torch import nn

from src.metrics import gflownet as gfn_metrics
from src.models.components import (
    CvtNodeInitializer,
    EmbeddingBackbone,
    FlowPredictor,
    GFlowNetActor,
    GraphEnv,
    RewardOutput,
    TrajectoryAgent,
)
from src.models.components.gflownet_actor import RolloutDiagnostics
from src.models.components.gflownet_env import STOP_RELATION
from src.models.components.gflownet_ops import (
    EDGE_POLICY_MASK_KEY,
    STOP_NODE_MASK_KEY,
    OutgoingEdges,
    apply_edge_policy_mask,
    compute_forward_log_probs,
    gather_outgoing_edges,
    neg_inf_value,
    segment_max,
)
from src.models.components.gflownet_ops import build_edge_head_csr
from src.models.components.trajectory_utils import (
    derive_trajectory,
    reflect_backward_to_forward,
    reflect_forward_to_backward,
    stack_steps,
)
from src.utils import log_metric, setup_optimizer
from src.utils.logging_utils import get_logger, log_event

logger = get_logger(__name__)

_ZERO = 0
_ONE = 1
_TWO = 2

_FLOW_STATS_DIM = 2

_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_CVT_INIT_ENABLED = True
_DEFAULT_CACHE_ACTION_KEYS = True
_DEFAULT_VALIDATE_EDGE_BATCH = False

_DEFAULT_TB_LOG_PROB_MIN = -20.0
_DEFAULT_TB_DELTA_MAX = 20.0
_DEFAULT_ALLOW_ZERO_HOP = False
_DEFAULT_SHAPING_WEIGHT = 0.0
_DEFAULT_SHAPING_EPS = 1.0e-6
_DEFAULT_SHAPING_ANNEAL_STEPS = 0
_DEFAULT_SHAPING_ANNEAL_START = 0

_EDGE_INV_PREVIEW = 5

_METRIC_P50 = 0.5
_METRIC_P90 = 0.9

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
class _TBDiagSpec:
    log_prob_min: float
    delta_max: float


@dataclass(frozen=True)
class _PreparedBatch:
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    node_batch: torch.Tensor
    flow_features: torch.Tensor
    q_local_indices: torch.Tensor
    a_local_indices: torch.Tensor
    q_ptr: torch.Tensor
    a_ptr: torch.Tensor
    start_nodes_q: torch.Tensor
    dummy_mask: torch.Tensor
    edge_is_inverse: torch.Tensor
    inverse_edge_ids: torch.Tensor
    edge_ids_by_head: torch.Tensor
    edge_ptr_by_head: torch.Tensor


@dataclass(frozen=True)
class _TBViewTerms:
    loss_per_graph: torch.Tensor
    weight_mask: torch.Tensor
    success_mask: torch.Tensor
    diag: dict[str, torch.Tensor]


@dataclass(frozen=True)
class _TBBackwardInputs:
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor
    stats_fwd: Any
    reward_fwd: RewardOutput
    log_reward: torch.Tensor
    success_mask: torch.Tensor
    log_f_start: torch.Tensor
    stats_bwd: Any
    bwd_diag: Optional[RolloutDiagnostics]
    fwd_diag: Optional[RolloutDiagnostics]


@dataclass(frozen=True)
class _TBForwardInputs:
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor
    stats_fwd: Any
    reward_fwd: RewardOutput
    success_mask: torch.Tensor
    log_f_start: torch.Tensor
    stats_bwd: Any
    bwd_diag: Optional[RolloutDiagnostics]
    fwd_diag: Optional[RolloutDiagnostics]
    actions_bwd: torch.Tensor
    stop_nodes_fwd: torch.Tensor


class GFlowNetModule(LightningModule):
    """Minimal Dual-Stream GFlowNet implementation in a single LightningModule."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        reward_fn: nn.Module,
        env: GraphEnv,
        emb_dim: int,
        inverse_relation_suffix: Optional[str] = None,
        backbone_finetune: bool = _DEFAULT_BACKBONE_FINETUNE,
        state_cfg: Optional[Mapping[str, Any]] = None,
        cvt_init_cfg: Optional[Mapping[str, Any]] = None,
        flow_cfg: Optional[Mapping[str, Any]] = None,
        backward_cfg: Optional[Mapping[str, Any]] = None,
        training_cfg: Mapping[str, Any] = None,
        evaluation_cfg: Mapping[str, Any] = None,
        actor_cfg: Optional[Mapping[str, Any]] = None,
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
        self.reward_fn = reward_fn
        self.env = env
        self.max_steps = int(self.env.max_steps)
        self._inverse_relation_suffix = inverse_relation_suffix

        self.training_cfg = training_cfg or {}
        self.evaluation_cfg = evaluation_cfg or {}
        self.actor_cfg = actor_cfg or {}
        self.state_cfg = state_cfg or {}
        self.cvt_init_cfg = cvt_init_cfg or {}
        self.flow_cfg = flow_cfg or {}
        self.backward_cfg = backward_cfg or {}
        self.runtime_cfg = runtime_cfg or {}
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.logging_cfg = logging_cfg or {}

        self._cache_action_keys = bool(self.runtime_cfg.get("cache_action_keys", _DEFAULT_CACHE_ACTION_KEYS))
        self._validate_edge_batch = bool(self.runtime_cfg.get("validate_edge_batch", _DEFAULT_VALIDATE_EDGE_BATCH))

        self._init_backbone(emb_dim=emb_dim, finetune=backbone_finetune)
        self._init_cvt_init()
        self._init_agents()
        self._init_actors()
        self._init_flow_predictors()
        self._validate_cfg_contract()
        self._save_serializable_hparams()

        self._cvt_mask: Optional[torch.Tensor] = None
        self._relation_inverse_map: Optional[torch.Tensor] = None
        self._relation_is_inverse: Optional[torch.Tensor] = None

    # ------------------------- Init -------------------------

    def _init_backbone(self, *, emb_dim: int, finetune: bool) -> None:
        self.backbone = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            finetune=finetune,
        )

    def _init_cvt_init(self) -> None:
        if not bool(self.cvt_init_cfg.get("enabled", _DEFAULT_CVT_INIT_ENABLED)):
            raise ValueError("cvt_init_cfg.enabled must be true; CVT initialization is mandatory.")
        self.cvt_init = CvtNodeInitializer()

    def _build_agent(self, *, cfg: Mapping[str, Any], name: str) -> TrajectoryAgent:
        hidden_dim = self._resolve_agent_hidden_dim(cfg=cfg, name=name)
        dropout = float(cfg.get("dropout", 0.0))
        return TrajectoryAgent(
            token_dim=self.hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def _init_agents(self) -> None:
        self.agent = self._build_agent(cfg=self.state_cfg, name="state_cfg")
        share_state = bool(self.backward_cfg.get("share_state_encoder", False))
        share_edge = bool(self.backward_cfg.get("share_edge_scorer", False))
        if share_state != share_edge:
            raise ValueError("backward_cfg.share_state_encoder/share_edge_scorer must match for TrajectoryAgent.")
        if share_state and share_edge:
            self.agent_backward = self.agent
        else:
            self.agent_backward = self._build_agent(cfg=self.state_cfg, name="backward_cfg.state_cfg")

    def _resolve_agent_hidden_dim(self, *, cfg: Mapping[str, Any], name: str) -> int:
        raw_state_dim = cfg.get("state_dim", None)
        if raw_state_dim is None:
            return int(self.hidden_dim)
        state_dim = int(raw_state_dim)
        if state_dim != int(self.hidden_dim):
            raise ValueError(
                f"{name}.state_dim must match model.hidden_dim ({self.hidden_dim}). "
                f"Got {state_dim}."
            )
        return int(self.hidden_dim)

    def _init_actors(self) -> None:
        if "score_mode" in self.actor_cfg:
            raise ValueError("actor_cfg.score_mode is no longer supported; use agent prior + h_transform only.")
        policy_temperature = float(self.actor_cfg.get("policy_temperature", 1.0))
        stop_bias_raw = self.actor_cfg.get("stop_bias_init")
        stop_bias_init = None if stop_bias_raw is None else float(stop_bias_raw)
        h_transform_bias = self.actor_cfg.get("h_transform_bias")
        if h_transform_bias is not None:
            h_transform_bias = float(h_transform_bias)
        h_transform_clip = self.actor_cfg.get("h_transform_clip")
        if h_transform_clip is not None:
            h_transform_clip = float(h_transform_clip)
        direction_embedding = self.actor_cfg.get("direction_embedding")
        if direction_embedding is not None:
            direction_embedding = bool(direction_embedding)
        direction_embedding_scale = self.actor_cfg.get("direction_embedding_scale")
        if direction_embedding_scale is not None:
            direction_embedding_scale = float(direction_embedding_scale)
        self.actor = GFlowNetActor(
            env=self.env,
            agent=self.agent,
            max_steps=self.max_steps,
            policy_temperature=policy_temperature,
            stop_bias_init=stop_bias_init,
            context_mode="question_start",
            default_mode="forward",
            h_transform_bias=h_transform_bias,
            h_transform_clip=h_transform_clip,
            direction_embedding=direction_embedding,
            direction_embedding_scale=direction_embedding_scale,
        )
        self.actor_backward = GFlowNetActor(
            env=self.env,
            agent=self.agent_backward,
            max_steps=self.max_steps,
            policy_temperature=policy_temperature,
            stop_bias_init=stop_bias_init,
            context_mode="question_start",
            default_mode="backward",
            h_transform_bias=h_transform_bias,
            h_transform_clip=h_transform_clip,
            direction_embedding=direction_embedding,
            direction_embedding_scale=direction_embedding_scale,
        )

    def _init_flow_predictors(self) -> None:
        self.flow_graph_stats_log1p = bool(self.flow_cfg.get("graph_stats_log1p", True))
        self.log_f = FlowPredictor(self.hidden_dim, _FLOW_STATS_DIM)

    def _validate_cfg_contract(self) -> None:
        allowed_training = {
            "num_train_rollouts",
            "num_miner_rollouts",
            "num_forward_rollouts",
            "accumulate_grad_batches",
            "allow_zero_hop",
            "shaping",
            "tb",
        }
        extra_training = set(self.training_cfg.keys()) - allowed_training
        if extra_training:
            raise ValueError(f"Unsupported training_cfg keys: {sorted(extra_training)}")
        allowed_eval = {"num_eval_rollouts", "rollout_temperature"}
        extra_eval = set(self.evaluation_cfg.keys()) - allowed_eval
        if extra_eval:
            raise ValueError(f"Unsupported evaluation_cfg keys: {sorted(extra_eval)}")

    def _save_serializable_hparams(self) -> None:
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "backbone",
                "cvt_init",
                "agent",
                "agent_backward",
                "actor_backward",
                "reward_fn",
                "env",
                "log_f",
                "log_f_backward",
                "training_cfg",
                "evaluation_cfg",
                "actor_cfg",
                "state_cfg",
                "cvt_init_cfg",
                "flow_cfg",
                "backward_cfg",
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
        if self._relation_inverse_map is not None and self._relation_is_inverse is not None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("datamodule is required to initialize relation inverse assets.")
        resources = getattr(datamodule, "shared_resources", None)
        if resources is None:
            raise RuntimeError("datamodule.shared_resources is required to initialize relation inverse assets.")
        self._cvt_mask = resources.cvt_mask
        inverse_map_cpu, inverse_mask_cpu = resources.relation_inverse_assets(suffix=self._inverse_relation_suffix)
        inverse_map = inverse_map_cpu.to(device=self.device, dtype=torch.long, non_blocking=True)
        inverse_mask = inverse_mask_cpu.to(device=self.device, dtype=torch.bool, non_blocking=True)
        self.register_buffer("relation_inverse_map", inverse_map, persistent=False)
        self.register_buffer("relation_is_inverse", inverse_mask, persistent=False)
        self._relation_inverse_map = inverse_map
        self._relation_is_inverse = inverse_mask

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
            return None
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

    # ------------------------- Lightning hooks -------------------------

    def forward(self, batch: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("GFlowNetModule.forward is not supported; use training_step/eval rollouts.")

    @staticmethod
    def _log_batch_scale(batch: Any, *, batch_idx: int) -> None:
        if batch_idx != _ZERO:
            return
        ptr = getattr(batch, "ptr", None)
        edge_index = getattr(batch, "edge_index", None)
        num_nodes_val = getattr(batch, "num_nodes", None)
        if not torch.is_tensor(ptr) or not torch.is_tensor(edge_index) or num_nodes_val is None:
            log_event(
                logger,
                "gfn_batch_scale_missing",
                has_ptr=torch.is_tensor(ptr),
                has_edge_index=torch.is_tensor(edge_index),
                has_num_nodes=num_nodes_val is not None,
            )
            return
        num_nodes = int(num_nodes_val.detach().tolist()) if torch.is_tensor(num_nodes_val) else int(num_nodes_val)
        num_graphs = int(ptr.numel() - _ONE)
        num_edges = int(edge_index.size(1))
        log_event(logger, "gfn_batch_scale", num_graphs=num_graphs, num_nodes=num_nodes, num_edges=num_edges)

    def training_step(self, batch: Any, batch_idx: int):
        self._ensure_runtime_initialized()
        optimizer = self.optimizers()
        accum = float(self._accumulate_grad_batches())
        if self._should_zero_grad(batch_idx):
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._compute_training_loss(batch)
        if not torch.isfinite(loss).all().item():
            raise ValueError("Non-finite loss detected.")
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
            if name.startswith("terminal_hit@"):
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
            if name.startswith("terminal_hit@"):
                log_metric(self, f"test/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)

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
    def _validate_packed_node_locals(
        *,
        node_locals: torch.Tensor,
        ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        name: str,
    ) -> None:
        node_ptr = node_ptr.to(dtype=torch.long).view(-1)
        node_locals = node_locals.to(device=node_ptr.device, dtype=torch.long).view(-1)
        ptr = ptr.to(device=node_ptr.device, dtype=torch.long).view(-1)
        num_graphs = int(node_ptr.numel() - _ONE)
        if ptr.numel() != num_graphs + _ONE:
            raise ValueError(f"{name}_ptr length mismatch with batch size.")
        if int(ptr[_ZERO].detach().tolist()) != _ZERO:
            raise ValueError(f"{name}_ptr must start at 0.")
        if int(ptr[-_ONE].detach().tolist()) != int(node_locals.numel()):
            raise ValueError(f"{name}_ptr must end at {int(node_locals.numel())}.")
        if bool((ptr[_ONE:] < ptr[:-_ONE]).any().detach().tolist()):
            raise ValueError(f"{name}_ptr must be non-decreasing.")
        if node_locals.numel() == _ZERO:
            return
        positions = torch.arange(node_locals.numel(), device=node_ptr.device, dtype=ptr.dtype)
        graph_ids = torch.bucketize(positions, ptr[_ONE:], right=True)
        node_start = node_ptr.index_select(0, graph_ids)
        node_end = node_ptr.index_select(0, graph_ids + _ONE)
        invalid = (node_locals < node_start) | (node_locals >= node_end)
        if bool(invalid.any().detach().tolist()):
            preview = node_locals[invalid][:_EDGE_INV_PREVIEW].to(device="cpu").tolist()
            raise ValueError(f"{name} indices fall outside per-graph node ranges (preview={preview}).")

    @staticmethod
    def _compute_node_batch(node_ptr: torch.Tensor) -> torch.Tensor:
        num_graphs = int(node_ptr.numel() - _ONE)
        node_counts = (node_ptr[_ONE:] - node_ptr[:-_ONE]).clamp(min=_ZERO)
        return torch.repeat_interleave(torch.arange(num_graphs, device=node_ptr.device), node_counts)

    @staticmethod
    def _safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        denom = denom.to(device=numer.device, dtype=numer.dtype)
        return torch.where(denom > float(_ZERO), numer / denom, torch.zeros_like(numer))

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.numel() == _ZERO:
            return values.new_zeros(())
        mask = mask.to(device=values.device, dtype=torch.bool)
        selected = values[mask]
        if selected.numel() == _ZERO:
            return values.new_zeros(())
        return selected.mean()

    @staticmethod
    def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        if values.numel() == _ZERO:
            return values.new_zeros(())
        mask = mask.to(device=values.device, dtype=torch.bool)
        selected = values[mask]
        if selected.numel() == _ZERO:
            return values.new_zeros(())
        return selected.to(dtype=torch.float32).quantile(q)

    @staticmethod
    def _apply_metric_prefix(metrics: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        if not prefix:
            return metrics
        return {f"{prefix}{name}": value for name, value in metrics.items()}

    @staticmethod
    def _init_rollout_diag_steps() -> dict[str, list[torch.Tensor]]:
        return {
            "has_edge": [],
            "stop_margin": [],
            "allow_stop": [],
            "max_edge_score": [],
            "stop_logits": [],
        }

    @staticmethod
    def _build_rollout_diagnostics(
        *,
        diag_steps: dict[str, list[torch.Tensor]],
        num_graphs: int,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RolloutDiagnostics:
        has_edge_seq = stack_steps(
            diag_steps["has_edge"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=torch.bool,
            fill_value=_ZERO,
        )
        stop_margin_seq = stack_steps(
            diag_steps["stop_margin"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            fill_value=float(_ZERO),
        )
        allow_stop_seq = stack_steps(
            diag_steps["allow_stop"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=torch.bool,
            fill_value=_ZERO,
        )
        max_edge_score_seq = stack_steps(
            diag_steps["max_edge_score"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            fill_value=float(_ZERO),
        )
        stop_logit_seq = stack_steps(
            diag_steps["stop_logits"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            fill_value=float(_ZERO),
        )
        return RolloutDiagnostics(
            has_edge_seq=has_edge_seq,
            stop_margin_seq=stop_margin_seq,
            allow_stop_seq=allow_stop_seq,
            max_edge_score_seq=max_edge_score_seq,
            stop_logit_seq=stop_logit_seq,
        )

    def _build_flow_features(self, *, node_ptr: torch.Tensor, edge_ptr: torch.Tensor) -> torch.Tensor:
        node_counts = (node_ptr[_ONE:] - node_ptr[:-_ONE]).to(dtype=torch.float32)
        edge_counts = (edge_ptr[_ONE:] - edge_ptr[:-_ONE]).to(dtype=torch.float32)
        stats = torch.stack((node_counts, edge_counts), dim=1)
        if self.flow_graph_stats_log1p:
            stats = torch.log1p(stats)
        return stats

    @staticmethod
    def _sample_one_start_nodes(*, node_locals: torch.Tensor, ptr: torch.Tensor, name: str) -> torch.Tensor:
        ptr = ptr.to(dtype=torch.long).view(-1)
        node_locals = node_locals.to(device=ptr.device, dtype=torch.long).view(-1)
        counts = (ptr[_ONE:] - ptr[:-_ONE]).clamp(min=_ZERO)
        if bool((counts <= _ZERO).any().detach().tolist()):
            raise ValueError(f"{name} missing in batch; filter data.")
        num_graphs = int(counts.numel())
        if num_graphs <= _ZERO:
            return torch.zeros((0,), device=ptr.device, dtype=torch.long)
        offsets = torch.floor(
            torch.rand((num_graphs,), device=ptr.device) * counts.to(dtype=torch.float32)
        ).to(dtype=torch.long)
        starts = ptr[:-_ONE]
        sample_idx = starts + offsets
        return node_locals.index_select(0, sample_idx)

    @staticmethod
    def _build_dummy_mask(*, answer_ptr: torch.Tensor) -> torch.Tensor:
        answer_counts = answer_ptr[1:] - answer_ptr[:-1]
        return (answer_counts == _ZERO).to(dtype=torch.bool)

    def _resolve_node_is_cvt(self, batch: Any, *, num_nodes_total: int, device: torch.device) -> torch.Tensor:
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

    def _build_inverse_edge_ids(self, *, edge_index: torch.Tensor, edge_relations: torch.Tensor, num_nodes: int) -> torch.Tensor:
        inverse_map = self.relation_inverse_map
        if edge_index.numel() == _ZERO:
            return torch.zeros((0,), device=edge_index.device, dtype=torch.long)
        edge_relations = edge_relations.to(device=edge_index.device, dtype=torch.long).view(-1)
        inv_relations = inverse_map.index_select(0, edge_relations)
        if bool((inv_relations < _ZERO).any().detach().tolist()):
            bad = inv_relations[inv_relations < _ZERO][:_EDGE_INV_PREVIEW].tolist()
            raise ValueError(f"relation_inverse_map missing ids for edges (preview: {bad}).")
        heads = edge_index[_ZERO].to(dtype=torch.long)
        tails = edge_index[_ONE].to(dtype=torch.long)
        num_relations = int(inverse_map.numel())
        stride_tail = num_relations
        stride_head = num_nodes * num_relations
        edge_keys = heads * stride_head + tails * stride_tail + edge_relations
        inv_keys = tails * stride_head + heads * stride_tail + inv_relations
        sorted_keys, sorted_idx = torch.sort(edge_keys)
        pos = torch.searchsorted(sorted_keys, inv_keys)
        num_keys = int(sorted_keys.numel())
        max_pos = max(num_keys - _ONE, _ZERO)
        pos_safe = pos.clamp(min=_ZERO, max=max_pos)
        matched = sorted_keys.index_select(0, pos_safe) == inv_keys
        within = pos < num_keys
        valid = within & matched
        if not bool(valid.all().detach().tolist()):
            preview = inv_keys[~valid][:_EDGE_INV_PREVIEW].tolist()
            raise ValueError(f"Missing inverse edges for backward policy (preview keys: {preview}).")
        return sorted_idx.index_select(0, pos_safe)

    def _prepare_batch(self, batch: Any) -> _PreparedBatch:
        device = self.device
        node_ptr = getattr(batch, "ptr", None)
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)
        if not torch.is_tensor(node_ptr) or not torch.is_tensor(edge_index) or not torch.is_tensor(edge_attr):
            raise AttributeError("Batch missing ptr/edge_index/edge_attr required for GFlowNet.")
        node_ptr = node_ptr.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_index = edge_index.to(device=device, dtype=torch.long, non_blocking=True)
        edge_relations = edge_attr.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        num_graphs = int(node_ptr.numel() - _ONE)
        num_nodes_total = int(node_ptr[-1].detach().tolist()) if node_ptr.numel() > 0 else _ZERO

        edge_batch = getattr(batch, "edge_batch", None)
        edge_ptr = getattr(batch, "edge_ptr", None)
        if edge_batch is None or edge_ptr is None:
            raise AttributeError(
                "Batch missing edge_batch/edge_ptr; enable data.precompute_edge_batch to avoid per-step CPU builds."
            )
        edge_batch = torch.as_tensor(edge_batch, dtype=torch.long, device=device).view(-1)
        edge_ptr = torch.as_tensor(edge_ptr, dtype=torch.long, device=device).view(-1)

        q_local_indices = getattr(batch, "q_local_indices", None)
        a_local_indices = getattr(batch, "a_local_indices", None)
        if not torch.is_tensor(q_local_indices) or not torch.is_tensor(a_local_indices):
            raise AttributeError("Batch missing q_local_indices/a_local_indices required for GFlowNet.")
        q_local_indices = q_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        a_local_indices = a_local_indices.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        slice_dict = getattr(batch, "_slice_dict")
        q_ptr = slice_dict["q_local_indices"].to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        a_ptr = slice_dict["a_local_indices"].to(device=device, dtype=torch.long, non_blocking=True).view(-1)

        dummy_mask = self._build_dummy_mask(answer_ptr=a_ptr)
        self._validate_packed_node_locals(node_locals=q_local_indices, ptr=q_ptr, node_ptr=node_ptr, name="q_local_indices")
        self._validate_packed_node_locals(node_locals=a_local_indices, ptr=a_ptr, node_ptr=node_ptr, name="a_local_indices")
        start_nodes_q = self._sample_one_start_nodes(node_locals=q_local_indices, ptr=q_ptr, name="q_local_indices")
        node_batch = self._compute_node_batch(node_ptr)

        node_is_cvt = self._resolve_node_is_cvt(batch, num_nodes_total=num_nodes_total, device=device)
        question_emb = getattr(batch, "question_emb", None)
        node_embeddings = getattr(batch, "node_embeddings", None)
        edge_embeddings = getattr(batch, "edge_embeddings", None)
        if not torch.is_tensor(question_emb):
            raise AttributeError("Batch missing question_emb required for GFlowNet.")
        if not torch.is_tensor(node_embeddings) or not torch.is_tensor(edge_embeddings):
            raise AttributeError("Batch missing node_embeddings/edge_embeddings required for GFlowNet.")
        question_emb = question_emb.to(device=device, non_blocking=True)
        node_embeddings = node_embeddings.to(device=device, non_blocking=True)
        edge_embeddings = edge_embeddings.to(device=device, non_blocking=True)
        node_embeddings = self.cvt_init(
            node_embeddings=node_embeddings,
            relation_embeddings=edge_embeddings,
            edge_index=edge_index,
            node_is_cvt=node_is_cvt,
        )
        node_tokens = self.backbone.project_node_embeddings(node_embeddings)
        relation_tokens = self.backbone.project_relation_embeddings(edge_embeddings)
        question_tokens = self.backbone.project_question_embeddings(question_emb)
        flow_features = self._build_flow_features(node_ptr=node_ptr, edge_ptr=edge_ptr)

        relation_is_inverse = self.relation_is_inverse
        edge_is_inverse = relation_is_inverse.index_select(0, edge_relations).to(device=device, dtype=torch.bool)
        inverse_edge_ids = self._build_inverse_edge_ids(edge_index=edge_index, edge_relations=edge_relations, num_nodes=num_nodes_total)
        edge_ids_by_head, edge_ptr_by_head = build_edge_head_csr(edge_index=edge_index, num_nodes_total=num_nodes_total, device=device)
        return _PreparedBatch(
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            question_tokens=question_tokens,
            node_batch=node_batch,
            flow_features=flow_features,
            q_local_indices=q_local_indices,
            a_local_indices=a_local_indices,
            q_ptr=q_ptr,
            a_ptr=a_ptr,
            start_nodes_q=start_nodes_q,
            dummy_mask=dummy_mask,
            edge_is_inverse=edge_is_inverse,
            inverse_edge_ids=inverse_edge_ids,
            edge_ids_by_head=edge_ids_by_head,
            edge_ptr_by_head=edge_ptr_by_head,
        )

    # ------------------------- Core math -------------------------

    @staticmethod
    def _build_state_nodes(*, actions: torch.Tensor, edge_index: torch.Tensor, start_nodes: torch.Tensor) -> torch.Tensor:
        if actions.dim() != 2:
            raise ValueError("actions must be [B, T].")
        num_graphs, num_steps = actions.shape
        device = actions.device
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            return torch.zeros((num_graphs, num_steps), device=device, dtype=torch.long)
        start_nodes = start_nodes.to(device=device, dtype=torch.long).view(-1)
        actions = actions.to(device=device, dtype=torch.long)
        action_ids = actions.clamp(min=_ZERO)
        tails = edge_index[_ONE].to(device=device, dtype=torch.long).index_select(0, action_ids.view(-1)).view(num_graphs, num_steps)
        step_ids = torch.arange(num_steps, device=device, dtype=torch.long).view(1, -1).expand(num_graphs, -1)
        valid_idx = torch.where(actions >= _ZERO, step_ids, torch.full_like(step_ids, -_ONE))
        last_idx = torch.cummax(valid_idx, dim=1).values
        tail_last = tails.gather(1, last_idx.clamp(min=_ZERO))
        node_after = torch.where(last_idx >= _ZERO, tail_last, start_nodes.view(-1, 1))
        state_nodes = torch.empty((num_graphs, num_steps), device=device, dtype=torch.long)
        state_nodes[:, 0] = start_nodes
        if num_steps > _ONE:
            state_nodes[:, 1:] = node_after[:, :-1]
        return state_nodes

    @staticmethod
    def _compute_stop_node_locals(*, state_nodes: torch.Tensor, stop_idx: torch.Tensor, node_ptr: torch.Tensor) -> torch.Tensor:
        stop_nodes = state_nodes.gather(1, stop_idx.view(-1, 1)).squeeze(1)
        offsets = node_ptr[:-1].to(device=stop_nodes.device, dtype=torch.long)
        locals_ = stop_nodes - offsets
        valid = stop_nodes >= _ZERO
        return torch.where(valid, locals_.to(dtype=torch.long), torch.full_like(locals_, -1))

    @staticmethod
    def _build_node_is_target(num_nodes_total: int, target_locals: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((num_nodes_total,), device=target_locals.device, dtype=torch.bool)
        if target_locals.numel() > 0:
            mask[target_locals.to(dtype=torch.long).clamp(min=_ZERO)] = True
        return mask

    @staticmethod
    def _compute_log_f_nodes(
        *,
        log_f_module: nn.Module,
        node_tokens: torch.Tensor,
        node_batch: torch.Tensor,
        question_tokens: torch.Tensor,
        flow_features: torch.Tensor,
    ) -> torch.Tensor:
        return log_f_module(
            node_tokens=node_tokens,
            question_tokens=question_tokens,
            graph_features=flow_features,
            node_batch=node_batch,
        )

    @staticmethod
    def _gather_log_f_start(*, log_f_nodes: torch.Tensor, start_nodes: torch.Tensor) -> torch.Tensor:
        start_nodes = start_nodes.to(device=log_f_nodes.device, dtype=torch.long).view(-1)
        valid = start_nodes >= _ZERO
        safe_nodes = start_nodes.clamp(min=_ZERO)
        log_f_start = log_f_nodes.index_select(0, safe_nodes)
        return torch.where(valid, log_f_start, torch.zeros_like(log_f_start))

    def _resolve_tb_diag_spec(self) -> _TBDiagSpec:
        tb_cfg = self.training_cfg.get("tb") or {}
        log_prob_min = float(tb_cfg.get("log_prob_min", _DEFAULT_TB_LOG_PROB_MIN))
        delta_max = float(tb_cfg.get("delta_max", _DEFAULT_TB_DELTA_MAX))
        return _TBDiagSpec(
            log_prob_min=log_prob_min,
            delta_max=delta_max,
        )

    def _resolve_allow_zero_hop(self) -> bool:
        return bool(self.training_cfg.get("allow_zero_hop", _DEFAULT_ALLOW_ZERO_HOP))

    def _resolve_shaping_weight(self) -> float:
        shaping_cfg = self.training_cfg.get("shaping") or {}
        if not isinstance(shaping_cfg, Mapping):
            return float(_DEFAULT_SHAPING_WEIGHT)
        enabled = shaping_cfg.get("enabled", True)
        if enabled is not None and not bool(enabled):
            return float(_ZERO)
        weight = float(shaping_cfg.get("weight", _DEFAULT_SHAPING_WEIGHT))
        if weight < float(_ZERO):
            raise ValueError("training_cfg.shaping.weight must be >= 0.")
        anneal_steps = int(shaping_cfg.get("anneal_steps", _DEFAULT_SHAPING_ANNEAL_STEPS))
        anneal_start = int(shaping_cfg.get("anneal_start_step", _DEFAULT_SHAPING_ANNEAL_START))
        if anneal_steps <= _ZERO:
            return weight
        step = int(getattr(self, "global_step", _ZERO))
        if step < anneal_start:
            return weight
        progress = (step - anneal_start) / float(anneal_steps)
        factor = max(float(_ZERO), float(_ONE) - float(progress))
        return weight * factor

    @staticmethod
    def _resolve_action_keys_from_graph(*, actor: GFlowNetActor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        action_keys = graph.get("action_keys_shared")
        if action_keys is None:
            action_keys = graph.get("action_keys_backward")
        if action_keys is None:
            action_keys = graph.get("action_keys_forward")
        if action_keys is None:
            action_keys = actor.agent.precompute_action_keys(
                relation_tokens=graph["relation_tokens"],
                node_tokens=graph["node_tokens"],
                edge_index=graph["edge_index"],
                question_tokens=graph["question_tokens"],
                edge_batch=graph["edge_batch"],
            )
        return action_keys

    def _encode_state_sequence(
        self,
        *,
        actor: GFlowNetActor,
        graph: dict[str, torch.Tensor],
        actions: torch.Tensor,
        stats: Any,
        start_nodes: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs, num_steps = actions.shape
        hidden0 = self._compute_initial_hidden(
            actor=actor,
            node_tokens=graph["node_tokens"],
            question_tokens=graph["question_tokens"],
            start_nodes=start_nodes,
        )
        edge_index = graph["edge_index"]
        actions_safe = actions.clamp(min=_ZERO)
        relation_tokens_step = graph["relation_tokens"].index_select(0, actions_safe.view(-1)).view(num_graphs, num_steps, -1)
        tails = edge_index[_ONE].index_select(0, actions_safe.view(-1)).view(num_graphs, num_steps)
        node_tokens_step = graph["node_tokens"].index_select(0, tails.view(-1)).view(num_graphs, num_steps, -1)
        state_vec, _ = actor.agent.encode_state_sequence(
            hidden=hidden0,
            relation_tokens=relation_tokens_step,
            node_tokens=node_tokens_step,
            action_mask=stats.move_mask,
        )
        return state_vec

    @staticmethod
    def _gather_policy_outgoing(
        *,
        curr_nodes: torch.Tensor,
        active_mask: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> OutgoingEdges:
        outgoing = gather_outgoing_edges(
            curr_nodes=curr_nodes,
            edge_ids_by_head=graph["edge_ids_by_head"],
            edge_ptr_by_head=graph["edge_ptr_by_head"],
            active_mask=active_mask,
        )
        policy_mask = graph.get(EDGE_POLICY_MASK_KEY)
        if policy_mask is None:
            raise ValueError("edge_policy_mask missing from graph cache; strict edge policy requires it.")
        num_graphs = int(curr_nodes.numel())
        if outgoing.edge_ids.numel() == _ZERO:
            empty = outgoing.edge_ids.new_empty((_ZERO,))
            edge_counts = outgoing.edge_ids.new_zeros((num_graphs,))
            has_edge = outgoing.edge_ids.new_zeros((num_graphs,), dtype=torch.bool)
            return OutgoingEdges(edge_ids=empty, edge_batch=empty, edge_counts=edge_counts, has_edge=has_edge)
        return apply_edge_policy_mask(outgoing=outgoing, edge_policy_mask=policy_mask, num_graphs=num_graphs)

    @staticmethod
    def _compute_cosine_potential(
        *,
        question_tokens: torch.Tensor,
        state_vec_seq: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if state_vec_seq.dim() != 3:
            raise ValueError("state_vec_seq must be [B, T, H] for potential shaping.")
        batch = int(state_vec_seq.size(0))
        if question_tokens.dim() < 2:
            raise ValueError("question_tokens must be at least [B, H] for potential shaping.")
        q_tokens = question_tokens.to(device=state_vec_seq.device, dtype=state_vec_seq.dtype).reshape(batch, -1)
        if q_tokens.size(0) != batch:
            raise ValueError("question_tokens batch mismatch with state_vec_seq.")
        if q_tokens.size(1) != state_vec_seq.size(2):
            raise ValueError("question_tokens hidden dim mismatch with state_vec_seq.")
        q_norm = torch.linalg.norm(q_tokens, dim=1).clamp(min=eps)
        s_norm = torch.linalg.norm(state_vec_seq, dim=2).clamp(min=eps)
        dot = (state_vec_seq * q_tokens.unsqueeze(1)).sum(dim=2)
        denom = q_norm.unsqueeze(1) * s_norm
        return dot / denom

    @classmethod
    def _compute_potential_shaping_sum(
        cls,
        *,
        question_tokens: torch.Tensor,
        state_vec_seq: torch.Tensor,
        move_mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        phi = cls._compute_cosine_potential(
            question_tokens=question_tokens,
            state_vec_seq=state_vec_seq,
            eps=eps,
        )
        if phi.shape != move_mask.shape:
            raise ValueError("Potential shaping phi/move_mask shape mismatch.")
        if phi.size(1) <= _ZERO:
            return phi.new_zeros((phi.size(0),))
        delta = phi[:, 1:] - phi[:, :-1]
        pad = delta.new_zeros((delta.size(0), _ONE))
        delta_steps = torch.cat((delta, pad), dim=1)
        mask = move_mask.to(device=delta_steps.device, dtype=delta_steps.dtype)
        return (delta_steps * mask).sum(dim=1)

    def _apply_potential_shaping(
        self,
        *,
        reward: RewardOutput,
        question_tokens: torch.Tensor,
        state_vec_seq: torch.Tensor,
        stats: Any,
        shaping_weight: float,
    ) -> RewardOutput:
        if shaping_weight <= float(_ZERO):
            return reward
        question_tokens = question_tokens.detach()
        state_vec_seq = state_vec_seq.detach()
        shaping_sum = self._compute_potential_shaping_sum(
            question_tokens=question_tokens,
            state_vec_seq=state_vec_seq,
            move_mask=stats.move_mask,
            eps=float(_DEFAULT_SHAPING_EPS),
        )
        log_reward = reward.log_reward + shaping_sum.to(dtype=reward.log_reward.dtype) * float(shaping_weight)
        return RewardOutput(
            log_reward=log_reward,
            success=reward.success,
        )

    @staticmethod
    def _reverse_steps(*, steps: torch.Tensor, stop_idx: torch.Tensor, fill_value: float) -> torch.Tensor:
        if steps.dim() != 2:
            raise ValueError("steps must be [B, T].")
        batch, max_steps = steps.shape
        if batch == _ZERO:
            return steps
        stop_idx = stop_idx.to(device=steps.device, dtype=torch.long).view(-1)
        if stop_idx.numel() != batch:
            raise ValueError("stop_idx batch size mismatch for reverse steps.")
        num_moves = stop_idx.clamp(min=_ZERO, max=max_steps)
        base_idx = torch.arange(max_steps, device=steps.device, dtype=torch.long).view(1, -1).expand(batch, -1)
        mask = base_idx < num_moves.unsqueeze(1)
        rev_idx = num_moves.unsqueeze(1) - _ONE - base_idx
        rev_idx = torch.where(mask, rev_idx, torch.zeros_like(rev_idx))
        gathered = steps.gather(1, rev_idx.clamp(min=_ZERO))
        fill = steps.new_full(steps.shape, float(fill_value))
        return torch.where(mask, gathered, fill)

    @staticmethod
    def _apply_zero_hop_mask(
        *,
        num_moves: torch.Tensor,
        weight: torch.Tensor,
        success_mask: torch.Tensor,
        allow_zero_hop: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_hop = num_moves.to(device=weight.device) == _ZERO
        if allow_zero_hop:
            return weight, success_mask
        weight = torch.where(zero_hop, torch.zeros_like(weight), weight)
        success_mask = success_mask & (~zero_hop)
        return weight, success_mask

    def _compute_log_pb_forward(
        self,
        *,
        log_pf_bwd_steps: torch.Tensor,
        stats_bwd: Any,
    ) -> torch.Tensor:
        masked = torch.where(stats_bwd.move_mask, log_pf_bwd_steps, torch.zeros_like(log_pf_bwd_steps))
        return self._reverse_steps(
            steps=masked,
            stop_idx=stats_bwd.stop_idx,
            fill_value=float(_ZERO),
        )

    @staticmethod
    def _compute_tb_loss_per_graph(
        *,
        log_f_start: torch.Tensor,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        log_reward: torch.Tensor,
        step_mask_incl_stop: torch.Tensor,
        move_mask: torch.Tensor,
    ) -> torch.Tensor:
        log_pf_steps = log_pf_steps.to(dtype=log_reward.dtype)
        log_pb_steps = log_pb_steps.to(dtype=log_reward.dtype)
        log_f_start = log_f_start.to(device=log_reward.device, dtype=log_reward.dtype)
        step_mask = step_mask_incl_stop.to(device=log_pf_steps.device, dtype=torch.bool)
        move_mask = move_mask.to(device=log_pf_steps.device, dtype=torch.bool)
        sum_pf = torch.where(step_mask, log_pf_steps, torch.zeros_like(log_pf_steps)).sum(dim=1)
        sum_pb = torch.where(move_mask, log_pb_steps, torch.zeros_like(log_pb_steps)).sum(dim=1)
        residual = log_f_start + sum_pf - sum_pb - log_reward
        return residual.pow(2)

    # ------------------------- Teacher log-probs (for reflected off-policy) -------------------------

    def _compute_initial_hidden(
        self,
        *,
        actor: GFlowNetActor,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        start_nodes: torch.Tensor,
    ) -> torch.Tensor:
        start_nodes = start_nodes.to(device=node_tokens.device, dtype=torch.long)
        context_nodes = node_tokens.index_select(0, start_nodes.clamp(min=_ZERO))
        valid = start_nodes >= _ZERO
        context_nodes = torch.where(valid.unsqueeze(-1), context_nodes, torch.zeros_like(context_nodes))
        return actor.agent.initialize_state(question_tokens=question_tokens, node_tokens=context_nodes)

    def _compute_teacher_log_pf_steps(
        self,
        *,
        actions: torch.Tensor,
        actor: GFlowNetActor,
        graph: dict[str, torch.Tensor],
        start_nodes: torch.Tensor,
        temperature: Optional[float],
    ) -> tuple[torch.Tensor, RolloutDiagnostics]:
        if actions.dim() != 2:
            raise ValueError("actions must be [B, T] for teacher forcing.")
        num_graphs, num_steps = actions.shape
        device = actions.device
        if num_graphs <= _ZERO or num_steps <= _ZERO:
            empty = torch.zeros((num_graphs, num_steps), device=device, dtype=torch.float32)
            diag = self._build_rollout_diagnostics(
                diag_steps=self._init_rollout_diag_steps(),
                num_graphs=num_graphs,
                num_steps=num_steps,
                device=device,
                dtype=empty.dtype,
            )
            return empty, diag
        temp, _ = actor.resolve_temperature(temperature)
        stats = derive_trajectory(actions_seq=actions, stop_value=STOP_RELATION)
        state_nodes = self._build_state_nodes(actions=actions, edge_index=graph["edge_index"], start_nodes=start_nodes)
        state_vec = self._encode_state_sequence(
            actor=actor,
            graph=graph,
            actions=actions,
            stats=stats,
            start_nodes=start_nodes,
        )
        num_states = int(num_graphs * num_steps)
        state_nodes_flat = state_nodes.reshape(-1)
        active_flat = stats.step_mask_incl_stop.reshape(-1)
        outgoing = self._gather_policy_outgoing(
            curr_nodes=state_nodes_flat,
            active_mask=active_flat,
            graph=graph,
        )
        if outgoing.edge_ids.numel() == _ZERO:
            edge_scores = outgoing.edge_ids.new_empty((_ZERO,), dtype=state_vec.dtype)
            edge_valid_mask = outgoing.edge_ids.new_empty((_ZERO,), dtype=torch.bool)
        else:
            step_ids = torch.arange(num_steps, device=device, dtype=torch.long).view(1, -1).expand(num_graphs, -1)
            horizon_exhausted = (step_ids >= int(self.env.max_steps)).reshape(-1)
            edge_valid_mask = ~horizon_exhausted.index_select(0, outgoing.edge_batch)
            action_keys = self._resolve_action_keys_from_graph(actor=actor, graph=graph)
            edge_scores = actor._compute_edge_scores(
                state_vec=state_vec.reshape(num_states, -1),
                action_keys=action_keys,
                edge_batch=outgoing.edge_batch,
                edge_ids=outgoing.edge_ids,
                graph=graph,
            )
        safe_nodes = state_nodes_flat.clamp(min=_ZERO)
        stop_node_mask = graph[STOP_NODE_MASK_KEY].to(device=device, dtype=torch.bool).view(-1)
        is_target = stop_node_mask.index_select(0, safe_nodes) & (state_nodes_flat >= _ZERO)
        last_step = (step_ids + _ONE >= int(self.env.max_steps)).reshape(-1)
        allow_stop = active_flat & (is_target | (~outgoing.has_edge) | last_step)
        stop_logits = actor._compute_stop_logits(
            state_vec=state_vec.reshape(num_states, -1),
            edge_scores=edge_scores,
            edge_batch=outgoing.edge_batch,
            num_graphs=num_states,
            edge_valid_mask=edge_valid_mask,
        )
        policy = compute_forward_log_probs(
            edge_scores=edge_scores,
            stop_logits=stop_logits,
            allow_stop=allow_stop,
            edge_batch=outgoing.edge_batch,
            num_graphs=num_states,
            temperature=temp,
            edge_valid_mask=edge_valid_mask,
        )
        max_edge_score, _ = actor._max_edge_score(
            edge_scores=edge_scores,
            edge_batch=outgoing.edge_batch,
            num_graphs=num_states,
            edge_valid_mask=edge_valid_mask,
        )
        stop_margin = policy.stop - policy.not_stop
        stop_margin = torch.where(policy.has_edge, stop_margin, torch.zeros_like(stop_margin))
        actions_flat = actions.reshape(-1).to(dtype=torch.long)
        choose_stop = actions_flat == STOP_RELATION
        if outgoing.edge_ids.numel() == _ZERO:
            selected_edge_lp = torch.full_like(policy.stop, neg_inf_value(policy.stop))
        else:
            desired = actions_flat.index_select(0, outgoing.edge_batch)
            match = outgoing.edge_ids == desired
            neg_inf = neg_inf_value(policy.edge)
            matched_lp = torch.where(match, policy.edge, torch.full_like(policy.edge, neg_inf))
            selected_edge_lp, _ = segment_max(matched_lp, outgoing.edge_batch, num_states)
        step_lp = torch.where(choose_stop, policy.stop, selected_edge_lp)
        step_lp = torch.where(active_flat, step_lp, torch.zeros_like(step_lp))
        log_pf_steps = step_lp.reshape(num_graphs, num_steps)
        diag = RolloutDiagnostics(
            has_edge_seq=policy.has_edge.reshape(num_graphs, num_steps),
            stop_margin_seq=stop_margin.reshape(num_graphs, num_steps),
            allow_stop_seq=allow_stop.reshape(num_graphs, num_steps),
            max_edge_score_seq=max_edge_score.reshape(num_graphs, num_steps),
            stop_logit_seq=stop_logits.reshape(num_graphs, num_steps),
        )
        return log_pf_steps, diag

    def _summarize_tb_stats(
        self,
        *,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        move_mask: torch.Tensor,
        num_moves: torch.Tensor,
        graph_mask: torch.Tensor,
        diag_spec: _TBDiagSpec,
    ) -> dict[str, torch.Tensor]:
        move_mask = move_mask.to(device=log_pf_steps.device, dtype=torch.bool)
        move_count = move_mask.to(dtype=log_pf_steps.dtype).sum()
        log_pf_clamped = log_pf_steps.clamp(min=float(diag_spec.log_prob_min))
        log_pb_clamped = log_pb_steps.clamp(min=float(diag_spec.log_prob_min))
        log_prob_clip = (log_pf_steps < float(diag_spec.log_prob_min)) | (log_pb_steps < float(diag_spec.log_prob_min))
        log_prob_clip = log_prob_clip & move_mask
        log_prob_clip_rate = self._safe_div(log_prob_clip.to(dtype=log_pf_steps.dtype).sum(), move_count)
        step_delta_raw = torch.abs(log_pf_clamped - log_pb_clamped)
        delta_clip = (step_delta_raw > float(diag_spec.delta_max)) & move_mask
        delta_clip_rate = self._safe_div(delta_clip.to(dtype=log_pf_steps.dtype).sum(), move_count)
        step_delta = step_delta_raw.clamp(max=float(diag_spec.delta_max))
        delta_sum = step_delta.sum(dim=1)
        moves = num_moves.to(device=delta_sum.device, dtype=delta_sum.dtype)
        moves_safe = moves.clamp(min=_ONE)
        avg_delta = torch.where(moves > _ZERO, delta_sum / moves_safe, torch.zeros_like(delta_sum))
        graph_mask = graph_mask.to(device=avg_delta.device, dtype=torch.bool)
        active_graphs = graph_mask & (moves > _ZERO)
        num_moves_f = num_moves.to(dtype=log_pf_steps.dtype)
        log_pf_mean = self._safe_div((log_pf_clamped * move_mask.to(dtype=log_pf_steps.dtype)).sum(), move_count)
        log_pb_mean = self._safe_div((log_pb_clamped * move_mask.to(dtype=log_pf_steps.dtype)).sum(), move_count)
        return {
            "zero_hop_rate": self._safe_div((moves == _ZERO).to(dtype=log_pf_steps.dtype)[graph_mask].sum(), graph_mask.sum()),
            "num_moves_mean": self._masked_mean(num_moves_f, graph_mask),
            "num_moves_p50": self._masked_quantile(num_moves_f, graph_mask, _METRIC_P50),
            "num_moves_p90": self._masked_quantile(num_moves_f, graph_mask, _METRIC_P90),
            "log_pf_mean": log_pf_mean,
            "log_pb_mean": log_pb_mean,
            "avg_delta_mean": self._masked_mean(avg_delta, active_graphs),
            "avg_delta_p50": self._masked_quantile(avg_delta, active_graphs, _METRIC_P50),
            "avg_delta_p90": self._masked_quantile(avg_delta, active_graphs, _METRIC_P90),
            "delta_clip_rate": delta_clip_rate,
            "log_prob_clip_rate": log_prob_clip_rate,
        }

    def _compute_stop_rate_stats(
        self,
        *,
        has_edge_seq: torch.Tensor,
        allow_stop_seq: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        active_mask = active_mask.to(device=has_edge_seq.device, dtype=torch.bool)
        active_count = active_mask.to(dtype=torch.float32).sum()
        has_edge_rate = self._safe_div((has_edge_seq & active_mask).to(dtype=torch.float32).sum(), active_count)
        allow_stop_rate = self._safe_div((allow_stop_seq & active_mask).to(dtype=torch.float32).sum(), active_count)
        return {
            "has_edge_rate": has_edge_rate,
            "allow_stop_rate": allow_stop_rate,
        }

    def _compute_stop_margin_stats(
        self,
        *,
        stop_margin_seq: torch.Tensor,
        has_edge_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "stop_margin_mean": self._masked_mean(stop_margin_seq, has_edge_mask),
            "stop_margin_p50": self._masked_quantile(stop_margin_seq, has_edge_mask, _METRIC_P50),
            "stop_margin_p90": self._masked_quantile(stop_margin_seq, has_edge_mask, _METRIC_P90),
        }

    def _compute_stop_logit_stats(
        self,
        *,
        stop_logit_seq: torch.Tensor,
        max_edge_score_seq: torch.Tensor,
        active_mask: torch.Tensor,
        has_edge_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "stop_logit_mean": self._masked_mean(stop_logit_seq, active_mask),
            "max_edge_score_mean": self._masked_mean(max_edge_score_seq, has_edge_mask),
        }

    def _compute_stop_event_rates(
        self,
        *,
        stop_idx: torch.Tensor,
        success_mask: torch.Tensor,
        has_edge_seq: torch.Tensor,
        max_steps: int,
    ) -> dict[str, torch.Tensor]:
        stop_idx = stop_idx.to(device=has_edge_seq.device, dtype=torch.long).view(-1)
        max_idx = max(int(has_edge_seq.size(1) - _ONE), int(_ZERO))
        stop_idx_safe = stop_idx.clamp(min=_ZERO, max=max_idx)
        stop_has_edge = has_edge_seq.gather(1, stop_idx_safe.view(-1, 1)).squeeze(1)
        horizon_stop = stop_idx >= int(max_steps)
        success_mask = success_mask.to(device=stop_idx.device, dtype=torch.bool)
        dead_end_stop = (~success_mask) & (~stop_has_edge) & (~horizon_stop)
        return {
            "stop_at_target_rate": success_mask.to(dtype=torch.float32).mean(),
            "stop_at_dead_end_rate": dead_end_stop.to(dtype=torch.float32).mean(),
            "stop_at_horizon_rate": horizon_stop.to(dtype=torch.float32).mean(),
        }

    def _summarize_stop_diagnostics(
        self,
        *,
        diag: RolloutDiagnostics,
        step_mask: torch.Tensor,
        stop_idx: torch.Tensor,
        success_mask: torch.Tensor,
        max_steps: int,
    ) -> dict[str, torch.Tensor]:
        active_mask = step_mask.to(device=diag.has_edge_seq.device, dtype=torch.bool)
        allow_stop_mask = diag.allow_stop_seq.to(dtype=torch.bool) & active_mask
        has_edge_mask = diag.has_edge_seq.to(dtype=torch.bool) & allow_stop_mask
        metrics = {}
        metrics.update(self._compute_stop_rate_stats(has_edge_seq=diag.has_edge_seq, allow_stop_seq=diag.allow_stop_seq, active_mask=active_mask))
        metrics.update(self._compute_stop_margin_stats(stop_margin_seq=diag.stop_margin_seq, has_edge_mask=has_edge_mask))
        metrics.update(self._compute_stop_logit_stats(stop_logit_seq=diag.stop_logit_seq, max_edge_score_seq=diag.max_edge_score_seq, active_mask=allow_stop_mask, has_edge_mask=has_edge_mask))
        metrics.update(self._compute_stop_event_rates(stop_idx=stop_idx, success_mask=success_mask, has_edge_seq=diag.has_edge_seq, max_steps=max_steps))
        return metrics

    # ------------------------- Training loss -------------------------

    def _build_graph_cache(
        self,
        *,
        prepared: _PreparedBatch,
        node_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        target_node_locals: torch.Tensor,
        target_ptr: torch.Tensor,
        edge_policy_mask: torch.Tensor,
        stop_node_mask: torch.Tensor,
        log_f_nodes: torch.Tensor,
        actor: GFlowNetActor,
    ) -> dict[str, torch.Tensor]:
        question_tokens = actor.condition_question_tokens(question_tokens)
        graph: dict[str, torch.Tensor] = {
            "node_ptr": prepared.node_ptr,
            "edge_index": prepared.edge_index,
            "edge_relations": prepared.edge_relations,
            "edge_batch": prepared.edge_batch,
            "edge_ptr": prepared.edge_ptr,
            "node_tokens": node_tokens,
            "relation_tokens": relation_tokens,
            "question_tokens": question_tokens,
            "dummy_mask": prepared.dummy_mask,
            "start_node_locals": start_node_locals,
            "start_ptr": start_ptr,
            "target_node_locals": target_node_locals,
            "target_ptr": target_ptr,
            "node_batch": prepared.node_batch,
            "edge_ids_by_head": prepared.edge_ids_by_head,
            "edge_ptr_by_head": prepared.edge_ptr_by_head,
            "log_f_nodes": log_f_nodes,
        }
        graph[EDGE_POLICY_MASK_KEY] = edge_policy_mask.to(device=node_tokens.device, dtype=torch.bool).view(-1)
        graph[STOP_NODE_MASK_KEY] = stop_node_mask.to(device=node_tokens.device, dtype=torch.bool).view(-1)
        if self._cache_action_keys:
            mode = actor.default_mode
            keys = actor.agent.precompute_action_keys(
                relation_tokens=relation_tokens,
                node_tokens=node_tokens,
                edge_index=prepared.edge_index,
                question_tokens=question_tokens,
                edge_batch=prepared.edge_batch,
            )
            cache_key = "action_keys_backward" if mode == "backward" else "action_keys_forward"
            graph[cache_key] = keys
        return graph

    def _sample_one_answer_start(self, prepared: _PreparedBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_ptr = prepared.a_ptr.to(dtype=torch.long).view(-1)
        a_local = prepared.a_local_indices.to(dtype=torch.long).view(-1)
        num_graphs = int(a_ptr.numel() - _ONE)
        counts = (a_ptr[_ONE:] - a_ptr[:-_ONE]).clamp(min=_ZERO)
        if bool((counts <= _ZERO).any().detach().tolist()):
            raise ValueError("Answer nodes missing; cannot sample miner starts.")
        offsets = torch.floor(torch.rand((num_graphs,), device=a_ptr.device) * counts.to(dtype=torch.float32)).to(dtype=torch.long)
        starts = a_ptr[:-_ONE]
        sample_idx = starts + offsets
        sampled = a_local.index_select(0, sample_idx)
        ptr = torch.arange(num_graphs + _ONE, device=a_ptr.device, dtype=torch.long)
        return sampled, ptr, sampled

    def _compute_graph_masks(
        self,
        prepared: _PreparedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_mask = ~prepared.dummy_mask
        num_nodes_total = int(prepared.node_ptr[-1].detach().tolist())
        node_is_answer = self._build_node_is_target(num_nodes_total, prepared.a_local_indices)
        node_is_question = self._build_node_is_target(num_nodes_total, prepared.q_local_indices)
        edge_policy_forward = (~prepared.edge_is_inverse).to(dtype=torch.bool)
        edge_policy_backward = prepared.edge_is_inverse.to(dtype=torch.bool)
        return graph_mask, node_is_answer, node_is_question, edge_policy_forward, edge_policy_backward

    @staticmethod
    def _resolve_backward_success_mask(
        *,
        stop_nodes: torch.Tensor,
        node_is_question: torch.Tensor,
        graph_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = stop_nodes >= _ZERO
        safe_nodes = stop_nodes.clamp(min=_ZERO)
        is_target = node_is_question.index_select(0, safe_nodes) & valid
        return is_target & graph_mask.to(device=stop_nodes.device, dtype=torch.bool)

    def _compute_backward_stats_and_success(
        self,
        *,
        prepared: _PreparedBatch,
        actions_bwd: torch.Tensor,
        start_bwd_nodes: torch.Tensor,
        node_is_question: torch.Tensor,
        graph_mask: torch.Tensor,
    ) -> tuple[Any, torch.Tensor, torch.Tensor]:
        stats_bwd = derive_trajectory(actions_seq=actions_bwd, stop_value=STOP_RELATION)
        state_nodes_bwd = self._build_state_nodes(
            actions=actions_bwd,
            edge_index=prepared.edge_index,
            start_nodes=start_bwd_nodes,
        )
        stop_nodes_bwd = state_nodes_bwd.gather(1, stats_bwd.stop_idx.view(-1, 1)).squeeze(1)
        success_mask = self._resolve_backward_success_mask(
            stop_nodes=stop_nodes_bwd,
            node_is_question=node_is_question,
            graph_mask=graph_mask,
        )
        return stats_bwd, stop_nodes_bwd, success_mask

    def _rollout_backward_trajectory(
        self,
        *,
        prepared: _PreparedBatch,
        graph_bwd: dict[str, torch.Tensor],
        node_is_question: torch.Tensor,
        graph_mask: torch.Tensor,
        policy_temperature: float,
    ) -> tuple[
        torch.Tensor,
        Any,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[RolloutDiagnostics],
        dict[str, torch.Tensor],
    ]:
        start_bwd_locals, start_bwd_ptr, start_bwd_nodes = self._sample_one_answer_start(prepared)
        graph_bwd_rollout = dict(graph_bwd)
        graph_bwd_rollout["start_node_locals"] = start_bwd_locals
        graph_bwd_rollout["start_ptr"] = start_bwd_ptr
        rollout_bwd = self.actor_backward.rollout(
            graph=graph_bwd_rollout,
            temperature=policy_temperature,
            record_actions=True,
            record_diagnostics=True,
            max_steps_override=None,
            mode="backward",
            init_node_locals=start_bwd_locals,
            init_ptr=start_bwd_ptr,
        )
        actions_bwd = rollout_bwd.actions_seq
        if actions_bwd is None:
            raise RuntimeError("actor_backward.rollout must return actions_seq when record_actions=True.")
        stats_bwd, stop_nodes_bwd, success_mask = self._compute_backward_stats_and_success(
            prepared=prepared,
            actions_bwd=actions_bwd,
            start_bwd_nodes=start_bwd_nodes,
            node_is_question=node_is_question,
            graph_mask=graph_mask,
        )
        return (
            actions_bwd,
            stats_bwd,
            start_bwd_nodes,
            stop_nodes_bwd,
            success_mask,
            rollout_bwd.diagnostics,
            graph_bwd_rollout,
        )

    def _compute_forward_terms_from_backward(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        graph_bwd: dict[str, torch.Tensor],
        actions_bwd: torch.Tensor,
        stats_bwd: Any,
        stop_nodes_bwd: torch.Tensor,
        start_bwd_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, Optional[RolloutDiagnostics]]:
        reflected, _ = reflect_backward_to_forward(
            actions_seq=actions_bwd,
            stop_idx=stats_bwd.stop_idx,
            edge_inverse_map=prepared.inverse_edge_ids,
            stop_value=STOP_RELATION,
        )
        stats_fwd = derive_trajectory(actions_seq=reflected, stop_value=STOP_RELATION)
        log_pf_steps, fwd_diag = self._compute_teacher_log_pf_steps(
            actions=reflected,
            actor=self.actor,
            graph=graph_fwd,
            start_nodes=stop_nodes_bwd,
            temperature=None,
        )
        log_pb_raw, _ = self._compute_teacher_log_pf_steps(
            actions=actions_bwd,
            actor=self.actor_backward,
            graph=graph_bwd,
            start_nodes=start_bwd_nodes,
            temperature=None,
        )
        log_pb_steps = self._compute_log_pb_forward(
            log_pf_bwd_steps=log_pb_raw,
            stats_bwd=stats_bwd,
        )
        return reflected, stats_fwd, log_pf_steps, log_pb_steps, fwd_diag

    def _compute_reward_from_states(
        self,
        *,
        prepared: _PreparedBatch,
        state_nodes: torch.Tensor,
        stop_idx: torch.Tensor,
        node_is_target: torch.Tensor,
    ) -> RewardOutput:
        stop_locals = self._compute_stop_node_locals(
            state_nodes=state_nodes,
            stop_idx=stop_idx,
            node_ptr=prepared.node_ptr,
        )
        return self.reward_fn(
            node_ptr=prepared.node_ptr,
            stop_node_locals=stop_locals,
            dummy_mask=prepared.dummy_mask,
            node_is_target=node_is_target,
        )

    def _update_stop_diagnostics(
        self,
        *,
        diag: dict[str, torch.Tensor],
        rollout_diag: Optional[RolloutDiagnostics],
        stats: Any,
        success_mask: torch.Tensor,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        if rollout_diag is None:
            return diag
        stats_dict = self._summarize_stop_diagnostics(
            diag=rollout_diag,
            step_mask=stats.step_mask_incl_stop,
            stop_idx=stats.stop_idx,
            success_mask=success_mask,
            max_steps=int(self.env.max_steps),
        )
        diag.update(self._apply_metric_prefix(stats_dict, prefix))
        return diag

    def _compute_tb_loss_and_diag(
        self,
        *,
        log_f_start: torch.Tensor,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        log_reward: torch.Tensor,
        stats: Any,
        graph_mask: torch.Tensor,
        diag_spec: _TBDiagSpec,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_per_graph = self._compute_tb_loss_per_graph(
            log_f_start=log_f_start,
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            log_reward=log_reward,
            step_mask_incl_stop=stats.step_mask_incl_stop,
            move_mask=stats.move_mask,
        )
        diag = self._summarize_tb_stats(
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            move_mask=stats.move_mask,
            num_moves=stats.num_moves,
            graph_mask=graph_mask,
            diag_spec=diag_spec,
        )
        return loss_per_graph, diag

    def _compute_tb_backward_inputs(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        graph_bwd: dict[str, torch.Tensor],
        graph_mask: torch.Tensor,
        node_is_answer: torch.Tensor,
        node_is_question: torch.Tensor,
        policy_temperature: float,
        shaping_weight: float,
    ) -> _TBBackwardInputs:
        actions_bwd, stats_bwd, start_bwd_nodes, stop_nodes_bwd, success_mask, bwd_diag, graph_bwd_rollout = (
            self._rollout_backward_trajectory(
                prepared=prepared,
                graph_bwd=graph_bwd,
                node_is_question=node_is_question,
                graph_mask=graph_mask,
                policy_temperature=policy_temperature,
            )
        )
        reflected, stats_fwd, log_pf_steps, log_pb_steps, fwd_diag = self._compute_forward_terms_from_backward(
            prepared=prepared,
            graph_fwd=graph_fwd,
            graph_bwd=graph_bwd_rollout,
            actions_bwd=actions_bwd,
            stats_bwd=stats_bwd,
            stop_nodes_bwd=stop_nodes_bwd,
            start_bwd_nodes=start_bwd_nodes,
        )
        state_nodes_fwd = self._build_state_nodes(
            actions=reflected,
            edge_index=prepared.edge_index,
            start_nodes=stop_nodes_bwd,
        )
        reward_fwd = self._compute_reward_from_states(
            prepared=prepared,
            state_nodes=state_nodes_fwd,
            stop_idx=stats_fwd.stop_idx,
            node_is_target=node_is_answer,
        )
        state_nodes_bwd = self._build_state_nodes(
            actions=actions_bwd,
            edge_index=prepared.edge_index,
            start_nodes=start_bwd_nodes,
        )
        reward_bwd = self._compute_reward_from_states(
            prepared=prepared,
            state_nodes=state_nodes_bwd,
            stop_idx=stats_bwd.stop_idx,
            node_is_target=node_is_question,
        )
        if shaping_weight > float(_ZERO):
            state_vec_seq = self._encode_state_sequence(
                actor=self.actor,
                graph=graph_fwd,
                actions=reflected,
                stats=stats_fwd,
                start_nodes=stop_nodes_bwd,
            )
            reward_fwd = self._apply_potential_shaping(
                reward=reward_fwd,
                question_tokens=graph_fwd["question_tokens"],
                state_vec_seq=state_vec_seq,
                stats=stats_fwd,
                shaping_weight=shaping_weight,
            )
        log_reward = reward_fwd.log_reward + reward_bwd.log_reward
        log_f_start = self._gather_log_f_start(
            log_f_nodes=graph_fwd["log_f_nodes"],
            start_nodes=stop_nodes_bwd,
        )
        return _TBBackwardInputs(
            log_pf_steps=log_pf_steps, log_pb_steps=log_pb_steps, stats_fwd=stats_fwd, reward_fwd=reward_fwd,
            log_reward=log_reward, success_mask=success_mask, log_f_start=log_f_start, stats_bwd=stats_bwd,
            bwd_diag=bwd_diag, fwd_diag=fwd_diag,
        )

    def _finalize_tb_backward_terms(
        self,
        *,
        inputs: _TBBackwardInputs,
        graph_mask: torch.Tensor,
        allow_zero_hop: bool,
        diag_spec: _TBDiagSpec,
    ) -> _TBViewTerms:
        loss_per_graph, diag = self._compute_tb_loss_and_diag(
            log_f_start=inputs.log_f_start,
            log_pf_steps=inputs.log_pf_steps,
            log_pb_steps=inputs.log_pb_steps,
            log_reward=inputs.log_reward,
            stats=inputs.stats_fwd,
            graph_mask=graph_mask,
            diag_spec=diag_spec,
        )
        weight = graph_mask.to(dtype=loss_per_graph.dtype)
        weight, success_mask = self._apply_zero_hop_mask(
            num_moves=inputs.stats_fwd.num_moves,
            weight=weight,
            success_mask=inputs.success_mask,
            allow_zero_hop=allow_zero_hop,
        )
        diag = self._update_stop_diagnostics(
            diag=diag,
            rollout_diag=inputs.bwd_diag,
            stats=inputs.stats_bwd,
            success_mask=success_mask,
            prefix="bwd_",
        )
        diag = self._update_stop_diagnostics(
            diag=diag,
            rollout_diag=inputs.fwd_diag,
            stats=inputs.stats_fwd,
            success_mask=inputs.reward_fwd.success,
            prefix="fwd_",
        )
        return _TBViewTerms(
            loss_per_graph=loss_per_graph,
            weight_mask=weight,
            success_mask=success_mask,
            diag=diag,
        )

    def _compute_tb_forward_inputs(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        graph_bwd: dict[str, torch.Tensor],
        graph_mask: torch.Tensor,
        node_is_answer: torch.Tensor,
        shaping_weight: float,
    ) -> _TBForwardInputs:
        actions_fwd, log_pf_steps, stats_fwd, stop_nodes_fwd, reward_fwd, success_mask, fwd_diag, _ = (
            self._rollout_forward_trajectory(
                prepared=prepared,
                graph_fwd=graph_fwd,
                node_is_answer=node_is_answer,
                graph_mask=graph_mask,
                record_state=shaping_weight > float(_ZERO),
                shaping_weight=shaping_weight,
            )
        )
        actions_bwd, stats_bwd, log_pb_steps, bwd_diag = self._compute_backward_terms_from_forward(
            prepared=prepared,
            graph_bwd=graph_bwd,
            actions_fwd=actions_fwd,
            stats_fwd=stats_fwd,
            stop_nodes_fwd=stop_nodes_fwd,
        )
        log_f_start = self._gather_log_f_start(
            log_f_nodes=graph_fwd["log_f_nodes"],
            start_nodes=graph_fwd["start_node_locals"],
        )
        return _TBForwardInputs(
            log_pf_steps=log_pf_steps, log_pb_steps=log_pb_steps, stats_fwd=stats_fwd, reward_fwd=reward_fwd,
            success_mask=success_mask, log_f_start=log_f_start, stats_bwd=stats_bwd, bwd_diag=bwd_diag, fwd_diag=fwd_diag,
            actions_bwd=actions_bwd, stop_nodes_fwd=stop_nodes_fwd,
        )

    def _finalize_tb_forward_terms(
        self,
        *,
        inputs: _TBForwardInputs,
        graph_mask: torch.Tensor,
        node_is_question: torch.Tensor,
        allow_zero_hop: bool,
        diag_spec: _TBDiagSpec,
        prepared: _PreparedBatch,
    ) -> _TBViewTerms:
        loss_per_graph, diag = self._compute_tb_loss_and_diag(
            log_f_start=inputs.log_f_start,
            log_pf_steps=inputs.log_pf_steps,
            log_pb_steps=inputs.log_pb_steps,
            log_reward=inputs.reward_fwd.log_reward,
            stats=inputs.stats_fwd,
            graph_mask=graph_mask,
            diag_spec=diag_spec,
        )
        weight = graph_mask.to(dtype=loss_per_graph.dtype)
        weight, success_mask = self._apply_zero_hop_mask(
            num_moves=inputs.stats_fwd.num_moves,
            weight=weight,
            success_mask=inputs.success_mask,
            allow_zero_hop=allow_zero_hop,
        )
        diag = self._update_stop_diagnostics(
            diag=diag,
            rollout_diag=inputs.fwd_diag,
            stats=inputs.stats_fwd,
            success_mask=inputs.reward_fwd.success,
            prefix="fwd_",
        )
        if inputs.bwd_diag is not None:
            state_nodes_bwd = self._build_state_nodes(
                actions=inputs.actions_bwd,
                edge_index=prepared.edge_index,
                start_nodes=inputs.stop_nodes_fwd,
            )
            stop_nodes_bwd = state_nodes_bwd.gather(1, inputs.stats_bwd.stop_idx.view(-1, 1)).squeeze(1)
            success_bwd = self._resolve_backward_success_mask(
                stop_nodes=stop_nodes_bwd,
                node_is_question=node_is_question,
                graph_mask=graph_mask,
            )
            diag = self._update_stop_diagnostics(
                diag=diag,
                rollout_diag=inputs.bwd_diag,
                stats=inputs.stats_bwd,
                success_mask=success_bwd,
                prefix="bwd_",
            )
        return _TBViewTerms(
            loss_per_graph=loss_per_graph,
            weight_mask=weight,
            success_mask=success_mask,
            diag=diag,
        )

    def _compute_tb_backward_rollout_terms(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        graph_bwd: dict[str, torch.Tensor],
        graph_mask: torch.Tensor,
        node_is_answer: torch.Tensor,
        node_is_question: torch.Tensor,
        policy_temperature: float,
        allow_zero_hop: bool,
        diag_spec: _TBDiagSpec,
        shaping_weight: float,
    ) -> _TBViewTerms:
        inputs = self._compute_tb_backward_inputs(
            prepared=prepared,
            graph_fwd=graph_fwd,
            graph_bwd=graph_bwd,
            graph_mask=graph_mask,
            node_is_answer=node_is_answer,
            node_is_question=node_is_question,
            policy_temperature=policy_temperature,
            shaping_weight=shaping_weight,
        )
        return self._finalize_tb_backward_terms(
            inputs=inputs,
            graph_mask=graph_mask,
            allow_zero_hop=allow_zero_hop,
            diag_spec=diag_spec,
        )

    def _rollout_forward_trajectory(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        node_is_answer: torch.Tensor,
        graph_mask: torch.Tensor,
        record_state: bool,
        shaping_weight: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Any,
        torch.Tensor,
        RewardOutput,
        torch.Tensor,
        Optional[RolloutDiagnostics],
        Optional[torch.Tensor],
    ]:
        rollout_fwd = self.actor.rollout(
            graph=graph_fwd,
            temperature=None,
            record_actions=True,
            record_diagnostics=True,
            record_state=record_state,
            max_steps_override=None,
            mode="forward",
            init_node_locals=graph_fwd["start_node_locals"],
            init_ptr=graph_fwd["start_ptr"],
        )
        actions_fwd = rollout_fwd.actions_seq
        if actions_fwd is None:
            raise RuntimeError("actor.rollout must return actions_seq when record_actions=True.")
        state_vec_seq = rollout_fwd.state_vec_seq
        stats_fwd = derive_trajectory(actions_seq=actions_fwd, stop_value=STOP_RELATION)
        state_nodes_fwd = self._build_state_nodes(
            actions=actions_fwd,
            edge_index=prepared.edge_index,
            start_nodes=graph_fwd["start_node_locals"],
        )
        stop_nodes_fwd = state_nodes_fwd.gather(1, stats_fwd.stop_idx.view(-1, 1)).squeeze(1)
        reward_fwd = self._compute_reward_from_states(
            prepared=prepared,
            state_nodes=state_nodes_fwd,
            stop_idx=stats_fwd.stop_idx,
            node_is_target=node_is_answer,
        )
        if shaping_weight > float(_ZERO):
            if state_vec_seq is None:
                state_vec_seq = self._encode_state_sequence(
                    actor=self.actor,
                    graph=graph_fwd,
                    actions=actions_fwd,
                    stats=stats_fwd,
                    start_nodes=graph_fwd["start_node_locals"],
                )
            reward_fwd = self._apply_potential_shaping(
                reward=reward_fwd,
                question_tokens=graph_fwd["question_tokens"],
                state_vec_seq=state_vec_seq,
                stats=stats_fwd,
                shaping_weight=shaping_weight,
            )
        success_mask = reward_fwd.success & graph_mask
        return (
            actions_fwd,
            rollout_fwd.log_pf_steps,
            stats_fwd,
            stop_nodes_fwd,
            reward_fwd,
            success_mask,
            rollout_fwd.diagnostics,
            rollout_fwd.state_vec_seq,
        )

    def _compute_backward_terms_from_forward(
        self,
        *,
        prepared: _PreparedBatch,
        graph_bwd: dict[str, torch.Tensor],
        actions_fwd: torch.Tensor,
        stats_fwd: Any,
        stop_nodes_fwd: torch.Tensor,
    ) -> tuple[torch.Tensor, Any, torch.Tensor, Optional[RolloutDiagnostics]]:
        actions_bwd, _ = reflect_forward_to_backward(
            actions_seq=actions_fwd,
            stop_idx=stats_fwd.stop_idx,
            edge_inverse_map=prepared.inverse_edge_ids,
            stop_value=STOP_RELATION,
        )
        stats_bwd = derive_trajectory(actions_seq=actions_bwd, stop_value=STOP_RELATION)
        log_pf_bwd_steps, bwd_diag = self._compute_teacher_log_pf_steps(
            actions=actions_bwd,
            actor=self.actor_backward,
            graph=graph_bwd,
            start_nodes=stop_nodes_fwd,
            temperature=None,
        )
        log_pb_steps = self._compute_log_pb_forward(
            log_pf_bwd_steps=log_pf_bwd_steps,
            stats_bwd=stats_bwd,
        )
        return actions_bwd, stats_bwd, log_pb_steps, bwd_diag

    def _compute_tb_forward_rollout_terms(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        graph_bwd: dict[str, torch.Tensor],
        graph_mask: torch.Tensor,
        node_is_answer: torch.Tensor,
        node_is_question: torch.Tensor,
        allow_zero_hop: bool,
        diag_spec: _TBDiagSpec,
        shaping_weight: float,
    ) -> _TBViewTerms:
        inputs = self._compute_tb_forward_inputs(
            prepared=prepared,
            graph_fwd=graph_fwd,
            graph_bwd=graph_bwd,
            graph_mask=graph_mask,
            node_is_answer=node_is_answer,
            shaping_weight=shaping_weight,
        )
        return self._finalize_tb_forward_terms(
            inputs=inputs,
            graph_mask=graph_mask,
            node_is_question=node_is_question,
            allow_zero_hop=allow_zero_hop,
            diag_spec=diag_spec,
            prepared=prepared,
        )

    @staticmethod
    def _merge_metric_totals(
        total: dict[str, torch.Tensor],
        update: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for name, value in update.items():
            if name in total:
                total[name] = total[name] + value
            else:
                total[name] = value
        return total

    def _finalize_tb_view_metrics(
        self,
        *,
        graph_mask: torch.Tensor,
        num_rollouts: int,
        loss_num: torch.Tensor,
        loss_den: torch.Tensor,
        success_total: torch.Tensor,
        diag_total: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        graph_count = graph_mask.to(dtype=loss_num.dtype).sum().clamp(min=float(_ONE))
        denom = graph_count * float(num_rollouts)
        metrics = {
            "success_rate": (success_total / denom).detach(),
            "weight_mean": (loss_den / denom).detach(),
            "loss_num": loss_num.detach(),
            "loss_den": loss_den.detach(),
        }
        if num_rollouts > _ZERO:
            denom_rollouts = torch.tensor(float(num_rollouts), device=loss_num.device, dtype=loss_num.dtype)
            for name, value in diag_total.items():
                metrics[name] = (value / denom_rollouts).detach()
        return metrics

    def _compute_tb_backward_view_loss(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        graph_bwd: dict[str, torch.Tensor],
        graph_mask: torch.Tensor,
        node_is_answer: torch.Tensor,
        node_is_question: torch.Tensor,
        num_rollouts: int,
        policy_temperature: float,
        allow_zero_hop: bool,
        diag_spec: _TBDiagSpec,
        shaping_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        loss_num = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_den = torch.zeros((), device=self.device, dtype=torch.float32)
        success_total = torch.zeros((), device=self.device, dtype=torch.float32)
        diag_total: dict[str, torch.Tensor] = {}
        for _ in range(num_rollouts):
            terms = self._compute_tb_backward_rollout_terms(
                prepared=prepared,
                graph_fwd=graph_fwd,
                graph_bwd=graph_bwd,
                graph_mask=graph_mask,
                node_is_answer=node_is_answer,
                node_is_question=node_is_question,
                policy_temperature=policy_temperature,
                allow_zero_hop=allow_zero_hop,
                diag_spec=diag_spec,
                shaping_weight=shaping_weight,
            )
            loss_per_graph = terms.loss_per_graph
            weight_mask = terms.weight_mask
            if not torch.isfinite(loss_per_graph).all().detach().tolist():
                finite = torch.isfinite(loss_per_graph)
                loss_per_graph = torch.where(finite, loss_per_graph, torch.zeros_like(loss_per_graph))
                weight_mask = torch.where(finite, weight_mask, torch.zeros_like(weight_mask))
            loss_num = loss_num + (loss_per_graph * weight_mask).sum()
            loss_den = loss_den + weight_mask.sum()
            success_total = success_total + terms.success_mask.to(dtype=loss_num.dtype).sum()
            diag_total = self._merge_metric_totals(diag_total, terms.diag)
        metrics = self._finalize_tb_view_metrics(
            graph_mask=graph_mask,
            num_rollouts=num_rollouts,
            loss_num=loss_num,
            loss_den=loss_den,
            success_total=success_total,
            diag_total=diag_total,
        )
        return loss_num, loss_den, self._apply_metric_prefix(metrics, "bwd_")

    def _compute_tb_forward_view_loss(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        graph_bwd: dict[str, torch.Tensor],
        graph_mask: torch.Tensor,
        node_is_answer: torch.Tensor,
        node_is_question: torch.Tensor,
        num_rollouts: int,
        allow_zero_hop: bool,
        diag_spec: _TBDiagSpec,
        shaping_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        loss_num = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_den = torch.zeros((), device=self.device, dtype=torch.float32)
        success_total = torch.zeros((), device=self.device, dtype=torch.float32)
        diag_total: dict[str, torch.Tensor] = {}
        for _ in range(num_rollouts):
            terms = self._compute_tb_forward_rollout_terms(
                prepared=prepared,
                graph_fwd=graph_fwd,
                graph_bwd=graph_bwd,
                graph_mask=graph_mask,
                node_is_answer=node_is_answer,
                node_is_question=node_is_question,
                allow_zero_hop=allow_zero_hop,
                diag_spec=diag_spec,
                shaping_weight=shaping_weight,
            )
            loss_per_graph = terms.loss_per_graph
            weight_mask = terms.weight_mask
            if not torch.isfinite(loss_per_graph).all().detach().tolist():
                finite = torch.isfinite(loss_per_graph)
                loss_per_graph = torch.where(finite, loss_per_graph, torch.zeros_like(loss_per_graph))
                weight_mask = torch.where(finite, weight_mask, torch.zeros_like(weight_mask))
            loss_num = loss_num + (loss_per_graph * weight_mask).sum()
            loss_den = loss_den + weight_mask.sum()
            success_total = success_total + terms.success_mask.to(dtype=loss_num.dtype).sum()
            diag_total = self._merge_metric_totals(diag_total, terms.diag)
        metrics = self._finalize_tb_view_metrics(
            graph_mask=graph_mask,
            num_rollouts=num_rollouts,
            loss_num=loss_num,
            loss_den=loss_den,
            success_total=success_total,
            diag_total=diag_total,
        )
        metrics = self._apply_metric_prefix(metrics, "fwd_")
        return loss_num, loss_den, metrics

    def _compute_tb_training_loss(
        self,
        *,
        prepared: _PreparedBatch,
        graph_fwd: dict[str, torch.Tensor],
        graph_bwd: dict[str, torch.Tensor],
        graph_mask: torch.Tensor,
        node_is_answer: torch.Tensor,
        node_is_question: torch.Tensor,
        num_miner_rollouts: int,
        num_forward_rollouts: int,
        policy_temperature: float,
        allow_zero_hop: bool,
        diag_spec: _TBDiagSpec,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shaping_weight = self._resolve_shaping_weight()
        bwd_num, bwd_den, bwd_metrics = self._compute_tb_backward_view_loss(
            prepared=prepared,
            graph_fwd=graph_fwd,
            graph_bwd=graph_bwd,
            graph_mask=graph_mask,
            node_is_answer=node_is_answer,
            node_is_question=node_is_question,
            num_rollouts=num_miner_rollouts,
            policy_temperature=policy_temperature,
            allow_zero_hop=allow_zero_hop,
            diag_spec=diag_spec,
            shaping_weight=shaping_weight,
        )
        fwd_num, fwd_den, fwd_metrics = self._compute_tb_forward_view_loss(
            prepared=prepared,
            graph_fwd=graph_fwd,
            graph_bwd=graph_bwd,
            graph_mask=graph_mask,
            node_is_answer=node_is_answer,
            node_is_question=node_is_question,
            num_rollouts=num_forward_rollouts,
            allow_zero_hop=allow_zero_hop,
            diag_spec=diag_spec,
            shaping_weight=shaping_weight,
        )
        loss_num = bwd_num + fwd_num
        loss_den = bwd_den + fwd_den
        tb_loss = torch.where(loss_den > float(_ZERO), loss_num / loss_den, torch.zeros_like(loss_num))
        loss = tb_loss
        metrics = {"loss_num": loss_num.detach(), "loss_den": loss_den.detach()}
        metrics.update(bwd_metrics)
        metrics.update(fwd_metrics)
        metrics["shaping_weight"] = torch.tensor(float(shaping_weight), device=loss_num.device, dtype=torch.float32)
        return loss, metrics

    @staticmethod
    def _validate_training_batch(prepared: _PreparedBatch) -> int:
        num_graphs = int(prepared.node_ptr.numel() - _ONE)
        if num_graphs <= _ZERO:
            raise ValueError("Empty batch.")
        if bool(prepared.dummy_mask.any().detach().tolist()):
            raise ValueError(
                "Training batch contains graphs with missing answers (dummy_mask=True). "
                "GFlowNet training must use the -sub dataset (filter_missing_answer=true)."
            )
        return num_graphs

    def _resolve_training_specs(
        self,
    ) -> tuple[int, int, float, bool]:
        num_miner_rollouts = int(self.training_cfg.get("num_miner_rollouts", _ONE))
        if num_miner_rollouts <= _ZERO:
            raise ValueError("training_cfg.num_miner_rollouts must be > 0.")
        num_forward_rollouts = int(self.training_cfg.get("num_forward_rollouts", num_miner_rollouts))
        if num_forward_rollouts <= _ZERO:
            raise ValueError("training_cfg.num_forward_rollouts must be > 0.")
        policy_temperature = float(self.actor.policy_temperature.detach().item())
        allow_zero_hop = self._resolve_allow_zero_hop()
        return num_miner_rollouts, num_forward_rollouts, policy_temperature, allow_zero_hop

    def _build_training_graphs(
        self,
        *,
        prepared: _PreparedBatch,
        edge_policy_forward: torch.Tensor,
        edge_policy_backward: torch.Tensor,
        node_is_answer: torch.Tensor,
        node_is_question: torch.Tensor,
        num_graphs: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        log_f_nodes = self._compute_log_f_nodes(
            log_f_module=self.log_f,
            node_tokens=prepared.node_tokens,
            node_batch=prepared.node_batch,
            question_tokens=prepared.question_tokens,
            flow_features=prepared.flow_features,
        )
        start_ptr_q = torch.arange(num_graphs + _ONE, device=prepared.start_nodes_q.device, dtype=torch.long)
        graph_fwd = self._build_graph_cache(
            prepared=prepared,
            node_tokens=prepared.node_tokens,
            relation_tokens=prepared.relation_tokens,
            question_tokens=prepared.question_tokens,
            start_node_locals=prepared.start_nodes_q,
            start_ptr=start_ptr_q,
            target_node_locals=prepared.a_local_indices,
            target_ptr=prepared.a_ptr,
            edge_policy_mask=edge_policy_forward,
            stop_node_mask=node_is_answer,
            log_f_nodes=log_f_nodes,
            actor=self.actor,
        )
        graph_bwd = self._build_graph_cache(
            prepared=prepared,
            node_tokens=prepared.node_tokens,
            relation_tokens=prepared.relation_tokens,
            question_tokens=prepared.question_tokens,
            start_node_locals=prepared.a_local_indices,
            start_ptr=prepared.a_ptr,
            target_node_locals=prepared.q_local_indices,
            target_ptr=prepared.q_ptr,
            edge_policy_mask=edge_policy_backward,
            stop_node_mask=node_is_question,
            log_f_nodes=log_f_nodes,
            actor=self.actor_backward,
        )
        return graph_fwd, graph_bwd

    @staticmethod
    def _augment_training_metrics(
        metrics: dict[str, torch.Tensor],
        *,
        prepared: _PreparedBatch,
        policy_temperature: float,
    ) -> dict[str, torch.Tensor]:
        edge_counts = (prepared.edge_ptr[_ONE:] - prepared.edge_ptr[:-_ONE]).to(dtype=torch.float32)
        node_counts = (prepared.node_ptr[_ONE:] - prepared.node_ptr[:-_ONE]).to(dtype=torch.float32)
        if prepared.edge_is_inverse.numel() > _ZERO:
            inverse_edge_ratio = prepared.edge_is_inverse.to(dtype=torch.float32).mean()
        else:
            inverse_edge_ratio = edge_counts.new_zeros(())
        metrics.update(
            {
                "edges_per_graph_mean": edge_counts.mean(),
                "nodes_per_graph_mean": node_counts.mean(),
                "inverse_edge_ratio": inverse_edge_ratio,
                "policy_temperature": torch.tensor(policy_temperature, device=edge_counts.device, dtype=torch.float32),
            }
        )
        return metrics

    def _compute_training_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prepared = self._prepare_batch(batch)
        num_graphs = self._validate_training_batch(prepared)
        graph_mask, node_is_answer, node_is_question, edge_policy_forward, edge_policy_backward = self._compute_graph_masks(
            prepared
        )
        num_miner_rollouts, num_forward_rollouts, policy_temperature, allow_zero_hop = self._resolve_training_specs()
        diag_spec = self._resolve_tb_diag_spec()
        graph_fwd, graph_bwd = self._build_training_graphs(
            prepared=prepared,
            edge_policy_forward=edge_policy_forward,
            edge_policy_backward=edge_policy_backward,
            node_is_answer=node_is_answer,
            node_is_question=node_is_question,
            num_graphs=num_graphs,
        )
        loss, metrics = self._compute_tb_training_loss(
            prepared=prepared,
            graph_fwd=graph_fwd,
            graph_bwd=graph_bwd,
            graph_mask=graph_mask,
            node_is_answer=node_is_answer,
            node_is_question=node_is_question,
            num_miner_rollouts=num_miner_rollouts,
            num_forward_rollouts=num_forward_rollouts,
            policy_temperature=policy_temperature,
            allow_zero_hop=allow_zero_hop,
            diag_spec=diag_spec,
        )
        metrics = self._augment_training_metrics(
            metrics,
            prepared=prepared,
            policy_temperature=policy_temperature,
        )
        return loss, metrics

    # ------------------------- Eval -------------------------

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

    def _resolve_dataset_scope(self) -> str:
        datamodule = getattr(self.trainer, "datamodule", None)
        cfg = getattr(datamodule, "dataset_cfg", None) if datamodule is not None else None
        scope = None
        if isinstance(cfg, Mapping):
            scope = cfg.get("dataset_scope")
        if not scope:
            return "unknown"
        return str(scope).strip().lower()

    @torch.no_grad()
    def _compute_eval_metrics(self, batch: Any) -> tuple[dict[str, torch.Tensor], int]:
        prepared = self._prepare_batch(batch)
        num_graphs = int(prepared.node_ptr.numel() - _ONE)
        if num_graphs <= _ZERO:
            return {}, _ZERO
        valid_mask = ~prepared.dummy_mask
        valid_count = int(valid_mask.sum().detach().tolist())
        if valid_count <= _ZERO:
            return {}, _ZERO
        num_nodes_total = int(prepared.node_ptr[-1].detach().tolist())
        node_is_answer = self._build_node_is_target(num_nodes_total, prepared.a_local_indices)
        edge_policy_forward = (~prepared.edge_is_inverse).to(dtype=torch.bool)
        start_ptr_q = torch.arange(num_graphs + _ONE, device=prepared.start_nodes_q.device, dtype=torch.long)
        log_f_nodes = self._compute_log_f_nodes(
            log_f_module=self.log_f,
            node_tokens=prepared.node_tokens,
            node_batch=prepared.node_batch,
            question_tokens=prepared.question_tokens,
            flow_features=prepared.flow_features,
        )
        graph_fwd = self._build_graph_cache(
            prepared=prepared,
            node_tokens=prepared.node_tokens,
            relation_tokens=prepared.relation_tokens,
            question_tokens=prepared.question_tokens,
            start_node_locals=prepared.start_nodes_q,
            start_ptr=start_ptr_q,
            target_node_locals=prepared.a_local_indices,
            target_ptr=prepared.a_ptr,
            edge_policy_mask=edge_policy_forward,
            stop_node_mask=node_is_answer,
            log_f_nodes=log_f_nodes,
            actor=self.actor,
        )
        num_eval_rollouts = int(self.evaluation_cfg["num_eval_rollouts"])
        temperature = float(self.evaluation_cfg.get("rollout_temperature", 1.0))
        hits: list[torch.Tensor] = []
        lengths: list[torch.Tensor] = []
        horizon_stops: list[torch.Tensor] = []
        diag_total: dict[str, torch.Tensor] = {}
        for _ in range(num_eval_rollouts):
            rollout = self.actor.rollout(
                graph=graph_fwd,
                temperature=temperature,
                record_actions=True,
                record_diagnostics=True,
                max_steps_override=None,
                mode="forward",
                init_node_locals=prepared.start_nodes_q,
                init_ptr=start_ptr_q,
            )
            actions_seq = rollout.actions_seq
            if actions_seq is None:
                raise RuntimeError("actor.rollout must return actions_seq for eval.")
            stats = derive_trajectory(actions_seq=actions_seq, stop_value=STOP_RELATION)
            lengths.append(stats.num_moves.to(dtype=torch.float32))
            horizon_stops.append((stats.stop_idx >= int(self.env.max_steps)).to(dtype=torch.float32))
            state_nodes = self._build_state_nodes(
                actions=actions_seq,
                edge_index=prepared.edge_index,
                start_nodes=prepared.start_nodes_q,
            )
            stop_locals = self._compute_stop_node_locals(state_nodes=state_nodes, stop_idx=stats.stop_idx, node_ptr=prepared.node_ptr)
            terminal_hits = gfn_metrics.compute_terminal_hits(
                stop_node_locals=stop_locals,
                node_ptr=prepared.node_ptr,
                node_is_target=node_is_answer,
            )
            hits.append(terminal_hits)
            if rollout.diagnostics is not None:
                diag = self._summarize_stop_diagnostics(
                    diag=rollout.diagnostics,
                    step_mask=stats.step_mask_incl_stop,
                    stop_idx=stats.stop_idx,
                    success_mask=terminal_hits,
                    max_steps=int(self.env.max_steps),
                )
                for name, value in diag.items():
                    if name in diag_total:
                        diag_total[name] = diag_total[name] + value
                    else:
                        diag_total[name] = value
        terminal_hits = torch.stack(hits, dim=0)
        lengths_tensor = torch.stack(lengths, dim=0) if lengths else terminal_hits.new_empty((0, num_graphs), dtype=torch.float32)
        horizon_tensor = (
            torch.stack(horizon_stops, dim=0)
            if horizon_stops
            else terminal_hits.new_empty((0, num_graphs), dtype=torch.float32)
        )
        k_values = [_ONE, num_eval_rollouts]
        metrics = gfn_metrics.compute_terminal_hit_prefixes(terminal_hits=terminal_hits, k_values=k_values)
        pass_rate = terminal_hits.to(dtype=torch.float32).mean(dim=0)
        metrics["pass@1"] = pass_rate
        metrics["length_mean"] = lengths_tensor.mean(dim=0) if lengths_tensor.numel() > _ZERO else pass_rate.new_zeros(pass_rate.shape)
        zero_hop = (lengths_tensor == _ZERO).to(dtype=torch.float32)
        metrics["zero_hop_rate"] = zero_hop.mean(dim=0) if zero_hop.numel() > _ZERO else pass_rate.new_zeros(pass_rate.shape)
        metrics["stop_at_horizon_rate"] = (
            horizon_tensor.mean(dim=0) if horizon_tensor.numel() > _ZERO else pass_rate.new_zeros(pass_rate.shape)
        )
        if num_eval_rollouts > _ZERO:
            denom_rollouts = torch.tensor(float(num_eval_rollouts), device=terminal_hits.device, dtype=torch.float32)
            diag_avg = {name: (value / denom_rollouts) for name, value in diag_total.items()}
            metrics.update(self._apply_metric_prefix(diag_avg, "fwd_"))
        metrics = self._reduce_eval_metrics(metrics, valid_mask=valid_mask)
        return metrics, valid_count


__all__ = ["GFlowNetModule"]
