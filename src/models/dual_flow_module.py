from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Optional

import torch
from lightning import LightningModule

from src.metrics.common import extract_sample_ids
from src.models.components import (
    CvtNodeInitializer,
    EmbeddingBackbone,
    LogZPredictor,
    SRM,
    SinusoidalPositionalEncoding,
)
from src.models.components.graph_ops import (
    OutgoingEdges,
    build_edge_head_csr_from_mask,
    build_edge_tail_csr_from_mask,
    gather_outgoing_edges,
    gumbel_noise_like,
    segment_logsumexp_1d,
    segment_max,
)
from src.utils import log_metric, setup_optimizer
from src.utils.batch_ops import (
    build_dummy_mask,
    build_node_batch,
    build_node_mask,
    edge_reorder_perm,
    reorder_edge_inverse_map,
)
from src.utils.config_utils import require_cfg_mapping, validate_cfg_keys
from src.utils.logging_utils import get_logger, log_event

from src.models.dual_flow_constants import (
    _DB_CFG_KEYS,
    _DB_CFG_OPTIONAL_KEYS,
    _DEFAULT_AVOID_REVISIT,
    _DEFAULT_BACKBONE_FINETUNE,
    _DEFAULT_DIVERSE_BEAM_ENABLED,
    _DEFAULT_DIVERSE_BEAM_GROUPS,
    _DEFAULT_DIVERSE_BEAM_LAMBDA,
    _DEFAULT_DIVERSE_BEAM_PENALTY,
    _DEFAULT_DIVERSE_BEAM_SIMILARITY,
    _DEFAULT_EDGE_DROPOUT,
    _DEFAULT_EDGE_INTER_DIM,
    _DEFAULT_GNN_DROPOUT,
    _DEFAULT_GNN_LAYERS,
    _DEFAULT_INVERSE_REL_SUFFIX,
    _DEFAULT_P0_COSINE_EPS,
    _DEFAULT_P0_MODE,
    _DEFAULT_P0_RESIDUAL,
    _DEFAULT_P0_TEMPERATURE,
    _DEFAULT_ONECYCLE_ANNEAL,
    _DEFAULT_ONECYCLE_BASE_MOMENTUM,
    _DEFAULT_ONECYCLE_CYCLE_MOMENTUM,
    _DEFAULT_ONECYCLE_DIV_FACTOR,
    _DEFAULT_ONECYCLE_FINAL_DIV_FACTOR,
    _DEFAULT_ONECYCLE_MAX_MOMENTUM,
    _DEFAULT_ONECYCLE_PCT_START,
    _DEFAULT_ONECYCLE_THREE_PHASE,
    _DEFAULT_SCHED_ETA_MIN,
    _DEFAULT_SCHED_T0,
    _DEFAULT_SCHED_T_MAX,
    _DEFAULT_SCHED_T_MULT,
    _DEFAULT_START_TEMPERATURE_END,
    _DEFAULT_START_TEMPERATURE_START,
    _DEFAULT_STRICT_INVERSE,
    _DEFAULT_TRAIN_ROLLOUTS,
    _DEFAULT_VALIDATE_EDGE_BATCH,
    _DIVERSE_BEAM_PENALTIES,
    _DIVERSE_BEAM_SIMILARITIES,
    _NEG_ONE,
    _ONE,
    _P0_MODE_DEGREE,
    _P0_MODE_INDEGREE,
    _P0_MODE_NONE,
    _P0_MODE_PREFERENTIAL,
    _P0_MODE_SEMANTIC,
    _P0_MODES,
    _SCHED_INTERVAL_EPOCH,
    _SCHED_INTERVAL_STEP,
    _SCHED_INTERVALS,
    _SCHED_TYPE_COSINE,
    _SCHED_TYPE_COSINE_WARM_RESTARTS,
    _SCHED_TYPE_ONECYCLE,
    _SELF_RELATION_ID,
    _STANDARD_METRICS,
    _TERMINAL_DEAD_END,
    _TERMINAL_HIT,
    _TERMINAL_INVALID_START,
    _TERMINAL_MAX_STEPS,
    _TERMINAL_NONE,
    _THREE,
    _TWO,
    _ZERO,
)
from src.models.dual_flow_types import _BeamCandidateMatrix, _BeamCandidates, _BeamState, _PreparedBatch, _RolloutResult

logger = get_logger(__name__)



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
        self._validate_cfg_contract()
        self._save_serializable_hparams()
        self._p0_cfg = self._resolve_p0_cfg()

        self._cvt_mask = None
        self._relation_inverse_map = None
        self._relation_inverse_mask = None
        self._relation_vocab_size = None

    def _validate_cfg_contract(self) -> None:
        allowed_training = {
            "accumulate_grad_batches",
            "db_cfg",
            "grad_clip_norm",
            "num_rollouts",
            "start_temperature_start",
            "start_temperature_end",
        }
        extra_training = set(self.training_cfg.keys()) - allowed_training
        if extra_training:
            raise ValueError(f"Unsupported training_cfg keys: {sorted(extra_training)}")
        allowed_eval = {"beam_size", "diverse_beam"}
        extra_eval = set(self.evaluation_cfg.keys()) - allowed_eval
        if extra_eval:
            raise ValueError(f"Unsupported evaluation_cfg keys: {sorted(extra_eval)}")

    def _save_serializable_hparams(self) -> None:
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "backbone_fwd",
                "cvt_init_fwd",
                "policy_fwd",
                "forward_ctx_proj",
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

    def _resolve_actor_cfg(self) -> dict[str, float | int | bool]:
        raw = self.actor_cfg or {}
        extra = set(raw.keys()) - {
            "edge_inter_dim",
            "edge_dropout",
            "use_spherical",
            "temperature_init",
            "norm_eps",
            "logit_scale_min",
            "logit_scale_max",
        }
        if extra:
            raise ValueError(f"Unsupported actor_cfg keys: {sorted(extra)}")
        edge_inter_dim = int(raw.get("edge_inter_dim", _DEFAULT_EDGE_INTER_DIM))
        edge_dropout = float(raw.get("edge_dropout", _DEFAULT_EDGE_DROPOUT))
        use_spherical = bool(raw.get("use_spherical", SRM.DEFAULT_USE_SPHERICAL))
        temperature_init = float(raw.get("temperature_init", SRM.DEFAULT_TEMPERATURE_INIT))
        norm_eps = float(raw.get("norm_eps", SRM.DEFAULT_NORM_EPS))
        logit_scale_min = float(raw.get("logit_scale_min", SRM.DEFAULT_LOGIT_SCALE_MIN))
        logit_scale_max = float(raw.get("logit_scale_max", SRM.DEFAULT_LOGIT_SCALE_MAX))
        if edge_inter_dim <= _ZERO:
            raise ValueError("actor_cfg.edge_inter_dim must be > 0.")
        if edge_dropout < float(_ZERO):
            raise ValueError("actor_cfg.edge_dropout must be >= 0.")
        if temperature_init <= float(_ZERO):
            raise ValueError("actor_cfg.temperature_init must be > 0.")
        if norm_eps <= float(_ZERO):
            raise ValueError("actor_cfg.norm_eps must be > 0.")
        if logit_scale_min <= float(_ZERO):
            raise ValueError("actor_cfg.logit_scale_min must be > 0.")
        if logit_scale_max <= logit_scale_min:
            raise ValueError("actor_cfg.logit_scale_max must be > logit_scale_min.")
        return {
            "edge_inter_dim": edge_inter_dim,
            "edge_dropout": edge_dropout,
            "use_spherical": use_spherical,
            "temperature_init": temperature_init,
            "norm_eps": norm_eps,
            "logit_scale_min": logit_scale_min,
            "logit_scale_max": logit_scale_max,
        }

    @staticmethod
    def _coerce_db_cfg(raw: Mapping[str, Any]) -> dict[str, float | int | str]:
        cfg: dict[str, float | int | str] = {
            "sampling_temperature_start": float(raw["sampling_temperature_start"]),
            "sampling_temperature_end": float(raw["sampling_temperature_end"]),
            "dead_end_log_reward": float(raw["dead_end_log_reward"]),
            "dead_end_weight": float(raw["dead_end_weight"]),
            "pb_edge_dropout": float(raw["pb_edge_dropout"]),
        }
        if "dead_end_log_reward_start" in raw and raw.get("dead_end_log_reward_start") is not None:
            cfg["dead_end_log_reward_start"] = float(raw["dead_end_log_reward_start"])
        return cfg

    @staticmethod
    def _validate_db_cfg_values(cfg: Mapping[str, float | int | str]) -> None:
        if float(cfg["sampling_temperature_start"]) <= float(_ZERO) or float(cfg["sampling_temperature_end"]) <= float(
            _ZERO
        ):
            raise ValueError("db_cfg.sampling_temperature_start/end must be > 0.")
        if float(cfg["sampling_temperature_start"]) < float(cfg["sampling_temperature_end"]):
            raise ValueError("db_cfg.sampling_temperature_start must be >= sampling_temperature_end for cosine.")
        if float(cfg["pb_edge_dropout"]) < float(_ZERO) or float(cfg["pb_edge_dropout"]) >= float(_ONE):
            raise ValueError("db_cfg.pb_edge_dropout must satisfy 0 <= p < 1.")
        if float(cfg["dead_end_weight"]) < float(_ZERO):
            raise ValueError("db_cfg.dead_end_weight must be >= 0.")

    def _resolve_db_cfg(self) -> dict[str, float | int | str]:
        raw = require_cfg_mapping(self.training_cfg.get("db_cfg"), "training_cfg.db_cfg")
        validate_cfg_keys(raw, required=_DB_CFG_KEYS, optional=_DB_CFG_OPTIONAL_KEYS, name="db_cfg")
        cfg = self._coerce_db_cfg(raw)
        self._validate_db_cfg_values(cfg)
        return cfg

    def _resolve_sampling_temperature(self) -> float:
        cfg = self._resolve_db_cfg()
        start = float(cfg["sampling_temperature_start"])
        end = float(cfg["sampling_temperature_end"])
        progress = self._resolve_training_progress()
        half = float(_ONE) / float(_TWO)
        cosine = half * (float(_ONE) + math.cos(math.pi * progress))
        return end + (start - end) * cosine

    def _resolve_dead_end_log_reward(self, *, cfg: Optional[Mapping[str, Any]] = None) -> float:
        """Resolve the effective dead-end grounding value.

        We support an optional cosine schedule from `dead_end_log_reward_start` -> `dead_end_log_reward`.
        This keeps the `-infinity` limiting behavior while avoiding overly sharp terminal penalties
        at cold start when most rollouts fail.
        """
        if cfg is None:
            cfg = self._resolve_db_cfg()
        end = float(cfg["dead_end_log_reward"])
        start = cfg.get("dead_end_log_reward_start", None) if isinstance(cfg, Mapping) else None
        if start is None:
            return end
        start = float(start)
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
            datamodule = getattr(trainer, "datamodule", None)
            train_dataset = getattr(datamodule, "train_dataset", None) if datamodule is not None else None
            if datamodule is None or train_dataset is not None:
                max_steps = getattr(trainer, "estimated_stepping_batches", None)
        total_steps = int(max_steps) if max_steps is not None else _ZERO
        if total_steps <= _ZERO:
            return float(_ZERO)
        step = float(getattr(trainer, "global_step", self.global_step))
        progress = step / float(total_steps)
        return min(max(progress, float(_ZERO)), float(_ONE))

    def _resolve_num_rollouts(self) -> int:
        raw = self.training_cfg.get("num_rollouts", _DEFAULT_TRAIN_ROLLOUTS)
        num_rollouts = int(raw)
        if num_rollouts <= _ZERO:
            raise ValueError("training_cfg.num_rollouts must be > 0.")
        return num_rollouts

    def _resolve_start_temperature(self) -> float:
        start = float(self.training_cfg.get("start_temperature_start", _DEFAULT_START_TEMPERATURE_START))
        end = float(self.training_cfg.get("start_temperature_end", _DEFAULT_START_TEMPERATURE_END))
        if start <= float(_ZERO) or end <= float(_ZERO):
            raise ValueError("training_cfg.start_temperature_start/end must be > 0.")
        progress = self._resolve_training_progress()
        half = float(_ONE) / float(_TWO)
        cosine = half * (float(_ONE) + math.cos(math.pi * progress))
        return end + (start - end) * cosine

    def _resolve_p0_cfg(self) -> dict[str, float | str | bool]:
        raw = self.runtime_cfg.get("p0_cfg")
        if raw is None:
            raw = {}
        raw = require_cfg_mapping(raw, "runtime_cfg.p0_cfg")
        mode = str(raw.get("mode", _DEFAULT_P0_MODE) or _DEFAULT_P0_MODE).strip().lower()
        if mode not in _P0_MODES:
            raise ValueError(f"runtime_cfg.p0_cfg.mode must be one of {sorted(_P0_MODES)}, got {mode!r}.")
        residual_enabled = bool(raw.get("residual_enabled", _DEFAULT_P0_RESIDUAL))
        temperature = float(raw.get("temperature", _DEFAULT_P0_TEMPERATURE))
        if temperature <= float(_ZERO):
            raise ValueError("runtime_cfg.p0_cfg.temperature must be > 0.")
        cosine_eps = float(raw.get("cosine_eps", _DEFAULT_P0_COSINE_EPS))
        if cosine_eps <= float(_ZERO):
            raise ValueError("runtime_cfg.p0_cfg.cosine_eps must be > 0.")
        return {
            "mode": mode,
            "residual_enabled": residual_enabled,
            "temperature": temperature,
            "cosine_eps": cosine_eps,
        }

    def _p0_enabled(self) -> bool:
        cfg = self._p0_cfg
        return bool(cfg.get("residual_enabled")) and str(cfg.get("mode")) != _P0_MODE_NONE

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

    def _resolve_diverse_beam_cfg(self) -> dict[str, Any]:
        raw = self.evaluation_cfg.get("diverse_beam", None)
        if raw is None:
            raw = {}
        raw = require_cfg_mapping(raw, "evaluation_cfg.diverse_beam")
        enabled = bool(raw.get("enabled", _DEFAULT_DIVERSE_BEAM_ENABLED))
        groups = int(raw.get("groups", _DEFAULT_DIVERSE_BEAM_GROUPS))
        penalty = str(raw.get("penalty", _DEFAULT_DIVERSE_BEAM_PENALTY)).strip().lower()
        similarity = str(raw.get("similarity", _DEFAULT_DIVERSE_BEAM_SIMILARITY)).strip().lower()
        penalty_lambda = float(raw.get("lambda", _DEFAULT_DIVERSE_BEAM_LAMBDA))
        max_candidates = raw.get("max_candidates_per_graph", None)
        if max_candidates is not None:
            max_candidates = int(max_candidates)
            if max_candidates < _ZERO:
                raise ValueError("evaluation_cfg.diverse_beam.max_candidates_per_graph must be >= 0.")
        if groups <= _ZERO:
            raise ValueError("evaluation_cfg.diverse_beam.groups must be > 0.")
        if penalty not in _DIVERSE_BEAM_PENALTIES:
            raise ValueError(f"diverse_beam.penalty must be one of {sorted(_DIVERSE_BEAM_PENALTIES)}, got {penalty!r}.")
        if similarity not in _DIVERSE_BEAM_SIMILARITIES:
            raise ValueError(
                f"diverse_beam.similarity must be one of {sorted(_DIVERSE_BEAM_SIMILARITIES)}, got {similarity!r}."
            )
        if penalty_lambda < float(_ZERO):
            raise ValueError("evaluation_cfg.diverse_beam.lambda must be >= 0.")
        return {
            "enabled": enabled,
            "groups": groups,
            "penalty": penalty,
            "similarity": similarity,
            "lambda": penalty_lambda,
            "max_candidates_per_graph": max_candidates,
        }

    def _resolve_beam_size(self) -> int:
        return self._resolve_beam_size_value()

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

    def _init_cvt_init(self) -> None:
        enabled = bool(self.cvt_init_cfg.get("enabled", True))
        self._cvt_enabled = enabled
        self.cvt_init_fwd = CvtNodeInitializer()

    def _init_actor(self) -> None:
        actor_cfg = self._resolve_actor_cfg()
        self.policy_fwd = SRM(
            d_plm=self.hidden_dim,
            d_kg=self.hidden_dim,
            d_inter=actor_cfg["edge_inter_dim"],
            dropout=actor_cfg["edge_dropout"],
            use_spherical=actor_cfg["use_spherical"],
            temperature_init=actor_cfg["temperature_init"],
            norm_eps=actor_cfg["norm_eps"],
            logit_scale_min=actor_cfg["logit_scale_min"],
            logit_scale_max=actor_cfg["logit_scale_max"],
        )
        self.forward_ctx_proj = self._build_context_mlp(in_dim=self.hidden_dim * _TWO)
        self.start_selector = self._build_start_selector()
        self.z_time_encoder = SinusoidalPositionalEncoding(self.hidden_dim)
        self.z_predictor = LogZPredictor(hidden_dim=self.hidden_dim, context_dim=self.hidden_dim)

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



    @staticmethod
    def _compute_log_denom(*, logits: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if num_graphs <= 0:
            return torch.zeros((0,), device=logits.device, dtype=logits.dtype)
        edge_batch = edge_batch.view(-1)
        if edge_batch.device != logits.device:
            edge_batch = edge_batch.to(device=logits.device)
        if edge_batch.dtype != torch.long:
            edge_batch = edge_batch.to(dtype=torch.long)
        counts = torch.bincount(edge_batch, minlength=num_graphs)
        log_denom = segment_logsumexp_1d(logits, edge_batch, num_graphs)
        neg_inf = torch.finfo(logits.dtype).min
        return torch.where(counts > 0, log_denom, torch.full_like(log_denom, neg_inf))

    @staticmethod
    def _ensure_tensor(
        value: torch.Tensor,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            return torch.as_tensor(value, dtype=dtype, device=device)
        tensor = value
        if tensor.device != device:
            return tensor.to(device=device, dtype=dtype or tensor.dtype, non_blocking=non_blocking)
        if dtype is not None and tensor.dtype != dtype:
            return tensor.to(dtype=dtype)
        return tensor


    def _resolve_node_is_cvt(
        self,
        node_global_ids: torch.Tensor,
        *,
        num_nodes_total: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not self._cvt_enabled:
            return torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
        cvt_mask = self._cvt_mask
        if cvt_mask is None:
            return torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
        node_global_ids = node_global_ids.view(-1)
        if node_global_ids.numel() != num_nodes_total:
            raise ValueError("node_global_ids length mismatch with ptr.")
        return cvt_mask.to(device=device, dtype=torch.bool).index_select(0, node_global_ids)

    @staticmethod
    def _resolve_context_tokens(context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.dim() == 2:
            return context_tokens
        if context_tokens.dim() == 3 and context_tokens.size(1) == 1:
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

    @staticmethod
    def _build_step_ids(*, num_graphs: int, step: int, device: torch.device) -> torch.Tensor:
        return torch.full((num_graphs,), step, device=device, dtype=torch.long)

    def _select_start_nodes(
        self,
        *,
        question_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        allow_empty: bool,
        name: str,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ptr = ptr.view(-1)
        local_indices = local_indices.view(-1)
        counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
        if not allow_empty:
            torch._assert((counts > 0).all(), f"{name} missing in batch; filter data.")
        num_graphs = counts.numel()
        out = torch.full((num_graphs,), _NEG_ONE, device=local_indices.device, dtype=torch.long)
        hidden_dim = node_tokens.size(-1)
        if local_indices.numel() == 0 or num_graphs == 0:
            zeros = torch.zeros((num_graphs, hidden_dim), device=node_tokens.device, dtype=node_tokens.dtype)
            return out, zeros
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=local_indices.device), counts)
        question_tokens = self._resolve_context_tokens(question_tokens)
        question_sel = question_tokens.index_select(0, graph_ids)
        node_sel = node_tokens.index_select(0, local_indices)
        if temperature <= 0:
            raise ValueError("start_selector temperature must be > 0.")
        logits = self.start_selector(torch.cat((question_sel, node_sel), dim=-1)).view(-1)
        logits_scaled = logits / temperature
        log_denom = segment_logsumexp_1d(logits_scaled, graph_ids, num_graphs)
        soft_weights = torch.exp(logits_scaled - log_denom.index_select(0, graph_ids))
        noise = gumbel_noise_like(torch.zeros_like(logits_scaled, dtype=torch.float32))
        scores = logits_scaled + noise.to(dtype=logits_scaled.dtype)
        _, argmax = segment_max(scores, graph_ids, num_graphs)
        valid = counts > 0
        hard_weights = torch.zeros_like(logits)
        argmax_valid = argmax[valid]
        if argmax_valid.numel() > 0:
            hard_weights.index_put_((argmax_valid,), torch.ones_like(argmax_valid, dtype=logits.dtype))
        # Straight-through: hard selection forward, soft gradients backward.
        weights = hard_weights - soft_weights.detach() + soft_weights
        start_nodes = torch.where(valid, local_indices.index_select(0, argmax), out)
        start_tokens = torch.zeros((num_graphs, hidden_dim), device=node_sel.device, dtype=node_sel.dtype)
        start_tokens.index_add_(0, graph_ids, node_sel * weights.unsqueeze(-1))
        return start_nodes, start_tokens

    @staticmethod
    def _sample_nodes_uniform(
        *,
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        allow_empty: bool,
        name: str,
    ) -> torch.Tensor:
        ptr = ptr.view(-1)
        local_indices = local_indices.view(-1)
        counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
        if not allow_empty:
            torch._assert((counts > 0).all(), f"{name} missing in batch; filter data.")
        num_graphs = counts.numel()
        out = torch.full((num_graphs,), _NEG_ONE, device=local_indices.device, dtype=torch.long)
        if local_indices.numel() == 0 or num_graphs == 0:
            return out
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=local_indices.device), counts)
        scores = gumbel_noise_like(torch.zeros_like(local_indices, dtype=torch.float32))
        _, argmax = segment_max(scores, graph_ids, num_graphs)
        valid = counts > 0
        out = torch.where(valid, local_indices.index_select(0, argmax), out)
        return out

    @staticmethod
    def _extract_graph_stats(batch: Any) -> tuple[int, int]:
        num_graphs = getattr(batch, "num_graphs", None)
        num_nodes_total = getattr(batch, "num_nodes_total", None)
        if num_graphs is None or num_nodes_total is None:
            raise AttributeError("Batch missing num_graphs/num_nodes_total; ensure collate precomputes graph stats.")
        return int(num_graphs), int(num_nodes_total)

    def _extract_graph_tensors(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        node_ptr = getattr(batch, "ptr", None)
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)
        if not torch.is_tensor(node_ptr) or not torch.is_tensor(edge_index) or not torch.is_tensor(edge_attr):
            raise AttributeError("Batch missing ptr/edge_index/edge_attr required for DualFlow.")
        node_ptr = self._ensure_tensor(node_ptr, device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_index = self._ensure_tensor(edge_index, device=device, dtype=torch.long, non_blocking=True)
        edge_relations = self._ensure_tensor(edge_attr, device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_batch = getattr(batch, "edge_batch", None)
        edge_ptr = getattr(batch, "edge_ptr", None)
        if edge_batch is None or edge_ptr is None:
            raise AttributeError(
                "Batch missing edge_batch/edge_ptr; enable data.precompute_edge_batch in the collator."
            )
        edge_batch = self._ensure_tensor(edge_batch, device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_ptr = self._ensure_tensor(edge_ptr, device=device, dtype=torch.long, non_blocking=True).view(-1)
        return node_ptr, edge_index, edge_relations, edge_batch, edge_ptr

    def _extract_index_tensors(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        q_local_indices = getattr(batch, "q_local_indices", None)
        a_local_indices = getattr(batch, "a_local_indices", None)
        if not torch.is_tensor(q_local_indices) or not torch.is_tensor(a_local_indices):
            raise AttributeError("Batch missing q_local_indices/a_local_indices required for DualFlow.")
        q_local_indices = self._ensure_tensor(q_local_indices, device=device, dtype=torch.long, non_blocking=True).view(-1)
        a_local_indices = self._ensure_tensor(a_local_indices, device=device, dtype=torch.long, non_blocking=True).view(-1)
        slice_dict = getattr(batch, "_slice_dict")
        q_ptr = self._ensure_tensor(slice_dict["q_local_indices"], device=device, dtype=torch.long, non_blocking=True).view(-1)
        a_ptr = self._ensure_tensor(slice_dict["a_local_indices"], device=device, dtype=torch.long, non_blocking=True).view(-1)
        answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
        if answer_ptr is None and hasattr(batch, "_slice_dict"):
            answer_ptr = batch._slice_dict.get("answer_entity_ids")
        if answer_ptr is None:
            raise AttributeError("Batch missing answer_entity_ids_ptr required for DualFlow.")
        answer_ptr = self._ensure_tensor(answer_ptr, device=device, dtype=torch.long, non_blocking=True).view(-1)
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
        question_emb = self._ensure_tensor(question_emb, device=device, non_blocking=True)
        node_embeddings = self._ensure_tensor(node_embeddings, device=device, non_blocking=True)
        edge_embeddings = self._ensure_tensor(edge_embeddings, device=device, non_blocking=True)
        return question_emb, node_embeddings, edge_embeddings

    def _prepare_batch(self, batch: Any) -> _PreparedBatch:
        num_graphs, num_nodes_total = self._extract_graph_stats(batch)
        node_ptr, edge_index, edge_relations, edge_batch, edge_ptr = self._extract_graph_tensors(batch)
        q_local_indices, a_local_indices, q_ptr, a_ptr, answer_ptr = self._extract_index_tensors(batch)
        if edge_index.numel() > 0:
            torch._assert((edge_index >= 0).all(), "edge_index contains negative values.")
            torch._assert((edge_index < num_nodes_total).all(), "edge_index out of range for num_nodes_total.")
        if q_local_indices.numel() > 0:
            torch._assert((q_local_indices >= 0).all(), "q_local_indices contains negative values.")
            torch._assert((q_local_indices < num_nodes_total).all(), "q_local_indices out of range.")
        if a_local_indices.numel() > 0:
            torch._assert((a_local_indices >= 0).all(), "a_local_indices contains negative values.")
            torch._assert((a_local_indices < num_nodes_total).all(), "a_local_indices out of range.")
        node_global_ids = getattr(batch, "node_global_ids", None)
        if not torch.is_tensor(node_global_ids):
            raise AttributeError("Batch missing node_global_ids required for DualFlow.")
        node_global_ids = self._ensure_tensor(
            node_global_ids, device=self.device, dtype=torch.long, non_blocking=True
        ).view(-1)
        edge_inverse_map = getattr(batch, "edge_inverse_map", None)
        if not torch.is_tensor(edge_inverse_map):
            raise AttributeError("Batch missing edge_inverse_map; enable data.precompute_edge_inverse_map in collator.")
        edge_inverse_map = self._ensure_tensor(
            edge_inverse_map, device=self.device, dtype=torch.long, non_blocking=True
        ).view(-1)
        if edge_inverse_map.numel() != edge_index.size(1):
            raise ValueError("edge_inverse_map length mismatch with edge_index.")
        dummy_mask = build_dummy_mask(answer_ptr=answer_ptr)
        node_batch = build_node_batch(node_ptr=node_ptr, device=self.device)
        node_is_cvt = self._resolve_node_is_cvt(node_global_ids, num_nodes_total=num_nodes_total, device=self.device)
        question_emb, node_embeddings, edge_embeddings = self._extract_embeddings(
            batch,
            edge_index=edge_index,
            node_is_cvt=node_is_cvt,
        )
        if self._cvt_enabled:
            node_embeddings = self.cvt_init_fwd(
                node_embeddings=node_embeddings,
                relation_embeddings=edge_embeddings,
                edge_index=edge_index,
                node_is_cvt=node_is_cvt,
            )
        if edge_embeddings.size(0) != edge_index.size(1):
            raise ValueError("edge_embeddings length must match edge_index.")
        perm = edge_reorder_perm(
            edge_index=edge_index,
            edge_batch=edge_batch,
            edge_relations=edge_relations,
            node_ptr=node_ptr,
            num_edges_before=edge_index.size(1),
        )
        if perm is not None:
            edge_index = edge_index.index_select(1, perm)
            edge_batch = edge_batch.index_select(0, perm)
            edge_relations = edge_relations.index_select(0, perm)
            edge_embeddings = edge_embeddings.index_select(0, perm)
        edge_inverse_map = reorder_edge_inverse_map(edge_inverse_map=edge_inverse_map, perm=perm)
        node_tokens = self.backbone_fwd.project_node_embeddings(node_embeddings)
        relation_tokens = self.backbone_fwd.project_relation_embeddings(edge_embeddings)
        node_tokens = self.backbone_fwd.encode_graph(
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            edge_index=edge_index,
            num_nodes=num_nodes_total,
        )
        question_tokens_fwd_base = self._resolve_context_tokens(
            self.backbone_fwd.project_question_embeddings(question_emb)
        )
        if edge_ptr.numel() != num_graphs + 1:
            raise ValueError("edge_ptr length mismatch with batch graph count.")
        start_temperature = self._resolve_start_temperature()
        start_nodes_fwd, start_tokens_fwd = self._select_start_nodes(
            question_tokens=question_tokens_fwd_base,
            node_tokens=node_tokens,
            local_indices=q_local_indices,
            ptr=q_ptr,
            allow_empty=False,
            name="q_local_indices",
            temperature=start_temperature,
        )
        context_tokens_fwd = self._build_forward_context(
            question_tokens=question_tokens_fwd_base,
            start_tokens=start_tokens_fwd,
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
        self._validate_edge_csr(
            edge_index=edge_index,
            edge_mask=edge_mask_fwd,
            edge_ids_by_head=edge_ids_by_head_fwd,
            edge_ptr_by_head=edge_ptr_by_head_fwd,
            edge_ids_by_tail=edge_ids_by_tail_fwd,
            edge_ptr_by_tail=edge_ptr_by_tail_fwd,
            num_nodes_total=num_nodes_total,
        )
        self._validate_edge_csr(
            edge_index=edge_index,
            edge_mask=edge_mask_bwd,
            edge_ids_by_head=edge_ids_by_head_bwd,
            edge_ptr_by_head=edge_ptr_by_head_bwd,
            edge_ids_by_tail=edge_ids_by_tail_bwd,
            edge_ptr_by_tail=edge_ptr_by_tail_bwd,
            num_nodes_total=num_nodes_total,
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
        answer_entity_ids = getattr(batch, "answer_entity_ids", None)
        if not torch.is_tensor(answer_entity_ids):
            raise AttributeError("Batch missing answer_entity_ids required for DualFlow.")
        answer_entity_ids = self._ensure_tensor(
            answer_entity_ids, device=self.device, dtype=torch.long, non_blocking=True
        ).view(-1)
        prepared_fwd = _PreparedBatch(
            num_graphs=num_graphs,
            num_nodes_total=num_nodes_total,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_relations=edge_relations,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            question_emb_raw=question_emb,
            edge_embeddings_raw=edge_embeddings,
            node_embeddings=node_embeddings,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
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
        return prepared_fwd

    @staticmethod
    def _build_edge_inverse_mask(*, edge_relations: torch.Tensor, inverse_mask: torch.Tensor) -> torch.Tensor:
        edge_relations = edge_relations.view(-1)
        mask = torch.zeros_like(edge_relations, dtype=torch.bool)
        valid = edge_relations >= 0
        if valid.any():
            mask[valid] = inverse_mask.index_select(0, edge_relations[valid])
        return mask

    @staticmethod
    def _build_edge_direction_mask(
        *,
        edge_is_inverse: torch.Tensor,
        self_loop_mask: torch.Tensor,
        forward: bool,
    ) -> torch.Tensor:
        base = ~edge_is_inverse if forward else edge_is_inverse
        return base | self_loop_mask

    @staticmethod
    def _validate_edge_csr(
        *,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        edge_ids_by_tail: torch.Tensor,
        edge_ptr_by_tail: torch.Tensor,
        num_nodes_total: int,
    ) -> None:
        if edge_index.numel() == 0:
            return
        num_edges = edge_index.size(1)
        mask = edge_mask.to(dtype=torch.bool)
        heads = edge_index[0]
        tails = edge_index[1]
        if edge_ids_by_head.numel() > 0:
            torch._assert((edge_ids_by_head >= 0).all(), "edge_ids_by_head contains negative values.")
            torch._assert((edge_ids_by_head < num_edges).all(), "edge_ids_by_head out of range.")
        if edge_ids_by_tail.numel() > 0:
            torch._assert((edge_ids_by_tail >= 0).all(), "edge_ids_by_tail contains negative values.")
            torch._assert((edge_ids_by_tail < num_edges).all(), "edge_ids_by_tail out of range.")
        expected = mask.sum()
        head_count = torch.tensor(edge_ids_by_head.numel(), device=edge_mask.device, dtype=expected.dtype)
        tail_count = torch.tensor(edge_ids_by_tail.numel(), device=edge_mask.device, dtype=expected.dtype)
        torch._assert(head_count == expected, "edge_ids_by_head length mismatch with edge_mask.")
        torch._assert(tail_count == expected, "edge_ids_by_tail length mismatch with edge_mask.")
        counts_head = torch.bincount(heads[mask], minlength=num_nodes_total)
        ptr_counts_head = edge_ptr_by_head[1:] - edge_ptr_by_head[:-1]
        torch._assert(torch.equal(counts_head, ptr_counts_head), "edge_ptr_by_head mismatch with edge_mask.")
        counts_tail = torch.bincount(tails[mask], minlength=num_nodes_total)
        ptr_counts_tail = edge_ptr_by_tail[1:] - edge_ptr_by_tail[:-1]
        torch._assert(torch.equal(counts_tail, ptr_counts_tail), "edge_ptr_by_tail mismatch with edge_mask.")

    @staticmethod
    def _validate_edge_inverse_map(
        *,
        edge_inverse_map: torch.Tensor,
        edge_relations: torch.Tensor,
        strict: bool,
    ) -> None:
        if not strict:
            return
        if edge_inverse_map.numel() == 0:
            return
        edge_inverse_map = edge_inverse_map.view(-1)
        edge_relations = edge_relations.view(-1)
        missing = (edge_relations >= 0) & (edge_inverse_map < 0)
        torch._assert(~missing.any(), "Missing inverse edges for relation pairs.")
        valid = edge_inverse_map >= 0
        inv_safe = edge_inverse_map[valid]
        idx = torch.arange(edge_inverse_map.numel(), device=edge_inverse_map.device, dtype=edge_inverse_map.dtype)[valid]
        back = edge_inverse_map.index_select(0, inv_safe)
        if not torch.equal(back, idx):
            raise ValueError("Edge inverse map is not symmetric.")



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
        max_batch = node_batch_sel.max()
        steps_num = torch.tensor(steps.numel(), device=max_batch.device)
        torch._assert(steps_num > max_batch, "steps length must cover max node batch index.")
        time_emb = self.z_time_encoder(steps).index_select(0, node_batch_sel)
        node_tokens_sel = node_tokens_sel + time_emb
        return self.z_predictor(
            node_tokens=node_tokens_sel,
            question_tokens=context_tokens,
            node_batch=node_batch_sel,
        )

    def _compute_log_p0(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_decisions: int,
        decision_graph_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cfg = self._p0_cfg
        mode = str(cfg["mode"])
        if mode == _P0_MODE_NONE:
            return torch.zeros_like(edge_batch, dtype=torch.float32)
        if num_decisions <= _ZERO:
            return torch.zeros_like(edge_batch, dtype=torch.float32)
        if mode == _P0_MODE_DEGREE:
            counts = torch.bincount(edge_batch, minlength=num_decisions).clamp(min=_ONE)
            log_p0 = -torch.log(counts.to(dtype=torch.float32)).index_select(0, edge_batch)
            return log_p0.detach()
        if mode == _P0_MODE_INDEGREE:
            tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
            indegree = prepared.edge_ptr_by_tail_fwd[1:] - prepared.edge_ptr_by_tail_fwd[:-1]
            indegree = indegree.to(device=edge_ids.device, dtype=torch.float32)
            counts = indegree.index_select(0, tails.clamp(min=_ZERO)).clamp(min=float(_ONE))
            log_p0 = -torch.log(counts)
            return log_p0.detach()
        if mode == _P0_MODE_PREFERENTIAL:
            tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
            indegree = prepared.edge_ptr_by_tail_fwd[1:] - prepared.edge_ptr_by_tail_fwd[:-1]
            indegree = indegree.to(device=edge_ids.device, dtype=torch.float32)
            counts = indegree.index_select(0, tails.clamp(min=_ZERO)).clamp(min=float(_ONE))
            log_p0 = torch.log(counts)
            return log_p0.detach()
        if mode == _P0_MODE_SEMANTIC:
            tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
            tail_emb = prepared.node_embeddings.index_select(0, tails)
            question_emb = self._resolve_context_tokens(prepared.question_emb_raw)
            if decision_graph_ids is not None:
                decision_graph_ids = decision_graph_ids.to(device=edge_batch.device, dtype=torch.long).view(-1)
                graph_ids = decision_graph_ids.index_select(0, edge_batch)
            else:
                graph_ids = edge_batch
            query = question_emb.index_select(0, graph_ids)
            cosine_eps = float(cfg["cosine_eps"])
            sim = self._cosine_similarity(tail_emb, query, eps=cosine_eps)
            temperature = float(cfg["temperature"])
            base = sim / temperature
            log_denom = self._compute_log_denom(logits=base, edge_batch=edge_batch, num_graphs=num_decisions)
            log_p0 = base - log_denom.index_select(0, edge_batch)
            return log_p0.detach()
        raise ValueError(f"Unsupported p0 mode: {mode!r}.")

    def _compute_edge_logits_components(
        self,
        *,
        policy: SRM,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        apply_p0: bool,
        compute_p0: bool,
        decision_graph_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_ids.numel() == _ZERO:
            empty = torch.zeros((_ZERO,), device=edge_ids.device, dtype=torch.float32)
            return empty, empty, empty
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
        nn_logits = policy(context_edge, head_tokens, relation_tokens, tail_tokens, None)
        log_p0 = torch.zeros_like(nn_logits, dtype=torch.float32)
        if compute_p0 and str(self._p0_cfg["mode"]) != _P0_MODE_NONE:
            num_decisions = int(steps.numel())
            log_p0 = self._compute_log_p0(
                prepared=prepared,
                edge_ids=edge_ids,
                edge_batch=edge_batch,
                num_decisions=num_decisions,
                decision_graph_ids=decision_graph_ids,
            ).to(dtype=nn_logits.dtype)
        logits = nn_logits
        if apply_p0 and self._p0_enabled():
            logits = logits + log_p0
        if temperature != float(_ONE):
            logits = logits / float(temperature)
        return nn_logits, log_p0, logits

    def _compute_edge_logits(
        self,
        *,
        policy: SRM,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        apply_p0: bool = False,
        decision_graph_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, logits = self._compute_edge_logits_components(
            policy=policy,
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            steps=steps,
            temperature=temperature,
            context_tokens=context_tokens,
            apply_p0=apply_p0,
            compute_p0=apply_p0,
            decision_graph_ids=decision_graph_ids,
        )
        return logits

    @staticmethod
    def _cosine_similarity(x: torch.Tensor, y: torch.Tensor, *, eps: float) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        x_norm = x / x.norm(dim=-1, keepdim=True).clamp(min=eps)
        y_norm = y / y.norm(dim=-1, keepdim=True).clamp(min=eps)
        return (x_norm * y_norm).sum(dim=-1)

    def _compute_pb_logits(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=edge_ids.device, dtype=torch.long).view(-1)
        logits = torch.zeros((edge_ids.numel(),), device=edge_ids.device, dtype=torch.float32)
        allowed = torch.ones_like(edge_ids, dtype=torch.bool)
        return logits, allowed

    def _compute_pb_log_prob(
        self,
        *,
        prepared: _PreparedBatch,
        chosen_edge: torch.Tensor,
        parent_nodes: torch.Tensor,
        move_mask: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        visited_nodes: Optional[torch.Tensor] = None,
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
        outgoing = self._apply_action_constraints_to_outgoing(
            outgoing,
            num_graphs=move_mask.numel(),
            edge_index=prepared.edge_index,
            edge_mask=edge_mask,
            visited_nodes=visited_nodes,
        )
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
        )
        num_graphs = move_mask.numel()
        # NOTE: We normalize PB over the full candidate set after applying static masks.
        # Dynamic constraints (e.g., avoid_revisit / pb_edge_dropout) are treated as an
        # approximation to keep training efficient; the residual absorbs the mismatch.
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        chosen_edge_safe = chosen_edge.clamp(min=_ZERO)
        chosen_for_edge = chosen_edge_safe.index_select(0, edge_batch)
        match = edge_ids == chosen_for_edge
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(match, logits, torch.full_like(logits, neg_inf))
        chosen_logits, _ = segment_max(masked, edge_batch, num_graphs)
        log_pb_edge = chosen_logits - log_denom
        allowed_batch = edge_batch[allowed]
        allowed_counts = torch.bincount(allowed_batch, minlength=num_graphs)
        no_allowed = allowed_counts == _ZERO
        log_pb_edge = torch.where(no_allowed, torch.zeros_like(log_pb_edge), log_pb_edge)
        log_pb_step = torch.where(move_mask, log_pb_edge, torch.zeros_like(log_pb_edge))
        if return_no_allowed:
            return log_pb_step, no_allowed
        return log_pb_step

    def _sample_pb_edges(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
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
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        log_probs = logits - log_denom.index_select(0, edge_batch)
        scores = log_probs + gumbel_noise_like(log_probs)
        _, argmax = segment_max(scores, edge_batch, num_graphs)
        chosen_edge = edge_ids.index_select(0, argmax)
        log_prob_chosen = log_probs.index_select(0, argmax)
        allowed_batch = edge_batch[allowed]
        allowed_counts = torch.bincount(allowed_batch, minlength=num_graphs)
        has_allowed = allowed_counts > _ZERO
        return chosen_edge, log_prob_chosen, has_allowed

    def _rollout_pb(
        self,
        *,
        prepared: _PreparedBatch,
        graph_mask: torch.Tensor,
        start_nodes: torch.Tensor,
        node_is_target: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        record_actions: bool,
        record_log_pf: bool,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> _RolloutResult:
        num_graphs = int(prepared.num_graphs)
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
        visited_nodes = None
        if self._avoid_revisit:
            visited_nodes = torch.zeros((prepared.node_batch.numel(),), device=device, dtype=torch.bool)
            visited_nodes.index_fill_(0, curr_nodes[active], True)
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
            outgoing = self._apply_action_constraints_to_outgoing(
                outgoing,
                num_graphs=num_graphs,
                edge_index=prepared.edge_index,
                edge_mask=edge_mask,
                visited_nodes=visited_nodes,
            )
            move_mask = active & outgoing.has_edge
            if outgoing.edge_ids.numel() > _ZERO:
                chosen_edge, log_pf_step, has_allowed = self._sample_pb_edges(
                    prepared=prepared,
                    edge_ids=outgoing.edge_ids,
                    edge_batch=outgoing.edge_batch,
                    num_graphs=num_graphs,
                )
                move_mask = move_mask & has_allowed
                chosen_edge = torch.where(move_mask, chosen_edge, torch.full_like(chosen_edge, _NEG_ONE))
                chosen_tail = prepared.edge_index[_ONE].index_select(0, chosen_edge.clamp(min=_ZERO))
                curr_nodes = torch.where(move_mask, chosen_tail, curr_nodes)
                if visited_nodes is not None:
                    visited_nodes.index_fill_(0, chosen_tail[move_mask], True)
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

    def _sample_pb_edge_dropout_mask(self, *, edge_index: torch.Tensor) -> Optional[torch.Tensor]:
        drop_prob = float(self._resolve_db_cfg()["pb_edge_dropout"])
        if drop_prob <= float(_ZERO):
            return None
        num_edges = int(edge_index.size(1))
        if num_edges <= _ZERO:
            return None
        keep = torch.rand((num_edges,), device=edge_index.device) >= drop_prob
        return keep

    def _sample_edges(
        self,
        *,
        policy: SRM,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        apply_p0: bool,
        collect_policy_metrics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        if edge_ids.numel() == _ZERO:
            zeros = torch.zeros((num_graphs,), device=prepared.edge_index.device, dtype=torch.float32)
            return (
                torch.full((num_graphs,), _NEG_ONE, device=prepared.edge_index.device, dtype=torch.long),
                zeros,
                zeros,
                None,
            )
        compute_p0 = collect_policy_metrics and str(self._p0_cfg["mode"]) != _P0_MODE_NONE
        nn_logits, log_p0, logits = self._compute_edge_logits_components(
            policy=policy,
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            steps=steps + _ONE,
            temperature=temperature,
            context_tokens=context_tokens,
            apply_p0=apply_p0,
            compute_p0=compute_p0,
            decision_graph_ids=None,
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        log_probs = logits - log_denom.index_select(0, edge_batch)
        scores = log_probs + gumbel_noise_like(log_probs)
        _, argmax = segment_max(scores, edge_batch, num_graphs)
        chosen_edge = edge_ids.index_select(0, argmax)
        log_prob_chosen = log_probs.index_select(0, argmax)
        policy_metrics = None
        if collect_policy_metrics:
            edge_count = torch.tensor(nn_logits.numel(), device=nn_logits.device, dtype=torch.float32)
            drift_abs_sum = nn_logits.abs().sum()
            drift_sq_sum = (nn_logits * nn_logits).sum()
            kl_sum = torch.zeros((), device=nn_logits.device, dtype=torch.float32)
            kl_count = torch.zeros((), device=nn_logits.device, dtype=torch.float32)
            if compute_p0:
                pi = torch.exp(log_probs)
                kl_edges = pi * (log_probs - log_p0.to(dtype=log_probs.dtype))
                kl_per_decision = torch.zeros((num_graphs,), device=nn_logits.device, dtype=torch.float32)
                kl_per_decision.index_add_(0, edge_batch, kl_edges.to(dtype=torch.float32))
                counts = torch.bincount(edge_batch, minlength=num_graphs)
                has_edge = counts > _ZERO
                if bool(has_edge.any().detach().tolist()):
                    kl_sum = kl_per_decision[has_edge].sum()
                    kl_count = has_edge.to(dtype=torch.float32).sum()
            policy_metrics = {
                "drift_abs_sum": drift_abs_sum,
                "drift_sq_sum": drift_sq_sum,
                "edge_count": edge_count,
                "kl_sum": kl_sum,
                "kl_count": kl_count,
            }
        return chosen_edge, log_prob_chosen, log_denom, policy_metrics

    def _compute_forward_log_prob(
        self,
        *,
        policy: SRM,
        prepared: _PreparedBatch,
        chosen_edge: torch.Tensor,
        parent_nodes: torch.Tensor,
        move_mask: torch.Tensor,
        steps: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        apply_p0: bool,
        visited_nodes: Optional[torch.Tensor] = None,
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
        outgoing = self._apply_action_constraints_to_outgoing(
            outgoing,
            num_graphs=move_mask.numel(),
            edge_index=prepared.edge_index,
            edge_mask=edge_mask,
            visited_nodes=visited_nodes,
        )
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
            apply_p0=apply_p0,
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=move_mask.numel())
        chosen_edge_safe = chosen_edge.clamp(min=_ZERO)
        chosen_for_edge = chosen_edge_safe.index_select(0, edge_batch)
        match = edge_ids == chosen_for_edge
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(match, logits, torch.full_like(logits, neg_inf))
        chosen_logits, _ = segment_max(masked, edge_batch, move_mask.numel())
        log_pf_edge = chosen_logits - log_denom
        has_edge = outgoing.has_edge.to(device=log_pf_edge.device, dtype=torch.bool)
        log_pf_edge = torch.where(has_edge, log_pf_edge, torch.zeros_like(log_pf_edge))
        log_pf_step = torch.where(move_mask & has_edge, log_pf_edge, torch.zeros_like(log_pf_edge))
        return log_pf_step

    def _rollout_policy(
        self,
        *,
        policy: SRM,
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
        apply_p0: bool,
        collect_policy_metrics: bool = False,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> _RolloutResult:
        num_graphs = int(prepared.num_graphs)
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
        visited_nodes = None
        if self._avoid_revisit:
            visited_nodes = torch.zeros((prepared.node_batch.numel(),), device=device, dtype=torch.bool)
            visited_nodes.index_fill_(0, curr_nodes[active], True)
        stop_nodes = torch.full((num_graphs,), _NEG_ONE, device=device, dtype=torch.long)
        actions = None
        log_pf_steps = None
        policy_accum = None
        out_degree = None
        if collect_policy_metrics:
            policy_accum = {
                "drift_abs_sum": torch.zeros((), device=device, dtype=torch.float32),
                "drift_sq_sum": torch.zeros((), device=device, dtype=torch.float32),
                "edge_count": torch.zeros((), device=device, dtype=torch.float32),
                "kl_sum": torch.zeros((), device=device, dtype=torch.float32),
                "kl_count": torch.zeros((), device=device, dtype=torch.float32),
                "deg_sum": torch.zeros((), device=device, dtype=torch.float32),
                "deg_count": torch.zeros((), device=device, dtype=torch.float32),
            }
            out_degree = (edge_ptr_by_head[1:] - edge_ptr_by_head[:-1]).to(device=device, dtype=torch.float32)
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
            outgoing = self._apply_action_constraints_to_outgoing(
                outgoing,
                num_graphs=num_graphs,
                edge_index=prepared.edge_index,
                edge_mask=edge_mask,
                visited_nodes=visited_nodes,
            )
            move_mask = active & outgoing.has_edge
            if collect_policy_metrics and policy_accum is not None and out_degree is not None:
                if bool(move_mask.any().detach().tolist()):
                    head_nodes = curr_nodes[move_mask].clamp(min=_ZERO)
                    deg_sum = out_degree.index_select(0, head_nodes).sum()
                    policy_accum["deg_sum"] = policy_accum["deg_sum"] + deg_sum
                    policy_accum["deg_count"] = policy_accum["deg_count"] + move_mask.to(dtype=torch.float32).sum()
            if outgoing.edge_ids.numel() > _ZERO:
                step_ids = self._build_step_ids(num_graphs=num_graphs, step=step, device=device)
                chosen_edge, log_pf_step, _, step_metrics = self._sample_edges(
                    policy=policy,
                    prepared=prepared,
                    edge_ids=outgoing.edge_ids,
                    edge_batch=outgoing.edge_batch,
                    num_graphs=num_graphs,
                    steps=step_ids,
                    temperature=temperature,
                    context_tokens=context_tokens,
                    apply_p0=apply_p0,
                    collect_policy_metrics=collect_policy_metrics,
                )
                if collect_policy_metrics and policy_accum is not None and step_metrics is not None:
                    for key, value in step_metrics.items():
                        policy_accum[key] = policy_accum[key] + value.to(device=device)
                chosen_edge = torch.where(outgoing.has_edge, chosen_edge, torch.full_like(chosen_edge, _NEG_ONE))
                chosen_tail = prepared.edge_index[_ONE].index_select(0, chosen_edge.clamp(min=_ZERO))
                curr_nodes = torch.where(move_mask, chosen_tail, curr_nodes)
                if visited_nodes is not None:
                    visited_nodes.index_fill_(0, chosen_tail[move_mask], True)
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
        policy_metrics = None
        if collect_policy_metrics and policy_accum is not None:
            edge_count = policy_accum["edge_count"]
            if edge_count > float(_ZERO):
                drift_abs_mean = policy_accum["drift_abs_sum"] / edge_count
                drift_rms = torch.sqrt(policy_accum["drift_sq_sum"] / edge_count)
            else:
                drift_abs_mean = torch.zeros((), device=device, dtype=torch.float32)
                drift_rms = torch.zeros((), device=device, dtype=torch.float32)
            kl_count = policy_accum["kl_count"]
            if kl_count > float(_ZERO):
                kl_mean = policy_accum["kl_sum"] / kl_count
            else:
                kl_mean = torch.zeros((), device=device, dtype=torch.float32)
            deg_count = policy_accum["deg_count"]
            if deg_count > float(_ZERO):
                degree_mean = policy_accum["deg_sum"] / deg_count
            else:
                degree_mean = torch.zeros((), device=device, dtype=torch.float32)
            policy_metrics = {
                "policy_drift_abs": drift_abs_mean.detach(),
                "policy_drift_rms": drift_rms.detach(),
                "policy_kl_p0": kl_mean.detach(),
                "policy_out_degree_mean": degree_mean.detach(),
            }
        return _RolloutResult(
            log_pf_sum=log_pf_sum,
            stop_nodes=stop_nodes,
            num_moves=num_moves,
            stop_reason=stop_reason,
            actions=actions,
            log_pf_steps=log_pf_steps,
            policy_metrics=policy_metrics,
        )

    def _apply_action_constraints_to_outgoing(
        self,
        outgoing: OutgoingEdges,
        *,
        num_graphs: int,
        edge_index: torch.Tensor,
        edge_mask: Optional[torch.Tensor],
        visited_nodes: Optional[torch.Tensor],
    ) -> OutgoingEdges:
        if edge_mask is not None:
            outgoing = self._apply_edge_mask_to_outgoing(outgoing, edge_mask=edge_mask, num_graphs=num_graphs)
        if visited_nodes is not None:
            outgoing = self._apply_no_revisit_to_outgoing(
                outgoing,
                visited_nodes=visited_nodes,
                edge_index=edge_index,
                num_graphs=num_graphs,
            )
        return outgoing

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

    @staticmethod
    def _apply_no_revisit_to_outgoing(
        outgoing: OutgoingEdges,
        *,
        visited_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        num_graphs: int,
    ) -> OutgoingEdges:
        edge_ids = outgoing.edge_ids
        edge_batch = outgoing.edge_batch
        if edge_ids.numel() == _ZERO:
            return outgoing
        visited_nodes = visited_nodes.to(device=edge_ids.device, dtype=torch.bool).view(-1)
        tails = edge_index[_ONE].index_select(0, edge_ids)
        keep = ~visited_nodes.index_select(0, tails)
        edge_ids = edge_ids[keep]
        edge_batch = edge_batch[keep]
        counts = torch.bincount(edge_batch, minlength=num_graphs).to(device=edge_ids.device, dtype=torch.long)
        has_edge = counts > _ZERO
        return OutgoingEdges(edge_ids=edge_ids, edge_batch=edge_batch, edge_counts=counts, has_edge=has_edge)

    def _compute_db_loss(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        actions: torch.Tensor,
        graph_mask: torch.Tensor,
        traj_lengths: torch.Tensor,
        stop_reason: torch.Tensor,
        node_is_target: torch.Tensor,
        sampling_temperature: float,
        edge_mask_bwd: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = prepared_fwd.node_ptr.device
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        num_graphs, max_steps = actions.shape
        if max_steps == _ZERO:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return self._ensure_loss_requires_grad(zero), {"db_loss": zero.detach()}

        db_cfg = self._resolve_db_cfg()
        dead_end_log_reward = float(self._resolve_dead_end_log_reward(cfg=db_cfg))
        dead_end_weight = float(db_cfg["dead_end_weight"])
        edge_mask = actions >= _ZERO
        failure_mask = (stop_reason != _TERMINAL_HIT) & graph_mask
        weight = torch.ones((num_graphs,), device=device, dtype=torch.float32)
        if dead_end_weight != float(_ONE):
            weight = torch.where(failure_mask, weight * dead_end_weight, weight)
        accum = self._init_db_accumulators(device=device)
        total = accum["total"]
        denom = accum["denom"]
        valid_count = accum["valid_count"]
        move_count = accum["move_count"]
        log_pb_sum = accum["log_pb_sum"]
        log_pb_min = accum["log_pb_min"]
        log_z_u_sum = accum["log_z_u_sum"]
        log_z_v_sum = accum["log_z_v_sum"]
        delta_sum = accum["delta_sum"]
        delta_sq_sum = accum["delta_sq_sum"]
        delta_count = accum["delta_count"]
        inv_invalid_count = accum["inv_invalid_count"]
        no_allowed_count = accum["no_allowed_count"]
        finite_pf_count = accum["finite_pf_count"]
        finite_pb_count = accum["finite_pb_count"]
        finite_z_u_count = accum["finite_z_u_count"]
        finite_z_v_count = accum["finite_z_v_count"]
        visited_fwd = None
        visited_bwd = None
        if self._avoid_revisit:
            num_nodes_total = int(prepared_fwd.node_batch.numel())
            visited_fwd = torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
            visited_bwd = torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
            mask_all = edge_mask & graph_mask.view(-1, _ONE)
            edges_taken = actions[mask_all].to(device=device, dtype=torch.long).clamp(min=_ZERO)
            tails_taken = prepared_fwd.edge_index[_ONE].index_select(0, edges_taken)
            visited_bwd.index_fill_(0, tails_taken, True)
        for step in range(max_steps):
            edge_ids = actions[:, step]
            move_mask = edge_mask[:, step] & graph_mask
            move_count = move_count + move_mask.to(dtype=torch.float32).sum()
            safe_edges = edge_ids.clamp(min=_ZERO)
            heads = prepared_fwd.edge_index[_ZERO].index_select(0, safe_edges)
            tails = prepared_fwd.edge_index[_ONE].index_select(0, safe_edges)
            if visited_fwd is not None:
                visited_fwd.index_fill_(0, heads[move_mask], True)
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
                apply_p0=self._p0_enabled(),
                visited_nodes=visited_fwd,
            )
            inv_edge = prepared_fwd.edge_inverse_map.index_select(0, safe_edges)
            inv_valid = inv_edge >= _ZERO
            inv_edge = torch.where(inv_valid, inv_edge, torch.full_like(inv_edge, _NEG_ONE))
            active_bwd = move_mask & inv_valid
            log_pb, no_allowed = self._compute_pb_log_prob(
                prepared=prepared_fwd,
                chosen_edge=inv_edge,
                parent_nodes=tails,
                move_mask=active_bwd,
                edge_ids_by_head=prepared_fwd.edge_ids_by_head_bwd,
                edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_bwd,
                visited_nodes=visited_bwd,
                edge_mask=edge_mask_bwd,
                return_no_allowed=True,
            )
            no_allowed_count = no_allowed_count + (no_allowed & active_bwd).to(dtype=torch.float32).sum()
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
            finite_pf = torch.isfinite(log_pf) & move_mask
            finite_pb = torch.isfinite(log_pb) & move_mask
            finite_z_u = torch.isfinite(log_z_u) & move_mask
            finite_z_v = torch.isfinite(log_z_v) & move_mask
            finite_pf_count = finite_pf_count + finite_pf.to(dtype=torch.float32).sum()
            finite_pb_count = finite_pb_count + finite_pb.to(dtype=torch.float32).sum()
            finite_z_u_count = finite_z_u_count + finite_z_u.to(dtype=torch.float32).sum()
            finite_z_v_count = finite_z_v_count + finite_z_v.to(dtype=torch.float32).sum()
            finite_all = finite_pf & finite_pb & finite_z_u & finite_z_v
            valid = move_mask & inv_valid & finite_all
            valid_f = valid.to(dtype=torch.float32)
            valid_count = valid_count + valid_f.sum()
            log_pb_sum = log_pb_sum + (log_pb * valid_f).sum()
            log_z_u_sum = log_z_u_sum + (log_z_u * valid_f).sum()
            log_z_v_sum = log_z_v_sum + (log_z_v * valid_f).sum()
            pb_for_min = torch.where(valid, log_pb, torch.full_like(log_pb, float("inf")))
            log_pb_min = torch.minimum(log_pb_min, pb_for_min.min())
            delta = (log_z_u + log_pf) - (log_z_v + log_pb)
            delta = torch.where(valid, delta, torch.zeros_like(delta))
            step_weight = weight * valid.to(dtype=weight.dtype)
            if bool(valid.any().detach().tolist()):
                delta_sum = delta_sum + delta[valid].sum()
                delta_sq_sum = delta_sq_sum + (delta[valid] * delta[valid]).sum()
                delta_count = delta_count + valid.to(dtype=torch.float32).sum()
            total = total + (delta.pow(_TWO) * step_weight).sum()
            denom = denom + step_weight.sum()
            if visited_fwd is not None and visited_bwd is not None:
                visited_fwd.index_fill_(0, tails[move_mask], True)
                visited_bwd.index_fill_(0, tails[move_mask], False)
        loss, metrics = self._finalize_db_metrics(
            total=total,
            denom=denom,
            valid_count=valid_count,
            move_count=move_count,
            log_pb_sum=log_pb_sum,
            log_pb_min=log_pb_min,
            log_z_u_sum=log_z_u_sum,
            log_z_v_sum=log_z_v_sum,
            delta_sum=delta_sum,
            delta_sq_sum=delta_sq_sum,
            delta_count=delta_count,
            inv_invalid_count=inv_invalid_count,
            no_allowed_count=no_allowed_count,
            finite_pf_count=finite_pf_count,
            finite_pb_count=finite_pb_count,
            finite_z_u_count=finite_z_u_count,
            finite_z_v_count=finite_z_v_count,
            device=device,
        )
        return self._ensure_loss_requires_grad(loss), metrics

    @staticmethod
    def _init_db_accumulators(*, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "total": torch.zeros((), device=device, dtype=torch.float32),
            "denom": torch.zeros((), device=device, dtype=torch.float32),
            "valid_count": torch.zeros((), device=device, dtype=torch.float32),
            "move_count": torch.zeros((), device=device, dtype=torch.float32),
            "log_pb_sum": torch.zeros((), device=device, dtype=torch.float32),
            "log_pb_min": torch.full((), float("inf"), device=device, dtype=torch.float32),
            "log_z_u_sum": torch.zeros((), device=device, dtype=torch.float32),
            "log_z_v_sum": torch.zeros((), device=device, dtype=torch.float32),
            "delta_sum": torch.zeros((), device=device, dtype=torch.float32),
            "delta_sq_sum": torch.zeros((), device=device, dtype=torch.float32),
            "delta_count": torch.zeros((), device=device, dtype=torch.float32),
            "inv_invalid_count": torch.zeros((), device=device, dtype=torch.float32),
            "no_allowed_count": torch.zeros((), device=device, dtype=torch.float32),
            "finite_pf_count": torch.zeros((), device=device, dtype=torch.float32),
            "finite_pb_count": torch.zeros((), device=device, dtype=torch.float32),
            "finite_z_u_count": torch.zeros((), device=device, dtype=torch.float32),
            "finite_z_v_count": torch.zeros((), device=device, dtype=torch.float32),
        }

    @staticmethod
    def _finalize_db_metrics(
        *,
        total: torch.Tensor,
        denom: torch.Tensor,
        valid_count: torch.Tensor,
        move_count: torch.Tensor,
        log_pb_sum: torch.Tensor,
        log_pb_min: torch.Tensor,
        log_z_u_sum: torch.Tensor,
        log_z_v_sum: torch.Tensor,
        delta_sum: torch.Tensor,
        delta_sq_sum: torch.Tensor,
        delta_count: torch.Tensor,
        inv_invalid_count: torch.Tensor,
        no_allowed_count: torch.Tensor,
        finite_pf_count: torch.Tensor,
        finite_pb_count: torch.Tensor,
        finite_z_u_count: torch.Tensor,
        finite_z_v_count: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        has_denom = denom > float(_ZERO)
        denom_safe = torch.where(has_denom, denom, torch.ones_like(denom))
        loss = total / denom_safe
        loss = torch.where(has_denom, loss, zero)
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
        valid_step_rate = torch.where(move_any, valid_count / move_count_safe, zero)
        finite_pf_rate = torch.where(move_any, finite_pf_count / move_count_safe, zero)
        finite_pb_rate = torch.where(move_any, finite_pb_count / move_count_safe, zero)
        finite_z_u_rate = torch.where(move_any, finite_z_u_count / move_count_safe, zero)
        finite_z_v_rate = torch.where(move_any, finite_z_v_count / move_count_safe, zero)
        delta_has = delta_count > float(_ZERO)
        delta_mean = torch.where(delta_has, delta_sum / delta_count.clamp(min=_ONE), zero)
        delta_var = torch.where(
            delta_has,
            delta_sq_sum / delta_count.clamp(min=_ONE) - delta_mean * delta_mean,
            zero,
        )
        metrics = {
            "db_loss": loss.detach(),
            "db_log_pb_mean": log_pb_mean.detach(),
            "db_log_pb_min": log_pb_min.detach(),
            "db_log_z_u_mean": log_z_u_mean.detach(),
            "db_log_z_v_mean": log_z_v_mean.detach(),
            "db_delta_var": delta_var.detach(),
            "db_inv_edge_invalid_rate": inv_edge_invalid_rate.detach(),
            "db_no_allowed_rate": no_allowed_rate.detach(),
            "db_valid_step_rate": valid_step_rate.detach(),
            "db_finite_pf_rate": finite_pf_rate.detach(),
            "db_finite_pb_rate": finite_pb_rate.detach(),
            "db_finite_z_u_rate": finite_z_u_rate.detach(),
            "db_finite_z_v_rate": finite_z_v_rate.detach(),
        }
        return loss, metrics

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

    @staticmethod
    def _validate_training_batch(prepared_fwd: _PreparedBatch) -> torch.Tensor:
        num_graphs = int(prepared_fwd.num_graphs)
        if num_graphs <= _ZERO:
            raise ValueError("Empty batch.")
        graph_mask = ~prepared_fwd.dummy_mask
        torch._assert(graph_mask.any(), "Training batch contains no valid graphs.")
        return graph_mask

    def _run_training_rollout(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        sampling_temperature: float,
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
                apply_p0=self._p0_enabled(),
                collect_policy_metrics=True,
            )
        if rollout_fwd.actions is None:
            raise RuntimeError("Rollout actions are required for detailed balance training.")
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            actions=rollout_fwd.actions,
            graph_mask=graph_mask,
            traj_lengths=rollout_fwd.num_moves,
            stop_reason=rollout_fwd.stop_reason,
            node_is_target=node_is_target,
            sampling_temperature=sampling_temperature,
        )
        success = (rollout_fwd.stop_reason == _TERMINAL_HIT) & graph_mask
        lengths = rollout_fwd.num_moves.to(dtype=torch.float32)
        denom = graph_mask.to(dtype=lengths.dtype).sum().clamp(min=_ONE)
        length_mean = (lengths * graph_mask.to(dtype=lengths.dtype)).sum() / denom
        metrics = {
            **db_metrics,
            "rollout_success_rate": success.to(dtype=torch.float32).mean(),
            "rollout_length_mean": length_mean,
        }
        if rollout_fwd.policy_metrics:
            metrics.update(rollout_fwd.policy_metrics)
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
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        node_is_start: torch.Tensor,
        start_nodes_bwd: torch.Tensor,
        sampling_temperature: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        edge_index_for_dropout = prepared_fwd.edge_index
        edge_mask_bwd = self._sample_pb_edge_dropout_mask(edge_index=edge_index_for_dropout)
        with torch.no_grad():
            rollout_bwd = self._rollout_pb(
                prepared=prepared_fwd,
                graph_mask=graph_mask,
                start_nodes=start_nodes_bwd,
                node_is_target=node_is_start,
                edge_ids_by_head=prepared_fwd.edge_ids_by_head_bwd,
                edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_bwd,
                record_actions=True,
                record_log_pf=False,
                edge_mask=edge_mask_bwd,
            )
        if rollout_bwd.actions is None:
            raise RuntimeError("Backward rollout actions are required for detailed balance training.")
        actions_fwd = self._map_inverse_actions(
            actions=rollout_bwd.actions,
            edge_inverse_map=prepared_fwd.edge_inverse_map,
        )
        actions_fwd = self._reverse_actions_by_length(actions=actions_fwd, lengths=rollout_bwd.num_moves)
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            actions=actions_fwd,
            graph_mask=graph_mask,
            traj_lengths=rollout_bwd.num_moves,
            stop_reason=rollout_bwd.stop_reason,
            node_is_target=node_is_target,
            sampling_temperature=sampling_temperature,
            edge_mask_bwd=edge_mask_bwd,
        )
        success = (rollout_bwd.stop_reason == _TERMINAL_HIT) & graph_mask
        lengths = rollout_bwd.num_moves.to(dtype=torch.float32)
        denom = graph_mask.to(dtype=lengths.dtype).sum().clamp(min=_ONE)
        length_mean = (lengths * graph_mask.to(dtype=lengths.dtype)).sum() / denom
        metrics = {
            **db_metrics,
            "rollout_bwd_success_rate": success.to(dtype=torch.float32).mean(),
            "rollout_bwd_length_mean": length_mean,
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
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        node_is_start: torch.Tensor,
        start_nodes_bwd: torch.Tensor,
        sampling_temperature: float,
        num_rollouts: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if num_rollouts <= _ZERO:
            raise ValueError("num_rollouts must be > 0.")
        losses: list[torch.Tensor] = []
        metric_series: dict[str, list[torch.Tensor]] = {}
        for _ in range(num_rollouts):
            db_loss_fwd, metrics_fwd = self._run_training_rollout(
                prepared_fwd=prepared_fwd,
                graph_mask=graph_mask,
                node_is_target=node_is_target,
                sampling_temperature=sampling_temperature,
            )
            db_loss_bwd, metrics_bwd = self._run_backward_rollout(
                prepared_fwd=prepared_fwd,
                graph_mask=graph_mask,
                node_is_target=node_is_target,
                node_is_start=node_is_start,
                start_nodes_bwd=start_nodes_bwd,
                sampling_temperature=sampling_temperature,
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
        torch._assert(~invalid.any(), "Backward rollout sampled edges without forward inverse.")
        return torch.where(actions >= _ZERO, mapped, actions)

    @staticmethod
    def _reverse_actions_by_length(*, actions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if actions.numel() == _ZERO:
            return actions
        lengths = lengths.to(device=actions.device, dtype=torch.long).view(-1)
        num_graphs, max_steps = actions.shape
        if lengths.numel() != num_graphs:
            raise ValueError("lengths length mismatch with actions batch dimension.")
        steps = torch.arange(max_steps, device=actions.device, dtype=torch.long).view(_ONE, -1)
        lengths = lengths.clamp(min=_ZERO, max=max_steps).view(-1, _ONE)
        idx = torch.where(steps < lengths, lengths - _ONE - steps, steps).expand(num_graphs, -1)
        return actions.gather(1, idx)

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

    def _compute_training_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prepared_fwd = self._prepare_batch(batch)
        graph_mask = self._validate_training_batch(prepared_fwd)
        num_nodes_total = int(prepared_fwd.num_nodes_total)
        start_nodes_bwd = self._sample_nodes_uniform(
            local_indices=prepared_fwd.a_local_indices,
            ptr=prepared_fwd.a_ptr,
            allow_empty=True,
            name="a_local_indices",
        )
        node_is_target = build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        node_is_start = build_node_mask(num_nodes_total, prepared_fwd.q_local_indices)
        sampling_temperature = self._resolve_sampling_temperature()
        num_rollouts = self._resolve_num_rollouts()
        loss, metrics = self._aggregate_training_rollouts(
            prepared_fwd=prepared_fwd,
            graph_mask=graph_mask,
            node_is_target=node_is_target,
            node_is_start=node_is_start,
            start_nodes_bwd=start_nodes_bwd,
            sampling_temperature=sampling_temperature,
            num_rollouts=num_rollouts,
        )
        metrics.update(self._compute_log_z_metrics(prepared_fwd=prepared_fwd, graph_mask=graph_mask))
        return loss, metrics

    def _compute_log_z_metrics(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        graph_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graph_mask = graph_mask.to(device=prepared_fwd.node_ptr.device, dtype=torch.bool)
        start_nodes = prepared_fwd.start_nodes_fwd.to(device=prepared_fwd.node_ptr.device, dtype=torch.long)
        valid = graph_mask & (start_nodes >= _ZERO)
        if not bool(valid.any().detach().tolist()):
            zero = torch.zeros((), device=prepared_fwd.node_ptr.device, dtype=torch.float32)
            return {"log_z_mean": zero, "log_z_std": zero}
        safe_start = torch.where(valid, start_nodes, torch.zeros_like(start_nodes))
        step_ids = torch.zeros((prepared_fwd.num_graphs,), device=prepared_fwd.node_ptr.device, dtype=torch.long)
        log_z = self._compute_log_z_for_nodes(
            node_tokens=prepared_fwd.node_tokens,
            context_tokens=prepared_fwd.context_tokens,
            node_batch=prepared_fwd.node_batch,
            steps=step_ids,
            node_ids=safe_start,
        )
        log_z = log_z[valid]
        mean = log_z.mean()
        std = log_z.std(unbiased=False) if log_z.numel() > _ONE else torch.zeros_like(mean)
        return {"log_z_mean": mean.detach(), "log_z_std": std.detach()}

    @staticmethod
    def _ensure_loss_requires_grad(loss: torch.Tensor) -> torch.Tensor:
        if loss.requires_grad:
            return loss
        return loss + torch.zeros((), device=loss.device, dtype=loss.dtype, requires_grad=True)

    def _collect_logit_scale_metrics(self) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        logit_scale = getattr(self.policy_fwd, "logit_scale", None)
        if logit_scale is not None:
            scale = logit_scale.exp()
            scale = scale.clamp(min=self.policy_fwd.logit_scale_min, max=self.policy_fwd.logit_scale_max)
            metrics["logit_scale_fwd"] = scale.detach()
        if metrics:
            metrics["logit_scale_max"] = torch.stack(list(metrics.values())).max()
        return metrics



    @staticmethod
    def _index_candidates(candidates: _BeamCandidates, index: torch.Tensor) -> _BeamCandidates:
        return _BeamCandidates(
            cand_scores=candidates.cand_scores.index_select(0, index),
            cand_nodes=candidates.cand_nodes.index_select(0, index),
            cand_graph=candidates.cand_graph.index_select(0, index),
            cand_src_beam=candidates.cand_src_beam.index_select(0, index),
            cand_edge_id=candidates.cand_edge_id.index_select(0, index),
            cand_is_edge=candidates.cand_is_edge.index_select(0, index),
            cand_done=candidates.cand_done.index_select(0, index),
        )

    @staticmethod
    def _merge_indices_by_graph(
        *,
        cand_graph_edge: torch.Tensor,
        cand_graph_stay: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        counts_edge = torch.bincount(cand_graph_edge, minlength=num_graphs)
        counts_stay = torch.bincount(cand_graph_stay, minlength=num_graphs)
        total_counts = counts_edge + counts_stay
        total = int(cand_graph_edge.numel() + cand_graph_stay.numel())
        offsets = total_counts.cumsum(0) - total_counts

        total_edges = cand_graph_edge.numel()
        start_edge = (counts_edge.cumsum(0) - counts_edge).index_select(0, cand_graph_edge)
        pos_edge = torch.arange(total_edges, device=cand_graph_edge.device) - start_edge
        idx_edge = offsets.index_select(0, cand_graph_edge) + pos_edge

        total_stay = cand_graph_stay.numel()
        start_stay = (counts_stay.cumsum(0) - counts_stay).index_select(0, cand_graph_stay)
        pos_stay = torch.arange(total_stay, device=cand_graph_stay.device) - start_stay
        idx_stay = offsets.index_select(0, cand_graph_stay) + counts_edge.index_select(0, cand_graph_stay) + pos_stay
        return idx_edge, idx_stay, total

    @staticmethod
    def _scatter_merged_candidates(
        *,
        idx_edge: torch.Tensor,
        idx_stay: torch.Tensor,
        total: int,
        cand_scores_edge: torch.Tensor,
        cand_nodes_edge: torch.Tensor,
        cand_graph_edge: torch.Tensor,
        cand_src_beam_edge: torch.Tensor,
        cand_edge_id_edge: torch.Tensor,
        cand_is_edge_edge: torch.Tensor,
        cand_done_edge: torch.Tensor,
        cand_scores_stay: torch.Tensor,
        cand_nodes_stay: torch.Tensor,
        cand_graph_stay: torch.Tensor,
        cand_src_beam_stay: torch.Tensor,
        cand_edge_id_stay: torch.Tensor,
        cand_is_edge_stay: torch.Tensor,
        cand_done_stay: torch.Tensor,
    ) -> _BeamCandidates:
        device = cand_scores_edge.device
        out_scores = torch.empty((total,), device=device, dtype=cand_scores_edge.dtype)
        out_nodes = torch.empty((total,), device=device, dtype=cand_nodes_edge.dtype)
        out_graph = torch.empty((total,), device=device, dtype=torch.long)
        out_src = torch.empty((total,), device=device, dtype=torch.long)
        out_edge_id = torch.empty((total,), device=device, dtype=cand_edge_id_edge.dtype)
        out_is_edge = torch.empty((total,), device=device, dtype=torch.bool)
        out_done = torch.empty((total,), device=device, dtype=torch.bool)

        out_scores.index_copy_(0, idx_edge, cand_scores_edge)
        out_nodes.index_copy_(0, idx_edge, cand_nodes_edge)
        out_graph.index_copy_(0, idx_edge, cand_graph_edge)
        out_src.index_copy_(0, idx_edge, cand_src_beam_edge)
        out_edge_id.index_copy_(0, idx_edge, cand_edge_id_edge)
        out_is_edge.index_copy_(0, idx_edge, cand_is_edge_edge)
        out_done.index_copy_(0, idx_edge, cand_done_edge)

        out_scores.index_copy_(0, idx_stay, cand_scores_stay)
        out_nodes.index_copy_(0, idx_stay, cand_nodes_stay)
        out_graph.index_copy_(0, idx_stay, cand_graph_stay)
        out_src.index_copy_(0, idx_stay, cand_src_beam_stay)
        out_edge_id.index_copy_(0, idx_stay, cand_edge_id_stay)
        out_is_edge.index_copy_(0, idx_stay, cand_is_edge_stay)
        out_done.index_copy_(0, idx_stay, cand_done_stay)

        return _BeamCandidates(
            cand_scores=out_scores,
            cand_nodes=out_nodes,
            cand_graph=out_graph,
            cand_src_beam=out_src,
            cand_edge_id=out_edge_id,
            cand_is_edge=out_is_edge,
            cand_done=out_done,
        )

    @staticmethod
    def _coerce_candidate_graph(cand_graph: torch.Tensor) -> torch.Tensor:
        if cand_graph.dtype != torch.long:
            return cand_graph.to(dtype=torch.long)
        return cand_graph

    @staticmethod
    def _coerce_candidates_graph(candidates: _BeamCandidates) -> _BeamCandidates:
        if candidates.cand_graph.dtype == torch.long:
            return candidates
        return _BeamCandidates(
            cand_scores=candidates.cand_scores,
            cand_nodes=candidates.cand_nodes,
            cand_graph=candidates.cand_graph.to(dtype=torch.long),
            cand_src_beam=candidates.cand_src_beam,
            cand_edge_id=candidates.cand_edge_id,
            cand_is_edge=candidates.cand_is_edge,
            cand_done=candidates.cand_done,
        )

    @staticmethod
    def _maybe_sort_candidates_by_graph(
        candidates: _BeamCandidates,
        cand_graph: torch.Tensor,
    ) -> tuple[_BeamCandidates, torch.Tensor]:
        if cand_graph.numel() <= 1:
            return candidates, cand_graph
        if (cand_graph[:-1] <= cand_graph[1:]).all().item():
            return candidates, cand_graph
        order = torch.argsort(cand_graph)
        candidates = DualFlowModule._index_candidates(candidates, order)
        return candidates, candidates.cand_graph

    @staticmethod
    def _truncate_candidates_by_score(
        candidates: _BeamCandidates,
        *,
        cand_graph: torch.Tensor,
        num_graphs: int,
        cap: Optional[int],
    ) -> tuple[Optional[_BeamCandidates], torch.Tensor, torch.Tensor, int, bool]:
        counts = torch.bincount(cand_graph, minlength=num_graphs)
        if cap is None or cap <= 0:
            raise ValueError("max_candidates_per_graph must be set for beam candidate truncation.")
        if int(counts.max().item()) <= int(cap):
            return candidates, cand_graph, counts, int(cap), False

        order_score = torch.argsort(candidates.cand_scores, descending=True)
        graph_sorted = cand_graph.index_select(0, order_score)
        order_graph = torch.argsort(graph_sorted, stable=True)
        order = order_score.index_select(0, order_graph)
        candidates = DualFlowModule._index_candidates(candidates, order)
        cand_graph = candidates.cand_graph

        counts = torch.bincount(cand_graph, minlength=num_graphs)
        start = (counts.cumsum(0) - counts).index_select(0, cand_graph)
        pos = torch.arange(cand_graph.numel(), device=cand_graph.device) - start
        keep = pos < cap
        keep_idx = torch.nonzero(keep, as_tuple=False).view(-1)
        if keep_idx.numel() == 0:
            return None, cand_graph, counts, 0, True
        candidates = DualFlowModule._index_candidates(candidates, keep_idx)
        cand_graph = candidates.cand_graph
        counts = torch.bincount(cand_graph, minlength=num_graphs)
        max_count = int(cap)
        return candidates, cand_graph, counts, max_count, True

    @staticmethod
    def _merge_candidates_by_graph(
        *,
        cand_scores_edge: torch.Tensor,
        cand_nodes_edge: torch.Tensor,
        cand_graph_edge: torch.Tensor,
        cand_src_beam_edge: torch.Tensor,
        cand_edge_id_edge: torch.Tensor,
        cand_is_edge_edge: torch.Tensor,
        cand_done_edge: torch.Tensor,
        cand_scores_stay: torch.Tensor,
        cand_nodes_stay: torch.Tensor,
        cand_graph_stay: torch.Tensor,
        cand_src_beam_stay: torch.Tensor,
        cand_edge_id_stay: torch.Tensor,
        cand_is_edge_stay: torch.Tensor,
        cand_done_stay: torch.Tensor,
        num_graphs: int,
    ) -> Optional[_BeamCandidates]:
        total_edges = cand_scores_edge.numel()
        total_stay = cand_scores_stay.numel()
        total = total_edges + total_stay
        if total == 0:
            return None
        if total_edges > 1 and not (cand_graph_edge[:-1] <= cand_graph_edge[1:]).all().item():
            order_edge = torch.argsort(cand_graph_edge)
            cand_scores_edge = cand_scores_edge.index_select(0, order_edge)
            cand_nodes_edge = cand_nodes_edge.index_select(0, order_edge)
            cand_graph_edge = cand_graph_edge.index_select(0, order_edge)
            cand_src_beam_edge = cand_src_beam_edge.index_select(0, order_edge)
            cand_edge_id_edge = cand_edge_id_edge.index_select(0, order_edge)
            cand_is_edge_edge = cand_is_edge_edge.index_select(0, order_edge)
            cand_done_edge = cand_done_edge.index_select(0, order_edge)
        if total_stay > 1 and not (cand_graph_stay[:-1] <= cand_graph_stay[1:]).all().item():
            order_stay = torch.argsort(cand_graph_stay)
            cand_scores_stay = cand_scores_stay.index_select(0, order_stay)
            cand_nodes_stay = cand_nodes_stay.index_select(0, order_stay)
            cand_graph_stay = cand_graph_stay.index_select(0, order_stay)
            cand_src_beam_stay = cand_src_beam_stay.index_select(0, order_stay)
            cand_edge_id_stay = cand_edge_id_stay.index_select(0, order_stay)
            cand_is_edge_stay = cand_is_edge_stay.index_select(0, order_stay)
            cand_done_stay = cand_done_stay.index_select(0, order_stay)
        if total_edges == 0:
            return _BeamCandidates(
                cand_scores=cand_scores_stay,
                cand_nodes=cand_nodes_stay,
                cand_graph=cand_graph_stay,
                cand_src_beam=cand_src_beam_stay,
                cand_edge_id=cand_edge_id_stay,
                cand_is_edge=cand_is_edge_stay,
                cand_done=cand_done_stay,
            )
        if total_stay == 0:
            return _BeamCandidates(
                cand_scores=cand_scores_edge,
                cand_nodes=cand_nodes_edge,
                cand_graph=cand_graph_edge,
                cand_src_beam=cand_src_beam_edge,
                cand_edge_id=cand_edge_id_edge,
                cand_is_edge=cand_is_edge_edge,
                cand_done=cand_done_edge,
            )
        idx_edge, idx_stay, total = DualFlowModule._merge_indices_by_graph(
            cand_graph_edge=cand_graph_edge,
            cand_graph_stay=cand_graph_stay,
            num_graphs=num_graphs,
        )
        return DualFlowModule._scatter_merged_candidates(
            idx_edge=idx_edge,
            idx_stay=idx_stay,
            total=total,
            cand_scores_edge=cand_scores_edge,
            cand_nodes_edge=cand_nodes_edge,
            cand_graph_edge=cand_graph_edge,
            cand_src_beam_edge=cand_src_beam_edge,
            cand_edge_id_edge=cand_edge_id_edge,
            cand_is_edge_edge=cand_is_edge_edge,
            cand_done_edge=cand_done_edge,
            cand_scores_stay=cand_scores_stay,
            cand_nodes_stay=cand_nodes_stay,
            cand_graph_stay=cand_graph_stay,
            cand_src_beam_stay=cand_src_beam_stay,
            cand_edge_id_stay=cand_edge_id_stay,
            cand_is_edge_stay=cand_is_edge_stay,
            cand_done_stay=cand_done_stay,
        )

    def _beam_search_state(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
    ) -> _BeamState:
        num_graphs = int(prepared.num_graphs)
        if num_graphs <= 0:
            return _BeamState(
                beam_nodes=torch.zeros((0, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_scores=torch.zeros((0, 0), device=prepared.node_ptr.device, dtype=torch.float32),
                beam_paths=torch.zeros((0, 0, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_lengths=torch.zeros((0, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_done=torch.zeros((0, 0), device=prepared.node_ptr.device, dtype=torch.bool),
                flat_graph_ids=torch.zeros((0,), device=prepared.node_ptr.device, dtype=torch.long),
                flat_beam_ids=torch.zeros((0,), device=prepared.node_ptr.device, dtype=torch.long),
                beam_context=torch.zeros((0, prepared.context_tokens.size(-1)), device=prepared.node_ptr.device, dtype=prepared.context_tokens.dtype),
                num_graphs=0,
                beam_size=0,
                max_steps=int(self.max_steps),
                neg_inf=float("-inf"),
            )
        if beam_size <= 0:
            return _BeamState(
                beam_nodes=torch.zeros((num_graphs, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_scores=torch.zeros((num_graphs, 0), device=prepared.node_ptr.device, dtype=torch.float32),
                beam_paths=torch.zeros((num_graphs, 0, int(self.max_steps)), device=prepared.node_ptr.device, dtype=torch.long),
                beam_lengths=torch.zeros((num_graphs, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_done=torch.zeros((num_graphs, 0), device=prepared.node_ptr.device, dtype=torch.bool),
                flat_graph_ids=torch.zeros((0,), device=prepared.node_ptr.device, dtype=torch.long),
                flat_beam_ids=torch.zeros((0,), device=prepared.node_ptr.device, dtype=torch.long),
                beam_context=prepared.context_tokens[:0],
                num_graphs=num_graphs,
                beam_size=0,
                max_steps=int(self.max_steps),
                neg_inf=float("-inf"),
            )
        state = self._init_beam_state(prepared=prepared, beam_size=beam_size, node_is_target=node_is_target)
        diverse_cfg = self._resolve_diverse_beam_cfg()
        for step in range(state.max_steps):
            candidates = self._beam_expand_candidates(
                prepared=prepared,
                state=state,
                step=step,
                node_is_target=node_is_target,
            )
            if candidates is None:
                break
            state = self._beam_update_from_candidates(
                state=state,
                candidates=candidates,
                step=step,
                diverse_cfg=diverse_cfg,
            )
        return state

    def _beam_search(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
    ) -> list[list[tuple[int, float, list[int]]]]:
        state = self._beam_search_state(prepared=prepared, beam_size=beam_size, node_is_target=node_is_target)
        if state.beam_nodes.numel() == 0:
            return []
        return self._beam_finalize(state)

    def _init_beam_state(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
    ) -> _BeamState:
        num_graphs = int(prepared.num_graphs)
        device = prepared.node_ptr.device
        max_steps = int(self.max_steps)
        neg_inf = float("-inf")
        start_nodes = prepared.start_nodes_fwd.to(device=device, dtype=torch.long)
        beam_nodes = torch.full((num_graphs, beam_size), _NEG_ONE, device=device, dtype=torch.long)
        beam_scores = torch.full((num_graphs, beam_size), neg_inf, device=device, dtype=torch.float32)
        beam_paths = torch.full((num_graphs, beam_size, max_steps), _NEG_ONE, device=device, dtype=torch.long)
        beam_lengths = torch.zeros((num_graphs, beam_size), device=device, dtype=torch.long)
        valid_start = start_nodes >= 0
        beam_nodes[:, 0] = start_nodes
        beam_scores[:, 0] = torch.where(valid_start, torch.zeros_like(beam_scores[:, 0]), beam_scores[:, 0])
        start_target = node_is_target.index_select(0, start_nodes.clamp(min=0))
        beam_done = torch.zeros((num_graphs, beam_size), device=device, dtype=torch.bool)
        beam_done[:, 0] = valid_start & start_target
        flat_graph_ids = torch.arange(num_graphs, device=device).repeat_interleave(beam_size)
        flat_beam_ids = torch.arange(beam_size, device=device).repeat(num_graphs)
        beam_context = prepared.context_tokens.index_select(0, flat_graph_ids)
        return _BeamState(
            beam_nodes=beam_nodes,
            beam_scores=beam_scores,
            beam_paths=beam_paths,
            beam_lengths=beam_lengths,
            beam_done=beam_done,
            flat_graph_ids=flat_graph_ids,
            flat_beam_ids=flat_beam_ids,
            beam_context=beam_context,
            num_graphs=num_graphs,
            beam_size=beam_size,
            max_steps=max_steps,
            neg_inf=neg_inf,
        )

    def _beam_expand_candidates(
        self,
        *,
        prepared: _PreparedBatch,
        state: _BeamState,
        step: int,
        node_is_target: torch.Tensor,
    ) -> Optional[_BeamCandidates]:
        flat_nodes = state.beam_nodes.view(-1)
        flat_scores = state.beam_scores.view(-1)
        flat_done = state.beam_done.view(-1)
        flat_valid = flat_nodes >= 0
        expand_mask = flat_valid & ~flat_done
        outgoing = gather_outgoing_edges(
            curr_nodes=flat_nodes,
            edge_ids_by_head=prepared.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared.edge_ptr_by_head_fwd,
            active_mask=expand_mask,
        )
        empty_long = torch.zeros((0,), device=flat_nodes.device, dtype=torch.long)
        empty_bool = torch.zeros((0,), device=flat_nodes.device, dtype=torch.bool)
        empty_float = torch.zeros((0,), device=flat_nodes.device, dtype=torch.float32)

        if outgoing.edge_ids.numel() > 0:
            step_ids = torch.full((state.num_graphs * state.beam_size,), step, device=flat_nodes.device, dtype=torch.long)
            logits = self._compute_edge_logits(
                policy=self.policy_fwd,
                prepared=prepared,
                edge_ids=outgoing.edge_ids,
                edge_batch=outgoing.edge_batch,
                steps=step_ids + 1,
                temperature=1.0,
                context_tokens=state.beam_context,
                apply_p0=self._p0_enabled(),
                decision_graph_ids=state.flat_graph_ids,
            )
            log_denom = self._compute_log_denom(
                logits=logits, edge_batch=outgoing.edge_batch, num_graphs=state.num_graphs * state.beam_size
            )
            log_probs = logits - log_denom.index_select(0, outgoing.edge_batch)
            cand_scores_edge = flat_scores.index_select(0, outgoing.edge_batch) + log_probs
            cand_nodes_edge = prepared.edge_index[1].index_select(0, outgoing.edge_ids)
            cand_graph_edge = state.flat_graph_ids.index_select(0, outgoing.edge_batch)
            cand_src_beam_edge = state.flat_beam_ids.index_select(0, outgoing.edge_batch)
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
        cand_graph_stay = state.flat_graph_ids[stay_mask]
        cand_src_beam_stay = state.flat_beam_ids[stay_mask]
        cand_edge_id_stay = torch.full_like(cand_nodes_stay, _NEG_ONE)
        cand_is_edge_stay = torch.zeros_like(cand_scores_stay, dtype=torch.bool)
        cand_done_stay = torch.ones_like(cand_scores_stay, dtype=torch.bool)

        if cand_scores_edge.numel() + cand_scores_stay.numel() == 0:
            return None
        return self._merge_candidates_by_graph(
            cand_scores_edge=cand_scores_edge,
            cand_nodes_edge=cand_nodes_edge,
            cand_graph_edge=cand_graph_edge,
            cand_src_beam_edge=cand_src_beam_edge,
            cand_edge_id_edge=cand_edge_id_edge,
            cand_is_edge_edge=cand_is_edge_edge,
            cand_done_edge=cand_done_edge,
            cand_scores_stay=cand_scores_stay,
            cand_nodes_stay=cand_nodes_stay,
            cand_graph_stay=cand_graph_stay,
            cand_src_beam_stay=cand_src_beam_stay,
            cand_edge_id_stay=cand_edge_id_stay,
            cand_is_edge_stay=cand_is_edge_stay,
            cand_done_stay=cand_done_stay,
            num_graphs=state.num_graphs,
        )

    @staticmethod
    def _build_candidate_matrix(
        candidates: _BeamCandidates,
        *,
        num_graphs: int,
        neg_inf: float,
        max_candidates_per_graph: Optional[int] = None,
    ) -> Optional[_BeamCandidateMatrix]:
        cand_graph = candidates.cand_graph
        if cand_graph.numel() == 0:
            return None
        candidates = DualFlowModule._coerce_candidates_graph(candidates)
        cand_graph = candidates.cand_graph
        cap = int(max_candidates_per_graph) if max_candidates_per_graph is not None else None
        if cap is None or cap <= 0:
            raise ValueError("max_candidates_per_graph must be set for beam candidate matrix.")
        candidates, cand_graph, counts, max_count, truncated = DualFlowModule._truncate_candidates_by_score(
            candidates,
            cand_graph=cand_graph,
            num_graphs=num_graphs,
            cap=cap,
        )
        if candidates is None:
            return None
        candidates, cand_graph = DualFlowModule._maybe_sort_candidates_by_graph(candidates, cand_graph)
        cand_graph = DualFlowModule._coerce_candidate_graph(cand_graph)
        counts = torch.bincount(cand_graph, minlength=num_graphs)
        max_count = int(cap)

        device = cand_graph.device
        start = (counts.cumsum(0) - counts).index_select(0, cand_graph)
        pos = torch.arange(cand_graph.numel(), device=device) - start
        scores = torch.full((num_graphs, max_count), neg_inf, device=device, dtype=torch.float32)
        nodes = torch.full((num_graphs, max_count), _NEG_ONE, device=device, dtype=torch.long)
        src_beam = torch.full((num_graphs, max_count), _NEG_ONE, device=device, dtype=torch.long)
        edge_id = torch.full((num_graphs, max_count), _NEG_ONE, device=device, dtype=torch.long)
        is_edge = torch.zeros((num_graphs, max_count), device=device, dtype=torch.bool)
        done = torch.zeros((num_graphs, max_count), device=device, dtype=torch.bool)
        scores[cand_graph, pos] = candidates.cand_scores
        nodes[cand_graph, pos] = candidates.cand_nodes
        src_beam[cand_graph, pos] = candidates.cand_src_beam
        edge_id[cand_graph, pos] = candidates.cand_edge_id
        is_edge[cand_graph, pos] = candidates.cand_is_edge
        done[cand_graph, pos] = candidates.cand_done
        return _BeamCandidateMatrix(
            scores=scores,
            nodes=nodes,
            src_beam=src_beam,
            edge_id=edge_id,
            is_edge=is_edge,
            done=done,
            counts=counts,
        )

    @staticmethod
    def _build_diverse_keys(
        *,
        similarity: str,
        nodes: torch.Tensor,
        edge_id: torch.Tensor,
        src_beam: torch.Tensor,
        is_edge: torch.Tensor,
    ) -> torch.Tensor:
        if similarity == "tail":
            return nodes.to(dtype=torch.long)
        if similarity == "edge":
            stay_keys = -src_beam.to(dtype=torch.long) - 2
            edge_keys = edge_id.to(dtype=torch.long)
            return torch.where(is_edge, edge_keys, stay_keys)
        if similarity == "source":
            return src_beam.to(dtype=torch.long)
        raise ValueError(f"Unsupported diverse beam similarity: {similarity!r}.")

    def _select_beam_positions(
        self,
        *,
        scores: torch.Tensor,
        keys: torch.Tensor,
        counts: torch.Tensor,
        beam_size: int,
        diverse_cfg: dict[str, Any],
        neg_inf: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs, max_count = scores.size()
        if beam_size <= 0 or max_count <= 0 or num_graphs <= 0:
            empty_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
            empty_scores = torch.full((num_graphs, beam_size), neg_inf, device=scores.device, dtype=torch.float32)
            return empty_pos, empty_scores
        k_per_graph = counts.clamp(max=beam_size)
        range_beam = torch.arange(beam_size, device=scores.device).unsqueeze(0)
        if not diverse_cfg["enabled"] or beam_size <= 1 or diverse_cfg["groups"] <= 1:
            k_top = min(int(beam_size), int(max_count))
            if k_top <= 0:
                empty_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
                empty_scores = torch.full((num_graphs, beam_size), neg_inf, device=scores.device, dtype=torch.float32)
                return empty_pos, empty_scores
            top_scores, top_pos = torch.topk(scores, k_top, dim=1)
            sel_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
            sel_scores = torch.full((num_graphs, beam_size), neg_inf, device=scores.device, dtype=torch.float32)
            sel_pos[:, :k_top] = top_pos
            sel_scores[:, :k_top] = top_scores
            valid = range_beam < k_per_graph.unsqueeze(1)
            sel_pos = torch.where(valid, sel_pos, torch.full_like(sel_pos, _NEG_ONE))
            sel_scores = torch.where(valid, sel_scores, torch.full_like(sel_scores, neg_inf))
            return sel_pos, sel_scores
        sel_pos = self._diverse_select_positions(
            scores=scores,
            keys=keys,
            counts=counts,
            beam_size=beam_size,
            groups=int(diverse_cfg["groups"]),
            penalty=str(diverse_cfg["penalty"]),
            penalty_lambda=float(diverse_cfg["lambda"]),
            neg_inf=neg_inf,
        )
        pos_safe = sel_pos.clamp(min=0)
        sel_scores = torch.gather(scores, 1, pos_safe)
        valid = range_beam < k_per_graph.unsqueeze(1)
        valid = valid & (sel_pos >= 0)
        sel_scores = torch.where(valid, sel_scores, torch.full_like(sel_scores, neg_inf))
        sel_pos = torch.where(valid, sel_pos, torch.full_like(sel_pos, _NEG_ONE))
        return sel_pos, sel_scores

    def _diverse_select_positions(
        self,
        *,
        scores: torch.Tensor,
        keys: torch.Tensor,
        counts: torch.Tensor,
        beam_size: int,
        groups: int,
        penalty: str,
        penalty_lambda: float,
        neg_inf: float,
    ) -> torch.Tensor:
        num_graphs, max_count = scores.size()
        if max_count <= 0 or beam_size <= 0 or num_graphs <= 0:
            return torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
        range_count = torch.arange(max_count, device=scores.device).unsqueeze(0)
        valid_mask = range_count < counts.unsqueeze(1)
        valid_mask = valid_mask & torch.isfinite(scores)
        graph_ids = torch.arange(num_graphs, device=scores.device).unsqueeze(1).expand_as(scores)
        pos_ids = range_count.expand_as(scores)
        flat_scores = scores[valid_mask]
        if flat_scores.numel() == 0:
            return torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
        flat_keys = keys[valid_mask].to(dtype=torch.long)
        flat_graph = graph_ids[valid_mask].to(dtype=torch.long)
        flat_pos = pos_ids[valid_mask]
        key_min = flat_keys.min().to(dtype=torch.long)
        key_max = flat_keys.max().to(dtype=torch.long)
        key_stride = (key_max - key_min + 1).clamp(min=1).to(dtype=torch.long)
        comp = flat_graph * key_stride + (flat_keys - key_min)
        order = torch.argsort(comp)
        comp_sorted = comp.index_select(0, order)
        scores_sorted = flat_scores.index_select(0, order)
        pos_sorted = flat_pos.index_select(0, order)
        graph_sorted = flat_graph.index_select(0, order)
        change = comp_sorted[1:] != comp_sorted[:-1]
        group_ids = torch.zeros_like(comp_sorted, dtype=torch.long)
        group_ids[1:] = torch.cumsum(change.to(dtype=torch.long), dim=0)
        num_groups = group_ids[-1] + 1
        if penalty == "soft":
            group_counts = torch.bincount(group_ids).to(dtype=scores_sorted.dtype)
            penalty_counts = group_counts.index_select(0, group_ids)
            scores_sorted = scores_sorted - penalty_lambda * penalty_counts
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
            scores_adj = scores.clone().masked_fill(~valid_mask, neg_inf)
            scores_adj_flat = scores_sorted.index_select(0, inv_order)
            scores_adj[valid_mask] = scores_adj_flat
            k_top = min(int(beam_size), int(max_count))
            top_scores, top_pos = torch.topk(scores_adj, k_top, dim=1)
            sel_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
            sel_pos[:, :k_top] = top_pos
            return sel_pos
        _, argmax = segment_max(scores_sorted, group_ids, num_groups)
        best_scores = scores_sorted.index_select(0, argmax)
        best_pos = pos_sorted.index_select(0, argmax)
        best_graph = graph_sorted.index_select(0, argmax)
        counts_unique = torch.bincount(best_graph, minlength=num_graphs)
        if counts_unique.numel() > 0:
            max_unique = counts_unique.max().clamp(min=0)
        else:
            max_unique = torch.zeros((), device=scores.device, dtype=torch.long)
        unique_scores = torch.full((num_graphs, max_unique), neg_inf, device=scores.device, dtype=scores.dtype)
        unique_pos = torch.full((num_graphs, max_unique), _NEG_ONE, device=scores.device, dtype=torch.long)
        start = (counts_unique.cumsum(0) - counts_unique).index_select(0, best_graph)
        pos_unique = torch.arange(best_graph.numel(), device=scores.device) - start
        unique_scores[best_graph, pos_unique] = best_scores
        unique_pos[best_graph, pos_unique] = best_pos
        max_unique_val = unique_scores.size(1)
        k_top = min(int(beam_size), int(max_unique_val))
        sel_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
        if k_top > 0:
            top_scores, top_idx = torch.topk(unique_scores, k_top, dim=1)
            _ = top_scores
            sel_pos[:, :k_top] = torch.gather(unique_pos, 1, top_idx)
        valid_sel = sel_pos >= 0
        selected_count = valid_sel.sum(dim=1)
        selected_mask = torch.zeros((num_graphs, max_count), device=scores.device, dtype=torch.bool)
        if torch.any(valid_sel):
            batch_idx = torch.arange(num_graphs, device=scores.device).unsqueeze(1).expand_as(sel_pos)
            selected_mask[batch_idx[valid_sel], sel_pos[valid_sel]] = True
        remaining = (beam_size - selected_count).clamp(min=0)
        scores_remain = scores.masked_fill(~valid_mask, neg_inf)
        scores_remain = scores_remain.masked_fill(selected_mask, neg_inf)
        k_remain = min(int(beam_size), int(max_count))
        top_scores_r, top_pos_r = torch.topk(scores_remain, k_remain, dim=1)
        range_k = torch.arange(beam_size, device=scores.device)
        take_mask = range_k.unsqueeze(0) < remaining.unsqueeze(1)
        rank = torch.cumsum(take_mask, dim=1) - 1
        flat_rows, flat_cols = torch.nonzero(take_mask, as_tuple=True)
        if flat_rows.numel() > 0:
            insert_pos = selected_count[flat_rows] + rank[flat_rows, flat_cols]
            linear = flat_rows * beam_size + insert_pos
            sel_pos.view(-1)[linear] = top_pos_r[flat_rows, flat_cols]
        return sel_pos

    def _beam_update_from_candidates(
        self,
        *,
        state: _BeamState,
        candidates: _BeamCandidates,
        step: int,
        diverse_cfg: dict[str, Any],
    ) -> _BeamState:
        max_candidates = diverse_cfg.get("max_candidates_per_graph")
        if max_candidates is None:
            groups = int(diverse_cfg.get("groups", 1))
            if not diverse_cfg.get("enabled", False):
                groups = 1
            max_candidates = state.beam_size * max(1, groups)
        else:
            max_candidates = int(max_candidates)
            if max_candidates <= 0:
                max_candidates = None
        matrix = self._build_candidate_matrix(
            candidates,
            num_graphs=state.num_graphs,
            neg_inf=state.neg_inf,
            max_candidates_per_graph=max_candidates,
        )
        if matrix is None:
            return _BeamState(
                beam_nodes=torch.full_like(state.beam_nodes, _NEG_ONE),
                beam_scores=torch.full_like(state.beam_scores, state.neg_inf),
                beam_paths=torch.full_like(state.beam_paths, _NEG_ONE),
                beam_lengths=torch.zeros_like(state.beam_lengths),
                beam_done=torch.zeros_like(state.beam_done),
                flat_graph_ids=state.flat_graph_ids,
                flat_beam_ids=state.flat_beam_ids,
                beam_context=state.beam_context,
                num_graphs=state.num_graphs,
                beam_size=state.beam_size,
                max_steps=state.max_steps,
                neg_inf=state.neg_inf,
            )
        keys = self._build_diverse_keys(
            similarity=str(diverse_cfg["similarity"]),
            nodes=matrix.nodes,
            edge_id=matrix.edge_id,
            src_beam=matrix.src_beam,
            is_edge=matrix.is_edge,
        )
        sel_pos, sel_scores = self._select_beam_positions(
            scores=matrix.scores,
            keys=keys,
            counts=matrix.counts,
            beam_size=state.beam_size,
            diverse_cfg=diverse_cfg,
            neg_inf=state.neg_inf,
        )
        pos_safe = sel_pos.clamp(min=0)
        sel_nodes = torch.gather(matrix.nodes, 1, pos_safe)
        sel_src = torch.gather(matrix.src_beam, 1, pos_safe)
        sel_edge_id = torch.gather(matrix.edge_id, 1, pos_safe)
        sel_is_edge = torch.gather(matrix.is_edge, 1, pos_safe)
        sel_done = torch.gather(matrix.done, 1, pos_safe)
        valid_sel = sel_pos >= 0
        sel_nodes = torch.where(valid_sel, sel_nodes, torch.full_like(sel_nodes, _NEG_ONE))
        sel_src = torch.where(valid_sel, sel_src, torch.full_like(sel_src, _NEG_ONE))
        sel_edge_id = torch.where(valid_sel, sel_edge_id, torch.full_like(sel_edge_id, _NEG_ONE))
        sel_is_edge = torch.where(valid_sel, sel_is_edge, torch.zeros_like(sel_is_edge))
        sel_done = torch.where(valid_sel, sel_done, torch.zeros_like(sel_done))
        sel_scores = torch.where(valid_sel, sel_scores, torch.full_like(sel_scores, state.neg_inf))
        batch_idx = torch.arange(state.num_graphs, device=state.beam_nodes.device).unsqueeze(1).expand_as(sel_src)
        sel_src_safe = sel_src.clamp(min=0)
        sel_paths = state.beam_paths[batch_idx, sel_src_safe]
        sel_lengths = state.beam_lengths[batch_idx, sel_src_safe]
        sel_paths = sel_paths.clone()
        sel_paths[:, :, step] = torch.where(sel_is_edge, sel_edge_id, sel_paths[:, :, step])
        sel_lengths = sel_lengths + sel_is_edge.to(dtype=sel_lengths.dtype)
        sel_paths = torch.where(valid_sel.unsqueeze(-1), sel_paths, torch.full_like(sel_paths, _NEG_ONE))
        sel_lengths = torch.where(valid_sel, sel_lengths, torch.zeros_like(sel_lengths))
        sel_done = torch.where(valid_sel, sel_done, torch.zeros_like(sel_done))
        return _BeamState(
            beam_nodes=sel_nodes,
            beam_scores=sel_scores,
            beam_paths=sel_paths,
            beam_lengths=sel_lengths,
            beam_done=sel_done,
            flat_graph_ids=state.flat_graph_ids,
            flat_beam_ids=state.flat_beam_ids,
            beam_context=state.beam_context,
            num_graphs=state.num_graphs,
            beam_size=state.beam_size,
            max_steps=state.max_steps,
            neg_inf=state.neg_inf,
        )

    @staticmethod
    def _beam_finalize(state: _BeamState) -> list[list[tuple[int, float, list[int]]]]:
        beam_nodes_np = state.beam_nodes.detach().cpu().numpy()
        beam_scores_np = state.beam_scores.detach().cpu().numpy()
        beam_paths_np = state.beam_paths.detach().cpu().numpy()
        beam_lengths_np = state.beam_lengths.detach().cpu().numpy()
        beams: list[list[tuple[int, float, list[int]]]] = []
        for graph_idx in range(state.num_graphs):
            graph_beams: list[tuple[int, float, list[int]]] = []
            for beam_idx in range(state.beam_size):
                node_id = int(beam_nodes_np[graph_idx, beam_idx])
                if node_id < 0:
                    continue
                score = float(beam_scores_np[graph_idx, beam_idx])
                length = int(beam_lengths_np[graph_idx, beam_idx])
                if length <= 0:
                    path = []
                else:
                    path = beam_paths_np[graph_idx, beam_idx, :length].tolist()
                graph_beams.append((node_id, score, path))
            beams.append(graph_beams)
        return beams



    @torch.no_grad()
    def _compute_eval_metrics(self, batch: Any) -> tuple[dict[str, torch.Tensor], int]:
        prepared_fwd = self._prepare_batch(batch)
        num_graphs = int(prepared_fwd.num_graphs)
        if num_graphs <= _ZERO:
            return {}, _ZERO
        valid_count_attr = getattr(batch, "num_valid_graphs", None)
        if valid_count_attr is None:
            raise AttributeError("Batch missing num_valid_graphs; collator must precompute answer counts.")
        valid_count = int(valid_count_attr)
        scope = self._resolve_dataset_scope()
        if scope == "full":
            graph_mask = torch.ones((num_graphs,), device=prepared_fwd.node_ptr.device, dtype=torch.bool)
            batch_size = num_graphs
        else:
            graph_mask = ~prepared_fwd.dummy_mask
            batch_size = valid_count
            if batch_size <= _ZERO:
                return {}, _ZERO
        num_nodes_total = int(prepared_fwd.num_nodes_total)
        node_is_target_all = build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        beam_size = self._resolve_beam_size()
        beam_state = self._beam_search_state(
            prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target_all
        )
        beam_nodes = beam_state.beam_nodes
        beam_lengths = beam_state.beam_lengths
        if beam_nodes.numel() == _ZERO:
            return {}, _ZERO
        beam_valid = beam_nodes >= _ZERO
        beam_nodes_safe = beam_nodes.clamp(min=_ZERO)
        beam_hits = node_is_target_all.index_select(0, beam_nodes_safe.view(-1)).view(num_graphs, -1)
        beam_hits = beam_hits & beam_valid
        hit_hits = beam_hits.any(dim=1).to(dtype=torch.float32)
        sorted_nodes, _ = torch.sort(beam_nodes, dim=1)
        sorted_valid = sorted_nodes >= _ZERO
        unique_mask = torch.ones_like(sorted_nodes, dtype=torch.bool)
        if sorted_nodes.size(1) > _ONE:
            unique_mask[:, _ONE:] = sorted_nodes[:, _ONE:] != sorted_nodes[:, :-_ONE]
        unique_mask = unique_mask & sorted_valid
        unique_nodes_safe = sorted_nodes.clamp(min=_ZERO)
        unique_hits = node_is_target_all.index_select(0, unique_nodes_safe.view(-1)).view(num_graphs, -1)
        unique_hit_counts = (unique_mask & unique_hits).sum(dim=1).to(dtype=torch.float32)
        unique_pred_counts = unique_mask.sum(dim=1).to(dtype=torch.float32)
        answer_counts = (prepared_fwd.a_ptr[_ONE:] - prepared_fwd.a_ptr[:-_ONE]).clamp(min=_ZERO).to(dtype=torch.float32)
        precision_scores = unique_hit_counts / unique_pred_counts.clamp(min=_ONE)
        recall_scores = unique_hit_counts / answer_counts.clamp(min=_ONE)
        denom = recall_scores + precision_scores
        f1_scores = torch.where(denom > float(_ZERO), (float(_TWO) * recall_scores * precision_scores / denom), torch.zeros_like(denom))
        beam_valid_counts = beam_valid.sum(dim=1).to(dtype=torch.float32)
        diversity_scores = unique_pred_counts / beam_valid_counts.clamp(min=_ONE)
        length = beam_lengths[:, _ZERO].to(dtype=torch.float32)
        modes_per_graph = None
        if beam_state.beam_paths.numel() > 0 and beam_state.beam_lengths.numel() > 0:
            beam_hits_cpu = beam_hits.detach().cpu().numpy()
            beam_paths_cpu = beam_state.beam_paths.detach().cpu().numpy()
            beam_lengths_cpu = beam_state.beam_lengths.detach().cpu().numpy()
            modes: list[int] = []
            for graph_idx in range(num_graphs):
                seen: set[tuple[int, ...]] = set()
                for beam_idx in range(beam_state.beam_size):
                    if not beam_hits_cpu[graph_idx, beam_idx]:
                        continue
                    length_i = int(beam_lengths_cpu[graph_idx, beam_idx])
                    if length_i <= 0:
                        path = ()
                    else:
                        path = tuple(beam_paths_cpu[graph_idx, beam_idx, :length_i].tolist())
                    seen.add(path)
                modes.append(len(seen))
            modes_per_graph = torch.as_tensor(modes, device=beam_nodes.device, dtype=torch.float32)
        metrics = {
            "hit@beam": hit_hits,
            "recall@beam": recall_scores,
            "precision@beam": precision_scores,
            "f1@beam": f1_scores,
            "diversity@beam": diversity_scores,
        }
        # Coverage-style diagnostics (used to separate retrieval failure from reasoning failure).
        metrics["coverage_rate"] = (~prepared_fwd.dummy_mask).to(dtype=torch.float32)
        metrics["retrieval_failure_rate"] = prepared_fwd.dummy_mask.to(dtype=torch.float32)
        metrics["length_mean"] = length
        if modes_per_graph is not None:
            metrics["modes@beam"] = modes_per_graph
        eval_temperature = float(_ONE)
        rollout_fwd = self._rollout_policy(
            policy=self.policy_fwd,
            prepared=prepared_fwd,
            graph_mask=graph_mask,
            start_nodes=prepared_fwd.start_nodes_fwd,
            node_is_target=node_is_target_all,
            edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
            record_actions=True,
            record_log_pf=False,
            temperature=eval_temperature,
            context_tokens=prepared_fwd.context_tokens,
            apply_p0=self._p0_enabled(),
            collect_policy_metrics=True,
        )
        fwd_actions = rollout_fwd.actions
        if fwd_actions is None:
            fwd_actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=self.device, dtype=torch.long)
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            actions=fwd_actions,
            graph_mask=graph_mask,
            traj_lengths=rollout_fwd.num_moves,
            stop_reason=rollout_fwd.stop_reason,
            node_is_target=node_is_target_all,
            sampling_temperature=eval_temperature,
        )
        success = (rollout_fwd.stop_reason == _TERMINAL_HIT) & graph_mask
        metrics.update(db_metrics)
        if rollout_fwd.policy_metrics:
            metrics.update(rollout_fwd.policy_metrics)
        metrics.update(self._compute_log_z_metrics(prepared_fwd=prepared_fwd, graph_mask=graph_mask))
        metrics["rollout_success_rate"] = success.to(dtype=torch.float32).mean()
        metrics.update(
            self._build_terminal_metrics(
                stop_reason=rollout_fwd.stop_reason,
                graph_mask=graph_mask,
                prefix="rollout",
            )
        )
        metrics = self._reduce_eval_metrics(metrics, valid_mask=graph_mask)
        return metrics, batch_size

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



    @staticmethod
    def _filter_metrics(metrics: dict[str, torch.Tensor], keep: set[str]) -> dict[str, torch.Tensor]:
        if not metrics:
            return {}
        return {name: value for name, value in metrics.items() if name in keep}

    def _get_standard_metrics(self, stage: str) -> set[str]:
        key = str(stage).strip().lower()
        if key not in _STANDARD_METRICS:
            raise ValueError(f"Unsupported metrics stage: {stage!r}.")
        return _STANDARD_METRICS[key]



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
        elif sched_type in {"onecycle", "one_cycle", "onecyclelr", _SCHED_TYPE_ONECYCLE}:
            if interval != _SCHED_INTERVAL_STEP:
                raise ValueError("OneCycleLR requires scheduler_cfg.interval='step'.")
            max_lr = self.scheduler_cfg.get("max_lr", None)
            if max_lr is None:
                raise ValueError("scheduler_cfg.max_lr must be set for OneCycleLR.")
            total_steps = self._resolve_onecycle_total_steps()
            pct_start = float(self.scheduler_cfg.get("pct_start", _DEFAULT_ONECYCLE_PCT_START))
            anneal_strategy = str(
                self.scheduler_cfg.get("anneal_strategy", _DEFAULT_ONECYCLE_ANNEAL) or _DEFAULT_ONECYCLE_ANNEAL
            ).strip().lower()
            cycle_momentum = bool(self.scheduler_cfg.get("cycle_momentum", _DEFAULT_ONECYCLE_CYCLE_MOMENTUM))
            base_momentum = float(self.scheduler_cfg.get("base_momentum", _DEFAULT_ONECYCLE_BASE_MOMENTUM))
            max_momentum = float(self.scheduler_cfg.get("max_momentum", _DEFAULT_ONECYCLE_MAX_MOMENTUM))
            div_factor = float(self.scheduler_cfg.get("div_factor", _DEFAULT_ONECYCLE_DIV_FACTOR))
            final_div_factor = float(self.scheduler_cfg.get("final_div_factor", _DEFAULT_ONECYCLE_FINAL_DIV_FACTOR))
            three_phase = bool(self.scheduler_cfg.get("three_phase", _DEFAULT_ONECYCLE_THREE_PHASE))
            last_epoch = int(self.scheduler_cfg.get("last_epoch", -1))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                three_phase=three_phase,
                last_epoch=last_epoch,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")
        return {"scheduler": scheduler, "interval": interval}

    def _resolve_onecycle_total_steps(self) -> int:
        total_steps = self.scheduler_cfg.get("total_steps", None)
        if total_steps is not None:
            total_steps = int(total_steps)
            if total_steps <= _ZERO:
                raise ValueError("scheduler_cfg.total_steps must be > 0.")
            return total_steps
        epochs = self.scheduler_cfg.get("epochs", None)
        steps_per_epoch = self.scheduler_cfg.get("steps_per_epoch", None)
        if epochs is not None or steps_per_epoch is not None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError("scheduler_cfg.epochs and scheduler_cfg.steps_per_epoch must be set together.")
            epochs = int(epochs)
            steps_per_epoch = int(steps_per_epoch)
            if epochs <= _ZERO or steps_per_epoch <= _ZERO:
                raise ValueError("scheduler_cfg.epochs and steps_per_epoch must be > 0.")
            return epochs * steps_per_epoch
        trainer = getattr(self, "trainer", None)
        estimated = getattr(trainer, "estimated_stepping_batches", None) if trainer is not None else None
        if estimated is None:
            raise ValueError("OneCycleLR requires total_steps or epochs+steps_per_epoch (trainer not initialized).")
        total_steps = int(estimated)
        if total_steps <= _ZERO:
            raise ValueError("trainer.estimated_stepping_batches must be > 0 for OneCycleLR.")
        return total_steps

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

    def on_fit_start(self) -> None:
        self._check_onecycle_total_steps()

    def _check_onecycle_total_steps(self) -> None:
        if self._onecycle_checked:
            return
        self._onecycle_checked = True
        sched_type = str(self.scheduler_cfg.get("type", "") or "").strip().lower()
        if sched_type not in {"onecycle", "one_cycle", "onecyclelr", _SCHED_TYPE_ONECYCLE}:
            return
        trainer = getattr(self, "trainer", None)
        estimated = getattr(trainer, "estimated_stepping_batches", None) if trainer is not None else None
        configured = None
        source = None
        if "total_steps" in self.scheduler_cfg and self.scheduler_cfg.get("total_steps") is not None:
            configured = int(self.scheduler_cfg.get("total_steps"))
            source = "total_steps"
        elif self.scheduler_cfg.get("epochs") is not None or self.scheduler_cfg.get("steps_per_epoch") is not None:
            epochs = self.scheduler_cfg.get("epochs")
            steps_per_epoch = self.scheduler_cfg.get("steps_per_epoch")
            if epochs is not None and steps_per_epoch is not None:
                configured = int(epochs) * int(steps_per_epoch)
                source = "epochs*steps_per_epoch"
        if estimated is None:
            log_event(logger, "onecycle_total_steps_check", configured=configured, source=source, estimated=None)
            return
        estimated = int(estimated)
        if configured is None:
            log_event(logger, "onecycle_total_steps_check", configured=None, source=None, estimated=estimated)
            return
        diff = abs(int(configured) - estimated)
        ratio = float(diff) / float(estimated) if estimated > _ZERO else 0.0
        log_event(
            logger,
            "onecycle_total_steps_check",
            level=logging.WARNING if ratio >= 0.05 else logging.INFO,
            configured=int(configured),
            source=source,
            estimated=estimated,
            diff=diff,
            ratio=ratio,
        )



    def forward(self, batch: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("DualFlowModule.forward is not supported; use training_step/eval.")

    def training_step(self, batch: Any, batch_idx: int):
        self._ensure_runtime_initialized()
        optimizer = self.optimizers()
        accum = float(self._accumulate_grad_batches())
        if self._should_zero_grad(batch_idx):
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._compute_training_loss(batch)
        metrics.update(self._collect_logit_scale_metrics())
        metrics = self._filter_metrics(metrics, self._get_standard_metrics("train"))
        self.manual_backward(loss / accum)
        if self._should_step_optimizer(batch_idx):
            clip_norm = self.training_cfg.get("grad_clip_norm", None)
            if clip_norm is not None:
                clip_norm = float(clip_norm)
                if clip_norm > float(_ZERO):
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            interval = str(self.scheduler_cfg.get("interval", _SCHED_INTERVAL_EPOCH) or _SCHED_INTERVAL_EPOCH).strip().lower()
            if interval == _SCHED_INTERVAL_STEP:
                self._step_scheduler()
        batch_size = getattr(batch, "num_graphs", None)
        if batch_size is None:
            ptr = getattr(batch, "ptr", None)
            if ptr is None:
                raise AttributeError("Batch missing num_graphs/ptr required for logging batch_size.")
            ptr = torch.as_tensor(ptr)
            batch_size = int(ptr.numel() - _ONE)
        batch_size = int(batch_size)
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
        metrics = self._filter_metrics(metrics, self._get_standard_metrics("val"))
        scope = self._resolve_dataset_scope()
        for name, value in metrics.items():
            scoped_name = f"val/{scope}/{name}"
            log_metric(self, scoped_name, value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            if scope == "full" and name.startswith(("hit@", "recall@", "precision@", "f1@")):
                log_metric(self, f"val/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._ensure_runtime_initialized()
        _ = batch_idx
        metrics, batch_size = self._compute_eval_metrics(batch)
        if batch_size <= _ZERO:
            return
        metrics = self._filter_metrics(metrics, self._get_standard_metrics("test"))
        scope = self._resolve_dataset_scope()
        for name, value in metrics.items():
            scoped_name = f"test/{scope}/{name}"
            log_metric(self, scoped_name, value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            if scope == "full" and name.startswith(("hit@", "recall@", "precision@", "f1@")):
                log_metric(self, f"test/{name}", value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._ensure_runtime_initialized()
        _ = batch_idx, dataloader_idx
        prepared_fwd = self._prepare_batch(batch)
        num_graphs = int(prepared_fwd.num_graphs)
        if num_graphs <= _ZERO:
            return []
        valid_mask = ~prepared_fwd.dummy_mask
        num_nodes_total = int(prepared_fwd.num_nodes_total)
        node_is_target = build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        beam_size = self._resolve_beam_size()
        sample_ids = extract_sample_ids(batch)
        if len(sample_ids) != num_graphs:
            raise ValueError("sample_id length mismatch with batch graph count.")

        predict_mode = str(self.runtime_cfg.get("predict_mode", "full")).strip().lower()
        lite_mode = predict_mode in {"lite", "light", "summary", "fast"}
        beams = None
        beam_state = None
        if lite_mode:
            beam_state = self._beam_search_state(
                prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target
            )
        else:
            beams = self._beam_search(prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target)
        rollouts_per_graph: list[list[dict[str, Any]]] = [[] for _ in range(num_graphs)]
        node_global_cpu = prepared_fwd.node_global_ids.detach().cpu()
        node_is_target_cpu = node_is_target.detach().cpu()
        node_global_np = node_global_cpu.numpy()
        node_is_target_np = node_is_target_cpu.numpy()
        edge_index_np = None
        edge_rel_np = None
        if lite_mode and beam_state is not None:
            beam_nodes_np = beam_state.beam_nodes.detach().cpu().numpy()
            beam_scores_np = beam_state.beam_scores.detach().cpu().numpy()
            beam_paths_np = beam_state.beam_paths.detach().cpu().numpy()
            beam_lengths_np = beam_state.beam_lengths.detach().cpu().numpy()
            for graph_idx in range(num_graphs):
                for beam_idx in range(beam_state.beam_size):
                    stop_node = int(beam_nodes_np[graph_idx, beam_idx])
                    if stop_node < _ZERO:
                        continue
                    score = float(beam_scores_np[graph_idx, beam_idx])
                    length = int(beam_lengths_np[graph_idx, beam_idx])
                    if length <= _ZERO:
                        path = []
                    else:
                        path = beam_paths_np[graph_idx, beam_idx, :length].tolist()
                    success = bool(node_is_target_np[stop_node]) if stop_node >= _ZERO else False
                    rollouts_per_graph[graph_idx].append(
                        {
                            "rollout_index": beam_idx,
                            "score": score,
                            "path_edge_ids": path,
                            "stop_node_id": stop_node,
                            "reach_success": success,
                        }
                    )
        else:
            edge_index_cpu = prepared_fwd.edge_index.detach().cpu()
            edge_rel_cpu = prepared_fwd.edge_relations.detach().cpu()
            edge_index_np = edge_index_cpu.numpy()
            edge_rel_np = edge_rel_cpu.numpy()
            for graph_idx in range(num_graphs):
                beam = beams[graph_idx]
                for beam_idx, (stop_node, score, path) in enumerate(beam):
                    edges_list: list[dict[str, Any]] = []
                    for edge_id in path:
                        head = int(edge_index_np[_ZERO, edge_id])
                        tail = int(edge_index_np[_ONE, edge_id])
                        rel = int(edge_rel_np[edge_id])
                        head_ent = int(node_global_np[head])
                        tail_ent = int(node_global_np[tail])
                        edges_list.append(
                            {
                                "src_entity_id": head_ent,
                                "dst_entity_id": tail_ent,
                                "head_entity_id": head_ent,
                                "tail_entity_id": tail_ent,
                                "relation_id": rel,
                            }
                        )
                    stop_entity = int(node_global_np[stop_node]) if stop_node >= _ZERO else None
                    success = bool(node_is_target_np[stop_node]) if stop_node >= _ZERO else False
                    rollouts_per_graph[graph_idx].append(
                        {
                            "rollout_index": beam_idx,
                            "score": float(score),
                            "edges": edges_list,
                            "stop_node_entity_id": stop_entity,
                            "reach_success": success,
                        }
                    )

        node_ptr_cpu = prepared_fwd.node_ptr.detach().cpu()
        q_ptr_cpu = prepared_fwd.q_ptr.detach().cpu()
        a_ptr_cpu = prepared_fwd.answer_ptr.detach().cpu()
        q_local_cpu = prepared_fwd.q_local_indices.detach().cpu()
        answer_ids_cpu = prepared_fwd.answer_entity_ids.detach().cpu()
        node_ptr_np = node_ptr_cpu.numpy()
        q_ptr_np = q_ptr_cpu.numpy()
        a_ptr_np = a_ptr_cpu.numpy()
        answer_ids_np = answer_ids_cpu.numpy()
        records: list[dict[str, Any]] = []
        for graph_idx in range(num_graphs):
            node_start = int(node_ptr_np[graph_idx])
            node_end = int(node_ptr_np[graph_idx + _ONE])
            q_start = int(q_ptr_np[graph_idx])
            q_end = int(q_ptr_np[graph_idx + _ONE])
            start_indices = q_local_cpu[q_start:q_end].to(dtype=torch.long)
            start_entity_ids: list[int]
            if start_indices.numel() == _ZERO:
                start_entity_ids = []
            else:
                start_indices_np = start_indices.numpy()
                if (start_indices_np < _ZERO).any():
                    raise ValueError(f"q_local_indices contain negative values for sample_id={sample_ids[graph_idx]!r}.")
                if (start_indices_np >= num_nodes_total).any():
                    raise ValueError(f"q_local_indices out of range for sample_id={sample_ids[graph_idx]!r}.")
                in_graph = (start_indices_np >= node_start) & (start_indices_np < node_end)
                if not in_graph.all():
                    raise ValueError(f"q_local_indices mismatch node_ptr for sample_id={sample_ids[graph_idx]!r}.")
                start_entity_ids = node_global_np[start_indices_np].tolist()
            a_start = int(a_ptr_np[graph_idx])
            a_end = int(a_ptr_np[graph_idx + _ONE])
            answer_ids = answer_ids_np[a_start:a_end].tolist() if a_end > a_start else []
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




__all__ = ["DualFlowModule"]
