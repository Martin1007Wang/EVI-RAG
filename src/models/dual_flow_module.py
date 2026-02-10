from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule

from src.metrics.common import extract_sample_ids
from src.models.components import (
    CvtNodeInitializer,
    EmbeddingBackbone,
    LogZPredictor,
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
from src.utils import log_metric
from src.utils.batch_ops import (
    build_dummy_mask,
    build_node_batch,
    build_node_mask,
)
from src.utils.config_utils import require_cfg_mapping, validate_cfg_keys
from src.utils.logging_utils import get_logger, log_event

from src.models.dual_flow_constants import (
    _DB_CFG_KEYS,
    _DB_CFG_OPTIONAL_KEYS,
    _DEFAULT_DEGREE_BUCKETS,
    _DEFAULT_DIVERSE_BEAM_ENABLED,
    _DEFAULT_DIVERSE_BEAM_GROUPS,
    _DEFAULT_DIVERSE_BEAM_LAMBDA,
    _DEFAULT_DIVERSE_BEAM_PENALTY,
    _DEFAULT_DIVERSE_BEAM_SIMILARITY,
    _DEFAULT_GNN_DROPOUT,
    _DEFAULT_GNN_LAYERS,
    _DEFAULT_LOG_Z_BIAS_INIT,
    _DEFAULT_LOGIT_SCALE_INIT,
    _DEFAULT_P0_MODE,
    _DEFAULT_P0_SEMANTIC_SCALE,
    _DEFAULT_EXPLORATION_EPS,
    _DEFAULT_EXPLORATION_WARMUP,
    _DEFAULT_BEAM_METRICS_TOPK,
    _DEFAULT_GAMMA,
    _DEFAULT_MAX_LOG_DEG,
    _DEFAULT_NONFINITE_DEBUG_MAX,
    _DEFAULT_NONFINITE_DEBUG_SEG_MAX,
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
    _DEFAULT_STOP_REWARD_EPSILON,
    _DEFAULT_STOP_REWARD_MODE,
    _STOP_ACTION_ID,
    _DEFAULT_TRAIN_ROLLOUTS,
    _DIVERSE_BEAM_PENALTIES,
    _DIVERSE_BEAM_SIMILARITIES,
    _NEG_ONE,
    _ONE,
    _P0_MODES,
    _STOP_REWARD_MODES,
    _SCHED_INTERVAL_EPOCH,
    _SCHED_INTERVAL_STEP,
    _SCHED_INTERVALS,
    _SCHED_TYPE_COSINE,
    _SCHED_TYPE_COSINE_WARM_RESTARTS,
    _SCHED_TYPE_ONECYCLE,
    _SELF_RELATION_ID,
    _STANDARD_METRICS,
    _TERMINAL_DEAD_END,
    _TERMINAL_EMIT,
    _TERMINAL_HIT,
    _TERMINAL_INVALID_START,
    _TERMINAL_MAX_STEPS,
    _TERMINAL_NONE,
    _THREE,
    _FOUR,
    _TWO,
    _ZERO,
)
from src.models.dual_flow_types import (
    _BeamCandidateMatrix,
    _BeamCandidates,
    _BeamState,
    _HierLogProbs,
    _PreparedBatch,
    _RolloutResult,
)

logger = get_logger(__name__)


class EdgeSetAttentionScorer(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_degree_buckets: int,
        max_log_deg: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_degree_buckets = int(num_degree_buckets)
        if self.num_degree_buckets <= _ZERO:
            raise ValueError("num_degree_buckets must be > 0.")
        if max_log_deg <= float(_ZERO):
            raise ValueError("max_log_deg must be > 0.")
        self.register_buffer("max_log_deg", torch.tensor(float(max_log_deg)), persistent=False)
        self.degree_emb_in = torch.nn.Embedding(self.num_degree_buckets, self.hidden_dim)
        self.degree_emb_out = torch.nn.Embedding(self.num_degree_buckets, self.hidden_dim)
        self.degree_proj = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * _TWO, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * _THREE, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.key_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * _FOUR, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.scale = float(_ONE) / math.sqrt(float(self.hidden_dim))
        self._zero_init_last(self.degree_proj)
        self._zero_init_last(self.query_mlp)
        self._zero_init_last(self.key_mlp)

    @staticmethod
    def _zero_init_last(seq: torch.nn.Sequential) -> None:
        last = None
        for layer in reversed(seq):
            if isinstance(layer, torch.nn.Linear):
                last = layer
                break
        if last is None:
            return
        torch.nn.init.zeros_(last.weight)
        if last.bias is not None:
            torch.nn.init.zeros_(last.bias)

    def _bucketize_degree(self, degree: torch.Tensor) -> torch.Tensor:
        degree = degree.to(device=self.max_log_deg.device, dtype=torch.float32).clamp(min=float(_ZERO))
        log_deg = torch.log(degree + float(_ONE))
        scaled = log_deg / self.max_log_deg * float(self.num_degree_buckets)
        bucket = torch.floor(scaled).to(dtype=torch.long)
        bucket = torch.clamp(bucket, min=_ZERO, max=self.num_degree_buckets - _ONE)
        return bucket

    def _embed_degree(self, deg_in: torch.Tensor, deg_out: torch.Tensor) -> torch.Tensor:
        bucket_in = self._bucketize_degree(deg_in)
        bucket_out = self._bucketize_degree(deg_out)
        emb_in = self.degree_emb_in(bucket_in)
        emb_out = self.degree_emb_out(bucket_out)
        merged = torch.cat((emb_in, emb_out), dim=-1)
        return self.degree_proj(merged)

    def encode_query(
        self,
        *,
        u_emb: torch.Tensor,
        q_emb: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        q_in = torch.cat((u_emb, q_emb, t_emb), dim=-1)
        return self.query_mlp(q_in)

    def encode_edge_key(
        self,
        *,
        u_emb: torch.Tensor,
        r_emb: torch.Tensor,
        v_emb: torch.Tensor,
        deg_in: torch.Tensor,
        deg_out: torch.Tensor,
    ) -> torch.Tensor:
        deg_feat = self._embed_degree(deg_in, deg_out)
        k_in = torch.cat((u_emb, r_emb, v_emb, deg_feat), dim=-1)
        return self.key_mlp(k_in)

    def forward(
        self,
        *,
        u_emb: torch.Tensor,
        r_emb: torch.Tensor,
        v_emb: torch.Tensor,
        q_emb: torch.Tensor,
        t_emb: torch.Tensor,
        deg_in: torch.Tensor,
        deg_out: torch.Tensor,
    ) -> torch.Tensor:
        if t_emb.size(-1) != self.hidden_dim:
            raise ValueError("t_emb must match hidden_dim.")
        query = self.encode_query(u_emb=u_emb, q_emb=q_emb, t_emb=t_emb)
        key = self.encode_edge_key(u_emb=u_emb, r_emb=r_emb, v_emb=v_emb, deg_in=deg_in, deg_out=deg_out)
        return (query * key).sum(dim=-1) * self.scale



class DualFlowModule(LightningModule):
    """Trajectory balance with off-policy rollouts."""


    def __init__(
        self,
        *,
        hidden_dim: int,
        max_steps: int,
        emb_dim: int,
        gnn_layers: int = _DEFAULT_GNN_LAYERS,
        gnn_dropout: float = _DEFAULT_GNN_DROPOUT,
        embedding_adapter_cfg: Optional[Mapping[str, Any]] = None,
        degree_bucket_cfg: Optional[Mapping[str, Any]] = None,
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
        self.embedding_adapter_cfg = embedding_adapter_cfg or {}
        self.degree_bucket_cfg = degree_bucket_cfg or {}
        self.actor_cfg = actor_cfg or {}
        self.runtime_cfg = runtime_cfg or {}
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.logging_cfg = logging_cfg or {}
        self._onecycle_checked = False

        self.register_buffer("prior_weight", torch.tensor(float(_ONE)))
        prior_override = self.runtime_cfg.get("prior_weight_override", None)
        if prior_override is not None:
            self.prior_weight.fill_(float(prior_override))

        self._init_backbone(
            emb_dim=emb_dim,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
        )
        self._init_cvt_init()
        self._init_actor()
        self._validate_cfg_contract()
        self._save_serializable_hparams()
        self._cvt_mask = None
        self._relation_vocab_size = None

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs: Any) -> Any:
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict, **kwargs)
        drop_keys = [key for key in state_dict.keys() if key.startswith("stop_predictor.") or key == "stop_bias"]
        if drop_keys:
            filtered = {key: value for key, value in state_dict.items() if key not in drop_keys}
            return super().load_state_dict(filtered, strict=strict, **kwargs)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def _validate_cfg_contract(self) -> None:
        allowed_training = {
            "accumulate_grad_batches",
            "db_cfg",
            "grad_clip_norm",
            "num_rollouts",
            "lookahead_cfg",
        }
        extra_training = set(self.training_cfg.keys()) - allowed_training
        if extra_training:
            raise ValueError(f"Unsupported training_cfg keys: {sorted(extra_training)}")
        allowed_eval = {"beam_size", "diverse_beam", "answer_gain_stop", "beam_metrics"}
        extra_eval = set(self.evaluation_cfg.keys()) - allowed_eval
        if extra_eval:
            raise ValueError(f"Unsupported evaluation_cfg keys: {sorted(extra_eval)}")
        allowed_actor: set[str] = set()
        extra_actor = set((self.actor_cfg or {}).keys()) - allowed_actor
        if extra_actor:
            raise ValueError(f"Unsupported actor_cfg keys: {sorted(extra_actor)}")

    def _save_serializable_hparams(self) -> None:
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "backbone_fwd",
                "cvt_init_fwd",
                "forward_ctx_proj",
                "z_time_encoder",
                "z_predictor",
                "training_cfg",
                "evaluation_cfg",
                "embedding_adapter_cfg",
                "degree_bucket_cfg",
                "actor_cfg",
                "runtime_cfg",
                "optimizer_cfg",
                "scheduler_cfg",
                "logging_cfg",
            ],
        )

    @staticmethod
    def _coerce_db_cfg(raw: Mapping[str, Any]) -> dict[str, float | int | str]:
        cfg: dict[str, float | int | str] = {
            "sampling_temperature_start": float(raw["sampling_temperature_start"]),
            "sampling_temperature_end": float(raw["sampling_temperature_end"]),
        }
        return cfg

    @staticmethod
    def _validate_db_cfg_values(cfg: Mapping[str, float | int | str]) -> None:
        if float(cfg["sampling_temperature_start"]) <= float(_ZERO) or float(cfg["sampling_temperature_end"]) <= float(
            _ZERO
        ):
            raise ValueError("db_cfg.sampling_temperature_start/end must be > 0.")
        if float(cfg["sampling_temperature_start"]) < float(cfg["sampling_temperature_end"]):
            raise ValueError("db_cfg.sampling_temperature_start must be >= sampling_temperature_end for cosine.")

    def _resolve_db_cfg(self) -> dict[str, float | int | str]:
        raw = require_cfg_mapping(self.training_cfg.get("db_cfg"), "training_cfg.db_cfg")
        validate_cfg_keys(raw, required=_DB_CFG_KEYS, optional=_DB_CFG_OPTIONAL_KEYS, name="db_cfg")
        cfg = self._coerce_db_cfg(raw)
        self._validate_db_cfg_values(cfg)
        return cfg

    def _resolve_degree_bucket_cfg(self) -> dict[str, float | int]:
        raw = require_cfg_mapping(self.degree_bucket_cfg, "degree_bucket_cfg")
        num_buckets = int(raw.get("num_buckets", _DEFAULT_DEGREE_BUCKETS))
        max_log_deg = raw.get("max_log_deg", _DEFAULT_MAX_LOG_DEG)
        if max_log_deg is None:
            raise ValueError("degree_bucket_cfg.max_log_deg is required.")
        max_log_deg = float(max_log_deg)
        if num_buckets <= _ZERO:
            raise ValueError("degree_bucket_cfg.num_buckets must be > 0.")
        if max_log_deg <= float(_ZERO):
            raise ValueError("degree_bucket_cfg.max_log_deg must be > 0.")
        return {"num_buckets": num_buckets, "max_log_deg": max_log_deg}

    def _resolve_p0_cfg(self) -> dict[str, str | float]:
        mode = _DEFAULT_P0_MODE
        semantic_scale = _DEFAULT_P0_SEMANTIC_SCALE
        if isinstance(self.runtime_cfg, Mapping):
            raw_mode = self.runtime_cfg.get("p0_mode", mode)
            if raw_mode is not None:
                mode = str(raw_mode).strip().lower()
            raw_scale = self.runtime_cfg.get("p0_semantic_scale", semantic_scale)
            if raw_scale is not None:
                semantic_scale = float(raw_scale)
        if mode not in _P0_MODES:
            raise ValueError(f"runtime_cfg.p0_mode must be one of {sorted(_P0_MODES)}, got {mode!r}.")
        if semantic_scale <= float(_ZERO):
            raise ValueError("runtime_cfg.p0_semantic_scale must be > 0.")
        return {"mode": mode, "semantic_scale": semantic_scale}

    def _resolve_stop_reward_cfg(self) -> dict[str, str | float]:
        mode = _DEFAULT_STOP_REWARD_MODE
        epsilon = _DEFAULT_STOP_REWARD_EPSILON
        if isinstance(self.runtime_cfg, Mapping):
            raw_mode = self.runtime_cfg.get("stop_reward_mode", mode)
            if raw_mode is not None:
                mode = str(raw_mode).strip().lower()
            raw_eps = self.runtime_cfg.get("stop_reward_epsilon", epsilon)
            if raw_eps is not None:
                epsilon = float(raw_eps)
        if mode not in _STOP_REWARD_MODES:
            raise ValueError(
                f"runtime_cfg.stop_reward_mode must be one of {sorted(_STOP_REWARD_MODES)}, got {mode!r}."
            )
        if epsilon <= float(_ZERO):
            raise ValueError("runtime_cfg.stop_reward_epsilon must be > 0.")
        return {"mode": mode, "epsilon": epsilon}

    def _resolve_logit_scale_init(self) -> float:
        init = _DEFAULT_LOGIT_SCALE_INIT
        if isinstance(self.runtime_cfg, Mapping):
            raw = self.runtime_cfg.get("logit_scale_init", init)
            if raw is not None:
                init = float(raw)
        return float(init)

    def _resolve_log_z_bias_init(self) -> float:
        bias = _DEFAULT_LOG_Z_BIAS_INIT
        if isinstance(self.runtime_cfg, Mapping):
            raw = self.runtime_cfg.get("log_z_bias_init", bias)
            if raw is not None:
                bias = float(raw)
        return float(bias)

    def _resolve_gamma(self) -> float:
        gamma = _DEFAULT_GAMMA
        if isinstance(self.runtime_cfg, Mapping):
            raw = self.runtime_cfg.get("gamma", gamma)
            if raw is not None:
                gamma = float(raw)
        if gamma <= float(_ZERO) or gamma > float(_ONE):
            raise ValueError("runtime_cfg.gamma must be in (0, 1].")
        return float(gamma)

    def _resolve_logit_scale_schedule_cfg(self) -> dict[str, float | bool]:
        raw = None
        if isinstance(self.runtime_cfg, Mapping):
            raw = self.runtime_cfg.get("logit_scale_schedule", None)
        if raw is None:
            return {"enabled": False, "start": 1.0, "end": 1.0}
        raw = require_cfg_mapping(raw, "runtime_cfg.logit_scale_schedule")
        enabled = bool(raw.get("enabled", True))
        start = float(raw.get("start", 1.0))
        end = float(raw.get("end", 1.0))
        if start <= float(_ZERO) or end <= float(_ZERO):
            raise ValueError("runtime_cfg.logit_scale_schedule.start/end must be > 0.")
        return {"enabled": enabled, "start": start, "end": end}

    def _maybe_update_logit_scale_schedule(self) -> None:
        cfg = self._resolve_logit_scale_schedule_cfg()
        if not bool(cfg.get("enabled", False)):
            return
        progress = self._resolve_training_progress()
        start = float(cfg["start"])
        end = float(cfg["end"])
        alpha = start + (end - start) * float(progress)
        log_value = math.log(alpha)
        with torch.no_grad():
            self.logit_scale.fill_(log_value)

    def _resolve_exploration_cfg(self) -> dict[str, float | int]:
        epsilon = _DEFAULT_EXPLORATION_EPS
        warmup_steps = _DEFAULT_EXPLORATION_WARMUP
        if isinstance(self.runtime_cfg, Mapping):
            raw = self.runtime_cfg.get("exploration_cfg", None)
            if raw is not None:
                raw = require_cfg_mapping(raw, "runtime_cfg.exploration_cfg")
                epsilon = float(raw.get("epsilon", epsilon))
                warmup_steps = int(raw.get("warmup_steps", warmup_steps))
        if epsilon < float(_ZERO) or epsilon > float(_ONE):
            raise ValueError("runtime_cfg.exploration_cfg.epsilon must be in [0, 1].")
        if warmup_steps < _ZERO:
            raise ValueError("runtime_cfg.exploration_cfg.warmup_steps must be >= 0.")
        return {"epsilon": epsilon, "warmup_steps": warmup_steps}

    def _resolve_lookahead_cfg(self) -> dict[str, float | bool]:
        raw = self.training_cfg.get("lookahead_cfg", None)
        if raw is None:
            return {"enabled": False, "log_f_floor": 0.0, "boost": 0.0}
        raw = require_cfg_mapping(raw, "training_cfg.lookahead_cfg")
        enabled = bool(raw.get("enabled", False))
        log_f_floor = float(raw.get("log_f_floor", 0.0))
        boost = float(raw.get("boost", 0.0))
        return {"enabled": enabled, "log_f_floor": log_f_floor, "boost": boost}

    def _resolve_sampling_temperature(self) -> float:
        cfg = self._resolve_db_cfg()
        start = float(cfg["sampling_temperature_start"])
        end = float(cfg["sampling_temperature_end"])
        progress = self._resolve_training_progress()
        half = float(_ONE) / float(_TWO)
        cosine = half * (float(_ONE) + math.cos(math.pi * progress))
        return end + (start - end) * cosine

    def _resolve_sampling_prior_weight_override(self) -> Optional[float]:
        if "prior_weight_sampling_override" not in self.runtime_cfg:
            return None
        raw = self.runtime_cfg.get("prior_weight_sampling_override", None)
        if raw is None:
            return None
        return float(raw)

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
        return self._resolve_sampling_temperature()

    def _stop_enabled(self) -> bool:
        return True

    def _resolve_stop_min_steps(self) -> int:
        raw = 0
        if isinstance(self.runtime_cfg, Mapping):
            raw = self.runtime_cfg.get("stop_min_steps", 0)
        min_steps = int(raw)
        if min_steps < 0:
            raise ValueError("runtime_cfg.stop_min_steps must be >= 0.")
        return min_steps

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

    def _resolve_answer_gain_cfg(self) -> dict[str, Any]:
        raw = self.evaluation_cfg.get("answer_gain_stop", None)
        if raw is None:
            return {"enabled": False, "patience": _ZERO, "epsilon": 0.0, "min_beam": _ONE}
        raw = require_cfg_mapping(raw, "evaluation_cfg.answer_gain_stop")
        extra = set(raw.keys()) - {"enabled", "patience", "epsilon", "min_beam"}
        if extra:
            raise ValueError(f"Unsupported evaluation_cfg.answer_gain_stop keys: {sorted(extra)}")
        enabled = bool(raw.get("enabled", False))
        patience = int(raw.get("patience", _ZERO))
        epsilon = float(raw.get("epsilon", 0.0))
        min_beam = int(raw.get("min_beam", _ONE))
        if patience < _ZERO:
            raise ValueError("evaluation_cfg.answer_gain_stop.patience must be >= 0.")
        if epsilon < float(_ZERO):
            raise ValueError("evaluation_cfg.answer_gain_stop.epsilon must be >= 0.")
        if min_beam <= _ZERO:
            raise ValueError("evaluation_cfg.answer_gain_stop.min_beam must be > 0.")
        return {
            "enabled": enabled,
            "patience": patience,
            "epsilon": epsilon,
            "min_beam": min_beam,
        }

    def _resolve_beam_metrics_cfg(self) -> dict[str, Any]:
        raw = self.evaluation_cfg.get("beam_metrics", None)
        if raw is None:
            return {"topk": list(_DEFAULT_BEAM_METRICS_TOPK)}
        raw = require_cfg_mapping(raw, "evaluation_cfg.beam_metrics")
        extra = set(raw.keys()) - {"topk"}
        if extra:
            raise ValueError(f"Unsupported evaluation_cfg.beam_metrics keys: {sorted(extra)}")
        topk_raw = raw.get("topk", _DEFAULT_BEAM_METRICS_TOPK)
        if not isinstance(topk_raw, (list, tuple)):
            raise ValueError("evaluation_cfg.beam_metrics.topk must be a list/tuple of positive ints.")
        topk = [int(value) for value in topk_raw]
        if any(value <= _ZERO for value in topk):
            raise ValueError("evaluation_cfg.beam_metrics.topk must contain positive integers.")
        topk = sorted(set(topk))
        return {"topk": topk}

    def _resolve_beam_size(self) -> int:
        return self._resolve_beam_size_value()

    @staticmethod
    def _apply_answer_dedup(
        *,
        beam_nodes: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_hits: torch.Tensor,
        beam_valid: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs, beam_size = beam_nodes.shape
        if beam_size <= _ONE:
            return beam_valid
        keep = beam_valid.clone()
        beam_nodes_cpu = beam_nodes.detach().cpu()
        beam_scores_cpu = beam_scores.detach().cpu()
        beam_hits_cpu = beam_hits.detach().cpu()
        for g in range(num_graphs):
            best: dict[int, tuple[float, int]] = {}
            for idx in range(beam_size):
                if not bool(beam_hits_cpu[g, idx]):
                    continue
                node_id = int(beam_nodes_cpu[g, idx].item())
                if node_id < _ZERO:
                    continue
                score = float(beam_scores_cpu[g, idx].item())
                prev = best.get(node_id)
                if prev is None:
                    best[node_id] = (score, idx)
                    continue
                prev_score, prev_idx = prev
                if score > prev_score:
                    keep[g, prev_idx] = False
                    best[node_id] = (score, idx)
                else:
                    keep[g, idx] = False
        return keep

    @staticmethod
    def _apply_answer_gain_stop(
        *,
        beam_nodes: torch.Tensor,
        beam_hits: torch.Tensor,
        patience: int,
        epsilon: float,
        min_beam: int,
    ) -> torch.Tensor:
        num_graphs, beam_size = beam_nodes.shape
        if beam_size <= 0 or patience <= 0:
            return torch.full((num_graphs,), beam_size, device=beam_nodes.device, dtype=torch.long)
        cutoffs = torch.full((num_graphs,), beam_size, device=beam_nodes.device, dtype=torch.long)
        beam_nodes_cpu = beam_nodes.detach().cpu()
        beam_hits_cpu = beam_hits.detach().cpu()
        for g in range(num_graphs):
            seen: set[int] = set()
            consec = _ZERO
            cutoff = beam_size
            for idx in range(beam_size):
                node_id = int(beam_nodes_cpu[g, idx].item())
                gain = _ZERO
                if node_id >= 0 and bool(beam_hits_cpu[g, idx].item()):
                    if node_id not in seen:
                        seen.add(node_id)
                        gain = _ONE
                if float(gain) <= float(epsilon):
                    consec += _ONE
                else:
                    consec = _ZERO
                if idx + _ONE >= min_beam and consec >= patience:
                    cutoff = idx + _ONE
                    break
            cutoffs[g] = int(cutoff)
        return cutoffs

    def _init_backbone(
        self,
        *,
        emb_dim: int,
        gnn_layers: int,
        gnn_dropout: float,
    ) -> None:
        self.backbone_fwd = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            gnn_layers=gnn_layers,
            gnn_dropout=gnn_dropout,
            adapter_cfg=self.embedding_adapter_cfg,
        )

    def _init_cvt_init(self) -> None:
        self._cvt_enabled = True
        self.cvt_init_fwd = CvtNodeInitializer()

    def _init_actor(self) -> None:
        self.forward_ctx_proj = self._build_context_mlp(in_dim=self.hidden_dim * _TWO)
        self.start_selector = self._build_start_selector()
        self.node_query_film = torch.nn.Linear(self.hidden_dim, self.hidden_dim * _TWO)
        self._zero_init_linear(self.node_query_film)
        self.edge_scorer = self._build_edge_scorer()
        self.edge_flow_predictor = self._build_edge_flow_predictor()
        self.relation_key_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        EdgeSetAttentionScorer._zero_init_last(self.relation_key_mlp)
        self.z_time_encoder = SinusoidalPositionalEncoding(self.hidden_dim)
        self.z_predictor = LogZPredictor(hidden_dim=self.hidden_dim, context_dim=self.hidden_dim)
        self._init_log_z_bias()
        self.logit_scale = torch.nn.Parameter(torch.tensor(float(self._resolve_logit_scale_init())))
        schedule_cfg = self._resolve_logit_scale_schedule_cfg()
        if schedule_cfg.get("enabled", False):
            start = float(schedule_cfg["start"])
            with torch.no_grad():
                self.logit_scale.fill_(math.log(start))
            self.logit_scale.requires_grad_(False)

    def _build_context_mlp(self, *, in_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _init_log_z_bias(self) -> None:
        self.z_predictor.set_output_bias(self._resolve_log_z_bias_init())

    def _build_start_selector(self) -> torch.nn.Module:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * _TWO, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, _ONE),
        )
        self._zero_init_linear(mlp[_NEG_ONE])
        return mlp

    def _build_edge_policy_mlp(self, *, in_dim: int) -> torch.nn.Module:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, _ONE),
        )
        self._zero_init_linear(mlp[_NEG_ONE])
        return mlp

    def _build_edge_flow_predictor(self) -> torch.nn.Module:
        in_dim = self.hidden_dim * _THREE
        mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, _ONE),
        )
        self._zero_init_linear(mlp[_NEG_ONE])
        return mlp

    def _build_edge_scorer(self) -> torch.nn.Module:
        cfg = self._resolve_degree_bucket_cfg()
        return EdgeSetAttentionScorer(
            hidden_dim=self.hidden_dim,
            num_degree_buckets=cfg["num_buckets"],
            max_log_deg=cfg["max_log_deg"],
        )

    @staticmethod
    def _zero_init_linear(layer: torch.nn.Linear) -> None:
        torch.nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)



    def setup(self, stage: Optional[str] = None) -> None:
        _ = stage
        self._ensure_runtime_initialized()

    def _ensure_runtime_initialized(self) -> None:
        if self._cvt_mask is not None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("datamodule is required to initialize CVT assets.")
        resources = getattr(datamodule, "shared_resources", None)
        if resources is None:
            raise RuntimeError("datamodule.shared_resources is required to initialize CVT assets.")
        self._cvt_mask = resources.cvt_mask



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

    def _inject_query_into_nodes(
        self,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        question_tokens = self._resolve_context_tokens(question_tokens)
        node_batch = node_batch.to(device=node_tokens.device, dtype=torch.long).view(-1)
        context = question_tokens.index_select(0, node_batch)
        modulation = self.node_query_film(context)
        gamma, beta = modulation.chunk(_TWO, dim=-1)
        return node_tokens * (float(_ONE) + gamma) + beta

    @staticmethod
    def _build_step_ids(*, num_graphs: int, step: int, device: torch.device) -> torch.Tensor:
        return torch.full((num_graphs,), step, device=device, dtype=torch.long)

    def _init_prev_relation(self, *, num_graphs: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((num_graphs, self.hidden_dim), device=device, dtype=torch.float32)

    def _update_prev_state(
        self,
        *,
        prev_state: torch.Tensor,
        rel_emb: torch.Tensor,
        update_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_state = prev_state.to(device=rel_emb.device, dtype=rel_emb.dtype)
        if prev_state.size(-1) != self.hidden_dim or rel_emb.size(-1) != self.hidden_dim:
            raise ValueError("prev_state/rel_emb must match hidden_dim.")
        return prev_state

    def _compute_prev_rel_sequences(
        self,
        *,
        rel_tokens: torch.Tensor,
        num_moves: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs, max_steps, hidden_dim = rel_tokens.shape
        device = rel_tokens.device
        dtype = rel_tokens.dtype
        if max_steps == _ZERO:
            empty = torch.zeros((num_graphs, _ZERO, hidden_dim), device=device, dtype=dtype)
            return empty, empty
        zeros = torch.zeros((num_graphs, max_steps, hidden_dim), device=device, dtype=dtype)
        return zeros, zeros

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
            torch._assert((edge_relations >= 0).all(), "edge_relations contains negative values.")
        if edge_relations.numel() > 0:
            rel_max = int(edge_relations.max().item())
            if rel_max < _ZERO:
                raise ValueError("edge_relations must be non-negative.")
            rel_vocab = rel_max + _ONE
        else:
            rel_vocab = _ZERO
        if self._relation_vocab_size is None or self._relation_vocab_size < int(rel_vocab):
            self._relation_vocab_size = int(rel_vocab)
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
        num_edges = int(edge_index.size(1))
        edge_inverse_map = torch.arange(num_edges, device=self.device, dtype=torch.long)
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
        node_tokens = self._inject_query_into_nodes(
            node_tokens=node_tokens,
            question_tokens=question_tokens_fwd_base,
            node_batch=node_batch,
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
        edge_mask_fwd = torch.ones((num_edges,), device=self.device, dtype=torch.bool)
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
        edge_ids_by_head_bwd = edge_ids_by_tail_fwd
        edge_ptr_by_head_bwd = edge_ptr_by_tail_fwd
        edge_ids_by_tail_bwd = edge_ids_by_head_fwd
        edge_ptr_by_tail_bwd = edge_ptr_by_head_fwd
        self._validate_edge_inverse_map(
            edge_inverse_map=edge_inverse_map,
            edge_relations=edge_relations,
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
    ) -> None:
        if edge_inverse_map.numel() == 0:
            return
        edge_inverse_map = edge_inverse_map.view(-1)
        edge_relations = edge_relations.view(-1)
        valid = edge_inverse_map >= 0
        inv_safe = edge_inverse_map[valid]
        idx = torch.arange(edge_inverse_map.numel(), device=edge_inverse_map.device, dtype=edge_inverse_map.dtype)[valid]
        back = edge_inverse_map.index_select(0, inv_safe)
        if not torch.equal(back, idx):
            raise ValueError("Edge inverse map is not symmetric.")

    @staticmethod
    def _raise_non_finite(
        name: str,
        tensor: torch.Tensor,
        *,
        segment_ids: Optional[torch.Tensor] = None,
        num_segments: Optional[int] = None,
        allow_neginf: bool = False,
    ) -> None:
        if allow_neginf:
            bad = torch.isnan(tensor) | torch.isposinf(tensor)
        else:
            bad = ~torch.isfinite(tensor)
        if not bool(bad.any().detach().tolist()):
            return
        bad_idx = torch.nonzero(bad, as_tuple=False).view(-1)
        msg = f"{name} contains non-finite values. count={int(bad_idx.numel())}"
        if (
            segment_ids is not None
            and num_segments is not None
            and segment_ids.numel() == tensor.numel()
            and bad_idx.numel() > 0
        ):
            seg = segment_ids.to(device=tensor.device, dtype=torch.long).view(-1)
            seg_bad = seg.index_select(0, bad_idx)
            unique = torch.unique(seg_bad)
            max_show = min(int(_DEFAULT_NONFINITE_DEBUG_SEG_MAX), int(unique.numel()))
            msg += f" bad_segments={unique[:max_show].tolist()}"
        max_show = min(int(_DEFAULT_NONFINITE_DEBUG_MAX), int(bad_idx.numel()))
        msg += f" sample_idx={bad_idx[:max_show].tolist()}"
        raise RuntimeError(msg)



    def _compute_log_z_for_nodes(
        self,
        *,
        node_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        node_batch: torch.Tensor,
        steps: torch.Tensor,
        node_ids: Optional[torch.Tensor],
        prev_rel_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context_tokens = self._resolve_context_tokens(context_tokens)
        if node_ids is None:
            node_tokens_sel = node_tokens
            node_batch_sel = node_batch
        else:
            node_ids = node_ids.to(device=node_tokens.device, dtype=torch.long).view(-1)
            num_nodes_total = int(node_tokens.size(0))
            if num_nodes_total <= _ZERO:
                raise ValueError("node_tokens must be non-empty when node_ids is provided.")
            node_tokens_sel = node_tokens.index_select(0, node_ids)
            node_batch_sel = node_batch.index_select(0, node_ids)
        steps = steps.to(device=node_tokens_sel.device, dtype=torch.long).view(-1)
        if steps.numel() > _ZERO:
            torch._assert(
                (steps == _ZERO).all(),
                "LogZ is defined at t=0; steps must be all zeros.",
            )
        if steps.numel() == node_tokens_sel.size(0):
            time_emb = self.z_time_encoder(steps)
        else:
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

    def _compute_log_f_for_nodes(
        self,
        *,
        node_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        node_batch: torch.Tensor,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        node_ids = node_ids.to(device=node_tokens.device, dtype=torch.long).view(-1)
        steps = torch.zeros((node_ids.numel(),), device=node_tokens.device, dtype=torch.long)
        return self._compute_log_z_for_nodes(
            node_tokens=node_tokens,
            context_tokens=context_tokens,
            node_batch=node_batch,
            steps=steps,
            node_ids=node_ids,
            prev_rel_emb=None,
        )

    def _compute_log_eps_for_graphs(self, *, prepared: _PreparedBatch) -> torch.Tensor:
        cfg = self._resolve_stop_reward_cfg()
        device = prepared.node_ptr.device
        if cfg["mode"] == "uniform_node":
            node_counts = (prepared.node_ptr[1:] - prepared.node_ptr[:-1]).clamp(min=_ONE)
            return -torch.log(node_counts.to(device=device, dtype=torch.float32))
        log_eps = math.log(float(cfg["epsilon"]))
        return torch.full((int(prepared.num_graphs),), log_eps, device=device, dtype=torch.float32)

    def _compute_log_reward_for_nodes(
        self,
        *,
        prepared: _PreparedBatch,
        node_ids: torch.Tensor,
        node_is_target: torch.Tensor,
    ) -> torch.Tensor:
        node_ids = node_ids.to(device=prepared.node_ptr.device, dtype=torch.long).view(-1)
        node_is_target = node_is_target.to(device=prepared.node_ptr.device, dtype=torch.bool)
        safe_ids = node_ids.clamp(min=_ZERO)
        target = node_is_target.index_select(0, safe_ids)
        node_batch = prepared.node_batch.index_select(0, safe_ids)
        log_eps = self._compute_log_eps_for_graphs(prepared=prepared).index_select(0, node_batch)
        log_r = torch.where(target, torch.zeros_like(log_eps), log_eps)
        return log_r

    def _compute_log_z_for_edges(
        self,
        *,
        prepared: _PreparedBatch,
        node_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
        context_tokens: torch.Tensor,
        prev_rel_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_ids = node_ids.to(device=prepared.node_tokens.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=node_ids.device, dtype=torch.long).view(-1)
        context_tokens = self._resolve_context_tokens(context_tokens)
        if edge_batch.numel() > _ZERO:
            max_batch = edge_batch.max()
            context_len = torch.tensor(context_tokens.size(0), device=max_batch.device)
            torch._assert(context_len > max_batch, "context_tokens length must cover max edge batch index.")
        steps = steps.to(device=node_ids.device, dtype=torch.long).view(-1)
        node_tokens_sel = prepared.node_tokens.index_select(0, node_ids)
        if steps.numel() == node_tokens_sel.size(0):
            time_emb = self.z_time_encoder(steps)
        else:
            max_batch = edge_batch.max()
            steps_num = torch.tensor(steps.numel(), device=max_batch.device)
            torch._assert(steps_num > max_batch, "steps length must cover max edge batch index.")
            time_emb = self.z_time_encoder(steps).index_select(0, edge_batch)
        node_tokens_sel = node_tokens_sel + time_emb
        return self.z_predictor(
            node_tokens=node_tokens_sel,
            question_tokens=context_tokens,
            node_batch=edge_batch,
        )

    def _compute_log_f_for_edges(
        self,
        *,
        prepared: _PreparedBatch,
        node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        _ = (edge_ids, edge_batch, steps)
        node_ids = node_ids.to(device=prepared.node_tokens.device, dtype=torch.long).view(-1)
        context_tokens = self._resolve_context_tokens(context_tokens)
        return self._compute_log_f_for_nodes(
            node_tokens=prepared.node_tokens,
            context_tokens=context_tokens,
            node_batch=prepared.node_batch,
            node_ids=node_ids,
        )

    @staticmethod
    def _compute_log_indegree_bias(
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        indegree = prepared.edge_ptr_by_tail_fwd[1:] - prepared.edge_ptr_by_tail_fwd[:-1]
        indegree = indegree.to(device=edge_ids.device, dtype=torch.float32)
        counts = indegree.index_select(0, tails.clamp(min=_ZERO)).clamp(min=float(_ONE))
        log_bias = -torch.log(counts)
        return log_bias.detach()

    def _compute_edge_logits_components(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        prev_rel_emb: Optional[torch.Tensor] = None,
        prior_weight_override: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_ids.numel() == _ZERO:
            empty = torch.zeros((_ZERO,), device=edge_ids.device, dtype=torch.float32)
            return empty, empty, empty
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        heads = prepared.edge_index[_ZERO].index_select(0, edge_ids)
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        steps = steps.to(device=prepared.node_tokens.device, dtype=torch.long).view(-1)
        if steps.numel() == edge_ids.numel():
            steps_edge = steps
        else:
            max_batch = edge_batch.max()
            steps_num = torch.tensor(steps.numel(), device=max_batch.device)
            torch._assert(steps_num > max_batch, "steps length must cover max edge batch index.")
            steps_edge = steps.index_select(0, edge_batch)
        head_tokens = prepared.node_tokens.index_select(0, heads)
        tail_tokens = prepared.node_tokens.index_select(0, tails)
        relation_tokens = prepared.relation_tokens.index_select(0, edge_ids)
        context_tokens = self._resolve_context_tokens(context_tokens)
        q_edge = context_tokens.index_select(0, edge_batch)
        time_emb = self.z_time_encoder(steps_edge)
        log_bias = self._compute_log_indegree_bias(prepared=prepared, edge_ids=edge_ids)
        log_deg = -log_bias
        in_degree = prepared.edge_ptr_by_tail_fwd[1:] - prepared.edge_ptr_by_tail_fwd[:-1]
        out_degree = prepared.edge_ptr_by_head_fwd[1:] - prepared.edge_ptr_by_head_fwd[:-1]
        in_degree = in_degree.to(device=prepared.edge_index.device, dtype=torch.float32)
        out_degree = out_degree.to(device=prepared.edge_index.device, dtype=torch.float32)
        deg_in = in_degree.index_select(0, tails.clamp(min=_ZERO))
        deg_out = out_degree.index_select(0, tails.clamp(min=_ZERO))
        nn_logits = self.edge_scorer(
            u_emb=head_tokens,
            r_emb=relation_tokens,
            v_emb=tail_tokens,
            q_emb=q_edge,
            t_emb=time_emb,
            deg_in=deg_in,
            deg_out=deg_out,
        ).view(-1)
        log_bias = log_bias.to(device=nn_logits.device, dtype=nn_logits.dtype)
        logits = nn_logits
        if temperature != float(_ONE):
            logits = logits / float(temperature)
        return nn_logits, log_bias, logits

    def _compute_edge_logits(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        prev_rel_emb: Optional[torch.Tensor] = None,
        prior_weight_override: Optional[float] = None,
    ) -> torch.Tensor:
        _, _, logits = self._compute_edge_logits_components(
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            steps=steps,
            temperature=temperature,
            context_tokens=context_tokens,
            prev_rel_emb=prev_rel_emb,
            prior_weight_override=prior_weight_override,
        )
        return logits

    @staticmethod
    def _compute_log_p0_uniform(
        *,
        relation_graph: torch.Tensor,
        relation_inv: torch.Tensor,
        num_graphs: int,
        num_relations: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rel_counts = torch.bincount(relation_graph, minlength=num_graphs).clamp(min=_ONE)
        log_p0_rel = -torch.log(rel_counts.to(dtype=torch.float32)).index_select(0, relation_graph)
        edge_counts = torch.bincount(relation_inv, minlength=num_relations).clamp(min=_ONE)
        log_p0_edge_given_rel = -torch.log(edge_counts.to(dtype=torch.float32)).index_select(0, relation_inv)
        log_p0_edge = log_p0_rel.index_select(0, relation_inv) + log_p0_edge_given_rel
        return log_p0_rel, log_p0_edge

    def _compute_log_p0_semantic(
        self,
        *,
        relation_repr: torch.Tensor,
        relation_graph: torch.Tensor,
        relation_inv: torch.Tensor,
        num_graphs: int,
        relation_tokens: torch.Tensor,
        tail_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation_key = self.relation_key_mlp(relation_repr)
        relation_query = context_tokens.index_select(0, relation_graph)
        relation_scores = (relation_query * relation_key).sum(dim=-1) * float(scale)
        relation_lse = segment_logsumexp_1d(relation_scores, relation_graph, num_graphs)
        log_p0_rel = relation_scores - relation_lse.index_select(0, relation_graph)

        q_edge = context_tokens.index_select(0, edge_batch)
        edge_sem = (q_edge * (tail_tokens + relation_tokens)).sum(dim=-1) * float(scale)
        num_relations = int(log_p0_rel.numel())
        edge_lse = segment_logsumexp_1d(edge_sem, relation_inv, num_relations)
        log_p0_edge_given_rel = edge_sem - edge_lse.index_select(0, relation_inv)
        log_p0_edge = log_p0_rel.index_select(0, relation_inv) + log_p0_edge_given_rel
        return log_p0_rel, log_p0_edge

    @staticmethod
    def _compute_log_stop_from_flows(
        *,
        log_z: torch.Tensor,
        log_sum_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_z = log_z.to(dtype=torch.float32)
        log_sum_z = log_sum_z.to(dtype=log_z.dtype)
        DualFlowModule._raise_non_finite("log_z", log_z)
        DualFlowModule._raise_non_finite("log_sum_z", log_sum_z, allow_neginf=True)
        log_z_total = torch.logaddexp(log_sum_z, log_z)
        log_stop_prob = log_z - log_z_total
        DualFlowModule._raise_non_finite("log_stop_prob", log_stop_prob)
        return log_stop_prob, log_z_total

    def _compute_hierarchical_log_probs(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        parent_nodes: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        node_is_target: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> _HierLogProbs:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        if num_graphs is None:
            num_graphs = int(edge_batch.max().item()) + 1 if edge_batch.numel() > 0 else 0
        num_graphs = int(num_graphs)
        parent_nodes = parent_nodes.to(device=prepared.node_tokens.device, dtype=torch.long).view(-1)
        if parent_nodes.numel() != num_graphs:
            raise ValueError("parent_nodes length mismatch with num_graphs.")
        num_nodes_total = int(prepared.node_tokens.size(0))
        if num_nodes_total <= _ZERO:
            raise ValueError("prepared.node_tokens must be non-empty.")
        if edge_batch.numel() > _ZERO:
            has_edge = torch.bincount(edge_batch, minlength=num_graphs) > _ZERO
            valid_parent = (parent_nodes >= _ZERO) & (parent_nodes < num_nodes_total)
            bad_parent = has_edge & ~valid_parent
            if bad_parent.any():
                bad = torch.nonzero(bad_parent, as_tuple=False).view(-1)
                max_show = min(int(_DEFAULT_NONFINITE_DEBUG_MAX), int(bad.numel()))
                raise RuntimeError(
                    "parent_nodes out of range for graphs with edges. "
                    f"bad_graphs={bad[:max_show].tolist()} parent_nodes={parent_nodes.index_select(0, bad)[:max_show].tolist()} "
                    f"num_nodes_total={num_nodes_total}"
                )
        parent_nodes_safe = parent_nodes.clamp(min=_ZERO, max=max(num_nodes_total - 1, 0))
        if edge_ids.numel() == _ZERO:
            empty = torch.zeros((_ZERO,), device=prepared.edge_index.device, dtype=torch.float32)
            neg_inf = torch.finfo(empty.dtype).min
            log_z = self._compute_log_f_for_nodes(
                node_tokens=prepared.node_tokens,
                context_tokens=context_tokens,
                node_batch=prepared.node_batch,
                node_ids=parent_nodes_safe,
            )
            relation_lse = torch.full((num_graphs,), neg_inf, device=prepared.edge_index.device, dtype=torch.float32)
            if self._stop_enabled():
                stop_log_prob, log_denom = self._compute_log_stop_from_flows(
                    log_z=log_z,
                    log_sum_z=relation_lse,
                )
            else:
                stop_log_prob = torch.full((num_graphs,), neg_inf, device=prepared.edge_index.device, dtype=torch.float32)
                log_denom = relation_lse
            return _HierLogProbs(
                edge_log_prob=empty,
                edge_log_prob_cond=empty,
                relation_log_prob=empty,
                relation_graph=empty,
                relation_id=empty,
                relation_inv=empty,
                stop_log_prob=stop_log_prob,
                relation_batch=empty,
                relation_lse=relation_lse,
                log_z=log_z,
                log_sum_z=relation_lse,
            )

        heads = prepared.edge_index[_ZERO].index_select(0, edge_ids)
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        steps = steps.to(device=prepared.node_tokens.device, dtype=torch.long).view(-1)
        if steps.numel() == num_graphs:
            steps_graph = steps
            steps_edge = steps_graph.index_select(0, edge_batch)
        elif steps.numel() == edge_ids.numel():
            steps_edge = steps
            edge_positions = torch.arange(edge_ids.numel(), device=edge_ids.device, dtype=torch.float32)
            _, argmax = segment_max(edge_positions, edge_batch, num_graphs)
            steps_graph = steps.index_select(0, argmax)
        else:
            raise ValueError("steps length must match num_graphs or edge_ids.")
        head_tokens = prepared.node_tokens.index_select(0, heads)
        tail_tokens = prepared.node_tokens.index_select(0, tails)
        relation_tokens = prepared.relation_tokens.index_select(0, edge_ids)
        context_tokens = self._resolve_context_tokens(context_tokens)
        q_edge = context_tokens.index_select(0, edge_batch)
        time_emb = self.z_time_encoder(steps_edge)

        if edge_batch.numel() > _ZERO:
            expected_heads = parent_nodes.index_select(0, edge_batch)
            mismatch = (heads != expected_heads) & (expected_heads >= _ZERO)
            torch._assert(~mismatch.any(), "edge heads must match parent_nodes for hierarchical logits.")

        parent_nodes_safe = parent_nodes.clamp(min=_ZERO)
        parent_tokens = prepared.node_tokens.index_select(0, parent_nodes_safe)
        q_graph = context_tokens
        time_graph = self.z_time_encoder(steps_graph)
        query_graph = self.edge_scorer.encode_query(u_emb=parent_tokens, q_emb=q_graph, t_emb=time_graph)

        relation_ids = prepared.edge_relations.index_select(0, edge_ids)
        relation_vocab = int(self._relation_vocab_size or 0)
        if relation_vocab <= _ZERO:
            raise RuntimeError("relation vocab size must be initialized before computing hierarchical logits.")
        torch._assert((relation_ids >= _ZERO).all(), "relation_ids must be non-negative.")
        torch._assert((relation_ids < relation_vocab).all(), "relation_ids out of range.")
        relation_batch = edge_batch * relation_vocab + relation_ids
        unique_relation_batch, relation_inv = torch.unique(relation_batch, sorted=True, return_inverse=True)
        relation_graph = unique_relation_batch // relation_vocab
        relation_id = unique_relation_batch % relation_vocab

        num_relations = unique_relation_batch.numel()
        p0_cfg = self._resolve_p0_cfg()
        if p0_cfg["mode"] == "uniform":
            log_p0_rel, log_p0_edge = self._compute_log_p0_uniform(
                relation_graph=relation_graph,
                relation_inv=relation_inv,
                num_graphs=num_graphs,
                num_relations=num_relations,
            )
        else:
            relation_repr = torch.zeros(
                (num_relations, relation_tokens.size(-1)),
                device=relation_tokens.device,
                dtype=relation_tokens.dtype,
            )
            relation_repr.index_add_(0, relation_inv, relation_tokens)
            relation_counts = torch.bincount(relation_inv, minlength=num_relations).to(
                device=relation_tokens.device, dtype=relation_tokens.dtype
            )
            relation_repr = relation_repr / relation_counts.clamp(min=float(_ONE)).unsqueeze(-1)
            log_p0_rel, log_p0_edge = self._compute_log_p0_semantic(
                relation_repr=relation_repr,
                relation_graph=relation_graph,
                relation_inv=relation_inv,
                num_graphs=num_graphs,
                relation_tokens=relation_tokens,
                tail_tokens=tail_tokens,
                context_tokens=query_graph,
                edge_batch=edge_batch,
                scale=float(p0_cfg["semantic_scale"]),
            )

        log_f_tail = self._compute_log_f_for_edges(
            prepared=prepared,
            node_ids=tails,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            steps=steps_edge,
            context_tokens=context_tokens,
        )
        self._raise_non_finite("log_f_tail", log_f_tail, segment_ids=edge_batch, num_segments=num_graphs)
        edge_logits_base = log_p0_edge + log_f_tail
        gamma = self._resolve_gamma()
        if gamma != float(_ONE):
            edge_logits_base = edge_logits_base + math.log(float(gamma))
        if temperature != float(_ONE):
            edge_logits = edge_logits_base / float(temperature)
        else:
            edge_logits = edge_logits_base
        self._raise_non_finite("edge_logits", edge_logits, segment_ids=edge_batch, num_segments=num_graphs)

        # Relation logits are derived by marginalizing edge logits for consistency.
        relation_edge_lse = segment_logsumexp_1d(edge_logits, relation_inv, num_relations)
        self._raise_non_finite(
            "relation_edge_lse",
            relation_edge_lse,
            segment_ids=relation_graph,
            num_segments=num_graphs,
        )
        relation_logits = relation_edge_lse

        relation_lse = segment_logsumexp_1d(relation_logits, relation_graph, num_graphs)
        self._raise_non_finite("relation_lse", relation_lse, allow_neginf=True)
        torch._assert(relation_lse.numel() == num_graphs, "relation_lse length mismatch with num_graphs.")
        log_z = self._compute_log_f_for_nodes(
            node_tokens=prepared.node_tokens,
            context_tokens=context_tokens,
            node_batch=prepared.node_batch,
            node_ids=parent_nodes_safe,
        )
        self._raise_non_finite("log_z", log_z)
        if self._stop_enabled():
            stop_log_prob, log_denom = self._compute_log_stop_from_flows(
                log_z=log_z,
                log_sum_z=relation_lse,
            )
        else:
            neg_inf = torch.finfo(relation_lse.dtype).min
            stop_log_prob = torch.full((num_graphs,), neg_inf, device=relation_lse.device, dtype=relation_lse.dtype)
            log_denom = relation_lse
        self._raise_non_finite("log_denom", log_denom)
        relation_log_prob = relation_logits - log_denom.index_select(0, relation_graph)
        self._raise_non_finite(
            "relation_log_prob",
            relation_log_prob,
            segment_ids=relation_graph,
            num_segments=num_graphs,
        )

        edge_log_prob_cond = edge_logits - relation_edge_lse.index_select(0, relation_inv)
        edge_log_prob = edge_log_prob_cond + relation_log_prob.index_select(0, relation_inv)
        self._raise_non_finite(
            "edge_log_prob_cond",
            edge_log_prob_cond,
            segment_ids=edge_batch,
            num_segments=num_graphs,
        )
        self._raise_non_finite("edge_log_prob", edge_log_prob, segment_ids=edge_batch, num_segments=num_graphs)

        return _HierLogProbs(
            edge_log_prob=edge_log_prob,
            edge_log_prob_cond=edge_log_prob_cond,
            relation_log_prob=relation_log_prob,
            relation_graph=relation_graph,
            relation_id=relation_id,
            relation_inv=relation_inv,
            stop_log_prob=stop_log_prob,
            relation_batch=relation_batch,
            relation_lse=relation_lse,
            log_z=log_z,
            log_sum_z=relation_lse,
        )

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
            edge_mask=edge_mask,
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
        # Any dynamic constraints would be an approximation; the residual absorbs the mismatch.
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
        prev_rel = self._init_prev_relation(num_graphs=num_graphs, device=device)
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
                edge_mask=edge_mask,
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

    def _sample_edges(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        parent_nodes: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        collect_policy_metrics: bool = False,
        prev_rel_emb: Optional[torch.Tensor] = None,
        force_stop_mask: Optional[torch.Tensor] = None,
        prior_weight_override: Optional[float] = None,
        node_is_target: Optional[torch.Tensor] = None,
        lookahead_cfg: Optional[dict[str, float | bool]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        if edge_ids.numel() == _ZERO:
            zeros = torch.zeros((num_graphs,), device=prepared.edge_index.device, dtype=torch.float32)
            return (
                torch.full((num_graphs,), _STOP_ACTION_ID, device=prepared.edge_index.device, dtype=torch.long),
                zeros,
                zeros,
                None,
            )
        if force_stop_mask is not None:
            force_stop_mask = force_stop_mask.to(device=edge_ids.device, dtype=torch.bool).view(-1)
            if force_stop_mask.numel() != num_graphs:
                raise ValueError("force_stop_mask length mismatch with num_graphs.")

        hier = self._compute_hierarchical_log_probs(
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            parent_nodes=parent_nodes,
            steps=steps,
            temperature=temperature,
            context_tokens=context_tokens,
            node_is_target=node_is_target,
            num_graphs=num_graphs,
        )
        edge_log_prob = hier.edge_log_prob
        edge_log_prob_cond = hier.edge_log_prob_cond
        relation_log_prob = hier.relation_log_prob
        relation_graph = hier.relation_graph
        relation_id = hier.relation_id
        stop_log_prob = hier.stop_log_prob
        relation_batch = hier.relation_batch
        relation_lse = hier.relation_lse

        if force_stop_mask is not None and force_stop_mask.any():
            neg_inf = torch.finfo(edge_log_prob.dtype).min
            force_edge = force_stop_mask.index_select(0, relation_graph)
            relation_log_prob = torch.where(force_edge, torch.full_like(relation_log_prob, neg_inf), relation_log_prob)
            stop_log_prob = torch.where(force_stop_mask, torch.zeros_like(stop_log_prob), stop_log_prob)

        sample_relation_log_prob = relation_log_prob
        sample_edge_log_prob_cond = edge_log_prob_cond
        sample_stop_log_prob = stop_log_prob
        if lookahead_cfg is not None and bool(lookahead_cfg.get("enabled", False)):
            raise RuntimeError("Lookahead sampling is not supported.")
        if force_stop_mask is not None and force_stop_mask.any():
            neg_inf = torch.finfo(sample_edge_log_prob_cond.dtype).min
            force_edge = force_stop_mask.index_select(0, relation_graph)
            sample_relation_log_prob = torch.where(force_edge, torch.full_like(sample_relation_log_prob, neg_inf), sample_relation_log_prob)
            sample_stop_log_prob = torch.where(force_stop_mask, torch.zeros_like(sample_stop_log_prob), sample_stop_log_prob)

        relation_scores = sample_relation_log_prob + gumbel_noise_like(sample_relation_log_prob)
        relation_best, relation_argmax = segment_max(relation_scores, relation_graph, num_graphs)
        relation_vocab = int(self._relation_vocab_size or 0)
        if relation_vocab <= _ZERO:
            raise RuntimeError("relation vocab size must be initialized before sampling.")
        selected_relation_batch = (relation_graph * relation_vocab) + relation_id
        selected_relation = selected_relation_batch.index_select(0, relation_argmax)
        selected_relation_id = relation_id.index_select(0, relation_argmax)

        stop_scores = sample_stop_log_prob + gumbel_noise_like(sample_stop_log_prob)
        stop_mask = stop_scores >= relation_best
        if force_stop_mask is not None:
            stop_mask = stop_mask | force_stop_mask

        selected_relation_by_edge = selected_relation.index_select(0, edge_batch)
        edge_keep = relation_batch == selected_relation_by_edge
        edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
        has_edge = edge_counts > 0
        relation_counts = torch.bincount(relation_graph, minlength=num_graphs)
        if (has_edge & (relation_counts == _ZERO)).any():
            bad = torch.nonzero(has_edge & (relation_counts == _ZERO), as_tuple=False).view(-1)
            raise RuntimeError(
                f"Relation groups missing for graphs with edges. bad_graphs={bad.tolist()} "
                f"edge_counts={edge_counts.index_select(0, bad).tolist()}."
            )
        if relation_argmax.numel() == num_graphs and relation_graph.numel() > 0:
            arg_graph = relation_graph.index_select(0, relation_argmax)
            bad_arg = (relation_counts > _ZERO) & (arg_graph != torch.arange(num_graphs, device=arg_graph.device))
            if bad_arg.any():
                bad = torch.nonzero(bad_arg, as_tuple=False).view(-1)
                raise RuntimeError(
                    f"segment_max returned argmax from wrong graph. bad_graphs={bad.tolist()} "
                    f"arg_graph={arg_graph.index_select(0, bad).tolist()}."
                )
        keep_counts = torch.bincount(edge_batch[edge_keep], minlength=num_graphs)
        missing_keep = has_edge & (keep_counts == _ZERO) & ~stop_mask
        if missing_keep.any():
            bad = torch.nonzero(missing_keep, as_tuple=False).view(-1)
            arg_rel = selected_relation_id.index_select(0, bad).tolist()
            rel_counts = relation_counts.index_select(0, bad).tolist()
            edge_ct = edge_counts.index_select(0, bad).tolist()
            finite_rel = torch.isfinite(sample_relation_log_prob)
            finite_counts = torch.bincount(relation_graph[finite_rel], minlength=num_graphs)
            finite_rel_counts = finite_counts.index_select(0, bad).tolist()
            raise RuntimeError(
                "Selected relation has no edges in graph. "
                f"bad_graphs={bad.tolist()} selected_rel={arg_rel} "
                f"relation_counts={rel_counts} edge_counts={edge_ct} finite_relation_counts={finite_rel_counts}."
            )
        neg_inf = torch.finfo(edge_log_prob_cond.dtype).min
        edge_scores = sample_edge_log_prob_cond + gumbel_noise_like(sample_edge_log_prob_cond)
        edge_scores = torch.where(edge_keep, edge_scores, torch.full_like(edge_scores, neg_inf))
        _, argmax = segment_max(edge_scores, edge_batch, num_graphs)
        chosen_edge = edge_ids.index_select(0, argmax)
        log_prob_edge = edge_log_prob.index_select(0, argmax)
        log_prob_chosen = log_prob_edge
        log_prob_chosen = torch.where(stop_mask, stop_log_prob, log_prob_chosen)
        chosen_edge = torch.where(stop_mask, torch.full_like(chosen_edge, _STOP_ACTION_ID), chosen_edge)
        if chosen_edge.numel() > 0:
            valid_check = (~stop_mask) & (chosen_edge >= _ZERO) & has_edge
            if valid_check.any():
                chosen_rel = prepared.edge_relations.index_select(0, chosen_edge.clamp(min=_ZERO))
                expected_rel = selected_relation_id.to(device=chosen_rel.device, dtype=chosen_rel.dtype)
                torch._assert(
                    (chosen_rel[valid_check] == expected_rel[valid_check]).all(),
                    "Chosen edge relation id mismatch with selected relation.",
                )
        no_edge = ~has_edge & ~stop_mask
        if no_edge.any():
            log_prob_chosen = torch.where(no_edge, torch.zeros_like(log_prob_chosen), log_prob_chosen)
            chosen_edge = torch.where(no_edge, torch.full_like(chosen_edge, _NEG_ONE), chosen_edge)
        policy_metrics = None
        if collect_policy_metrics:
            nn_logits, log_bias, _ = self._compute_edge_logits_components(
                prepared=prepared,
                edge_ids=edge_ids,
                edge_batch=edge_batch,
                steps=steps,
                temperature=temperature,
                context_tokens=context_tokens,
                prev_rel_emb=prev_rel_emb,
                prior_weight_override=prior_weight_override,
            )
            edge_count = torch.tensor(nn_logits.numel(), device=nn_logits.device, dtype=torch.float32)
            drift_abs_sum = nn_logits.abs().sum()
            drift_sq_sum = (nn_logits * nn_logits).sum()
            log_deg = (-log_bias).to(device=nn_logits.device, dtype=nn_logits.dtype)
            log_deg_sum = log_deg.sum()
            log_deg_sq_sum = (log_deg * log_deg).sum()
            nn_sum = nn_logits.sum()
            nn_log_deg_sum = (nn_logits * log_deg).sum()
            policy_metrics = {
                "drift_abs_sum": drift_abs_sum,
                "drift_sq_sum": drift_sq_sum,
                "edge_count": edge_count,
                "log_deg_sum": log_deg_sum,
                "log_deg_sq_sum": log_deg_sq_sum,
                "nn_sum": nn_sum,
                "nn_log_deg_sum": nn_log_deg_sum,
            }
            stop_log_prob_graph = stop_log_prob
            log_sum_z = hier.log_sum_z
            log_z = hier.log_z
            valid = torch.isfinite(stop_log_prob_graph)
            if log_sum_z is not None:
                valid = valid & torch.isfinite(log_sum_z)
            if log_z is not None:
                valid = valid & torch.isfinite(log_z)
            if valid.any() and log_sum_z is not None and log_z is not None:
                stop_log_mass = log_z
                stop_minus_relation = stop_log_mass - log_sum_z
                policy_metrics.update(
                    {
                        "stop_logit_sum": stop_log_mass[valid].sum(),
                        "relation_lse_sum": log_sum_z[valid].sum(),
                        "stop_minus_relation_sum": stop_minus_relation[valid].sum(),
                        "stop_stat_count": valid.to(dtype=stop_log_prob_graph.dtype).sum(),
                    }
                )
        log_denom = torch.zeros((num_graphs,), device=edge_ids.device, dtype=log_prob_chosen.dtype)
        return chosen_edge, log_prob_chosen, log_denom, policy_metrics

    def _compute_forward_log_prob(
        self,
        *,
        prepared: _PreparedBatch,
        chosen_edge: torch.Tensor,
        parent_nodes: torch.Tensor,
        move_mask: torch.Tensor,
        steps: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        context_tokens_edge: Optional[torch.Tensor] = None,
        prev_rel_emb: Optional[torch.Tensor] = None,
        node_is_target: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        prior_weight_override: Optional[float] = None,
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
            edge_mask=edge_mask,
        )
        if outgoing.edge_ids.numel() == _ZERO:
            return torch.zeros_like(move_mask, dtype=torch.float32)
        edge_ids = outgoing.edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = outgoing.edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_context = context_tokens_edge if context_tokens_edge is not None else context_tokens
        hier = self._compute_hierarchical_log_probs(
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            parent_nodes=parent_nodes,
            steps=steps,
            temperature=temperature,
            context_tokens=edge_context,
            node_is_target=node_is_target,
            num_graphs=move_mask.numel(),
        )
        edge_log_prob = hier.edge_log_prob
        chosen_edge_safe = chosen_edge.clamp(min=_ZERO)
        chosen_for_edge = chosen_edge_safe.index_select(0, edge_batch)
        match = edge_ids == chosen_for_edge
        neg_inf = torch.finfo(edge_log_prob.dtype).min
        masked = torch.where(match, edge_log_prob, torch.full_like(edge_log_prob, neg_inf))
        chosen_logits, _ = segment_max(masked, edge_batch, move_mask.numel())
        log_pf_edge = chosen_logits
        has_edge = outgoing.has_edge.to(device=log_pf_edge.device, dtype=torch.bool)
        bad = has_edge & ~torch.isfinite(log_pf_edge)
        if bad.any():
            bad_idx = torch.nonzero(bad, as_tuple=False).view(-1)
            max_show = min(int(_DEFAULT_NONFINITE_DEBUG_MAX), int(bad_idx.numel()))
            chosen_bad = chosen_edge.index_select(0, bad_idx)
            parent_bad = parent_nodes.index_select(0, bad_idx)
            msg = (
                "forward log prob non-finite for selected edges. "
                f"count={int(bad_idx.numel())} idx={bad_idx[:max_show].tolist()} "
                f"parent_nodes={parent_bad[:max_show].tolist()} chosen_edge={chosen_bad[:max_show].tolist()}"
            )
            raise RuntimeError(msg)
        log_pf_edge = torch.where(has_edge, log_pf_edge, torch.zeros_like(log_pf_edge))
        log_pf_step = torch.where(move_mask & has_edge, log_pf_edge, torch.zeros_like(log_pf_edge))
        return log_pf_step

    def _compute_stop_log_prob(
        self,
        *,
        prepared: _PreparedBatch,
        parent_nodes: torch.Tensor,
        move_mask: torch.Tensor,
        steps: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        prev_rel_emb: Optional[torch.Tensor] = None,
        node_is_target: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        force_stop_mask: Optional[torch.Tensor] = None,
        prior_weight_override: Optional[float] = None,
    ) -> torch.Tensor:
        if not self._stop_enabled():
            return torch.zeros_like(move_mask, dtype=torch.float32)
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for stop log prob.")
        outgoing = gather_outgoing_edges(
            curr_nodes=parent_nodes,
            edge_ids_by_head=edge_ids_by_head,
            edge_ptr_by_head=edge_ptr_by_head,
            active_mask=move_mask,
        )
        outgoing = self._apply_action_constraints_to_outgoing(
            outgoing,
            num_graphs=move_mask.numel(),
            edge_mask=edge_mask,
        )
        if outgoing.edge_ids.numel() == _ZERO:
            return torch.zeros_like(move_mask, dtype=torch.float32)
        edge_ids = outgoing.edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = outgoing.edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        hier = self._compute_hierarchical_log_probs(
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            parent_nodes=parent_nodes,
            steps=steps,
            temperature=temperature,
            context_tokens=context_tokens,
            node_is_target=node_is_target,
            num_graphs=move_mask.numel(),
        )
        stop_log_prob = hier.stop_log_prob
        bad_stop = move_mask & ~torch.isfinite(stop_log_prob)
        if bad_stop.any():
            bad_idx = torch.nonzero(bad_stop, as_tuple=False).view(-1)
            max_show = min(int(_DEFAULT_NONFINITE_DEBUG_MAX), int(bad_idx.numel()))
            msg = (
                "stop log prob non-finite for active graphs. "
                f"count={int(bad_idx.numel())} idx={bad_idx[:max_show].tolist()}"
            )
            if hier.log_z is not None and hier.log_sum_z is not None:
                log_z_bad = hier.log_z.index_select(0, bad_idx)
                log_sum_bad = hier.log_sum_z.index_select(0, bad_idx)
                msg += f" log_z={log_z_bad[:max_show].tolist()} log_sum_z={log_sum_bad[:max_show].tolist()}"
            raise RuntimeError(msg)
        if force_stop_mask is not None:
            stop_log_prob = torch.where(force_stop_mask, torch.zeros_like(stop_log_prob), stop_log_prob)
        log_pf_stop = stop_log_prob
        log_pf_stop = torch.where(move_mask, log_pf_stop, torch.zeros_like(log_pf_stop))
        return log_pf_stop

    def _rollout_policy(
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
        temperature: float,
        context_tokens: torch.Tensor,
        collect_policy_metrics: bool = False,
        exploration_cfg: Optional[Mapping[str, float | int]] = None,
        edge_mask: Optional[torch.Tensor] = None,
        prior_weight_override: Optional[float] = None,
        lookahead_cfg: Optional[dict[str, float | bool]] = None,
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
        prev_rel = self._init_prev_relation(num_graphs=num_graphs, device=device)
        stop_nodes = torch.full((num_graphs,), _NEG_ONE, device=device, dtype=torch.long)
        actions = None
        log_pf_steps = None
        policy_accum = None
        out_degree = None
        in_degree = None
        if collect_policy_metrics:
            policy_accum = {
                "drift_abs_sum": torch.zeros((), device=device, dtype=torch.float32),
                "drift_sq_sum": torch.zeros((), device=device, dtype=torch.float32),
                "edge_count": torch.zeros((), device=device, dtype=torch.float32),
                "deg_sum": torch.zeros((), device=device, dtype=torch.float32),
                "deg_count": torch.zeros((), device=device, dtype=torch.float32),
                "tail_deg_sum": torch.zeros((), device=device, dtype=torch.float32),
                "tail_deg_count": torch.zeros((), device=device, dtype=torch.float32),
                "log_deg_sum": torch.zeros((), device=device, dtype=torch.float32),
                "log_deg_sq_sum": torch.zeros((), device=device, dtype=torch.float32),
                "nn_sum": torch.zeros((), device=device, dtype=torch.float32),
                "nn_log_deg_sum": torch.zeros((), device=device, dtype=torch.float32),
                "tail_log_deg_sum": torch.zeros((), device=device, dtype=torch.float32),
                "tail_log_deg_count": torch.zeros((), device=device, dtype=torch.float32),
                "stop_logit_sum": torch.zeros((), device=device, dtype=torch.float32),
                "relation_lse_sum": torch.zeros((), device=device, dtype=torch.float32),
                "stop_minus_relation_sum": torch.zeros((), device=device, dtype=torch.float32),
                "stop_stat_count": torch.zeros((), device=device, dtype=torch.float32),
            }
            out_degree = (edge_ptr_by_head[1:] - edge_ptr_by_head[:-1]).to(device=device, dtype=torch.float32)
            in_degree = (prepared.edge_ptr_by_tail_fwd[1:] - prepared.edge_ptr_by_tail_fwd[:-1]).to(
                device=device, dtype=torch.float32
            )
        if record_actions:
            actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=device, dtype=torch.long)
        if record_log_pf:
            log_pf_steps = torch.zeros((num_graphs, self.max_steps), device=device, dtype=torch.float32)
        explore_eps = float(_ZERO)
        explore_active = False
        if exploration_cfg is not None:
            explore_eps = float(exploration_cfg.get("epsilon", 0.0))
            warmup_steps = int(exploration_cfg.get("warmup_steps", 0))
            if explore_eps > float(_ZERO) and warmup_steps > _ZERO:
                global_step = int(getattr(self.trainer, "global_step", self.global_step))
                explore_active = global_step < warmup_steps
        for step in range(int(self.max_steps)):
            force_last = active & (step >= (self.max_steps - 1))
            force_stop_mask = force_last
            outgoing = gather_outgoing_edges(
                curr_nodes=curr_nodes,
                edge_ids_by_head=edge_ids_by_head,
                edge_ptr_by_head=edge_ptr_by_head,
                active_mask=active,
            )
            outgoing = self._apply_action_constraints_to_outgoing(
                outgoing,
                num_graphs=num_graphs,
                edge_mask=edge_mask,
            )
            move_mask = active & outgoing.has_edge
            step_ids = self._build_step_ids(num_graphs=num_graphs, step=step, device=device)
            if collect_policy_metrics and policy_accum is not None and out_degree is not None:
                head_nodes = curr_nodes[move_mask].clamp(min=_ZERO)
                deg_sum = out_degree.index_select(0, head_nodes).sum()
                policy_accum["deg_sum"] = policy_accum["deg_sum"] + deg_sum
                policy_accum["deg_count"] = policy_accum["deg_count"] + move_mask.to(dtype=torch.float32).sum()
            if outgoing.edge_ids.numel() > _ZERO or self._stop_enabled():
                explore_mask = None
                if explore_active and explore_eps > float(_ZERO):
                    rand = torch.rand((num_graphs,), device=device)
                    explore_mask = active & outgoing.has_edge & ~force_stop_mask & (rand < explore_eps)
                chosen_edge, log_pf_step, _, step_metrics = self._sample_edges(
                    prepared=prepared,
                    edge_ids=outgoing.edge_ids,
                    edge_batch=outgoing.edge_batch,
                    num_graphs=num_graphs,
                    parent_nodes=curr_nodes,
                    steps=step_ids,
                    temperature=temperature,
                    context_tokens=context_tokens,
                    collect_policy_metrics=collect_policy_metrics,
                    prev_rel_emb=prev_rel,
                    force_stop_mask=force_stop_mask,
                    prior_weight_override=prior_weight_override,
                    node_is_target=node_is_target,
                    lookahead_cfg=lookahead_cfg,
                )
                if explore_mask is not None and bool(explore_mask.any().detach().tolist()):
                    edge_batch = outgoing.edge_batch
                    edge_ids = outgoing.edge_ids
                    explore_edge = explore_mask.index_select(0, edge_batch)
                    scores = gumbel_noise_like(edge_ids.to(dtype=torch.float32))
                    neg_inf = torch.finfo(scores.dtype).min
                    scores = torch.where(explore_edge, scores, torch.full_like(scores, neg_inf))
                    _, argmax = segment_max(scores, edge_batch, num_graphs)
                    random_edge = edge_ids.index_select(0, argmax)
                    chosen_edge = torch.where(explore_mask, random_edge, chosen_edge)
                    log_pf_step = torch.where(explore_mask, torch.zeros_like(log_pf_step), log_pf_step)
                if collect_policy_metrics and policy_accum is not None and step_metrics is not None:
                    for key, value in step_metrics.items():
                        policy_accum[key] = policy_accum[key] + value.to(device=device)
                stop_mask = (chosen_edge == _STOP_ACTION_ID) & active
                stop_mask = stop_mask | force_stop_mask
                stop_nodes = torch.where(stop_mask, curr_nodes, stop_nodes)
                stop_hits = node_is_target.index_select(0, curr_nodes.clamp(min=_ZERO)) & stop_mask
                stop_reason = torch.where(stop_hits, torch.full_like(stop_reason, _TERMINAL_HIT), stop_reason)
                stop_reason = torch.where(
                    stop_mask & ~stop_hits & force_stop_mask,
                    torch.full_like(stop_reason, _TERMINAL_MAX_STEPS),
                    stop_reason,
                )
                stop_reason = torch.where(
                    stop_mask & ~stop_hits & ~force_stop_mask,
                    torch.full_like(stop_reason, _TERMINAL_EMIT),
                    stop_reason,
                )
                move_mask = active & ~stop_mask & outgoing.has_edge
                chosen_edge = torch.where(move_mask, chosen_edge, torch.full_like(chosen_edge, _NEG_ONE))
                chosen_tail = prepared.edge_index[_ONE].index_select(0, chosen_edge.clamp(min=_ZERO))
                curr_nodes = torch.where(move_mask, chosen_tail, curr_nodes)
                if collect_policy_metrics and policy_accum is not None and in_degree is not None:
                    tail_nodes = chosen_tail[move_mask].clamp(min=_ZERO)
                    tail_deg_sum = in_degree.index_select(0, tail_nodes).sum()
                    policy_accum["tail_deg_sum"] = policy_accum["tail_deg_sum"] + tail_deg_sum
                    policy_accum["tail_deg_count"] = policy_accum["tail_deg_count"] + move_mask.to(
                        dtype=torch.float32
                    ).sum()
                    tail_log_deg = torch.log(in_degree.index_select(0, tail_nodes).clamp(min=float(_ONE)))
                    policy_accum["tail_log_deg_sum"] = policy_accum["tail_log_deg_sum"] + tail_log_deg.sum()
                    policy_accum["tail_log_deg_count"] = policy_accum["tail_log_deg_count"] + move_mask.to(
                        dtype=torch.float32
                    ).sum()
                chosen_rel = prepared.relation_tokens.index_select(0, chosen_edge.clamp(min=_ZERO))
                prev_rel = self._update_prev_state(prev_state=prev_rel, rel_emb=chosen_rel, update_mask=move_mask)
                log_pf_step = torch.where(active, log_pf_step, torch.zeros_like(log_pf_step))
                log_pf_sum = log_pf_sum + log_pf_step
                num_moves = num_moves + move_mask.to(dtype=torch.long)
                if record_actions and actions is not None:
                    actions[:, step] = torch.where(active, chosen_edge, actions[:, step])
                if record_log_pf and log_pf_steps is not None:
                    log_pf_steps[:, step] = torch.where(active, log_pf_step, log_pf_steps[:, step])
                active = active & ~stop_mask & move_mask
            else:
                no_edge = active & ~outgoing.has_edge
                stop_nodes = torch.where(no_edge, curr_nodes, stop_nodes)
                stop_reason = torch.where(no_edge, torch.full_like(stop_reason, _TERMINAL_DEAD_END), stop_reason)
                active = active & outgoing.has_edge
        stop_nodes = torch.where(
            stop_nodes >= _ZERO,
            stop_nodes,
            torch.where(active, curr_nodes, torch.full_like(curr_nodes, _NEG_ONE)),
        )
        if self._stop_enabled():
            stop_reason = torch.where(active, torch.full_like(stop_reason, _TERMINAL_MAX_STEPS), stop_reason)
        else:
            at_target = node_is_target.index_select(0, curr_nodes.clamp(min=_ZERO)) & active
            stop_reason = torch.where(at_target, torch.full_like(stop_reason, _TERMINAL_HIT), stop_reason)
            stop_reason = torch.where(active & ~at_target, torch.full_like(stop_reason, _TERMINAL_MAX_STEPS), stop_reason)
        policy_metrics = None
        if collect_policy_metrics and policy_accum is not None:
            edge_count = policy_accum["edge_count"]
            drift_abs_mean = torch.where(
                edge_count > float(_ZERO),
                policy_accum["drift_abs_sum"] / edge_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            drift_rms = torch.where(
                edge_count > float(_ZERO),
                torch.sqrt(policy_accum["drift_sq_sum"] / edge_count),
                torch.zeros((), device=device, dtype=torch.float32),
            )
            deg_count = policy_accum["deg_count"]
            degree_mean = torch.where(
                deg_count > float(_ZERO),
                policy_accum["deg_sum"] / deg_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            tail_deg_count = policy_accum["tail_deg_count"]
            tail_degree_mean = torch.where(
                tail_deg_count > float(_ZERO),
                policy_accum["tail_deg_sum"] / tail_deg_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            log_deg_sum = policy_accum["log_deg_sum"]
            log_deg_sq_sum = policy_accum["log_deg_sq_sum"]
            nn_sum = policy_accum["nn_sum"]
            nn_log_deg_sum = policy_accum["nn_log_deg_sum"]
            log_deg_mean = torch.where(
                edge_count > float(_ZERO),
                log_deg_sum / edge_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            nn_mean = torch.where(
                edge_count > float(_ZERO),
                nn_sum / edge_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            log_deg_var = torch.where(
                edge_count > float(_ZERO),
                log_deg_sq_sum / edge_count - log_deg_mean * log_deg_mean,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            log_deg_var = torch.clamp(log_deg_var, min=0.0)
            log_deg_std = torch.sqrt(log_deg_var)
            cov = torch.where(
                edge_count > float(_ZERO),
                nn_log_deg_sum / edge_count - nn_mean * log_deg_mean,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            w_eff = torch.where(
                log_deg_var > float(_ZERO),
                cov / log_deg_var,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            tail_log_deg_count = policy_accum["tail_log_deg_count"]
            tail_log_deg_mean = torch.where(
                tail_log_deg_count > float(_ZERO),
                policy_accum["tail_log_deg_sum"] / tail_log_deg_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            log_deg_bias = tail_log_deg_mean - log_deg_mean
            stop_stat_count = policy_accum["stop_stat_count"]
            stop_logit_mean = torch.where(
                stop_stat_count > float(_ZERO),
                policy_accum["stop_logit_sum"] / stop_stat_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            relation_lse_mean = torch.where(
                stop_stat_count > float(_ZERO),
                policy_accum["relation_lse_sum"] / stop_stat_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            stop_minus_relation_mean = torch.where(
                stop_stat_count > float(_ZERO),
                policy_accum["stop_minus_relation_sum"] / stop_stat_count,
                torch.zeros((), device=device, dtype=torch.float32),
            )
            policy_metrics = {
                "policy/candidate_edge_count": edge_count.detach(),
                "policy/decision_step_count": deg_count.detach(),
                "policy/move_step_count": tail_deg_count.detach(),
                "policy/stop/stat_count": stop_stat_count.detach(),
                "policy/candidate_edge/nn_logit_abs_mean": drift_abs_mean.detach(),
                "policy/candidate_edge/nn_logit_rms": drift_rms.detach(),
                "policy/decision_head/out_degree_mean": degree_mean.detach(),
                "policy/move_tail/in_degree_mean": tail_degree_mean.detach(),
                "policy/candidate_edge/tail_log_in_degree_mean": log_deg_mean.detach(),
                "policy/candidate_edge/tail_log_in_degree_std": log_deg_std.detach(),
                "policy/candidate_edge/nn_logit_vs_tail_log_in_degree_slope": w_eff.detach(),
                "policy/move_tail/log_in_degree_mean": tail_log_deg_mean.detach(),
                "policy/move_tail/log_in_degree_minus_candidate_mean": log_deg_bias.detach(),
                "policy/stop/logit_mean": stop_logit_mean.detach(),
                "policy/stop/relation_lse_mean": relation_lse_mean.detach(),
                "policy/stop/logit_minus_relation_lse_mean": stop_minus_relation_mean.detach(),
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
        edge_mask: Optional[torch.Tensor],
    ) -> OutgoingEdges:
        if edge_mask is not None:
            outgoing = self._apply_edge_mask_to_outgoing(outgoing, edge_mask=edge_mask, num_graphs=num_graphs)
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


    def _compute_tb_loss(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        actions: torch.Tensor,
        graph_mask: torch.Tensor,
        stop_reason: torch.Tensor,
        stop_nodes: torch.Tensor,
        log_pf_sum: torch.Tensor,
        sampling_temperature: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = prepared_fwd.node_ptr.device
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        actions = actions.to(device=device, dtype=torch.long)
        stop_reason = stop_reason.to(device=device, dtype=torch.long).view(-1)
        stop_nodes = stop_nodes.to(device=device, dtype=torch.long).view(-1)
        log_pf_sum = log_pf_sum.to(device=device, dtype=torch.float32).view(-1)
        num_graphs, max_steps = actions.shape
        num_edges = int(prepared_fwd.edge_index.size(1))
        if num_edges <= _ZERO:
            raise ValueError("edge_index is empty for this batch; check dataset filtering.")

        start_nodes = prepared_fwd.start_nodes_fwd.to(device=device, dtype=torch.long)
        valid_start = (start_nodes >= _ZERO) & graph_mask
        safe_start = torch.where(valid_start, start_nodes, torch.zeros_like(start_nodes))
        step_ids = torch.zeros((num_graphs,), device=device, dtype=torch.long)
        log_f_start = self._compute_log_z_for_nodes(
            node_tokens=prepared_fwd.node_tokens,
            context_tokens=prepared_fwd.context_tokens,
            node_batch=prepared_fwd.node_batch,
            steps=step_ids,
            node_ids=safe_start,
            prev_rel_emb=None,
        )
        log_f_start = torch.where(valid_start, log_f_start, torch.zeros_like(log_f_start))
        outgoing = gather_outgoing_edges(
            curr_nodes=safe_start,
            edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
            active_mask=valid_start,
        )
        outgoing = self._apply_action_constraints_to_outgoing(
            outgoing,
            num_graphs=num_graphs,
            edge_mask=None,
        )
        if outgoing.edge_ids.numel() > _ZERO:
            hier_start = self._compute_hierarchical_log_probs(
                prepared=prepared_fwd,
                edge_ids=outgoing.edge_ids,
                edge_batch=outgoing.edge_batch,
                parent_nodes=safe_start,
                steps=step_ids,
                temperature=sampling_temperature,
                context_tokens=prepared_fwd.context_tokens,
                node_is_target=None,
                num_graphs=num_graphs,
            )
            log_sum_z_start = hier_start.relation_lse
        else:
            neg_inf = torch.finfo(log_f_start.dtype).min
            log_sum_z_start = torch.full((num_graphs,), neg_inf, device=device, dtype=log_f_start.dtype)
        log_z_start = torch.logaddexp(log_sum_z_start, log_f_start)
        log_z_start = torch.where(valid_start, log_z_start, torch.zeros_like(log_z_start))
        log_eps = self._compute_log_eps_for_graphs(prepared=prepared_fwd).to(
            device=device, dtype=log_z_start.dtype
        )
        log_reward = torch.where(stop_reason == _TERMINAL_HIT, torch.zeros_like(log_z_start), log_eps)

        move_mask = (actions >= _ZERO) & graph_mask.view(-1, _ONE)
        safe_edges = actions.clamp(min=_ZERO)
        tails = prepared_fwd.edge_index[_ONE].index_select(0, safe_edges.reshape(-1)).reshape(num_graphs, max_steps)
        inv_edge = prepared_fwd.edge_inverse_map.index_select(0, safe_edges.reshape(-1)).reshape(num_graphs, max_steps)
        inv_valid = inv_edge >= _ZERO
        active_bwd = move_mask & inv_valid
        log_pb_step, no_allowed = self._compute_pb_log_prob(
            prepared=prepared_fwd,
            chosen_edge=inv_edge.reshape(-1),
            parent_nodes=tails.reshape(-1),
            move_mask=active_bwd.reshape(-1),
            edge_ids_by_head=prepared_fwd.edge_ids_by_head_bwd,
            edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_bwd,
            return_no_allowed=True,
        )
        log_pb_step = log_pb_step.reshape(num_graphs, max_steps)
        log_pb_sum = (log_pb_step * active_bwd.to(dtype=log_pb_step.dtype)).sum(dim=1)
        stop_action = (
            (stop_reason == _TERMINAL_HIT)
            | (stop_reason == _TERMINAL_EMIT)
            | (stop_reason == _TERMINAL_MAX_STEPS)
        )
        if stop_action.any():
            log_pb_stop = torch.zeros_like(log_pb_sum)
            log_pb_sum = log_pb_sum + torch.where(stop_action & graph_mask, log_pb_stop, torch.zeros_like(log_pb_stop))

        weight = torch.ones((num_graphs,), device=device, dtype=torch.float32)

        self._raise_non_finite("tb/log_z_start", log_z_start)
        self._raise_non_finite("tb/log_pf_sum", log_pf_sum)
        self._raise_non_finite("tb/log_pb_sum", log_pb_sum)
        self._raise_non_finite("tb/log_reward", log_reward)
        valid = valid_start
        delta = log_z_start + log_pf_sum - log_reward - log_pb_sum
        delta = torch.where(valid, delta, torch.zeros_like(delta))
        weight = weight * valid.to(dtype=weight.dtype)
        total = (delta.pow(_TWO) * weight).sum()
        denom = weight.sum().clamp(min=_ONE)
        loss = torch.where(denom > float(_ZERO), total / denom, torch.zeros_like(total))

        valid_f = valid.to(dtype=torch.float32)
        denom_valid = valid_f.sum().clamp(min=_ONE)
        log_z_mean = (log_z_start * valid_f).sum() / denom_valid
        log_pf_mean = (log_pf_sum * valid_f).sum() / denom_valid
        log_pb_mean = (log_pb_sum * valid_f).sum() / denom_valid
        log_reward_mean = (log_reward * valid_f).sum() / denom_valid
        delta_mean = (delta * valid_f).sum() / denom_valid
        delta_var = (delta * delta * valid_f).sum() / denom_valid - delta_mean * delta_mean

        move_count = move_mask.to(dtype=torch.float32).sum()
        inv_invalid_count = (move_mask & ~inv_valid).to(dtype=torch.float32).sum()
        move_count_safe = move_count.clamp(min=_ONE)
        zero = torch.zeros_like(move_count)
        inv_edge_missing_rate = torch.where(move_count > _ZERO, inv_invalid_count / move_count_safe, zero)
        pb_step_count = active_bwd.to(dtype=torch.float32).sum()
        pb_no_allowed_count = (no_allowed & active_bwd.reshape(-1)).to(dtype=torch.float32).sum()
        pb_step_count_safe = pb_step_count.clamp(min=_ONE)
        pb_no_allowed_rate = torch.where(pb_step_count > _ZERO, pb_no_allowed_count / pb_step_count_safe, zero)
        tb_valid_graph_count = valid_f.sum()

        metrics = {
            "tb/loss": loss.detach(),
            "tb/valid_graph_count": tb_valid_graph_count.detach(),
            "tb/log_z/start/mean": log_z_mean.detach(),
            "tb/log_pf/mean": log_pf_mean.detach(),
            "tb/log_pb/mean": log_pb_mean.detach(),
            "tb/log_reward/mean": log_reward_mean.detach(),
            "tb/delta/mean": delta_mean.detach(),
            "tb/delta/var_batch": delta_var.detach(),
            "tb/forward_move_count": move_count.detach(),
            "tb/inverse_edge/missing_count": inv_invalid_count.detach(),
            "tb/inverse_edge/missing_rate": inv_edge_missing_rate.detach(),
            "tb/pb/step_count": pb_step_count.detach(),
            "tb/pb/no_allowed_count": pb_no_allowed_count.detach(),
            "tb/pb/no_allowed_rate": pb_no_allowed_rate.detach(),
        }
        return self._ensure_loss_requires_grad(loss), metrics

    def _recompute_log_pf_sum(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        actions: torch.Tensor,
        num_moves: torch.Tensor,
        stop_reason: torch.Tensor,
        stop_nodes: torch.Tensor,
        node_is_target: torch.Tensor,
        sampling_temperature: float,
        graph_mask: torch.Tensor,
        prior_weight_override: Optional[float] = None,
    ) -> torch.Tensor:
        device = prepared_fwd.node_ptr.device
        actions = actions.to(device=device, dtype=torch.long)
        num_moves = num_moves.to(device=device, dtype=torch.long).view(-1)
        stop_reason = stop_reason.to(device=device, dtype=torch.long).view(-1)
        stop_nodes = stop_nodes.to(device=device, dtype=torch.long).view(-1)
        graph_mask = graph_mask.to(device=device, dtype=torch.bool).view(-1)
        num_graphs, max_steps = actions.shape

        move_mask = actions >= _ZERO
        safe_edges = actions.clamp(min=_ZERO)
        heads = prepared_fwd.edge_index[_ZERO].index_select(0, safe_edges.reshape(-1)).reshape(num_graphs, max_steps)
        steps = torch.arange(max_steps, device=device, dtype=torch.long).view(1, -1).expand(num_graphs, max_steps)
        flat_size = num_graphs * max_steps
        log_pf_edges = self._compute_forward_log_prob(
            prepared=prepared_fwd,
            chosen_edge=actions.reshape(-1),
            parent_nodes=heads.reshape(-1),
            move_mask=move_mask.reshape(-1),
            steps=steps.reshape(-1),
            edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
            temperature=sampling_temperature,
            context_tokens=prepared_fwd.context_tokens,
            context_tokens_edge=self._resolve_context_tokens(prepared_fwd.context_tokens).repeat_interleave(
                max_steps, dim=0
            ),
            prev_rel_emb=None,
            node_is_target=node_is_target,
            prior_weight_override=prior_weight_override,
        ).reshape(num_graphs, max_steps)
        log_pf_edges_sum = (log_pf_edges * move_mask.to(dtype=log_pf_edges.dtype)).sum(dim=1)

        stop_steps = num_moves.clamp(min=_ZERO, max=max_steps - 1)
        stop_active = (
            (stop_reason == _TERMINAL_EMIT)
            | (stop_reason == _TERMINAL_HIT)
            | (stop_reason == _TERMINAL_MAX_STEPS)
        ) & graph_mask
        safe_stop_nodes = torch.where(stop_active, stop_nodes, torch.zeros_like(stop_nodes)).clamp(min=_ZERO)
        log_pf_stop = self._compute_stop_log_prob(
            prepared=prepared_fwd,
            parent_nodes=safe_stop_nodes,
            move_mask=stop_active,
            steps=stop_steps,
            edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
            temperature=sampling_temperature,
            context_tokens=prepared_fwd.context_tokens,
            prev_rel_emb=None,
            node_is_target=node_is_target,
            force_stop_mask=None,
            prior_weight_override=prior_weight_override,
        )

        log_pf_sum = log_pf_edges_sum + log_pf_stop
        log_pf_sum = torch.where(graph_mask, log_pf_sum, torch.zeros_like(log_pf_sum))
        return log_pf_sum

    @staticmethod
    def _build_rollout_step_nodes(
        *,
        edge_index: torch.Tensor,
        actions: torch.Tensor,
        start_nodes: torch.Tensor,
    ) -> torch.Tensor:
        actions = actions.to(device=edge_index.device, dtype=torch.long)
        start_nodes = start_nodes.to(device=edge_index.device, dtype=torch.long)
        num_graphs, max_steps = actions.shape
        if max_steps <= _ZERO:
            return torch.zeros((num_graphs, 0), device=edge_index.device, dtype=torch.long)
        tails = edge_index[_ONE].index_select(0, actions.clamp(min=_ZERO).view(-1)).view(num_graphs, max_steps)
        nodes_before = torch.empty_like(tails)
        nodes_before[:, 0] = start_nodes
        if max_steps > _ONE:
            nodes_before[:, _ONE:] = tails[:, : max_steps - _ONE]
        return nodes_before

    def _compute_rollout_reach_metrics(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        actions: torch.Tensor,
        num_moves: torch.Tensor,
        stop_reason: torch.Tensor,
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graph_mask = graph_mask.to(device=prepared_fwd.node_ptr.device, dtype=torch.bool)
        actions = actions.to(device=graph_mask.device, dtype=torch.long)
        num_moves = num_moves.to(device=graph_mask.device, dtype=torch.long).view(-1)
        stop_reason = stop_reason.to(device=graph_mask.device, dtype=torch.long).view(-1)
        nodes_before = self._build_rollout_step_nodes(
            edge_index=prepared_fwd.edge_index,
            actions=actions,
            start_nodes=prepared_fwd.start_nodes_fwd,
        )
        num_graphs, max_steps = actions.shape
        if max_steps <= _ZERO:
            zero = torch.zeros((), device=graph_mask.device, dtype=torch.float32)
            return {
                "rollout/reach/target_any_count": zero,
                "rollout/reach/target_any_rate": zero,
                "rollout/terminal/hit_given_reach_target_any_rate": zero,
            }
        steps = torch.arange(max_steps, device=graph_mask.device, dtype=torch.long).view(1, -1)
        start_nodes = prepared_fwd.start_nodes_fwd.to(device=graph_mask.device, dtype=torch.long).view(-1)
        start_hit = node_is_target.index_select(0, start_nodes.clamp(min=_ZERO)) & graph_mask
        tails = prepared_fwd.edge_index[_ONE].index_select(0, actions.clamp(min=_ZERO).view(-1)).view(num_graphs, max_steps)
        move_mask = steps < num_moves.view(-1, _ONE)
        tail_hits = node_is_target.index_select(0, tails.clamp(min=_ZERO).view(-1)).view(num_graphs, max_steps)
        reach_any = start_hit | (tail_hits & move_mask & graph_mask.view(-1, _ONE)).any(dim=1)
        reach_count = reach_any.to(dtype=torch.float32).sum()
        valid_count = graph_mask.to(dtype=torch.float32).sum()
        denom = valid_count.clamp(min=_ONE)
        reach_rate = reach_count / denom
        hit_count = ((stop_reason == _TERMINAL_HIT) & graph_mask).to(dtype=torch.float32).sum()
        hit_given_reach_rate = torch.where(
            reach_count > float(_ZERO),
            hit_count / reach_count,
            torch.zeros_like(hit_count),
        )
        return {
            "rollout/reach/target_any_count": reach_count,
            "rollout/reach/target_any_rate": reach_rate,
            "rollout/terminal/hit_given_reach_target_any_rate": hit_given_reach_rate,
        }

    def _compute_beam_reach_mask(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        beam_paths: torch.Tensor,
        beam_lengths: torch.Tensor,
        beam_nodes: torch.Tensor,
        node_is_target: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs, beam_size, max_steps = beam_paths.shape
        if max_steps <= _ZERO or beam_size <= _ZERO:
            return torch.zeros((num_graphs, beam_size), device=beam_paths.device, dtype=torch.bool)
        flat_paths = beam_paths.view(-1, max_steps)
        flat_lengths = beam_lengths.view(-1)
        flat_nodes = beam_nodes.view(-1)
        first_edge = flat_paths[:, _ZERO]
        first_head = prepared_fwd.edge_index[_ZERO].index_select(0, first_edge.clamp(min=_ZERO))
        start_nodes = torch.where(flat_lengths > _ZERO, first_head, flat_nodes)
        nodes_before = self._build_rollout_step_nodes(
            edge_index=prepared_fwd.edge_index,
            actions=flat_paths,
            start_nodes=start_nodes,
        )
        tails = prepared_fwd.edge_index[_ONE].index_select(0, flat_paths.clamp(min=_ZERO).view(-1)).view(
            -1, max_steps
        )
        steps = torch.arange(max_steps, device=beam_paths.device, dtype=torch.long).view(_ONE, -1)
        active = steps < flat_lengths.view(-1, _ONE)
        nodes_before_safe = nodes_before.clamp(min=_ZERO)
        tails_safe = tails.clamp(min=_ZERO)
        hits_before = node_is_target.index_select(0, nodes_before_safe.view(-1)).view(-1, max_steps) & active
        hits_after = node_is_target.index_select(0, tails_safe.view(-1)).view(-1, max_steps) & active
        reach = (hits_before | hits_after).any(dim=1)
        start_hits = node_is_target.index_select(0, start_nodes.clamp(min=_ZERO))
        reach = torch.where(flat_lengths > _ZERO, reach, start_hits)
        return reach.view(num_graphs, beam_size)

    @staticmethod
    def _build_terminal_metrics(
        *,
        stop_reason: torch.Tensor,
        graph_mask: torch.Tensor,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        stop_reason = stop_reason.to(device=graph_mask.device, dtype=torch.long)
        graph_mask = graph_mask.to(device=stop_reason.device, dtype=torch.bool)
        valid_count = graph_mask.to(dtype=torch.float32).sum()
        denom = valid_count.clamp(min=_ONE)
        hit_count = ((stop_reason == _TERMINAL_HIT) & graph_mask).to(dtype=torch.float32).sum()
        dead_count = ((stop_reason == _TERMINAL_DEAD_END) & graph_mask).to(dtype=torch.float32).sum()
        max_steps_count = ((stop_reason == _TERMINAL_MAX_STEPS) & graph_mask).to(dtype=torch.float32).sum()
        invalid_start_count = ((stop_reason == _TERMINAL_INVALID_START) & graph_mask).to(dtype=torch.float32).sum()
        emit_count = ((stop_reason == _TERMINAL_EMIT) & graph_mask).to(dtype=torch.float32).sum()
        other_count = ((stop_reason == _TERMINAL_NONE) & graph_mask).to(dtype=torch.float32).sum()
        hit_rate = hit_count / denom
        dead_rate = dead_count / denom
        max_steps_rate = max_steps_count / denom
        invalid_start_rate = invalid_start_count / denom
        emit_rate = emit_count / denom
        other_rate = other_count / denom
        return {
            f"{prefix}/valid_graph_count": valid_count,
            f"{prefix}/terminal/hit_count": hit_count,
            f"{prefix}/terminal/dead_end_count": dead_count,
            f"{prefix}/terminal/max_steps_count": max_steps_count,
            f"{prefix}/terminal/invalid_start_count": invalid_start_count,
            f"{prefix}/terminal/emit_count": emit_count,
            f"{prefix}/terminal/other_count": other_count,
            f"{prefix}/terminal/hit_rate": hit_rate,
            f"{prefix}/terminal/dead_end_rate": dead_rate,
            f"{prefix}/terminal/max_steps_rate": max_steps_rate,
            f"{prefix}/terminal/invalid_start_rate": invalid_start_rate,
            f"{prefix}/terminal/emit_rate": emit_rate,
            f"{prefix}/terminal/other_rate": other_rate,
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
        sampling_prior_weight = self._resolve_sampling_prior_weight_override()
        with torch.no_grad():
            rollout_fwd = self._rollout_policy(
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
                collect_policy_metrics=True,
                exploration_cfg=self._resolve_exploration_cfg(),
                prior_weight_override=sampling_prior_weight,
                lookahead_cfg=self._resolve_lookahead_cfg(),
            )
        if rollout_fwd.actions is None:
            raise RuntimeError("Rollout actions are required for trajectory balance training.")
        log_pf_sum = self._recompute_log_pf_sum(
            prepared_fwd=prepared_fwd,
            actions=rollout_fwd.actions,
            num_moves=rollout_fwd.num_moves,
            stop_reason=rollout_fwd.stop_reason,
            stop_nodes=rollout_fwd.stop_nodes,
            node_is_target=node_is_target,
            sampling_temperature=sampling_temperature,
            graph_mask=graph_mask,
        )
        tb_loss, tb_metrics = self._compute_tb_loss(
            prepared_fwd=prepared_fwd,
            actions=rollout_fwd.actions,
            graph_mask=graph_mask,
            stop_reason=rollout_fwd.stop_reason,
            stop_nodes=rollout_fwd.stop_nodes,
            log_pf_sum=log_pf_sum,
            sampling_temperature=sampling_temperature,
        )
        lengths = rollout_fwd.num_moves.to(dtype=torch.float32)
        denom = graph_mask.to(dtype=lengths.dtype).sum().clamp(min=_ONE)
        length_mean = (lengths * graph_mask.to(dtype=lengths.dtype)).sum() / denom
        min_steps = self._resolve_stop_min_steps()
        emit_mask = (rollout_fwd.stop_reason == _TERMINAL_EMIT) & graph_mask
        emit_count = emit_mask.to(dtype=torch.float32).sum()
        emit_at_min_steps_count = (emit_mask & (rollout_fwd.num_moves == min_steps)).to(dtype=torch.float32).sum()
        emit_at_min_steps_rate = torch.where(
            emit_count > float(_ZERO),
            emit_at_min_steps_count / emit_count.clamp(min=float(_ONE)),
            torch.zeros_like(emit_count),
        )
        metrics = {
            **tb_metrics,
            "rollout/num_moves_mean": length_mean,
            "rollout/terminal/emit_at_stop_min_steps_count": emit_at_min_steps_count,
            "rollout/terminal/emit_at_stop_min_steps_given_emit_rate": emit_at_min_steps_rate,
        }
        metrics.update(
            self._compute_rollout_reach_metrics(
                prepared_fwd=prepared_fwd,
                actions=rollout_fwd.actions,
                num_moves=rollout_fwd.num_moves,
                stop_reason=rollout_fwd.stop_reason,
                graph_mask=graph_mask,
                node_is_target=node_is_target,
            )
        )
        if rollout_fwd.policy_metrics:
            metrics.update(rollout_fwd.policy_metrics)
        metrics.update(
            self._build_terminal_metrics(
                stop_reason=rollout_fwd.stop_reason,
                graph_mask=graph_mask,
                prefix="rollout",
            )
        )
        return tb_loss, metrics

    def _run_backward_rollout(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        graph_mask: torch.Tensor,
        node_is_start: torch.Tensor,
        start_nodes_bwd: torch.Tensor,
        sampling_temperature: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _ = (prepared_fwd, graph_mask, node_is_start, start_nodes_bwd, sampling_temperature)
        raise RuntimeError("Backward rollouts are disabled under trajectory balance training.")

    @staticmethod
    def _metric_is_count(name: str) -> bool:
        return name.endswith("_count")

    @staticmethod
    def _resolve_metric_denom_key(name: str) -> Optional[str]:
        if name.endswith("_count"):
            return None
        if name == "rollout/terminal/hit_given_reach_target_any_rate":
            return "rollout/reach/target_any_count"
        if name == "rollout/terminal/emit_at_stop_min_steps_given_emit_rate":
            return "rollout/terminal/emit_count"
        if name == "tb/inverse_edge/missing_rate":
            return "tb/forward_move_count"
        if name == "tb/pb/no_allowed_rate":
            return "tb/pb/step_count"
        if name == "policy/decision_head/out_degree_mean":
            return "policy/decision_step_count"
        if name == "policy/move_tail/in_degree_mean":
            return "policy/move_step_count"
        if name.startswith("policy/candidate_edge/"):
            return "policy/candidate_edge_count"
        if name.startswith("rollout/"):
            return "rollout/valid_graph_count"
        if name.startswith("tb/"):
            return "tb/valid_graph_count"
        return None

    def _aggregate_training_rollouts(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        sampling_temperature: float,
        num_rollouts: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if num_rollouts <= _ZERO:
            raise ValueError("num_rollouts must be > 0.")
        losses: list[torch.Tensor] = []
        metric_series: dict[str, list[torch.Tensor]] = {}
        for _ in range(num_rollouts):
            tb_loss, metrics = self._run_training_rollout(
                prepared_fwd=prepared_fwd,
                graph_mask=graph_mask,
                node_is_target=node_is_target,
                sampling_temperature=sampling_temperature,
            )
            losses.append(tb_loss)
            for name, value in metrics.items():
                metric_series.setdefault(name, []).append(value)
        loss = torch.stack(losses).mean()
        aggregated: dict[str, torch.Tensor] = {}
        for name, values in metric_series.items():
            stacked = torch.stack(values)
            if self._metric_is_count(name):
                aggregated[name] = stacked.sum()
                continue
            denom_key = self._resolve_metric_denom_key(name)
            if denom_key is not None and denom_key in metric_series:
                denom = torch.stack(metric_series[denom_key]).to(dtype=torch.float32)
                weight = denom
                weight_sum = weight.sum()
                value_f = stacked.to(dtype=torch.float32)
                weighted_sum = (value_f * weight).sum()
                aggregated[name] = torch.where(
                    weight_sum > float(_ZERO),
                    weighted_sum / weight_sum.clamp(min=float(_ONE)),
                    torch.zeros_like(weighted_sum),
                )
                continue
            aggregated[name] = stacked.mean()
        aggregated["loss_total"] = loss.detach()
        return loss, aggregated

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

    def _compute_training_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prepared_fwd = self._prepare_batch(batch)
        graph_mask = self._validate_training_batch(prepared_fwd)
        num_nodes_total = int(prepared_fwd.num_nodes_total)
        node_is_target = build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        sampling_temperature = self._resolve_sampling_temperature()
        num_rollouts = self._resolve_num_rollouts()
        loss, metrics = self._aggregate_training_rollouts(
            prepared_fwd=prepared_fwd,
            graph_mask=graph_mask,
            node_is_target=node_is_target,
            sampling_temperature=sampling_temperature,
            num_rollouts=num_rollouts,
        )
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
        safe_start = torch.where(valid, start_nodes, torch.zeros_like(start_nodes))
        step_ids = torch.zeros((prepared_fwd.num_graphs,), device=prepared_fwd.node_ptr.device, dtype=torch.long)
        log_z = self._compute_log_z_for_nodes(
            node_tokens=prepared_fwd.node_tokens,
            context_tokens=prepared_fwd.context_tokens,
            node_batch=prepared_fwd.node_batch,
            steps=step_ids,
            node_ids=safe_start,
            prev_rel_emb=None,
        )
        valid_f = valid.to(dtype=log_z.dtype)
        denom = valid_f.sum().clamp(min=_ONE)
        masked = log_z * valid_f
        mean = masked.sum() / denom
        var = ((masked - mean) * valid_f).pow(_TWO).sum() / denom
        std = torch.sqrt(var)
        empty = valid_f.sum() <= float(_ZERO)
        mean = torch.where(empty, torch.zeros_like(mean), mean)
        std = torch.where(empty, torch.zeros_like(std), std)
        return {"log_z_mean": mean.detach(), "log_z_std": std.detach()}

    @staticmethod
    def _ensure_loss_requires_grad(loss: torch.Tensor) -> torch.Tensor:
        if loss.requires_grad:
            return loss
        return loss + torch.zeros((), device=loss.device, dtype=loss.dtype, requires_grad=True)



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

    def _build_eval_start_expansion(
        self,
        *,
        prepared: _PreparedBatch,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        counts = (prepared.q_ptr[1:] - prepared.q_ptr[:-1]).clamp(min=0)
        num_graphs = counts.numel()
        if num_graphs <= 0:
            return None
        total = int(counts.sum().item())
        if total <= 0:
            return None
        start_nodes = prepared.q_local_indices.to(device=prepared.node_ptr.device, dtype=torch.long).view(-1)
        graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=start_nodes.device), counts)
        if graph_ids.numel() != start_nodes.numel():
            raise ValueError("q_local_indices length mismatch with q_ptr.")
        question_tokens = self._resolve_context_tokens(
            self.backbone_fwd.project_question_embeddings(prepared.question_emb_raw)
        )
        question_sel = question_tokens.index_select(0, graph_ids)
        start_tokens = prepared.node_tokens.index_select(0, start_nodes)
        context_tokens = self._build_forward_context(question_tokens=question_sel, start_tokens=start_tokens)
        return start_nodes, context_tokens, graph_ids, counts

    def _merge_start_beams(
        self,
        *,
        beam_state: _BeamState,
        start_graph_ids: torch.Tensor,
        counts: torch.Tensor,
    ) -> _BeamState:
        num_graphs = int(counts.numel())
        if num_graphs <= 0 or start_graph_ids.numel() == 0:
            return _BeamState(
                beam_nodes=torch.zeros((num_graphs, 0), device=beam_state.beam_nodes.device, dtype=torch.long),
                beam_scores=torch.zeros((num_graphs, 0), device=beam_state.beam_nodes.device, dtype=torch.float32),
                beam_paths=torch.zeros((num_graphs, 0, 0), device=beam_state.beam_nodes.device, dtype=torch.long),
                beam_lengths=torch.zeros((num_graphs, 0), device=beam_state.beam_nodes.device, dtype=torch.long),
                beam_done=torch.zeros((num_graphs, 0), device=beam_state.beam_nodes.device, dtype=torch.bool),
                flat_graph_ids=torch.zeros((0,), device=beam_state.beam_nodes.device, dtype=torch.long),
                flat_beam_ids=torch.zeros((0,), device=beam_state.beam_nodes.device, dtype=torch.long),
                beam_context=beam_state.beam_context[:0],
                beam_prev_rel=torch.zeros((num_graphs, 0, self.hidden_dim), device=beam_state.beam_nodes.device, dtype=beam_state.beam_context.dtype),
                num_graphs=num_graphs,
                beam_size=0,
                max_steps=beam_state.max_steps,
                neg_inf=beam_state.neg_inf,
            )
        device = beam_state.beam_nodes.device
        counts = counts.to(device=device, dtype=torch.long).view(-1)
        start_graph_ids = start_graph_ids.to(device=device, dtype=torch.long).view(-1)
        max_starts = int(counts.max().item()) if counts.numel() > 0 else 0
        total_beam = int(beam_state.beam_size) * max_starts
        if total_beam <= 0:
            return _BeamState(
                beam_nodes=torch.zeros((num_graphs, 0), device=device, dtype=torch.long),
                beam_scores=torch.zeros((num_graphs, 0), device=device, dtype=torch.float32),
                beam_paths=torch.zeros((num_graphs, 0, 0), device=device, dtype=torch.long),
                beam_lengths=torch.zeros((num_graphs, 0), device=device, dtype=torch.long),
                beam_done=torch.zeros((num_graphs, 0), device=device, dtype=torch.bool),
                flat_graph_ids=torch.zeros((0,), device=device, dtype=torch.long),
                flat_beam_ids=torch.zeros((0,), device=device, dtype=torch.long),
                beam_context=beam_state.beam_context[:0],
                beam_prev_rel=torch.zeros((num_graphs, 0, self.hidden_dim), device=device, dtype=beam_state.beam_context.dtype),
                num_graphs=num_graphs,
                beam_size=0,
                max_steps=beam_state.max_steps,
                neg_inf=beam_state.neg_inf,
            )
        neg_inf = beam_state.neg_inf
        max_steps = int(beam_state.max_steps)
        beam_nodes = torch.full((num_graphs, total_beam), _NEG_ONE, device=device, dtype=torch.long)
        beam_scores = torch.full((num_graphs, total_beam), neg_inf, device=device, dtype=beam_state.beam_scores.dtype)
        beam_lengths = torch.zeros((num_graphs, total_beam), device=device, dtype=beam_state.beam_lengths.dtype)
        beam_done = torch.zeros((num_graphs, total_beam), device=device, dtype=beam_state.beam_done.dtype)
        beam_paths = torch.full((num_graphs, total_beam, max_steps), _NEG_ONE, device=device, dtype=beam_state.beam_paths.dtype)
        start_offsets = (counts.cumsum(0) - counts).index_select(0, start_graph_ids)
        start_pos = torch.arange(start_graph_ids.numel(), device=device, dtype=torch.long) - start_offsets
        beam_size = int(beam_state.beam_size)
        pos = start_pos.unsqueeze(1) * beam_size + torch.arange(beam_size, device=device).unsqueeze(0)
        flat_pos = pos.reshape(-1)
        flat_graph = start_graph_ids.unsqueeze(1).expand_as(pos).reshape(-1)
        linear = flat_graph * total_beam + flat_pos
        beam_nodes.view(-1).index_copy_(0, linear, beam_state.beam_nodes.reshape(-1))
        beam_scores.view(-1).index_copy_(0, linear, beam_state.beam_scores.reshape(-1))
        beam_lengths.view(-1).index_copy_(0, linear, beam_state.beam_lengths.reshape(-1))
        beam_done.view(-1).index_copy_(0, linear, beam_state.beam_done.reshape(-1))
        beam_paths.view(-1, max_steps).index_copy_(0, linear, beam_state.beam_paths.reshape(-1, max_steps))
        sort_scores, sort_idx = torch.sort(beam_scores, dim=1, descending=True)
        beam_scores = sort_scores
        beam_nodes = beam_nodes.gather(1, sort_idx)
        beam_lengths = beam_lengths.gather(1, sort_idx)
        beam_done = beam_done.gather(1, sort_idx)
        path_idx = sort_idx.unsqueeze(-1).expand(-1, -1, max_steps)
        beam_paths = beam_paths.gather(1, path_idx)
        flat_graph_ids = torch.arange(num_graphs, device=device).repeat_interleave(total_beam)
        flat_beam_ids = torch.arange(total_beam, device=device).repeat(num_graphs)
        beam_prev_rel = torch.zeros((num_graphs, total_beam, self.hidden_dim), device=device, dtype=beam_state.beam_context.dtype)
        return _BeamState(
            beam_nodes=beam_nodes,
            beam_scores=beam_scores,
            beam_paths=beam_paths,
            beam_lengths=beam_lengths,
            beam_done=beam_done,
            flat_graph_ids=flat_graph_ids,
            flat_beam_ids=flat_beam_ids,
            beam_context=beam_state.beam_context[:0],
            beam_prev_rel=beam_prev_rel,
            num_graphs=num_graphs,
            beam_size=total_beam,
            max_steps=max_steps,
            neg_inf=neg_inf,
        )

    def _beam_search_multi_start_state(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
    ) -> _BeamState:
        bundle = self._build_eval_start_expansion(prepared=prepared)
        if bundle is None:
            return self._beam_search_state(prepared=prepared, beam_size=0, node_is_target=node_is_target)
        start_nodes, context_tokens, start_graph_ids, counts = bundle
        num_starts = int(start_nodes.numel())
        state = self._beam_search_state(
            prepared=prepared,
            beam_size=beam_size,
            node_is_target=node_is_target,
            start_nodes=start_nodes,
            context_tokens=context_tokens,
            num_graphs_override=num_starts,
        )
        return self._merge_start_beams(
            beam_state=state,
            start_graph_ids=start_graph_ids,
            counts=counts,
        )

    def _beam_search_state(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
        start_nodes: Optional[torch.Tensor] = None,
        context_tokens: Optional[torch.Tensor] = None,
        num_graphs_override: Optional[int] = None,
    ) -> _BeamState:
        num_graphs = int(num_graphs_override) if num_graphs_override is not None else int(prepared.num_graphs)
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
                beam_prev_rel=torch.zeros((0, 0, self.hidden_dim), device=prepared.node_ptr.device, dtype=prepared.context_tokens.dtype),
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
                beam_prev_rel=torch.zeros((num_graphs, 0, self.hidden_dim), device=prepared.node_ptr.device, dtype=prepared.context_tokens.dtype),
                num_graphs=num_graphs,
                beam_size=0,
                max_steps=int(self.max_steps),
                neg_inf=float("-inf"),
            )
        state = self._init_beam_state(
            prepared=prepared,
            beam_size=beam_size,
            node_is_target=node_is_target,
            start_nodes=start_nodes,
            context_tokens=context_tokens,
            num_graphs_override=num_graphs,
        )
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
                relation_tokens=prepared.relation_tokens,
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
        return self._beam_finalize(state, require_done=self._stop_enabled())

    def _init_beam_state(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
        start_nodes: Optional[torch.Tensor] = None,
        context_tokens: Optional[torch.Tensor] = None,
        num_graphs_override: Optional[int] = None,
    ) -> _BeamState:
        num_graphs = int(num_graphs_override) if num_graphs_override is not None else int(prepared.num_graphs)
        device = prepared.node_ptr.device
        max_steps = int(self.max_steps)
        neg_inf = float("-inf")
        if start_nodes is None:
            start_nodes = prepared.start_nodes_fwd
        start_nodes = start_nodes.to(device=device, dtype=torch.long).view(-1)
        if start_nodes.numel() != num_graphs:
            raise ValueError("start_nodes length mismatch with num_graphs.")
        beam_nodes = torch.full((num_graphs, beam_size), _NEG_ONE, device=device, dtype=torch.long)
        beam_scores = torch.full((num_graphs, beam_size), neg_inf, device=device, dtype=torch.float32)
        beam_paths = torch.full((num_graphs, beam_size, max_steps), _NEG_ONE, device=device, dtype=torch.long)
        beam_lengths = torch.zeros((num_graphs, beam_size), device=device, dtype=torch.long)
        valid_start = start_nodes >= 0
        beam_nodes[:, 0] = start_nodes
        beam_scores[:, 0] = torch.where(valid_start, torch.zeros_like(beam_scores[:, 0]), beam_scores[:, 0])
        beam_done = torch.zeros((num_graphs, beam_size), device=device, dtype=torch.bool)
        flat_graph_ids = torch.arange(num_graphs, device=device).repeat_interleave(beam_size)
        flat_beam_ids = torch.arange(beam_size, device=device).repeat(num_graphs)
        if context_tokens is None:
            context_tokens = prepared.context_tokens
        context_tokens = self._resolve_context_tokens(context_tokens)
        if context_tokens.size(0) != num_graphs:
            raise ValueError("context_tokens length mismatch with num_graphs.")
        beam_context = context_tokens.index_select(0, flat_graph_ids)
        prev_rel = self._init_prev_relation(num_graphs=num_graphs, device=device)
        beam_prev_rel = prev_rel.unsqueeze(1).expand(num_graphs, beam_size, -1)
        return _BeamState(
            beam_nodes=beam_nodes,
            beam_scores=beam_scores,
            beam_paths=beam_paths,
            beam_lengths=beam_lengths,
            beam_done=beam_done,
            flat_graph_ids=flat_graph_ids,
            flat_beam_ids=flat_beam_ids,
            beam_context=beam_context,
            beam_prev_rel=beam_prev_rel,
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
        prev_rel_flat = state.beam_prev_rel.reshape(-1, state.beam_prev_rel.size(-1))
        flat_valid = flat_nodes >= 0
        expand_mask = flat_valid & ~flat_done
        safe_nodes = flat_nodes.clamp(min=_ZERO)
        at_target = node_is_target.index_select(0, safe_nodes) & expand_mask
        outgoing = gather_outgoing_edges(
            curr_nodes=flat_nodes,
            edge_ids_by_head=prepared.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared.edge_ptr_by_head_fwd,
            active_mask=expand_mask,
        )
        empty_long = torch.zeros((0,), device=flat_nodes.device, dtype=torch.long)
        empty_bool = torch.zeros((0,), device=flat_nodes.device, dtype=torch.bool)
        empty_float = torch.zeros((0,), device=flat_nodes.device, dtype=torch.float32)
        step_ids = torch.full(
            (state.num_graphs * state.beam_size,),
            step,
            device=flat_nodes.device,
            dtype=torch.long,
        )

        if outgoing.edge_ids.numel() > 0:
            hier = self._compute_hierarchical_log_probs(
                prepared=prepared,
                edge_ids=outgoing.edge_ids,
                edge_batch=outgoing.edge_batch,
                parent_nodes=flat_nodes,
                steps=step_ids,
                temperature=1.0,
                context_tokens=state.beam_context,
                node_is_target=node_is_target,
                num_graphs=state.num_graphs * state.beam_size,
            )
            edge_log_prob = hier.edge_log_prob
            stop_log_prob = hier.stop_log_prob
            force_last = expand_mask & (step >= (state.max_steps - 1))
            force_mask = force_last | at_target
            if force_mask.any():
                force_edge = force_mask.index_select(0, outgoing.edge_batch)
                neg_inf = torch.finfo(edge_log_prob.dtype).min
                edge_log_prob = torch.where(force_edge, torch.full_like(edge_log_prob, neg_inf), edge_log_prob)
                stop_log_prob = torch.where(force_mask, torch.zeros_like(stop_log_prob), stop_log_prob)
            log_probs = edge_log_prob
            cand_scores_edge = flat_scores.index_select(0, outgoing.edge_batch) + log_probs
            cand_nodes_edge = prepared.edge_index[1].index_select(0, outgoing.edge_ids)
            cand_graph_edge = state.flat_graph_ids.index_select(0, outgoing.edge_batch)
            cand_src_beam_edge = state.flat_beam_ids.index_select(0, outgoing.edge_batch)
            cand_edge_id_edge = outgoing.edge_ids
            cand_is_edge_edge = torch.ones_like(cand_scores_edge, dtype=torch.bool)
            cand_done_edge = torch.zeros_like(cand_scores_edge, dtype=torch.bool)
        else:
            cand_scores_edge = empty_float
            cand_nodes_edge = empty_long
            cand_graph_edge = empty_long
            cand_src_beam_edge = empty_long
            cand_edge_id_edge = empty_long
            cand_is_edge_edge = empty_bool
            cand_done_edge = empty_bool

        cand_scores_stop = empty_float
        cand_nodes_stop = empty_long
        cand_graph_stop = empty_long
        cand_src_beam_stop = empty_long
        cand_edge_id_stop = empty_long
        cand_is_edge_stop = empty_bool
        cand_done_stop = empty_bool
        if outgoing.edge_ids.numel() == 0:
            log_prob_stop = torch.zeros_like(flat_scores)
        else:
            log_prob_stop = stop_log_prob
        stop_mask = expand_mask
        if stop_mask.any():
            cand_scores_stop = flat_scores[stop_mask] + log_prob_stop[stop_mask]
            cand_nodes_stop = flat_nodes[stop_mask]
            cand_graph_stop = state.flat_graph_ids[stop_mask]
            cand_src_beam_stop = state.flat_beam_ids[stop_mask]
            cand_edge_id_stop = torch.full_like(cand_nodes_stop, _NEG_ONE)
            cand_is_edge_stop = torch.zeros_like(cand_scores_stop, dtype=torch.bool)
            cand_done_stop = torch.ones_like(cand_scores_stop, dtype=torch.bool)

        allow_stay = flat_done
        stay_mask = flat_valid & allow_stay
        cand_scores_stay = flat_scores[stay_mask]
        cand_nodes_stay = flat_nodes[stay_mask]
        cand_graph_stay = state.flat_graph_ids[stay_mask]
        cand_src_beam_stay = state.flat_beam_ids[stay_mask]
        cand_edge_id_stay = torch.full_like(cand_nodes_stay, _NEG_ONE)
        cand_is_edge_stay = torch.zeros_like(cand_scores_stay, dtype=torch.bool)
        cand_done_stay = torch.ones_like(cand_scores_stay, dtype=torch.bool)

        if cand_scores_stop.numel() > 0:
            cand_scores_stay = torch.cat((cand_scores_stay, cand_scores_stop), dim=0)
            cand_nodes_stay = torch.cat((cand_nodes_stay, cand_nodes_stop), dim=0)
            cand_graph_stay = torch.cat((cand_graph_stay, cand_graph_stop), dim=0)
            cand_src_beam_stay = torch.cat((cand_src_beam_stay, cand_src_beam_stop), dim=0)
            cand_edge_id_stay = torch.cat((cand_edge_id_stay, cand_edge_id_stop), dim=0)
            cand_is_edge_stay = torch.cat((cand_is_edge_stay, cand_is_edge_stop), dim=0)
            cand_done_stay = torch.cat((cand_done_stay, cand_done_stop), dim=0)

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
        self._raise_non_finite("diverse/scores", scores, allow_neginf=True)
        valid_mask = range_count < counts.unsqueeze(1)
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
        relation_tokens: torch.Tensor,
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
                beam_prev_rel=torch.zeros_like(state.beam_prev_rel),
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
        sel_prev_rel = state.beam_prev_rel[batch_idx, sel_src_safe]
        sel_paths = sel_paths.clone()
        sel_paths[:, :, step] = torch.where(sel_is_edge, sel_edge_id, sel_paths[:, :, step])
        sel_lengths = sel_lengths + sel_is_edge.to(dtype=sel_lengths.dtype)
        sel_paths = torch.where(valid_sel.unsqueeze(-1), sel_paths, torch.full_like(sel_paths, _NEG_ONE))
        sel_lengths = torch.where(valid_sel, sel_lengths, torch.zeros_like(sel_lengths))
        sel_done = torch.where(valid_sel, sel_done, torch.zeros_like(sel_done))
        rel_tokens = relation_tokens.index_select(0, sel_edge_id.clamp(min=_ZERO).view(-1))
        rel_tokens = rel_tokens.view(sel_edge_id.size(0), sel_edge_id.size(1), -1)
        sel_prev_rel = self._update_prev_state(prev_state=sel_prev_rel, rel_emb=rel_tokens, update_mask=sel_is_edge)
        null_rel = torch.zeros_like(sel_prev_rel)
        sel_prev_rel = torch.where(valid_sel.unsqueeze(-1), sel_prev_rel, null_rel)
        return _BeamState(
            beam_nodes=sel_nodes,
            beam_scores=sel_scores,
            beam_paths=sel_paths,
            beam_lengths=sel_lengths,
            beam_done=sel_done,
            flat_graph_ids=state.flat_graph_ids,
            flat_beam_ids=state.flat_beam_ids,
            beam_context=state.beam_context,
            beam_prev_rel=sel_prev_rel,
            num_graphs=state.num_graphs,
            beam_size=state.beam_size,
            max_steps=state.max_steps,
            neg_inf=state.neg_inf,
        )

    @staticmethod
    def _beam_finalize(state: _BeamState, *, require_done: bool) -> list[list[tuple[int, float, list[int]]]]:
        beam_nodes_np = state.beam_nodes.detach().cpu().numpy()
        beam_scores_np = state.beam_scores.detach().cpu().numpy()
        beam_paths_np = state.beam_paths.detach().cpu().numpy()
        beam_lengths_np = state.beam_lengths.detach().cpu().numpy()
        beam_done_np = state.beam_done.detach().cpu().numpy()
        beams: list[list[tuple[int, float, list[int]]]] = []
        for graph_idx in range(state.num_graphs):
            graph_beams: list[tuple[int, float, list[int]]] = []
            for beam_idx in range(state.beam_size):
                node_id = int(beam_nodes_np[graph_idx, beam_idx])
                if require_done and not bool(beam_done_np[graph_idx, beam_idx]):
                    continue
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

    @staticmethod
    def _compute_unique_counts_for_mask(
        *,
        nodes: torch.Tensor,
        mask: torch.Tensor,
        node_is_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes_masked = torch.where(mask, nodes, torch.full_like(nodes, _NEG_ONE))
        sorted_nodes, _ = torch.sort(nodes_masked, dim=1)
        sorted_valid = sorted_nodes >= _ZERO
        unique_mask = torch.ones_like(sorted_nodes, dtype=torch.bool)
        if sorted_nodes.size(1) > _ONE:
            unique_mask[:, _ONE:] = sorted_nodes[:, _ONE:] != sorted_nodes[:, :-_ONE]
        unique_mask = unique_mask & sorted_valid
        unique_nodes_safe = sorted_nodes.clamp(min=_ZERO)
        unique_hits = node_is_target.index_select(0, unique_nodes_safe.view(-1)).view(nodes.size(0), -1)
        unique_hit_counts = (unique_mask & unique_hits).sum(dim=1).to(dtype=torch.float32)
        unique_pred_counts = unique_mask.sum(dim=1).to(dtype=torch.float32)
        return unique_hit_counts, unique_pred_counts

    def _compute_topk_metrics(
        self,
        *,
        beam_nodes: torch.Tensor,
        beam_hits: torch.Tensor,
        beam_valid: torch.Tensor,
        beam_scores: torch.Tensor,
        node_is_target: torch.Tensor,
        answer_counts: torch.Tensor,
        topk: list[int],
        neg_inf: float,
    ) -> dict[str, torch.Tensor]:
        if not topk:
            return {}
        beam_size = int(beam_nodes.size(1))
        if beam_size <= _ZERO:
            return {}
        beam_rank = beam_valid.cumsum(dim=1)
        beam_scores = torch.where(beam_valid, beam_scores, torch.full_like(beam_scores, neg_inf))
        large_rank = beam_size + _ONE
        hit_rank = torch.where(beam_hits & beam_valid, beam_rank, torch.full_like(beam_rank, large_rank))
        first_hit_rank = hit_rank.min(dim=1).values
        metrics: dict[str, torch.Tensor] = {}
        for k in topk:
            k_eff = min(int(k), beam_size)
            if k_eff <= _ZERO:
                continue
            topk_mask = beam_valid & (beam_rank <= k_eff)
            topk_hits = beam_hits & topk_mask
            topk_counts = topk_mask.sum(dim=1).to(dtype=torch.float32)
            unique_hit_counts, unique_pred_counts = self._compute_unique_counts_for_mask(
                nodes=beam_nodes,
                mask=topk_mask,
                node_is_target=node_is_target,
            )
            precision = unique_hit_counts / unique_pred_counts.clamp(min=_ONE)
            recall = unique_hit_counts / answer_counts.clamp(min=_ONE)
            denom = precision + recall
            f1 = torch.where(
                denom > float(_ZERO),
                (float(_TWO) * precision * recall / denom),
                torch.zeros_like(denom),
            )
            hit_any = topk_hits.any(dim=1).to(dtype=torch.float32)
            diversity = unique_pred_counts / topk_counts.clamp(min=_ONE)
            rank_f = first_hit_rank.to(dtype=torch.float32).clamp(min=_ONE)
            mrr = torch.where(first_hit_rank <= k_eff, float(_ONE) / rank_f, torch.zeros_like(rank_f))
            noise = (float(_ONE) - precision).clamp(min=float(_ZERO))
            hit_scores = torch.where(topk_hits, beam_scores, torch.full_like(beam_scores, neg_inf))
            miss_scores = torch.where(topk_mask & (~beam_hits), beam_scores, torch.full_like(beam_scores, neg_inf))
            best_hit = hit_scores.max(dim=1).values
            best_miss = miss_scores.max(dim=1).values
            has_hit = topk_hits.any(dim=1)
            has_miss = (topk_mask & (~beam_hits)).any(dim=1)
            safe_hit = torch.where(has_hit, best_hit, torch.zeros_like(best_hit))
            safe_miss = torch.where(has_miss, best_miss, torch.zeros_like(best_miss))
            gap_raw = safe_hit - safe_miss
            gap_valid = (has_hit & has_miss).to(dtype=torch.float32)
            gap = torch.where(gap_valid > float(_ZERO), gap_raw, torch.zeros_like(gap_raw))
            suffix = f"@{k}"
            metrics[f"hit{suffix}"] = hit_any
            metrics[f"recall{suffix}"] = recall
            metrics[f"precision{suffix}"] = precision
            metrics[f"f1{suffix}"] = f1
            metrics[f"mrr{suffix}"] = mrr
            metrics[f"noise{suffix}"] = noise
            metrics[f"diversity{suffix}"] = diversity
            metrics[f"score_gap{suffix}"] = gap
            metrics[f"score_gap_valid{suffix}"] = gap_valid
            metrics[f"unique_answers{suffix}"] = unique_hit_counts
        return metrics



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
        beam_state = self._beam_search_multi_start_state(
            prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target_all
        )
        beam_nodes_all = beam_state.beam_nodes
        beam_lengths_all = beam_state.beam_lengths
        beam_paths_all = beam_state.beam_paths
        beam_valid_all = beam_nodes_all >= _ZERO
        beam_reach = self._compute_beam_reach_mask(
            prepared_fwd=prepared_fwd,
            beam_paths=beam_paths_all,
            beam_lengths=beam_lengths_all,
            beam_nodes=beam_nodes_all,
            node_is_target=node_is_target_all,
        )
        beam_reach = beam_reach & beam_valid_all
        beam_nodes = beam_nodes_all
        beam_lengths = beam_lengths_all
        if beam_nodes.numel() == _ZERO:
            return {}, _ZERO
        beam_done = beam_state.beam_done
        beam_nodes = torch.where(beam_done, beam_nodes, torch.full_like(beam_nodes, _NEG_ONE))
        beam_lengths = torch.where(beam_done, beam_lengths, torch.zeros_like(beam_lengths))
        beam_valid = beam_nodes >= _ZERO
        beam_nodes_safe = beam_nodes.clamp(min=_ZERO)
        beam_hits = node_is_target_all.index_select(0, beam_nodes_safe.view(-1)).view(num_graphs, -1)
        beam_hits = beam_hits & beam_valid
        beam_pass = beam_reach & ~beam_hits
        beam_keep = self._apply_answer_dedup(
            beam_nodes=beam_nodes,
            beam_scores=beam_state.beam_scores,
            beam_hits=beam_hits,
            beam_valid=beam_valid,
        )
        beam_valid = beam_valid & beam_keep
        beam_hits = beam_hits & beam_valid
        beam_nodes = torch.where(beam_valid, beam_nodes, torch.full_like(beam_nodes, _NEG_ONE))
        beam_lengths = torch.where(beam_valid, beam_lengths, torch.zeros_like(beam_lengths))
        answer_gain_cfg = self._resolve_answer_gain_cfg()
        if answer_gain_cfg["enabled"]:
            cutoffs = self._apply_answer_gain_stop(
                beam_nodes=beam_nodes,
                beam_hits=beam_hits,
                patience=int(answer_gain_cfg["patience"]),
                epsilon=float(answer_gain_cfg["epsilon"]),
                min_beam=int(answer_gain_cfg["min_beam"]),
            )
            idx = torch.arange(beam_nodes.size(1), device=beam_nodes.device).unsqueeze(0)
            beam_keep = idx < cutoffs.unsqueeze(1)
            beam_valid = beam_valid & beam_keep
            beam_hits = beam_hits & beam_keep
            beam_nodes = torch.where(beam_valid, beam_nodes, torch.full_like(beam_nodes, _NEG_ONE))
            beam_lengths = torch.where(beam_keep, beam_lengths, torch.zeros_like(beam_lengths))
        hit_terminal = beam_hits.any(dim=1).to(dtype=torch.float32)
        hit_reach = beam_reach.any(dim=1).to(dtype=torch.float32)
        hit_pass = beam_pass.any(dim=1).to(dtype=torch.float32)
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
        beam_size_eval = beam_lengths.size(1)
        if beam_size_eval <= 0:
            length = torch.zeros((beam_lengths.size(0),), device=beam_lengths.device, dtype=torch.float32)
        else:
            idx = torch.arange(beam_size_eval, device=beam_lengths.device).unsqueeze(0)
            idx = idx.expand_as(beam_valid)
            idx_masked = torch.where(beam_valid, idx, torch.full_like(idx, beam_size_eval))
            first_idx = idx_masked.min(dim=1).values
            has_valid = first_idx < beam_size_eval
            first_idx = first_idx.clamp(max=beam_size_eval - 1)
            length = beam_lengths.gather(1, first_idx.unsqueeze(1)).squeeze(1).to(dtype=torch.float32)
            length = torch.where(has_valid, length, torch.zeros_like(length))
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
            "hit@beam": hit_reach,
            "hit@beam_terminal": hit_terminal,
            "hit@beam_pass": hit_pass,
            "recall@beam": recall_scores,
            "precision@beam": precision_scores,
            "f1@beam": f1_scores,
            "diversity@beam": diversity_scores,
        }
        beam_metrics_cfg = self._resolve_beam_metrics_cfg()
        topk_metrics = self._compute_topk_metrics(
            beam_nodes=beam_nodes,
            beam_hits=beam_hits,
            beam_valid=beam_valid,
            beam_scores=beam_state.beam_scores,
            node_is_target=node_is_target_all,
            answer_counts=answer_counts,
            topk=list(beam_metrics_cfg.get("topk", [])),
            neg_inf=float(beam_state.neg_inf),
        )
        metrics.update(topk_metrics)
        if answer_gain_cfg["enabled"]:
            metrics["beam_size_adaptive"] = beam_valid_counts
        # Coverage-style diagnostics for answer presence in subgraph.
        metrics["coverage_rate"] = (answer_counts > _ZERO).to(dtype=torch.float32)
        metrics["length_mean"] = length
        if modes_per_graph is not None:
            metrics["modes@beam"] = modes_per_graph
        eval_temperature = float(_ONE)
        rollout_fwd = self._rollout_policy(
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
            collect_policy_metrics=True,
        )
        fwd_actions = rollout_fwd.actions
        if fwd_actions is None:
            fwd_actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=self.device, dtype=torch.long)
        log_pf_sum = self._recompute_log_pf_sum(
            prepared_fwd=prepared_fwd,
            actions=fwd_actions,
            num_moves=rollout_fwd.num_moves,
            stop_reason=rollout_fwd.stop_reason,
            stop_nodes=rollout_fwd.stop_nodes,
            node_is_target=node_is_target_all,
            sampling_temperature=eval_temperature,
            graph_mask=graph_mask,
        )
        tb_loss, tb_metrics = self._compute_tb_loss(
            prepared_fwd=prepared_fwd,
            actions=fwd_actions,
            graph_mask=graph_mask,
            stop_reason=rollout_fwd.stop_reason,
            stop_nodes=rollout_fwd.stop_nodes,
            log_pf_sum=log_pf_sum,
            sampling_temperature=eval_temperature,
        )
        lengths = rollout_fwd.num_moves.to(dtype=torch.float32)
        denom = graph_mask.to(dtype=lengths.dtype).sum().clamp(min=_ONE)
        rollout_num_moves_mean = (lengths * graph_mask.to(dtype=lengths.dtype)).sum() / denom
        min_steps = self._resolve_stop_min_steps()
        emit_mask = (rollout_fwd.stop_reason == _TERMINAL_EMIT) & graph_mask
        emit_count = emit_mask.to(dtype=torch.float32).sum()
        emit_at_min_steps_count = (emit_mask & (rollout_fwd.num_moves == min_steps)).to(dtype=torch.float32).sum()
        emit_at_min_steps_rate = torch.where(
            emit_count > float(_ZERO),
            emit_at_min_steps_count / emit_count.clamp(min=float(_ONE)),
            torch.zeros_like(emit_count),
        )
        metrics.update(tb_metrics)
        metrics["rollout/num_moves_mean"] = rollout_num_moves_mean
        metrics["rollout/terminal/emit_at_stop_min_steps_count"] = emit_at_min_steps_count
        metrics["rollout/terminal/emit_at_stop_min_steps_given_emit_rate"] = emit_at_min_steps_rate
        metrics.update(
            self._compute_rollout_reach_metrics(
                prepared_fwd=prepared_fwd,
                actions=fwd_actions,
                num_moves=rollout_fwd.num_moves,
                stop_reason=rollout_fwd.stop_reason,
                graph_mask=graph_mask,
                node_is_target=node_is_target_all,
            )
        )
        if rollout_fwd.policy_metrics:
            metrics.update(rollout_fwd.policy_metrics)
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

    def _build_beam_metric_names(self) -> set[str]:
        cfg = self._resolve_beam_metrics_cfg()
        topk = cfg.get("topk", [])
        names: set[str] = set()
        for k in topk:
            suffix = f"@{k}"
            names.update(
                {
                    f"hit{suffix}",
                    f"recall{suffix}",
                    f"precision{suffix}",
                    f"f1{suffix}",
                    f"mrr{suffix}",
                    f"noise{suffix}",
                    f"diversity{suffix}",
                    f"score_gap{suffix}",
                    f"score_gap_valid{suffix}",
                    f"unique_answers{suffix}",
                }
            )
        return names

    def _get_standard_metrics(self, stage: str) -> set[str]:
        key = str(stage).strip().lower()
        if key not in _STANDARD_METRICS:
            raise ValueError(f"Unsupported metrics stage: {stage!r}.")
        metrics = set(_STANDARD_METRICS[key])
        if key in {"val", "test"}:
            metrics |= self._build_beam_metric_names()
        return metrics

    @staticmethod
    def _resolve_optimizer_class(name: str):
        name = name.lower()
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "adamax": torch.optim.Adamax,
            "sgd": torch.optim.SGD,
            "adagrad": torch.optim.Adagrad,
            "adadelta": torch.optim.Adadelta,
            "rmsprop": torch.optim.RMSprop,
            "nadam": getattr(torch.optim, "NAdam", torch.optim.Adam),
            "lbfgs": torch.optim.LBFGS,
        }
        if name in optimizers:
            return optimizers[name]
        if name in {"muon", "singledevicemuon", "single_device_muon", "distributedmuon", "muon_distributed"}:
            force_single = name in {"singledevicemuon", "single_device_muon"}
            force_distributed = name in {"distributedmuon", "muon_distributed"}

            def _factory(params, **cfg):
                try:
                    from muon import Muon, SingleDeviceMuon  # type: ignore
                except ImportError as exc:  # pragma: no cover
                    raise ImportError(
                        "Muon optimizer requested but the 'muon-optimizer' package is not installed. "
                        "Install it via `pip install git+https://github.com/KellerJordan/Muon`."
                    ) from exc

                if force_single and force_distributed:
                    raise ValueError("Cannot force both distributed and single-device Muon modes.")
                if force_distributed and not torch.distributed.is_available():
                    raise RuntimeError("Distributed Muon requested but torch.distributed is unavailable.")
                if force_distributed:
                    return Muon(params, **cfg)
                return SingleDeviceMuon(params, **cfg)

            return _factory
        for attr in dir(torch.optim):
            cls = getattr(torch.optim, attr)
            if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer) and attr.lower() == name:
                return cls
        raise ValueError(f"Unsupported optimizer type '{name}'.")

    def _build_optimizer_param_groups(self, *, weight_decay: float) -> list[dict[str, Any]]:
        no_decay: set[str] = set()
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.modules.batchnorm._BatchNorm),
            ):
                for param_name, _ in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    no_decay.add(full_name)
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or name in no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        if not decay_params and not no_decay_params:
            raise ValueError("No trainable parameters found for optimizer.")
        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": float(_ZERO)})
        return param_groups



    def configure_optimizers(self):
        cfg = require_cfg_mapping(self.optimizer_cfg, "optimizer_cfg") if self.optimizer_cfg is not None else {}
        cfg = dict(cfg)
        if any(key in cfg for key in ("auto_param_groups", "param_group_overrides", "param_groups")):
            raise ValueError("optimizer_cfg parameter groups are fixed in code; remove param group keys from config.")
        opt_type = str(cfg.pop("type", cfg.pop("name", "adamw"))).lower()
        weight_decay = cfg.pop("weight_decay", None)
        if weight_decay is None:
            raise ValueError("optimizer_cfg.weight_decay must be set.")
        params = self._build_optimizer_param_groups(weight_decay=float(weight_decay))
        optimizer_cls = self._resolve_optimizer_class(opt_type)
        optimizer = optimizer_cls(params, **cfg)
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

    @staticmethod
    def _coerce_log_batch_size(value: Any, *, fallback: int) -> int:
        if value is None:
            return int(fallback)
        if torch.is_tensor(value):
            if value.numel() != _ONE:
                raise ValueError(f"Batch size metric must be scalar; got {tuple(value.shape)}.")
            value = float(value.detach().cpu().item())
        else:
            value = float(value)
        if not math.isfinite(value):
            return _ZERO
        size = int(round(value))
        if size <= _ZERO:
            return _ZERO
        return size

    def _log_metrics(
        self,
        *,
        prefix: str,
        metrics: dict[str, torch.Tensor],
        fallback_batch_size: int,
        prog_bar: bool = False,
    ) -> None:
        fallback_batch_size = int(fallback_batch_size)
        if fallback_batch_size <= _ZERO:
            raise ValueError("fallback_batch_size must be > 0.")
        for name, value in metrics.items():
            reduce_fx = "sum" if self._metric_is_count(name) else "mean"
            batch_size = fallback_batch_size
            if reduce_fx != "sum":
                denom_key = self._resolve_metric_denom_key(name)
                if denom_key is not None:
                    batch_size = self._coerce_log_batch_size(metrics.get(denom_key), fallback=fallback_batch_size)
                if batch_size <= _ZERO:
                    continue
            log_metric(
                self,
                f"{prefix}/{name}",
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
                reduce_fx=reduce_fx,
            )

    def training_step(self, batch: Any, batch_idx: int):
        self._ensure_runtime_initialized()
        self._maybe_update_logit_scale_schedule()
        optimizer = self.optimizers()
        accum = float(self._accumulate_grad_batches())
        if self._should_zero_grad(batch_idx):
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._compute_training_loss(batch)
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
        self._log_metrics(prefix="train", metrics=metrics, fallback_batch_size=batch_size, prog_bar=False)
        loss_batch_size = self._coerce_log_batch_size(metrics.get("tb/valid_graph_count"), fallback=batch_size)
        if loss_batch_size <= _ZERO:
            loss_batch_size = batch_size
        log_metric(
            self,
            "train/loss",
            loss.detach(),
            batch_size=loss_batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            reduce_fx="mean",
        )
        return loss.detach()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._ensure_runtime_initialized()
        _ = batch_idx
        metrics, batch_size = self._compute_eval_metrics(batch)
        if batch_size <= _ZERO:
            return
        metrics = self._filter_metrics(metrics, self._get_standard_metrics("val"))
        scope = self._resolve_dataset_scope()
        self._log_metrics(prefix=f"val/{scope}", metrics=metrics, fallback_batch_size=batch_size, prog_bar=False)
        if scope == "full":
            topk = {
                name: value
                for name, value in metrics.items()
                if name.startswith(("hit@", "recall@", "precision@", "f1@"))
            }
            self._log_metrics(prefix="val", metrics=topk, fallback_batch_size=batch_size, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._ensure_runtime_initialized()
        _ = batch_idx
        metrics, batch_size = self._compute_eval_metrics(batch)
        if batch_size <= _ZERO:
            return
        metrics = self._filter_metrics(metrics, self._get_standard_metrics("test"))
        scope = self._resolve_dataset_scope()
        self._log_metrics(prefix=f"test/{scope}", metrics=metrics, fallback_batch_size=batch_size, prog_bar=False)
        if scope == "full":
            topk = {
                name: value
                for name, value in metrics.items()
                if name.startswith(("hit@", "recall@", "precision@", "f1@"))
            }
            self._log_metrics(prefix="test", metrics=topk, fallback_batch_size=batch_size, prog_bar=False)

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._ensure_runtime_initialized()
        _ = batch_idx, dataloader_idx
        prepared_fwd = self._prepare_batch(batch)
        num_graphs = int(prepared_fwd.num_graphs)
        if num_graphs <= _ZERO:
            return []
        num_nodes_total = int(prepared_fwd.num_nodes_total)
        node_is_target = build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        beam_size = self._resolve_beam_size()
        sample_ids = extract_sample_ids(batch)
        if len(sample_ids) != num_graphs:
            raise ValueError("sample_id length mismatch with batch graph count.")

        beam_state = self._beam_search_multi_start_state(
            prepared=prepared_fwd, beam_size=beam_size, node_is_target=node_is_target
        )
        if beam_state.beam_nodes.numel() == _ZERO:
            return []
        beams = self._beam_finalize(beam_state, require_done=True)
        rollouts_per_graph: list[list[dict[str, Any]]] = [[] for _ in range(num_graphs)]
        node_global_cpu = prepared_fwd.node_global_ids.detach().cpu()
        node_is_target_cpu = node_is_target.detach().cpu()
        edge_index_cpu = prepared_fwd.edge_index.detach().cpu()
        edge_rel_cpu = prepared_fwd.edge_relations.detach().cpu()
        for graph_idx in range(num_graphs):
            beam = beams[graph_idx]
            for beam_idx, (stop_node, score, path) in enumerate(beam):
                edges_list: list[dict[str, Any]] = []
                if path:
                    edge_ids = torch.as_tensor(path, dtype=torch.long, device=edge_index_cpu.device)
                    edge_sel = edge_index_cpu.index_select(1, edge_ids)
                    rel_sel = edge_rel_cpu.index_select(0, edge_ids)
                    head_local = edge_sel[_ZERO]
                    tail_local = edge_sel[_ONE]
                    head_ent = node_global_cpu.index_select(0, head_local).tolist()
                    tail_ent = node_global_cpu.index_select(0, tail_local).tolist()
                    rel_list = rel_sel.tolist()
                    edges_list = [
                        {
                            "src_entity_id": int(h_ent),
                            "dst_entity_id": int(t_ent),
                            "head_entity_id": int(h_ent),
                            "tail_entity_id": int(t_ent),
                            "relation_id": int(rel),
                        }
                        for h_ent, t_ent, rel in zip(head_ent, tail_ent, rel_list)
                    ]
                stop_entity = int(node_global_cpu[stop_node].item()) if stop_node >= _ZERO else None
                success = bool(node_is_target_cpu[stop_node].item()) if stop_node >= _ZERO else False
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
        a_local_ptr_cpu = prepared_fwd.a_ptr.detach().cpu()
        answer_ptr_cpu = prepared_fwd.answer_ptr.detach().cpu()
        q_local_cpu = prepared_fwd.q_local_indices.detach().cpu()
        answer_ids_cpu = prepared_fwd.answer_entity_ids.detach().cpu()
        records: list[dict[str, Any]] = []
        for graph_idx in range(num_graphs):
            node_start = int(node_ptr_cpu[graph_idx].item())
            node_end = int(node_ptr_cpu[graph_idx + _ONE].item())
            q_start = int(q_ptr_cpu[graph_idx].item())
            q_end = int(q_ptr_cpu[graph_idx + _ONE].item())
            start_indices = q_local_cpu[q_start:q_end].to(dtype=torch.long)
            start_entity_ids: list[int]
            if start_indices.numel() == _ZERO:
                start_entity_ids = []
            else:
                if bool((start_indices < _ZERO).any().item()):
                    raise ValueError(f"q_local_indices contain negative values for sample_id={sample_ids[graph_idx]!r}.")
                if bool((start_indices >= num_nodes_total).any().item()):
                    raise ValueError(f"q_local_indices out of range for sample_id={sample_ids[graph_idx]!r}.")
                in_graph = (start_indices >= node_start) & (start_indices < node_end)
                if not bool(in_graph.all().item()):
                    raise ValueError(f"q_local_indices mismatch node_ptr for sample_id={sample_ids[graph_idx]!r}.")
                start_entity_ids = node_global_cpu.index_select(0, start_indices).tolist()
            a_start = int(answer_ptr_cpu[graph_idx].item())
            a_end = int(answer_ptr_cpu[graph_idx + _ONE].item())
            answer_ids = answer_ids_cpu[a_start:a_end].tolist() if a_end > a_start else []
            a_local_start = int(a_local_ptr_cpu[graph_idx].item())
            a_local_end = int(a_local_ptr_cpu[graph_idx + _ONE].item())
            a_entity_in_graph = a_local_end > a_local_start
            record = {
                "sample_id": sample_ids[graph_idx],
                "start_entity_ids": start_entity_ids,
                "answer_entity_ids": answer_ids,
                "a_entity_in_graph": bool(a_entity_in_graph),
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
