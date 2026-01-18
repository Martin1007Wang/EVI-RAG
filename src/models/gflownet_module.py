from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torch import nn

from src.models.components import (
    EmbeddingBackbone,
    FlowPredictor,
    GFlowNetActor,
    GraphEnv,
    CvtNodeInitializer,
    SinkSelector,
    StartSelector,
    TrajectoryAgent,
)
from src.data.components.embeddings import attach_embeddings_to_batch
from src.data.schema.constants import _INVERSE_RELATION_SUFFIX_DEFAULT
from src.gfn.engine import FlowFeatureSpec, GFlowNetEngine, GFlowNetRolloutConfig, resolve_flow_spec
from src.gfn.training import GFlowNetTrainingLoop
from src.gfn.ops import GFlowNetBatchProcessor, GFlowNetInputValidator
from src.metrics import gflownet as gfn_metrics
from src.utils.logging_utils import get_logger, log_event

logger = get_logger(__name__)

_ZERO = 0
_ONE = 1
_HALF = 0.5
_TWO = 2
_THREE = 3
_NAN = float("nan")
_DEFAULT_POLICY_TEMPERATURE = 1.0
_DEFAULT_VALIDATE_EDGE_BATCH = True
_DEFAULT_VALIDATE_ROLLOUT_BATCH = True
_DEFAULT_VALIDATE_ON_DEVICE = False
_DEFAULT_REQUIRE_PRECOMPUTED_EDGE_BATCH = True
_DEFAULT_FORCE_FP32 = False
_INVALID_RELATION_ID = -1
_DEFAULT_ROLLOUT_CHUNK_SIZE = 1
_DEFAULT_LOG_ON_STEP_TRAIN = False
_DEFAULT_EVAL_TEMPERATURE_EXTRAS: tuple[float, ...] = ()
_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_AGENT_DROPOUT = 0.0
_DEFAULT_CVT_INIT_ENABLED = True
_DEFAULT_START_SELECTION_ENABLED = False
_DEFAULT_START_SELECTION_DROPOUT = 0.0
_DEFAULT_SINK_SELECTION_ENABLED = False
_DEFAULT_SINK_SELECTION_DROPOUT = 0.0
_DEFAULT_BACKWARD_SHARE_STATE_ENCODER = True
_DEFAULT_BACKWARD_SHARE_EDGE_SCORER = True
_BACKWARD_SHARE_STATE_KEY = "share_state_encoder"
_BACKWARD_SHARE_EDGE_KEY = "share_edge_scorer"
_DEFAULT_MANUAL_GRAD_CLIP_ALGO = "norm"
_DEFAULT_MANUAL_GRAD_CLIP_NORM_TYPE = 2.0
_DEFAULT_GRAD_CLIP_ADAPTIVE = False
_DEFAULT_GRAD_CLIP_LOG_EPS = 1.0e-8
_DEFAULT_GRAD_CLIP_TAIL_PROB_EPS = 1.0e-6
_DEFAULT_GRAD_CLIP_LOG_INTERVAL = 1
_DEFAULT_GRAD_NONFINITE_MAX_PARAMS = 20
_POTENTIAL_SCHEDULE_COSINE = "cosine"
_POTENTIAL_SCHEDULE_LINEAR = "linear"
_POTENTIAL_SCHEDULE_NONE = "none"
_POTENTIAL_SCHEDULES = {
    _POTENTIAL_SCHEDULE_COSINE,
    _POTENTIAL_SCHEDULE_LINEAR,
    _POTENTIAL_SCHEDULE_NONE,
}
_MIN_PURE_PHASE_RATIO = 0.0
_MAX_PURE_PHASE_RATIO = 1.0
_BATCH_DEVICE_KEYS = (
    "edge_index",
    "edge_attr",
    "ptr",
    "q_local_indices",
    "a_local_indices",
    "question_emb",
    "node_embeddings",
    "edge_embeddings",
    "node_min_dists",
    "edge_batch",
    "edge_ptr",
    "is_dummy_agent",
)
_BATCH_FLOAT_KEYS = {
    "question_emb",
    "node_embeddings",
    "edge_embeddings",
}


def _load_vocab_size_from_parquet(path: Path, *, id_column: str, label: str) -> int:
    vocab_path = Path(path).expanduser().resolve()
    if not vocab_path.exists():
        raise FileNotFoundError(f"{label} not found: {vocab_path}")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(f"pyarrow is required to load {label}.") from exc
    table = pq.read_table(vocab_path, columns=[id_column])
    ids = torch.as_tensor(table.column(id_column).to_numpy(), dtype=torch.long)
    if ids.numel() == _ZERO:
        raise ValueError(f"{label} is empty.")
    if int(ids.min().detach().tolist()) < _ZERO:
        raise ValueError(f"{label} contains negative {id_column} values.")
    max_id = int(ids.max().detach().tolist())
    return max_id + _ONE


def _load_relation_vocab_size(path: Path) -> int:
    return _load_vocab_size_from_parquet(
        path,
        id_column="relation_id",
        label="relation_vocab.parquet",
    )


def _load_entity_vocab_size(path: Path) -> int:
    return _load_vocab_size_from_parquet(
        path,
        id_column="entity_id",
        label="entity_vocab.parquet",
    )


class GFlowNetModule(LightningModule):
    """扁平 PyG batch 版本的 GFlowNet，移除 dense padding。"""

    def __init__(
        self,
        *,
        hidden_dim: int,
        reward_fn: nn.Module,
        env: GraphEnv,
        emb_dim: int,
        entity_vocab_path: Optional[str | Path] = None,
        relation_vocab_path: Optional[str | Path] = None,
        relation_vocab_size: Optional[int] = None,
        inverse_relation_suffix: Optional[str] = None,
        backbone_finetune: bool = _DEFAULT_BACKBONE_FINETUNE,
        edge_score_cfg: Optional[Mapping[str, Any]] = None,
        entity_score_cfg: Optional[Mapping[str, Any]] = None,
        state_cfg: Optional[Mapping[str, Any]] = None,
        cvt_init_cfg: Optional[Mapping[str, Any]] = None,
        flow_cfg: Optional[Mapping[str, Any]] = None,
        start_selection_cfg: Optional[Mapping[str, Any]] = None,
        sink_selection_cfg: Optional[Mapping[str, Any]] = None,
        backward_cfg: Optional[Mapping[str, Any]] = None,
        training_cfg: Mapping[str, Any],
        evaluation_cfg: Mapping[str, Any],
        actor_cfg: Optional[Mapping[str, Any]] = None,
        runtime_cfg: Optional[Mapping[str, Any]] = None,
        optimizer_cfg: Optional[Mapping[str, Any]] = None,
        scheduler_cfg: Optional[Mapping[str, Any]] = None,
        logging_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.hidden_dim = int(hidden_dim)

        self._init_config_maps(
            training_cfg=training_cfg,
            evaluation_cfg=evaluation_cfg,
            runtime_cfg=runtime_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            logging_cfg=logging_cfg,
            actor_cfg=actor_cfg,
            edge_score_cfg=edge_score_cfg,
            entity_score_cfg=entity_score_cfg,
            state_cfg=state_cfg,
            cvt_init_cfg=cvt_init_cfg,
            flow_cfg=flow_cfg,
            start_selection_cfg=start_selection_cfg,
            sink_selection_cfg=sink_selection_cfg,
            backward_cfg=backward_cfg,
        )
        self._validate_runtime_cfg()
        self._init_vocab_settings(
            entity_vocab_path=entity_vocab_path,
            relation_vocab_path=relation_vocab_path,
            relation_vocab_size=relation_vocab_size,
            inverse_relation_suffix=inverse_relation_suffix,
        )
        self._validate_structured_cfg()
        self._init_eval_settings()
        self._init_logging_settings()
        self._init_grad_clip_settings()
        self._init_backward_policy_settings()

        self.reward_fn = reward_fn
        self.env = env
        self.max_steps = int(self.env.max_steps)
        self._init_backbone(emb_dim=emb_dim, finetune=backbone_finetune)
        self.cvt_init = None
        self._state_input_dim = self._init_agent_components()
        self._init_backward_agent_components()
        self._init_reward_schedule()
        self.flow_spec: FlowFeatureSpec = resolve_flow_spec(self.flow_cfg)
        self._init_start_selection()
        self._init_sink_selection()
        self.log_f = FlowPredictor(self.hidden_dim, self._state_input_dim, self.flow_spec.stats_dim)
        self.log_f_backward = FlowPredictor(
            self.hidden_dim,
            getattr(self, "_state_input_dim_backward", self._state_input_dim),
            self.flow_spec.stats_dim,
        )
        self.actor = None
        self.actor_backward = None
        self.input_validator = None
        self.batch_processor = None
        self.engine = None
        self.training_loop = GFlowNetTrainingLoop(self)
        self.training_loop.init_metric_stores()
        self._assert_training_cfg_contract()
        self._save_serializable_hparams()

    def _init_config_maps(
        self,
        *,
        training_cfg: Mapping[str, Any],
        evaluation_cfg: Mapping[str, Any],
        runtime_cfg: Optional[Mapping[str, Any]],
        optimizer_cfg: Optional[Mapping[str, Any]],
        scheduler_cfg: Optional[Mapping[str, Any]],
        logging_cfg: Optional[Mapping[str, Any]],
        actor_cfg: Optional[Mapping[str, Any]],
        edge_score_cfg: Optional[Mapping[str, Any]],
        entity_score_cfg: Optional[Mapping[str, Any]],
        state_cfg: Optional[Mapping[str, Any]],
        cvt_init_cfg: Optional[Mapping[str, Any]],
        flow_cfg: Optional[Mapping[str, Any]],
        start_selection_cfg: Optional[Mapping[str, Any]],
        sink_selection_cfg: Optional[Mapping[str, Any]],
        backward_cfg: Optional[Mapping[str, Any]],
    ) -> None:
        self.training_cfg = self._require_mapping(training_cfg, "training_cfg")
        self.evaluation_cfg = self._require_mapping(evaluation_cfg, "evaluation_cfg")
        self.runtime_cfg = self._optional_mapping(runtime_cfg, "runtime_cfg")
        self.optimizer_cfg = self._optional_mapping(optimizer_cfg, "optimizer_cfg")
        self.scheduler_cfg = self._optional_mapping(scheduler_cfg, "scheduler_cfg")
        self.logging_cfg = self._optional_mapping(logging_cfg, "logging_cfg")
        self.actor_cfg = self._optional_mapping(actor_cfg, "actor_cfg")
        self.edge_score_cfg = self._optional_mapping(edge_score_cfg, "edge_score_cfg")
        self.entity_score_cfg = self._optional_mapping(entity_score_cfg, "entity_score_cfg")
        self.state_cfg = self._optional_mapping(state_cfg, "state_cfg")
        self.cvt_init_cfg = self._optional_mapping(cvt_init_cfg, "cvt_init_cfg")
        self.flow_cfg = self._optional_mapping(flow_cfg, "flow_cfg")
        self.start_selection_cfg = self._optional_mapping(start_selection_cfg, "start_selection_cfg")
        self.sink_selection_cfg = self._optional_mapping(sink_selection_cfg, "sink_selection_cfg")
        self.backward_cfg = self._optional_mapping(backward_cfg, "backward_cfg")
        self._validate_edge_batch = bool(self.runtime_cfg.get("validate_edge_batch", _DEFAULT_VALIDATE_EDGE_BATCH))
        self._validate_rollout_batch = bool(
            self.runtime_cfg.get("validate_rollout_batch", _DEFAULT_VALIDATE_ROLLOUT_BATCH)
        )
        self._validate_on_device = bool(self.runtime_cfg.get("validate_on_device", _DEFAULT_VALIDATE_ON_DEVICE))
        self._require_precomputed_edge_batch = bool(
            self.runtime_cfg.get(
                "require_precomputed_edge_batch",
                _DEFAULT_REQUIRE_PRECOMPUTED_EDGE_BATCH,
            )
        )
        self._force_fp32 = bool(self.runtime_cfg.get("force_fp32", _DEFAULT_FORCE_FP32))

    def _validate_runtime_cfg(self) -> None:
        if "vectorized_rollouts" in self.runtime_cfg:
            raise ValueError("runtime_cfg.vectorized_rollouts has been removed; use rollout_chunk_size.")

    @staticmethod
    def _coerce_path(value: str | Path | None, name: str) -> Optional[Path]:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError(f"{name} must be a non-empty path string.")
            return Path(text)
        raise TypeError(f"{name} must be a path or string, got {type(value).__name__}.")

    def _init_vocab_settings(
        self,
        *,
        entity_vocab_path: Optional[str | Path],
        relation_vocab_path: Optional[str | Path],
        relation_vocab_size: Optional[int],
        inverse_relation_suffix: Optional[str],
    ) -> None:
        self._entity_vocab_path = self._coerce_path(entity_vocab_path, "entity_vocab_path")
        self._relation_vocab_path = self._coerce_path(relation_vocab_path, "relation_vocab_path")
        self._relation_vocab_size_cfg = relation_vocab_size
        self._inverse_relation_suffix = inverse_relation_suffix
        self.relation_vocab_size = None
        self.entity_vocab_size = None
        self._relation_inverse_map = None
        self._cvt_mask = None
        self._vocab_initialized = False
        self._runtime_initialized = False

    def _cvt_init_enabled(self) -> bool:
        return bool(self.cvt_init_cfg.get("enabled", _DEFAULT_CVT_INIT_ENABLED))

    def _validate_structured_cfg(self) -> None:
        if self.edge_score_cfg:
            raise ValueError("edge_score_cfg has been removed; TrajectoryAgent handles edge scoring.")
        if self.entity_score_cfg:
            raise ValueError("entity_score_cfg has been removed; TrajectoryAgent handles edge scoring.")
        if "bias_init" in self.flow_cfg or "bias_init_value" in self.flow_cfg:
            raise ValueError("flow_cfg.bias_init has been removed; drop bias_init and bias_init_value.")
        if self._relation_vocab_path is None:
            raise ValueError("relation_vocab_path must be set to build relation_inverse_map required for backward flow.")
        if self._relation_vocab_size_cfg is not None:
            self._require_positive_int(self._relation_vocab_size_cfg, "relation_vocab_size")
        suffix = self._inverse_relation_suffix
        if suffix is not None and (not isinstance(suffix, str) or not suffix):
            raise ValueError("inverse_relation_suffix must be a non-empty string.")
        if self._entity_vocab_path is None:
            raise ValueError("entity_vocab_path must be set for CVT initialization.")

    def _load_vocab_assets(self) -> None:
        if self._vocab_initialized:
            return
        self._init_relation_vocab(
            self._relation_vocab_path,
            self._relation_vocab_size_cfg,
            self._entity_vocab_path,
            load_entity=False,
        )
        self._init_relation_inverse_map(self._relation_vocab_path, self._inverse_relation_suffix)
        self._init_cvt_mask(self._entity_vocab_path)
        self._vocab_initialized = True

    def _ensure_runtime_initialized(self) -> None:
        if self._runtime_initialized:
            return
        self._load_vocab_assets()
        self._init_cvt_init()
        self.actor = self._build_actor(agent=self.agent, context_mode="question")
        if self._backward_share_agent:
            self.actor_backward = self.actor
        else:
            self.actor_backward = self._build_actor(
                agent=self.agent_backward,
                context_mode="start_node",
            )
        self._init_engine_components()
        self._runtime_initialized = True

    def _init_relation_vocab(
        self,
        relation_vocab_path: Optional[Path],
        relation_vocab_size: Optional[int],
        entity_vocab_path: Optional[Path],
        *,
        load_entity: bool,
    ) -> None:
        if relation_vocab_path is None:
            raise ValueError("relation_vocab_path must be set to resolve relation vocab size.")
        relation_count = _load_relation_vocab_size(relation_vocab_path)
        if relation_vocab_size is not None:
            expected_size = self._require_positive_int(relation_vocab_size, "relation_vocab_size")
            if int(expected_size) != int(relation_count):
                raise ValueError("relation_vocab_size does not match relation_vocab.parquet.")
        self.relation_vocab_size = int(relation_count)
        if load_entity:
            if entity_vocab_path is None:
                raise ValueError("entity_vocab_path must be set to load entity vocab size.")
            self.entity_vocab_size = _load_entity_vocab_size(entity_vocab_path)
        else:
            self.entity_vocab_size = None

    def _init_relation_inverse_map(
        self,
        relation_vocab_path: Optional[str | Path],
        inverse_relation_suffix: Optional[str],
    ) -> None:
        self._relation_inverse_map = None
        self._relation_is_inverse = None
        if relation_vocab_path is None:
            return
        if self.relation_vocab_size is None:
            raise RuntimeError("relation_vocab_size must be initialized before loading inverse relations.")
        path = Path(relation_vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"relation_vocab.parquet not found: {path}")
        suffix = inverse_relation_suffix or _INVERSE_RELATION_SUFFIX_DEFAULT
        if not isinstance(suffix, str) or not suffix:
            raise ValueError("inverse_relation_suffix must be a non-empty string.")
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("pyarrow is required to load relation_vocab.parquet.") from exc
        table = pq.read_table(path, columns=["relation_id", "kg_id"])
        relation_ids = torch.as_tensor(table.column("relation_id").to_numpy(), dtype=torch.long)
        kg_ids = [str(val) for val in table.column("kg_id").to_pylist()]
        if relation_ids.numel() == _ZERO:
            raise ValueError("relation_vocab.parquet is empty.")
        max_id = int(relation_ids.max().detach().tolist())
        if max_id >= self.relation_vocab_size:
            raise ValueError("relation_vocab.parquet relation_id exceeds relation_vocab_size.")
        id_lookup = {kg_id: int(rel_id) for rel_id, kg_id in zip(relation_ids.tolist(), kg_ids)}
        inverse_map = torch.full(
            (self.relation_vocab_size,),
            _INVALID_RELATION_ID,
            dtype=torch.long,
            device=self.device,
        )
        inverse_mask = torch.zeros(
            (self.relation_vocab_size,),
            dtype=torch.bool,
            device=self.device,
        )
        for rel_id, kg_id in zip(relation_ids.tolist(), kg_ids):
            inv_key = kg_id[:-len(suffix)] if kg_id.endswith(suffix) else f"{kg_id}{suffix}"
            inv_id = id_lookup.get(inv_key)
            if inv_id is not None:
                inverse_map[int(rel_id)] = int(inv_id)
            if kg_id.endswith(suffix):
                inverse_mask[int(rel_id)] = True
        self.register_buffer("relation_inverse_map", inverse_map, persistent=False)
        self._relation_inverse_map = inverse_map
        self.register_buffer("relation_is_inverse", inverse_mask, persistent=False)
        self._relation_is_inverse = inverse_mask

    def _init_cvt_mask(self, entity_vocab_path: Optional[str | Path]) -> None:
        self._cvt_mask = None
        if entity_vocab_path is None:
            raise ValueError("entity_vocab_path must be set for CVT initialization.")
        path = Path(entity_vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"entity_vocab.parquet not found: {path}")
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("pyarrow is required to load entity_vocab.parquet.") from exc
        table = pq.read_table(path, columns=["entity_id", "is_cvt"])
        entity_ids = torch.as_tensor(table.column("entity_id").to_numpy(), dtype=torch.long)
        is_cvt = torch.as_tensor(table.column("is_cvt").to_numpy(), dtype=torch.bool)
        if entity_ids.numel() == _ZERO:
            raise ValueError("entity_vocab.parquet is empty.")
        if entity_ids.numel() != is_cvt.numel():
            raise ValueError("entity_vocab.parquet entity_id/is_cvt length mismatch.")
        max_id = int(entity_ids.max().detach().tolist())
        if max_id < _ZERO:
            raise ValueError("entity_vocab.parquet contains negative entity_id values.")
        mask = torch.zeros((max_id + _ONE,), dtype=torch.bool)
        mask[entity_ids] = is_cvt
        self._cvt_mask = mask

    def _init_eval_settings(self) -> None:
        self._eval_rollout_prefixes, self._eval_rollouts = self._parse_eval_rollouts(self.evaluation_cfg)
        eval_temp_cfg = self.evaluation_cfg.get("rollout_temperature")
        if eval_temp_cfg is None:
            raise ValueError("evaluation_cfg.rollout_temperature must be set explicitly (no implicit eval temperature).")
        self._eval_rollout_temperature = float(eval_temp_cfg)
        if self._eval_rollout_temperature < 0.0:
            raise ValueError(f"evaluation_cfg.rollout_temperature must be >= 0, got {self._eval_rollout_temperature}.")
        self._composite_score_cfg = gfn_metrics.resolve_composite_score_cfg(
            self.evaluation_cfg.get("composite_score_cfg")
        )
        self._eval_temp_extras_default = _DEFAULT_EVAL_TEMPERATURE_EXTRAS

    def _init_logging_settings(self) -> None:
        self._train_prog_bar = self._coerce_str_set(
            self.logging_cfg.get("train_prog_bar"),
            "logging_cfg.train_prog_bar",
        )
        self._eval_prog_bar = self._coerce_str_set(
            self.logging_cfg.get("eval_prog_bar"),
            "logging_cfg.eval_prog_bar",
        )
        self._log_on_step_train = bool(self.logging_cfg.get("log_on_step_train", _DEFAULT_LOG_ON_STEP_TRAIN))

    def _init_grad_clip_settings(self) -> None:
        self._adaptive_grad_clip = bool(self.training_cfg.get("adaptive_gradient_clip", _DEFAULT_GRAD_CLIP_ADAPTIVE))
        log_eps = self.training_cfg.get("grad_clip_log_eps", _DEFAULT_GRAD_CLIP_LOG_EPS)
        self._grad_clip_log_eps = self._require_positive_float(log_eps, "training_cfg.grad_clip_log_eps")
        tail_eps = self.training_cfg.get("grad_clip_tail_prob_eps", _DEFAULT_GRAD_CLIP_TAIL_PROB_EPS)
        self._grad_clip_tail_prob_eps = self._require_positive_float(
            tail_eps,
            "training_cfg.grad_clip_tail_prob_eps",
        )
        self.register_buffer("grad_log_norm_ema", torch.tensor(float(_ZERO)), persistent=True)
        self.register_buffer("grad_log_norm_sq_ema", torch.tensor(float(_ZERO)), persistent=True)
        self.register_buffer("grad_ema_initialized", torch.tensor(_ZERO, dtype=torch.long), persistent=True)
        self._grad_clip_algorithm_default = _DEFAULT_MANUAL_GRAD_CLIP_ALGO
        self._grad_clip_norm_type_default = _DEFAULT_MANUAL_GRAD_CLIP_NORM_TYPE
        self._grad_nonfinite_max_params_default = _DEFAULT_GRAD_NONFINITE_MAX_PARAMS
        log_interval = self.training_cfg.get("grad_clip_log_interval", _DEFAULT_GRAD_CLIP_LOG_INTERVAL)
        self._grad_clip_log_interval = self._require_positive_int(
            log_interval,
            "training_cfg.grad_clip_log_interval",
        )

    def _init_backbone(self, *, emb_dim: int, finetune: bool) -> None:
        self.backbone = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            finetune=finetune,
            force_fp32=self._force_fp32,
        )

    def _init_cvt_init(self) -> None:
        cfg = self.cvt_init_cfg
        if cfg is None:
            cfg = {}
        if not bool(cfg.get("enabled", _DEFAULT_CVT_INIT_ENABLED)):
            raise ValueError("cvt_init_cfg.enabled must be true; CVT initialization is mandatory.")
        self.cvt_init = CvtNodeInitializer(
            hidden_dim=self.hidden_dim,
            force_fp32=self._force_fp32,
        )

    def _init_start_selection(self) -> None:
        cfg = self.start_selection_cfg
        enabled = bool(cfg.get("enabled", _DEFAULT_START_SELECTION_ENABLED)) if cfg else _DEFAULT_START_SELECTION_ENABLED
        if not enabled:
            self.start_selector = None
            return
        dropout = self._require_non_negative_float(
            cfg.get("dropout", _DEFAULT_START_SELECTION_DROPOUT),
            "start_selection_cfg.dropout",
        )
        self.start_selector = StartSelector(
            token_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            force_fp32=self._force_fp32,
        )

    def _init_sink_selection(self) -> None:
        cfg = self.sink_selection_cfg
        enabled = bool(cfg.get("enabled", _DEFAULT_SINK_SELECTION_ENABLED)) if cfg else _DEFAULT_SINK_SELECTION_ENABLED
        if not enabled:
            self.sink_selector = None
            return
        dropout = self._require_non_negative_float(
            cfg.get("dropout", _DEFAULT_SINK_SELECTION_DROPOUT),
            "sink_selection_cfg.dropout",
        )
        self.sink_selector = SinkSelector(
            token_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            force_fp32=self._force_fp32,
        )

    def _init_backward_policy_settings(self) -> None:
        cfg = self.backward_cfg
        share_state = bool(cfg.get(_BACKWARD_SHARE_STATE_KEY, _DEFAULT_BACKWARD_SHARE_STATE_ENCODER))
        share_edge = bool(cfg.get(_BACKWARD_SHARE_EDGE_KEY, _DEFAULT_BACKWARD_SHARE_EDGE_SCORER))
        if share_state != share_edge:
            raise ValueError("backward_cfg.share_state_encoder/share_edge_scorer must match when using TrajectoryAgent.")
        self._backward_share_agent = share_state and share_edge

    def _build_trajectory_agent(
        self,
        *,
        cfg: Mapping[str, Any],
        name: str,
    ) -> tuple[TrajectoryAgent, int]:
        hidden_dim_raw = cfg.get("state_dim", cfg.get("hidden_dim", self.hidden_dim))
        hidden_dim = self._require_positive_int(hidden_dim_raw, f"{name}.state_dim")
        dropout = float(cfg.get("dropout", _DEFAULT_AGENT_DROPOUT))
        if "gate_bias_init" in cfg or "gate_feature_dim" in cfg or "step_scale" in cfg:
            raise ValueError(f"{name} uses TrajectoryAgent; remove gate_bias_init, gate_feature_dim, step_scale.")
        agent = TrajectoryAgent(
            token_dim=self.hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            force_fp32=self._force_fp32,
        )
        return agent, hidden_dim

    def _init_agent_components(self) -> int:
        agent, hidden_dim = self._build_trajectory_agent(cfg=self.state_cfg, name="state_cfg")
        self.agent = agent
        return hidden_dim

    def _init_backward_agent_components(self) -> None:
        if self._backward_share_agent:
            self.agent_backward = self.agent
            self._state_input_dim_backward = self._state_input_dim
            return
        agent, hidden_dim = self._build_trajectory_agent(cfg=self.state_cfg, name="backward_cfg.state_cfg")
        self.agent_backward = agent
        self._state_input_dim_backward = hidden_dim

    def _init_reward_schedule(self) -> None:
        self._potential_weight_init = float(getattr(self.reward_fn, "potential_weight", _ZERO))
        self._potential_weight_end = float(getattr(self.reward_fn, "potential_weight_end", _ZERO))
        schedule_raw = getattr(self.reward_fn, "potential_schedule", _POTENTIAL_SCHEDULE_LINEAR)
        self._potential_schedule = str(schedule_raw or "").strip().lower()
        if self._potential_schedule not in _POTENTIAL_SCHEDULES:
            raise ValueError(f"reward_fn.potential_schedule must be one of {sorted(_POTENTIAL_SCHEDULES)}.")
        anneal_cfg = getattr(self.reward_fn, "potential_anneal_epochs", None)
        decay_cfg = getattr(self.reward_fn, "potential_weight_decay_epochs", None)
        if anneal_cfg is not None:
            self._potential_anneal_epochs = int(anneal_cfg)
        elif decay_cfg is not None:
            self._potential_anneal_epochs = int(decay_cfg)
        else:
            self._potential_anneal_epochs = None
        if self._potential_anneal_epochs is not None and self._potential_anneal_epochs < _ZERO:
            raise ValueError("reward_fn.potential_anneal_epochs must be >= 0.")
        pure_cfg = getattr(self.reward_fn, "potential_pure_phase_ratio", None)
        if pure_cfg is None:
            self._potential_pure_phase_ratio = None
        else:
            self._potential_pure_phase_ratio = float(pure_cfg)
            if not (_MIN_PURE_PHASE_RATIO <= self._potential_pure_phase_ratio <= _MAX_PURE_PHASE_RATIO):
                raise ValueError("reward_fn.potential_pure_phase_ratio must be in [0, 1].")

    def _build_actor(
        self,
        *,
        agent: TrajectoryAgent,
        context_mode: str,
        actor_cfg: Optional[Mapping[str, Any]] = None,
    ) -> GFlowNetActor:
        cfg = self.actor_cfg if actor_cfg is None else actor_cfg
        policy_temperature = float(cfg.get("policy_temperature", _DEFAULT_POLICY_TEMPERATURE))
        stop_bias_init = cfg.get("stop_bias_init")
        if stop_bias_init is None:
            stop_bias = None
        else:
            stop_bias = self._require_float(stop_bias_init, "actor_cfg.stop_bias_init")
        return GFlowNetActor(
            env=self.env,
            agent=agent,
            max_steps=self.max_steps,
            policy_temperature=policy_temperature,
            stop_bias_init=stop_bias,
            context_mode=context_mode,
        )

    def _init_engine_components(self) -> None:
        if self.actor is None:
            raise RuntimeError("actor must be initialized before engine components.")
        self.input_validator = GFlowNetInputValidator(
            validate_edge_batch=self._validate_edge_batch,
            validate_rollout_batch=self._validate_rollout_batch,
            move_to_device=self._validate_on_device,
        )
        self.batch_processor = GFlowNetBatchProcessor(
            backbone=self.backbone,
            cvt_init=self.cvt_init,
            cvt_mask=self._cvt_mask,
            require_precomputed_edge_batch=self._require_precomputed_edge_batch,
        )
        self.engine = GFlowNetEngine(
            actor=self.actor,
            actor_backward=self.actor_backward,
            reward_fn=self.reward_fn,
            env=self.env,
            log_f=self.log_f,
            log_f_backward=self.log_f_backward,
            start_selector=self.start_selector,
            sink_selector=self.sink_selector,
            flow_spec=self.flow_spec,
            relation_inverse_map=self._relation_inverse_map,
            relation_is_inverse=self._relation_is_inverse,
            batch_processor=self.batch_processor,
            input_validator=self.input_validator,
            composite_score_cfg=self._composite_score_cfg,
            dual_stream_cfg=self.training_cfg.get("dual_stream"),
            subtb_cfg=self.training_cfg.get("subtb"),
            start_selection_cfg=self.start_selection_cfg,
            sink_selection_cfg=self.sink_selection_cfg,
            z_align_cfg=self.training_cfg.get("z_align"),
            h_guidance_cfg=self.training_cfg.get("h_guidance"),
            replay_cfg=self.training_cfg.get("replay"),
        )

    def _assert_training_cfg_contract(self) -> None:
        if self.training_cfg.get("safety_net") is not None:
            raise ValueError("training_cfg.safety_net has been removed; no implicit shortcuts are allowed.")
        if self.training_cfg.get("sp_dropout") is not None:
            raise ValueError("training_cfg.sp_dropout has been removed; do not configure SP-dropout.")
        if self.training_cfg.get("start_backtrack") is not None:
            raise ValueError("training_cfg.start_backtrack has been removed; use training_cfg.dual_stream.")
        if self.training_cfg.get("backward_warmup") is not None:
            raise ValueError("training_cfg.backward_warmup has been removed; use training_cfg.dual_stream.")
        if self.training_cfg.get("stitching") is not None:
            raise ValueError("training_cfg.stitching has been removed; SubTB is the only TB loss.")
        if self.training_cfg.get("curriculum") is not None:
            raise ValueError("training_cfg.curriculum has been removed; disable curriculum scheduling.")
        if self.training_cfg.get("log_f_target") is not None:
            raise ValueError("training_cfg.log_f_target has been removed; no target networks are used.")
        subtb_cfg = self.training_cfg.get("subtb")
        if subtb_cfg is None:
            raise ValueError("training_cfg.subtb is required; SubTB is the only trajectory balance loss.")
        if isinstance(subtb_cfg, bool):
            if not subtb_cfg:
                raise ValueError("training_cfg.subtb must be enabled; SubTB is required.")
        elif isinstance(subtb_cfg, Mapping):
            if not bool(subtb_cfg.get("enabled", True)):
                raise ValueError("training_cfg.subtb.enabled must be true; SubTB is required.")
        else:
            raise TypeError("training_cfg.subtb must be a mapping or bool.")

    def _save_serializable_hparams(self) -> None:
        # 仅保存可序列化的标量，避免将配置映射写入 checkpoint。
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "backbone",
                "cvt_init",
                "agent",
                "agent_backward",
                "log_f",
                "log_f_backward",
                "reward_fn",
                "env",
                "actor_backward",
                "start_selector",
                "sink_selector",
                "actor_cfg",
                "backward_cfg",
                "training_cfg",
                "evaluation_cfg",
                "flow_cfg",
                "optimizer_cfg",
                "scheduler_cfg",
                "logging_cfg",
            ],
        )

    def configure_optimizers(self):
        self._ensure_runtime_initialized()
        return self.training_loop.configure_optimizers()

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: Optional[int] = None,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        _ = optimizer_idx, gradient_clip_val, gradient_clip_algorithm
        self.training_loop.grad_clipper.clip_gradients(optimizer)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self._ensure_runtime_initialized()
        return super().load_state_dict(state_dict, strict=strict)

    def setup(self, stage: Optional[str] = None) -> None:
        _ = stage
        self._ensure_runtime_initialized()

    def forward(self, batch: Any) -> torch.Tensor:
        raise NotImplementedError("GFlowNetModule.forward is not supported; use actor.rollout() or engine APIs.")

    @staticmethod
    def _require_mapping(cfg: Any, name: str) -> Mapping[str, Any]:
        if cfg is None:
            raise ValueError(f"{name} must be provided as a mapping (got None).")
        if not isinstance(cfg, Mapping):
            raise TypeError(f"{name} must be a mapping, got {type(cfg).__name__}.")
        return cfg

    @staticmethod
    def _optional_mapping(cfg: Any, name: str) -> Mapping[str, Any]:
        if cfg is None:
            return {}
        if not isinstance(cfg, Mapping):
            raise TypeError(f"{name} must be a mapping or None, got {type(cfg).__name__}.")
        return cfg

    @staticmethod
    def _coerce_str_set(value: Any, name: str) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of strings, got {type(value).__name__}.")
        if isinstance(value, Sequence):
            return set(str(v) for v in value)
        raise TypeError(f"{name} must be a sequence of strings, got {type(value).__name__}.")

    @staticmethod
    def _require_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a float, got bool.")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise TypeError(f"{name} must be a float, got empty string.")
            try:
                return float(text)
            except ValueError as exc:
                raise TypeError(f"{name} must be a float, got {value!r}.") from exc
        raise TypeError(f"{name} must be a float, got {type(value).__name__}.")

    def require_float(self, value: Any, name: str) -> float:
        return self._require_float(value, name)

    @staticmethod
    def _require_positive_float(value: Any, name: str) -> float:
        parsed = GFlowNetModule._require_float(value, name)
        if parsed <= float(_ZERO):
            raise ValueError(f"{name} must be > 0, got {parsed}.")
        return parsed

    def require_positive_float(self, value: Any, name: str) -> float:
        return self._require_positive_float(value, name)

    @staticmethod
    def _require_non_negative_float(value: Any, name: str) -> float:
        parsed = GFlowNetModule._require_float(value, name)
        if parsed < float(_ZERO):
            raise ValueError(f"{name} must be >= 0, got {parsed}.")
        return parsed

    def require_non_negative_float(self, value: Any, name: str) -> float:
        return self._require_non_negative_float(value, name)

    @staticmethod
    def _require_probability_open(value: Any, name: str) -> float:
        parsed = GFlowNetModule._require_float(value, name)
        if not (float(_ZERO) < parsed < float(_ONE)):
            raise ValueError(f"{name} must be in (0, 1), got {parsed}.")
        return parsed

    def require_probability_open(self, value: Any, name: str) -> float:
        return self._require_probability_open(value, name)

    @staticmethod
    def _require_positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a positive int, got bool.")
        if isinstance(value, int):
            parsed = int(value)
        elif isinstance(value, float):
            if not value.is_integer():
                raise TypeError(f"{name} must be a positive int, got {value}.")
            parsed = int(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text or not text.lstrip("-").isdigit():
                raise TypeError(f"{name} must be a positive int, got {value!r}.")
            parsed = int(text)
        else:
            raise TypeError(f"{name} must be a positive int, got {type(value).__name__}.")
        if parsed <= 0:
            raise ValueError(f"{name} must be > 0, got {parsed}.")
        return parsed

    def require_positive_int(self, value: Any, name: str) -> int:
        return self._require_positive_int(value, name)

    @staticmethod
    def _parse_eval_rollouts(cfg: Mapping[str, Any]) -> tuple[list[int], int]:
        value = cfg.get("num_eval_rollouts", 1)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            prefixes: list[int] = []
            for idx, raw in enumerate(value):
                prefixes.append(GFlowNetModule._require_positive_int(raw, f"evaluation_cfg.num_eval_rollouts[{idx}]"))
            if not prefixes:
                raise ValueError("evaluation_cfg.num_eval_rollouts must be a non-empty list.")
            prefixes = sorted(set(prefixes))
            return prefixes, max(prefixes)
        if isinstance(value, int):
            count = GFlowNetModule._require_positive_int(value, "evaluation_cfg.num_eval_rollouts")
            return [count], count
        raise TypeError("evaluation_cfg.num_eval_rollouts must be an int or sequence " f"(got {type(value).__name__}).")

    def _build_rollout_cfg(self, *, is_training: bool) -> GFlowNetRolloutConfig:
        if not is_training:
            self._refresh_eval_settings()
        num_rollouts = self._resolve_num_rollouts(is_training)
        chunk_raw = (
            self.training_cfg.get("rollout_chunk_size", _DEFAULT_ROLLOUT_CHUNK_SIZE)
            if is_training
            else self.evaluation_cfg.get("rollout_chunk_size", _DEFAULT_ROLLOUT_CHUNK_SIZE)
        )
        chunk_name = "training_cfg.rollout_chunk_size" if is_training else "evaluation_cfg.rollout_chunk_size"
        rollout_chunk_size = self._resolve_rollout_chunk_size(
            chunk_raw,
            num_rollouts=num_rollouts,
            name=chunk_name,
        )
        return GFlowNetRolloutConfig(
            num_rollouts=num_rollouts,
            eval_rollout_prefixes=self._eval_rollout_prefixes,
            eval_rollout_temperature=self._eval_rollout_temperature,
            rollout_chunk_size=rollout_chunk_size,
            is_training=is_training,
        )

    def build_rollout_cfg(self, *, is_training: bool) -> GFlowNetRolloutConfig:
        return self._build_rollout_cfg(is_training=is_training)

    def _build_eval_rollout_cfg(self, *, temperature: float) -> GFlowNetRolloutConfig:
        self._refresh_eval_settings()
        num_rollouts = self._resolve_num_rollouts(is_training=False)
        chunk_raw = self.evaluation_cfg.get("rollout_chunk_size", _DEFAULT_ROLLOUT_CHUNK_SIZE)
        rollout_chunk_size = self._resolve_rollout_chunk_size(
            chunk_raw,
            num_rollouts=num_rollouts,
            name="evaluation_cfg.rollout_chunk_size",
        )
        return GFlowNetRolloutConfig(
            num_rollouts=num_rollouts,
            eval_rollout_prefixes=self._eval_rollout_prefixes,
            eval_rollout_temperature=float(temperature),
            rollout_chunk_size=rollout_chunk_size,
            is_training=False,
        )

    def build_eval_rollout_cfg(self, *, temperature: float) -> GFlowNetRolloutConfig:
        return self._build_eval_rollout_cfg(temperature=temperature)

    def _resolve_num_rollouts(self, is_training: bool) -> int:
        if is_training:
            return self._require_positive_int(
                self.training_cfg.get("num_train_rollouts"),
                "training_cfg.num_train_rollouts",
            )
        return self._require_positive_int(self._eval_rollouts, "evaluation_cfg.num_eval_rollouts")

    @classmethod
    def _resolve_rollout_chunk_size(
        cls,
        value: Any,
        *,
        num_rollouts: int,
        name: str,
    ) -> int:
        chunk_size = cls._require_positive_int(value, name)
        if chunk_size > num_rollouts:
            return num_rollouts
        return chunk_size

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        # Move only hot-path tensors to avoid repeated H2D copies while keeping large ID tensors on CPU.
        node_embeddings = getattr(batch, "node_embeddings", None)
        edge_embeddings = getattr(batch, "edge_embeddings", None)
        has_node = torch.is_tensor(node_embeddings)
        has_edge = torch.is_tensor(edge_embeddings)
        if has_node != has_edge:
            raise ValueError("node_embeddings and edge_embeddings must be attached together.")
        if not has_node:
            self._attach_embeddings(batch, device=device)
        self._ensure_answer_ids_device(batch, device=device)
        if device.type == "cpu":
            return batch
        for key in _BATCH_DEVICE_KEYS:
            value = getattr(batch, key, None)
            if torch.is_tensor(value):
                if key in _BATCH_FLOAT_KEYS:
                    setattr(batch, key, value.to(device=device, dtype=torch.float32, non_blocking=True))
                else:
                    setattr(batch, key, value.to(device=device, non_blocking=True))
        return batch

    def _attach_embeddings(self, batch: Any, *, device: torch.device) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("datamodule is required to attach embeddings.")
        resources = getattr(datamodule, "shared_resources", None)
        if resources is None:
            raise RuntimeError("datamodule.shared_resources is required to attach embeddings.")
        global_embeddings = resources.global_embeddings
        attach_embeddings_to_batch(batch, global_embeddings=global_embeddings, embeddings_device=device)

    @staticmethod
    def _ensure_answer_ids_device(batch: Any, *, device: torch.device) -> None:
        answer_ids = getattr(batch, "answer_entity_ids", None)
        answer_ptr = getattr(batch, "answer_entity_ids_ptr", None)
        if not torch.is_tensor(answer_ids) or not torch.is_tensor(answer_ptr):
            return
        target = device
        if device.type == "cpu":
            target = torch.device("cpu")
        batch.answer_entity_ids = answer_ids.to(device=target, dtype=torch.long, non_blocking=True)
        batch.answer_entity_ids_ptr = answer_ptr.to(device=target, dtype=torch.long, non_blocking=True)

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.training_loop.on_before_optimizer_step(optimizer, optimizer_idx)

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

    def training_step(self, batch, batch_idx: int):
        self._ensure_runtime_initialized()
        self._log_batch_scale(batch, batch_idx=batch_idx)
        return self.training_loop.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx: int):
        self._ensure_runtime_initialized()
        self.training_loop.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx: int):
        self._ensure_runtime_initialized()
        self.training_loop.test_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self._ensure_runtime_initialized()
        return self.training_loop.predict_step(batch, batch_idx, dataloader_idx)

    def on_train_epoch_start(self) -> None:
        self._ensure_runtime_initialized()
        self.training_loop.on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        self.training_loop.on_train_epoch_end()

    def _validate_batch_inputs(self, batch: Any, *, is_training: bool, require_rollout: bool) -> None:
        if require_rollout:
            self.input_validator.validate_rollout_batch(
                batch,
                device=self.device,
                is_training=is_training,
            )
            return
        self.input_validator.validate_edge_batch(batch, device=self.device)

    def validate_batch_inputs(self, batch: Any, *, is_training: bool, require_rollout: bool) -> None:
        self._ensure_runtime_initialized()
        self._validate_batch_inputs(batch, is_training=is_training, require_rollout=require_rollout)

    def _refresh_eval_settings(self) -> None:
        """Ensure eval rollouts reflect current config (even when loading old checkpoints)."""
        prefixes, count = self._parse_eval_rollouts(self.evaluation_cfg)
        self._eval_rollout_prefixes = prefixes
        self._eval_rollouts = count
        eval_temp_cfg = self.evaluation_cfg.get("rollout_temperature")
        if eval_temp_cfg is None:
            raise ValueError("evaluation_cfg.rollout_temperature must be set explicitly (no implicit eval temperature).")
        if float(eval_temp_cfg) < 0.0:
            raise ValueError(f"evaluation_cfg.rollout_temperature must be >= 0, got {eval_temp_cfg}.")
        self._eval_rollout_temperature = float(eval_temp_cfg)
        self._composite_score_cfg = gfn_metrics.resolve_composite_score_cfg(
            self.evaluation_cfg.get("composite_score_cfg")
        )
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.set_composite_score_cfg(self._composite_score_cfg)

__all__ = ["GFlowNetModule"]
