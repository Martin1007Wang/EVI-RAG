from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import lmdb
import torch
from lightning import LightningModule
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchmetrics import MeanMetric, MetricCollection

from src.models.components import (
    GFlowNetActor,
    GraphEnv,
)
from src.models.components.logz_features import FlowFeatureSpec, resolve_flow_spec
from src.gfn.training import GFlowNetTrainingLoop
from src.gfn.ops import GFlowNetBatchProcessor, GFlowNetInputValidator, RolloutInputs
from src.gfn.engine import GFlowNetEngine, GFlowNetRolloutConfig
from src.metrics import gflownet as gfn_metrics

_ZERO = 0
_ONE = 1
_HALF = 0.5
_NAN = float("nan")
_DEFAULT_POLICY_TEMPERATURE = 1.0
_DEFAULT_CHECK_FINITE = False
_DEFAULT_COSINE_BIAS_ALPHA = 0.0
_DEFAULT_COSINE_RELATION_BIAS_ALPHA = 0.0
_DEFAULT_VALIDATE_EDGE_BATCH = True
_DEFAULT_VALIDATE_ROLLOUT_BATCH = True
_DEFAULT_REQUIRE_PRECOMPUTED_EDGE_BATCH = True
_DEFAULT_REQUIRE_PRECOMPUTED_NODE_IN_DEGREE = True
_DEFAULT_ROLLOUT_CHUNK_SIZE = 1
_DEFAULT_LOG_ON_STEP_TRAIN = False
_DEFAULT_EVAL_TEMPERATURE_EXTRAS: tuple[float, ...] = ()
_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_EDGE_SCORE_LAYERS = 2
_DEFAULT_EDGE_SCORE_DROPOUT = 0.1
_DEFAULT_EDGE_SCORE_BIAS_INIT = 0.0
_DEFAULT_EDGE_SCORE_ACTIVATION_CHECKPOINTING = False
_DEFAULT_STATE_DROPOUT = 0.0
_DEFAULT_RELATION_USE_ACTIVE_NODES = False
_DEFAULT_MANUAL_GRAD_CLIP_ALGO = "norm"
_DEFAULT_MANUAL_GRAD_CLIP_NORM_TYPE = 2.0
_DEFAULT_GRAD_CLIP_ADAPTIVE = False
_DEFAULT_GRAD_CLIP_LOG_EPS = 1.0e-8
_DEFAULT_GRAD_CLIP_TAIL_PROB_EPS = 1.0e-6
_DEFAULT_GRAD_NONFINITE_MAX_PARAMS = 20
_DEFAULT_CONTROL_ENABLED = False
_DEFAULT_CONTROL_TARGET_MODE = "reachable_horizon_frac"
_CONTROL_TARGET_MODE_REACHABLE = "reachable_horizon_frac"
_CONTROL_TARGET_MODE_FIXED = "fixed"
_CONTROL_SUCCESS_KEY = "pass@1"
_CONTROL_REACHABLE_KEY = "control/reachable_horizon_frac"
_FLOW_OUTPUT_DIM = 1
_EMBED_INIT_STD_POWER = 0.25
_GRU_NUM_GATES = 3
_GRU_ORTHO_GAIN = 1.0
_VOCAB_MAX_READERS = 1
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
_FLOW_BIAS_INIT_NONE = "none"
_FLOW_BIAS_INIT_MIN_LOG_REWARD = "min_log_reward"
_FLOW_BIAS_INIT_BATCH_LOG_REWARD = "batch_log_reward"
_FLOW_BIAS_INIT_VALUE = "value"
_FLOW_BIAS_INIT_OPTIONS = {
    _FLOW_BIAS_INIT_NONE,
    _FLOW_BIAS_INIT_MIN_LOG_REWARD,
    _FLOW_BIAS_INIT_BATCH_LOG_REWARD,
    _FLOW_BIAS_INIT_VALUE,
}
_FLOW_BIAS_INIT_VALUE_KEY = "bias_init_value"
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
    "node_in_degree",
    "edge_batch",
    "edge_ptr",
    "answer_entity_ids",
    "answer_entity_ids_ptr",
    "is_dummy_agent",
)
_BATCH_FLOAT_KEYS = {
    "question_emb",
    "node_embeddings",
    "edge_embeddings",
}


def _init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _init_embedding_linear(layer: nn.Linear, *, out_dim: int) -> None:
    std = float(out_dim) ** (-_EMBED_INIT_STD_POWER)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _init_gru_cell(gru: nn.GRUCell) -> None:
    for weight in (gru.weight_ih, gru.weight_hh):
        for chunk in weight.chunk(_GRU_NUM_GATES, dim=0):
            nn.init.orthogonal_(chunk, gain=_GRU_ORTHO_GAIN)
    if gru.bias_ih is not None:
        nn.init.zeros_(gru.bias_ih)
    if gru.bias_hh is not None:
        nn.init.zeros_(gru.bias_hh)


def _load_relation_count(vocabulary_path: Path) -> int:
    vocab_path = Path(vocabulary_path).expanduser().resolve()
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary LMDB not found: {vocab_path}")
    env = lmdb.open(str(vocab_path), readonly=True, lock=False, max_readers=_VOCAB_MAX_READERS)
    try:
        with env.begin(write=False) as txn:
            payload = txn.get(b"relation_labels")
            if payload is None:
                raise ValueError("Vocabulary LMDB missing relation_labels.")
            labels = json.loads(payload.decode("utf-8"))
    finally:
        env.close()
    if not isinstance(labels, list):
        raise ValueError("relation_labels must decode to a list.")
    if len(labels) <= _ZERO:
        raise ValueError("relation_labels is empty; relation vocab required.")
    return len(labels)


class EmbeddingBackbone(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        finetune: bool = _DEFAULT_BACKBONE_FINETUNE,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.finetune = bool(finetune)

        self.node_norm = nn.LayerNorm(self.emb_dim)
        self.rel_norm = nn.LayerNorm(self.emb_dim)
        self.node_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.rel_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        self.q_proj = nn.Linear(self.emb_dim, self.hidden_dim)
        _init_embedding_linear(self.node_proj, out_dim=self.hidden_dim)
        _init_linear(self.rel_proj)
        _init_linear(self.q_proj)
        if not self.finetune:
            for param in self.node_proj.parameters():
                param.requires_grad = False
            for param in self.rel_proj.parameters():
                param.requires_grad = False
            for param in self.q_proj.parameters():
                param.requires_grad = False

    def forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        question_emb = batch.question_emb.to(device=device, dtype=torch.float32, non_blocking=True)
        node_embeddings = batch.node_embeddings.to(device=device, dtype=torch.float32, non_blocking=True)
        edge_embeddings = batch.edge_embeddings.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=False):
            question_tokens = self.q_proj(question_emb)
            node_tokens = self.node_proj(self.node_norm(node_embeddings))
            relation_tokens = self.rel_proj(self.rel_norm(edge_embeddings))
        return node_tokens, relation_tokens, question_tokens


class GraphStateEncoder(nn.Module):
    """GRU-based relation-history encoder."""

    def __init__(
        self,
        *,
        input_dim: int,
        state_dim: int,
        dropout: float = _DEFAULT_STATE_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.dropout = float(dropout)
        if self.input_dim <= 0 or self.state_dim <= 0:
            raise ValueError("input_dim/state_dim must be positive for GraphStateEncoder.")
        self.input_proj = nn.Identity() if self.input_dim == self.state_dim else nn.Linear(
            self.input_dim,
            self.state_dim,
            bias=False,
        )
        self.gru = nn.GRUCell(self.state_dim, self.state_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        _init_gru_cell(self.gru)
        if isinstance(self.input_proj, nn.Linear):
            _init_linear(self.input_proj)

    def init_state(self, num_graphs: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((num_graphs, self.state_dim), device=device, dtype=dtype)

    def update_state(
        self,
        state: torch.Tensor,
        *,
        relation_tokens: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> torch.Tensor:
        device_type = relation_tokens.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            state = state.to(dtype=torch.float32)
            relation_tokens = relation_tokens.to(dtype=torch.float32)
            relation_tokens = self.input_proj(relation_tokens)
            if self.dropout > 0.0:
                relation_tokens = self.dropout_layer(relation_tokens)
            updated = self.gru(relation_tokens, state)
        update_mask = update_mask.to(device=updated.device, dtype=torch.bool).view(-1, 1)
        return torch.where(update_mask, updated, state)


class StateRelationMLP(nn.Module):
    """MLP over [q ⊕ e_t ⊕ h_t] to produce relation logits."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bias_init: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim/hidden_dim/output_dim must be positive for StateRelationMLP.")
        self.norm = nn.LayerNorm(int(input_dim))
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(max(_ZERO, num_layers - _ONE)):
            linear = nn.Linear(in_dim, int(hidden_dim))
            _init_linear(linear)
            layers.append(linear)
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        final = nn.Linear(in_dim, int(output_dim))
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, float(bias_init))
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, state_inputs: torch.Tensor) -> torch.Tensor:
        if state_inputs.dim() != 2:
            raise ValueError("state_inputs must be [B,H] for StateRelationMLP.")
        device_type = state_inputs.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            state_inputs = state_inputs.to(dtype=torch.float32)
            return self.net(self.norm(state_inputs))


class EdgeEntityScorer(nn.Module):
    """Edge-level scorer for P(e|s,r) using state + relation + entity tokens."""

    def __init__(
        self,
        *,
        state_dim: int,
        token_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation_checkpointing: bool = _DEFAULT_EDGE_SCORE_ACTIVATION_CHECKPOINTING,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if state_dim <= 0 or token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim/token_dim/hidden_dim must be positive for EdgeEntityScorer.")
        self.state_norm = nn.LayerNorm(int(state_dim))
        layers: list[nn.Module] = []
        in_dim = int(state_dim)
        for _ in range(max(_ZERO, num_layers - _ONE)):
            linear = nn.Linear(in_dim, int(hidden_dim))
            _init_linear(linear)
            layers.append(linear)
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        final = nn.Linear(in_dim, int(hidden_dim))
        _init_linear(final)
        layers.append(final)
        self.state_mlp = nn.Sequential(*layers)
        self.rel_proj = nn.Linear(int(token_dim), int(hidden_dim), bias=False)
        self.node_proj = nn.Linear(int(token_dim), int(hidden_dim), bias=False)
        _init_linear(self.rel_proj)
        _init_linear(self.node_proj)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.activation_checkpointing = bool(activation_checkpointing)

    def _score_edges(
        self,
        state_inputs: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> torch.Tensor:
        state_ctx = self.state_mlp(self.state_norm(state_inputs))
        edge_state = state_ctx.index_select(0, edge_batch)
        edge_ctx = edge_state + self.rel_proj(relation_tokens)
        edge_ctx = self.dropout(edge_ctx)
        node_ctx = self.node_proj(node_tokens)
        return (edge_ctx * node_ctx).sum(dim=-1)

    def forward(
        self,
        *,
        state_inputs: torch.Tensor,
        relation_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> torch.Tensor:
        if state_inputs.dim() != 2 or relation_tokens.dim() != 2 or node_tokens.dim() != 2:
            raise ValueError("state_inputs/relation_tokens/node_tokens must be [*, H] for EdgeEntityScorer.")
        device_type = state_inputs.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            state_inputs = state_inputs.to(dtype=torch.float32)
            relation_tokens = relation_tokens.to(dtype=torch.float32)
            node_tokens = node_tokens.to(dtype=torch.float32)
            edge_batch = edge_batch.to(device=state_inputs.device, dtype=torch.long).view(-1)
            if relation_tokens.size(0) != edge_batch.numel() or node_tokens.size(0) != edge_batch.numel():
                raise ValueError("edge_batch length mismatch with relation_tokens/node_tokens.")
            if self.activation_checkpointing and self.training:
                return checkpoint(
                    self._score_edges,
                    state_inputs,
                    relation_tokens,
                    node_tokens,
                    edge_batch,
                    use_reentrant=False,
                )
            return self._score_edges(
                state_inputs,
                relation_tokens,
                node_tokens,
                edge_batch,
            )


class FlowPredictor(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        input_dim = self.hidden_dim + self.hidden_dim + self.feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, _FLOW_OUTPUT_DIM),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_linear(layer)

    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        graph_features: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if node_tokens.dim() != 2 or question_tokens.dim() != 2:
            raise ValueError("node_tokens and question_tokens must be [*, H] for FlowPredictor.")
        if graph_features.dim() != 2:
            raise ValueError("graph_features must be [B, F] for FlowPredictor.")
        if node_batch.dim() != 1:
            raise ValueError("node_batch must be [N] for FlowPredictor.")
        q_tokens = question_tokens.index_select(0, node_batch)
        g_tokens = graph_features.index_select(0, node_batch)
        context = torch.cat((q_tokens, node_tokens, g_tokens), dim=-1)
        return self.net(context).squeeze(-1)

    def set_output_bias(self, bias: float) -> None:
        last_linear = None
        for layer in reversed(self.net):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break
        if last_linear is None or last_linear.bias is None:
            raise RuntimeError("FlowPredictor missing output bias for initialization.")
        with torch.no_grad():
            last_linear.bias.fill_(float(bias))


class GFlowNetModule(LightningModule):
    """扁平 PyG batch 版本的 GFlowNet，移除 dense padding。"""

    def __init__(
        self,
        *,
        hidden_dim: int,
        policy: nn.Module,
        reward_fn: nn.Module,
        env: GraphEnv,
        emb_dim: int,
        vocabulary_path: str | Path,
        relation_vocab_size: Optional[int] = None,
        backbone_finetune: bool = _DEFAULT_BACKBONE_FINETUNE,
        edge_score_cfg: Optional[Mapping[str, Any]] = None,
        entity_score_cfg: Optional[Mapping[str, Any]] = None,
        state_cfg: Optional[Mapping[str, Any]] = None,
        flow_cfg: Optional[Mapping[str, Any]] = None,
        training_cfg: Mapping[str, Any],
        evaluation_cfg: Mapping[str, Any],
        actor_cfg: Optional[Mapping[str, Any]] = None,
        control_cfg: Optional[Mapping[str, Any]] = None,
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
            control_cfg=control_cfg,
            edge_score_cfg=edge_score_cfg,
            entity_score_cfg=entity_score_cfg,
            state_cfg=state_cfg,
            flow_cfg=flow_cfg,
        )
        self._validate_runtime_cfg()
        self._init_relation_vocab(vocabulary_path, relation_vocab_size)
        self._init_eval_settings()
        self._init_logging_settings()
        self._init_control_settings()
        self._init_grad_clip_settings()

        self.policy = policy
        self.reward_fn = reward_fn
        self.env = env
        self.max_steps = int(self.env.max_steps)
        self._init_backbone(emb_dim=emb_dim, finetune=backbone_finetune)
        (
            relation_state_input_dim,
            edge_state_input_dim,
            relation_use_active_nodes,
        ) = self._init_state_components()
        score_cfg = self._resolve_edge_score_cfg()
        entity_cfg = self._resolve_entity_score_cfg(score_cfg)
        self._init_relation_heads(relation_state_input_dim, score_cfg)
        self._init_edge_heads(edge_state_input_dim, entity_cfg)
        self._init_reward_schedule()
        self.flow_spec: FlowFeatureSpec = resolve_flow_spec(self.flow_cfg)
        self.log_f = FlowPredictor(self.hidden_dim, self.flow_spec.stats_dim)
        self._init_flow_bias_settings()
        self.actor = self._build_actor(
            relation_state_input_dim=relation_state_input_dim,
            relation_use_active_nodes=relation_use_active_nodes,
        )
        self._init_engine_components()
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
        control_cfg: Optional[Mapping[str, Any]],
        edge_score_cfg: Optional[Mapping[str, Any]],
        entity_score_cfg: Optional[Mapping[str, Any]],
        state_cfg: Optional[Mapping[str, Any]],
        flow_cfg: Optional[Mapping[str, Any]],
    ) -> None:
        self.training_cfg = self._require_mapping(training_cfg, "training_cfg")
        self.evaluation_cfg = self._require_mapping(evaluation_cfg, "evaluation_cfg")
        self.runtime_cfg = self._optional_mapping(runtime_cfg, "runtime_cfg")
        self.optimizer_cfg = self._optional_mapping(optimizer_cfg, "optimizer_cfg")
        self.scheduler_cfg = self._optional_mapping(scheduler_cfg, "scheduler_cfg")
        self.logging_cfg = self._optional_mapping(logging_cfg, "logging_cfg")
        self.actor_cfg = self._optional_mapping(actor_cfg, "actor_cfg")
        self.control_cfg = self._optional_mapping(control_cfg, "control_cfg")
        self.edge_score_cfg = self._optional_mapping(edge_score_cfg, "edge_score_cfg")
        self.entity_score_cfg = self._optional_mapping(entity_score_cfg, "entity_score_cfg")
        self.state_cfg = self._optional_mapping(state_cfg, "state_cfg")
        self.flow_cfg = self._optional_mapping(flow_cfg, "flow_cfg")
        self._validate_edge_batch = bool(self.runtime_cfg.get("validate_edge_batch", _DEFAULT_VALIDATE_EDGE_BATCH))
        self._validate_rollout_batch = bool(
            self.runtime_cfg.get("validate_rollout_batch", _DEFAULT_VALIDATE_ROLLOUT_BATCH)
        )
        self._require_precomputed_edge_batch = bool(
            self.runtime_cfg.get(
                "require_precomputed_edge_batch",
                _DEFAULT_REQUIRE_PRECOMPUTED_EDGE_BATCH,
            )
        )
        self._require_precomputed_node_in_degree = bool(
            self.runtime_cfg.get(
                "require_precomputed_node_in_degree",
                _DEFAULT_REQUIRE_PRECOMPUTED_NODE_IN_DEGREE,
            )
        )

    def _validate_runtime_cfg(self) -> None:
        if "vectorized_rollouts" in self.runtime_cfg:
            raise ValueError("runtime_cfg.vectorized_rollouts has been removed; use rollout_chunk_size.")

    def _init_relation_vocab(self, vocabulary_path: str | Path, relation_vocab_size: Optional[int]) -> None:
        if relation_vocab_size is None:
            relation_count = _load_relation_count(Path(vocabulary_path))
        else:
            relation_count = self._require_positive_int(relation_vocab_size, "relation_vocab_size")
        self.relation_vocab_size = int(relation_count)

    def _init_eval_settings(self) -> None:
        self._eval_rollout_prefixes, self._eval_rollouts = self._parse_eval_rollouts(self.evaluation_cfg)
        eval_temp_cfg = self.evaluation_cfg.get("rollout_temperature")
        if eval_temp_cfg is None:
            raise ValueError("evaluation_cfg.rollout_temperature must be set explicitly (no implicit eval temperature).")
        self._eval_rollout_temperature = float(eval_temp_cfg)
        if self._eval_rollout_temperature < 0.0:
            raise ValueError(f"evaluation_cfg.rollout_temperature must be >= 0, got {self._eval_rollout_temperature}.")
        self._context_debug_enabled = bool(self.evaluation_cfg.get("context_debug", False))
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

    def _init_control_settings(self) -> None:
        self._control_enabled = bool(self.control_cfg.get("enabled", _DEFAULT_CONTROL_ENABLED))
        if not self._control_enabled:
            self.register_buffer("lambda_success", torch.tensor(float(_ZERO)), persistent=True)
            return
        target_mode = str(self.control_cfg.get("target_mode", _DEFAULT_CONTROL_TARGET_MODE)).strip().lower()
        if target_mode not in {_CONTROL_TARGET_MODE_REACHABLE, _CONTROL_TARGET_MODE_FIXED}:
            raise ValueError(
                "control_cfg.target_mode must be "
                f"'{_CONTROL_TARGET_MODE_REACHABLE}' or '{_CONTROL_TARGET_MODE_FIXED}', "
                f"got {target_mode!r}."
            )
        target_min = self._require_non_negative_float(self.control_cfg.get("target_min"), "control_cfg.target_min")
        target_max = self._require_non_negative_float(self.control_cfg.get("target_max"), "control_cfg.target_max")
        if target_max < target_min:
            raise ValueError("control_cfg.target_max must be >= control_cfg.target_min.")
        temperature_min = self._require_positive_float(
            self.control_cfg.get("temperature_min"),
            "control_cfg.temperature_min",
        )
        temperature_max = self._require_positive_float(
            self.control_cfg.get("temperature_max"),
            "control_cfg.temperature_max",
        )
        temperature_base = self._require_positive_float(
            self.control_cfg.get("temperature_base"),
            "control_cfg.temperature_base",
        )
        if temperature_max < temperature_min:
            raise ValueError("control_cfg.temperature_max must be >= control_cfg.temperature_min.")
        if not (temperature_min <= temperature_base <= temperature_max):
            raise ValueError("control_cfg.temperature_base must lie in [temperature_min, temperature_max].")
        dual_lr = self._resolve_control_dual_lr()
        lambda_min = math.log(temperature_min / temperature_base)
        lambda_max = math.log(temperature_max / temperature_base)
        lambda_init = self._require_float(self.control_cfg.get("lambda_init", _ZERO), "control_cfg.lambda_init")
        if not (lambda_min <= lambda_init <= lambda_max):
            raise ValueError("control_cfg.lambda_init must be within [lambda_min, lambda_max].")
        self.register_buffer("lambda_success", torch.tensor(lambda_init, dtype=torch.float32), persistent=True)
        self._control_target_mode = target_mode
        self._control_target_min = float(target_min)
        self._control_target_max = float(target_max)
        self._control_temperature_min = float(temperature_min)
        self._control_temperature_max = float(temperature_max)
        self._control_temperature_base = float(temperature_base)
        self._control_dual_lr = float(dual_lr)
        self._control_lambda_min = float(lambda_min)
        self._control_lambda_max = float(lambda_max)

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

    def _get_param_name_map(self) -> Dict[int, str]:
        return self.training_loop.grad_clipper._get_param_name_map()

    def _init_backbone(self, *, emb_dim: int, finetune: bool) -> None:
        self.backbone = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            finetune=finetune,
        )

    def _init_state_components(self) -> tuple[int, int, bool]:
        state_dim = int(self.state_cfg.get("state_dim", self.hidden_dim))
        state_dropout = float(self.state_cfg.get("dropout", _DEFAULT_STATE_DROPOUT))
        relation_use_active_nodes = bool(
            self.state_cfg.get("relation_use_active_nodes", _DEFAULT_RELATION_USE_ACTIVE_NODES)
        )
        relation_state_input_dim = self._resolve_state_input_dim(
            hidden_dim=self.hidden_dim,
            state_dim=state_dim,
            include_active_nodes=relation_use_active_nodes,
        )
        edge_state_input_dim = self._resolve_state_input_dim(
            hidden_dim=self.hidden_dim,
            state_dim=state_dim,
            include_active_nodes=True,
        )
        self.state_encoder = GraphStateEncoder(
            input_dim=self.hidden_dim,
            state_dim=state_dim,
            dropout=state_dropout,
        )
        return relation_state_input_dim, edge_state_input_dim, relation_use_active_nodes

    def _resolve_edge_score_cfg(self) -> Dict[str, Any]:
        score_hidden_dim = int(self.edge_score_cfg.get("hidden_dim", self.hidden_dim))
        score_layers = self._require_positive_int(
            self.edge_score_cfg.get("num_layers", _DEFAULT_EDGE_SCORE_LAYERS),
            "edge_score_cfg.num_layers",
        )
        score_dropout = float(self.edge_score_cfg.get("dropout", _DEFAULT_EDGE_SCORE_DROPOUT))
        score_bias_init = float(self.edge_score_cfg.get("bias_init", _DEFAULT_EDGE_SCORE_BIAS_INIT))
        score_checkpointing = bool(
            self.edge_score_cfg.get(
                "activation_checkpointing",
                _DEFAULT_EDGE_SCORE_ACTIVATION_CHECKPOINTING,
            )
        )
        return {
            "hidden_dim": score_hidden_dim,
            "num_layers": score_layers,
            "dropout": score_dropout,
            "bias_init": score_bias_init,
            "activation_checkpointing": score_checkpointing,
        }

    def _resolve_entity_score_cfg(self, score_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not self.entity_score_cfg:
            self.entity_score_cfg = dict(self.edge_score_cfg)
        entity_hidden_dim = int(self.entity_score_cfg.get("hidden_dim", score_cfg["hidden_dim"]))
        entity_layers = self._require_positive_int(
            self.entity_score_cfg.get("num_layers", score_cfg["num_layers"]),
            "entity_score_cfg.num_layers",
        )
        entity_dropout = float(self.entity_score_cfg.get("dropout", score_cfg["dropout"]))
        entity_checkpointing = bool(
            self.entity_score_cfg.get("activation_checkpointing", score_cfg["activation_checkpointing"])
        )
        return {
            "hidden_dim": entity_hidden_dim,
            "num_layers": entity_layers,
            "dropout": entity_dropout,
            "activation_checkpointing": entity_checkpointing,
        }

    def _init_relation_heads(self, relation_state_input_dim: int, score_cfg: Dict[str, Any]) -> None:
        output_dim = self.relation_vocab_size
        self.forward_head = StateRelationMLP(
            input_dim=relation_state_input_dim,
            hidden_dim=score_cfg["hidden_dim"],
            num_layers=score_cfg["num_layers"],
            dropout=score_cfg["dropout"],
            bias_init=score_cfg["bias_init"],
            output_dim=output_dim,
        )
        self.backward_head = StateRelationMLP(
            input_dim=relation_state_input_dim,
            hidden_dim=score_cfg["hidden_dim"],
            num_layers=score_cfg["num_layers"],
            dropout=score_cfg["dropout"],
            bias_init=score_cfg["bias_init"],
            output_dim=output_dim,
        )

    def _init_edge_heads(self, edge_state_input_dim: int, entity_cfg: Dict[str, Any]) -> None:
        token_dim = self.hidden_dim
        self.edge_forward_head = EdgeEntityScorer(
            state_dim=edge_state_input_dim,
            token_dim=token_dim,
            hidden_dim=entity_cfg["hidden_dim"],
            num_layers=entity_cfg["num_layers"],
            dropout=entity_cfg["dropout"],
            activation_checkpointing=entity_cfg["activation_checkpointing"],
        )
        self.edge_backward_head = EdgeEntityScorer(
            state_dim=edge_state_input_dim,
            token_dim=token_dim,
            hidden_dim=entity_cfg["hidden_dim"],
            num_layers=entity_cfg["num_layers"],
            dropout=entity_cfg["dropout"],
            activation_checkpointing=entity_cfg["activation_checkpointing"],
        )

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

    def _init_flow_bias_settings(self) -> None:
        bias_raw = self.flow_cfg.get("bias_init", _FLOW_BIAS_INIT_NONE)
        bias_init = str(bias_raw or _FLOW_BIAS_INIT_NONE).strip().lower()
        if bias_init not in _FLOW_BIAS_INIT_OPTIONS:
            raise ValueError(f"flow_cfg.bias_init must be one of {sorted(_FLOW_BIAS_INIT_OPTIONS)}.")
        self._flow_bias_init = bias_init
        self._flow_bias_initialized = False
        if bias_init == _FLOW_BIAS_INIT_VALUE:
            raw_value = self.flow_cfg.get(_FLOW_BIAS_INIT_VALUE_KEY)
            if raw_value is None:
                raise ValueError(f"flow_cfg.{_FLOW_BIAS_INIT_VALUE_KEY} must be set when bias_init='value'.")
            self._flow_bias_value = self._require_float(raw_value, f"flow_cfg.{_FLOW_BIAS_INIT_VALUE_KEY}")
        else:
            self._flow_bias_value = None

    def _build_actor(
        self,
        *,
        relation_state_input_dim: int,
        relation_use_active_nodes: bool,
    ) -> GFlowNetActor:
        policy_temperature = float(self.actor_cfg.get("policy_temperature", _DEFAULT_POLICY_TEMPERATURE))
        backward_temperature = self.actor_cfg.get("backward_temperature")
        stop_bias_init = self.actor_cfg.get("stop_bias_init")
        if stop_bias_init is None:
            stop_bias = None
        else:
            stop_bias = self._require_float(stop_bias_init, "actor_cfg.stop_bias_init")
        cosine_bias_alpha = float(self.actor_cfg.get("cosine_bias_alpha", _DEFAULT_COSINE_BIAS_ALPHA))
        cosine_relation_bias_alpha = float(
            self.actor_cfg.get("cosine_relation_bias_alpha", _DEFAULT_COSINE_RELATION_BIAS_ALPHA)
        )
        check_finite = bool(self.actor_cfg.get("check_finite", _DEFAULT_CHECK_FINITE))
        return GFlowNetActor(
            policy=self.policy,
            env=self.env,
            forward_head=self.forward_head,
            backward_head=self.backward_head,
            edge_forward_head=self.edge_forward_head,
            edge_backward_head=self.edge_backward_head,
            state_encoder=self.state_encoder,
            state_input_dim=relation_state_input_dim,
            max_steps=self.max_steps,
            hidden_dim=self.hidden_dim,
            policy_temperature=policy_temperature,
            backward_temperature=backward_temperature,
            stop_bias_init=stop_bias,
            cosine_bias_alpha=cosine_bias_alpha,
            cosine_relation_bias_alpha=cosine_relation_bias_alpha,
            relation_use_active_nodes=relation_use_active_nodes,
            check_finite=check_finite,
        )

    def _init_engine_components(self) -> None:
        self.input_validator = GFlowNetInputValidator(
            validate_edge_batch=self._validate_edge_batch,
            validate_rollout_batch=self._validate_rollout_batch,
        )
        self.batch_processor = GFlowNetBatchProcessor(
            backbone=self.backbone,
            require_precomputed_edge_batch=self._require_precomputed_edge_batch,
            require_precomputed_node_in_degree=self._require_precomputed_node_in_degree,
            corridor_cfg=self.training_cfg.get("corridor"),
        )
        self.engine = GFlowNetEngine(
            actor=self.actor,
            reward_fn=self.reward_fn,
            env=self.env,
            log_f=self.log_f,
            flow_spec=self.flow_spec,
            batch_processor=self.batch_processor,
            input_validator=self.input_validator,
            context_debug_enabled=self._context_debug_enabled,
            composite_score_cfg=self._composite_score_cfg,
            dual_stream_cfg=self.training_cfg.get("dual_stream"),
            distance_prior_cfg=self.training_cfg.get("distance_prior"),
        )

    def _init_metric_stores(self) -> None:
        if not hasattr(self, "training_loop"):
            self.training_loop = GFlowNetTrainingLoop(self)
        self.training_loop.init_metric_stores()

    def _assert_training_cfg_contract(self) -> None:
        if self.training_cfg.get("safety_net") is not None:
            raise ValueError("training_cfg.safety_net has been removed; no implicit shortcuts are allowed.")
        if self.training_cfg.get("sp_dropout") is not None:
            raise ValueError("training_cfg.sp_dropout has been removed; do not configure SP-dropout.")
        if self.training_cfg.get("replay") is not None:
            raise ValueError("training_cfg.replay has been removed; do not configure replay trajectories.")
        if self.training_cfg.get("start_backtrack") is not None:
            raise ValueError("training_cfg.start_backtrack has been removed; use training_cfg.dual_stream.")
        if self.training_cfg.get("backward_warmup") is not None:
            raise ValueError("training_cfg.backward_warmup has been removed; use training_cfg.dual_stream.")

    def _save_serializable_hparams(self) -> None:
        # 仅保存可序列化的标量，避免将配置映射写入 checkpoint。
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "policy",
                "backbone",
                "forward_head",
                "backward_head",
                "edge_forward_head",
                "edge_backward_head",
                "state_encoder",
                "log_f",
                "reward_fn",
                "env",
                "actor_cfg",
                "training_cfg",
                "evaluation_cfg",
                "flow_cfg",
                "optimizer_cfg",
                "scheduler_cfg",
                "logging_cfg",
                "control_cfg",
            ],
        )

    def configure_optimizers(self):
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

    def forward(self, batch: Any) -> torch.Tensor:
        device = self.device
        self._validate_batch_inputs(batch, is_training=self.training, require_rollout=False)
        inputs = self.batch_processor.prepare_full_rollout_inputs(batch, device)
        graph_cache = self.batch_processor.build_graph_cache(inputs, device=device)
        state_vec = self._init_state_vec(inputs)
        return self.actor._score_edges_forward(
            graph_cache,
            state_vec,
            active_nodes=graph_cache["node_is_start"],
            autocast_ctx=self.actor._autocast_context(device),
        )

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

    def _resolve_control_dual_lr(self) -> float:
        raw = self.control_cfg.get("dual_lr")
        if raw is None:
            num_rollouts = self._require_positive_int(
                self.training_cfg.get("num_train_rollouts"),
                "training_cfg.num_train_rollouts",
            )
            return float(_ONE) / float(num_rollouts)
        return self._require_positive_float(raw, "control_cfg.dual_lr")

    @staticmethod
    def _resolve_state_input_dim(*, hidden_dim: int, state_dim: int, include_active_nodes: bool) -> int:
        base = int(hidden_dim) + int(state_dim)
        if include_active_nodes:
            base += int(hidden_dim)
        return base

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

    def _resolve_eval_temperature_extras(self) -> list[float]:
        return self.training_loop._resolve_eval_temperature_extras()

    def _format_temperature_suffix(self, temperature: float) -> str:
        return self.training_loop._format_temperature_suffix(temperature)

    def _suffix_metrics(self, metrics: Dict[str, torch.Tensor], *, suffix: str) -> Dict[str, torch.Tensor]:
        return self.training_loop._suffix_metrics(metrics, suffix=suffix)

    def _extract_scalar_metric(
        self,
        metrics: Dict[str, torch.Tensor],
        key: str,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        return self.training_loop._extract_scalar_metric(metrics, key, device=device)

    def _resolve_control_target(
        self,
        metrics: Dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        return self.training_loop._resolve_control_target(metrics, device=device, dtype=dtype)

    def _temperature_from_lambda(self, lambda_value: torch.Tensor) -> torch.Tensor:
        return self.training_loop._temperature_from_lambda(lambda_value)

    def _apply_control_temperature(self) -> None:
        self.training_loop._apply_control_temperature()

    def _update_success_controller(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.training_loop._update_success_controller(metrics)

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

    def _accumulate_grad_batches(self) -> int:
        return self.training_loop._accumulate_grad_batches()

    def _estimate_optimizer_steps_per_epoch(self) -> Optional[int]:
        return self.training_loop.grad_clipper._estimate_optimizer_steps_per_epoch()

    def _resolve_grad_clip_ema_beta(self) -> float:
        return self.training_loop.grad_clipper._resolve_grad_clip_ema_beta()

    def _resolve_grad_clip_tail_prob(self) -> float:
        return self.training_loop.grad_clipper._resolve_grad_clip_tail_prob()

    def _is_last_train_batch(self, batch_idx: int) -> bool:
        return self.training_loop._is_last_train_batch(batch_idx)

    def _should_zero_grad(self, batch_idx: int) -> bool:
        return self.training_loop._should_zero_grad(batch_idx)

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        return self.training_loop._should_step_optimizer(batch_idx)

    def _normal_icdf(self, prob: torch.Tensor) -> torch.Tensor:
        return self.training_loop.grad_clipper._normal_icdf(prob)

    def _grad_log_norm_stats(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        return self.training_loop.grad_clipper._grad_log_norm_stats(device=device, dtype=dtype)

    def _update_grad_log_norm_ema(self, log_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.training_loop.grad_clipper._update_grad_log_norm_ema(log_norm)

    def _resolve_adaptive_clip_val(
        self,
        *,
        fallback: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        return self.training_loop.grad_clipper._resolve_adaptive_clip_val(fallback, device=device, dtype=dtype)

    def _compute_grad_norm(
        self,
        params: Sequence[torch.nn.Parameter],
        *,
        norm_type: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.training_loop.grad_clipper._compute_grad_norm(
            params,
            norm_type=norm_type,
            device=device,
            dtype=dtype,
        )

    def _collect_grad_params(
        self,
        optimizer_ref: torch.optim.Optimizer,
    ) -> tuple[list[torch.nn.Parameter], Optional[torch.device], Optional[torch.dtype]]:
        return self.training_loop.grad_clipper._collect_grad_params(optimizer_ref)

    def _assert_finite_grads(self, params: Sequence[torch.nn.Parameter]) -> None:
        self.training_loop.grad_clipper._assert_finite_grads(params)

    def _summarize_grad(self, name: str, grad: torch.Tensor) -> str:
        return self.training_loop.grad_clipper._summarize_grad(name, grad)

    def _build_grad_metrics(
        self,
        *,
        pre_norm: torch.Tensor,
        post_norm: torch.Tensor,
        clip_val: float,
        clip_coef: float,
        flow_pre: torch.Tensor,
        flow_post: torch.Tensor,
        actor_pre: torch.Tensor,
        actor_post: torch.Tensor,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        return self.training_loop.grad_clipper._build_grad_metrics(
            pre_norm=pre_norm,
            post_norm=post_norm,
            clip_val=clip_val,
            clip_coef=clip_coef,
            flow_pre=flow_pre,
            flow_post=flow_post,
            actor_pre=actor_pre,
            actor_post=actor_post,
            log_mean=log_mean,
            log_std=log_std,
            device=device,
            dtype=dtype,
        )

    def _collect_module_grad_params(self, module: nn.Module) -> list[torch.nn.Parameter]:
        return self.training_loop.grad_clipper._collect_module_grad_params(module)

    def _compute_module_grad_norm(
        self,
        module: nn.Module,
        *,
        norm_type: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.training_loop.grad_clipper._compute_module_grad_norm(
            module,
            norm_type=norm_type,
            device=device,
            dtype=dtype,
        )

    def _clip_gradients_if_needed(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        return self.training_loop.grad_clipper.clip_gradients(optimizer)

    def _step_scheduler(self) -> None:
        self.training_loop._step_scheduler()

    def _step_optimizer(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        return self.training_loop._step_optimizer(optimizer)

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.training_loop.on_before_optimizer_step(optimizer, optimizer_idx)

    def training_step(self, batch, batch_idx: int):
        return self.training_loop.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx: int):
        self.training_loop.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx: int):
        self.training_loop.test_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self.training_loop.predict_step(batch, batch_idx, dataloader_idx)

    def on_train_epoch_start(self) -> None:
        self.training_loop.on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        self.training_loop.on_train_epoch_end()

    def _resolve_min_log_reward(self) -> float:
        return self.training_loop._resolve_min_log_reward()

    @torch.no_grad()
    def _estimate_log_reward_bias(self, batch: Any) -> float:
        return self.training_loop._estimate_log_reward_bias(batch)

    def _get_metric_store(self, prefix: str) -> MetricCollection:
        return self.training_loop.metric_logger._get_metric_store(prefix)

    def _get_or_create_metric(self, store: MetricCollection, name: str) -> MeanMetric:
        return self.training_loop.metric_logger._get_or_create_metric(store, name)

    def _update_metrics(self, metrics: Dict[str, torch.Tensor], *, prefix: str, batch_size: int) -> None:
        self.training_loop.metric_logger.update_metrics(metrics, prefix=prefix, batch_size=batch_size)

    def _log_metric_store(self, *, prefix: str, batch_size: int) -> None:
        self.training_loop.metric_logger.log_metric_store(prefix=prefix, batch_size=batch_size)

    def _compute_rollout_records(
        self,
        *,
        batch: Any,
        batch_idx: int | None = None,
    ) -> list[Dict[str, Any]]:
        return self.training_loop._compute_rollout_records(batch=batch, batch_idx=batch_idx)

    def sample_edge_targets(
        self,
        batch: Any,
        *,
        num_rollouts: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rollout_cfg = self._build_rollout_cfg(is_training=False)
        rollout_count = num_rollouts if num_rollouts is not None else rollout_cfg.num_rollouts
        rollout_count = self._require_positive_int(rollout_count, "num_rollouts")
        rollout_temperature = rollout_cfg.eval_rollout_temperature if temperature is None else float(temperature)
        return self.engine.sample_edge_targets(
            batch=batch,
            device=self.device,
            num_rollouts=rollout_count,
            temperature=rollout_temperature,
        )

    def _compute_batch_loss(
        self,
        batch: Any,
        batch_idx: int | None = None,
        *,
        rollout_cfg: GFlowNetRolloutConfig | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.training_loop._compute_batch_loss(batch, batch_idx=batch_idx, rollout_cfg=rollout_cfg)

    def _init_state_vec(
        self,
        inputs: RolloutInputs,
    ) -> torch.Tensor:
        num_graphs = int(inputs.node_ptr.numel() - 1)
        return self.state_encoder.init_state(
            num_graphs,
            device=inputs.node_tokens.device,
            dtype=inputs.node_tokens.dtype,
        )

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
        if hasattr(self, "engine"):
            self.engine.set_composite_score_cfg(self._composite_score_cfg)

    def _apply_potential_weight_schedule(self) -> None:
        self.training_loop._apply_potential_weight_schedule()

    def _resolve_potential_anneal_epochs(self) -> int:
        return self.training_loop._resolve_potential_anneal_epochs()


__all__ = ["GFlowNetModule"]
