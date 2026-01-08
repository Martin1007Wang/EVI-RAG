from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Dict, Optional

import lmdb
import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric, Metric, MetricCollection

from src.models.components import (
    GFlowNetActor,
    GraphEnv,
)
from src.utils import log_metric, setup_optimizer
from src.utils.gfn import GFlowNetBatchProcessor, GFlowNetInputValidator, RolloutInputs
from src.utils.gfn_engine import GFlowNetEngine, GFlowNetRolloutConfig

_ZERO = 0
_ONE = 1
_DEFAULT_POLICY_TEMPERATURE = 1.0
_DEFAULT_CHECK_FINITE = False
_DEFAULT_VALIDATE_EDGE_BATCH = True
_DEFAULT_VECTORIZED_ROLLOUTS = True
_DEFAULT_LOG_ON_STEP_TRAIN = False
_DEFAULT_BACKBONE_FINETUNE = True
_DEFAULT_EDGE_SCORE_LAYERS = 2
_DEFAULT_EDGE_SCORE_DROPOUT = 0.1
_DEFAULT_EDGE_SCORE_BIAS_INIT = 0.0
_DEFAULT_STATE_DROPOUT = 0.0
_LOGZ_OUTPUT_DIM = 1
_LOGZ_STATS_DIM = 3
_EMBED_INIT_STD_POWER = 0.25
_DEFAULT_REWARD_EMBEDDING_SOURCE = "raw"
_ALLOWED_REWARD_EMBEDDING_SOURCES = {"raw", "backbone"}
_GRU_NUM_GATES = 3
_GRU_ORTHO_GAIN = 1.0
_VOCAB_MAX_READERS = 1
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
        return self.net(self.norm(state_inputs))


class LogZPredictor(nn.Module):
    def __init__(self, hidden_dim: int, stats_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.stats_dim = int(stats_dim)
        input_dim = self.hidden_dim + self.hidden_dim + self.stats_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, _LOGZ_OUTPUT_DIM),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_linear(layer)

    def forward(
        self,
        *,
        question_tokens: torch.Tensor,
        start_tokens: torch.Tensor,
        graph_stats: torch.Tensor,
    ) -> torch.Tensor:
        if question_tokens.shape != start_tokens.shape:
            raise ValueError("question_tokens and start_tokens must have the same shape for LogZ.")
        if question_tokens.size(0) != graph_stats.size(0):
            raise ValueError("graph_stats batch size must match question_tokens for LogZ.")
        context = torch.cat((question_tokens, start_tokens, graph_stats), dim=-1)
        return self.net(context).squeeze(-1)


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
        backbone_finetune: bool = _DEFAULT_BACKBONE_FINETUNE,
        edge_score_cfg: Optional[Mapping[str, Any]] = None,
        state_cfg: Optional[Mapping[str, Any]] = None,
        training_cfg: Mapping[str, Any],
        evaluation_cfg: Mapping[str, Any],
        reward_cfg: Optional[Mapping[str, Any]] = None,
        actor_cfg: Optional[Mapping[str, Any]] = None,
        runtime_cfg: Optional[Mapping[str, Any]] = None,
        optimizer_cfg: Optional[Mapping[str, Any]] = None,
        scheduler_cfg: Optional[Mapping[str, Any]] = None,
        logging_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.training_cfg = self._require_mapping(training_cfg, "training_cfg")
        self.evaluation_cfg = self._require_mapping(evaluation_cfg, "evaluation_cfg")
        self.reward_cfg = self._optional_mapping(reward_cfg, "reward_cfg")
        self.runtime_cfg = self._optional_mapping(runtime_cfg, "runtime_cfg")
        self.optimizer_cfg = self._optional_mapping(optimizer_cfg, "optimizer_cfg")
        self.scheduler_cfg = self._optional_mapping(scheduler_cfg, "scheduler_cfg")
        self.logging_cfg = self._optional_mapping(logging_cfg, "logging_cfg")
        self.actor_cfg = self._optional_mapping(actor_cfg, "actor_cfg")
        self.edge_score_cfg = self._optional_mapping(edge_score_cfg, "edge_score_cfg")
        self.state_cfg = self._optional_mapping(state_cfg, "state_cfg")
        self._validate_edge_batch = bool(self.runtime_cfg.get("validate_edge_batch", _DEFAULT_VALIDATE_EDGE_BATCH))
        self._vectorized_rollouts = bool(self.runtime_cfg.get("vectorized_rollouts", _DEFAULT_VECTORIZED_ROLLOUTS))

        reward_source = str(
            self.reward_cfg.get("embedding_source", _DEFAULT_REWARD_EMBEDDING_SOURCE)
        ).strip().lower()
        if reward_source not in _ALLOWED_REWARD_EMBEDDING_SOURCES:
            raise ValueError(
                "reward_cfg.embedding_source must be one of "
                f"{sorted(_ALLOWED_REWARD_EMBEDDING_SOURCES)}, got {reward_source!r}."
            )
        if reward_source == "backbone" and bool(backbone_finetune):
            raise ValueError(
                "reward_cfg.embedding_source='backbone' requires backbone_finetune=false "
                "to keep the reward distribution fixed."
            )
        self.reward_embedding_source = reward_source

        relation_count = _load_relation_count(Path(vocabulary_path))
        self.relation_vocab_size = int(relation_count)

        self._eval_rollout_prefixes, self._eval_rollouts = self._parse_eval_rollouts(self.evaluation_cfg)
        eval_temp_cfg = self.evaluation_cfg.get("rollout_temperature")
        if eval_temp_cfg is None:
            raise ValueError("evaluation_cfg.rollout_temperature must be set explicitly (no implicit eval temperature).")
        self._eval_rollout_temperature = float(eval_temp_cfg)
        if self._eval_rollout_temperature < 0.0:
            raise ValueError(f"evaluation_cfg.rollout_temperature must be >= 0, got {self._eval_rollout_temperature}.")
        self._train_prog_bar = self._coerce_str_set(self.logging_cfg.get("train_prog_bar"), "logging_cfg.train_prog_bar")
        self._eval_prog_bar = self._coerce_str_set(self.logging_cfg.get("eval_prog_bar"), "logging_cfg.eval_prog_bar")
        self._log_on_step_train = bool(self.logging_cfg.get("log_on_step_train", _DEFAULT_LOG_ON_STEP_TRAIN))

        self.policy = policy
        self.backbone = EmbeddingBackbone(
            emb_dim=emb_dim,
            hidden_dim=self.hidden_dim,
            finetune=backbone_finetune,
        )
        state_dim = int(self.state_cfg.get("state_dim", self.hidden_dim))
        state_dropout = float(self.state_cfg.get("dropout", _DEFAULT_STATE_DROPOUT))
        self.state_encoder = GraphStateEncoder(
            input_dim=self.hidden_dim,
            state_dim=state_dim,
            dropout=state_dropout,
        )
        score_hidden_dim = int(self.edge_score_cfg.get("hidden_dim", self.hidden_dim))
        score_layers = self._require_positive_int(
            self.edge_score_cfg.get("num_layers", _DEFAULT_EDGE_SCORE_LAYERS),
            "edge_score_cfg.num_layers",
        )
        score_dropout = float(self.edge_score_cfg.get("dropout", _DEFAULT_EDGE_SCORE_DROPOUT))
        score_bias_init = float(self.edge_score_cfg.get("bias_init", _DEFAULT_EDGE_SCORE_BIAS_INIT))
        state_input_dim = (self.hidden_dim + self.hidden_dim + state_dim)
        output_dim = self.relation_vocab_size
        self.forward_head = StateRelationMLP(
            input_dim=state_input_dim,
            hidden_dim=score_hidden_dim,
            num_layers=score_layers,
            dropout=score_dropout,
            bias_init=score_bias_init,
            output_dim=output_dim,
        )
        self.backward_head = StateRelationMLP(
            input_dim=state_input_dim,
            hidden_dim=score_hidden_dim,
            num_layers=score_layers,
            dropout=score_dropout,
            bias_init=score_bias_init,
            output_dim=output_dim,
        )
        self.log_z = LogZPredictor(self.hidden_dim, _LOGZ_STATS_DIM)
        self.reward_fn = reward_fn
        self.env = env
        self.max_steps = int(self.env.max_steps)
        self._potential_weight_init = float(getattr(self.reward_fn, "potential_weight", _ZERO))
        self._potential_weight_end = float(
            self.reward_cfg.get("potential_weight_end", self._potential_weight_init)
        )
        self._potential_weight_decay_epochs = int(
            self.reward_cfg.get("potential_weight_decay_epochs", _ZERO)
        )
        if self._potential_weight_decay_epochs < _ZERO:
            raise ValueError("reward_cfg.potential_weight_decay_epochs must be >= 0.")
        policy_temperature = float(self.actor_cfg.get("policy_temperature", _DEFAULT_POLICY_TEMPERATURE))
        backward_temperature = self.actor_cfg.get("backward_temperature")
        check_finite = bool(self.actor_cfg.get("check_finite", _DEFAULT_CHECK_FINITE))
        self.actor = GFlowNetActor(
            policy=self.policy,
            env=self.env,
            forward_head=self.forward_head,
            backward_head=self.backward_head,
            state_encoder=self.state_encoder,
            state_input_dim=state_input_dim,
            max_steps=self.max_steps,
            hidden_dim=self.hidden_dim,
            policy_temperature=policy_temperature,
            backward_temperature=backward_temperature,
            check_finite=check_finite,
        )
        self.input_validator = GFlowNetInputValidator(validate_edge_batch=self._validate_edge_batch)
        self.batch_processor = GFlowNetBatchProcessor(backbone=self.backbone)
        self.engine = GFlowNetEngine(
            actor=self.actor,
            reward_fn=self.reward_fn,
            env=self.env,
            log_z=self.log_z,
            batch_processor=self.batch_processor,
            input_validator=self.input_validator,
            reward_embedding_source=self.reward_embedding_source,
            vectorized_rollouts=self._vectorized_rollouts,
        )
        self.train_metrics = MetricCollection({})
        self.val_metrics = MetricCollection({})
        self.test_metrics = MetricCollection({})
        if self.training_cfg.get("safety_net") is not None:
            raise ValueError("training_cfg.safety_net has been removed; no implicit shortcuts are allowed.")
        if self.training_cfg.get("sp_dropout") is not None:
            raise ValueError("training_cfg.sp_dropout has been removed; do not configure SP-dropout.")
        if self.training_cfg.get("replay") is not None:
            raise ValueError("training_cfg.replay has been removed; do not configure replay trajectories.")
        # 仅保存可序列化的标量，避免将配置映射写入 checkpoint。
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "policy",
                "backbone",
                "forward_head",
                "backward_head",
                "state_encoder",
                "log_z",
                "reward_fn",
                "env",
                "actor_cfg",
                "training_cfg",
                "evaluation_cfg",
                "reward_cfg",
                "optimizer_cfg",
                "scheduler_cfg",
                "logging_cfg",
            ],
        )

    def configure_optimizers(self):
        optimizer = setup_optimizer(self, self.optimizer_cfg)
        scheduler_type = str(self.scheduler_cfg.get("type", "") or "").lower()
        if not scheduler_type:
            return optimizer
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", 10)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.scheduler_cfg.get("interval", "epoch"),
                    "monitor": self.scheduler_cfg.get("monitor", "val/loss"),
                },
            }
        if scheduler_type in {"cosine_restart", "cosine_warm_restarts", "cosine_restarts"}:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.scheduler_cfg.get("t_0", 10)),
                T_mult=int(self.scheduler_cfg.get("t_mult", 1)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.scheduler_cfg.get("interval", "epoch"),
                    "monitor": self.scheduler_cfg.get("monitor", "val/loss"),
                },
            }
        return optimizer

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
        return GFlowNetRolloutConfig(
            num_rollouts=num_rollouts,
            eval_rollout_prefixes=self._eval_rollout_prefixes,
            eval_rollout_temperature=self._eval_rollout_temperature,
            vectorized_rollouts=self._vectorized_rollouts,
            is_training=is_training,
        )

    def _resolve_num_rollouts(self, is_training: bool) -> int:
        if is_training:
            return self._require_positive_int(
                self.training_cfg.get("num_train_rollouts"),
                "training_cfg.num_train_rollouts",
            )
        return self._require_positive_int(self._eval_rollouts, "evaluation_cfg.num_eval_rollouts")

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        # Move only hot-path tensors to avoid repeated H2D copies while keeping large ID tensors on CPU.
        if device.type == "cpu":
            return batch
        for key in _BATCH_DEVICE_KEYS:
            value = getattr(batch, key, None)
            if torch.is_tensor(value):
                setattr(batch, key, value.to(device=device, non_blocking=True))
        return batch

    def training_step(self, batch, batch_idx: int):
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        batch_size = int(batch.num_graphs)
        self._update_metrics(metrics, prefix="train", batch_size=batch_size)
        self._log_metric_store(prefix="train", batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        batch_size = int(batch.num_graphs)
        self._update_metrics(metrics, prefix="val", batch_size=batch_size)
        self._log_metric_store(prefix="val", batch_size=batch_size)

    def test_step(self, batch, batch_idx: int):
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        batch_size = int(batch.num_graphs)
        self._update_metrics(metrics, prefix="test", batch_size=batch_size)
        self._log_metric_store(prefix="test", batch_size=batch_size)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self._compute_rollout_records(batch=batch, batch_idx=batch_idx)

    def on_train_epoch_start(self) -> None:
        self._apply_potential_weight_schedule()

    def _get_metric_store(self, prefix: str) -> MetricCollection:
        if prefix == "train":
            return self.train_metrics
        if prefix == "val":
            return self.val_metrics
        if prefix == "test":
            return self.test_metrics
        raise ValueError(f"Unknown metric prefix: {prefix}")

    def _get_or_create_metric(self, store: MetricCollection, name: str) -> MeanMetric:
        if name in store:
            return store[name]  # type: ignore[return-value]
        metric = MeanMetric().to(self.device)
        if hasattr(store, "add_metrics"):
            store.add_metrics({name: metric})
        else:
            store[name] = metric
        return metric

    def _update_metrics(self, metrics: Dict[str, torch.Tensor], *, prefix: str, batch_size: int) -> None:
        store = self._get_metric_store(prefix)
        for name, value in metrics.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            if not torch.is_floating_point(value):
                value = value.float()
            value = value.detach().to(device=self.device)
            metric = self._get_or_create_metric(store, name)
            if value.numel() == 1:
                metric.update(value, weight=batch_size)
            else:
                metric.update(value)

    def _log_metric_store(self, *, prefix: str, batch_size: int) -> None:
        if prefix == "predict":
            return
        store = self._get_metric_store(prefix)
        sync_dist = bool(self.trainer and getattr(self.trainer, "num_devices", 1) > 1)
        is_train = prefix == "train"
        prog_bar_set = set(self._train_prog_bar if is_train else self._eval_prog_bar)
        log_on_step = self._log_on_step_train if is_train else False
        for name, metric in store.items():
            prog_bar = name in prog_bar_set or (is_train and name == "loss")
            metric_attribute = f"{prefix}_metrics.{name}" if isinstance(metric, Metric) else None
            log_metric(
                self,
                f"{prefix}/{name}",
                metric,
                sync_dist=sync_dist,
                prog_bar=prog_bar,
                on_step=log_on_step,
                on_epoch=True,
                batch_size=batch_size,
                metric_attribute=metric_attribute,
            )

    def _compute_rollout_records(
        self,
        *,
        batch: Any,
        batch_idx: int | None = None,
    ) -> list[Dict[str, Any]]:
        rollout_cfg = self._build_rollout_cfg(is_training=False)
        return self.engine.compute_rollout_records(
            batch=batch,
            device=self.device,
            rollout_cfg=rollout_cfg,
        )

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

    def _compute_batch_loss(self, batch: Any, batch_idx: int | None = None) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rollout_cfg = self._build_rollout_cfg(is_training=self.training)
        return self.engine.compute_batch_loss(
            batch=batch,
            device=self.device,
            rollout_cfg=rollout_cfg,
        )

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

    def _apply_potential_weight_schedule(self) -> None:
        if self._potential_weight_decay_epochs <= _ZERO:
            return
        if not hasattr(self.reward_fn, "potential_weight"):
            return
        current_epoch = int(self.current_epoch)
        ratio = float(current_epoch) / float(self._potential_weight_decay_epochs)
        ratio = min(ratio, float(_ONE))
        weight = self._potential_weight_init + (
            (self._potential_weight_end - self._potential_weight_init) * ratio
        )
        self.reward_fn.potential_weight = float(weight)


__all__ = ["GFlowNetModule"]
