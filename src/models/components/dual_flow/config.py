from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from src.models.components import QCBiANetwork

from .constants import (
    _DB_CFG_KEYS,
    _DEFAULT_START_TEMPERATURE_END,
    _DEFAULT_START_TEMPERATURE_START,
    _DEFAULT_ANSWER_POOLING,
    _ANSWER_POOLINGS,
    _DEFAULT_DIVERSE_BEAM_ENABLED,
    _DEFAULT_DIVERSE_BEAM_GROUPS,
    _DEFAULT_DIVERSE_BEAM_LAMBDA,
    _DEFAULT_DIVERSE_BEAM_PENALTY,
    _DEFAULT_DIVERSE_BEAM_SIMILARITY,
    _DEFAULT_EDGE_DROPOUT,
    _DEFAULT_EDGE_INTER_DIM,
    _DEFAULT_INVERSE_REL_SUFFIX,
    _DEFAULT_STRICT_INVERSE,
    _DEFAULT_TRAIN_ROLLOUTS,
    _DIVERSE_BEAM_PENALTIES,
    _DIVERSE_BEAM_SIMILARITIES,
    _NEG_ONE,
    _ONE,
    _PB_MODES,
    _PB_MODE_TOPO_SEMANTIC,
    _PB_MODE_UNIFORM,
    _TWO,
    _ZERO,
)


class DualFlowConfigMixin:
    def _validate_cfg_contract(self) -> None:
        allowed_training = {
            "accumulate_grad_batches",
            "db_cfg",
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
        use_spherical = bool(raw.get("use_spherical", QCBiANetwork.DEFAULT_USE_SPHERICAL))
        temperature_init = float(raw.get("temperature_init", QCBiANetwork.DEFAULT_TEMPERATURE_INIT))
        norm_eps = float(raw.get("norm_eps", QCBiANetwork.DEFAULT_NORM_EPS))
        logit_scale_min = float(raw.get("logit_scale_min", QCBiANetwork.DEFAULT_LOGIT_SCALE_MIN))
        logit_scale_max = float(raw.get("logit_scale_max", QCBiANetwork.DEFAULT_LOGIT_SCALE_MAX))
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
            "sampling_temperature_start": float(raw["sampling_temperature_start"]),
            "sampling_temperature_end": float(raw["sampling_temperature_end"]),
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
        if float(cfg["sampling_temperature_start"]) <= float(_ZERO) or float(cfg["sampling_temperature_end"]) <= float(
            _ZERO
        ):
            raise ValueError("db_cfg.sampling_temperature_start/end must be > 0.")
        if float(cfg["sampling_temperature_start"]) < float(cfg["sampling_temperature_end"]):
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

    def _resolve_answer_pooling(self) -> str:
        pooling = str(self.runtime_cfg.get("answer_pooling", _DEFAULT_ANSWER_POOLING) or _DEFAULT_ANSWER_POOLING)
        pooling = pooling.strip().lower()
        if pooling not in _ANSWER_POOLINGS:
            raise ValueError(f"runtime_cfg.answer_pooling must be one of {sorted(_ANSWER_POOLINGS)}, got {pooling!r}.")
        return pooling

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
        raw = self._require_cfg_mapping(raw, "evaluation_cfg.diverse_beam")
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
