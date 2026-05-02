from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.weaver.rollout.stop_advantage import StopAdvantageConfig


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    hidden_dim: int
    feature_encoder_cfg: dict[str, Any]
    state_readout_dropout: float
    state_readout_cfg: dict[str, Any]
    stop_scorer_cfg: dict[str, Any]
    edge_scorer_cfg: dict[str, Any]
    flow_head_cfg: dict[str, Any]
    action_parameterization: str
    doob_stop_mode: str
    doob_successor_value_mode: str


@dataclass(frozen=True)
class RolloutRuntimeConfig:
    expand_budget: int
    train_num_rollout: int
    eval_num_rollout: int
    train_chunk_size: int
    eval_chunk_size: int
    stop_advantage_cfg: StopAdvantageConfig


@dataclass(frozen=True)
class EvalRuntimeConfig:
    budgets: tuple[int, ...]
    debug_metrics: bool
    exclude_anchors_from_retrieved: bool
    use_reachable_targets: bool


@dataclass(frozen=True)
class ScheduleRuntimeConfig:
    temperature: float
    eval_temperature: float
    temperature_cfg: dict[str, Any] | None


@dataclass(frozen=True)
class DiagnosticsRuntimeConfig:
    train_rollout_diagnostics: bool
    train_rollout_diagnostics_interval: int
    train_stop_counterfactual: bool
    train_policy_diagnostics: bool
    train_validate_rollout_depth: bool
    eval_stop_counterfactual: bool
    eval_validate_rollout_depth: bool
    grad_norm_interval: int


def build_policy_runtime_config(
    *,
    policy_cfg: dict[str, Any] | None,
    entity_text_embeddings: torch.Tensor,
    entity_embedding_map: torch.Tensor,
    relation_embeddings: torch.Tensor,
) -> PolicyRuntimeConfig:
    cfg = dict(policy_cfg or {})

    hidden_dim = int(cfg.pop("hidden_dim", 1024))
    state_readout_dropout = float(cfg.pop("state_readout_dropout", 0.0))
    action_parameterization = str(
        cfg.pop("action_parameterization", "doob_value_prior")
    )

    feature_encoder_cfg = dict(cfg.pop("feature_encoder", {}))
    state_readout_cfg = dict(cfg.pop("state_readout", {}))
    transition_features_cfg = dict(cfg.pop("transition_features", {}))
    legacy_action_features_cfg = dict(cfg.pop("action_features", {}))
    if transition_features_cfg:
        raise ValueError(
            "policy_cfg.transition_features was removed with the residual edge "
            f"scorer. Got: {sorted(transition_features_cfg)}."
        )
    if legacy_action_features_cfg:
        raise ValueError(
            "policy_cfg.action_features was removed with the residual edge scorer. "
            f"Got: {sorted(legacy_action_features_cfg)}."
        )
    stop_scorer_cfg = dict(cfg.pop("stop_scorer", {}))
    edge_scorer_cfg = dict(cfg.pop("edge_scorer", {}))
    flow_head_cfg = dict(cfg.pop("flow_head", {}))
    doob_cfg = normalize_doob_config(cfg.pop("doob", {}))

    state_readout_cfg = normalize_state_readout_config(state_readout_cfg)
    stop_scorer_cfg = normalize_stop_scorer_config(stop_scorer_cfg)

    if "share_edge_encoder_with_readout" in edge_scorer_cfg:
        raise ValueError(
            "policy_cfg.edge_scorer.share_edge_encoder_with_readout was removed "
            "with the residual edge scorer."
        )

    if cfg:
        raise ValueError(f"Unused policy_cfg keys: {sorted(cfg)}.")

    if action_parameterization not in {"doob_value_prior", "semantic_gate"}:
        raise ValueError(
            "policy_cfg.action_parameterization must be 'doob_value_prior' or "
            f"'semantic_gate', got {action_parameterization!r}."
        )

    feature_encoder_cfg = build_feature_encoder_config(
        cfg=feature_encoder_cfg,
        entity_text_embeddings=entity_text_embeddings,
        entity_embedding_map=entity_embedding_map,
        relation_embeddings=relation_embeddings,
        hidden_dim=hidden_dim,
    )

    return PolicyRuntimeConfig(
        hidden_dim=hidden_dim,
        feature_encoder_cfg=feature_encoder_cfg,
        state_readout_dropout=state_readout_dropout,
        state_readout_cfg=state_readout_cfg,
        stop_scorer_cfg=stop_scorer_cfg,
        edge_scorer_cfg=edge_scorer_cfg,
        flow_head_cfg=flow_head_cfg,
        action_parameterization=action_parameterization,
        doob_stop_mode=doob_cfg["stop_mode"],
        doob_successor_value_mode=doob_cfg["successor_value_mode"],
    )


def build_rollout_runtime_config(
    rollout_cfg: dict[str, Any] | None,
) -> RolloutRuntimeConfig:
    cfg = dict(rollout_cfg or {})

    expand_budget = int(cfg.pop("expand_budget", 3))
    train_num_rollout = int(cfg.pop("train_num_rollout", 8))
    eval_num_rollout = int(cfg.pop("eval_num_rollout", 8))

    train_chunk_size = cfg.pop("train_chunk_size", train_num_rollout)
    eval_chunk_size = cfg.pop("eval_chunk_size", eval_num_rollout)
    stop_advantage_cfg = StopAdvantageConfig.from_dict(cfg.pop("stop_adv", None))

    if cfg:
        raise ValueError(f"Unused rollout_cfg keys: {sorted(cfg)}.")
    if stop_advantage_cfg.enabled:
        raise ValueError(
            "rollout_cfg.stop_adv.enabled=true is not supported by fused-only "
            "rollouts yet."
        )

    validate_rollout_counts(
        expand_budget=expand_budget,
        train_num_rollout=train_num_rollout,
        eval_num_rollout=eval_num_rollout,
    )

    return RolloutRuntimeConfig(
        expand_budget=expand_budget,
        train_num_rollout=train_num_rollout,
        eval_num_rollout=eval_num_rollout,
        train_chunk_size=normalize_chunk_size(
            train_chunk_size,
            fallback=train_num_rollout,
            name="train_chunk_size",
        ),
        eval_chunk_size=normalize_chunk_size(
            eval_chunk_size,
            fallback=eval_num_rollout,
            name="eval_chunk_size",
        ),
        stop_advantage_cfg=stop_advantage_cfg,
    )


def build_eval_runtime_config(
    *,
    eval_cfg: dict[str, Any] | None,
    eval_num_rollout: int,
) -> EvalRuntimeConfig:
    cfg = dict(eval_cfg or {})

    raw_budgets = cfg.pop("budgets", (1, 2, 4, 8))
    debug_metrics = bool(cfg.pop("debug_metrics", False))
    exclude_anchors_from_retrieved = bool(
        cfg.pop("exclude_anchors_from_retrieved", True)
    )
    use_reachable_targets = bool(cfg.pop("use_reachable_targets", True))

    if cfg:
        raise ValueError(f"Unused eval_cfg keys: {sorted(cfg)}.")

    budgets = tuple(sorted({int(k) for k in raw_budgets}))

    if not budgets:
        raise ValueError("eval_cfg.budgets must be non-empty.")
    if any(k < 1 for k in budgets):
        raise ValueError(f"eval_cfg.budgets must all be >= 1, got {budgets}.")
    if max(budgets) > int(eval_num_rollout):
        raise ValueError(
            f"max(eval_cfg.budgets)={max(budgets)} cannot exceed "
            f"eval_num_rollout={eval_num_rollout}."
        )

    return EvalRuntimeConfig(
        budgets=budgets,
        debug_metrics=debug_metrics,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )


def build_schedule_runtime_config(
    schedule_cfg: dict[str, Any] | None,
) -> ScheduleRuntimeConfig:
    cfg = dict(schedule_cfg or {})

    temperature = float(cfg.pop("temperature", 1.0))
    eval_temperature = float(cfg.pop("eval_temperature", temperature))
    temperature_cfg = cfg.pop("temperature_cfg", None)

    if cfg:
        raise ValueError(f"Unused schedule_cfg keys: {sorted(cfg)}.")

    return ScheduleRuntimeConfig(
        temperature=temperature,
        eval_temperature=eval_temperature,
        temperature_cfg=dict(temperature_cfg) if temperature_cfg is not None else None,
    )


def build_diagnostics_runtime_config(
    diagnostic_cfg: dict[str, Any] | None,
) -> DiagnosticsRuntimeConfig:
    cfg = dict(diagnostic_cfg or {})

    train_rollout_diagnostics = bool(cfg.pop("train_rollout_diagnostics", False))
    train_rollout_diagnostics_interval = int(
        cfg.pop("train_rollout_diagnostics_interval", 0)
    )
    train_stop_counterfactual = bool(cfg.pop("train_stop_counterfactual", False))
    train_policy_diagnostics = bool(cfg.pop("train_policy_diagnostics", False))
    train_validate_rollout_depth = bool(cfg.pop("train_validate_rollout_depth", False))
    eval_stop_counterfactual = bool(cfg.pop("eval_stop_counterfactual", False))
    eval_validate_rollout_depth = bool(cfg.pop("eval_validate_rollout_depth", False))
    grad_norm_interval = int(cfg.pop("grad_norm_interval", 0))

    if cfg:
        raise ValueError(f"Unused diagnostic_cfg keys: {sorted(cfg)}.")
    if train_rollout_diagnostics_interval < 0:
        raise ValueError(
            "diagnostic_cfg.train_rollout_diagnostics_interval must be >= 0, "
            f"got {train_rollout_diagnostics_interval}."
        )
    if grad_norm_interval < 0:
        raise ValueError(
            f"diagnostic_cfg.grad_norm_interval must be >= 0, got {grad_norm_interval}."
        )

    return DiagnosticsRuntimeConfig(
        train_rollout_diagnostics=train_rollout_diagnostics,
        train_rollout_diagnostics_interval=train_rollout_diagnostics_interval,
        train_stop_counterfactual=train_stop_counterfactual,
        train_policy_diagnostics=train_policy_diagnostics,
        train_validate_rollout_depth=train_validate_rollout_depth,
        eval_stop_counterfactual=eval_stop_counterfactual,
        eval_validate_rollout_depth=eval_validate_rollout_depth,
        grad_norm_interval=grad_norm_interval,
    )


def build_feature_encoder_config(
    *,
    cfg: dict[str, Any],
    entity_text_embeddings: torch.Tensor,
    entity_embedding_map: torch.Tensor,
    relation_embeddings: torch.Tensor,
    hidden_dim: int,
) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("hidden_dim", int(hidden_dim))
    if int(cfg["hidden_dim"]) != int(hidden_dim):
        raise ValueError(
            "policy_cfg.feature_encoder.hidden_dim must match policy_cfg.hidden_dim: "
            f"{cfg['hidden_dim']} != {hidden_dim}."
        )

    embedding_dim = cfg.pop("embedding_dim", None)
    if embedding_dim is not None:
        actual_dim = int(entity_text_embeddings.size(-1))
        if int(embedding_dim) != actual_dim:
            raise ValueError(
                "policy_cfg.feature_encoder.embedding_dim must match runtime "
                f"entity_text_embeddings dim: {embedding_dim} != {actual_dim}."
            )

    role_projection = cfg.pop("role_projection", None)
    if role_projection not in (None, "linear_layernorm"):
        raise ValueError(
            "policy_cfg.feature_encoder.role_projection must be 'linear_layernorm' "
            f"when provided, got {role_projection!r}."
        )

    forbidden = {
        "entity_text_embeddings",
        "entity_embedding_map",
        "relation_embeddings",
    }

    overlap = forbidden.intersection(cfg)
    if overlap:
        raise ValueError(
            "policy_cfg.feature_encoder must not contain runtime embedding tensors: "
            f"{sorted(overlap)}."
        )

    cfg.update(
        {
            "entity_text_embeddings": entity_text_embeddings,
            "entity_embedding_map": entity_embedding_map,
            "relation_embeddings": relation_embeddings,
        }
    )

    return cfg


def normalize_state_readout_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)

    _pop_expected(cfg, "node_pooling", "query_dot", "policy_cfg.state_readout")
    _pop_expected(cfg, "edge_pooling", "query_dot", "policy_cfg.state_readout")
    _pop_expected(cfg, "edge_encoder", "linear_layernorm", "policy_cfg.state_readout")
    _pop_expected(cfg, "include_anchor_pool", False, "policy_cfg.state_readout")
    _pop_expected(
        cfg,
        "include_progress_in_state",
        False,
        "policy_cfg.state_readout",
    )

    return cfg


def normalize_stop_scorer_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg)


def normalize_doob_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg or {})

    top_k = cfg.pop("top_k", None)
    if top_k is not None:
        raise ValueError(
            "policy_cfg.doob.top_k was removed because Doob policy now uses "
            "full frontier support. Remove this key instead of truncating the prior."
        )

    stop_mode = str(cfg.pop("stop_mode", "reward"))
    if stop_mode not in {"reward", "learned"}:
        raise ValueError(
            "policy_cfg.doob.stop_mode must be 'reward' or 'learned', "
            f"got {stop_mode!r}."
        )

    successor_value_mode = str(cfg.pop("successor_value_mode", "flow"))
    if successor_value_mode != "flow":
        raise ValueError(
            "policy_cfg.doob.successor_value_mode must be 'flow', "
            f"got {successor_value_mode!r}."
        )

    if cfg:
        raise ValueError(f"Unused policy_cfg.doob keys: {sorted(cfg)}.")

    return {
        "stop_mode": stop_mode,
        "successor_value_mode": successor_value_mode,
    }


def _pop_expected(
    cfg: dict[str, Any],
    key: str,
    expected: Any,
    namespace: str,
) -> None:
    if key not in cfg:
        return
    value = cfg.pop(key)
    if value != expected:
        raise ValueError(
            f"{namespace}.{key}={value!r} is no longer supported; expected "
            f"{expected!r} for the current policy implementation."
        )


def validate_rollout_counts(
    *,
    expand_budget: int,
    train_num_rollout: int,
    eval_num_rollout: int,
) -> None:
    if expand_budget < 0:
        raise ValueError(f"expand_budget must be >= 0, got {expand_budget}.")
    if train_num_rollout < 1:
        raise ValueError(f"train_num_rollout must be >= 1, got {train_num_rollout}.")
    if eval_num_rollout < 1:
        raise ValueError(f"eval_num_rollout must be >= 1, got {eval_num_rollout}.")


def normalize_chunk_size(
    value: Any,
    *,
    fallback: int,
    name: str,
) -> int:
    if value is None:
        return int(fallback)

    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1 or None, got {value}.")

    return value


__all__ = [
    "DiagnosticsRuntimeConfig",
    "EvalRuntimeConfig",
    "PolicyRuntimeConfig",
    "RolloutRuntimeConfig",
    "ScheduleRuntimeConfig",
    "build_diagnostics_runtime_config",
    "build_eval_runtime_config",
    "build_feature_encoder_config",
    "build_policy_runtime_config",
    "build_rollout_runtime_config",
    "build_schedule_runtime_config",
    "normalize_chunk_size",
    "validate_rollout_counts",
]
