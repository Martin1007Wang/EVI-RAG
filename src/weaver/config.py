from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class PolicyDefaults:
    mode: str = "bdb"
    flow_budget_conditioning: str = "additive"
    evidence_state_encoder_dropout: float = 0.0
    edge_scorer: str = "pointer"
    continuation_logit_bias_init: float = -5.912023
    continuation_mass_reduction: str = "logsumexp"
    feature_dde_cfg: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "num_forward_rounds": 2,
            "num_backward_rounds": 2,
            "include_anchor_indicator": True,
        }
    )
    feature_role_projection: str = "linear_layernorm"
    feature_role_projection_init: str = "identity"
    non_text_init_std: float = 0.02
    evidence_state_encoder_cfg: dict[str, Any] = field(
        default_factory=lambda: {
            "num_layers": 1,
            "dropout": 0.0,
        }
    )
    frontier_pointer_cfg: dict[str, Any] = field(default_factory=dict)
    stop_head_cfg: dict[str, Any] = field(default_factory=dict)
    flow_head_cfg: dict[str, Any] = field(
        default_factory=lambda: {
            "num_layers": 1,
            "dropout": 0.0,
            "zero_init": True,
            "bias_init": 0.0,
        }
    )


@dataclass(frozen=True)
class RewardDefaults:
    reward_floor: float = 1.0e-6
    edge_cost: float = 0.1
    beta: float = 2.0
    debug_checks: bool = False


@dataclass(frozen=True)
class LossDefaults:
    type: str = "bdb"
    child_flow_target: str = "detach_current"
    backward_kernel: str = "uniform_boundary"
    edge_mode: str = "full"
    child_chunk_size: int = 2048


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    hidden_dim: int
    mode: str
    flow_budget_conditioning: str
    edge_scorer: str
    continuation_logit_bias_init: float
    continuation_mass_reduction: str
    feature_encoder_cfg: dict[str, Any]
    evidence_state_encoder_dropout: float
    evidence_state_encoder_cfg: dict[str, Any]
    flow_head_cfg: dict[str, Any]
    frontier_pointer_cfg: dict[str, Any]
    stop_head_cfg: dict[str, Any]


@dataclass(frozen=True)
class RolloutRuntimeConfig:
    expand_budget: int
    train_num_rollout: int
    eval_num_rollout: int
    train_chunk_size: int
    eval_chunk_size: int


@dataclass(frozen=True)
class EvalRuntimeConfig:
    budgets: tuple[int, ...]
    debug_metrics: bool
    exclude_anchors_from_retrieved: bool
    use_reachable_targets: bool
    compute_loss: bool


@dataclass(frozen=True)
class SamplingRuntimeConfig:
    train_temperature: float
    eval_temperature: float


@dataclass(frozen=True)
class DiagnosticsRuntimeConfig:
    train_rollout_diagnostics: bool
    train_rollout_diagnostics_interval: int
    train_policy_diagnostics: bool
    train_validate_rollout_depth: bool
    eval_validate_rollout_depth: bool
    grad_norm_interval: int


def build_policy_runtime_config(
    *,
    hidden_dim: int,
    entity_text_embeddings: torch.Tensor,
    entity_embedding_map: torch.Tensor,
    relation_embeddings: torch.Tensor,
    policy: dict[str, Any] | None = None,
    defaults: PolicyDefaults | None = None,
) -> PolicyRuntimeConfig:
    defaults = defaults or PolicyDefaults()
    cfg = dict(policy or {})
    hidden_dim = int(hidden_dim)
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

    evidence_state_encoder_dropout = float(
        cfg.pop(
            "evidence_state_encoder_dropout",
            defaults.evidence_state_encoder_dropout,
        )
    )
    mode = str(cfg.pop("mode", defaults.mode)).lower()
    allowed_modes = {"bdb"}
    if mode not in allowed_modes:
        raise ValueError(f"policy.mode must be one of {sorted(allowed_modes)}, got {mode!r}.")
    flow_budget_conditioning = str(
        cfg.pop("flow_budget_conditioning", defaults.flow_budget_conditioning)
    ).lower()
    allowed_budget_conditioning = {"none", "additive"}
    if flow_budget_conditioning not in allowed_budget_conditioning:
        raise ValueError(
            "policy.flow_budget_conditioning must be one of "
            f"{sorted(allowed_budget_conditioning)}, got {flow_budget_conditioning!r}."
        )
    edge_scorer_default = defaults.edge_scorer
    edge_scorer = str(cfg.pop("edge_scorer", edge_scorer_default))
    continuation_logit_bias_init = float(
        cfg.pop(
            "continuation_logit_bias_init",
            defaults.continuation_logit_bias_init,
        )
    )
    continuation_mass_reduction = str(
        cfg.pop(
            "continuation_mass_reduction",
            defaults.continuation_mass_reduction,
        )
    )
    allowed_edge_scorers = {"pointer"}
    if edge_scorer not in allowed_edge_scorers:
        raise ValueError(
            "policy.edge_scorer must be one of "
            f"{sorted(allowed_edge_scorers)}, got {edge_scorer!r}."
        )
    allowed_mass_reductions = {"logsumexp", "logmeanexp"}
    if continuation_mass_reduction not in allowed_mass_reductions:
        raise ValueError(
            "policy.continuation_mass_reduction must be one of "
            f"{sorted(allowed_mass_reductions)}, "
            f"got {continuation_mass_reduction!r}."
        )
    feature_encoder_cfg = dict(cfg.pop("feature_encoder", {}))
    evidence_state_encoder_cfg = dict(
        cfg.pop("evidence_state_encoder", defaults.evidence_state_encoder_cfg)
    )
    removed_state_keys = {"use_path_memory", "use_frontier_summary"}.intersection(
        evidence_state_encoder_cfg
    )
    if removed_state_keys:
        raise ValueError(
            "Removed evidence_state_encoder keys: "
            f"{sorted(removed_state_keys)}."
        )
    removed_policy_keys = {"edge_policy_head", "state_readout", "state_readout_dropout"}.intersection(cfg)
    if removed_policy_keys:
        raise ValueError(
            "Removed policy keys: "
            f"{sorted(removed_policy_keys)}."
    )
    frontier_pointer_cfg = dict(
        cfg.pop("frontier_pointer", defaults.frontier_pointer_cfg)
    )
    stop_head_cfg = dict(cfg.pop("stop_head", defaults.stop_head_cfg))
    flow_head_cfg = dict(cfg.pop("flow_head", defaults.flow_head_cfg))
    if cfg:
        raise ValueError(f"Unused policy keys: {sorted(cfg)}.")
    # REMOVED: TE-BFM/SubTB policy modes are outside the BDB method — see methodology.md §3.9
    if mode == "bdb" and flow_budget_conditioning == "none":
        raise ValueError(
            "policy.flow_budget_conditioning cannot be 'none' when "
            "policy.mode='bdb'. Use additive budget conditioning."
        )

    feature_encoder_cfg = build_feature_encoder_config(
        cfg=feature_encoder_cfg,
        entity_text_embeddings=entity_text_embeddings,
        entity_embedding_map=entity_embedding_map,
        relation_embeddings=relation_embeddings,
        hidden_dim=hidden_dim,
        defaults=defaults,
    )

    return PolicyRuntimeConfig(
        hidden_dim=hidden_dim,
        mode=mode,
        flow_budget_conditioning=flow_budget_conditioning,
        edge_scorer=edge_scorer,
        continuation_logit_bias_init=continuation_logit_bias_init,
        continuation_mass_reduction=continuation_mass_reduction,
        feature_encoder_cfg=feature_encoder_cfg,
        evidence_state_encoder_dropout=evidence_state_encoder_dropout,
        evidence_state_encoder_cfg=evidence_state_encoder_cfg,
        flow_head_cfg=flow_head_cfg,
        frontier_pointer_cfg=frontier_pointer_cfg,
        stop_head_cfg=stop_head_cfg,
    )


def build_rollout_runtime_config(
    rollout: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
) -> RolloutRuntimeConfig:
    rollout_options = dict(rollout or {})
    runtime_cfg = dict(runtime or {})

    expand_budget = int(rollout_options.pop("expand_budget", 2))
    train_num_rollout = int(rollout_options.pop("train_num_rollout", 8))
    eval_num_rollout = int(rollout_options.pop("eval_num_rollout", 8))
    if rollout_options:
        raise ValueError(f"Unused rollout keys: {sorted(rollout_options)}.")

    train_chunk_size = runtime_cfg.pop("train_chunk_size", train_num_rollout)
    eval_chunk_size = runtime_cfg.pop("eval_chunk_size", eval_num_rollout)
    if runtime_cfg:
        raise ValueError(f"Unused runtime keys: {sorted(runtime_cfg)}.")

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
            name="runtime.train_chunk_size",
        ),
        eval_chunk_size=normalize_chunk_size(
            eval_chunk_size,
            fallback=eval_num_rollout,
            name="runtime.eval_chunk_size",
        ),
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
    compute_loss = bool(cfg.pop("compute_loss", False))

    if cfg:
        raise ValueError(f"Unused eval keys: {sorted(cfg)}.")

    budgets = tuple(sorted({int(k) for k in raw_budgets}))

    if not budgets:
        raise ValueError("eval.budgets must be non-empty.")
    if any(k < 1 for k in budgets):
        raise ValueError(f"eval.budgets must all be >= 1, got {budgets}.")
    if max(budgets) > int(eval_num_rollout):
        raise ValueError(
            f"max(eval.budgets)={max(budgets)} cannot exceed "
            f"rollout.eval_num_rollout={eval_num_rollout}."
        )

    return EvalRuntimeConfig(
        budgets=budgets,
        debug_metrics=debug_metrics,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
        compute_loss=compute_loss,
    )


def build_sampling_runtime_config(
    sampling: dict[str, Any] | None,
) -> SamplingRuntimeConfig:
    cfg = dict(sampling or {})

    train_temperature = float(cfg.pop("train_temperature", 1.0))
    eval_temperature = float(cfg.pop("eval_temperature", train_temperature))
    if cfg:
        raise ValueError(f"Unused sampling keys: {sorted(cfg)}.")
    if train_temperature <= 0.0:
        raise ValueError(
            f"sampling.train_temperature must be positive, got {train_temperature}."
        )
    if eval_temperature <= 0.0:
        raise ValueError(
            f"sampling.eval_temperature must be positive, got {eval_temperature}."
        )

    return SamplingRuntimeConfig(
        train_temperature=train_temperature,
        eval_temperature=eval_temperature,
    )


def build_reward_config(
    reward: dict[str, Any] | None,
    *,
    defaults: RewardDefaults | None = None,
) -> dict[str, Any]:
    defaults = defaults or RewardDefaults()
    cfg = dict(reward or {})

    removed_reward_keys = {
        "score_mode",
        "length_discount",
        "path_weight",
        "prefix_answer_bonus",
        "wrong_branch_penalty",
        "path_prefix_weight",
    }.intersection(cfg)
    if removed_reward_keys:
        raise ValueError(
            "Removed reward keys: "
            f"{sorted(removed_reward_keys)}. Reward is terminal F-beta plus edge cost."
        )
    reward_floor = float(cfg.pop("reward_floor", defaults.reward_floor))
    edge_cost = float(cfg.pop("edge_cost", defaults.edge_cost))
    beta = float(cfg.pop("beta", defaults.beta))
    if reward_floor <= 0.0:
        raise ValueError(
            f"reward.reward_floor must be > 0, got {reward_floor}."
        )
    if edge_cost < 0.0:
        raise ValueError(f"reward.edge_cost must be >= 0, got {edge_cost}.")
    if beta <= 0.0:
        raise ValueError(f"reward.beta must be > 0, got {beta}.")
    debug_checks = bool(cfg.pop("debug_checks", defaults.debug_checks))
    if cfg:
        raise ValueError(f"Unused reward keys: {sorted(cfg)}.")

    return {
        "reward_floor": reward_floor,
        "edge_cost": edge_cost,
        "beta": beta,
        "debug_checks": debug_checks,
    }


def build_loss_config(
    loss: dict[str, Any] | None,
    *,
    max_trajectory_len: int,
    defaults: LossDefaults | None = None,
) -> dict[str, Any]:
    defaults = defaults or LossDefaults()
    del max_trajectory_len
    cfg = dict(loss or {})

    loss_type = str(cfg.pop("type", defaults.type)).lower()
    if loss_type not in {"bdb"}:
        raise ValueError(
            "Only loss.type='bdb' is supported, "
            f"got {loss_type!r}."
        )
    child_flow_target = str(
        cfg.pop("child_flow_target", defaults.child_flow_target)
    )
    backward_kernel = str(cfg.pop("backward_kernel", defaults.backward_kernel))
    edge_mode = str(cfg.pop("edge_mode", defaults.edge_mode))
    child_chunk_size = int(cfg.pop("child_chunk_size", defaults.child_chunk_size))
    state_sources = dict(
        cfg.pop(
            "state_sources",
            {"rollout": True, "oracle_prefix": False, "counterfactual": False},
        )
    )
    removed_loss_keys = {
        "prefix_stop_lambda",
        "stop_advantage_lambda",
        "stop_advantage_margin",
        "stop_advantage_temperature",
        "stop_advantage_child_topk",
        "stop_boundary",
    }.intersection(cfg)
    if removed_loss_keys:
        raise ValueError(
            "Removed loss keys: "
            f"{sorted(removed_loss_keys)}. Stop is learned by the policy "
            "and training uses pure BDB."
        )
    removed_alt_loss_keys = {
        "subtb_lambda",
        "stop_weight",
        "edge_weight",
        "base_weight",
        "lookahead_depth",
        "terminal_chunk_size",
        "max_backup_edges_per_state",
        "max_expanded_states",
        "include_counterfactual_internal_states",
    }.intersection(cfg)
    if removed_alt_loss_keys:
        raise ValueError(
            "Removed non-BDB loss keys: "
            f"{sorted(removed_alt_loss_keys)}. "
            "REMOVED: alternative objectives/backups — see methodology.md §3.9"
        )
    if cfg:
        raise ValueError(f"Unused loss keys: {sorted(cfg)}.")
    if loss_type == "bdb":
        if edge_mode != "full":
            raise ValueError(
                "BDB v1 only supports loss.edge_mode='full'; "
                f"got {edge_mode!r}."
            )
        if child_flow_target != "detach_current":
            raise ValueError(
                "BDB v1 only supports loss.child_flow_target='detach_current'; "
                f"got {child_flow_target!r}."
            )
        if backward_kernel != "uniform_boundary":
            raise ValueError(
                "BDB v1 only supports loss.backward_kernel='uniform_boundary'; "
                f"got {backward_kernel!r}."
            )
        if not bool(state_sources.get("rollout", False)):
            raise ValueError("BDB v1 requires loss.state_sources.rollout=true.")
        unsupported_sources = [
            name
            for name, enabled in state_sources.items()
            if name != "rollout" and bool(enabled)
        ]
        if unsupported_sources:
            raise ValueError(
                "BDB v1 only supports rollout state sources; unsupported enabled "
                f"sources={unsupported_sources}."
            )
    if child_chunk_size < 1:
        raise ValueError(
            f"loss.child_chunk_size must be >= 1, got {child_chunk_size}."
        )

    return {
        "type": loss_type,
        "child_flow_target": child_flow_target,
        "backward_kernel": backward_kernel,
        "edge_mode": edge_mode,
        "child_chunk_size": child_chunk_size,
        "state_sources": state_sources,
    }


def validate_algorithm_coupling(
    *,
    policy: PolicyRuntimeConfig,
    loss: dict[str, Any],
    rollout: RolloutRuntimeConfig,
    reward: dict[str, Any],
) -> None:
    loss_type = str(loss.get("type", "bdb")).lower()
    if policy.mode == "bdb" and loss_type != "bdb":
        raise ValueError("policy.mode='bdb' requires loss.type='bdb'.")
    if loss_type == "bdb" and policy.mode != "bdb":
        raise ValueError("loss.type='bdb' requires policy.mode='bdb'.")
    if policy.flow_budget_conditioning == "none":
        raise ValueError("policy.flow_budget_conditioning cannot be 'none' for BDB.")
    if rollout.expand_budget < 1:
        raise ValueError("rollout.expand_budget must be >= 1 for BDB.")
    del reward
    # REMOVED: TE-BFM coupling checks and backup caps — see methodology.md §3.9


def build_diagnostics_runtime_config(
    diagnostics: dict[str, Any] | None,
) -> DiagnosticsRuntimeConfig:
    cfg = dict(diagnostics or {})

    train_rollout_diagnostics = bool(cfg.pop("train_rollout_diagnostics", False))
    train_rollout_diagnostics_interval = int(
        cfg.pop("train_rollout_diagnostics_interval", 0)
    )
    train_policy_diagnostics = bool(cfg.pop("train_policy_diagnostics", False))
    train_validate_rollout_depth = bool(cfg.pop("train_validate_rollout_depth", False))
    eval_validate_rollout_depth = bool(cfg.pop("eval_validate_rollout_depth", False))
    grad_norm_interval = int(cfg.pop("grad_norm_interval", 0))

    if cfg:
        raise ValueError(f"Unused diagnostics keys: {sorted(cfg)}.")
    if train_rollout_diagnostics_interval < 0:
        raise ValueError(
            "diagnostics.train_rollout_diagnostics_interval must be >= 0, "
            f"got {train_rollout_diagnostics_interval}."
        )
    if grad_norm_interval < 0:
        raise ValueError(
            f"diagnostics.grad_norm_interval must be >= 0, got {grad_norm_interval}."
        )

    return DiagnosticsRuntimeConfig(
        train_rollout_diagnostics=train_rollout_diagnostics,
        train_rollout_diagnostics_interval=train_rollout_diagnostics_interval,
        train_policy_diagnostics=train_policy_diagnostics,
        train_validate_rollout_depth=train_validate_rollout_depth,
        eval_validate_rollout_depth=eval_validate_rollout_depth,
        grad_norm_interval=grad_norm_interval,
    )


def build_feature_encoder_config(
    *,
    cfg: dict[str, Any] | None = None,
    entity_text_embeddings: torch.Tensor,
    entity_embedding_map: torch.Tensor,
    relation_embeddings: torch.Tensor,
    hidden_dim: int,
    defaults: PolicyDefaults,
) -> dict[str, Any]:
    cfg = dict(cfg or {})
    dde_cfg = dict(cfg.pop("dde", defaults.feature_dde_cfg))
    non_text_init_std = float(cfg.pop("non_text_init_std", defaults.non_text_init_std))
    role_projection = str(cfg.pop("role_projection", defaults.feature_role_projection))
    role_projection_init = str(
        cfg.pop("role_projection_init", defaults.feature_role_projection_init)
    )
    if cfg:
        raise ValueError(f"Unused policy.feature_encoder keys: {sorted(cfg)}.")

    return {
        "entity_text_embeddings": entity_text_embeddings,
        "entity_embedding_map": entity_embedding_map,
        "relation_embeddings": relation_embeddings,
        "hidden_dim": int(hidden_dim),
        "dde": dde_cfg,
        "non_text_init_std": non_text_init_std,
        "role_projection": role_projection,
        "role_projection_init": role_projection_init,
    }


def validate_rollout_counts(
    *,
    expand_budget: int,
    train_num_rollout: int,
    eval_num_rollout: int,
) -> None:
    if expand_budget < 0:
        raise ValueError(f"rollout.expand_budget must be >= 0, got {expand_budget}.")
    if train_num_rollout < 1:
        raise ValueError(
            f"rollout.train_num_rollout must be >= 1, got {train_num_rollout}."
        )
    if eval_num_rollout < 1:
        raise ValueError(
            f"rollout.eval_num_rollout must be >= 1, got {eval_num_rollout}."
        )


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
    "LossDefaults",
    "PolicyDefaults",
    "PolicyRuntimeConfig",
    "RewardDefaults",
    "RolloutRuntimeConfig",
    "SamplingRuntimeConfig",
    "build_diagnostics_runtime_config",
    "build_eval_runtime_config",
    "build_feature_encoder_config",
    "build_loss_config",
    "build_policy_runtime_config",
    "build_reward_config",
    "build_rollout_runtime_config",
    "build_sampling_runtime_config",
    "normalize_chunk_size",
    "validate_algorithm_coupling",
    "validate_rollout_counts",
]
