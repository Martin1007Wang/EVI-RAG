from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf


_DEFAULT_TRAINING_CFG: dict[str, Any] = {
    "rollouts_per_graph": 8,
    "sampling_temperature": 1.0,
    "force_stop_on_answer_hit": False,
    "terminal_failure_log_reward": -3.0,
    "step_log_penalty": 0.0,
    "answer_stop_log_reward_bonus": 0.0,
    "sampling_temperature_schedule": {
        "type": "constant",
        "initial_temperature": None,
        "final_temperature": None,
        "total_steps": None,
        "hold_steps": 0,
    },
    "proposal_bias_schedule": {
        "type": "constant",
        "initial_scale": None,
        "final_scale": None,
        "total_steps": None,
        "hold_steps": 0,
    },
    "success_replay": {
        "mix_alpha": 0.0,
        "capacity": 1024,
        "min_buffer_size": 64,
        "replay_trajectories_per_step": None,
        "deduplicate": True,
        "add_shortest_path_guidance": False,
        "expand_imitation_weight": 0.0,
        "expand_imitation_from_anchor_bonus": 0.0,
        "expand_imitation_answer_finish_bonus": 0.0,
        "mask_stop_loss": True,
    },
    "replay_mix_schedule": {
        "type": "constant",
        "initial_alpha": None,
        "final_alpha": None,
        "total_steps": None,
        "hold_steps": 0,
    },
    "answer_quotient": {
        "enabled": False,
        "weight": 0.0,
        "direct_entity_ranking_weight": 0.0,
        "replace_terminal_loss": False,
        "gold_reward_mode": "shared",
        "allocate_stop_mass": False,
    },
    "potential_reward": {
        "answer_distance_weight": 0.0,
        "unreachable_distance": None,
    },
    "subgraph_reward": {
        "c_step": 0.1,
        "lambda_conn": 0.5,
        "beta_answer_bits": 0.0,
        "beta_answer_full": 0.0,
        "beta_hit": 2.0,
        "beta_cnt": 0.25,
        "beta_early": 1.0,
        "min_stop_edges": 1,
    },
    "subgraph_proposal": {
        "oracle_answer_distance_weight": 0.0,
        "prior_question_similarity_weight": 0.0,
        "prior_component_merge_weight": 0.0,
        "stop_hit_bias": 0.0,
    },
    "subtb": {
        "lambda_weight": 1.0,
        "normalize": True,
        "root_loss_weight": 1.0,
        "pairwise_loss_weight": 1.0,
        "terminal_loss_weight": 1.0,
    },
}


def _to_plain_mapping(node: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(node, DictConfig):
        container = OmegaConf.to_container(node, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"Expected {field_name} to resolve to a mapping.")
        return dict(container)
    if isinstance(node, Mapping):
        return {str(key): deepcopy(value) for key, value in node.items()}
    raise TypeError(f"Expected {field_name} to be a mapping, got {type(node)!r}.")


def _deep_merge(
    *, base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(base=merged[key], override=value)
            continue
        merged[key] = deepcopy(value)
    return merged


def normalize_answer_quotient_cfg(answer_quotient_cfg: Any) -> dict[str, Any]:
    cfg = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["answer_quotient"],
        override=_to_plain_mapping(
            answer_quotient_cfg,
            field_name="training_cfg.answer_quotient",
        ),
    )
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["weight"] = float(cfg.get("weight", 0.0))
    if cfg["weight"] < 0.0:
        raise ValueError("training.answer_quotient.weight must be >= 0.")
    cfg["direct_entity_ranking_weight"] = float(
        cfg.get("direct_entity_ranking_weight", 0.0)
    )
    if cfg["direct_entity_ranking_weight"] < 0.0:
        raise ValueError(
            "training.answer_quotient.direct_entity_ranking_weight must be >= 0."
        )
    cfg["replace_terminal_loss"] = bool(cfg.get("replace_terminal_loss", False))
    cfg["allocate_stop_mass"] = bool(cfg.get("allocate_stop_mass", False))
    cfg["gold_reward_mode"] = str(cfg.get("gold_reward_mode", "shared"))
    if cfg["gold_reward_mode"] not in {"shared", "unit"}:
        raise ValueError(
            "training.answer_quotient.gold_reward_mode must be one of {'shared', 'unit'}."
        )
    if cfg["direct_entity_ranking_weight"] > 0.0 and not cfg["enabled"]:
        raise ValueError(
            "training.answer_quotient.direct_entity_ranking_weight requires enabled=True."
        )
    if cfg["replace_terminal_loss"] and not cfg["enabled"]:
        raise ValueError(
            "training.answer_quotient.replace_terminal_loss requires enabled=True."
        )
    if cfg["replace_terminal_loss"] and cfg["weight"] <= 0.0:
        raise ValueError(
            "training.answer_quotient.replace_terminal_loss requires weight > 0."
        )
    return cfg


def answer_quotient_active(answer_quotient_cfg: Mapping[str, Any]) -> bool:
    return bool(answer_quotient_cfg.get("enabled", False)) and (
        float(answer_quotient_cfg.get("weight", 0.0)) > 0.0
        or bool(answer_quotient_cfg.get("replace_terminal_loss", False))
    )


def answer_quotient_stop_allocation_active(
    answer_quotient_cfg: Mapping[str, Any],
) -> bool:
    return bool(answer_quotient_cfg.get("enabled", False)) and bool(
        answer_quotient_cfg.get("allocate_stop_mass", False)
    )


def answer_quotient_direct_entity_ranking_active(
    answer_quotient_cfg: Mapping[str, Any],
) -> bool:
    return bool(answer_quotient_cfg.get("enabled", False)) and (
        float(answer_quotient_cfg.get("direct_entity_ranking_weight", 0.0)) > 0.0
    )


def normalize_potential_reward_cfg(potential_reward_cfg: Any) -> dict[str, Any]:
    cfg = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["potential_reward"],
        override=_to_plain_mapping(
            potential_reward_cfg,
            field_name="training_cfg.potential_reward",
        ),
    )
    cfg["answer_distance_weight"] = float(cfg.get("answer_distance_weight", 0.0))
    if cfg["answer_distance_weight"] < 0.0:
        raise ValueError(
            "training.potential_reward.answer_distance_weight must be >= 0."
        )
    unreachable_distance = cfg.get("unreachable_distance")
    if unreachable_distance is not None:
        cfg["unreachable_distance"] = int(unreachable_distance)
        if cfg["unreachable_distance"] < 0:
            raise ValueError(
                "training.potential_reward.unreachable_distance must be >= 0 when set."
            )
    return cfg


def potential_reward_active(potential_reward_cfg: Mapping[str, Any]) -> bool:
    return float(potential_reward_cfg.get("answer_distance_weight", 0.0)) > 0.0


def normalize_training_cfg(training_cfg: Any) -> dict[str, Any]:
    cfg = _deep_merge(
        base=_DEFAULT_TRAINING_CFG,
        override=_to_plain_mapping(training_cfg, field_name="training_cfg"),
    )
    cfg["rollouts_per_graph"] = int(cfg.get("rollouts_per_graph", 8))
    if cfg["rollouts_per_graph"] < 1:
        raise ValueError("training.rollouts_per_graph must be >= 1.")
    cfg["sampling_temperature"] = float(cfg.get("sampling_temperature", 1.0))
    if cfg["sampling_temperature"] <= 0.0:
        raise ValueError("training.sampling_temperature must be > 0.")
    cfg["force_stop_on_answer_hit"] = bool(cfg.get("force_stop_on_answer_hit", False))
    cfg["terminal_failure_log_reward"] = float(
        cfg.get("terminal_failure_log_reward", -3.0)
    )
    if cfg["terminal_failure_log_reward"] > 0.0:
        raise ValueError("training.terminal_failure_log_reward must be <= 0.")
    cfg["step_log_penalty"] = float(cfg.get("step_log_penalty", 0.0))
    if cfg["step_log_penalty"] > 0.0:
        raise ValueError("training.step_log_penalty must be <= 0.")
    cfg["answer_stop_log_reward_bonus"] = float(
        cfg.get("answer_stop_log_reward_bonus", 0.0)
    )
    if cfg["answer_stop_log_reward_bonus"] < 0.0:
        raise ValueError("training.answer_stop_log_reward_bonus must be >= 0.")
    cfg["answer_quotient"] = normalize_answer_quotient_cfg(cfg["answer_quotient"])
    cfg["potential_reward"] = normalize_potential_reward_cfg(cfg["potential_reward"])
    for key in (
        "sampling_temperature_schedule",
        "proposal_bias_schedule",
        "success_replay",
        "replay_mix_schedule",
        "subgraph_reward",
        "subgraph_proposal",
        "subtb",
    ):
        cfg[key] = _deep_merge(
            base=_DEFAULT_TRAINING_CFG[key],
            override=_to_plain_mapping(cfg[key], field_name=f"training_cfg.{key}"),
        )
    cfg["success_replay"]["expand_imitation_weight"] = float(
        cfg["success_replay"].get("expand_imitation_weight", 0.0)
    )
    if cfg["success_replay"]["expand_imitation_weight"] < 0.0:
        raise ValueError(
            "training.success_replay.expand_imitation_weight must be >= 0."
        )
    cfg["success_replay"]["expand_imitation_from_anchor_bonus"] = float(
        cfg["success_replay"].get("expand_imitation_from_anchor_bonus", 0.0)
    )
    if cfg["success_replay"]["expand_imitation_from_anchor_bonus"] < 0.0:
        raise ValueError(
            "training.success_replay.expand_imitation_from_anchor_bonus must be >= 0."
        )
    cfg["success_replay"]["expand_imitation_answer_finish_bonus"] = float(
        cfg["success_replay"].get("expand_imitation_answer_finish_bonus", 0.0)
    )
    if cfg["success_replay"]["expand_imitation_answer_finish_bonus"] < 0.0:
        raise ValueError(
            "training.success_replay.expand_imitation_answer_finish_bonus must be >= 0."
        )
    cfg["success_replay"]["mask_stop_loss"] = bool(
        cfg["success_replay"].get("mask_stop_loss", True)
    )
    return cfg


__all__ = [
    "answer_quotient_active",
    "answer_quotient_direct_entity_ranking_active",
    "answer_quotient_stop_allocation_active",
    "normalize_answer_quotient_cfg",
    "normalize_potential_reward_cfg",
    "normalize_training_cfg",
    "potential_reward_active",
]
