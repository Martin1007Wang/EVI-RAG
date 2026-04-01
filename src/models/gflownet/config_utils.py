from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from omegaconf import DictConfig, OmegaConf


_DEFAULT_TRAINING_CFG: dict[str, Any] = {
    "rollouts_per_graph": 8,
    "sampling_temperature": 1.0,
    "sampling_temperature_schedule": {
        "type": "constant",
        "initial_temperature": None,
        "final_temperature": None,
        "total_steps": None,
        "hold_steps": 0,
    },
    "action_pruning": {
        "per_node_top_k": 0,
        "per_state_top_k": 0,
    },
    "answer_reward": {
        "gold_answer_bonus": 2.0,
        "wrong_answer_penalty": 2.0,
        "failure_penalty": 4.0,
        "size_penalty": 0.1,
        "redundancy_penalty": 0.25,
        "component_penalty": 0.5,
    },
    "subtb": {
        "lambda_weight": 1.0,
        "topology_weight_alpha": 0.0,
    },
    "auxiliary": {
        "proposal": {
            "enabled": False,
            "prior": {
                "oracle_answer_distance_weight": 0.0,
                "prior_question_similarity_weight": 0.0,
                "prior_component_merge_weight": 0.0,
                "stop_hit_bias": 0.0,
            },
            "schedule": {
                "type": "constant",
                "initial_scale": None,
                "final_scale": None,
                "total_steps": None,
                "hold_steps": 0,
            },
        },
        "replay": {
            "enabled": False,
            "mix_alpha": 0.0,
            "buffer": {
                "capacity": 1024,
                "min_buffer_size": 64,
                "replay_trajectories_per_step": None,
                "deduplicate": True,
            },
            "guidance": {
                "add_shortest_path_guidance": False,
                "expand_imitation_weight": 0.0,
                "expand_imitation_from_anchor_bonus": 0.0,
                "expand_imitation_answer_finish_bonus": 0.0,
                "mask_stop_loss": True,
            },
            "schedule": {
                "type": "constant",
                "initial_alpha": None,
                "final_alpha": None,
                "total_steps": None,
                "hold_steps": 0,
            },
        },
    },
}

_REMOVED_TRAINING_KEYS: dict[str, str] = {
    "proposal_bias_schedule": "use training_cfg.auxiliary.proposal.schedule",
    "subgraph_proposal": "use training_cfg.auxiliary.proposal.prior",
    "success_replay": "use training_cfg.auxiliary.replay",
    "replay_mix_schedule": "use training_cfg.auxiliary.replay.schedule",
    "answer_quotient": "answer-quotient configs were removed; answer selection is now part of the main stop action",
    "potential_reward": "potential-reward shaping was removed from the mainline",
    "force_stop_on_answer_hit": "legacy stop shaping was removed from the mainline",
    "terminal_failure_log_reward": "configure training_cfg.answer_reward.failure_penalty instead",
    "step_log_penalty": "step penalties were removed from the mainline",
    "answer_stop_log_reward_bonus": "answer-stop bonuses were removed from the mainline",
}

_REMOVED_SUBTB_KEYS: dict[str, str] = {
    "normalize": "subtb normalization is always derived from the active trajectory weights",
    "root_loss_weight": "root/pairwise/terminal loss reweighting was removed from the mainline",
    "pairwise_loss_weight": "root/pairwise/terminal loss reweighting was removed from the mainline",
    "terminal_loss_weight": "root/pairwise/terminal loss reweighting was removed from the mainline",
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


def _validate_non_negative(
    *, cfg: dict[str, Any], field_names: tuple[str, ...], prefix: str
) -> None:
    for field_name in field_names:
        cfg[field_name] = float(cfg.get(field_name, 0.0))
        if cfg[field_name] < 0.0:
            raise ValueError(f"{prefix}.{field_name} must be >= 0.")


def _raise_on_removed_training_keys(training_cfg: Mapping[str, Any]) -> None:
    removed = [
        f"{key}: {message}"
        for key, message in _REMOVED_TRAINING_KEYS.items()
        if key in training_cfg
    ]
    if removed:
        raise ValueError(
            "Removed training config detected in the answer-centric mainline. "
            + " ".join(sorted(removed))
        )


def _raise_on_removed_subtb_keys(subtb_cfg: Mapping[str, Any]) -> None:
    removed = [
        f"{key}: {message}"
        for key, message in _REMOVED_SUBTB_KEYS.items()
        if key in subtb_cfg
    ]
    if removed:
        raise ValueError(
            "Removed SubTB config detected in the answer-centric mainline. "
            + " ".join(sorted(removed))
        )


def normalize_training_action_pruning_cfg(action_pruning_cfg: Any) -> dict[str, Any]:
    cfg = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["action_pruning"],
        override=_to_plain_mapping(
            action_pruning_cfg,
            field_name="training_cfg.action_pruning",
        ),
    )
    cfg["per_node_top_k"] = int(cfg.get("per_node_top_k", 0))
    if cfg["per_node_top_k"] < 0:
        raise ValueError("training.action_pruning.per_node_top_k must be >= 0.")
    cfg["per_state_top_k"] = int(cfg.get("per_state_top_k", 0))
    if cfg["per_state_top_k"] < 0:
        raise ValueError("training.action_pruning.per_state_top_k must be >= 0.")
    return cfg


def normalize_auxiliary_cfg(auxiliary_cfg: Any) -> dict[str, Any]:
    cfg = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"],
        override=_to_plain_mapping(
            auxiliary_cfg,
            field_name="training_cfg.auxiliary",
        ),
    )
    proposal_cfg = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"]["proposal"],
        override=_to_plain_mapping(
            cfg["proposal"],
            field_name="training_cfg.auxiliary.proposal",
        ),
    )
    proposal_cfg["enabled"] = bool(proposal_cfg.get("enabled", False))
    proposal_cfg["prior"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"]["proposal"]["prior"],
        override=_to_plain_mapping(
            proposal_cfg["prior"],
            field_name="training_cfg.auxiliary.proposal.prior",
        ),
    )
    _validate_non_negative(
        cfg=proposal_cfg["prior"],
        field_names=(
            "oracle_answer_distance_weight",
            "prior_question_similarity_weight",
            "prior_component_merge_weight",
            "stop_hit_bias",
        ),
        prefix="training.auxiliary.proposal.prior",
    )
    proposal_cfg["schedule"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"]["proposal"]["schedule"],
        override=_to_plain_mapping(
            proposal_cfg["schedule"],
            field_name="training_cfg.auxiliary.proposal.schedule",
        ),
    )
    if proposal_cfg["enabled"]:
        raise ValueError(
            "training.auxiliary.proposal.enabled=true is not supported in the answer-centric "
            "mainline yet; disable it until proposal bias is implemented end-to-end."
        )

    replay_cfg = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"]["replay"],
        override=_to_plain_mapping(
            cfg["replay"],
            field_name="training_cfg.auxiliary.replay",
        ),
    )
    replay_cfg["enabled"] = bool(replay_cfg.get("enabled", False))
    replay_cfg["mix_alpha"] = float(replay_cfg.get("mix_alpha", 0.0))
    if not 0.0 <= replay_cfg["mix_alpha"] < 1.0:
        raise ValueError("training.auxiliary.replay.mix_alpha must be in [0, 1).")
    replay_cfg["buffer"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"]["replay"]["buffer"],
        override=_to_plain_mapping(
            replay_cfg["buffer"],
            field_name="training_cfg.auxiliary.replay.buffer",
        ),
    )
    replay_cfg["buffer"]["capacity"] = int(replay_cfg["buffer"].get("capacity", 1024))
    if replay_cfg["buffer"]["capacity"] < 1:
        raise ValueError("training.auxiliary.replay.buffer.capacity must be >= 1.")
    replay_cfg["buffer"]["min_buffer_size"] = int(
        replay_cfg["buffer"].get("min_buffer_size", 64)
    )
    if replay_cfg["buffer"]["min_buffer_size"] < 0:
        raise ValueError(
            "training.auxiliary.replay.buffer.min_buffer_size must be >= 0."
        )
    replay_trajectories_per_step = replay_cfg["buffer"].get(
        "replay_trajectories_per_step"
    )
    if replay_trajectories_per_step is not None:
        replay_cfg["buffer"]["replay_trajectories_per_step"] = int(
            replay_trajectories_per_step
        )
        if replay_cfg["buffer"]["replay_trajectories_per_step"] < 0:
            raise ValueError(
                "training.auxiliary.replay.buffer.replay_trajectories_per_step must be >= 0 when set."
            )
    replay_cfg["buffer"]["deduplicate"] = bool(
        replay_cfg["buffer"].get("deduplicate", True)
    )
    replay_cfg["guidance"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"]["replay"]["guidance"],
        override=_to_plain_mapping(
            replay_cfg["guidance"],
            field_name="training_cfg.auxiliary.replay.guidance",
        ),
    )
    replay_cfg["guidance"]["add_shortest_path_guidance"] = bool(
        replay_cfg["guidance"].get("add_shortest_path_guidance", False)
    )
    _validate_non_negative(
        cfg=replay_cfg["guidance"],
        field_names=(
            "expand_imitation_weight",
            "expand_imitation_from_anchor_bonus",
            "expand_imitation_answer_finish_bonus",
        ),
        prefix="training.auxiliary.replay.guidance",
    )
    replay_cfg["guidance"]["mask_stop_loss"] = bool(
        replay_cfg["guidance"].get("mask_stop_loss", True)
    )
    replay_cfg["schedule"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"]["replay"]["schedule"],
        override=_to_plain_mapping(
            replay_cfg["schedule"],
            field_name="training_cfg.auxiliary.replay.schedule",
        ),
    )

    return {
        "proposal": proposal_cfg,
        "replay": replay_cfg,
    }


def normalize_training_cfg(training_cfg: Any) -> dict[str, Any]:
    training_cfg_mapping = _to_plain_mapping(training_cfg, field_name="training_cfg")
    _raise_on_removed_training_keys(training_cfg_mapping)
    cfg = _deep_merge(base=_DEFAULT_TRAINING_CFG, override=training_cfg_mapping)
    cfg["rollouts_per_graph"] = int(cfg.get("rollouts_per_graph", 8))
    if cfg["rollouts_per_graph"] < 1:
        raise ValueError("training.rollouts_per_graph must be >= 1.")
    cfg["sampling_temperature"] = float(cfg.get("sampling_temperature", 1.0))
    if cfg["sampling_temperature"] <= 0.0:
        raise ValueError("training.sampling_temperature must be > 0.")
    cfg["sampling_temperature_schedule"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["sampling_temperature_schedule"],
        override=_to_plain_mapping(
            cfg["sampling_temperature_schedule"],
            field_name="training_cfg.sampling_temperature_schedule",
        ),
    )
    cfg["action_pruning"] = normalize_training_action_pruning_cfg(
        cfg.get("action_pruning", {})
    )
    cfg["answer_reward"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["answer_reward"],
        override=_to_plain_mapping(
            cfg["answer_reward"],
            field_name="training_cfg.answer_reward",
        ),
    )
    _validate_non_negative(
        cfg=cfg["answer_reward"],
        field_names=(
            "gold_answer_bonus",
            "wrong_answer_penalty",
            "failure_penalty",
            "size_penalty",
            "redundancy_penalty",
            "component_penalty",
        ),
        prefix="training.answer_reward",
    )
    subtb_cfg = _to_plain_mapping(
        cfg["subtb"],
        field_name="training_cfg.subtb",
    )
    _raise_on_removed_subtb_keys(subtb_cfg)
    cfg["subtb"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["subtb"],
        override=subtb_cfg,
    )
    cfg["subtb"]["lambda_weight"] = float(cfg["subtb"].get("lambda_weight", 1.0))
    if not 0.0 <= cfg["subtb"]["lambda_weight"] <= 1.0:
        raise ValueError("training.subtb.lambda_weight must be in [0, 1].")
    cfg["subtb"]["topology_weight_alpha"] = float(
        cfg["subtb"].get("topology_weight_alpha", 0.0)
    )
    if cfg["subtb"]["topology_weight_alpha"] < 0.0:
        raise ValueError("training.subtb.topology_weight_alpha must be >= 0.")
    cfg["auxiliary"] = normalize_auxiliary_cfg(cfg.get("auxiliary", {}))
    return cfg


__all__ = [
    "normalize_auxiliary_cfg",
    "normalize_training_action_pruning_cfg",
    "normalize_training_cfg",
]
