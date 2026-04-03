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
        "hit_bonus": 5.0,
        "frontier_bonus": 1.0,
        "coverage_bonus": 0.2,
        "size_penalty": 0.1,
        "component_penalty": 0.5,
    },
    "detailed_balance": {},
    "auxiliary": {
        "proposal": {
            "prior": {
                "oracle_answer_distance_weight": 0.0,
                "prior_question_similarity_weight": 0.0,
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
        "supervision": {
            "enabled": True,
            "warmup_steps": 0,
            "warmup_db_weight": 0.0,
            "db_weight": 1.0,
            "imitation_weight": 0.0,
            "success_action_weight": 0.0,
        },
    },
}

_REMOVED_TRAINING_KEYS: dict[str, str] = {
    "proposal_bias_schedule": "proposal-bias scheduling was removed from the answer-centric mainline",
    "subgraph_proposal": "use training_cfg.auxiliary.proposal.prior feature hints only; proposal bias itself was removed from the mainline",
    "success_replay": "use training_cfg.auxiliary.replay",
    "replay_mix_schedule": "use training_cfg.auxiliary.replay.schedule",
    "answer_quotient": "answer-quotient configs were removed; answer selection is now part of the main stop action",
    "potential_reward": "potential-reward shaping was removed from the mainline",
    "force_stop_on_answer_hit": "legacy stop shaping was removed from the mainline",
    "terminal_failure_log_reward": "terminal failure penalties were removed from the paper-aligned mainline reward",
    "step_log_penalty": "step penalties were removed from the mainline",
    "answer_stop_log_reward_bonus": "answer-stop bonuses were removed from the mainline",
    "subtb": "SubTB was removed; use training_cfg.detailed_balance for the paper-aligned single-step Detailed Balance objective",
}

_REMOVED_ANSWER_REWARD_KEYS: dict[str, str] = {
    "gold_answer_bonus": "use training_cfg.answer_reward.hit_bonus",
    "wrong_answer_penalty": "wrong-answer penalties were removed from the paper-aligned mainline reward",
    "failure_penalty": "terminal failure penalties were removed from the paper-aligned mainline reward",
    "redundancy_penalty": "use training_cfg.answer_reward.size_penalty and training_cfg.answer_reward.component_penalty instead of legacy redundancy shaping",
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


def _raise_on_removed_answer_reward_keys(answer_reward_cfg: Mapping[str, Any]) -> None:
    removed = [
        f"{key}: {message}"
        for key, message in _REMOVED_ANSWER_REWARD_KEYS.items()
        if key in answer_reward_cfg
    ]
    if removed:
        raise ValueError(
            "Removed answer reward keys detected in the answer-centric mainline. "
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
        ),
        prefix="training.auxiliary.proposal.prior",
    )
    if "enabled" in proposal_cfg:
        raise ValueError(
            "training.auxiliary.proposal.enabled was removed. Proposal bias is not implemented "
            "end-to-end in the answer-centric mainline; keep only "
            "training_cfg.auxiliary.proposal.prior.* feature hints."
        )
    if "schedule" in proposal_cfg:
        raise ValueError(
            "training.auxiliary.proposal.schedule was removed. Proposal bias scheduling is not "
            "implemented in the answer-centric mainline."
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

    supervision_cfg = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["auxiliary"]["supervision"],
        override=_to_plain_mapping(
            cfg.get("supervision", {}),
            field_name="training_cfg.auxiliary.supervision",
        ),
    )
    supervision_cfg["enabled"] = bool(supervision_cfg.get("enabled", True))
    supervision_cfg["warmup_steps"] = int(supervision_cfg.get("warmup_steps", 0))
    if supervision_cfg["warmup_steps"] < 0:
        raise ValueError("training.auxiliary.supervision.warmup_steps must be >= 0.")
    _validate_non_negative(
        cfg=supervision_cfg,
        field_names=(
            "warmup_db_weight",
            "db_weight",
            "imitation_weight",
            "success_action_weight",
        ),
        prefix="training.auxiliary.supervision",
    )

    return {
        "proposal": proposal_cfg,
        "replay": replay_cfg,
        "supervision": supervision_cfg,
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
    answer_reward_override = _to_plain_mapping(
        cfg["answer_reward"],
        field_name="training_cfg.answer_reward",
    )
    _raise_on_removed_answer_reward_keys(answer_reward_override)
    cfg["answer_reward"] = _deep_merge(
        base=_DEFAULT_TRAINING_CFG["answer_reward"],
        override=answer_reward_override,
    )
    _validate_non_negative(
        cfg=cfg["answer_reward"],
        field_names=(
            "hit_bonus",
            "frontier_bonus",
            "coverage_bonus",
            "size_penalty",
            "component_penalty",
        ),
        prefix="training.answer_reward",
    )
    detailed_balance_cfg = _to_plain_mapping(
        cfg.get("detailed_balance", {}),
        field_name="training_cfg.detailed_balance",
    )
    if detailed_balance_cfg:
        raise ValueError(
            "training.detailed_balance does not accept additional loss-shaping keys; "
            "remove them to use the paper-aligned single-step Detailed Balance objective."
        )
    cfg["detailed_balance"] = {}
    cfg["auxiliary"] = normalize_auxiliary_cfg(cfg.get("auxiliary", {}))
    return cfg


__all__ = [
    "normalize_auxiliary_cfg",
    "normalize_training_action_pruning_cfg",
    "normalize_training_cfg",
]
