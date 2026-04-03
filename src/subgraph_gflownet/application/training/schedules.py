from __future__ import annotations

from collections.abc import Mapping


def resolve_action_pruning_cfg(
    training_cfg: Mapping[str, object],
) -> Mapping[str, int] | None:
    action_pruning_cfg = training_cfg.get("action_pruning")
    if action_pruning_cfg is None:
        return None
    if not isinstance(action_pruning_cfg, Mapping):
        raise TypeError("training_cfg.action_pruning must be a mapping when provided.")
    per_node_top_k = int(action_pruning_cfg.get("per_node_top_k", 0))
    per_state_top_k = int(action_pruning_cfg.get("per_state_top_k", 0))
    if per_node_top_k <= 0 and per_state_top_k <= 0:
        return None
    return {
        "per_node_top_k": per_node_top_k,
        "per_state_top_k": per_state_top_k,
    }


def resolve_supervision_phase(
    supervision_cfg: Mapping[str, object], *, current_step: int
) -> dict[str, float | bool]:
    if not bool(supervision_cfg.get("enabled", True)):
        return {
            "enabled": False,
            "warmup_active": False,
            "db_weight": 1.0,
            "imitation_weight": 0.0,
            "success_action_weight": 0.0,
        }
    warmup_steps = int(supervision_cfg.get("warmup_steps", 0))
    warmup_active = int(current_step) < int(warmup_steps)
    return {
        "enabled": True,
        "warmup_active": bool(warmup_active),
        "db_weight": float(
            supervision_cfg["warmup_db_weight"]
            if warmup_active
            else supervision_cfg["db_weight"]
        ),
        "imitation_weight": float(supervision_cfg["imitation_weight"]),
        "success_action_weight": float(supervision_cfg["success_action_weight"]),
    }


__all__ = ["resolve_action_pruning_cfg", "resolve_supervision_phase"]
