from __future__ import annotations

from typing import Any, Protocol, cast

import torch
from lightning import LightningModule
from omegaconf import DictConfig


class PretrainedLoadFn(Protocol):
    def __call__(
        self,
        path: str,
        *,
        strict: bool,
    ) -> tuple[Any, Any]:
        ...


def load_checkpoint_weights(
    model: LightningModule,
    checkpoint_path: str,
    *,
    strict: bool,
) -> tuple[list[str], list[str]]:
    load_fn_obj = getattr(model, "load_pretrained_weights", None)
    if callable(load_fn_obj):
        load_fn = cast(PretrainedLoadFn, load_fn_obj)
        missing, unexpected = load_fn(str(checkpoint_path), strict=strict)
        return list(missing), list(unexpected)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    result = model.load_state_dict(state_dict, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


def filter_compatible_state_dict(
    *,
    state_dict: dict[str, Any],
    current_state: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    if strict:
        return state_dict

    filtered: dict[str, Any] = {}
    reset_stop_gate = False

    for key, value in state_dict.items():
        current_value = current_state.get(key)
        if (
            isinstance(value, torch.Tensor)
            and isinstance(current_value, torch.Tensor)
            and value.shape != current_value.shape
        ):
            if key.startswith("policy.stop_gate.net.0."):
                reset_stop_gate = True
            continue
        filtered[key] = value

    if reset_stop_gate:
        stop_gate_prefixes = (
            "policy.stop_gate.net.",
            "policy.stop_gate.stop_bias",
            "policy.stop_gate.expand_bias",
        )
        filtered = {
            key: value
            for key, value in filtered.items()
            if not key.startswith(stop_gate_prefixes)
        }

    return filtered


def load_pretrained_if_requested(cfg: DictConfig, model: LightningModule) -> None:
    checkpoint_path = cfg.get("pretrained_ckpt_path", None)
    if checkpoint_path in (None, ""):
        return

    strict = bool(cfg.get("strict_pretrained_load", False))
    missing, unexpected = load_checkpoint_weights(
        model,
        str(checkpoint_path),
        strict=strict,
    )

    print(
        f"Loaded pretrained weights from {checkpoint_path!r}; "
        f"missing={missing}, unexpected={unexpected}"
    )
