from __future__ import annotations

from collections import OrderedDict
from fnmatch import fnmatch
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch import nn
from torch.optim import Optimizer
import torch.distributed as dist

try:  # Optional Hydra dependency
    from omegaconf import DictConfig, ListConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf is optional
    DictConfig = None  # type: ignore[assignment]
    ListConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


def setup_optimizer(module: nn.Module, optimizer_cfg: Mapping[str, Any] | None) -> Optimizer:
    """Instantiate a torch optimizer from a simple config dictionary."""
    if module is None:
        raise ValueError("setup_optimizer requires a valid nn.Module.")

    cfg = _to_plain_object(optimizer_cfg) or {}
    if not isinstance(cfg, Mapping):
        raise TypeError(f"optimizer_cfg must be a mapping, got {type(cfg)}.")

    cfg = dict(cfg)
    opt_type = str(cfg.pop("type", cfg.pop("name", "adamw"))).lower()
    param_groups_cfg = cfg.pop("param_groups", None)

    params = _build_param_groups(module, param_groups_cfg)
    optimizer_cls = _resolve_optimizer_class(opt_type)
    return optimizer_cls(params, **cfg)


def _to_plain_object(obj: Any) -> Any:
    if OmegaConf is None or obj is None:
        return obj
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _resolve_optimizer_class(name: str):
    name = name.lower()
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "rmsprop": torch.optim.RMSprop,
        "nadam": getattr(torch.optim, "NAdam", torch.optim.Adam),
        "lbfgs": torch.optim.LBFGS,
    }
    if name in optimizers:
        return optimizers[name]
    if name in {"muon", "singledevicemuon", "single_device_muon", "distributedmuon", "muon_distributed"}:
        force_single = name in {"singledevicemuon", "single_device_muon"}
        force_distributed = name in {"distributedmuon", "muon_distributed"}

        def _factory(params, **cfg):
            return _create_muon_optimizer(
                params,
                force_single=force_single,
                force_distributed=force_distributed,
                **cfg,
            )

        return _factory

    for attr in dir(torch.optim):
        cls = getattr(torch.optim, attr)
        if isinstance(cls, type) and issubclass(cls, Optimizer) and attr.lower() == name:
            return cls
    raise ValueError(f"Unsupported optimizer type '{name}'.")


def _build_param_groups(module: nn.Module, groups_cfg: Any):
    named_params = OrderedDict(
        (name, param) for name, param in module.named_parameters() if param.requires_grad
    )
    if not named_params:
        raise ValueError("Module has no trainable parameters.")

    if not groups_cfg:
        return list(named_params.values())

    normalized_groups = _normalize_group_config(groups_cfg)
    assigned: set[str] = set()
    built_groups: List[Dict[str, Any]] = []

    for raw_cfg in normalized_groups:
        group_cfg = dict(raw_cfg or {})
        patterns = group_cfg.pop("params", None) or group_cfg.pop("names", None)
        if not patterns:
            continue

        if isinstance(patterns, str):
            patterns = [patterns]
        elif isinstance(patterns, Sequence):
            patterns = list(patterns)
        else:
            raise TypeError("param group patterns must be a string or sequence of strings.")

        matched = []
        for name, param in named_params.items():
            if name in assigned:
                continue
            if _matches_any(name, patterns):
                matched.append(param)
                assigned.add(name)

        if not matched:
            continue

        overrides = {k: v for k, v in group_cfg.items() if k not in {"params", "names"}}
        overrides["params"] = matched
        built_groups.append(overrides)

    remaining = [param for name, param in named_params.items() if name not in assigned]
    if remaining:
        built_groups.append({"params": remaining})

    if not built_groups:
        raise ValueError("Param group configuration did not match any parameters.")

    return built_groups


def _normalize_group_config(config: Any) -> List[Mapping[str, Any]]:
    config = _to_plain_object(config)
    if config is None:
        return []

    if isinstance(config, Mapping):
        if "params" in config or "names" in config:
            return [config]
        return [group for group in config.values() if isinstance(group, Mapping)]

    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        groups: List[Mapping[str, Any]] = []
        for group in config:
            if isinstance(group, Mapping):
                groups.append(group)
        return groups

    raise TypeError("param_groups must be a mapping or sequence of mappings.")


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(_match_pattern(name, pattern) for pattern in patterns)


def _match_pattern(name: str, pattern: str) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return False
    if any(token in pattern for token in ("*", "?", "[")):
        return fnmatch(name, pattern)
    if name == pattern:
        return True
    if name.startswith(f"{pattern}."):
        return True
    if name.endswith(f".{pattern}") or name.endswith(pattern):
        return True
    return False


def _create_muon_optimizer(
    params,
    *,
    force_single: bool = False,
    force_distributed: bool = False,
    **cfg,
):
    try:
        from muon import Muon, SingleDeviceMuon  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Muon optimizer requested but the 'muon-optimizer' package is not installed. "
            "Install it via `pip install git+https://github.com/KellerJordan/Muon`."
        ) from exc

    if force_single and force_distributed:
        raise ValueError("Cannot force both distributed and single-device Muon modes.")

    if not isinstance(params, list):
        params = list(params)

    if not params:
        raise ValueError("Muon optimizer received an empty parameter list.")

    if isinstance(params[0], dict):
        raise ValueError(
            "Muon optimizer does not support param group dictionaries. "
            "Remove `param_groups` from the optimizer config or instantiate "
            "`MuonWithAuxAdam` manually for fine-grained control."
        )

    allowed_keys = {"lr", "weight_decay", "momentum"}
    unexpected = set(cfg) - allowed_keys
    if unexpected:
        raise ValueError(f"Unsupported Muon optimizer arguments: {sorted(unexpected)}")

    if force_distributed:
        cls = Muon
    elif force_single:
        cls = SingleDeviceMuon
    else:
        use_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        cls = Muon if use_distributed else SingleDeviceMuon

    sorted_params = sorted(params, key=lambda param: param.numel(), reverse=True)
    return cls(sorted_params, **cfg)


__all__ = ["setup_optimizer"]
