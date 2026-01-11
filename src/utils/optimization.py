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

_AUTO_PARAM_GROUPS_KEY = "auto_param_groups"
_PARAM_GROUP_OVERRIDES_KEY = "param_group_overrides"
_PARAM_GROUP_NAMES_KEYS = {"params", "names"}
_LR_MULT_KEY = "lr_mult"
_NO_WEIGHT_DECAY = 0.0
_DEFAULT_NO_DECAY_MODULE_TYPES = (
    nn.LayerNorm,
    nn.Embedding,
    nn.modules.batchnorm._BatchNorm,
)
_NO_DECAY_MODULE_NAME_MAP = {
    "layernorm": nn.LayerNorm,
    "batchnorm": nn.modules.batchnorm._BatchNorm,
    "embedding": nn.Embedding,
}


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
    auto_groups_cfg = cfg.pop(_AUTO_PARAM_GROUPS_KEY, None)
    override_groups_cfg = cfg.pop(_PARAM_GROUP_OVERRIDES_KEY, None)

    params = _resolve_param_groups(
        module,
        param_groups_cfg,
        auto_groups_cfg,
        override_groups_cfg,
        base_lr=cfg.get("lr"),
        has_weight_decay="weight_decay" in cfg,
    )
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


def _resolve_param_groups(
    module: nn.Module,
    groups_cfg: Any,
    auto_groups_cfg: Any,
    override_groups_cfg: Any,
    *,
    base_lr: Any,
    has_weight_decay: bool,
):
    named_params = _named_trainable_params(module)
    auto_cfg = _normalize_auto_group_config(auto_groups_cfg)
    if override_groups_cfg and groups_cfg:
        raise ValueError("param_group_overrides cannot be combined with param_groups.")
    if override_groups_cfg and auto_cfg is None:
        raise ValueError("param_group_overrides requires auto_param_groups to be enabled.")
    if groups_cfg and auto_cfg is not None:
        raise ValueError("auto_param_groups cannot be combined with explicit param_groups.")
    if groups_cfg:
        return _build_param_groups(named_params, groups_cfg)
    if override_groups_cfg:
        if not has_weight_decay:
            raise ValueError("auto_param_groups requires optimizer_cfg.weight_decay to be set.")
        return _build_param_groups_with_overrides(
            module,
            named_params,
            auto_cfg,
            override_groups_cfg,
            base_lr=base_lr,
        )
    if auto_cfg is None:
        return list(named_params.values())
    if not has_weight_decay:
        raise ValueError("auto_param_groups requires optimizer_cfg.weight_decay to be set.")
    return _build_auto_param_groups(module, named_params, auto_cfg)


def _named_trainable_params(module: nn.Module) -> OrderedDict[str, nn.Parameter]:
    named_params = OrderedDict(
        (name, param) for name, param in module.named_parameters() if param.requires_grad
    )
    if not named_params:
        raise ValueError("Module has no trainable parameters.")
    return named_params


def _build_param_groups(named_params: Mapping[str, nn.Parameter], groups_cfg: Any):
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


def _build_param_groups_with_overrides(
    module: nn.Module,
    named_params: Mapping[str, nn.Parameter],
    auto_cfg: Mapping[str, Any],
    overrides_cfg: Any,
    *,
    base_lr: Any,
) -> List[Dict[str, Any]]:
    normalized_overrides = _normalize_group_config(overrides_cfg)
    assigned: set[str] = set()
    built_groups: List[Dict[str, Any]] = []
    for raw_cfg in normalized_overrides:
        group_cfg = dict(raw_cfg or {})
        patterns = _extract_param_patterns(group_cfg)
        if not patterns:
            continue
        matched = OrderedDict(
            (name, param)
            for name, param in named_params.items()
            if name not in assigned and _matches_any(name, patterns)
        )
        if not matched:
            continue
        assigned.update(matched.keys())
        auto_groups = _build_auto_param_groups(module, matched, auto_cfg)
        _apply_group_overrides(auto_groups, group_cfg, base_lr=base_lr)
        built_groups.extend(auto_groups)
    remaining = OrderedDict((name, param) for name, param in named_params.items() if name not in assigned)
    if remaining:
        built_groups.extend(_build_auto_param_groups(module, remaining, auto_cfg))
    if not built_groups:
        raise ValueError("param_group_overrides did not match any parameters.")
    return built_groups


def _extract_param_patterns(group_cfg: Dict[str, Any]) -> List[str]:
    patterns = None
    for key in _PARAM_GROUP_NAMES_KEYS:
        if key in group_cfg:
            patterns = group_cfg.pop(key)
            break
    if not patterns:
        return []
    if isinstance(patterns, str):
        return [patterns]
    if isinstance(patterns, Sequence) and not isinstance(patterns, (bytes, str)):
        return list(patterns)
    raise TypeError("param group patterns must be a string or sequence of strings.")


def _apply_group_overrides(
    groups: List[Dict[str, Any]],
    override_cfg: Dict[str, Any],
    *,
    base_lr: Any,
) -> None:
    group_cfg = {k: v for k, v in override_cfg.items() if k not in _PARAM_GROUP_NAMES_KEYS}
    lr_mult = group_cfg.pop(_LR_MULT_KEY, None)
    if lr_mult is not None:
        if "lr" in group_cfg:
            raise ValueError("param_group_overrides cannot set both lr and lr_mult.")
        if base_lr is None:
            raise ValueError("param_group_overrides.lr_mult requires optimizer_cfg.lr to be set.")
        group_cfg["lr"] = float(base_lr) * float(lr_mult)
    for group in groups:
        group.update(group_cfg)


def _normalize_auto_group_config(config: Any) -> Mapping[str, Any] | None:
    config = _to_plain_object(config)
    if config is None or config is False:
        return None
    if config is True:
        return {}
    if isinstance(config, Mapping):
        if not config.get("enabled", True):
            return None
        return dict(config)
    raise TypeError("auto_param_groups must be a bool or mapping.")


def _build_auto_param_groups(
    module: nn.Module,
    named_params: Mapping[str, nn.Parameter],
    auto_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    module_types = _resolve_no_decay_module_types(auto_cfg)
    include_bias = _resolve_include_bias(auto_cfg)
    no_decay_param_ids = _collect_no_decay_param_ids(module, module_types)
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    for name, param in named_params.items():
        if _should_exclude_from_weight_decay(name, param, no_decay_param_ids, include_bias):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": _NO_WEIGHT_DECAY})
    if not groups:
        raise ValueError("auto_param_groups produced no parameter groups.")
    return groups


def _resolve_no_decay_module_types(auto_cfg: Mapping[str, Any]) -> tuple[type, ...]:
    names = auto_cfg.get("no_decay_modules")
    if names is None:
        return _DEFAULT_NO_DECAY_MODULE_TYPES
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, Sequence) or isinstance(names, (bytes, str)):
        raise TypeError("auto_param_groups.no_decay_modules must be a string or sequence.")
    resolved: List[type] = []
    unknown: List[str] = []
    for name in names:
        key = str(name).strip().lower()
        module_type = _NO_DECAY_MODULE_NAME_MAP.get(key)
        if module_type is None:
            unknown.append(str(name))
        else:
            resolved.append(module_type)
    if unknown:
        raise ValueError(f"Unknown no_decay_modules entries: {unknown}")
    return tuple(resolved)


def _resolve_include_bias(auto_cfg: Mapping[str, Any]) -> bool:
    include_bias = auto_cfg.get("include_bias", True)
    if not isinstance(include_bias, bool):
        raise TypeError("auto_param_groups.include_bias must be a bool.")
    return include_bias


def _collect_no_decay_param_ids(
    module: nn.Module,
    module_types: Sequence[type],
) -> set[int]:
    param_ids: set[int] = set()
    for submodule in module.modules():
        if isinstance(submodule, tuple(module_types)):
            for param in submodule.parameters(recurse=False):
                if param.requires_grad:
                    param_ids.add(id(param))
    return param_ids


def _should_exclude_from_weight_decay(
    name: str,
    param: nn.Parameter,
    no_decay_param_ids: set[int],
    include_bias: bool,
) -> bool:
    if id(param) in no_decay_param_ids:
        return True
    if include_bias and _is_bias_parameter(name):
        return True
    return False


def _is_bias_parameter(name: str) -> bool:
    if name == "bias":
        return True
    return name.endswith(".bias")


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
            "Disable `auto_param_groups` or remove `param_groups` from the optimizer config, "
            "or instantiate `MuonWithAuxAdam` manually for fine-grained control."
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
