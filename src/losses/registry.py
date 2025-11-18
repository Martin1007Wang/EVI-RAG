from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from src.utils.registry import Registry


@dataclass(frozen=True)
class LossMetadata:
    name: str
    description: str
    family: str
    aliases: tuple[str, ...]


LOSS_REGISTRY: Registry[type] = Registry("loss")
_LOSS_METADATA: Dict[str, LossMetadata] = {}


def register_loss(
    name: str,
    *,
    description: str,
    family: str,
    aliases: Optional[Iterable[str]] = None,
) -> callable:
    alias_list = tuple(sorted({a.strip().lower() for a in (aliases or []) if a}))

    def decorator(cls: type) -> type:
        LOSS_REGISTRY.register(name, cls)
        key = name.strip().lower()
        _LOSS_METADATA[key] = LossMetadata(
            name=key,
            description=description,
            family=family,
            aliases=alias_list,
        )
        for alias in alias_list:
            LOSS_REGISTRY.register(alias, cls)
            if alias not in _LOSS_METADATA:
                _LOSS_METADATA[alias] = LossMetadata(
                    name=alias,
                    description=f"Alias of {key}: {description}",
                    family=family,
                    aliases=(),
                )
        return cls

    return decorator


def create_loss_function(config: Dict[str, Any]) -> object:
    """Instantiate a registered loss from a plain dictionary config."""
    cfg = dict(config)
    loss_type = str(cfg.pop("type", "")).strip().lower()
    if not loss_type:
        available = ", ".join(sorted(LOSS_REGISTRY.available().keys())) or "<empty>"
        raise ValueError(f"Loss configuration missing 'type'. Available: {available}")
    try:
        return LOSS_REGISTRY.build(loss_type, **cfg)
    except KeyError as exc:  # pragma: no cover - defensive
        available = ", ".join(sorted(LOSS_REGISTRY.available().keys())) or "<empty>"
        raise ValueError(f"Unknown loss '{loss_type}'. Available: {available}") from exc


def get_available_losses() -> Dict[str, str]:
    """Return a human readable summary for CLI tooling."""
    summary: Dict[str, str] = {}
    for key, info in sorted(_LOSS_METADATA.items(), key=lambda item: item[0]):
        if key in summary:
            continue
        alias_note = f" (aliases: {', '.join(info.aliases)})" if info.aliases else ""
        summary[key] = f"[{info.family}] {info.description}{alias_note}"
    return summary


__all__ = ["LOSS_REGISTRY", "register_loss", "create_loss_function", "get_available_losses", "LossMetadata"]
