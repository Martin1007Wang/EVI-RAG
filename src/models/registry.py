from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from src.utils.registry import Registry

RETRIEVER_REGISTRY: Registry[type] = Registry("retriever")
_MODEL_INFO: Dict[str, str] = {}


def register_retriever(
    name: str,
    *,
    description: str = "",
    aliases: Optional[Iterable[str]] = None,
) -> callable:
    alias_list = list(aliases or [])

    def decorator(cls: type) -> type:
        RETRIEVER_REGISTRY.register(name, cls)
        key = name.strip().lower()
        if description:
            _MODEL_INFO[key] = description
        for alias in alias_list:
            RETRIEVER_REGISTRY.register(alias, cls)
            alias_key = alias.strip().lower()
            if description and alias_key not in _MODEL_INFO:
                _MODEL_INFO[alias_key] = f"Alias of {name}: {description}"
        return cls

    return decorator


def create_retriever(config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> object:
    cfg = dict(config or {})
    cfg.update(kwargs)
    model_type = str(cfg.pop("model_type", "")).strip().lower()
    if not model_type:
        available = ", ".join(RETRIEVER_REGISTRY.available().keys()) or "<empty>"
        raise ValueError(f"Configuration missing 'model_type'. Available: {available}")
    try:
        return RETRIEVER_REGISTRY.build(model_type, **cfg)
    except KeyError as exc:
        available = ", ".join(RETRIEVER_REGISTRY.available().keys()) or "<empty>"
        raise ValueError(f"Unknown retriever '{model_type}'. Available: {available}") from exc


def get_model_info() -> Dict[str, str]:
    return dict(sorted(_MODEL_INFO.items(), key=lambda item: item[0]))


__all__ = ["RETRIEVER_REGISTRY", "register_retriever", "create_retriever", "get_model_info"]
