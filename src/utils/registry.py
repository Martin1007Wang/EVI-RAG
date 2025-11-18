from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple name -> callable registry used by the training stack."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._builders: Dict[str, Callable[..., T]] = {}

    def register(self, key: str, builder: Callable[..., T]) -> Callable[..., T]:
        normalized = self._normalize_key(key)
        if normalized in self._builders:
            raise KeyError(f"Object {normalized!r} already registered in registry {self._name!r}")
        self._builders[normalized] = builder
        return builder

    def build(self, key: str, *args, **kwargs) -> T:
        normalized = self._normalize_key(key)
        if normalized not in self._builders:
            available = ", ".join(sorted(self._builders)) or "<empty>"
            raise KeyError(f"Unknown object {normalized!r} requested from registry {self._name!r}. " f"Available: {available}")
        return self._builders[normalized](*args, **kwargs)

    def available(self) -> Dict[str, Callable[..., T]]:
        return dict(self._builders)

    def _normalize_key(self, key: str) -> str:
        if normalized := key.strip().lower():
            return normalized
        else:
            raise ValueError("Registry keys must be non-empty strings")


__all__ = ["Registry"]
