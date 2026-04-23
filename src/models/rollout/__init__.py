from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["RolloutBatch", "RolloutEngine", "StateReconstructor"]

if TYPE_CHECKING:  # pragma: no cover
    from .engine import RolloutEngine
    from .reconstruct import StateReconstructor
    from .types import RolloutBatch


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "RolloutBatch":
        from .types import RolloutBatch

        return RolloutBatch
    if name == "RolloutEngine":
        from .engine import RolloutEngine

        return RolloutEngine
    if name == "StateReconstructor":
        from .reconstruct import StateReconstructor

        return StateReconstructor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
