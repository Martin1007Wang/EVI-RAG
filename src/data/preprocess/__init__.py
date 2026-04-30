"""Internal preprocessing stage modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["run_preprocess_pipeline"]

if TYPE_CHECKING:  # pragma: no cover
    from src.data.pipeline import run_preprocess_pipeline


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "run_preprocess_pipeline":
        from src.data.pipeline import run_preprocess_pipeline

        return run_preprocess_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
