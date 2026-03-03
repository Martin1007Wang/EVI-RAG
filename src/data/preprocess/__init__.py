"""Preprocess (ETL) pipeline surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "PreprocessContext",
    "run_preprocess_pipeline",
]

if TYPE_CHECKING:  # pragma: no cover
    from .context import PreprocessContext
    from .main import run_preprocess_pipeline


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "PreprocessContext":
        from .context import PreprocessContext

        return PreprocessContext
    if name == "run_preprocess_pipeline":
        from .main import run_preprocess_pipeline

        return run_preprocess_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
