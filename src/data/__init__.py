"""Preprocess pipeline surface for retrieval datasets.

Keep this module import-lightweight: heavy dependencies (Hydra/OmegaConf,
datasets, etc.) are imported lazily to avoid side effects in small utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "PreprocessContext",
    "run_preprocess_pipeline",
]

if TYPE_CHECKING:  # pragma: no cover
    from .preprocess.context import PreprocessContext
    from .preprocess.main import run_preprocess_pipeline


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "PreprocessContext":
        from .preprocess.context import PreprocessContext

        return PreprocessContext
    if name == "run_preprocess_pipeline":
        from .preprocess.main import run_preprocess_pipeline

        return run_preprocess_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
