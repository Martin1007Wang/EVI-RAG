"""Public data API exposed to Lightning & Hydra configs.

Keep this module import-lightweight: heavy dependencies (reasoner datasets,
Hydra/OmegaConf, etc.) are imported lazily to avoid side effects during unit
tests and simple utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "GRetrievalDataset",
    "create_g_retrieval_dataset",
    "GRetrievalDataModule",
    "UnifiedDataLoader",
    "GraphStore",
    "EmbeddingStore",
    "GlobalEmbeddingStore",
    "SharedDataResources",
]

if TYPE_CHECKING:  # pragma: no cover
    from .components import EmbeddingStore, GlobalEmbeddingStore, GraphStore, SharedDataResources
    from .components.loader import UnifiedDataLoader
    from .g_retrieval_datamodule import GRetrievalDataModule
    from .g_retrieval_dataset import GRetrievalDataset, create_g_retrieval_dataset


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in ("SharedDataResources", "EmbeddingStore", "GlobalEmbeddingStore", "GraphStore"):
        from .components import EmbeddingStore, GlobalEmbeddingStore, GraphStore, SharedDataResources

        return {
            "SharedDataResources": SharedDataResources,
            "EmbeddingStore": EmbeddingStore,
            "GlobalEmbeddingStore": GlobalEmbeddingStore,
            "GraphStore": GraphStore,
        }[name]

    if name == "UnifiedDataLoader":
        from .components.loader import UnifiedDataLoader

        return UnifiedDataLoader

    if name in ("GRetrievalDataset", "create_g_retrieval_dataset"):
        from .g_retrieval_dataset import GRetrievalDataset, create_g_retrieval_dataset

        return {"GRetrievalDataset": GRetrievalDataset, "create_g_retrieval_dataset": create_g_retrieval_dataset}[name]

    if name == "GRetrievalDataModule":
        from .g_retrieval_datamodule import GRetrievalDataModule

        return GRetrievalDataModule

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
