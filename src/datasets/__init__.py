"""Runtime dataset API exposed to Lightning & Hydra configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "GraphRetrievalDataset",
    "create_graph_retrieval_dataset",
    "GraphRetrievalDataModule",
    "EmbeddingStore",
    "GlobalEmbeddingStore",
    "SharedDataResources",
]

if TYPE_CHECKING:  # pragma: no cover
    from .components import EmbeddingStore, GlobalEmbeddingStore, SharedDataResources
    from .graph_retrieval_datamodule import GraphRetrievalDataModule
    from .graph_retrieval_dataset import GraphRetrievalDataset, create_graph_retrieval_dataset


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in ("SharedDataResources", "EmbeddingStore", "GlobalEmbeddingStore"):
        from .components import (
            EmbeddingStore,
            GlobalEmbeddingStore,
            SharedDataResources,
        )

        return {
            "SharedDataResources": SharedDataResources,
            "EmbeddingStore": EmbeddingStore,
            "GlobalEmbeddingStore": GlobalEmbeddingStore,
        }[name]

    if name in ("GraphRetrievalDataset", "create_graph_retrieval_dataset"):
        from .graph_retrieval_dataset import GraphRetrievalDataset, create_graph_retrieval_dataset

        return {
            "GraphRetrievalDataset": GraphRetrievalDataset,
            "create_graph_retrieval_dataset": create_graph_retrieval_dataset,
        }[name]

    if name == "GraphRetrievalDataModule":
        from .graph_retrieval_datamodule import GraphRetrievalDataModule

        return GraphRetrievalDataModule

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
