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
    "GAgentPyGDataset",
    "GAgentDataModule",
    "SelectorEvalDataModule",
    "ReasonerPathDataset",
    "ReasonerPathDataModule",
    "ReasonerTripletDataModule",
    "ReasonerOracleDataModule",
]

if TYPE_CHECKING:  # pragma: no cover
    from .components import EmbeddingStore, GlobalEmbeddingStore, GraphStore, SharedDataResources
    from .components.loader import UnifiedDataLoader
    from .g_agent_datamodule import GAgentDataModule
    from .g_agent_dataset import GAgentPyGDataset
    from .g_retrieval_datamodule import GRetrievalDataModule
    from .g_retrieval_dataset import GRetrievalDataset, create_g_retrieval_dataset
    from .selector_eval_datamodule import SelectorEvalDataModule
    from .reasoner_path_datamodule import ReasonerPathDataModule
    from .reasoner_path_dataset import ReasonerPathDataset
    from .reasoner_triplet_datamodule import ReasonerTripletDataModule
    from .reasoner_oracle_datamodule import ReasonerOracleDataModule


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

    if name == "GAgentPyGDataset":
        from .g_agent_dataset import GAgentPyGDataset

        return GAgentPyGDataset

    if name == "GAgentDataModule":
        from .g_agent_datamodule import GAgentDataModule

        return GAgentDataModule

    if name == "SelectorEvalDataModule":
        from .selector_eval_datamodule import SelectorEvalDataModule

        return SelectorEvalDataModule

    if name == "ReasonerPathDataset":
        from .reasoner_path_dataset import ReasonerPathDataset

        return ReasonerPathDataset

    if name == "ReasonerPathDataModule":
        from .reasoner_path_datamodule import ReasonerPathDataModule

        return ReasonerPathDataModule

    if name == "ReasonerTripletDataModule":
        from .reasoner_triplet_datamodule import ReasonerTripletDataModule

        return ReasonerTripletDataModule

    if name == "ReasonerOracleDataModule":
        from .reasoner_oracle_datamodule import ReasonerOracleDataModule

        return ReasonerOracleDataModule

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
