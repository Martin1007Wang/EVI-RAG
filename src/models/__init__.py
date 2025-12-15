from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["RetrieverModule", "GFlowNetModule", "LLMReasonerModule"]

if TYPE_CHECKING:  # pragma: no cover
    from .gflownet_module import GFlowNetModule
    from .llm_reasoner_module import LLMReasonerModule
    from .retriever_module import RetrieverModule


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "RetrieverModule":
        from .retriever_module import RetrieverModule

        return RetrieverModule
    if name == "GFlowNetModule":
        from .gflownet_module import GFlowNetModule

        return GFlowNetModule
    if name == "LLMReasonerModule":
        from .llm_reasoner_module import LLMReasonerModule

        return LLMReasonerModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
