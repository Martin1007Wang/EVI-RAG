from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "ActionHead",
    "BackboneOutput",
    "ExpandEdgeScorer",
    "FlowHead",
    "NBFBackbone",
    "ZHead",
]

if TYPE_CHECKING:  # pragma: no cover
    from .backbone import BackboneOutput, NBFBackbone
    from .heads import ActionHead, ExpandEdgeScorer, FlowHead, ZHead


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"BackboneOutput", "NBFBackbone"}:
        from .backbone import BackboneOutput, NBFBackbone

        return {"BackboneOutput": BackboneOutput, "NBFBackbone": NBFBackbone}[name]
    if name in {"ActionHead", "ExpandEdgeScorer", "FlowHead", "ZHead"}:
        from .heads import ActionHead, ExpandEdgeScorer, FlowHead, ZHead

        return {
            "ActionHead": ActionHead,
            "ExpandEdgeScorer": ExpandEdgeScorer,
            "FlowHead": FlowHead,
            "ZHead": ZHead,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
