from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

DEFAULT_COMPOSITE_ENABLED = False
DEFAULT_COMPOSITE_WEIGHT_CONTEXT_HIT = 0.6
DEFAULT_COMPOSITE_WEIGHT_TERMINAL_HIT = 0.3
DEFAULT_COMPOSITE_WEIGHT_PASS_BEST = 0.1


@dataclass(frozen=True)
class CompositeScoreConfig:
    enabled: bool = DEFAULT_COMPOSITE_ENABLED
    weight_context_hit: float = DEFAULT_COMPOSITE_WEIGHT_CONTEXT_HIT
    weight_terminal_hit: float = DEFAULT_COMPOSITE_WEIGHT_TERMINAL_HIT
    weight_pass_best: float = DEFAULT_COMPOSITE_WEIGHT_PASS_BEST

    @property
    def weight_sum(self) -> float:
        return float(
            self.weight_context_hit + self.weight_terminal_hit + self.weight_pass_best
        )


def resolve_composite_score_cfg(raw_cfg: Optional[Any]) -> CompositeScoreConfig:
    if isinstance(raw_cfg, CompositeScoreConfig):
        return raw_cfg
    if raw_cfg is None:
        return CompositeScoreConfig()
    cfg = raw_cfg if isinstance(raw_cfg, Mapping) else {}
    enabled = bool(cfg.get("enabled", DEFAULT_COMPOSITE_ENABLED))
    weight_context = float(
        cfg.get("weight_context_hit", DEFAULT_COMPOSITE_WEIGHT_CONTEXT_HIT)
    )
    weight_terminal = float(
        cfg.get("weight_terminal_hit", DEFAULT_COMPOSITE_WEIGHT_TERMINAL_HIT)
    )
    weight_pass_best = float(
        cfg.get("weight_pass_best", DEFAULT_COMPOSITE_WEIGHT_PASS_BEST)
    )
    return CompositeScoreConfig(
        enabled=enabled,
        weight_context_hit=weight_context,
        weight_terminal_hit=weight_terminal,
        weight_pass_best=weight_pass_best,
    )


__all__ = ["CompositeScoreConfig", "resolve_composite_score_cfg"]
