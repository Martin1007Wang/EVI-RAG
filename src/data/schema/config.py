from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    """Placeholder config schema for future pydantic validation."""

    dataset_name: Optional[str] = None
