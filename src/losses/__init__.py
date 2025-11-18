"""Unified loss package for retriever training."""

from .base import BaseLoss, ContrastiveLoss, DeterministicLoss, PointwiseLoss, ProbabilisticLoss
from .output import LossOutput
from .registry import (
    LOSS_REGISTRY,
    create_loss_function,
    get_available_losses,
    register_loss,
)
from .evidential import EDLLoss, REDLLoss, Type2Loss, VERALiteLoss
from .hybrid import HybridVERALoss

try:  # Optional deterministic losses (torch_scatter dependency)
    from .deterministic import DeterministicBCELoss, DeterministicInfoNCELoss, DeterministicMSELoss
except Exception:  # pragma: no cover - keep package importable without torch_scatter
    DeterministicBCELoss = DeterministicInfoNCELoss = DeterministicMSELoss = None  # type: ignore[assignment]

__all__ = [
    "BaseLoss",
    "PointwiseLoss",
    "DeterministicLoss",
    "ProbabilisticLoss",
    "ContrastiveLoss",
    "LossOutput",
    "LOSS_REGISTRY",
    "register_loss",
    "create_loss_function",
    "get_available_losses",
    "EDLLoss",
    "REDLLoss",
    "Type2Loss",
    "VERALiteLoss",
    "HybridVERALoss",
    "DeterministicBCELoss",
    "DeterministicMSELoss",
    "DeterministicInfoNCELoss",
]
