# src/losses/retriever_loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.components.retriever import RetrieverOutput

@dataclass
class LossOutput:
    loss: torch.Tensor
    components: Dict[str, float]
    metrics: Dict[str, float]

class RetrieverBCELoss(nn.Module):
    """
    Standard Binary Cross Entropy with Logits.
    Targets are expected to be 0 or 1.
    """
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        # If dataset is heavily imbalanced, pos_weight > 1 helps
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(
        self, 
        output: RetrieverOutput, 
        targets: torch.Tensor, 
        training_step: int = 0
    ) -> LossOutput:
        
        # 1. Sanity Check
        logits = output.logits
        if logits is None:
            # Fallback if model only outputs scores (probabilities)
            # Inverse sigmoid is unstable, so prefer logits if available.
            logits = torch.logit(torch.clamp(output.scores, 1e-6, 1-1e-6))
            
        # 2. Compute BCE
        # reduction='mean' is standard. 
        # For 'pos_weight', we need to reshape it if we used it.
        # But simpler F.binary_cross_entropy_with_logits handles pos_weight automatically if passed.
        
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets.float(), 
            pos_weight=self.pos_weight if self.pos_weight != 1.0 else None,
            reduction="mean"
        )

        # 3. Metrics for monitoring
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pos_mask = targets > 0.5
            pos_avg = probs[pos_mask].mean() if pos_mask.any() else 0.0
            neg_avg = probs[~pos_mask].mean() if (~pos_mask).any() else 0.0
            
        return LossOutput(
            loss=loss,
            components={}, # No aux terms
            metrics={
                "pos_prob": float(pos_avg),
                "neg_prob": float(neg_avg),
                "separation": float(pos_avg - neg_avg)
            }
        )

# Factory function compatible with your Hydra setup
def create_loss_function(cfg: Dict[str, Any] | DictConfig) -> RetrieverBCELoss:
    # We ignore 'type' because we only have one type now.
    # Or keep a simple switch if you plan to add InfoNCE back later.
    
    # Simple conversion
    if isinstance(cfg, DictConfig):
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(cfg, resolve=True)
    
    pos_weight = cfg.get("pos_weight", 1.0)
    return RetrieverBCELoss(pos_weight=pos_weight)
