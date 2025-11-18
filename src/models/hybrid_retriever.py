from __future__ import annotations

from typing import Any, Optional, List

import torch
from torch_geometric.data import Batch

from .base_retriever import BaseRetriever
from .components import DenseFeatureExtractor, DeterministicHead, EvidentialHead
from .outputs import ProbabilisticOutput
from .registry import register_retriever


@register_retriever(
    "hybrid",
    description="Hybrid retriever with dedicated ranking and evidential heads",
    aliases=("vera", "vera_hybrid"),
)
class HybridRetriever(BaseRetriever):
    """Retriever that decouples ranking logits from evidential uncertainty."""

    def __init__(
        self,
        *,
        hidden_dim: Optional[int] = None,
        dropout_p: float = 0.1,
        lambda_prior_pos: float = 1.0,
        lambda_prior_neg: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(dropout_p=dropout_p, **kwargs)
        self.lambda_prior_pos = float(lambda_prior_pos)
        self.lambda_prior_neg = float(lambda_prior_neg)
        self.lambda_prior = 0.5 * (self.lambda_prior_pos + self.lambda_prior_neg)
        self._alpha_beta_eps = 1e-6
        input_feature_dim = self.get_input_feature_dim()
        hidden = hidden_dim or max(self.emb_dim // 2, 128)

        self.feature_extractor = DenseFeatureExtractor(
            input_dim=input_feature_dim,
            emb_dim=self.emb_dim,
            hidden_dim=hidden,
            dropout_p=self.dropout_p,
        )
        self.ranking_head = DeterministicHead(hidden)
        self.evidence_head = EvidentialHead(hidden)
        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            for module in self.feature_extractor.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.zeros_(module.bias)
            torch.nn.init.xavier_uniform_(self.ranking_head.linear.weight)
            torch.nn.init.zeros_(self.ranking_head.linear.bias)
            torch.nn.init.xavier_uniform_(self.evidence_head.evidence.weight)
            torch.nn.init.zeros_(self.evidence_head.evidence.bias)

    def _score_triples(
        self,
        data: Batch,
        base_semantic_features_override: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> ProbabilisticOutput:
        q_emb, h_emb, r_emb, t_emb, query_ids = self._prepare_embeddings(
            data,
            base_semantic_features_override=base_semantic_features_override,
            align_q_to_edges=True,
        )
        triple_repr = self._compute_triple_representation(h_emb, r_emb, t_emb)
        combined = torch.cat([q_emb, triple_repr], dim=-1)
        hidden = self.feature_extractor(combined)
        logits = self.ranking_head(hidden)
        scores = torch.sigmoid(logits)
        evidence_logits = self.evidence_head(hidden)

        return ProbabilisticOutput(
            scores=scores,
            logits=logits,
            query_ids=query_ids,
            evidence_logits=evidence_logits,
            lambda_prior=self.lambda_prior,
            lambda_prior_pos=self.lambda_prior_pos,
            lambda_prior_neg=self.lambda_prior_neg,
        )

    def retrieve(
        self,
        data: Batch,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.eval()
        with torch.no_grad():
            model_output = self.forward(data, **kwargs)
            model_output.ensure_moments(self._alpha_beta_eps)
            scores = model_output.scores
            groups, orders = self.group_and_rank(scores, model_output.query_ids, k=k)
            ranked_local = [order.detach().cpu().tolist() for order in orders]
            ranked_global = [
                group[order].detach().cpu().tolist() for group, order in zip(groups, orders)
            ]
            final_scores: List[List[float]] = self.ranker.gather(scores, groups, orders)  # type: ignore[arg-type]
            aleatoric_vals = self.ranker.gather(model_output.aleatoric, groups, orders)  # type: ignore[arg-type]
            epistemic_vals = (
                self.ranker.gather(model_output.epistemic, groups, orders) if model_output.epistemic is not None else None  # type: ignore[arg-type]
            )
            alpha_vals = self.ranker.gather(model_output.alpha, groups, orders)  # type: ignore[arg-type]
            beta_vals = self.ranker.gather(model_output.beta, groups, orders)  # type: ignore[arg-type]

            return {
                "ranked_local_indices": ranked_local,
                "ranked_global_indices": ranked_global,
                "probabilities": final_scores,
                "aleatoric_uncertainties": aleatoric_vals,
                "epistemic_uncertainties": epistemic_vals,
                "alpha": alpha_vals,
                "beta": beta_vals,
                "metadata": {
                    "model_type": "hybrid",
                    "sample_ids": data.sample_id,
                },
            }


__all__ = ["HybridRetriever"]
