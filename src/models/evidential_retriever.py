from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .base_retriever import BaseRetriever
from .components import DenseFeatureExtractor, EvidentialHead
from .outputs import (
    ProbabilisticOutput,
    beta_expected_entropy,
    beta_mutual_information,
)
from .registry import register_retriever

logger = logging.getLogger(__name__)


@register_retriever(
    "evidential",
    description="Evidential retriever built on deterministic backbone",
)
class EvidentialRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        lambda_prior_pos: float = 1.0,
        lambda_prior_neg: float = 1.0,
        hidden_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lambda_prior_pos = float(lambda_prior_pos)
        self.lambda_prior_neg = float(lambda_prior_neg)
        self.lambda_prior = 0.5 * (self.lambda_prior_pos + self.lambda_prior_neg)
        self._alpha_beta_eps = 1e-6
        self._mu_eps = 1e-6

        input_feature_dim = self.get_input_feature_dim()
        hidden = hidden_dim or max(self.emb_dim // 2, 128)
        self.feature_extractor = DenseFeatureExtractor(
            input_dim=input_feature_dim,
            emb_dim=self.emb_dim,
            hidden_dim=hidden,
            dropout_p=self.dropout_p,
        )
        self.evidence_head = EvidentialHead(hidden)
        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            for module in self.feature_extractor.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.zeros_(module.bias)
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
        evidence_logits = self.evidence_head(hidden)
        v_pos, v_neg = evidence_logits.unbind(dim=-1)
        s_pos = F.softplus(v_pos)
        s_neg = F.softplus(v_neg)
        alpha = torch.clamp(self.lambda_prior_pos + s_pos, min=self._alpha_beta_eps)
        beta = torch.clamp(self.lambda_prior_neg + s_neg, min=self._alpha_beta_eps)
        total = alpha + beta
        mu = torch.clamp(alpha / total, self._mu_eps, 1.0 - self._mu_eps)
        aleatoric = beta_expected_entropy(alpha, beta)
        epistemic = beta_mutual_information(alpha, beta)

        return ProbabilisticOutput(
            scores=mu,
            query_ids=query_ids,
            evidence_logits=evidence_logits,
            alpha=alpha,
            beta=beta,
            lambda_prior=self.lambda_prior,
            lambda_prior_pos=self.lambda_prior_pos,
            lambda_prior_neg=self.lambda_prior_neg,
            aleatoric=aleatoric,
            epistemic=epistemic,
        )

    def retrieve(
        self,
        data: Batch,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            model_output = self.forward(data, **kwargs)
            model_output.ensure_moments(self._mu_eps)
            ranking_scores = model_output.scores

            groups, orders = self.group_and_rank(ranking_scores, model_output.query_ids, k=k)
            ranked_local = [order.detach().cpu().tolist() for order in orders]
            ranked_global = [
                group[order].detach().cpu().tolist() for group, order in zip(groups, orders)
            ]

            final_scores = self.ranker.gather(model_output.scores, groups, orders)  # type: ignore[arg-type]
            aleatoric_vals = self.ranker.gather(model_output.aleatoric, groups, orders)  # type: ignore[arg-type]
            epistemic_vals = self.ranker.gather(model_output.epistemic, groups, orders) if model_output.epistemic is not None else None  # type: ignore[arg-type]
            alpha_vals = self.ranker.gather(model_output.alpha, groups, orders)  # type: ignore[arg-type]
            beta_vals = self.ranker.gather(model_output.beta, groups, orders)  # type: ignore[arg-type]

            return self._format_retrieve_output(
                ranked_local_indices=ranked_local,
                ranked_global_indices=ranked_global,
                probabilities=final_scores,
                aleatoric_uncertainties=aleatoric_vals,
                epistemic_uncertainties=epistemic_vals,
                alpha=alpha_vals,
                beta=beta_vals,
                sample_ids=data.sample_id,
            )

    def _format_retrieve_output(
        self,
        *,
        ranked_local_indices: List[List[int]],
        ranked_global_indices: List[List[int]],
        probabilities: List[List[float]],
        aleatoric_uncertainties: List[List[float]],
        epistemic_uncertainties: List[List[float]] | None,
        alpha: List[List[float]],
        beta: List[List[float]],
        sample_ids: List[str],
    ) -> Dict[str, Any]:
        return {
            "ranked_local_indices": ranked_local_indices,
            "ranked_global_indices": ranked_global_indices,
            "probabilities": probabilities,
            "aleatoric_uncertainties": aleatoric_uncertainties,
            "epistemic_uncertainties": epistemic_uncertainties,
            "alpha": alpha,
            "beta": beta,
            "metadata": {
                "model_type": "evidential",
                "sample_ids": sample_ids,
            },
        }


__all__ = ["EvidentialRetriever"]
