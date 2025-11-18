from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import Batch

from .base_retriever import BaseRetriever
from .components import DenseFeatureExtractor, DeterministicHead
from .outputs import DeterministicOutput
from .registry import register_retriever

logger = logging.getLogger(__name__)


@register_retriever(
    "deterministic",
    description="Deterministic scoring head with sigmoid calibration",
    aliases=("det",),
)
class DeterministicRetriever(BaseRetriever):
    def __init__(self, *, hidden_dim: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        input_feature_dim = self.get_input_feature_dim()
        hidden = hidden_dim or max(self.emb_dim // 2, 128)
        self.feature_extractor = DenseFeatureExtractor(
            input_dim=input_feature_dim,
            emb_dim=self.emb_dim,
            hidden_dim=hidden,
            dropout_p=self.dropout_p,
        )
        self.score_head = DeterministicHead(hidden)

    def _score_triples(
        self,
        data: Batch,
        base_semantic_features_override: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> DeterministicOutput:
        q_emb, h_emb, r_emb, t_emb, query_ids = self._prepare_embeddings(
            data,
            base_semantic_features_override=base_semantic_features_override,
            align_q_to_edges=True,
        )
        triple_repr = self._compute_triple_representation(h_emb, r_emb, t_emb)
        features = torch.cat([q_emb, triple_repr], dim=1)
        extracted = self.feature_extractor(features)
        logits = self.score_head(extracted)
        probs = torch.sigmoid(logits)
        return DeterministicOutput(scores=probs, query_ids=query_ids, logits=logits)

    def retrieve(
        self,
        data: Batch,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            model_output = self.forward(data, **kwargs)
            ranking_scores = model_output.scores
            groups, orders = self.group_and_rank(ranking_scores, model_output.query_ids, k=k)

            ranked_local = [order.detach().cpu().tolist() for order in orders]
            probabilities = self.ranker.gather(model_output.scores, groups, orders)
            aleatoric = [[0.0] * len(sample) for sample in probabilities]
            return self._format_retrieve_output(
                ranked_local_indices=ranked_local,
                probabilities=probabilities,
                aleatoric_uncertainties=aleatoric,
                sample_ids=data.sample_id,
            )

    def _format_retrieve_output(
        self,
        *,
        ranked_local_indices: List[List[int]],
        probabilities: List[List[float]],
        aleatoric_uncertainties: List[List[float]],
        sample_ids: List[str],
    ) -> Dict[str, Any]:
        return {
            "ranked_local_indices": ranked_local_indices,
            "probabilities": probabilities,
            "aleatoric_uncertainties": aleatoric_uncertainties,
            "epistemic_uncertainties": None,
            "alpha": None,
            "beta": None,
            "metadata": {
                "model_type": "deterministic",
                "sample_ids": sample_ids,
            },
        }


__all__ = ["DeterministicRetriever"]
