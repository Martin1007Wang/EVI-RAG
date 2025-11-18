from __future__ import annotations

from typing import Iterable, List, Optional

import torch


class GroupRanker:
    """Utility that groups edges by query id and ranks them by score."""

    def group(self, query_ids: torch.Tensor) -> list[torch.Tensor]:
        ids_cpu = query_ids.detach().cpu().tolist()
        order: list[int] = []
        mapping: dict[int, int] = {}
        groups: list[list[int]] = []

        for idx, qid in enumerate(ids_cpu):
            if qid not in mapping:
                mapping[qid] = len(groups)
                groups.append([])
                order.append(qid)
            groups[mapping[qid]].append(idx)

        device = query_ids.device
        return [torch.tensor(g, device=device, dtype=torch.long) for g in groups]

    def rank(
        self,
        scores: torch.Tensor,
        groups: Iterable[torch.Tensor],
        *,
        k: Optional[int] = None,
    ) -> list[torch.Tensor]:
        ranked: list[torch.Tensor] = []
        for group in groups:
            group_scores = scores[group]
            _, local_sorted = torch.sort(group_scores, descending=True)
            if k is not None:
                local_sorted = local_sorted[:k]
            ranked.append(local_sorted)
        return ranked

    def gather(
        self,
        tensor: torch.Tensor,
        groups: Iterable[torch.Tensor],
        orders: Iterable[torch.Tensor],
    ) -> List[List[float]]:
        gathered: List[List[float]] = []
        for group, order in zip(groups, orders):
            values = tensor[group][order]
            gathered.append(values.detach().cpu().tolist())
        return gathered


__all__ = ["GroupRanker"]
