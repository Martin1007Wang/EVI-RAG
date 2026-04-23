from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch, SampleFields
from src.models.policy import CandidateEdges
from src.models.state import State


class TeacherGuidance:
    def __init__(
        self,
        *,
        score_exponent: float,
    ) -> None:
        if score_exponent < 0.0:
            raise ValueError(f"score_exponent must be >= 0, got {score_exponent}.")
        self.score_exponent = float(score_exponent)

    def candidate_scores(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        remaining_expand_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (valid_mask, teacher_scores) aligned with ``candidates``."""
        num_candidates = len(candidates)
        device = candidates.edge_ids.device
        valid_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
        scores = torch.zeros(num_candidates, dtype=torch.float32, device=device)
        if num_candidates == 0 or remaining_expand_budget < 0:
            return valid_mask, scores

        self._require_per_target_teacher_labels(retrieval_batch)

        num_graphs = int(retrieval_batch.ptr.numel()) - 1
        train_target_ids = _get_train_target_node_ids(retrieval_batch)
        if train_target_ids.numel() == 0 or num_graphs <= 0:
            return valid_mask, scores

        ptr = retrieval_batch.ptr.long()
        edge_ptr = retrieval_batch.edge_ptr.long()
        node_counts = ptr[1:] - ptr[:-1]
        edge_counts = edge_ptr[1:] - edge_ptr[:-1]

        target_batch = retrieval_batch.batch.index_select(0, train_target_ids)
        target_counts = torch.bincount(target_batch, minlength=num_graphs).long()
        target_ptr = _exclusive_cumsum(target_counts)
        node_flat_ptr = _exclusive_cumsum(target_counts * node_counts)
        edge_flat_ptr = _exclusive_cumsum(target_counts * edge_counts)

        src_all = retrieval_batch.edge_index[0].index_select(
            0, candidates.edge_ids.long()
        )
        dst_all = retrieval_batch.edge_index[1].index_select(
            0, candidates.edge_ids.long()
        )

        present_graphs = torch.unique(candidates.batch_index, sorted=True)
        for graph_id_tensor in present_graphs:
            graph_id = int(graph_id_tensor.item())
            graph_mask = candidates.batch_index.eq(graph_id)
            target_lo = int(target_ptr[graph_id].item())
            target_hi = int(target_ptr[graph_id + 1].item())
            if target_lo == target_hi:
                continue

            graph_target_ids = train_target_ids[target_lo:target_hi]
            uncovered_targets = ~state.active_nodes.index_select(0, graph_target_ids)
            if not bool(uncovered_targets.any().item()):
                continue

            graph_node_count = int(node_counts[graph_id].item())
            graph_edge_count = int(edge_counts[graph_id].item())
            graph_node_flat = retrieval_batch.target_node_distance_flat[
                int(node_flat_ptr[graph_id].item()) : int(
                    node_flat_ptr[graph_id + 1].item()
                )
            ]
            graph_count_flat = retrieval_batch.target_shortest_path_count_flat[
                int(node_flat_ptr[graph_id].item()) : int(
                    node_flat_ptr[graph_id + 1].item()
                )
            ]
            graph_edge_flat = retrieval_batch.target_shortest_path_edge_mask_flat[
                int(edge_flat_ptr[graph_id].item()) : int(
                    edge_flat_ptr[graph_id + 1].item()
                )
            ]
            graph_target_dist = graph_node_flat.view(-1, graph_node_count)[
                uncovered_targets
            ]
            graph_target_counts = graph_count_flat.view(-1, graph_node_count)[
                uncovered_targets
            ]
            graph_target_edge_mask = graph_edge_flat.view(-1, graph_edge_count)[
                uncovered_targets
            ]

            matched = torch.nonzero(graph_mask, as_tuple=False).view(-1)
            candidate_edge_ids = candidates.edge_ids.index_select(0, matched).long()
            local_edge_ids = candidate_edge_ids - edge_ptr[graph_id]
            local_src = src_all.index_select(0, matched) - ptr[graph_id]
            local_dst = dst_all.index_select(0, matched) - ptr[graph_id]
            src_active = state.active_nodes.index_select(
                0, src_all.index_select(0, matched)
            )
            dst_active = state.active_nodes.index_select(
                0, dst_all.index_select(0, matched)
            )

            positive = graph_target_edge_mask.index_select(1, local_edge_ids)
            src_dist = graph_target_dist.index_select(1, local_src)
            dst_dist = graph_target_dist.index_select(1, local_dst)
            dst_count = graph_target_counts.index_select(1, local_dst).float()

            valid_for_target = (
                positive
                & src_active.unsqueeze(0)
                & ~dst_active.unsqueeze(0)
                & src_dist.ge(1)
                & dst_dist.ge(0)
                & src_dist.eq(dst_dist + 1)
                & dst_dist.le(remaining_expand_budget)
            )
            if not bool(valid_for_target.any().item()):
                continue

            matched_valid = valid_for_target.any(dim=0)
            valid_mask[matched[matched_valid]] = True
            support = (dst_count * valid_for_target.float()).sum(dim=0)
            if self.score_exponent == 0.0:
                scores[matched[matched_valid]] = 1.0
            else:
                scores[matched[matched_valid]] = (
                    support[matched_valid].clamp_min(1.0).pow(self.score_exponent)
                )
        return valid_mask, scores

    def graph_has_teacher_expand(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        remaining_expand_budget: int,
        num_graphs: int,
    ) -> torch.Tensor:
        valid_mask, _ = self.candidate_scores(
            retrieval_batch=retrieval_batch,
            state=state,
            candidates=candidates,
            remaining_expand_budget=remaining_expand_budget,
        )
        counts = torch.bincount(
            candidates.batch_index[valid_mask], minlength=num_graphs
        )
        return counts.gt(0)

    def graph_should_stop(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        remaining_expand_budget: int,
        num_graphs: int,
    ) -> torch.Tensor:
        return ~self.graph_has_teacher_expand(
            retrieval_batch=retrieval_batch,
            state=state,
            candidates=candidates,
            remaining_expand_budget=remaining_expand_budget,
            num_graphs=num_graphs,
        )

    @staticmethod
    def _require_per_target_teacher_labels(retrieval_batch: RetrievalBatch) -> None:
        required = (
            SampleFields.TARGET_NODE_DISTANCE_FLAT,
            SampleFields.TARGET_SHORTEST_PATH_COUNT_FLAT,
            SampleFields.TARGET_SHORTEST_PATH_EDGE_MASK_FLAT,
        )
        missing = [name for name in required if not hasattr(retrieval_batch, name)]
        if _get_train_target_node_ids(retrieval_batch).numel() == 0:
            return
        if missing:
            raise RuntimeError(
                "RetrievalBatch is missing target-conditioned teacher labels "
                f"{missing}. Rebuild preprocessing/materialization artifacts to "
                "enable coverage-aware teacher guidance."
            )


def _exclusive_cumsum(values: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(values.numel() + 1, dtype=values.dtype, device=values.device)
    if values.numel() > 0:
        out[1:] = torch.cumsum(values, dim=0)
    return out


def _get_train_target_node_ids(retrieval_batch: RetrievalBatch) -> torch.Tensor:
    return getattr(retrieval_batch, SampleFields.TRAIN_TARGET_NODE_IDS).long()


__all__ = ["TeacherGuidance"]
