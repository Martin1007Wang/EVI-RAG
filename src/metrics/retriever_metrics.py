from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torchmetrics import Metric

from src.utils import normalize_k_values
from src.utils.graph_utils import compute_qa_edge_mask

_LABEL_POSITIVE_THRESHOLD = 0.5
_COUNT_EPS = 1.0
_ZERO = 0
_ONE = 1


def _infer_num_graphs(batch: Any, indexes: torch.Tensor, num_graphs: Optional[int]) -> int:
    if num_graphs is not None:
        return int(num_graphs)
    node_ptr = getattr(batch, "ptr", None)
    if node_ptr is not None:
        return int(node_ptr.numel() - 1)
    if indexes.numel() == 0:
        return _ZERO
    return int(indexes.max().item()) + _ONE


def _group_edges_by_graph(indexes: torch.Tensor, num_graphs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_graphs <= _ZERO or indexes.numel() == 0:
        return indexes.new_empty((0,), dtype=torch.long), indexes.new_empty((0,), dtype=torch.long)
    order = torch.argsort(indexes, stable=True)
    counts = torch.bincount(indexes, minlength=num_graphs)
    offsets = torch.cumsum(counts, dim=0)
    return order, offsets


def _prepare_metric_inputs(
    preds: torch.Tensor,
    target: torch.Tensor,
    indexes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores_all = preds.detach().view(-1)
    labels_all = target.detach().view(-1) > _LABEL_POSITIVE_THRESHOLD
    graph_ids = indexes.detach().view(-1).to(dtype=torch.long)
    if scores_all.numel() == 0:
        return scores_all, labels_all, graph_ids
    if scores_all.numel() != labels_all.numel() or scores_all.numel() != graph_ids.numel():
        raise ValueError(
            f"preds/target/indexes mismatch: {scores_all.shape} vs {labels_all.shape} vs {graph_ids.shape}"
        )
    return scores_all, labels_all, graph_ids


def _iter_graph_edge_slices(order: torch.Tensor, offsets: torch.Tensor):
    start = _ZERO
    for end in offsets.tolist():
        end_int = int(end)
        if end_int <= start:
            start = end_int
            yield None
            continue
        yield order[start:end_int]
        start = end_int


def _compute_bridge_mask(batch: Any, *, edge_index: torch.Tensor) -> torch.Tensor:
    num_nodes = getattr(batch, "num_nodes", None)
    if num_nodes is None:
        raise ValueError("Batch missing num_nodes required for bridge metrics.")
    q_local_indices = getattr(batch, "q_local_indices", None)
    a_local_indices = getattr(batch, "a_local_indices", None)
    if q_local_indices is None or a_local_indices is None:
        raise ValueError("Batch missing q_local_indices/a_local_indices required for bridge metrics.")
    near_mask = compute_qa_edge_mask(
        edge_index,
        num_nodes=int(num_nodes),
        q_local_indices=q_local_indices,
        a_local_indices=a_local_indices,
    )
    return ~near_mask


class EdgeRecallAtK(Metric):
    full_state_update: bool = False

    def __init__(self, k_values: Optional[Sequence[int]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k_values = normalize_k_values(k_values)
        for k in self.k_values:
            self.add_state(f"recall_sum_at_{k}", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("graph_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        batch: Any,
        num_graphs: Optional[int] = None,
        **_: Any,
    ) -> None:
        if not self.k_values:
            return
        scores_all, labels_all, graph_ids = _prepare_metric_inputs(preds, target, indexes)
        if scores_all.numel() == 0:
            return

        num_graphs = _infer_num_graphs(batch, graph_ids, num_graphs)
        if num_graphs <= _ZERO:
            return

        order, offsets = _group_edges_by_graph(graph_ids, num_graphs)
        if order.numel() == 0:
            return
        self._accumulate_edge_recall(scores_all, labels_all, order, offsets)

    def _accumulate_edge_recall(
        self,
        scores_all: torch.Tensor,
        labels_all: torch.Tensor,
        order: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        max_k = max(self.k_values)
        for edge_idx in _iter_graph_edge_slices(order, offsets):
            if edge_idx is None:
                continue
            scores = scores_all.index_select(0, edge_idx)
            labels = labels_all.index_select(0, edge_idx)
            if scores.numel() == 0:
                continue
            self._update_graph_recall(scores, labels, max_k)

    def _update_graph_recall(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        max_k: int,
    ) -> None:
        pos_count = labels.sum().to(dtype=torch.float32)
        k_top = min(int(scores.numel()), max_k)
        if k_top <= _ZERO:
            self.graph_count += 1.0
            return
        top_idx = torch.topk(scores, k=k_top, largest=True, sorted=True).indices
        top_labels = labels.index_select(0, top_idx).to(dtype=torch.float32)
        cum_hits = torch.cumsum(top_labels, dim=0)
        denom = pos_count.clamp(min=_COUNT_EPS)
        for k in self.k_values:
            k_eff = min(int(k), k_top)
            if k_eff <= _ZERO:
                hits = cum_hits.new_zeros(())
            else:
                hits = cum_hits[k_eff - 1]
            recall = hits / denom
            getattr(self, f"recall_sum_at_{int(k)}").add_(recall)
        self.graph_count += 1.0

    def compute(self) -> Dict[str, torch.Tensor]:
        if not self.k_values:
            return {}
        denom = self.graph_count.clamp(min=_COUNT_EPS)
        return {
            f"edge/recall@{k}": getattr(self, f"recall_sum_at_{k}") / denom
            for k in self.k_values
        }


class BridgeEdgeRecallAtK(Metric):
    full_state_update: bool = False

    def __init__(self, k_values: Optional[Sequence[int]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k_values = normalize_k_values(k_values)
        for k in self.k_values:
            self.add_state(f"recall_sum_at_{k}", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("graph_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        batch: Any,
        num_graphs: Optional[int] = None,
        **_: Any,
    ) -> None:
        if not self.k_values:
            return
        scores_all, labels_all, graph_ids = _prepare_metric_inputs(preds, target, indexes)
        if scores_all.numel() == 0:
            return
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            raise ValueError("Batch missing edge_index required for bridge recall.")
        edge_index = edge_index.to(device=scores_all.device)
        bridge_mask = _compute_bridge_mask(batch, edge_index=edge_index)
        if bridge_mask.numel() != scores_all.numel():
            raise ValueError(
                f"bridge_mask length mismatch: {bridge_mask.numel()} vs scores {scores_all.numel()}"
            )
        scores_all = scores_all[bridge_mask]
        labels_all = labels_all[bridge_mask]
        graph_ids = graph_ids[bridge_mask]
        if scores_all.numel() == 0:
            return

        num_graphs = _infer_num_graphs(batch, graph_ids, num_graphs)
        if num_graphs <= _ZERO:
            return

        order, offsets = _group_edges_by_graph(graph_ids, num_graphs)
        if order.numel() == 0:
            return
        self._accumulate_bridge_recall(scores_all, labels_all, order, offsets)

    def _accumulate_bridge_recall(
        self,
        scores_all: torch.Tensor,
        labels_all: torch.Tensor,
        order: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        max_k = max(self.k_values)
        for edge_idx in _iter_graph_edge_slices(order, offsets):
            if edge_idx is None:
                continue
            scores = scores_all.index_select(0, edge_idx)
            labels = labels_all.index_select(0, edge_idx)
            if scores.numel() == 0:
                continue
            self._update_graph_bridge_recall(scores, labels, max_k)

    def _update_graph_bridge_recall(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        max_k: int,
    ) -> None:
        pos_count = labels.sum().to(dtype=torch.float32)
        if pos_count <= _ZERO:
            return
        k_top = min(int(scores.numel()), max_k)
        if k_top <= _ZERO:
            return
        top_idx = torch.topk(scores, k=k_top, largest=True, sorted=True).indices
        top_labels = labels.index_select(0, top_idx).to(dtype=torch.float32)
        cum_hits = torch.cumsum(top_labels, dim=0)
        denom = pos_count.clamp(min=_COUNT_EPS)
        for k in self.k_values:
            k_eff = min(int(k), k_top)
            if k_eff <= _ZERO:
                hits = cum_hits.new_zeros(())
            else:
                hits = cum_hits[k_eff - 1]
            recall = hits / denom
            getattr(self, f"recall_sum_at_{int(k)}").add_(recall)
        self.graph_count += 1.0

    def compute(self) -> Dict[str, torch.Tensor]:
        if not self.k_values:
            return {}
        denom = self.graph_count.clamp(min=_COUNT_EPS)
        return {
            f"bridge/recall@{k}": getattr(self, f"recall_sum_at_{k}") / denom
            for k in self.k_values
        }


class BridgePositiveCoverage(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("bridge_pos_edges", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pos_edges", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("graphs_with_pos", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("graphs_with_bridge_pos", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        batch: Any,
        num_graphs: Optional[int] = None,
        **_: Any,
    ) -> None:
        scores_all, labels_all, graph_ids = _prepare_metric_inputs(preds, target, indexes)
        if scores_all.numel() == 0:
            return
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            raise ValueError("Batch missing edge_index required for bridge coverage.")
        edge_index = edge_index.to(device=scores_all.device)
        bridge_mask = _compute_bridge_mask(batch, edge_index=edge_index)
        if bridge_mask.numel() != labels_all.numel():
            raise ValueError(
                f"bridge_mask length mismatch: {bridge_mask.numel()} vs labels {labels_all.numel()}"
            )

        labels_float = labels_all.to(dtype=torch.float32)
        bridge_labels = labels_float[bridge_mask]
        self.total_pos_edges += labels_float.sum()
        self.bridge_pos_edges += bridge_labels.sum()

        num_graphs = _infer_num_graphs(batch, graph_ids, num_graphs)
        if num_graphs <= _ZERO:
            return
        pos_counts = torch.zeros(num_graphs, device=labels_float.device, dtype=labels_float.dtype)
        pos_counts.scatter_add_(0, graph_ids, labels_float)
        bridge_pos_counts = torch.zeros(num_graphs, device=labels_float.device, dtype=labels_float.dtype)
        if bridge_labels.numel() > 0:
            bridge_graph_ids = graph_ids[bridge_mask]
            bridge_pos_counts.scatter_add_(0, bridge_graph_ids, bridge_labels)
        has_pos = pos_counts > 0
        has_bridge_pos = bridge_pos_counts > 0
        self.graphs_with_pos += has_pos.sum()
        self.graphs_with_bridge_pos += (has_pos & has_bridge_pos).sum()

    def compute(self) -> Dict[str, torch.Tensor]:
        edge_denom = self.total_pos_edges.clamp(min=_COUNT_EPS)
        graph_denom = self.graphs_with_pos.clamp(min=_COUNT_EPS)
        return {
            "bridge/pos_edge_frac": self.bridge_pos_edges / edge_denom,
            "bridge/pos_graph_frac": self.graphs_with_bridge_pos / graph_denom,
        }


class ScoreMargin(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("margin_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("graph_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        batch: Any,
        num_graphs: Optional[int] = None,
        **_: Any,
    ) -> None:
        scores_all, labels_all, graph_ids = _prepare_metric_inputs(preds, target, indexes)
        if scores_all.numel() == 0:
            return

        num_graphs = _infer_num_graphs(batch, graph_ids, num_graphs)
        if num_graphs <= _ZERO:
            return

        order, offsets = _group_edges_by_graph(graph_ids, num_graphs)
        if order.numel() == 0:
            return
        self._accumulate_score_margin(scores_all, labels_all, order, offsets)

    def _accumulate_score_margin(
        self,
        scores_all: torch.Tensor,
        labels_all: torch.Tensor,
        order: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        for edge_idx in _iter_graph_edge_slices(order, offsets):
            if edge_idx is None:
                continue
            scores = scores_all.index_select(0, edge_idx)
            labels = labels_all.index_select(0, edge_idx)
            if scores.numel() == 0:
                continue
            self._update_graph_margin(scores, labels)

    def _update_graph_margin(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        has_pos = bool(labels.any().item())
        has_neg = bool((~labels).any().item())
        if not has_pos or not has_neg:
            return
        pos_scores = scores[labels]
        neg_scores = scores[~labels]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return
        margin = pos_scores.min() - neg_scores.max()
        self.margin_sum += margin
        self.graph_count += 1.0

    def compute(self) -> Dict[str, torch.Tensor]:
        denom = self.graph_count.clamp(min=_COUNT_EPS)
        return {"edge/score_margin": self.margin_sum / denom}


class BridgeProbQuality(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("pos_prob_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("neg_prob_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sep_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("graph_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor,
        batch: Any,
        num_graphs: Optional[int] = None,
        **_: Any,
    ) -> None:
        scores_all, labels_all, graph_ids = _prepare_metric_inputs(preds, target, indexes)
        if scores_all.numel() == 0:
            return
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            raise ValueError("Batch missing edge_index required for bridge quality.")
        edge_index = edge_index.to(device=scores_all.device)
        bridge_mask = _compute_bridge_mask(batch, edge_index=edge_index)
        if bridge_mask.numel() != scores_all.numel():
            raise ValueError(
                f"bridge_mask length mismatch: {bridge_mask.numel()} vs scores {scores_all.numel()}"
            )
        scores_all = scores_all[bridge_mask]
        labels_all = labels_all[bridge_mask]
        graph_ids = graph_ids[bridge_mask]
        if scores_all.numel() == 0:
            return

        num_graphs = _infer_num_graphs(batch, graph_ids, num_graphs)
        if num_graphs <= _ZERO:
            return

        order, offsets = _group_edges_by_graph(graph_ids, num_graphs)
        if order.numel() == 0:
            return
        self._accumulate_bridge_quality(scores_all, labels_all, order, offsets)

    def _accumulate_bridge_quality(
        self,
        scores_all: torch.Tensor,
        labels_all: torch.Tensor,
        order: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        for edge_idx in _iter_graph_edge_slices(order, offsets):
            if edge_idx is None:
                continue
            scores = scores_all.index_select(0, edge_idx)
            labels = labels_all.index_select(0, edge_idx)
            if scores.numel() == 0:
                continue
            has_pos = bool(labels.any().item())
            has_neg = bool((~labels).any().item())
            if not has_pos or not has_neg:
                continue
            probs = torch.sigmoid(scores)
            pos_mean = probs[labels].mean()
            neg_mean = probs[~labels].mean()
            self.pos_prob_sum += pos_mean
            self.neg_prob_sum += neg_mean
            self.sep_sum += pos_mean - neg_mean
            self.graph_count += 1.0

    def compute(self) -> Dict[str, torch.Tensor]:
        denom = self.graph_count.clamp(min=_COUNT_EPS)
        return {
            "bridge/pos_prob": self.pos_prob_sum / denom,
            "bridge/neg_prob": self.neg_prob_sum / denom,
            "bridge/separation": self.sep_sum / denom,
        }
