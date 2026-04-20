from __future__ import annotations

from typing import Any

import torch

from src.data.schema import RetrievalBatch
from src.utils.logging_utils import get_logger


log = get_logger(__name__)


def collect_mask_debug_summaries(
    batch: RetrievalBatch,
    *,
    max_graphs: int | None = None,
) -> list[dict[str, Any]]:
    edge_batch = getattr(batch, "edge_batch", None)
    if not isinstance(edge_batch, torch.Tensor):
        raise TypeError(
            "RetrievalBatch.edge_batch is required for mask debug summaries. "
            "Ensure the collator attaches edge_batch before training/eval."
        )

    summaries: list[dict[str, Any]] = []
    num_graphs = int(batch.num_graphs)
    edge_index = batch.edge_index
    graph_limit = num_graphs if max_graphs is None else min(max_graphs, num_graphs)

    for graph_idx in range(graph_limit):
        node_mask = batch.batch == graph_idx
        graph_edge_mask = edge_batch == graph_idx
        anchor_nodes = batch.is_anchor_mask[node_mask]
        target_nodes = batch.is_target_mask[node_mask]
        pos_edges = batch.positive_edge_mask[graph_edge_mask]
        src = edge_index[0][graph_edge_mask]
        dst = edge_index[1][graph_edge_mask]

        global_node_ids = node_mask.nonzero(as_tuple=False).flatten()
        target_global_ids = global_node_ids[target_nodes]
        node_start = (
            int(global_node_ids[0].item()) if global_node_ids.numel() > 0 else 0
        )
        pos_edge_src = src[pos_edges]
        pos_edge_dst = dst[pos_edges]
        positive_hits = pos_edge_dst.new_zeros(pos_edge_dst.shape, dtype=torch.bool)

        positive_to_target_ratio: float | None = None
        if pos_edge_dst.numel() > 0:
            if target_global_ids.numel() == 0:
                positive_to_target_ratio = 0.0
            else:
                positive_hits = (
                    pos_edge_dst.unsqueeze(1) == target_global_ids.unsqueeze(0)
                ).any(dim=1)
                positive_to_target_ratio = float(positive_hits.float().mean().item())

        non_target_dst_global_ids = pos_edge_dst[~positive_hits]
        non_target_dst_local_ids = (
            non_target_dst_global_ids - node_start
            if non_target_dst_global_ids.numel() > 0
            else non_target_dst_global_ids
        )
        graph_target_distance = getattr(batch, "node_to_target_distance", None)
        src_distance_sample: tuple[int, ...] = ()
        dst_distance_sample: tuple[int, ...] = ()
        non_target_dst_distance_sample: tuple[int, ...] = ()
        positive_reachable_dst_ratio: float | None = None
        if isinstance(graph_target_distance, torch.Tensor):
            per_graph_distance = graph_target_distance[node_mask]
            if pos_edge_src.numel() > 0:
                pos_edge_src_local_ids = pos_edge_src - node_start
                pos_edge_dst_local_ids = pos_edge_dst - node_start
                src_distances = per_graph_distance.index_select(
                    0, pos_edge_src_local_ids.long()
                )
                dst_distances = per_graph_distance.index_select(
                    0, pos_edge_dst_local_ids.long()
                )
                src_distance_sample = tuple(
                    int(value) for value in src_distances[:5].tolist()
                )
                dst_distance_sample = tuple(
                    int(value) for value in dst_distances[:5].tolist()
                )
                positive_reachable_dst_ratio = float(
                    dst_distances.ge(0).float().mean().item()
                )
            if non_target_dst_local_ids.numel() > 0:
                non_target_dst_distance_sample = tuple(
                    int(value)
                    for value in per_graph_distance.index_select(
                        0,
                        non_target_dst_local_ids.long(),
                    )[:5].tolist()
                )

        sample_id_value = getattr(batch, "sample_id", None)
        sample_id: str | None = None
        if isinstance(sample_id_value, (list, tuple)) and graph_idx < len(
            sample_id_value
        ):
            sample_id = str(sample_id_value[graph_idx])
        elif isinstance(sample_id_value, str) and num_graphs == 1:
            sample_id = sample_id_value

        summaries.append(
            {
                "graph_idx": graph_idx,
                "sample_id": sample_id,
                "num_nodes": int(node_mask.sum().item()),
                "anchor_nodes": int(anchor_nodes.sum().item()),
                "target_nodes": int(target_nodes.sum().item()),
                "num_edges": int(graph_edge_mask.sum().item()),
                "positive_edges": int(pos_edges.sum().item()),
                "positive_edge_target_dst_ratio": positive_to_target_ratio,
                "positive_edge_target_dst_hits": int(positive_hits.sum().item()),
                "positive_edge_reachable_dst_ratio": positive_reachable_dst_ratio,
                "positive_edge_source_count": int(pos_edge_src.numel()),
                "positive_edge_target_count": int(pos_edge_dst.numel()),
                "positive_edge_src_distance_sample": src_distance_sample,
                "positive_edge_dst_distance_sample": dst_distance_sample,
                "non_target_positive_dst_local_ids_sample": tuple(
                    int(value) for value in non_target_dst_local_ids[:5].tolist()
                ),
                "non_target_positive_dst_global_ids_sample": tuple(
                    int(value) for value in non_target_dst_global_ids[:5].tolist()
                ),
                "non_target_positive_dst_distance_sample": non_target_dst_distance_sample,
            }
        )

    return summaries


def log_mask_debug_summaries(
    batch: RetrievalBatch,
    *,
    stage: str,
    batch_idx: int,
    max_graphs: int | None = None,
) -> None:
    summaries = collect_mask_debug_summaries(batch, max_graphs=max_graphs)
    for summary in summaries:
        ratio = summary["positive_edge_target_dst_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.2f}"
        reachable_ratio = summary["positive_edge_reachable_dst_ratio"]
        if reachable_ratio is None or reachable_ratio >= 1.0:
            continue
        reachable_ratio_text = f"{reachable_ratio:.2f}"
        sample_id_suffix = (
            "" if summary["sample_id"] is None else f" sample_id={summary['sample_id']}"
        )
        log.warning(
            "Mask debug [%s batch=%d graph=%d%s]: nodes=%d anchor=%d target=%d edges=%d positive=%d positive_edge_to_target_ratio=%s positive_edge_reachable_dst_ratio=%s",
            stage,
            batch_idx,
            summary["graph_idx"],
            sample_id_suffix,
            summary["num_nodes"],
            summary["anchor_nodes"],
            summary["target_nodes"],
            summary["num_edges"],
            summary["positive_edges"],
            ratio_text,
            reachable_ratio_text,
        )
        log.warning(
            "Mask debug destination [%s batch=%d graph=%d%s]: is_target=%d/%d src_distance=%s dst_distance=%s non_target_dst_local_ids=%s non_target_dst_global_ids=%s non_target_dst_distance=%s",
            stage,
            batch_idx,
            summary["graph_idx"],
            sample_id_suffix,
            summary["positive_edge_target_dst_hits"],
            summary["positive_edges"],
            list(summary["positive_edge_src_distance_sample"]),
            list(summary["positive_edge_dst_distance_sample"]),
            list(summary["non_target_positive_dst_local_ids_sample"]),
            list(summary["non_target_positive_dst_global_ids_sample"]),
            list(summary["non_target_positive_dst_distance_sample"]),
        )


__all__ = ["collect_mask_debug_summaries", "log_mask_debug_summaries"]
