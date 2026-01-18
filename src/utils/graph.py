from __future__ import annotations

from dataclasses import dataclass

import torch

_ZERO = 0
_DIST_UNREACHABLE = -1
_DIST_ORIGIN = 0
_DIST_STEP = 1

try:
    from torch import _dynamo as _torch_dynamo
except Exception:  # pragma: no cover - optional torch compile dependency
    _torch_dynamo = None

_EDGE_BATCH_INVERSION_PREVIEW = 5
_EDGE_BATCH_MISMATCH_PREVIEW = 5
_EDGE_BATCH_MIN = 0
_EDGE_BATCH_SAMPLE_PREVIEW = 5
_EDGE_BATCH_POS_PREVIEW = 5
_EDGE_PTR_MIN_LEN = 2
_QA_MASK_DIM = 2
_QA_MASK_EDGE_DIM = 2


def _dynamo_disable(fn):
    if _torch_dynamo is None:
        return fn
    return _torch_dynamo.disable(fn)


@dataclass(frozen=True)
class EdgeBatchDebugContext:
    sample_ids: list[str]
    edge_ptr: torch.Tensor


def build_edge_batch_debug_context(debug_batch: object) -> EdgeBatchDebugContext | None:
    sample_ids = getattr(debug_batch, "sample_id", None)
    slice_dict = getattr(debug_batch, "_slice_dict", None)
    if sample_ids is None or not isinstance(slice_dict, dict):
        return None
    edge_ptr = slice_dict.get("edge_index")
    if edge_ptr is None:
        return None
    if not torch.is_tensor(edge_ptr):
        edge_ptr = torch.as_tensor(edge_ptr, dtype=torch.long)
    edge_ptr = edge_ptr.detach().to(device="cpu")
    if edge_ptr.numel() < _EDGE_PTR_MIN_LEN:
        return None
    return EdgeBatchDebugContext(sample_ids=[str(sid) for sid in sample_ids], edge_ptr=edge_ptr)


def _preview_sample_ids(debug_context: EdgeBatchDebugContext, edge_positions: torch.Tensor) -> list[str]:
    edge_ptr = debug_context.edge_ptr
    if edge_ptr.numel() < _EDGE_PTR_MIN_LEN:
        return []
    edge_positions = edge_positions.detach().to(device="cpu")
    graph_ids = torch.bucketize(edge_positions, edge_ptr[1:], right=False)
    unique_graphs = sorted(set(graph_ids.tolist()))
    preview: list[str] = []
    for gid in unique_graphs[:_EDGE_BATCH_SAMPLE_PREVIEW]:
        if 0 <= gid < len(debug_context.sample_ids):
            preview.append(debug_context.sample_ids[gid])
    return preview


@_dynamo_disable
def compute_edge_batch(
    edge_index: torch.Tensor,
    *,
    node_ptr: torch.Tensor,
    num_graphs: int,
    device: torch.device,
    validate: bool = True,
    debug_context: EdgeBatchDebugContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if node_ptr.numel() != num_graphs + 1:
        raise ValueError(f"node_ptr length mismatch: got {node_ptr.numel()} expected {num_graphs + 1}")
    # NOTE: right=True assigns boundary nodes to their owning graph: ptr[g] <= i < ptr[g+1].
    edge_batch = torch.bucketize(edge_index[0], node_ptr[1:], right=True)
    tail_batch = None
    if validate:
        tail_batch = torch.bucketize(edge_index[1], node_ptr[1:], right=True)
    if validate:
        if edge_batch.numel() > 0:
            min_idx = int(edge_batch.min().detach().tolist())
            max_idx = int(edge_batch.max().detach().tolist())
            if min_idx < _EDGE_BATCH_MIN or max_idx >= num_graphs:
                invalid = torch.nonzero(edge_batch >= num_graphs, as_tuple=False).view(-1)
                preview = invalid[:_EDGE_BATCH_POS_PREVIEW].detach().to(device="cpu").tolist()
                sample_preview = _preview_sample_ids(debug_context, invalid) if debug_context is not None else []
                detail = f"min={min_idx} max={max_idx} num_graphs={num_graphs}"
                if preview:
                    detail += f" invalid_edge_positions={preview}"
                if sample_preview:
                    detail += f" sample_id_preview={sample_preview}"
                raise ValueError(f"edge_batch contains out-of-range indices; {detail}.")
        if tail_batch is not None and not torch.equal(edge_batch, tail_batch):
            mismatch = torch.nonzero(edge_batch != tail_batch, as_tuple=False).view(-1)
            preview = mismatch[:_EDGE_BATCH_MISMATCH_PREVIEW].detach().to(device="cpu").tolist()
            sample_preview = _preview_sample_ids(debug_context, mismatch) if debug_context is not None else []
            detail = f"first_mismatches={preview}"
            if sample_preview:
                detail += f" sample_id_preview={sample_preview}"
            raise ValueError(
                "edge_index crosses graph boundaries; head/tail graph assignments differ. "
                f"{detail}"
            )
        if edge_batch.numel() > 1 and not bool((edge_batch[:-1] <= edge_batch[1:]).all().detach().tolist()):
            inv = torch.nonzero(edge_batch[:-1] > edge_batch[1:], as_tuple=False).view(-1)
            preview = inv[:_EDGE_BATCH_INVERSION_PREVIEW].detach().to(device="cpu").tolist()
            sample_preview = _preview_sample_ids(debug_context, inv) if debug_context is not None else []
            detail = f"first_inversions={preview}"
            if sample_preview:
                detail += f" sample_id_preview={sample_preview}"
            raise ValueError(
                "edge_batch is not non-decreasing along the flattened edge list, which breaks per-graph slicing; "
                f"{detail}. Ensure edges are concatenated per-graph (PyG Batch)."
            )
    edge_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
    edge_counts.scatter_add_(0, edge_batch, torch.ones_like(edge_batch, dtype=torch.long))
    edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    edge_ptr[1:] = edge_counts.cumsum(0)
    return edge_batch, edge_ptr


def compute_invalid_nodes(
    *,
    edge_index: torch.Tensor,
    node_is_start: torch.Tensor,
    node_is_answer: torch.Tensor,
) -> torch.Tensor:
    num_nodes_total = int(node_is_start.numel())
    neighbors = torch.zeros(num_nodes_total, device=edge_index.device, dtype=torch.bool)
    if edge_index.numel() > _ZERO:
        heads = edge_index[0]
        tails = edge_index[1]
        start_heads = node_is_start[heads]
        if bool(start_heads.any().detach().tolist()):
            neighbors[tails[start_heads]] = True
        start_tails = node_is_start[tails]
        if bool(start_tails.any().detach().tolist()):
            neighbors[heads[start_tails]] = True
    return (node_is_start | neighbors) & (~node_is_answer)


def _undirected_bfs_distances_impl(
    edge_index: torch.Tensor,
    num_nodes: int,
    start_nodes: torch.Tensor,
) -> torch.Tensor:
    dist_unreachable = -1
    dist_origin = 0
    dist_step = 1
    device = edge_index.device
    dist = torch.full((num_nodes,), dist_unreachable, device=device, dtype=torch.long)
    if num_nodes <= 0:
        return dist
    if start_nodes.numel() == 0:
        return dist
    dist[start_nodes] = dist_origin
    frontier = dist == dist_origin
    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    if edge_src.numel() == 0:
        return dist
    step = dist_origin
    next_counts = torch.zeros(num_nodes, device=device, dtype=torch.long)
    for _ in range(num_nodes):
        if not bool(frontier.any().detach().tolist()):
            break
        next_counts.zero_()
        next_counts.index_add_(0, edge_dst, frontier[edge_src].to(dtype=torch.long))
        next_counts.index_add_(0, edge_src, frontier[edge_dst].to(dtype=torch.long))
        next_mask = (next_counts > 0) & (dist < 0)
        if not bool(next_mask.any().detach().tolist()):
            break
        step += dist_step
        dist[next_mask] = step
        frontier = next_mask
    return dist


_UNDIRECTED_BFS_SCRIPT = None
_DIRECTED_BFS_SCRIPT = None


@_dynamo_disable
def _directed_bfs_distances_impl(
    edge_index: torch.Tensor,
    num_nodes: int,
    start_nodes: torch.Tensor,
    dist_unreachable: int = _DIST_UNREACHABLE,
    dist_origin: int = _DIST_ORIGIN,
    dist_step: int = _DIST_STEP,
) -> torch.Tensor:
    device = edge_index.device
    dist = torch.full((num_nodes,), dist_unreachable, device=device, dtype=torch.long)
    if num_nodes <= 0:
        return dist
    if start_nodes.numel() == 0:
        return dist
    dist[start_nodes] = dist_origin
    frontier = dist == dist_origin
    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    if edge_src.numel() == 0:
        return dist
    step = dist_origin
    next_counts = torch.zeros(num_nodes, device=device, dtype=torch.long)
    for _ in range(num_nodes):
        if not bool(frontier.any().detach().tolist()):
            break
        next_counts.zero_()
        next_counts.index_add_(0, edge_dst, frontier[edge_src].to(dtype=torch.long))
        next_mask = (next_counts > 0) & (dist < 0)
        if not bool(next_mask.any().detach().tolist()):
            break
        step += dist_step
        dist[next_mask] = step
        frontier = next_mask
    return dist


def _get_undirected_bfs_script():
    global _UNDIRECTED_BFS_SCRIPT
    if _UNDIRECTED_BFS_SCRIPT is not None:
        return _UNDIRECTED_BFS_SCRIPT
    try:  # pragma: no cover - TorchScript availability depends on build
        _UNDIRECTED_BFS_SCRIPT = torch.jit.script(_undirected_bfs_distances_impl)
    except Exception as exc:  # pragma: no cover - TorchScript is required
        raise RuntimeError("TorchScript is required for undirected_bfs_distances; scripting failed.") from exc
    return _UNDIRECTED_BFS_SCRIPT


def _get_directed_bfs_script():
    global _DIRECTED_BFS_SCRIPT
    if _DIRECTED_BFS_SCRIPT is not None:
        return _DIRECTED_BFS_SCRIPT
    try:  # pragma: no cover - TorchScript availability depends on build
        _DIRECTED_BFS_SCRIPT = torch.jit.script(_directed_bfs_distances_impl)
    except Exception as exc:  # pragma: no cover - TorchScript is required
        raise RuntimeError("TorchScript is required for directed_bfs_distances; scripting failed.") from exc
    return _DIRECTED_BFS_SCRIPT


def undirected_bfs_distances(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    start_nodes: torch.Tensor,
) -> torch.Tensor:
    """Tensor-only multi-source BFS over an undirected edge list (TorchScript compiled if available)."""
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return torch.full((0,), _DIST_UNREACHABLE, device=edge_index.device, dtype=torch.long)
    if not torch.is_tensor(start_nodes):
        start_nodes = torch.as_tensor(start_nodes, device=edge_index.device, dtype=torch.long)
    else:
        start_nodes = start_nodes.to(device=edge_index.device, dtype=torch.long)
    start_nodes = start_nodes.view(-1)
    if start_nodes.numel() == 0:
        return torch.full((num_nodes,), _DIST_UNREACHABLE, device=edge_index.device, dtype=torch.long)
    torch._assert((start_nodes >= 0).all(), "start_nodes contain negatives for BFS distances.")
    torch._assert((start_nodes < num_nodes).all(), "start_nodes out of range for BFS distances.")
    return _get_undirected_bfs_script()(edge_index, num_nodes, start_nodes)


def directed_bfs_distances(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    start_nodes: torch.Tensor,
) -> torch.Tensor:
    """Tensor-only multi-source BFS over directed edges (TorchScript compiled if available)."""
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return torch.full((0,), _DIST_UNREACHABLE, device=edge_index.device, dtype=torch.long)
    if not torch.is_tensor(start_nodes):
        start_nodes = torch.as_tensor(start_nodes, device=edge_index.device, dtype=torch.long)
    else:
        start_nodes = start_nodes.to(device=edge_index.device, dtype=torch.long)
    start_nodes = start_nodes.view(-1)
    if start_nodes.numel() == 0:
        return torch.full((num_nodes,), _DIST_UNREACHABLE, device=edge_index.device, dtype=torch.long)
    torch._assert((start_nodes >= 0).all(), "start_nodes contain negatives for BFS distances.")
    torch._assert((start_nodes < num_nodes).all(), "start_nodes out of range for BFS distances.")
    return _get_directed_bfs_script()(edge_index, num_nodes, start_nodes)


def compute_qa_edge_mask(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    q_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
) -> torch.Tensor:
    if edge_index.dim() != _QA_MASK_DIM or edge_index.size(0) != _QA_MASK_EDGE_DIM:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if not torch.is_tensor(q_local_indices):
        q_local_indices = torch.as_tensor(q_local_indices, dtype=torch.long, device=edge_index.device)
    else:
        q_local_indices = q_local_indices.to(device=edge_index.device, dtype=torch.long)
    if not torch.is_tensor(a_local_indices):
        a_local_indices = torch.as_tensor(a_local_indices, dtype=torch.long, device=edge_index.device)
    else:
        a_local_indices = a_local_indices.to(device=edge_index.device, dtype=torch.long)

    q_local_indices = q_local_indices.view(-1)
    a_local_indices = a_local_indices.view(-1)
    if q_local_indices.numel() == 0 and a_local_indices.numel() == 0:
        return edge_index.new_zeros(edge_index.size(1), dtype=torch.bool)
    if q_local_indices.numel() == 0:
        qa_indices = a_local_indices
    elif a_local_indices.numel() == 0:
        qa_indices = q_local_indices
    else:
        qa_indices = torch.cat([q_local_indices, a_local_indices], dim=0)

    if qa_indices.numel() > 0:
        torch._assert(
            (qa_indices >= 0).all(),
            "q/a local indices contain negative values.",
        )
        torch._assert(
            (qa_indices < num_nodes).all(),
            "q/a local indices exceed num_nodes; batch collation is invalid.",
        )

    node_mask = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.bool)
    if qa_indices.numel() > 0:
        node_mask[qa_indices] = True
    head_idx, tail_idx = edge_index
    return node_mask[head_idx] | node_mask[tail_idx]


__all__ = [
    "EdgeBatchDebugContext",
    "build_edge_batch_debug_context",
    "compute_edge_batch",
    "compute_qa_edge_mask",
    "directed_bfs_distances",
    "undirected_bfs_distances",
]
