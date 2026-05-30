from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.data.schema import ReplayProgramBatch, RetrievalBatch

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class DirectedAdjacencyIndex:
    """
    CSR-style indices over physical directed KG edges.

    out_ptr / edge_ids_by_src:
        edge ids grouped by source node.

    in_ptr / edge_ids_by_dst:
        edge ids grouped by destination node.

    Incoming edges are original physical KG edge ids.
    No inverse edges are fabricated.
    """

    out_ptr: Tensor
    edge_ids_by_src: Tensor
    in_ptr: Tensor
    edge_ids_by_dst: Tensor


@dataclass(frozen=True, slots=True)
class GraphContext:
    """
    Static label-free graph context.

    Coordinates:
    - edge_index[:, e] uses batch-physical node ids.
    - edge id e is the physical column id in edge_index.
    - node_to_graph[n] maps physical node id -> graph id.
    - edge_to_graph[e] maps physical edge id -> graph id.
    - edge_ptr[g]:edge_ptr[g + 1] gives graph g's physical edge range.
    - anchor_node_ids contains physical node ids grouped by graph.
    """

    edge_index: Tensor  # [2, E]
    node_to_graph: Tensor  # [N]
    edge_to_graph: Tensor  # [E]
    edge_ptr: Tensor  # [G + 1]

    anchor_mask: Tensor  # [N]
    anchor_ptr: Tensor  # [G + 1]
    anchor_node_ids: Tensor  # [A], grouped by graph

    adjacency: DirectedAdjacencyIndex

    num_nodes: int
    num_edges: int
    num_graphs: int

    @classmethod
    def from_batch(
        cls,
        batch: RetrievalBatch,
        *,
        validate: bool = False,
    ) -> GraphContext:
        edge_index = batch.edge_index
        node_to_graph = batch.batch.to(
            device=edge_index.device,
            dtype=torch.long,
        )

        num_nodes = int(batch.num_nodes_total)
        num_edges = int(batch.num_edges_total)
        num_graphs = int(batch.num_graphs_total)

        if validate:
            _validate_graph_inputs(
                edge_index=edge_index,
                node_to_graph=node_to_graph,
                num_nodes=num_nodes,
                num_edges=num_edges,
                num_graphs=num_graphs,
            )

        edge_to_graph = infer_edge_to_graph(
            edge_index=edge_index,
            node_to_graph=node_to_graph,
            validate=validate,
        )

        edge_ptr = build_graph_ptr_from_grouped_ids(
            graph_ids=edge_to_graph,
            num_graphs=num_graphs,
            device=edge_index.device,
        )

        if validate:
            _validate_grouped_graph_ids(
                graph_ids=edge_to_graph,
                name="edge_to_graph",
            )

        raw_anchor_node_ids = batch.anchor_node_ids.to(
            device=edge_index.device,
            dtype=torch.long,
        )

        if validate:
            _validate_id_range(
                ids=raw_anchor_node_ids,
                upper=num_nodes,
                name="anchor_node_ids",
            )

        anchor_mask = torch.zeros(
            num_nodes,
            dtype=torch.bool,
            device=edge_index.device,
        )
        if int(raw_anchor_node_ids.numel()) > 0:
            anchor_mask[raw_anchor_node_ids] = True

        anchor_ptr, anchor_node_ids = build_graph_node_csr(
            node_ids=raw_anchor_node_ids,
            node_to_graph=node_to_graph,
            num_graphs=num_graphs,
        )

        adjacency = build_directed_adjacency_index(
            edge_index=edge_index,
            num_nodes=num_nodes,
        )

        return cls(
            edge_index=edge_index,
            node_to_graph=node_to_graph,
            edge_to_graph=edge_to_graph,
            edge_ptr=edge_ptr,
            anchor_mask=anchor_mask,
            anchor_ptr=anchor_ptr,
            anchor_node_ids=anchor_node_ids,
            adjacency=adjacency,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_graphs=num_graphs,
        )

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    @property
    def edge_src(self) -> Tensor:
        return self.edge_index[0]

    @property
    def edge_dst(self) -> Tensor:
        return self.edge_index[1]


@dataclass(frozen=True, slots=True)
class TargetContext:
    """
    Target-label context for reward computation and evaluation.

    Not inference-safe.

    Coordinates:
    - target_mask[n] is indexed by batch-physical node id.
    - node_target_distance[n] is indexed by batch-physical node id.
    - reachable_target_node_ids are batch-physical node ids grouped by graph.
    """

    target_mask: Tensor  # [N]
    reachable_target_node_ids: Tensor  # [T]
    reachable_target_node_ids_ptr: Tensor  # [G + 1]
    target_count_by_graph: Tensor  # [G]

    node_target_distance: Tensor  # [N]
    anchor_target_count_by_graph: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))

    @classmethod
    def from_batch(
        cls,
        *,
        batch: RetrievalBatch,
        graph_context: GraphContext,
        validate: bool = False,
    ) -> TargetContext:
        target_node_ids = batch.reachable_target_node_ids.to(
            device=graph_context.device,
            dtype=torch.long,
        )

        target_mask = torch.zeros(
            int(graph_context.num_nodes),
            dtype=torch.bool,
            device=graph_context.device,
        )

        if int(target_node_ids.numel()) > 0:
            target_mask[target_node_ids] = True

        target_graph_ids = graph_context.node_to_graph.index_select(
            0,
            target_node_ids,
        )

        target_count_by_graph = torch.bincount(
            target_graph_ids,
            minlength=int(graph_context.num_graphs),
        ).to(dtype=torch.long)

        target_ptr = getattr(batch, "reachable_target_node_ids_ptr", None)
        if target_ptr is None:
            target_ptr = build_ptr_from_counts(
                counts=target_count_by_graph,
                device=graph_context.device,
            )
        else:
            target_ptr = target_ptr.to(
                device=graph_context.device,
                dtype=torch.long,
            )

        node_target_distance = batch.node_target_distance.to(
            device=graph_context.device,
            dtype=torch.long,
        )

        anchor_target_count_by_graph = torch.bincount(
            graph_context.node_to_graph.index_select(0, graph_context.anchor_node_ids[target_mask.index_select(0, graph_context.anchor_node_ids)])
            if int(graph_context.anchor_node_ids.numel()) > 0
            else torch.empty(0, dtype=torch.long, device=graph_context.device),
            minlength=int(graph_context.num_graphs),
        ).to(dtype=torch.long)

        if validate:
            _validate_target_context_tensors(
                target_mask=target_mask,
                reachable_target_node_ids=target_node_ids,
                reachable_target_node_ids_ptr=target_ptr,
                target_count_by_graph=target_count_by_graph,
                node_target_distance=node_target_distance,
                graph_context=graph_context,
            )

        return cls(
            target_mask=target_mask,
            reachable_target_node_ids=target_node_ids,
            reachable_target_node_ids_ptr=target_ptr,
            target_count_by_graph=target_count_by_graph,
            node_target_distance=node_target_distance,
            anchor_target_count_by_graph=anchor_target_count_by_graph,
        )

    @property
    def device(self) -> torch.device:
        return self.target_mask.device

    @property
    def valid_graph_mask(self) -> Tensor:
        return self.target_count_by_graph.gt(0)


@dataclass(frozen=True, slots=True)
class ReplayContext:
    candidate_edge_ids: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    candidate_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    candidate_target_positions: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    candidate_target_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    edge_to_candidate_ids: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    edge_to_candidate_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    candidate_graph_ptr: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    path_truncated_by_graph: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))

    @classmethod
    def from_batch(
        cls,
        *,
        batch: RetrievalBatch,
        graph_context: GraphContext,
        target_context: TargetContext,
        validate: bool = False,
    ) -> "ReplayContext":
        program = getattr(batch, "replay_program", None)
        if not isinstance(program, ReplayProgramBatch):
            raise TypeError("batch.replay_program must be a ReplayProgramBatch.")

        replay = cls(
            candidate_edge_ids=program.candidate_edge_ids.to(dtype=torch.long).contiguous(),
            candidate_ptr=program.candidate_ptr.to(dtype=torch.long).contiguous(),
            candidate_target_positions=program.candidate_target_positions.to(dtype=torch.long).contiguous(),
            candidate_target_ptr=program.candidate_target_ptr.to(dtype=torch.long).contiguous(),
            edge_to_candidate_ids=program.edge_to_candidate_ids.to(dtype=torch.long).contiguous(),
            edge_to_candidate_ptr=program.edge_to_candidate_ptr.to(dtype=torch.long).contiguous(),
            candidate_graph_ptr=program.candidate_graph_ptr.to(dtype=torch.long).contiguous(),
            path_truncated_by_graph=program.path_truncated_by_graph.to(dtype=torch.long).contiguous(),
        )

        if validate:
            _validate_replay_context_tensors(
                replay_context=replay,
                graph_context=graph_context,
                target_context=target_context,
            )

        return replay


def infer_edge_to_graph(
    *,
    edge_index: Tensor,
    node_to_graph: Tensor,
    validate: bool,
) -> Tensor:
    if int(edge_index.size(1)) == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=edge_index.device,
        )

    src_graph = node_to_graph.index_select(0, edge_index[0])

    if validate:
        dst_graph = node_to_graph.index_select(0, edge_index[1])
        mismatch = src_graph.ne(dst_graph)
        if bool(mismatch.any()):
            first = int(mismatch.nonzero(as_tuple=False).flatten()[0].item())
            raise ValueError(
                "Cross-graph edge detected: "
                f"edge_id={first}, "
                f"src={int(edge_index[0, first].item())}, "
                f"dst={int(edge_index[1, first].item())}, "
                f"src_graph={int(src_graph[first].item())}, "
                f"dst_graph={int(dst_graph[first].item())}."
            )

    return src_graph


def build_directed_adjacency_index(
    *,
    edge_index: Tensor,
    num_nodes: int,
) -> DirectedAdjacencyIndex:
    edge_ids = torch.arange(
        int(edge_index.size(1)),
        dtype=torch.long,
        device=edge_index.device,
    )

    out_ptr, edge_ids_by_src = build_node_to_edge_csr(
        node_ids=edge_index[0],
        edge_ids=edge_ids,
        num_nodes=int(num_nodes),
    )

    in_ptr, edge_ids_by_dst = build_node_to_edge_csr(
        node_ids=edge_index[1],
        edge_ids=edge_ids,
        num_nodes=int(num_nodes),
    )

    return DirectedAdjacencyIndex(
        out_ptr=out_ptr,
        edge_ids_by_src=edge_ids_by_src,
        in_ptr=in_ptr,
        edge_ids_by_dst=edge_ids_by_dst,
    )


def build_node_to_edge_csr(
    *,
    node_ids: Tensor,
    edge_ids: Tensor,
    num_nodes: int,
) -> tuple[Tensor, Tensor]:
    node_ids = node_ids.to(dtype=torch.long)
    edge_ids = edge_ids.to(dtype=torch.long)

    if int(node_ids.numel()) != int(edge_ids.numel()):
        raise ValueError("node_ids and edge_ids must have the same length: " f"{node_ids.numel()} vs {edge_ids.numel()}.")

    if int(node_ids.numel()) == 0:
        ptr = torch.zeros(
            int(num_nodes) + 1,
            dtype=torch.long,
            device=node_ids.device,
        )
        return ptr, edge_ids

    order = torch.argsort(node_ids)
    sorted_node_ids = node_ids.index_select(0, order)
    sorted_edge_ids = edge_ids.index_select(0, order)

    counts = torch.bincount(
        sorted_node_ids,
        minlength=int(num_nodes),
    ).to(dtype=torch.long)

    ptr = build_ptr_from_counts(
        counts=counts,
        device=node_ids.device,
    )

    return ptr, sorted_edge_ids


def build_graph_node_csr(
    *,
    node_ids: Tensor,
    node_to_graph: Tensor,
    num_graphs: int,
) -> tuple[Tensor, Tensor]:
    node_ids = node_ids.to(
        device=node_to_graph.device,
        dtype=torch.long,
    )

    if int(node_ids.numel()) == 0:
        ptr = torch.zeros(
            int(num_graphs) + 1,
            dtype=torch.long,
            device=node_to_graph.device,
        )
        return ptr, node_ids

    graph_ids = node_to_graph.index_select(0, node_ids)

    order = torch.argsort(graph_ids)
    sorted_graph_ids = graph_ids.index_select(0, order)
    sorted_node_ids = node_ids.index_select(0, order)

    counts = torch.bincount(
        sorted_graph_ids,
        minlength=int(num_graphs),
    ).to(dtype=torch.long)

    ptr = build_ptr_from_counts(
        counts=counts,
        device=node_to_graph.device,
    )

    return ptr, sorted_node_ids


def build_graph_ptr_from_grouped_ids(
    *,
    graph_ids: Tensor,
    num_graphs: int,
    device: torch.device,
) -> Tensor:
    graph_ids = graph_ids.to(
        device=device,
        dtype=torch.long,
    )

    if int(graph_ids.numel()) == 0:
        return torch.zeros(
            int(num_graphs) + 1,
            dtype=torch.long,
            device=device,
        )

    counts = torch.bincount(
        graph_ids,
        minlength=int(num_graphs),
    ).to(dtype=torch.long)

    return build_ptr_from_counts(
        counts=counts,
        device=device,
    )


def build_ptr_from_counts(
    *,
    counts: Tensor,
    device: torch.device,
) -> Tensor:
    counts = counts.to(
        device=device,
        dtype=torch.long,
    ).view(-1)

    ptr = torch.empty(
        int(counts.numel()) + 1,
        dtype=torch.long,
        device=device,
    )
    ptr[0] = 0
    ptr[1:] = torch.cumsum(counts, dim=0)

    return ptr


def _validate_target_context_tensors(
    *,
    target_mask: Tensor,
    reachable_target_node_ids: Tensor,
    reachable_target_node_ids_ptr: Tensor,
    target_count_by_graph: Tensor,
    node_target_distance: Tensor,
    graph_context: GraphContext,
) -> None:
    if target_mask.ndim != 1:
        raise ValueError(f"target_mask must have shape [N], got {tuple(target_mask.shape)}.")

    if int(target_mask.numel()) != int(graph_context.num_nodes):
        raise ValueError("target_mask length must equal graph_context.num_nodes: " f"{int(target_mask.numel())} vs {int(graph_context.num_nodes)}.")

    if reachable_target_node_ids.ndim != 1:
        raise ValueError("reachable_target_node_ids must have shape [T], " f"got {tuple(reachable_target_node_ids.shape)}.")

    if reachable_target_node_ids.dtype != torch.long:
        raise TypeError("reachable_target_node_ids must have dtype torch.long, " f"got {reachable_target_node_ids.dtype}.")

    _validate_id_range(
        ids=reachable_target_node_ids,
        upper=int(graph_context.num_nodes),
        name="reachable_target_node_ids",
    )

    if reachable_target_node_ids_ptr.ndim != 1:
        raise ValueError("reachable_target_node_ids_ptr must have shape [G + 1], " f"got {tuple(reachable_target_node_ids_ptr.shape)}.")

    if int(reachable_target_node_ids_ptr.numel()) != int(graph_context.num_graphs) + 1:
        raise ValueError(
            "reachable_target_node_ids_ptr length must be num_graphs + 1: "
            f"{int(reachable_target_node_ids_ptr.numel())} vs "
            f"{int(graph_context.num_graphs) + 1}."
        )

    if target_count_by_graph.ndim != 1:
        raise ValueError("target_count_by_graph must have shape [G], " f"got {tuple(target_count_by_graph.shape)}.")

    if int(target_count_by_graph.numel()) != int(graph_context.num_graphs):
        raise ValueError(
            "target_count_by_graph length must equal num_graphs: " f"{int(target_count_by_graph.numel())} vs {int(graph_context.num_graphs)}."
        )

    if node_target_distance.ndim != 1:
        raise ValueError("node_target_distance must have shape [N], " f"got {tuple(node_target_distance.shape)}.")

    if int(node_target_distance.numel()) != int(graph_context.num_nodes):
        raise ValueError(
            "node_target_distance length must equal num_nodes: " f"{int(node_target_distance.numel())} vs {int(graph_context.num_nodes)}."
        )

def _validate_replay_context_tensors(
    *,
    replay_context: ReplayContext,
    graph_context: GraphContext,
    target_context: TargetContext,
) -> None:
    if replay_context.candidate_edge_ids.ndim != 1:
        raise ValueError("candidate_edge_ids must have shape [C_edges].")
    if replay_context.candidate_ptr.ndim != 1:
        raise ValueError("candidate_ptr must have shape [C + 1].")
    if replay_context.candidate_target_positions.ndim != 1:
        raise ValueError("candidate_target_positions must have shape [C_targets].")
    if replay_context.candidate_target_ptr.ndim != 1:
        raise ValueError("candidate_target_ptr must have shape [C + 1].")
    if replay_context.edge_to_candidate_ids.ndim != 1:
        raise ValueError("edge_to_candidate_ids must have shape [R].")
    if replay_context.edge_to_candidate_ptr.ndim != 1:
        raise ValueError("edge_to_candidate_ptr must have shape [E + 1].")
    if replay_context.candidate_graph_ptr.ndim != 1:
        raise ValueError("candidate_graph_ptr must have shape [G + 1].")
    if replay_context.path_truncated_by_graph.ndim != 1:
        raise ValueError("path_truncated_by_graph must have shape [G].")

    candidate_count = int(replay_context.candidate_ptr.numel()) - 1
    if candidate_count < 0:
        raise ValueError("candidate_ptr must contain at least one element.")
    if int(replay_context.candidate_target_ptr.numel()) != candidate_count + 1:
        raise ValueError("candidate_target_ptr length must match candidate count.")
    if int(replay_context.candidate_ptr[-1].item()) != int(replay_context.candidate_edge_ids.numel()):
        raise ValueError("candidate_ptr must terminate at candidate_edge_ids length.")
    if int(replay_context.candidate_target_ptr[-1].item()) != int(replay_context.candidate_target_positions.numel()):
        raise ValueError("candidate_target_ptr must terminate at candidate_target_positions length.")
    if int(replay_context.edge_to_candidate_ptr.numel()) != int(graph_context.num_edges) + 1:
        raise ValueError("edge_to_candidate_ptr length must equal num_edges + 1.")
    if int(replay_context.edge_to_candidate_ptr[-1].item()) != int(replay_context.edge_to_candidate_ids.numel()):
        raise ValueError("edge_to_candidate_ptr must terminate at edge_to_candidate_ids length.")
    if int(replay_context.candidate_graph_ptr.numel()) != int(graph_context.num_graphs) + 1:
        raise ValueError("candidate_graph_ptr length must equal num_graphs + 1.")
    if int(replay_context.candidate_graph_ptr[-1].item()) != candidate_count:
        raise ValueError("candidate_graph_ptr must terminate at candidate count.")
    if int(replay_context.path_truncated_by_graph.numel()) != int(graph_context.num_graphs):
        raise ValueError("path_truncated_by_graph length must equal num_graphs.")

    _validate_id_range(
        ids=replay_context.candidate_edge_ids,
        upper=int(graph_context.num_edges),
        name="candidate_edge_ids",
    )
    if int(replay_context.candidate_target_positions.numel()) > 0:
        max_targets = int(target_context.reachable_target_node_ids.numel())
        _validate_id_range(
            ids=replay_context.candidate_target_positions,
            upper=max(max_targets, 1),
            name="candidate_target_positions",
        )
    if int(replay_context.edge_to_candidate_ids.numel()) > 0:
        _validate_id_range(
            ids=replay_context.edge_to_candidate_ids,
            upper=max(candidate_count, 1),
            name="edge_to_candidate_ids",
        )


def _validate_graph_inputs(
    *,
    edge_index: Tensor,
    node_to_graph: Tensor,
    num_nodes: int,
    num_edges: int,
    num_graphs: int,
) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")

    if edge_index.dtype != torch.long:
        raise TypeError(f"edge_index must have dtype torch.long, got {edge_index.dtype}.")

    if int(edge_index.size(1)) != int(num_edges):
        raise ValueError(f"edge_index has {edge_index.size(1)} edges, but num_edges={num_edges}.")

    if node_to_graph.ndim != 1:
        raise ValueError(f"node_to_graph must have shape [N], got {tuple(node_to_graph.shape)}.")

    if node_to_graph.dtype != torch.long:
        raise TypeError(f"node_to_graph must have dtype torch.long, got {node_to_graph.dtype}.")

    if int(node_to_graph.numel()) != int(num_nodes):
        raise ValueError(f"node_to_graph has length {node_to_graph.numel()}, " f"but num_nodes={num_nodes}.")

    _validate_id_range(
        ids=node_to_graph,
        upper=int(num_graphs),
        name="node_to_graph",
    )

    _validate_id_range(
        ids=edge_index.reshape(-1),
        upper=int(num_nodes),
        name="edge_index",
    )


def _validate_grouped_graph_ids(
    *,
    graph_ids: Tensor,
    name: str,
) -> None:
    if int(graph_ids.numel()) <= 1:
        return

    if bool(graph_ids[1:].lt(graph_ids[:-1]).any()):
        raise ValueError(f"{name} must be grouped by graph id. " "Graph-local edge id cannot be converted by edge_ptr offsets.")


def _validate_id_range(
    *,
    ids: Tensor,
    upper: int,
    name: str,
) -> None:
    if int(ids.numel()) == 0:
        return

    if ids.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {ids.dtype}.")

    min_id = int(ids.min().item())
    max_id = int(ids.max().item())

    if min_id < 0 or max_id >= int(upper):
        raise ValueError(f"{name} contains ids outside [0, {upper}): " f"min={min_id}, max={max_id}.")


__all__ = [
    "DirectedAdjacencyIndex",
    "GraphContext",
    "TargetContext",
    "build_directed_adjacency_index",
    "build_graph_node_csr",
    "build_graph_ptr_from_grouped_ids",
    "build_node_to_edge_csr",
    "build_ptr_from_counts",
    "infer_edge_to_graph",
]
