from __future__ import annotations

from typing import Any

import torch


def require_tensor(value: Any, *, device: torch.device, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value


def require_1d_long(value: Any, *, device: torch.device, name: str) -> torch.Tensor:
    tensor = require_tensor(value, device=device, name=name)
    if tensor.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long, got {tensor.dtype}")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}")
    return tensor


def require_1d_float(value: Any, *, device: torch.device, name: str) -> torch.Tensor:
    tensor = require_tensor(value, device=device, name=name)
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}")
    return tensor


def require_2d_float(value: Any, *, device: torch.device, name: str) -> torch.Tensor:
    tensor = require_tensor(value, device=device, name=name)
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    return tensor


def require_3d_float(value: Any, *, device: torch.device, name: str) -> torch.Tensor:
    tensor = require_tensor(value, device=device, name=name)
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if tensor.dim() != 3:
        raise ValueError(f"{name} must be 3D, got shape {tuple(tensor.shape)}")
    return tensor


def require_edge_index(value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = require_tensor(value, device=device, name="edge_index")
    if tensor.dtype != torch.long:
        raise TypeError(f"edge_index must be torch.long, got {tensor.dtype}")
    if tensor.dim() != 2 or tensor.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(tensor.shape)}")
    return tensor


def require_bool_1d(value: Any, *, device: torch.device, name: str) -> torch.Tensor:
    tensor = require_tensor(value, device=device, name=name)
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must be torch.bool, got {tensor.dtype}")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}")
    return tensor


def require_bool_2d(value: Any, *, device: torch.device, name: str) -> torch.Tensor:
    tensor = require_tensor(value, device=device, name=name)
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must be torch.bool, got {tensor.dtype}")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    return tensor


def require_str_list(value: Any, *, expected_size: int, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a list/tuple of strings.")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field} must contain strings only.")
    if len(value) != expected_size:
        raise ValueError(f"{field} length mismatch: expected {expected_size}, got {len(value)}")
    return list(value)


def coerce_optional_str_list(value: Any, *, expected_size: int, field: str, default: str = "") -> list[str]:
    if value is None:
        return [default for _ in range(expected_size)]
    if isinstance(value, str):
        if expected_size == 1:
            return [value]
        return [value for _ in range(expected_size)]
    return require_str_list(value, expected_size=expected_size, field=field)


def require_int_attr(batch: Any, name: str, *, min_value: int) -> int:
    value = getattr(batch, name, None)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value)!r}")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def extract_graph_tensors(
    batch: Any, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_ptr = require_1d_long(getattr(batch, "node_ptr", None), device=device, name="node_ptr")
    edge_index = require_edge_index(getattr(batch, "edge_index", None), device=device)
    edge_rel_global = require_1d_long(getattr(batch, "edge_attr", None), device=device, name="edge_attr")
    if edge_rel_global.numel() != int(edge_index.size(1)):
        raise ValueError(
            "edge_attr length mismatch with edge_index. "
            f"edge_attr={edge_rel_global.numel()} edge_index={int(edge_index.size(1))}."
        )
    edge_batch = require_1d_long(getattr(batch, "edge_batch", None), device=device, name="edge_batch")
    node_batch = require_1d_long(getattr(batch, "batch", None), device=device, name="batch")
    return node_ptr, edge_index, edge_rel_global, edge_batch, node_batch


def extract_alignment_tensors(
    batch: Any, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_local_indices = require_1d_long(getattr(batch, "q_local_indices", None), device=device, name="q_local_indices")
    a_local_indices = require_1d_long(getattr(batch, "a_local_indices", None), device=device, name="a_local_indices")
    q_ptr = require_1d_long(getattr(batch, "q_ptr", None), device=device, name="q_ptr")
    a_ptr = require_1d_long(getattr(batch, "a_ptr", None), device=device, name="a_ptr")
    answer_ptr = require_1d_long(getattr(batch, "answer_ptr", None), device=device, name="answer_ptr")
    answer_entity_ids = require_1d_long(
        getattr(batch, "answer_entity_ids", None), device=device, name="answer_entity_ids"
    )
    return q_local_indices, a_local_indices, q_ptr, a_ptr, answer_ptr, answer_entity_ids


def extract_embeddings(
    batch: Any, *, device: torch.device, num_graphs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_embeddings = require_2d_float(getattr(batch, "node_embeddings", None), device=device, name="node_embeddings")
    edge_embeddings = require_2d_float(getattr(batch, "edge_embeddings", None), device=device, name="edge_embeddings")
    question_emb = require_2d_float(getattr(batch, "question_emb", None), device=device, name="question_emb")
    if int(question_emb.size(0)) != num_graphs:
        raise ValueError("question_emb length mismatch with num_graphs.")
    return node_embeddings, edge_embeddings, question_emb


def extract_optional_heuristic_log_v(batch: Any, *, device: torch.device) -> torch.Tensor | None:
    raw_heuristic_log_v = getattr(batch, "heuristic_log_v", None)
    if raw_heuristic_log_v is None:
        return None
    return require_1d_float(raw_heuristic_log_v, device=device, name="heuristic_log_v")


def extract_optional_question_context(
    batch: Any,
    *,
    device: torch.device,
    num_graphs: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    raw_question_ctx = getattr(batch, "question_ctx", None)
    if raw_question_ctx is None:
        return None, None
    question_ctx = require_3d_float(raw_question_ctx, device=device, name="question_ctx")
    if int(question_ctx.size(0)) != num_graphs:
        raise ValueError(
            "question_ctx batch mismatch with num_graphs: "
            f"question_ctx={int(question_ctx.size(0))}, num_graphs={num_graphs}."
        )
    raw_mask = getattr(batch, "question_ctx_mask", None)
    if raw_mask is None:
        return question_ctx, None
    question_ctx_mask = require_bool_2d(raw_mask, device=device, name="question_ctx_mask")
    if tuple(question_ctx_mask.shape) != tuple(question_ctx.shape[:2]):
        raise ValueError(
            "question_ctx_mask shape mismatch with question_ctx: "
            f"mask={tuple(question_ctx_mask.shape)}, context={tuple(question_ctx.shape[:2])}."
        )
    return question_ctx, question_ctx_mask


def extract_metadata(
    batch: Any, *, device: torch.device, num_graphs: int
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    node_global_ids = require_1d_long(getattr(batch, "node_global_ids", None), device=device, name="node_global_ids")
    dummy_mask = require_bool_1d(getattr(batch, "dummy_mask", None), device=device, name="dummy_mask")
    if int(dummy_mask.numel()) != num_graphs:
        raise ValueError("dummy_mask length mismatch with num_graphs.")
    sample_ids = require_str_list(getattr(batch, "sample_id", None), expected_size=num_graphs, field="sample_id")
    questions = coerce_optional_str_list(
        getattr(batch, "question", None), expected_size=num_graphs, field="question", default=""
    )
    return node_global_ids, dummy_mask, sample_ids, questions


class DualFlowBatchAdapter:
    def __init__(self, *, super_source_enabled: bool) -> None:
        self.super_source_enabled = bool(super_source_enabled)

    @staticmethod
    def build_relation_tokens(
        edge_rel_global: torch.Tensor,
        edge_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_rel_global.numel() == 0:
            rel_dim = int(edge_embeddings.size(-1)) if edge_embeddings.dim() == 2 else 0
            return edge_embeddings.new_zeros((0, rel_dim)), edge_rel_global.new_zeros((0,))

        _, edge_relations = torch.unique(edge_rel_global, sorted=True, return_inverse=True)
        num_rel = int(edge_relations.max().item()) + 1
        edge_ids = torch.arange(edge_relations.numel(), device=edge_relations.device, dtype=torch.long)
        first_occ = torch.full((num_rel,), edge_relations.numel(), device=edge_relations.device, dtype=torch.long)
        first_occ.scatter_reduce_(0, edge_relations, edge_ids, reduce="amin", include_self=True)
        return edge_embeddings.index_select(0, first_occ), edge_relations

    def maybe_attach_super_source(
        self,
        *,
        node_ptr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rel_global: torch.Tensor,
        edge_batch: torch.Tensor,
        node_batch: torch.Tensor,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        question_emb: torch.Tensor,
        q_local_indices: torch.Tensor,
        q_ptr: torch.Tensor,
        a_local_indices: torch.Tensor,
        a_ptr: torch.Tensor,
        node_global_ids: torch.Tensor,
        relation_tokens: torch.Tensor,
        edge_relations: torch.Tensor,
        heuristic_log_v: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not self.super_source_enabled:
            return (
                node_ptr,
                edge_index,
                edge_rel_global,
                edge_batch,
                node_batch,
                node_embeddings,
                edge_embeddings,
                node_global_ids,
                relation_tokens,
                edge_relations,
                None,
                None,
                heuristic_log_v,
            )
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        q_counts = (q_ptr[1:] - q_ptr[:-1]).clamp(min=0)
        a_counts = (a_ptr[1:] - a_ptr[:-1]).clamp(min=0)
        if bool((q_counts <= 0).any().item()):
            raise ValueError("super_source_enabled requires non-empty q_local_indices per graph.")
        num_graphs = int(node_counts.numel())
        graph_ids = torch.arange(num_graphs, device=node_ptr.device, dtype=torch.long)
        num_super_nodes_per_graph = 2
        new_counts = node_counts + num_super_nodes_per_graph
        new_node_ptr = torch.zeros_like(node_ptr)
        new_node_ptr[1:] = new_counts.cumsum(dim=0)
        forward_super_local_indices = node_counts
        backward_super_local_indices = node_counts + 1
        forward_super_abs = new_node_ptr[:-1] + forward_super_local_indices
        backward_super_abs = new_node_ptr[:-1] + backward_super_local_indices

        old_num_nodes = int(node_embeddings.size(0))
        new_num_nodes = old_num_nodes + num_graphs * num_super_nodes_per_graph
        old_to_new = (
            torch.arange(old_num_nodes, device=node_ptr.device, dtype=torch.long)
            + node_batch * num_super_nodes_per_graph
        )
        new_node_embeddings = node_embeddings.new_zeros((new_num_nodes, int(node_embeddings.size(1))))
        new_node_embeddings.index_copy_(0, old_to_new, node_embeddings)
        new_heuristic_log_v = None
        if heuristic_log_v is not None:
            if int(heuristic_log_v.numel()) != old_num_nodes:
                raise ValueError(
                    "super_source_enabled requires heuristic_log_v length to match pre-augmentation nodes. "
                    f"heuristic={int(heuristic_log_v.numel())} nodes={old_num_nodes}."
                )
            new_heuristic_log_v = heuristic_log_v.new_zeros((new_num_nodes,))
            new_heuristic_log_v.index_copy_(0, old_to_new, heuristic_log_v)
        if int(question_emb.size(1)) != int(node_embeddings.size(1)):
            raise ValueError(
                "super_source_enabled requires question_emb dim to match node_embeddings dim. "
                f"got question_emb={int(question_emb.size(1))}, node_embeddings={int(node_embeddings.size(1))}"
            )
        new_node_embeddings.index_copy_(0, forward_super_abs, question_emb)
        new_node_embeddings.index_copy_(0, backward_super_abs, question_emb)

        new_node_batch = torch.empty((new_num_nodes,), device=node_batch.device, dtype=torch.long)
        new_node_batch.index_copy_(0, old_to_new, node_batch)
        new_node_batch.index_copy_(0, forward_super_abs, graph_ids)
        new_node_batch.index_copy_(0, backward_super_abs, graph_ids)

        new_node_global_ids = node_global_ids.new_full((new_num_nodes,), fill_value=-1)
        new_node_global_ids.index_copy_(0, old_to_new, node_global_ids)

        src_shift = node_batch.index_select(0, edge_index[0]) * num_super_nodes_per_graph
        dst_shift = node_batch.index_select(0, edge_index[1]) * num_super_nodes_per_graph
        shifted_edge_index = torch.stack(
            [edge_index[0] + src_shift, edge_index[1] + dst_shift],
            dim=0,
        )

        q_offsets = new_node_ptr[:-1].repeat_interleave(q_counts)
        q_abs = q_local_indices + q_offsets
        forward_super_heads = forward_super_abs.repeat_interleave(q_counts)
        forward_super_edge_index = torch.stack([forward_super_heads, q_abs], dim=0)

        a_offsets = new_node_ptr[:-1].repeat_interleave(a_counts)
        a_abs = a_local_indices + a_offsets
        backward_super_heads = backward_super_abs.repeat_interleave(a_counts)
        backward_super_edge_index = torch.stack([backward_super_heads, a_abs], dim=0)

        new_edge_index = shifted_edge_index
        if int(forward_super_heads.numel()) > 0:
            new_edge_index = torch.cat([new_edge_index, forward_super_edge_index], dim=1)
        if int(backward_super_heads.numel()) > 0:
            new_edge_index = torch.cat([new_edge_index, backward_super_edge_index], dim=1)

        base_super_rel_id = int(edge_rel_global.max().item()) + 1 if edge_rel_global.numel() > 0 else 0
        new_edge_rel_global = edge_rel_global
        if int(forward_super_heads.numel()) > 0:
            forward_super_rel = edge_rel_global.new_full((forward_super_heads.numel(),), fill_value=base_super_rel_id)
            new_edge_rel_global = torch.cat([new_edge_rel_global, forward_super_rel], dim=0)
        if int(backward_super_heads.numel()) > 0:
            backward_super_rel = edge_rel_global.new_full(
                (backward_super_heads.numel(),),
                fill_value=base_super_rel_id + 1,
            )
            new_edge_rel_global = torch.cat([new_edge_rel_global, backward_super_rel], dim=0)

        rel_dim = int(edge_embeddings.size(1))
        new_edge_embeddings = edge_embeddings
        new_edge_batch = edge_batch
        if int(forward_super_heads.numel()) > 0:
            forward_super_edge_embeddings = edge_embeddings.new_zeros((forward_super_heads.numel(), rel_dim))
            new_edge_embeddings = torch.cat([new_edge_embeddings, forward_super_edge_embeddings], dim=0)
            new_edge_batch = torch.cat([new_edge_batch, graph_ids.repeat_interleave(q_counts)], dim=0)
        if int(backward_super_heads.numel()) > 0:
            backward_super_edge_embeddings = edge_embeddings.new_zeros((backward_super_heads.numel(), rel_dim))
            new_edge_embeddings = torch.cat([new_edge_embeddings, backward_super_edge_embeddings], dim=0)
            new_edge_batch = torch.cat([new_edge_batch, graph_ids.repeat_interleave(a_counts)], dim=0)

        new_relation_tokens, new_edge_relations = self.build_relation_tokens(new_edge_rel_global, new_edge_embeddings)
        return (
            new_node_ptr,
            new_edge_index,
            new_edge_rel_global,
            new_edge_batch,
            new_node_batch,
            new_node_embeddings,
            new_edge_embeddings,
            new_node_global_ids,
            new_relation_tokens,
            new_edge_relations,
            forward_super_local_indices,
            backward_super_local_indices,
            new_heuristic_log_v,
        )

    def prepare_batch(self, batch: Any, *, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
        num_graphs = require_int_attr(batch, "num_graphs", min_value=1)
        _ = require_int_attr(batch, "num_nodes_total", min_value=0)

        node_ptr, edge_index, edge_rel_global, edge_batch, node_batch = extract_graph_tensors(batch, device=device)
        q_local_indices, a_local_indices, q_ptr, a_ptr, answer_ptr, answer_entity_ids = extract_alignment_tensors(
            batch, device=device
        )
        node_embeddings, edge_embeddings, question_emb = extract_embeddings(batch, device=device, num_graphs=num_graphs)
        heuristic_log_v = extract_optional_heuristic_log_v(batch, device=device)
        question_ctx, question_ctx_mask = extract_optional_question_context(
            batch,
            device=device,
            num_graphs=num_graphs,
        )
        if heuristic_log_v is not None and int(heuristic_log_v.numel()) != int(node_embeddings.size(0)):
            raise ValueError(
                "heuristic_log_v length mismatch with node_embeddings. "
                f"heuristic={int(heuristic_log_v.numel())} nodes={int(node_embeddings.size(0))}."
            )
        relation_tokens, edge_relations = self.build_relation_tokens(edge_rel_global, edge_embeddings)
        node_global_ids, dummy_mask, sample_ids, questions = extract_metadata(
            batch, device=device, num_graphs=num_graphs
        )
        (
            node_ptr,
            edge_index,
            edge_rel_global,
            edge_batch,
            node_batch,
            node_embeddings,
            edge_embeddings,
            node_global_ids,
            relation_tokens,
            edge_relations,
            start_local_indices,
            backward_start_local_indices,
            heuristic_log_v,
        ) = self.maybe_attach_super_source(
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_rel_global=edge_rel_global,
            edge_batch=edge_batch,
            node_batch=node_batch,
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            question_emb=question_emb,
            q_local_indices=q_local_indices,
            q_ptr=q_ptr,
            a_local_indices=a_local_indices,
            a_ptr=a_ptr,
            node_global_ids=node_global_ids,
            relation_tokens=relation_tokens,
            edge_relations=edge_relations,
            heuristic_log_v=heuristic_log_v,
        )
        num_nodes_total = int(node_ptr[-1].item()) if int(node_ptr.numel()) > 0 else 0

        prepared = {
            "num_graphs": num_graphs,
            "num_nodes_total": num_nodes_total,
            "node_ptr": node_ptr,
            "edge_index": edge_index,
            "edge_relations": edge_relations,
            "edge_rel_global": edge_rel_global,
            "edge_batch": edge_batch,
            "node_batch": node_batch,
            "node_embeddings": node_embeddings,
            "node_tokens": node_embeddings,
            "relation_tokens": relation_tokens,
            "question_emb": question_emb,
            "question_ctx": question_ctx,
            "question_ctx_mask": question_ctx_mask,
            "q_local_indices": q_local_indices,
            "a_local_indices": a_local_indices,
            "q_ptr": q_ptr,
            "a_ptr": a_ptr,
            "answer_entity_ids": answer_entity_ids,
            "answer_ptr": answer_ptr,
            "node_global_ids": node_global_ids,
            "dummy_mask": dummy_mask,
            "sample_ids": sample_ids,
            "heuristic_log_v": heuristic_log_v,
            "start_local_indices": start_local_indices,
            "backward_start_local_indices": backward_start_local_indices,
        }
        return prepared, {"questions": questions}


__all__ = ["DualFlowBatchAdapter"]
