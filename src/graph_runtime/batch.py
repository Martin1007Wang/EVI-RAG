from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _require_tensor(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}.")
    return value


def _require_1d_long(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = _require_tensor(value, name=name, device=device)
    if tensor.dtype != torch.long or tensor.dim() != 1:
        raise ValueError(
            f"{name} must be 1D torch.long, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _require_2d_float(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = _require_tensor(value, name=name, device=device)
    if not torch.is_floating_point(tensor) or tensor.dim() != 2:
        raise ValueError(
            f"{name} must be 2D floating point, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _require_3d_float(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = _require_tensor(value, name=name, device=device)
    if not torch.is_floating_point(tensor) or tensor.dim() != 3:
        raise ValueError(
            f"{name} must be 3D floating point, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _require_bool_2d(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    tensor = _require_tensor(value, name=name, device=device)
    if tensor.dtype != torch.bool or tensor.dim() != 2:
        raise ValueError(
            f"{name} must be 2D bool, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _require_edge_index(value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = _require_tensor(value, name="edge_index", device=device)
    if tensor.dtype != torch.long or tensor.dim() != 2 or int(tensor.size(0)) != 2:
        raise ValueError(
            f"edge_index must be [2, E] torch.long, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _coerce_str_list(value: Any, *, expected_size: int, name: str) -> list[str]:
    if value is None:
        return ["" for _ in range(expected_size)]
    if isinstance(value, str):
        if expected_size != 1:
            raise ValueError(
                f"{name} single string cannot represent {expected_size} items."
            )
        return [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be list/tuple[str], got {type(value)!r}.")
    values = [str(item or "") for item in value]
    if len(values) != expected_size:
        raise ValueError(
            f"{name} length mismatch: expected {expected_size}, got {len(values)}."
        )
    return values


@dataclass(frozen=True)
class TrajectoryBatch:
    """Typed runtime batch adapted from a PyG retrieval batch."""

    num_graphs: int
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_rel_global: torch.Tensor
    edge_batch: torch.Tensor
    node_batch: torch.Tensor
    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    question_emb: torch.Tensor
    question_ctx: torch.Tensor
    question_ctx_mask: torch.Tensor
    q_local_indices: torch.Tensor
    q_ptr: torch.Tensor
    a_local_indices: torch.Tensor
    a_ptr: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor
    node_global_ids: torch.Tensor
    sample_ids: list[str]
    questions: list[str]
    dataset_scope: str
    heuristic_log_v: torch.Tensor | None = None

    @property
    def num_nodes_total(self) -> int:
        return int(self.node_ptr[-1].item()) if int(self.node_ptr.numel()) > 0 else 0

    @property
    def dummy_mask(self) -> torch.Tensor:
        counts = self.answer_ptr[1:] - self.answer_ptr[:-1]
        return counts <= 0

    def validate(self) -> None:
        if self.num_graphs < 1:
            raise ValueError("TrajectoryBatch.num_graphs must be >= 1.")
        if int(self.node_ptr.numel()) != self.num_graphs + 1:
            raise ValueError("node_ptr must have length num_graphs + 1.")
        if int(self.q_ptr.numel()) != self.num_graphs + 1:
            raise ValueError("q_ptr must have length num_graphs + 1.")
        if int(self.a_ptr.numel()) != self.num_graphs + 1:
            raise ValueError("a_ptr must have length num_graphs + 1.")
        if int(self.answer_ptr.numel()) != self.num_graphs + 1:
            raise ValueError("answer_ptr must have length num_graphs + 1.")
        if int(self.node_embeddings.size(0)) != self.num_nodes_total:
            raise ValueError("node_embeddings row count mismatch with node_ptr.")
        if int(self.node_batch.numel()) != self.num_nodes_total:
            raise ValueError("node_batch length mismatch with node_ptr.")
        if int(self.node_global_ids.numel()) != self.num_nodes_total:
            raise ValueError("node_global_ids length mismatch with node_ptr.")
        if int(self.edge_index.size(1)) != int(self.edge_rel_global.numel()):
            raise ValueError("edge_index/edge_rel_global mismatch.")
        if int(self.edge_index.size(1)) != int(self.edge_batch.numel()):
            raise ValueError("edge_index/edge_batch mismatch.")
        if int(self.edge_index.size(1)) != int(self.edge_embeddings.size(0)):
            raise ValueError("edge_index/edge_embeddings mismatch.")
        if int(self.question_emb.size(0)) != self.num_graphs:
            raise ValueError("question_emb batch mismatch.")
        if int(self.question_ctx.size(0)) != self.num_graphs:
            raise ValueError("question_ctx batch mismatch.")
        if tuple(self.question_ctx_mask.shape) != tuple(self.question_ctx.shape[:2]):
            raise ValueError("question_ctx_mask shape mismatch with question_ctx.")
        if bool((~self.question_ctx_mask).all(dim=1).any().item()):
            raise ValueError("question_ctx_mask contains rows without valid tokens.")
        if int(self.q_local_indices.numel()) != int(
            (self.q_ptr[1:] - self.q_ptr[:-1]).sum().item()
        ):
            raise ValueError("q_ptr mismatch with q_local_indices length.")
        if int(self.a_local_indices.numel()) != int(
            (self.a_ptr[1:] - self.a_ptr[:-1]).sum().item()
        ):
            raise ValueError("a_ptr mismatch with a_local_indices length.")
        if int(self.answer_entity_ids.numel()) != int(
            (self.answer_ptr[1:] - self.answer_ptr[:-1]).sum().item()
        ):
            raise ValueError("answer_ptr mismatch with answer_entity_ids length.")
        if len(self.sample_ids) != self.num_graphs:
            raise ValueError("sample_ids length mismatch with num_graphs.")
        if len(self.questions) != self.num_graphs:
            raise ValueError("questions length mismatch with num_graphs.")

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        target_device = torch.device(device)
        heuristic_log_v = None
        if self.heuristic_log_v is not None:
            heuristic_log_v = self.heuristic_log_v.to(device=target_device)
        moved = TrajectoryBatch(
            num_graphs=self.num_graphs,
            node_ptr=self.node_ptr.to(device=target_device),
            edge_index=self.edge_index.to(device=target_device),
            edge_rel_global=self.edge_rel_global.to(device=target_device),
            edge_batch=self.edge_batch.to(device=target_device),
            node_batch=self.node_batch.to(device=target_device),
            node_embeddings=self.node_embeddings.to(device=target_device),
            edge_embeddings=self.edge_embeddings.to(device=target_device),
            question_emb=self.question_emb.to(device=target_device),
            question_ctx=self.question_ctx.to(device=target_device),
            question_ctx_mask=self.question_ctx_mask.to(device=target_device),
            q_local_indices=self.q_local_indices.to(device=target_device),
            q_ptr=self.q_ptr.to(device=target_device),
            a_local_indices=self.a_local_indices.to(device=target_device),
            a_ptr=self.a_ptr.to(device=target_device),
            answer_entity_ids=self.answer_entity_ids.to(device=target_device),
            answer_ptr=self.answer_ptr.to(device=target_device),
            node_global_ids=self.node_global_ids.to(device=target_device),
            sample_ids=list(self.sample_ids),
            questions=list(self.questions),
            dataset_scope=self.dataset_scope,
            heuristic_log_v=heuristic_log_v,
        )
        moved.validate()
        return moved

    @classmethod
    def from_pyg_batch(
        cls,
        batch: Any,
        *,
        device: torch.device,
        dataset_scope: str,
    ) -> "TrajectoryBatch":
        num_graphs = int(getattr(batch, "num_graphs", 0))
        if num_graphs < 1:
            raise ValueError("PyG batch must define num_graphs >= 1.")
        node_ptr = _require_1d_long(
            getattr(batch, "node_ptr", None), name="node_ptr", device=device
        )
        edge_index = _require_edge_index(
            getattr(batch, "edge_index", None), device=device
        )
        edge_rel_global = _require_1d_long(
            getattr(batch, "edge_attr", None), name="edge_attr", device=device
        )
        edge_batch = _require_1d_long(
            getattr(batch, "edge_batch", None), name="edge_batch", device=device
        )
        node_batch = _require_1d_long(
            getattr(batch, "batch", None), name="batch", device=device
        )
        node_embeddings = _require_2d_float(
            getattr(batch, "node_embeddings", None),
            name="node_embeddings",
            device=device,
        )
        edge_embeddings = _require_2d_float(
            getattr(batch, "edge_embeddings", None),
            name="edge_embeddings",
            device=device,
        )
        question_emb = _require_2d_float(
            getattr(batch, "question_emb", None), name="question_emb", device=device
        )
        question_ctx = _require_3d_float(
            getattr(batch, "question_ctx", None), name="question_ctx", device=device
        )
        question_ctx_mask = _require_bool_2d(
            getattr(batch, "question_ctx_mask", None),
            name="question_ctx_mask",
            device=device,
        )
        q_local_indices = _require_1d_long(
            getattr(batch, "q_local_indices", None),
            name="q_local_indices",
            device=device,
        )
        q_ptr = _require_1d_long(
            getattr(batch, "q_ptr", None), name="q_ptr", device=device
        )
        a_local_indices = _require_1d_long(
            getattr(batch, "a_local_indices", None),
            name="a_local_indices",
            device=device,
        )
        a_ptr = _require_1d_long(
            getattr(batch, "a_ptr", None), name="a_ptr", device=device
        )
        answer_entity_ids = _require_1d_long(
            getattr(batch, "answer_entity_ids", None),
            name="answer_entity_ids",
            device=device,
        )
        answer_ptr = _require_1d_long(
            getattr(batch, "answer_ptr", None), name="answer_ptr", device=device
        )
        node_global_ids = _require_1d_long(
            getattr(batch, "node_global_ids", None),
            name="node_global_ids",
            device=device,
        )
        heuristic_log_v = getattr(batch, "heuristic_log_v", None)
        if heuristic_log_v is not None:
            heuristic_log_v = _require_tensor(
                heuristic_log_v, name="heuristic_log_v", device=device
            )
        trajectory_batch = cls(
            num_graphs=num_graphs,
            node_ptr=node_ptr,
            edge_index=edge_index,
            edge_rel_global=edge_rel_global,
            edge_batch=edge_batch,
            node_batch=node_batch,
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            question_emb=question_emb,
            question_ctx=question_ctx,
            question_ctx_mask=question_ctx_mask,
            q_local_indices=q_local_indices,
            q_ptr=q_ptr,
            a_local_indices=a_local_indices,
            a_ptr=a_ptr,
            answer_entity_ids=answer_entity_ids,
            answer_ptr=answer_ptr,
            node_global_ids=node_global_ids,
            sample_ids=_coerce_str_list(
                getattr(batch, "sample_id", None),
                expected_size=num_graphs,
                name="sample_id",
            ),
            questions=_coerce_str_list(
                getattr(batch, "question", None),
                expected_size=num_graphs,
                name="question",
            ),
            dataset_scope=str(dataset_scope),
            heuristic_log_v=heuristic_log_v,
        )
        trajectory_batch.validate()
        return trajectory_batch

    @classmethod
    def concatenate(cls, batches: list["TrajectoryBatch"]) -> "TrajectoryBatch":
        if not batches:
            raise ValueError("TrajectoryBatch.concatenate requires at least one batch.")
        if len(batches) == 1:
            return batches[0]

        device = batches[0].node_ptr.device
        dataset_scope = str(batches[0].dataset_scope)
        has_heuristic = batches[0].heuristic_log_v is not None

        num_graphs = 0
        node_offset = 0
        node_ptr_values = [0]
        q_ptr_values = [0]
        a_ptr_values = [0]
        answer_ptr_values = [0]

        edge_index_parts: list[torch.Tensor] = []
        edge_rel_parts: list[torch.Tensor] = []
        edge_batch_parts: list[torch.Tensor] = []
        node_batch_parts: list[torch.Tensor] = []
        node_embedding_parts: list[torch.Tensor] = []
        edge_embedding_parts: list[torch.Tensor] = []
        question_emb_parts: list[torch.Tensor] = []
        question_ctx_parts: list[torch.Tensor] = []
        question_ctx_mask_parts: list[torch.Tensor] = []
        q_local_parts: list[torch.Tensor] = []
        a_local_parts: list[torch.Tensor] = []
        answer_entity_parts: list[torch.Tensor] = []
        node_global_parts: list[torch.Tensor] = []
        heuristic_parts: list[torch.Tensor] = []
        sample_ids: list[str] = []
        questions: list[str] = []

        for batch in batches:
            batch.validate()
            if batch.node_ptr.device != device:
                raise ValueError(
                    "All TrajectoryBatch instances must share the same device."
                )
            if str(batch.dataset_scope) != dataset_scope:
                raise ValueError(
                    "All TrajectoryBatch instances must share dataset_scope."
                )
            if (batch.heuristic_log_v is not None) != has_heuristic:
                raise ValueError(
                    "All TrajectoryBatch instances must either all include heuristic_log_v or all omit it."
                )

            node_counts = (batch.node_ptr[1:] - batch.node_ptr[:-1]).tolist()
            q_counts = (batch.q_ptr[1:] - batch.q_ptr[:-1]).tolist()
            a_counts = (batch.a_ptr[1:] - batch.a_ptr[:-1]).tolist()
            answer_counts = (batch.answer_ptr[1:] - batch.answer_ptr[:-1]).tolist()

            for count in node_counts:
                node_ptr_values.append(node_ptr_values[-1] + int(count))
            for count in q_counts:
                q_ptr_values.append(q_ptr_values[-1] + int(count))
            for count in a_counts:
                a_ptr_values.append(a_ptr_values[-1] + int(count))
            for count in answer_counts:
                answer_ptr_values.append(answer_ptr_values[-1] + int(count))

            edge_index_parts.append(batch.edge_index + int(node_offset))
            edge_rel_parts.append(batch.edge_rel_global)
            edge_batch_parts.append(batch.edge_batch + int(num_graphs))
            node_batch_parts.append(batch.node_batch + int(num_graphs))
            node_embedding_parts.append(batch.node_embeddings)
            edge_embedding_parts.append(batch.edge_embeddings)
            question_emb_parts.append(batch.question_emb)
            question_ctx_parts.append(batch.question_ctx)
            question_ctx_mask_parts.append(batch.question_ctx_mask)
            q_local_parts.append(batch.q_local_indices)
            a_local_parts.append(batch.a_local_indices)
            answer_entity_parts.append(batch.answer_entity_ids)
            node_global_parts.append(batch.node_global_ids)
            if has_heuristic and batch.heuristic_log_v is not None:
                heuristic_parts.append(batch.heuristic_log_v)

            sample_ids.extend(batch.sample_ids)
            questions.extend(batch.questions)
            node_offset += batch.num_nodes_total
            num_graphs += int(batch.num_graphs)

        heuristic_log_v = None
        if has_heuristic:
            heuristic_log_v = torch.cat(heuristic_parts, dim=0)

        concatenated = cls(
            num_graphs=num_graphs,
            node_ptr=torch.tensor(node_ptr_values, device=device, dtype=torch.long),
            edge_index=torch.cat(edge_index_parts, dim=1),
            edge_rel_global=torch.cat(edge_rel_parts, dim=0),
            edge_batch=torch.cat(edge_batch_parts, dim=0),
            node_batch=torch.cat(node_batch_parts, dim=0),
            node_embeddings=torch.cat(node_embedding_parts, dim=0),
            edge_embeddings=torch.cat(edge_embedding_parts, dim=0),
            question_emb=torch.cat(question_emb_parts, dim=0),
            question_ctx=torch.cat(question_ctx_parts, dim=0),
            question_ctx_mask=torch.cat(question_ctx_mask_parts, dim=0),
            q_local_indices=torch.cat(q_local_parts, dim=0),
            q_ptr=torch.tensor(q_ptr_values, device=device, dtype=torch.long),
            a_local_indices=torch.cat(a_local_parts, dim=0),
            a_ptr=torch.tensor(a_ptr_values, device=device, dtype=torch.long),
            answer_entity_ids=torch.cat(answer_entity_parts, dim=0),
            answer_ptr=torch.tensor(answer_ptr_values, device=device, dtype=torch.long),
            node_global_ids=torch.cat(node_global_parts, dim=0),
            sample_ids=sample_ids,
            questions=questions,
            dataset_scope=dataset_scope,
            heuristic_log_v=heuristic_log_v,
        )
        concatenated.validate()
        return concatenated

    def select_graph(self, graph_idx: int) -> "TrajectoryBatch":
        if graph_idx < 0 or graph_idx >= self.num_graphs:
            raise IndexError(f"graph_idx out of range: {graph_idx}.")
        node_start = int(self.node_ptr[graph_idx].item())
        node_end = int(self.node_ptr[graph_idx + 1].item())
        edge_mask = self.edge_batch == graph_idx
        edge_index = self.edge_index[:, edge_mask] - node_start
        edge_rel_global = self.edge_rel_global[edge_mask]
        edge_embeddings = self.edge_embeddings[edge_mask]
        num_nodes = node_end - node_start
        q_start = int(self.q_ptr[graph_idx].item())
        q_end = int(self.q_ptr[graph_idx + 1].item())
        a_start = int(self.a_ptr[graph_idx].item())
        a_end = int(self.a_ptr[graph_idx + 1].item())
        answer_start = int(self.answer_ptr[graph_idx].item())
        answer_end = int(self.answer_ptr[graph_idx + 1].item())
        heuristic_log_v = None
        if self.heuristic_log_v is not None:
            heuristic_log_v = self.heuristic_log_v[node_start:node_end]
        sub_batch = TrajectoryBatch(
            num_graphs=1,
            node_ptr=torch.tensor(
                [0, num_nodes], device=self.node_ptr.device, dtype=torch.long
            ),
            edge_index=edge_index,
            edge_rel_global=edge_rel_global,
            edge_batch=torch.zeros_like(edge_rel_global),
            node_batch=torch.zeros(
                (num_nodes,), device=self.node_batch.device, dtype=torch.long
            ),
            node_embeddings=self.node_embeddings[node_start:node_end],
            edge_embeddings=edge_embeddings,
            question_emb=self.question_emb[graph_idx : graph_idx + 1],
            question_ctx=self.question_ctx[graph_idx : graph_idx + 1],
            question_ctx_mask=self.question_ctx_mask[graph_idx : graph_idx + 1],
            q_local_indices=self.q_local_indices[q_start:q_end],
            q_ptr=torch.tensor(
                [0, q_end - q_start], device=self.q_ptr.device, dtype=torch.long
            ),
            a_local_indices=self.a_local_indices[a_start:a_end],
            a_ptr=torch.tensor(
                [0, a_end - a_start], device=self.a_ptr.device, dtype=torch.long
            ),
            answer_entity_ids=self.answer_entity_ids[answer_start:answer_end],
            answer_ptr=torch.tensor(
                [0, answer_end - answer_start],
                device=self.answer_ptr.device,
                dtype=torch.long,
            ),
            node_global_ids=self.node_global_ids[node_start:node_end],
            sample_ids=[self.sample_ids[graph_idx]],
            questions=[self.questions[graph_idx]],
            dataset_scope=self.dataset_scope,
            heuristic_log_v=heuristic_log_v,
        )
        sub_batch.validate()
        return sub_batch


__all__ = ["TrajectoryBatch"]
