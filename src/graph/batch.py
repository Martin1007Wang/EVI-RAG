from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from .batch_utils import (
    build_edge_ptr_from_edge_batch,
    build_relation_table_from_rows,
    coerce_str_list,
    compact_relation_table,
    compute_edge_batch_and_ptr,
    move_float_feature,
    require_1d_long,
    require_2d_float,
    require_3d_float,
    require_bool_2d,
    require_edge_index,
    require_tensor,
)


@dataclass(frozen=True)
class TrajectoryBatch:
    """Typed runtime batch adapted from a PyG retrieval batch."""

    num_graphs: int
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_rel_global: torch.Tensor
    edge_batch: torch.Tensor
    node_batch: torch.Tensor
    node_embeddings: torch.Tensor | None
    edge_embeddings: torch.Tensor | None
    question_emb: torch.Tensor | None
    question_ctx: torch.Tensor | None
    question_ctx_mask: torch.Tensor | None
    # These are the in-graph grounded question entities that seed rollout.
    anchor_local_indices: torch.Tensor
    anchor_ptr: torch.Tensor
    a_local_indices: torch.Tensor
    a_ptr: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor
    node_entity_ids: torch.Tensor
    sample_ids: list[str]
    questions: list[str]
    dataset_scope: str
    heuristic_log_v: torch.Tensor | None = None
    relation_embeddings: torch.Tensor | None = None
    edge_rel_local: torch.Tensor | None = None

    @property
    def num_nodes_total(self) -> int:
        return int(self.node_ptr[-1].item()) if int(self.node_ptr.numel()) > 0 else 0

    @cached_property
    def edge_ptr(self) -> torch.Tensor:
        return build_edge_ptr_from_edge_batch(
            self.edge_batch,
            num_graphs=self.num_graphs,
            device=self.edge_batch.device,
            validate=True,
        )

    @property
    def dummy_mask(self) -> torch.Tensor:
        counts = self.answer_ptr[1:] - self.answer_ptr[:-1]
        return counts <= 0

    @property
    def has_raw_features(self) -> bool:
        return all(
            value is not None
            for value in (
                self.node_embeddings,
                self.question_emb,
                self.question_ctx,
                self.question_ctx_mask,
            )
        )

    def require_raw_features(self) -> None:
        if self.has_raw_features:
            return
        raise ValueError(
            "TrajectoryBatch is missing raw float features required for encoding. "
            "Use the pre-encoded PreparedBatch path or keep node/question features attached."
        )

    def without_raw_features(self) -> "TrajectoryBatch":
        return TrajectoryBatch(
            num_graphs=self.num_graphs,
            node_ptr=self.node_ptr,
            edge_index=self.edge_index,
            edge_rel_global=self.edge_rel_global,
            edge_batch=self.edge_batch,
            node_batch=self.node_batch,
            node_embeddings=None,
            edge_embeddings=None,
            question_emb=None,
            question_ctx=None,
            question_ctx_mask=None,
            anchor_local_indices=self.anchor_local_indices,
            anchor_ptr=self.anchor_ptr,
            a_local_indices=self.a_local_indices,
            a_ptr=self.a_ptr,
            answer_entity_ids=self.answer_entity_ids,
            answer_ptr=self.answer_ptr,
            node_entity_ids=self.node_entity_ids,
            sample_ids=list(self.sample_ids),
            questions=list(self.questions),
            dataset_scope=self.dataset_scope,
            heuristic_log_v=None,
            relation_embeddings=None,
            edge_rel_local=None,
        )

    def validate(self) -> None:
        if self.num_graphs < 1:
            raise ValueError("TrajectoryBatch.num_graphs must be >= 1.")
        if int(self.node_ptr.numel()) != self.num_graphs + 1:
            raise ValueError("node_ptr must have length num_graphs + 1.")
        if int(self.anchor_ptr.numel()) != self.num_graphs + 1:
            raise ValueError("anchor_ptr must have length num_graphs + 1.")
        if int(self.a_ptr.numel()) != self.num_graphs + 1:
            raise ValueError("a_ptr must have length num_graphs + 1.")
        if int(self.answer_ptr.numel()) != self.num_graphs + 1:
            raise ValueError("answer_ptr must have length num_graphs + 1.")
        if int(self.node_batch.numel()) != self.num_nodes_total:
            raise ValueError("node_batch length mismatch with node_ptr.")
        if int(self.node_entity_ids.numel()) != self.num_nodes_total:
            raise ValueError("node_entity_ids length mismatch with node_ptr.")
        if int(self.edge_index.size(1)) != int(self.edge_rel_global.numel()):
            raise ValueError("edge_index/edge_rel_global mismatch.")
        if int(self.edge_index.size(1)) != int(self.edge_batch.numel()):
            raise ValueError("edge_index/edge_batch mismatch.")
        actual_edge_ptr = self.edge_ptr
        computed_edge_batch, computed_edge_ptr = compute_edge_batch_and_ptr(
            self.edge_index,
            node_ptr=self.node_ptr,
            num_graphs=self.num_graphs,
            device=self.edge_index.device,
            validate=True,
        )
        if not torch.equal(self.edge_batch, computed_edge_batch):
            raise ValueError("edge_batch mismatch with edge_index/node_ptr.")
        if not torch.equal(actual_edge_ptr, computed_edge_ptr):
            raise ValueError("edge_ptr mismatch with edge_batch.")
        if int(self.anchor_local_indices.numel()) != int(
            (self.anchor_ptr[1:] - self.anchor_ptr[:-1]).sum().item()
        ):
            raise ValueError("anchor_ptr mismatch with anchor_local_indices length.")
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

        raw_feature_presence = (
            self.node_embeddings,
            self.question_emb,
            self.question_ctx,
            self.question_ctx_mask,
        )
        has_any_raw_feature = any(value is not None for value in raw_feature_presence)
        if has_any_raw_feature and not self.has_raw_features:
            raise ValueError(
                "node_embeddings/question_emb/question_ctx/question_ctx_mask must either "
                "all be populated or all be omitted."
            )
        if self.has_raw_features:
            node_embeddings = self.node_embeddings
            question_emb = self.question_emb
            question_ctx = self.question_ctx
            question_ctx_mask = self.question_ctx_mask
            assert node_embeddings is not None
            assert question_emb is not None
            assert question_ctx is not None
            assert question_ctx_mask is not None
            if int(node_embeddings.size(0)) != self.num_nodes_total:
                raise ValueError("node_embeddings row count mismatch with node_ptr.")
            if int(question_emb.size(0)) != self.num_graphs:
                raise ValueError("question_emb batch mismatch.")
            if int(question_ctx.size(0)) != self.num_graphs:
                raise ValueError("question_ctx batch mismatch.")
            if tuple(question_ctx_mask.shape) != tuple(question_ctx.shape[:2]):
                raise ValueError("question_ctx_mask shape mismatch with question_ctx.")
            if bool((~question_ctx_mask).all(dim=1).any().item()):
                raise ValueError(
                    "question_ctx_mask contains rows without valid tokens."
                )

        has_relation_table = (
            self.relation_embeddings is not None or self.edge_rel_local is not None
        )
        if has_relation_table:
            if self.relation_embeddings is None or self.edge_rel_local is None:
                raise ValueError(
                    "relation_embeddings and edge_rel_local must be provided together."
                )
            if (
                not torch.is_floating_point(self.relation_embeddings)
                or self.relation_embeddings.dim() != 2
            ):
                raise ValueError(
                    "relation_embeddings must be 2D floating point when provided."
                )
            if (
                self.edge_rel_local.dtype != torch.long
                or self.edge_rel_local.dim() != 1
            ):
                raise ValueError("edge_rel_local must be 1D torch.long when provided.")
            if int(self.edge_rel_local.numel()) != int(self.edge_index.size(1)):
                raise ValueError("edge_index/edge_rel_local mismatch.")
            if int(self.edge_rel_local.numel()) > 0:
                if bool((self.edge_rel_local < 0).any().item()) or bool(
                    (self.edge_rel_local >= int(self.relation_embeddings.size(0)))
                    .any()
                    .item()
                ):
                    raise ValueError(
                        "edge_rel_local contains out-of-range indices for relation_embeddings."
                    )
        elif self.edge_embeddings is not None:
            if int(self.edge_index.size(1)) != int(self.edge_embeddings.size(0)):
                raise ValueError("edge_index/edge_embeddings mismatch.")
        elif self.has_raw_features and int(self.edge_index.size(1)) > 0:
            raise ValueError(
                "Raw TrajectoryBatch with edges must provide either edge_embeddings or "
                "(relation_embeddings, edge_rel_local)."
            )

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype | None = None,
    ) -> "TrajectoryBatch":
        target_device = torch.device(device)
        heuristic_log_v = None
        if self.heuristic_log_v is not None:
            heuristic_log_v = move_float_feature(
                self.heuristic_log_v,
                device=target_device,
                dtype=feature_dtype,
            )
        node_embeddings = None
        if self.node_embeddings is not None:
            node_embeddings = move_float_feature(
                self.node_embeddings,
                device=target_device,
                dtype=feature_dtype,
            )
        edge_embeddings = None
        if self.edge_embeddings is not None:
            edge_embeddings = move_float_feature(
                self.edge_embeddings,
                device=target_device,
                dtype=feature_dtype,
            )
        question_emb = None
        if self.question_emb is not None:
            question_emb = move_float_feature(
                self.question_emb,
                device=target_device,
                dtype=feature_dtype,
            )
        question_ctx = None
        if self.question_ctx is not None:
            question_ctx = move_float_feature(
                self.question_ctx,
                device=target_device,
                dtype=feature_dtype,
            )
        question_ctx_mask = None
        if self.question_ctx_mask is not None:
            question_ctx_mask = self.question_ctx_mask.to(device=target_device)
        relation_embeddings = None
        if self.relation_embeddings is not None:
            relation_embeddings = move_float_feature(
                self.relation_embeddings,
                device=target_device,
                dtype=feature_dtype,
            )
        edge_rel_local = None
        if self.edge_rel_local is not None:
            edge_rel_local = self.edge_rel_local.to(device=target_device)
        return TrajectoryBatch(
            num_graphs=self.num_graphs,
            node_ptr=self.node_ptr.to(device=target_device),
            edge_index=self.edge_index.to(device=target_device),
            edge_rel_global=self.edge_rel_global.to(device=target_device),
            edge_batch=self.edge_batch.to(device=target_device),
            node_batch=self.node_batch.to(device=target_device),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            question_emb=question_emb,
            question_ctx=question_ctx,
            question_ctx_mask=question_ctx_mask,
            anchor_local_indices=self.anchor_local_indices.to(device=target_device),
            anchor_ptr=self.anchor_ptr.to(device=target_device),
            a_local_indices=self.a_local_indices.to(device=target_device),
            a_ptr=self.a_ptr.to(device=target_device),
            answer_entity_ids=self.answer_entity_ids.to(device=target_device),
            answer_ptr=self.answer_ptr.to(device=target_device),
            node_entity_ids=self.node_entity_ids.to(device=target_device),
            sample_ids=list(self.sample_ids),
            questions=list(self.questions),
            dataset_scope=self.dataset_scope,
            heuristic_log_v=heuristic_log_v,
            relation_embeddings=relation_embeddings,
            edge_rel_local=edge_rel_local,
        )

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
        node_ptr = require_1d_long(
            getattr(batch, "node_ptr", None), name="node_ptr", device=device
        )
        edge_index = require_edge_index(
            getattr(batch, "edge_index", None), device=device
        )
        edge_rel_global = require_1d_long(
            getattr(batch, "edge_attr", None), name="edge_attr", device=device
        )
        edge_batch_value = getattr(batch, "edge_batch", None)
        if edge_batch_value is None:
            edge_batch, _ = compute_edge_batch_and_ptr(
                edge_index,
                node_ptr=node_ptr,
                num_graphs=num_graphs,
                device=device,
                validate=True,
            )
        else:
            edge_batch = require_1d_long(
                edge_batch_value, name="edge_batch", device=device
            )
        node_batch = require_1d_long(
            getattr(batch, "batch", None), name="batch", device=device
        )
        node_embeddings = require_2d_float(
            getattr(batch, "node_embeddings", None),
            name="node_embeddings",
            device=device,
        )
        edge_embeddings_value = getattr(batch, "edge_embeddings", None)
        edge_embeddings = None
        if edge_embeddings_value is not None:
            edge_embeddings = require_2d_float(
                edge_embeddings_value,
                name="edge_embeddings",
                device=device,
            )
        relation_embeddings_value = getattr(batch, "relation_embeddings", None)
        relation_embeddings = None
        if relation_embeddings_value is not None:
            relation_embeddings = require_2d_float(
                relation_embeddings_value,
                name="relation_embeddings",
                device=device,
            )
        edge_rel_local_value = getattr(batch, "edge_rel_local", None)
        edge_rel_local = None
        if edge_rel_local_value is not None:
            edge_rel_local = require_1d_long(
                edge_rel_local_value,
                name="edge_rel_local",
                device=device,
            )
        question_emb = require_2d_float(
            getattr(batch, "question_emb", None), name="question_emb", device=device
        )
        question_ctx = require_3d_float(
            getattr(batch, "question_ctx", None), name="question_ctx", device=device
        )
        question_ctx_mask = require_bool_2d(
            getattr(batch, "question_ctx_mask", None),
            name="question_ctx_mask",
            device=device,
        )
        anchor_local_indices = require_1d_long(
            getattr(batch, "anchor_local_indices", None),
            name="anchor_local_indices",
            device=device,
        )
        anchor_ptr = require_1d_long(
            getattr(batch, "anchor_ptr", None), name="anchor_ptr", device=device
        )
        a_local_indices = require_1d_long(
            getattr(batch, "a_local_indices", None),
            name="a_local_indices",
            device=device,
        )
        a_ptr = require_1d_long(
            getattr(batch, "a_ptr", None), name="a_ptr", device=device
        )
        answer_entity_ids = require_1d_long(
            getattr(batch, "answer_entity_ids", None),
            name="answer_entity_ids",
            device=device,
        )
        answer_ptr = require_1d_long(
            getattr(batch, "answer_ptr", None), name="answer_ptr", device=device
        )
        node_entity_ids = require_1d_long(
            getattr(batch, "node_entity_ids", None),
            name="node_entity_ids",
            device=device,
        )
        heuristic_log_v = getattr(batch, "heuristic_log_v", None)
        if heuristic_log_v is not None:
            heuristic_log_v = require_tensor(
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
            anchor_local_indices=anchor_local_indices,
            anchor_ptr=anchor_ptr,
            a_local_indices=a_local_indices,
            a_ptr=a_ptr,
            answer_entity_ids=answer_entity_ids,
            answer_ptr=answer_ptr,
            node_entity_ids=node_entity_ids,
            sample_ids=coerce_str_list(
                getattr(batch, "sample_id", None),
                expected_size=num_graphs,
                name="sample_id",
            ),
            questions=coerce_str_list(
                getattr(batch, "question", None),
                expected_size=num_graphs,
                name="question",
            ),
            dataset_scope=str(dataset_scope),
            heuristic_log_v=heuristic_log_v,
            relation_embeddings=relation_embeddings,
            edge_rel_local=edge_rel_local,
        )
        trajectory_batch.validate()
        return trajectory_batch

    @classmethod
    def concatenate(
        cls, batches: list["TrajectoryBatch"], *, validate: bool = True
    ) -> "TrajectoryBatch":
        if not batches:
            raise ValueError("TrajectoryBatch.concatenate requires at least one batch.")
        if len(batches) == 1:
            return batches[0]

        device = batches[0].node_ptr.device
        dataset_scope = str(batches[0].dataset_scope)
        has_heuristic = batches[0].heuristic_log_v is not None
        has_node_embeddings = batches[0].node_embeddings is not None
        has_question_features = batches[0].question_emb is not None
        has_edge_embeddings = batches[0].edge_embeddings is not None
        has_relation_tables = (
            batches[0].relation_embeddings is not None
            and batches[0].edge_rel_local is not None
        )

        num_graphs = 0
        node_offset = 0
        node_ptr_values = [0]
        anchor_ptr_values = [0]
        a_ptr_values = [0]
        answer_ptr_values = [0]

        edge_index_parts: list[torch.Tensor] = []
        edge_rel_parts: list[torch.Tensor] = []
        edge_batch_parts: list[torch.Tensor] = []
        node_batch_parts: list[torch.Tensor] = []
        node_embedding_parts: list[torch.Tensor] = []
        edge_embedding_parts: list[torch.Tensor] = []
        relation_global_parts: list[torch.Tensor] = []
        relation_embedding_parts: list[torch.Tensor] = []
        question_emb_parts: list[torch.Tensor] = []
        question_ctx_parts: list[torch.Tensor] = []
        question_ctx_mask_parts: list[torch.Tensor] = []
        anchor_local_parts: list[torch.Tensor] = []
        a_local_parts: list[torch.Tensor] = []
        answer_entity_parts: list[torch.Tensor] = []
        node_entity_parts: list[torch.Tensor] = []
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
            if (batch.node_embeddings is not None) != has_node_embeddings:
                raise ValueError(
                    "All TrajectoryBatch instances must either all include node_embeddings or all omit them."
                )
            if (batch.question_emb is not None) != has_question_features:
                raise ValueError(
                    "All TrajectoryBatch instances must either all include question features or all omit them."
                )
            batch_has_relation_table = (
                batch.relation_embeddings is not None
                and batch.edge_rel_local is not None
            )
            if (batch.edge_embeddings is not None) != has_edge_embeddings:
                raise ValueError(
                    "All TrajectoryBatch instances must either all include edge_embeddings or all omit them."
                )
            if batch_has_relation_table != has_relation_tables:
                raise ValueError(
                    "All TrajectoryBatch instances must share the same relation-table representation."
                )

            node_counts = (batch.node_ptr[1:] - batch.node_ptr[:-1]).tolist()
            anchor_counts = (batch.anchor_ptr[1:] - batch.anchor_ptr[:-1]).tolist()
            a_counts = (batch.a_ptr[1:] - batch.a_ptr[:-1]).tolist()
            answer_counts = (batch.answer_ptr[1:] - batch.answer_ptr[:-1]).tolist()

            for count in node_counts:
                node_ptr_values.append(node_ptr_values[-1] + int(count))
            for count in anchor_counts:
                anchor_ptr_values.append(anchor_ptr_values[-1] + int(count))
            for count in a_counts:
                a_ptr_values.append(a_ptr_values[-1] + int(count))
            for count in answer_counts:
                answer_ptr_values.append(answer_ptr_values[-1] + int(count))

            edge_index_parts.append(batch.edge_index + int(node_offset))
            edge_rel_parts.append(batch.edge_rel_global)
            edge_batch_parts.append(batch.edge_batch + int(num_graphs))
            node_batch_parts.append(batch.node_batch + int(num_graphs))
            if has_node_embeddings and batch.node_embeddings is not None:
                node_embedding_parts.append(batch.node_embeddings)
            if has_question_features and batch.question_emb is not None:
                question_emb_parts.append(batch.question_emb)
                question_ctx_parts.append(batch.question_ctx)
                question_ctx_mask_parts.append(batch.question_ctx_mask)
            if has_edge_embeddings and batch.edge_embeddings is not None:
                edge_embedding_parts.append(batch.edge_embeddings)
            elif (
                has_relation_tables
                and batch.relation_embeddings is not None
                and batch.edge_rel_local is not None
            ):
                compact_relation_ids, compact_relation_embeddings, _ = (
                    compact_relation_table(
                        edge_rel_global=batch.edge_rel_global,
                        relation_embeddings=batch.relation_embeddings,
                        edge_rel_local=batch.edge_rel_local,
                    )
                )
                relation_global_parts.append(compact_relation_ids)
                relation_embedding_parts.append(compact_relation_embeddings)
            anchor_local_parts.append(batch.anchor_local_indices)
            a_local_parts.append(batch.a_local_indices)
            answer_entity_parts.append(batch.answer_entity_ids)
            node_entity_parts.append(batch.node_entity_ids)
            if has_heuristic and batch.heuristic_log_v is not None:
                heuristic_parts.append(batch.heuristic_log_v)

            sample_ids.extend(batch.sample_ids)
            questions.extend(batch.questions)
            node_offset += batch.num_nodes_total
            num_graphs += int(batch.num_graphs)

        heuristic_log_v = None
        if has_heuristic:
            heuristic_log_v = torch.cat(heuristic_parts, dim=0)

        edge_rel_global = torch.cat(edge_rel_parts, dim=0)
        node_embeddings = None
        if has_node_embeddings:
            node_embeddings = torch.cat(node_embedding_parts, dim=0)
        edge_embeddings = None
        relation_embeddings = None
        edge_rel_local = None
        if has_edge_embeddings:
            edge_embeddings = torch.cat(edge_embedding_parts, dim=0)
        elif has_relation_tables:
            flat_relation_global_ids = torch.cat(relation_global_parts, dim=0)
            flat_relation_embeddings = torch.cat(relation_embedding_parts, dim=0)
            relation_ids, relation_embeddings = build_relation_table_from_rows(
                relation_global_ids=flat_relation_global_ids,
                relation_embeddings=flat_relation_embeddings,
            )
            relation_ids_from_edges, edge_rel_local = torch.unique(
                edge_rel_global, sorted=True, return_inverse=True
            )
            if not torch.equal(relation_ids, relation_ids_from_edges):
                raise ValueError(
                    "Relation table rows are inconsistent with concatenated edge_rel_global values."
                )
        question_emb = None
        question_ctx = None
        question_ctx_mask = None
        if has_question_features:
            question_emb = torch.cat(question_emb_parts, dim=0)
            question_ctx = torch.cat(question_ctx_parts, dim=0)
            question_ctx_mask = torch.cat(question_ctx_mask_parts, dim=0)

        concatenated = cls(
            num_graphs=num_graphs,
            node_ptr=torch.tensor(node_ptr_values, device=device, dtype=torch.long),
            edge_index=torch.cat(edge_index_parts, dim=1),
            edge_rel_global=edge_rel_global,
            edge_batch=torch.cat(edge_batch_parts, dim=0),
            node_batch=torch.cat(node_batch_parts, dim=0),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            question_emb=question_emb,
            question_ctx=question_ctx,
            question_ctx_mask=question_ctx_mask,
            anchor_local_indices=torch.cat(anchor_local_parts, dim=0),
            anchor_ptr=torch.tensor(anchor_ptr_values, device=device, dtype=torch.long),
            a_local_indices=torch.cat(a_local_parts, dim=0),
            a_ptr=torch.tensor(a_ptr_values, device=device, dtype=torch.long),
            answer_entity_ids=torch.cat(answer_entity_parts, dim=0),
            answer_ptr=torch.tensor(answer_ptr_values, device=device, dtype=torch.long),
            node_entity_ids=torch.cat(node_entity_parts, dim=0),
            sample_ids=sample_ids,
            questions=questions,
            dataset_scope=dataset_scope,
            heuristic_log_v=heuristic_log_v,
            relation_embeddings=relation_embeddings,
            edge_rel_local=edge_rel_local,
        )
        if validate:
            concatenated.validate()
        return concatenated

    def select_graph(
        self, graph_idx: int, *, validate: bool = True
    ) -> "TrajectoryBatch":
        if graph_idx < 0 or graph_idx >= self.num_graphs:
            raise IndexError(f"graph_idx out of range: {graph_idx}.")
        node_start = int(self.node_ptr[graph_idx].item())
        node_end = int(self.node_ptr[graph_idx + 1].item())
        edge_start = int(self.edge_ptr[graph_idx].item())
        edge_end = int(self.edge_ptr[graph_idx + 1].item())
        edge_index = self.edge_index[:, edge_start:edge_end] - node_start
        edge_rel_global = self.edge_rel_global[edge_start:edge_end]
        edge_embeddings = None
        if self.edge_embeddings is not None:
            edge_embeddings = self.edge_embeddings[edge_start:edge_end]
        relation_embeddings = None
        edge_rel_local = None
        if self.relation_embeddings is not None and self.edge_rel_local is not None:
            relation_edge_rel_local = self.edge_rel_local[edge_start:edge_end]
            _, relation_embeddings, edge_rel_local = compact_relation_table(
                edge_rel_global=edge_rel_global,
                relation_embeddings=self.relation_embeddings,
                edge_rel_local=relation_edge_rel_local,
            )
        num_nodes = node_end - node_start
        anchor_start = int(self.anchor_ptr[graph_idx].item())
        anchor_end = int(self.anchor_ptr[graph_idx + 1].item())
        a_start = int(self.a_ptr[graph_idx].item())
        a_end = int(self.a_ptr[graph_idx + 1].item())
        answer_start = int(self.answer_ptr[graph_idx].item())
        answer_end = int(self.answer_ptr[graph_idx + 1].item())
        heuristic_log_v = None
        if self.heuristic_log_v is not None:
            heuristic_log_v = self.heuristic_log_v[node_start:node_end]
        node_embeddings = None
        if self.node_embeddings is not None:
            node_embeddings = self.node_embeddings[node_start:node_end]
        question_emb = None
        question_ctx = None
        question_ctx_mask = None
        if (
            self.question_emb is not None
            and self.question_ctx is not None
            and self.question_ctx_mask is not None
        ):
            question_emb = self.question_emb[graph_idx : graph_idx + 1]
            question_ctx = self.question_ctx[graph_idx : graph_idx + 1]
            question_ctx_mask = self.question_ctx_mask[graph_idx : graph_idx + 1]
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
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            question_emb=question_emb,
            question_ctx=question_ctx,
            question_ctx_mask=question_ctx_mask,
            anchor_local_indices=self.anchor_local_indices[anchor_start:anchor_end],
            anchor_ptr=torch.tensor(
                [0, anchor_end - anchor_start],
                device=self.anchor_ptr.device,
                dtype=torch.long,
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
            node_entity_ids=self.node_entity_ids[node_start:node_end],
            sample_ids=[self.sample_ids[graph_idx]],
            questions=[self.questions[graph_idx]],
            dataset_scope=self.dataset_scope,
            heuristic_log_v=heuristic_log_v,
            relation_embeddings=relation_embeddings,
            edge_rel_local=edge_rel_local,
        )
        if validate:
            sub_batch.validate()
        return sub_batch

    def select_graphs(
        self, graph_indices: list[int] | tuple[int, ...], *, validate: bool = True
    ) -> "TrajectoryBatch":
        if not graph_indices:
            raise ValueError("graph_indices must be non-empty.")
        selected = [
            self.select_graph(int(graph_idx), validate=False)
            for graph_idx in graph_indices
        ]
        combined = TrajectoryBatch.concatenate(selected, validate=validate)
        return combined


__all__ = ["TrajectoryBatch"]
