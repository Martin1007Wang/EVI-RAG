from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import lightning as L
import torch
import torch.nn.functional as F

from src.data.edge_retriever_labels import EdgeLabelStore
from src.metrics.common import extract_sample_ids
from src.models.components.edge_retriever import EdgeRetriever

_ZERO = 0
_ONE = 1


def _as_single_graph(batch: Any) -> None:
    num_graphs = int(getattr(batch, "num_graphs", 0) or 0)
    if num_graphs != _ONE:
        raise ValueError(
            "EdgeRetrieverModule currently requires data.batch_size=1 " f"(got num_graphs={num_graphs})."
        )


def _resolve_question_emb(batch: Any) -> torch.Tensor:
    q = getattr(batch, "question_emb", None)
    if q is None:
        raise AttributeError("Batch missing question_emb.")
    return torch.as_tensor(q)


def _resolve_node_emb(batch: Any) -> torch.Tensor:
    x = getattr(batch, "node_embeddings", None)
    if x is None:
        raise AttributeError(
            "Batch missing node_embeddings; set data.embeddings_device and ensure transfer_batch_to_device attaches them."
        )
    return torch.as_tensor(x)


def _resolve_edge_emb(batch: Any) -> torch.Tensor:
    e = getattr(batch, "edge_embeddings", None)
    if e is None:
        raise AttributeError(
            "Batch missing edge_embeddings; set data.embeddings_device and ensure transfer_batch_to_device attaches them."
        )
    return torch.as_tensor(e)


def _resolve_node_embedding_ids(batch: Any) -> torch.Tensor:
    ids = getattr(batch, "node_embedding_ids", None)
    if ids is None:
        raise AttributeError("Batch missing node_embedding_ids.")
    return torch.as_tensor(ids, dtype=torch.long)


def _resolve_q_local_indices(batch: Any) -> torch.Tensor:
    q_local = getattr(batch, "q_local_indices", None)
    if q_local is None:
        raise AttributeError("Batch missing q_local_indices.")
    return torch.as_tensor(q_local, dtype=torch.long)


def _resolve_answer_local_indices(batch: Any) -> torch.Tensor:
    a_local = getattr(batch, "a_local_indices", None)
    if a_local is None:
        raise AttributeError("Batch missing a_local_indices.")
    return torch.as_tensor(a_local, dtype=torch.long)


def _resolve_answer_entity_ids(batch: Any) -> torch.Tensor:
    answer_ids = getattr(batch, "answer_entity_ids", None)
    if answer_ids is None:
        raise AttributeError("Batch missing answer_entity_ids.")
    return torch.as_tensor(answer_ids, dtype=torch.long)


def _topk_edge_ids(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = int(k)
    if k <= _ZERO or scores.numel() == _ZERO:
        return scores.new_empty((_ZERO,), dtype=torch.long)
    k_eff = min(k, int(scores.numel()))
    return torch.topk(scores, k_eff, largest=True, sorted=True).indices.to(dtype=torch.long)


class EdgeRetrieverModule(L.LightningModule):
    """Supervised edge-iid triple scorer baseline.

    This module implements the SubgraphRAG-style supervised retriever baseline:
    score each edge independently, trained with strict shortest-path labels.
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        topic_pe: bool,
        dde_num_rounds: int,
        dde_num_reverse_rounds: int,
        lr: float,
        weight_decay: float,
        eval_k_list: Sequence[int],
        export_top_k: int,
        skip_no_path: bool = True,
        label_store_dir: Optional[str] = None,
        label_store_template: str = "{split}.pt",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.retriever = EdgeRetriever(
            emb_dim=int(emb_dim),
            topic_pe=bool(topic_pe),
            dde_num_rounds=int(dde_num_rounds),
            dde_num_reverse_rounds=int(dde_num_reverse_rounds),
        )
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        self._eval_k_list = [int(k) for k in list(eval_k_list)]
        self._export_top_k = int(export_top_k)
        self._skip_no_path = bool(skip_no_path)
        self._label_store_dir = None if label_store_dir in (None, "") else str(label_store_dir)
        self._label_store_template = str(label_store_template)
        self._label_store: dict[str, EdgeLabelStore] = {}

    # ----------------------- #
    # Label access (lazy)
    # ----------------------- #
    def _get_label_store(self, split: str) -> EdgeLabelStore:
        split = str(split)
        store = self._label_store.get(split)
        if store is not None:
            return store
        if self._label_store_dir is None:
            raise RuntimeError("label_store_dir is not configured; required for supervised training/validation.")
        path = Path(self._label_store_dir) / self._label_store_template.format(split=split)
        store = EdgeLabelStore(path)
        self._label_store[split] = store
        return store

    # ----------------------- #
    # Forward + loss
    # ----------------------- #
    def _edge_logits(self, batch: Any) -> torch.Tensor:
        _as_single_graph(batch)
        edge_index = torch.as_tensor(getattr(batch, "edge_index"), dtype=torch.long)
        return self.retriever(
            edge_index=edge_index,
            edge_rel_emb=_resolve_edge_emb(batch).to(device=self.device),
            node_emb=_resolve_node_emb(batch).to(device=self.device),
            node_embedding_ids=_resolve_node_embedding_ids(batch).to(device=self.device),
            question_emb=_resolve_question_emb(batch).to(device=self.device),
            q_local_indices=_resolve_q_local_indices(batch).to(device=self.device),
        )

    def _target_edge_mask(self, batch: Any, *, split: str) -> torch.Tensor:
        _as_single_graph(batch)
        sample_id = extract_sample_ids(batch)[0]
        store = self._get_label_store(split)
        entry = store.get(sample_id)
        edge_index = torch.as_tensor(getattr(batch, "edge_index"), dtype=torch.long, device="cpu")
        num_edges = int(edge_index.size(1))
        if entry.num_edges and entry.num_edges != num_edges:
            raise ValueError(
                f"Label num_edges mismatch for sample_id={sample_id!r}: label={entry.num_edges} batch={num_edges}"
            )
        target = torch.zeros((num_edges,), dtype=torch.float32, device=self.device)
        if entry.positive_edge_ids.numel() > _ZERO:
            idx = entry.positive_edge_ids.to(device=self.device, dtype=torch.long)
            idx = idx[(idx >= _ZERO) & (idx < num_edges)]
            if idx.numel() > _ZERO:
                target.index_fill_(0, idx, 1.0)
        return target

    # ----------------------- #
    # Lightning hooks
    # ----------------------- #
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        sample_id = extract_sample_ids(batch)[0]
        store = self._get_label_store("train")
        entry = store.get(sample_id)
        if self._skip_no_path and entry.max_path_length in (None, 0):
            # Match SubgraphRAG's `skip_no_path=true` behavior.
            loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
            self.log("train/skip_no_path", 1.0, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True)
            return loss

        logits = self._edge_logits(batch)
        if logits.numel() == _ZERO:
            loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
            self.log("train/loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True)
            return loss
        target = self._target_edge_mask(batch, split="train")
        loss = F.binary_cross_entropy_with_logits(logits, target)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        _ = batch_idx
        logits = self._edge_logits(batch)
        if logits.numel() == _ZERO:
            return
        scores = torch.sigmoid(logits)
        target = self._target_edge_mask(batch, split="validation")
        pos = (target > 0.5).nonzero().view(-1)
        if pos.numel() == _ZERO:
            return
        ranks = torch.argsort(scores, descending=True)
        for k in self._eval_k_list:
            k_eff = min(int(k), int(ranks.numel()))
            topk = ranks[:k_eff]
            recall = float(torch.isin(pos, topk).sum().item()) / float(max(int(pos.numel()), 1))
            self.log(
                f"val/triple_recall@{int(k)}",
                recall,
                prog_bar=(int(k) == max(self._eval_k_list)),
                on_epoch=True,
            )

    # ----------------------- #
    # Predict (export rollouts)
    # ----------------------- #
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        _ = batch_idx
        _ = dataloader_idx
        _as_single_graph(batch)
        sample_id = extract_sample_ids(batch)[0]
        edge_index = torch.as_tensor(getattr(batch, "edge_index"), dtype=torch.long)
        edge_attr = torch.as_tensor(getattr(batch, "edge_attr"), dtype=torch.long)
        node_global_ids = torch.as_tensor(getattr(batch, "node_global_ids"), dtype=torch.long)
        q_local = _resolve_q_local_indices(batch).view(-1).to(dtype=torch.long)
        a_local = _resolve_answer_local_indices(batch).view(-1).to(dtype=torch.long)

        start_entity_ids = (
            node_global_ids.index_select(0, q_local.clamp(min=_ZERO)).detach().cpu().unique().tolist()
            if q_local.numel() > _ZERO
            else []
        )
        answer_entity_ids = _resolve_answer_entity_ids(batch).detach().cpu().unique().tolist()
        a_entity_in_graph = a_local.numel() > _ZERO

        logits = self._edge_logits(batch)
        scores = torch.sigmoid(logits).detach().cpu()
        topk_ids = _topk_edge_ids(scores, self._export_top_k).tolist()
        rollouts: list[Dict[str, Any]] = []
        answer_set = set(int(x) for x in answer_entity_ids)
        for rank, e_id in enumerate(topk_ids):
            u = int(edge_index[_ZERO, e_id].detach().cpu().tolist())
            v = int(edge_index[_ONE, e_id].detach().cpu().tolist())
            h_ent = int(node_global_ids[u].detach().cpu().tolist())
            t_ent = int(node_global_ids[v].detach().cpu().tolist())
            rel = int(edge_attr[e_id].detach().cpu().tolist())
            success = t_ent in answer_set
            rollouts.append(
                {
                    "rollout_index": int(rank),
                    "score": float(scores[e_id].item()),
                    "edges": [
                        {
                            "src_entity_id": h_ent,
                            "dst_entity_id": t_ent,
                            "head_entity_id": h_ent,
                            "tail_entity_id": t_ent,
                            "relation_id": rel,
                        }
                    ],
                    "stop_node_entity_id": t_ent,
                    "reach_success": bool(success),
                }
            )

        return [
            {
                "sample_id": sample_id,
                "start_entity_ids": start_entity_ids,
                "answer_entity_ids": answer_entity_ids,
                "a_entity_in_graph": bool(a_entity_in_graph),
                "rollouts": rollouts,
            }
        ]


__all__ = ["EdgeRetrieverModule"]
