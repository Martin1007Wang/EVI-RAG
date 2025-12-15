from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from src.models.components.retriever import RetrieverOutput
from src.utils import (
    extract_sample_ids,
    infer_batch_size,
    log_metric,
    normalize_k_values,
)

logger = logging.getLogger(__name__)


class RetrieverModule(LightningModule):
    def __init__(
        self,
        retriever: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: Any,
        scheduler: Any = None,
        evaluation_cfg: Optional[DictConfig | Dict] = None,
        compile_model: bool = False,
        compile_dynamic: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["retriever"])

        self.model = retriever
        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling retriever with torch.compile (dynamic=%s)...", compile_dynamic)
            self.model = torch.compile(self.model, dynamic=compile_dynamic)

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        eval_cfg = self._to_dict(evaluation_cfg or {})
        self._ranking_k = normalize_k_values(eval_cfg.get("ranking_k", [1, 5, 10]), default=[1, 5, 10])
        self._answer_k = normalize_k_values(eval_cfg.get("answer_recall_k", [5, 10]), default=[5, 10])

        self._streaming_state: Dict[str, Dict[str, Any]] = {}
        self.eval_persist_cfg: Optional[Dict[str, Any]] = None

    @staticmethod
    def _to_dict(cfg: Union[DictConfig, Dict]) -> Dict[str, Any]:
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
        return dict(cfg)

    # ------------------------------------------------------------------ #
    # Core Forward & Prediction
    # ------------------------------------------------------------------ #
    def forward(self, batch: Any) -> RetrieverOutput:
        return self.model(batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        """
        Interface for g_agent / inference.
        Returns a lightweight dict on CPU (mostly) to avoid VRAM retention in downstream graph ops.
        """
        output = self(batch)
        scores = output.scores.detach().cpu()
        logits = output.logits.detach().cpu()
        query_ids = output.query_ids.detach().cpu()
        relation_ids = output.relation_ids.detach().cpu() if output.relation_ids is not None else None

        def _maybe_cpu(attr_name: str) -> Any:
            val = getattr(batch, attr_name, None)
            if val is None:
                return None
            return val.detach().cpu() if hasattr(val, "detach") else val

        q_ptr = getattr(batch, "q_local_indices_ptr", None)
        if q_ptr is None and hasattr(batch, "_slice_dict"):
            q_ptr = batch._slice_dict.get("q_local_indices")
        if torch.is_tensor(q_ptr):
            q_ptr = q_ptr.detach().cpu()

        return {
            "query_ids": query_ids,
            "scores": scores,
            "logits": logits,
            "relation_ids": relation_ids,
            "edge_index": _maybe_cpu("edge_index"),
            "edge_attr": _maybe_cpu("edge_attr"),
            "node_global_ids": _maybe_cpu("node_global_ids"),
            "ptr": _maybe_cpu("ptr"),
            "answer_entity_ids": _maybe_cpu("answer_entity_ids"),
            "answer_entity_ids_ptr": _maybe_cpu("answer_entity_ids_ptr"),
            "sample_id": getattr(batch, "sample_id", None),
            "q_local_indices": _maybe_cpu("q_local_indices"),
            "q_local_indices_ptr": q_ptr,
        }

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        num_graphs = infer_batch_size(batch)
        output = self(batch)
        loss_output = self.loss(
            output,
            batch.labels,
            training_step=self.global_step,
            edge_batch=output.query_ids,
            num_graphs=num_graphs,
        )
        log_metric(self, "train/loss", loss_output.loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=num_graphs)

        for k, v in getattr(loss_output, "components", {}).items():
            log_metric(self, f"train/loss/{k}", v, on_step=False, on_epoch=True, batch_size=num_graphs)
        for k, v in getattr(loss_output, "metrics", {}).items():
            log_metric(self, f"train/metric/{k}", v, on_step=False, on_epoch=True, batch_size=num_graphs)

        return loss_output.loss

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }

    # ------------------------------------------------------------------ #
    # Evaluation Loop (Validation & Test)
    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self) -> None:
        self._reset_streaming_state(split="val")

    def on_test_epoch_start(self) -> None:
        self._reset_streaming_state(split="test")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_eval_step(batch, batch_idx, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_eval_step(batch, batch_idx, split="test")

    def on_validation_epoch_end(self) -> None:
        self._finalize_streaming_state(split="val")

    def on_test_epoch_end(self) -> None:
        self._finalize_streaming_state(split="test")

    def _shared_eval_step(self, batch: Any, batch_idx: int, *, split: str) -> None:
        num_graphs = infer_batch_size(batch)
        output = self(batch)
        loss_out = self.loss(
            output,
            batch.labels,
            training_step=self.global_step,
            edge_batch=output.query_ids,
            num_graphs=num_graphs,
        )

        log_metric(
            self,
            f"{split}/loss",
            loss_out.loss,
            batch_size=num_graphs,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        self._update_streaming_state(split=split, batch=batch, output=output, num_graphs=num_graphs)

    # ------------------------------------------------------------------ #
    # Streaming Metrics (no epoch-sized buffering)
    # ------------------------------------------------------------------ #
    def _reset_streaming_state(self, *, split: str) -> None:
        device = self.device
        ranking_k = [int(k) for k in self._ranking_k]
        max_ranking_k = max(ranking_k) if ranking_k else 1
        answer_k = [int(k) for k in self._answer_k]
        max_answer_k = max(answer_k) if answer_k else 0
        max_top_k = max(max_ranking_k, max_answer_k)

        positions = torch.arange(1, max_ranking_k + 1, device=device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1.0)

        self._streaming_state[split] = {
            "device": device,
            "max_ranking_k": int(max_ranking_k),
            "max_answer_k": int(max_answer_k),
            "max_top_k": int(max_top_k),
            "discounts": discounts,
            "discounts_cumsum": discounts.cumsum(0),
            "ranking_count": torch.zeros((), device=device, dtype=torch.float32),
            "mrr_sum": torch.zeros((), device=device, dtype=torch.float32),
            "precision_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in ranking_k},
            "recall_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in ranking_k},
            "f1_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in ranking_k},
            "ndcg_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in ranking_k},
            "answer_count": torch.zeros((), device=device, dtype=torch.float32),
            "answer_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in answer_k},
            "persist_samples": [],
        }

    @staticmethod
    def _all_reduce_inplace(value: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    def _finalize_streaming_state(self, *, split: str) -> None:
        state = self._streaming_state.get(split)
        if not state:
            return

        self._all_reduce_inplace(state["ranking_count"])
        self._all_reduce_inplace(state["mrr_sum"])
        for metric in ("precision_sum", "recall_sum", "f1_sum", "ndcg_sum"):
            for tensor in state[metric].values():
                self._all_reduce_inplace(tensor)
        self._all_reduce_inplace(state["answer_count"])
        for tensor in state["answer_sum"].values():
            self._all_reduce_inplace(tensor)

        ranking_count = state["ranking_count"]
        denom = ranking_count.clamp_min(1.0)
        effective_ranking_count = int(ranking_count.detach().cpu().item())
        batch_size = max(effective_ranking_count, 1)

        log_metric(
            self,
            f"{split}/ranking/mrr",
            state["mrr_sum"] / denom,
            batch_size=batch_size,
            sync_dist=False,
            on_step=False,
            on_epoch=True,
        )
        for k in sorted(state["recall_sum"].keys()):
            log_metric(
                self,
                f"{split}/ranking/recall@{k}",
                state["recall_sum"][k] / denom,
                batch_size=batch_size,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        for k in sorted(state["precision_sum"].keys()):
            log_metric(
                self,
                f"{split}/ranking/precision@{k}",
                state["precision_sum"][k] / denom,
                batch_size=batch_size,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        for k in sorted(state["f1_sum"].keys()):
            log_metric(
                self,
                f"{split}/ranking/f1@{k}",
                state["f1_sum"][k] / denom,
                batch_size=batch_size,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        for k in sorted(state["ndcg_sum"].keys()):
            log_metric(
                self,
                f"{split}/ranking/ndcg@{k}",
                state["ndcg_sum"][k] / denom,
                batch_size=batch_size,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )

        answer_count = state["answer_count"]
        answer_denom = answer_count.clamp_min(1.0)
        effective_answer_count = int(answer_count.detach().cpu().item())
        answer_batch_size = max(effective_answer_count, 1)
        for k in sorted(state["answer_sum"].keys()):
            log_metric(
                self,
                f"{split}/answer/answer_recall@{k}",
                state["answer_sum"][k] / answer_denom,
                batch_size=answer_batch_size,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )

        self._maybe_persist_retriever_outputs(split, state["persist_samples"])
        state["persist_samples"] = []

    def _update_streaming_state(
        self,
        *,
        split: str,
        batch: Any,
        output: RetrieverOutput,
        num_graphs: int,
    ) -> None:
        state = self._streaming_state.get(split)
        if state is None or state.get("device") != output.scores.device:
            self._reset_streaming_state(split=split)
            state = self._streaming_state[split]

        scores_all = output.scores.detach().view(-1)
        labels_all = batch.labels.detach().view(-1).to(dtype=torch.float32)
        if scores_all.numel() != labels_all.numel():
            raise ValueError(f"scores/labels shape mismatch: {scores_all.shape} vs {labels_all.shape}")

        answer_ids_all = getattr(batch, "answer_entity_ids", None)
        if answer_ids_all is None:
            raise ValueError("Batch missing answer_entity_ids required for metrics.")
        slice_dict = getattr(batch, "_slice_dict", None)
        answer_ptr_raw = getattr(batch, "answer_entity_ids_ptr", None)
        if answer_ptr_raw is None and isinstance(slice_dict, dict):
            answer_ptr_raw = slice_dict.get("answer_entity_ids")
        if answer_ptr_raw is None:
            raise ValueError("Batch missing answer_entity_ids_ptr required for metrics.")
        answer_ptr = torch.as_tensor(answer_ptr_raw, dtype=torch.long).view(-1).tolist()
        if len(answer_ptr) != num_graphs + 1:
            raise ValueError(f"answer_entity_ids_ptr length mismatch: {len(answer_ptr)} vs expected {num_graphs + 1}")

        edge_ptr: Optional[List[int]] = None
        if isinstance(slice_dict, dict) and "edge_index" in slice_dict:
            candidate = torch.as_tensor(slice_dict.get("edge_index"), dtype=torch.long).view(-1)
            if candidate.numel() == num_graphs + 1:
                edge_ptr = candidate.tolist()
        query_ids = output.query_ids.detach().view(-1)
        if edge_ptr is None and query_ids.numel() != scores_all.numel():
            raise ValueError(f"query_ids/scores mismatch: {query_ids.shape} vs {scores_all.shape}")

        persist_cfg = getattr(self, "eval_persist_cfg", None) or {}
        persist_splits = persist_cfg.get("splits") or ["test"]
        persist_enabled = bool(persist_cfg.get("enabled")) and split in persist_splits
        sample_ids = extract_sample_ids(batch) if persist_enabled else []
        questions = self._extract_questions(batch, num_graphs) if persist_enabled else []

        max_ranking_k = int(state["max_ranking_k"])
        max_answer_k = int(state["max_answer_k"])
        max_top_k = int(state["max_top_k"])
        discounts = state["discounts"]
        discounts_cumsum = state["discounts_cumsum"]

        for gid in range(int(num_graphs)):
            if edge_ptr is not None:
                start = int(edge_ptr[gid])
                end = int(edge_ptr[gid + 1])
                if end <= start:
                    continue
                scores = scores_all[start:end]
                labels = labels_all[start:end]
                edge_index_g = batch.edge_index[:, start:end] if (max_answer_k > 0 or persist_enabled) else None
                edge_attr_g = batch.edge_attr[start:end] if persist_enabled else None
            else:
                mask = query_ids == gid
                if not bool(mask.any().item()):
                    continue
                scores = scores_all[mask]
                labels = labels_all[mask]
                edge_index_g = batch.edge_index[:, mask] if (max_answer_k > 0 or persist_enabled) else None
                edge_attr_g = batch.edge_attr[mask] if persist_enabled else None

            num_edges = int(scores.numel())
            if num_edges <= 0:
                continue

            k_top = min(num_edges, max_top_k)
            top_scores, top_idx = torch.topk(scores, k=k_top, largest=True, sorted=True)

            # --- Ranking metrics (only defined when positives exist) ---
            pos_mask = labels > 0.5
            pos_count = int(pos_mask.sum().item())
            if pos_count > 0:
                best_pos = scores[pos_mask].max()
                rank = (scores > best_pos).sum() + 1
                state["mrr_sum"].add_(1.0 / rank.to(dtype=torch.float32))
                state["ranking_count"].add_(1.0)

                k_rank = min(k_top, max_ranking_k)
                ranked_labels = labels[top_idx[:k_rank]].to(dtype=torch.float32)
                hits_prefix = ranked_labels.cumsum(0)
                dcg_prefix = (ranked_labels * discounts[:k_rank]).cumsum(0)
                pos_denom = ranked_labels.new_tensor(float(pos_count))

                for k in state["recall_sum"].keys():
                    k_int = int(k)
                    k_used = min(k_int, k_rank)
                    hits = hits_prefix[k_used - 1] if k_used > 0 else ranked_labels.new_zeros(())
                    recall = hits / pos_denom
                    precision = hits / ranked_labels.new_tensor(float(k_int))
                    denom_f1 = (precision + recall).clamp_min(1e-12)
                    f1 = (2 * precision * recall) / denom_f1

                    state["precision_sum"][k].add_(precision)
                    state["recall_sum"][k].add_(recall)
                    state["f1_sum"][k].add_(f1)

                    ideal_k = min(pos_count, k_used)
                    if ideal_k <= 0:
                        state["ndcg_sum"][k].add_(ranked_labels.new_zeros(()))
                    else:
                        ideal_dcg = discounts_cumsum[ideal_k - 1]
                        state["ndcg_sum"][k].add_(dcg_prefix[k_used - 1] / ideal_dcg)

            # --- Answer recall (depends on top-k endpoints, independent of positives) ---
            if state["answer_sum"]:
                a_start = int(answer_ptr[gid])
                a_end = int(answer_ptr[gid + 1])
                answer_ids = answer_ids_all[a_start:a_end]
                if answer_ids.numel() > 0 and max_answer_k > 0:
                    if edge_index_g is None:
                        raise ValueError("edge_index_g required for answer recall but missing.")
                    k_ans = min(k_top, max_answer_k)
                    top_edges = top_idx[:k_ans]
                    edge_index_top = edge_index_g[:, top_edges]
                    head_ids = batch.node_global_ids[edge_index_top[0]]
                    tail_ids = batch.node_global_ids[edge_index_top[1]]

                    answer_unique = torch.unique(answer_ids)
                    if answer_unique.numel() > 0:
                        answer_unique, _ = torch.sort(answer_unique)
                        num_answers = int(answer_unique.numel())
                        ranks = torch.arange(1, k_ans + 1, device=scores.device, dtype=torch.long)
                        first_rank = torch.full((num_answers,), k_ans + 1, dtype=torch.long, device=scores.device)

                        def _update_first_rank(entity_ids: torch.Tensor) -> None:
                            idx = torch.searchsorted(answer_unique, entity_ids)
                            valid = idx < num_answers
                            if not bool(valid.any().item()):
                                return
                            idx_valid = idx[valid]
                            ent_valid = entity_ids[valid]
                            match = answer_unique[idx_valid] == ent_valid
                            if not bool(match.any().item()):
                                return
                            first_rank.scatter_reduce_(
                                0,
                                idx_valid[match],
                                ranks[valid][match],
                                reduce="amin",
                                include_self=True,
                            )

                        _update_first_rank(head_ids)
                        _update_first_rank(tail_ids)

                        state["answer_count"].add_(1.0)
                        for k in state["answer_sum"].keys():
                            k_used = min(int(k), k_ans)
                            state["answer_sum"][k].add_((first_rank <= k_used).to(dtype=torch.float32).mean())

            # --- Optional persistence (store only top max_ranking_k edges) ---
            if persist_enabled:
                k_persist = min(k_top, max_ranking_k)
                if k_persist <= 0:
                    continue
                top_edges = top_idx[:k_persist]
                if edge_index_g is None or edge_attr_g is None:
                    raise ValueError("edge_index_g/edge_attr_g required for persistence but missing.")
                edge_index_top = edge_index_g[:, top_edges]
                head_ids = batch.node_global_ids[edge_index_top[0]].detach().cpu()
                tail_ids = batch.node_global_ids[edge_index_top[1]].detach().cpu()
                relation_ids = edge_attr_g[top_edges].detach().cpu()

                sample_id = sample_ids[gid] if gid < len(sample_ids) else str(gid)
                question = questions[gid] if gid < len(questions) else ""
                a_start = int(answer_ptr[gid])
                a_end = int(answer_ptr[gid + 1])
                answer_ids = answer_ids_all[a_start:a_end].detach().cpu()

                state["persist_samples"].append(
                    {
                        "scores": top_scores[:k_persist].detach().cpu(),
                        "labels": labels[top_edges].detach().cpu(),
                        "head_ids": head_ids,
                        "tail_ids": tail_ids,
                        "relation_ids": relation_ids,
                        "sample_id": sample_id,
                        "question": question,
                        "answer_ids": answer_ids,
                    }
                )

    @staticmethod
    def _extract_questions(batch: Any, num_graphs: int) -> List[str]:
        raw = getattr(batch, "question", None)
        if raw is None:
            return ["" for _ in range(num_graphs)]
        if isinstance(raw, (list, tuple)):
            return [str(q) for q in raw]
        if torch.is_tensor(raw):
            if raw.numel() == num_graphs:
                return [str(v.item()) for v in raw]
            return [str(raw.cpu().tolist()) for _ in range(num_graphs)]
        return [str(raw) for _ in range(num_graphs)]

    # ------------------------------------------------------------------ #
    # Optional persistence for downstream evaluation
    # ------------------------------------------------------------------ #
    def _maybe_persist_retriever_outputs(self, split: str, samples: List[Dict[str, Any]]) -> None:
        cfg = getattr(self, "eval_persist_cfg", None) or {}
        if not cfg.get("enabled"):
            return
        splits = cfg.get("splits") or ["test"]
        if split not in splits or not samples:
            return

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        merged_samples = samples
        if world_size > 1:
            gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, samples)
            if rank != 0:
                return
            merged_samples = []
            for part in gathered:
                if part:
                    merged_samples.extend(part)

        output_dir = Path(cfg.get("output_dir", "eval_retriever"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split}_retriever_eval.pt"
        entity_map, relation_map = self._resolve_vocab_maps(cfg)

        ranking_k = [int(k) for k in self._ranking_k]
        records: List[Dict[str, Any]] = []
        for sample in merged_samples:
            scores = sample["scores"]
            labels = sample["labels"]
            head_ids = sample.get("head_ids")
            tail_ids = sample.get("tail_ids")
            relation_ids = sample.get("relation_ids")
            sample_id = sample.get("sample_id", "")
            question = sample.get("question", "")
            answer_ids = sample.get("answer_ids")

            triplets_by_k: Dict[int, List[Dict[str, Any]]] = {}
            for k in ranking_k:
                top_k = min(int(k), int(scores.numel()))
                edges: List[Dict[str, Any]] = []
                for rank_idx in range(top_k):
                    head_val = int(head_ids[rank_idx].item()) if head_ids is not None else None
                    rel_val = int(relation_ids[rank_idx].item()) if relation_ids is not None else None
                    tail_val = int(tail_ids[rank_idx].item()) if tail_ids is not None else None
                    edges.append(
                        {
                            "head_entity_id": head_val,
                            "relation_id": rel_val,
                            "tail_entity_id": tail_val,
                            "head_text": entity_map.get(head_val) if entity_map is not None else None,
                            "relation_text": relation_map.get(rel_val) if relation_map is not None else None,
                            "tail_text": entity_map.get(tail_val) if entity_map is not None else None,
                            "score": float(scores[rank_idx].item()),
                            "label": float(labels[rank_idx].item()),
                            "rank": int(rank_idx + 1),
                        }
                    )
                triplets_by_k[int(k)] = edges

            record: Dict[str, Any] = {
                "sample_id": str(sample_id),
                "question": str(question),
                "triplets_by_k": triplets_by_k,
            }
            if answer_ids is not None:
                record["answer_entity_ids"] = [int(x) for x in answer_ids.tolist()]
            records.append(record)

        payload = {
            "settings": {
                "split": split,
                "ranking_k": ranking_k,
                "answer_k": [int(k) for k in self._answer_k],
            },
            "samples": records,
        }
        torch.save(payload, output_path)
        logger.info("Persisted retriever eval outputs to %s (samples=%d)", output_path, len(records))

    @staticmethod
    def _resolve_vocab_maps(cfg: Dict[str, Any]) -> tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
        if not cfg.get("textualize"):
            return None, None
        entity_path = cfg.get("entity_vocab_path")
        relation_path = cfg.get("relation_vocab_path")
        if not entity_path or not relation_path:
            ds = cfg.get("dataset_cfg") or {}
            out_dir = ds.get("out_dir")
            if out_dir:
                base = Path(out_dir)
                if not entity_path:
                    entity_path = base / "entity_vocab.parquet"
                if not relation_path:
                    relation_path = base / "relation_vocab.parquet"

        ent_map: Optional[Dict[int, str]] = None
        rel_map: Optional[Dict[int, str]] = None
        try:
            import pyarrow.parquet as pq

            if entity_path and Path(entity_path).exists():
                ent_table = pq.read_table(entity_path, columns=["entity_id", "label"])
                ent_ids = ent_table.column("entity_id").to_pylist()
                ent_labels = ent_table.column("label").to_pylist()
                ent_map = {int(i): str(l) for i, l in zip(ent_ids, ent_labels) if i is not None and l is not None}
            if relation_path and Path(relation_path).exists():
                rel_table = pq.read_table(relation_path, columns=["relation_id", "label"])
                rel_ids = rel_table.column("relation_id").to_pylist()
                rel_labels = rel_table.column("label").to_pylist()
                rel_map = {int(i): str(l) for i, l in zip(rel_ids, rel_labels) if i is not None and l is not None}
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load vocab for textualize: %s", exc)
        return ent_map, rel_map
