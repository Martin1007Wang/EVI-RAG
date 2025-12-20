from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import hydra
import pandas as pd
from omegaconf import ListConfig
import torch
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig
from torch import nn
from torch_scatter import scatter_max

from src.models.components import (
    GFlowNetActor,
    GFlowNetEstimator,
    GraphEmbedder,
    GraphEnv,
    RewardOutput,
    TrajectoryStateEncoder,
)
from src.models.components import GTPathAlignedReward
from src.utils import setup_optimizer
from src.utils.logging_utils import log_metric
from src.utils.pylogger import RankedLogger
from src.data.components import SharedDataResources

logger = logging.getLogger(__name__)
debug_logger = RankedLogger("gflownet.debug", rank_zero_only=True)


class GFlowNetModule(LightningModule):
    """扁平 PyG batch 版本的 GFlowNet，移除 dense padding。"""

    def __init__(
        self,
        *,
        hidden_dim: int,
        policy_cfg: Any,
        reward_cfg: Any,
        env_cfg: Any,
        actor_cfg: Any,
        embedder_cfg: Any,
        state_encoder_cfg: Any,
        estimator_cfg: Any,
        training_cfg: Any,
        evaluation_cfg: Optional[Dict[str, Any]] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        logging_cfg: Optional[Dict[str, Any]] = None,
        eval_persist_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.reward_cfg = reward_cfg
        self.actor_cfg = actor_cfg
        self.state_encoder_cfg = state_encoder_cfg
        self.estimator_cfg = estimator_cfg
        self.training_cfg = training_cfg
        self.evaluation_cfg = evaluation_cfg or {}
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.logging_cfg = logging_cfg or {}
        eval_rollouts_cfg = self._cfg_get(self.evaluation_cfg, "num_eval_rollouts", 1)
        if isinstance(eval_rollouts_cfg, (list, tuple, ListConfig)):
            self._eval_rollout_prefixes = sorted({int(max(1, v)) for v in eval_rollouts_cfg})
            self._eval_rollouts = max(self._eval_rollout_prefixes)
        else:
            self._eval_rollouts = int(max(1, self._cfg_get_int(self.evaluation_cfg, "num_eval_rollouts", 1)))
            self._eval_rollout_prefixes = [self._eval_rollouts]
        self._path_hit_k = self._parse_int_list(self._cfg_get(self.evaluation_cfg, "path_hit_k", [5]))
        self._debug_batches_to_log = max(int(self._cfg_get(self.training_cfg, "debug_batches_to_log", 0)), 0)
        self._debug_graphs_to_log = max(int(self._cfg_get(self.training_cfg, "debug_graphs_to_log", 1)), 1)
        self._debug_batches_logged = 0
        self._subtb_shapes_logged = False
        self._train_prog_bar = set(self._cfg_get(self.logging_cfg, "train_prog_bar", []))
        self._eval_prog_bar = set(self._cfg_get(self.logging_cfg, "eval_prog_bar", []))
        self._auto_success_k = bool(self._cfg_get(self.logging_cfg, "auto_add_success_at_k", True))
        self._auto_path_hit_f1 = bool(self._cfg_get(self.logging_cfg, "auto_add_path_hit_f1", True))
        self._log_on_step_train = bool(self._cfg_get(self.logging_cfg, "log_on_step_train", False))
        # Eval 持久化配置（由 Hydra 注入），仅在 eval 阶段使用
        self.eval_persist_cfg: Dict[str, Any] = dict(eval_persist_cfg or {})
        self._eval_rollout_storage: Dict[str, list] = {"val": [], "test": [], "predict": []}
        self._stream_persist = bool(self.eval_persist_cfg.get("stream", False))
        self._stream_processor: Optional["_EvalPersistProcessor"] = None
        self._stream_output_dir: Optional[Path] = None
        self._active_eval_split: Optional[str] = None
        self._predict_metric_sums: Dict[str, torch.Tensor] = {}
        self._predict_metric_counts: Dict[str, torch.Tensor] = {}
        self._predict_metrics: Dict[str, torch.Tensor] = {}
        self._last_debug_epoch: int = -1

        self.policy: nn.Module = hydra.utils.instantiate(policy_cfg)
        self.reward_fn: nn.Module = hydra.utils.instantiate(reward_cfg)
        self.env: GraphEnv = hydra.utils.instantiate(env_cfg)
        self.max_steps = int(self.env.max_steps)
        self.embedder = hydra.utils.instantiate(embedder_cfg)
        self.state_encoder: TrajectoryStateEncoder = hydra.utils.instantiate(state_encoder_cfg)
        self.estimator = hydra.utils.instantiate(estimator_cfg)
        self.actor = hydra.utils.instantiate(
            actor_cfg,
            policy=self.policy,
            env=self.env,
            max_steps=self.max_steps,
        )
        # 仅保存可序列化的标量，避免将 Hydra 配置对象写入 checkpoint。
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "policy_cfg",
                "reward_cfg",
                "env_cfg",
                "actor_cfg",
                "embedder_cfg",
                "state_encoder_cfg",
                "estimator_cfg",
                "training_cfg",
                "evaluation_cfg",
                "optimizer_cfg",
                "scheduler_cfg",
                "logging_cfg",
                "eval_persist_cfg",
            ],
        )

    def configure_optimizers(self):
        optimizer = setup_optimizer(self, self.optimizer_cfg)
        if not self.scheduler_cfg:
            return optimizer
        scheduler_type = (self.scheduler_cfg.get("type") or "").lower()
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", 10)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self._cfg_get(self.scheduler_cfg, "interval", "epoch"),
                    "monitor": self._cfg_get(self.scheduler_cfg, "monitor", "val/loss"),
                },
            }
        return optimizer

    @staticmethod
    def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, DictConfig):
            return cfg.get(key, default)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    @staticmethod
    def _cfg_get_int(cfg: Any, key: str, default: int) -> int:
        value = GFlowNetModule._cfg_get(cfg, key, default)
        if value is None:
            return int(default)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            try:
                return int(value[0]) if len(value) > 0 else int(default)
            except Exception:
                return int(default)
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _parse_int_list(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [int(value)]
        if isinstance(value, (list, tuple, ListConfig)):
            result: list[int] = []
            for v in value:
                iv = int(v)
                if iv <= 0:
                    continue
                result.append(iv)
            return result or [1]
        try:
            iv = int(value)
            return [iv] if iv > 0 else [1]
        except Exception:
            return [1]

    def setup(self, stage: str | None = None) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None or getattr(trainer, "datamodule", None) is None:
            raise ValueError("Shared resources must be provided via datamodule.shared_resources; trainer/datamodule missing.")
        resources = getattr(trainer.datamodule, "shared_resources", None)
        if resources is None:
            raise ValueError("Shared resources not found on datamodule; ensure GAgentDataModule constructs SharedDataResources.")
        self.embedder.setup(resources, device=self.device)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        # Keep the PyG Batch on CPU to avoid unnecessary GPU residency/copies for large integer ID tensors.
        # The model explicitly moves only the tensors it needs to `self.device` inside embedder/actor/env.
        return batch

    def training_step(self, batch, batch_idx: int):
        # 强制开启梯度，防止上游误用了 no_grad/inference_mode 使图被剥离。
        with torch.autograd.enable_grad():
            loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        batch_size = int(batch.num_graphs)
        log_metric(self, "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self._log_metrics(metrics, prefix="train", batch_size=batch_size)
        self._last_debug = {
            "log_reward": metrics.get("log_reward"),
            "answer_hit": metrics.get("answer_hit"),
            "path_prefix_ratio": metrics.get("path_prefix_ratio"),
            "path_full_hit": metrics.get("path_full_hit"),
            "gt_path_exists_ratio": metrics.get("gt_path_exists_ratio"),
        }
        return loss

    def validation_step(self, batch, batch_idx: int):
        self._active_eval_split = "val"
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        self._log_metrics(metrics, prefix="val", batch_size=int(batch.num_graphs))
        self._active_eval_split = None

    def test_step(self, batch, batch_idx: int):
        self._active_eval_split = "test"
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        self._log_metrics(metrics, prefix="test", batch_size=int(batch.num_graphs))
        self._active_eval_split = None

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self._active_eval_split = "predict"
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        self._log_metrics(metrics, prefix="predict", batch_size=int(batch.num_graphs))
        self._accumulate_predict_metrics(metrics, batch_size=int(batch.num_graphs))
        self._active_eval_split = None
        return None

    def on_validation_epoch_end(self):
        self._persist_eval_rollouts("val")

    def on_test_epoch_end(self):
        self._persist_eval_rollouts("test")

    def on_predict_epoch_end(self):
        self._persist_eval_rollouts("predict")
        metrics = self._finalize_predict_metrics()
        self._predict_metrics = metrics
        if metrics:
            for name, value in metrics.items():
                log_metric(self, f"predict/{name}", value, batch_size=1, on_step=False, on_epoch=True)

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str, batch_size: int) -> None:
        sync_dist = bool(self.trainer and getattr(self.trainer, "num_devices", 1) > 1)
        is_train = prefix == "train"
        max_eval_prefix = max(self._eval_rollout_prefixes) if hasattr(self, "_eval_rollout_prefixes") else self._eval_rollouts
        prog_bar_set = set(self._train_prog_bar if is_train else self._eval_prog_bar)
        if not is_train and self._auto_success_k and max_eval_prefix > 1:
            prog_bar_set.add(f"success@{max_eval_prefix}")
        if not is_train and self._auto_path_hit_f1 and self._path_hit_k:
            prog_bar_set.add(f"path_hit_f1@{max(self._path_hit_k)}")
        for name, value in metrics.items():
            if not torch.is_floating_point(value):
                value = value.float()
            scalar = value.mean()
            log_on_step = self._log_on_step_train if is_train else False
            prog_bar = name in prog_bar_set
            log_metric(
                self,
                f"{prefix}/{name}",
                scalar,
                sync_dist=sync_dist,
                prog_bar=prog_bar,
                on_step=log_on_step,
                on_epoch=True,
                batch_size=batch_size,
            )

    def _accumulate_predict_metrics(self, metrics: Dict[str, torch.Tensor], *, batch_size: int) -> None:
        bs = torch.tensor(float(batch_size), device=self.device)
        for name, value in metrics.items():
            if not torch.is_floating_point(value):
                value = value.float()
            scalar = value.mean()
            self._predict_metric_sums[name] = self._predict_metric_sums.get(name, torch.zeros((), device=self.device)) + scalar * bs
            self._predict_metric_counts[name] = self._predict_metric_counts.get(name, torch.zeros((), device=self.device)) + bs

    def _finalize_predict_metrics(self) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}
        if dist.is_available() and dist.is_initialized():
            for name in list(self._predict_metric_sums.keys()):
                dist.all_reduce(self._predict_metric_sums[name], op=dist.ReduceOp.SUM)
                dist.all_reduce(self._predict_metric_counts[name], op=dist.ReduceOp.SUM)
        for name, total in self._predict_metric_sums.items():
            denom = self._predict_metric_counts.get(name, torch.tensor(1.0, device=total.device)).clamp(min=1.0)
            metrics[name] = (total / denom).detach().cpu()
        self._predict_metric_sums.clear()
        self._predict_metric_counts.clear()
        return metrics

    def _compute_batch_loss(self, batch: Any, batch_idx: int | None = None) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self.device
        should_debug = self._should_log_debug(batch_idx)
        if not self.training:
            self._refresh_eval_settings()
        split = self._active_eval_split
        collect_rollouts = (not self.training) and self._should_persist(split)

        embed = self.embedder.embed_batch(batch, device=device)
        if should_debug:
            self._log_batch_debug(batch=batch, embed=embed, batch_idx=batch_idx)
            self._log_gt_path_feasibility_debug(batch=batch, embed=embed, device=device, batch_idx=batch_idx)
            self._debug_batches_logged += 1
        edge_tokens = embed.edge_tokens
        edge_batch = embed.edge_batch
        edge_ptr = embed.edge_ptr
        node_ptr = embed.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        edge_scores = embed.edge_scores
        edge_index = embed.edge_index
        edge_relations = embed.edge_relations
        edge_labels = embed.edge_labels
        path_mask = embed.path_mask
        path_exists = embed.path_exists
        heads_global = embed.heads_global
        tails_global = embed.tails_global
        node_tokens = embed.node_tokens
        question_tokens = embed.question_tokens
        answer_ptr = batch._slice_dict["answer_entity_ids"].to(device)
        answer_node_ptr = batch._slice_dict["answer_node_locals"].to(device)
        gt_path_ptr = batch._slice_dict["gt_path_edge_local_ids"].to(device)
        # SubTB：logF(s_t) 在每一步状态上预测；不再使用单独的 start_summary/logZ 常数项。

        graph_cache: Dict[str, torch.Tensor] = {
            "edge_index": edge_index,
            "edge_batch": edge_batch,
            "node_global_ids": batch.node_global_ids.to(device),
            "heads_global": heads_global,
            "tails_global": tails_global,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "start_node_locals": batch.start_node_locals.to(device),
            "start_ptr": batch._slice_dict["start_node_locals"].to(device),
            "start_entity_ids": batch.start_entity_ids.to(device),
            "start_entity_ptr": batch._slice_dict["start_entity_ids"].to(device),
            "answer_node_locals": batch.answer_node_locals.to(device),
            "answer_ptr": batch._slice_dict["answer_entity_ids"].to(device),
            "answer_entity_ids": batch.answer_entity_ids.to(device),
            "edge_relations": batch.edge_attr.to(device),
            "edge_labels": edge_labels,
            "is_answer_reachable": batch.is_answer_reachable.to(device),
        }

        gt_counts = gt_path_ptr[1:] - gt_path_ptr[:-1]

        step0_weight = float(self._cfg_get(self.training_cfg, "step0_supervision_weight", 0.0))
        step0_loss: Optional[torch.Tensor] = None
        step0_acc: Optional[torch.Tensor] = None
        if self.training and step0_weight > 0.0:
            step0_loss, step0_acc = self._compute_step0_supervision(
                edge_tokens=edge_tokens,
                question_tokens=question_tokens,
                edge_batch=edge_batch,
                edge_ptr=edge_ptr,
                edge_scores=edge_scores,
                edge_index=edge_index,
                node_ptr=node_ptr,
                heads_global=heads_global,
                tails_global=tails_global,
                start_node_locals=batch.start_node_locals.to(device),
                gt_path_edge_local_ids=batch.gt_path_edge_local_ids.to(device),
                gt_path_ptr=gt_path_ptr,
                training=self.training,
            )

        gt_stats: Dict[str, torch.Tensor] = {}
        if path_mask is not None and path_mask.numel() == edge_batch.numel():
            gt_edge_counts = torch.bincount(edge_batch, weights=path_mask.float(), minlength=num_graphs)
            valid_gt = path_exists.bool() if path_exists is not None else torch.ones(num_graphs, device=device, dtype=torch.bool)
            if valid_gt.any():
                gt_counts_valid = gt_edge_counts[valid_gt]
                gt_stats["gt_path_edges_mean"] = gt_counts_valid.mean()
                gt_stats["gt_path_edges_p90"] = torch.quantile(gt_counts_valid, 0.9)
                gt_stats["gt_path_edges_max"] = gt_counts_valid.max()
                if self._path_hit_k:
                    for k_val in self._path_hit_k:
                        k_int = int(k_val)
                        gt_stats[f"gt_path_edges_gt@{k_int}"] = (gt_counts_valid > k_int).float().mean()
            else:
                zero = torch.tensor(0.0, device=device)
                gt_stats["gt_path_edges_mean"] = zero
                gt_stats["gt_path_edges_p90"] = zero
                gt_stats["gt_path_edges_max"] = zero
                if self._path_hit_k:
                    for k_val in self._path_hit_k:
                        k_int = int(k_val)
                        gt_stats[f"gt_path_edges_gt@{k_int}"] = zero

        if self.training:
            num_rollouts = 1
        else:
            try:
                num_rollouts = int(self._eval_rollouts)
            except Exception:
                num_rollouts = 1
            if num_rollouts <= 0:
                num_rollouts = 1
                debug_logger.warning(
                    "[EVAL_ROLLOUTS_FIX] _eval_rollouts<=0 detected; clamped to 1 (batch_idx=%s).",
                    str(batch_idx),
                )
        loss_list = []
        metrics_list: list[Dict[str, torch.Tensor]] = []
        path_hits: list[torch.Tensor] = []
        rollout_logs: list[Dict[str, torch.Tensor]] = []
        rollout_selected_masks: list[torch.Tensor] = []
        rollout_prefix_ratios: list[torch.Tensor] = []
        rollout_full_hits: list[torch.Tensor] = []
        rollout_answer_hits: list[torch.Tensor] = []
        rollout_lengths: list[torch.Tensor] = []

        for _ in range(num_rollouts):
            rollout = self.actor.rollout(
                batch=batch,
                edge_tokens=edge_tokens,
                question_tokens=question_tokens,
                edge_batch=edge_batch,
                edge_ptr=edge_ptr,
                node_ptr=node_ptr,
                edge_scores=edge_scores,
                training=self.training,
                batch_idx=batch_idx,
                graph_cache=graph_cache,
            )
            # Residual PF: rollout 已返回 clean log_pf/log_pf_steps/actions_seq 等用于 SubTB。
            state_emb_seq = self.state_encoder(
                actions_seq=rollout["actions_seq"],
                edge_tokens=edge_tokens,
                stop_indices=edge_ptr[1:],
                question_tokens=question_tokens,
                node_tokens=node_tokens,
                start_node_locals=batch.start_node_locals.to(device),
                start_ptr=batch._slice_dict["start_node_locals"].to(device),
            )

            reward_out: RewardOutput = self.reward_fn(
                actions_seq=rollout["actions_seq"],
                selected_mask=rollout["selected_mask"],
                selection_order=rollout["selection_order"],
                length=rollout["length"],
                edge_labels=edge_labels,
                edge_batch=edge_batch,
                edge_heads=heads_global,
                edge_tails=tails_global,
                edge_scores=edge_scores,
                answer_entity_ids=batch.answer_entity_ids.to(device),
                answer_ptr=batch._slice_dict["answer_entity_ids"].to(device),
                path_mask=path_mask,
                path_exists=path_exists,
                reach_success=rollout["reach_success"],
                reach_fraction=rollout["reach_fraction"],
                edge_index=edge_index,
                node_ptr=node_ptr,
                edge_ptr=edge_ptr,
                gt_path_edge_local_ids=batch.gt_path_edge_local_ids.to(device),
                gt_path_ptr=gt_path_ptr,
                max_steps=torch.tensor(self.max_steps, device=device),
                answer_node_locals=batch.answer_node_locals.to(device),
                answer_node_ptr=answer_node_ptr,
                is_answer_reachable=batch.is_answer_reachable.view(-1).to(device),
            )

            log_reward = reward_out.log_reward
            self._assert_finite(log_reward, "log_reward")
            self._assert_finite(rollout["log_pf"], "log_pf")
            log_flow_states = self._compute_log_flow_states(
                state_emb_seq=state_emb_seq,
                question_tokens=question_tokens,
                log_reward=log_reward,
                edge_lengths=rollout["length"].long(),
            )
            # Backward policy P_B is deterministic under this state definition:
            # given the full trajectory state (selected edges + order), the predecessor is uniquely
            # obtained by removing the last selected edge. Hence log P_B = 0 for every realized step.
            log_pb_steps = self._compute_deterministic_log_pb_steps(
                actions_seq=rollout["actions_seq"],
                stop_indices=edge_ptr[1:],
                dtype=rollout["log_pf_steps"].dtype,
                device=device,
            )
            subtb_loss_raw, subtb_per_graph = self._compute_subtb_loss(
                log_flow_states=log_flow_states,
                log_pf_steps=rollout["log_pf_steps"],
                log_pb_steps=log_pb_steps,
                edge_lengths=rollout["length"].long(),
                return_per_graph=True,
            )
            subtb_loss = subtb_loss_raw
            mask_subtb_by_path_exists = bool(self._cfg_get(self.training_cfg, "mask_subtb_by_path_exists", False))
            if mask_subtb_by_path_exists and isinstance(self.reward_fn, GTPathAlignedReward):
                subtb_loss = self._mask_subtb_loss_by_path_exists(
                    subtb_per_graph=subtb_per_graph,
                    path_exists=path_exists,
                )

            loss = subtb_loss
            if step0_loss is not None:
                loss = loss + step0_weight * step0_loss

            if should_debug:
                self._log_reward_prior_debug(
                    rollout=rollout,
                    reward_out=reward_out,
                    edge_scores=edge_scores,
                    edge_batch=edge_batch,
                    log_reward=log_reward,
                    batch_idx=batch_idx,
                )
                self._log_rollout_sanity_debug(
                    rollout=rollout,
                    reward_out=reward_out,
                    log_reward=log_reward,
                    log_flow_states=log_flow_states,
                    edge_batch=edge_batch,
                    edge_ptr=edge_ptr,
                    path_mask=path_mask,
                    gt_path_edge_local_ids=batch.gt_path_edge_local_ids.to(device),
                    gt_path_ptr=gt_path_ptr,
                    batch_idx=batch_idx,
                )

            reward_metrics = reward_out.as_dict()
            log_reward_metric = reward_metrics.pop("log_reward")
            answer_hit = reward_metrics.pop("answer_hit", None)
            success = reward_metrics.pop("success", None)
            path_prefix_len = reward_metrics.pop("path_prefix_len", None)
            path_prefix_ratio = reward_metrics.pop("path_prefix_ratio", None)
            path_full_hit = reward_metrics.pop("path_full_hit", None)
            answer_reach_frac = reward_metrics.pop("answer_reach_frac", None)
            reward_metrics.pop("reward", None)
            reward_metrics.pop("pos_precision", None)
            reward_metrics.pop("pos_recall", None)
            reward_metrics.pop("pos_f1", None)
            reward_metrics.pop("answer_precision", None)
            reward_metrics.pop("answer_recall", None)
            reward_metrics.pop("answer_f1", None)
            reward_metrics.pop("path_exists", None)

            valid_graphs = (
                path_exists.bool()
                if path_exists is not None
                else torch.ones(num_graphs, device=device, dtype=torch.bool)
            )

            # Top-K 边回溯命中率（按选择顺序前 K 条边）；仅在存在 GT 边的图上聚合。
            if not self.training and self._path_hit_k:
                sel_order = rollout["selection_order"]
                for k_val in self._path_hit_k:
                    k_int = int(k_val)
                    topk_mask = (sel_order >= 0) & (sel_order < k_int)
                    if path_mask is not None and path_mask.numel() == topk_mask.numel():
                        edge_valid = path_mask.bool() & topk_mask & valid_graphs[edge_batch]
                    else:
                        edge_valid = torch.zeros_like(topk_mask, dtype=torch.bool)
                    hits_topk = torch.bincount(edge_batch, weights=edge_valid.float(), minlength=num_graphs)
                    if path_mask is not None and path_mask.numel() == topk_mask.numel():
                        gt_edge_mask = path_mask.bool() & valid_graphs[edge_batch]
                        gt_counts = torch.bincount(edge_batch, weights=gt_edge_mask.float(), minlength=num_graphs)
                    else:
                        gt_counts = torch.zeros(num_graphs, device=device, dtype=torch.float32)
                    selected_counts = torch.bincount(
                        edge_batch,
                        weights=(rollout["selected_mask"].bool() & valid_graphs[edge_batch]).float(),
                        minlength=num_graphs,
                    )
                    denom = torch.minimum(torch.full_like(selected_counts, float(k_int)), selected_counts)
                    has_gt = valid_graphs & (gt_counts > 0)
                    valid_mask = has_gt & (denom > 0)
                    precision_topk = torch.zeros(num_graphs, device=device, dtype=torch.float32)
                    recall_topk = torch.zeros_like(precision_topk)
                    f1_topk = torch.zeros_like(precision_topk)
                    precision_topk[valid_mask] = hits_topk[valid_mask] / denom[valid_mask].clamp(min=1.0)
                    recall_topk[has_gt] = hits_topk[has_gt] / gt_counts[has_gt].clamp(min=1.0)
                with torch.no_grad():
                    denom_f1 = precision_topk + recall_topk
                    mask_f1 = denom_f1 > 0
                    f1_topk[mask_f1] = (
                        2 * precision_topk[mask_f1] * recall_topk[mask_f1] / denom_f1[mask_f1]
                    )

                    # 仅在存在 GT 的图上取均值，避免把“无 GT”混入 0 噪声
                    def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                        if not mask.any():
                            return torch.tensor(0.0, device=device)
                        return (t * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

                    reward_metrics[f"path_hit_precision@{k_int}"] = _masked_mean(precision_topk, valid_mask)
                    reward_metrics[f"path_hit_recall@{k_int}"] = _masked_mean(recall_topk, has_gt)
                reward_metrics[f"path_hit_f1@{k_int}"] = _masked_mean(f1_topk, has_gt)
            # 确保所有配置的 K 都有键（即便 path_mask 为空）
            if not self.training and self._path_hit_k:
                zero = torch.tensor(0.0, device=device)
                for k_val in self._path_hit_k:
                    k_int = int(k_val)
                    reward_metrics.setdefault(f"path_hit_precision@{k_int}", zero)
                    reward_metrics.setdefault(f"path_hit_recall@{k_int}", zero)
                    reward_metrics.setdefault(f"path_hit_f1@{k_int}", zero)

            metrics: Dict[str, torch.Tensor] = {
                "log_reward": log_reward_metric,
                "path_prefix_len": (
                    path_prefix_len.detach()
                    if isinstance(path_prefix_len, torch.Tensor)
                    else torch.tensor(0.0, device=device)
                ),
                "path_prefix_ratio": (
                    path_prefix_ratio.detach()
                    if isinstance(path_prefix_ratio, torch.Tensor)
                    else torch.tensor(0.0, device=device)
                ),
                "path_full_hit": (
                    path_full_hit.detach()
                    if isinstance(path_full_hit, torch.Tensor)
                    else torch.tensor(0.0, device=device)
                ),
                "answer_hit": (
                    answer_hit.detach()
                    if isinstance(answer_hit, torch.Tensor)
                    else (success.detach() if isinstance(success, torch.Tensor) else rollout["reach_success"].detach())
                ),
                "length_mean": rollout["length"].detach(),
                "subtb_loss": subtb_loss.detach(),
                "subtb_loss_unmasked": subtb_loss_raw.detach(),
                "subtb_loss_gt_exists": (
                    subtb_per_graph[path_exists.bool()].mean().detach()
                    if path_exists is not None and bool(path_exists.any().item())
                    else torch.tensor(0.0, device=device)
                ),
                "subtb_loss_gt_missing": (
                    subtb_per_graph[(~path_exists.bool())].mean().detach()
                    if path_exists is not None and bool((~path_exists.bool()).any().item())
                    else torch.tensor(0.0, device=device)
                ),
                "gt_path_exists_ratio": path_exists.float().mean().detach()
                if path_exists is not None
                else torch.tensor(0.0, device=device),
            }
            if isinstance(answer_reach_frac, torch.Tensor) and not isinstance(self.reward_fn, GTPathAlignedReward):
                metrics["answer_reach_frac"] = answer_reach_frac.detach()
            if "policy_shift_rate" in rollout:
                metrics["policy_shift_rate"] = rollout["policy_shift_rate"].detach()
            if "policy_residual_l1" in rollout:
                metrics["policy_residual_l1"] = rollout["policy_residual_l1"].detach()
            metrics.update(reward_metrics)
            if gt_stats:
                metrics.update(gt_stats)
            if step0_loss is not None:
                metrics["step0_loss"] = step0_loss.detach()
                if step0_acc is not None:
                    metrics["step0_acc@1"] = step0_acc.detach()

            if not self.training:
                zeros_f = torch.zeros(num_graphs, device=device, dtype=log_reward_metric.dtype)
                prefix_val = (
                    path_prefix_ratio.detach().to(dtype=zeros_f.dtype)
                    if isinstance(path_prefix_ratio, torch.Tensor)
                    else zeros_f
                )
                full_val = (
                    path_full_hit.detach().to(dtype=torch.bool)
                    if isinstance(path_full_hit, torch.Tensor)
                    else torch.zeros(num_graphs, device=device, dtype=torch.bool)
                )
                if isinstance(answer_hit, torch.Tensor):
                    answer_val = answer_hit.detach().to(dtype=torch.bool)
                elif isinstance(success, torch.Tensor):
                    answer_val = success.detach().to(dtype=torch.bool)
                else:
                    answer_val = rollout["reach_success"].detach().to(dtype=torch.bool)
                rollout_prefix_ratios.append(prefix_val)
                rollout_full_hits.append(full_val)
                rollout_answer_hits.append(answer_val)
                rollout_lengths.append(rollout["length"].detach().to(dtype=zeros_f.dtype))
                rollout_selected_masks.append(rollout["selected_mask"].detach())

            loss_list.append(loss)
            metrics_list.append(metrics)
            if collect_rollouts:
                rollout_logs.append(
                    {
                        "selected_mask": rollout["selected_mask"].detach().cpu(),
                        "selection_order": rollout["selection_order"].detach().cpu(),
                        "log_pf": rollout["log_pf"].detach().cpu(),
                        "log_reward": log_reward.detach().cpu(),
                        "reach_success": rollout["reach_success"].detach().cpu(),
                    }
                )

            if not self.training:
                if path_mask is not None and path_mask.numel() == rollout["selected_mask"].numel():
                    hit_edge = rollout["selected_mask"].bool() & path_mask.bool() & valid_graphs[edge_batch]
                else:
                    hit_edge = torch.zeros_like(rollout["selected_mask"], dtype=torch.bool)
                per_graph_hit = torch.zeros(num_graphs, device=device, dtype=torch.bool)
                if hit_edge.any():
                    per_graph_hit = torch.bincount(edge_batch, weights=hit_edge.float(), minlength=num_graphs) > 0
                per_graph_hit = per_graph_hit & valid_graphs
                path_hits.append(per_graph_hit)

        if not loss_list:
            # Defensive: avoid empty TensorList when eval rollouts unexpectedly skipped.
            loss = torch.zeros((), device=device, dtype=torch.float32)
            metrics: Dict[str, torch.Tensor] = {}
            if not self.training:
                debug_logger.warning(
                    "[EVAL_EMPTY] No eval rollouts recorded (batch_idx=%s, num_rollouts=%s).",
                    str(batch_idx),
                    str(self._eval_rollouts),
                )
        elif num_rollouts > 1:
            loss = torch.stack(loss_list, dim=0).mean()
            metrics = self._aggregate_metrics(metrics_list, best_of=(not self.training))
        else:
            loss = loss_list[0]
            metrics = metrics_list[0]

        if self.training and not loss.requires_grad:

            def _flag(t: torch.Tensor, name: str) -> str:
                return f"{name}: req={t.requires_grad} grad_fn={t.grad_fn is not None} shape={tuple(t.shape)}"

            debug_logger.error(
                "[LOSS_STATUS] grad_enabled=%s loss_is_leaf=%s len_loss_list=%d loss_list_req=%s | %s | %s | %s",
                torch.is_grad_enabled(),
                loss.is_leaf,
                len(loss_list),
                [t.requires_grad for t in loss_list],
                _flag(loss, "loss"),
                _flag(subtb_loss, "subtb_loss"),
                _flag(rollout["log_pf"], "log_pf"),
            )
            raise RuntimeError("Training loss has no grad; check LOSS_STATUS log for detached SubTB tensors.")

        if not self.training:
            if self._eval_rollout_prefixes:
                k_values = [int(k) for k in self._eval_rollout_prefixes]
                valid_graphs = (
                    path_exists.bool() if path_exists is not None else torch.ones(num_graphs, device=device, dtype=torch.bool)
                )
                path_stack = (
                    torch.stack(path_hits, dim=0) if path_hits else torch.zeros(1, num_graphs, device=device, dtype=torch.bool)
                )
                rollouts_avail = path_stack.size(0)
                answer_counts = (answer_ptr[1:] - answer_ptr[:-1]).to(device)
                valid_answers = (answer_counts > 0) & valid_graphs
                for k_int in k_values:
                    k_clamped = min(max(k_int, 1), rollouts_avail)
                    hit_any = path_stack[:k_clamped].any(dim=0)
                    if valid_graphs.any():
                        denom = valid_graphs.float().sum().clamp(min=1.0)
                        hit_k = (hit_any & valid_graphs).float().sum() / denom
                    else:
                        hit_k = torch.tensor(0.0, device=device)
                    metrics[f"path_hit_any@{k_int}"] = hit_k
            metrics.update(
                self._compute_rollout_eval_metrics(
                    path_exists=path_exists,
                    path_mask=path_mask,
                    edge_batch=edge_batch,
                    edge_index=edge_index,
                    node_ptr=node_ptr,
                    gt_counts=gt_counts.to(device=device),
                    answer_node_locals=batch.answer_node_locals.to(device),
                    answer_node_ptr=answer_node_ptr,
                    is_answer_reachable=batch.is_answer_reachable.to(device),
                    rollout_selected_masks=rollout_selected_masks,
                    rollout_prefix_ratios=rollout_prefix_ratios,
                    rollout_full_hits=rollout_full_hits,
                    rollout_answer_hits=rollout_answer_hits,
                    rollout_lengths=rollout_lengths,
                )
            )
        if collect_rollouts and rollout_logs:
            self._buffer_rollout_records(
                split=split or "test",
                batch=batch,
                rollout_logs=rollout_logs,
                heads_global=heads_global.detach().cpu(),
                tails_global=tails_global.detach().cpu(),
                edge_relations=edge_relations.detach().cpu(),
                edge_scores=edge_scores.detach().cpu(),
                edge_labels=edge_labels.detach().cpu(),
                edge_index=edge_index.detach().cpu(),
                edge_ptr=edge_ptr.detach().cpu(),
                node_ptr=node_ptr.detach().cpu(),
                start_node_ptr=batch._slice_dict["start_node_locals"].detach().cpu(),
                start_node_locals=batch.start_node_locals.detach().cpu(),
            )
        return loss, metrics

    @staticmethod
    def _sort_edges_by_graph(batch: Any) -> Any:
        edge_index = batch.edge_index
        edge_batch = batch.batch[edge_index[0]]
        perm = torch.argsort(edge_batch)
        if perm.numel() == 0:
            return batch
        # 重排所有按边对齐的字段，确保同图边连续，edge_ptr 合法。
        batch.edge_index = edge_index[:, perm]
        if hasattr(batch, "edge_attr"):
            batch.edge_attr = batch.edge_attr[perm]
        if hasattr(batch, "edge_labels"):
            batch.edge_labels = batch.edge_labels[perm]
        if hasattr(batch, "top_edge_mask"):
            batch.top_edge_mask = batch.top_edge_mask[perm]
        if hasattr(batch, "edge_relations"):
            batch.edge_relations = batch.edge_relations[perm]

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
        if hasattr(batch, "gt_path_edge_local_ids") and batch.gt_path_edge_local_ids.numel() > 0:
            batch.gt_path_edge_local_ids = inv_perm[batch.gt_path_edge_local_ids]
        return batch

    def _should_log_debug(self, batch_idx: int | None) -> bool:
        if self._debug_batches_to_log <= 0:
            return False
        if not debug_logger.isEnabledFor(logging.INFO):
            return False
        # 重置计数：每个 epoch 的首个 batch 触发一次。
        epoch = int(getattr(self, "current_epoch", -1))
        if epoch != self._last_debug_epoch:
            self._debug_batches_logged = 0
            self._last_debug_epoch = epoch
        if batch_idx not in (0, None):
            return False
        return self._debug_batches_logged < self._debug_batches_to_log

    def _compute_step0_supervision(
        self,
        *,
        edge_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_index: torch.Tensor,
        node_ptr: torch.Tensor,
        heads_global: torch.Tensor,
        tails_global: torch.Tensor,
        start_node_locals: torch.Tensor,
        gt_path_edge_local_ids: torch.Tensor,
        gt_path_ptr: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = edge_tokens.device
        num_graphs = int(edge_ptr.numel() - 1)
        if num_graphs <= 0:
            raise ValueError("edge_ptr implies no graphs for step0 supervision.")
        if gt_path_ptr.numel() != num_graphs + 1:
            raise ValueError("gt_path_ptr length mismatch with edge_ptr.")
        if edge_index.size(1) != edge_batch.numel():
            raise ValueError("edge_index must align with edge_batch for step0 supervision.")

        gt_counts = gt_path_ptr[1:] - gt_path_ptr[:-1]
        has_gt = gt_counts > 0
        if not bool(has_gt.any().item()):
            return torch.zeros((), device=device), None

        gt_edges = gt_path_edge_local_ids.to(device=device, dtype=torch.long)
        first_idx = gt_path_ptr[:-1][has_gt]
        gt_first = torch.full((num_graphs,), -1, device=device, dtype=torch.long)
        gt_first[has_gt] = gt_edges[first_idx]
        gt_first_sel = gt_first[has_gt]
        if (gt_first_sel < 0).any() or (gt_first_sel >= edge_batch.numel()).any():
            raise ValueError("GT step0 edge indices out of batch edge range.")

        num_nodes_total = int(node_ptr[-1].item())
        if num_nodes_total <= 0:
            raise ValueError("node_ptr implies empty batch for step0 supervision.")
        start_nodes = start_node_locals.to(device=device, dtype=torch.long)
        if start_nodes.numel() == 0:
            raise ValueError("start_node_locals empty; cannot compute step0 supervision.")
        if (start_nodes < 0).any() or int(start_nodes.max().item()) >= num_nodes_total:
            raise ValueError("start_node_locals out of range for step0 supervision.")
        node_is_start = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        node_is_start[start_nodes] = True
        edge_starts_mask = node_is_start[edge_index[0]] | node_is_start[edge_index[1]]
        if edge_starts_mask.numel() != edge_batch.numel():
            raise ValueError("edge_starts_mask shape mismatch for step0 supervision.")

        valid_counts = torch.bincount(edge_batch[edge_starts_mask], minlength=num_graphs)
        if bool((valid_counts[has_gt] <= 0).any().item()):
            raise ValueError("Graphs with GT path have no valid step0 edges.")
        invalid_gt = ~edge_starts_mask[gt_first_sel]
        if bool(invalid_gt.any().item()):
            raise ValueError("GT step0 edge is not in the step0 action mask.")

        selected_mask = torch.zeros_like(edge_batch, dtype=torch.bool, device=device)
        selection_order = torch.full_like(edge_batch, -1, dtype=torch.long, device=device)
        edge_residual, stop_residual, _ = self.policy(
            edge_tokens,
            question_tokens,
            edge_batch,
            selected_mask,
            selection_order=selection_order,
            edge_heads=heads_global,
            edge_tails=tails_global,
            frontier_mask=edge_starts_mask,
        )
        log_edge, _ = self.actor.compute_step0_log_probs(
            edge_scores=edge_scores,
            edge_residual=edge_residual,
            stop_residual=stop_residual,
            edge_batch=edge_batch,
            valid_edges=edge_starts_mask,
            training=training,
        )
        logp = log_edge[gt_first[has_gt]]
        loss = -logp.mean()

        masked_log_edge = log_edge.masked_fill(~edge_starts_mask, float("-inf"))
        _, edge_argmax = scatter_max(masked_log_edge, edge_batch, dim=0, dim_size=num_graphs)
        acc = (edge_argmax[has_gt] == gt_first[has_gt]).float().mean()
        return loss, acc

    def _log_reward_prior_debug(
        self,
        *,
        rollout: Dict[str, torch.Tensor],
        reward_out: RewardOutput,
        edge_scores: torch.Tensor,
        edge_batch: torch.Tensor,
        log_reward: torch.Tensor,
        batch_idx: int | None,
        top_pct: float = 0.1,
    ) -> None:
        """粗粒度打印 reward vs retriever score，用于 overfit / 逸出先验的对拍。"""
        try:
            sel = rollout["selected_mask"].detach().cpu()
            scores = edge_scores.detach().cpu()
            e_batch = edge_batch.detach().cpu()
            reward = reward_out.reward.detach().cpu()
            log_r = log_reward.detach().cpu()
            success = reward_out.success.detach().cpu()
        except Exception as exc:  # pragma: no cover - debug only
            debug_logger.error("Failed to collect reward/prior debug tensors (batch_idx=%s): %s", str(batch_idx), exc)
            return

        num_graphs = int(reward.numel())
        graphs_to_log = min(num_graphs, self._debug_graphs_to_log)
        for g in range(graphs_to_log):
            mask_g = e_batch == g
            sel_g = sel[mask_g]
            scores_g = scores[mask_g]
            if scores_g.numel() == 0:
                continue
            sel_any = sel_g.any().item()
            mean_all = float(scores_g.mean().item())
            mean_sel = float(scores_g[sel_g].mean().item()) if sel_any else 0.0
            max_sel = float(scores_g[sel_g].max().item()) if sel_any else 0.0
            try:
                # torch.quantile 不可用时回退 numpy
                q_thr = float(torch.quantile(scores_g, 1.0 - top_pct).item())
            except Exception:
                import numpy as np  # lazy import

                q_thr = float(np.quantile(scores_g.numpy(), 1.0 - top_pct))
            sel_frac_top = (
                float(((sel_g) & (scores_g >= q_thr)).float().sum().item()) / float(max(sel_g.float().sum().item(), 1.0))
                if sel_any
                else 0.0
            )

    def _log_rollout_sanity_debug(
        self,
        *,
        rollout: Dict[str, torch.Tensor],
        reward_out: RewardOutput,
        log_reward: torch.Tensor,
        log_flow_states: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
        path_mask: Optional[torch.Tensor],
        gt_path_edge_local_ids: Optional[torch.Tensor],
        gt_path_ptr: Optional[torch.Tensor],
        batch_idx: int | None,
    ) -> None:
        """打印 SubTB 关键量的确定性一致性检查（可直接粘贴日志定位问题）。"""
        try:
            log_pf_steps = rollout["log_pf_steps"].detach()
            log_pf = rollout["log_pf"].detach()
            length = rollout["length"].detach().long()
            success = rollout["reach_success"].detach().float()
        except Exception as exc:  # pragma: no cover - debug only
            debug_logger.error("Failed to collect rollout sanity tensors (batch_idx=%s): %s", str(batch_idx), exc)
            return

        with torch.no_grad():
            try:
                debug_logger.info(
                    "[ROLL_STATS] batch_idx=%s log_reward_mean=%.3f log_pf_mean=%.3f len_mean=%.3f success=%.3f",
                    str(batch_idx),
                    float(log_reward.mean().item()),
                    float(log_pf.mean().item()),
                    float(length.float().mean().item()),
                    float(success.mean().item()),
                )
                if "policy_shift_rate" in rollout and "policy_residual_l1" in rollout:
                    debug_logger.info(
                        "[POLICY_STATS] batch_idx=%s shift_rate=%.6g residual_l1=%.6g",
                        str(batch_idx),
                        float(rollout["policy_shift_rate"].detach().item()),
                        float(rollout["policy_residual_l1"].detach().item()),
                    )
                stop_idx = edge_ptr[1:].view(-1, 1)  # [B,1] for broadcast with actions[B,T]
                stop_counts = (rollout["actions"] == stop_idx).sum(dim=1).float()
                debug_logger.info(
                    "[ROLL_DISTR] batch_idx=%s len(min/med/max)=%.1f/%.1f/%.1f stop_cnt(min/med/max)=%.1f/%.1f/%.1f",
                    str(batch_idx),
                    float(length.min().item()),
                    float(torch.quantile(length.float(), 0.5).item()),
                    float(length.max().item()),
                    float(stop_counts.min().item()),
                    float(torch.quantile(stop_counts, 0.5).item()),
                    float(stop_counts.max().item()),
                )
            except Exception:
                pass
            # 1) PF 是否按 step 累加（这能直接抓住“累计 log_pf 错了”的结构性 bug）
            log_pf_sum = log_pf_steps.sum(dim=1)
            max_abs_diff = (log_pf_sum - log_pf).abs().max()
            debug_logger.info(
                "[PF_SUM_CHECK] batch_idx=%s max_abs_diff=%.6g",
                str(batch_idx),
                float(max_abs_diff.item()),
            )

            # 2) 终端 TB 残差：logF(s0) + Σ logPF - logR（若符号/terminal_state/stop 对齐错，会直接爆炸）
            num_graphs, num_actions = log_pf_steps.shape
            terminal_state = length.clamp(min=0, max=max(num_actions - 1, 0)) + 1  # state index in [1, T_action]
            log_pf_prefix = torch.zeros(num_graphs, num_actions + 1, device=log_pf_steps.device, dtype=log_pf_steps.dtype)
            log_pf_prefix[:, 1:] = log_pf_steps.cumsum(dim=1)
            term_log_pf = log_pf_prefix.gather(1, terminal_state.view(-1, 1)).view(-1)
            logF_s0 = log_flow_states[:, 0].to(dtype=term_log_pf.dtype)
            tb_residual = logF_s0 + term_log_pf - log_reward.to(dtype=term_log_pf.dtype)
            debug_logger.info(
                "[TB_TERMINAL_RESID] batch_idx=%s mean=%.6g std=%.6g min=%.6g max=%.6g",
                str(batch_idx),
                float(tb_residual.mean().item()),
                float(tb_residual.std(unbiased=False).item()),
                float(tb_residual.min().item()),
                float(tb_residual.max().item()),
            )

            # 3) Reward 定义与 rollout 前缀匹配的自洽性检查（仅路径奖励）
            if isinstance(self.reward_fn, GTPathAlignedReward):
                reward_prefix = reward_out.path_prefix_ratio.detach()
                if (
                    gt_path_edge_local_ids is not None
                    and gt_path_ptr is not None
                    and reward_prefix.numel() == num_graphs
                    and gt_path_ptr.numel() == num_graphs + 1
                ):
                    gt_ptr = gt_path_ptr.to(device=log_pf_steps.device)
                    gt_edges = gt_path_edge_local_ids.to(device=log_pf_steps.device, dtype=torch.long)
                    gt_counts = gt_ptr[1:] - gt_ptr[:-1]
                    max_gt = int(gt_counts.max().item()) if gt_counts.numel() > 0 else 0
                    if max_gt > 0:
                        gt_path = torch.full((num_graphs, max_gt), -1, dtype=torch.long, device=log_pf_steps.device)
                        if gt_edges.numel() > 0:
                            gt_batch = torch.repeat_interleave(torch.arange(num_graphs, device=log_pf_steps.device), gt_counts)
                            gt_pos = torch.arange(gt_edges.numel(), device=log_pf_steps.device) - gt_ptr[gt_batch]
                            edge_start = edge_ptr[:-1].to(device=log_pf_steps.device)
                            edge_end = edge_ptr[1:].to(device=log_pf_steps.device)
                            in_range = (gt_edges >= edge_start[gt_batch]) & (gt_edges < edge_end[gt_batch])
                            if not bool(in_range.all().item()):
                                bad = torch.nonzero(~in_range, as_tuple=False).view(-1)
                                preview = bad[:5].detach().cpu().tolist()
                                raise ValueError(
                                    "gt_path_edge_local_ids must be batch-global edge indices; "
                                    f"found out-of-range entries at positions={preview}."
                                )
                            local_edges = gt_edges - edge_start[gt_batch]
                            gt_path[gt_batch, gt_pos] = local_edges
                    else:
                        gt_path = torch.empty(num_graphs, 0, dtype=torch.long, device=log_pf_steps.device)

                    actions = rollout["actions_seq"].to(device=log_pf_steps.device, dtype=torch.long)
                    stop_idx = edge_ptr[1:].view(-1, 1).to(device=log_pf_steps.device)
                    is_stop = actions == stop_idx
                    actions_local = actions - edge_ptr[:-1].view(-1, 1).to(device=log_pf_steps.device)
                    actions_local = torch.where(is_stop, torch.full_like(actions_local, -1), actions_local)

                    max_compare = min(actions_local.size(1), gt_path.size(1))
                    if max_compare > 0:
                        actions_cmp = actions_local[:, :max_compare]
                        gt_cmp = gt_path[:, :max_compare]
                        valid = gt_cmp >= 0
                        match = (actions_cmp == gt_cmp) & valid & (actions_cmp >= 0)
                        prefix_mask = match.float().cumprod(dim=1)
                        prefix_len = prefix_mask.sum(dim=1)
                    else:
                        prefix_len = torch.zeros(num_graphs, device=log_pf_steps.device, dtype=torch.float32)

                    prefix_ratio = torch.zeros_like(prefix_len)
                    has_gt = gt_counts > 0
                    if has_gt.any():
                        prefix_ratio[has_gt] = prefix_len[has_gt] / gt_counts[has_gt].to(prefix_len.dtype).clamp(min=1.0)
                    diff = (prefix_ratio - reward_prefix.to(dtype=prefix_ratio.dtype)).abs()
                    debug_logger.info(
                        "[PREFIX_ALIGN] batch_idx=%s max_abs_diff=%.6g",
                        str(batch_idx),
                        float(diff.max().item()),
                    )
                return

            # 3) Reward 内部的 Top-K recall 与 path_mask 计算是否一致（非路径奖励）
            reward_recall = reward_out.answer_reach_frac.detach()
            if (
                path_mask is not None
                and path_mask.numel() == rollout["selection_order"].numel()
                and reward_recall.numel() == num_graphs
            ):
                k = max(self._path_hit_k) if self._path_hit_k else 5
                sel_order = rollout["selection_order"]
                topk_mask = (sel_order >= 0) & (sel_order < int(k))
                hit_topk = path_mask.bool() & topk_mask
                hits = torch.bincount(edge_batch, weights=hit_topk.float(), minlength=num_graphs)
                gt_counts = torch.bincount(edge_batch, weights=path_mask.bool().float(), minlength=num_graphs)
                recall = torch.zeros(num_graphs, device=log_pf_steps.device, dtype=torch.float32)
                has_gt = gt_counts > 0
                recall[has_gt] = hits[has_gt] / gt_counts[has_gt].clamp(min=1.0)
                diff = (recall - reward_recall.to(dtype=recall.dtype)).abs()
                debug_logger.info(
                    "[REWARD_ALIGN] batch_idx=%s k=%d max_abs_diff=%.6g",
                    str(batch_idx),
                    int(k),
                    float(diff.max().item()),
                )

    def _log_gt_path_feasibility_debug(self, *, batch: Any, embed: Any, device: torch.device, batch_idx: int | None) -> None:
        """验证 GT 路径是否在 GraphEnv 约束下可执行；不可执行则训练目标本身不成立。"""
        try:
            node_ptr = embed.node_ptr.to(device)
            edge_ptr = embed.edge_ptr.to(device)
            edge_batch = embed.edge_batch.to(device)
            edge_index = embed.edge_index.to(device)
            heads_global = embed.heads_global.to(device)
            tails_global = embed.tails_global.to(device)
            edge_labels = embed.edge_labels.to(device)

            graph_dict: Dict[str, torch.Tensor] = {
                "edge_index": edge_index,
                "edge_batch": edge_batch,
                "node_global_ids": batch.node_global_ids.to(device),
                "heads_global": heads_global,
                "tails_global": tails_global,
                "node_ptr": node_ptr,
                "edge_ptr": edge_ptr,
                "start_node_locals": batch.start_node_locals.to(device),
                "start_ptr": batch._slice_dict["start_node_locals"].to(device),
                "start_entity_ids": batch.start_entity_ids.to(device),
                "start_entity_ptr": batch._slice_dict["start_entity_ids"].to(device),
                "answer_node_locals": batch.answer_node_locals.to(device),
                "answer_ptr": batch._slice_dict["answer_entity_ids"].to(device),
                "answer_entity_ids": batch.answer_entity_ids.to(device),
                "edge_relations": batch.edge_attr.to(device),
                "edge_labels": edge_labels,
                "is_answer_reachable": batch.is_answer_reachable.to(device),
            }

            gt_edges_all = batch.gt_path_edge_local_ids.to(device)
            gt_ptr = batch._slice_dict["gt_path_edge_local_ids"].to(device)
        except Exception as exc:  # pragma: no cover - debug only
            debug_logger.error("Failed to collect GT feasibility tensors (batch_idx=%s): %s", str(batch_idx), exc)
            return

        num_graphs = int(node_ptr.numel() - 1)
        graphs_to_check = min(num_graphs, self._debug_graphs_to_log)
        if graphs_to_check <= 0:
            return
        if gt_ptr.numel() != num_graphs + 1:
            debug_logger.error(
                "[GT_FEASIBILITY] batch_idx=%s gt_ptr has wrong shape: %s (expected B+1=%d)",
                str(batch_idx),
                tuple(gt_ptr.shape),
                num_graphs + 1,
            )
            return

        # NOTE: gt_path_edge_local_ids 已保序；验证 GT 路径在环境约束下是否可执行。
        stop_indices = edge_ptr[1:].to(device=device)

        for g in range(graphs_to_check):
            es, ee = int(edge_ptr[g].item()), int(edge_ptr[g + 1].item())
            ns, ne = int(node_ptr[g].item()), int(node_ptr[g + 1].item())
            if ne <= ns or ee <= es:
                continue

            gt_s, gt_e = int(gt_ptr[g].item()), int(gt_ptr[g + 1].item())
            gt_edges = gt_edges_all[gt_s:gt_e]
            if gt_edges.numel() == 0:
                continue

            # 按 GT 顺序逐步检查可行性（step0/前缀/全路径）
            state = self.env.reset(graph_dict, device=device)
            actions = stop_indices.clone()
            prefix_ok = 0
            for step_idx, edge_id in enumerate(gt_edges.tolist()):
                if edge_id < es or edge_id >= ee:
                    debug_logger.error(
                        "[GT_FEASIBILITY] batch_idx=%s graph=%d edge_id=%d out_of_range=[%d,%d)",
                        str(batch_idx),
                        g,
                        int(edge_id),
                        es,
                        ee,
                    )
                    break
                action_mask = self.env.action_mask_edges(state)
                if not bool(action_mask[edge_id].item()):
                    break
                actions[:] = stop_indices
                actions[g] = int(edge_id)
                state = self.env.step(state, actions, step_index=step_idx)
                prefix_ok += 1

            full_ok = prefix_ok == int(gt_edges.numel())
            step0_ok = prefix_ok >= 1
            debug_logger.info(
                "[GT_FEASIBILITY] batch_idx=%s graph=%d gt_edges=%d step0_ok=%s prefix_len=%d full_ok=%s answer_hit=%s",
                str(batch_idx),
                g,
                int(gt_edges.numel()),
                str(bool(step0_ok)),
                int(prefix_ok),
                str(bool(full_ok)),
                str(bool(state.answer_hits[g].item())),
            )

    def _log_batch_debug(self, *, batch: Any, embed, batch_idx: int | None) -> None:
        try:
            node_ptr = batch.ptr.detach().cpu()
            edge_ptr = embed.edge_ptr.detach().cpu()
            edge_index = batch.edge_index.detach().cpu()
            edge_batch = embed.edge_batch.detach().cpu()
            node_global_ids = batch.node_global_ids.detach().cpu()
            start_node_locals = batch.start_node_locals.detach().cpu()
            answer_node_locals = batch.answer_node_locals.detach().cpu()
            gt_edges = batch.gt_path_edge_local_ids.detach().cpu()
            start_ptr = batch._slice_dict.get("start_node_locals")
            answer_ptr = batch._slice_dict.get("answer_node_locals")
            start_entity_ptr = batch._slice_dict.get("start_entity_ids")
            answer_entity_ptr = batch._slice_dict.get("answer_entity_ids")
            path_ptr = batch._slice_dict.get("gt_path_edge_local_ids")
        except Exception as exc:  # pragma: no cover - defensive
            debug_logger.error("Failed to collect debug tensors (batch_idx=%s): %s", str(batch_idx), exc)
            return

        num_graphs = int(node_ptr.numel() - 1)
        total_edges = int(edge_index.size(1))
        total_nodes = int(node_global_ids.numel())
        monotonic_edge_batch = bool(edge_batch.numel() < 2 or torch.all(edge_batch[:-1] <= edge_batch[1:]))
        debug_logger.info(
            "[BATCH_DEBUG] batch_idx=%s num_graphs=%d nodes=%d edges=%d edge_batch_sorted=%s",
            str(batch_idx),
            num_graphs,
            total_nodes,
            total_edges,
            monotonic_edge_batch,
        )

        graphs_to_log = min(num_graphs, self._debug_graphs_to_log)
        for g in range(graphs_to_log):
            ns, ne = int(node_ptr[g].item()), int(node_ptr[g + 1].item())
            es, ee = int(edge_ptr[g].item()), int(edge_ptr[g + 1].item())
            nodes = node_global_ids[ns:ne]
            start_slice = slice(int(start_ptr[g].item()), int(start_ptr[g + 1].item())) if start_ptr is not None else slice(0, 0)
            answer_slice = (
                slice(int(answer_ptr[g].item()), int(answer_ptr[g + 1].item())) if answer_ptr is not None else slice(0, 0)
            )
            path_slice = slice(int(path_ptr[g].item()), int(path_ptr[g + 1].item())) if path_ptr is not None else slice(0, 0)

            start_nodes = start_node_locals[start_slice]
            answer_nodes = answer_node_locals[answer_slice]
            path_edges = gt_edges[path_slice]

            start_oob = start_nodes.numel() > 0 and ((start_nodes < ns) | (start_nodes >= ne)).any().item()
            answer_oob = answer_nodes.numel() > 0 and ((answer_nodes < ns) | (answer_nodes >= ne)).any().item()
            path_oob = path_edges.numel() > 0 and ((path_edges < es) | (path_edges >= ee)).any().item()

            edge_batch_slice = edge_batch[es:ee]
            edge_batch_mismatch = bool(edge_batch_slice.numel() > 0 and torch.unique(edge_batch_slice).numel() != 1)
            node_dup = max(int(nodes.numel() - torch.unique(nodes).numel()), 0)

            start_entities = (
                int(start_entity_ptr[g + 1].item() - start_entity_ptr[g].item()) if start_entity_ptr is not None else 0
            )
            answer_entities = (
                int(answer_entity_ptr[g + 1].item() - answer_entity_ptr[g].item()) if answer_entity_ptr is not None else 0
            )

            summary: Dict[str, Any] = {
                "graph": g,
                "node_span": [ns, ne],
                "edge_span": [es, ee],
                "node_dup": node_dup,
                "edge_batch_ok": not edge_batch_mismatch,
                "start_locals": (start_nodes - ns).tolist(),
                "start_entities": start_entities,
                "answer_locals": (answer_nodes - ns).tolist(),
                "answer_entities": answer_entities,
                "path_edges_local": (path_edges - es).tolist(),
                "flags": {
                    "start_oob": bool(start_oob),
                    "answer_oob": bool(answer_oob),
                    "path_oob": bool(path_oob),
                },
            }
            debug_logger.info("[GRAPH_DEBUG] %s", summary)

    @staticmethod
    @staticmethod
    def _aggregate_metrics(metrics_list: list[Dict[str, torch.Tensor]], *, best_of: bool = False) -> Dict[str, torch.Tensor]:
        """Aggregate metrics across eval rollouts.

        best_of=True: hit/recall/F1/precision/success 取 max，length 取 min，其余取 mean。
        """
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        aggregated: Dict[str, torch.Tensor] = {}
        for key in keys:
            stack = torch.stack([m[key] for m in metrics_list], dim=0).float()
            if not best_of:
                aggregated[key] = stack.mean(dim=0)
                continue

            k_lower = key.lower()
            if any(token in k_lower for token in ("hit", "recall", "f1", "precision", "success")):
                aggregated[key] = stack.max(dim=0).values
            elif "length" in k_lower:
                aggregated[key] = stack.min(dim=0).values
            else:
                aggregated[key] = stack.mean(dim=0)
        return aggregated

    def _compute_rollout_eval_metrics(
        self,
        *,
        path_exists: Optional[torch.Tensor],
        path_mask: Optional[torch.Tensor],
        edge_batch: torch.Tensor,
        edge_index: torch.Tensor,
        node_ptr: torch.Tensor,
        gt_counts: torch.Tensor,
        answer_node_locals: torch.Tensor,
        answer_node_ptr: torch.Tensor,
        is_answer_reachable: torch.Tensor,
        rollout_selected_masks: list[torch.Tensor],
        rollout_prefix_ratios: list[torch.Tensor],
        rollout_full_hits: list[torch.Tensor],
        rollout_answer_hits: list[torch.Tensor],
        rollout_lengths: list[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not rollout_selected_masks:
            return {}
        device = rollout_selected_masks[0].device
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            return {}

        path_valid = (
            path_exists.bool()
            if isinstance(path_exists, torch.Tensor)
            else (gt_counts.to(device=device) > 0)
        )
        answer_counts = answer_node_ptr[1:] - answer_node_ptr[:-1]
        answer_valid = (
            is_answer_reachable.view(-1).bool()
            if isinstance(is_answer_reachable, torch.Tensor) and is_answer_reachable.numel() == num_graphs
            else (answer_counts > 0)
        )

        def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if not mask.any():
                return torch.tensor(0.0, device=device)
            return (value * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        rollout_count = len(rollout_selected_masks)
        selected_stack = torch.stack(rollout_selected_masks, dim=0).bool()
        prefix_stack = torch.stack(rollout_prefix_ratios, dim=0)
        full_stack = torch.stack(rollout_full_hits, dim=0).bool()
        answer_stack = torch.stack(rollout_answer_hits, dim=0).bool()
        length_stack = torch.stack(rollout_lengths, dim=0)
        gt_len = gt_counts.to(device=device, dtype=length_stack.dtype)

        k_values = [int(k) for k in self._eval_rollout_prefixes] if self._eval_rollout_prefixes else [rollout_count]
        metrics: Dict[str, torch.Tensor] = {}
        for k_int in k_values:
            k_clamped = min(max(k_int, 1), rollout_count)
            prefix_best = prefix_stack[:k_clamped].max(dim=0).values
            metrics[f"path_prefix@{k_int}"] = _masked_mean(prefix_best, path_valid)

            exact_any = full_stack[:k_clamped].any(dim=0)
            metrics[f"path_exact@{k_int}"] = _masked_mean(exact_any.float(), path_valid)

            len_err = (length_stack[:k_clamped] - gt_len).abs()
            min_err = len_err.min(dim=0).values
            metrics[f"path_len_err@{k_int}"] = _masked_mean(min_err, path_valid)

            answer_any = answer_stack[:k_clamped].any(dim=0)
            metrics[f"answer_hit@{k_int}"] = _masked_mean(answer_any.float(), answer_valid)

            union_mask = selected_stack[:k_clamped].any(dim=0)
            if path_mask is not None and path_mask.numel() == edge_batch.numel():
                path_mask_bool = path_mask.bool()
                hit_mask = union_mask & path_mask_bool
                hits = torch.bincount(edge_batch, weights=hit_mask.float(), minlength=num_graphs)
                selected = torch.bincount(edge_batch, weights=union_mask.float(), minlength=num_graphs)
                precision = torch.zeros(num_graphs, device=device, dtype=hits.dtype)
                mask_sel = selected > 0
                precision[mask_sel] = hits[mask_sel] / selected[mask_sel].clamp(min=1.0)
                recall = torch.zeros_like(precision)
                mask_gt = gt_counts > 0
                recall[mask_gt] = hits[mask_gt] / gt_counts[mask_gt].to(dtype=hits.dtype).clamp(min=1.0)
                metrics[f"path_edge_precision@{k_int}"] = _masked_mean(precision, path_valid & mask_sel)
                metrics[f"path_edge_recall@{k_int}"] = _masked_mean(recall, path_valid)

            if answer_node_locals.numel() > 0 and answer_counts.numel() == num_graphs:
                total_nodes = int(node_ptr[-1].item())
                visited = torch.zeros(total_nodes, device=device, dtype=torch.bool)
                if union_mask.any():
                    visited[edge_index[0, union_mask]] = True
                    visited[edge_index[1, union_mask]] = True
                answer_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), answer_counts)
                hit_answers = visited[answer_node_locals.to(device=device, dtype=torch.long)]
                hits = torch.bincount(answer_batch, weights=hit_answers.float(), minlength=num_graphs)
                recall = torch.zeros(num_graphs, device=device, dtype=hits.dtype)
                valid_answers = answer_counts > 0
                recall[valid_answers] = hits[valid_answers] / answer_counts[valid_answers].to(dtype=hits.dtype).clamp(min=1.0)
                metrics[f"answer_recall@{k_int}"] = _masked_mean(recall, answer_valid)

        return metrics

    def _refresh_eval_settings(self) -> None:
        """Ensure eval rollouts and path_hit_k reflect current config (even when loading old checkpoints)."""
        eval_cfg = self.evaluation_cfg or {}
        rollouts_cfg = self._cfg_get(eval_cfg, "num_eval_rollouts", self._eval_rollouts if hasattr(self, "_eval_rollouts") else 1)
        if isinstance(rollouts_cfg, (list, tuple, ListConfig)):
            prefixes = sorted({int(max(1, v)) for v in rollouts_cfg})
            self._eval_rollouts = max(prefixes) if prefixes else 1
            self._eval_rollout_prefixes = prefixes or [self._eval_rollouts]
        else:
            self._eval_rollouts = int(max(1, self._cfg_get_int(eval_cfg, "num_eval_rollouts", 1)))
            self._eval_rollout_prefixes = [self._eval_rollouts]
        self._path_hit_k = self._parse_int_list(self._cfg_get(eval_cfg, "path_hit_k", getattr(self, "_path_hit_k", [5])))

    def _should_persist(self, split: Optional[str]) -> bool:
        cfg = getattr(self, "eval_persist_cfg", None) or {}
        if not cfg.get("enabled"):
            return False
        splits = cfg.get("splits") or ["test"]
        return (split or "test") in splits

    @staticmethod
    def _extract_batch_meta(batch: Any, num_graphs: int) -> tuple[list[str], list[str]]:
        raw_ids = getattr(batch, "sample_id", None)
        if raw_ids is None:
            sample_ids = [str(i) for i in range(num_graphs)]
        elif isinstance(raw_ids, (list, tuple)):
            sample_ids = [str(s) for s in raw_ids]
        elif torch.is_tensor(raw_ids):
            sample_ids = [str(x.item()) for x in raw_ids.view(-1)]
        else:
            sample_ids = [str(raw_ids) for _ in range(num_graphs)]

        raw_q = getattr(batch, "question", None)
        if raw_q is None:
            questions = ["" for _ in range(num_graphs)]
        elif isinstance(raw_q, (list, tuple)):
            questions = [str(q) for q in raw_q]
        elif torch.is_tensor(raw_q):
            if raw_q.numel() == num_graphs:
                questions = [str(v.item()) for v in raw_q.view(-1)]
            else:
                questions = [str(raw_q.cpu().tolist()) for _ in range(num_graphs)]
        else:
            questions = [str(raw_q) for _ in range(num_graphs)]
        return sample_ids, questions

    @staticmethod
    def _value_at(tensor: torch.Tensor, idx: int) -> torch.Tensor:
        flat = tensor.view(-1)
        if flat.numel() == 0:
            return torch.tensor(0.0)
        if idx < flat.numel():
            return flat[idx]
        return flat[-1]

    def _buffer_rollout_records(
        self,
        *,
        split: str,
        batch: Any,
        rollout_logs: list[Dict[str, torch.Tensor]],
        heads_global: torch.Tensor,
        tails_global: torch.Tensor,
        edge_relations: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_labels: torch.Tensor,
        edge_index: torch.Tensor,
        edge_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        start_node_ptr: torch.Tensor,
        start_node_locals: torch.Tensor,
    ) -> None:
        if not self._should_persist(split):
            return

        num_graphs = int(edge_ptr.numel() - 1)
        sample_ids, questions = self._extract_batch_meta(batch, num_graphs)
        storage = self._eval_rollout_storage.setdefault(split, [])
        stream_records: list[Dict[str, Any]] = []

        for g in range(num_graphs):
            es, ee = int(edge_ptr[g].item()), int(edge_ptr[g + 1].item())
            s0, s1 = int(start_node_ptr[g].item()), int(start_node_ptr[g + 1].item())
            start_nodes = start_node_locals[s0:s1]
            start_set = set(start_nodes.tolist())

            edge_range = torch.arange(es, ee)
            rollouts: list[Dict[str, Any]] = []
            for ridx, log in enumerate(rollout_logs):
                sel_mask = log["selected_mask"][es:ee].bool()
                sel_order = log["selection_order"][es:ee]
                selected_global = edge_range[sel_mask]
                order_selected = sel_order[sel_mask]
                if order_selected.numel() > 0:
                    perm = torch.argsort(order_selected)
                    ordered_global = selected_global[perm]
                else:
                    ordered_global = torch.empty(0, dtype=torch.long)

                edges: list[Dict[str, Any]] = []
                current_tail_local: Optional[int] = None
                for step_idx, edge_idx in enumerate(ordered_global.tolist()):
                    h_gid = int(heads_global[edge_idx].item())
                    t_gid = int(tails_global[edge_idx].item())
                    head_local = int(edge_index[0, edge_idx].item())
                    tail_local = int(edge_index[1, edge_idx].item())

                    if step_idx == 0:
                        head_is_start = head_local in start_set
                        tail_is_start = tail_local in start_set
                        if head_is_start or tail_is_start:
                            src_is_head = head_is_start
                        else:
                            src_is_head = True
                    else:
                        if current_tail_local == head_local:
                            src_is_head = True
                        elif current_tail_local == tail_local:
                            src_is_head = False
                        else:
                            src_is_head = True

                    if src_is_head:
                        src_global = h_gid
                        dst_global = t_gid
                        current_tail_local = tail_local
                    else:
                        src_global = t_gid
                        dst_global = h_gid
                        current_tail_local = head_local

                    edges.append(
                        {
                            "head_entity_id": h_gid,
                            "relation_id": int(edge_relations[edge_idx].item()),
                            "tail_entity_id": t_gid,
                            "edge_score": float(edge_scores[edge_idx].item()),
                            "edge_label": float(edge_labels[edge_idx].item()),
                            "edge_local_index": int(edge_idx - es),
                            "src_entity_id": src_global,
                            "dst_entity_id": dst_global,
                        }
                    )

                rollouts.append(
                    {
                        "rollout_index": ridx,
                        "success": bool(self._value_at(log["reach_success"], g).item()),
                        "log_pf": float(self._value_at(log["log_pf"], g).item()),
                        "log_reward": float(self._value_at(log["log_reward"], g).item()),
                        "edges": edges,
                    }
                )

            record = {
                "sample_id": sample_ids[g] if g < len(sample_ids) else str(g),
                "question": questions[g] if g < len(questions) else "",
                "rollouts": rollouts,
            }
            if self._stream_persist:
                stream_records.append(record)
            else:
                storage.append(record)
        if self._stream_persist and stream_records:
            self._stream_rollouts(stream_records, split=split)

    def _persist_eval_rollouts(self, split: str) -> None:
        if not self._should_persist(split):
            return
        if self._stream_persist:
            return  # streaming模式已写盘
        records = self._eval_rollout_storage.get(split, [])
        if not records:
            return

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        merged = records
        if world_size > 1:
            gathered: list[Optional[list]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, records)
            if rank != 0:
                self._eval_rollout_storage[split] = []
                return
            merged = []
            for part in gathered:
                if part:
                    merged.extend(part)

        self._eval_rollout_storage[split] = []
        cfg = getattr(self, "eval_persist_cfg", None) or {}
        output_dir = Path(cfg.get("output_dir", "eval_gflownet"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split}_gflownet_eval.pt"
        k_values = [int(k) for k in getattr(self, "_eval_rollout_prefixes", [self._eval_rollouts])]
        processor = _EvalPersistProcessor(cfg)
        merged = processor.process(merged)

        payload = {
            "settings": {
                "split": split,
                "num_eval_rollouts": k_values,
                "path_hit_k": [int(k) for k in getattr(self, "_path_hit_k", [])],
            },
            "samples": merged,
        }
        torch.save(payload, output_path)
        logger.info("Persisted gflownet eval outputs to %s (samples=%d)", output_path, len(merged))

    def _stream_rollouts(self, records: list[Dict[str, Any]], *, split: str) -> None:
        """Stream rollouts to JSONL to避免堆积内存。支持多进程：gather 到 rank0 写盘。"""
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        merged = records
        if world_size > 1:
            gathered: list[Optional[list]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, records)
            if rank != 0:
                return
            merged = []
            for part in gathered:
                if part:
                    merged.extend(part)

        if self._stream_processor is None:
            cfg = getattr(self, "eval_persist_cfg", None) or {}
            self._stream_processor = _EvalPersistProcessor(cfg)
            output_dir = Path(cfg.get("output_dir", "eval_gflownet_stream"))
            output_dir.mkdir(parents=True, exist_ok=True)
            self._stream_output_dir = output_dir

        processed = self._stream_processor.process(merged)
        assert self._stream_output_dir is not None
        output_path = self._stream_output_dir / f"{split}_gflownet_eval.jsonl"
        with output_path.open("a", encoding="utf-8") as f:
            for rec in processed:
                f.write(json.dumps(rec) + "\n")
        logger.info("Stream-persisted %d samples to %s", len(processed), output_path)

    def _compute_log_flow_states(
        self,
        *,
        state_emb_seq: torch.Tensor,  # [B, T_action, H]
        question_tokens: torch.Tensor,  # [B, H]
        log_reward: torch.Tensor,  # [B]
        edge_lengths: torch.Tensor,  # [B] number of selected edges
    ) -> torch.Tensor:
        """Compute logF(s_t) for SubTB, with terminal logF(s_T)=logR."""
        if state_emb_seq.dim() != 3:
            raise ValueError(f"state_emb_seq must be [B, T, H], got shape={tuple(state_emb_seq.shape)}")
        num_graphs, num_steps, hidden_dim = state_emb_seq.shape
        if question_tokens.size(0) != num_graphs:
            raise ValueError("question_tokens batch size mismatch with state_emb_seq.")
        if question_tokens.size(-1) != hidden_dim:
            raise ValueError("question_tokens hidden_dim mismatch with state_emb_seq.")
        if log_reward.numel() != num_graphs:
            raise ValueError("log_reward batch size mismatch with state_emb_seq.")

        state_flat = state_emb_seq.reshape(num_graphs * num_steps, hidden_dim)
        q_flat = question_tokens.unsqueeze(1).expand(num_graphs, num_steps, hidden_dim).reshape(num_graphs * num_steps, hidden_dim)
        context = self.estimator.build_context(state_flat, q_flat)
        log_flow_flat = self.estimator.log_z(context)
        log_flow_pred = log_flow_flat.view(num_graphs, num_steps)

        # Pad one extra terminal state; fill it with log_reward for the max-length case.
        log_flow_states = torch.zeros(num_graphs, num_steps + 1, device=state_emb_seq.device, dtype=log_flow_pred.dtype)
        log_flow_states[:, :num_steps] = log_flow_pred
        log_flow_states[:, num_steps] = log_reward.to(dtype=log_flow_pred.dtype)

        # Terminal state index is (stop_step + 1) where stop_step == #selected_edges.
        terminal_state = edge_lengths.clamp(min=0, max=num_steps - 1) + 1
        log_flow_states.scatter_(1, terminal_state.view(-1, 1), log_reward.view(-1, 1).to(dtype=log_flow_pred.dtype))
        return log_flow_states

    @staticmethod
    def _compute_deterministic_log_pb_steps(
        *,
        actions_seq: torch.Tensor,  # [B, T_action]
        stop_indices: torch.Tensor,  # [B]
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Deterministic P_B: each non-terminal state has a unique predecessor, so log P_B = 0."""
        if actions_seq.dim() != 2:
            raise ValueError(f"actions_seq must be 2D [B,T], got shape={tuple(actions_seq.shape)}")
        if stop_indices.dim() != 1 or stop_indices.numel() != actions_seq.size(0):
            raise ValueError("stop_indices must be [B] and aligned with actions_seq batch size.")
        return torch.zeros_like(actions_seq, dtype=dtype, device=device)

    def _compute_subtb_loss(
        self,
        *,
        log_flow_states: torch.Tensor,  # [B, T_state]
        log_pf_steps: torch.Tensor,  # [B, T_action]
        log_pb_steps: torch.Tensor,  # [B, T_action]
        edge_lengths: torch.Tensor,  # [B] number of selected edges
        return_per_graph: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sub-Trajectory Balance with λ=1 (uniform weights over all sub-trajectories)."""
        device = log_pf_steps.device
        num_graphs, num_actions = log_pf_steps.shape
        if debug_logger.isEnabledFor(logging.INFO) and not self._subtb_shapes_logged:
            term_state_preview = edge_lengths.clamp(min=0, max=max(num_actions - 1, 0)) + 1
            debug_logger.info(
                "[SUBTB_SHAPE] log_flow_states=%s log_pf_steps=%s log_pb_steps=%s edge_lengths=%s term_state(min=%d,max=%d)",
                tuple(log_flow_states.shape),
                tuple(log_pf_steps.shape),
                tuple(log_pb_steps.shape),
                tuple(edge_lengths.shape),
                int(term_state_preview.min().item()) if term_state_preview.numel() else -1,
                int(term_state_preview.max().item()) if term_state_preview.numel() else -1,
            )
            self._subtb_shapes_logged = True
        if log_flow_states.shape != (num_graphs, num_actions + 1):
            raise ValueError(
                f"log_flow_states shape {tuple(log_flow_states.shape)} != (B, T_action+1)=({num_graphs},{num_actions + 1})"
            )
        if log_pb_steps.shape != log_pf_steps.shape:
            raise ValueError("log_pb_steps must have the same shape as log_pf_steps.")
        if log_pb_steps.numel() > 0 and bool((log_pb_steps != 0).any().item()):
            raise ValueError("Deterministic P_B expects log_pb_steps to be all zeros.")

        log_pf_prefix = torch.zeros(num_graphs, num_actions + 1, device=device, dtype=log_pf_steps.dtype)
        log_pf_prefix[:, 1:] = log_pf_steps.cumsum(dim=1)

        # Segment log-prob from state i -> j (i<j): sum_{t=i}^{j-1} log_pf_steps = prefix[j] - prefix[i].
        pf_seg = log_pf_prefix.unsqueeze(1) - log_pf_prefix.unsqueeze(2)
        residual = log_flow_states.unsqueeze(2) + pf_seg - log_flow_states.unsqueeze(1)

        t_state = num_actions + 1
        idx = torch.arange(t_state, device=device)
        i_idx = idx.view(1, t_state, 1)
        j_idx = idx.view(1, 1, t_state)
        term_state = edge_lengths.clamp(min=0, max=num_actions - 1) + 1
        term_state = term_state.view(num_graphs, 1, 1)
        valid = (i_idx < j_idx) & (i_idx <= term_state) & (j_idx <= term_state)
        valid_f = valid.to(dtype=residual.dtype)
        sq = residual.pow(2) * valid_f
        denom = valid_f.sum(dim=(1, 2)).clamp(min=1.0)
        per_graph = sq.sum(dim=(1, 2)) / denom
        mean_loss = per_graph.mean()
        if return_per_graph:
            return mean_loss, per_graph
        return mean_loss

    @staticmethod
    def _mask_subtb_loss_by_path_exists(*, subtb_per_graph: torch.Tensor, path_exists: torch.Tensor) -> torch.Tensor:
        valid_graphs_for_loss = path_exists.bool()
        if valid_graphs_for_loss.any():
            return subtb_per_graph[valid_graphs_for_loss].mean()
        return torch.zeros((), device=subtb_per_graph.device, dtype=subtb_per_graph.dtype)

    def _assert_finite(self, tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).sum().item()
            raise ValueError(f"{name} contains {bad} non-finite values.")


__all__ = ["GFlowNetModule"]


class _EvalPersistProcessor:
    """
    精简的持久化处理器：文本化 edges，构造 chains（连续 head==prev_tail），再聚合为 candidate_chains。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.ent_map, self.rel_map = self._resolve_vocab_maps(cfg)

    def process(self, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if self.ent_map is not None or self.rel_map is not None:
            self._inject_text(records)
        for sample in records:
            sample["candidate_chains"] = self._build_candidate_paths(sample)
        return records

    def _inject_text(self, records: list[Dict[str, Any]]) -> None:
        for sample in records:
            for rollout in sample.get("rollouts", []):
                for edge in rollout.get("edges", []):
                    h = edge.get("head_entity_id")
                    r = edge.get("relation_id")
                    t = edge.get("tail_entity_id")
                    s_id = edge.get("src_entity_id")
                    d_id = edge.get("dst_entity_id")
                    if self.ent_map is not None:
                        edge["head_text"] = self.ent_map.get(h, str(h) if h is not None else None)
                        edge["tail_text"] = self.ent_map.get(t, str(t) if t is not None else None)
                        edge["src_text"] = self.ent_map.get(s_id, str(s_id) if s_id is not None else None)
                        edge["dst_text"] = self.ent_map.get(d_id, str(d_id) if d_id is not None else None)
                    if self.rel_map is not None:
                        edge["relation_text"] = self.rel_map.get(r, str(r) if r is not None else None)

    def _build_candidate_paths(self, sample: Dict[str, Any]) -> list[Dict[str, Any]]:
        chain_stats: Dict[tuple, Dict[str, Any]] = {}
        for rollout in sample.get("rollouts", []):
            ridx = int(rollout.get("rollout_index", 0) or 0)
            path = rollout.get("edges") or []
            sig = []
            for e in path:
                src = e.get("src_entity_id")
                if src is None:
                    src = e.get("head_entity_id")
                dst = e.get("dst_entity_id")
                if dst is None:
                    dst = e.get("tail_entity_id")
                sig.append((src, e.get("relation_id"), dst))
            sig = tuple(sig)
            if not sig:
                continue
            stat = chain_stats.setdefault(
                sig,
                {
                    "frequency": 0,
                    "from_rollouts": set(),
                    "example_edges": path,
                },
            )
            stat["frequency"] += 1
            stat["from_rollouts"].add(ridx)

        candidates: list[Dict[str, Any]] = []
        for sig, stat in chain_stats.items():
            edges = stat["example_edges"]
            chain_text = " -> ".join(self._fmt_edge(e) for e in edges)
            candidates.append(
                {
                    "signature": sig,
                    "length": len(edges),
                    "frequency": stat["frequency"],
                    "from_rollouts": sorted(stat["from_rollouts"]),
                    "chain_edges": [
                        {
                            "head_entity_id": e.get("head_entity_id"),
                            "relation_id": e.get("relation_id"),
                            "tail_entity_id": e.get("tail_entity_id"),
                            "head_text": e.get("head_text"),
                            "relation_text": e.get("relation_text"),
                            "tail_text": e.get("tail_text"),
                            "edge_score": e.get("edge_score"),
                            # 保留实际行走方向，便于文本化时使用 src->dst 而非图定义的 head->tail
                            "src_entity_id": e.get("src_entity_id"),
                            "dst_entity_id": e.get("dst_entity_id"),
                        }
                        for e in edges
                    ],
                    "chain_text": chain_text,
                }
            )

        candidates.sort(key=lambda c: (-c["frequency"], -c["length"]))
        for i, c in enumerate(candidates, 1):
            c["rank"] = i
        return candidates

    @staticmethod
    def _fmt_edge(e: Dict[str, Any]) -> str:
        def _txt(val_text: Any, val_id: Any) -> str:
            if val_text is not None:
                return str(val_text)
            if val_id is None:
                return "UNK"
            return str(val_id)

        # 优先使用 roll-out 实际的 src/dst，以避免 head/tail 方向与游走方向不一致
        h = _txt(e.get("src_text"), e.get("src_entity_id"))
        t = _txt(e.get("dst_text"), e.get("dst_entity_id"))
        if h == "UNK" and t == "UNK":
            # 回退到图定义的 head/tail
            h = _txt(e.get("head_text"), e.get("head_entity_id"))
            t = _txt(e.get("tail_text"), e.get("tail_entity_id"))
        r = _txt(e.get("relation_text"), e.get("relation_id"))
        return f"{h} -[{r}]-> {t}"

    def _resolve_vocab_maps(self, cfg: Dict[str, Any]) -> tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
        if not cfg.get("textualize"):
            return None, None
        entity_path = cfg.get("entity_vocab_path")
        relation_path = cfg.get("relation_vocab_path")
        if not entity_path or not relation_path:
            raise ValueError("eval_persist_cfg.textualize=true requires both entity_vocab_path and relation_vocab_path.")
        if not Path(entity_path).exists():
            raise FileNotFoundError(f"entity_vocab_path not found: {entity_path}")
        if not Path(relation_path).exists():
            raise FileNotFoundError(f"relation_vocab_path not found: {relation_path}")
        ent_map: Optional[Dict[int, str]] = None
        rel_map: Optional[Dict[int, str]] = None
        try:
            if entity_path and Path(entity_path).exists():
                ent_df = pd.read_parquet(entity_path)
                if "entity_id" in ent_df.columns:
                    ent_map = dict(zip(ent_df.entity_id.astype(int), ent_df.label.astype(str)))
                elif "embedding_id" in ent_df.columns:
                    ent_map = dict(zip(ent_df.embedding_id.astype(int), ent_df.label.astype(str)))
            if relation_path and Path(relation_path).exists():
                rel_df = pd.read_parquet(relation_path)
                rel_map = dict(zip(rel_df.relation_id.astype(int), rel_df.label.astype(str)))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load vocab for textualize: %s", exc)
        return ent_map, rel_map
