from __future__ import annotations

import json
import contextlib
import math
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

from src.models.components import (
    GFlowNetActor,
    GFlowNetEstimator,
    GraphEmbedder,
    GraphEnv,
    GNNStateEncoder,
    RewardOutput,
)
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
        eval_temp_cfg = self._cfg_get_float(self.evaluation_cfg, "rollout_temperature", None)
        if eval_temp_cfg is None:
            raise ValueError("evaluation_cfg.rollout_temperature must be set explicitly (no implicit eval temperature).")
        if eval_temp_cfg < 0.0:
            raise ValueError(f"evaluation_cfg.rollout_temperature must be >= 0, got {eval_temp_cfg}.")
        self._eval_rollout_temperature = float(eval_temp_cfg)
        self._debug_batches_to_log = max(int(self._cfg_get(self.training_cfg, "debug_batches_to_log", 0)), 0)
        self._debug_graphs_to_log = max(int(self._cfg_get(self.training_cfg, "debug_graphs_to_log", 1)), 1)
        self._debug_batches_logged = 0
        self._subtb_shapes_logged = False
        self._train_prog_bar = set(self._cfg_get(self.logging_cfg, "train_prog_bar", []))
        self._eval_prog_bar = set(self._cfg_get(self.logging_cfg, "eval_prog_bar", []))
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
        self.predict_metrics: Dict[str, torch.Tensor] = {}
        self._last_debug_epoch: int = -1
        self._estimated_stepping_batches: Optional[int] = None

        self.policy: nn.Module = hydra.utils.instantiate(policy_cfg)
        self.reward_fn: nn.Module = hydra.utils.instantiate(reward_cfg)
        self.env: GraphEnv = hydra.utils.instantiate(env_cfg)
        self.max_steps = int(self.env.max_steps)
        self.embedder = hydra.utils.instantiate(embedder_cfg)
        self.estimator = hydra.utils.instantiate(estimator_cfg)
        state_encoder: GNNStateEncoder = hydra.utils.instantiate(state_encoder_cfg)
        self.actor = hydra.utils.instantiate(
            actor_cfg,
            policy=self.policy,
            env=self.env,
            state_encoder=state_encoder,
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

    def on_fit_start(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        est = getattr(trainer, "estimated_stepping_batches", None)
        if est is None:
            return
        try:
            est_int = int(est)
        except Exception:
            return
        self._estimated_stepping_batches = est_int if est_int > 0 else None

    def _train_progress(self) -> Optional[float]:
        total = self._estimated_stepping_batches
        if total is None:
            trainer = getattr(self, "trainer", None)
            est = getattr(trainer, "estimated_stepping_batches", None) if trainer is not None else None
            try:
                total = int(est) if est is not None else None
            except Exception:
                total = None
        if total is None or total <= 0:
            return None
        return min(1.0, max(0.0, float(self.global_step) / float(max(1, total))))

    def _scheduled_coef(self, base_coef: float, schedule_cfg: Any) -> float:
        if not schedule_cfg:
            return float(base_coef)
        schedule_type = str(self._cfg_get(schedule_cfg, "type", "cosine")).strip().lower()
        if schedule_type in {"none", "null", "constant", "const", "identity", "id"}:
            return float(base_coef)
        progress = self._train_progress()
        if progress is None:
            return float(base_coef)
        final_coef = float(self._cfg_get_float(schedule_cfg, "final_coef", 0.0) or 0.0)
        start_frac = float(self._cfg_get_float(schedule_cfg, "start_frac", 0.0) or 0.0)
        duration_frac = float(self._cfg_get_float(schedule_cfg, "duration_frac", 1.0) or 1.0)
        denom = max(duration_frac, 1e-12)
        phase = (progress - start_frac) / denom
        phase = min(1.0, max(0.0, float(phase)))
        if schedule_type in {"linear", "lin"}:
            coef = float(base_coef) + (final_coef - float(base_coef)) * phase
        elif schedule_type in {"cosine", "cos"}:
            coef = final_coef + 0.5 * (float(base_coef) - final_coef) * (1.0 + math.cos(math.pi * phase))
        else:
            raise ValueError(
                f"Unsupported schedule.type={schedule_type!r} (expected 'none'|'linear'|'cosine')."
            )
        return max(0.0, float(coef))

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            checkpoint["retriever_meta"] = self.embedder.export_retriever_meta()
        except Exception as exc:
            raise RuntimeError(f"Failed to export retriever_meta for checkpoint: {exc}") from exc

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        meta = checkpoint.get("retriever_meta")
        if meta is None:
            if getattr(self.embedder, "allow_deferred_init", False):
                raise RuntimeError("Missing retriever_meta in gflownet checkpoint; cannot init embedder for eval.")
            return
        try:
            self.embedder.init_from_retriever_meta(meta)
        except Exception as exc:
            raise RuntimeError(f"Failed to apply retriever_meta from checkpoint: {exc}") from exc

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
    def _cfg_get_float(cfg: Any, key: str, default: Optional[float]) -> Optional[float]:
        value = GFlowNetModule._cfg_get(cfg, key, default)
        if value is None:
            return default
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            try:
                return float(value[0]) if len(value) > 0 else default
            except Exception:
                return default
        try:
            return float(value)
        except Exception:
            return default

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
            "answer_reach_frac": metrics.get("answer_reach_frac"),
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
        self.predict_metrics = metrics

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str, batch_size: int) -> None:
        if prefix == "predict":
            return
        sync_dist = bool(self.trainer and getattr(self.trainer, "num_devices", 1) > 1)
        is_train = prefix == "train"
        prog_bar_set = set(self._train_prog_bar if is_train else self._eval_prog_bar)
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
            self._debug_batches_logged += 1
        edge_tokens = embed.edge_tokens.to(dtype=torch.float32)
        edge_batch = embed.edge_batch
        edge_ptr = embed.edge_ptr
        node_ptr = embed.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        edge_index = embed.edge_index
        edge_relations = embed.edge_relations
        edge_labels = batch.edge_labels.to(device)
        heads_global = embed.heads_global
        tails_global = embed.tails_global
        node_tokens = embed.node_tokens.to(dtype=torch.float32)
        question_tokens = embed.question_tokens.to(dtype=torch.float32)
        answer_node_ptr = batch._slice_dict["answer_node_locals"].to(device)
        # SubTB：logF(s_t) 在每一步状态上预测；不再使用单独的 start_summary/logZ 常数项。
        subtb_penalty = str(self._cfg_get(self.training_cfg, "subtb_penalty", "l2")).strip().lower()

        graph_cache: Dict[str, torch.Tensor] = {
            "edge_index": edge_index,
            "edge_batch": edge_batch,
            "node_global_ids": batch.node_global_ids.to(device),
            "heads_global": heads_global,
            "tails_global": tails_global,
            "edge_scores": batch.edge_scores.to(device=device, dtype=torch.float32).view(-1),
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "node_tokens": node_tokens,
            "start_node_locals": batch.start_node_locals.to(device),
            "start_ptr": batch._slice_dict["start_node_locals"].to(device),
            "answer_node_locals": batch.answer_node_locals.to(device),
            "answer_ptr": batch._slice_dict["answer_node_locals"].to(device),
        }

        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=False)
            if device.type != "cpu"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            state_encoder_cache = self.actor.state_encoder.precompute(
                edge_index=edge_index,
                edge_batch=edge_batch,
                node_ptr=node_ptr,
                start_node_locals=graph_cache["start_node_locals"],
                start_ptr=graph_cache["start_ptr"],
                node_tokens=node_tokens,
                edge_tokens=edge_tokens,
                question_tokens=question_tokens,
            )

        bc_coef_base = float(self._cfg_get_float(self.training_cfg, "bc_coef", 0.0) or 0.0)
        bc_schedule_cfg = self._cfg_get(self.training_cfg, "bc_schedule", None)
        bc_coef = self._scheduled_coef(bc_coef_base, bc_schedule_cfg) if (self.training and bc_schedule_cfg) else bc_coef_base
        bc_include_stop = bool(self._cfg_get(self.training_cfg, "bc_include_stop", False))

        gt_flow_coef_base = float(self._cfg_get_float(self.training_cfg, "gt_flow_coef", 0.0) or 0.0)
        gt_flow_schedule_cfg = self._cfg_get(self.training_cfg, "gt_flow_schedule", None)
        gt_flow_coef = (
            self._scheduled_coef(gt_flow_coef_base, gt_flow_schedule_cfg)
            if (self.training and gt_flow_schedule_cfg)
            else gt_flow_coef_base
        )

        bc_loss: Optional[torch.Tensor] = None
        bc_metrics: Dict[str, torch.Tensor] = {}
        gt_subtb_loss: Optional[torch.Tensor] = None
        gt_flow_metrics: Dict[str, torch.Tensor] = {}

        if self.training and (bc_coef > 0.0 or gt_flow_coef > 0.0):
            gt_actions_seq, gt_lens, usable = self._sample_gt_actions_from_dag(
                edge_index=edge_index,
                edge_labels=edge_labels,
                edge_ptr=edge_ptr,
                node_ptr=node_ptr,
                start_node_locals=graph_cache["start_node_locals"],
                start_ptr=graph_cache["start_ptr"],
                answer_node_locals=graph_cache["answer_node_locals"],
                answer_ptr=graph_cache["answer_ptr"],
                max_steps=self.max_steps,
            )
            usable_frac = usable.float().mean() if num_graphs > 0 else torch.zeros((), device=device)

            stop_indices = edge_ptr[1:].to(device=device, dtype=torch.long)
            num_steps_bc = self.max_steps + 1
            if gt_actions_seq.shape != (num_graphs, num_steps_bc):
                raise ValueError(
                    f"gt_actions_seq shape {tuple(gt_actions_seq.shape)} != ({num_graphs},{num_steps_bc})"
                )

            rollout_gt = self.actor.rollout(
                batch=batch,
                edge_tokens=edge_tokens,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
                edge_batch=edge_batch,
                edge_ptr=edge_ptr,
                node_ptr=node_ptr,
                temperature=None,
                batch_idx=batch_idx,
                graph_cache=graph_cache,
                forced_actions_seq=gt_actions_seq,
                state_encoder_cache=state_encoder_cache,
            )
            log_pf_steps_gt = rollout_gt["log_pf_steps"]
            if log_pf_steps_gt.shape != (num_graphs, num_steps_bc):
                raise ValueError(
                    f"GT log_pf_steps shape {tuple(log_pf_steps_gt.shape)} != ({num_graphs},{num_steps_bc})"
                )

            if bc_coef > 0.0:
                target_len = gt_lens + (1 if bc_include_stop else 0)
                target_len = torch.clamp(target_len, min=0, max=num_steps_bc)
                steps = torch.arange(num_steps_bc, device=device).view(1, -1)
                step_mask = steps < target_len.view(-1, 1)
                step_mask = step_mask & usable.view(-1, 1)
                denom = step_mask.sum().to(dtype=log_pf_steps_gt.dtype)
                if float(denom.item()) > 0.0:
                    bc_loss = -(log_pf_steps_gt * step_mask.to(dtype=log_pf_steps_gt.dtype)).sum() / denom
                else:
                    bc_loss = torch.zeros((), device=device, dtype=log_pf_steps_gt.dtype)
                bc_metrics = {
                    "bc_loss": bc_loss.detach(),
                    "bc_coef": torch.tensor(bc_coef, device=device, dtype=bc_loss.dtype),
                    "bc_usable_frac": usable_frac.detach(),
                    "bc_steps_mean": (target_len.to(dtype=log_pf_steps_gt.dtype)[usable].mean().detach() if usable.any() else torch.zeros((), device=device)),
                }

            if gt_flow_coef > 0.0:
                reward_out_gt: RewardOutput = self.reward_fn(
                    selected_mask=rollout_gt["selected_mask"],
                    edge_labels=edge_labels,
                    edge_batch=edge_batch,
                    edge_index=edge_index,
                    node_ptr=node_ptr,
                    start_node_locals=batch.start_node_locals.to(device),
                    answer_node_locals=batch.answer_node_locals.to(device),
                    answer_node_ptr=answer_node_ptr,
                )
                log_reward_gt = reward_out_gt.log_reward
                self._assert_finite(log_reward_gt, "log_reward_gt")

                state_emb_seq_gt = rollout_gt.get("state_emb_seq")
                if state_emb_seq_gt is None:
                    raise ValueError("state_emb_seq missing in GT rollout; actor/state_encoder must return per-step embeddings.")

                log_flow_states_gt = self._compute_log_flow_states(
                    state_emb_seq=state_emb_seq_gt,
                    question_tokens=question_tokens,
                    log_reward=log_reward_gt,
                    edge_lengths=rollout_gt["length"].long(),
                )
                phi_states_gt = rollout_gt.get("phi_states", None)
                if phi_states_gt is None:
                    raise ValueError("phi_states missing in GT rollout; GraphEnv.potential must be enabled.")
                if phi_states_gt.shape != log_flow_states_gt.shape:
                    raise ValueError(
                        f"phi_states_gt shape {tuple(phi_states_gt.shape)} != log_flow_states_gt {tuple(log_flow_states_gt.shape)}"
                    )
                phi_states_gt = phi_states_gt.to(device=device, dtype=log_flow_states_gt.dtype).detach()
                term_state_gt = rollout_gt["length"].long().clamp(min=0, max=rollout_gt["log_pf_steps"].size(1) - 1) + 1
                phi_states_gt.scatter_(
                    1,
                    term_state_gt.view(-1, 1),
                    torch.zeros_like(term_state_gt, dtype=phi_states_gt.dtype).view(-1, 1),
                )
                log_flow_states_gt = log_flow_states_gt + phi_states_gt

                log_pb_steps_gt = self._compute_deterministic_log_pb_steps(
                    actions_seq=rollout_gt["actions_seq"],
                    stop_indices=edge_ptr[1:],
                    dtype=rollout_gt["log_pf_steps"].dtype,
                    device=device,
                )
                gt_subtb_loss = self._compute_subtb_loss(
                    log_flow_states=log_flow_states_gt,
                    log_pf_steps=rollout_gt["log_pf_steps"],
                    log_pb_steps=log_pb_steps_gt,
                    edge_lengths=rollout_gt["length"].long(),
                    graph_mask=usable,
                    penalty=subtb_penalty,
                )
                gt_flow_metrics = {
                    "gt_subtb_loss": gt_subtb_loss.detach(),
                    "gt_flow_coef": torch.tensor(gt_flow_coef, device=device, dtype=gt_subtb_loss.dtype),
                    "gt_usable_frac": usable_frac.detach(),
                }

        if self.training:
            num_rollouts = max(int(self._cfg_get_int(self.training_cfg, "num_train_rollouts", 1)), 1)
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
        rollout_logs: list[Dict[str, torch.Tensor]] = []
        rollout_answer_hits: list[torch.Tensor] = []
        rollout_lengths: list[torch.Tensor] = []

        for _ in range(num_rollouts):
            rollout = self.actor.rollout(
                batch=batch,
                edge_tokens=edge_tokens,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
                edge_batch=edge_batch,
                edge_ptr=edge_ptr,
                node_ptr=node_ptr,
                temperature=None if self.training else self._eval_rollout_temperature,
                batch_idx=batch_idx,
                graph_cache=graph_cache,
                state_encoder_cache=state_encoder_cache,
            )
            # Rollout returns log_pf/log_pf_steps/actions_seq for SubTB.
            state_emb_seq = rollout.get("state_emb_seq")
            if state_emb_seq is None:
                raise ValueError("state_emb_seq missing in rollout; actor/state_encoder must return per-step embeddings.")

            reward_out: RewardOutput = self.reward_fn(
                selected_mask=rollout["selected_mask"],
                edge_labels=edge_labels,
                edge_batch=edge_batch,
                edge_index=edge_index,
                node_ptr=node_ptr,
                start_node_locals=batch.start_node_locals.to(device),
                answer_node_locals=batch.answer_node_locals.to(device),
                answer_node_ptr=answer_node_ptr,
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
            phi_states = rollout.get("phi_states", None)
            if phi_states is None:
                raise ValueError("phi_states missing in rollout; GraphEnv.potential must be enabled.")
            if phi_states.shape != log_flow_states.shape:
                raise ValueError(
                    f"phi_states shape {tuple(phi_states.shape)} != log_flow_states {tuple(log_flow_states.shape)}"
                )
            phi_states = phi_states.to(device=device, dtype=log_flow_states.dtype).detach()
            term_state = rollout["length"].long().clamp(min=0, max=rollout["log_pf_steps"].size(1) - 1) + 1
            phi_states.scatter_(
                1,
                term_state.view(-1, 1),
                torch.zeros_like(term_state, dtype=phi_states.dtype).view(-1, 1),
            )
            log_flow_states = log_flow_states + phi_states
            # Backward policy P_B is deterministic under this state definition:
            # given the full trajectory state (selected edges + order), the predecessor is uniquely
            # obtained by removing the last selected edge. Hence log P_B = 0 for every realized step.
            log_pb_steps = self._compute_deterministic_log_pb_steps(
                actions_seq=rollout["actions_seq"],
                stop_indices=edge_ptr[1:],
                dtype=rollout["log_pf_steps"].dtype,
                device=device,
            )
            subtb_loss = self._compute_subtb_loss(
                log_flow_states=log_flow_states,
                log_pf_steps=rollout["log_pf_steps"],
                log_pb_steps=log_pb_steps,
                edge_lengths=rollout["length"].long(),
                penalty=subtb_penalty,
            )
            loss = subtb_loss
            if bc_loss is not None and bc_coef > 0.0:
                loss = loss + bc_coef * bc_loss
            if gt_subtb_loss is not None and gt_flow_coef > 0.0:
                loss = loss + gt_flow_coef * gt_subtb_loss

            if should_debug:
                self._log_rollout_sanity_debug(
                    rollout=rollout,
                    log_reward=log_reward,
                    log_flow_states=log_flow_states,
                    edge_ptr=edge_ptr,
                    batch_idx=batch_idx,
                )

            reward_metrics = reward_out.as_dict()
            log_reward_metric = reward_metrics.pop("log_reward")
            answer_hit = reward_metrics.pop("answer_hit", None)
            answer_reach_frac = reward_metrics.pop("answer_reach_frac", None)
            reward_metrics.pop("reward", None)
            reward_metrics.pop("success", None)

            metrics: Dict[str, torch.Tensor] = {
                "log_reward": log_reward_metric,
                "answer_hit": (
                    answer_hit.detach()
                    if isinstance(answer_hit, torch.Tensor)
                    else rollout["reach_success"].detach()
                ),
                "answer_reach_frac": (
                    answer_reach_frac.detach()
                    if isinstance(answer_reach_frac, torch.Tensor)
                    else torch.zeros(num_graphs, device=device, dtype=log_reward_metric.dtype)
                ),
                "length_mean": rollout["length"].detach(),
                "subtb_loss": subtb_loss.detach(),
                **{k: v.detach() for k, v in reward_metrics.items()},
                **bc_metrics,
                **gt_flow_metrics,
            }

            if not self.training:
                if isinstance(answer_hit, torch.Tensor):
                    answer_val = answer_hit.detach().to(dtype=torch.bool)
                else:
                    answer_val = rollout["reach_success"].detach().to(dtype=torch.bool)
                rollout_answer_hits.append(answer_val)
                rollout_lengths.append(rollout["length"].detach().to(dtype=log_reward_metric.dtype))

            loss_list.append(loss)
            metrics_list.append(metrics)
            if collect_rollouts:
                rollout_logs.append(
                    {
                        "selected_mask": rollout["selected_mask"].detach().cpu(),
                        "selection_order": rollout["selection_order"].detach().cpu(),
                        "log_pf": rollout["log_pf"].detach().cpu(),
                        "log_reward": log_reward.detach().cpu(),
                    }
                )

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
                if rollout_answer_hits:
                    answer_stack = torch.stack(rollout_answer_hits, dim=0).bool()
                else:
                    answer_stack = torch.zeros(1, num_graphs, device=device, dtype=torch.bool)
                rollouts_avail = answer_stack.size(0)
                for k_int in k_values:
                    k_clamped = min(max(k_int, 1), rollouts_avail)
                    hit_any = answer_stack[:k_clamped].any(dim=0)
                    metrics[f"answer_hit@{k_int}"] = hit_any.float().mean()
        if collect_rollouts and rollout_logs:
            self._buffer_rollout_records(
                split=split or "test",
                batch=batch,
                rollout_logs=rollout_logs,
                heads_global=heads_global.detach().cpu(),
                tails_global=tails_global.detach().cpu(),
                edge_relations=edge_relations.detach().cpu(),
                edge_index=edge_index.detach().cpu(),
                edge_ptr=edge_ptr.detach().cpu(),
                node_ptr=node_ptr.detach().cpu(),
                start_node_ptr=batch._slice_dict["start_node_locals"].detach().cpu(),
                start_node_locals=batch.start_node_locals.detach().cpu(),
            )
        return loss, metrics

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

    def _log_rollout_sanity_debug(
        self,
        *,
        rollout: Dict[str, torch.Tensor],
        log_reward: torch.Tensor,
        log_flow_states: torch.Tensor,
        edge_ptr: torch.Tensor,
        batch_idx: int | None,
    ) -> None:
        """打印 SubTB 关键量的确定性一致性检查（可直接粘贴日志定位问题）。"""
        try:
            log_pf_steps = rollout["log_pf_steps"].detach()
            log_pf = rollout["log_pf"].detach()
            actions_seq = rollout["actions_seq"].detach()
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
                stop_idx = edge_ptr[1:].view(-1, 1)  # [B,1] for broadcast with actions[B,T]
                stop_counts = (actions_seq == stop_idx).sum(dim=1).float()
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

    def _log_batch_debug(self, *, batch: Any, embed, batch_idx: int | None) -> None:
        try:
            node_ptr = batch.ptr.detach().cpu()
            edge_ptr = embed.edge_ptr.detach().cpu()
            edge_index = batch.edge_index.detach().cpu()
            edge_batch = embed.edge_batch.detach().cpu()
            node_global_ids = batch.node_global_ids.detach().cpu()
            start_node_locals = batch.start_node_locals.detach().cpu()
            answer_node_locals = batch.answer_node_locals.detach().cpu()
            start_ptr = batch._slice_dict.get("start_node_locals")
            answer_ptr = batch._slice_dict.get("answer_node_locals")
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

            start_nodes = start_node_locals[start_slice]
            answer_nodes = answer_node_locals[answer_slice]

            start_oob = start_nodes.numel() > 0 and ((start_nodes < ns) | (start_nodes >= ne)).any().item()
            answer_oob = answer_nodes.numel() > 0 and ((answer_nodes < ns) | (answer_nodes >= ne)).any().item()

            edge_batch_slice = edge_batch[es:ee]
            edge_batch_mismatch = bool(edge_batch_slice.numel() > 0 and torch.unique(edge_batch_slice).numel() != 1)
            node_dup = max(int(nodes.numel() - torch.unique(nodes).numel()), 0)

            start_count = int(start_nodes.numel())
            answer_count = int(answer_nodes.numel())

            summary: Dict[str, Any] = {
                "graph": g,
                "node_span": [ns, ne],
                "edge_span": [es, ee],
                "node_dup": node_dup,
                "edge_batch_ok": not edge_batch_mismatch,
                "start_locals": (start_nodes - ns).tolist(),
                "start_count": start_count,
                "answer_locals": (answer_nodes - ns).tolist(),
                "answer_count": answer_count,
                "flags": {
                    "start_oob": bool(start_oob),
                    "answer_oob": bool(answer_oob),
                },
            }
            debug_logger.info("[GRAPH_DEBUG] %s", summary)

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

    def _refresh_eval_settings(self) -> None:
        """Ensure eval rollouts reflect current config (even when loading old checkpoints)."""
        eval_cfg = self.evaluation_cfg or {}
        rollouts_cfg = self._cfg_get(eval_cfg, "num_eval_rollouts", self._eval_rollouts if hasattr(self, "_eval_rollouts") else 1)
        if isinstance(rollouts_cfg, (list, tuple, ListConfig)):
            prefixes = sorted({int(max(1, v)) for v in rollouts_cfg})
            self._eval_rollouts = max(prefixes) if prefixes else 1
            self._eval_rollout_prefixes = prefixes or [self._eval_rollouts]
        else:
            self._eval_rollouts = int(max(1, self._cfg_get_int(eval_cfg, "num_eval_rollouts", 1)))
            self._eval_rollout_prefixes = [self._eval_rollouts]
        eval_temp_cfg = self._cfg_get_float(eval_cfg, "rollout_temperature", self._eval_rollout_temperature)
        if eval_temp_cfg is None:
            raise ValueError("evaluation_cfg.rollout_temperature must be set explicitly (no implicit eval temperature).")
        if eval_temp_cfg < 0.0:
            raise ValueError(f"evaluation_cfg.rollout_temperature must be >= 0, got {eval_temp_cfg}.")
        self._eval_rollout_temperature = float(eval_temp_cfg)

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
                            "edge_local_index": int(edge_idx - es),
                            "src_entity_id": src_global,
                            "dst_entity_id": dst_global,
                        }
                    )

                rollouts.append(
                    {
                        "rollout_index": ridx,
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
    def _sample_gt_actions_from_dag(
        *,
        edge_index: torch.Tensor,
        edge_labels: torch.Tensor,
        edge_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        start_node_locals: torch.Tensor,
        start_ptr: torch.Tensor,
        answer_node_locals: torch.Tensor,
        answer_ptr: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_index.device
        num_graphs = int(node_ptr.numel() - 1)
        num_steps = int(max_steps) + 1
        if num_graphs <= 0:
            empty_actions = torch.empty((0, num_steps), dtype=torch.long, device=device)
            empty_lens = torch.empty((0,), dtype=torch.long, device=device)
            empty_mask = torch.empty((0,), dtype=torch.bool, device=device)
            return empty_actions, empty_lens, empty_mask

        edge_index_cpu = edge_index.detach().cpu()
        edge_labels_cpu = edge_labels.detach().cpu().view(-1)
        edge_ptr_cpu = edge_ptr.detach().cpu().view(-1)
        node_ptr_cpu = node_ptr.detach().cpu().view(-1)
        start_node_locals_cpu = start_node_locals.detach().cpu().view(-1)
        start_ptr_cpu = start_ptr.detach().cpu().view(-1)
        answer_node_locals_cpu = answer_node_locals.detach().cpu().view(-1)
        answer_ptr_cpu = answer_ptr.detach().cpu().view(-1)

        actions_buf: list[list[int]] = []
        lens_buf: list[int] = []
        usable_buf: list[bool] = []

        from collections import deque

        for g in range(num_graphs):
            stop_idx = int(edge_ptr_cpu[g + 1].item())
            actions = [stop_idx] * num_steps
            lens = 0
            usable = False

            e0 = int(edge_ptr_cpu[g].item())
            e1 = int(edge_ptr_cpu[g + 1].item())
            n0 = int(node_ptr_cpu[g].item())
            n1 = int(node_ptr_cpu[g + 1].item())
            if e1 <= e0 or n1 <= n0:
                actions_buf.append(actions)
                lens_buf.append(lens)
                usable_buf.append(usable)
                continue

            pos_mask = edge_labels_cpu[e0:e1] > 0.5
            if not bool(pos_mask.any()):
                actions_buf.append(actions)
                lens_buf.append(lens)
                usable_buf.append(usable)
                continue

            num_nodes = n1 - n0
            heads = (edge_index_cpu[0, e0:e1] - n0).tolist()
            tails = (edge_index_cpu[1, e0:e1] - n0).tolist()
            pos_mask_list = pos_mask.tolist()

            s0 = int(start_ptr_cpu[g].item())
            s1 = int(start_ptr_cpu[g + 1].item())
            starts = [
                int(x) - n0
                for x in start_node_locals_cpu[s0:s1].tolist()
                if 0 <= int(x) - n0 < num_nodes
            ]
            a0 = int(answer_ptr_cpu[g].item())
            a1 = int(answer_ptr_cpu[g + 1].item())
            answers = [
                int(x) - n0
                for x in answer_node_locals_cpu[a0:a1].tolist()
                if 0 <= int(x) - n0 < num_nodes
            ]
            if not starts or not answers:
                actions_buf.append(actions)
                lens_buf.append(lens)
                usable_buf.append(usable)
                continue

            adj_fwd: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
            adj_rev: list[list[int]] = [[] for _ in range(num_nodes)]
            for local_eid, keep in enumerate(pos_mask_list):
                if not keep:
                    continue
                h = heads[local_eid]
                t = tails[local_eid]
                if h < 0 or t < 0 or h >= num_nodes or t >= num_nodes:
                    continue
                global_eid = e0 + local_eid
                adj_fwd[h].append((t, global_eid))
                adj_rev[t].append(h)
            for nbrs in adj_fwd:
                nbrs.sort(key=lambda item: (item[0], item[1]))
            for nbrs in adj_rev:
                nbrs.sort()

            def bfs_dist(neighbors: list[list[int]], src_nodes: list[int]) -> list[int]:
                dist = [-1] * num_nodes
                q: deque[int] = deque()
                for s in src_nodes:
                    if 0 <= s < num_nodes and dist[s] < 0:
                        dist[s] = 0
                        q.append(s)
                while q:
                    u = q.popleft()
                    du = dist[u] + 1
                    for v in neighbors[u]:
                        if dist[v] >= 0:
                            continue
                        dist[v] = du
                        q.append(v)
                return dist

            adj_fwd_neighbors = [[nb for nb, _ in nbrs] for nbrs in adj_fwd]
            dist_start = bfs_dist(adj_fwd_neighbors, starts)
            reachable_answers = [a for a in answers if dist_start[a] >= 0]
            if not reachable_answers:
                actions_buf.append(actions)
                lens_buf.append(lens)
                usable_buf.append(usable)
                continue

            start_set = set(starts)
            perm = torch.randperm(len(reachable_answers)).tolist()
            for idx in perm:
                target = reachable_answers[idx]
                dist_to_t = bfs_dist(adj_rev, [target])
                if dist_to_t[target] != 0:
                    continue
                candidates: list[tuple[int, int, int]] = []
                for local_eid, keep in enumerate(pos_mask_list):
                    if not keep:
                        continue
                    h = heads[local_eid]
                    t = tails[local_eid]
                    if h not in start_set:
                        continue
                    src = h
                    dst = t
                    if dst in start_set:
                        continue
                    if dist_to_t[src] <= 0 or dist_to_t[dst] < 0:
                        continue
                    if dist_to_t[src] > max_steps:
                        continue
                    if dist_to_t[dst] != dist_to_t[src] - 1:
                        continue
                    candidates.append((src, dst, e0 + local_eid))
                if not candidates:
                    continue

                first_idx = int(torch.randint(len(candidates), (1,)).item())
                _, cur, first_edge = candidates[first_idx]
                path_edges = [int(first_edge)]
                steps = 1
                visited = set(start_set)
                visited.add(cur)
                while cur != target:
                    if steps >= max_steps:
                        path_edges = []
                        break
                    options = [
                        (nb, eid)
                        for nb, eid in adj_fwd[cur]
                        if dist_to_t[nb] == dist_to_t[cur] - 1 and nb not in visited
                    ]
                    if not options:
                        path_edges = []
                        break
                    opt_idx = int(torch.randint(len(options), (1,)).item())
                    cur, eid = options[opt_idx]
                    visited.add(cur)
                    path_edges.append(int(eid))
                    steps += 1
                if path_edges and cur == target and len(path_edges) <= max_steps:
                    for step_idx, eid in enumerate(path_edges):
                        if step_idx >= max_steps:
                            break
                        actions[step_idx] = int(eid)
                    lens = len(path_edges)
                    usable = lens > 0
                    break

            actions_buf.append(actions)
            lens_buf.append(lens)
            usable_buf.append(usable)

        gt_actions = torch.tensor(actions_buf, dtype=torch.long, device=device)
        gt_lens = torch.tensor(lens_buf, dtype=torch.long, device=device)
        usable_mask = torch.tensor(usable_buf, dtype=torch.bool, device=device)
        return gt_actions, gt_lens, usable_mask

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
        graph_mask: torch.Tensor | None = None,  # [B] optional bool mask for averaging
        penalty: str = "l2",
    ) -> torch.Tensor:
        """Sub-Trajectory Balance with λ=1 (uniform weights over all sub-trajectories).

        `penalty` controls the element-wise residual penalty:
          - "l2": residual^2 (default, classic SubTB)
          - "log1p_l2": log(1 + residual^2) (robust, reduces rare explosion without extra hyperparams)
        """
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
        penalty_key = str(penalty).strip().lower()
        if penalty_key in {"l2", "mse"}:
            elem = residual.pow(2)
        elif penalty_key in {"log1p_l2", "log1p"}:
            elem = torch.log1p(residual.pow(2))
        else:
            raise ValueError(f"Unsupported SubTB penalty={penalty!r}. Use 'l2' or 'log1p_l2'.")
        sq = elem * valid_f
        denom = valid_f.sum(dim=(1, 2)).clamp(min=1.0)
        per_graph = sq.sum(dim=(1, 2)) / denom
        if graph_mask is not None:
            graph_mask = graph_mask.to(device=device, dtype=torch.bool).view(-1)
            if graph_mask.numel() != num_graphs:
                raise ValueError(f"graph_mask length {graph_mask.numel()} != num_graphs {num_graphs}")
            weights = graph_mask.to(dtype=per_graph.dtype)
            denom_g = weights.sum().clamp(min=1.0)
            return (per_graph * weights).sum() / denom_g
        return per_graph.mean()

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
