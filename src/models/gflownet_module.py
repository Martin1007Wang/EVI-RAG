from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.data.components import SharedDataResources
from src.models.components import GraphEnv, GraphState, RewardOutput
from src.models.components.projections import EmbeddingProjector
from src.utils import setup_optimizer
from src.utils.logging_utils import log_metric
from src.utils.pylogger import RankedLogger

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
        evaluation_cfg: Optional[Dict[str, Any]] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
        exploration_epsilon: float = 0.1,
        eval_exploration_epsilon: float | None = None,
        policy_edge_score_bias: float = 0.0,
        policy_temperature: float = 1.0,
        log_reward_inv_temp: float = 1.0,
        gt_replay_weight: float = 1.0,
        log_z_init_bias: float = 15.0,
        lr_actor: float = 1e-4,
        lr_z: float = 1e-3,
        debug_numeric: bool = False,
        debug_numeric_freq: int = 1000,
        debug_actions: bool = False,
        debug_actions_steps: int = 1,
        use_retriever_projectors: bool = True,
        projector_checkpoint: Optional[str] = None,
        freeze_projectors: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.exploration_epsilon = float(exploration_epsilon)
        self.eval_exploration_epsilon = float(exploration_epsilon if eval_exploration_epsilon is None else eval_exploration_epsilon)
        self.policy_edge_score_bias = float(policy_edge_score_bias)
        self.policy_temperature = max(float(policy_temperature), 1e-6)
        self.log_reward_inv_temp = float(log_reward_inv_temp)
        self.gt_replay_weight = float(gt_replay_weight)
        self.log_z_init_bias = float(log_z_init_bias)
        self.lr_actor = float(lr_actor)
        self.lr_z = float(lr_z)

        self.reward_cfg = self._to_plain_dict(reward_cfg)
        self.evaluation_cfg = self._to_plain_dict(evaluation_cfg)
        self.optimizer_cfg = self._to_plain_dict(optimizer_cfg)
        self.scheduler_cfg = self._to_plain_dict(scheduler_cfg)
        self.resources_cfg = resources or {}
        self.debug_numeric = bool(debug_numeric)
        self.debug_numeric_freq = max(int(debug_numeric_freq), 1)
        self.debug_actions = bool(debug_actions)
        self.debug_actions_steps = max(int(debug_actions_steps), 0)
        self.use_retriever_projectors = bool(use_retriever_projectors)
        self.projector_checkpoint = projector_checkpoint
        self.freeze_projectors = bool(freeze_projectors)
        self._projectors_loaded = False
        self._proj_dropout = self._extract_dropout(policy_cfg)

        self.policy: nn.Module = policy_cfg if isinstance(policy_cfg, nn.Module) else hydra.utils.instantiate(policy_cfg)
        self.reward_fn: nn.Module = reward_cfg if isinstance(reward_cfg, nn.Module) else hydra.utils.instantiate(reward_cfg)
        self.env: GraphEnv = env_cfg if isinstance(env_cfg, GraphEnv) else hydra.utils.instantiate(env_cfg)
        self.max_steps = int(self.env.max_steps)
        self.bidir_token = bool(getattr(self.env, "bidir_token", False))

        self.log_z_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        if self.log_z_head[2].bias is not None:
            nn.init.constant_(self.log_z_head[2].bias, self.log_z_init_bias)
        self.log_z_condition: Optional[nn.Module] = None

        self._shared_resources: Optional[SharedDataResources] = None
        self._global_embeddings = None
        self._entity_dim: Optional[int] = None
        self._relation_dim: Optional[int] = None
        self._num_entities: Optional[int] = None
        self._num_relations: Optional[int] = None
        self._question_dim: Optional[int] = None
        self.retriever_entity_projector: Optional[EmbeddingProjector] = None
        self.retriever_relation_projector: Optional[EmbeddingProjector] = None
        self.retriever_query_projector: Optional[EmbeddingProjector] = None
        self.entity_projector: Optional[nn.Module] = None
        self.relation_projector: Optional[nn.Module] = None
        self.query_projector: Optional[nn.Module] = None

        self._init_projection_layers(dropout=self._proj_dropout)
        ignore_params = []
        if isinstance(policy_cfg, nn.Module):
            ignore_params.append("policy_cfg")
        if isinstance(reward_cfg, nn.Module):
            ignore_params.append("reward_cfg")
        if isinstance(env_cfg, nn.Module):
            ignore_params.append("env_cfg")
        self.save_hyperparameters(logger=False, ignore=ignore_params or None)

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
                    "interval": self.scheduler_cfg.get("interval", "epoch"),
                    "monitor": self.scheduler_cfg.get("monitor", "val/loss"),
                },
            }
        return optimizer

    @staticmethod
    def _to_plain_dict(config: Any) -> Dict[str, Any]:
        if isinstance(config, DictConfig):
            return dict(OmegaConf.to_container(config, resolve=True))  # type: ignore[arg-type]
        if isinstance(config, dict):
            return dict(config)
        return {}

    def setup(self, stage: str | None = None) -> None:
        self._setup_resources()
        self._setup_components()

    def on_after_backward(self) -> None:
        """梯度范数调试：按 debug_numeric_freq 将 Z 与 Actor 的梯度比写入独立日志。"""
        if not self.debug_numeric:
            return
        trainer = getattr(self, "trainer", None)
        global_step = int(getattr(trainer, "global_step", 0)) if trainer is not None else 0
        if global_step % self.debug_numeric_freq != 0:
            return

        z_params = list(self.log_z_head.parameters())
        if self.log_z_condition is not None:
            z_params += list(self.log_z_condition.parameters())
        z_norm_sq = 0.0
        for p in z_params:
            if p.grad is not None:
                g = p.grad.detach()
                z_norm_sq += float(g.norm().item() ** 2)

        actor_norm_sq = 0.0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("log_z_head") or name.startswith("log_z_condition"):
                continue
            if p.grad is not None:
                g = p.grad.detach()
                actor_norm_sq += float(g.norm().item() ** 2)

        z_norm = z_norm_sq ** 0.5
        actor_norm = actor_norm_sq ** 0.5
        ratio = z_norm / (actor_norm + 1e-8)

        debug_logger.info(
            "[GRAD_DEBUG] global_step=%d z_norm=%.4g actor_norm=%.4g ratio=%.4g",
            global_step,
            z_norm,
            actor_norm,
            ratio,
        )
        return

    def training_step(self, batch, batch_idx: int):
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        batch_size = int(batch.num_graphs)
        log_metric(self, "train/loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)
        self._log_metrics(metrics, prefix="train", batch_size=batch_size)
        self._last_debug = {
            "reward": metrics.get("reward"),
            "success": metrics.get("success"),
            "gt_path_exists": metrics.get("gt_path_exists"),
            "gt_path_f1": metrics.get("gt_path_f1"),
            "answer_reach_frac": metrics.get("answer_reach_frac"),
        }
        return loss

    def validation_step(self, batch, batch_idx: int):
        _, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        self._log_metrics(metrics, prefix="val", batch_size=int(batch.num_graphs))

    def test_step(self, batch, batch_idx: int):
        _, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        self._log_metrics(metrics, prefix="test", batch_size=int(batch.num_graphs))

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str, batch_size: int) -> None:
        sync_dist = bool(self.trainer and getattr(self.trainer, "num_devices", 1) > 1)
        is_train = prefix == "train"
        step_keys = {"reward", "success", "answer_reach_frac", "length", "avg_step_entropy"}
        prog_bar_keys = {"reward", "success", "answer_reach_frac", "length"}
        val_prog_bar_keys = {"success", "answer_f1"}
        prog_bar_set = prog_bar_keys if is_train else val_prog_bar_keys
        for name, value in metrics.items():
            if not torch.is_floating_point(value):
                value = value.float()
            scalar = value.mean()
            log_on_step = is_train and name in step_keys
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

    def _compute_batch_loss(self, batch: Any, batch_idx: int | None = None) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self.device
        self._ensure_embeddings()
        self._validate_batch(batch)

        edge_index = batch.edge_index.to(device)
        edge_batch = batch.batch[edge_index[0]].to(device)
        node_global_ids = batch.node_global_ids.to(device)
        node_ptr = batch.ptr.to(device)
        num_graphs = int(node_ptr.numel() - 1)

        edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
        edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        edge_ptr[1:] = edge_counts.cumsum(0)

        edge_labels = batch.edge_labels.to(device)
        edge_scores = batch.edge_scores.to(device)
        edge_relations = batch.edge_attr.to(device)

        path_mask = None
        if hasattr(batch, "gt_path_edge_local_ids") and batch.gt_path_edge_local_ids.numel() > 0:
            path_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
            gt_ids = batch.gt_path_edge_local_ids.to(device)
            path_mask[gt_ids] = True
        path_exists = batch.gt_path_exists.to(device)

        heads_global = node_global_ids[edge_index[0]]
        tails_global = node_global_ids[edge_index[1]]
        head_emb = self._lookup_entities(heads_global)
        tail_emb = self._lookup_entities(tails_global)
        rel_emb = self._lookup_relations(edge_relations)
        edge_features = torch.cat([head_emb, rel_emb, tail_emb], dim=-1)
        edge_tokens = self.edge_projector(edge_features)
        safe_scores = torch.nan_to_num(edge_scores, nan=0.0, posinf=0.0, neginf=0.0)
        score_std = self._standardize_scores_flat(safe_scores, edge_batch)

        question_tokens = self._prepare_question_tokens(batch.question_emb, batch_size=num_graphs, device=device)

        rollout = self._rollout(
            edge_tokens,
            question_tokens,
            batch,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
            node_ptr=node_ptr,
            score_std=score_std,
            safe_edge_scores=safe_scores,
            path_mask=path_mask if path_mask is not None and path_mask.any() else None,
            path_exists=path_exists,
            batch_idx=batch_idx,
        )

        reward_out: RewardOutput = self.reward_fn(
            selected_mask=rollout["selected_mask"],
            edge_labels=edge_labels,
            edge_scores=edge_scores,
            edge_batch=edge_batch,
            edge_heads=heads_global,
            edge_tails=tails_global,
            answer_entity_ids=batch.answer_entity_ids.to(device),
            answer_ptr=batch._slice_dict["answer_entity_ids"].to(device),
            path_mask=path_mask,
            path_exists=path_exists,
            reach_success=rollout["reach_success"],
            reach_fraction=self._compute_answer_reach_fraction(
                selected_mask=rollout["selected_mask"],
                edge_batch=edge_batch,
                edge_heads=heads_global,
                edge_tails=tails_global,
                answer_entity_ids=batch.answer_entity_ids.to(device),
                answer_ptr=batch._slice_dict["answer_entity_ids"].to(device),
            ),
        )

        log_reward = torch.log(reward_out.reward.clamp(min=1e-6)) * self.log_reward_inv_temp
        log_z = self.log_z_head(rollout["log_z_context"]).squeeze(-1)

        length_int = rollout["length"].detach().clamp(min=0)
        log_pb = torch.zeros_like(length_int)
        self._assert_finite(log_reward, "log_reward")
        self._assert_finite(log_z, "log_z")
        self._assert_finite(rollout["log_pf"], "log_pf")
        tb_loss = torch.mean((log_z + rollout["log_pf"] - log_pb - log_reward) ** 2)
        loss = tb_loss

        gt_log_pf = None
        gt_loss_value = None
        if self.gt_replay_weight > 0 and hasattr(batch, "gt_path_edge_local_ids") and batch.gt_path_edge_local_ids.numel() > 0:
            gt_log_pf = self._compute_gt_log_pf(
                edge_tokens=edge_tokens,
                question_tokens=question_tokens,
                batch=batch,
                edge_batch=edge_batch,
                edge_ptr=edge_ptr,
                score_bias=score_std,
                path_exists=path_exists,
                heads_global=heads_global,
                tails_global=tails_global,
            )
            mask_gt = path_exists.bool() if path_exists is not None else torch.zeros_like(length_int, dtype=torch.bool)
            if mask_gt.any():
                gt_loss_value = -gt_log_pf[mask_gt].mean()
                loss = loss + self.gt_replay_weight * gt_loss_value
        else:
            gt_log_pf = None

        length_safe = rollout["length"].detach().clamp(min=1.0)
        avg_step_entropy = (-rollout["log_pf"].detach() / length_safe).mean()

        metrics: Dict[str, torch.Tensor] = {
            **reward_out.as_dict(),
            "length": rollout["length"].detach(),
            "path_exists_ratio": reward_out.path_exists.detach().float().mean() if hasattr(reward_out, "path_exists") else torch.tensor(0.0, device=device),
            "tb_loss": tb_loss.detach(),
            "avg_step_entropy": avg_step_entropy,
        }

        # Debug: reward / reach_fraction / success 关系 + reach_fraction 对拍
        if self.debug_numeric and batch_idx is not None and batch_idx % self.debug_numeric_freq == 0:
            try:
                reward_vals = reward_out.reward.detach()
                success_vals = reward_out.success.detach()
                answer_frac = reward_out.answer_reach_frac.detach()
                rollout_frac = rollout["reach_fraction"].detach()
                length_vals = rollout["length"].detach()
                path_exists_vals = reward_out.path_exists.detach().float() if hasattr(reward_out, "path_exists") else torch.zeros_like(success_vals)

                debug_logger.info(
                    "[REWARD_DEBUG] batch=%d reward(min/mean/max)=%.4g/%.4g/%.4g "
                    "success(mean)=%.4g answer_reach_frac(mean)=%.4g rollout_reach_frac(mean)=%.4g "
                    "length(mean)=%.4g path_exists(mean)=%.4g",
                    batch_idx,
                    reward_vals.min().item(), reward_vals.mean().item(), reward_vals.max().item(),
                    success_vals.mean().item(),
                    answer_frac.mean().item(),
                    rollout_frac.mean().item(),
                    length_vals.mean().item(),
                    path_exists_vals.mean().item(),
                )

                # 小样本对拍 reach_fraction：impl vs naive（最多前 3 个图）
                num_graphs = int(batch.ptr.numel() - 1)
                num_check = min(3, num_graphs)
                edge_batch_debug = edge_batch
                heads_debug = heads_global
                tails_debug = tails_global
                selected_mask_debug = rollout["selected_mask"].detach()
                success_debug = reward_out.success.detach()
                answer_ids_debug = batch.answer_entity_ids.to(device)
                answer_ptr_debug = batch._slice_dict["answer_entity_ids"].to(device)

                frac_impl = self._compute_answer_reach_fraction(
                    selected_mask=selected_mask_debug,
                    edge_batch=edge_batch_debug,
                    edge_heads=heads_debug,
                    edge_tails=tails_debug,
                    answer_entity_ids=answer_ids_debug,
                    answer_ptr=answer_ptr_debug,
                )

                edge_ptr_debug = rollout.get("edge_ptr") if isinstance(rollout, dict) else None
                if edge_ptr_debug is None:
                    edge_ptr_debug = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
                    edge_counts_debug = torch.bincount(edge_batch_debug, minlength=num_graphs)
                    edge_ptr_debug[1:] = edge_counts_debug.cumsum(0)

                for g in range(num_check):
                    f_impl = frac_impl[g].item()

                    es, ee = int(edge_ptr_debug[g].item()), int(edge_ptr_debug[g + 1].item())
                    mask_g = (edge_batch_debug == g)
                    selected_g = selected_mask_debug[mask_g]
                    heads_g = heads_debug[mask_g]
                    tails_g = tails_debug[mask_g]

                    a_start, a_end = int(answer_ptr_debug[g].item()), int(answer_ptr_debug[g + 1].item())
                    answers_g = answer_ids_debug[a_start:a_end]
                    if answers_g.numel() == 0:
                        continue

                    selected_nodes = torch.cat([heads_g[selected_g], tails_g[selected_g]], dim=0)
                    if selected_nodes.numel() == 0:
                        f_naive = 0.0
                    else:
                        hits = []
                        for a in answers_g.tolist():
                            hits.append(int((selected_nodes == a).any().item()))
                        f_naive = sum(hits) / max(len(hits), 1)

                    success_g = float(success_debug[g].item()) if success_debug.numel() > g else float("nan")
                    debug_logger.info(
                        "[REACH_CHECK] batch=%d graph=%d success=%.4g frac_impl=%.4g frac_naive=%.4g answers=%d",
                        batch_idx, g, success_g, f_impl, f_naive, answers_g.numel(),
                    )

                # 统计 success 与 frac_impl 的不一致情况，便于快速定位比例
                if frac_impl.numel() == success_debug.numel():
                    mismatch_pos = ((success_debug == 0) & (frac_impl > 0)).sum().item()
                    mismatch_neg = ((success_debug > 0) & (frac_impl == 0)).sum().item()
                    debug_logger.info(
                        "[SUCCESS_MISMATCH] batch=%d mismatch_pos=%d mismatch_neg=%d total=%d",
                        batch_idx,
                        int(mismatch_pos),
                        int(mismatch_neg),
                        int(success_debug.numel()),
                    )
            except Exception as exc:  # pragma: no cover - debug 期间不抛训练
                debug_logger.error("[REWARD_DEBUG_ERROR] batch=%s exc=%s", str(batch_idx), exc)

        if gt_log_pf is not None:
            metrics["gt_log_pf"] = gt_log_pf.detach()
            mask_gt = path_exists.bool() if path_exists is not None else torch.zeros_like(length_int, dtype=torch.bool)
            if mask_gt.any() and self.gt_replay_weight > 0:
                if gt_loss_value is None:
                    gt_loss_value = -gt_log_pf[mask_gt].mean()
                metrics["gt_replay_loss"] = gt_loss_value.detach()

        if not self.training:
            extra_eval = self._compute_multi_path_metrics(
                edge_tokens,
                question_tokens,
                batch,
                edge_mask=None,
                path_mask=path_mask,
                path_exists=path_exists,
            )
            for name, tensor in extra_eval.items():
                metrics[name] = tensor.detach()
            if gt_log_pf is None and path_exists.any() and hasattr(batch, "gt_path_edge_local_ids") and batch.gt_path_edge_local_ids.numel() > 0:
                gt_log_pf = self._compute_gt_log_pf(
                    edge_tokens=edge_tokens,
                    question_tokens=question_tokens,
                    batch=batch,
                    edge_batch=edge_batch,
                    edge_ptr=edge_ptr,
                    score_bias=score_std,
                    path_exists=path_exists,
                    heads_global=heads_global,
                    tails_global=tails_global,
                )
                metrics["gt_log_pf"] = gt_log_pf.detach()

        if self.debug_numeric:
            split = "train" if self.training else "eval"
            debug_logger.info(
                f"[NUMERIC] {split} batch={batch_idx} "
                f"reward(min/mean/max)=({reward_out.reward.min().item():.4g}/"
                f"{reward_out.reward.mean().item():.4g}/"
                f"{reward_out.reward.max().item():.4g}) "
                f"log_z(min/mean/max)=({log_z.min().item():.4g}/"
                f"{log_z.mean().item():.4g}/"
                f"{log_z.max().item():.4g}) "
                f"log_pf(min/mean/max)=({rollout['log_pf'].min().item():.4g}/"
                f"{rollout['log_pf'].mean().item():.4g}/"
                f"{rollout['log_pf'].max().item():.4g}) "
                f"tb_loss={tb_loss.item():.4g}"
            )
        return loss, metrics

    def _rollout(
        self,
        edge_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        batch: Any,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        score_std: torch.Tensor,
        safe_edge_scores: torch.Tensor,
        path_mask: Optional[torch.Tensor],
        path_exists: Optional[torch.Tensor],
        batch_idx: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        device = edge_tokens.device
        num_graphs = int(node_ptr.numel() - 1)
        # PyG slice sanity: pointers must be per-field, not cross-field reused.
        assert batch._slice_dict["answer_entity_ids"].shape == batch._slice_dict["answer_node_locals"].shape, (
            "answer_entity_ids ptr must align with answer_node_locals ptr"
        )
        if batch.gt_path_edge_local_ids.numel() > 0:
            max_gt = int(batch.gt_path_edge_local_ids.max().item())
            assert max_gt < batch.edge_index.size(1), "gt_path_edge_local_ids exceeds edge_index size"
        graph_dict = {
            "edge_index": batch.edge_index.to(device),
            "edge_batch": edge_batch,
            "node_global_ids": batch.node_global_ids.to(device),
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
            "edge_scores": batch.edge_scores.to(device),
            "edge_labels": batch.edge_labels.to(device),
            "top_edge_mask": batch.top_edge_mask.to(device),
            "gt_path_edge_local_ids": batch.gt_path_edge_local_ids.to(device),
            "gt_edge_ptr": batch._slice_dict["gt_path_edge_local_ids"].to(device),
            "gt_path_exists": batch.gt_path_exists.to(device),
            "is_answer_reachable": batch.is_answer_reachable.to(device),
        }
        state = self.env.reset(graph_dict, device=device)

        log_pf = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        log_z_context: Optional[torch.Tensor] = None

        edge_logit_bias = self.policy_edge_score_bias * score_std
        debug_logged = False

        for step in range(self.max_steps + 1):
            if state.done.all():
                break

            action_mask_edges = self.env.action_mask_edges(state)
            edge_logits, stop_logits, cls_out = self.policy(
                edge_tokens,
                question_tokens,
                edge_batch,
                state.selected_mask,
                edge_heads=batch.node_global_ids[batch.edge_index[0]].to(device),
                edge_tails=batch.node_global_ids[batch.edge_index[1]].to(device),
                current_tail=state.current_tail,
            )

            if (
                self.debug_actions
                and self.debug_actions_steps > 0
                and step < self.debug_actions_steps
                and (batch_idx is None or batch_idx == 0)
                and not debug_logged
            ):
                g_idx = 0
                start_idx = int(edge_ptr[g_idx].item())
                end_idx = int(edge_ptr[g_idx + 1].item())
                edge_count = end_idx - start_idx
                start_ptr = graph_dict["start_ptr"]
                s0, s1 = int(start_ptr[g_idx].item()), int(start_ptr[g_idx + 1].item())
                start_nodes = graph_dict["start_node_locals"][s0:s1].detach().cpu().tolist()
                current_tail = int(state.current_tail[g_idx].item()) if state.current_tail.numel() > g_idx else -1
                debug_logger.info(
                    "[DEBUG_ACTION] g0 step=%d edges=%d starts_local=%s current_tail=%d",
                    step,
                    edge_count,
                    start_nodes,
                    current_tail,
                )
                if edge_count > 0:
                    mask_slice = action_mask_edges[start_idx:end_idx]
                    local_logits = edge_logits[start_idx:end_idx]
                    local_bias = edge_logit_bias[start_idx:end_idx]
                    local_scores = safe_edge_scores[start_idx:end_idx]
                    if mask_slice.any():
                        masked_logits = local_logits.masked_fill(~mask_slice, float("-inf"))
                        top_local_idx = int(torch.argmax(masked_logits).item())
                        edge_id = start_idx + top_local_idx
                        head_id = int(batch.node_global_ids[batch.edge_index[0, edge_id]].item())
                        tail_id = int(batch.node_global_ids[batch.edge_index[1, edge_id]].item())
                        debug_logger.info(
                            "[DEBUG_ACTION] g0 top_edge=%d head=%d tail=%d logit=%.4g bias=%.4g score=%.4g masked=%s",
                            edge_id,
                            head_id,
                            tail_id,
                            float(local_logits[top_local_idx].item()),
                            float(local_bias[top_local_idx].item()),
                            float(local_scores[top_local_idx].item()),
                            bool(mask_slice[top_local_idx].item()),
                        )
                    else:
                        debug_logger.info("[DEBUG_ACTION] g0 has no valid edges under mask at step %d", step)
                debug_logged = True

            if log_z_context is None:
                start_ptr = batch._slice_dict["start_entity_ids"].to(device)
                max_start = int((start_ptr[1:] - start_ptr[:-1]).max().item()) if start_ptr.numel() > 1 else 0
                start_entity_ids = torch.full((num_graphs, max_start), -1, dtype=torch.long, device=device) if max_start > 0 else torch.empty(num_graphs, 0, dtype=torch.long, device=device)
                start_entity_mask = torch.zeros_like(start_entity_ids, dtype=torch.bool)
                for g in range(num_graphs):
                    s, e = int(start_ptr[g].item()), int(start_ptr[g + 1].item())
                    if s == e:
                        continue
                    vals = batch.start_entity_ids[s:e].to(device)
                    l = min(max_start, vals.numel())
                    start_entity_ids[g, :l] = vals[:l]
                    start_entity_mask[g, :l] = True
                log_z_context = self._build_log_z_context(
                    cls_out,
                    start_entity_ids,
                    start_entity_mask if start_entity_ids.numel() > 0 else torch.zeros(num_graphs, 0, dtype=torch.bool, device=device),
                    question_tokens,
                )

            actions = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
            log_pf_vec = torch.zeros(num_graphs, dtype=torch.float32, device=device)

            epsilon = self.exploration_epsilon if self.training else self.eval_exploration_epsilon
            for g in range(num_graphs):
                es, ee = int(edge_ptr[g].item()), int(edge_ptr[g + 1].item())
                if es == ee or state.done[g]:
                    actions[g] = ee
                    continue
                mask_slice = action_mask_edges[es:ee]
                if not mask_slice.any():
                    actions[g] = ee
                    continue
                logits_slice = edge_logits[es:ee] + edge_logit_bias[es:ee]
                stop_logit = stop_logits[g]
                # 只对合法动作分布混入 epsilon，非法动作概率为 0
                valid_mask = torch.cat([mask_slice, torch.tensor([True], device=device)])  # stop 一直合法
                logits_slice = logits_slice.masked_fill(~mask_slice, -1e9)
                cat_logits = torch.cat([logits_slice, stop_logit.view(1)])
                cat_logits = cat_logits.masked_fill(~valid_mask, -1e9)
                if self.policy_temperature != 1.0:
                    cat_logits = cat_logits / self.policy_temperature
                probs = torch.softmax(cat_logits, dim=0)
                if epsilon and epsilon > 0:
                    uniform = valid_mask.float()
                    uniform = uniform / uniform.sum().clamp(min=1.0)
                    probs = (1 - epsilon) * probs + epsilon * uniform
                action = torch.distributions.Categorical(probs=probs).sample()
                log_pf_vec[g] = torch.log(probs[action].clamp(min=torch.finfo(probs.dtype).eps))
                if action == logits_slice.numel():
                    actions[g] = ee
                else:
                    actions[g] = es + action

            log_pf = log_pf + log_pf_vec
            state = self.env.step(state, actions, step_index=step)

        reach_success = state.answer_hits.float()
        reach_fraction = reach_success
        length = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        for g in range(num_graphs):
            es, ee = int(edge_ptr[g].item()), int(edge_ptr[g + 1].item())
            length[g] = state.selected_mask[es:ee].sum().float()

        if log_z_context is None:
            raise RuntimeError("log_z_context was not set during rollout.")

        return {
            "log_pf": log_pf,
            "log_z_context": log_z_context,
            "selected_mask": state.selected_mask,
            "selection_order": state.selection_order,
            "actions": actions,
            "reach_fraction": reach_fraction,
            "reach_success": reach_success.float(),
            "length": length,
        }

    def _compute_gt_log_pf(
        self,
        *,
        edge_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        batch: Any,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
        score_bias: torch.Tensor,
        path_exists: torch.Tensor,
        heads_global: torch.Tensor,
        tails_global: torch.Tensor,
    ) -> torch.Tensor:
        """Off-policy 计算 GT 路径的 log_pf（仅在 eval 调用，避免训练期开销）。"""
        device = edge_tokens.device
        num_graphs = int(edge_ptr.numel() - 1)
        gt_edges = batch.gt_path_edge_local_ids.to(device)
        gt_ptr = batch._slice_dict["gt_path_edge_local_ids"].to(device)
        if gt_edges.numel() == 0 or gt_ptr.numel() != num_graphs + 1:
            return torch.zeros(num_graphs, device=device)

        selected_mask = torch.zeros_like(edge_batch, dtype=torch.bool, device=device)
        edge_logits, _, _ = self.policy(
            edge_tokens,
            question_tokens,
            edge_batch,
            selected_mask,
            edge_heads=heads_global,
            edge_tails=tails_global,
            current_tail=None,
        )
        if self.policy_edge_score_bias != 0.0:
            edge_logits = edge_logits + self.policy_edge_score_bias * score_bias

        log_probs = torch.empty_like(edge_logits)
        for g in range(num_graphs):
            es, ee = int(edge_ptr[g].item()), int(edge_ptr[g + 1].item())
            if es == ee:
                continue
            log_probs[es:ee] = torch.log_softmax(edge_logits[es:ee], dim=0)

        log_pf = torch.zeros(num_graphs, device=device)
        for g in range(num_graphs):
            if not path_exists[g]:
                continue
            gs, ge = int(gt_ptr[g].item()), int(gt_ptr[g + 1].item())
            if gs == ge:
                continue
            local_gt_indices = gt_edges[gs:ge]
            log_pf[g] = log_probs[local_gt_indices].sum()
        return log_pf

    def _prepare_question_tokens(self, question_emb: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if question_emb is None or question_emb.numel() == 0:
            raise ValueError(
                "question_emb is missing or empty; g_agent cache must provide question embeddings for every graph."
            )
        if question_emb.dim() == 1:
            if question_emb.numel() % max(batch_size, 1) != 0:
                raise ValueError(f"question_emb flatten length {question_emb.numel()} not divisible by batch_size={batch_size}")
            dim = question_emb.numel() // max(batch_size, 1)
            question_emb = question_emb.view(batch_size, dim)
        elif question_emb.dim() == 2:
            if question_emb.size(0) == 1 and batch_size > 1:
                question_emb = question_emb.expand(batch_size, -1)
            elif question_emb.size(0) != batch_size:
                raise ValueError(f"question_emb batch mismatch: {question_emb.size(0)} vs {batch_size}")
        else:
            raise ValueError(f"Unsupported question_emb rank={question_emb.dim()}")

        tokens = question_emb.to(device)
        if self._question_dim is None:
            self._question_dim = int(tokens.size(1))
        elif int(tokens.size(1)) != self._question_dim:
            raise ValueError(f"Inconsistent question_dim: current {tokens.size(1)} vs expected {self._question_dim}")
        if self.retriever_query_projector is not None:
            tokens = self.retriever_query_projector(tokens)
        tokens = self.query_projector(tokens)
        return tokens

    def _standardize_scores_flat(self, scores: torch.Tensor, edge_batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(edge_batch.max().item()) + 1 if edge_batch.numel() > 0 else 0
        mean = torch.zeros(num_graphs, device=scores.device)
        var = torch.zeros(num_graphs, device=scores.device)
        counts = torch.bincount(edge_batch, minlength=num_graphs).clamp(min=1).float()
        mean.scatter_add_(0, edge_batch, scores)
        mean = mean / counts
        diff = scores - mean[edge_batch]
        var.scatter_add_(0, edge_batch, diff * diff)
        var = var / counts
        std = torch.sqrt(var).clamp(min=1e-6)
        standardized = (scores - mean[edge_batch]) / std[edge_batch]
        return standardized

    def _validate_batch(self, batch: Any) -> None:
        num_graphs = int(batch.ptr.numel() - 1)
        num_edges = int(batch.edge_index.size(1))
        required_fields = (
            "start_node_locals",
            "answer_node_locals",
            "start_entity_ids",
            "answer_entity_ids",
            "top_edge_mask",
            "gt_path_edge_local_ids",
            "gt_path_exists",
            "is_answer_reachable",
        )
        for field in required_fields:
            if not hasattr(batch, field):
                raise ValueError(f"Batch missing required field {field}; g_agent cache must materialize it.")
        if not hasattr(batch, "question_emb"):
            raise ValueError("Batch missing question_emb; g_agent cache must store question embeddings for each sample.")
        start_ptr = batch._slice_dict.get("start_node_locals")
        if start_ptr is None:
            raise ValueError("Batch missing start_node_locals slice info; PyG collate may be broken.")
        per_graph_start = start_ptr[1:] - start_ptr[:-1]
        if per_graph_start.numel() != num_graphs:
            raise ValueError("start_ptr length mismatch; expected one count per graph.")
        if (per_graph_start <= 0).any():
            missing = torch.nonzero(per_graph_start <= 0, as_tuple=False).view(-1).cpu().tolist()
            raise ValueError(
                f"Batch contains graphs without start_node_locals at indices {missing}; "
                "g_agent cache must provide non-empty start anchors per graph."
            )
        start_entity_ptr = batch._slice_dict.get("start_entity_ids")
        if start_entity_ptr is None:
            raise ValueError("Batch missing start_entity_ids slice info; g_agent cache may be corrupt.")
        per_graph_start_entities = start_entity_ptr[1:] - start_entity_ptr[:-1]
        if per_graph_start_entities.numel() != num_graphs:
            raise ValueError("start_entity_ptr length mismatch; expected one count per graph.")
        if (per_graph_start_entities <= 0).any():
            missing = torch.nonzero(per_graph_start_entities <= 0, as_tuple=False).view(-1).cpu().tolist()
            raise ValueError(
                f"Batch contains graphs without start_entity_ids at indices {missing}; "
                "g_agent cache must preserve the seed entities instead of relying on runtime defaults."
            )
        answer_entity_ptr = batch._slice_dict.get("answer_entity_ids")
        if answer_entity_ptr is None:
            raise ValueError("Batch missing answer_entity_ids slice info; g_agent cache may be corrupt.")
        if answer_entity_ptr.numel() != num_graphs + 1:
            raise ValueError("answer_entity_ptr length mismatch; expected one offset per graph.")
        if (answer_entity_ptr[1:] - answer_entity_ptr[:-1] <= 0).any():
            missing = torch.nonzero((answer_entity_ptr[1:] - answer_entity_ptr[:-1]) <= 0, as_tuple=False).view(-1).cpu().tolist()
            raise ValueError(
                f"Batch contains graphs without answer_entity_ids at indices {missing}; "
                "g_agent cache must include supervised answer anchors."
            )
        if "gt_path_edge_local_ids" not in batch._slice_dict:
            raise ValueError("Batch missing gt_path_edge_local_ids slice info; ensure g_agent cache persisted gt paths.")
        if hasattr(batch, "edge_scores") and batch.edge_scores.numel() != num_edges:
            raise ValueError(f"edge_scores numel {batch.edge_scores.numel()} != num_edges {num_edges}")
        if hasattr(batch, "edge_labels") and batch.edge_labels.numel() != num_edges:
            raise ValueError(f"edge_labels numel {batch.edge_labels.numel()} != num_edges {num_edges}")
        if hasattr(batch, "edge_attr") and batch.edge_attr.numel() != num_edges:
            raise ValueError(f"edge_attr numel {batch.edge_attr.numel()} != num_edges {num_edges}")
        if hasattr(batch, "top_edge_mask") and batch.top_edge_mask.numel() != num_edges:
            raise ValueError(f"top_edge_mask numel {batch.top_edge_mask.numel()} != num_edges {num_edges}")
        if hasattr(batch, "question_emb"):
            q = batch.question_emb
            if q.dim() == 1:
                if q.numel() % max(num_graphs, 1) != 0:
                    raise ValueError(f"question_emb numel {q.numel()} not divisible by num_graphs {num_graphs}")
            elif q.dim() == 2:
                if q.size(0) not in (1, num_graphs):
                    raise ValueError(f"question_emb first dim {q.size(0)} incompatible with num_graphs {num_graphs}")
            else:
                raise ValueError(f"Unsupported question_emb rank={q.dim()}")

    def _compute_answer_reach_fraction(
        self,
        *,
        selected_mask: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_heads: torch.Tensor,
        edge_tails: torch.Tensor,
        answer_entity_ids: torch.Tensor,
        answer_ptr: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(answer_ptr.numel() - 1)
        if num_graphs == 0:
            return torch.zeros(0, device=selected_mask.device)
        edge_nodes = torch.cat([edge_heads, edge_tails], dim=0)
        selected_twice = torch.cat([selected_mask, selected_mask], dim=0)
        selected_nodes = torch.where(
            selected_twice,
            edge_nodes,
            torch.full_like(edge_nodes, -1),
        )
        edge_batch_twice = torch.cat([edge_batch, edge_batch], dim=0)
        frac = torch.zeros(num_graphs, device=selected_mask.device)
        for g in range(num_graphs):
            a_start, a_end = int(answer_ptr[g].item()), int(answer_ptr[g + 1].item())
            if a_start == a_end:
                continue
            answers = answer_entity_ids[a_start:a_end]
            mask_g = edge_batch_twice == g
            selected_g = selected_nodes[mask_g]
            if selected_g.numel() == 0:
                continue
            hits = (selected_g.unsqueeze(1) == answers).any(dim=0).float().sum()
            frac[g] = hits / float(answers.numel())
        return frac

    def _assert_finite(self, tensor: torch.Tensor, name: str) -> None:
        if not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).sum().item()
            raise ValueError(f"{name} contains {bad} non-finite values.")

    def _setup_resources(self):
        if self._shared_resources is not None:
            return
        self._shared_resources = SharedDataResources(**self.resources_cfg)
        self._global_embeddings = self._shared_resources.global_embeddings
        self._entity_dim = self._global_embeddings.entity_embeddings.size(1)
        self._relation_dim = self._global_embeddings.relation_embeddings.size(1)
        self._num_entities = self._global_embeddings.entity_embeddings.size(0)
        self._num_relations = self._global_embeddings.relation_embeddings.size(0)

    def _ensure_embeddings(self) -> None:
        if self._shared_resources is None or self._global_embeddings is None:
            self._setup_resources()
        if self.entity_projector is None or self.relation_projector is None or self.query_projector is None or self.edge_projector is None:
            self._init_projection_layers(dropout=self._proj_dropout)

    def _load_projector_checkpoint(self) -> None:
        if self.projector_checkpoint is None:
            raise ValueError("use_retriever_projectors=True 但未提供 projector_checkpoint。")
        ckpt_path = Path(self.projector_checkpoint).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"projector_checkpoint 不存在: {ckpt_path}")
        try:
            import collections
            import functools
            import typing
            from torch.serialization import add_safe_globals
            import omegaconf
            from omegaconf.base import ContainerMetadata, Metadata
            from omegaconf.nodes import AnyNode
            from src.losses.retriever_loss import RetrieverBCELoss

            add_safe_globals(
                [
                    omegaconf.listconfig.ListConfig,
                    omegaconf.dictconfig.DictConfig,
                    ContainerMetadata,
                    Metadata,
                    AnyNode,
                    typing.Any,
                    list,
                    dict,
                    tuple,
                    set,
                    frozenset,
                    type(None),
                    str,
                    int,
                    float,
                    bool,
                    collections.defaultdict,
                    functools.partial,
                    torch.optim.AdamW,
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    RetrieverBCELoss,
                ]
            )
        except Exception as exc:  # pragma: no cover - 仅用于兼容老 PyTorch
            logger.warning("注册 checkpoint safe globals 失败，尝试继续加载: %s", exc)

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.retriever_entity_projector = self._load_single_projector(
            state_dict,
            name="entity_proj",
            prefixes=[
                "model._orig_mod.entity_proj.network.0",
                "model.entity_proj.network.0",
                "entity_proj.network.0",
            ],
        )
        self.retriever_relation_projector = self._load_single_projector(
            state_dict,
            name="relation_proj",
            prefixes=[
                "model._orig_mod.relation_proj.network.0",
                "model.relation_proj.network.0",
                "relation_proj.network.0",
            ],
        )
        self.retriever_query_projector = self._load_single_projector(
            state_dict,
            name="query_proj",
            prefixes=[
                "model._orig_mod.query_proj.network.0",
                "model.query_proj.network.0",
                "query_proj.network.0",
            ],
        )
        self._projectors_loaded = True

    def _load_single_projector(self, state_dict: Dict[str, torch.Tensor], *, name: str, prefixes: list[str]) -> EmbeddingProjector:
        weight_key = None
        bias_key = None
        for prefix in prefixes:
            candidate_weight = f"{prefix}.weight"
            candidate_bias = f"{prefix}.bias"
            if candidate_weight in state_dict and candidate_bias in state_dict:
                weight_key, bias_key = candidate_weight, candidate_bias
                break
        if weight_key is None or bias_key is None:
            raise KeyError(f"在 checkpoint 中未找到 {name} 权重，prefixes={prefixes}")
        weight = state_dict[weight_key]
        bias = state_dict[bias_key]
        projector = EmbeddingProjector(output_dim=int(weight.shape[0]), finetune=False)
        load_state = {"network.0.weight": weight, "network.0.bias": bias}
        missing, unexpected = projector.load_state_dict(load_state, strict=False)
        if missing:
            raise RuntimeError(f"{name} projector 缺失权重: {missing}")
        if unexpected and any("network.0" not in key for key in unexpected):
            raise RuntimeError(f"{name} projector 出现多余键: {unexpected}")
        projector.eval()
        return projector

    def _freeze_retriever_projectors(self) -> None:
        for module in [self.retriever_entity_projector, self.retriever_relation_projector, self.retriever_query_projector]:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    def _setup_components(self):
        self._ensure_embeddings()
        if self.use_retriever_projectors and not self._projectors_loaded:
            self._load_projector_checkpoint()
        if self.freeze_projectors:
            self._freeze_retriever_projectors()
        self.log_z_condition = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _init_projection_layers(self, dropout: float) -> None:
        self.entity_projector = self._build_projection_head(dropout)
        self.relation_projector = self._build_projection_head(dropout)
        self.query_projector = self._build_projection_head(dropout)
        self.edge_projector = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

    def _build_projection_head(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _lookup_entities(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.clamp(min=0, max=self._num_entities - 1)
        embeddings = self._global_embeddings.get_entity_embeddings(ids)
        if self.retriever_entity_projector is not None:
            embeddings = self.retriever_entity_projector(embeddings)
        return self.entity_projector(embeddings)

    def _lookup_relations(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.clamp(min=0, max=self._num_relations - 1)
        embeddings = self._global_embeddings.get_relation_embeddings(ids)
        if self.retriever_relation_projector is not None:
            embeddings = self.retriever_relation_projector(embeddings)
        return self.relation_projector(embeddings)

    def _build_log_z_context(
        self,
        cls_out: torch.Tensor,
        start_entity_ids: torch.Tensor,
        start_entity_mask: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        start_summary = self._aggregate_start_entities(start_entity_ids, start_entity_mask)
        conditioned = self.log_z_condition(start_summary) if self.log_z_condition is not None else start_summary
        return cls_out + conditioned + question_tokens

    def _aggregate_start_entities(self, start_entity_ids: torch.Tensor, start_entity_mask: torch.Tensor) -> torch.Tensor:
        device = start_entity_ids.device
        if start_entity_ids.numel() == 0:
            return torch.zeros(start_entity_ids.size(0), self.hidden_dim, device=device)
        safe_ids = torch.where(start_entity_mask, start_entity_ids, torch.zeros_like(start_entity_ids))
        flat = safe_ids.view(-1)
        projected = self._lookup_entities(flat).view(*start_entity_ids.shape, -1)
        weights = start_entity_mask.float().unsqueeze(-1)
        total = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (projected * weights).sum(dim=1) / total
        return pooled

    def _gather_entity_embeddings(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num_entities = int(self._num_entities or self._global_embeddings.entity_embeddings.size(0))
        max_id = max(num_entities - 1, 0)
        safe_ids = torch.where(mask, ids, torch.zeros_like(ids))
        safe_ids = safe_ids.clamp(min=0, max=max_id)
        flat = safe_ids.view(-1)
        embeddings = self._global_embeddings.get_entity_embeddings(flat)
        embeddings = embeddings.view(*ids.shape, -1)
        return embeddings

    @staticmethod
    def _extract_dropout(policy_cfg: Any) -> float:
        if isinstance(policy_cfg, DictConfig):
            try:
                return float(policy_cfg.get("dropout", 0.0))
            except Exception:
                return 0.0
        if isinstance(policy_cfg, nn.Module):
            return float(getattr(policy_cfg, "dropout", 0.0) or 0.0)
        return 0.0

    def _edge_ranks_flat(self, edge_scores: torch.Tensor, edge_batch: torch.Tensor, path_mask: torch.Tensor, num_graphs: int) -> torch.Tensor:
        device = edge_scores.device
        ranks = torch.full_like(edge_scores, fill_value=-1.0)
        for g in range(num_graphs):
            mask = (edge_batch == g)
            if not mask.any():
                continue
            scores_g = torch.where(path_mask[mask], edge_scores[mask], torch.full_like(edge_scores[mask], -1e9))
            order = scores_g.argsort(descending=True)
            rank = torch.full_like(scores_g, fill_value=float(scores_g.numel() + 1))
            rank[order] = torch.arange(scores_g.numel(), device=device, dtype=torch.float32)
            ranks[mask] = rank
        return ranks

    def _compute_multi_path_metrics(
        self,
        edge_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        batch: Any,
        edge_mask: Optional[torch.Tensor],
        path_mask: Optional[torch.Tensor],
        path_exists: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if path_mask is None or path_exists is None or not path_mask.any():
            zeros = torch.zeros(1, device=edge_tokens.device)
            return {"vpr": zeros, "path_recall@1": zeros, "wlrr": zeros}
        device = edge_tokens.device
        num_graphs = int(batch.ptr.numel() - 1)
        if path_mask.numel() != edge_tokens.size(0):
            raise ValueError(f"path_mask length {path_mask.numel()} != num_edges {edge_tokens.size(0)}")
        if path_exists.numel() != num_graphs:
            raise ValueError(f"path_exists length {path_exists.numel()} != num_graphs {num_graphs}")
        scores = batch.edge_scores.to(device).float()
        edge_batch = batch.batch[batch.edge_index[0]].to(device)
        ranks = self._edge_ranks_flat(scores, edge_batch, path_mask, num_graphs)
        k = 1
        vpr = torch.zeros(1, device=device)
        path_recall_at_k = torch.zeros(1, device=device)
        wlrr = torch.zeros(1, device=device)
        for g in range(num_graphs):
            mask = (edge_batch == g)
            if not mask.any():
                continue
            ranks_g = ranks[mask]
            path_mask_g = path_mask[mask]
            if not path_mask_g.any():
                continue
            path_ranks = ranks_g[path_mask_g]
            path_recall_at_k += (path_ranks < k).float().mean()
            vpr += (path_ranks < 5).float().mean()
            hard = path_ranks > 10
            if hard.any():
                wlrr += (path_ranks[hard] < 1000).float().mean()
        denom = max(num_graphs, 1)
        return {
            "vpr": vpr / denom,
            f"path_recall@{k}": path_recall_at_k / denom,
            "wlrr": wlrr / denom,
        }


__all__ = ["GFlowNetModule"]
