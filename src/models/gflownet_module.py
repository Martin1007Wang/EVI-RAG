from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from lightning import LightningModule
from torch import nn

from src.models.components import (
    GFlowNetActor,
    GFlowNetEstimator,
    GraphEmbedder,
    GraphEnv,
    StateEncoder,
    RewardOutput,
)
from src.models.components.gflownet_env import STOP_RELATION, DIRECTION_FORWARD, DIRECTION_BACKWARD
from src.utils import setup_optimizer
from src.utils.graph_utils import POS_LABEL_THRESHOLD
from src.utils.logging_utils import log_metric

_ZERO = 0
_ONE = 1


class GFlowNetModule(LightningModule):
    """扁平 PyG batch 版本的 GFlowNet，移除 dense padding。"""

    def __init__(
        self,
        *,
        hidden_dim: int,
        policy_cfg: DictConfig,
        reward_cfg: DictConfig,
        env_cfg: DictConfig,
        actor_cfg: DictConfig,
        embedder_cfg: DictConfig,
        state_encoder_cfg: DictConfig,
        estimator_cfg: DictConfig,
        training_cfg: DictConfig,
        evaluation_cfg: DictConfig,
        optimizer_cfg: Optional[DictConfig] = None,
        scheduler_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.reward_cfg = reward_cfg
        self.actor_cfg = actor_cfg
        self.state_encoder_cfg = state_encoder_cfg
        self.estimator_cfg = estimator_cfg
        self.training_cfg = self._require_cfg(training_cfg, "training_cfg")
        self.evaluation_cfg = self._require_cfg(evaluation_cfg, "evaluation_cfg")
        self.optimizer_cfg = self._optional_cfg(optimizer_cfg, "optimizer_cfg")
        self.scheduler_cfg = self._optional_cfg(scheduler_cfg, "scheduler_cfg")
        self.logging_cfg = self._optional_cfg(logging_cfg, "logging_cfg")

        self._eval_rollout_prefixes, self._eval_rollouts = self._parse_eval_rollouts(self.evaluation_cfg)
        eval_temp_cfg = self.evaluation_cfg.get("rollout_temperature")
        if eval_temp_cfg is None:
            raise ValueError("evaluation_cfg.rollout_temperature must be set explicitly (no implicit eval temperature).")
        self._eval_rollout_temperature = float(eval_temp_cfg)
        if self._eval_rollout_temperature < 0.0:
            raise ValueError(
                f"evaluation_cfg.rollout_temperature must be >= 0, got {self._eval_rollout_temperature}."
            )
        self._train_prog_bar = self._require_list_cfg(self.logging_cfg, "train_prog_bar")
        self._eval_prog_bar = self._require_list_cfg(self.logging_cfg, "eval_prog_bar")
        self._log_on_step_train = bool(self.logging_cfg.get("log_on_step_train", False))

        self.policy: nn.Module = hydra.utils.instantiate(policy_cfg)
        self.reward_fn: nn.Module = hydra.utils.instantiate(reward_cfg)
        self.env: GraphEnv = hydra.utils.instantiate(env_cfg)
        self.max_steps = int(self.env.max_steps)
        self.embedder = hydra.utils.instantiate(embedder_cfg)
        self.estimator = hydra.utils.instantiate(estimator_cfg)
        state_encoder: StateEncoder = hydra.utils.instantiate(state_encoder_cfg)
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
            ],
        )

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
        scheduler_type = str(self.scheduler_cfg.get("type", "") or "").lower()
        if not scheduler_type:
            return optimizer
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
        if scheduler_type in {"cosine_restart", "cosine_warm_restarts", "cosine_restarts"}:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.scheduler_cfg.get("t_0", 10)),
                T_mult=int(self.scheduler_cfg.get("t_mult", 1)),
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
    def _require_cfg(cfg: Any, name: str) -> DictConfig:
        if cfg is None:
            raise ValueError(f"{name} must be provided as a DictConfig (got None).")
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"{name} must be a DictConfig, got {type(cfg).__name__}.")
        return cfg

    @staticmethod
    def _optional_cfg(cfg: Any, name: str) -> DictConfig:
        if cfg is None:
            return OmegaConf.create({})
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"{name} must be a DictConfig or None, got {type(cfg).__name__}.")
        return cfg

    @staticmethod
    def _require_list_cfg(cfg: DictConfig, key: str) -> set[str]:
        value = cfg.get(key)
        if value is None:
            return set()
        if not isinstance(value, ListConfig):
            raise TypeError(f"{key} must be a ListConfig, got {type(value).__name__}.")
        return set(str(v) for v in value)

    @staticmethod
    def _require_positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a positive int, got bool.")
        if isinstance(value, int):
            parsed = int(value)
        elif isinstance(value, float):
            if not value.is_integer():
                raise TypeError(f"{name} must be a positive int, got {value}.")
            parsed = int(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text or not text.lstrip("-").isdigit():
                raise TypeError(f"{name} must be a positive int, got {value!r}.")
            parsed = int(text)
        else:
            raise TypeError(f"{name} must be a positive int, got {type(value).__name__}.")
        if parsed <= 0:
            raise ValueError(f"{name} must be > 0, got {parsed}.")
        return parsed

    @staticmethod
    def _parse_eval_rollouts(cfg: DictConfig) -> tuple[list[int], int]:
        value = cfg.get("num_eval_rollouts", 1)
        if isinstance(value, ListConfig):
            prefixes: list[int] = []
            for idx, raw in enumerate(value):
                prefixes.append(GFlowNetModule._require_positive_int(raw, f"evaluation_cfg.num_eval_rollouts[{idx}]"))
            if not prefixes:
                raise ValueError("evaluation_cfg.num_eval_rollouts must be a non-empty list.")
            prefixes = sorted(set(prefixes))
            return prefixes, max(prefixes)
        if isinstance(value, int):
            count = GFlowNetModule._require_positive_int(value, "evaluation_cfg.num_eval_rollouts")
            return [count], count
        raise TypeError(
            "evaluation_cfg.num_eval_rollouts must be an int or ListConfig "
            f"(got {type(value).__name__})."
        )

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
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        batch_size = int(batch.num_graphs)
        log_metric(self, "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self._log_metrics(metrics, prefix="train", batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        self._log_metrics(metrics, prefix="val", batch_size=int(batch.num_graphs))

    def test_step(self, batch, batch_idx: int):
        loss, metrics = self._compute_batch_loss(batch, batch_idx=batch_idx)
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        self._log_metrics(metrics, prefix="test", batch_size=int(batch.num_graphs))

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self._compute_rollout_records(batch=batch, batch_idx=batch_idx)

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

    def _compute_rollout_records(
        self,
        *,
        batch: Any,
        batch_idx: int | None = None,
    ) -> list[Dict[str, Any]]:
        device = self.device
        self._refresh_eval_settings()

        embed = self.embedder.embed_batch(batch, device=device)
        edge_tokens = embed.edge_tokens.to(dtype=torch.float32)
        edge_batch = embed.edge_batch
        edge_ptr = embed.edge_ptr
        node_ptr = embed.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        edge_index = embed.edge_index
        edge_relations = embed.edge_relations
        node_tokens = embed.node_tokens.to(dtype=torch.float32)
        question_tokens = embed.question_tokens.to(dtype=torch.float32)
        node_struct_raw = embed.node_struct_raw
        base_edge_scores = batch.edge_scores.to(device=device, dtype=torch.float32).view(-1)
        num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        node_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), node_counts)
        node_is_start = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        start_node_locals = batch.start_node_locals.to(device)
        if start_node_locals.numel() > 0:
            node_is_start[start_node_locals] = True
        node_is_answer = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        answer_node_locals = batch.answer_node_locals.to(device)
        if answer_node_locals.numel() > 0:
            node_is_answer[answer_node_locals] = True
        dummy_mask = getattr(batch, "is_dummy_agent", None)
        if torch.is_tensor(dummy_mask):
            dummy_mask = dummy_mask.to(device=device, dtype=torch.bool).view(-1)
        else:
            dummy_mask = torch.zeros(num_graphs, device=device, dtype=torch.bool)

        graph_cache: Dict[str, torch.Tensor] = {
            "edge_index": edge_index,
            "edge_batch": edge_batch,
            "edge_relations": edge_relations,
            "edge_scores": base_edge_scores,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "node_tokens": node_tokens,
            "node_batch": node_batch,
            "node_is_start": node_is_start,
            "node_is_answer": node_is_answer,
            "start_node_locals": start_node_locals,
            "start_ptr": batch._slice_dict["start_node_locals"].to(device),
            "answer_node_locals": answer_node_locals,
            "answer_ptr": batch._slice_dict["answer_node_locals"].to(device),
            "dummy_mask": dummy_mask,
        }

        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=torch.is_autocast_enabled())
            if device.type != "cpu"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            state_encoder_cache = self.actor.state_encoder.precompute(
                node_ptr=node_ptr,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
                edge_index=edge_index,
                start_node_locals=start_node_locals,
                node_struct_raw=node_struct_raw,
            )

        num_rollouts = self._require_positive_int(self._eval_rollouts, "evaluation_cfg.num_eval_rollouts")
        rollout_logs: list[Dict[str, torch.Tensor]] = []
        for _ in range(num_rollouts):
            rollout = self.actor.rollout(
                batch=batch,
                edge_tokens=edge_tokens,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
                edge_batch=edge_batch,
                edge_ptr=edge_ptr,
                node_ptr=node_ptr,
                temperature=self._eval_rollout_temperature,
                batch_idx=batch_idx,
                graph_cache=graph_cache,
                state_encoder_cache=state_encoder_cache,
            )
            rollout_logs.append(
                {
                    "actions_seq": rollout["actions_seq"].detach().cpu(),
                    "log_pf": rollout["log_pf"].detach().cpu(),
                    "directions_seq": rollout["directions_seq"].detach().cpu(),
                }
            )

        return self._build_rollout_records(
            batch=batch,
            rollout_logs=rollout_logs,
            node_ptr=node_ptr.detach().cpu(),
            edge_ptr=edge_ptr.detach().cpu(),
        )

    def _compute_batch_loss(self, batch: Any, batch_idx: int | None = None) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self.device
        if not self.training:
            self._refresh_eval_settings()

        embed = self.embedder.embed_batch(batch, device=device)
        edge_tokens = embed.edge_tokens.to(dtype=torch.float32)
        edge_batch = embed.edge_batch
        edge_ptr = embed.edge_ptr
        node_ptr = embed.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        edge_index = embed.edge_index
        edge_relations = embed.edge_relations
        node_tokens = embed.node_tokens.to(dtype=torch.float32)
        question_tokens = embed.question_tokens.to(dtype=torch.float32)
        node_struct_raw = embed.node_struct_raw
        answer_node_ptr = batch._slice_dict["answer_node_locals"].to(device)
        base_edge_scores = batch.edge_scores.to(device=device, dtype=torch.float32).view(-1)
        num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        node_counts = (node_ptr[1:] - node_ptr[:-1]).clamp(min=0)
        node_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), node_counts)
        node_is_start = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        start_node_locals = batch.start_node_locals.to(device)
        if start_node_locals.numel() > 0:
            node_is_start[start_node_locals] = True
        node_is_answer = torch.zeros(num_nodes_total, device=device, dtype=torch.bool)
        answer_node_locals = batch.answer_node_locals.to(device)
        if answer_node_locals.numel() > 0:
            node_is_answer[answer_node_locals] = True
        dummy_mask = getattr(batch, "is_dummy_agent", None)
        if torch.is_tensor(dummy_mask):
            dummy_mask = dummy_mask.to(device=device, dtype=torch.bool).view(-1)
        else:
            dummy_mask = torch.zeros(num_graphs, device=device, dtype=torch.bool)

        graph_cache: Dict[str, torch.Tensor] = {
            "edge_index": edge_index,
            "edge_batch": edge_batch,
            "edge_relations": edge_relations,
            "edge_scores": base_edge_scores,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "node_tokens": node_tokens,
            "node_batch": node_batch,
            "node_is_start": node_is_start,
            "node_is_answer": node_is_answer,
            "start_node_locals": start_node_locals,
            "start_ptr": batch._slice_dict["start_node_locals"].to(device),
            "answer_node_locals": answer_node_locals,
            "answer_ptr": batch._slice_dict["answer_node_locals"].to(device),
            "dummy_mask": dummy_mask,
        }

        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=torch.is_autocast_enabled())
            if device.type != "cpu"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            state_encoder_cache = self.actor.state_encoder.precompute(
                node_ptr=node_ptr,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
                edge_index=edge_index,
                start_node_locals=start_node_locals,
                node_struct_raw=node_struct_raw,
            )
        bc_weight = self._compute_bc_weight() if self.training else 0.0
        bc_loss_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        bc_steps_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        bc_valid_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        bc_pair_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        bc_loss_scalar = torch.zeros((), device=device, dtype=torch.float32)

        if self.training:
            num_rollouts = self._require_positive_int(
                self.training_cfg.get("num_train_rollouts"),
                "training_cfg.num_train_rollouts",
            )
        else:
            num_rollouts = self._require_positive_int(self._eval_rollouts, "evaluation_cfg.num_eval_rollouts")
        loss_list = []
        metrics_list: list[Dict[str, torch.Tensor]] = []
        rollout_answer_hits: list[torch.Tensor] = []
        need_bc = self.training and bc_weight > 0.0
        dag_edge_mask = None
        if need_bc:
            edge_labels = getattr(batch, "edge_labels", None)
            if not torch.is_tensor(edge_labels):
                raise ValueError("edge_labels missing in g_agent batch; cannot derive dag_edge_mask.")
            dag_edge_mask = edge_labels.to(device=device, dtype=torch.float32).view(-1) > POS_LABEL_THRESHOLD
            if dag_edge_mask.numel() != edge_tokens.size(0):
                raise ValueError(
                    f"dag_edge_mask length {dag_edge_mask.numel()} != num_edges {edge_tokens.size(0)}."
                )

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
                dag_edge_mask=dag_edge_mask if need_bc else None,
                return_bc_stats=need_bc,
            )
            # Rollout returns log_pf/log_pf_steps/actions_seq for SubTB.
            state_emb_seq = rollout.get("state_emb_seq")
            if state_emb_seq is None:
                raise ValueError("state_emb_seq missing in rollout; actor/state_encoder must return per-step embeddings.")

            reward_out: RewardOutput = self.reward_fn(
                selected_mask=rollout["selected_mask"],
                edge_scores=base_edge_scores,
                edge_batch=edge_batch,
                answer_hit=rollout["reach_success"],
                dummy_mask=dummy_mask,
                pair_start_node_locals=batch.pair_start_node_locals.to(device),
                pair_answer_node_locals=batch.pair_answer_node_locals.to(device),
                pair_shortest_lengths=batch.pair_shortest_lengths.to(device),
                start_node_hit=rollout.get("start_node_hit"),
                answer_node_hit=rollout.get("answer_node_hit"),
                node_ptr=node_ptr,
            )

            log_reward = reward_out.log_reward
            if bool(dummy_mask.any().item()):
                log_reward_for_loss = torch.where(dummy_mask, torch.zeros_like(log_reward), log_reward)
                self._assert_finite(log_reward_for_loss, "log_reward")
            else:
                self._assert_finite(log_reward, "log_reward")
                log_reward_for_loss = log_reward
            self._assert_finite(rollout["log_pf"], "log_pf")
            log_flow_states = self._compute_log_flow_states(
                state_emb_seq=state_emb_seq,
                question_tokens=question_tokens,
                log_reward=log_reward_for_loss,
                edge_lengths=rollout["length"].long(),
            )
            # Backward policy P_B is deterministic under this state definition:
            # given the full trajectory state, the predecessor is uniquely obtained by removing the last action.
            log_pb_steps = self._compute_deterministic_log_pb_steps(
                actions_seq=rollout["actions_seq"],
                stop_indices=torch.full((num_graphs,), STOP_RELATION, device=device, dtype=torch.long),
                dtype=rollout["log_pf_steps"].dtype,
                device=device,
            )
            graph_mask = None
            if bool(dummy_mask.any().item()):
                graph_mask = ~dummy_mask
            subtb_loss = self._compute_subtb_loss(
                log_flow_states=log_flow_states,
                log_pf_steps=rollout["log_pf_steps"],
                log_pb_steps=log_pb_steps,
                edge_lengths=rollout["length"].long(),
                graph_mask=graph_mask,
            )
            if need_bc:
                bc_loss_per_graph = rollout.get("bc_loss_per_graph")
                bc_steps_per_graph = rollout.get("bc_steps_per_graph")
                bc_pair_per_graph = rollout.get("bc_has_dag")
                if bc_loss_per_graph is None or bc_steps_per_graph is None or bc_pair_per_graph is None:
                    bc_loss_per_graph, bc_steps_per_graph, bc_pair_per_graph = self._compute_dag_bc_loss(
                        log_prob_edge_seq=rollout.get("log_prob_edge_seq"),
                        valid_edges_seq=rollout.get("valid_edges_seq"),
                        dag_edge_mask=dag_edge_mask,
                        edge_batch=edge_batch,
                        num_graphs=num_graphs,
                    )
                bc_valid_per_graph = (bc_steps_per_graph > 0).to(dtype=bc_loss_per_graph.dtype)
                bc_loss_scalar = bc_loss_per_graph.mean()
            else:
                bc_loss_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.float32)
                bc_steps_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.float32)
                bc_valid_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.float32)
                bc_pair_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.float32)
                bc_loss_scalar = torch.zeros((), device=device, dtype=torch.float32)

            loss = subtb_loss + bc_weight * bc_loss_scalar

            reward_metrics = reward_out.as_dict()
            log_reward_metric = reward_metrics.pop("log_reward")
            reward_metrics.pop("reward", None)
            answer_hit = reward_metrics.pop("answer_hit", None)
            success = reward_metrics.pop("success", None)
            if answer_hit is None:
                answer_hit = success

            metrics: Dict[str, torch.Tensor] = {
                "log_reward": log_reward_metric,
                "answer_hit": (
                    answer_hit.detach()
                    if isinstance(answer_hit, torch.Tensor)
                    else rollout["reach_success"].detach()
                ),
                "prob_answer": (
                    answer_hit.detach()
                    if isinstance(answer_hit, torch.Tensor)
                    else rollout["reach_success"].detach()
                ),
                "length_mean": rollout["length"].detach(),
                "subtb_loss": subtb_loss.detach(),
                "bc_loss": bc_loss_per_graph.detach(),
                "bc_steps": bc_steps_per_graph.detach(),
                "bc_valid": bc_valid_per_graph.detach(),
                "bc_has_pair": bc_pair_per_graph.detach(),
                "bc_weight": torch.full_like(bc_steps_per_graph, float(bc_weight)),
                **{k: v.detach() for k, v in reward_metrics.items()},
            }

            if not self.training:
                if isinstance(answer_hit, torch.Tensor):
                    answer_val = answer_hit.detach().to(dtype=torch.bool)
                else:
                    answer_val = rollout["reach_success"].detach().to(dtype=torch.bool)
                rollout_answer_hits.append(answer_val)

            loss_list.append(loss)
            metrics_list.append(metrics)

        if not loss_list:
            raise RuntimeError(f"No rollouts recorded (batch_idx={batch_idx}, num_rollouts={num_rollouts}).")
        if num_rollouts > 1:
            loss = torch.stack(loss_list, dim=0).mean()
            metrics = self._aggregate_metrics(metrics_list, best_of=(not self.training))
        else:
            loss = loss_list[0]
            metrics = metrics_list[0]

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
        return loss, metrics


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

    @staticmethod
    def _segment_logsumexp_1d(
        logits: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        if logits.dim() != 1 or segment_ids.dim() != 1:
            raise ValueError("logits and segment_ids must be 1D tensors.")
        if logits.numel() != segment_ids.numel():
            raise ValueError("logits and segment_ids must have the same length.")
        if num_segments <= 0:
            raise ValueError(f"num_segments must be positive, got {num_segments}")
        if logits.numel() == 0:
            neg_inf = torch.finfo(logits.dtype).min
            return torch.full((num_segments,), neg_inf, device=logits.device, dtype=logits.dtype)
        neg_inf = torch.finfo(logits.dtype).min
        max_per = torch.full((num_segments,), neg_inf, device=logits.device, dtype=logits.dtype)
        max_per.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
        shifted = logits - max_per[segment_ids]
        exp = torch.exp(shifted)
        sum_per = torch.zeros((num_segments,), device=logits.device, dtype=logits.dtype)
        sum_per.index_add_(0, segment_ids, exp)
        eps = torch.finfo(logits.dtype).eps
        return torch.log(sum_per.clamp(min=eps)) + max_per

    def _compute_dag_bc_loss(
        self,
        *,
        log_prob_edge_seq: torch.Tensor | None,
        valid_edges_seq: torch.Tensor | None,
        dag_edge_mask: torch.Tensor | None,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_batch.device
        if log_prob_edge_seq is None or valid_edges_seq is None or dag_edge_mask is None:
            raise ValueError("dag BC loss requires log_prob_edge_seq/valid_edges_seq/dag_edge_mask.")
        if log_prob_edge_seq.dim() != 2 or valid_edges_seq.shape != log_prob_edge_seq.shape:
            raise ValueError("log_prob_edge_seq/valid_edges_seq must be [T,E] with matching shapes.")
        dag_edge_mask = dag_edge_mask.to(device=device, dtype=torch.bool).view(1, -1)
        if dag_edge_mask.size(1) != log_prob_edge_seq.size(1):
            raise ValueError("dag_edge_mask length mismatch with log_prob_edge_seq.")

        mask = valid_edges_seq.to(device=device, dtype=torch.bool) & dag_edge_mask
        neg_inf = torch.finfo(log_prob_edge_seq.dtype).min
        masked_logits = torch.where(mask, log_prob_edge_seq, torch.full_like(log_prob_edge_seq, neg_inf))
        num_steps = int(log_prob_edge_seq.size(0))
        edge_batch = edge_batch.to(device=device, dtype=torch.long).view(1, -1)
        step_offsets = torch.arange(num_steps, device=device, dtype=edge_batch.dtype).view(-1, 1) * int(num_graphs)
        segment_ids = (edge_batch + step_offsets).reshape(-1)
        logsumexp = self._segment_logsumexp_1d(
            masked_logits.reshape(-1),
            segment_ids,
            num_segments=num_steps * int(num_graphs),
        ).view(num_steps, num_graphs)
        mask_f = mask.reshape(-1).to(dtype=torch.float32)
        counts = torch.zeros(num_steps * int(num_graphs), device=device, dtype=mask_f.dtype)
        counts.index_add_(0, segment_ids, mask_f)
        counts = counts.view(num_steps, num_graphs)
        has_valid = counts > 0
        loss_steps = torch.where(has_valid, -logsumexp, torch.zeros_like(logsumexp))
        step_counts = has_valid.to(dtype=loss_steps.dtype).sum(dim=0)
        bc_loss = loss_steps.sum(dim=0) / step_counts.clamp(min=1.0)

        dag_counts = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        dag_counts.index_add_(0, edge_batch.view(-1), dag_edge_mask.view(-1).to(dtype=torch.float32))
        bc_has_dag = (dag_counts > 0).to(dtype=bc_loss.dtype)
        return bc_loss, step_counts, bc_has_dag

    def _refresh_eval_settings(self) -> None:
        """Ensure eval rollouts reflect current config (even when loading old checkpoints)."""
        prefixes, count = self._parse_eval_rollouts(self.evaluation_cfg)
        self._eval_rollout_prefixes = prefixes
        self._eval_rollouts = count
        eval_temp_cfg = self.evaluation_cfg.get("rollout_temperature")
        if eval_temp_cfg is None:
            raise ValueError("evaluation_cfg.rollout_temperature must be set explicitly (no implicit eval temperature).")
        if float(eval_temp_cfg) < 0.0:
            raise ValueError(f"evaluation_cfg.rollout_temperature must be >= 0, got {eval_temp_cfg}.")
        self._eval_rollout_temperature = float(eval_temp_cfg)

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

    def _build_rollout_records(
        self,
        *,
        batch: Any,
        rollout_logs: list[Dict[str, torch.Tensor]],
        node_ptr: torch.Tensor,
        edge_ptr: torch.Tensor,
    ) -> list[Dict[str, Any]]:
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            raise ValueError("node_ptr must encode at least one graph.")
        edge_ptr = edge_ptr.to(dtype=torch.long).view(-1)
        if edge_ptr.numel() != num_graphs + 1:
            raise ValueError(f"edge_ptr length {edge_ptr.numel()} != num_graphs+1 ({num_graphs + 1}).")
        if edge_ptr.numel() == 0:
            raise ValueError("edge_ptr must be non-empty.")
        if int(edge_ptr[0].item()) != 0:
            raise ValueError(f"edge_ptr must start at 0, got {int(edge_ptr[0].item())}.")
        if bool((edge_ptr[1:] < edge_ptr[:-1]).any().item()):
            raise ValueError("edge_ptr must be non-decreasing.")
        total_edges = int(edge_ptr[-1].item())
        sample_ids, questions = self._extract_batch_meta(batch, num_graphs)
        if not rollout_logs:
            raise ValueError("rollout_logs must be non-empty.")

        normalized_logs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for ridx, log in enumerate(rollout_logs):
            actions_seq = log.get("actions_seq")
            directions_seq = log.get("directions_seq")
            log_pf = log.get("log_pf")
            if actions_seq is None or directions_seq is None or log_pf is None:
                raise ValueError(f"rollout_logs[{ridx}] missing required keys (actions_seq/directions_seq/log_pf).")
            if actions_seq.dim() != 2:
                raise ValueError(
                    f"rollout_logs[{ridx}].actions_seq must be [B,T], got shape={tuple(actions_seq.shape)}."
                )
            if directions_seq.shape != actions_seq.shape:
                raise ValueError(
                    f"rollout_logs[{ridx}].directions_seq shape mismatch with actions_seq: "
                    f"{tuple(directions_seq.shape)} vs {tuple(actions_seq.shape)}."
                )
            if actions_seq.size(0) != num_graphs:
                raise ValueError(
                    f"rollout_logs[{ridx}].actions_seq batch {actions_seq.size(0)} != num_graphs {num_graphs}."
                )
            if log_pf.dim() != 1 or log_pf.numel() != num_graphs:
                raise ValueError(
                    f"rollout_logs[{ridx}].log_pf must be [B], got shape={tuple(log_pf.shape)}."
                )
            normalized_logs.append(
                (
                    actions_seq.to(dtype=torch.long),
                    directions_seq.to(dtype=torch.long),
                    log_pf,
                )
            )

        records: list[Dict[str, Any]] = []
        for g in range(num_graphs):
            rollouts: list[Dict[str, Any]] = []
            edge_start = int(edge_ptr[g].item())
            edge_end = int(edge_ptr[g + 1].item())
            for ridx, (actions_seq, directions_seq, log_pf) in enumerate(normalized_logs):
                actions = actions_seq[g].to(dtype=torch.long)
                directions = directions_seq[g].to(dtype=torch.long)
                if actions.numel() != directions.numel():
                    raise ValueError(
                        f"rollout_logs[{ridx}] actions/directions length mismatch for graph {g}: "
                        f"{actions.numel()} vs {directions.numel()}."
                    )
                if bool((actions < STOP_RELATION).any().item()):
                    bad = actions[actions < STOP_RELATION][:5].tolist()
                    raise ValueError(f"rollout_logs[{ridx}] actions_seq contains invalid negatives: {bad}.")
                keep = actions >= 0
                edge_ids_global = actions[keep].tolist()
                dir_ids = directions[keep].tolist()
                if edge_ids_global:
                    if total_edges <= 0:
                        raise ValueError("actions_seq selects edges but edge_ptr indicates zero total edges.")
                    if min(edge_ids_global) < edge_start or max(edge_ids_global) >= edge_end:
                        raise ValueError(
                            f"rollout_logs[{ridx}] edge ids out of range for graph {g}: "
                            f"expected [{edge_start},{edge_end}), got min={min(edge_ids_global)} "
                            f"max={max(edge_ids_global)}."
                        )
                    invalid_dir = [
                        d for d in dir_ids if d not in (DIRECTION_FORWARD, DIRECTION_BACKWARD)
                    ]
                    if invalid_dir:
                        raise ValueError(
                            f"rollout_logs[{ridx}] directions_seq contains invalid values: {invalid_dir[:5]}."
                        )

                edge_ids = [int(eid - edge_start) for eid in edge_ids_global]
                rollouts.append(
                    {
                        "rollout_index": ridx,
                        "log_pf": float(log_pf[g].item()),
                        "edge_ids": edge_ids,
                        "directions": dir_ids,
                    }
                )

            records.append(
                {
                    "sample_id": sample_ids[g] if g < len(sample_ids) else str(g),
                    "question": questions[g] if g < len(questions) else "",
                    "rollouts": rollouts,
                }
            )
        return records

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

    def _estimate_total_steps(self) -> int | None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        total_steps = getattr(trainer, "estimated_stepping_batches", None)
        if total_steps is not None:
            try:
                total_steps = int(total_steps)
            except (TypeError, ValueError):
                total_steps = None
        if total_steps is None or total_steps <= 0:
            max_steps = getattr(trainer, "max_steps", None)
            if isinstance(max_steps, int) and max_steps > 0:
                total_steps = max_steps
        if total_steps is None or total_steps <= 0:
            return None
        return int(total_steps)

    def _compute_bc_weight(self) -> float:
        bc_weight = float(self.training_cfg.get("bc_weight", 0.0))
        if bc_weight <= 0.0:
            return 0.0
        bc_floor = float(self.training_cfg.get("bc_weight_floor", 0.0))
        bc_floor = max(0.0, min(bc_floor, bc_weight))
        hold_ratio = float(self.training_cfg.get("bc_hold_ratio", 0.0))
        decay_ratio = float(self.training_cfg.get("bc_decay_ratio", 0.0))
        hold_ratio = max(0.0, hold_ratio)
        decay_ratio = max(0.0, decay_ratio)

        total_steps = self._estimate_total_steps()
        if total_steps is None or (hold_ratio == 0.0 and decay_ratio == 0.0):
            return bc_weight

        hold_steps = int(round(total_steps * hold_ratio))
        decay_steps = int(round(total_steps * decay_ratio))
        hold_steps = max(0, hold_steps)
        decay_steps = max(0, decay_steps)

        step = int(getattr(self, "global_step", 0))
        if step < hold_steps:
            scale = 1.0
        elif decay_steps <= 0:
            scale = 0.0
        else:
            t = min(max(step - hold_steps, 0), decay_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * (t / decay_steps)))
        return bc_floor + (bc_weight - bc_floor) * scale

    def _compute_subtb_loss(
        self,
        *,
        log_flow_states: torch.Tensor,  # [B, T_state]
        log_pf_steps: torch.Tensor,  # [B, T_action]
        log_pb_steps: torch.Tensor,  # [B, T_action]
        edge_lengths: torch.Tensor,  # [B] number of selected edges
        graph_mask: torch.Tensor | None = None,  # [B] optional bool mask for averaging
    ) -> torch.Tensor:
        """Sub-Trajectory Balance with λ=1 (uniform weights over all sub-trajectories).
        """
        device = log_pf_steps.device
        num_graphs, num_actions = log_pf_steps.shape
        if log_flow_states.shape != (num_graphs, num_actions + 1):
            raise ValueError(
                f"log_flow_states shape {tuple(log_flow_states.shape)} != (B, T_action+1)=({num_graphs},{num_actions + 1})"
            )
        if log_pb_steps.shape != log_pf_steps.shape:
            raise ValueError("log_pb_steps must have the same shape as log_pf_steps.")
        if log_pb_steps.numel() > 0 and bool((log_pb_steps != 0).any().item()):
            raise ValueError("Deterministic P_B expects log_pb_steps to be all zeros.")

        if num_actions == 0:
            per_graph = torch.zeros(num_graphs, device=device, dtype=log_pf_steps.dtype)
        else:
            log_pf_prefix = torch.zeros(num_graphs, num_actions + 1, device=device, dtype=log_pf_steps.dtype)
            log_pf_prefix[:, 1:] = log_pf_steps.cumsum(dim=1)

            # r_{i,j} = logF_i + (prefix_j - prefix_i) - logF_j = a_i + b_j.
            a = log_flow_states - log_pf_prefix
            b = log_pf_prefix - log_flow_states

            a_cumsum = a.cumsum(dim=1)
            a2_cumsum = (a * a).cumsum(dim=1)
            prefix_a = a_cumsum - a
            prefix_a2 = a2_cumsum - (a * a)

            idx = torch.arange(num_actions + 1, device=device, dtype=log_pf_steps.dtype).view(1, -1)
            b2 = b * b
            contrib = prefix_a2 + 2.0 * b * prefix_a + idx * b2

            term_state = edge_lengths.clamp(min=0, max=num_actions - 1) + 1
            mask = idx <= term_state.view(-1, 1)
            mask_f = mask.to(dtype=contrib.dtype)

            sum_sq = (contrib * mask_f).sum(dim=1)
            denom = (idx * mask_f).sum(dim=1).clamp(min=1.0)
            per_graph = sum_sq / denom

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
