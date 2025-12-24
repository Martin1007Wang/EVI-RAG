from __future__ import annotations

import contextlib
import logging
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
from src.models.components.gflownet_env import STOP_RELATION, DIRECTION_FORWARD
from src.utils import setup_optimizer
from src.utils.logging_utils import log_metric
logger = logging.getLogger(__name__)


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
        self.predict_metrics: Dict[str, torch.Tensor] = {}

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
    def _parse_eval_rollouts(cfg: DictConfig) -> tuple[list[int], int]:
        value = cfg.get("num_eval_rollouts", 1)
        if isinstance(value, ListConfig):
            prefixes = sorted({int(max(1, v)) for v in value})
            return (prefixes or [1]), (max(prefixes) if prefixes else 1)
        if isinstance(value, int):
            count = max(1, int(value))
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

    def _compute_rollout_records(self, *, batch: Any, batch_idx: int | None = None) -> list[Dict[str, Any]]:
        device = self.device
        self._refresh_eval_settings()

        embed = self.embedder.embed_batch(batch, device=device)
        edge_tokens = embed.edge_tokens.to(dtype=torch.float32)
        edge_batch = embed.edge_batch
        edge_ptr = embed.edge_ptr
        node_ptr = embed.node_ptr
        edge_index = embed.edge_index
        edge_relations = embed.edge_relations
        node_tokens = embed.node_tokens.to(dtype=torch.float32)
        question_tokens = embed.question_tokens.to(dtype=torch.float32)
        base_edge_scores = batch.edge_scores.to(device=device, dtype=torch.float32).view(-1)

        graph_cache: Dict[str, torch.Tensor] = {
            "edge_index": edge_index,
            "edge_batch": edge_batch,
            "edge_relations": edge_relations,
            "edge_scores": base_edge_scores,
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
                node_ptr=node_ptr,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
            )

        num_rollouts = max(1, int(self._eval_rollouts))
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
                }
            )

        return self._build_rollout_records(
            batch=batch,
            rollout_logs=rollout_logs,
            node_ptr=node_ptr.detach().cpu(),
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
        answer_node_ptr = batch._slice_dict["answer_node_locals"].to(device)
        base_edge_scores = batch.edge_scores.to(device=device, dtype=torch.float32).view(-1)

        graph_cache: Dict[str, torch.Tensor] = {
            "edge_index": edge_index,
            "edge_batch": edge_batch,
            "edge_relations": edge_relations,
            "edge_scores": base_edge_scores,
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
                node_ptr=node_ptr,
                node_tokens=node_tokens,
                question_tokens=question_tokens,
            )

        if self.training:
            num_rollouts = max(int(self.training_cfg.num_train_rollouts), 1)
        else:
            try:
                num_rollouts = int(self._eval_rollouts)
            except Exception:
                num_rollouts = 1
            if num_rollouts <= 0:
                num_rollouts = 1
                logger.warning(
                    "[EVAL_ROLLOUTS_FIX] _eval_rollouts<=0 detected; clamped to 1 (batch_idx=%s).",
                    str(batch_idx),
                )
        loss_list = []
        metrics_list: list[Dict[str, torch.Tensor]] = []
        rollout_answer_hits: list[torch.Tensor] = []

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
                edge_labels=batch.edge_labels.to(device),
                edge_batch=edge_batch,
                answer_hit=rollout["reach_success"],
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
            # given the full trajectory state, the predecessor is uniquely obtained by removing the last action.
            log_pb_steps = self._compute_deterministic_log_pb_steps(
                actions_seq=rollout["actions_seq"],
                stop_indices=torch.full((num_graphs,), STOP_RELATION, device=device, dtype=torch.long),
                dtype=rollout["log_pf_steps"].dtype,
                device=device,
            )
            subtb_loss = self._compute_subtb_loss(
                log_flow_states=log_flow_states,
                log_pf_steps=rollout["log_pf_steps"],
                log_pb_steps=log_pb_steps,
                edge_lengths=rollout["length"].long(),
            )
            loss = subtb_loss

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
                "length_mean": rollout["length"].detach(),
                "subtb_loss": subtb_loss.detach(),
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
            # Defensive: avoid empty TensorList when eval rollouts unexpectedly skipped.
            loss = torch.zeros((), device=device, dtype=torch.float32)
            metrics: Dict[str, torch.Tensor] = {}
            if not self.training:
                logger.warning(
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

    @staticmethod
    def _value_at(tensor: torch.Tensor, idx: int) -> torch.Tensor:
        flat = tensor.view(-1)
        if flat.numel() == 0:
            return torch.tensor(0.0)
        if idx < flat.numel():
            return flat[idx]
        return flat[-1]

    def _build_rollout_records(
        self,
        *,
        batch: Any,
        rollout_logs: list[Dict[str, torch.Tensor]],
        node_ptr: torch.Tensor,
    ) -> list[Dict[str, Any]]:
        num_graphs = int(node_ptr.numel() - 1)
        sample_ids, questions = self._extract_batch_meta(batch, num_graphs)
        try:
            edge_index = batch.edge_index.detach().cpu()
            edge_relations = batch.edge_attr.detach().cpu()
            node_global_ids = batch.node_global_ids.detach().cpu()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to collect edge metadata for rollouts: %s", exc)
            edge_index = None
            edge_relations = None
            node_global_ids = None

        records: list[Dict[str, Any]] = []
        for g in range(num_graphs):
            rollouts: list[Dict[str, Any]] = []
            for ridx, log in enumerate(rollout_logs):
                actions = log["actions_seq"][g].to(dtype=torch.long)
                edge_ids = actions[actions >= 0].tolist()
                edges: list[Dict[str, Any]] = []
                if edge_index is not None and edge_relations is not None and node_global_ids is not None:
                    for e_id in edge_ids:
                        if e_id < 0 or e_id >= int(edge_index.size(1)):
                            continue
                        h_local = int(edge_index[0, e_id].item())
                        t_local = int(edge_index[1, e_id].item())
                        if h_local < 0 or t_local < 0 or h_local >= int(node_global_ids.numel()) or t_local >= int(node_global_ids.numel()):
                            continue
                        h_gid = int(node_global_ids[h_local].item())
                        t_gid = int(node_global_ids[t_local].item())
                        rel_id = int(edge_relations[e_id].item()) if e_id < int(edge_relations.numel()) else None
                        edges.append(
                            {
                                "edge_id": int(e_id),
                                "head_entity_id": h_gid,
                                "relation_id": rel_id,
                                "tail_entity_id": t_gid,
                                "src_entity_id": h_gid,
                                "dst_entity_id": t_gid,
                            }
                        )
                relation_seq = [int(edge.get("relation_id")) for edge in edges if edge.get("relation_id") is not None]
                direction_seq = [int(DIRECTION_FORWARD) for _ in relation_seq]

                rollouts.append(
                    {
                        "rollout_index": ridx,
                        "log_pf": float(self._value_at(log["log_pf"], g).item())
                        if "log_pf" in log
                        else None,
                        "relations": relation_seq,
                        "directions": direction_seq,
                        "edges": edges,
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
