from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Any

import torch
from lightning import LightningModule

from src.models.batch_adapter import DualFlowBatchAdapter
from src.models.components.policy import DualFlowPolicy
from src.models.components.sampler import RolloutResult, RolloutSampler
from src.models.configs.dual_flow_cfg import DualFlowConfig
from src.models.configs.environment import EnvironmentConfig
from src.models.configs.objective import SubTBConfig
from src.models.configs.policy import PolicyConfig
from src.models.configs.search import BeamSearchConfig, RolloutConfig
from src.models.configs.training import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.models.eval_export import DualFlowEvalExporter
from src.models.environment.builder import GraphEnvironmentBuilder
from src.models.environment.contracts import GraphEnvContext
from src.models.environment.masks import build_node_membership_mask as build_graph_node_membership_mask
from src.models.metrics.subtb_loss import SubTrajectoryBalanceLoss
from src.models.optimizers import build_optimizer_and_scheduler
from src.models.reward_engine import DualFlowRewardEngine
from src.utils.logging_utils import log_metric

EncodedPolicyContext = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DualFlowModule(LightningModule):
    """DualFlow orchestration module with strict typed configs and automatic optimization."""

    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        policy_cfg: PolicyConfig,
        sampling_cfg: RolloutConfig,
        eval_cfg: BeamSearchConfig,
        subtb_cfg: SubTBConfig,
        training_cfg: TrainingConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
    ) -> None:
        super().__init__()
        self.cfg = DualFlowConfig(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            sampling_cfg=sampling_cfg,
            eval_cfg=eval_cfg,
            subtb_cfg=subtb_cfg,
            training_cfg=training_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
        )
        self.save_hyperparameters({"config": asdict(self.cfg)}, logger=False)

        self.env_builder = GraphEnvironmentBuilder(self.cfg.env_cfg)
        self.policy = DualFlowPolicy(
            self.cfg.policy_cfg,
            backward_prior_mode=self.cfg.sampling_cfg.backward_prior_mode,
        )
        self.sampler = RolloutSampler(self.cfg.sampling_cfg)
        self.subtb_loss_fn = SubTrajectoryBalanceLoss(self.cfg.subtb_cfg)
        self.batch_adapter = DualFlowBatchAdapter(super_source_enabled=bool(self.cfg.env_cfg.super_source_enabled))
        self.reward_engine = DualFlowRewardEngine(stop_cfg=self.cfg.env_cfg.stop)
        self.eval_exporter = DualFlowEvalExporter()

    def _sample_online_rollout(
        self,
        *,
        base_context: GraphEnvContext,
    ) -> tuple[RolloutResult, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], EncodedPolicyContext]:
        rollout_context = self.override_forward_start_nodes(base_context, deterministic=False)
        encoded_context = self._encode_policy_context(rollout_context)
        online_rollout = self.sampler.sample_forward(
            rollout_context,
            self.policy,
            deterministic=False,
            encoded_context=encoded_context,
        )
        rewards_online_raw, reward_metrics_online = self.compute_rewards(online_rollout.stop_nodes, base_context)
        hit_mask_online = self.compute_hit_mask(online_rollout.stop_nodes, base_context)
        return online_rollout, rewards_online_raw, hit_mask_online, reward_metrics_online, encoded_context

    @staticmethod
    def _prefix_metric_namespace(
        metrics: dict[str, torch.Tensor],
        *,
        namespace: str,
    ) -> dict[str, torch.Tensor]:
        return {f"{namespace}/{key}": value for key, value in metrics.items()}

    @staticmethod
    def _sample_uniform_start_nodes(
        *,
        local_indices: torch.Tensor,
        ptr: torch.Tensor,
        num_graphs: int,
        device: torch.device,
        field_name: str,
        require_non_empty: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ptr.dim() != 1 or int(ptr.numel()) != num_graphs + 1:
            raise ValueError(f"{field_name}_ptr must have shape [B+1], got {tuple(ptr.shape)} for B={num_graphs}.")
        counts = (ptr[1:] - ptr[:-1]).to(device=device, dtype=torch.long)
        valid_graph = counts > 0
        if require_non_empty and bool((~valid_graph).any().item()):
            raise ValueError(f"{field_name} must be non-empty for every graph when backward_weight > 0.")
        starts = torch.zeros((num_graphs,), dtype=torch.long, device=device)
        if int(local_indices.numel()) == 0:
            return starts, valid_graph
        safe_counts = counts.clamp(min=1)
        rand = torch.rand((num_graphs,), device=device, dtype=torch.float32)
        offsets = torch.floor(rand * safe_counts.to(dtype=torch.float32)).to(dtype=torch.long)
        base = ptr[:-1].to(device=device, dtype=torch.long)
        flat_idx = (base + offsets).clamp(min=0, max=int(local_indices.numel()) - 1)
        gathered = local_indices.to(device=device, dtype=torch.long).index_select(0, flat_idx)
        starts = torch.where(valid_graph, gathered, starts)
        return starts, valid_graph

    def _sample_backward_rollout(
        self,
        *,
        base_context: GraphEnvContext,
        encoded_context: EncodedPolicyContext,
        current_beta: float,
    ) -> tuple[RolloutResult, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        num_graphs = int(base_context.num_graphs)
        a_counts = (base_context.a_ptr[1:] - base_context.a_ptr[:-1]).to(device=self.device, dtype=torch.long)
        valid_start_graph = a_counts > 0
        if bool((~valid_start_graph).any().item()):
            raise ValueError("a_local_indices must be non-empty for every graph when backward_weight > 0.")
        start_local_bwd, valid_start_graph = self._sample_uniform_start_nodes(
            local_indices=base_context.a_local_indices,
            ptr=base_context.a_ptr,
            num_graphs=num_graphs,
            device=self.device,
            field_name="a_local_indices",
            require_non_empty=True,
        )
        _, valid_target_graph = self._sample_uniform_start_nodes(
            local_indices=base_context.q_local_indices,
            ptr=base_context.q_ptr,
            num_graphs=num_graphs,
            device=self.device,
            field_name="q_local_indices",
            require_non_empty=True,
        )
        valid_graph = valid_start_graph & valid_target_graph
        # Backward rollout must walk reversed graph edges (adj_t_bwd as rollout topology).
        bwd_context = replace(
            base_context,
            start_local_indices=start_local_bwd,
            adj_t_fwd=base_context.adj_t_bwd,
            adj_t_bwd=base_context.adj_t_fwd,
        )
        bwd_rollout = self.sampler.sample_forward(
            bwd_context,
            self.policy,
            deterministic=False,
            encoded_context=encoded_context,
        )
        num_rollouts = int(bwd_rollout.stop_nodes.size(1))
        valid_by_graph = valid_graph.view(num_graphs, 1).expand(num_graphs, num_rollouts)
        if bwd_rollout.valid_mask is None:
            bwd_valid_mask = valid_by_graph
        else:
            bwd_valid_mask = bwd_rollout.valid_mask.to(device=self.device, dtype=torch.bool) & valid_by_graph
        bwd_rollout = replace(bwd_rollout, valid_mask=bwd_valid_mask)
        bwd_rewards_raw, bwd_reward_metrics = self.compute_rewards(
            stop_nodes_abs=bwd_rollout.stop_nodes,
            context=base_context,
            reward_beta=current_beta,
            target_local_indices=base_context.q_local_indices,
            target_ptr=base_context.q_ptr,
            target_field_name="q_local_indices",
        )
        bwd_hit_mask = self.compute_hit_mask(
            bwd_rollout.stop_nodes,
            base_context,
            target_local_indices=base_context.q_local_indices,
            target_ptr=base_context.q_ptr,
            target_field_name="q_local_indices",
        )
        bwd_valid_ratio = bwd_valid_mask.float().mean()
        return bwd_rollout, bwd_rewards_raw, bwd_hit_mask, bwd_reward_metrics, bwd_valid_ratio

    def _compute_backward_subtb_loss(
        self,
        *,
        context: GraphEnvContext,
        encoded_context: EncodedPolicyContext,
        current_beta: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        backward_weight = float(self.cfg.subtb_cfg.backward_weight)
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        if backward_weight <= 0.0:
            return zero, {"subtb/backward_weight": zero}
        bwd_rollout, bwd_rewards_raw, bwd_hit_mask, bwd_reward_metrics, bwd_valid_ratio = self._sample_backward_rollout(
            base_context=context,
            encoded_context=encoded_context,
            current_beta=current_beta,
        )
        bwd_loss_raw, bwd_loss_metrics = self.subtb_loss_fn(
            fwd_rollout=bwd_rollout,
            rewards=bwd_rewards_raw,
            reward_beta=current_beta,
            hit_mask=bwd_hit_mask,
        )
        if not torch.isfinite(bwd_loss_raw):
            raise RuntimeError("Non-finite backward SubTB loss in DualFlowModule.")
        bwd_loss_weighted = bwd_loss_raw * backward_weight
        metrics = {
            "subtb/backward_weight": torch.tensor(backward_weight, device=self.device, dtype=torch.float32),
            "subtb/backward_loss_raw": bwd_loss_raw.detach(),
            "subtb/backward_loss_weighted": bwd_loss_weighted.detach(),
            "subtb/backward_valid_ratio": bwd_valid_ratio.detach(),
        }
        metrics.update(self._prefix_metric_namespace(bwd_loss_metrics, namespace="bwd"))
        metrics.update(self._prefix_metric_namespace(bwd_reward_metrics, namespace="bwd"))
        return bwd_loss_weighted, metrics

    def _build_train_scalar_metrics(
        self,
        *,
        loss: torch.Tensor,
        current_beta: float,
    ) -> dict[str, torch.Tensor]:
        return {
            "train/reward_beta": torch.tensor(current_beta, device=loss.device, dtype=loss.dtype),
        }

    def _compute_training_loss(
        self,
        *,
        rollout: RolloutResult,
        rewards_raw: torch.Tensor,
        hit_mask: torch.Tensor,
        context: GraphEnvContext,
        encoded_context: EncodedPolicyContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current_beta = self.current_reward_beta()
        forward_subtb_loss, loss_metrics = self.subtb_loss_fn(
            fwd_rollout=rollout,
            rewards=rewards_raw,
            reward_beta=current_beta,
            hit_mask=hit_mask,
        )
        if not torch.isfinite(forward_subtb_loss):
            raise RuntimeError("Non-finite training loss in DualFlowModule.")
        backward_subtb_loss, backward_metrics = self._compute_backward_subtb_loss(
            context=context,
            encoded_context=encoded_context,
            current_beta=current_beta,
        )
        ranking_loss, ranking_metrics = self._compute_ranking_aux_loss(
            context=context,
            encoded_context=encoded_context,
        )
        loss = forward_subtb_loss + backward_subtb_loss + ranking_loss
        loss_metrics["subtb/forward_loss_raw"] = forward_subtb_loss.detach()
        loss_metrics.update(backward_metrics)
        loss_metrics.update(
            self._build_train_scalar_metrics(
                loss=loss,
                current_beta=current_beta,
            )
        )
        loss_metrics.update(ranking_metrics)
        return loss, loss_metrics

    def _compute_ranking_aux_loss(
        self,
        *,
        context: GraphEnvContext,
        encoded_context: EncodedPolicyContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weight = float(self.cfg.subtb_cfg.ranking_weight)
        node_tokens, _, question_tokens = encoded_context
        zero = torch.zeros((), device=node_tokens.device, dtype=torch.float32)
        if weight <= 0.0:
            metrics = {
                "subtb/ranking_raw": zero.detach(),
                "subtb/ranking_weighted": zero.detach(),
                "subtb/ranking_listwise": zero.detach(),
                "subtb/ranking_bce": zero.detach(),
                "subtb/ranking_margin": zero.detach(),
                "subtb/ranking_pos_mean": zero.detach(),
                "subtb/ranking_neg_mean": zero.detach(),
                "subtb/ranking_answer_mass": zero.detach(),
            }
            return zero, metrics

        node_scores_raw = self.policy.compute_node_priority_scores(
            env_context=context,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
        )
        node_scores = (node_scores_raw / float(self.cfg.subtb_cfg.ranking_temperature)).to(dtype=torch.float32)
        if node_scores.dim() != 1 or int(node_scores.numel()) != int(context.num_nodes_total):
            raise ValueError(
                "Node priority scores must be 1D with num_nodes_total entries, "
                f"got shape={tuple(node_scores.shape)}, num_nodes_total={context.num_nodes_total}."
            )

        node_graph_ids = context.node_batch.to(device=node_scores.device, dtype=torch.long)
        if int(node_graph_ids.numel()) != int(node_scores.numel()):
            raise ValueError(
                "node_batch length mismatch with node_scores in ranking auxiliary loss: "
                f"node_batch={int(node_graph_ids.numel())}, node_scores={int(node_scores.numel())}."
            )
        num_graphs = int(context.num_graphs)
        all_logsumexp = self._segment_logsumexp(
            values=node_scores,
            segment_ids=node_graph_ids,
            num_segments=num_graphs,
        )

        answer_mask = build_graph_node_membership_mask(
            local_indices=context.a_local_indices,
            ptr=context.a_ptr,
            node_ptr=context.node_ptr,
            num_nodes_total=context.num_nodes_total,
            device=node_scores.device,
            field_name="a_local_indices",
        )
        if int(answer_mask.numel()) != int(node_scores.numel()):
            raise ValueError(
                "Answer mask length mismatch with node scores in ranking auxiliary loss: "
                f"answer_mask={int(answer_mask.numel())}, node_scores={int(node_scores.numel())}."
            )

        has_positive = bool(answer_mask.any().item())
        has_negative = bool((~answer_mask).any().item())
        pos_mean = node_scores[answer_mask].mean() if has_positive else zero
        neg_mean = node_scores[~answer_mask].mean() if has_negative else zero

        if has_positive:
            answer_scores = node_scores[answer_mask]
            answer_graph_ids = node_graph_ids[answer_mask]
            pos_logsumexp = self._segment_logsumexp(
                values=answer_scores,
                segment_ids=answer_graph_ids,
                num_segments=num_graphs,
            )
        else:
            pos_logsumexp = torch.full(
                (num_graphs,),
                fill_value=float("-inf"),
                device=node_scores.device,
                dtype=torch.float32,
            )

        valid_graph = (~context.dummy_mask.to(device=node_scores.device, dtype=torch.bool)) & torch.isfinite(all_logsumexp)
        valid_graph = valid_graph & torch.isfinite(pos_logsumexp)
        if bool(valid_graph.any().item()):
            log_answer_mass = pos_logsumexp - all_logsumexp
            ranking_raw = (-log_answer_mass[valid_graph]).mean()
            answer_mass_mean = torch.exp(log_answer_mass[valid_graph]).mean()
        else:
            ranking_raw = zero
            answer_mass_mean = zero
        ranking_weighted = ranking_raw * weight

        metrics = {
            "subtb/ranking_raw": ranking_raw.detach(),
            "subtb/ranking_weighted": ranking_weighted.detach(),
            "subtb/ranking_listwise": ranking_raw.detach(),
            "subtb/ranking_bce": zero.detach(),
            "subtb/ranking_margin": zero.detach(),
            "subtb/ranking_pos_mean": pos_mean.detach(),
            "subtb/ranking_neg_mean": neg_mean.detach(),
            "subtb/ranking_answer_mass": answer_mass_mean.detach(),
        }
        return ranking_weighted, metrics

    @staticmethod
    def _segment_logsumexp(
        *,
        values: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        if values.dim() != 1:
            raise ValueError(f"values must be 1D for segment_logsumexp, got {tuple(values.shape)}")
        if segment_ids.dim() != 1:
            raise ValueError(f"segment_ids must be 1D for segment_logsumexp, got {tuple(segment_ids.shape)}")
        if int(values.numel()) != int(segment_ids.numel()):
            raise ValueError(
                "segment_logsumexp size mismatch between values and segment_ids: "
                f"values={int(values.numel())}, segment_ids={int(segment_ids.numel())}."
            )
        if num_segments < 0:
            raise ValueError(f"num_segments must be >= 0, got {num_segments}.")
        if num_segments == 0:
            return torch.empty((0,), device=values.device, dtype=torch.float32)

        ids = segment_ids.to(device=values.device, dtype=torch.long)
        if int(ids.numel()) == 0:
            return torch.full((num_segments,), fill_value=float("-inf"), device=values.device, dtype=torch.float32)
        if bool((ids < 0).any().item()) or bool((ids >= num_segments).any().item()):
            raise ValueError("segment_ids contains out-of-range values in segment_logsumexp.")

        values_fp32 = values.to(dtype=torch.float32)
        max_per_segment = torch.full(
            (num_segments,),
            fill_value=float("-inf"),
            device=values.device,
            dtype=torch.float32,
        )
        max_per_segment.scatter_reduce_(0, ids, values_fp32, reduce="amax", include_self=True)
        has_values = torch.zeros((num_segments,), dtype=torch.bool, device=values.device)
        has_values.scatter_(0, ids, True)
        safe_max = torch.where(has_values, max_per_segment, torch.zeros_like(max_per_segment))

        shifted = torch.exp(values_fp32 - safe_max.index_select(0, ids))
        sum_per_segment = torch.zeros((num_segments,), device=values.device, dtype=torch.float32)
        sum_per_segment.scatter_add_(0, ids, shifted)
        lse = safe_max + torch.log(sum_per_segment.clamp(min=torch.finfo(torch.float32).tiny))
        return torch.where(
            has_values,
            lse,
            torch.full((num_segments,), fill_value=float("-inf"), device=values.device, dtype=torch.float32),
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        if int(self.cfg.sampling_cfg.num_rollouts) < 4:
            raise ValueError("sampling_cfg.num_rollouts must be >= 4 for SubTB multi-rollout training.")

        base_context, _ = self.build_context(batch)
        fwd_rollout, rewards_raw, hit_mask, reward_metrics, encoded_context = self._sample_online_rollout(
            base_context=base_context
        )
        loss, loss_metrics = self._compute_training_loss(
            rollout=fwd_rollout,
            rewards_raw=rewards_raw,
            hit_mask=hit_mask,
            context=base_context,
            encoded_context=encoded_context,
        )
        self.log_train_metrics(loss, loss_metrics, reward_metrics, base_context.num_graphs)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_eval_step(batch=batch, stage="val", batch_idx=batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_eval_step(batch=batch, stage="test", batch_idx=batch_idx)

    def _encode_policy_context(self, context: GraphEnvContext) -> EncodedPolicyContext:
        return self.policy.encode_context(context)

    def _run_eval_rollout(
        self,
        *,
        context: GraphEnvContext,
        encoded_context: EncodedPolicyContext,
    ) -> RolloutResult:
        eval_cfg = self.cfg.eval_cfg
        return self.sampler.beam_search_forward(
            context,
            self.policy,
            beam_size=int(eval_cfg.beam_size),
            max_steps=int(eval_cfg.max_steps),
            require_done=bool(eval_cfg.require_done),
            diverse_penalty=float(eval_cfg.diverse_penalty),
            encoded_context=encoded_context,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> list[dict[str, Any]]:
        del batch_idx, dataloader_idx
        context, metadata = self.build_context(batch)
        rollout_context = self.override_forward_start_nodes(context, deterministic=True)
        encoded_context = self._encode_policy_context(rollout_context)
        rollout = self._run_eval_rollout(
            context=rollout_context,
            encoded_context=encoded_context,
        )
        return self.eval_exporter.build_predict_records(
            rollout=rollout,
            context=context,
            questions=metadata["questions"],
        )

    def configure_optimizers(self) -> dict[str, Any]:
        return build_optimizer_and_scheduler(
            model_parameters=self.named_parameters(),
            optimizer_cfg=asdict(self.cfg.optimizer_cfg),
            scheduler_cfg=asdict(self.cfg.scheduler_cfg),
            estimated_stepping_batches=(
                int(self.trainer.estimated_stepping_batches) if self.trainer is not None else None
            ),
        )

    def build_context(self, batch: Any) -> tuple[GraphEnvContext, dict[str, Any]]:
        prepared, metadata = self.prepare_batch(batch)
        return self.env_builder.build_context(prepared), metadata

    def prepare_batch(self, batch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.batch_adapter.prepare_batch(batch=batch, device=self.device)

    def override_forward_start_nodes(self, context: GraphEnvContext, *, deterministic: bool) -> GraphEnvContext:
        del deterministic
        return context

    def current_reward_beta(self) -> float:
        stop_cfg = self.cfg.env_cfg.stop
        beta_init = float(stop_cfg.reward_beta_init)
        beta_max = float(stop_cfg.reward_beta_max)
        anneal_steps = int(stop_cfg.reward_beta_anneal_steps)
        anneal_start = int(stop_cfg.reward_beta_anneal_start_step)
        schedule = str(stop_cfg.reward_beta_schedule).strip().lower()

        if beta_init <= 0 or beta_max <= 0:
            raise ValueError("stop.reward_beta_init and stop.reward_beta_max must be > 0.")
        if anneal_steps <= 0 or beta_init == beta_max:
            return beta_init

        progress_num = max(int(self.global_step) - anneal_start, 0)
        progress = min(float(progress_num) / float(anneal_steps), 1.0)
        if schedule == "linear":
            return beta_init + (beta_max - beta_init) * progress
        if schedule == "exponential":
            return math.exp(math.log(beta_init) + (math.log(beta_max) - math.log(beta_init)) * progress)
        raise ValueError("stop.reward_beta_schedule must be one of {'linear', 'exponential'}.")

    def compute_rewards(
        self,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        reward_beta: float | None = None,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.reward_engine.compute_rewards(
            stop_nodes_abs=stop_nodes_abs,
            context=context,
            reward_beta=self.current_reward_beta() if reward_beta is None else float(reward_beta),
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
        )

    def compute_hit_mask(
        self,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
    ) -> torch.Tensor:
        return self.reward_engine.compute_hit_mask(
            stop_nodes_abs,
            context,
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
        )

    def log_train_metrics(
        self,
        loss: torch.Tensor,
        loss_metrics: dict[str, Any],
        reward_metrics: dict[str, Any],
        batch_size: int,
    ) -> None:
        log_metric(
            self,
            "train/loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for key, value in {**loss_metrics, **reward_metrics}.items():
            metric_name = key if key.startswith("train/") else f"train/{key}"
            log_metric(
                self,
                metric_name,
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def shared_eval_step(self, *, batch: Any, stage: str, batch_idx: int = 0) -> None:
        context, _ = self.build_context(batch)
        rollout_every_n = int(self.cfg.eval_cfg.rollout_metrics_every_n_batches)
        run_rollout_metrics = rollout_every_n > 0 and (batch_idx % rollout_every_n == 0)

        beam_context = self.override_forward_start_nodes(context, deterministic=True)
        encoded_context = self._encode_policy_context(beam_context)
        beam_rollout = self._run_eval_rollout(
            context=beam_context,
            encoded_context=encoded_context,
        )
        rewards, reward_metrics = self.compute_rewards(beam_rollout.stop_nodes, context)

        scope = self.resolve_dataset_scope()
        prefix = f"{stage}/{scope}"
        metrics = self.eval_exporter.build_eval_metrics(
            prefix=prefix,
            reward_metrics=reward_metrics,
            rollout=beam_rollout,
            num_graphs=context.num_graphs,
            device=rewards.device,
        )
        if run_rollout_metrics:
            rollout_context = self.override_forward_start_nodes(context, deterministic=False)
            rollout = self.sampler.sample_forward(
                rollout_context,
                self.policy,
                deterministic=False,
                temperature=float(self.cfg.sampling_cfg.eval_sampling_temperature),
                encoded_context=encoded_context,
                collect_traces=False,
            )
            _, rollout_reward_metrics = self.compute_rewards(rollout.stop_nodes, context)
            metrics.update(
                self.eval_exporter.build_rollout_probe_metrics(
                    prefix=prefix,
                    reward_metrics=rollout_reward_metrics,
                    rollout=rollout,
                )
            )
        for key, value in metrics.items():
            log_metric(self, key, value, batch_size=context.num_graphs, on_step=False, on_epoch=True, sync_dist=True)

    def resolve_dataset_scope(self) -> str:
        trainer = self.trainer
        if trainer is None:
            return "full"
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return "full"
        dataset_cfg = getattr(datamodule, "dataset_cfg", None)
        if dataset_cfg is None:
            return "full"

        scope = str(
            dataset_cfg.get("dataset_scope", "")
            if isinstance(dataset_cfg, dict)
            else getattr(dataset_cfg, "dataset_scope", "")
        )
        scope = scope.strip().lower()
        if scope in {"full", "sub"}:
            return scope

        dataset_name = str(
            dataset_cfg.get("name", "") if isinstance(dataset_cfg, dict) else getattr(dataset_cfg, "name", "")
        )
        return "sub" if dataset_name.endswith("-sub") else "full"


__all__ = ["DualFlowModule"]
