from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule

from src.models.batch_adapter import DualFlowBatchAdapter
from src.models.components.high_energy_replay import HighEnergyReplayBuffer
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
        self.replay_buffer = HighEnergyReplayBuffer(self.cfg.training_cfg.replay_cfg)
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

    def _apply_replay_rollout_mix(
        self,
        *,
        base_context: GraphEnvContext,
        online_rollout: RolloutResult,
        rewards_online_raw: torch.Tensor,
        hit_mask_online: torch.Tensor,
        reward_metrics_online: dict[str, torch.Tensor],
        replay_alpha: float,
    ) -> tuple[
        RolloutResult,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        replay_offline_ratio = torch.tensor(0.0, device=self.device)
        replay_invalid_ratio = torch.tensor(0.0, device=self.device)
        replay_oracle_graph_ratio = torch.tensor(0.0, device=self.device)
        mixed_rollout = online_rollout
        rewards_raw = rewards_online_raw
        hit_mask = hit_mask_online
        reward_metrics: dict[str, torch.Tensor] = {}
        for key, value in reward_metrics_online.items():
            reward_metrics[f"train/online/{key}"] = value

        # Keep train/reward/* aligned with the rollout used for the training loss.
        def attach_final_metrics(final_metrics: dict[str, torch.Tensor]) -> None:
            for key, value in final_metrics.items():
                reward_metrics[key] = value
                reward_metrics[f"train/final/{key}"] = value

        attach_final_metrics(reward_metrics_online)

        replay_cfg = self.cfg.training_cfg.replay_cfg
        if not bool(replay_cfg.enabled):
            return (
                mixed_rollout,
                rewards_raw,
                hit_mask,
                reward_metrics,
                replay_offline_ratio,
                replay_invalid_ratio,
                replay_oracle_graph_ratio,
            )

        replay_batch = self.replay_buffer.build_and_sample(
            context=base_context,
            num_rollouts=int(self.cfg.sampling_cfg.num_rollouts),
            max_steps=int(self.cfg.sampling_cfg.max_steps),
            alpha=replay_alpha,
            stop_min_steps=int(self.cfg.sampling_cfg.stop_min_steps),
            device=self.device,
        )
        replay_mask_raw = replay_batch.use_offline_mask.to(device=self.device, dtype=torch.bool)
        replay_mask = replay_mask_raw
        replay_oracle_graph_ratio = replay_batch.graph_has_oracle.float().mean()
        if not bool(replay_mask_raw.any().item()):
            return (
                mixed_rollout,
                rewards_raw,
                hit_mask,
                reward_metrics,
                replay_offline_ratio,
                replay_invalid_ratio,
                replay_oracle_graph_ratio,
            )

        replay_rollout = self.sampler.evaluate_forced_paths(
            env_context=base_context,
            policy=self.policy,
            start_local_indices=replay_batch.start_local_indices,
            forced_edge_ids=replay_batch.edge_ids,
            path_lengths=replay_batch.path_lengths,
            collect_traces=True,
            use_visited_mask=bool(replay_cfg.track_visited_mask),
        )
        if replay_rollout.valid_mask is not None:
            replay_valid_mask = replay_rollout.valid_mask.to(device=self.device, dtype=torch.bool)
            if tuple(replay_valid_mask.shape) != tuple(replay_mask_raw.shape):
                raise ValueError(
                    "Replay valid_mask shape mismatch: "
                    f"got={tuple(replay_valid_mask.shape)} expected={tuple(replay_mask_raw.shape)}."
                )
            replay_mask = replay_mask_raw & replay_valid_mask
            replay_invalid_ratio = (replay_mask_raw & (~replay_valid_mask)).float().mean()
        replay_offline_ratio = replay_mask.float().mean()
        if not bool(replay_mask.any().item()):
            return (
                mixed_rollout,
                rewards_raw,
                hit_mask,
                reward_metrics,
                replay_offline_ratio,
                replay_invalid_ratio,
                replay_oracle_graph_ratio,
            )

        self.validate_rollout_merge_inputs(
            base_rollout=online_rollout,
            replay_rollout=replay_rollout,
            replay_mask=replay_mask,
        )
        mixed_rollout = self.merge_rollouts(
            base_rollout=online_rollout,
            replay_rollout=replay_rollout,
            replay_mask=replay_mask,
        )
        rewards_raw, reward_metrics_mixed = self.compute_rewards(mixed_rollout.stop_nodes, base_context)
        hit_mask = self.compute_hit_mask(mixed_rollout.stop_nodes, base_context)
        for key, value in reward_metrics_mixed.items():
            reward_metrics[f"train/mixed/{key}"] = value
        attach_final_metrics(reward_metrics_mixed)
        return (
            mixed_rollout,
            rewards_raw,
            hit_mask,
            reward_metrics,
            replay_offline_ratio,
            replay_invalid_ratio,
            replay_oracle_graph_ratio,
        )

    def _build_train_scalar_metrics(
        self,
        *,
        loss: torch.Tensor,
        replay_alpha: float,
        replay_offline_ratio: torch.Tensor,
        replay_invalid_ratio: torch.Tensor,
        replay_oracle_graph_ratio: torch.Tensor,
        current_beta: float,
    ) -> dict[str, torch.Tensor]:
        return {
            "train/qcbia_alpha": self.policy.current_qcbia_alpha(device=loss.device, dtype=loss.dtype),
            "train/reward_beta": torch.tensor(current_beta, device=loss.device, dtype=loss.dtype),
            "train/replay_alpha": torch.tensor(replay_alpha, device=loss.device, dtype=loss.dtype),
            "train/replay_offline_ratio": replay_offline_ratio.to(device=loss.device, dtype=loss.dtype),
            "train/replay_invalid_ratio": replay_invalid_ratio.to(device=loss.device, dtype=loss.dtype),
            "train/replay_oracle_graph_ratio": replay_oracle_graph_ratio.to(device=loss.device, dtype=loss.dtype),
        }

    def _compute_training_loss(
        self,
        *,
        rollout: RolloutResult,
        rewards_raw: torch.Tensor,
        hit_mask: torch.Tensor,
        context: GraphEnvContext,
        encoded_context: EncodedPolicyContext,
        replay_alpha: float,
        replay_offline_ratio: torch.Tensor,
        replay_invalid_ratio: torch.Tensor,
        replay_oracle_graph_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current_beta = self.current_reward_beta()
        loss, loss_metrics = self.subtb_loss_fn(
            fwd_rollout=rollout,
            rewards=rewards_raw,
            reward_beta=current_beta,
            hit_mask=hit_mask,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss in DualFlowModule.")
        ranking_loss, ranking_metrics = self._compute_ranking_aux_loss(
            context=context,
            encoded_context=encoded_context,
        )
        loss = loss + ranking_loss
        loss_metrics.update(
            self._build_train_scalar_metrics(
                loss=loss,
                replay_alpha=replay_alpha,
                replay_offline_ratio=replay_offline_ratio,
                replay_invalid_ratio=replay_invalid_ratio,
                replay_oracle_graph_ratio=replay_oracle_graph_ratio,
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
        node_tokens, _, question_tokens = encoded_context
        node_scores_raw = self.policy.compute_node_priority_scores(
            env_context=context,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
        )
        weight = float(self.cfg.subtb_cfg.ranking_weight)
        listwise_weight = float(self.cfg.subtb_cfg.ranking_listwise_weight)
        bce_weight = float(self.cfg.subtb_cfg.ranking_bce_weight)
        temperature = float(self.cfg.subtb_cfg.ranking_temperature)
        margin = float(self.cfg.subtb_cfg.ranking_margin)
        hard_negative_k = int(self.cfg.subtb_cfg.ranking_hard_negative_k)

        node_scores = node_scores_raw / temperature
        answer_mask = build_graph_node_membership_mask(
            local_indices=context.a_local_indices,
            ptr=context.a_ptr,
            node_ptr=context.node_ptr,
            num_nodes_total=context.num_nodes_total,
            device=node_scores.device,
            field_name="a_local_indices",
        )
        has_positive = bool(answer_mask.any().item())
        has_negative = bool((~answer_mask).any().item())
        if (not has_positive) or (not has_negative) or weight <= 0.0:
            zero = node_scores.new_zeros(())
            pos_mean = node_scores[answer_mask].mean() if has_positive else zero
            neg_mean = node_scores[~answer_mask].mean() if has_negative else zero
            metrics = {
                "subtb/ranking_weight": torch.tensor(weight, device=zero.device, dtype=zero.dtype),
                "subtb/ranking_listwise_weight": torch.tensor(listwise_weight, device=zero.device, dtype=zero.dtype),
                "subtb/ranking_bce_weight": torch.tensor(bce_weight, device=zero.device, dtype=zero.dtype),
                "subtb/ranking_raw": zero.detach(),
                "subtb/ranking_weighted": zero.detach(),
                "subtb/ranking_listwise": zero.detach(),
                "subtb/ranking_bce": zero.detach(),
                "subtb/ranking_margin": zero.detach(),
                "subtb/ranking_pos_mean": pos_mean.detach(),
                "subtb/ranking_neg_mean": neg_mean.detach(),
            }
            return zero, metrics

        listwise_terms: list[torch.Tensor] = []
        bce_terms: list[torch.Tensor] = []
        margin_terms: list[torch.Tensor] = []
        num_graphs = int(context.num_graphs)
        for graph_idx in range(num_graphs):
            node_start = int(context.node_ptr[graph_idx].item())
            node_end = int(context.node_ptr[graph_idx + 1].item())
            if node_end <= node_start:
                continue
            answer_start = int(context.a_ptr[graph_idx].item())
            answer_end = int(context.a_ptr[graph_idx + 1].item())
            if answer_end <= answer_start:
                continue
            graph_scores = node_scores[node_start:node_end]
            local_answers = context.a_local_indices[answer_start:answer_end].to(device=node_scores.device, dtype=torch.long)
            if bool((local_answers < 0).any().item()) or bool((local_answers >= int(graph_scores.numel())).any().item()):
                raise ValueError("a_local_indices out of range in ranking auxiliary loss.")
            positive_local = torch.zeros((int(graph_scores.numel()),), dtype=torch.bool, device=node_scores.device)
            positive_local.scatter_(0, local_answers, True)
            pos_scores = graph_scores[positive_local]
            neg_scores = graph_scores[~positive_local]
            if int(pos_scores.numel()) == 0 or int(neg_scores.numel()) == 0:
                continue

            if listwise_weight > 0.0:
                listwise = -(torch.logsumexp(pos_scores, dim=0) - torch.logsumexp(graph_scores, dim=0))
                listwise_terms.append(listwise)

            if bce_weight > 0.0:
                pos_bce = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                neg_bce = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                bce_terms.append((pos_bce + neg_bce) * 0.5)

            if margin > 0.0 and hard_negative_k > 0:
                topk = min(hard_negative_k, int(neg_scores.numel()))
                hardest = torch.topk(neg_scores, k=topk, dim=0).values
                pos_anchor = pos_scores.mean()
                margin_term = torch.relu(margin + hardest - pos_anchor).mean()
                margin_terms.append(margin_term)

        if len(listwise_terms) == 0 and len(bce_terms) == 0 and len(margin_terms) == 0:
            zero = node_scores.new_zeros(())
            metrics = {
                "subtb/ranking_weight": torch.tensor(weight, device=zero.device, dtype=zero.dtype),
                "subtb/ranking_listwise_weight": torch.tensor(listwise_weight, device=zero.device, dtype=zero.dtype),
                "subtb/ranking_bce_weight": torch.tensor(bce_weight, device=zero.device, dtype=zero.dtype),
                "subtb/ranking_raw": zero.detach(),
                "subtb/ranking_weighted": zero.detach(),
                "subtb/ranking_listwise": zero.detach(),
                "subtb/ranking_bce": zero.detach(),
                "subtb/ranking_margin": zero.detach(),
                "subtb/ranking_pos_mean": node_scores[answer_mask].mean().detach(),
                "subtb/ranking_neg_mean": node_scores[~answer_mask].mean().detach(),
            }
            return zero, metrics

        zero = node_scores.new_zeros(())
        listwise_loss = torch.stack(listwise_terms).mean() if len(listwise_terms) > 0 else zero
        bce_loss = torch.stack(bce_terms).mean() if len(bce_terms) > 0 else zero
        margin_loss = torch.stack(margin_terms).mean() if len(margin_terms) > 0 else node_scores.new_zeros(())
        ranking_raw = listwise_weight * listwise_loss + bce_weight * bce_loss + margin_loss
        ranking_weighted = ranking_raw * weight
        metrics = {
            "subtb/ranking_weight": torch.tensor(weight, device=node_scores.device, dtype=node_scores.dtype),
            "subtb/ranking_listwise_weight": torch.tensor(
                listwise_weight,
                device=node_scores.device,
                dtype=node_scores.dtype,
            ),
            "subtb/ranking_bce_weight": torch.tensor(
                bce_weight,
                device=node_scores.device,
                dtype=node_scores.dtype,
            ),
            "subtb/ranking_raw": ranking_raw.detach(),
            "subtb/ranking_weighted": ranking_weighted.detach(),
            "subtb/ranking_listwise": listwise_loss.detach(),
            "subtb/ranking_bce": bce_loss.detach(),
            "subtb/ranking_margin": margin_loss.detach(),
            "subtb/ranking_pos_mean": node_scores[answer_mask].mean().detach(),
            "subtb/ranking_neg_mean": node_scores[~answer_mask].mean().detach(),
        }
        return ranking_weighted, metrics

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        if int(self.cfg.sampling_cfg.num_rollouts) < 4:
            raise ValueError("sampling_cfg.num_rollouts must be >= 4 for SubTB multi-rollout training.")
        self.policy.set_training_step(int(self.global_step))

        base_context, _ = self.build_context(batch)
        replay_alpha = self.current_replay_alpha()
        (
            online_rollout,
            rewards_online_raw,
            hit_mask_online,
            reward_metrics_online,
            encoded_context,
        ) = self._sample_online_rollout(
            base_context=base_context
        )
        (
            fwd_rollout,
            rewards_raw,
            hit_mask,
            reward_metrics,
            replay_offline_ratio,
            replay_invalid_ratio,
            replay_oracle_graph_ratio,
        ) = self._apply_replay_rollout_mix(
            base_context=base_context,
            online_rollout=online_rollout,
            rewards_online_raw=rewards_online_raw,
            hit_mask_online=hit_mask_online,
            reward_metrics_online=reward_metrics_online,
            replay_alpha=replay_alpha,
        )
        loss, loss_metrics = self._compute_training_loss(
            rollout=fwd_rollout,
            rewards_raw=rewards_raw,
            hit_mask=hit_mask,
            context=base_context,
            encoded_context=encoded_context,
            replay_alpha=replay_alpha,
            replay_offline_ratio=replay_offline_ratio,
            replay_invalid_ratio=replay_invalid_ratio,
            replay_oracle_graph_ratio=replay_oracle_graph_ratio,
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

    def current_replay_alpha(self) -> float:
        replay_cfg = self.cfg.training_cfg.replay_cfg
        if not bool(replay_cfg.enabled):
            return 0.0
        alpha_init = float(replay_cfg.alpha_init)
        alpha_final = float(replay_cfg.alpha_final)
        anneal_epochs = int(replay_cfg.alpha_anneal_epochs)
        if anneal_epochs <= 0:
            return max(min(alpha_final, 1.0), 0.0)
        progress = min(max(float(self.current_epoch), 0.0) / float(anneal_epochs), 1.0)
        alpha = alpha_init + (alpha_final - alpha_init) * progress
        return max(min(alpha, 1.0), 0.0)

    @staticmethod
    def validate_rollout_merge_inputs(
        *,
        base_rollout: RolloutResult,
        replay_rollout: RolloutResult,
        replay_mask: torch.Tensor,
    ) -> None:
        if replay_mask.dim() != 2:
            raise ValueError(f"replay_mask must be [B, K], got shape={tuple(replay_mask.shape)}")
        expected_shape = tuple(replay_mask.shape)
        required_dense_fields = (
            "log_pf_sum",
            "stop_nodes",
            "num_moves",
            "num_steps",
            "stop_reason",
        )
        for field_name in required_dense_fields:
            base_value = getattr(base_rollout, field_name)
            replay_value = getattr(replay_rollout, field_name)
            if base_value is None or replay_value is None:
                raise ValueError(f"Rollout merge requires `{field_name}` on both base and replay rollouts.")
            if tuple(base_value.shape) != expected_shape or tuple(replay_value.shape) != expected_shape:
                raise ValueError(
                    f"Rollout merge shape mismatch for `{field_name}`: "
                    f"base={tuple(base_value.shape)} replay={tuple(replay_value.shape)} expected={expected_shape}."
                )
        required_trace_fields = ("log_pf_steps", "log_pb_steps", "log_f_steps")
        for field_name in required_trace_fields:
            base_value = getattr(base_rollout, field_name)
            replay_value = getattr(replay_rollout, field_name)
            if base_value is None or replay_value is None:
                raise ValueError(f"Rollout merge requires `{field_name}` traces on both rollouts.")
            if base_value.dim() != 3 or replay_value.dim() != 3:
                raise ValueError(
                    f"Rollout trace `{field_name}` must be 3D [B,K,T], "
                    f"got base={tuple(base_value.shape)} replay={tuple(replay_value.shape)}."
                )
            if tuple(base_value.shape[:2]) != expected_shape or tuple(replay_value.shape[:2]) != expected_shape:
                raise ValueError(
                    f"Rollout trace shape mismatch for `{field_name}`: "
                    f"base={tuple(base_value.shape)} replay={tuple(replay_value.shape)} expected_prefix={expected_shape}."
                )
            if int(base_value.size(-1)) != int(replay_value.size(-1)):
                raise ValueError(
                    f"Rollout trace horizon mismatch for `{field_name}`: "
                    f"base_T={int(base_value.size(-1))} replay_T={int(replay_value.size(-1))}."
                )
        optional_mask_fields = ("valid_mask",)
        for field_name in optional_mask_fields:
            base_value = getattr(base_rollout, field_name)
            replay_value = getattr(replay_rollout, field_name)
            if base_value is not None and tuple(base_value.shape) != expected_shape:
                raise ValueError(
                    f"Rollout optional mask `{field_name}` has invalid base shape: "
                    f"{tuple(base_value.shape)} expected={expected_shape}."
                )
            if replay_value is not None and tuple(replay_value.shape) != expected_shape:
                raise ValueError(
                    f"Rollout optional mask `{field_name}` has invalid replay shape: "
                    f"{tuple(replay_value.shape)} expected={expected_shape}."
                )

    @staticmethod
    def merge_rollouts(
        *,
        base_rollout: RolloutResult,
        replay_rollout: RolloutResult,
        replay_mask: torch.Tensor,
    ) -> RolloutResult:
        if replay_mask.dim() != 2:
            raise ValueError(f"replay_mask must be [B, K], got shape={tuple(replay_mask.shape)}")
        mask = replay_mask.to(dtype=torch.bool)

        def pick_required(base: torch.Tensor, replay: torch.Tensor) -> torch.Tensor:
            local_mask = mask
            while local_mask.dim() < base.dim():
                local_mask = local_mask.unsqueeze(-1)
            return torch.where(local_mask, replay, base)

        def pick_optional(base: torch.Tensor | None, replay: torch.Tensor | None) -> torch.Tensor | None:
            if base is None or replay is None:
                return base
            return pick_required(base, replay)

        return RolloutResult(
            log_pf_sum=pick_required(base_rollout.log_pf_sum, replay_rollout.log_pf_sum),
            stop_nodes=pick_required(base_rollout.stop_nodes, replay_rollout.stop_nodes),
            num_moves=pick_required(base_rollout.num_moves, replay_rollout.num_moves),
            num_steps=pick_required(base_rollout.num_steps, replay_rollout.num_steps),
            stop_reason=pick_required(base_rollout.stop_reason, replay_rollout.stop_reason),
            actions=pick_optional(base_rollout.actions, replay_rollout.actions),
            log_pf_steps=pick_optional(base_rollout.log_pf_steps, replay_rollout.log_pf_steps),
            log_pb_steps=pick_optional(base_rollout.log_pb_steps, replay_rollout.log_pb_steps),
            log_f_steps=pick_optional(base_rollout.log_f_steps, replay_rollout.log_f_steps),
            log_pb_sum=pick_optional(base_rollout.log_pb_sum, replay_rollout.log_pb_sum),
            valid_mask=pick_optional(base_rollout.valid_mask, replay_rollout.valid_mask),
            policy_metrics=base_rollout.policy_metrics,
        )

    def compute_rewards(
        self, stop_nodes_abs: torch.Tensor, context: GraphEnvContext
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.reward_engine.compute_rewards(
            stop_nodes_abs=stop_nodes_abs,
            context=context,
            reward_beta=self.current_reward_beta(),
        )

    def compute_hit_mask(self, stop_nodes_abs: torch.Tensor, context: GraphEnvContext) -> torch.Tensor:
        return self.reward_engine.compute_hit_mask(stop_nodes_abs, context)

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
            on_step=True,
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
                on_step=True,
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
