from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Any

import torch
import torch.nn.functional as F
from src.models.algorithms.base import (
    AlgorithmModule,
    BatchPayload,
    build_optimizer_and_scheduler,
)

from src.models.policy import DualFlowPolicy
from src.models.rollout import (
    STOP_REASON_ACTION,
    STOP_REASON_DEAD_END,
    STOP_REASON_MAX_STEPS_REACHED,
)
from src.models.rollout import RolloutResult, RolloutSampler
from src.models.configs.dual_flow_cfg import DualFlowConfig
from src.models.configs.environment import EnvironmentConfig
from src.models.configs.objective import SubTBConfig
from src.models.configs.policy import PolicyConfig
from src.models.configs.search import BeamSearchConfig, RolloutConfig
from src.models.configs.training import OptimizerConfig, SchedulerConfig
from src.metrics.dual_flow.export import (
    DualFlowRolloutExporter,
    GraphExportInputs,
    RolloutExportInputs,
)
from src.models.environment.builder import GraphEnvironmentBuilder
from src.models.environment.context import GraphEnvContext
from src.models.environment.ops import (
    build_node_membership_mask,
    has_super_source_layout,
)
from src.models.objectives.subtb_loss import SubTrajectoryBalanceLoss
from src.models.reward.reward_engine import DualFlowRewardEngine
from src.utils.logging_utils import log_metric
from src.utils.segment_ops import segment_logsumexp_1d

EncodedPolicyContext = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DualFlowModule(AlgorithmModule):
    """DualFlow orchestration module with strict typed configs and automatic optimization."""

    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        policy_cfg: PolicyConfig,
        sampling_cfg: RolloutConfig,
        eval_cfg: BeamSearchConfig,
        subtb_cfg: SubTBConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
    ) -> None:
        super().__init__(optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg)
        self.cfg = DualFlowConfig(
            env_cfg=env_cfg,
            policy_cfg=policy_cfg,
            sampling_cfg=sampling_cfg,
            eval_cfg=eval_cfg,
            subtb_cfg=subtb_cfg,
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
        self.reward_engine = DualFlowRewardEngine(stop_cfg=self.cfg.env_cfg.stop)
        self.eval_exporter = DualFlowRolloutExporter()

    @staticmethod
    def _forward_start_step_offset(context: GraphEnvContext) -> int:
        if has_super_source_layout(
            node_ptr=context.node_ptr,
            node_global_ids=context.node_global_ids,
            num_nodes_total=context.num_nodes_total,
            device=context.node_ptr.device,
        ):
            return 1
        return 0

    @staticmethod
    def _build_rollout_export_inputs(rollout: RolloutResult) -> RolloutExportInputs:
        return RolloutExportInputs(
            stop_nodes=rollout.stop_nodes,
            num_moves=rollout.num_moves,
            log_pf_sum=rollout.log_pf_sum,
            stop_reason=rollout.stop_reason,
        )

    @staticmethod
    def _build_graph_export_inputs(context: GraphEnvContext) -> GraphExportInputs:
        return GraphExportInputs(
            q_local_indices=context.q_local_indices,
            q_ptr=context.q_ptr,
            a_ptr=context.a_ptr,
            node_ptr=context.node_ptr,
            node_global_ids=context.node_global_ids,
            answer_entity_ids=context.answer_entity_ids,
            answer_ptr=context.answer_ptr,
            sample_ids=list(context.sample_ids),
            num_graphs=int(context.num_graphs),
        )

    def _sample_online_rollout(
        self,
        *,
        base_context: GraphEnvContext,
    ) -> tuple[
        RolloutResult,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        EncodedPolicyContext,
    ]:
        rollout_context = self.override_forward_start_nodes(
            base_context, deterministic=False
        )
        encoded_context = self._encode_policy_context(rollout_context)
        online_rollout = self.sampler.sample_forward(
            rollout_context,
            self.policy,
            flow_direction="forward",
            deterministic=False,
            encoded_context=encoded_context,
        )
        rollout_diag, done_mask_online = self._build_rollout_diagnostics(
            rollout=online_rollout,
            context=base_context,
            flow_direction="forward",
        )
        rewards_online_raw, reward_metrics_online = self.compute_rewards(
            online_rollout.stop_nodes,
            base_context,
            terminal_done_mask=done_mask_online,
        )
        reward_metrics_online.update(rollout_diag)
        hit_mask_online = self.compute_hit_mask(
            online_rollout.stop_nodes,
            base_context,
            terminal_done_mask=done_mask_online,
        )
        return (
            online_rollout,
            rewards_online_raw,
            hit_mask_online,
            reward_metrics_online,
            encoded_context,
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = (
            checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        )
        if not isinstance(state_dict, dict):
            return
        if not any(
            "._orig_mod." in key or key.startswith("_orig_mod.") for key in state_dict
        ):
            return
        stripped: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "", 1)
            else:
                new_key = key.replace("._orig_mod.", ".", 1)
            stripped[new_key] = value
        checkpoint["state_dict"] = stripped

    @staticmethod
    def _prefix_metric_namespace(
        metrics: dict[str, torch.Tensor],
        *,
        namespace: str,
    ) -> dict[str, torch.Tensor]:
        return {f"{namespace}/{key}": value for key, value in metrics.items()}

    def _sample_backward_rollout(
        self,
        *,
        base_context: GraphEnvContext,
        encoded_context: EncodedPolicyContext,
        current_beta: float,
    ) -> tuple[
        RolloutResult, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor
    ]:
        num_graphs = int(base_context.num_graphs)
        a_counts = (base_context.a_ptr[1:] - base_context.a_ptr[:-1]).to(
            device=self.device, dtype=torch.long
        )
        q_counts = (base_context.q_ptr[1:] - base_context.q_ptr[:-1]).to(
            device=self.device, dtype=torch.long
        )
        if not has_super_source_layout(
            node_ptr=base_context.node_ptr,
            node_global_ids=base_context.node_global_ids,
            num_nodes_total=base_context.num_nodes_total,
            device=self.device,
        ):
            raise ValueError(
                "Backward SubTB requires dual super-source layout (forward/backward super nodes per graph)."
            )
        valid_start_graph = a_counts > 0
        valid_target_graph = q_counts > 0
        if bool((~valid_start_graph).any().item()):
            raise ValueError(
                "a_local_indices must be non-empty for every graph when backward_weight > 0."
            )
        if bool((~valid_target_graph).any().item()):
            raise ValueError(
                "q_local_indices must be non-empty for every graph when backward_weight > 0."
            )
        valid_graph = valid_start_graph & valid_target_graph
        # Backward rollout must walk reversed graph edges (adj_t_bwd as rollout topology).
        bwd_context = replace(
            base_context,
            adj_t_fwd=base_context.adj_t_bwd,
            adj_t_bwd=base_context.adj_t_fwd,
        )
        bwd_rollout = self.sampler.sample_forward(
            bwd_context,
            self.policy,
            flow_direction="backward",
            deterministic=False,
            encoded_context=encoded_context,
        )
        num_rollouts = int(bwd_rollout.stop_nodes.size(1))
        valid_by_graph = valid_graph.view(num_graphs, 1).expand(
            num_graphs, num_rollouts
        )
        if bwd_rollout.valid_mask is None:
            bwd_valid_mask = valid_by_graph
        else:
            bwd_valid_mask = (
                bwd_rollout.valid_mask.to(device=self.device, dtype=torch.bool)
                & valid_by_graph
            )
        bwd_rollout = replace(bwd_rollout, valid_mask=bwd_valid_mask)
        bwd_rollout_diag, bwd_done_mask = self._build_rollout_diagnostics(
            rollout=bwd_rollout,
            context=base_context,
            flow_direction="backward",
        )
        bwd_done_mask = bwd_done_mask & bwd_valid_mask
        bwd_rewards_raw, bwd_reward_metrics = self.compute_rewards(
            stop_nodes_abs=bwd_rollout.stop_nodes,
            context=base_context,
            reward_beta=current_beta,
            target_local_indices=base_context.q_local_indices,
            target_ptr=base_context.q_ptr,
            target_field_name="q_local_indices",
            terminal_done_mask=bwd_done_mask,
        )
        bwd_reward_metrics.update(bwd_rollout_diag)
        bwd_hit_mask = self.compute_hit_mask(
            bwd_rollout.stop_nodes,
            base_context,
            target_local_indices=base_context.q_local_indices,
            target_ptr=base_context.q_ptr,
            target_field_name="q_local_indices",
            terminal_done_mask=bwd_done_mask,
        )
        bwd_valid_ratio = bwd_valid_mask.float().mean()
        return (
            bwd_rollout,
            bwd_rewards_raw,
            bwd_hit_mask,
            bwd_reward_metrics,
            bwd_valid_ratio,
        )

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
            return zero, {}
        (
            bwd_rollout,
            bwd_rewards_raw,
            bwd_hit_mask,
            bwd_reward_metrics,
            bwd_valid_ratio,
        ) = self._sample_backward_rollout(
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
            "subtb/backward_weight": torch.tensor(
                backward_weight, device=self.device, dtype=torch.float32
            ),
            "subtb/backward_loss_raw": bwd_loss_raw.detach(),
            "subtb/backward_loss_weighted": bwd_loss_weighted.detach(),
            "subtb/backward_valid_ratio": bwd_valid_ratio.detach(),
        }
        metrics.update(self._prefix_metric_namespace(bwd_loss_metrics, namespace="bwd"))
        metrics.update(
            self._prefix_metric_namespace(bwd_reward_metrics, namespace="bwd")
        )
        return bwd_loss_weighted, metrics

    def _build_train_scalar_metrics(
        self,
        *,
        loss: torch.Tensor,
        current_beta: float,
    ) -> dict[str, torch.Tensor]:
        stop_cfg = self.cfg.env_cfg.stop
        beta_init = float(stop_cfg.reward_beta_init)
        beta_max = float(stop_cfg.reward_beta_max)
        anneal_steps = int(stop_cfg.reward_beta_anneal_steps)
        if anneal_steps <= 0 or beta_init == beta_max:
            return {}
        return {
            "train/reward_beta": torch.tensor(
                current_beta, device=loss.device, dtype=loss.dtype
            )
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
        doob_h_loss, doob_h_metrics = self._compute_doob_h_aux_loss(
            rollout=rollout,
            rewards_raw=rewards_raw,
            context=context,
            encoded_context=encoded_context,
        )
        loss = forward_subtb_loss + backward_subtb_loss + ranking_loss + doob_h_loss
        loss_metrics["subtb/forward_loss_raw"] = forward_subtb_loss.detach()
        loss_metrics.update(backward_metrics)
        loss_metrics.update(
            self._build_train_scalar_metrics(
                loss=loss,
                current_beta=current_beta,
            )
        )
        loss_metrics.update(ranking_metrics)
        loss_metrics.update(doob_h_metrics)
        return loss, loss_metrics

    def _compute_doob_h_aux_loss(
        self,
        *,
        rollout: RolloutResult,
        rewards_raw: torch.Tensor,
        context: GraphEnvContext,
        encoded_context: EncodedPolicyContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weight = float(self.cfg.subtb_cfg.doob_h_weight)
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        if weight <= 0.0:
            metrics = {
                "subtb/doob_h_raw": zero.detach(),
                "subtb/doob_h_weighted": zero.detach(),
                "subtb/doob_h_valid_ratio": zero.detach(),
                "subtb/doob_h_target_mean": zero.detach(),
                "subtb/doob_h_pred_mean": zero.detach(),
                "subtb/doob_h_mc_return_mean": zero.detach(),
            }
            return zero, metrics
        state_nodes_steps = rollout.state_nodes_steps
        if state_nodes_steps is None:
            raise ValueError(
                "doob_h_weight > 0 requires rollout.state_nodes_steps traces."
            )
        if state_nodes_steps.dim() != 3:
            raise ValueError(
                f"rollout.state_nodes_steps must be 3D [B, R, T], got {tuple(state_nodes_steps.shape)}."
            )
        if rewards_raw.dim() == 1:
            rewards_2d = rewards_raw.unsqueeze(1)
        elif rewards_raw.dim() == 2:
            rewards_2d = rewards_raw
        else:
            raise ValueError(
                f"rewards_raw must be 1D or 2D, got shape={tuple(rewards_raw.shape)}"
            )
        if tuple(rewards_2d.shape) != tuple(rollout.num_steps.shape):
            raise ValueError(
                "rewards_raw shape mismatch with rollout.num_steps for doob-h loss: "
                f"rewards={tuple(rewards_2d.shape)}, num_steps={tuple(rollout.num_steps.shape)}."
            )

        node_tokens, _, question_tokens = encoded_context
        node_log_h = self.policy.compute_node_log_h(
            env_context=context,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
        ).to(device=self.device, dtype=torch.float32)
        if node_log_h.dim() != 1 or int(node_log_h.numel()) != int(
            context.num_nodes_total
        ):
            raise ValueError(
                "policy node_log_h shape mismatch in doob-h loss: "
                f"node_log_h={tuple(node_log_h.shape)}, num_nodes_total={int(context.num_nodes_total)}."
            )

        stop_cfg = self.cfg.env_cfg.stop
        reward_base = float(stop_cfg.reward_base)
        reward_epsilon = float(stop_cfg.reward_epsilon)
        if reward_base <= 0.0 or reward_epsilon <= 0.0:
            raise ValueError(
                "stop reward_base/reward_epsilon must be > 0 for doob-h loss."
            )
        if reward_base <= reward_epsilon:
            raise ValueError(
                "doob-h MC regression requires reward_base > reward_epsilon to normalize returns into [0, 1]."
            )

        # Monte Carlo target for h(S): return-to-go normalized into [0, 1].
        denom = reward_base - reward_epsilon
        mc_return = (
            (rewards_2d.to(device=self.device, dtype=torch.float32) - reward_epsilon)
            / denom
        ).clamp(0.0, 1.0)
        target_h = mc_return.unsqueeze(-1).expand_as(
            state_nodes_steps.to(device=self.device, dtype=torch.float32)
        )

        num_steps = rollout.num_steps.to(device=self.device, dtype=torch.long)
        step_idx = torch.arange(
            state_nodes_steps.size(-1), device=self.device, dtype=torch.long
        ).view(1, 1, -1)
        valid_t = step_idx < num_steps.unsqueeze(-1)
        state_nodes = state_nodes_steps.to(device=self.device, dtype=torch.long)
        valid_nodes = state_nodes >= 0
        train_mask = valid_t & valid_nodes

        safe_nodes = state_nodes.clamp(
            min=0, max=max(int(context.num_nodes_total) - 1, 0)
        )
        pred_log_h = node_log_h.index_select(0, safe_nodes.view(-1)).view_as(
            state_nodes_steps
        )
        pred_h = pred_log_h.exp().clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        pred_logits = torch.logit(pred_h, eps=1.0e-6)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits,
            target_h,
            reduction="none",
        )

        if bool(train_mask.any().item()):
            raw_loss = bce[train_mask].mean()
            target_h_mean = target_h[train_mask].mean()
            pred_h_mean = pred_h[train_mask].mean()
            mc_return_mean = mc_return.mean()
        else:
            raw_loss = zero
            target_h_mean = zero
            pred_h_mean = zero
            mc_return_mean = zero
        weighted_loss = raw_loss * weight
        metrics = {
            "subtb/doob_h_raw": raw_loss.detach(),
            "subtb/doob_h_weighted": weighted_loss.detach(),
            "subtb/doob_h_valid_ratio": train_mask.float().mean().detach(),
            "subtb/doob_h_target_mean": target_h_mean.detach(),
            "subtb/doob_h_pred_mean": pred_h_mean.detach(),
            "subtb/doob_h_mc_return_mean": mc_return_mean.detach(),
        }
        return weighted_loss, metrics

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
            return zero, {}

        node_scores_raw = self.policy.compute_node_priority_scores(
            env_context=context,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
        )
        node_scores = (
            node_scores_raw / float(self.cfg.subtb_cfg.ranking_temperature)
        ).to(dtype=torch.float32)
        if node_scores.dim() != 1 or int(node_scores.numel()) != int(
            context.num_nodes_total
        ):
            raise ValueError(
                "Node priority scores must be 1D with num_nodes_total entries, "
                f"got shape={tuple(node_scores.shape)}, num_nodes_total={context.num_nodes_total}."
            )

        node_graph_ids = context.node_batch.to(
            device=node_scores.device, dtype=torch.long
        )
        if int(node_graph_ids.numel()) != int(node_scores.numel()):
            raise ValueError(
                "node_batch length mismatch with node_scores in ranking auxiliary loss: "
                f"node_batch={int(node_graph_ids.numel())}, node_scores={int(node_scores.numel())}."
            )
        num_graphs = int(context.num_graphs)
        all_logsumexp, _ = segment_logsumexp_1d(
            values=node_scores,
            segment_ids=node_graph_ids,
            num_segments=num_graphs,
            dtype=torch.float32,
            ignore_non_finite=False,
            empty_value=float("-inf"),
        )

        answer_mask = build_node_membership_mask(
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
            pos_logsumexp, _ = segment_logsumexp_1d(
                values=answer_scores,
                segment_ids=answer_graph_ids,
                num_segments=num_graphs,
                dtype=torch.float32,
                ignore_non_finite=False,
                empty_value=float("-inf"),
            )
        else:
            pos_logsumexp = torch.full(
                (num_graphs,),
                fill_value=float("-inf"),
                device=node_scores.device,
                dtype=torch.float32,
            )

        valid_graph = (
            ~context.dummy_mask.to(device=node_scores.device, dtype=torch.bool)
        ) & torch.isfinite(all_logsumexp)
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
            "subtb/ranking_pos_mean": pos_mean.detach(),
            "subtb/ranking_neg_mean": neg_mean.detach(),
            "subtb/ranking_answer_mass": answer_mass_mean.detach(),
        }
        return ranking_weighted, metrics

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        if int(self.cfg.sampling_cfg.num_rollouts) < 4:
            raise ValueError(
                "sampling_cfg.num_rollouts must be >= 4 for SubTB multi-rollout training."
            )

        base_context, _ = self.build_context(batch)
        fwd_rollout, rewards_raw, hit_mask, reward_metrics, encoded_context = (
            self._sample_online_rollout(base_context=base_context)
        )
        loss, loss_metrics = self._compute_training_loss(
            rollout=fwd_rollout,
            rewards_raw=rewards_raw,
            hit_mask=hit_mask,
            context=base_context,
            encoded_context=encoded_context,
        )
        self.log_train_metrics(
            loss, loss_metrics, reward_metrics, base_context.num_graphs
        )
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
            flow_direction="forward",
            beam_size=int(eval_cfg.beam_size),
            max_steps=int(eval_cfg.max_steps),
            require_done=bool(eval_cfg.require_done),
            diverse_penalty=float(eval_cfg.diverse_penalty),
            candidate_expansion_factor=int(eval_cfg.candidate_expansion_factor),
            encoded_context=encoded_context,
        )

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> list[dict[str, Any]]:
        del batch_idx, dataloader_idx
        context, metadata = self.build_context(batch)
        rollout_context = self.override_forward_start_nodes(context, deterministic=True)
        encoded_context = self._encode_policy_context(rollout_context)
        rollout = self._run_eval_rollout(
            context=rollout_context,
            encoded_context=encoded_context,
        )
        questions = metadata.get("questions")
        if not isinstance(questions, list) or len(questions) != int(context.num_graphs):
            questions = ["" for _ in range(int(context.num_graphs))]
        export_rollout = self._build_rollout_export_inputs(rollout)
        export_graph = self._build_graph_export_inputs(context)
        step_offset = self._forward_start_step_offset(context)
        return self.eval_exporter.build_predict_records(
            rollout=export_rollout,
            graph=export_graph,
            questions=questions,
            step_offset=step_offset,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        return build_optimizer_and_scheduler(
            model_parameters=self.named_parameters(),
            optimizer_cfg=asdict(self.cfg.optimizer_cfg),
            scheduler_cfg=asdict(self.cfg.scheduler_cfg),
            estimated_stepping_batches=(
                int(self.trainer.estimated_stepping_batches)
                if self.trainer is not None
                else None
            ),
        )

    def build_context(self, batch: Any) -> tuple[GraphEnvContext, dict[str, Any]]:
        prepared, metadata = self._unpack_batch(batch)
        return self.env_builder.build_context(prepared), metadata

    @staticmethod
    def _unpack_batch(batch: Any) -> BatchPayload:
        if isinstance(batch, dict) and "inputs" in batch and "metadata" in batch:
            prepared = batch.get("inputs")
            metadata = batch.get("metadata")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            prepared, metadata = batch
        else:
            raise RuntimeError(
                "DualFlowModule expects batches prepared by GRetrievalDataModule hooks "
                "(src.datasets.g_retrieval_datamodule, on_after_batch_transfer)."
            )
        if not isinstance(prepared, dict) or not isinstance(metadata, dict):
            raise TypeError("Prepared batch must be a dict with metadata dict.")
        return prepared, metadata

    def override_forward_start_nodes(
        self, context: GraphEnvContext, *, deterministic: bool
    ) -> GraphEnvContext:
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
            raise ValueError(
                "stop.reward_beta_init and stop.reward_beta_max must be > 0."
            )
        if anneal_steps <= 0 or beta_init == beta_max:
            return beta_init

        progress_num = max(int(self.global_step) - anneal_start, 0)
        progress = min(float(progress_num) / float(anneal_steps), 1.0)
        if schedule == "linear":
            return beta_init + (beta_max - beta_init) * progress
        if schedule == "exponential":
            return math.exp(
                math.log(beta_init)
                + (math.log(beta_max) - math.log(beta_init)) * progress
            )
        raise ValueError(
            "stop.reward_beta_schedule must be one of {'linear', 'exponential'}."
        )

    def compute_rewards(
        self,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        reward_beta: float | None = None,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
        terminal_done_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.reward_engine.compute_rewards(
            stop_nodes_abs=stop_nodes_abs,
            context=context,
            reward_beta=self.current_reward_beta()
            if reward_beta is None
            else float(reward_beta),
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
            terminal_done_mask=terminal_done_mask,
        )

    def compute_hit_mask(
        self,
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
        terminal_done_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.reward_engine.compute_hit_mask(
            stop_nodes_abs,
            context,
            target_local_indices=target_local_indices,
            target_ptr=target_ptr,
            target_field_name=target_field_name,
            terminal_done_mask=terminal_done_mask,
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
            if metric_name.endswith("subtb/var_loss"):
                continue
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
        context, metadata = self.build_context(batch)
        rollout_every_n = int(self.cfg.eval_cfg.rollout_metrics_every_n_batches)
        run_rollout_metrics = rollout_every_n > 0 and (batch_idx % rollout_every_n == 0)

        beam_context = self.override_forward_start_nodes(context, deterministic=True)
        encoded_context = self._encode_policy_context(beam_context)
        beam_rollout = self._run_eval_rollout(
            context=beam_context,
            encoded_context=encoded_context,
        )
        beam_rollout_diag, beam_done_mask = self._build_rollout_diagnostics(
            rollout=beam_rollout,
            context=context,
            flow_direction="forward",
        )
        rewards, reward_metrics = self.compute_rewards(
            beam_rollout.stop_nodes,
            context,
            terminal_done_mask=beam_done_mask,
        )

        scope = self._resolve_dataset_scope(metadata)
        prefix = f"{stage}/{scope}"
        step_offset = self._forward_start_step_offset(context)
        beam_export = self._build_rollout_export_inputs(beam_rollout)
        metrics = self.eval_exporter.build_eval_metrics(
            prefix=prefix,
            reward_metrics=reward_metrics,
            rollout=beam_export,
            step_offset=step_offset,
            num_graphs=int(context.num_graphs),
            device=rewards.device,
        )
        metrics.update(
            {
                f"{prefix}/non_max_steps_terminated_ratio": beam_rollout_diag[
                    "rollout/non_max_steps_terminated_ratio"
                ],
                f"{prefix}/max_steps_reached_ratio": beam_rollout_diag[
                    "rollout/max_steps_reached_ratio"
                ],
                f"{prefix}/max_steps_reached_on_target_ratio": beam_rollout_diag[
                    "rollout/max_steps_reached_on_target_ratio"
                ],
            }
        )
        if run_rollout_metrics:
            rollout_context = self.override_forward_start_nodes(
                context, deterministic=False
            )
            rollout_encoded_context = (
                encoded_context
                if rollout_context is beam_context
                else self._encode_policy_context(rollout_context)
            )
            rollout = self.sampler.sample_forward(
                rollout_context,
                self.policy,
                flow_direction="forward",
                deterministic=False,
                temperature=float(self.cfg.sampling_cfg.eval_sampling_temperature),
                encoded_context=rollout_encoded_context,
                collect_traces=False,
            )
            rollout_diag, rollout_done_mask = self._build_rollout_diagnostics(
                rollout=rollout,
                context=context,
                flow_direction="forward",
            )
            _, rollout_reward_metrics = self.compute_rewards(
                rollout.stop_nodes,
                context,
                terminal_done_mask=rollout_done_mask,
            )
            rollout_export = self._build_rollout_export_inputs(rollout)
            metrics.update(
                self.eval_exporter.build_rollout_probe_metrics(
                    prefix=prefix,
                    reward_metrics=rollout_reward_metrics,
                    rollout=rollout_export,
                    step_offset=step_offset,
                )
            )
            metrics.update(
                {
                    f"{prefix}/non_max_steps_terminated_ratio_rollout": rollout_diag[
                        "rollout/non_max_steps_terminated_ratio"
                    ],
                    f"{prefix}/max_steps_reached_ratio_rollout": rollout_diag[
                        "rollout/max_steps_reached_ratio"
                    ],
                    f"{prefix}/max_steps_reached_on_target_ratio_rollout": rollout_diag[
                        "rollout/max_steps_reached_on_target_ratio"
                    ],
                }
            )
        for key, value in metrics.items():
            log_metric(
                self,
                key,
                value,
                batch_size=context.num_graphs,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def _build_rollout_diagnostics(
        self,
        *,
        rollout: RolloutResult,
        context: GraphEnvContext,
        flow_direction: str,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        stop_reason = rollout.stop_reason.to(device=self.device, dtype=torch.long)
        non_max_steps_terminated_mask = stop_reason != STOP_REASON_MAX_STEPS_REACHED
        terminal_mask = stop_reason != 0
        max_steps_reached_mask = stop_reason == STOP_REASON_MAX_STEPS_REACHED
        action_stop_mask = stop_reason == STOP_REASON_ACTION
        dead_end_mask = stop_reason == STOP_REASON_DEAD_END
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        max_steps_reached_on_target = zero
        if bool(max_steps_reached_mask.any().item()):
            if flow_direction == "forward":
                max_steps_reached_hits = self.compute_hit_mask(
                    rollout.stop_nodes,
                    context,
                    target_local_indices=context.a_local_indices,
                    target_ptr=context.a_ptr,
                    target_field_name="a_local_indices",
                )
            elif flow_direction == "backward":
                max_steps_reached_hits = self.compute_hit_mask(
                    rollout.stop_nodes,
                    context,
                    target_local_indices=context.q_local_indices,
                    target_ptr=context.q_ptr,
                    target_field_name="q_local_indices",
                )
            else:
                raise ValueError(
                    f"Unsupported flow_direction for rollout diagnostics: {flow_direction!r}."
                )
            max_steps_reached_hits = max_steps_reached_hits.to(
                device=self.device, dtype=torch.bool
            )
            max_steps_reached_on_target = (
                max_steps_reached_hits & max_steps_reached_mask
            ).float().sum() / max_steps_reached_mask.float().sum().clamp(min=1.0)
        metrics = {
            "rollout/non_max_steps_terminated_ratio": non_max_steps_terminated_mask.float()
            .mean()
            .detach(),
            "rollout/max_steps_reached_ratio": max_steps_reached_mask.float()
            .mean()
            .detach(),
            "rollout/action_stop_ratio": action_stop_mask.float().mean().detach(),
            "rollout/dead_end_ratio": dead_end_mask.float().mean().detach(),
            "rollout/max_steps_reached_on_target_ratio": max_steps_reached_on_target.detach(),
        }
        return metrics, terminal_mask

    @staticmethod
    def _resolve_dataset_scope(metadata: dict[str, Any]) -> str:
        scope = str(metadata.get("dataset_scope", "")).strip().lower()
        if scope in {"full", "sub"}:
            return scope
        return "full"


__all__ = ["DualFlowModule"]
