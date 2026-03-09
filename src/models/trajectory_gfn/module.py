from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
from lightning import LightningModule

from src.models.algorithms.base import build_optimizer_and_scheduler
from src.models.configs.policy import PolicyConfig
from src.models.configs.trajectory_gfn import (
    HorizonConfig,
    TrajectoryAnalyzerConfig,
    TrajectoryInferenceConfig,
    TrajectoryTrainingConfig,
)
from src.models.configs.training import OptimizerConfig, SchedulerConfig
from src.utils.logging_utils import get_logger, log_event, log_metric

from .analyzer import AnswerMassAnalysis, AnswerMassAnalyzer
from .batch import TrajectoryBatch
from .inference import AdaptivePosteriorInference
from .losses import StepwiseDetailedBalanceLoss
from .metrics import compute_elastic_metrics
from .policy import InvalidStartCandidatesError, TrajectoryPolicy
from .posterior import build_window_result
from .reward import TrajectoryReward
from .sampler import ForwardRolloutSampler
from .schema import ElasticEvalBatch, ElasticLabelRecord, ElasticWindowResult
from .search import MassAdaptiveTrajectorySearch


logger = get_logger(__name__)
_INVALID_START_STOP_REASON = "invalid_start_support"


@dataclass(frozen=True)
class TrajectoryGFlowNetConfig:
    horizon_cfg: HorizonConfig
    policy_cfg: PolicyConfig
    training_cfg: TrajectoryTrainingConfig
    inference_cfg: TrajectoryInferenceConfig
    analyzer_cfg: TrajectoryAnalyzerConfig
    optimizer_cfg: OptimizerConfig
    scheduler_cfg: SchedulerConfig


class TrajectoryGFlowNetModule(LightningModule):
    def __init__(
        self,
        *,
        horizon_cfg: HorizonConfig,
        policy_cfg: PolicyConfig,
        training_cfg: TrajectoryTrainingConfig,
        inference_cfg: TrajectoryInferenceConfig,
        analyzer_cfg: TrajectoryAnalyzerConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
    ) -> None:
        super().__init__()
        self.cfg = TrajectoryGFlowNetConfig(
            horizon_cfg=horizon_cfg,
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            inference_cfg=inference_cfg,
            analyzer_cfg=analyzer_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
        )
        self.save_hyperparameters({"config": asdict(self.cfg)}, logger=False)
        self.policy = TrajectoryPolicy(
            policy_cfg,
            max_steps=int(horizon_cfg.max_steps),
            min_stop_steps=int(horizon_cfg.min_stop_steps),
        )
        self.reward = TrajectoryReward(
            epsilon=float(training_cfg.reward_epsilon),
            wrong_stop_reward_mode=str(training_cfg.wrong_stop_reward_mode),
        )
        self.sampler = ForwardRolloutSampler(
            horizon_cfg=horizon_cfg,
            training_cfg=training_cfg,
            reward=self.reward,
        )
        self.loss_fn = StepwiseDetailedBalanceLoss(config=training_cfg)
        self.posterior_inference = AdaptivePosteriorInference(
            answer_mass_threshold=float(inference_cfg.answer_mass_threshold),
            support_mass_threshold=float(inference_cfg.support_mass_threshold),
            rollout_chunk_size=int(inference_cfg.rollout_chunk_size),
            max_rollouts=int(inference_cfg.max_rollouts),
            answer_top_ks=tuple(int(k) for k in inference_cfg.answer_top_ks),
        )
        self.analyzer = AnswerMassAnalyzer(
            max_steps=int(horizon_cfg.max_steps),
            min_stop_steps=int(horizon_cfg.min_stop_steps),
        )
        self.search = MassAdaptiveTrajectorySearch(
            horizon_cfg=horizon_cfg,
            inference_cfg=inference_cfg,
            analyzer=self.analyzer,
        )
        self.predict_results: list[ElasticWindowResult] = []
        self.predict_labels: list[ElasticLabelRecord] = []
        self.predict_metrics: dict[str, Any] = {}

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

    def _log_metric_bundle(
        self,
        *,
        metrics: dict[str, Any],
        prefix: str,
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
        prog_bar_key: str | None = None,
    ) -> None:
        for name, value in metrics.items():
            if torch.is_tensor(value):
                metric_value = value.detach()
            else:
                metric_value = torch.tensor(float(value), device=self.device)
            key = f"{prefix}/{name}"
            log_metric(
                self,
                key,
                metric_value,
                batch_size=batch_size,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=(key == prog_bar_key),
                sync_dist=True,
            )

    def _ensure_batch(self, batch: Any) -> TrajectoryBatch:
        if not isinstance(batch, TrajectoryBatch):
            raise TypeError(
                "TrajectoryGFlowNetModule expects TrajectoryBatch inputs from the datamodule."
            )
        model_device = next(self.parameters()).device
        if batch.node_embeddings.device != model_device:
            return batch.to(model_device)
        return batch

    def _predict_inference_mode(self, *, include_support_windows: bool) -> str:
        mode = str(self.cfg.inference_cfg.mode)
        if mode == "exact" and not include_support_windows:
            return "exact_rank_only"
        return mode

    @staticmethod
    def _empty_answer_mass_analysis(batch: TrajectoryBatch) -> AnswerMassAnalysis:
        device = batch.node_ptr.device
        return AnswerMassAnalysis(
            terminal_mass=torch.zeros(
                (batch.num_nodes_total,), device=device, dtype=torch.float32
            ),
            answer_entity_ids=torch.empty((0,), device=device, dtype=torch.long),
            answer_probs=torch.empty((0,), device=device, dtype=torch.float32),
            gold_total_mass=0.0,
        )

    def _build_invalid_start_result(
        self,
        batch: TrajectoryBatch,
        *,
        include_support_windows: bool,
    ) -> ElasticWindowResult:
        return build_window_result(
            batch=batch,
            discovered_paths=[],
            analysis=self._empty_answer_mass_analysis(batch),
            inference_mode=self._predict_inference_mode(
                include_support_windows=include_support_windows
            ),
            answer_mass_threshold=float(self.cfg.inference_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.cfg.inference_cfg.support_mass_threshold),
            probe_count=0,
            remaining_mass_upper=1.0,
            stop_reason=_INVALID_START_STOP_REASON,
        )

    def _infer_single_graph(
        self,
        batch: TrajectoryBatch,
        *,
        include_support_windows: bool,
    ) -> ElasticWindowResult:
        context = self.policy.encode(batch)
        if str(self.cfg.inference_cfg.mode) == "sampled_rank_only":
            return self.posterior_inference.infer_sampled_rank_only_graph(
                batch=batch,
                policy=self.policy,
                context=context,
                sampler=self.sampler,
            )
        analysis = self.analyzer.analyze(
            batch=batch,
            policy=self.policy,
            context=context,
        )
        if str(self.cfg.inference_cfg.mode) == "exact":
            if not include_support_windows:
                return build_window_result(
                    batch=batch,
                    discovered_paths=[],
                    analysis=analysis,
                    inference_mode="exact_rank_only",
                    answer_mass_threshold=float(
                        self.cfg.inference_cfg.answer_mass_threshold
                    ),
                    support_mass_threshold=float(
                        self.cfg.inference_cfg.support_mass_threshold
                    ),
                    probe_count=0,
                    remaining_mass_upper=1.0,
                    stop_reason="rank_only",
                )
            return self.search.generate_window(
                batch=batch,
                policy=self.policy,
                context=context,
            )
        return self.posterior_inference.infer_sampled_graph(
            batch=batch,
            policy=self.policy,
            context=context,
            sampler=self.sampler,
            analysis=analysis,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        trajectory_batch = self._ensure_batch(batch)
        context = self.policy.encode(trajectory_batch)
        sample_batch = self.sampler.sample(
            batch=trajectory_batch,
            policy=self.policy,
            context=context,
        )
        loss_output = self.loss_fn.compute(sample_batch)
        metrics = {
            "loss": loss_output.loss.detach(),
            "db_start": loss_output.start_loss,
            "db_move": loss_output.move_loss,
            "db_stop": loss_output.stop_loss,
            "rollout_hit": loss_output.hit_rate,
        }
        self._log_metric_bundle(
            metrics=metrics,
            prefix="train",
            batch_size=int(trajectory_batch.num_graphs),
            on_step=True,
            on_epoch=False,
            prog_bar_key="train/loss",
        )
        return loss_output.loss

    def _evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
    ) -> tuple[
        dict[str, float],
        list[ElasticWindowResult],
        dict[str, torch.Tensor],
        dict[str, float],
    ]:
        with torch.no_grad():
            context = self.policy.encode(batch)
            sample_batch = self.sampler.sample(
                batch=batch,
                policy=self.policy,
                context=context,
                num_rollouts=int(self.cfg.inference_cfg.rollout_chunk_size),
                is_training=False,
            )
            loss_output = self.loss_fn.compute(sample_batch)
            include_support_windows = (
                bool(self.cfg.inference_cfg.compute_support_windows)
                or str(self.cfg.inference_cfg.mode) != "exact"
            )
            window_results = self._infer_results(
                batch,
                include_support_windows=include_support_windows,
            )
        elastic_metrics: dict[str, float] = {}
        if include_support_windows:
            elastic_metrics = compute_elastic_metrics(
                ElasticEvalBatch(
                    dataset_scope=batch.dataset_scope,
                    mass_threshold=float(self.cfg.inference_cfg.support_mass_threshold),
                    results=window_results,
                )
            )
        db_metrics = {
            "db_loss": loss_output.loss.detach(),
            "db_start": loss_output.start_loss,
            "db_move": loss_output.move_loss,
            "db_stop": loss_output.stop_loss,
            "rollout_hit": loss_output.hit_rate,
        }
        rank_metrics = self.posterior_inference.aggregate_rank_metrics(
            results=window_results,
        )
        return elastic_metrics, window_results, db_metrics, rank_metrics

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        trajectory_batch = self._ensure_batch(batch)
        elastic_metrics, _, db_metrics, rank_metrics = self._evaluate_batch(
            batch=trajectory_batch
        )
        self._log_metric_bundle(
            metrics=db_metrics,
            prefix=f"val/{trajectory_batch.dataset_scope}",
            batch_size=int(trajectory_batch.num_graphs),
            on_step=False,
            on_epoch=True,
        )
        self._log_metric_bundle(
            metrics=rank_metrics,
            prefix=f"val/{trajectory_batch.dataset_scope}",
            batch_size=int(trajectory_batch.num_graphs),
            on_step=False,
            on_epoch=True,
        )
        self._log_metric_bundle(
            metrics=elastic_metrics,
            prefix=f"val/{trajectory_batch.dataset_scope}",
            batch_size=int(trajectory_batch.num_graphs),
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        trajectory_batch = self._ensure_batch(batch)
        elastic_metrics, _, db_metrics, rank_metrics = self._evaluate_batch(
            batch=trajectory_batch
        )
        self._log_metric_bundle(
            metrics=db_metrics,
            prefix=f"test/{trajectory_batch.dataset_scope}",
            batch_size=int(trajectory_batch.num_graphs),
            on_step=False,
            on_epoch=True,
        )
        self._log_metric_bundle(
            metrics=rank_metrics,
            prefix=f"test/{trajectory_batch.dataset_scope}",
            batch_size=int(trajectory_batch.num_graphs),
            on_step=False,
            on_epoch=True,
        )
        self._log_metric_bundle(
            metrics=elastic_metrics,
            prefix=f"test/{trajectory_batch.dataset_scope}",
            batch_size=int(trajectory_batch.num_graphs),
            on_step=False,
            on_epoch=True,
        )

    def _infer_results(
        self,
        batch: TrajectoryBatch,
        *,
        include_support_windows: bool = True,
    ) -> list[ElasticWindowResult]:
        results: list[ElasticWindowResult] = []
        for graph_idx in range(batch.num_graphs):
            sub_batch = batch.select_graph(graph_idx)
            try:
                results.append(
                    self._infer_single_graph(
                        sub_batch,
                        include_support_windows=include_support_windows,
                    )
                )
            except InvalidStartCandidatesError as exc:
                sample_id = sub_batch.sample_ids[0]
                log_event(
                    logger,
                    "trajectory_gfn_invalid_start_skipped",
                    level=logging.WARNING,
                    sample_id=sample_id,
                    dataset_scope=sub_batch.dataset_scope,
                    inference_mode=self._predict_inference_mode(
                        include_support_windows=include_support_windows
                    ),
                    min_stop_steps=exc.min_stop_steps,
                )
                results.append(
                    self._build_invalid_start_result(
                        sub_batch,
                        include_support_windows=include_support_windows,
                    )
                )
        return results

    def on_predict_epoch_start(self) -> None:
        self.predict_results = []
        self.predict_labels = []
        self.predict_metrics = {}

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> list[ElasticWindowResult]:
        del batch_idx, dataloader_idx
        trajectory_batch = self._ensure_batch(batch)
        return self._infer_results(trajectory_batch)

    def on_predict_batch_end(
        self,
        outputs: list[ElasticWindowResult] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch_idx, dataloader_idx
        if outputs:
            trajectory_batch = self._ensure_batch(batch)
            self.predict_results.extend(outputs)
            self.predict_labels.extend(
                self._build_predict_labels(trajectory_batch, outputs)
            )

    @staticmethod
    def _build_predict_labels(
        batch: TrajectoryBatch,
        outputs: list[ElasticWindowResult],
    ) -> list[ElasticLabelRecord]:
        if len(outputs) != batch.num_graphs:
            raise ValueError(
                "Predict outputs must align with TrajectoryBatch graph count. "
                f"outputs={len(outputs)} num_graphs={batch.num_graphs}."
            )
        labels: list[ElasticLabelRecord] = []
        a_counts = batch.a_ptr[1:] - batch.a_ptr[:-1]
        for graph_idx, result in enumerate(outputs):
            answer_start = int(batch.answer_ptr[graph_idx].item())
            answer_end = int(batch.answer_ptr[graph_idx + 1].item())
            labels.append(
                ElasticLabelRecord(
                    sample_id=result.sample_id,
                    question=batch.questions[graph_idx],
                    start_entity_ids=list(result.start_entity_ids),
                    answer_entity_ids=[
                        int(value)
                        for value in batch.answer_entity_ids[
                            answer_start:answer_end
                        ].tolist()
                    ],
                    a_entity_in_graph=bool(int(a_counts[graph_idx].item()) > 0),
                )
            )
        return labels

    def on_predict_epoch_end(self) -> None:
        if not self.predict_results:
            self.predict_metrics = {}
            return
        invalid_start_sample_ids = [
            result.sample_id
            for result in self.predict_results
            if result.stop_reason == _INVALID_START_STOP_REASON
        ]
        invalid_start_count = len(invalid_start_sample_ids)
        eval_batch = ElasticEvalBatch(
            dataset_scope=self.predict_results[0].dataset_scope,
            mass_threshold=float(self.cfg.inference_cfg.support_mass_threshold),
            results=self.predict_results,
        )
        self.predict_metrics = {
            **self.posterior_inference.aggregate_rank_metrics(
                results=self.predict_results
            ),
            **compute_elastic_metrics(eval_batch),
            "invalid_start_count": invalid_start_count,
            "invalid_start_rate": (
                float(invalid_start_count) / float(len(self.predict_results))
            ),
        }
        if invalid_start_count > 0:
            log_event(
                logger,
                "trajectory_gfn_invalid_start_summary",
                level=logging.WARNING,
                count=invalid_start_count,
                dataset_scope=self.predict_results[0].dataset_scope,
                sample_ids=invalid_start_sample_ids,
            )
