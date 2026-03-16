from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

from src.models.components import (
    EmbeddingBackbone,
    GraphLogZHead,
    NodeFlowHead,
    StartLogitHead,
)
from src.models.components.heuristic_heads import LearnedHeuristicHead
from src.models.configs import (
    GFlowNetTrainingConfig,
    HeuristicConfig,
    HorizonConfig,
    OptimizerConfig,
    PolicyConfig,
    SchedulerConfig,
)
from src.graph_runtime import TrajectoryBatch
from src.metrics.protocol import MetricRuntimeFactoryProtocol
from src.utils.logging_utils import get_logger, log_event, log_metric

from .evaluation_controller import MetricRuntimeController
from .policy import GFlowNetPolicy, TrajectoryHeuristic, TrajectoryPolicy
from .training import (
    SubTrajectoryBalanceLoss,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class GFlowNetConfig:
    horizon_cfg: HorizonConfig
    training_cfg: GFlowNetTrainingConfig
    heuristic_cfg: HeuristicConfig
    policy_cfg: PolicyConfig
    inference_cfg: Any
    optimizer_cfg: OptimizerConfig
    scheduler_cfg: SchedulerConfig


class GFlowNetModule(LightningModule):
    @staticmethod
    def _build_base_policy(
        *,
        policy_cfg: PolicyConfig,
        max_steps: int,
    ) -> TrajectoryPolicy:
        graph_hidden_dim = int(policy_cfg.backbone.hidden_dim)
        backbone = EmbeddingBackbone(policy_cfg.backbone)
        state_score_head = NodeFlowHead(
            node_dim=graph_hidden_dim,
            question_dim=graph_hidden_dim,
            hidden_dim=int(policy_cfg.state_score_head.hidden_dim),
            num_layers=int(policy_cfg.state_score_head.num_layers),
            dropout=float(policy_cfg.state_score_head.dropout),
        )
        start_head = StartLogitHead(
            policy_dim=graph_hidden_dim,
            hidden_dim=int(policy_cfg.start_head.hidden_dim),
            dropout=float(policy_cfg.start_head.dropout),
        )
        return TrajectoryPolicy(
            config=policy_cfg,
            max_steps=max_steps,
            backbone=backbone,
            state_score_head=state_score_head,
            start_head=start_head,
        )

    @staticmethod
    def _build_trajectory_heuristic(
        *,
        heuristic_cfg: HeuristicConfig,
        graph_hidden_dim: int,
    ) -> TrajectoryHeuristic:
        learned_head = None
        if heuristic_cfg.canonical_kind == "learned":
            learned_head = LearnedHeuristicHead(
                hidden_dim=int(heuristic_cfg.critic_hidden_dim),
                dropout=float(heuristic_cfg.critic_dropout),
                feature_dim=graph_hidden_dim,
            )
        return TrajectoryHeuristic(
            config=heuristic_cfg,
            learned_head=learned_head,
        )

    @staticmethod
    def _build_policy(
        *,
        policy_cfg: PolicyConfig,
        heuristic_cfg: HeuristicConfig,
        max_steps: int,
    ) -> GFlowNetPolicy:
        graph_hidden_dim = int(policy_cfg.backbone.hidden_dim)
        base_policy = GFlowNetModule._build_base_policy(
            policy_cfg=policy_cfg,
            max_steps=max_steps,
        )
        trajectory_heuristic = GFlowNetModule._build_trajectory_heuristic(
            heuristic_cfg=heuristic_cfg,
            graph_hidden_dim=graph_hidden_dim,
        )
        graph_log_z_head = GraphLogZHead(
            feature_dim=graph_hidden_dim,
            hidden_dim=int(policy_cfg.graph_log_z_head.hidden_dim),
            dropout=float(policy_cfg.graph_log_z_head.dropout),
        )
        return GFlowNetPolicy(
            base_policy=base_policy,
            heuristic_cfg=heuristic_cfg,
            graph_log_z_head=graph_log_z_head,
            trajectory_heuristic=trajectory_heuristic,
        )

    @staticmethod
    def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
        if is_dataclass(cfg):
            return asdict(cfg)  # type: ignore[arg-type]
        if isinstance(cfg, dict):
            return dict(cfg)
        raise TypeError(f"Expected dataclass or dict config, got {type(cfg)!r}.")

    @staticmethod
    def _normalize_scheduler_interval(scheduler_cfg: dict[str, Any]) -> str:
        interval = str(scheduler_cfg.get("interval", "step")).lower()
        if interval not in {"step", "epoch"}:
            raise ValueError(
                f"Unsupported scheduler interval: {interval!r}. Expected 'step' or 'epoch'."
            )
        return interval

    @staticmethod
    def _resolve_schedule_horizon(
        *,
        scheduler_cfg: dict[str, Any],
        interval: str,
        estimated_stepping_batches: int | None,
        trainer_max_epochs: int | None,
    ) -> int | None:
        explicit_t_max = scheduler_cfg.get("t_max")
        if explicit_t_max is not None:
            horizon = int(explicit_t_max)
            if horizon <= 0:
                raise ValueError(f"scheduler requires t_max > 0, got {horizon}.")
        elif interval == "step":
            if estimated_stepping_batches is None:
                return None
            horizon = int(estimated_stepping_batches)
        else:
            if trainer_max_epochs is None:
                return None
            horizon = int(trainer_max_epochs)
        if horizon <= 0:
            raise ValueError(f"Scheduler horizon must be > 0, got {horizon}.")
        return horizon

    @classmethod
    def _build_optimizer_and_scheduler(
        cls,
        *,
        model_parameters: list[tuple[str, torch.nn.Parameter]] | Any,
        optimizer_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any],
        estimated_stepping_batches: int | None,
        trainer_max_epochs: int | None = None,
    ) -> dict[str, Any]:
        trainable_params = [
            parameter for _, parameter in model_parameters if parameter.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in model.")

        opt_type = str(optimizer_cfg.get("type", "adamw")).lower()
        if opt_type != "adamw":
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        optimizer = AdamW(
            trainable_params,
            lr=float(optimizer_cfg.get("lr", 1.0e-4)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
            betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        )
        scheduler = None
        scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
        interval = cls._normalize_scheduler_interval(scheduler_cfg)
        schedule_horizon = cls._resolve_schedule_horizon(
            scheduler_cfg=scheduler_cfg,
            interval=interval,
            estimated_stepping_batches=estimated_stepping_batches,
            trainer_max_epochs=trainer_max_epochs,
        )
        if schedule_horizon is not None:
            eta_min = float(scheduler_cfg.get("eta_min", 0.0))
            if scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=schedule_horizon,
                    eta_min=eta_min,
                )
            elif scheduler_type == "cosine_warm_restarts":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=schedule_horizon,
                    T_mult=int(scheduler_cfg.get("t_mult", 1)),
                    eta_min=eta_min,
                )
            elif scheduler_type == "onecycle":
                if interval != "step":
                    raise ValueError(
                        "onecycle scheduler requires interval='step' because it must advance per optimizer step."
                    )
                if estimated_stepping_batches is not None and schedule_horizon < int(
                    estimated_stepping_batches
                ):
                    raise ValueError(
                        "onecycle scheduler would exhaust before training ends: "
                        f"t_max={schedule_horizon} estimated_steps={int(estimated_stepping_batches)}. "
                        "Set trainer.max_steps and scheduler t_max consistently."
                    )
                scheduler_lr = scheduler_cfg.get("lr", optimizer_cfg.get("lr", 1.0e-4))
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=float(scheduler_lr),
                    total_steps=schedule_horizon,
                    pct_start=float(scheduler_cfg.get("pct_start", 0.3)),
                    anneal_strategy=str(scheduler_cfg.get("anneal", "cos")),
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        result: dict[str, Any] = {"optimizer": optimizer}
        if scheduler is not None:
            result["lr_scheduler"] = {"scheduler": scheduler, "interval": interval}
        return result

    def __init__(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: GFlowNetTrainingConfig,
        policy_cfg: PolicyConfig,
        inference_cfg: Any,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
        metric_runtime_factory: MetricRuntimeFactoryProtocol,
        heuristic_cfg: HeuristicConfig = HeuristicConfig(),
    ) -> None:
        super().__init__()
        self.cfg = GFlowNetConfig(
            horizon_cfg=horizon_cfg,
            training_cfg=training_cfg,
            heuristic_cfg=heuristic_cfg,
            policy_cfg=policy_cfg,
            inference_cfg=inference_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
        )
        self.save_hyperparameters({"config": asdict(self.cfg)}, logger=False)
        self.policy = self._build_policy(
            policy_cfg=policy_cfg,
            heuristic_cfg=heuristic_cfg,
            max_steps=int(horizon_cfg.max_steps),
        )
        self.metric_runtime = metric_runtime_factory.build_runtime(
            horizon_cfg=horizon_cfg,
            training_cfg=training_cfg,
            inference_cfg=inference_cfg,
            policy=self.policy,
        )
        self.runtime_controller = MetricRuntimeController(
            metric_runtime=self.metric_runtime,
            metrics_profile=str(self.cfg.inference_cfg.metrics_profile),
            on_invalid_start=self._log_invalid_start,
        )
        self.sampler = self.runtime_controller.sampler
        self.loss_fn = SubTrajectoryBalanceLoss(config=training_cfg.subtb)
        self.search = self.runtime_controller.search

    @property
    def metrics_profile(self) -> str:
        return str(self.runtime_controller.metrics_profile)

    @property
    def task_view(self) -> str:
        return str(self.cfg.inference_cfg.task_view)

    @property
    def predict_results(self) -> list[Any]:
        return self.runtime_controller.prediction_state.results

    @predict_results.setter
    def predict_results(self, value: list[Any]) -> None:
        self.runtime_controller.prediction_state.results = list(value)

    @property
    def predict_labels(self) -> list[Any]:
        return self.runtime_controller.prediction_state.labels

    @predict_labels.setter
    def predict_labels(self, value: list[Any]) -> None:
        self.runtime_controller.prediction_state.labels = list(value)

    @property
    def predict_metrics(self) -> dict[str, Any]:
        return self.runtime_controller.prediction_state.metrics

    @predict_metrics.setter
    def predict_metrics(self, value: dict[str, Any]) -> None:
        self.runtime_controller.prediction_state.metrics = dict(value)

    def _ensure_batch(self, batch: Any) -> TrajectoryBatch:
        if not isinstance(batch, TrajectoryBatch):
            raise TypeError(
                "GFlowNetModule expects TrajectoryBatch inputs from the datamodule."
            )
        model_device = next(self.parameters()).device
        if batch.node_embeddings.device != model_device:
            return batch.to(model_device)
        return batch

    def configure_optimizers(self) -> dict[str, Any]:
        return self._build_optimizer_and_scheduler(
            model_parameters=self.named_parameters(),
            optimizer_cfg=self._cfg_to_dict(self.cfg.optimizer_cfg),
            scheduler_cfg=self._cfg_to_dict(self.cfg.scheduler_cfg),
            estimated_stepping_batches=(
                int(self.trainer.estimated_stepping_batches)
                if self.trainer is not None
                else None
            ),
            trainer_max_epochs=(
                int(self.trainer.max_epochs)
                if self.trainer is not None and int(self.trainer.max_epochs) > 0
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
            metric_value = (
                value.detach()
                if torch.is_tensor(value)
                else torch.tensor(float(value), device=self.device)
            )
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

    def _log_invalid_start(self, batch: TrajectoryBatch) -> None:
        log_event(
            logger,
            "gflownet_invalid_start_skipped",
            level=logging.WARNING,
            sample_id=batch.sample_ids[0],
            dataset_scope=batch.dataset_scope,
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        trajectory_batch = self._ensure_batch(batch)
        if self.sampler is None:
            raise RuntimeError(
                "Current metric runtime does not define a training sampler; this model cannot train with the configured metric_runtime_factory."
            )
        prepared_batch = self.policy.prepare_batch(trajectory_batch)
        sample_batch = self.sampler.sample(
            batch=trajectory_batch,
            policy=self.policy,
            prepared_batch=prepared_batch,
            rollout_batch_size=int(self.cfg.training_cfg.rollout_batch_size),
            temperature=float(self.cfg.training_cfg.sampling_temperature),
        )
        loss_output = self.loss_fn.compute(sample_batch)
        metrics: dict[str, Any] = {
            "loss": loss_output.loss.detach(),
            "subtb_loss": loss_output.subtb_loss,
            "subtb_residual": loss_output.residual_abs,
            "subtb_root": loss_output.root_abs,
            "rollout_success": loss_output.success_rate,
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

    def _evaluate_batch_output(self, *, batch: TrajectoryBatch) -> Any:
        return self.runtime_controller.evaluate_batch_output(
            batch=batch,
            include_answer_support=False,
        )

    def _evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
    ) -> tuple[
        dict[str, float],
        list[Any],
        dict[str, torch.Tensor],
        dict[str, float],
    ]:
        return self.runtime_controller.evaluate_batch(
            batch=batch,
            include_answer_support=False,
        )

    def _log_eval_outputs(
        self,
        *,
        stage: str,
        batch: TrajectoryBatch,
        outputs: Any,
    ) -> None:
        prefix = f"{stage}/{batch.dataset_scope}"
        batch_size = int(batch.num_graphs)
        for metrics in (
            outputs.model_metrics,
            outputs.secondary_metrics,
            outputs.primary_metrics,
        ):
            self._log_metric_bundle(
                metrics=metrics,
                prefix=prefix,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        trajectory_batch = self._ensure_batch(batch)
        outputs = self._evaluate_batch_output(batch=trajectory_batch)
        self._log_eval_outputs(stage="val", batch=trajectory_batch, outputs=outputs)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        trajectory_batch = self._ensure_batch(batch)
        outputs = self._evaluate_batch_output(batch=trajectory_batch)
        self._log_eval_outputs(stage="test", batch=trajectory_batch, outputs=outputs)

    def on_predict_epoch_start(self) -> None:
        self.runtime_controller.reset_prediction_state()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> list[Any]:
        del batch_idx, dataloader_idx
        trajectory_batch = self._ensure_batch(batch)
        return self.runtime_controller.predict_batch(
            batch=trajectory_batch,
        )

    def on_predict_batch_end(
        self,
        outputs: list[Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch_idx, dataloader_idx
        trajectory_batch = self._ensure_batch(batch)
        self.runtime_controller.record_prediction_batch(
            batch=trajectory_batch,
            outputs=outputs,
        )

    def on_predict_epoch_end(self) -> None:
        self.runtime_controller.finalize_prediction_epoch()

    def get_predict_metrics(self) -> dict[str, Any]:
        return self.runtime_controller.get_predict_metrics()

    def write_prediction_artifacts(
        self,
        *,
        output_dir: str | Path,
        split: str,
        artifact_name: str = "eval_answer_reachability",
        schema_version: int = 1,
        entity_vocab_path: str | Path | None = None,
        relation_vocab_path: str | Path | None = None,
        questions_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> dict[str, Path] | None:
        return self.runtime_controller.write_prediction_artifacts(
            output_dir=output_dir,
            split=split,
            artifact_name=artifact_name,
            schema_version=schema_version,
            entity_vocab_path=entity_vocab_path,
            relation_vocab_path=relation_vocab_path,
            questions_path=questions_path,
            overwrite=overwrite,
        )


__all__ = [
    "GFlowNetConfig",
    "GFlowNetModule",
]
