from __future__ import annotations

from typing import Any
import torch
from lightning import LightningModule
from src.data.schema import RetrievalBatch
from src.models.policy import Policy
from src.models.rollout import RolloutEngine
from src.models.losses import DetailedBalanceLoss
from src.models.reward import RewardModel

from src.models.optimization import build_optimizer_and_scheduler
from src.models.schedules import SamplingTemperatureScheduler

# 引入重构后的、针对 RAG 检索器定义的评估接口
from src.eval.metrics import (
    build_union_context_graph,
    compute_union_coverage_metrics,
    compute_exploration_diversity,
    compute_context_efficiency,
)


class GFlowNetModule(LightningModule):
    def __init__(
        self,
        *,
        max_steps: int = 20,
        rollouts_per_graph=8,
        sampling_temperature_schedule: dict[str, Any] | None = None,
        backbone: dict[str, Any] | None = None,
        policy_hidden_dim: int = 512,
        answer_reward: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
        eval_num_rollouts: int = 36,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.policy = Policy(
            backbone_cfg=backbone or {},
            hidden_dim=policy_hidden_dim,
        )
        self.reward_model = RewardModel(**(answer_reward or {}))
        self.rollout_engine = RolloutEngine(max_steps=max_steps)
        self.loss_fn = DetailedBalanceLoss()
        self.temp_scheduler = SamplingTemperatureScheduler(
            **(sampling_temperature_schedule or {})
        )

    def training_step(self, batch: RetrievalBatch, batch_idx: int) -> torch.Tensor:
        """核心训练步：计算 Detailed Balance Loss"""
        temp = self.temp_scheduler.value(self.global_step)

        rollout_batch = self.rollout_engine.run_exploration(
            policy=self.policy,
            base_graph=batch,
            reward_model=self.reward_model,
        )

        loss_output = self.loss_fn.compute(rollout_batch)
        total_loss = loss_output.loss

        self.log_dict(
            {
                "train/db_loss": total_loss,
                "train/success_rate": loss_output.success_rate,
                "train/residual_abs": loss_output.residual_abs,
                "train/temp": temp,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def configure_optimizers(self) -> dict[str, Any]:
        return build_optimizer_and_scheduler(
            module=self,
            optimizer_cfg=self.hparams.get("optimizer", {}),
            scheduler_cfg=self.hparams.get("scheduler", {}),
        )

    def evaluate_subgraph_retrieval(self, batch: RetrievalBatch) -> dict[str, Any]:
        num_rollouts = self.hparams.get("eval_num_rollouts")

        # 1. 无偏蒙特卡洛采样 (产出 N 条完整轨迹)
        rollouts = self.rollout_engine.run_multiple_exploration(
            policy=self.policy,
            base_graph=batch,
            reward_model=self.reward_model,
            num_rollouts=num_rollouts,
        )

        # 2. 拓扑聚合：构建联合交付图 G_ctx
        union_graph = build_union_context_graph(rollouts, batch)

        # 3. 核心维度计算：覆盖率、效率、多样性
        coverage_metrics = compute_union_coverage_metrics(union_graph, batch)
        efficiency_metrics = compute_context_efficiency(union_graph, batch)
        diversity_metrics = compute_exploration_diversity(union_graph, batch)

        return {
            "coverage": coverage_metrics,
            "efficiency": efficiency_metrics,
            "diversity": diversity_metrics,
        }

    def validation_step(self, batch: RetrievalBatch, batch_idx: int) -> dict[str, Any]:
        results = self.evaluate_subgraph_retrieval(batch)

        # 验证阶段仅提取用于监控健康度的核心标量
        self.log_dict(
            {
                "val/union_recall": results["coverage"]["union_answer_recall"],
                "val/ctx_nodes": results["efficiency"]["num_nodes"],
                "val/diversity_score": results["diversity"]["sink_entropy"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return results

    def test_step(self, batch: RetrievalBatch, batch_idx: int) -> dict[str, Any]:
        results = self.evaluate_subgraph_retrieval(batch)

        # 测试阶段记录最完备的实证指标数据字典
        metrics_to_log = {}
        for k, v in results["coverage"].items():
            metrics_to_log[f"test/coverage/{k}"] = v
        for k, v in results["efficiency"].items():
            metrics_to_log[f"test/efficiency/{k}"] = v
        for k, v in results["diversity"].items():
            metrics_to_log[f"test/diversity/{k}"] = v

        self.log_dict(
            metrics_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return results

    def load_pretrained_weights(
        self, checkpoint_path: str, strict: bool = False
    ) -> tuple[list[str], list[str]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        return self.load_state_dict(state_dict, strict=strict)


__all__ = ["GFlowNetModule"]
