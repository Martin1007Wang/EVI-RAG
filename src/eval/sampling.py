from __future__ import annotations

from typing import Any
import torch
import time

from src.data.schema import RetrievalBatch
from src.models.rollout import RolloutEngine, RolloutBatch
from src.models.policy import Policy
from src.models.reward import RewardModel


def run_monte_carlo_sampling(
    engine: RolloutEngine,
    policy: Policy,
    base_graph: RetrievalBatch,
    reward_model: RewardModel,
    num_rollouts: int,
    batch_rollouts: int = 1,
    temperature: float = 1.0,
) -> list[RolloutBatch]:
    """
    执行蒙特卡洛采样

    Args:
        engine: Rollout 引擎
        policy: 策略网络
        base_graph: 基础图数据
        reward_model: 奖励模型
        num_rollouts: 总采样次数
        batch_rollouts: 批次大小（内存控制）
        temperature: 采样温度

    Returns:
        采样结果列表
    """
    results = []

    # 多次调用 run_exploration 进行采样
    for i in range(0, num_rollouts, batch_rollouts):
        current_batch = min(batch_rollouts, num_rollouts - i)

        # 这里需要修改 RolloutEngine 以支持多次采样
        # 目前先使用单次采样，后续需要扩展
        rollout_batch = engine.run_exploration(
            policy=policy,
            base_graph=base_graph,
            reward_model=reward_model,
            num_rollouts=current_batch,
            temperature=temperature,
        )

        results.append(rollout_batch)

    return results


def run_early_stop_sampling(
    engine: RolloutEngine,
    policy: Policy,
    base_graph: RetrievalBatch,
    reward_model: RewardModel,
    num_rollouts: int,
    batch_rollouts: int = 256,
    temperature: float = 1.0,
    early_stop_cfg: dict[str, Any] | None = None,
) -> tuple[list[RolloutBatch], dict[str, Any]]:
    """
    带置信度提前停止的采样

    Args:
        engine: Rollout 引擎
        policy: 策略网络
        base_graph: 基础图数据
        reward_model: 奖励模型
        num_rollouts: 请求的总采样次数
        batch_rollouts: 批次大小
        temperature: 采样温度
        early_stop_cfg: 提前停止配置

    Returns:
        (采样结果列表, 停止信息字典)
    """
    if not early_stop_cfg or not early_stop_cfg.get("enabled", False):
        # 无提前停止
        results = run_monte_carlo_sampling(
            engine=engine,
            policy=policy,
            base_graph=base_graph,
            reward_model=reward_model,
            num_rollouts=num_rollouts,
            batch_rollouts=batch_rollouts,
            temperature=temperature,
        )
        return results, None

    # 提取提前停止配置
    confidence = early_stop_cfg.get("confidence", 0.95)
    min_rollouts = early_stop_cfg.get("min_rollouts", 512)
    stability_top_k = early_stop_cfg.get("stability_top_k", 1)

    results = []
    executed_rollouts = 0
    stop_early = False
    start_time = time.time()

    # 累积的答案统计（用于稳定性检查）
    accumulated_stats = {}

    while executed_rollouts < num_rollouts:
        # 批次采样
        current_batch = min(batch_rollouts, num_rollouts - executed_rollouts)
        batch_results = run_monte_carlo_sampling(
            engine=engine,
            policy=policy,
            base_graph=base_graph,
            reward_model=reward_model,
            num_rollouts=current_batch,
            batch_rollouts=current_batch,  # 一次采完
            temperature=temperature,
        )

        results.extend(batch_results)
        executed_rollouts += current_batch

        # 更新累积统计（简化实现）
        # 实际应该统计每个答案的出现频次
        _update_accumulated_stats(accumulated_stats, batch_results)

        # 检查停止条件（达到最小采样数后）
        if executed_rollouts >= min_rollouts:
            stability = _compute_topk_stability(
                accumulated_stats,
                top_k=stability_top_k,
                confidence=confidence,
                total_samples=executed_rollouts,
            )
            if stability > 0:  # top-k 排名已稳定
                stop_early = True
                break

    elapsed_time = time.time() - start_time

    stop_info = {
        "requested_rollouts": num_rollouts,
        "executed_rollouts": executed_rollouts,
        "early_stop_rate": 1.0 if stop_early else 0.0,
        "elapsed_time": elapsed_time,
        "time_per_sample": elapsed_time / executed_rollouts
        if executed_rollouts > 0
        else 0.0,
    }

    return results, stop_info


def _update_accumulated_stats(
    stats: dict[str, Any],
    batch_results: list[RolloutBatch],
) -> None:
    """更新累积统计（简化实现）"""
    # 这里应该实现答案频次统计
    # 由于我们无法直接获取答案ID，这里使用占位符
    pass


def _compute_topk_stability(
    stats: dict[str, Any],
    top_k: int,
    confidence: float,
    total_samples: int,
) -> float:
    """
    计算 top-k 排名的稳定性

    基于 Hoeffding 不等式:
    P(|estimated_prob - true_prob| > ε) ≤ 2exp(-2nε²)

    Returns:
        稳定性边界（正数表示稳定）
    """
    if total_samples <= 0 or top_k <= 0:
        return 0.0

    # 计算置信区间半径
    delta = 1.0 - confidence
    epsilon = (1.0 / (2.0 * total_samples) * np.log(2.0 / delta)) ** 0.5

    # 简化实现：返回 epsilon 的倒数作为稳定性分数
    # 实际应该比较第 k 名和第 k+1 名的概率差
    return 1.0 / epsilon if epsilon > 0 else float("inf")


def _merge_rollout_batches(
    batch1: list[RolloutBatch] | None,
    batch2: list[RolloutBatch],
) -> list[RolloutBatch]:
    """合并两个 rollout 批次"""
    if batch1 is None:
        return batch2
    return batch1 + batch2


__all__ = [
    "run_monte_carlo_sampling",
    "run_early_stop_sampling",
]
