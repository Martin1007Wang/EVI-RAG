# GFlowNet Architecture

本文只描述当前主线实现，不再沿用历史 `trajectory_gfn` / `tasks/answer_reachability`
命名。

## 1. 当前代码边界

```
src/train.py
src/eval.py
src/runs/
src/models/
src/models/policy/
src/models/training/
src/metrics/answer_reachability/
src/graph_runtime/
src/archive/policy/
```

- `src/train.py`: 训练入口；负责 Hydra 组装、Lightning 实例化、fit/test 调度。
- `src/eval.py`: 评估入口；负责 checkpoint 评估、split 选择、predict/test 调度。
- `src/runs/`: 任务级 runner；封装 train/eval contract、dataset variant、split replay、输出落盘。
- `src/models/gflownet_module.py`: 主线 LightningModule；负责组装 policy、sampler、loss、metric runtime。
- `src/models/gflownet/`: 当前 GFlowNet 主体；包括共享 encoder、三头 policy、sampler、replay、SubTB loss。
- `src/metrics/answer_reachability/`: 验证/测试/预测逻辑；包括 exact analysis、search、posterior、metrics、artifact writer。
- `src/graph_runtime/`: 图批处理运行时表示；把 DataLoader/PyG batch 转为统一 `TrajectoryBatch`。
- `src/archive/policy/`: 历史 policy 实验代码，仅保留兼容/追溯用途，不属于当前主线。

## 2. 训练链路

1. `src/train.py` 读取 Hydra 配置并实例化 `run`、`data`、`model`、`trainer`。
2. `configs/run/train_answer_reachability.yaml` 选择 `AnswerReachabilityTrainRunner`。
3. `src/models/gflownet_module.py` 在 `training_step()` 中调用：
   - `policy.prepare_batch()` 编码图与问题；
   - `ForwardTrajectoryGFNSampler.sample()` 采样 rollout；
   - `AnswerReachabilityTrajectorySupervisor` 给出 terminal target/reward；
   - `SubTrajectoryBalanceLoss.compute()` 用解耦 `log F / log P_F / log P_B` 计算 SubTB 目标。
4. Lightning 负责优化器、scheduler、日志、checkpoint。

训练期验证默认走低成本 `rank_only`，只看 answer posterior，不跑 support-window
search。相关默认值见 `configs/experiment/train_answer_reachability.yaml`。

## 3. 评估链路

1. `src/eval.py` 读取 `run.execution_mode`，选择 `trainer.predict()` 或 `trainer.test()`。
2. `AnswerReachabilityEvalRunner` 支持：
   - `run.split`
   - `run.run_all_splits`
   - `run.dataset_variants`
3. `GraphRetrievalDataModule.set_eval_split()` 接收 runner 当前请求的 split。
4. `GFlowNetModule` 把评估委托给 `EvaluationController` 和 metric runtime。
5. `src/metrics/answer_reachability/runtime.py` 决定当前任务视图：
   - `answer_reachability`
   - `edge_retrieval`
6. `ReachabilityBatchEvaluator` 负责：
   - exact reachability analysis
   - rank-only result 构造
   - support-window search
   - batch metrics 聚合
7. `RunOutputOrchestrator` 在 runner 末尾统一落盘 metrics 和 prediction artifacts。

## 4. 推荐阅读顺序

如果你第一次进仓库，建议按下面顺序读：

1. `configs/experiment/train_answer_reachability.yaml`
2. `configs/experiment/answer_reachability.yaml`
3. `src/train.py`
4. `src/eval.py`
5. `src/runs/answer_reachability.py`
6. `src/models/gflownet_module.py`
7. `src/models/gflownet/policy.py`
8. `src/models/gflownet/sampler.py`
9. `src/models/gflownet/losses.py`
10. `src/metrics/answer_reachability/runtime.py`
11. `src/metrics/answer_reachability/batch_evaluator.py`
12. `src/metrics/answer_reachability/support_search.py`

## 5. 历史路径映射

旧文档中的以下路径已经迁移：

- `src/models/trajectory_gfn/module.py` -> `src/models/gflownet_module.py`
- `src/models/trajectory_gfn/sampler.py` -> `src/models/gflownet/sampler.py`
- `src/models/trajectory_gfn/losses.py` -> `src/models/gflownet/losses.py`
- `src/tasks/answer_reachability/execution.py` -> `src/metrics/answer_reachability/batch_evaluator.py`
- `src/tasks/answer_reachability/metrics.py` -> `src/metrics/answer_reachability/metrics.py`
- `src/tasks/answer_reachability/posterior.py` -> `src/metrics/answer_reachability/posterior.py`
- `src/tasks/answer_reachability/search.py` -> `src/metrics/answer_reachability/support_search.py`
- `src/tasks/answer_reachability/exact_reachability.py` -> `src/metrics/answer_reachability/exact_analysis.py`

如果你看到旧路径，请优先以上面的当前实现为准。
