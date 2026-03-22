# GFlowNet Architecture

本文只描述当前主线实现，不再沿用历史 `trajectory_gfn` / `tasks/answer_reachability`
命名。

## 1. 当前代码边界

```
src/train.py
src/eval.py
src/runs/
src/models/gflownet_module.py
src/models/gflownet/
src/models/components/
src/models/configs/
src/metrics/answer_reachability/
src/graph_runtime/
src/archive/
```

- `src/train.py`: 训练入口；负责 Hydra 组装、Lightning 实例化、fit/test 调度。
- `src/eval.py`: 评估入口；负责 checkpoint 评估、split 选择、predict/test 调度。
- `src/runs/`: 任务级 runner；封装 train/eval contract、dataset variant、split replay、输出落盘。
- `src/models/gflownet_module.py`: 主线 LightningModule；负责组装 policy、sampler、loss、metric runtime。
- `src/models/gflownet/`: 当前 GFlowNet 主体；包括共享 encoder、flow/policy 头、sampler、replay、SubTB loss。
- `src/metrics/answer_reachability/`: 验证/测试/预测逻辑；包括 flow-frontier analysis、legacy Monte Carlo、posterior、metrics、artifact writer。
- `src/graph_runtime/`: 图批处理运行时表示；把 DataLoader/PyG batch 转为统一 `TrajectoryBatch`。
- `src/archive/`: 历史实验代码，仅保留兼容/追溯用途，不属于当前主线。

## 2. 训练链路

1. `src/train.py` 读取 Hydra 配置并实例化 `run`、`data`、`model`、`trainer`。
2. `configs/run/train_answer_reachability.yaml` 选择 `AnswerReachabilityTrainRunner`。
3. `src/models/gflownet_module.py` 在 `training_step()` 中调用：
   - `policy.prepare_batch()` 编码图与问题；
   - `ForwardTrajectoryGFNSampler.sample()` 采样 rollout；
   - `AnswerReachabilityTrajectorySupervisor` 给出 terminal target/reward；
   - `SubTrajectoryBalanceLoss.compute()` 用 `log F / log P_F / log R` 的前向子轨迹一致性残差计算 SubTB 目标。
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
   - runtime-selected reachability analysis（answer-reachability 默认 flow-frontier，edge retrieval 固定 Monte Carlo）
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
12. `src/metrics/answer_reachability/flow_frontier.py`
13. `src/metrics/answer_reachability/monte_carlo.py`

## 5. 当前策略内部结构

`src/models/gflownet/policy.py` 现在采用“静态编码 + recurrent prefix controller”的拆分：

- `prepare_batch()` 只在每个 batch 开头运行一次，得到：
  - 节点表示 `node_tokens`
  - 关系表示 `relation_tokens`
  - 全局问题向量 `question_tokens`
  - 问题 token 序列 `question_context_tokens`
- `SearchState` 同时保留两层状态：
  - 离散环境状态：`path_token_ids` 精确记录 prefix，用于 backward / replay / trace
  - 连续控制状态：`control_state` 压缩问题条件下的前缀历史，用于前向打分
- 起点和后续 graph move 都通过同一个 controller 更新：先让当前 `control_state`
  注意问题 token，再把注意结果、关系表示、目标节点表示送入 `GRUCell`。
- `NodeFlowHead` 只读取当前 prefix 的 state feature，输出 `log F`。
- `TransitionPolicyHead` 只读取：
  - 当前 prefix 的 state feature
  - 静态候选节点表示
  - 关系表示
  不再做 path self-attention，也不再为每个 candidate 重新编码整段前缀。
- 非根 backward transition 不再学习一个独立 backward head，而是从 `path_token_ids`
  直接恢复 prefix-tree 上唯一合法 parent。

这意味着当前主线是：

- 环境语义上：exact prefix state
- 神经表征上：recurrent control state
- 前向 actor 上：`control-state + static candidate` 打分
- backward 上：exact parent recovery

## 6. 历史路径映射

旧文档中的以下路径已经迁移：

- `src/models/trajectory_gfn/module.py` -> `src/models/gflownet_module.py`
- `src/models/trajectory_gfn/sampler.py` -> `src/models/gflownet/sampler.py`
- `src/models/trajectory_gfn/losses.py` -> `src/models/gflownet/losses.py`
- `src/tasks/answer_reachability/execution.py` -> `src/metrics/answer_reachability/batch_evaluator.py`
- `src/tasks/answer_reachability/metrics.py` -> `src/metrics/answer_reachability/metrics.py`
- `src/tasks/answer_reachability/posterior.py` -> `src/metrics/answer_reachability/posterior.py`
- `src/tasks/answer_reachability/search.py` -> `src/metrics/answer_reachability/flow_frontier.py`
- `src/tasks/answer_reachability/exact_reachability.py` -> `src/metrics/answer_reachability/exact_analysis.py`

如果你看到旧路径，请优先以上面的当前实现为准。
