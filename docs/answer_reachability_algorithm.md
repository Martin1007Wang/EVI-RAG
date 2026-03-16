# Answer Reachability Algorithm

本文描述当前 answer-reachability 主线的训练、验证、测试与指标语义。

## 1. 任务定义

给定问题节点集合 `Q`、图 `G`、答案实体集合 `A`，模型学习一个从起点到终点的
trajectory policy。训练时采样轨迹并优化 GFlowNet/SubTB 目标；评估时用 exact
analysis 和 guided search 分析答案概率与 support window。

## 2. 训练逻辑

训练主链在：

- `src/models/gflownet_module.py`
- `src/models/training/sampler.py`
- `src/models/training/answer_reachability.py`
- `src/models/training/losses.py`

步骤如下：

1. `TrajectoryPolicy` 或 `GFlowNetPolicy` 先编码图与问题。
2. `ForwardTrajectoryGFNSampler` 从 start distribution 采样起点，并按 forward
   distribution rollout。
3. `AnswerReachabilityTrajectorySupervisor` 决定哪些 terminal node 算成功，并为成败
   轨迹提供 reward / log_reward。
4. `SubTrajectoryBalanceLoss` 用 sampled trajectories 计算 SubTB 残差并回传梯度。

训练期默认不会跑昂贵的 support-window search；只保留 answer ranking 所需的轻量验证。

## 3. 验证 / 测试逻辑

验证与测试主链在：

- `src/metrics/answer_reachability/runtime.py`
- `src/metrics/answer_reachability/execution.py`
- `src/metrics/answer_reachability/exact.py`
- `src/metrics/answer_reachability/search.py`
- `src/metrics/answer_reachability/posterior.py`
- `src/metrics/answer_reachability/metrics.py`

### 3.1 rank_only

`metrics_profile=rank_only` 时：

- 只做 exact reachability analysis。
- 构建 answer posterior。
- 输出 `answer/*` 指标。
- 不做 support-window search，因此没有 `window/*` 指标。

这一路径适合训练期验证或 edge retrieval view。

### 3.2 full

`metrics_profile=full` 时：

- 先做 exact reachability analysis，得到 gold mass 和 answer posterior。
- 再用 `ReachabilityGuidedSearch` 生成 support window。
- 最终同时输出：
  - `answer/*`
  - `window/*`
  - `cert/*`

这一路径适合正式评估与 artifact 生成。

## 4. 指标词典

### 4.1 `answer/*`

定义见 `src/metrics/answer_reachability/posterior.py`。

- `answer/gold_mass`: gold answers 在 exact posterior 中的总概率质量。
- `answer/selected_mass`: 满足 answer mass threshold 的 posterior 前缀总质量。
- `answer/hit@k`: posterior 前 `k` 个 answer 中是否命中任一 gold answer。
- `answer/recall@k`: posterior 前 `k` 个 answer 覆盖 gold answers 的比例。

### 4.2 `window/*`

定义见 `src/metrics/answer_reachability/metrics.py`。

- `window/adaptive/*`: 自适应窗口整体统计。
- `window/top{k}/*`: support window 前缀 `top-k` 的 hit/recall/precision/f1。
- `window/adaptive/path_count`: 最终窗口发出的 path 数。
- `window/adaptive/path_mass`: 窗口覆盖到的总概率质量。
- `window/adaptive/gold_mass`: 窗口覆盖到的 gold probability mass。
- `window/adaptive/missed_gold_mass`: 仍未覆盖的 gold mass。

### 4.3 `cert/*`

- `cert/remaining_mass_upper`: 当前 search 仍未探索部分的质量上界。
- `cert/coverage_rate`: 当前窗口是否给出 coverage certificate。

### 4.4 `edge/*`

定义见 `src/metrics/answer_reachability/edge_eval.py`。

- `edge/mrr`: 第一个正例边的倒数排名。
- `edge/hit@k`: 前 `k` 条边是否包含任一 shortest-path positive edge。
- `edge/precision@k`: 前 `k` 条边中正例比例。
- `edge/recall@k`: 前 `k` 条边覆盖正例边的比例。
- `edge/gold_mass`: 正例边对应的 exact success mass。

## 5. split 与执行模式

### 5.1 `run.execution_mode`

- `predict`: 推荐评估模式；会汇总 predict metrics 并写 prediction artifacts。
- `test`: 调用 Lightning `trainer.test()`；保留为更接近标准 test loop 的模式。

### 5.2 `run.split`

当前评估 split 由 `run.split` 控制，`src/eval.py` 会显式把它传给
`GRetrievalDataModule.set_eval_split()`。

### 5.3 `run.run_all_splits`

如果启用，`BaseEvalRunner` 会按顺序重放多个 split，并在每次评估前临时覆盖
`cfg.run.split`。datamodule 会据此加载对应的 train/validation/test 数据。

### 5.4 `run.dataset_variants`

answer-reachability 正式评估通常会同时跑 `full` 和 `sub` 两个 dataset scope；runner
会为每个 variant 单独重放 split 并分别保存 metrics/artifacts。

## 6. 关键配置

主要配置位于 `src/models/configs/gflownet.py`：

- `eval_profile`: `full` 或 `rank_only`
- `eval_view`: `answer_reachability` 或 `edge_retrieval`
- `answer_mass_threshold`: answer posterior 截断阈值
- `support_mass_threshold`: support window 目标阈值
- `support_path_overlap_penalty`: 多条 support path 的重叠惩罚
- `max_expansions`, `max_frontier_size`, `strict_search`: search 预算与严格性

内部代码统一优先使用别名 `metrics_profile` 与 `task_view`，但 Hydra 对外仍保持
`eval_profile` / `eval_view` 兼容。
