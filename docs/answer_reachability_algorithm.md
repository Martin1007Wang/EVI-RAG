# Answer Reachability Algorithm

本文描述当前 answer-reachability 主线的训练、验证、测试与指标语义。

如果你希望先看当前实现对应的数学推导，请先读：

- `docs/answer_reachability_math_derivation.md`

如果你希望直接按当前代码实现复盘“训练/验证/最终评估到底算了什么、最小原子操作是什么”，请读：

- `docs/answer_reachability_codepath_review.md`

## 1. 任务定义

给定问题节点集合 `Q`、图 `G`、答案实体集合 `A`，模型学习一个从起点到终点的
trajectory policy。当前主线把多起点集合建模成一个隐式虚拟源 `s_emptyset` 指向
`Q` 上真实起点状态的 root-flow decomposition：起点分布由起点状态流归一化得到，
`log Z` 则由所有起点状态流的 `logsumexp` 给出。训练时采样轨迹并优化解耦三头
GFlowNet 的 `SubTB` 目标；默认 answer-reachability 评估时则沿 learned flow 做
deterministic flow-frontier expansion，直接构造答案 posterior 与 support window。
Monte Carlo 现在只保留为 legacy 后端，以及 edge retrieval 的唯一评估路径。

## 2. 训练逻辑

训练主链在：

- `src/models/gflownet_module.py`
- `src/models/gflownet/sampler.py`
- `src/models/gflownet/replay.py`
- `src/models/gflownet/losses.py`

步骤如下：

1. `BaseSearchPolicy` 或 `GFlowNetPolicy` 先编码图与问题；这一步只做一次，得到静态节点/关系表示、全局问题向量和问题 token 序列。
2. `ForwardTrajectoryGFNSampler` 先在隐式虚拟源下构造 start distribution，随后从中
   采样真实起点，并初始化 recurrent `control_state` 后 rollout。
3. `AnswerReachabilityTrajectorySupervisor` 决定哪些 terminal node 算成功，并为成败
   轨迹提供 reward / log_reward。
4. `SubTrajectoryBalanceLoss` 用 `log F / log P_F / log R` 的前向子轨迹一致性残差计算
   `SubTB` 并回传梯度。

当前 rollout 的搜索状态不是单纯 `(node, time)`：环境侧仍保留精确离散 prefix
`path_token_ids`，而前向打分侧用一个轻量 recurrent `control_state` 压缩前缀历史。
因此：

- 前向 `log F` 读取 `node + step/remaining + control_state`
- forward actor 读取 `current_state_feature + relation + candidate node`
- backward graph move 则通过 prefix-tree 上的唯一 parent 精确恢复

当前默认训练配置已经关闭 heuristic guidance：behavior sampling 不再额外乘任何
learned/topology heuristic bias，`guidance.loss_weight` 也保持为 `0.0`。如果后续实验想
重新打开，需要显式覆盖 `heuristic_cfg.kind`、`heuristic_cfg.beta` 和
`training_cfg.guidance.loss_weight`。

主训练实验现在还会对 terminal reward 加一个很小的长度惩罚：reward 仍然按 entity 定义，
但会再乘 `exp(-alpha * t)`，从而让同一答案的更短推理链条拿到更高的目标质量。
5. 如果开启 replay，buffer 中缓存的成功离散路径会在当前参数下重新打分，并与
   on-policy loss 按轨迹数加权平均。

训练期默认不会跑昂贵的 support-window search；只保留 answer ranking 所需的轻量验证。

当前主训练实验把 answer reward 配成 `entity_sink`：reward 先在 entity 层定义，再对最终
`submit` 动作附加一个显式 terminal backward 近似，而不是继续把 alias-count 因子直接塞进
reward 本身。

## 3. 验证 / 测试逻辑

验证与测试主链在：

- `src/metrics/answer_reachability/runtime.py`
- `src/metrics/answer_reachability/batch_evaluator.py`
- `src/metrics/answer_reachability/analysis.py`
- `src/metrics/answer_reachability/flow_frontier.py`
- `src/metrics/answer_reachability/monte_carlo.py`
- `src/metrics/answer_reachability/posterior.py`
- `src/metrics/answer_reachability/metrics.py`

### 3.1 rank_only

`metrics_profile=rank_only` 时：

- 只做 reachability analysis，不发 support window。
- answer-reachability 默认走 `FlowFrontierReachabilityAnalyzer`，在共享
  `prepared_batch` 上按图做 deterministic frontier expansion。
- 如果显式配置 `support_search_method=monte_carlo`，或者任务是 `edge_retrieval`，
  则退回 legacy Monte Carlo analyzer。
- 构建 answer posterior。
- 输出 `answer/*` 指标。
- 不做 support-window search，因此没有 `window/*` 指标。

这一路径适合训练期验证或 edge retrieval view。

### 3.2 full

`metrics_profile=full` 时：

- 默认先做 flow-frontier reachability analysis：
  - 用起点 flow 归一化得到 per-graph `log Z`
  - 从起点集合建立 deterministic frontier
  - 对每个 child state 计算 `F(s)`，并用 `F(s) / Z < flow_prune_epsilon` 做 pruning
  - 用保留下来的 terminal trajectories 直接累积 answer posterior 与 gold mass
- 再用 `FlowFrontierSupportSearch` 从 discovered trajectories 组装 support window。
- 如果显式切到 `support_search_method=monte_carlo`，则使用 legacy Monte Carlo
  analysis/search。
- 最终同时输出：
  - `answer/*`
  - `window/*`
  - `cert/*`

这一路径适合正式评估与 artifact 生成。

## 4. 指标词典

### 4.1 `answer/*`

定义见 `src/metrics/answer_reachability/posterior.py`。

- `answer/gold_mass`: gold answers 的总概率质量；exhaustive flow-frontier 下是精确值，
  budget/pruning 命中时配合 `cert/*` 解读，legacy Monte Carlo 路径下则是估计值。
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

- `cert/remaining_mass_upper`: 未被当前 support window 覆盖的保守质量上界；
  flow-frontier 路径来自被 prune / 未展开 frontier 的 normalized flow 质量，legacy
  Monte Carlo 路径来自区间估计。
- `cert/coverage_rate`: 当前窗口是否给出 coverage certificate；flow-frontier 在未命中
  搜索预算时可以置真，Monte Carlo 路径默认不会置真。

### 4.4 `edge/*`

定义见 `src/metrics/answer_reachability/edge_eval.py`。

- `edge/mrr`: 第一个正例边的倒数排名。
- `edge/hit@k`: 前 `k` 条边是否包含任一 shortest-path positive edge。
- `edge/precision@k`: 前 `k` 条边中正例比例。
- `edge/recall@k`: 前 `k` 条边覆盖正例边的比例。
- `edge/gold_mass`: 正例边对应的 success mass 估计；当前 edge retrieval 固定走 Monte
  Carlo backend。

## 5. split 与执行模式

### 5.1 `run.execution_mode`

- `predict`: 推荐评估模式；会汇总 predict metrics 并写 prediction artifacts。
- `test`: 调用 Lightning `trainer.test()`；保留为更接近标准 test loop 的模式。

### 5.2 `run.split`

当前评估 split 由 `run.split` 控制，`src/eval.py` 会显式把它传给
`GraphRetrievalDataModule.set_eval_split()`。

### 5.3 `run.run_all_splits`

如果启用，`BaseEvalRunner` 会按顺序重放多个 split，并在每次评估前临时覆盖
`cfg.run.split`。datamodule 会据此加载对应的 train/validation/test 数据。

### 5.4 `run.dataset_variants`

answer-reachability 正式评估通常会同时跑 `full` 和 `sub` 两个 dataset scope；runner
会为每个 variant 单独重放 split 并分别保存 metrics/artifacts。

## 6. 关键配置

主要配置位于 `src/models/configs/gflownet.py`：

- `metrics_profile`: `full` 或 `rank_only`
- `task`: `answer_reachability` 或 `edge_retrieval`
- `support_search_method`: `flow_frontier` 或 `monte_carlo`
- `flow_prune_epsilon`: flow-frontier pruning 阈值
- `answer_mass_threshold`: answer posterior 截断阈值
- `support_mass_threshold`: support window 目标阈值
- `support_path_overlap_penalty`: 多条 support path 的重叠惩罚
- `monte_carlo_rollouts`, `monte_carlo_confidence`: legacy / edge-retrieval Monte Carlo 预算
- `max_expansions`, `max_frontier_size`, `strict_search`: search 预算与严格性

代码与 Hydra 配置统一使用 `metrics_profile` 与 `task`，不再保留旧别名。
