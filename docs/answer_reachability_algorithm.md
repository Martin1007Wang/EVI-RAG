# Answer Reachability 当前主线算法说明

本文档描述仓库当前真正运行的主线：

- 模型只有一个：`src.models.gflownet_module.GFlowNetModule`
- 训练目标只有一个：`SubTB`
- `h` 函数只有三个变体：`topology`、`embedding`、`learned`
- answer reachability 仍然是唯一任务层，但它现在负责 exact 评估、support-window search 和产物写出，而不是再维护第二套模型实现

如果你想先看目录边界而不是算法细节，优先读
`docs/gflownet_architecture.md`。

如果你想看当前训练目标的公式化解释，继续读
`docs/answer_reachability_math_derivation.md`。

## 1. 当前代码主链路

最重要的文件是：

- `configs/model/gflownet.yaml`
- `src/models/gflownet_module.py`
- `src/models/policy/search_policy.py`
- `src/models/policy/trajectory_policy.py`
- `src/models/components/scoring.py`
- `src/models/policy/heuristic_utils.py`
- `src/models/training/sampler.py`
- `src/models/training/losses.py`
- `src/tasks/answer_reachability/rollout.py`
- `src/tasks/answer_reachability/execution.py`
- `src/tasks/answer_reachability/exact_reachability.py`

它们的分工是：

- `src/models/gflownet_module.py` 负责 Lightning 训练、验证、预测和 artifact 导出
- `src/models/policy/search_policy.py` 负责起点分布、node-time state score、graph log Z 和 trajectory heuristic wiring
- `src/models/policy/trajectory_policy.py` 负责基础 start / move 参数化与 batch encoding glue
- `components/scoring.py` 提供主线公开的 scoring heads
- `src/models/policy/heuristic_utils.py` 负责 topology / embedding 两种无参 heuristic 计算
- `src/models/policy/heuristic.py` 负责根据配置选择并应用 trajectory heuristic
- `src/models/training/sampler.py` 负责前向 rollout 采样，并把轨迹整理成 SubTB 需要的张量
- `src/tasks/answer_reachability/rollout.py` 负责 answer-terminal 判定与 reward 语义
- `src/models/training/losses.py` 只实现单一 `SubTrajectoryBalanceLoss`
- `tasks/answer_reachability/*` 负责 exact gold-mass 分析、rank metrics、support-window search 和预测产物

## 2. 问题表述

给定问题 `x` 和样本图 `G_x = (V_x, E_x)`，模型要在图上定义一个有限 horizon 的前向过程。

当前主线的训练与评估已经明确分开：

- 训练使用 sampled rollouts + SubTB
- 评估和预测使用 exact answer reachability analysis

两者共享同一套 start / move policy，而不是维护两套不同模型。

## 3. 状态与策略

当前状态是 node-time 形式：

```text
s_t = (v_t, t)
```

其中：

- `v_t` 是当前节点
- `t` 是已经执行的 move 数

主线策略不再显式建模完整 prefix 历史。时间信息通过：

- `step_embedding(t)`
- `remaining_embedding(T - t)`

与当前节点特征相加后得到 state feature。

### 3.1 起点分布

起点只从 `q_local_indices` 对应的 question entities 中选。

`GFlowNetPolicy.compute_start_distribution(...)` 会：

- 取出 question entity 候选节点
- 用 `StartLogitHead` 计算 logits
- 在每个图的候选集合内做 softmax

如果某个图没有任何有限起点候选，会抛出 `InvalidStartCandidatesError`。

### 3.2 前向 move 分布

`GFlowNetPolicy.compute_forward_distribution(...)` 会：

- 构造 child node-time state
- 对 child state 计算 score
- 把 child score 当作 edge logits
- 再通过 `apply_forward_constraints(...)` 执行 horizon 约束

主线没有第二套 sampled policy。训练采样、exact analyzer 和 support-window search 都共享这一套 move 参数化。

### 3.3 graph log Z

除了 start / move 两类打分外，主线还保留一个 graph-level `log Z(x)` 头：

- 输入是 question summary 与 graph summary
- 输出用于 SubTB 根边界项

这个头只在训练损失里使用，不直接决定 answer posterior 的 exact 评估逻辑。

## 4. trajectory heuristic 三个变体

当前主线只有三种 trajectory heuristic：

1. `topology`
2. `embedding`
3. `learned`

learned trajectory heuristic 的内部 head 在 `src/models/components/heuristic_heads.py`，无参辅助计算在
`src/models/policy/heuristic_utils.py`。

### 4.1 `topology`

- 从 question entities 出发做图上传播
- 生成每个节点的 `log h(v)`
- 适合把局部拓扑接近性直接作为 bias

### 4.2 `embedding`

- 直接比较 node token 与 question token 的语义相似度
- 用 cosine similarity 构造 `log h(v)`
- 适合快速引入语义先验

### 4.3 `learned`

- 对 state feature 和 question feature 再接一个小 MLP
- 由模型自己学出 `h(s)`
- 它是当前唯一会额外引入 heuristic head 参数的变体

### 4.4 trajectory heuristic 的作用位置

`h` 只做 bias，不引入第二个训练目标：

- start logits 上加 `beta * log h`
- edge logits 上加 `beta * log h`

也就是说，`h` 影响采样分布和 exact search 排序，但训练目标仍然只有 `SubTB`。

## 5. 训练：sampled rollouts + SubTB

训练主链路在 `GFlowNetModule.training_step(...)`：

```text
prepare_batch
-> sample rollouts
-> compute SubTB
-> log train metrics
```

### 5.1 rollout 语义

`ForwardTrajectoryGFNSampler` 的 rollouts 采用 absorbing 语义；具体的 answer-terminal 判定和 reward 规则则由 `src/tasks/answer_reachability/rollout.py` 提供：

- 当前节点命中 gold answer 时，轨迹记为 hit 并停止
- 没有合法 move 时，轨迹停止
- 达到 `max_steps` 时，轨迹停止

reward 定义也只有一套：

- 命中 gold：reward = 1
- 否则：reward = `epsilon` 或 graph-normalized `epsilon`

### 5.2 SubTB 是唯一损失

`SubTrajectoryBalanceLoss` 会把一条 rollout 转成：

- 根边界 `log Z(x)`
- 起点状态值 `log F(s_0)`
- 每一步 child 状态值 `log F(s_t)`
- 轨迹前缀概率 `log P(prefix_t)`
- 终点 reward anchor `log R`

然后在整条子轨迹上最小化加权残差平方。当前版本没有：

- DB / TB / SubTB 混合权重
- 独立 critic loss
- 第二套 objective head

训练日志里保留的核心指标只有：

- `train/loss`
- `train/subtb_loss`
- `train/subtb_residual`
- `train/subtb_root`
- `train/rollout_success`

## 6. 评估：exact answer reachability

验证和预测不直接复用 sampled SubTB，而是通过任务层执行 exact 分析。

`AnswerReachabilityExecution` 会：

- 调用 `ExactReachabilityAnalyzer` 精确求 answer mass
- 构造 rank-only 或 support-window 结果
- 聚合 answer posterior metrics 和 support metrics

因此当前主线是：

- 训练目标：GFlowNet + sampled SubTB
- 验证目标：exact answer reachability metrics

这不是两套模型，而是同一策略的两种使用方式。

## 7. 当前不再保留的东西

下面这些都已经不属于主线：

- `trajectory_policy` 命名和兼容层
- `guidance_cfg` / `GuidanceConfig` 命名
- 多损失并行训练接口
- 多个模型配置 alias

现在唯一推荐入口就是：

- model config: `configs/model/gflownet.yaml`
- model class: `src.models.gflownet_module.GFlowNetModule`

## 8. 推荐阅读顺序

如果你在继续改这个仓库，建议按下面顺序读：

1. `docs/gflownet_architecture.md`
2. `docs/answer_reachability_algorithm.md`
3. `docs/answer_reachability_math_derivation.md`
4. `src/models/gflownet_module.py`
5. `src/tasks/answer_reachability/execution.py`
