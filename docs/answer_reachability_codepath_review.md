# Answer Reachability Codepath Review

本文不是数学推导，而是一次“按当前代码实现复盘”的说明文档。

目标只有三个：

1. 说清楚训练/验证/最终评估到底算了什么。
2. 说清楚这些量是怎么一步一步被算出来的。
3. 说清楚当前实现里最小的重复计算单元，也就是 runtime 真正花时间的“原子操作”。

如果你希望先看更抽象的算法说明，可以配合阅读：

- `docs/answer_reachability_algorithm.md`
- `docs/answer_reachability_math_derivation.md`
- `docs/gflownet_architecture.md`

## 1. 一句话总览

当前主线是一个 answer-reachability GFlowNet：

- 训练时，对每张图先编码，再从问题节点集合里采样起点；
- 然后在图上 rollout 若干条 trajectory；
- 对每条 trajectory 记录 target-policy `log P_F`、state flow `log F`、terminal reward `log R`；
- 用 `SubTB` 目标训练；
- 验证时默认做 deterministic flow-frontier reachability analysis，不做 support-window search；
- 最终评估时复用同一套 flow-frontier 搜索结果，再组装 answer posterior 和 support window；
- edge retrieval 和显式 legacy fallback 仍保留 Monte Carlo 路径。

如果只看计算量，当前系统最核心的代价不是 loss 本身，而是：

- 图编码；
- rollout 过程中对 active state 的特征构造；
- 对候选边的打分；
- 评估期 retained frontier 上的大量 prefix state 扩展与 candidate 打分。

## 2. 主链路入口

### 2.1 训练入口

训练主链路：

`src/train.py` -> `AnswerReachabilityTrainRunner.run()` -> `train_model()` -> `trainer.fit()` -> `GFlowNetModule.training_step()`

关键文件：

- `src/train.py`
- `src/runs/answer_reachability.py`
- `src/utils/entrypoint_utils.py`
- `src/models/gflownet_module.py`

### 2.2 评估入口

评估主链路：

`src/eval.py` -> `AnswerReachabilityEvalRunner.run()` -> `evaluate_model()` -> `trainer.predict()` / `trainer.test()` -> `GFlowNetModule.predict_step()` / `validation_step()` / `test_step()`

关键文件：

- `src/eval.py`
- `src/runs/common.py`
- `src/metrics/answer_reachability/runtime.py`
- `src/metrics/answer_reachability/batch_evaluator.py`

## 3. 训练时到底算了什么

当前 `training_step()` 的主链路非常明确：

1. 把 dataloader 输出整理成 `TrajectoryBatch`
2. `policy.prepare_batch()` 编码图与问题
3. adaptive controller 决定本步 rollout 数和采样温度
4. `sampler.sample()` 做 on-policy rollout
5. `SubTrajectoryBalanceLoss.compute()` 计算 actor loss
6. 统计 rollout 指标
7. 如果 replay 打开，则重放历史成功路径并计算 replay loss
8. 可选 guidance loss
9. 把成功路径写入 replay buffer
10. 记录日志，更新 adaptive controller 的 EMA 统计

对应代码在：

- `src/models/gflownet_module.py`
- `src/models/gflownet/sampler.py`
- `src/models/gflownet/losses.py`
- `src/models/gflownet/replay.py`
- `src/models/gflownet/adaptive_sampling.py`

### 3.1 输入批次是什么

训练 batch 最终会被整理成 `TrajectoryBatch`。它不是普通 token batch，而是“图 + 问题 + 实体/关系 embedding + 图内索引结构”的统一运行时表示。

这里面最重要的字段是：

- `node_ptr`, `edge_index`, `edge_batch`: 表示一个 disconnected graph batch
- `node_global_ids`, `edge_rel_global`: 图节点和边的全局实体/关系 id
- `q_local_indices`: 问题起点候选节点
- `a_local_indices`, `answer_entity_ids`: gold answer supervision
- `sample_ids`: replay / metric / artifact 对应的样本标识

所以从训练视角看，一次 batch 不是一串 token，而是一组带问题条件的图检索样本。

### 3.2 `policy.prepare_batch()` 算了什么

这一步是“每个训练 batch 只做一次”的图编码阶段。

它做的事情是：

1. 把 `TrajectoryBatch` 变成 `GraphTopology + GraphObservation`
2. 提取 node / relation / question / question-context 表征
3. 过 GNN backbone 编码图
4. 构建 heuristic cache

输出是 `PreparedGFlowNetBatch`，核心字段包括：

- `topology`
- `observation`
- `node_tokens`
- `relation_tokens`
- `question_tokens`
- `question_context_tokens`
- `question_context_mask`
- `heuristic_cache`

这一步本质上是在做“图级共享前向缓存”。后面的多条 rollout 会反复复用这份编码结果。

### 3.3 起点分布算了什么

每张图的 start candidates 来自 `q_local_indices`。当前实现里，起点不是全图任意节点，而是问题节点集合。

计算流程：

1. 枚举 start candidates
2. 用 state-flow head 对每个 candidate 计算 `start_log_flow`
3. 对同一张图内的 start candidates 做 segmented `logsumexp`
4. 得到每张图的 `graph_log_z`
5. 对候选起点归一化得到 `log_probs`
6. 每张图每条 rollout 采样一个 start node

所以当前 start distribution 真实在算的是：

- 每个起点状态的未归一化状态流 `F(s_0)`
- 每张图的 partition-like quantity `Z = sum F(s_0)`
- 每条 rollout 的起点采样概率

从张量形状上看，采样后会得到：

- `start_nodes [num_graphs, num_rollouts]`
- `start_log_probs [num_graphs, num_rollouts]`
- `start_state_log_f [num_graphs, num_rollouts]`

### 3.4 rollout 过程中每一步算了什么

这是训练最核心也最耗时的部分。

对于每个 active rollout state，当前代码会构造 `SearchState`，其中包含：

- 当前所在节点 `current_nodes`
- 已走步数 `num_steps`
- 已经积累的 path token 序列 `path_token_ids`
- 当前 recurrent controller `control_state`

然后每一步都会做以下操作：

1. 找出当前节点的 outgoing edge candidates
2. 构造 state feature
3. 如果度太大，可先做 candidate shortlist
4. 对 surviving candidates 计算 forward logits
5. 加上 submit action
6. 形成 forward action distribution
7. 用 behavior logits 采样动作
8. 同时记录 target-policy `log P_F`
9. 对 graph move 更新路径 token、step count 和 recurrent `control_state`
10. 计算 next-state `log F` 与 backward log-prob
11. 更新 terminal node、success mask 和 trace

这里要特别注意，当前 state feature 不是“当前节点 embedding 直接拿来打分”，而是两层状态的组合：

- 离散 prefix bookkeeping：`path_token_ids` 精确保留整段路径，用于 backward / replay / trace
- 连续前向表征：`control_state` 先注意问题 token，再通过 GRU 吸收“关系 + 目标节点”信息
- 当前节点 token 与 `step_embed(t)` / `remaining_embed(H-t)`

真正进入前向打分的 state feature 来自：

```text
state_feature = MLP([node + time embeddings; control_state])
```

forward actor 再把这个 `state_feature` 和“静态候选节点表示 + relation 表示”拼起来打分。
所以当前模型的环境状态是 exact prefix state，而前向神经表征是它的 recurrent summary。

### 3.5 terminal reward 算了什么

当前主训练实验默认 reward mode 是 `entity_sink`。

从代码语义上看，现在这条分支做的是：

1. 把 terminal node 映射成 terminal entity id
2. 判断 terminal entity 是否属于 gold answer set
3. 构造 base reward

当前 base reward 形式是：

- gold answer: `epsilon + exp(beta * positive_utility)`
- non-gold answer: `epsilon + exp(beta * negative_utility)`

然后再叠加两个惩罚：

- cycle penalty: 对重复/环路步数做衰减
- failure length penalty: 对失败轨迹随长度增加而衰减

因此训练时真正进入 loss 的 terminal supervision 是：

- `terminal_rewards`
- `terminal_log_rewards`
- `success_mask`
- `terminal_num_steps`

### 3.6 `SubTB` loss 算了什么

`SubTrajectoryBalanceLoss.compute()` 不是只看 terminal reward，也不是只看一跳 transition。它会把整条 trajectory 上很多子段都拿来构造残差。

它主要使用这些量：

- `start_state_log_f`
- `next_state_log_f_steps`
- `log_pf_steps`
- `log_pb_steps` 的形状一致性（当前实现记录它，但 residual 主体并未直接使用）
- `terminal_log_rewards`
- `terminal_action_counts` / `terminal_num_steps`

内部核心步骤是：

1. 构造 forward prefix sums
2. 得到每个 prefix state 的 `log F`
3. 构造任意 prefix-to-prefix 的 pairwise residual
4. 构造 prefix-to-terminal 的 residual
5. 用 `lambda_weight` 给长子段衰减加权
6. 对残差平方求和并平均

因此它算的不是一个单点 loss，而是一整组 sub-trajectory consistency residual 的加权均值。

当前实现还会额外记录：

- `success_rate`
- `residual_abs`
- `residual_variance`
- `root_abs`
- `log_z_mean`
- `log_z_variance`

这些主要用于监控，而不是都直接进入优化目标。

### 3.7 replay 和 adaptive sampling 各算了什么

#### success replay

replay 的逻辑是：

1. 从当前 on-policy batch 中收集成功轨迹
2. 去重后写入 buffer
3. 下一些 step 开始，从 buffer 中抽取当前 batch 对应 sample 的成功轨迹
4. 对这些轨迹做 teacher-forced 重打分
5. 用同一个 `SubTB` loss 再算一遍 replay loss
6. 按 trajectory 数量把 on-policy loss 和 replay loss 加权平均

所以 replay 没有改变目标函数形式，只是改变了“哪些 trajectory 被重新看了一遍”。

#### adaptive sampling

adaptive sampling 不改 loss，只改下一步的 rollout budget 和采样温度。

它会缓冲和 EMA 的量包括：

- success rate
- 每百条 rollout 的新成功路径数
- SubTB residual variance
- normalized start entropy

这些量的作用是控制：

- 下一个 step 用多少条 rollout
- behavior sampling 的 temperature multiplier

换句话说，adaptive sampling 是“采样控制器”，不是“新损失项”。

## 4. 评估时到底算了什么

### 4.1 训练期 `rank_only` validation

训练期间默认只跑 `rank_only`。

这一路径只关心 answer posterior，不关心 support window。

它做的事是：

1. 对一整个 disconnected graph batch 做一次图编码
2. 计算 batched start distribution 与每张图的 `graph_log_z`
3. 在共享 `prepared_batch` 上按图运行 flow-frontier search
4. 直接累积 terminal mass、answer posterior 和 gold total mass
5. 构造 answer ranking 指标

输出的核心量有：

- `answer_probs`
- `gold_total_mass`
- `probe_count`
- `remaining_mass_upper`
- `stop_reason`
- `answer/hit@k`
- `answer/recall@k`
- `answer/precision@k`
- `answer/f1@k`

这一路径不会做：

- support-window generation
- support path selection
- `window/*` 指标

如果显式配置 `support_search_method=monte_carlo`，或者当前任务视图是
`edge_retrieval`，同一个 batch evaluator 仍会切回 legacy Monte Carlo analyzer。

### 4.2 full final eval

正式 final eval 的 `metrics_profile=full` 时，评估不只是 answer posterior，还要把
deterministically discovered trajectories 组织成 support window。

当前默认 full flow-frontier 路径的步骤是：

1. 对 batched disconnected graphs 做一次图编码
2. 计算起点 flow、per-graph `log Z` 和初始 frontier
3. 对每张图按 frontier 分层展开，并用 `F(s) / Z < flow_prune_epsilon` 做 pruning
4. 直接累计 discovered trajectories 的 path probability，构造 answer posterior
5. 选择满足 `answer_mass_threshold` 的答案前缀
6. 对每个选中答案，从 discovered paths 中贪心选 support paths
7. 直到达到 `support_mass_threshold * answer_upper_bound`
8. 输出 `SupportWindowResult`

所以 full eval 多出来的计算主要是：

- deterministic frontier expansion 期间的 path bookkeeping
- support path greedy selection
- `window/*` 和 `cert/*` 统计

### 4.3 为什么 full eval 明显更慢

当前慢的主要原因也不是单一 bug，而是 frontier 展开预算乘法：

- full eval 通常还会跑两个 dataset variant：`full` 和 `sub`
- 每张图都会展开一批 retained prefix states
- 每个 retained state 都要再做 candidate 打分与 controller 更新
- 当前默认 `rankflow` eval 用的是 `predict` 模式，还要做 artifact/result 构造

可以粗略地把总成本理解为：

`dataset_variants * expanded_prefix_states * average_candidate_actions * candidate_scoring_cost`

其中真正最贵的仍然是 `candidate_scoring_cost`；`flow_prune_epsilon`、
`max_expansions` 和 `max_frontier_size` 则直接决定评估要走多深、多宽。

## 5. 当前代码里的“原子操作”

这里的原子操作，不是 CUDA kernel 级别，而是从算法实现角度最小、可重复、会被大量调用的计算单元。

| 原子操作 | 重复轴 | 输入 | 输出 | 为什么贵 |
| --- | --- | --- | --- | --- |
| 图编码 | 每个 batch 一次 | 图结构 + entity/relation/question embedding | `PreparedGFlowNetBatch` | GNN 编码整批图 |
| 起点打分 | 每个 start candidate | candidate node + graph context | `start_log_flow` | 所有 rollout 之前都要做 |
| 起点采样 | 每图每 rollout 一次 | graph 内归一化 start probs | sampled start node | 图数和 rollout 数相乘 |
| prefix-state 特征构造 | 每个 unique active prefix state | 当前节点 + time embedding + recurrent control state | state feature | 仍然依赖 prefix 历史，但不再重跑 path self-attention |
| controller 更新 | 每个发生 graph move 的 active rollout | 上一步 `control_state` + question context + relation + next node | next `control_state` | 每步都要做，但成本远小于整段 prefix attention |
| candidate shortlist | 每个高出度 state | 当前 state + candidate edges | top-k candidates | 高度节点会多一次筛选 |
| transition 打分 | 每个 surviving candidate edge | state feature + relation + target node token | edge logits | rollout 中最重的高频操作之一 |
| 动作采样/更新 | 每个 active rollout step | action distribution | next state / submit / success | 每步都做 |
| terminal reward | 每个 rollout 终止时 | terminal entity + gold answers + penalties | `log R(x)` | 相对便宜，但所有 rollout 都有 |
| SubTB 残差汇总 | 每条 rollout | `log F`, `log P_F`, `log R` | per-rollout loss | horizon 小时不算最重 |
| frontier expansion / path packaging | 每张图每次 eval | discovered prefix states / trajectories | `ReachabilityAnalysis` / `SupportWindowResult` | answer-reachability 主线评估成本已明显转到这里 |

## 6. 哪些量是“真正被估计”的

为了防止概念混淆，这里把训练和评估中真正被估计的量拆开写。

### 6.1 训练期

训练期主要估计/学习的是：

- start state flow `F(s_0)`
- 中间状态 flow `F(s_t)`
- forward policy `P_F(a_t | s_t)`
- terminal reward induced target `R(x)`

然后通过 `SubTB` 让这些量在子轨迹层面满足一致性关系。

### 6.2 验证 / 评估期

默认 answer-reachability 验证和评估会通过 flow-frontier deterministically 构造：

- terminal node mass
- answer entity posterior
- gold total mass
- discovered trajectory probability
- support coverage statistics
- omitted mass upper bound

如果显式切回 Monte Carlo backend，上面这些量才重新变成 sampled estimates 和区间。
因此当前主线 answer-reachability 评估已经不再是纯 sampling-based estimator；真正仍然
完全依赖 Monte Carlo 的是 edge retrieval 和 legacy fallback。

## 7. 当前实现里几个容易误判的点

### 7.1 `rank_only` 和 `full` 的 posterior 来源其实一样

默认 answer-reachability 下，两者都来自同一个 flow-frontier search summary。`full`
只是多了一层 support window 组装，并不是另一个完全不同的 ranking backend。

### 7.2 `strict_search` / `max_expansions` / `max_frontier_size` 现在会直接影响 runtime 和证书质量

这些参数已经重新成为主线 runtime 的真实预算：它们决定 frontier 能展开多深、多宽，以及
`remaining_mass_upper` 会不会因为 budget 命中而变松。

### 7.3 `entity_sink` 在当前代码里更像 entity-level reward，而不是独立的新搜索器

它主要决定 terminal reward 的语义，不会把训练逻辑改成另一种 rollout 算法。

### 7.4 `graph_log_z` 不再只是监控量

训练期 `SubTB` residual 主体仍然不是显式对 `graph_log_z` 单独回传一个独立 loss 项；但在
评估期，`graph_log_z` 已经是 flow-frontier search 的全局归一化常数，也是 pruning 与质量
证书的分母。

### 7.5 current full eval 的慢，不是因为 batch 没做起来

`rank_only` 和现在的 full flow-frontier support-search 都已经能复用同一个 disconnected
graph batch encode。真正的慢主要来自 retained frontier 太大、suite 太宽，以及每步
candidate 打分太重。edge retrieval 那条分支仍然会受 Monte Carlo rollout 预算影响。

## 8. 用这份复盘去理解“为什么现在慢”

如果只从当前代码看，速度问题可以按下面顺序理解：

1. 一个 batch 先做一次图编码
2. 然后对每张图、每个 retained prefix state、每个候选 action，不断做 state feature、controller update 和 candidate edge scoring
3. full eval 还要把 discovered trajectories 组装成 support window 和 artifacts
4. 还会对 `full` 和 `sub` 两个 dataset variant 各跑一遍

因此系统慢的根源更接近：

- `frontier budget` 太大
- `candidate scoring` 太贵
- `suite width` 太宽

而不是某一个单独的日志打印或 metrics 聚合函数。只有 edge retrieval / legacy fallback
那条分支，才仍然主要受 Monte Carlo rollout budget 支配。

## 9. 建议阅读顺序

如果后续还要继续做性能优化，建议按这个顺序读代码：

1. `src/models/gflownet_module.py`
2. `src/models/gflownet/policy.py`
3. `src/models/gflownet/sampler.py`
4. `src/models/gflownet/losses.py`
5. `src/models/gflownet/replay.py`
6. `src/metrics/answer_reachability/batch_evaluator.py`
7. `src/metrics/answer_reachability/flow_frontier.py`
8. `src/metrics/answer_reachability/monte_carlo.py`
9. `src/metrics/answer_reachability/posterior.py`

只要把这 9 个文件串起来看，就能把“训练在算什么、验证在算什么、full eval 为什么慢”连成一条完整主线。
