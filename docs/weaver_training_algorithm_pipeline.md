# Weaver 当前算法复盘

本文按当前仓库实现复盘 `Weaver` 的训练与推理算法，重点回答三个问题：

- 当前状态空间到底怎么定义
- 当前训练目标到底在优化什么
- 当前 rollout / replay / reward / evaluation 怎样串起来

本文只描述代码现在实际做的事情，不追溯旧版本设计。主要对应代码：

- [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:1)
- [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:1)
- [src/weaver/feature.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/feature.py:1)
- [src/weaver/policy/forward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy/forward.py:1)
- [src/weaver/policy/backward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy/backward.py:1)
- [src/weaver/objectives/subtb/](/mnt/wangjingxiong/EVI-RAG/src/weaver/objectives/subtb/:1)
- [src/weaver/reward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/reward.py:1)
- [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:1)
- [src/weaver/rollout/replay.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/replay.py:1)
- [src/eval/rollout.py](/mnt/wangjingxiong/EVI-RAG/src/eval/rollout.py:1)

## 1. 与旧版文档的关键差异

当前实现已经不是旧文档描述的那套算法，至少有五处必须纠正：

- 训练目标已从 `Detailed Balance` 切换到 `SubTrajectory Balance`
- 旧的 `edge_flow_matching`、`transition_batch`、`transition_builder` 已经删除
- `StateEncoder` 不再使用 `active-node mean pool`，而是用 `question_h` 对 selected edges 做 attention pooling
- 当前边动作打分是 `question_h · relation_h` 谓词对齐、`question_h · edge_h` 边语义项和 state-conditioned marginal MLP 的加和
- backward kernel 不再能简化成 `-log(1 + |S_z|)`，因为合法前驱数量取决于 root-reachability 和 parent active set

后文都以当前代码为准。

## 2. 任务建模与状态空间

对每个样本，已知：

- 查询 `q`
- 有向候选图 `G = (V, E)`
- 锚点集合 `A`
- 目标答案节点集合 `Y`
- 最大扩展预算 `B`

算法不直接对整张图排序，而是在预算内逐步选择边，构造证据子图。

当前状态 `z` 用 canonical edge-set state 表示，也就是只记录“当前已经选了哪些边”，与到达顺序无关。实现里 `StateBatch` 的状态键由以下部分组成：

- `graph_ids`
- 排序后的 `edge_ids`
- `edge_count`
- `budget`

对应 [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:47)。

### 2.1 active nodes

当前状态的 active nodes 不是抽象概念，而是运行时真正在用的可扩展节点集合：

```text
X_z = A ∪ endpoints(S_z)
```

也就是：

- 初始 active nodes 来自锚点
- 每条已选边会把自己的 `src` 和 `dst` 都并入 active set
- 最终做去重

对应 [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:287)。

### 2.2 frontier

frontier 定义为从当前 active nodes 出发、且尚未被选中的出边：

```text
C(z) = { e = (u, v) ∈ E \ S_z : u ∈ X_z }
```

动作空间为：

```text
𝒜(z) = {STOP} ∪ C(z)
```

若 frontier 为空，或者预算已经耗尽，则该状态只能终止。对应 [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:234)。

### 2.3 root-reachable 约束

当前状态必须满足 root-reachable：每条已选边都必须能通过一条从锚点出发的已选路径到达其源点。实现采用迭代传播判定，见 [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:321)。

这个约束不仅限制合法状态，也会影响 backward kernel，因为不是每个“删去一条边”的 parent 都合法。

## 3. 离线特征与预处理产物

训练 batch 进入模型前，依赖三类离线语义产物：

- 实体文本语义表
- relation 语义表
- relation-neighborhood 伪特征表

三类原始语义向量都来自同一个 BGE encoder，并在预处理时做 L2 normalize。`FeatureEncoder` 将它们分别投影到统一的模型空间：

```text
question_h = LN(Linear_question(question_emb))
entity_h   = LN(Linear_entity(entity_sem))
relation_h = LN(Linear_relation(relation_sem))
```

三个 `Linear` 都使用 `bias=False`。`LayerNorm` 代替输出端 L2 约束，允许模型学习各向异性的决策空间。对应 [src/weaver/feature.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/feature.py:76)。

### 3.1 Entity 特征

实体特征优先使用实体文本语义；只有实体没有文本时，才回落到 relation-neighborhood 伪特征。对应 [src/weaver/feature.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/feature.py:143)。

relation-neighborhood 的构造方式为：

```text
mid_sem(i) = Normalize( Σ relation_sem(r) ),   r ∈ incident_relation_types(i)
```

其中：

- relation type 会先按实体去重
- 当前版本故意不区分入边和出边关系类型
- 任何无文本实体都必须至少在一条保留图边中出现，否则预处理直接报错

对应 [src/data/preprocess/relation_neighborhood.py](/mnt/wangjingxiong/EVI-RAG/src/data/preprocess/relation_neighborhood.py:1)。

### 3.2 Relation 和 Edge 特征

`relation_h` 先按 `edge_relation_catalog_ids` 去重投影，再按边展开为 `[E, H]`，同一 relation type 的所有边共享同一个向量。

`EdgeEncoder` 内部不再做 question/entity/relation 对齐，只对已经在统一模型空间中的三元组表示做一次线性融合：

1. 从 `entity_h` 取 `src_h` 和 `dst_h`
2. 拼接 `[src_h || relation_h || dst_h]`
3. 做 `Linear(3H→H, bias=False) + LayerNorm(H)`

```text
edge_h(e) = LN(W_e [src_h || relation_h || dst_h])
```

`relation_h != edge_h`：前者是纯 relation 语义，供 FlowEstimator Path1 做谓词对齐；后者包含 `src/relation/dst` 结构交互。对应 [src/weaver/feature.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/feature.py:20)。

### 3.3 replay bank

replay 不是在线经验回放缓存，而是预处理阶段构造的弱监督轨迹库。训练时只按当前预算和 round variant 取出一批已排序好的合法扩展序列。运行期逻辑在 [src/weaver/rollout/replay.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/replay.py:1)。

## 4. 前向策略参数化

### 4.1 StateEncoder 的输入

当前 `StateEncoder` 编码两路信息：

- selected edges 的 question-conditioned attention summary
- 原始 `question_h`

对应 [src/weaver/feature.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/feature.py:31)。

更具体地说：

```text
h_sel(z) = AttnPool(question_h, selected_edge_h)
state_h(z) = LN(W_s [h_sel(z) || question_h])
```

要点有两条：

- question 作为 attention 的 Q 向量进入状态编码
- fusion 时显式拼回 `question_h`

### 4.2 边动作打分

当前边动作打分由两个路径相加：

```text
phi_relation(e) = (question_h · relation_h(e) + λ question_h · edge_h(e)) / sqrt(H)
phi_mgn(z, e)   = MLP([state_h(z) || edge_h(e) || state_h(z) ⊙ edge_h(e)])
edge_logit(z,e) = phi_relation(e) + phi_mgn(z,e)
```

对应 [src/weaver/policy/forward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy/forward.py:50)。

实现上：

- `PolicyCache` 缓存 `question_h_by_graph`、`edge_h` 和 `relation_h`
- rollout 时按 frontier 取对应 `edge_h` 和 `relation_h`

### 4.3 STOP 和 state flow

停止动作由 `StopPolicyHead([state_h || question_h])` 产生一个标量 logit：

```text
stop_logit(z) = StopPolicyHead([state_h(z) || question_h])
```

训练目标中的状态流由 `StateFlowHead(state_h)` 直接输出：

```text
log F(z) = StateFlowHead(state_h(z))
```

其中：

- `stop_logit` 参与 rollout 动作采样
- `log F(z)` 只在训练 objective 中使用，不参与推理动作选择

### 4.4 前向分布

最终前向策略是 parent-local 的：

```text
P_F(. | z) = softmax({stop_logit(z)} ∪ {edge_logit(z, e) : e ∈ C(z)})
```

若 `C(z)` 为空，则该状态退化为 forced terminal，`STOP` 的对数概率为 `0`。这一行为由 `PolicyOutput` 和边界状态测试覆盖，见 [tests/test_detailed_balance.py](/mnt/wangjingxiong/EVI-RAG/tests/test_detailed_balance.py:15)。

## 5. 训练数据流

一次 `training_step` 的主链路在 [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:74)：

1. 从 batch 构造 `GraphContext`
2. 构造 `FeaturePack`
3. 构造 `PolicyCache`
4. 构造 `TargetContext`
5. 构造 `ReplayContext`
6. 在 `torch.no_grad()` 下采样训练轨迹
7. 构造 `SubTrajectoryBalanceBatch`
8. 用 objective 重算前向并求损失
9. 额外记录 reward 指标

训练时 rollout 不回传梯度；参数更新来自 objective 对 unique states 的重算前向。

### 5.1 policy rollout

`RolloutRunner.train_rollouts()` 会先采样 policy trajectories。每条轨迹从空状态出发，直到：

- 采样到 `STOP`
- frontier 为空
- 预算耗尽

这部分提供当前策略真实访问到的状态分布。对应 [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:41)。

### 5.2 replay trajectories

若配置了 `ReplaySource`，训练还会拼接 replay trajectories。当前 replay 逻辑有三点值得明确：

- replay bank 按 `(budget, variant, slot)` 组织
- 当前 round 使用 `variant = replay_round mod round_variants`
- replay 可按 `anneal_steps` 线性衰减保留比例

对应 [src/weaver/rollout/replay.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/replay.py:12)。

### 5.3 SubTB batch 的构造

当前 `build_subtrajectory_balance_batch()` 做的事情很具体：

1. 对每条终止轨迹展开所有 prefix
2. 把每个 prefix 转成排序后的 canonical edge set
3. 对 prefix states 去重
4. 记录每条轨迹每个 prefix 对应的 unique state id

输出包含：

- `trajectories`
- `states`
- `prefix_state_ids`
- `prefix_valid_mask`

对应 [src/weaver/objectives/subtrajectory_balance_batch.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/objectives/subtrajectory_balance_batch.py:14)。

需要特别说明：当前实现没有再单独构造“unique transitions 表”；SubTB 直接在 trajectory prefix 上累积前向与后向对数概率。

## 6. SubTrajectory Balance 数学目标

当前目标函数在 [src/weaver/objectives/subtrajectory_balance.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/objectives/subtrajectory_balance.py:14)。

### 6.1 训练里会先重算什么

objective 不直接复用 rollout 时记录的 logits，而是先对 unique states 重算：

- `log F(z)`
- `log P_F(STOP | z)`
- 各 frontier 边的 `log P_F(e | z)`
- terminal `log R(z)`

当前实现会对 unique states 一次性重算这些量。

### 6.2 transition residual

对一条轨迹上的任意连续子轨迹 `s_i -> ... -> s_j`，当前实现约束：

```text
δ_trans(i, j) =
    log F(s_i)
  + Σ_{t=i}^{j-1} log P_F(a_t | s_t)
  - Σ_{t=i+1}^{j} log P_B(s_{t-1} | s_t)
  - log F(s_j)
```

如果约束成立，就有：

```text
F(s_i) · Π P_F = F(s_j) · Π P_B
```

当前实现会枚举每条轨迹上的所有连续子轨迹，除非 `max_subtrajectory_length` 限制了最大跨度。

### 6.3 terminal residual

对每个起点 `s_i` 到终止状态 `s_T` 的 suffix，还会加一条 terminal 边界条件：

```text
δ_term(i) =
    log F(s_i)
  + Σ_{t=i}^{T-1} log P_F(a_t | s_t)
  + log P_F(STOP | s_T)
  - Σ_{t=i+1}^{T} log P_B(s_{t-1} | s_t)
  - log R(s_T)
```

如果成立，则：

```text
F(s_i) · Π P_F · P_F(STOP | s_T) = R(s_T) · Π P_B
```

### 6.4 加权方式

当前损失对每条 residual 做平方，并按子轨迹长度做指数衰减加权：

```text
weight(span_len) = subtb_lambda^(span_len - 1)
```

因此：

- `subtb_lambda = 1` 时，各跨度权重相同
- `subtb_lambda < 1` 时，更偏向短子轨迹约束

总损失就是所有 transition residual 和 terminal residual 的加权平方平均。

## 7. backward kernel 的当前实现

旧文档里把 backward 写成：

```text
log P_B(z | z + e) = -log(1 + |S_z|)
```

这在当前实现里已经不成立。

当前 `uniform_backward_log_prob()` 的定义是：对 child state 的所有合法前驱做均匀分布。一个前驱是否“合法”，要同时满足：

- 从 child 删除一条边后，parent 仍然 root-reachable
- 被删边的源点仍然在 parent 的 active set 里

对应 [src/weaver/policy/backward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy/backward.py:9)。

因此：

```text
log P_B(parent | child) = -log(legal_predecessor_count(child))
```

而 `legal_predecessor_count(child)` 一般不等于 `child_edge_count`。这也是当前文档必须改掉 `-log(1 + |S_z|)` 写法的原因。

## 8. Terminal Reward

当前终止奖励由 `TerminalRewardModel` 定义，见 [src/weaver/reward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/reward.py:21)。

对任一状态 `z`，定义：

```text
answer_count(z) = |X_z ∩ Y|
target_count(z) = |Y|
recall(z) = answer_count(z) / max(target_count(z), 1)
success(z) = 1[answer_count(z) > 0 and target_count(z) > 0]
edge_count(z) = |S_z|
coverage(z) = log(answer_count(z) · exp(answer_prize) + eps)
```

则当前实现的 `log_reward` 为：

```text
log R(z) =
    answer_weight · recall(z)
  + coverage_weight · coverage(z)
  - edge_cost · edge_count(z)
  - fail_cost · (1 - success(z))
```

默认超参数来自 [configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:1)：

- `answer_weight = 6.0`
- `coverage_weight = 1.0`
- `edge_cost = 0.15`
- `fail_cost = 6.0`
- `answer_prize = 2.0`
- `eps = 1.0`

这意味着当前 reward 同时鼓励：

- 召回更多答案节点
- 覆盖更多答案
- 用更少的边
- 避免完全 miss 答案

## 9. 推理与评估

验证、测试和预测阶段都不使用 replay，也不计算 SubTB objective。推理只依赖前向策略：

```text
z_0 = ∅
for t = 0, 1, ..., B:
    采样或选择 a_t ~ P_F(. | z_t)
    if a_t = STOP:
        break
    z_{t+1} = z_t + a_t
```

最终输出的是终止状态对应的 canonical edge set。

评估时会先从 `TrajectoryBatch` 直接恢复 terminal `StateBatch`，再基于终止子图计算：

- sample-level retrieval 指标
- union / top-k 指标
- diversity 指标
- terminal diagnostics

对应 [src/eval/rollout.py](/mnt/wangjingxiong/EVI-RAG/src/eval/rollout.py:1)。

因此推理行为完全由 `P_F(. | z)` 决定；`StateFlowHead` 只服务于训练。

## 10. 一句话总结

当前版本 `Weaver` 可以概括为：

```text
在 canonical root-reachable edge-set 状态空间上，
用编码 selected/frontier/budget 的 parent-local policy 逐步扩展证据子图，
并用 SubTrajectory Balance 把前向概率、backward kernel、state flow 和 terminal reward 耦合起来。
```
