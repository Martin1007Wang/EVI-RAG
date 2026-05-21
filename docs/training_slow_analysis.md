# 训练过程缓慢的算法分析报告

## 1. 结论先行

当前仓库里的训练慢，不是单一原因，而是几类成本叠加：

1. 每个 `training_step` 不是一次前向/反向，而是“多次 rollout 采样 + 多次策略重算 + 手工反传累积”。
2. 默认配置把一个 step 切成了大量很小的微批计算：`batch_size=1`、`train_num_rollout=8`、`train_chunk_size=1`。
3. 特征编码器默认是 `lazy: true` 且 `embedding_tables_on_cpu: true`，会在策略热路径里反复做按需查表和 CPU->GPU 搬运。
4. 策略网络内部有大量 Python 级循环，尤其是按 `row`、按 `action`、按 `graph` 重建 forest / successor 表示。
5. 状态表示使用稠密布尔 mask：`node_mask [R, N]`、`edge_mask [R, E]`，rollout 数一上来，内存和带宽成本都会放大。
6. 训练目标需要对同一批 transition 同时算 parent policy、child policy、reward、backward correction，不是一次简单分类 loss。
7. 验证阶段同样不轻，会重复 rollout，并把结果搬到 CPU 上做重建和指标统计。

如果只问“为什么 GPU 利用率不高但训练还是很慢”，当前实现很可能就是典型的 **GPU 计算碎片化 + Python 控制流过重 + CPU/GPU 来回取数**。

---

## 2. 本报告范围

本报告只基于当前仓库代码做静态分析，不假设外部训练环境，也不假设具体数据规模。

我没有在当前工作区找到可直接复用的训练数据物化目录，因此下面的“慢点”分为两类：

- 确定性结论：由当前代码结构直接推出。
- 需要实测量化的结论：代码上高度可疑，但具体占比要结合真实样本规模、图大小、GPU/CPU 带宽来测。

仓库里已经有一个为此准备的探针脚本：[scripts/debug_webqsp_memory_probe.py](/mnt/wangjingxiong/EVI-RAG/scripts/debug_webqsp_memory_probe.py:1)，它本身也说明作者已经在排查 `train_chunk` / `forward_chunk` / `backward` 的显存和 successor 开销。

---

## 3. 训练入口与真实调用链

### 3.1 训练入口

训练入口在 [src/train.py](/mnt/wangjingxiong/EVI-RAG/src/train.py:51)：

1. 读取 Hydra 配置
2. 构建 `datamodule`
3. `setup_datamodule(datamodule)` 载入资源
4. 构建 `model`
5. `trainer.fit(...)`

关键代码：

- `main`: [src/train.py](/mnt/wangjingxiong/EVI-RAG/src/train.py:56)
- `build_datamodule/build_model/build_trainer`: [src/training/factory.py](/mnt/wangjingxiong/EVI-RAG/src/training/factory.py:27)

### 3.2 默认训练配置

默认配置直接放大了训练内环成本：

- `batch_size: 1`: [configs/datamodule/default.yaml](/mnt/wangjingxiong/EVI-RAG/configs/datamodule/default.yaml:5)
- `num_workers: 1`: [configs/datamodule/default.yaml](/mnt/wangjingxiong/EVI-RAG/configs/datamodule/default.yaml:6)
- `train_num_rollout: 8`: [configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:39)
- `train_chunk_size: 1`: [configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:41)
- `expand_budget: 3`: [configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:38)
- `lazy: true`: [configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:14)
- `embedding_tables_on_cpu: true`: [configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:15)

这几项组合起来的直接含义是：

- 每个 step 只处理 1 个图样本。
- 但对这 1 个图样本，要做 8 次 rollout 采样。
- 这 8 次 rollout 不是合成一次大的训练前向，而是按 `chunk_size=1` 切成 8 个 chunk。
- 每个 chunk 都会重新走 `_forward_chunk -> _forward_transitions -> policy(parent) -> policy(child)`。

因此，**一个训练 step 的计算粒度非常碎**。

---

## 4. 数据模块逻辑与接口

### 4.1 DataModule 逻辑

`RetrievalDataModule` 位于 [src/data/datamodule.py](/mnt/wangjingxiong/EVI-RAG/src/data/datamodule.py:11)。

它的职责很克制：

- 读取物化后的数据路径
- 构建 `RetrievalDataset`
- 构建 `DataLoader`
- 暴露 `model_resources`

关键接口：

- `setup`: [src/data/datamodule.py](/mnt/wangjingxiong/EVI-RAG/src/data/datamodule.py:78)
- `train_dataloader`: [src/data/datamodule.py](/mnt/wangjingxiong/EVI-RAG/src/data/datamodule.py:91)
- `_build_loader`: [src/data/datamodule.py](/mnt/wangjingxiong/EVI-RAG/src/data/datamodule.py:137)

### 4.2 Dataset 逻辑

`RetrievalDataset` 位于 [src/data/dataset.py](/mnt/wangjingxiong/EVI-RAG/src/data/dataset.py:58)。

每次 `get(idx)` 会做：

1. 从 split index 读取 `sample_id`
2. 从问题向量表里取 `question_emb`
3. 从 LMDB 读取样本 payload
4. 反序列化为 `RetrievalData`

关键接口：

- `get`: [src/data/dataset.py](/mnt/wangjingxiong/EVI-RAG/src/data/dataset.py:118)
- `LMDBSampleStore.load_sample`: [src/data/dataset.py](/mnt/wangjingxiong/EVI-RAG/src/data/dataset.py:43)
- `_build_retrieval_data`: [src/data/dataset.py](/mnt/wangjingxiong/EVI-RAG/src/data/dataset.py:163)

### 4.3 Collate 逻辑

`RetrievalCollator` 位于 [src/data/collate.py](/mnt/wangjingxiong/EVI-RAG/src/data/collate.py:12)。

它会：

1. 用 PyG 的 `from_data_list` 合并图字段
2. 手工 stack `question_emb`
3. 构造 `edge_batch`

关键接口：

- `__call__`: [src/data/collate.py](/mnt/wangjingxiong/EVI-RAG/src/data/collate.py:37)
- `stack_question_embeddings`: [src/data/collate.py](/mnt/wangjingxiong/EVI-RAG/src/data/collate.py:60)
- `edge_batch_from_node_batch`: [src/data/collate.py](/mnt/wangjingxiong/EVI-RAG/src/data/collate.py:79)

### 4.4 数据阶段是否是主要瓶颈

从代码看，数据阶段不是最可疑的主瓶颈，但有几个次级放大项：

- `batch_size=1`，DataLoader 很难通过 batch 并行摊薄开销。
- `num_workers=1`，预取能力弱。
- 每个样本都要从 LMDB 读取并反序列化多个图张量。

这部分通常会造成：

- step 启动延迟
- CPU 侧吞吐不足
- GPU 等数据

但相比训练内环，**数据加载看起来不是第一优先级瓶颈**。

---

## 5. 训练内环的真实算法

### 5.1 `training_step` 的实际行为

核心训练逻辑在 [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:94)。

单个 `training_step(batch)` 的真实流程如下：

1. `optimizer.zero_grad`
2. `with torch.no_grad()` 下先做一份 rollout 用的特征编码
3. 构造 `rollout_context`
4. 构造 `reward_context`
5. 通过 `runner.train_chunks(...)` 产生多个 chunk
6. 对每个 chunk：
   - 重新执行 `self.feature_encoder(batch)`
   - 执行 `_forward_chunk`
   - `manual_backward`
7. 累积完所有 chunk 后，手工除以 `total_weight`
8. `optimizer.step`

直接对应代码：

- rollout 特征准备: [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:114)
- chunk 循环: [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:135)
- 每个 chunk 重新编码 batch: [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:145)
- 手工反传: [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:155)
- 手工归一化梯度: [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:175)

### 5.2 一个 step 到底会调用多少次特征编码

默认配置下：

- `train_num_rollout = 8`
- `train_chunk_size = 1`

因此 `runner.train_chunks(...)` 会产出 8 个 chunk，见 [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:99)。

于是一个 `training_step` 至少会发生：

1. rollout 前的 `feature_encoder(batch)` 一次
2. 每个 chunk 的 `feature_encoder(batch)` 一次，共 8 次

合计至少 **9 次 `feature_encoder(batch)` 调用 / step**。

虽然默认 `lazy: true` 时，`feature_encoder(batch)` 本身只返回 `LazyFeatureBank`，不是一次完整 dense 编码，但它仍然意味着：

- rollout 和训练前向是两套独立特征访问路径
- 每个 chunk 的梯度图重新建立
- 后续 `node_rows/rel_rows/query_rows` 都会重新触发特征查询

这正是“看起来没有大矩阵前向，但 step 仍然很慢”的典型原因。

---

## 6. Rollout 模块逻辑与成本

### 6.1 RolloutRunner

`RolloutRunner` 位于 [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:69)。

训练时：

- `train_chunks(...)` 按 `train_chunk_size` 切块
- 每个 `train_chunk(...)` 内部会：
  - `engine.sample_rollouts(...)`
  - `transitions_from_rollouts(...)`
  - 可选 replay
  - 拼成 `TransitionBatch`

关键接口：

- `train_chunks`: [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:99)
- `train_chunk`: [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:119)

### 6.2 RolloutEngine

`RolloutEngine` 位于 [src/weaver/rollout/engine.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/engine.py:42)。

核心逻辑：

1. 初始化 `State.initial_from_graph_context(...)`
2. 对 `t in [0, expand_budget]` 迭代
3. 取当前活跃行 `active_rows`
4. 对活跃行跑一次 `policy(...)`
5. 用 `sample_action(...)` 采样 STOP 或 EXPAND
6. 更新 trace
7. `state.apply_edges_(...)`

关键接口：

- `prepare_context`: [src/weaver/rollout/engine.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/engine.py:50)
- `sample_rollouts`: [src/weaver/rollout/engine.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/engine.py:66)
- `_sample_fused_rollouts`: [src/weaver/rollout/engine.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/engine.py:87)

### 6.3 Rollout 的算法复杂度直观理解

设：

- `B` = batch 中图个数，默认这里基本是 1
- `K` = 每图 rollout 数，默认 8
- `T` = `expand_budget + 1`，默认 4
- `R = B * K`

那么 rollout 部分大致会做：

- 最多 `T` 轮策略评估
- 每轮处理最多 `R` 个状态行
- 每轮还要根据 frontier 大小计算所有候选边的 logits

也就是说，rollout 本身就不是 O(1) 的附加步骤，而是一个 **嵌套的状态搜索过程**。

默认配置下，哪怕只训练 1 个图，rollout 也已经在做：

- 最多 8 条轨迹
- 每条轨迹最多 3 次扩展
- 每次扩展前都要重算当前 frontier 的策略

---

## 7. State 与 Frontier 的逻辑和成本

### 7.1 状态表示

`State` 定义在 [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:167)。

核心字段：

- `node_mask: [R, N]`
- `edge_mask: [R, E]`
- `max_budget_by_row: [R]`
- `row_to_graph: [R]`

初始化位置：

- `initial`: [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:188)
- `initial_from_graph_context`: [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:245)

### 7.2 为什么这个表示容易慢

这是一个非常直观但很重的设计：

- 每个 rollout row 都持有一整张图的节点/边布尔掩码。
- 如果某个样本图有 `N` 个节点、`E` 条边，`R` 个并行 row，就要存 `R*N + R*E` 个布尔值。

即使 `bool` 只算 1 byte，单个状态批的显存/内存也是：

`O(R * (N + E))`

这会在两个地方放大：

1. rollout 期间 `State` 本身就要维护这些稠密 mask
2. `TransitionBatch` 会保存 parent/child 两份状态

仓库里的探针脚本也明确打印了这项成本，见 [scripts/debug_webqsp_memory_probe.py](/mnt/wangjingxiong/EVI-RAG/scripts/debug_webqsp_memory_probe.py:126)。

### 7.3 FrontierBuilder 的热路径

`FrontierBuilder.build(state)` 在 [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:76)。

它做了这些事：

1. 对 `state.node_mask` 做 `nonzero`
2. 根据每个活跃节点取 outgoing edges
3. 过滤跨图边
4. 过滤已经选过的边
5. `torch.unique(keys)` 去重
6. 再检查 `src_active & ~dst_active`

这意味着 frontier 构造不是常数开销，而是依赖：

- 当前活跃节点数
- 当前边出度
- 当前 row 数

并且包含：

- `nonzero`
- `repeat_interleave`
- `index_select`
- `unique`

这些操作对小 batch、高频调用的场景很容易形成高内存带宽开销。

---

## 8. Policy 模块逻辑与成本

### 8.1 Policy 接口

`Policy` 位于 [src/weaver/policy.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy.py:37)。

前向接口：

```python
policy(
    context: GraphContext,
    state: State,
    features: FeatureBankLike,
    frontier_builder: FrontierBuilder,
) -> PolicyOutput
```

核心输出：

- `stop_log_flow`
- `edge_log_flow`
- `state_log_flow`
- `action_log_prob`

### 8.2 Policy 内部流程

`Policy.forward(...)` 会做：

1. `forest_encoder(...)`
2. `row_forests(...)`
3. `frontier_builder.build(state)`
4. 计算 STOP head
5. 计算每个 frontier action 的 successor logits
6. 用 `segment_log_softmax` / `segment_logsumexp` 得到状态值和动作概率

对应代码：

- `forward`: [src/weaver/policy.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy.py:67)
- `_expand_logits`: [src/weaver/policy.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy.py:125)
- `_state_values_and_action_log_probs`: [src/weaver/policy.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy.py:186)

### 8.3 为什么 Policy 是第一主瓶颈候选

原因不是 head 很深，而是它的控制流复杂：

1. `forest_encoder` 本身就重。
2. `row_forests` 会重建每个 row 的 rooted forest。
3. frontier action 数不固定。
4. successor action 编码要逐 action 处理父节点、子节点、关系边。
5. 一个 transition batch 里 parent state 要算一次 policy，child state 还要再算一次 policy。

换句话说，这不是常规 Transformer/MLP 那种“把一个大张量塞进去算完”，而是 **图搜索状态 + 结构重建 + 动态候选集评分**。

---

## 9. ForestEncoder 模块逻辑与成本

### 9.1 ForestEncoder 接口

`ForestEncoder` 位于 [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:47)。

接口：

```python
forest_encoder(
    features: FeatureBankLike,
    state: State,
    context: GraphContext,
) -> ForestEncoding
```

还暴露：

- `row_forests(...)`
- `encode_successor_actions(...)`

### 9.2 `forward` 的核心逻辑

`ForestEncoder.forward(...)` 的成本很高，原因是它按 row 重建森林表示：

1. 取每个 row 的 query embedding
2. `row_forests = self.row_forests(...)`
3. 计算 `max_paths`
4. 创建 `path_memory [num_rows, max_paths, hidden_dim]`
5. 双层 Python 循环：
   - 外层遍历 `row`
   - 内层遍历 `forest.active_nodes`
6. 每个节点根据是不是根节点，分别读取 node/rel/query 特征并做投影
7. 最后做 `_pool_rows(...)`

对应代码：

- `forward`: [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:64)

### 9.3 `row_forests` 的重建开销

`row_forests(...)` 会对每个 row 调用 `reconstruct_row_forest(...)`，见：

- `row_forests`: [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:135)
- `reconstruct_row_forest`: [src/weaver/forest.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/forest.py:26)

`reconstruct_row_forest(...)` 本身是 Python BFS 风格：

1. 从 `state.edge_mask[row].nonzero()` 收集选中边
2. 用 `set` 和 `dict` 做 BFS
3. 遍历 outgoing edges
4. 构造 `parent_by_node / parent_edge_by_node / depth_by_node`

这是非常不利于吞吐的，因为：

- 它几乎完全在 Python 控制流里
- 不能有效利用 GPU 大张量并行
- row 数一多就线性放大

### 9.4 `encode_successor_actions` 的重路径

`encode_successor_actions(...)` 在 [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:150)。

这是本仓库最值得怀疑的热点之一。

它的主要成本来源：

1. 一次性取所有候选 action 的 child node / relation 特征
2. 但接着又有两段 Python 循环：
   - 第一段按 `idx` 遍历所有 action，查 `parent_slot`
   - 第二段按 `idx` 再遍历所有 action，做 row cache pooling

关键位置：

- 第一段循环: [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:189)
- 第二段循环: [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:216)

这意味着即使前面部分张量化了，**action 级别的后处理仍然是 Python 主导**。

如果一个状态的 frontier 很大，这里会非常慢。

---

## 10. FeatureEncoder 模块逻辑与成本

### 10.1 接口

`FeatureEncoder` 位于 [src/weaver/nn/feature_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/feature_encoder.py:70)。

它支持两种模式：

- `lazy=False`: 一次性预编码 node / relation / query
- `lazy=True`: 返回 `LazyFeatureBank`

默认配置是：

- `lazy: true`
- `embedding_tables_on_cpu: true`

见 [configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:14)。

### 10.2 懒模式的真实行为

`lazy=True` 时，`forward(batch)` 不会直接生成 dense 特征，而是返回 `LazyFeatureBank`：

- `forward`: [src/weaver/nn/feature_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/feature_encoder.py:118)

后续热路径里一旦调用：

- `node_feature_rows(...)`
- `relation_feature_rows(...)`
- `query_feature_rows(...)`

都会触发按需查询。

### 10.3 为什么 `embedding_tables_on_cpu: true` 很危险

当 `embedding_tables_on_cpu=True` 时：

- `entity_text_embeddings`、`entity_embedding_map`、`relation_semantic_table` 常驻 CPU，见 [src/weaver/nn/feature_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/feature_encoder.py:96)
- `_entity_observations(...)` 会在 CPU 查表后再 `.to(device=self.compute_device)`，见 [src/weaver/nn/feature_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/feature_encoder.py:143)
- `project_relation_ids(...)` 也会先在 relation table 所在设备查表，再搬到 compute device，见 [src/weaver/nn/feature_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/feature_encoder.py:155)

如果 compute device 是 GPU，这意味着训练热路径里会不断发生：

- 小批量索引
- CPU 内存读取
- CPU->GPU 传输
- 再做线性投影

这非常容易导致：

- GPU 等 CPU
- kernel 很碎
- PCIe / NVLink 传输频繁

所以这两项配置组合：

```yaml
lazy: true
embedding_tables_on_cpu: true
```

不是“省显存但还挺快”的轻量配置，而更像是 **用吞吐换显存**。

---

## 11. Reward 与 Loss 模块逻辑

### 11.1 Reward

`EvidenceLogReward` 在 [src/weaver/reward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/reward.py:33)。

训练时先通过 `prepare_context(...)` 预构建：

- `target_mask`
- `target_count_by_graph`
- `anchor_mask`

之后 `forward(...)` 会基于 `state.node_mask` 和 `state.edge_mask` 计算：

- `answer_gain`
- `fail_penalty`
- `edge_penalty`
- `log_reward`

接口：

- `prepare_context`: [src/weaver/reward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/reward.py:60)
- `forward`: [src/weaver/reward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/reward.py:106)

这一部分本身不算最重，但它依赖稠密 `node_mask`、`edge_mask`，因此成本也会随 `R/N/E` 放大。

### 11.2 Loss

`BellmanDecisionLoss` 在 [src/weaver/loss.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/loss.py:46)。

训练目标是：

- `expand_residual = edge_log_flow - child_state_log_flow_target`
- `stop_residual = stop_log_flow - terminal_log_reward`
- 二者平方后求均值

接口：

- `forward`: [src/weaver/loss.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/loss.py:53)

loss 本身不复杂，真正贵的是为了构造这些量，前面需要先算出：

- parent `policy`
- child `policy`
- reward(parent)
- reward(child)
- `num_forest_parents(child_state)`

所以这里属于 **loss 便宜，loss 所需中间量昂贵**。

---

## 12. 训练目标为什么天然比普通监督学习更重

看 `_forward_transitions(...)` 就能明白，见 [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:461)。

对一个 `TransitionBatch`，它至少要做：

1. `parent_out = self.policy(parent_state)`
2. `child_out = self.policy(child_state)`，虽然 `no_grad`，但计算量仍在
3. `_match_transition_actions(...)` 把 rollout 选中的动作回对齐到 parent frontier
4. `num_forest_parents(child_state, ...)`
5. `reward(parent_state)`
6. `reward(child_state)`
7. 再喂给 loss

因此单个 transition 不是“给一个 label 做交叉熵”，而是要同时估：

- 当前状态值
- 后继状态值
- 停止回报
- backward correction

这就是算法层面的刚性成本，不是简单换个优化器就能解决。

---

## 13. Replay / Transition 构造的逻辑与成本

### 13.1 transitions_from_rollouts

`transitions_from_rollouts(...)` 在 [src/weaver/rollout/replay.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/replay.py:94)。

它会：

1. 遍历每个 rollout
2. 从初始状态开始回放
3. 对每个 step：
   - 取 expand rows
   - clone parent
   - clone child
   - `child.apply_edges_`
   - `current.apply_edges_`

这说明 transition 构造并不是“rollout 时顺便记录好的现成张量”，而是 **训练前又根据 rollout trace 重建了一遍 parent/child 状态**。

这部分成本包括：

- Python 循环
- 多次 `clone()`
- 多次 `State.select_rows()`
- 多次 `apply_edges_()`

### 13.2 oracle_prefix_transitions

`oracle_prefix_transitions(...)` 在 [src/weaver/rollout/replay.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/replay.py:141)。

虽然默认 replay schedule 关闭，但代码实现表明一旦打开，会额外带来：

- 按图遍历
- 按 target 遍历
- BFS 风格 shortest path 搜索
- 为路径每一步构造 transition

这部分很明显不是轻量附加项。

### 13.3 dedupe_transitions

`dedupe_transitions(...)` 在 [src/weaver/rollout/replay.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/replay.py:190)。

这里还会把 `edge_mask` 和 graph id 拉到 CPU：

- `edge_masks = ...cpu()`
- `graphs = ...cpu()`
- `action_edge_ids = ...cpu()`

然后用 Python `set` 去重。

这进一步说明 replay 路径对吞吐不友好。

---

## 14. Sampling 模块的隐藏热点

`sample_action(...)` 在 [src/weaver/rollout/sampling.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/sampling.py:41)。

它不是简单 `argmax`，而是要：

1. 找哪些 row 有 frontier
2. 对 active rows 构造 stop + edge 的联合 logits
3. 做 `segment_log_softmax`
4. 用 Gumbel 或 argmax 在每个 row 内采样

这里一个额外的潜在热点是：

```python
active_frontier_mask = tensors.row_ids.unsqueeze(0).eq(active_rows.unsqueeze(1)).any(dim=0)
```

位置在 [src/weaver/rollout/sampling.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/sampling.py:162)。

这个表达式会构造一个二维比较矩阵，复杂度接近：

`O(num_active_rows * num_frontier_actions)`

如果 frontier 较大，就会产生不必要的广播比较成本。

---

## 15. 验证阶段为什么也可能慢

验证逻辑在 [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:257) 和 [src/eval/rollout.py](/mnt/wangjingxiong/EVI-RAG/src/eval/rollout.py:31)。

验证不是简单跑 loss，而是：

1. 再次 rollout
2. 结果送到 CPU 上重建 subgraph
3. 统计 best-of-k recall、compactness 等指标

特别是这段：

```python
SubgraphReconstructor(batch, device=torch.device("cpu")).stack(rollout_samples)
```

位置在 [src/eval/rollout.py](/mnt/wangjingxiong/EVI-RAG/src/eval/rollout.py:50)。

因此如果训练中频繁验证，即使训练 step 本身不算，验证也会插入大量额外时间。

---

## 16. 为什么当前实现会呈现“训练非常慢”的表观现象

结合默认配置，可以把一个 step 近似展开成下面这张逻辑图：

1. 读 1 个图样本
2. rollout 前准备上下文
3. 做 8 个 rollout chunk
4. 每个 chunk：
   - rollout 采样
   - 重建 transitions
   - 重新建立特征访问图
   - parent policy
   - child policy
   - reward
   - backward
5. 所有 chunk 累积完再 step

所以用户感受到的“慢”，本质上不是：

- 模型层太深
- 参数量太大

而更像是：

- 单 step 的算法工作量大
- 单 step 里的 GPU 工作被切得很碎
- 很多工作在 Python 和 CPU 上完成
- 有不少重复构造

---

## 17. 分模块慢点归因

### 17.1 数据读取层

确定性慢点：

- LMDB 逐样本读取
- 样本反序列化
- batch=1 无法摊薄固定开销

结论：

- 是次级瓶颈，不太像首要瓶颈。

### 17.2 rollout 层

确定性慢点：

- 每步多轨迹采样
- 每轨迹多时刻状态更新
- 每时刻都要重跑 policy

结论：

- 是主瓶颈之一。

### 17.3 policy / forest 编码层

确定性慢点：

- 多次重建 row forest
- 多层 Python 循环
- successor action 级别循环
- 动态 frontier 大小

结论：

- 很可能是第一主瓶颈。

### 17.4 feature 访问层

确定性慢点：

- lazy 查表
- embedding table 常驻 CPU
- 小批量多次 `.to(cuda)`

结论：

- 是造成 GPU 利用率低和 step 时延长尾的关键原因之一。

### 17.5 transition / replay 层

确定性慢点：

- rollout 后又重建 parent/child 状态
- clone/select_rows/apply_edges 多次调用

结论：

- 是训练 step 内额外的结构性成本。

### 17.6 loss 层

确定性慢点：

- loss 自身便宜
- 但需要的中间量很贵

结论：

- 不是直接瓶颈，但决定了前面必须做双 policy 计算。

---

## 18. 最关键的三个放大器

如果只挑最关键的三个慢因，我会给出这三个：

### 18.1 `train_chunk_size=1`

位置：[configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:41)

影响：

- 8 个 rollout 被拆成 8 个训练 chunk
- 每个 chunk 单独前向、单独反向
- GPU 计算过于碎片化

这是最直接的吞吐杀手之一。

### 18.2 `lazy: true` + `embedding_tables_on_cpu: true`

位置：[configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:14)

影响：

- 特征不预编码
- 按需访问时反复查表
- CPU->GPU 传输进入热路径

这是最直接的设备带宽杀手之一。

### 18.3 `ForestEncoder` / `reconstruct_row_forest` 的 Python 循环

位置：

- [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:85)
- [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:216)
- [src/weaver/forest.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/forest.py:26)

影响：

- 算法不是张量主导，而是 Python 控制流主导
- rollout row 和 frontier action 一多就线性放大

这是最直接的 CPU 调度与解释器开销来源。

---

## 19. 可以如何理解每个模块的职责

### 19.1 `RetrievalDataModule`

职责：把物化数据接到训练系统，不参与图算法。

接口：

- `setup`
- `train_dataloader`
- `val_dataloader`
- `test_dataloader`

### 19.2 `RetrievalDataset`

职责：按样本 id 从物化存储恢复图样本。

接口：

- `get(idx) -> RetrievalData`

### 19.3 `FeatureEncoder`

职责：把实体文本向量、关系向量、问题向量投影到统一 hidden space。

接口：

- `forward(batch) -> FeatureBankLike`
- `project_entity_ids`
- `project_relation_ids`
- `project_observations`

### 19.4 `RolloutEngine`

职责：在当前 policy 下进行有限预算的图扩展搜索。

接口：

- `prepare_context`
- `sample_rollouts`

### 19.5 `Policy`

职责：对当前状态给出 STOP 分数、各候选边的 EXPAND 分数，以及状态值。

接口：

- `forward(...) -> PolicyOutput`

### 19.6 `ForestEncoder`

职责：把当前已选证据子图编码成 row-level 表示，并为候选 successor 动作构造新状态表示。

接口：

- `forward(...) -> ForestEncoding`
- `row_forests(...)`
- `encode_successor_actions(...)`

### 19.7 `EvidenceLogReward`

职责：基于是否命中目标答案节点与边数惩罚给出确定性对数奖励。

接口：

- `prepare_context`
- `forward`

### 19.8 `BellmanDecisionLoss`

职责：约束 stop / expand 决策与后继状态值、终止奖励一致。

接口：

- `forward(...) -> LossOutput`

### 19.9 `WeaverModule`

职责：把 rollout、policy、reward、loss、优化器更新串起来，形成完整训练 step。

接口：

- `training_step`
- `validation_step`
- `test_step`
- `_forward_transitions`

---

## 20. 如果必须排序，最可能的耗时占比顺序

在不做实测的前提下，我对耗时排序的判断是：

1. `Policy` 内部的 `ForestEncoder + successor action encoding`
2. rollout 中重复调用 policy
3. lazy feature access 带来的 CPU->GPU 取数
4. transition 重建与状态 clone
5. 数据读取与反序列化
6. reward / loss 本体

这个排序是从代码结构推导出来的，和常见训练瓶颈分布是一致的。

---

## 21. 可执行的优化建议

下面按“风险最低”到“结构改动最大”排序。

### 21.1 低风险配置优化

1. 把 `train_chunk_size` 从 `1` 提高到 `train_num_rollout` 或至少更大的值。
2. 评估 `batch_size > 1` 是否可行。
3. 增大 `num_workers`，让数据加载不只 1 个 worker。
4. 减少验证频率，避免 rollout-heavy 验证过密。

预期收益：

- 减少微小 kernel
- 提高 GPU 利用率
- 降低 Python 循环调度频率

### 21.2 中风险配置优化

1. 把 `embedding_tables_on_cpu` 改为 `false`
2. 评估把 `lazy` 改为 `false`

预期收益：

- 避免热路径中 CPU->GPU 小块传输
- 用一次预编码换取后续多次索引更快

风险：

- 显存占用可能显著升高

### 21.3 中高风险代码优化

1. 缓存 `row_forests` 或避免 parent/child 重复完整重建
2. 将 `encode_successor_actions` 的 action 级 Python 循环进一步张量化
3. 避免 `sample_action` 中 `active_rows x frontier_rows` 广播比较
4. 减少 `TransitionBatch` 里 parent/child `State` 的全量 clone

预期收益：

- 显著降低 CPU 解释器开销
- 提升大 frontier 情况下吞吐

### 21.4 高风险结构优化

1. 改写状态表示，避免 `node_mask [R, N]` / `edge_mask [R, E]` 的稠密存储
2. 将 rollout trace 直接保存成可训练 transition，减少二次重建
3. 设计增量式 forest encoding，而不是每次从 `State` 重建

预期收益：

- 这是决定性优化方向

风险：

- 改动大，需要系统性回归验证

---

## 22. 我认为最值得先做的三件事

如果目标是“先把训练速度明显拉起来”，优先级建议如下：

1. 先把 `train_chunk_size` 调大，观察 step time 和 GPU 利用率变化。
2. 再测试 `embedding_tables_on_cpu=false`，确认是不是 CPU->GPU 取数在拖慢热路径。
3. 最后 profile `ForestEncoder.encode_successor_actions` 和 `reconstruct_row_forest`，这两处大概率是代码级最大热点。

这三件事覆盖了：

- 粒度问题
- 设备带宽问题
- Python 控制流问题

---

## 23. 附：关键源码接口索引

### 训练入口

- [src/train.py](/mnt/wangjingxiong/EVI-RAG/src/train.py:56)
- [src/training/factory.py](/mnt/wangjingxiong/EVI-RAG/src/training/factory.py:27)

### 数据

- [src/data/datamodule.py](/mnt/wangjingxiong/EVI-RAG/src/data/datamodule.py:11)
- [src/data/dataset.py](/mnt/wangjingxiong/EVI-RAG/src/data/dataset.py:58)
- [src/data/collate.py](/mnt/wangjingxiong/EVI-RAG/src/data/collate.py:12)

### 训练主循环

- [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:94)
- [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:461)

### rollout

- [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:69)
- [src/weaver/rollout/engine.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/engine.py:42)
- [src/weaver/rollout/sampling.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/sampling.py:41)
- [src/weaver/rollout/replay.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/replay.py:39)

### 状态与图结构

- [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:37)
- [src/weaver/forest.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/forest.py:26)
- [src/weaver/context.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/context.py:8)

### 编码与策略

- [src/weaver/nn/feature_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/feature_encoder.py:70)
- [src/weaver/nn/forest_encoder.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/nn/forest_encoder.py:47)
- [src/weaver/policy.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy.py:37)

### 奖励与损失

- [src/weaver/reward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/reward.py:33)
- [src/weaver/loss.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/loss.py:46)

### 评估

- [src/eval/rollout.py](/mnt/wangjingxiong/EVI-RAG/src/eval/rollout.py:31)

---

## 24. 最终判断

当前训练慢，根因不是“某一层算子没优化”，而是当前实现本身选择了一个 **高结构复杂度、低批处理粒度、CPU/GPU 混合特征访问** 的训练路径。

更准确地说：

- 算法上，它在做状态搜索式训练，不是普通监督学习。
- 实现上，它又把这套搜索拆成了大量微小 chunk，并在热路径里保留了很多 Python 循环和 CPU 查表。

所以训练慢是符合代码结构预期的，不是偶发异常。

如果需要，我下一步可以继续做两件事中的任意一个：

1. 再写一份“面向优化落地”的 Markdown，给出具体改哪些配置、改哪些函数、预期收益和风险。
2. 直接在代码里加 profile / timer / CUDA event，把每个模块的耗时打出来。
