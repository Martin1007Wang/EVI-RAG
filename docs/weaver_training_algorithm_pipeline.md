# Weaver 当前训练算法管线复盘

本文按当前工作区实现复盘 `Weaver` 的训练、推理和评估算法。目标是把代码现在实际做的事情讲清楚，不追溯旧版本设计，也不描述尚未实现的设想。

核心问题有四个：

- 状态空间和动作空间到底是什么
- policy / backward / state flow / reward 怎样参数化
- rollout / replay / SubTB objective 怎样串起来训练
- evaluation 最终怎样从 sampled subgraph 计算检索指标

主要对应代码：

- [src/train.py](/mnt/wangjingxiong/EVI-RAG/src/train.py:1)
- [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:1)
- [src/weaver/context.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/context.py:1)
- [src/weaver/state.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/state.py:1)
- [src/weaver/feature.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/feature.py:1)
- [src/weaver/policy/forward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy/forward.py:1)
- [src/weaver/policy/backward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy/backward.py:1)
- [src/weaver/policy/edge_scorer.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/policy/edge_scorer.py:1)
- [src/weaver/objectives/subtb/batch.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/objectives/subtb/batch.py:1)
- [src/weaver/objectives/subtb/scoring.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/objectives/subtb/scoring.py:1)
- [src/weaver/objectives/subtb/loss.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/objectives/subtb/loss.py:1)
- [src/weaver/reward.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/reward.py:1)
- [src/weaver/rollout/engine.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/engine.py:1)
- [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:1)
- [src/weaver/rollout/replay.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/replay.py:1)
- [src/eval/rollout.py](/mnt/wangjingxiong/EVI-RAG/src/eval/rollout.py:1)

## 1. 当前实现和旧设计的关键差异

当前实现已经不是旧文档里的 Detailed Balance / transition table 版本。需要先把几个容易误解的点钉死：

- 训练目标是 `ForwardLookingSubTBObjective`，不是 Detailed Balance。
- 旧的 `edge_flow_matching`、`transition_batch`、`transition_builder` 路径已经不存在。
- 状态是 canonical edge-set，不是当前节点、路径序列、active node embedding 或 budget-aware state。
- `StateEncoder` 不再 mean-pool active nodes，而是用 `question_h` 对 selected edges 做 cross-attention；空状态有可学习 `empty_state_emb`。
- 前向边打分和后向删边打分都使用 question-conditioned edge scorer，但 forward / backward 参数独立。
- backward 不再是 `-log(1 + |S_z|)`，也不是简单 `-log(edge_count)`；合法 predecessor 取决于删除后是否仍 root-reachable，以及被删边的 `src` 是否在 parent active set 中。
- reward 当前是 forward-looking 形式：terminal reward 用 `log(eps + recall)` 加 compactness cost，dense potential 用 non-anchor active node 到 target 的 bounded proximity 加同样 cost；`fail_cost` 已移除。
- replay 是预处理生成的弱监督 replay bank，不是在线缓存，也不是 runtime 根据当前模型重新搜索 oracle。
- frontier pruning 默认开启，基于原始 BGE question-relation dot score 静态剪枝；训练时会保留 recorded trajectory edges，避免 SubTB scoring 找不到 replay / rollout 中的动作。
- backward policy 默认是 learned question-conditioned scorer；SubTB residual 使用它的 log-prob 数值但 detach，不通过 residual 训练 backward。默认 `backward_aux_weight = 0.0`，所以 backward head 当前不接收训练梯度。

默认模型配置来自 [configs/model/weaver.yaml](/mnt/wangjingxiong/EVI-RAG/configs/model/weaver.yaml:1)：

- `budget: 8`
- `hidden_dim: 512`
- `train_policy_rollouts: 16`
- `eval_rollouts: 32`
- `subtb_lambda: 0.9`
- `terminal_loss_weight: 2.0`
- `replay_loss_weight: 0.25`
- `path_nce_weight: 0.0`

## 2. 端到端入口

训练入口是 [src/train.py](/mnt/wangjingxiong/EVI-RAG/src/train.py:1)。

主流程：

1. Hydra 读取 [configs/train.yaml](/mnt/wangjingxiong/EVI-RAG/configs/train.yaml:1)。
2. `seed_everything()` 固定随机种子。
3. `prepare_training_components(cfg, stage="fit")` 构建 datamodule 并加载 materialized resources。
4. `build_model(cfg, resources)` 实例化 `FeatureEncoder`、`ForwardPolicy`、`EvidenceStateScorer`、`ForwardLookingSubTBObjective`、`RolloutRunner`。
5. `load_pretrained_if_requested()` 可选加载预训练 checkpoint。
6. `trainer.fit(model, datamodule, ckpt_path=cfg.fit_ckpt_path)` 开始 Lightning 训练。
7. 若 `test_after_fit` 为真，则用 best checkpoint 跑 test。

`build_model()` 的关键点见 [src/training/factory.py](/mnt/wangjingxiong/EVI-RAG/src/training/factory.py:1)：

- `FeatureEncoder` 需要 materialization 里加载出的全局语义表：
  - `entity_text_semantic_table`
  - `text_row_by_entity_id`
  - `entity_relation_neighborhood_semantic_table`
  - `relation_neighborhood_row_by_entity_id`
  - `relation_semantic_table`
- policy 通过 `_build_policy(cfg.model.policy)` 实例化。
- `reward_model`、`objective`、`runner` 直接来自 `configs/model/weaver.yaml`。
- `debug_lookup` 若存在会挂到 module 上，仅用于调试，不进入训练算法。

## 3. 离线预处理和 materialization

训练时 dataset 不在线解析原始 KG，也不在线生成 replay。当前 runtime 依赖预处理产物，特别是 `replay_bank_v4` 和边级最短路径标签。如果 materialization provenance 不是 `replay_bank_v4`，`RetrievalDataset` 会直接拒绝加载，见 [src/data/dataset.py](/mnt/wangjingxiong/EVI-RAG/src/data/dataset.py:184)。

### 3.1 原始样本读取

预处理入口是 [src/preprocess.py](/mnt/wangjingxiong/EVI-RAG/src/preprocess.py:1)，默认配置是：

- [configs/preprocess.yaml](/mnt/wangjingxiong/EVI-RAG/configs/preprocess.yaml:1)
- [configs/preprocess/default.yaml](/mnt/wangjingxiong/EVI-RAG/configs/preprocess/default.yaml:1)

`iter_samples()` 从 Hugging Face dataset 读原始行，见 [src/data/preprocess/source.py](/mnt/wangjingxiong/EVI-RAG/src/data/preprocess/source.py:1)。它做的事情很窄：

- 按 dataset config 的 column map 取 `graph`、question entity、answer entity、question id、question text。
- graph 中每条 triple 标准化成 `(head, relation, tail)`。
- malformed triple、空 head、空 tail、空 relation 会被丢弃并计数。
- 输出 `RawSample`，不做图算法、不做 embedding、不建 catalog。

### 3.2 图清洗和监督标签

`prepare_sample()` 负责把 `RawSample` 变成 `PreparedSample`，见 [src/data/preprocess/graph_collect.py](/mnt/wangjingxiong/EVI-RAG/src/data/preprocess/graph_collect.py:1)。

图清洗规则：

- `remove_self_loops: true` 时删除 head == tail 的 self-loop。
- `dedup_edges: true` 时按完整 triple 去重。
- relation-distinct parallel triples 会保留，因为 `(head, relation, tail)` 不同。
- `build_local_graph()` 产生 sample-local `edge_index`。

样本过滤规则：

- graph 为空则丢弃。
- question entities 中没有任何实体落在 graph 中则丢弃。
- 若 split filter 要求 answer in graph，而 answer entities 都不在 graph 中，则丢弃。
- 计算 path labels 后，若要求 reachable answer 而没有 reachable target，则丢弃。

当前默认所有 train / validation / test split 都要求：

- `require_answer_in_graph: true`
- `require_reachable_answer: true`

监督字段：

- `anchor_node_ids`：question entities 在 sample-local graph 中对应的 node id。
- `target_node_ids`：answer entities 在 graph 中对应的 node id。
- `reachable_target_node_ids`：从 anchors 可达的 target nodes。
- `node_target_distance`：每个 node 到 target 的距离标签；不可达通常为负值。
- `edge_on_shortest_path`：每条 edge 是否位于任一 anchor-answer 最短路径集合上。

这些字段后续分别用于：

- anchors：定义状态初始 active set 和 root-reachable。
- reachable targets：训练 reward 和 evaluation 的目标集合。
- `node_target_distance`：保留为路径标签字段，供其它路径相关分析或诊断使用。
- `edge_on_shortest_path`：训练阶段的 frontier InfoNCE 弱监督标签。

### 3.3 replay bank 预处理

`build_replay_bank()` 在预处理阶段生成弱监督轨迹库，见 [src/graph/oracle_replay.py](/mnt/wangjingxiong/EVI-RAG/src/graph/oracle_replay.py:1)。

默认 replay 配置：

- `max_edges: 8`
- `round_variants: 8`
- `trajectories_per_graph: 4`
- `beam_width: 32`
- `path_variants_per_pair: 2`
- `max_expansions_per_state: 32`
- `seed: 42`

构造逻辑：

1. 对每个 anchor-target pair 建 shortest-path DAG。
2. 对每个 pair 枚举最多 `path_variants_per_pair` 条路径变体。
3. beam search 组合多个 pair 的路径，偏好：
   - 覆盖更多 targets
   - 使用更少 edges
   - 较少和已选候选重叠
   - 稳定 hash tie-break
4. 只保留 `len(edges) <= max_edges` 的候选。
5. 用 `_frontier_legal_order()` 把 unordered edge set 转成合法 rollout 顺序。

replay bank 的 tensor 形状：

```text
replay_bank_edge_ids    [round_variants, trajectories_per_graph, max_edges]
replay_bank_edge_count  [round_variants, trajectories_per_graph]
```

其中 `edge_count = -1` 表示该 slot 无有效 replay 轨迹。

注意：

- replay bank 不按 runtime budget 生成多份 oracle。
- runtime budget 只在训练时过滤 `edge_count > budget` 的 replay rows。
- replay edge ids 在 sample 里是 sample-local edge id，collator 会平移成 batch-physical edge id。
- 如果答案本身就是 anchor，replay 可以产生 zero-edge external terminal。

### 3.4 语义表

文本 encoder 在 [src/data/preprocess/text_encode.py](/mnt/wangjingxiong/EVI-RAG/src/data/preprocess/text_encode.py:1)。

默认 encoder：

- `BAAI/bge-large-en-v1.5`
- 固定 revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`
- CLS pooling
- L2 normalize
- float32 CPU tensor 输出

预处理会产出：

- question embeddings：每个 split 单独存储，shape `[num_samples, sem_dim]`。
- entity text semantic table：有文本实体的 BGE embedding。
- relation semantic table：relation text 的 BGE embedding。
- entity relation-neighborhood semantic table：无文本实体的 fallback 伪特征。

relation-neighborhood 逻辑见 [src/data/preprocess/relation_neighborhood.py](/mnt/wangjingxiong/EVI-RAG/src/data/preprocess/relation_neighborhood.py:1)：

```text
mid_sem(entity) = Normalize(sum relation_sem(relation_type))
```

实现细节：

- 只为无文本实体构造 fallback。
- 对每个实体 incident relation type 去重。
- 当前版本故意不区分 incoming / outgoing relation。
- relation embedding 求和后再 L2 normalize。
- 任一无文本实体如果没有出现在 retained graph edge 中，预处理直接报错。

### 3.5 materialization 和 dataset runtime 边界

`Materializer` 负责把 prepared samples 和语义表写入 manifest-addressed materialization，见 [src/data/preprocess/materialize.py](/mnt/wangjingxiong/EVI-RAG/src/data/preprocess/materialize.py:1)。

runtime dataset 只做：

- 从 LMDB 反序列化 sample tensor。
- 读取 split question embedding table。
- 将 sample-local tensors 包成 `RetrievalData`。
- 要求 replay bank 字段必须存在。

runtime dataset 不做：

- 原始 graph 解析。
- 样本过滤。
- path label 重新计算。
- replay bank 重新生成。
- entity/relation embedding 加载。

## 4. Batch 坐标约定

PyG collator 在 [src/data/collate.py](/mnt/wangjingxiong/EVI-RAG/src/data/collate.py:1) 和 [src/data/schema/batch.py](/mnt/wangjingxiong/EVI-RAG/src/data/schema/batch.py:1)。

collate 后的核心坐标约定：

- `edge_index[:, e]` 使用 batch-physical node ids。
- `batch[n]` 是 node id 到 graph id 的映射。
- `anchor_node_ids`、`target_node_ids`、`reachable_target_node_ids` 都被 PyG increment 成 batch-physical node ids。
- `node_entity_catalog_ids` 和 `edge_relation_catalog_ids` 仍是全局 catalog id，不随 batching increment。
- `edge_batch[e]` 由 `batch[edge_index[0, e]]` 推导，主要供 evaluation 使用。
- `ReplayBankBatch.edge_ids` 会把 sample-local replay edge ids 加上 edge offset，变成 batch-physical edge ids。

`GraphContext.from_batch()` 在模型内构造 label-free graph context：

- `edge_index`
- `node_to_graph`
- `edge_to_graph`
- `edge_ptr`
- `anchor_mask`
- `anchor_ptr`
- `anchor_node_ids`
- CSR 风格 directed adjacency index

`TargetContext.from_batch()` 构造训练 reward / evaluation 需要的 label context：

- `target_mask`
- `reachable_target_node_ids`
- `reachable_target_node_ids_ptr`
- `target_count_by_graph`
- `node_target_distance`
- `edge_on_shortest_path`
- `anchor_target_count_by_graph`

`ReplayContext.from_batch()` 只包 replay bank：

- `edge_ids`
- `edge_count`
- `priority`

## 5. 状态空间

对每个样本，已知：

- 查询 `q`
- 有向候选图 `G = (V, E)`
- anchor 集合 `A`
- reachable target answer node 集合 `Y`
- rollout 系统 cap `B`

算法不直接对整张图排序，而是在 cap 内逐步选择边，构造证据子图。

### 5.1 Canonical edge-set state

当前状态 `z` 是 canonical selected-edge set：

```text
z = (graph_id, sorted(S_z), |S_z|)
S_z ⊆ E
```

实现是 `StateBatch`：

- `graph_ids: [S]`
- `edge_ids: [S, B]`，有效 selected edge 排序，padding 为 `-1`
- `edge_count: [S]`

重要语义：

- 状态只记录已经选择了哪些 edge。
- action 到达顺序不进入状态身份。
- `edge_ids` storage width 只是 padding capacity，不进入状态 identity。
- `budget` 不是状态语义，只是 tensor storage width 和 rollout cap。
- `StateBatch.from_selected_edges()` 会排序、检查重复、检查 edge 属于对应 graph、检查 root-reachable。

这就是为什么两条轨迹 `[0, 1]` 和 `[1, 0]` 如果最终 edge set 一样，会在 SubTB prefix 去重时映射到同一个 canonical state。

### 5.2 Active nodes

active nodes 是当前状态下可扩展节点集合：

```text
X_z = A ∪ endpoints(S_z)
```

实现上：

- 初始 active nodes 来自当前 graph 的所有 anchors。
- 每条 selected edge 的 `src` 和 `dst` 都加入 active set。
- 同一个 state 内按 `(row_id, node_id)` 去重。

这意味着状态不是“当前游标节点”。只要某条边曾经被选过，它的两端都会保留在 active set 中，后续可以从这些节点继续扩展。

### 5.3 Frontier

frontier 是从 active nodes 出发、且尚未被选中的出边：

```text
C(z) = { e = (u, v) ∈ E \ S_z : u ∈ X_z }
```

动作空间：

```text
A(z) = {STOP} ∪ C(z)
```

实现细节：

- `frontier_from_graph()` 通过 CSR out adjacency 找 active nodes 的出边。
- 已选边会被过滤掉。
- 如果多个 active node 路径产生同一个 edge，会按 `(row, edge)` 去重。
- frontier 按 key 排序，保证 action lookup 可向量化。

若 frontier 为空，policy action space 仍有 STOP，但该 STOP 是结构性 forced terminal。

### 5.4 Multi-source root-reachable

合法 selected-edge state 必须满足 multi-source root-reachable：

```text
每条 selected edge e = (u, v) 都必须能通过 selected edges
从同一 graph 的某个 anchor 到达它的 src u。
```

实现是 `root_reachable_mask_from_edges()`：

1. 把 selected edges 按 `(state row, src node)` 建索引。
2. 初始 frontier nodes 是 selected edge src 中也是 anchor 的节点。
3. 从这些节点沿 selected edges 迭代传播。
4. 所有 selected edges 都被传播到才算 root-reachable。

多 anchor 的含义：

- 一个合法终止状态可以是多个 anchor 分别长出的 forest。
- 不要求所有 anchors 连成一个 component。
- 也不要求所有 selected edges 属于同一条 path。

这个约束直接影响 backward action space：不是所有删边操作都会得到合法 parent。

### 5.5 状态空间的图结构

在集合包含关系下，合法 states 形成有限偏序结构。一次 forward expand 等价于：

```text
z -> z + e
e ∈ C(z)
```

它是 Hasse DAG 上的一条覆盖边。一般不能称为 lattice，因为 multi-source root-reachable selected-edge sets 不保证对交集封闭。

## 6. 特征编码

`FeatureEncoder` 输入 `RetrievalBatch`，输出 `FeaturePack`：

```text
question_h  [G, H]
entity_h    [N, H]
edge_h      [E, H]
relation_h  [E, H]
frontier_prune_score [E]
```

其中 `G` 是 batch 内 graph 数，`N` 是 batch-physical node 数，`E` 是 batch-physical edge 数，`H` 是 `hidden_dim`。

### 6.1 Question / entity / relation 投影

三类原始语义向量都来自 BGE 且已 L2 normalize。模型内投影为：

```text
question_h = LN(W_q question_sem)
entity_h   = LN(W_ent entity_sem)
relation_h = LN(W_rel relation_sem)
```

实现细节：

- 三个 `Linear` 都是 `bias=False`。
- 三个 projection 参数独立。
- 输出用 `LayerNorm`，不再强制 L2 normalize。
- `FeatureEncoder` 会检查输入和输出 finite。

### 6.2 Entity semantic fallback

实体语义查表优先级：

1. 若 `text_row_by_entity_id[id] >= 0`，使用 entity text semantic table。
2. 否则若 `relation_neighborhood_row_by_entity_id[id] >= 0`，使用 relation-neighborhood fallback。
3. 两者都没有则 `ValueError`。

文本实体永远优先于 relation-neighborhood fallback。

### 6.3 Relation_h 和 edge_h 的区别

`relation_h` 是纯 relation semantic projection，按边展开：

- 先对 `edge_relation_catalog_ids` 去重。
- 对 unique relation ids 投影。
- 再按 inverse index 展开回 `[E, H]`。

因此同一 relation type 的不同 edges 共享同一个 `relation_h`。

`edge_h` 是结构化三元组表示：

```text
edge_h(e) = LN(W_e [src_h(e) || relation_h(e) || dst_h(e)])
```

其中：

- `src_h` 和 `dst_h` 来自 `entity_h`。
- `relation_h` 是上面的纯 relation 表示。
- `EdgeEncoder` 只有一次 `Linear(3H -> H, bias=False) + LayerNorm`。

`relation_h != edge_h`：

- `relation_h` 用于构造三元组级 `edge_h`；frontier pruning 的 question-relation 对齐使用原始 `relation_semantic_table`，不是投影后的 `relation_h`。
- `edge_h` 包含 src / relation / dst 三路融合，用于状态条件的边际贡献估计。

## 7. 前向 policy

`ForwardPolicy` 由三部分组成：

- `StateEncoder`
- `FlowEstimator`
- `StateFlowHead`

### 7.1 Policy cache

每个 batch 内 graph 和 edge 特征不随 rollout step 改变，因此 policy 先构造 `PolicyInput`：

```text
question_h_by_graph = features.question_h.float()
edge_h              = features.edge_h.float()
frontier_prune_score = features.frontier_prune_score.float()
align_score         = scorer(question_h_by_edge, edge_h)
```

`align_score` 是当前 policy 参数下的 question-edge alignment，若传入 `graph_context` 会预先算好并缓存。rollout 可以复用 no-grad cache；SubTB scoring 会从带梯度的 `FeaturePack` 重新构造 fresh `PolicyInput`，避免复用 rollout cache 导致梯度断开。

### 7.2 StateEncoder

`StateEncoder` 把 `(question, selected edges)` 编成 `state_h(z)`。

非空状态：

```text
attn_out(z) = MultiHeadAttention(
    query = question_h(z),
    key   = selected_edge_h(z),
    value = selected_edge_h(z)
)

state_h(z) = LN(W_s [question_h(z) || attn_out(z)])
```

空状态：

```text
attn_out(z_0) = empty_state_emb
state_h(z_0) = LN(W_s [question_h(z_0) || empty_state_emb])
```

关键点：

- query 是 question embedding。
- selected edges 是 key/value。
- padding 通过 `key_padding_mask` 屏蔽。
- 空状态不跑 attention，直接用可学习 `empty_state_emb`。
- `empty_state_emb` 能从初始状态的 flow / stop / action loss 收到梯度。

实现里 `ForwardPolicy._build_state_h_batched()` 会把变长 selected edges pad 成 `[S, L_max, H]`，一次性批量送入 `StateEncoder.forward_batched()`。

### 7.3 Question-conditioned edge scorer

forward 和 backward 都使用 `QuestionConditionedEdgeScorer` 这一结构，但各自有独立 module 参数。

对任意候选边 `e`：

```text
phi_align(e)
  = (question_h · edge_h(e)) / sqrt(H)

phi_state(z, e)
  = MLP([state_h(z)
         || edge_h(e)
         || state_h(z) ⊙ edge_h(e)])

rank_logit(z, e) = phi_align(e) + phi_state(z, e)
```

两个路径的职责：

- `question_h · edge_h`：问题和完整三元组 edge 表示的直接对齐。
- `state_h ⊙ edge_h`：当前 selected subgraph 和候选边的兼容性。
- `phi_state` 当前不直接使用 `question_h ⊙ edge_h`；问题影响通过 `state_h` 间接传递。

frontier pruning 使用单独的静态分数：

```text
frontier_prune_score(e)
  = question_emb(g(e)) · relation_semantic_table(rel(e))
```

它工作在原始 L2-normalized 语义空间上，不依赖可训练投影层。

### 7.4 STOP / edge flat energy

STOP 只依赖 state：

```text
stop_logit(z) = StopHead(state_h(z))
```

每条 edge action 直接使用 flat energy：

```text
edge_logit(z, e) = align_scale * a(q, e) + c_theta(z, e)
```

其中 `v_theta(z, e)` 由两路 edge scorer 相加：

```text
v_theta(z, e) = align_scale * φ_align(q, edge_h(e)) + φ_state(state_h(z), edge_h(e))
```

因此：

- STOP 和每条 frontier edge 在同一个 action softmax 里竞争。
- `CONTINUE` 不是显式采样 token；它的概率质量是所有 edge action 概率之和。
- STOP-vs-CONTINUE 由 `stop_logit(z)` 和 `logsumexp_{e ∈ C(z)} edge_logit(z,e)` 共同决定。
- edge logits 的整体平移会改变 STOP-vs-CONTINUE，这是 flat energy 的预期行为。
- 当前 STOP 不显式看 frontier summary，不拼接 question，因为 question 已经进入 `state_h`。
- 如果 frontier 为空，`FlowEstimator` 返回空 edge logits 和全零 stop logits；该 state 的唯一动作是 STOP。
- 如果 root state 有非空 frontier，STOP logit 会被置为 `-1e9`，防止空证据子图立即停止。
- `frontier_size_correction` 默认 `0.0`；若设为正值，会从 edge logits 中减去 `frontier_size_correction * log |C(z)|`。

### 7.5 State flow

训练目标中的 state flow：

```text
log F_base(z) = StateFlowHead(state_h(z))
```

`log F_base(z)` 只在 objective 中使用。rollout / validation / test / predict 的动作选择不使用 state flow。SubTB scoring 阶段会再加上 reward 的 dense potential：

```text
log F(z) = log F_base(z) + state_potential(z)
```

### 7.6 Forward distribution

对每个 parent state：

```text
P_F(. | z) = softmax({stop_logit(z)} ∪ {edge_logit(z, e) : e ∈ C(z)})
```

对应的隐式 continue mass 是：

```text
P_F(CONTINUE | z) =
    Σ_e exp(edge_logit(z, e))
  / (exp(stop_logit(z)) + Σ_e exp(edge_logit(z, e)))
```

`PolicyOutput` 把所有 action logits flatten 成：

- `action_logits: [S + F]`
- `action_row_ids: [S + F]`
- `action_edge_ids: [S + F]`，STOP 用 `-1`

其中每个 state 至少有一个 STOP action，所以 `segment_logsumexp` 总有值。

如果 frontier 为空，则该 state 的 action space 只有 STOP，STOP log-prob 为 `0`。

## 8. Backward policy

Backward policy 只参与训练 objective，不参与 rollout 或 evaluation。

### 8.1 Removable edges

对 child state `z'`，backward action 是删除一个 selected edge。合法 removable edge 定义：

```text
e ∈ removable(z')
iff
  parent z = z' \ {e} 仍 multi-source root-reachable
  and src(e) ∈ X_z
```

第二个条件使用 `src(e)`，不是 `dst(e)`。因此叶子边可以被删除，即使删除后 `dst(e)` 不再 active。

实现步骤：

1. 枚举 child state 中每条 selected edge 作为删除候选。
2. 构造 parent selected-edge tensor。
3. 调用 `root_reachable_mask_from_edges()` 检查 parent root-reachable。
4. 计算 parent active nodes。
5. 检查 removed edge 的 `src` 是否在 parent active nodes 内。
6. 输出 `FrontierEncoding(row_ids, edge_ids, graph_ids)` 形式的 removable action space。

`legal_predecessor_count()` 和 `uniform_backward_log_prob()` 仍保留给 uniform backward 消融，但默认训练配置不走 uniform backward。

### 8.2 Learned backward distribution

默认 `BackwardPolicy` 的参数化形式是：

```text
P_B(z | z')
  = softmax_e {
      (question_h · edge_h(e)) / sqrt(H)
      + MLP_B([state_h(z')
               || edge_h(e)
               || state_h(z') ⊙ edge_h(e)])
    }
    where e ∈ removable(z') and z = z' \ {e}
```

实现细节：

- child state 的 `state_h(z')` 直接复用 forward scoring 时保留的 `state_h`。
- question 按 removable action 的 `graph_ids` 展开。
- softmax 在每个 child state 的 removable edges 内分段归一化。
- `BackwardPolicy` 和 forward `FlowEstimator` 不共享 scorer 参数。
- `score_subtb_batch()` 会对每个 trajectory step 的 actual parent edge gather learned backward log-prob。

当前默认目标的一个重要梯度边界：

- `backward_step_log_prob` 写入 SubTB residual 后会被 detach。
- `backward_aux_weight = 0.0`，因此 auxiliary `-log P_B` 也不参与 loss。
- 所以 learned backward 当前影响 residual 数值和诊断指标，但默认不会被训练更新。
- 若把 `backward_aux_weight` 设为正数，auxiliary loss 会只在 on-policy steps 上训练 backward head。

### 8.3 STOP 的 backward 概率

STOP 没有可学习 backward head。

`BackwardPolicyOutput.gather_log_prob()` 对 STOP edge id `-1` 返回 `0`，含义是：

```text
log P_B(pre-stop state | terminal-after-stop) = log 1 = 0
```

SubTB 里 backward prefix 只累积边删除动作；terminal STOP 只作为 forward terminal action 进入 terminal residual。

## 9. Rollout 和 trajectory record

rollout 代码在：

- [src/weaver/rollout/engine.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/engine.py:1)
- [src/weaver/rollout/runner.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/runner.py:1)
- [src/weaver/rollout/trajectory.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/rollout/trajectory.py:1)

### 9.1 TrajectoryBatch 字段

`TrajectoryBatch` 是 completed trajectory record：

```text
graph_ids    [T]
edge_ids     [T, B]   padding = -1
edge_logp    [T, B]   padding = 0.0
edge_count   [T]
stop_reason  [T]
stop_logp    [T]
source       [T]      False = policy, True = replay
```

terminal kind：

- `POLICY_STOP = 0`：policy 显式采样 STOP。
- `NO_FRONTIER = 1`：frontier 为空，只能 STOP。
- `BUDGET_TRUNCATED = 2`：硬 edge budget 截断。
- `EXTERNAL_TERMINAL = 3`：replay 外部终止。

source：

- `SRC_POLICY = 0`
- `SRC_REPLAY = 1`

`has_trainable_stop` 当前只保留作 endpoint provenance 兼容/诊断字段：

```text
POLICY_STOP or BUDGET_TRUNCATED or EXTERNAL_TERMINAL
```

训练 objective 不再用该字段决定 terminal equation。STOP 训练由 state legality 决定。

`is_forced_terminal`：

```text
NO_FRONTIER or BUDGET_TRUNCATED or EXTERNAL_TERMINAL
```

### 9.2 Policy rollout

`RolloutEngine.sample()` 从空状态开始：

```text
state = StateBatch.initial(graph_ids, budget)
for step in range(budget + 1):
    active rows = rows not done
    if edge_count >= budget:
        mark BUDGET_TRUNCATED
        continue
    policy_out = policy(decision_state)
    sampled action ~ P_F(. | state)
    if action == STOP:
        if frontier empty: NO_FRONTIER
        else: POLICY_STOP
        record stop_logp
        done
    else:
        record edge id and edge logp
        state.advance(edge)
unfinished rows -> BUDGET_TRUNCATED
```

实现细节：

- rollout 在 `torch.no_grad()` 下执行。
- `trusted=True` advance 跳过重复 frontier validation，因为 action 来自 policy output。
- 如果 policy 在 no-frontier row 采样 expand，会报 RuntimeError。
- 到达 budget truncation 时 rollout 会在该 terminal state 上重算当前 policy 的 STOP log-prob。训练时该 terminal state 仍作为 reward boundary，并将该 STOP log-prob 计入 terminal residual，给 STOP head 梯度。

### 9.3 Train rollout batch

`RolloutRunner.train_rollouts()` 会拼接两类 trajectories：

- policy trajectories：由当前 forward policy 采样。
- replay trajectories：从 `ReplaySource` 取预处理 replay bank。

若二者合计为 0，直接报错，因为 objective 没有训练项。

train rollout metrics 包含：

- policy / replay trajectory 数量。
- replay raw / kept count。
- 当前 replay fraction。
- replay priority mean。
- replay edge_count mean。
- replay coverage recall。
- replay unique edge set rate。
- replay target diversity。
- replay anchor pair diversity。
- policy terminal system-cap / model-stop / structural-stop rate。

### 9.4 ReplaySource runtime 行为

`ReplaySource` 从 `ReplayContext` 选择当前 round 的 replay bank slice：

```text
variant = replay_round % replay_context.edge_ids.size(1)
edge_ids   = replay_context.edge_ids[:, variant, :, :]
edge_count = replay_context.edge_count[:, variant, :]
priority   = replay_context.priority[:, variant, :]
```

然后：

- 保留 `0 <= edge_count <= runtime budget` 的 rows。
- replay fraction 初始为 `1.0`。
- 每次 validation 结束时，如果能读到 `ReplaySource.metric_name` 指标，则更新当前 best metric。
- 当 best metric 低于 `threshold` 时，fraction 保持 `1.0`。
- 当 best metric 达到或超过 `threshold` 后，按下式缩小：

```text
fraction = clamp((1 - best_metric) / max(1 - threshold, 1e-6), 0, 1)
```

- 对每个 graph 独立按当前 fraction 保留 replay slots；`0 < fraction < 1` 时至少保留 1 条有效 slot。
- 保留后按 `priority` 降序稳定排序。
- 输出 `TrajectoryBatch`：
  - `stop_reason = EXTERNAL_TERMINAL`
  - `source = SRC_REPLAY`
  - `edge_logp = 0`
  - `stop_logp = 0`

replay rows 没有历史 policy logits，SubTB scoring 会用当前 policy 参数重新计算这些边动作的 log-prob，不做 importance correction。

### 9.5 Eval rollout 和 diversity penalty

`RolloutRunner.eval_rollouts()` 默认只采样 forward policy，不使用 replay。

如果 `diversity_edge_penalty > 0`：

1. 每轮对每个 graph 采样 1 条 trajectory。
2. 记录已经被采样过的 edge 使用次数。
3. 下一轮对 edge action logit 加 bias：

```text
edge_logit_bias[e] = - diversity_edge_penalty * edge_use_count[e]
```

4. 多轮结果按 graph id 稳定排序后合并。

默认 `diversity_edge_penalty = 0.0`。

## 10. 训练 step 数据流

`WeaverModule.training_step()` 是当前训练主链路，见 [src/weaver/module.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/module.py:61)。

一次 step：

1. `_build_step_contexts(batch)` 构造 `GraphContext`、`TargetContext`、`ReplayContext`。
2. `with torch.no_grad()`：
   - `FeatureEncoder(batch) -> FeaturePack`
   - `policy.build_policy_input(features, graph_context) -> PolicyInput`
   - `runner.train_rollouts(...)` 采样 policy trajectories 并拼接 replay trajectories。
3. 重新调用 `FeatureEncoder(batch)` 和 `policy.build_policy_input(...)`，为 objective scoring 建立带梯度的新计算图。
4. `_build_objective_inputs(...)`：
   - `prepare_subtb_batch(...)`
   - `policy.prepare_action_space(...)`
   - `reward_model(...)`
   - `score_subtb_batch(...)`
5. `objective(batch, scores, reward, global_step)`
6. log train metrics
7. return loss

关键梯度边界：

- rollout 采样不回传梯度。
- replay 采样不回传梯度。
- objective 阶段会对 unique states 重新跑 policy，因此 feature encoder、forward policy、state flow 可获得梯度。
- backward log-prob 来自 learned backward head，但写入 SubTB residual 前会 detach；默认 `backward_aux_weight = 0.0`，所以 backward head 当前不获得训练梯度。
- reward model 当前 `forward()` 是 `@torch.no_grad()`，reward 不学习参数。

## 11. SubTB batch 构造

`prepare_subtb_batch()` 把 completed trajectories 转成 SubTB 训练表。

### 11.1 Prefix states

对每条 trajectory，枚举所有 prefix：

```text
prefix step k contains first k edges
k = 0..edge_count
```

然后：

- padding 位置置 `-1`。
- 对 prefix edge ids 排序，变成 canonical selected-edge state。
- state key 是：

```text
[graph_id, edge_count, sorted_edge_ids...]
```

- 对所有 valid prefix states 做 `torch.unique(dim=0)`。
- 用 `prefix_state_ids[trajectory_id, prefix_step]` 记录每个 prefix 映射到哪个 unique state。

输出的 `states` 是 `StateBatch.from_selected_edges()`，所以会再次检查 root-reachable。

### 11.2 Edge steps

`valid_steps = trajectories.valid_edge_mask()`。

对每个真实 edge step `t`：

- parent state 是 prefix `t`
- child state 是 prefix `t + 1`
- action edge 是 `trajectories.edge_ids[traj, t]`

这些被展开成：

- `step_traj_ids`
- `step_ids`
- `step_parent_state_ids`
- `step_edge_ids`

并按 `step_parent_state_ids` 排序，便于后续 gather。

### 11.3 Terminal metadata

每条 trajectory 的 terminal prefix：

```text
terminal_step_by_traj = edge_count
terminal_state_ids = prefix_state_ids[traj, edge_count]
terminal_kind_by_traj = stop_reason
```

endpoint metadata：

```text
terminal_state_ids = prefix_state_ids[traj, edge_count]
terminal_trainable_stop_mask = edge_count(terminal_state) > 0
```

含义：

- `terminal_trainable_stop_mask` 是 endpoint 兼容/诊断字段，不决定 loss 公式。
- `NO_FRONTIER`、`BUDGET_TRUNCATED`、`EXTERNAL_TERMINAL` 只记录轨迹来源，不改变 STOP terminal boundary。
- terminal reward terms 对 policy/replay rows 的所有非空 prefix endpoint 构建。

### 11.4 Transition terms

SubTB transition terms 是单张 `transition_terms` 表。row mask 是：

```text
trajectories.is_policy | trajectories.is_replay
```

对于轨迹长度 `T`，transition term 枚举：

```text
0 <= start < end <= T
```

也就是说 transition terms 只覆盖边动作之间的子轨迹，不包含 terminal STOP / reward 边界。

`lambda_exponent = end - start - 1`。

### 11.5 Terminal terms

terminal terms 对 policy/replay rows 的所有非空 prefix endpoint 构建。对轨迹长度 `T`，枚举：

```text
0 <= start <= end <= T
end > 0
```

`lambda_exponent = max(end - start, 1) - 1`。

因此：

- zero-edge trajectories 不产生 terminal term，因为空 evidence subgraph 不是合法 STOP reward target。
- 任意非空 prefix state 只要 `EvidenceStateScorer.terminal_valid_mask=True`，都会训练 STOP boundary。
- replay prefix 使用当前 policy 的 edge log-prob 和 STOP log-prob 重新打分，不使用历史 policy logits。

## 12. SubTB scoring

`score_subtb_batch()` 会对 unique states 一次性重算当前 policy 下的所有分数。

### 12.1 Forward scores

先准备 action space：

```text
action_space = policy.prepare_action_space(states)
```

再跑：

```text
output = policy(
    state=states,
    action_space=action_space,
    compute_log_flow=True
)
```

得到：

- `log_flow[state_id] = log_flow_base[state_id] + state_potential[state_id]`
- `state_potential[state_id]`
- `stop_log_prob_by_state[state_id]`
- 所有 frontier edge action log-prob

对 trajectory 记录的每个 edge step，重新 gather：

```text
step_log_prob[traj, step] =
    log P_F(edge_id | parent_state)
```

然后构建 prefix cumulative sum：

```text
forward_prefix_by_traj[:, 0] = 0
forward_prefix_by_traj[:, k] = sum_{t < k} step_log_prob[:, t]
```

### 12.2 Backward scores

如果 batch 中存在 edge steps，则：

1. 为所有 non-root child states 枚举合法 removable predecessors。
2. 用 learned `BackwardPolicy` 对 removable edges 做分段 softmax。
3. 对 trajectory 实际走过的 child state gather：

```text
backward_step_logp[traj, step] =
    log P_B(parent_state | child_state)
```

然后构建：

```text
backward_prefix_by_traj[:, 0] = 0
backward_prefix_by_traj[:, k] = sum_{t < k} backward_step_logp[:, t]
```

实现上 `backward_step_logp` 会 detach 后进入 SubTB residual，因此 residual 不反向训练 backward head。`backward_aux_logprob` 仍保留；只有 `backward_aux_weight > 0` 时才会训练 backward head。

### 12.3 STOP scores

每个 unique state 都有当前 policy 下的 STOP log-prob：

```text
stop_log_prob_by_state[z] = log P_F(STOP | z)
```

terminal residual 总是使用 end state 的 `stop_log_prob_by_state[end_state_id]`，再由 `EvidenceStateScorer.terminal_valid_mask` 过滤非法 STOP reward target。

## 13. Terminal reward

`EvidenceStateScorer` 当前不是可学习模型，而是固定 reward function。

对任一 state `z`：

```text
answer_count(z)    = |X_z ∩ (Y_reachable \ anchors)|
candidate_count(z) = |X_z|
target_count(z)    = |Y_reachable \ anchors|
recall(z)          = answer_count(z) / max(target_count(z), 1)
precision(z)       = answer_count(z) / max(edge_count(z), 1)
success(z)         = 1[answer_count(z) > 0 and target_count(z) > 0]
terminal_valid(z)  = 1[target_count(z) > 0 and edge_count(z) > 0]
edge_count(z)      = |S_z|
```

terminal reward 的 beta-free utility：

```text
compactness(z) = edge_cost_lambda * edge_count(z) / budget
U_rec(z)       = log(reward_epsilon + recall(z)) - log(1 + reward_epsilon)
U_R(z)         = U_rec(z) - compactness(z)
raw_log_R(z)   = reward_beta * U_R(z)
```

若 `center_log_reward = true`，会在同一 graph 的 terminal-valid states 内做 max-centering：

```text
log R(z) = raw_log_R(z) - max_{z' in same graph, terminal_valid(z')} raw_log_R(z')
```

否则 `log R(z) = raw_log_R(z)`。invalid terminal rows 保持有限的 0 值，并由 `terminal_valid_mask` 在 terminal residual 中过滤。

dense forward-looking potential 使用 bounded target proximity：

```text
Prox(z) =
  max_{x in X_z \ anchors}
    clamp(1 - d(x, Y) / (d_max(g) + 1), 0, 1)

U_Psi(z) = Prox(z) - compactness(z)
Psi(z)   = reward_beta * U_Psi(z)
```

实现把 shaped flow 写成：

```text
log F(z) = log F_base(z) + Psi(z)
remaining_log_reward(z) = log R(z) - Psi(z)
```

但当前 terminal residual 直接使用 `log R(z)`，`remaining_log_reward` 主要作为诊断字段。root / empty / invalid-target states 的 `Psi(z)` 为 0。

默认配置：

- `reward_beta = 3.0`
- `edge_cost_lambda = 0.5`
- `reward_epsilon = 1.0e-4`
- `center_log_reward = true`

reward 指标：

- `reward/beta`
- `reward/edge_cost_lambda`
- `reward/epsilon`
- `reward/center_log_reward`
- `reward/log_reward_mean`
- `reward/log_reward_std`
- `reward/log_reward_min`
- `reward/log_reward_max`
- `reward/log_reward_uncentered_mean`
- `reward/reward_shift_mean`
- `reward/potential_mean`
- `reward/potential_std`
- `reward/proximity_potential_mean`
- `reward/residual_mean`
- `reward/residual_std`
- `reward/recall_mean`
- `reward/terminal_recall_mean`
- `reward/precision_mean`
- `reward/terminal_precision_mean`
- `reward/terminal_quality_mean`
- `reward/edge_count_mean`
- `reward/terminal_edge_count_mean`
- `reward/hit_rate`
- `reward/valid_rate`
- `reward/terminal_valid_rate`
- `reward/target_hit_per_edge`
- `reward/terminal_target_hit_per_edge`

## 14. SubTrajectory Balance objective

`ForwardLookingSubTBObjective` 在 [src/weaver/objectives/subtb/loss.py](/mnt/wangjingxiong/EVI-RAG/src/weaver/objectives/subtb/loss.py:1)。

总损失由 on-policy FL-SubTB、replay FL-SubTB，以及可选辅助项组成：

```text
onpolicy_base_loss =
    (
        onpolicy_transition_loss
      + terminal_loss_weight * onpolicy_terminal_loss
    )
  / max(
        onpolicy_transition_weight
      + terminal_loss_weight * onpolicy_terminal_weight,
        1,
    )

replay_base_loss =
    (
        replay_transition_loss
      + terminal_loss_weight * replay_terminal_loss
    )
  / max(
        replay_transition_weight
      + terminal_loss_weight * replay_terminal_weight,
        1,
    )

base_loss = onpolicy_base_loss + replay_loss_weight * replay_base_loss

loss =
    base_loss
  + backward_aux_weight * backward_aux_loss
  + path_nce_weight * path_nce_loss
```

### 14.1 Transition residual

对一条轨迹上的连续子轨迹：

```text
s_i -> s_{i+1} -> ... -> s_j
```

transition residual：

```text
δ_transition(i, j) =
    log F(s_i)
  + log P_F(a_i, ..., a_{j-1})
  - log P_B(s_i, ..., s_{j-1} | s_{i+1}, ..., s_j)
  - log F(s_j)
```

用 prefix sum 写成实现形式：

```text
δ_transition(i, j) =
    log_flow[start_state]
  + (forward_prefix[j] - forward_prefix[i])
  - (backward_prefix[j] - backward_prefix[i])
  - log_flow[end_state]
```

约束成立时：

```text
F(s_i) * Π P_F = F(s_j) * Π P_B
```

policy trajectories 和 replay trajectories 都会产生 transition residual；实现会分别构造 on-policy / replay term tables 并分别归一，再用 `replay_loss_weight` 把 replay base loss 加到总 loss。

### 14.2 Terminal residual

对 start prefix 到 terminal state 的 suffix：

```text
s_i -> ... -> s_T -> STOP
```

terminal residual：

```text
δ_terminal(i) =
    log F(s_i)
  + log P_F(a_i, ..., a_{j-1})
  + log P_F(STOP | s_j)
  - log P_B(s_i, ..., s_{j-1} | s_{i+1}, ..., s_j)
  - log R(s_j)
```

```text
0 <= i <= j <= T
j > 0
```

terminality 是 state-action pair `(z, STOP)` 的属性，不由 trajectory provenance 决定。同一个 canonical state 不会因为这次来自 policy STOP、budget、no-frontier 或 replay endpoint 而切换成 `log F(z)=log R(z)`。

### 14.3 λ 加权

transition / terminal residual 先用 span 做指数加权，再进入平方 penalty：

```text
weight = subtb_lambda ^ lambda_exponent
phi(delta) = delta^2
```

transition term：

```text
lambda_exponent = end - start - 1
```

terminal term：

```text
lambda_exponent = max(end - start, 1) - 1
```

默认 `subtb_lambda = 0.9`，长 span 会按指数略微降权。

transition / terminal 两类 residual 分别按 `weight * phi(residual)` 累加。`terminal_loss_weight` 只在最终聚合时放大 terminal residual 的总 loss 和总 weight。

onpolicy_base_loss 和 replay_base_loss 分别归一；总 base loss 再加权合并：

```text
base_loss = onpolicy_base_loss + replay_loss_weight * replay_base_loss
```

当前实现还保留两个可选辅助项：

- `path_nce_loss`：预处理标记每条边是否位于任一 anchor-answer 最短路径集合上，训练时对每个 frontier row 的多正样本集合做 `logsumexp(all) - logsumexp(pos)`。该项只监督 frontier 内边排序；已经命中答案的 state 不再使用该辅助监督。默认 `path_nce_weight = 0.0`，因此不影响训练。
- `backward_aux_loss`：on-policy steps 上的 `-log P_B(parent | child)`。默认 `backward_aux_weight = 0.0`，因此不训练 backward head。

replay 轨迹仍通过 SubTB transition terms 和 terminal reward terms 参与训练，且 `EXTERNAL_TERMINAL` 会用当前 policy 的 terminal STOP log-prob 训练 replay 终点停止。

## 15. 训练日志

`WeaverModule._log_train()` 记录：

- `train/loss`：step + epoch，prog bar。
- residual mean 指标：step-level logging。
- 其他 objective metrics：epoch-level logging。
- rollout metrics：epoch-level logging，前缀 `train/rollout/`。

重要 objective 指标：

- `objective/loss`
- `objective/base_loss`
- `objective/onpolicy_base_loss`
- `objective/replay_base_loss`
- `objective/state_count`
- `objective/trajectory_count`
- `objective/transition_term_count`
- `objective/terminal_term_count`
- `objective/frontier_size_total`
- `objective/frontier_size_max`
- `objective/frontier_size_mean`
- `objective/log_flow_mean`
- `objective/stop_log_prob_all_mean`
- `objective/terminal_stop_log_prob_mean`
- `objective/forward_log_prob_mean`
- `objective/backward_log_prob_abs_mean`
- `objective/subtb_transition_abs_residual_mean`
- `objective/subtb_transition_abs_residual_p95`
- `objective/subtb_transition_abs_residual_max`
- `objective/subtb_terminal_abs_residual_mean`
- `objective/subtb_terminal_abs_residual_p95`
- `objective/subtb_terminal_abs_residual_max`
- `objective/terminal_loss_weight`
- `objective/replay_loss_weight`
- `objective/backward_aux_loss`
- `objective/backward_aux_weight`
- `objective/path_nce_loss`
- `objective/path_nce_weight`
- `objective/terminal_stop_trainable_count`
- `objective/terminal_budget_truncated_count`
- `objective/terminal_external_count`

## 16. Validation / test / predict

validation 和 test 都调用 `_eval_step()`。

流程：

1. `_build_policy_inputs(batch)`：
   - `GraphContext`
   - `FeaturePack`
   - `PolicyInput`
2. `runner.eval_rollouts()` 采样 forward policy。
3. `evaluate_rollout_samples()` 计算 retrieval metrics。
4. 如果 `diversity_edge_penalty > 0`，额外采样 diverse rollouts 并加 `diverse_` 前缀指标。
5. `_log_eval()` 记录 split metrics。

predict 只返回 `TrajectoryBatch`，不算 metrics。

验证、测试、预测都不使用：

- replay source
- SubTB objective
- reward
- backward probability
- state flow

这些阶段的行为完全由 forward policy 决定。

## 17. Evaluation 指标

evaluation 代码在 [src/eval/rollout.py](/mnt/wangjingxiong/EVI-RAG/src/eval/rollout.py:1)。

### 17.1 Terminal state 恢复

评估先把 completed trajectories 转成 terminal `StateBatch`：

```text
terminal_state = StateBatch.from_selected_edges(
    graph_ids = trajectories.graph_ids,
    edge_ids = trajectories.edge_ids,
    edge_count = trajectories.edge_count,
    budget = trajectories.budget
)
```

它不逐步 replay trajectory，只把 terminal canonical edge set 包起来并校验。

### 17.2 Dense masks

`stacked_terminal_masks()` 构造：

```text
node_masks [K, num_nodes]
edge_masks [K, num_edges]
```

其中 `K` 是每个 graph 最大 sampled trajectory 数。

node mask 来自：

```text
state.covered_node_pairs(context)
```

也就是 active nodes：

```text
anchors ∪ selected edge endpoints
```

edge mask 来自 selected edges。

### 17.3 Retrieval 指标

`retrieval_from_masks()`：

- 可选排除 anchors：`exclude_anchors_from_retrieved`
- 可选使用 reachable targets：`use_reachable_targets`
- 计算 per graph precision / recall / F1
- 若 `exclude_anchors_from_retrieved: true`，则 retrieved set、hit set、gold target set 都会同步排除 anchor-target overlap
- 只对有 gold targets 的 valid graphs 求平均

默认：

- `exclude_anchors_from_retrieved: true`
- `use_reachable_targets: true`

### 17.4 Single rollout metrics

每个 sample 单独看：

- `single_rollout/mean_recall`
- `single_rollout/mean_f1`
- `rollout/edge_count_mean`
- `rollout/recall_per_edge`
- `rollout/edge_count_rate_{k}`
- `rollout/edge_budget_full_rate`

### 17.5 Union@k metrics

对前 k 个 sampled trajectories 做 union：

```text
union_node_mask@k = any(node_masks[:k])
union_edge_mask@k = any(edge_masks[:k])
```

记录：

- `rollout_union@k/recall`
- `rollout_union@k/edges`
- `rollout_union@k/redundancy`

`k_windows` 默认 `[1, 2, 4, 8]`，但实际会被当前最大 sampled count 和 `MAX_LOGGED_K = 8` 裁剪。

### 17.6 Answer support probability

`marginal/answer_support_prob`：

- 对每个 node 计算它被 sampled terminal node mask 覆盖的频率。
- 只在 target nodes 上取均值。
- valid graph 再平均。

它衡量多次 rollout 中答案节点被支持的边际概率。

### 17.7 Terminal diagnostics

若 `enable_terminal_diagnostics = true`，额外记录：

- `terminal/policy_stop_rate`
- `terminal/structural_stop_rate`
- `terminal/budget_truncated_rate`
- `terminal/policy_terminal_rate`
- `terminal/forced_terminal_rate`
- `terminal/hit_then_continue_rate`
- `terminal/wasted_edge_rate`
- `terminal/stop_after_hit_rate`

其中 hit-then-continue / wasted-edge 通过逐步扫描 trajectory edge prefix 判断：一旦 active nodes 已经 hit target，后续继续扩展的边会被视为 wasted。

## 18. 当前算法的一句话总结

当前 `Weaver` 是：

```text
在 canonical multi-source root-reachable edge-set 状态空间上，
用 question-conditioned forward policy 逐步扩展证据子图，
用 learned-but-detached backward probability、state flow 和 terminal reward 构成 SubTrajectory Balance 约束，
再用 replay bank 的弱监督轨迹给早期训练提供额外 SubTB 约束。
```

更具体地说：

- forward 决定推理时怎样从 anchors 扩展边或停止。
- backward 只在训练中为 SubTB 提供 parent-child 反向概率；默认 residual 中 detach，辅助训练权重为 0。
- state flow 只在训练中承载 GFlowNet 的流一致性。
- reward 只评价 terminal canonical state。
- replay 只提供预处理 oracle trajectories，不是在线缓存。
- evaluation 只看 terminal active nodes / selected edges 的检索效果。
