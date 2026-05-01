# Weaver 特征工程实现复盘

更新时间：2026-05-01

本文复盘当前特征工程相关代码。范围包含离线预处理、物化 schema、数据读取、在线
`FeatureBank`/`StateContext`/transition features、edge/stop 打分链路。Reward、loss、LLM
eval、训练诊断只在影响特征语义或调用链时提及。

核对方式：按当前源码用 `rg` 枚举相关 `class`/`def`，重点核对
`src/data/preprocess/*`、`src/graph/*`、`src/data/schema/*`、`src/data/datamodule.py`、
`src/weaver/config.py`、`src/weaver/policy.py`、`src/weaver/state*.py` 和
`src/weaver/nn/*`。函数清单只覆盖特征工程及其直接调用链，不展开 reward/loss/LLM eval
内部实现。

## 1. 总体判断

当前实现的主线是合理的：

- 离线阶段把原始检索图转为局部图，物化文本 embedding、anchor 结构距离和 target
  shortest-path 标签。
- 在线 policy 不读取 `target_*`、`node_target_distance` 等答案派生字段，策略特征主要是
  semantic prior + anchor-conditioned DDE + 当前子图 readout + action residual。
- `EdgeScorer` 用 `semantic_prior_residual`：PLM 语义相似度作为 base logit，state/action
  特征只学习 residual correction，逻辑上能降低早期训练破坏 prior 排序的风险。
- State 实现把理论上的完整 Markov state 拆成静态 batch 上下文 + 动态
  `active_nodes/active_edges/root_edges`，内存语义清楚。

主要工程风险：

- 默认 rollout 配置仍走物理重复 batch，`train_num_rollout=8` 时静态图和静态特征会放大 8 倍。
- `StateReadout` 每步扫描全图边生成 frontier；预算小可接受，图或 rollout 数变大后会成为在线热点。
- `compute_target_path_labels` 对 anchor/target 分别做 BFS，多 target 图的离线成本偏高。
- `repeat_retrieval_batch` 没有重复 `target_shortest_path_count_flat`，当前主策略不用它，但 legacy
  coverage 或未来 teacher 若依赖该字段会有缺字段风险。
- 非文本实体共享一个 learnable embedding，表达能力受限；好处是避免无文本实体非法索引和 NaN。

## 2. 目录结构

### 2.1 配置

| 路径 | 职责 |
| --- | --- |
| `configs/preprocess.yaml` | Hydra 预处理入口，默认组合 paths/dataset/preprocess。 |
| `configs/preprocess/default.yaml` | 图清洗、过滤、encoder、batch size、text cache 配置。 |
| `configs/dataset/*.yaml` | 数据源、字段映射、LMDB/metadata/embedding 路径。 |
| `configs/datamodule/default.yaml` | DataLoader 参数。 |
| `configs/model/weaver.yaml` | `FeatureEncoder`、`StateReadout`、`TransitionFeatureBuilder`、`EdgeScorer`、`StopExpandGate` 默认配置。 |

### 2.2 离线预处理

| 路径 | 职责 |
| --- | --- |
| `src/preprocess.py` | Hydra CLI 入口。 |
| `src/data/pipeline.py` | 预处理总控：source -> graph_collect -> text_encode -> materialize。 |
| `src/data/preprocess/source.py` | 读取 HF 数据集，按 `column_map` 转成 `RawSample`。 |
| `src/data/preprocess/samples.py` | `RawSample`、`PreparedSample`、`SplitFilter` 数据结构。 |
| `src/data/preprocess/graph_collect.py` | 清洗边、过滤样本、构造 local graph、计算路径标签、积累 vocab。 |
| `src/graph/ops.py` | local graph、root edge、active node 重建等图操作。 |
| `src/graph/paths.py` | anchor/target BFS 标签、shortest suffix count、shortest-path edge mask。 |
| `src/data/preprocess/vocab.py` | Entity/Relation vocab 和 catalog，处理 text/non-text entity 映射。 |
| `src/data/preprocess/text_encode.py` | Transformer 文本编码，输出 L2-normalized embeddings，支持磁盘缓存。 |
| `src/data/preprocess/materialize.py` | 写 embedding artifact、catalog、LMDB sample、split index。 |

### 2.3 数据读取和 batch

| 路径 | 职责 |
| --- | --- |
| `src/data/schema/fields.py` | LMDB record 字段名和存储 schema 校验。 |
| `src/data/schema/batch.py` | PyG `RetrievalData`/`RetrievalBatch` 增量规则。 |
| `src/data/dataset.py` | 从 split index + LMDB 读取样本并构造成 `RetrievalData`。 |
| `src/data/collate.py` | PyG batching、`question_emb` stack、`edge_batch/edge_ptr` 构造。 |
| `src/data/datamodule.py` | 加载 dataset 和模型资源 embedding 表，并做资源校验。 |
| `src/data/schema/repeat.py` | 多 rollout 物理重复 `RetrievalBatch`。 |

### 2.4 在线特征和 policy

| 路径 | 职责 |
| --- | --- |
| `src/weaver/config.py` | 把 Hydra config 标准化为 runtime config，并注入 embedding tensors。 |
| `src/weaver/module.py` | LightningModule，组装 policy、rollout、reward、loss。 |
| `src/weaver/policy.py` | 单步 policy 主链路：feature/readout/candidate/action/edge/stop/flow。 |
| `src/weaver/state.py` | 动态 state：`State` 和 `RolloutState`。 |
| `src/weaver/state_ops.py` | frontier edge 枚举。 |
| `src/weaver/nn/feature_encoder.py` | 静态语义和模型空间特征。 |
| `src/weaver/nn/dde.py` | anchor-conditioned directional diffusion encoding。 |
| `src/weaver/nn/static_graph_features.py` | 静态 degree/frequency 特征。 |
| `src/weaver/nn/edge_encoder.py` | `(src, relation, dst)` 模型空间 edge 表征。 |
| `src/weaver/nn/state_readout.py` | 当前 active subgraph readout、path memory、frontier summary/cache。 |
| `src/weaver/nn/candidate_context.py` | 候选边坐标、active/new 状态、semantic prior 输入。 |
| `src/weaver/nn/transition_features.py` | 候选边 MDP transition type 编码。 |
| `src/weaver/nn/action_features.py` | 旧 action feature 名称兼容导出。 |
| `src/weaver/nn/edge_scorer.py` | semantic prior + residual edge logit。 |
| `src/weaver/nn/stop_gate.py` | Stop/Expand option logit。 |
| `src/weaver/nn/flow_head.py` | state log-flow head。 |

## 3. 端到端调用链

### 3.1 预处理链路

```text
python src/preprocess.py dataset=webqsp
  -> src/preprocess.py:main
  -> run_preprocess_pipeline(raw_cfg)
      -> iter_samples(...)
          -> datasets.load_dataset(...)
          -> _row_to_sample(...)
      -> collect_and_filter_graphs(...)
          -> _clean_edges(...)
          -> build_local_graph(...)
          -> _filter_graph_entities(...)
          -> compute_path_labels(...)
              -> compute_target_path_labels(...)
              -> compute_anchor_path_labels(...)
          -> _build_graph_catalog_ids(...)
      -> EntityCatalog.build(...) / RelationCatalog.build(...)
      -> encode_text_features(...)
          -> _encode_text_table(...)
          -> cache hit: _load_cached_embeddings(...)
          -> cache miss: TextEncoder.encode(...) -> _write_cached_embeddings(...)
      -> materialize_preprocessed_data(...)
          -> _save_catalog_artifacts(...)
          -> _save_embedding_artifacts(...)
          -> _write_lmdb_samples(...)
          -> _save_split_indices(...)
```

物化后的核心字段：

| 字段 | shape | 来源 | 在线策略是否使用 |
| --- | --- | --- | --- |
| `edge_index` | `[2, E]` | local graph | 是 |
| `node_entity_catalog_ids` | `[N]` | entity vocab | 是 |
| `edge_relation_catalog_ids` | `[E]` | relation vocab | 是 |
| `question_emb` | `[D]` 或 batch 后 `[B, D]` | text encoder | 是 |
| `anchor_node_ids` | `[A]` | question entities grounded into graph | 是 |
| `target_node_ids` | `[T_all]` | answer entities in graph | 否，reward/eval/diagnostics 用 |
| `reachable_target_node_ids` | `[T]` | anchor 可达 targets | 否，reward/eval/diagnostics 用 |
| `anchor_node_forward_distances_flat` | `[N]` | multi-source BFS | 当前不直接用；DDE 替代距离 embedding |
| `anchor_node_backward_distances_flat` | `[N]` | reverse multi-source BFS | 当前不直接用；DDE 替代距离 embedding |
| `node_target_distance` | `[N]` | nearest target distance | 否，reward/diagnostics 相关 |
| `target_node_distances_flat` | `[T*N]` | per-target reverse BFS | 否，teacher/diagnostics/legacy 用 |
| `target_shortest_path_count_flat` | `[T*N]` | distance-bucket DP | 否，legacy teacher 可能用 |
| `target_shortest_path_edge_mask_flat` | `[T*E]` | shortest path edge marking | 否，diagnostics/legacy 用 |

### 3.2 训练和在线特征链路

```text
train.py
  -> RetrievalDataModule.prepare_data/setup
      -> _load_tensor_artifact(entity_text_embeddings)
      -> _extract_entity_embedding_map(entity_metadata)
      -> _load_tensor_artifact(relation_embeddings)
      -> _validate_model_resources(...)
  -> WeaverModule(...)
      -> build_policy_runtime_config(...)
      -> Policy(...)
  -> RolloutRunner.run_training_rollouts_and_backward(...)
      -> RolloutEngine.run_vectorized(...)
          -> policy.prepare_rollout_context(batch)
              -> FeatureEncoder.forward(batch) -> FeatureBank
          -> State.create_initial(...) 或 RolloutState.create_initial(...)
          -> 每一步 policy(batch, state, rollout_context)
```

### 3.3 单步 policy 链路

```text
Policy.forward(batch, state, rollout_context)
  -> StateReadout.forward(fb, batch, state)
      -> pool active nodes
      -> pool active edges
      -> pool active relations/path memory
      -> compute frontier + frontier edge_h cache
      -> state_h, progress, frontier ids/batch/h
  -> FlowHead(state_h) -> state_log_flow
  -> build_candidate_context(...)
  -> TransitionFeatureBuilder.forward(...)
      -> candidate_semantic_scores(...)
      -> transition_type = [src_active_dst_new, dst_active_src_new, both_active]
  -> EdgeScorer.forward(...)
      -> semantic_logits
      -> residual_logits from [state_h, edge_h, transition_type, stop_gradient(semantic_logits)]
      -> final edge logits
  -> frontier_logit_summary(...)
  -> StopExpandGate.forward(...)
      -> stop_logits, expand_logits
```

## 4. 当前特征定义

### 4.1 `FeatureBank`

`FeatureEncoder.forward(batch)` 输出：

| 字段 | shape | 含义 |
| --- | --- | --- |
| `node_sem_h` | `[N, D]` | entity semantic space；文本实体查 PLM 表，非文本实体用共享 learnable embedding。 |
| `rel_sem_h` | `[E, D]` | edge relation semantic space。 |
| `query_sem_h` | `[B, D]` | question semantic space。 |
| `node_h` | `[N, H]` | `RoleProjection(node_sem_h) + DDE structural bias` 后 LayerNorm。 |
| `rel_h` | `[E, H]` | relation model-space projection。 |
| `query_h` | `[B, H]` | question model-space projection。 |
| `node_dde` | `[N, D_dde]` | anchor indicator + forward/backward diffusion coordinates。 |
| `node_is_non_text` | `[N]` | 节点是否缺文本 embedding。 |
| `node_log_degree` | `[N]` | undirected `log1p(degree)`。 |
| `edge_relation_log_frequency` | `[E]` | 同图内该 relation 的 `log1p(freq)`。 |

默认 `D_dde = 1 + 2 + 2 = 5`，`H=1024`，`D=1024`。

### 4.2 `StateContext`

`StateReadout.forward(...)` 输出：

| 字段 | shape | 含义 |
| --- | --- | --- |
| `state_h` | `[B_or_R, H]` | 当前子图状态 readout。 |
| `progress` | `[B_or_R]` | 已选 non-root edge 数 / `expand_budget`。 |
| `relation_path_h` | `[B_or_R, H]` 或 `None` | active relation 的 query-attentive pooling。 |
| `frontier_summary` | `[B_or_R, 3]` 或 `None` | frontier edge score 的 max/logmeanexp/log_size。 |
| `frontier_edge_ids` | `[C]` | 当前合法 Expand 候选边 ids。 |
| `frontier_edge_batch` | `[C]` | 每个候选属于哪个 graph/rollout row。 |
| `frontier_edge_h` | `[C, H]` | 已编码候选边，可被 `EdgeScorer` 复用。 |

### 4.3 `TransitionFeatureBuilder`

当前 transition feature 只保留 3 维：

| 组 | 维度 | 含义 |
| --- | ---: | --- |
| `transition_type` | 3 | `src_active_dst_new`、`dst_active_src_new`、`both_active`。 |

已删除旧的 DDE transition、semantic weak、path history、branching、status 拼接特征。原因是：

- DDE 已通过 `FeatureEncoder` 的 structural bias 进入 `node_h`，再由 `EdgeEncoder(h_u,h_r,h_v)`
  进入 residual。
- `<q,r>` 和 text new-node relevance 属于 semantic base measure，不再作为 residual feature 重复输入。
- relation path memory 属于 state readout，应通过 `h_s` 影响 residual。
- progress/frontier size 属于 Stop/Expand option calibration，应交给 `StopExpandGate`。
- `neither_active` 在合法 frontier 下是非法状态，现在作为 invariant 检查，不再进入模型输入。

### 4.4 `EdgeScorer`

默认公式：

```text
semantic(e) = <q_sem, r_sem> + alpha * 1[new endpoint is text] * <q_sem, new_sem>
b_e = tau * semantic(e)
delta_e = MLP([state_h, edge_h, transition_type, stop_gradient(b_e)])
z_e = b_e + lambda_eff * delta_e
```

设计意图：

- PLM semantic prior 作为基础排序。
- residual 学习 state-dependent correction。
- residual output 默认 zero-init，并有 warmup multiplier，早期 `z_e ~= b_e`。
- `EdgeScorer` 优先复用 `StateReadout` 缓存的 `frontier_edge_h`，避免候选边重复编码。
- residual 只看状态、edge representation、transition type 和 detached prior，不重新学习语义排序器。

### 4.5 `EdgeScorer` 关键检查结论

已按 semantic-prior residual 的职责边界检查：

- `final_logits` 使用 `effective_residual_scale()`，实际公式是
  `semantic_logits + residual_scale_multiplier * residual_scale * residual_logits`，没有使用裸
  `residual_scale` 绕过 warmup。
- residual MLP 最后一层强制 zero-init；`zero_init_residual_output=false` 会直接报错，避免 step 0
  policy 偏离 semantic prior。
- base `semantic_logits` 不 detach，`logit_scale` 和 semantic prior 路径仍可学习；但 residual 输入中的
  `b_e` 已 detach，形式为
  `MLP([state_h, edge_h, transition_type, stop_gradient(b_e)])`，避免 residual 反向重塑 prior 计算链。
- root edge 诊断已补充并挂到 `train_policy_diagnostics`：`edge/base_logit_std`、
  `edge/residual_logit_std`、`edge/residual_to_base_std_ratio`、
  `edge/prior_rank_vs_final_rank_kendall`、`edge/answer_edge_prior_rank`、
  `edge/answer_edge_final_rank`。其中 `edge/residual_logit_std` 统计的是
  `lambda_eff * delta_e` 的标准差，可直接和 base logit 标准差比较。

## 5. 函数级清单

### 5.1 入口和 pipeline

| 文件 | 函数/类 | 主要参数 | 含义 |
| --- | --- | --- | --- |
| `src/preprocess.py` | `maybe_print_config(cfg)` | `cfg: DictConfig` | 可选打印 Hydra config。 |
| `src/preprocess.py` | `main(cfg)` | `cfg: DictConfig` | Hydra 入口，调用 `run_preprocess_pipeline`。 |
| `src/data/pipeline.py` | `_resolve_path(value, name)` | 路径值、错误字段名 | 将配置值转成 `Path`，空值报错。 |
| `src/data/pipeline.py` | `_build_split_filter(section)` | split filter 配置段 | 生成 `SplitFilter(require_answer_in_graph, require_reachable_answer)`。 |
| `src/data/pipeline.py` | `_resolve_dataset_paths(dataset_cfg)` | dataset config | 解析 LMDB、metadata、embedding、catalog 路径。 |
| `src/data/pipeline.py` | `run_preprocess_pipeline(raw_cfg)` | 全量 Hydra config | 串起读取、图收集、文本编码、物化三个阶段。 |

### 5.2 数据源和样本结构

| 文件 | 函数/类 | 主要参数 | 含义 |
| --- | --- | --- | --- |
| `source.py` | `iter_samples(dataset, splits, column_map, dataset_source, hf_dataset, hf_cache_dir)` | 数据集名、split 列表、字段映射、HF 配置 | 从 HF 数据集逐行 yield `RawSample`；`stark` 当前禁用。 |
| `source.py` | `_row_to_sample(row, dataset, split, column_map)` | 原始行和字段映射 | 解析 question、graph、question entities、answer entities。 |
| `source.py` | `_parse_graph(graph_raw)` | 原始 graph 字段 | 转成 `(head, relation, tail)` tuple 列表，跳过非法三元组。 |
| `source.py` | `_normalize_entity(value)` | 任意实体值 | `str(value).strip()`。 |
| `source.py` | `_coerce_string_list(value)` | 字符串/列表/空值 | 规范化为非空字符串 list。 |
| `samples.py` | `SplitFilter` | `require_answer_in_graph`, `require_reachable_answer` | split 级过滤规则。 |
| `samples.py` | `RawSample` | dataset/split/question/graph/entities | 原始样本中间表示。 |
| `samples.py` | `PreparedSample` | graph tensors、anchor/target labels、catalog ids | 预处理完成、准备写 LMDB 的样本。 |

### 5.3 图收集和路径标签

| 文件 | 函数/类 | 主要参数 | 含义 |
| --- | --- | --- | --- |
| `graph_collect.py` | `collect_and_filter_graphs(sample_iter, split_filters, dedup_edges, remove_self_loops, validate_alignment)` | 样本迭代器、过滤和清洗开关 | 主函数：清洗图、过滤 anchor/answer/reachable、计算路径标签、积累 vocab。 |
| `graph_collect.py` | `_clean_edges(graph, remove_self_loops, dedup_edges)` | 原始边、清洗开关 | 删除 self-loop 和重复三元组。 |
| `graph_collect.py` | `_filter_graph_entities(entities, node_index)` | entity 列表和 local node map | 保留在图内且去重的 entities。 |
| `graph_collect.py` | `_build_graph_catalog_ids(graph_edges, node_index, entity_vocab, relation_vocab)` | 局部图和全局 vocab | 生成 node entity catalog ids、edge relation catalog ids。 |
| `graph_collect.py` | `_validate_edge_alignment(graph_edges, node_index, edge_index)` | 边列表和 tensor | 保证 `edge_index` 顺序和 `graph_edges` 对齐。 |
| `graph/ops.py` | `build_local_graph(graph_edges)` | `(h,r,t)` 序列 | 构造 local `node_index` 和 `edge_index`。 |
| `graph/ops.py` | `build_anchor_induced_edge_mask(edge_index, anchor_mask)` | 图边、anchor node mask | 初始 root edges：两个端点都为 anchor 的边。 |
| `graph/ops.py` | `rebuild_active_nodes(active_edges, edge_index, anchor_mask)` | active edge mask、图边、anchors | 从 anchors + active edge endpoints 重建 `V_s`。 |
| `graph/ops.py` | `compute_uniform_nonroot_backward_removals(...)` | active/root edges、edge_batch、anchor_mask | backward policy 的 non-root removable edges，属于 rollout 反向概率，不是主特征。 |
| `graph/ops.py` | `prune_to_protected_core(active_nodes, active_edges, edge_index, protected_nodes, max_iters)` | active 子图和保护节点 | eval 辅助：迭代剪掉非保护叶子。 |
| `paths.py` | `compute_path_labels(edge_index, anchor_node_ids, target_node_ids, num_nodes)` | 图、anchors、targets、节点数 | 统一生成 anchor labels、target labels、nearest target distance。 |
| `paths.py` | `compute_target_path_labels(edge_index, anchor_node_ids, target_node_ids, num_nodes)` | 图、anchors、targets | 生成 reachable targets、`d(v,target)`、shortest suffix counts、target path edge mask。 |
| `paths.py` | `compute_anchor_path_labels(edge_index, anchor_node_ids, num_nodes)` | 图、anchors、节点数 | 生成 anchor forward/backward min distances。 |
| `paths.py` | `_shortest_suffix_count_matrix(adjacency, target_node_ids, target_distances)` | adjacency、targets、距离矩阵 | 对每个 target 调 `_shortest_suffix_counts`。 |
| `paths.py` | `_shortest_suffix_counts(adjacency, target_node_id, node_to_target_dist)` | 单 target 距离 | 按距离 bucket 做 DP，统计每个节点到 target 的 shortest suffix 数。 |
| `paths.py` | `_shortest_path_edge_mask(src, dst, anchor_distances, target_node_ids, target_distances)` | 边列表、anchor/target 距离 | chunked tensor 判断每条边是否在某条 anchor-to-target shortest path 上。 |
| `paths.py` | `_multi_source_min_dist(adjacency, starts)` | adjacency、起点集合 | 多源 BFS 最短距离。 |
| `paths.py` | `_nearest_target_distance(target_node_distances_flat, num_targets, num_nodes)` | per-target 距离 | 压成每个节点到最近 target 的距离，不可达用大整数。 |
| `paths.py` | `_bfs(adjacency, start)` | adjacency、起点 | 单源 BFS。 |
| `paths.py` | `_build_adjacency(num_nodes, src, dst)` | 节点数、边 src/dst | 构造正向和反向 adjacency。 |
| `paths.py` | `_edge_lists(edge_index)` | `[2,E]` tensor | 校验并转成 Python list。 |
| `paths.py` | `_valid_unique_nodes(node_ids, num_nodes)` | node id tensor、节点数 | 保留范围内、去重 node ids。 |
| `paths.py` | `_empty_target_labels()` / `_empty_anchor_labels()` | 无 | 空标签兜底。 |

### 5.4 Vocab、文本编码、物化

| 文件 | 函数/类 | 主要参数 | 含义 |
| --- | --- | --- | --- |
| `vocab.py` | `EntityVocab.add/id/labels/__len__` | entity string | 全局 entity id 映射。 |
| `vocab.py` | `RelationVocab.add/id/labels/__len__` | relation string | 全局 relation id 映射。 |
| `vocab.py` | `EntityTyping(non_text_prefixes)` | 默认 `("m.", "g.")` | 判断 entity 是否有文本 embedding 来源。 |
| `vocab.py` | `EntityCatalog.build(vocab, typing, sort_text_entities)` | entity vocab 和 typing | 生成 entity labels、text embedding ids、non-text mask、text labels。 |
| `vocab.py` | `EntityCatalog.to_dict/save/load` | path | catalog 序列化。 |
| `vocab.py` | `RelationCatalog.build(vocab)` | relation vocab | relation label 转 text label，替换 `/` 和 `_`。 |
| `vocab.py` | `RelationCatalog.to_dict/save/load` | path | catalog 序列化。 |
| `text_encode.py` | `TextEncoder(model_name, device, progress_bar)` | HF model 和设备 | 加载 tokenizer/model，CLS pooling + L2 normalize。 |
| `text_encode.py` | `TextEncoder._forward_batch(texts)` | 文本 batch | Transformer 前向，返回 CPU float32 normalized embeddings。 |
| `text_encode.py` | `TextEncoder.encode(texts, batch_size, desc, query_prefix)` | 文本列表、batch size、可选 prefix | 分 batch 编码并拼接。 |
| `text_encode.py` | `EncodedFeatures` | 三类 embedding tensor | entity/relation/question embedding 容器。 |
| `text_encode.py` | `encode_text_features(entity_text_labels, relation_text_labels, question_texts, encoder_name, device, batch_size, progress_bar, cache_dir)` | 三类文本和 encoder 配置 | 统一编码三类文本，问题加 `Represent this sentence: ` prefix。 |
| `text_encode.py` | `_encode_text_table(texts, encoder_factory, batch_size, desc, encoder_name, cache_dir, cache_kind, query_prefix)` | 单类文本表 | 优先读缓存，miss 后调用 encoder 并写缓存。 |
| `text_encode.py` | `_text_cache_path(cache_dir, kind, encoder_name, texts, query_prefix)` | 缓存参数 | 用 schema version + 文本列表 hash 生成 cache path。 |
| `text_encode.py` | `_load_cached_embeddings(path, expected_rows)` | cache 文件和行数 | 校验 schema、shape、dtype、finite，失败返回 `None`。 |
| `text_encode.py` | `_write_cached_embeddings(path, embeddings)` | path、embedding tensor | 原子写缓存。 |
| `materialize.py` | `materialize_preprocessed_data(...)` | prepared samples、catalogs、embeddings、输出路径、LMDB 参数 | 写所有离线 artifact。 |
| `materialize.py` | `_validate_materialize_inputs(...)` | samples/catalogs/embeddings | 校验非空、维度和行数一致。 |
| `materialize.py` | `_save_catalog_artifacts(...)` | catalogs 和路径 | 写 entity/relation catalog 与 entity metadata。 |
| `materialize.py` | `_save_embedding_artifacts(...)` | catalogs、embedding tensors、目录 | 写 embedding `.pt` 文件。 |
| `materialize.py` | `_entity_text_embedding_ids_to_map(entity_text_embedding_ids)` | 1-based text ids | 转成 0-based map，非文本 0 -> -1。 |
| `materialize.py` | `_write_lmdb_samples(...)` | samples、question embeddings、LMDB 参数 | 分 split 写 LMDB，按 `commit_frequency` commit。 |
| `materialize.py` | `_sample_to_lmdb_record(sample, question_embedding)` | prepared sample 和问题 embedding | 转成 `SampleFields` record。 |
| `materialize.py` | `_validate_prepared_sample_shapes(sample)` | prepared sample | 写入前 shape/length 校验。 |
| `materialize.py` | `_save_split_indices(prepared_samples, metadata_dir, schema_version)` | samples、metadata 目录 | 写 `{split}.index.pt`。 |
| `materialize.py` | `_require_numel(sample_id, name, tensor, expected)` | tensor 和期望长度 | 长度校验。 |
| `materialize.py` | `_lmdb_path(lmdb_dir, split)` | LMDB 根目录和 split | 返回 `{split}.lmdb`。 |
| `materialize.py` | `_reset_output_path(path, overwrite)` | 输出路径、覆盖开关 | 覆盖时删除旧 LMDB。 |

### 5.5 数据 schema、dataset、collate、datamodule

| 文件 | 函数/类 | 主要参数 | 含义 |
| --- | --- | --- | --- |
| `fields.py` | `SampleFields` | 常量 | 定义 LMDB record 字段和 PyG increment 规则集合。 |
| `fields.py` | `StorageSchema.validate(data)` | record dict | 校验必需字段、shape、dtype、node id 范围和 reachable subset。 |
| `fields.py` | `_require_tensor/_require_1d/_require_1d_length` | value、name、dtype/length | 通用 tensor 校验。 |
| `fields.py` | `_scalar_int(value, name)` | int 或 scalar LongTensor | 转成 Python int。 |
| `fields.py` | `_validate_node_ids(value, name, num_nodes)` | node ids | 范围校验。 |
| `fields.py` | `_validate_subset(subset, superset, subset_name, superset_name)` | 两组 ids | reachable targets 必须是 targets 子集。 |
| `batch.py` | `RetrievalData.__inc__(key, value, ...)` | PyG key/value | node id 字段按 `num_nodes` 偏移，静态/label 字段不偏移。 |
| `batch.py` | `RetrievalBatch.__inc__(key, value, ...)` | PyG key/value | batch 级 node id 字段按总节点数偏移。 |
| `batch.py` | `num_nodes_total/num_edges_total/num_graphs_total` | 无 | 推断 batch 总节点、边、图数。 |
| `dataset.py` | `LMDBSampleStore(path, readahead, max_readers)` | LMDB path | 只读打开 LMDB。 |
| `dataset.py` | `LMDBSampleStore.load_sample(sample_id)` | sample id | 从 LMDB 反序列化样本。 |
| `dataset.py` | `RetrievalDataset(lmdb_dir, metadata_dir, split, lmdb_readahead, max_readers)` | 路径和 split | split dataset。 |
| `dataset.py` | `RetrievalDataset.get(idx)` | 样本下标 | 读取 raw record 并 `_build_retrieval_data`。 |
| `dataset.py` | `_build_retrieval_data(raw, sample_id)` | LMDB record | 转成 `RetrievalData` 并运行 runtime shape 校验。 |
| `dataset.py` | `_validate_runtime_shapes(...)` | 样本字段 | 运行时 shape/length/node id 校验。 |
| `dataset.py` | `_load_split_index(metadata_dir, split)` | metadata 路径 | 读取 sample id 列表。 |
| `dataset.py` | `_tensor(value, dtype)` | 任意 tensor-like | 转成指定 dtype contiguous tensor。 |
| `dataset.py` | `_scalar_int/_require_shape/_require_numel/_require_node_ids/_lmdb_path` | 校验参数 | dataset 读取辅助校验。 |
| `collate.py` | `RetrievalCollator(follow_batch, exclude_keys)` | PyG collate 参数 | collator 配置。 |
| `collate.py` | `RetrievalCollator.__call__(batch_list)` | `RetrievalData` 列表 | PyG batch，stack question embedding，补 `node_ptr/edge_batch/edge_ptr`。 |
| `collate.py` | `_stack_question_embeddings(batch_list)` | 样本列表 | 把 `[D]` 问题 embedding stack 成 `[B,D]`。 |
| `collate.py` | `_attach_edge_batch(batch)` | `RetrievalBatch` | 用 node ptr 推断每条边所属 graph，并构造 edge ptr。 |
| `datamodule.py` | `ModelResources` | entity embeddings、entity map、relation embeddings | 初始化 `WeaverModule/Policy` 所需静态资源。 |
| `datamodule.py` | `RetrievalDataModule(...)` | dataset cfg、batch/worker/LMDB 参数 | Lightning datamodule。 |
| `datamodule.py` | `prepare_data()` | 无 | 检查 LMDB、metadata、embedding artifact 存在。 |
| `datamodule.py` | `setup(stage)` | Lightning stage | 加载 model resources，构造 train/val/test dataset。 |
| `datamodule.py` | `train_dataloader/val_dataloader/test_dataloader()` | 无 | 返回对应 split 的 DataLoader。 |
| `datamodule.py` | `teardown(stage)` | Lightning stage | 关闭已经打开的 LMDB dataset。 |
| `datamodule.py` | `_ensure_model_resources_loaded()` | 无 | 加载并校验三类 embedding 资源。 |
| `datamodule.py` | `_build_loader(dataset, training)` | dataset、训练/评估标志 | 按 batch size、worker、shuffle/drop_last 配置构造 DataLoader。 |
| `datamodule.py` | `_load_artifact/_load_tensor_artifact` | path/name/keys | 兼容 tensor 或 mapping artifact。 |
| `datamodule.py` | `_extract_tensor(artifact, name, key)` | artifact、字段名 | 从 mapping/dataclass-like 对象中取 tensor；当前保留为通用兼容 helper。 |
| `datamodule.py` | `_extract_entity_embedding_map(artifact, name)` | entity metadata | 优先读 0-based map，兼容 1-based ids。 |
| `datamodule.py` | `_validate_model_resources(...)` | 三类资源 tensor | 校验维度、map 范围、finite 和 L2 norm。 |
| `datamodule.py` | `_validate_l2_normalized_rows(tensor, name, atol)` | embedding tensor | 行向量 L2 norm 校验。 |
| `datamodule.py` | `_path_from_dataset_cfg(cfg, key)` | dataset cfg、路径 key | 从 `dataset.paths` 解析必需路径。 |
| `datamodule.py` | `_split_name(cfg, key, default)` | dataset cfg、split key、默认值 | 解析可覆盖的 split 名。 |
| `datamodule.py` | `_require_dir/_require_file(path, name)` | 路径和错误名 | 启动前检查必要目录/文件存在。 |
| `repeat.py` | `repeat_retrieval_batch(batch, repeats)` | batch、重复次数 | 物理复制 batch，用于多 rollout 默认路径。 |
| `repeat.py` | `_repeat_node_offset_fields(...)` | node id 字段 | 对 node id 增加节点偏移。 |
| `repeat.py` | `_repeat_tensor_fields(...)` | tensor 字段名 | 沿 dim0 复制 node/edge/label tensors。 |
| `repeat.py` | `_repeat_question_emb(...)` | question embedding | 重复 `[B,D]` 问题 embedding。 |
| `repeat.py` | `_repeat_graph_sequence_fields(...)` | graph-level 字符串/list | 重复 graph 序列字段。 |
| `repeat.py` | `_repeat_tensor(tensor, repeats)` | tensor、次数 | 0D 用 repeat，其他 dim0 concat。 |
| `repeat.py` | `_validate_batch(batch)` | `RetrievalBatch` | 检查重复前必要字段。 |

### 5.6 Runtime config 和 module

| 文件 | 函数/类 | 主要参数 | 含义 |
| --- | --- | --- | --- |
| `config.py` | `PolicyRuntimeConfig` | hidden_dim 和各子模块 cfg | policy runtime 配置容器。 |
| `config.py` | `RolloutRuntimeConfig` | budget/rollout/chunk/static flags | rollout runtime 配置容器。 |
| `config.py` | `EvalRuntimeConfig/ScheduleRuntimeConfig/DiagnosticsRuntimeConfig` | eval/schedule/diagnostic cfg | 其他 runtime 配置容器。 |
| `config.py` | `build_policy_runtime_config(policy_cfg, entity_text_embeddings, entity_embedding_map, relation_embeddings)` | policy config 和 runtime embedding | 规范化 policy cfg，注入 embedding tensors。 |
| `config.py` | `build_rollout_runtime_config(rollout_cfg)` | rollout cfg | 校验 budget/rollout/chunk/static rollout flags。 |
| `config.py` | `build_eval_runtime_config(eval_cfg, eval_num_rollout)` | eval cfg、eval rollout 数 | 校验 budgets 等评估参数。 |
| `config.py` | `build_schedule_runtime_config(schedule_cfg)` | schedule cfg | 温度 schedule 配置。 |
| `config.py` | `build_diagnostics_runtime_config(diagnostic_cfg)` | diagnostic cfg | 诊断开关配置。 |
| `config.py` | `build_feature_encoder_config(cfg, entity_text_embeddings, entity_embedding_map, relation_embeddings, hidden_dim)` | feature encoder cfg 和资源 | 校验 embedding dim，禁止 config 内直接放 runtime tensors。 |
| `config.py` | `normalize_state_readout_config(cfg)` | state readout cfg | 移除/校验已固定的旧配置项。 |
| `config.py` | `normalize_transition_features_config(cfg)` | transition cfg | 禁止旧 handcrafted transition feature 开关，保持 residual feature 固定。 |
| `config.py` | `normalize_stop_scorer_config(cfg)` | stop scorer cfg | 兼容旧 stop 配置命名并返回规范化配置。 |
| `config.py` | `_pop_expected(cfg, key, expected, namespace)` | cfg key 和期望值 | 若旧 key 非期望值则报错。 |
| `config.py` | `validate_rollout_counts(expand_budget, train_num_rollout, eval_num_rollout)` | rollout 数 | 范围校验。 |
| `config.py` | `normalize_chunk_size(value, fallback, name)` | chunk size | `None` 用 fallback，其他必须 >=1。 |
| `module.py` | `WeaverModule(...)` | embeddings、policy/rollout/eval/reward/loss cfg | LightningModule，组装所有训练部件。 |
| `module.py` | `training_step(batch, batch_idx)` | batch、batch idx | 更新 residual schedule，运行 rollout + backward + optimizer/logging。 |
| `module.py` | `validation_step/test_step` | batch、batch idx | 调 `eval_step`。 |
| `module.py` | `generate_subgraph_masks(batch, num_rollouts, temperature)` | batch、rollout 数、温度 | 推理生成 union subgraph masks。 |
| `module.py` | `eval_step(batch, prefix)` | batch、val/test 前缀 | 生成 eval rollouts 并计算 retrieval metrics。 |
| `module.py` | `log_training_step(...)` | rollout result、batch、optimizer、温度 | 聚合训练诊断并 log。 |
| `module.py` | `normalize_loss_config(loss_cfg, max_trajectory_len)` | loss cfg、最大轨迹长度 | 补默认 `max_trajectory_len`。 |

### 5.7 State、feature encoder、readout、scorer

| 文件 | 函数/类 | 主要参数 | 含义 |
| --- | --- | --- | --- |
| `state.py` | `State.create_initial(batch, expand_budget, validate_anchor_ids)` | batch、budget | 初始化 active nodes 为 anchors，root edges 为 anchor-anchor edges。 |
| `state.py` | `State.apply_expansion(chosen_edges, edge_index)` | 选中边、图边 | 原地加入边和两端节点。 |
| `state.py` | `expanded_edge_mask/ids/count_per_graph` | edge_batch、num_graphs | non-root learned edges 统计。 |
| `state.py` | `remaining_budget_per_graph/expand_ratio_per_graph` | edge_batch、num_graphs | 每图剩余 budget 和 progress。 |
| `state.py` | `synchronous_rollout_depth(edge_batch, num_graphs, active_graphs)` | edge batch、active mask | 校验同步 rollout 深度。 |
| `state.py` | `RolloutState.create_initial(batch, expand_budget, rollout_to_graph, validate_anchor_ids)` | 静态 batch、动态 rollout->graph 映射 | fused static rollout 的 2D state 初始化。 |
| `state.py` | `RolloutState.apply_expansion(rollout_ids, chosen_edges, edge_index)` | rollout rows、原图边 ids | 对 2D state 原地加入边。 |
| `state_ops.py` | `frontier_edges(batch, state, device)` | batch、1D state | 返回 inactive 且 incident to active node 的候选边。 |
| `state_ops.py` | `has_frontier_edge_per_graph(edge_batch, frontier_edge_ids, num_graphs, device)` | frontier ids | 每图是否有候选边。 |
| `feature_encoder.py` | `EntityEmbeddingLayer(entity_text_embeddings, entity_embedding_map, non_text_init_std)` | 文本 embedding 表、entity->text map | 文本实体查表，非文本实体共享参数。 |
| `feature_encoder.py` | `EntityEmbeddingLayer.forward(entity_ids)` | entity catalog ids | 返回 semantic-space node embeddings。 |
| `feature_encoder.py` | `RoleProjection(input_dim, hidden_dim, init)` | 输入/输出维度、初始化 | 线性投影 + LayerNorm，可加结构 bias。 |
| `feature_encoder.py` | `RoleProjection.forward(x, bias)` | 输入特征、可选 bias | 输出 model-space role feature。 |
| `feature_encoder.py` | `FeatureEncoder(...)` | embeddings、hidden dim、DDE cfg、projection cfg | 静态特征编码器。 |
| `feature_encoder.py` | `FeatureEncoder.forward(batch)` | `RetrievalBatch` | 输出 `FeatureBank`。 |
| `feature_encoder.py` | `_node_dde(batch, num_nodes, device, dtype)` | batch 和节点数 | 调 `DirectionalDDE` 或返回 0-width tensor。 |
| `feature_encoder.py` | `_structural_bias(node_dde, device, dtype)` | DDE tensor | 线性投影成 node structural bias。 |
| `feature_encoder.py` | `_node_is_non_text(batch, num_nodes, device, entity_embedding_map)` | batch 和 entity map | 从 batch 字段或 map 推导 non-text mask。 |
| `feature_encoder.py` | `_edge_relation_log_frequency(batch, rel_ids, device, dtype)` | relation ids、edge batch | 计算/缓存 per-edge relation frequency。 |
| `dde.py` | `DirectionalDDE(num_forward_rounds, num_backward_rounds, include_anchor_indicator)` | diffusion 轮数和 anchor indicator 开关 | anchor-conditioned 结构坐标。 |
| `dde.py` | `DirectionalDDE.forward(edge_index, anchor_node_ids, num_nodes)` | 图边、anchors、节点数 | 输出 `[N, D_dde]`。 |
| `dde.py` | `_mean_messages(values, source_index, target_index, num_nodes)` | 节点值和边方向 | 一轮 mean aggregation。 |
| `static_graph_features.py` | `node_log_degree(edge_index, num_nodes, dtype)` | 图边、节点数 | undirected `log1p(degree)`。 |
| `static_graph_features.py` | `edge_relation_log_frequency(relation_ids, edge_batch, dtype)` | relation ids、edge graph ids | 同图内 relation 频率。 |
| `edge_encoder.py` | `EdgeEncoder(hidden_dim)` | hidden dim | `W[src_h, rel_h, dst_h]`。 |
| `edge_encoder.py` | `EdgeEncoder.forward(src_h, rel_h, dst_h)` | 三个 `[E,H]` tensor | 输出 edge model-space 表征。 |
| `state_readout.py` | `StateReadout(hidden_dim, num_layers, dropout, edge_encoder, use_path_memory, use_frontier_summary)` | readout 配置 | query-conditioned state readout。 |
| `state_readout.py` | `StateReadout.forward(fb, batch, state)` | FeatureBank、batch、State/RolloutState | 输出 `StateContext`，并生成 frontier。 |
| `state_readout.py` | `_forward_rollout_state(fb, batch, state)` | 2D RolloutState | fused static rollout readout。 |
| `state_readout.py` | `_pool_nodes/_pool_edges/_pool_relations(...)` | active masks 和 batch ids | 对 active nodes/edges/relations 做 query-attentive pooling。 |
| `state_readout.py` | `_pool_rollout_nodes/_pool_rollout_edges/_pool_rollout_relations(...)` | 2D masks | RolloutState 版本 pooling。 |
| `state_readout.py` | `_frontier_readout(...)` | active nodes/edges | 扫描 frontier、编码候选边、算 summary。 |
| `state_readout.py` | `_frontier_rollout_readout(...)` | 2D active masks、rollout_to_graph | RolloutState 版本 frontier。 |
| `state_readout.py` | `_query_pool(query_h, values, batch_index, num_graphs)` | query、values、segment ids | segmented attention pooling。 |
| `candidate_context.py` | `build_candidate_context(batch, state, candidate_edge_ids, candidate_batch_ids, candidate_static_graph_ids, device)` | batch、state、候选边 ids | 构造候选边 src/dst、graph id、active 状态、static graph id。 |
| `candidate_context.py` | `candidate_semantic_scores(fb, candidates)` | FeatureBank、候选上下文 | 计算 `<q,r>`、`<q,src>`、`<q,dst>`、new text node score。 |
| `transition_features.py` | `TransitionFeatureBuilder(dde_dim, hidden_dim)` | 配置 transition type 编码器；`dde_dim` 只作构造兼容。 |
| `transition_features.py` | `TransitionFeatureBuilder.forward(fb, context, batch, state, candidate_edge_ids, candidate_context)` | 当前上下文和候选边 | 输出 3 维 transition type 和 semantic prior 诊断分量。 |
| `action_features.py` | `ActionFeatureBuilder/ActionFeatureOutput` | 兼容别名 | 指向 `TransitionFeatureBuilder/TransitionFeatureOutput`。 |
| `edge_scorer.py` | `EdgeScorer(...)` | hidden dim、edge encoder、transition dim、semantic/residual 参数 | edge logit scorer。 |
| `edge_scorer.py` | `EdgeScorer.forward(fb, context, edge_index, edge_batch_index, active_nodes, candidate_edge_ids, candidate_context, transition_features, return_breakdown)` | FeatureBank、StateContext、候选边 | 输出 final logits 或 breakdown。 |
| `edge_scorer.py` | `_edge_h(fb, context, candidates)` | FeatureBank、StateContext、候选 | 复用缓存或编码候选边。 |
| `edge_scorer.py` | `_cached_frontier_edge_h(context, candidates, device, dtype)` | StateContext、候选 | 若 ids 完全一致，返回缓存 edge_h。 |
| `edge_scorer.py` | `_residual_logits_from_edge_h(...)` | state_h、edge_h、transition type、detached semantic logits | 计算 residual logits。 |
| `edge_scorer.py` | `effective_residual_scale(device, dtype)` | 可选设备/dtype | 当前 residual 有效 scale。 |
| `edge_scorer.py` | `update_residual_schedule(step)` | global step | 更新 residual warmup multiplier 和 requires_grad。 |
| `edge_scorer.py` | `_scheduled_residual_multiplier(step)` | step | 线性 warmup multiplier。 |
| `stop_gate.py` | `StopExpandGate(...)` | hidden dim、bias、progress/frontier summary 开关 | Stop/Expand option scorer。 |
| `stop_gate.py` | `reset_parameters()` | 无 | zero-init stop net 最后一层并重置 bias。 |
| `stop_gate.py` | `forward(state_h, edge_logits, edge_batch_index, num_graphs, progress_ratio, frontier_summary, edge_logmeanexp, edge_max, edge_log_size, has_candidate_edge)` | state 和 edge logits | 输出 stop/expand logits。 |
| `stop_gate.py` | `_expand_logit(edge_logits, edge_batch_index, batch_size, device, dtype)` | edge logits 和分段 ids | 每图 `scatter_logsumexp(edge_logits)`，无候选为 `-inf`。 |
| `stop_gate.py` | `_resolve_frontier_summary(...)` | summary 或三条向量 | 统一得到 max/logmeanexp/log_size。 |
| `stop_gate.py` | `_vector_or_zeros(value, batch_size, device, dtype)` | 可选向量 | 缺省时补零。 |
| `flow_head.py` | `FlowHead(hidden_dim, num_layers, dropout, zero_init)` | head 配置 | state log-flow MLP。 |
| `flow_head.py` | `FlowHead.forward(state_h)` | `[B,H]` state 表征 | 输出 `[B]` log flow。 |
| `policy.py` | `Policy(...)` | feature/readout/action/stop/edge/flow cfg | 组装在线 policy。 |
| `policy.py` | `prepare_rollout_context(batch)` | batch | 预先计算静态 `FeatureBank`。 |
| `policy.py` | `forward(batch, state, rollout_context, return_edge_breakdown, edge_logit_mode)` | batch、state、可选 FeatureBank | 单步 policy 输出 flow、stop/expand、edge logits。 |
| `policy.py` | `_validate_feature_bank(fb, batch, num_graphs)` | FeatureBank、batch | shape 校验。 |
| `policy.py` | `frontier_logit_summary(edge_logits, edge_batch, num_graphs, device)` | candidate logits 和 graph ids | 每图 edge max/logmeanexp/sharpness/log_size。 |

### 5.8 测试锚点

| 测试文件 | 覆盖点 |
| --- | --- |
| `tests/test_semantic_model_space_features.py` | DDE 方向性、semantic/model space 分离、非文本 mask 推导、transition feature 固定、shared edge encoder、semantic prior residual、frontier 缓存。 |
| `tests/test_non_text_entity_embeddings.py` | 非文本 entity 不进入文本 embedding 表、text encode cache 命中、共享 non-text embedding 兜底。 |
| `tests/test_entity_embedding_map_compat.py` | `entity_text_embedding_ids -> entity_embedding_map` 兼容、embedding L2 normalize 校验。 |
| `tests/test_graph_path_labels.py` | target shortest suffix count、shortest-path edge mask、多 anchor/target 顺序和 unreachable target 空标签。 |
| `tests/test_vectorized_online_rollouts.py` / `tests/test_gflownet_rollout_split.py` | 物理重复 rollout 与 static/fused rollout 路径的一致性和 batch 语义。 |

## 6. 效率分析

### 6.1 做得比较高效的部分

- 文本编码离线完成，在线只查 embedding tensor；`encode_text_features` 还有磁盘缓存，重复预处理时能跳过 Transformer。
- `FeatureBank` 明确区分 semantic space 和 model space，避免每个 policy head 重复投影。
- `StateReadout` 生成 frontier 时顺手缓存 `frontier_edge_h`，`EdgeScorer` 可直接复用。
- `TransitionFeatureBuilder` 把 semantic prior 诊断分量放进 `TransitionFeatureOutput`，`EdgeScorer`
  不再重复算 `<q,r>` 和 new-node score；但这些分量不进入 residual feature 向量。
- shortest suffix count 已是按 target distance bucket 的 DP，不是从每个节点枚举路径。
- shortest path edge mask 使用 target/edge chunk 张量判断，避免 Python 三重循环。

### 6.2 主要低效点

1. 离线 target labels 的 BFS 成本偏高。
   `compute_target_path_labels` 会先对每个 anchor 做正向 BFS，再对每个 reachable target 做反向 BFS。
   复杂度约为 `O(A*(V+E) + T*(V+E))`，edge mask 还带 `T*E*A` 级别的张量判断。
   WebQSP 局部图可接受，target 很多或图很大时会慢。

2. 默认多 rollout 仍物理复制 batch。
   `configs/model/weaver.yaml` 里 `use_static_batch_rollouts=false`、
   `use_fused_static_batch_rollouts=false`。因此 `RolloutEngine` 默认调用
   `repeat_retrieval_batch`，然后对重复后的大 batch 计算 `FeatureBank`。
   这简单可靠，但显存和静态特征计算约随 rollout chunk 线性增长。

3. 每步 frontier 扫描全图边。
   `StateReadout._frontier_readout` 和 `frontier_edges` 都是：

   ```text
   frontier = (active[src] or active[dst]) and not active_edge
   ```

   每步 `O(E)`。`expand_budget=3` 时问题不大；预算或图规模扩大后，应考虑增量 frontier set。

4. fused static rollout 的 frontier mask 是 `[R,E]`。
   `RolloutState` 避免复制静态 `FeatureBank`，但 `_frontier_rollout_readout` 会构造
   dynamic row 与所有边的归属/incident mask。R 或 E 很大时需要关注峰值显存。

5. text cache 是整表粒度。
   cache key 包含完整文本列表。任何 question 顺序或内容变化都会让 questions cache 整体 miss。
   对稳定数据集这是合理的；对频繁增量数据，per-text cache 会更省。

## 7. 逻辑合理性和风险

### 7.1 合理点

- 策略特征没有 label leakage：`FeatureEncoder`、`StateReadout`、`TransitionFeatureBuilder`、
  `EdgeScorer` 都不读取 `target_*` 字段。
- `anchor_*` 和 `question_*` 语义分层清楚：离线 `question_entities` grounding 后变成在线
  `anchor_node_ids`。
- root edges 语义清楚：anchor-anchor edges 初始就是 active evidence，不计入 learned expansion budget。
- `State` 的 canonical invariant 是 `V_s = anchors union endpoints(E_s)`，executor 负责校验和变更。
- non-text entity 用 `entity_embedding_map=-1` 兜底，避免旧方案里无文本 row 的非法查表。
- `EdgeScorer` 的 residual warmup 和 zero-init 与 semantic prior 的设计目标一致。

### 7.2 需要注意的点

- `anchor_node_forward_distances_flat` 和 `anchor_node_backward_distances_flat` 仍被物化和读取，
  但当前在线特征没有直接使用它们；DDE 已经取代旧的 distance embedding。保留它们会增加
  artifact 体积，但给诊断/兼容留了余地。
- `repeat_retrieval_batch` 未重复 `target_shortest_path_count_flat`。当前 reward/eval/主 policy
  不依赖它，`rollout_diagnostics` 也主要用 distance 和 edge mask；但 legacy coverage proposal
  读取该字段。如果启用 legacy teacher 并走物理重复 batch，需要补上这个字段。
- 非文本实体全部共享一个 learnable embedding。若结构位置、关系路径也相似，模型很难区分不同
  non-text entity；应先用错误样本验证是否成为瓶颈，再考虑 per-entity/hash/type embedding。
- `candidate_semantic_scores` 仍会计算 `<q,src>`、`<q,dst>` 作为 new-text bonus 的中间量和诊断分量，
  但 non-text node dot product 不再进入 residual feature。

## 8. 优化建议优先级

1. 优先评估并打开 `use_static_batch_rollouts` 或 `use_fused_static_batch_rollouts`。
   目标是静态 `FeatureBank` 只算一次，rollout 维度只复制动态 `State`。需要用现有测试和指标确认
   与物理重复路径数值一致。

2. 给 frontier 做增量维护。
   当前每步全边扫描对小预算可接受；如果后续预算、图规模或 rollout 数上升，增量 frontier 是最直接的
   在线加速点。

3. 补齐 `repeat_retrieval_batch` 的 label 字段一致性。
   至少把 `target_shortest_path_count_flat` 加入重复字段，避免未来 teacher/legacy 路径踩坑。

4. 对 target label 计算做批处理或按需计算。
   如果训练不再需要 shortest suffix count 或 shortest path edge mask，可以把 heavy labels 变成可配置输出。

5. 继续用 prior/final rank 诊断约束 residual。
   重点看 `edge/residual_to_base_std_ratio` 和 answer-edge final rank 是否系统性劣于 prior rank。

## 9. 最终判断

当前特征工程不是答案标签驱动的 policy，而是：

```text
semantic prior
+ anchor-conditioned DDE
+ current subgraph state readout
+ action/state residual correction
+ frontier-aware Stop/Expand
```

逻辑上是自洽的，短期维护重点是保持 semantic prior 和 residual 的职责边界，并监控 final rank
相对 prior rank 的变化。真正值得投入的工程优化是在线效率：减少 repeated batch 的静态特征复制，
以及把 frontier 从每步全图扫描改成增量维护。
