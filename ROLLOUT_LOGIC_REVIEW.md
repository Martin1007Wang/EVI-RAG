# Weaver Rollout 逻辑审阅

更新时间：2026-05-01

本文基于当前工作区代码梳理 Weaver/GFlowNet rollout 的实现。已有
`FEATURE_ENGINEERING_REVIEW.md` 偏重特征工程，本文偏重 rollout 执行链路、
跨文件调用关系、命名、效率和结构合理性。

## 总体结论

当前 rollout 主线是清楚且基本合理的：

- 静态 batch 特征通过 `Policy.prepare_rollout_context -> FeatureEncoder` 每个
  rollout chunk 计算一次，形成 `FeatureBank`。
- 动态 Markov 状态由静态 batch 图结构加当前 active node/edge mask 表示；写入
  rollout 结果后，`selected_edge_ids` 是重建终态子图的动态句柄。
- `Policy` 只做神经策略和 flow 估计，不直接看 reward、target shortest-path
  labels 或 teacher 结果。
- `RolloutEngine` 负责逐步调用 policy、reward、diagnostics、auxiliary 和
  executor。
- `StepExecutor` 只做动作合法性、采样、状态转移、终止 reward 写入和 backward
  log-prob。
- `SubTrajectoryBalanceLoss` 只消费 `RolloutBatch`，不重新执行环境逻辑。

主要问题和风险：

1. `WeaverModule.generate_subgraph_masks` 调用
   `self.rollout_runner.generate_online_rollouts(...)`，但当前
   `RolloutRunner` 没有这个方法，在线 forward/inference 路径会运行时报错。
2. 多 rollout 通过 `repeat_retrieval_batch` 物理复制整批图和标签，简单可靠，
   但显存和静态特征计算随 rollout 数线性增长。
3. 每个 step 至少两次扫描全图边找 frontier：`StateReadout._frontier_readout`
   和 `state_ops.frontier_edges`。虽然 `EdgeScorer` 复用了 frontier edge
   embedding，但 frontier mask 本身仍重复构造。
4. 若不需要 StopTB/StopAdv/stop counterfactual，`RolloutEngine` 仍会每步先评估
   `stop_now_reward`，存在可避免开销。
5. backward removal、StopAdv one-step child reward 和 expand edge sampling 中有小
   规模 Python loop。当前 `expand_budget`、`topk` 很小，逻辑可接受；大 batch 或大
   budget 时会成为 GPU 同步瓶颈。

## 当前特征如何进入 rollout

### 静态特征

`FeatureEncoder.forward` 生成 `FeatureBank`：

- `node_sem_h`：实体语义空间表示。文本实体查表，非文本实体使用共享 learnable
  embedding。
- `rel_sem_h`：关系语义空间表示。
- `query_sem_h`：问题 embedding。
- `node_h`、`rel_h`、`query_h`：通过 role projection 到模型空间。
- `node_dde`：anchor-conditioned directional diffusion encoding。
- `node_log_degree`：节点静态 log-degree。
- `edge_relation_log_frequency`：每图内关系频次。
- `node_is_non_text`：供 new text node semantic score 使用。

这些特征与 rollout 当前状态无关，因此由
`Policy.prepare_rollout_context(batch)` 在 chunk 开始时缓存。

### 动态状态

`State` 保存：

- `active_nodes`：当前子图节点 mask。
- `active_edges`：当前子图边 mask，包含 root edges。
- `root_edges`：anchor-anchor induced 初始边，不计入 learned expansion budget。
- `expand_budget`：每图最大 learned non-root expansion 数。

`State.create_initial` 以 `anchor_node_ids` 初始化 active nodes，并把
`build_anchor_induced_edge_mask` 得到的 anchor-anchor edges 作为 root edges。

### 每步候选边特征

`Policy.forward` 每步做：

1. `StateReadout.forward` 读 active subgraph，得到 `state_h`、relation path memory、
   frontier summary 和 cached frontier edge embeddings。
2. `frontier_edges` 枚举合法 Expand 候选边：
   `e not in E_s and (src in V_s or dst in V_s)`。
3. `build_candidate_context` 缓存候选边的 src/dst、graph id、端点 active 状态。
4. `ActionFeatureBuilder.forward` 构造候选边特征：
   - 端点 active/new 几何状态。
   - src/dst DDE、差分、乘积。
   - query-relation/query-src/query-dst semantic weak features。
   - relation path history score。
   - src/dst degree、relation frequency、frontier size。
   - progress 和端点 status。
5. `EdgeScorer.forward` 用 semantic prior 加 residual 得到 candidate edge logits。
6. `StopExpandGate.forward` 输出每图 Stop/Expand logits，其中 Expand logit 是该图
   frontier edge logits 的 logsumexp。

策略侧没有读取 `target_node_ids`、`target_shortest_path_*` 或
`node_target_distance`。这些 target-derived 字段只服务 reward、teacher diagnostics
和 eval。

## 跨文件调用链

### 训练主链路

```text
src/train.py
  -> WeaverModule.training_step
      -> policy.edge_scorer.update_residual_schedule
      -> temperature_schedule.current
      -> RolloutRunner.run_training_rollouts_and_backward
          -> rollout_chunk_sizes
          -> RolloutRunner._generate_chunk
              -> RolloutEngine.run_vectorized
                  -> repeat_retrieval_batch              # K > 1
                  -> Policy.prepare_rollout_context
                      -> FeatureEncoder.forward
                  -> RolloutEngine._run_one_batch
                      -> State.create_initial
                      -> RolloutBuffer
                      -> StepExecutor
                      -> for t in range(expand_budget + 1)
                          -> RewardModel.evaluate_terminal_state
                          -> Policy.forward
                              -> StateReadout.forward
                              -> FlowHead.forward
                              -> frontier_edges
                              -> build_candidate_context
                              -> ActionFeatureBuilder.forward
                              -> EdgeScorer.forward
                              -> frontier_logit_summary
                              -> StopExpandGate.forward
                          -> RolloutEngine._write_flow
                          -> RolloutBuffer.write_stop_counterfactual
                          -> write_policy_diagnostics
                          -> StopAdvantageAuxiliary.write_step   # optional
                          -> StepExecutor.execute_step
                              -> validate_policy_output
                              -> has_candidate
                              -> budget_exhausted_mask
                              -> sample_policy_actions
                              -> State.apply_expansion or _stop
                              -> UniformRemovalBackwardPolicy.log_prob_after_continue
                          -> RolloutBuffer.write_step
                      -> RolloutBatch.from_buffer
                  -> split_repeated_rollout_batch         # K > 1
          -> backward_rollouts
              -> concat_rollout_batches
              -> SubTrajectoryBalanceLoss.forward
              -> manual_backward
      -> TrainingDiagnosticsCollector.collect
```

### Eval 主链路

```text
WeaverModule.eval_step
  -> RolloutRunner.generate_eval_rollouts
      -> RolloutRunner.generate_rollouts
          -> RolloutEngine.run_vectorized
  -> evaluate_rollouts
      -> compute_expected_node_retrieval_quality
      -> compute_best_of_k_node_retrieval_quality
      -> compute_compactness_expectations
      -> compute_exploration_diversity_at_ks
      -> compute_stop_behavior_diagnostics
      -> compute_policy_behavior_diagnostics
      -> compute_stop_counterfactual_diagnostics     # debug/val
      -> compute_teacher_edge_diagnostics            # debug
```

### 在线生成子图路径

```text
WeaverModule.forward
  -> WeaverModule.generate_subgraph_masks
      -> self.rollout_runner.generate_online_rollouts   # 当前缺失
      -> compute_union_subgraph_masks
```

这里是当前明确的结构缺口：应该改成已有的
`RolloutRunner.generate_rollouts(...)`，或给 `RolloutRunner` 增加
`generate_online_rollouts` 别名并补测试。

## 代码文件和函数职责

### Runtime 入口和配置

| 文件 | 名称 | 职责 |
| --- | --- | --- |
| `src/weaver/module.py` | `WeaverModule.__init__` | 组装 `Policy`、`RewardModel`、loss、temperature schedule、`RolloutRunner`、StopAdv auxiliary。 |
| `src/weaver/module.py` | `training_step` | 训练 step：更新 residual schedule、生成 rollout、反传、优化器 step、记录指标。 |
| `src/weaver/module.py` | `forward` | 调用 `generate_subgraph_masks` 的推理入口。 |
| `src/weaver/module.py` | `generate_subgraph_masks` | 生成多个 rollout 并取 union terminal subgraph；当前调用了缺失方法。 |
| `src/weaver/module.py` | `eval_step` | eval/test 入口，生成 rollout 后调用 `evaluate_rollouts`。 |
| `src/weaver/module.py` | `log_training_step` | 汇总 loss、rollout diagnostics、调度和优化器指标。 |
| `src/weaver/config.py` | `build_policy_runtime_config` | 标准化 policy config 并注入 embedding tensors。 |
| `src/weaver/config.py` | `build_rollout_runtime_config` | 标准化 rollout budget、rollout count、chunk size、StopAdv config。 |
| `src/weaver/config.py` | `build_eval_runtime_config` | 标准化 eval budgets 和检索指标开关。 |
| `src/weaver/config.py` | `build_schedule_runtime_config` | 标准化采样温度配置。 |
| `src/weaver/config.py` | `build_diagnostics_runtime_config` | 标准化训练/eval rollout diagnostics 开关。 |

### `src/weaver/rollout/runner.py`

| 名称 | 职责 |
| --- | --- |
| `TrainingRolloutResult` | 训练 rollout 的 loss output 和 rollout batch 集合。 |
| `RolloutRunConfig` | 单次 rollout run 的 temperature、diagnostics、depth validation、edge logit mode。 |
| `RolloutRunner` | 分 chunk 调度 rollout，训练时负责每 chunk loss 和 backward。 |
| `RolloutRunner.run_training_rollouts_and_backward` | 训练路径：按 `train_chunk_size` 生成 rollout，调用 loss 并按完整 rollout 数归一化反传。 |
| `RolloutRunner.generate_eval_rollouts` | eval 路径入口，默认使用 `eval_num_rollout` 和 `eval_chunk_size`。 |
| `RolloutRunner.generate_rollouts` | 通用 no-grad rollout 生成。 |
| `RolloutRunner._generate_chunk` | 调用 `RolloutEngine.run_vectorized`。 |
| `backward_rollouts` | concat rollout chunk，计算 loss，按 `len(chunk)/normalize_by` 缩放后 backward。 |
| `concat_rollout_batches` | 合并多个 `RolloutBatch`。 |
| `concat_rollout_stats` | 合并 trajectory-level stats。 |
| `concat_rollout_traces` | 合并 step-level traces。 |
| `_cat` | 拼接必选 tensor 字段。 |
| `_cat_optional` | 拼接可选 tensor 字段，并禁止部分存在部分缺失。 |
| `rollout_chunk_sizes` | 生成 chunk size 序列。 |
| `positive_int` | 正整数校验。 |
| `non_negative_int` | 非负整数校验。 |

命名评价：`Runner` 这个层级划分合理，明确高于 `Engine`。缺少
`generate_online_rollouts` 是当前最大命名/API 不一致。

### `src/weaver/rollout/engine.py`

| 名称 | 职责 |
| --- | --- |
| `StepAuxiliary` | 可插拔 step-level auxiliary writer 协议，例如 StopAdv。 |
| `RolloutEngineConfig` | 引擎级执行配置。 |
| `RolloutEngine` | 有限 horizon 的向量化 rollout driver。 |
| `RolloutEngine.run_vectorized` | K 个 rollout 的入口；K>1 时物理 repeat batch，执行后 split 回逻辑 rollout。 |
| `RolloutEngine._run_one_batch` | 逐步执行一个物理 batched rollout。 |
| `RolloutEngine._evaluate_stop_now` | 当前状态直接停止的 terminal reward。 |
| `RolloutEngine._assert_synchronous_depth` | debug 校验未终止图的 expansion depth 一致。 |
| `RolloutEngine._write_flow` | 写入 `log F(s_t | q)`，t=0 也作为 root `log Z(q)`。 |

命名评价：`Engine` 职责清楚。`edge_logit_mode="semantic"` 已支持但 runner 没暴露，
更像诊断开关。`current_reward` 传给 StopAdv 时实际含义是 `stop_now_reward`，
可以考虑改名。

### `src/weaver/rollout/executor.py`

| 名称 | 职责 |
| --- | --- |
| `StepGraphContext` | 缓存物理 batched graph 的 edge/node/graph 坐标、anchor mask、root edges。 |
| `StepGraphContext.from_batch` | 从 `RetrievalBatch` 构造 executor 所需图上下文。 |
| `BackwardPolicy` | backward log-prob 协议。 |
| `UniformRemovalBackwardPolicy` | 对所有可逆 non-root removable edges 均匀分布。 |
| `UniformRemovalBackwardPolicy.log_prob_after_continue` | Continue 后计算 `-log |R(child)|`。 |
| `StepExecutor` | 单步环境转移，不调用 policy。 |
| `StepExecutor.execute_step` | 校验 policy output、mask 非法 Expand、采样、更新 state、写 terminal reward。 |
| `StepExecutor._continue_with_edge` | 对 Continue 图添加 chosen edge，更新 `log_pb`。 |
| `StepExecutor._stop` | 对 Stop 图写 terminal reward。 |
| `StepExecutor._write_terminal_reward` | 把 `TerminalRewardOutput` 指定 graph 拷入 step tensor。 |
| `StepExecutor._validate_state_root` | 校验 state root edges 与 batch anchor-induced root 一致。 |
| `TerminalStepTensors` | 单步 terminal reward 临时 tensor 容器。 |
| `TerminalStepTensors.zeros` | 创建全零 terminal step tensors。 |
| `has_candidate` | 按图判断是否有 candidate edge。 |
| `budget_exhausted_mask` | 按图判断 expansion budget 是否耗尽。 |
| `validate_policy_output` | 校验 `PolicyOutput` shape/device。 |
| `validate_candidate_tensors` | 校验 candidate edge logits/ids/batch ids。 |
| `validate_frontier_candidates` | debug 校验候选边确实是 frontier。 |
| `validate_terminal_reward` | 校验 terminal reward 输出 shape。 |
| `_validate_step_vector` | 单步 `[B]` tensor 校验。 |
| `_validate_node_ids` | node id 范围校验。 |
| `_validate_edge_ids` | edge id 范围校验。 |
| `_validate_graph_ids` | graph id 范围校验。 |

命名评价：`Executor` 层抽得合理。`validate_frontier` 默认关闭，依赖 policy 正确生成
candidate。`State.apply_expansion` 本身会忽略非法 edge，这和 repo 偏好的 fail-fast
略不一致，不过 executor 前面已有 policy output 校验。

### `src/weaver/rollout/sampling.py`

| 名称 | 职责 |
| --- | --- |
| `CONTINUE_ACTION` / `STOP_ACTION` | action type 编码。 |
| `STOP_OPTION_INDEX` / `EXPAND_OPTION_INDEX` | Stop/Expand option softmax 列索引。 |
| `ActionSample` | 每图一次采样结果：action type、chosen edge、target log-prob。 |
| `sample_policy_actions` | 用 temperature 行为分布采样动作，同时返回未加温 target policy log-prob。 |
| `option_action_log_probs` | 计算 option log-prob 和 conditional edge log-prob。不能 expand 时强制 Stop。 |
| `option_action_probs` | log-prob 的概率版本，用于 diagnostics。 |
| `_sample_expand_edges` | 对采到 Expand 的图逐图采一个 edge。 |
| `_validate_inputs` | 采样入口校验。 |
| `_validate_option_inputs` | option 概率入口校验。 |
| `_validate_candidate_tensors` | candidate tensor 校验。 |
| `_validate_graph_ids` | graph id 范围校验。 |

命名评价：Stop/Expand option 和 edge conditional policy 的分层命名清楚。
`_sample_expand_edges` 用 per-graph loop，可读但不是最强性能实现。

### `src/weaver/rollout/buffer.py`

| 名称 | 职责 |
| --- | --- |
| `RolloutBuffer` | mutable rollout 存储，最终转为 immutable `RolloutBatch`。 |
| `__post_init__` | 初始化所有 `[B]` 和 `[B,T]` tensor。 |
| `active` | 当前未终止图 mask。 |
| `write_state_log_flow` | 写 `log F(s_t | q)`。 |
| `write_step` | 写 step action、log_pf/log_pb、selected edge、stop/continue mask，并处理终止。 |
| `write_stop_counterfactual` | 写 stop-now reward/F1。 |
| `write_policy_step_diagnostics` | 写 Stop/Expand 概率、entropy、budget exhausted 等 diagnostics。 |
| `write_stop_advantage` | 写 StopAdv target、valid mask 和 continue log reward。 |
| `finalize_unfinished` | horizon 结束时强制终止的备用方法，目前主链路未使用。 |
| `_write_terminal_from_step` | 从 `StepResult` 写 trajectory-level terminal stats。 |
| `_check_t` | step index 校验。 |
| `_as_float` / `_as_long` / `_as_bool` | dtype/device/shape 规范化。 |
| `_zeros_float` / `_zeros_long` / `_zeros_bool` | tensor 创建 helper。 |

结构评价：buffer 把所有 rollout trace 写入集中化，便于 loss/eval。字段很多，但
`RolloutTraces` 与 loss/diagnostics 的依赖相对明确。

### `src/weaver/rollout/schema.py`

| 名称 | 职责 |
| --- | --- |
| `StepResult` | executor 单步输出，shape 全为 `[B]`。 |
| `RolloutTraces` | step-level traces，shape 主要为 `[B,T]`。 |
| `RolloutTraces.__post_init__` | 若未提供 `stop_adv_loss`，从 target 和 stop prob 计算诊断 loss。 |
| `RolloutTraces._compute_stop_adv_loss` | backward-compatible StopAdv diagnostic loss。 |
| `RolloutStats` | trajectory-level stats，shape 主要为 `[B]`。 |
| `RolloutBatch` | loss/eval 消费的 immutable rollout 结果。 |
| `RolloutBatch.from_buffer` | 从 `RolloutBuffer` 构造 `RolloutBatch`。 |

命名评价：`Stats` 和 `Traces` 的粒度好。`stop_adv_loss` 放在 schema 里只作为诊断，
真正训练 loss 在 `loss.py`，这个注释已写清楚。

### `src/weaver/rollout/batch_ops.py`

| 名称 | 职责 |
| --- | --- |
| `split_repeated_rollout_batch` | 把物理重复 batch 的 rollout 切回 K 个逻辑 rollout，并把 edge id 映回原 batch。 |
| `_slice_rollout_batch` | 切一个 repeat slice。 |
| `_slice_stats` | 切 stats。 |
| `_slice_traces` | 切 traces，并处理 selected edge id offset。 |
| `_unrepeat_edge_ids` | repeated edge id 减 offset，Stop 的 -1 保持不变。 |
| `_slice_optional` | 切可选 tensor。 |
| `_validate_rollout_first_dim` | 校验 rollout 第一维是否等于 `B * repeats`。 |
| `_validate_first_dim` | 单字段第一维校验。 |
| `_num_graphs` / `_num_edges` | 从原 batch 取图数和边数。 |

效率评价：实现简单。`_unrepeat_edge_ids` 没校验 edge id 是否落在当前 repeat slice，
如果上游出错可能延后暴露。

### `src/weaver/rollout/diagnostics.py`

| 名称 | 职责 |
| --- | --- |
| `write_policy_diagnostics` | 写 target Stop/Expand prob、StopTB valid mask、edge entropy、budget exhausted。 |
| `edge_entropy_by_graph` | 计算每图 conditional edge entropy。 |

命名评价：清楚。`has_candidate`、`budget_exhausted_mask` 在 executor、diagnostics、
StopAdv 多处重复调用，可考虑在 engine step 级缓存。

### `src/weaver/rollout/stop_advantage.py`

| 名称 | 职责 |
| --- | --- |
| `StopAdvantageConfig` | StopAdv auxiliary 配置。 |
| `StopAdvantageConfig.from_dict` | 解析/校验配置。 |
| `StopAdvantageAuxiliary` | 训练时写 optimal-stopping 软标签。 |
| `StopAdvantageAuxiliary.write_step` | 比较 stop-now reward 和候选 one-step child reward，写 target。 |
| `StopAdvantageAuxiliary._write_empty` | 无有效 candidate 时写空 target。 |
| `StopAdvantageAuxiliary._select_candidate_positions` | 按 semantic top-k、final top-k、random 选择候选 child。 |
| `OneStepChildReward` | one-step child reward 输出。 |
| `evaluate_one_step_child_rewards` | 快速估计 `s + e` 的 stop reward。 |
| `pool_continue_value` | 对 child rewards 做 max/softmax/logmeanexp pooling。 |
| `_topk_positions` | 每图 top-k candidate position。 |
| `_random_positions` | 每图随机 candidate position。 |
| `_node_mask` | node id 转 mask。 |
| `_connected_nodes_from_anchors` | Python BFS/DFS 计算 anchor-connected nodes。 |
| `_non_negative_int` / `_positive_float` | 配置校验。 |

效率评价：StopAdv 是训练辅助，候选量上限较小，因此 Python loop 可以接受。但
`evaluate_one_step_child_rewards` 会逐 candidate 做 connected set，GPU 上会有同步；
如果 top-k 或 batch 扩大，这里应优先重构。

### `src/weaver/rollout/terminal_subgraph.py`

| 名称 | 职责 |
| --- | --- |
| `UnionSubgraphMasks` | 多 rollout terminal subgraph union mask。 |
| `default_eval_device` | eval mask 默认放 CPU。 |
| `batch_num_graphs` | 取 batch graph 数。 |
| `node_mask_from_ids` | id 转 node mask，并做范围校验。 |
| `anchor_node_mask` | 构造 anchor mask。 |
| `eval_target_node_mask` | 选择 reachable targets 或 all targets 的 eval target mask。 |
| `root_edge_mask` | 构造 anchor-induced root edge mask。 |
| `terminal_subgraph_mask` | 用 root state 和 `selected_edge_ids` 重建一个 rollout 的 terminal subgraph。 |
| `stack_terminal_subgraph_masks` | stack 多 rollout terminal masks。 |
| `compute_union_subgraph_masks` | 多 rollout terminal masks 求 union。 |
| `_check_id_range` | id 范围校验。 |

结构评价：eval 子图重建与训练环境解耦合理。`root_edge_mask` 与 `reward.py` 有重复概念，
未来可集中到 `graph.ops`。

## Policy、State 和特征文件

| 文件 | 名称 | 职责 |
| --- | --- | --- |
| `src/weaver/state.py` | `State` | 当前 batched subgraph state。 |
| `src/weaver/state.py` | `State.create_initial` | 从 anchors/root edges 初始化。 |
| `src/weaver/state.py` | `State.detach` | 给 policy/reward snapshot，避免 executor 后续 mutation 影响当前 step。 |
| `src/weaver/state.py` | `State.apply_expansion` | 添加 chosen edges 和 endpoints。 |
| `src/weaver/state.py` | `expanded_edge_count_per_graph` | 每图 learned non-root edge 数。 |
| `src/weaver/state.py` | `remaining_budget_per_graph` | 每图剩余 expansion budget。 |
| `src/weaver/state.py` | `expand_ratio_per_graph` | 每图 budget 使用比例。 |
| `src/weaver/state.py` | `synchronous_rollout_depth` | debug 校验同步 depth。 |
| `src/weaver/state_ops.py` | `frontier_edges` | 枚举合法 Expand candidate。 |
| `src/weaver/state_ops.py` | `has_frontier_edge_per_graph` | 每图是否存在 frontier edge。 |
| `src/weaver/policy.py` | `PolicyOutput` | 单步 policy 输出。 |
| `src/weaver/policy.py` | `Policy.prepare_rollout_context` | 计算静态 `FeatureBank`。 |
| `src/weaver/policy.py` | `Policy.forward` | 单步 state flow、Stop/Expand logits、candidate edge logits。 |
| `src/weaver/policy.py` | `Policy._validate_feature_bank` | 校验 `FeatureBank` 与 batch 维度一致。 |
| `src/weaver/policy.py` | `FrontierLogitSummary` | per-graph frontier logit summary。 |
| `src/weaver/policy.py` | `frontier_logit_summary` | 计算 edge max、logmeanexp、sharpness、log size。 |
| `src/weaver/nn/feature_encoder.py` | `EntityEmbeddingLayer` | entity catalog id 到 semantic embedding。 |
| `src/weaver/nn/feature_encoder.py` | `RoleProjection` | semantic space 到 model space。 |
| `src/weaver/nn/feature_encoder.py` | `FeatureBank` | 静态 query/node/relation 特征容器。 |
| `src/weaver/nn/feature_encoder.py` | `FeatureEncoder.forward` | 构造 `FeatureBank`。 |
| `src/weaver/nn/dde.py` | `DirectionalDDE.forward` | anchor indicator 加正/反向 diffusion 坐标。 |
| `src/weaver/nn/static_graph_features.py` | `node_log_degree` | 静态 log-degree。 |
| `src/weaver/nn/static_graph_features.py` | `edge_relation_log_frequency` | 每图 relation log-frequency。 |
| `src/weaver/nn/state_readout.py` | `StateContext` | state readout 和下游复用特征。 |
| `src/weaver/nn/state_readout.py` | `StateReadout.forward` | active nodes/edges/relation/frontier summary readout。 |
| `src/weaver/nn/state_readout.py` | `_pool_nodes` / `_pool_edges` / `_pool_relations` | query attention pooling。 |
| `src/weaver/nn/state_readout.py` | `_frontier_readout` | frontier summary 和 cached frontier edge embeddings。 |
| `src/weaver/nn/candidate_context.py` | `CandidateContext` | 候选边坐标和端点状态。 |
| `src/weaver/nn/candidate_context.py` | `build_candidate_context` | 从 batch/state/candidate ids 构造 context。 |
| `src/weaver/nn/candidate_context.py` | `candidate_semantic_scores` | 计算 semantic prior 所需 scores。 |
| `src/weaver/nn/action_features.py` | `ActionFeatureBuilder.forward` | 构造 candidate edge action features。 |
| `src/weaver/nn/action_features.py` | `_resolve_node_log_degree` | 取缓存 degree 或现场计算。 |
| `src/weaver/nn/action_features.py` | `_resolve_edge_relation_log_frequency` | 取缓存 relation frequency 或现场计算。 |
| `src/weaver/nn/action_features.py` | `_candidate_frontier_log_size` | 每候选边对应图的 frontier size。 |
| `src/weaver/nn/edge_encoder.py` | `EdgeEncoder.forward` | `W_E [h_src, h_rel, h_dst]`。 |
| `src/weaver/nn/edge_scorer.py` | `EdgeScorer.forward` | semantic prior residual edge logit。 |
| `src/weaver/nn/edge_scorer.py` | `_edge_h` / `_cached_frontier_edge_h` | 复用 state readout 的 frontier edge embedding。 |
| `src/weaver/nn/edge_scorer.py` | `_residual_logits_from_edge_h` | residual MLP。 |
| `src/weaver/nn/edge_scorer.py` | `update_residual_schedule` | residual warmup/freezing 调度。 |
| `src/weaver/nn/stop_gate.py` | `StopExpandGate.forward` | Stop logit 和 Expand option logit。 |
| `src/weaver/nn/stop_gate.py` | `_expand_logit` | 每图 frontier edge logits 的 logsumexp。 |
| `src/weaver/nn/flow_head.py` | `FlowHead.forward` | 估计 `log F(s | q)`。 |

## Reward、Loss、Eval 和数据支持文件

| 文件 | 关键名称 | rollout 中的作用 |
| --- | --- | --- |
| `src/weaver/reward.py` | `RewardModel.evaluate_terminal_state` | 对当前 active subgraph 算 terminal reward、answer stats、support、compactness。 |
| `src/weaver/reward.py` | `root_edge_mask`、`answer_stats`、`anchor_answer_support`、`compactness_stats` | terminal reward 的内部组成。 |
| `src/weaver/loss.py` | `SubTrajectoryBalanceLoss.forward` | 从 `RolloutBatch` 计算 SubTB、StopTB、StopAdv loss。 |
| `src/weaver/loss.py` | `_subtrajectory_mask`、`_segment_sums`、`_subtrajectory_targets` | SubTB 目标和 segment 累积。 |
| `src/weaver/loss.py` | `stop_now_tb_loss`、`stop_advantage_loss` | Stop option 相关辅助 loss。 |
| `src/training/rollout_eval.py` | `evaluate_rollouts` | eval/test rollout 指标总入口。 |
| `src/training/rollout_diagnostics.py` | `compute_policy_behavior_diagnostics` | Stop/Expand prob、edge entropy 等策略行为诊断。 |
| `src/training/rollout_diagnostics.py` | `compute_stop_counterfactual_diagnostics` | 比较继续后最终 reward 和当时 stop-now reward。 |
| `src/training/rollout_diagnostics.py` | `compute_teacher_edge_diagnostics` | 用 target shortest-path labels 诊断采样边，不参与 policy。 |
| `src/training/diagnostics.py` | `TrainingDiagnosticsCollector.collect` | 汇总训练 loss 和 rollout diagnostics。 |
| `src/data/schema/batch.py` | `RetrievalData` / `RetrievalBatch` | PyG 数据对象和 batching 增量规则。 |
| `src/data/collate.py` | `RetrievalCollator` | 构造 `question_emb [B,D]`、`edge_batch`、`edge_ptr`。 |
| `src/data/schema/repeat.py` | `repeat_retrieval_batch` | 多 rollout 时物理重复 batch。 |
| `src/graph/ops.py` | `build_anchor_induced_edge_mask` | root edges。 |
| `src/graph/ops.py` | `compute_uniform_nonroot_backward_removals` | backward policy 的可逆 parent 集合。 |
| `src/graph/paths.py` | `compute_path_labels` | 离线生成 anchor/target path labels，policy 不直接使用 target labels。 |

## 算法逻辑审视

### 合理之处

- **层次清楚**：runner、engine、executor、policy、reward、loss 的边界基本正确。
- **无明显 label leakage**：policy 特征只用 graph/question/anchor/current state，target
  labels 在 reward 和 diagnostics 中使用。
- **Stop/Expand 分解合理**：先采 option，再在 Expand 条件下采边。loss 中 log_pf
  与 log_pb 的定义清楚。
- **root edges 语义一致**：root edges 是初始 evidence，不计入 learned expansion。
- **终止保证合理**：当无 frontier 或 budget exhausted 时采样逻辑强制 Stop；循环长度为
  `expand_budget + 1`，预算用完后下一步会终止。
- **多 rollout 的 split 语义清楚**：repeated batch 的 edge ids 会映射回原 batch。
- **Feature reuse 有意识**：`FeatureBank` 静态缓存，frontier edge embedding 在
  `StateReadout` 和 `EdgeScorer` 间复用。

### 需要修正或澄清

1. **在线生成 API 缺失**
   - 位置：`WeaverModule.generate_subgraph_masks`。
   - 问题：调用不存在的 `RolloutRunner.generate_online_rollouts`。
   - 建议：直接改用 `generate_rollouts`，或增加别名方法并补 `forward` 测试。

2. **`PolicyOutput.root_log_z` 命名容易误导**
   - 当前 `Policy.forward` 每个 state 都设置 `root_log_z=state_log_flow`。
   - `RolloutEngine._write_flow` 只在 `t==0` 使用它，所以训练主链路没问题。
   - 建议：要么只在 root state 填 `root_log_z`，要么改名/注释强调 engine 仅取
     root step。

3. **`State.create_initial` 对非法 anchor 静默过滤**
   - repo 风格更偏 fail-fast。
   - 当前实现对 invalid anchors 只忽略，可能隐藏数据物化错误。
   - 建议：训练/eval 主路径至少加 debug 或严格校验开关。

4. **frontier summary 命名顺序可读性不足**
   - `FrontierLogitSummary` dataclass 顺序含 `edge_sharpness`，但 `as_tensor` 返回
     `[edge_max, edge_logmeanexp, edge_log_size]`，不含 sharpness。
   - 当前逻辑可运行，但长期维护时容易误读。

5. **重复的 root edge 逻辑**
   - `reward.py`、`terminal_subgraph.py`、`state.py` 都构造 root edge mask。
   - 建议统一依赖 `graph.ops.build_anchor_induced_edge_mask`，减少未来语义漂移。

6. **`repeat_retrieval_batch` 是全量复制**
   - rollout 执行不需要所有 target shortest-path heavy labels。
   - 当前全量复制简单，但 memory-heavy。
   - 如果未来在 repeated batch 内启用 teacher/diagnostics，还要确认所有 label 字段都完整
     repeat。

## 效率审视

### 已经做得比较好的点

- `FeatureBank` 不在每个 step 重算。
- policy、sampling、diagnostics 大部分 tensor 化，按 candidate 或 graph 分段计算。
- `torch_scatter` 用于 per-graph logsumexp、max、sum。
- `RolloutRunner` chunking 控制显存峰值。
- `EdgeScorer` 可复用 `StateReadout` 的 frontier edge embeddings。
- residual head zero-init 和 warmup 减少早期破坏 semantic prior。

### 主要热点

| 热点 | 当前实现 | 风险 | 优先级 |
| --- | --- | --- | --- |
| 多 rollout | `repeat_retrieval_batch` 复制整批 node/edge/label tensors | 显存和静态 feature 计算随 K 线性增长 | 高 |
| frontier 枚举 | `StateReadout._frontier_readout` 和 `frontier_edges` 都扫全图边 | 大图或大 budget 下重复开销 | 高 |
| stop-now reward | engine 每 step 对所有 active graph 先算 reward | 关闭 StopTB/StopAdv 时仍可能浪费 | 中 |
| state snapshot | `State.detach` 每 step clone active masks | N/E 大时有额外显存带宽 | 中 |
| backward policy | 可逆 removal 用小规模 Python/CPU list 逻辑 | GPU 同步，大 batch 时慢 | 中 |
| StopAdv child reward | 每候选 child Python connected set | top-k/batch 扩大时慢 | 中 |
| expand edge sampling | 对 Expand 图逐图 `Categorical` | B 很大时 Python loop 开销 | 低到中 |
| repeated diagnostics | `has_candidate`、budget mask 多处重复算 | 小但容易累积 | 低 |

### 建议优化顺序

1. 修复 `generate_online_rollouts` API 缺口，并加回归测试。
2. 在 `Policy.forward` 中直接复用 `context.frontier_edge_ids` 作为 candidate ids，
   避免第二次 full-edge frontier scan。
3. 把 `has_candidate`、`remaining_budget`、`can_expand` 做成 engine step-level
   context，executor/diagnostics/StopAdv 共享。
4. 当 StopTB、StopAdv 和 stop counterfactual 都关闭时，允许 lazy terminal reward：
   只对本步实际 Stop 的图算 reward。
5. 为多 rollout 引入逻辑 rollout 维度，静态 batch 和 `FeatureBank` 不物理复制。
   这是较大重构，适合单独做。
6. 如果 batch/budget 继续变大，再考虑 vectorize `_sample_expand_edges`、
   `compute_uniform_nonroot_backward_removals` 和 StopAdv child reward。

## 结构评价

当前结构整体达到了研究代码里比较重要的几个目标：

- **可解释**：state、action、reward、loss 的对应关系能追踪。
- **可测**：已有测试覆盖 repeat/split、sampling probability、StopAdv、feature leakage、
  cached frontier edge embedding 等关键点。
- **可调参**：policy feature、state readout、stop gate、edge scorer、rollout count/budget
  都由 config 控制。
- **可诊断**：stop counterfactual、teacher edge、root answer-edge rank、policy entropy 等
  diagnostics 能定位 rollout 行为。

但结构上还存在两类债务：

- **API 债务**：`generate_online_rollouts` 缺失，`root_log_z` 非 root step 仍填值，
  `current_reward` 命名不够准确。
- **性能债务**：物理 repeat、多次 full-edge scan、eager stop-now reward 和小规模
  Python loop。

短期应先修 API 和重复 frontier scan。中期再处理多 rollout 不复制静态图特征。

## 测试建议

建议新增或补强这些测试：

1. `WeaverModule.forward` 或 `generate_subgraph_masks` smoke test，覆盖当前缺失 API。
2. `Policy.forward` 复用 `StateReadout.frontier_edge_ids` 后，断言 candidate ids 与
   `frontier_edges` 一致。
3. `State.create_initial` 对非法 anchor 的严格/非严格行为测试。
4. `repeat_retrieval_batch` 对 rollout 必需字段和 heavy label 字段的完整性测试。
5. StopTB/StopAdv 都关闭时的 lazy reward 分支测试，如果后续实施该优化。
6. 大 batch 下 sampling/backward removal 的性能基准，防止 Python loop 成为训练瓶颈。

## 最终判断

当前 rollout 算法逻辑是成立的：它把静态语义/结构特征、当前动态子图状态、候选边
action features 和 terminal reward 解耦得比较干净，适合继续迭代。最应该立即处理的不是
数学逻辑，而是一个在线 API 缺口和两个效率问题：重复 frontier scan、物理重复 batch。
