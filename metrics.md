# Metrics Audit

本文描述当前仓库已经重构后的指标系统。

目标：

1. 去掉同名字段重复计算和覆盖
2. 修正字段名和真实语义不一致的问题
3. 把默认监控指标、默认诊断指标、深诊断指标分层

适用代码路径：

- `src/weaver/losses.py`
- `src/weaver/module.py`
- `src/eval/metrics.py`
- `configs/model/gflownet.yaml`
- `configs/callbacks/default.yaml`

## 指标命名空间

当前指标按职责分为八类：

- `loss/*`
- `subtb/*`
- `flow/*`
- `reward/*`
- `prob/*`
- `rollout/*`
- `proposal/*`
- `policy/*`
- `optim/*`

评估阶段使用三类：

- `sample/*`
- `best_of_k/*`
- `diversity/*`

职责边界：

- `losses.py` 只产出 `loss/subtb/flow/reward/prob`
- `module.py` 只产出 `rollout/proposal/policy/optim`
- `eval/metrics.py` 只产出 `sample/best_of_k/diversity`

## 默认监控指标

默认 checkpoint 和 early stopping 监控：

| 字段名 | 说明 | mode |
|---|---|---:|
| `val/best_of_k/max_recall_at_4` | 前 4 条 rollout 中的最优 recall 均值 | `max` |

配置位置：`configs/callbacks/default.yaml`

## 默认训练指标

这些指标默认写入 logger/W&B。

### Loss / SubTB / Flow / Reward

| 字段名 | 含义 | 说明 |
|---|---|---|
| `train/loss/total` | 总训练损失 | 最终反向传播目标 |
| `train/loss/subtb` | SubTB 主损失 | 子轨迹平衡主项 |
| `train/subtb/residual_abs_mean` | SubTB 残差绝对值均值 | 看平衡误差大小 |
| `train/subtb/residual_square_mean` | SubTB 残差平方均值 | 看误差能量 |
| `train/subtb/residual_std` | SubTB 残差标准差 | 看残差波动 |
| `train/flow/log_z_mean` | 根流 `log Z` 均值 | 看 flow scale 是否漂移 |
| `train/flow/log_z_std` | 根流 `log Z` 标准差 | 看 flow 稳定性 |
| `train/reward/log_reward_mean` | terminal log reward 均值 | 看 reward 是否过低/过稀疏 |
| `train/reward/log_reward_std` | terminal log reward 标准差 | 看 reward 分布离散程度 |
| `train/reward/clipped_ratio` | reward 截断比例 | 低于 `log_reward_clip_min` 的比例 |

### Rollout / Proposal / Policy / Optim

| 字段名 | 含义 | 说明 |
|---|---|---|
| `train/rollout/trajectory_length_mean` | 轨迹长度均值 | 包含最终 `Stop` 步 |
| `train/rollout/max_length_ratio` | 走满预算比例 | 轨迹长度等于 `expand_budget + 1` 的比例 |
| `train/rollout/nonzero_reward_ratio` | 非零 reward 比例 | terminal answer F1 大于 0 的轨迹比例 |
| `train/rollout/terminal_f1_mean` | 终止 answer-node F1 均值 | 总体 rollout 质量 |
| `train/rollout/coverage/nonzero_reward_ratio` | coverage 组非零 reward 比例 | 仅统计 coverage rollout 轨迹 |
| `train/rollout/coverage/terminal_f1_mean` | coverage 组终止 F1 均值 | 仅统计 coverage rollout 轨迹 |
| `train/rollout/coverage/trajectory_count` | coverage 组轨迹数 | 真实 trajectory count，不是 rollout group 数 |
| `train/rollout/online/nonzero_reward_ratio` | online 组非零 reward 比例 | 仅统计 online rollout 轨迹 |
| `train/rollout/online/terminal_f1_mean` | online 组终止 F1 均值 | 仅统计 online rollout 轨迹 |
| `train/rollout/online/trajectory_count` | online 组轨迹数 | 真实 trajectory count，不是 rollout group 数 |
| `train/proposal/intervention_prob` | 当前 proposal 干预概率 | 调度器给定值 |
| `train/proposal/intervention_step_ratio` | proposal 干预步比例 | `intervention_count / total_decision_steps` |
| `train/proposal/forced_expand_step_ratio` | proposal 强制 Expand 步比例 | `expand_override_count / total_decision_steps` |
| `train/proposal/forced_stop_step_ratio` | proposal 强制 Stop 步比例 | `stop_override_count / total_decision_steps` |
| `train/proposal/coverage_rollout_count` | coverage rollout 个数 | rollout group 数，不是 trajectory count |
| `train/proposal/online_rollout_count` | online rollout 个数 | rollout group 数，不是 trajectory count |
| `train/policy/target_stop_prob_mean` | target policy 平均 Stop 概率 | 所有有效决策状态上的平均值 |
| `train/policy/target_expand_prob_mean` | target policy 平均 Expand 概率 | 所有有效决策状态上的平均值 |
| `train/policy/edge_entropy_mean` | 边分布熵均值 | 看边分布是否塌缩 |
| `train/optim/lr` | 当前学习率 | optimizer 第一个 param group 的 `lr` |
| `train/optim/temperature` | 当前 rollout temperature | 训练采样温度 |

## 深诊断指标

以下指标仅在 `model.debug_metrics=true` 时写入 logger。

### Train Deep Debug

| 字段名 | 含义 | 说明 |
|---|---|---|
| `train/loss/reward_matching` | reward matching 损失 | 仅当对应系数启用时有意义 |
| `train/loss/edge_auxiliary` | edge auxiliary 损失 | coverage 辅助项 |
| `train/loss/edge_auxiliary_count` | edge auxiliary 样本数 | 辅助监督计数 |
| `train/prob/trajectory_log_pf_mean` | 轨迹级前向 log-prob 均值 | 深诊断概率项 |
| `train/prob/trajectory_log_pb_mean` | 轨迹级后向 log-prob 均值 | 深诊断概率项 |
| `train/prob/step_log_pf_mean` | 步级前向 log-prob 均值 | 深诊断概率项 |
| `train/prob/step_log_pb_mean` | 步级后向 log-prob 均值 | 深诊断概率项 |
| `train/flow/state_log_flow_mean` | 状态流均值 | 中间状态 flow 量级 |
| `train/flow/state_log_flow_std` | 状态流标准差 | 中间状态 flow 波动 |
| `train/flow/terminal_flow_minus_reward_mean` | 终止 flow 与 reward 差值均值 | 看 terminal flow 是否贴合 reward |
| `train/subtb/subtrajectory_length_mean` | 子轨迹长度均值 | 深诊断 SubTB 展开结构 |
| `train/subtb/subtrajectory_count_mean` | 子轨迹数均值 | 深诊断 SubTB 展开结构 |
| `train/rollout/coverage/full_recall_rate_at_<K>` | coverage 组真实 K 的完美召回率 | K 使用实际 `effective_k=min(requested_k, coverage_rollouts)` |
| `train/rollout/online/full_recall_rate_at_<K>` | online 组真实 K 的完美召回率 | 不再使用假 `success_at_8` 命名 |
| `train/proposal/pure_proposal_trajectory_count` | 纯 proposal 轨迹数 | 全轨迹每一步都被 proposal 干预 |
| `train/proposal/pure_online_trajectory_count` | 纯 online 轨迹数 | 全轨迹没有任何 proposal 干预 |
| `train/proposal/mixed_trajectory_count` | mixed 轨迹数 | 同时包含 proposal 和 online 步 |
| `train/policy/edge_relation_scale` | relation scale | scorer 诊断参数 |
| `train/policy/edge_src_text_scale` | src text scale | scorer 诊断参数 |
| `train/policy/edge_dst_text_scale` | dst text scale | scorer 诊断参数 |
| `train/policy/edge_structural_scale` | structural scale | scorer 诊断参数 |

## 默认验证 / 测试指标

### Val Default

| 字段名 | 含义 | 说明 |
|---|---|---|
| `val/sample/expected_recall` | 单次采样期望召回率 | 随机采一条 rollout 的平均 recall |
| `val/sample/expected_nodes` | 单次采样期望节点数 | 终止子图平均节点数 |
| `val/best_of_k/max_recall_at_1` | 前 1 条 rollout 最优 recall | best-of-K 视角 |
| `val/best_of_k/max_recall_at_4` | 前 4 条 rollout 最优 recall | 默认 monitor |
| `val/best_of_k/max_recall_at_8` | 前 8 条 rollout 最优 recall | best-of-K 视角 |
| `val/best_of_k/full_recall_rate_at_1` | 前 1 条 rollout 完美召回率 | recall 等于 1.0 的图比例 |
| `val/best_of_k/full_recall_rate_at_4` | 前 4 条 rollout 完美召回率 | recall 等于 1.0 的图比例 |
| `val/best_of_k/full_recall_rate_at_8` | 前 8 条 rollout 完美召回率 | recall 等于 1.0 的图比例 |

### Test Default

| 字段名 | 含义 | 说明 |
|---|---|---|
| `test/sample/expected_recall` | 测试集单次采样期望召回率 | 与 `val` 同义 |
| `test/best_of_k/max_recall_at_4` | 测试集前 4 条 rollout 最优 recall | 与 `val` 同义 |
| `test/best_of_k/full_recall_rate_at_4` | 测试集前 4 条 rollout 完美召回率 | 与 `val` 同义 |

## 评估深诊断指标

以下指标只在 `model.debug_metrics=true` 时写入：

| 字段名 | 含义 | 说明 |
|---|---|---|
| `val/sample/dangling_edge_ratio` | 悬挂边比例 | added edges 中被 protected core 剪掉的边比例 |
| `test/sample/dangling_edge_ratio` | 悬挂边比例 | 与 `val` 同义 |
| `val/diversity/edge_jaccard` | rollout 多样性 | 终止子图边集合的 Jaccard distance |
| `test/diversity/edge_jaccard` | rollout 多样性 | 与 `val` 同义 |
| 其余 `val/best_of_k/*` 与 `test/best_of_k/*` 中未进入默认集合的 K | 可选 best-of-K 诊断 | 例如 `at_2` |

## 已删除或改名的旧字段

| 旧字段 | 处理 |
|---|---|
| `train/proposal_intervention_ratio` | 改为 `train/proposal/intervention_step_ratio` |
| `train/proposal_expand_ratio` | 改为 `train/proposal/forced_expand_step_ratio` |
| `train/proposal_stop_ratio` | 改为 `train/proposal/forced_stop_step_ratio` |
| `train/coverage_success_at_8` | 删除默认写入；调试时改为真实 K 的 `train/rollout/coverage/full_recall_rate_at_<K>` |
| `train/online_success_at_8` | 删除默认写入；调试时改为真实 K 的 `train/rollout/online/full_recall_rate_at_<K>` |
| `val/high_reward/oracle_max_recall_at_4` | 改为 `val/best_of_k/max_recall_at_4` |
| `val/high_reward/success_at_4` | 改为 `val/best_of_k/full_recall_rate_at_4` |
| `val/distribution/expected_dangling_ratio` | 改为 `val/sample/dangling_edge_ratio` |
| `val/diversity/edge_jaccard_diversity` | 改为 `val/diversity/edge_jaccard` |

## 当前系统解决了什么问题

当前实现已经修复三类结构性问题：

1. 不再由 `losses.py` 和 `module.py` 重复计算并覆盖 proposal ratio 字段。
2. 不再使用虚假的 `success_at_8` 命名；训练期 full recall 字段只在深诊断模式下按真实 `effective_k` 写出。
3. 默认 logger 只保留核心训练和评估指标，深诊断指标通过 `model.debug_metrics=true` 显式开启。

## 性能策略

默认训练指标走 fast path：

- `train/rollout/*terminal_f1_mean`
- `train/rollout/*nonzero_reward_ratio`
- `train/rollout/*trajectory_count`

这些只从 `rollout.stats.terminal_answer_f1` 汇总，不重建 terminal subgraph，也不调用 recall matrix。

只有在 `model.debug_metrics=true` 时，训练才会额外计算：

- `train/rollout/coverage/full_recall_rate_at_<K>`
- `train/rollout/online/full_recall_rate_at_<K>`

这些字段需要重建 terminal subgraph，因此默认关闭。

默认验证/测试也在计算前裁剪深诊断指标：

- 默认不计算 `sample/dangling_edge_ratio`
- 默认不计算 `diversity/edge_jaccard`
- `test` 默认不计算 `sample/expected_nodes`

这些指标在 `model.debug_metrics=true` 时才计算。
