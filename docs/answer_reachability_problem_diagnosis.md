# Answer Reachability Current Problem Diagnosis

本文记录当前 answer-reachability 主线已经暴露出的核心问题。

这不是一份调参备忘录，也不是最终方案设计文档。它的目的只有一个：把当前系统
真正卡住的地方说清楚，避免后续继续把主要精力放在 horizon、搜索预算或一般性
超参上。

相关背景文档：

- `docs/answer_reachability_performance_vs_upper_bound.md`
- `docs/answer_reachability_algorithm.md`
- `docs/answer_reachability_math_derivation.md`

## 1. 结论先说

- 当前主矛盾已经不是图可达性，而是训练目标、信用分配粒度、以及 answer ranking
  目标之间的不对齐。
- 在当前过滤后的 `cwq-sub` 和 `webqsp-sub` 数据上，gold answers 都在图中且 `4`
  步内可达，因此当前 gap 不能再主要归因于“图里没路”或“步数太短”。
- 当前 GFlowNet 真正在学的是“成功终止比失败终止更好”的 reward-normalized
  trajectory distribution，而不是“把最优答案排在最前面”的 ranking distribution。
- 因此，即使模型把当前 reward 分布学得更好，也不等价于它会自动逼近
  `recall@10` 的 oracle ceiling。

## 2. 已经基本排除的问题

根据 `docs/answer_reachability_performance_vs_upper_bound.md`：

- 当前保留下来的 `-sub` 验证/测试样本中，gold answers 全部都在图里。
- 而且这些 gold answers 全部都在当前 horizon `max_steps=4` 内可达。
- 因此当前 `gold_mass` 的 oracle upper bound 是 `1.0`。
- `hit@k` 的 oracle upper bound 也是 `1.0`。
- `recall@k` 的 upper bound 小于 `1.0`，主要来自多答案样本，而不是不可达。

这说明当前问题已经从：

```text
有没有路到答案？
```

转成了：

```text
在很多可达答案和很多可达路径里，模型为什么没有把概率质量放到更优答案上？
```

## 3. 当前系统真正优化的目标是什么

当前主线训练和评估共享同一个前向策略参数化，但不共享同一个目标语义。

### 3.1 训练目标

当前训练期 sampled rollout 的 terminal reward 是：

```text
R_x(tau) = 1,          if tau ends at any gold answer
R_x(tau) = epsilon_x,  otherwise
```

其中：

- `epsilon_x = epsilon`，或者
- `epsilon_x = epsilon / non_answer_count(x)`

对应实现：

- `src/models/gflownet/sampler.py`
- `src/models/configs/gflownet.py`

当前唯一真正被优化的损失是 `SubTB`，不是 `gold_mass`、不是 `recall@10`，也不是
任何显式的 ranking loss。对应推导见 `docs/answer_reachability_math_derivation.md`。

### 3.2 评估目标

验证和测试时，answer-reachability 默认通过 `FlowFrontierReachabilityAnalyzer`
deterministically 构造：

```text
M_gold(x) = P_theta(eventually hit any gold answer | x)
```

并进一步聚合成：

- `answer/gold_mass`
- `answer/hit@k`
- `answer/recall@k`

对应实现：

- `src/metrics/answer_reachability/analysis.py`
- `src/metrics/answer_reachability/flow_frontier.py`
- `src/metrics/answer_reachability/monte_carlo.py`
- `src/metrics/answer_reachability/posterior.py`

其中：

- answer-reachability 默认后端是 flow-frontier；
- edge retrieval 和显式 legacy fallback 仍会使用 Monte Carlo。

### 3.3 根本错位

因此当前系统天然存在一个结构性错位：

```text
训练学的是 reward-normalized trajectory distribution
评估看的是 answer-level ranking quality
```

这两者相关，但并不等价。

## 4. 问题一：奖励定义没有表达“答案排序”目标

如果把理想化的 GFlowNet 目标写成：

```text
P_theta(tau | x) ∝ R_x(tau)
```

那么在当前 reward 下，所有成功轨迹的 reward 都是 `1`，所有失败轨迹的 reward 都是
同一量级的 `epsilon_x`。这意味着：

- 模型被要求区分“成功 vs 失败”，
- 但没有被要求区分“更好的成功答案 vs 一般的成功答案”，
- 也没有被要求区分“更合理的成功路径 vs 冗余的成功路径”。

进一步地，answer-level 的理想终止分布近似满足：

```text
P_theta(a | x)
∝ sum_{tau: term(tau)=a} R_x(tau)
```

在当前 reward 下，这近似退化成：

```text
P_theta(a | x)
∝ number_or_mass_of_successful_paths_that_end_at_a
```

因此当前系统更容易学到：

- 哪些答案“容易通过很多路径被到达”，

而不是：

- 哪些答案“更应该排在前面”。

这正是当前 `gold_mass` 可以提升，但 `recall@10` 仍显著落后于 oracle ceiling 的一个
核心原因。

## 5. 问题二：信用分配仍然受 prefix 表征瓶颈限制

### 5.1 旧的 `(node, time)` 抽象已经移除，但表示瓶颈没有完全消失

当前主线已经不再把搜索状态写成纯 `(node, time)`。

现在的 `SearchState` 同时保留：

- 精确离散前缀 `path_token_ids`
- 当前节点与步数
- recurrent `control_state`

这带来了两个实质改进：

- 不同 prefix 不会在环境语义上被错误合并成同一个状态；
- non-root backward 可以直接从离散 prefix 恢复唯一 parent，而不需要再靠粗糙近似。

但这并不等于“路径质量区分问题已经彻底解决”。

前向 `log F` 和 `P_F` 看到的不是原始整段 prefix，而是一个压缩后的连续 summary：

```text
c_t = recurrent summary of prefix history conditioned on the question
```

因此当前真正的限制变成：

- 如果 `control_state` 容量不足，
- 或者对问题 token 的注意无法稳定挑出关键约束，
- 那么两条语义差异很大的 prefix 仍可能被压缩到过于接近的表示。

也就是说，主问题已经从“状态根本不保留路径信息”变成了：

```text
路径信息只通过一个轻量 recurrent controller 进入前向打分，表达力仍然可能不够。
```

### 5.2 backward 已经变精确，但 terminal credit 仍有近似

当前 graph move 的 backward 不再是按入度均摊的 uniform backward，而是 exact
prefix-tree backward：

```text
P_B(parent(x_t) | x_t) = 1
```

这修复了过去 credit assignment 里一个非常粗的近似。

但 terminal `submit -> sink(entity)` 这一步仍然需要 entity-level 的近似 backward
kernel，例如 alias-count 归一化。它解决的是“同一实体多个节点副本”的一部分问题，但仍
然没有完全校正 tree policy 下的 path multiplicity。

所以当前 backward 面临的核心残留问题已经不是“父边太粗”，而是：

- terminal entity aggregation 仍有近似；
- answer-level credit 仍然会受到 alias/path multiplicity 的影响。

### 5.3 已经修复的表示级问题，不再是当前主矛盾

这轮主线重构已经顺手修掉了几个以前确实存在的问题：

- 起点状态和中间状态现在共享同一个 `NodeFlowHead`
- 起点分布由起点 flow 直接归一化，不再依赖独立 `log Z` 头
- heuristic 默认不再污染 target policy；若开启，也只进入 behavior sampling

因此“起点头/状态头割裂”以及“heuristic 与 target flow 方程不一致”都不再是当前主线的
主要矛盾。

换句话说，当前还需要继续解决的是：

```text
如何让轻量 control-state 表征承载足够细的 prefix 语义与答案排序信用分配。
```

## 6. 问题三：非零失败奖励和大失败空间会稀释 gold mass

当前失败终点不是零奖励，而是小正奖励：

```text
R_fail = epsilon_x > 0
```

同时，当前前向约束基本只限制 horizon：

- 不禁止 revisit
- 不剪掉“未来必然失败”的边
- 不把失败 support 从前向 policy 中剔除

因此在 `max_steps=4` 的有限 horizon 内，仍然可能存在大量 failure prefixes 和 failure
trajectories。

这会导致一个重要后果：

```text
即使所有 gold 都可达，成功质量也不会自动逼近 1.0。
```

直观上，如果理想 GFlowNet 学到的是：

```text
P_theta(tau) ∝ R(tau)
```

那么 success mass 会同时受到两件事影响：

- successful trajectories 的总数和总质量
- failure trajectories 的总数和总质量

因此当前 gap 不能简单理解成“还没学会找到答案”，而更像：

```text
模型仍然把大量概率预算分给了虽然低奖励、但数量庞大的失败区域。
```

## 7. 问题四：`Z` 只负责总量归一，不负责排序信号

### 7.1 旧实现中的 `Z` 参数化问题

在最初的问题诊断阶段，系统没有显式解析求解真实 partition function `Z(x)`，而是
学习一个 graph-level 标量 `log Z_theta(x)`。

它由 `GraphLogZHead` 预测，输入是：

- question feature
- 所有 start 节点的平均 summary

可以写成：

```text
log Z_theta(x)
= g_theta(question_feature(x), start_summary(x))
```

这会带来一个结构性问题：

```text
root 归一化锚点
起点状态流
起点选择概率
```

是三套松耦合参数化，而不是同一个流守恒系统里自然推出的量。

### 7.2 当前实现中的修正

当前主线已经改成隐式虚拟源参数化：

```text
log Z_theta(x) = logsumexp_{q in Q(x)} V_start(q; x)
P_theta(s_0=q | x) = exp(V_start(q; x) - log Z_theta(x))
```

也就是说：

- `Z` 不再由独立 graph head 回归
- 起点概率由起点状态流直接归一化得到
- 根边界与起点边界满足严格的一致性关系

这个修正解决了“多起点集合下 root-flow 参数化不严谨”的问题，但没有改变下面这件
更根本的事实：

### 7.3 当前 `Z` 的作用

当前 `log Z_theta(x)` 的作用是：

- 作为 SubTB 根边界项，
- 帮助把 sampled trajectories 的 flow relation 调到自洽，
- 近似承担“总 reward mass 归一化锚点”的角色。

但它不直接承担：

- answer ranking supervision
- path preference supervision
- 多答案 top-k 排序 supervision

因此当前 `Z` 可以帮助系统学到：

```text
这个问题整体上有多少 reward mass
```

却不能单独解决：

```text
在多个正确答案之间，谁应该排前面
在多个成功路径之间，谁更合理
```

### 7.4 当前 `Z` 不等于 oracle ceiling

需要特别区分两件事：

- `Z_theta(x)`：当前模型内部学习到的归一化锚点
- oracle upper bound：假设可达 gold answers 都能被排在最前面时的评估 ceiling

这两者不是同一个对象。

因此，即使 `Z_theta(x)` 学得更稳定，也不意味着 `recall@10` 会自动接近 oracle。

## 8. 当前系统实际会偏向什么

在当前设计下，模型更容易偏向以下行为：

1. 学会“尽量进入任一成功区域”，而不是学会精确排序成功答案。
2. 偏好“成功路径更多、成功更容易累积质量”的答案。
3. 在多答案样本上，把质量集中到少数容易答案，而不是尽量覆盖更多 gold answers。
4. 在合流节点之后丢失 prefix 级别的路径优劣信息。

这与当前评估重点之间存在明显张力：

- `gold_mass` 关注成功质量是否进入 gold region。
- `recall@10` 关注 top-k 是否覆盖更多 gold answers。

当前训练目标对前者是弱对齐，对后者基本不是直接对齐。

## 9. 研究问题已经升级成什么

当前真正的问题，不再适合表述成：

```text
如何让模型在图里找到答案？
```

更准确的表述应该是：

```text
如何让模型在大量可达答案和大量可达路径中，
学习到足够细粒度的语义与路径质量信息，
并把更多概率质量稳定分配给更优答案，
从而提升 answer-level ranking，尤其是 recall@10。
```

换句话说，当前挑战已经升级为：

```text
从 feasibility 学习，升级为 ranking-generalization 学习。
```

## 10. 本文档给出的正式诊断

当前 answer-reachability 主线的核心问题可以归纳为三条：

1. 当前 reward 只表达“是否成功”，没有表达“哪个成功答案更值得优先生成”。
2. 当前状态与 backward 分解不足以对不同成功路径做细粒度信用分配。
3. 当前 `Z` 只是在学总 reward mass 的归一化，不承担答案排序职责。

因此，当前 gap 的本质不是：

```text
模型还没学会 reachability
```

而是：

```text
当前目标和表示还不足以支持 answer ranking 所需的精细信用分配与排序泛化。
```

## 11. 后续文档接口

本文档只负责把问题定义清楚。

后续如果继续推进，应当至少补两类文档：

1. 一个严格数学文档，明确当前 `R`、`Z`、SubTB、oracle upper bound 之间的关系。
2. 一个新目标设计文档，明确哪些改动属于：
   - reward shaping / answer ranking
   - prefix-aware / control-state credit assignment
   - state representation 升级

在这两类文档完成之前，不建议把主要精力继续放在常规调参上。
