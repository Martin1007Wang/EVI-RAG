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

验证和测试时，`ExactReachabilityAnalyzer` 通过 exact DP 计算：

```text
M_gold(x) = P_theta(eventually hit any gold answer | x)
```

并进一步聚合成：

- `answer/gold_mass`
- `answer/hit@k`
- `answer/recall@k`

对应实现：

- `src/metrics/answer_reachability/exact_analysis.py`
- `src/metrics/answer_reachability/posterior.py`

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

## 5. 问题二：当前信用分配粒度不足以区分路径质量

### 5.1 当前状态抽象是 `(node, time)`，不是 path-aware state

当前数学推导中，状态定义为：

```text
s_t = (v_t, t)
```

这意味着：

- 只要两条轨迹在同一步到达同一个节点，
- 它们之后共享同一个 `F(s)`，
- 也共享同一个 forward policy `P_F(s' | s)`。

因此如果两条不同前缀路径都在第 `t` 步到达节点 `v`：

```text
tau_1 -> (v, t)
tau_2 -> (v, t)
```

当前模型无法继续区分：

- 哪条前缀更合理，
- 哪条前缀更短、信息更干净，
- 哪条前缀更符合语义约束，
- 哪条前缀更值得保留更多概率质量。

这不是“训练得还不够好”，而是“当前状态表达本身没有保留这些信息”。

### 5.2 backward 分解也比较粗

当前采样时使用的 backward log-prob 是按 target 节点入度构造的 uniform backward：

```text
log P_B(parent | child) = -log indegree(child)
```

这会进一步让 credit assignment 更偏向拓扑均摊，而不是语义区分。

换句话说，当前 SubTB 不是在一个强表达的 path-aware 流模型上工作，而是在一个较粗
的 node-time 流模型上工作。

### 5.3 起点流和中间状态流的参数化已经统一

这个位置曾经存在一个重要问题：

- 起点状态 `(q, 0)` 由专门起点头估计
- 后续状态 `(v, t), t >= 1` 由状态流头估计

这会在 `t=0 -> t=1` 之间引入人为的参数鸿沟，让 SubTB 额外承担“两套头对齐”的负担。

当前主线已经改成统一状态流头：

```text
a_theta(q, x) := f_theta(q, 0, x)
```

也就是说：

- 起点状态和后续状态现在共享同一个 `NodeFlowHead`
- 起点特殊性由 `step_embed(0)` 和 `remaining_embed(T)` 表达
- `Y_start` 与后续中间状态锚点之间现在对应同一流函数的时间延拓

因此“起点头/状态头割裂”不再是当前主线的主要问题。

同样地，过去还存在一个配套问题：heuristic 被注入 forward policy，但没有进入状态锚点，
会让 SubTB 去拟合被 heuristic 污染后的残差。当前主线也已经把 heuristic 并入有效状态值
定义，并在 forward logit 中显式加入固定 backward 项，因此这条数学不一致也不再是当前主
线的主问题。

换句话说，当前需要继续学习的是真正的流量关系：

```text
同一个状态流函数如何随 node 和 time 变化
```

而不是：

```text
两个不同网络如何先彼此对齐
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
   - path-aware credit assignment
   - state representation 升级

在这两类文档完成之前，不建议把主要精力继续放在常规调参上。
