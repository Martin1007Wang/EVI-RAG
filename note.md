1. RewardModel 的 precision 定义在 anchor 与 target 有重叠时是错误的，甚至可能导致 F1 > 1
文件：src/models/reward.py:129-159
代码当前做的是：
hits        = |active_nodes ∩ target_nodes|
retrieved   = |active_nodes \ anchor_nodes|
precision   = hits / retrieved
recall      = hits / gold
F1          = 2PR / (P+R)
这里分子 hits 把 anchor 上的 target 也算进去了，但分母 retrieved 又把 anchor 全部排除了。
这会造成两个悖论：
- 若答案节点本身就是 anchor，且模型一开始就 stop：
  - recall = 1
  - retrieved = 0，代码把 precision = 0
  - 最终 F1 = 0
  - 这在语义上明显不对，因为答案明明已经在根状态里。
- 若有多个答案都在 anchor 中，而模型又额外扩了一点点非 anchor 节点：
  - hits 可能大于 retrieved
  - 导致 precision > 1
  - 进而 F1 > 1
  - 这在数学上直接不合法。
如果数据保证 anchor ∩ target = ∅，这个 bug 不会触发；但我在预处理代码里没看到这种排除。reachable_answer_entities 只是按图中可达答案构建，没有去除 question entities。见 src/data/preprocess_steps/graph_collect.py:147-177、src/data/preprocess_steps/materialize.py:150-190。
这是一类真正的逻辑错误，不只是“理论近似”。


2. teacher forcing 是 off-policy 的，但损失没有做任何校正
文件：src/models/rollout/sampling.py:118-153，src/models/rollout/executor.py:78-160，src/models/losses.py:313-370
实现里真实采样分布不是纯模型前向策略：
- 动作类型采样先按 behavior_logits 采样
- 之后可能被 teacher 强行改写
- 边采样也可能走 teacher sampler
- 另外 expand 采样还带温度 temperature
但写入 loss 的 step_log_pf 却始终是“未加温度、未做行为修正的 target policy log-prob”。
也就是说：
轨迹是按 behavior policy 采出来的
loss 却按 target policy 的 log P_F 在做 SubTB
且没有 importance weighting
因此，这不是严格意义上的 on-policy GFlowNet/SubTB 训练，而是“teacher-guided / tempered off-policy approximation”。
这不一定导致代码跑错，但如果你要问“数学上是不是严格对应标准 GFlowNet 目标”，答案是否定的。

3. 配置里有 reward shaping，但训练时根本没用上
文件：configs/model/gflownet.yaml:30-41，src/models/reward.py:410-474，src/models/rollout/executor.py:152-160
RewardModel 里实现了：
- potential(...)
- step_shaping(...)
配置里也暴露了：
- relation_shaping_scale
- positive_coverage_shaping_scale
- distance_progress_shaping_scale
但 rollout/executor 实际只在 stop 时调用了终止奖励：
reward_model(...)
没有任何地方调用 step_shaping()。
所以当前实现里，这几个 shaping 配置项是“看起来能用，实际上无效”的。  
这是实现与配置语义不一致。

4. teacher 的 stop 条件和 reward 优化目标不一致
文件：src/models/guidance.py:107-118，src/models/reward.py:18-33,129-159
teacher 的 stop 判据是：
只要当前 active subgraph 里出现任意一个 target，就可以 stop
但 reward 的主目标是所有答案实体上的 F1，不是“命中任一答案”：
recall = |active_gold| / |gold|
precision = ...
F1 over all answer nodes
因此在多答案问题上，teacher 会偏向过早停止，而 reward 期望继续扩展以覆盖更多答案。  
这会造成 supervision signal 和 final objective 不完全一致。

5. 数据可达性与 rollout horizon 没对齐
文件：configs/model/gflownet.yaml:6，src/utils/path_utils.py:183-214，src/data/preprocess_steps/graph_collect.py:141-177，configs/pipeline/default.yaml:13-21
当前 max_steps = 3。  
但预处理保留样本时，默认并没有要求“答案必须在 3 步内可达”，只要求“至少有路径”或者在 *-sub 数据集中“至少有 reachable in-graph answer”。
这意味着很多样本可能：
- 理论上答案在图中可达
- 但在当前 rollout budget 下根本不可能到达
这不是代码 bug，但会让奖励上界被结构性压低，也会让 teacher guidance 在剩余 budget 条件下经常失效。

算法主线
把每个问题对应的候选知识图记作 G = (V, E)。
预处理阶段先从图里构造几个监督/引导信号：
- anchor 集合 A：问题实体在图中的节点，见 src/data/preprocess_steps/graph_collect.py:97-119
- target 集合 Y：答案实体中“在图中且从 anchor 可达”的那些节点，见 src/utils/path_utils.py:149-214
- shortest_path_edge_mask：所有 anchor 到 reachable target 的最短路径边并集
- node_to_target_distance(v)：节点到任意 reachable target 的最短有向距离
- shortest_path_count(v)：从该节点沿最短路走到 target 的最短后缀条数
这些量被 materialize 到 RetrievalBatch。见 src/data/preprocess_steps/materialize.py:167-191、src/data/schema/batch.py:10-38。
MDP 定义
状态：
s_t = (V(E_t), E_t)
其中：
- E_t 是当前 active edges
- V(E_t) = A ∪ {u, v : (u, r, v) ∈ E_t}
- rollout depth / budget feature 由 |E_t \ E_0| 派生，不是独立状态分量
- 初始状态 s_0：
  - E_0 = anchor-induced edges
  - V(E_0) 由 anchors 与 root edges 共同确定
见 src/models/state.py:17-42
动作有两类：
a_t ∈ { expand(e), stop }
其中 expand(e) 只允许选 frontier edge：
Frontier(s_t) = { e ∈ E \ E_t : 至少有一个端点在 V(E_t) 中 }
见 src/models/policy.py:411-418。
转移：
expand(e=(u,v)):
  E_{t+1} = E_t ∪ {e}
  V_{t+1} = V_t ∪ {u,v}
stop:
  trajectory terminates
见 src/models/state.py:54-68。
前向策略分解
实现里把前向策略拆成两层：
P_F(a_t | s_t)
= P_type(c_t | s_t) *
  P_edge(e_t | s_t, c_t = expand)
其中：
- P_type 是 Expand vs Stop 的图级二分类，见 ActionHead
- P_edge 是在 candidate frontier 上按 softmax 选边，见 ExpandEdgeScorer
代码位置：
- src/models/policy.py:433-456
- src/models/modules/heads.py:93-185
- src/models/modules/heads.py:187-247
边打分又拆成：
logit(e) = prior(q, rel_e) + residual(src_dyn, dst_static, q)
其中 prior 基本是 query 和 relation 的 cosine prior，residual 是学习到的修正项。  
见 src/models/modules/heads.py:154-176。
反向策略
反向策略不是学出来的，而是精确构造的：
P_B(s_t | s_{t+1}) = 1 / |Parents(s_{t+1})|
这里 Parents(s_{t+1}) 不是“任意删一条边”，而是“删掉一条非 root edge 后，得到的状态仍然是合法前驱，并且把这条边重新加回去恰好能恢复当前子图”的那些状态。  
见 src/utils/graph_utils.py:120-256。
这一点实现得其实很认真，是本仓库数学上比较扎实的部分。
---
双通道网络的数学意义
这是本实现最关键的设计：
- node_h_state：只沿 active_edges 做消息传递，供 FlowHead 使用
- node_h_policy：沿 active_edges ∪ frontier_edges 做消息传递，供 actor/edge scorer 使用
见 src/models/modules/backbone.py:27-60,261-355。
直觉上：
- 流函数 F(s) 必须是严格的状态函数，不能偷偷看到“frontier 的未来信息”
- 但 actor 如果只看 active subgraph，又会对候选边感知不足
所以它做了一个分离：
critic 看 Markov-faithful 通道
actor  看 frontier-aware 通道
这在数学上是合理的，也和代码注释的意图一致。
---
损失函数推导
设一条轨迹为：
s_0 --a_0--> s_1 --a_1--> ... --a_{L-2}--> s_{L-1} --stop--> x
这里 rollout 明确保留 trajectory 作为训练载体，但每个 s_t 都是 canonical subgraph state，而不是把 history 本身塞进状态定义里。
注意这里：
- L = traj_len
- traj_len 包含最后的 stop 动作
- 所以若有 k 次扩张后停止，则 L = k + 1
代码中 state_log_flows 的实际含义是：
u_0 = root_log_z = log Z_theta(q)
u_t = log F_theta(s_t | q),  t = 1, ..., L-1
也就是说，第一列不是普通 FlowHead(s_0)，而是单独的 ZHead 输出。  
见 src/models/rollout/engine.py:79-99、src/models/policy.py:375-381。
每一步记录：
η_t = log P_F(a_t | s_t) - log P_B(s_t | s_{t+1})
那么 forward-looking SubTB 的子轨迹残差就是：
δ(i,j) = u_i + Σ_{t=i}^j η_t - target(j)
其中：
target(j) = u_{j+1},      if j < L-1
target(j) = log R(x),     if j = L-1
于是：
δ(i,j)
= u_i + Σ_{t=i}^j [log P_F(a_t|s_t) - log P_B(s_t|s_{t+1})] - u_{j+1},   j < L-1
δ(i,L-1)
= u_i + Σ_{t=i}^{L-1} [log P_F(a_t|s_t) - log P_B(s_t|s_{t+1})] - log R(x)
损失为：
L_SubTB
= Σ_{i<=j} λ^(j-i) * δ(i,j)^2
再加一个 reward matching 项来消除 flow 的加法常数自由度：
L_RM = (u_{L-1} - log R(x))^2
L = L_SubTB + α * L_RM
对应代码：
- 残差构造：src/models/losses.py:139-177
- 权重矩阵：src/models/losses.py:104-132
- 总损失：src/models/losses.py:223-443
我专门检查了这里的索引关系，结论是：没有发现 off-by-one bug。
一个细节是，代码注释里对 P_B 的记号略混乱，但张量本身和 loss 对齐是对的。
---
终止奖励的数学逻辑
当前终止奖励定义为：
recall    = |V_t ∩ Y| / |Y|
precision = |V_t ∩ Y| / |V_t \ A|
F1        = 2PR / (P+R)
log R     = log(F1), if F1 > 0
         = log_r_min + semantic_bonus, otherwise
见 src/models/reward.py:13-179。
如果 F1 = 0，它会回退到一个语义 bonus：
semantic_bonus
= η * max_{v ∈ active nodes} cosine(h_v, mean(h_targets))
见 src/models/reward.py:185-264。
这个设计的本意是对的：避免所有失败轨迹都拿到完全相同的奖励下界。  
问题不在这里，而在前面 precision 的定义混合了“hits 含 anchor / denominator 不含 anchor”。
---
实现上成立的部分
这些部分我认为是“数学逻辑基本正确”的：
- rollout 与 SubTB loss 的索引是一致的，见 src/models/rollout/engine.py:80-105 与 src/models/losses.py:139-177
- horizon 强制 stop 的处理是自洽的，见 src/models/rollout/executor.py:58-76,140-150
- backward policy 不是拍脑袋的 uniform，而是 exact parent uniform，见 src/utils/graph_utils.py:171-256
- node_h_state / node_h_policy 分离很合理，避免把 frontier 信息污染到 flow estimation，见 src/models/modules/backbone.py:297-355
---
次要代码味道
这些不是核心数学 bug，但说明代码还有未清理的设计残留：
- anchor_signed_distance 预处理出来了，但下游几乎没用到。见 src/data/preprocess_steps/materialize.py:159-190、src/data/dataset.py:76-98
- State.phase / apply_stop() 在 batched rollout 中实际上没被使用，真实的终止状态是 RolloutBuffer.is_terminated。见 src/models/state.py:54-80、src/models/rollout/buffers.py:65-88
