# Weaver Rollout、特征工程与 Policy 算子说明

本文说明当前 Weaver 的 rollout 与 policy 设计。当前实现采用
Reference-Measure-Corrected Action-Flow：STOP 和每条 frontier edge 都是
primitive action log-flow，continue flow 只由 edge action flows 聚合得到。

## 1. 状态真相

rollout state 的唯一集合真相是：

```text
selected_edge_ids
```

其余量都是派生视图：

```text
active_nodes      = anchors ∪ selected edges touched nodes
frontier          = active nodes 出发、尚未选择、当前图内的有向 KG edges
remaining_budget  = budget - |selected_edge_ids|
```

frontier 构造不读取 target label，也不读取答案节点。target 信息只通过 terminal
reward 进入训练监督。

合法动作是：

```text
A(z) = {STOP} ∪ Frontier(z)
```

如果 budget 已耗尽或 frontier 为空：

```text
A(z) = {STOP}
```

其中 budget 耗尽是环境边界，不等价于 policy 主动认为应该停止。诊断中应分开统计
`policy_stop` 与 `budget_boundary`。

## 2. FeatureEncoder：projection-free semantic reader

`FeatureEncoder` 读取 upstream embedding，保持 projection-free 语义空间：

```text
node_embedding
relation_embedding
query_embedding
node_text_mask
```

语义 prior 直接在 query、relation、destination 的原始兼容空间里计算。这样做的前提是
upstream embedding 已归一化或至少处于可比较的语义空间；Weaver 内部不再额外投影来改变该
空间。

这避免了一个常见错位：edge residual 可以学习状态依赖的残差，但 semantic prior 本身仍然
表达 query 与 KG 文本/关系的直接相似度。

## 3. StateEncoder：状态视图编码

policy 读取四类 state-level 表示：

```text
query_h
selected_h
active_h
budget_h
```

含义如下：

```text
query_h     当前问题表示
selected_h 由 selected_edge_ids 派生的已选证据摘要
active_h   由 active nodes 派生的当前可扩展区域摘要
budget_h   remaining budget 的可学习编码
```

`selected_h` 与 `active_h` 都来自当前 state，不读取目标标签。`remaining_budget` 只表达环境剩余
步数，不决定 reward。

## 4. EdgeEncoder：角色保持算子

每条 frontier edge：

```text
e = (src, rel, dst)
```

保留角色结构编码：

```text
edge_h = W_src h_src + W_rel h_rel + W_dst h_dst
```

该算子显式区分 source、relation、destination，不把三者当成无序集合。这个设计对有向 KG edge
是必要的，因为同一 relation 在不同方向、不同端点角色下语义不同。

## 5. STOP action log-flow

STOP head 输出：

```text
uθ(z) ∈ R
```

它是 STOP action log-flow，不是 Bernoulli probability，也不叫 terminal probability。

输入是：

```text
[query_h, selected_h, active_h, budget_h]
```

代码字段：

```text
stop_log_flow
```

STOP probability 只在和所有 edge action flows 归一化后得到：

```text
log P_F(STOP | z) = stop_log_flow - state_log_flow
```

## 6. Edge action log-flow

每条 edge 先计算 semantic prior：

```text
s_sem(q,e) =
    α <q, rel(e)> + β 1[dst has text] <q, dst(e)>
```

然后计算状态依赖 residual：

```text
rθ(z,e) =
MLPθ([query_h[row], selected_h[row], active_h[row], budget_h[row], edge_h])
```

raw edge score：

```text
edge_raw_score = sθ(z,e) = s_sem(q,e) + rθ(z,e)
```

最后做 reference-measure correction：

```text
edge_log_flow = qθ(z,e) = edge_raw_score - log |Frontier(z)|
```

`edge_log_flow` 是 primitive edge action log-flow，不是 conditional edge logit。

## 7. 为什么需要 `- log |Frontier(z)|`

如果直接使用 `edge_raw_score` 聚合 continue flow：

```text
continue_log_flow = logsumexp_e edge_raw_score
```

当所有 edge 分数近似相同：

```text
edge_raw_score = c
```

则：

```text
continue_log_flow = c + log |Frontier(z)|
```

frontier 越大，continue mass 天然越大。这会把节点度数或 frontier size 误当成继续扩展证据。

修正后：

```text
continue_log_flow =
logsumexp_e [edge_raw_score - log |Frontier(z)|]
= logmeanexp_e edge_raw_score
```

当所有 edge 质量相同时：

```text
continue_log_flow = c
```

边多不会自动压制 STOP。

## 8. Continue flow 是派生量

当前实现没有独立 continuation head。

```text
continue_log_flow = Cθ(z) = logsumexp_{e in Frontier(z)} edge_log_flow(z,e)
```

如果 frontier 为空：

```text
continue_log_flow = -∞
```

这意味着“是否继续”由所有 edge action flows 的聚合决定，而不是另一个只看 frontier summary 的
scalar head 决定。

## 9. State flow 和 forward policy

state flow：

```text
state_log_flow = Φθ(z)
               = logaddexp(stop_log_flow, continue_log_flow)
               = logsumexp(uθ(z), {qθ(z,e)})
```

action probabilities：

```text
stop_log_prob     = stop_log_flow - state_log_flow
edge_log_prob     = edge_log_flow - state_log_flow[row]
continue_log_prob = continue_log_flow - state_log_flow
```

条件 edge policy：

```text
conditional_edge_log_prob = edge_log_flow - continue_log_flow[row]
```

同一个 frontier 内，`-log |Frontier(z)|` 会在条件 softmax 中抵消：

```text
P_F(e | continue,z) = softmax_e edge_raw_score
```

因此：

```text
edge selection      看 edge 间相对质量
continue decision   看 edge action-flow aggregate
STOP competition    看 stop_log_flow 与 edge flow aggregate
```

## 10. PolicyOutput 字段与硬约束

`PolicyOutput` 的核心字段：

```text
state_log_flow
stop_log_flow
continue_log_flow

frontier_row_ids
frontier_edge_ids

edge_raw_score
edge_log_flow

stop_log_prob
edge_log_prob
continue_log_prob
conditional_edge_log_prob
```

validation 必须检查：

```text
state_log_flow =
    logaddexp(stop_log_flow, continue_log_flow)

continue_log_flow =
    segment_logsumexp(edge_log_flow by row)

stop_log_prob =
    stop_log_flow - state_log_flow

edge_log_prob =
    edge_log_flow - state_log_flow[row]

continue_log_prob =
    continue_log_flow - state_log_flow

conditional_edge_log_prob =
    edge_log_flow - continue_log_flow[row]
```

对非空 frontier row：

```text
sum_e exp(conditional_edge_log_prob) = 1
```

## 11. Rollout 逻辑

每一步：

```text
frontier = StateOps.frontier(state, graph)
out = policy(features, state, graph, frontier)
sample action from STOP ∪ frontier edges
```

采样分布：

```text
STOP: out.stop_log_prob[row]
EDGE: out.edge_log_prob[frontier item]
```

如果采到 STOP：

```text
terminal state = current state
```

如果采到 edge：

```text
state = expand(state, edge)
```

实现可以 two-stage sample：

```text
continue ~ P(continue | z)
edge     ~ P(e | continue,z)
```

但必须满足：

```text
log P(edge | z)
= continue_log_prob + conditional_edge_log_prob
= edge_log_prob
```

two-stage 只是代数分解，不是第二套模型。

## 12. Training objective：单一 SubTB 语义

训练只使用一套 flow ontology：

```text
STOP action flow = stop_log_flow
EDGE action flow = edge_log_flow
state flow       = state_log_flow
forward logprob  = action_log_flow - state_log_flow
```

对扩展 transition：

```text
z --e--> z'
```

长度 1 residual：

```text
edge_log_flow(z,e) - state_log_flow(z') - log P_B(z | z')
```

对 STOP transition：

```text
z --STOP--> terminal(z)
```

长度 1 residual：

```text
stop_log_flow(z) - log R(z)
```

一般 SubTB residual：

```text
Φθ(z_i)
+ Σ log P_F(z_{t+1}|z_t)
- Φθ(z_j)
- Σ log P_B(z_t|z_{t+1})
```

如果 `z_j` 是 terminal endpoint：

```text
Φθ(z_j) := log R(z_j)
```

当前实现保留 `BackwardPolicy`：child state 的有效 predecessor 上均匀。

## 13. Replay

replay 只提供更好的 state/transition 覆盖，不作为 behavior cloning。

主 loss 对 policy rollout 与 replay/oracle prefix transitions 使用同一公式：

```text
edge_log_flow - child_state_log_flow - backward_log_prob
```

或对应 SubTB segment。`behavior_action_log_prob` 不参与主 loss，当前实现已从 rollout tape/result 中移除。

## 14. Reward

本次实现保持 reward model 不变。reward 仍然只在 terminal boundary 上提供监督：

```text
log R(z)
```

推理时 policy 不读取答案；训练时答案只通过 terminal reward 进入 objective。这不是 label leakage，
leakage 只会发生在 forward feature 或 frontier 构造读取 target label 时。

## 15. Budget boundary

当 budget 耗尽：

```text
legal action set = {STOP}
terminal boundary is used for reward supervision
diagnostic reason = BUDGET_BOUNDARY
```

它不应计入 policy 主动 stop rate。否则会混淆：

```text
模型认为该停
预算耗尽不得不停
```

## 16. Evaluation 解释

评估应拆成四类。

Sampling quality：

```text
expected_recall
expected_hit
expected_edge_count
```

Model selectors：

```text
terminal_stop_flow_best: score = stop_log_flow(z_terminal)
trajectory_prob_best:   score = Σ log P_F(a_t | z_t)
state_flow_best:        score = state_log_flow(z_terminal)
```

Oracle upper bound：

```text
reward_best
oracle_best
```

这类指标只说明候选集上界，不说明模型选择能力。

Failure diagnostics：

```text
policy_stop_rate_by_depth
budget_boundary_rate_by_depth
hit_then_continue_rate
continue_log_flow_by_depth
stop_log_flow_by_depth
edge_log_flow_mean/max_by_depth
frontier_size_by_depth
terminal_flow_best - reward_best gap
```

## 17. 当前已删除或改名的旧设计

删除：

```text
continuation_head
frontier_h
frontier_summary_top_k
raw_continuation_log_flow
edge_action_log_flow = continuation_log_flow + edge_log_prob
behavior_action_log_prob in rollout result/tape
```

改名：

```text
terminal_log_flow -> stop_log_flow
terminal_head     -> stop_head
```

保留：

```text
FeatureEncoder projection-free
EdgeEncoder role-preserving
selected_h attention
active_h attention
budget_h
BackwardPolicy
selected_edge_ids as single truth
```

