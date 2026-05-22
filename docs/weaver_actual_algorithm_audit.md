# Weaver 实际算法与代码逻辑复盘

本文只基于当前仓库代码实际执行路径复盘，不根据命名、注释或预期论文算法推断。

## 0. Executive finding

当前仓库实现的是一个有限步 evidence subgraph 生成器：从 question anchors 初始化 active node set，每步在 active source nodes 的 outgoing physical edges 上采样扩展，或采样 STOP，最终得到一个 anchor-connected edge set/subgraph。它不是 DAG-GFlowNet；生成对象可以有环，frontier 只包含原始 physical outgoing edges。更准确地说，状态转移图是按 edge set inclusion 单调增长的 DAG，但生成对象不是 DAG。训练目标是 `SubTBLoss`，不是旧配置名里的 edge flow matching，也不是纯 DB/TB；`subtb_lambda=0.9` 时枚举采样轨迹内所有连续 subtrajectory，长度 1 项退化为 DB 风格，完整起止项包含 TB 风格项。STOP 被建模为 `edge_id=-1` 的 terminal action，不在 `Frontier` 中，但 policy 归一化时与 frontier edges 一起进入 `logsumexp`。frontier 是 active source outgoing frontier：`src in active` 的原始有向 KG 边，不制造 inverse edge。reward 是 label-dependent terminal reward：reachable answer recall 的 log，加 edge penalty 和 no-answer penalty；只用 `reachable_target_node_ids`，不是 full `target_node_ids`。训练信号不是直接监督 next edge，而是监督 policy 的 action logits/derived flows 在 policy rollouts + replay shortest paths 上满足 SubTB residual。

当前最核心的理论或实现错位是：代码与配置标签接近 edge-flow/GFlowNet，但实际是 sampled SubTB over anchor-connected outgoing-expanded edge sets；STOP flow 由 learned stop head 直接对齐 reward。验证现在默认用 `val/selector_stop_flow@8/f1` 选模型，并同时报告 candidate coverage、union、trajectory diagnostic、calibration 和 stop 行为。

## 1. Code structure map

| 文件路径 | 类 / 函数 | 输入 | 输出 | 是否有参数 | 被谁调用 | 算法中的真实作用 |
|---|---|---|---|---|---|---|
| `src/train.py:56` | `main` | Hydra cfg | `trainer.fit` | 否 | CLI | 训练入口：构造 datamodule、model、trainer |
| `src/weaver/module.py:38` | `WeaverModule` | `RetrievalBatch` | loss / metrics | 是 | Lightning | 主训练与验证模块 |
| `src/weaver/module.py:150` | `compute_step` | batch | `StepOutput` | 否 | `training_step` | 构造 graph/target/features，采样 rollout，计算 objective |
| `src/data/schema/batch.py:39` | `RetrievalBatch` | PyG batched sample | 图、label、path tensors | 否 | dataloader/model | 样本协议；包含 anchors、reachable targets、precomputed shortest path labels |
| `src/weaver/context.py:32` | `GraphContext.from_batch` | `RetrievalBatch` | static graph context | 否 | module/rollout | 无标签图上下文：`edge_index`、node graph id、anchor mask、CSR adjacency |
| `src/weaver/context.py:110` | `TargetContext.from_batch` | batch + graph | reachable target mask/count | 否 | module/reward/eval | label context，只用 `reachable_target_node_ids` |
| `src/weaver/state.py:35` | `State` | graph ids + masks | dynamic state | 否 | rollout/loss/replay | 状态：selected edge set、active node cache、depth |
| `src/weaver/state.py:172` | `State.frontier` | graph, budget | `Frontier(row_ids, edge_ids)` | 否 | rollout/policy/loss | 枚举 active source outgoing legal expansion edges；STOP 不在里面 |
| `src/weaver/state.py:264` | `State.expand` | rows, edge_ids | child `State` | 否 | rollout/replay | 选择 edge，加入 endpoints，`step + 1` |
| `src/weaver/nn/feature_encoder.py:35` | `FeatureEncoder` | batch catalog ids + question emb | `EncodedFeatures` | 是 | module | 把 PLM semantic features 投影到 Weaver model space |
| `src/weaver/nn/state_encoder.py:87` | `StateEncoder` | features + state + graph | state/query/node/edge encodings | 是 | policy | mean-pool active nodes 和 selected edges，生成 row state |
| `src/weaver/nn/edge_encoder.py:7` | `EdgeEncoder` | src/rel/dst embeddings | concat edge token | 否 | state/policy encoder | edge token = `[h_src, h_rel, h_dst]` |
| `src/weaver/policy/forward.py:16` | `ForwardPolicy` | features/state/context/frontier | `PolicyOutput` | 是 | rollout/module | 输出 STOP logit、edge logits、derived state log flow |
| `src/weaver/policy/output.py:80` | `stop_log_prob` | `PolicyOutput` | `[R]` | 否 | sampler/loss/eval diag | `stop_log_flow - state_log_flow` |
| `src/weaver/rollout/engine.py:16` | `RolloutEngine` | policy/context/features | `RolloutResult` | 否 | runner | finite-horizon sampler；budget/forced stop 在这里 |
| `src/weaver/rollout/sampler.py:10` | `sample_step` | `PolicyOutput`, rows, temperature | `StepAction` | 否 | engine | Gumbel argmax over STOP + frontier edges |
| `src/weaver/rollout/result.py:9` | `RolloutResult` | trajectory tensors | masks/logprob props | 否 | eval/replay | 保存 selected edges、stop step、forced stop |
| `src/weaver/rollout/replay.py:133` | `training_from_rollouts` | rollouts + graph | `TrainingBatch` | 否 | runner | 把采样轨迹还原成 expansion/terminal transitions |
| `src/weaver/rollout/replay.py:254` | `replay_trajectories` | batch labels + rollouts | oracle shortest path trajectories | 否 | `ReplaySource` | 对未命中 graph 生成 shortest-path replay |
| `src/weaver/utility/reward.py:26` | `TrueTerminalReward` | terminal state + target context | `RewardOutput` | 超参，无训练参数 | module/loss/eval | terminal label reward |
| `src/weaver/objectives/subtb.py:47` | `SubTBLoss` | `SubTBInput` | loss/metrics | 超参，无训练参数 | module | sampled trajectory SubTB residual |
| `src/weaver/policy/backward.py:29` | `UniformValidPredecessorBackwardPolicy` | child state + action edge | log P_B | 否 | module | hard-coded backward policy，不学习 |
| `src/eval/rollout.py:45` | `evaluate_rollout_samples` | rollouts + batch | metrics | 否 | `WeaverMetricSuite` | validation/test retrieval metrics |

当前没有主路径文件 `src/weaver/loss.py`、`src/weaver/objective.py`、`src/weaver/transitions.py`；当前实现对应 `src/weaver/objectives/subtb.py` 与 `src/weaver/transition.py`。

## 2. Actual execution path

### 训练路径

1. `src/train.py:56` 通过 Hydra 读取 `configs/train.yaml`，默认 model 是 `configs/model/weaver.yaml`。
2. dataloader 产出 `RetrievalBatch`。核心图字段来自 `src/data/schema/batch.py:43-73`，question embedding 在 collator 中 stack，`edge_batch` 由 source node graph id 构造，见 `src/data/collate.py:51-53`。
3. `WeaverModule.compute_step` 构造：
   - 静态图上下文 `GraphContext.from_batch`：`edge_index,node_to_graph,edge_to_graph,anchor_mask,adjacency`，见 `src/weaver/context.py:58-94`。
   - 监督上下文 `TargetContext.from_batch`：只读取 `batch.reachable_target_node_ids`，见 `src/weaver/context.py:134-150`。
   - features：`FeatureEncoder.forward` 产出 node/relation/query semantic 和 model tensors，见 `src/weaver/nn/feature_encoder.py:106-124`。
4. `sample_train_rollout` 在 `torch.no_grad()` 下调用 runner，见 `src/weaver/module.py:272-286`。
5. `RolloutRunner.train_rollouts` 按 replay schedule 分配 policy rollout 与 replay，默认 8 个训练 rollout 中 policy/replay 权重 1:1，见 `src/weaver/rollout/runner.py:112-130` 和 `configs/model/weaver.yaml:43-57`。
6. policy rollout：
   - `State.initial`：每个 graph/rollout row 从 anchors 初始化 active nodes，selected edges 全 false，step=0，见 `src/weaver/state.py:59-87`。
   - 每步 `State.frontier` 枚举 active source outgoing edges。
   - `ForwardPolicy.forward` 对当前 state/frontier 输出 logits/flows。
   - `sample_step` 用 `stop_log_flow/edge_log_flow` 除以 temperature 后 Gumbel argmax，见 `src/weaver/rollout/sampler.py:74-91`。
   - 采样用 tempered logits，但 `StepAction.log_prob` 用未加 temperature 的 `gather_action_log_prob`，见 `src/weaver/rollout/sampler.py:40-44`。
   - expansion 调用 `State.expand`，STOP 写入 tape。
7. forced stop：如果无 frontier 或 `depth >= expand_budget`，engine 不调用 policy sampling，直接写 `STOP_EDGE_ID=-1` 且 log_prob=0，见 `src/weaver/rollout/engine.py:82-131` 和 `src/weaver/rollout/action.py:43-72`。
8. replay：
   - `training_from_rollouts` 把 sampled rollouts 还原为 `ExpansionBatch` 和 `TerminalBatch`，见 `src/weaver/rollout/replay.py:133-251`。
   - `ReplaySource` 对 policy 没命中的 graph，用 precomputed shortest path labels 生成 replay trajectories，见 `src/weaver/rollout/replay.py:254-317`。
   - replay trajectories 再进入同一个 `TrainingBatch`，不是 CE label。
9. loss 前重新 forward：
   - 对 expansion parent/child states 分别算 policy output，见 `src/weaver/module.py:196-224`。
   - 对 terminal state 算 policy output 和 reward，见 `src/weaver/module.py:237-253`。
   - `build_subtb_input` 抽取 `parent_log_flow, child_log_flow, action_log_prob, backward_log_prob, terminal_log_reward, terminal_stop_log_prob`，见 `src/weaver/objectives/subtb.py:118-156`。
10. `SubTBLoss.forward` 枚举同一 trajectory 内连续 subtrajectory residual，做 Huber/MSE 加权均值，见 `src/weaver/objectives/subtb.py:64-115`。
11. optimizer 是手动优化：`zero_grad -> backward -> clip -> step`，见 `src/weaver/module.py:129-134`。

### 验证 / 评估路径

1. `validation_step/test_step` 调 `eval_step`，见 `src/weaver/module.py:288-348`。
2. eval 只采样 `runner.eval_rollouts`，不构造 loss，不用 replay。
3. metrics 用 `SubgraphReconstructor` 从 trajectory 重建 terminal node/edge masks，见 `src/weaver/rollout/subgraph.py:49-124`。
4. validation 默认监控 `val/selector_stop_flow@8/f1`，见 `configs/callbacks/train.yaml:7-18`，不是 loss。

静态图对象：`RetrievalBatch.edge_index/batch/ptr/catalog ids`、`GraphContext`、`EncodedFeatures`。动态 state：`State.selected_edge_mask/active_node_mask/step`、`RolloutTape`、`RolloutResult`。监督信号：`reachable_target_node_ids` 用于 reward/eval，shortest-path distance/count tensors 用于 replay path construction。

## 3. Mathematical algorithm reconstructed from code

- 输入样本 `x` = `RetrievalBatch`，字段包括 `edge_index, batch, ptr, question_emb, anchor_node_ids, target_node_ids, reachable_target_node_ids, node_target_*`；对应 `src/data/schema/batch.py:43-73`。
- local graph `G` = `GraphContext(edge_index,node_to_graph,edge_to_graph,anchor_mask,adjacency)`；对应 `src/weaver/context.py:48-56`。
- anchor set `A_g` = `batch.anchor_node_ids` 写入 `GraphContext.anchor_mask`；对应 `src/weaver/context.py:71-79`。
- answer/target set `Y_g` = `batch.reachable_target_node_ids`，不是 full `target_node_ids`；对应 `src/weaver/context.py:134-145`。
- state `z` = `(graph_id, S_E, X_V, t)`，对应 `State.graph_ids, selected_edge_mask, active_node_mask, step`；对应 `src/weaver/state.py:54-57`。
- action space `C(z)` = `{STOP} union Frontier(z)`。
- `Frontier(z)` = `{e=(u,v): u in X or v in X, e not in S, edge_to_graph[e]=g, t<budget}`；对应 `src/weaver/state.py:181-187`。
- STOP action = `edge_id=-1`；对应 `src/weaver/rollout/action.py:8`。
- transition `T(z,e)` = `S'=S union {e}`, `X'=X union {src(e),dst(e)}`, `t'=t+1`；对应 `src/weaver/state.py:282-309`。
- terminal reward：

```text
log R(z) =
  [log(epsilon + answer_weight * |X cap Y| / |Y|)
   - edge_cost * |S|
   - fail_cost * 1[|X cap Y| = 0]]
  / log_reward_scale
```

对应 `src/weaver/utility/reward.py:80-97`。

- policy：

```text
q_stop(z) = MLP([h_q, h_z])
q_edge(z,e) = MLP([h_q, h_z, h_e, 1[src in X], 1[dst in X]])
F(z) = logsumexp(q_stop(z), {q_edge(z,e): e in Frontier(z)})
pi(a|z) = exp(q(a,z) - F(z))
```

对应 `src/weaver/policy/forward.py:101-132` 与 `src/weaver/policy/output.py:67-81`。

- loss `L` = sampled SubTB。terminal 终点用 `log_reward`，nonterminal 终点用 `child_log_flow`；对应 `src/weaver/objectives/subtb.py:197-218`。
- 代码中没有 learned backward policy 参数；`P_B` 是 hard-coded uniform valid predecessor，见 `src/weaver/policy/backward.py:29-66`。

## 4. State and frontier audit

`State` 由 4 个 tensor 定义：`graph_ids [R]`、`selected_edge_mask [R,E]`、`active_node_mask [R,N]`、`step [R]`，见 `src/weaver/state.py:54-57`。

`selected_edge_mask` 是 canonical evidence edge set；`active_node_mask` 是维护出来的 cache，语义上应等于 anchors 加 selected edge endpoints，见 `src/weaver/state.py:44-49`。因此 node/edge 没有完全独立的双真值源，但 active node cache 与 selected edges 可以在非法构造 `State` 时不一致；代码没有 invariant checker。evaluation 又从 `RolloutResult` 重建 node/edge masks，见 `src/weaver/rollout/subgraph.py:66-123`，这是另一套下游重建视图。

active nodes = initial anchors for graph row，加所有 selected edge endpoints。selected edges = `selected_edge_mask.nonzero()`，见 `src/weaver/state.py:117-124`。

frontier 是 outgoing physical directed frontier：只枚举 `src in active` 的原始 KG 边，不制造 inverse edge，见 `src/weaver/state.py:178-191`。duplicate edge 被 `unique_row_edge_pairs` 去重，见 `src/weaver/state.py:243-247`。同一 edge 重选被 `~selected_edge_mask` 禁止，见 `src/weaver/state.py:234-239`。cycle 没有被禁止；物理 inverse edge 如果存在于 KG 中，也没有被禁止。

最小伪代码：

```python
def frontier(state, graph, budget):
    if budget is not None and all(state.step >= budget):
        return []
    pairs = []
    for row, node in nonzero(state.active_node_mask):
        for e in outgoing_edges_by_src[node]:
            pairs.append((row, e))
    pairs = [(r, e) for r, e in pairs if graph.edge_to_graph[e] == state.graph_ids[r]]
    pairs = [(r, e) for r, e in pairs if not state.selected_edge_mask[r, e]]
    pairs = unique_sorted_by_key(r * E + e)
    if budget is not None:
        pairs = [(r, e) for r, e in pairs if state.step[r] < budget]
    return pairs

def expand(state, rows, edge_ids):
    child = clone(state)
    child.selected_edge_mask[rows, edge_ids] = True
    src = graph.edge_index[0, edge_ids]
    dst = graph.edge_index[1, edge_ids]
    child.active_node_mask[rows, src] = True
    child.active_node_mask[rows, dst] = True
    child.step[rows] += 1
    return child
```

这个 transition system 表达的是 anchor-connected physical edge set / subgraph，不是 path、不是 tree、不是 DAG。状态本身对边顺序不敏感，因为 `selected_edge_mask` 是集合；训练 loss 对轨迹顺序敏感，因为 `SampleMeta.step_ids` 会按顺序组装 subtrajectory，见 `src/weaver/objectives/subtb.py:175-180`。它对 edge-set inclusion state graph 是闭包的：合法 expansion 后仍是同一类 state；但它不是 DAG-GFlowNet 的 DAG object space。

## 5. Policy audit

policy 输出 `stop_log_flow [R]`、`edge_log_flow [F]`、`state_log_flow [R]`、`edge_row_ids [F]`、`edge_ids [F]`，见 `src/weaver/policy/output.py:23-29`。变量名叫 `*_log_flow`，但 `stop_head` 和 `edge_head` 实际直接输出未归一化 scalar logits；`state_log_flow` 是对这些 logits 做 `logsumexp` 得到的 derived value，见 `src/weaver/policy/forward.py:33-43` 和 `src/weaver/policy/forward.py:134-161`。

STOP 是同一 action space 里的 terminal action，但不在 `Frontier` 数据结构里；概率归一化时它和 frontier edges 一起进入 `logsumexp`。edge scorer 输入是 query encoding、row state encoding、edge token `[src,rel,dst]`、`src_active/dst_active` 两个 bit，见 `src/weaver/policy/forward.py:110-130`。state encoder 输入不含 reward，只含 features/state/context，见 `src/weaver/nn/state_encoder.py:135-181`。reward 没有进入 policy forward；reward 只在 loss terminal 项中使用，见 `src/weaver/module.py:248-263`。

实际概率公式：

```text
F(z) = logsumexp(q_stop(z), q_e1(z), ..., q_em(z))
P(STOP|z) = exp(q_stop(z) - F(z))
P(e|z) = exp(q_e(z) - F(z))
```

frontier 大小会系统性影响 STOP probability：在 logits 同分布或相近时，`P(STOP)` 约为 `1 / (1 + |frontier|)`；代码没有 degree correction。证据是 `state_log_flow = logsumexp(stop + all edge logits)`，见 `src/weaver/policy/forward.py:145-161`。

learned terminal flow / reward terminal flow 的关系：STOP head 是 learned terminal action log-flow；reward 不参与 forward，terminal residual 中 `state_log_flow + stop_log_prob = stop_log_flow`，所以 STOP head 被直接拉向 `log_reward`，见 `src/weaver/objectives/subtb.py:147-149` 和 `src/weaver/objectives/subtb.py:205-211`。

## 6. Reward audit

reward 代码只在 terminal states 上被主训练调用：`terminals.state` 传给 `reward_model`，见 `src/weaver/module.py:248-253`。`TrueTerminalReward.forward` 本身可对任意 `State` 计算，但训练路径只传 terminal batch。reward 依赖 gold/reachable answer；它读取 `TargetContext.target_mask` 和 `target_count_by_graph`，见 `src/weaver/utility/reward.py:80-87`。

公式：

```text
supported = |active_node_mask cap reachable_target_mask|
recall = supported / max(reachable_target_count, 1)
raw_log_R = log(epsilon + answer_weight * recall)
            - edge_cost * selected_edge_count
            - fail_cost * 1[supported == 0]
log_R = raw_log_R / log_reward_scale
```

对应 `src/weaver/utility/reward.py:84-97`。分母是 reachable answer count，不是 full answer count；`TargetContext` 明确使用 `batch.reachable_target_node_ids`，见 `src/weaver/context.py:134`。reward 在真实推理时不可用，因为需要 target labels；eval 里也只是用于 label metric/log_reward 计算。

这个 reward 鼓励 answer hit + edge penalty + no-answer penalty。它没有显式最小充分证据约束、路径正确性约束、关系语义约束或 answer justification 约束；最小性只通过 `edge_cost * |S_E|` 间接体现。

## 7. Loss / objective audit

当前 loss 是 sampled SubTB loss，不是 full frontier flow matching，不是纯 DB，不是纯 TB。证据：`SubTBLoss.forward -> subtrajectory_terms`，见 `src/weaver/objectives/subtb.py:64-80`；并且 `state_minus_action_lse` 被显式标为 undefined for action-log-flow parameterization，见 `src/weaver/objectives/diagnostics.py:24-26`。

`build_subtb_input` 张量 shape：

| 字段 | shape | 代码来源 |
|---|---:|---|
| `parent_log_flow` | `[M_exp]` | `log_flow(parent_out)` |
| `child_log_flow` | `[M_exp]` | `log_flow(child_out)` |
| `action_log_prob` | `[M_exp]` | chosen expansion edge log prob |
| `backward_log_prob` | `[M_exp]` | uniform valid predecessor |
| `terminal_log_reward` | `[M_term]` | reward output |
| `terminal_parent_log_flow` | `[M_term]` | terminal state flow |
| `terminal_action_log_prob` | `[M_term]` | STOP log prob |

对应 `src/weaver/objectives/subtb.py:138-156`。

代码 residual：

```text
running_forward += log P_F(a_t | z_t)
running_backward += log P_B(z_t | z_{t+1})  # only nonterminal expansion events

if end event is terminal:
    residual = F(z_start) + running_forward - log_R(z_end) - running_backward
else:
    residual = F(z_start) + running_forward - F(z_end) - running_backward

loss = weighted mean Huber(residual), weight = lambda^(length - 1)
```

对应 `src/weaver/objectives/subtb.py:197-218` 和 `src/weaver/objectives/subtb.py:290-305`。

loss 需要 state flow，但 state flow 不是独立 head，而是 action logits 的 `logsumexp`。loss 不枚举所有 possible states，只枚举 sampled/replay trajectories 内 states；对每个 sampled state 的 policy forward 会枚举该 state 的 full frontier 用于 normalization。STOP transition 和 expand transition 不完全对称：expand 用 chosen edge log prob + child state flow + backward log prob；terminal STOP 用 STOP log prob + reward，没有 backward term，见 `src/weaver/objectives/subtb.py:205-218`。

backward policy 真实存在为 hard-coded uniform count，不学习，见 `src/weaver/policy/backward.py:61-66`。`log P_B` 只进入 loss residual，不进入 forward policy，见 `src/weaver/module.py:219-224`。

detach / stop-gradient：reward model forward 被 `@torch.no_grad()` 包裹，见 `src/weaver/utility/reward.py:71`；rollout sampling 也在 `torch.no_grad()` 下，见 `src/weaver/module.py:279`。loss residual 本身没有 detach；metrics 才 detach，见 `src/weaver/objectives/subtb.py:84-109`。

`lambda` 退化：`lambda=0` 时只有 length=1 residual 权重大于 0，等价 local DB-style terms；`lambda=1` 时所有连续 subtrajectory 等权；`max_len` 限制最长 subtrajectory，见 `src/weaver/objectives/subtb.py:188-204`。

## 8. Rollout / replay / evaluation audit

behavior policy 和 trained policy 不完全一致：采样用 logits / temperature，默认 `train_temperature=0.7`，见 `src/weaver/rollout/sampler.py:74-81` 和 `configs/model/weaver.yaml:81`；但存储的 action log prob 是未加 temperature 的 policy log prob，见 `src/weaver/rollout/sampler.py:40-44`。代码中未找到 epsilon-greedy。forced stop 存在：无 frontier 或 budget exhausted，见 `src/weaver/rollout/engine.py:156-171`。budget stop 存在：`expand_budget=3` 默认，见 `configs/model/weaver.yaml:42`。

replay 生成完整 shortest-path trajectories，并转成 expansion/terminal transitions；不是直接 CE 监督 next edge。`precomputed_shortest_edge_path` 从 best anchor 沿 outgoing edge 走向 target，见 `src/weaver/rollout/replay.py:706-743`。replay 只对 policy rollouts 未命中的 graph 生成，见 `src/weaver/rollout/replay.py:384-401`。

validation metric 现在拆成 candidate coverage、terminal selector、calibration 和 stop 行为；checkpoint/early stopping 监控 `val/selector_stop_flow@8/f1`，见 `configs/callbacks/train.yaml:7-18`。`candidate_oracle_best@k` 用真实 eval metric 选候选池上限；`candidate_reward_best@k` 用 label-dependent terminal reward 选 reward oracle；`selector_traj_prob@k` 只诊断 trajectory log probability；`selector_stop_flow@k` 用 `log_flow + log P(STOP)` 作为 terminal graph selector；`candidate_union@k` 是前 k 个 rollout 的 node/edge union object。forced stop rate 来自 terminal action 是否 forced。

## 9. Minimal pseudocode of the actual algorithm

```python
for batch in train_loader:
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    features = policy_feature_encoder(batch)

    budget = allocate_replay_budget(
        total=train_num_rollouts,
        policy_weight=1.0,
        replay_weight=1.0,
    )

    policy_rollouts = []
    for rollout_id in range(budget.policy_rollout):
        graph_ids = arange(graph.num_graphs)
        state = State.initial(graph, graph_ids)
        tape = RolloutTape(R=graph.num_graphs, T=expand_budget + 1)

        for t in range(expand_budget + 1):
            active_rows = rows_where_not_stopped(tape)
            if active_rows.empty:
                break

            active_state = state.select_rows(active_rows)
            frontier = active_state.frontier(graph, expand_budget=expand_budget)
            forced_local = rows_without_frontier_or_at_budget(
                active_state, frontier, expand_budget
            )

            sample_rows = active_rows - active_rows[forced_local]

            sampled_action = empty_action()
            if sample_rows.nonempty:
                sample_state = state.select_rows(sample_rows)
                sample_frontier = sample_state.frontier(graph, expand_budget)
                out = policy(
                    features=features,
                    state=sample_state,
                    context=graph,
                    frontier=sample_frontier,
                )

                # candidates are STOP plus all frontier edges for each row
                logits = concat(out.stop_log_flow, out.edge_log_flow) / temperature
                picked_edge_ids = segment_gumbel_argmax(logits)
                log_prob = gather_action_log_prob(out, rows, picked_edge_ids)

                sampled_action = StepAction(
                    row_ids=sample_rows,
                    edge_ids=picked_edge_ids,
                    log_prob=log_prob,
                    forced=False,
                )

            forced_action = StepAction.forced_stop(
                rows=active_rows[forced_local],
                log_prob=0.0,
                forced=True,
            )

            action = concat_and_sort(sampled_action, forced_action)
            tape.write(t, action)

            if any(action.edge_ids >= 0):
                state = state.expand(
                    graph,
                    rows=action.expand_rows,
                    edge_ids=action.expand_edge_ids,
                )

        policy_rollouts.append(tape.to_rollout_result())

    replay = None
    if budget.replay_expand > 0:
        trajectories = []
        for graph_id with reachable targets and not hit_by(policy_rollouts):
            for target_node in reachable_targets[graph_id]:
                path = precomputed_shortest_edge_path(
                    batch, graph, target_node, expand_budget
                )
                if path reaches target_node:
                    trajectories.append(
                        ReplayTrajectory(graph_id, path[:expand_budget])
                    )
        replay = sample_at_most(trajectories, budget.replay_expand)

    training_parts = []
    training_parts.append(training_from_rollouts(policy_rollouts, graph))
    if replay:
        training_parts.append(training_from_trajectories(replay, graph))
    training = concat_reindex_trajectories(training_parts)

    parent_frontier = training.expansions.parent.frontier(graph, expand_budget)
    child_frontier = training.expansions.child.frontier(graph, expand_budget)
    terminal_frontier = training.terminals.state.frontier(graph, expand_budget)

    parent_out = policy(features, training.expansions.parent, graph, parent_frontier)
    child_out = policy(features, training.expansions.child, graph, child_frontier)
    terminal_out = policy(features, training.terminals.state, graph, terminal_frontier)

    backward_log_prob = -log(valid_predecessor_count(training.expansions.child))
    reward_out = TrueTerminalReward(training.terminals.state, graph, target)

    x = SubTBInput(
        parent_log_flow=parent_out.state_log_flow,
        child_log_flow=child_out.state_log_flow,
        action_log_prob=log_prob_of_chosen_expansion(parent_out),
        backward_log_prob=backward_log_prob,
        terminal_log_reward=reward_out.log_reward,
        terminal_parent_log_flow=terminal_out.state_log_flow,
        terminal_action_log_prob=terminal_out.stop_log_flow
            - terminal_out.state_log_flow,
        trajectory_ids=training.meta.trajectory_ids,
        step_ids=training.meta.step_ids,
    )

    events = sort_events_by_trajectory_then_step(x)
    residuals = []
    for each contiguous subtrajectory i..j:
        running_forward = sum(log_PF[a_k | z_k])
        running_backward = sum(log_PB[z_k | z_{k+1}] for expansion events)
        if event_j is terminal:
            residual = F(z_i) + running_forward - log_R(z_j) - running_backward
        else:
            residual = F(z_i) + running_forward - F(z_j) - running_backward
        residuals.append(lambda_ ** (length - 1) * huber(residual))

    loss = source_balanced_mean(policy_residuals, replay_residuals)
    backward(loss)
    optimizer.step()
```

## 10. Mismatch table

| Claim / intended algorithm | Actual code behavior | Evidence in code | Consequence | Severity | Minimal fix direction |
|---|---|---|---|---|---|
| DAG-GFlowNet | Generated object 是 anchor-connected edge set/subgraph；cycles allowed | `src/weaver/state.py:234-247` 只禁止重复 edge | DAG theory claims invalid | major | 重命名算法或强制 DAG 约束 |
| Edge flow matching | `state_minus_action_lse` undefined；objective 是 SubTB | `src/weaver/objectives/diagnostics.py:24-26`, `src/weaver/objectives/subtb.py:47` | `edge_flow_matching` tag 误导 | minor | 重命名 configs/metrics |
| STOP separate branch | STOP 是 normalization 中的普通 terminal action，但不在 `Frontier` | `src/weaver/state.py:16-20`, `src/weaver/policy/output.py:80-85` | 高 degree state 系统性压低 STOP | major | Degree-normalize STOP 或独立建模 stop hazard |
| Reward as terminal flow | Terminal residual 使 `stop_log_flow` 直接匹配 `log_reward` | `src/weaver/objectives/subtb.py:205-211` | STOP head 学 reward scale | major | 分离 terminal reward calibration 或明确 terminal-flow 参数化 |
| Learned backward policy | Backward 是 hard-coded uniform valid predecessor | `src/weaver/policy/backward.py:29-66` | 没有 learned P_B | minor | 重命名为 fixed backward kernel |
| Directed KG traversal | Frontier 只包含 active source 的 outgoing 原始边 | `src/weaver/state.py:200-204` | relation direction 与 replay path 现在对齐 | major | 评估 outgoing-only 对 coverage / recall 的影响 |
| Replay as oracle CE | Replay transitions 进入同一个 SubTB，无 CE | `src/weaver/rollout/runner.py:216-221`, `src/weaver/objectives/subtb.py:75-80` | Oracle path signal 是间接的 | minor | 如有需要添加 auxiliary CE |
| Behavior policy equals learned policy | Sampling uses temperature，stored/eval logprob uses untempered policy | `src/weaver/rollout/sampler.py:74-81`, `src/weaver/rollout/sampler.py:40-44` | `temperature != 1` 时 off-policy | major | 存 behavior logprob 或设 `temperature=1` |
| Budget-free stopping | Budget forced stop exists；默认 `expand_budget=3` | `src/weaver/rollout/engine.py:170-171`, `configs/model/weaver.yaml:42` | 模型可能学 expand-to-budget | major | 跟踪并惩罚 forced stop；调 budget |
| Validation aligns with loss | Checkpoint monitors `selector_stop_flow@8/f1` | `configs/callbacks/train.yaml:7` | best checkpoint 侧重 deploy-time terminal selection，不直接最小化 SubTB loss | minor | 同时报 coverage、selector、calibration 和 loss 指标 |

## 11. Load-bearing assumptions

| assumption | falsifiable variable | dataset / split | expected failure direction | how to test in this repo |
|---|---|---|---|---|
| Reachable targets 是正确训练分母 | `reachable_target_node_ids` coverage vs `target_node_ids` | WebQSP train/val/test | reward overestimates quality when unreachable answers are excluded | 比较 `use_reachable_targets=true/false` eval |
| Outgoing-only expansion 可能损失可逆证据覆盖 | reachable target recall / forced stop rate | WebQSP val | 无法借由 incoming hit 到某些答案邻域 | 对比 outgoing-only vs incident frontier |
| STOP logit 能克服 frontier degree | frontier size vs `stop_prob` | WebQSP val | high-degree states stop too late | 记录 `frontier.num_edges` 与 `stop_prob` |
| Budget 3 足够覆盖 answer paths | shortest path length to reachable targets | WebQSP train/val | forced stop rate high, recall ceiling low | 用 `anchor_node_forward_distances_flat` / replay path length histogram |
| Replay shortest paths 有代表性 | replay fraction and replay hit graphs | WebQSP train | model overfits shortest outgoing paths | ablate `replay_expand=0` vs default |
| Edge penalty scale 合理 | `edge_cost/log_reward_scale` | WebQSP val | too small: expand-to-budget；too large: early STOP | sweep `edge_cost`，监控 forced stop/reward/recall |
| Terminal score 能排序有用 rollouts | `selector_stop_flow@k` gap to `candidate_oracle_best@k` | WebQSP val | high oracle recall but low stop_flow recall | 使用 `selector_stop_flow@k/oracle_gap` 和 calibration metrics |
| Temperature off-policy effect 可忽略 | `train_temperature`, `eval_temperature` | WebQSP train/val | `tau < 1` 降低 diversity 或 bias replay coverage | 跑 `tau=1.0` vs `0.7` 同 seed |

## 12. What this implementation breaks

相比标准 GFlowNet，这个实现改变或破坏了这些条件：

- Object space 不是 DAG objects，而是 edge sets with anchor-connected outgoing expansion。
- Forward action space 包含 learned STOP terminal action，但 STOP probability 与 variable-size frontier 一起归一化，产生 degree dependence。
- 训练是 sampled SubTB over policy/replay trajectories，不是 exhaustive flow matching over all transitions。
- Backward policy 是 fixed uniform valid predecessor，不学习。
- 当 temperature 不等于 1 时，rollout collection 是 off-policy。
- Terminal reward 是 supervised label reward，推理时不可用。

相比普通 KGQA retrieval，这个实现新增的可验证能力是：随机生成多个 compact-ish anchor-connected evidence subgraphs，并支持 diversity、candidate coverage、terminal selector calibration 和 union evaluation。如果实验不能证明 `selector_stop_flow@k`、`candidate_union@k` 或 downstream LLM 指标超过 shortest-path/BFS baselines，那么这套实现主要是在 answer hit + edge penalty reward 外包了一层 GFN-style trajectory consistency，复杂化了 retrieval。

## 13. Final verdict

当前代码实际算法的一句话定义：一个用 action-log-flow 参数化的 sampled SubTB 训练器，在 KG local graph 上从 anchors 生成 bounded anchor-connected outgoing edge subgraphs，terminal STOP flow 用 reachable-answer reward 监督。

值得继续修，但前提是先承认它不是 DAG-GFlowNet，也不是 edge flow matching。第一优先级应改 STOP/frontier/validation 三件事：校正 STOP 与 frontier degree 的耦合，验证 outgoing-only frontier 的覆盖代价，并持续同时监控 `selector_stop_flow@k`、`candidate_union@k`、calibration 和 loss。

应删除或重命名旧语义模块名和配置 tag：`edge_flow_matching`、DAG-GFlowNet 表述、旧 `loss/objective/transitions` 叫法；`state_log_flow` 可保留但文档必须写明它是 derived `logsumexp(action logits)`。

必须先跑的实验是：`tau=1` vs `0.7`、replay on/off、frontier incident vs outgoing、edge_cost sweep、budget sweep，以及 `candidate_oracle_best@k` 到 `selector_stop_flow@k` 的 gap；这些不跑，无法判断 GFN 部分是否带来超过 shortest-path replay 的收益。
