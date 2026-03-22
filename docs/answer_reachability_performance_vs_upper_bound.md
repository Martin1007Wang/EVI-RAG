# Answer Reachability Performance vs Theoretical Upper Bound

这份文档记录当前 `-sub` answer-reachability 训练结果、在现有搜索约束下的数据集理论上限，以及这些数字意味着什么。

## 1. 结论先说

- 对当前真正参与训练/验证的过滤后 `cwq-sub` 和 `webqsp-sub` 数据，图可达性本身几乎不是瓶颈。
- 在当前运行口径下，验证集和测试集里的 gold answer 都是 `4` 步内可达的，因此 `gold_mass` 的理论上限是 `1.0`。
- `hit@k` 的理论上限也都是 `1.0`，因为 oracle 可以把任一可达 gold answer 排到前 `k`。
- 真正限制 `recall@1/5/10` 上限的，不是图里没有答案，而是很多样本有多个 gold answers；`k` 固定时，不可能一次覆盖全部 gold。
- 当前模型和理论上限之间的主要差距，已经不是 feasibility gap，而是 ranking / mass allocation gap。

## 2. 这份上限是怎么定义的

这里的“理论上限”不是无约束上限，而是和当前训练/验证完全同口径的 oracle 上限：

- 数据使用当前 runtime filters 之后的真实 `-sub` 数据。
- 搜索约束使用当前模型的有向边遍历规则和 horizon `max_steps=4`。
- 评价口径使用当前 `answer/*` rank metrics 的定义。
- 对每个样本，假设存在一个 oracle，能把所有“4 步内可达的 gold answers”排在最前面。

对应代码语义：

- horizon 约束：`configs/dataset/cwq-sub.yaml`、`configs/dataset/webqsp-sub.yaml`
- 运行时过滤：`configs/dataset/base.yaml`
- 前向步约束：`src/models/gflownet/transitions.py`
- rank metric 定义：`src/metrics/answer_reachability/posterior.py`
- reachability analysis runtime：`src/metrics/answer_reachability/analysis.py`、`src/metrics/answer_reachability/flow_frontier.py`、`src/metrics/answer_reachability/monte_carlo.py`

## 3. 当前最终验证性能

以下数字来自 replay 版最终 run 的最新验证结果：

- `cwq-sub`: `logs/train_answer_reachability_cwq-sub/runs/2026-03-17_23-08-04/metrics/val.jsonl`
- `webqsp-sub`: `logs/train_answer_reachability_webqsp-sub/runs/2026-03-17_23-08-04/metrics/val.jsonl`

| Dataset | Final step | Effective pass | Gold mass | Selected mass | Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cwq-sub` | 26326 | 39.835 | 0.4811 | 0.9104 | 0.2412 | 0.4126 | 0.4705 | 0.1762 | 0.3688 | 0.4350 |
| `webqsp-sub` | 4920 | 60.299 | 0.6216 | 0.9146 | 0.2105 | 0.3684 | 0.4737 | 0.1207 | 0.2940 | 0.3856 |

训练侧的最终状态也说明模型已经学到了大量成功轨迹，但还没有把概率质量完全压到 gold answers 上：

- `cwq-sub`: `train/rollout_success ~= 0.742`, `train/success_replay_ratio ~= 0.147`
- `webqsp-sub`: `train/rollout_success ~= 0.945`, `train/success_replay_ratio ~= 0.200`

## 4. 验证集理论上限

### 4.1 Oracle 上限

| Dataset | Any gold in graph | All gold in graph | Any gold reachable <=4 | All gold reachable <=4 | Oracle gold mass | Oracle hit@1 | Oracle recall@1 | Oracle recall@5 | Oracle recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cwq-sub` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8652 | 0.9747 | 0.9903 |
| `webqsp-sub` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6331 | 0.9161 | 0.9656 |

这张表最重要的信息是：

- 当前保留下来的验证样本里，所有 gold answers 都在图里。
- 而且所有 gold answers 都在当前 horizon `4` 内可达。
- 所以当前 `gold_mass` 的理论上限就是 `1.0`，不是 `0.6`、`0.7` 之类的数据瓶颈值。

### 4.2 当前结果占上限的比例

| Dataset | Gold mass / upper | Hit@1 / upper | Recall@1 / upper | Recall@5 / upper | Recall@10 / upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cwq-sub` | 0.4811 | 0.2412 | 0.2037 | 0.3784 | 0.4392 |
| `webqsp-sub` | 0.6216 | 0.2105 | 0.1907 | 0.3210 | 0.3994 |

换句话说：

- `cwq-sub` 当前大约拿到了 `43.9%` 的 `recall@10` ceiling。
- `webqsp-sub` 当前大约拿到了 `39.9%` 的 `recall@10` ceiling。
- `webqsp-sub` 在 `gold_mass` 上更接近上限，但在 `recall@10` 上并没有明显更接近上限。

## 5. 测试集理论上限

测试集也呈现同样结论：过滤后的数据在当前约束下是完全可达的。

| Dataset | Any gold reachable <=4 | All gold reachable <=4 | Oracle gold mass | Oracle recall@1 | Oracle recall@5 | Oracle recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cwq-sub` | 1.0000 | 1.0000 | 1.0000 | 0.8799 | 0.9861 | 0.9965 |
| `webqsp-sub` | 1.0000 | 1.0000 | 1.0000 | 0.6472 | 0.9150 | 0.9571 |

因此，如果后续测试结果没有达到很高的 `gold_mass` / `recall@10`，主要也不应归因于图里没有答案或步数约束过紧。

## 6. 为什么 `recall@1` 的上限不是 1.0

原因不是不可达，而是多答案样本很多。

### 6.1 `cwq-sub` 验证集 gold answer 分布

- 平均每题 gold answer 数：`1.8610`
- 多答案样本占比：`19.06%`
- gold answer 数大于 `5` 的样本占比：`5.01%`
- gold answer 数大于 `10` 的样本占比：`2.80%`
- 最大 gold answer 数：`27`

### 6.2 `webqsp-sub` 验证集 gold answer 分布

- 平均每题 gold answer 数：`4.1491`
- 多答案样本占比：`51.75%`
- gold answer 数大于 `5` 的样本占比：`16.23%`
- gold answer 数大于 `10` 的样本占比：`7.46%`
- 最大 gold answer 数：`91`

这直接解释了为什么：

- `webqsp-sub` 的 `recall@1` ceiling 只有 `0.6331`
- 即使 oracle，`recall@5` 和 `recall@10` 也不会到 `1.0`

## 7. 这些数字意味着什么

### 7.1 数据可达性不是主矛盾

当前过滤后的 `-sub` 验证/测试数据里，gold answers 全部都在图中且 `4` 步内可达。说明：

- 不是 preprocessing 把答案裁掉了。
- 不是 horizon `4` 本身把答案截断了。
- 也不是“图里没路”导致性能天然封顶。

### 7.2 当前主矛盾是排序和质量分配

既然 oracle `gold_mass` 上限是 `1.0`，而当前只有：

- `cwq-sub`: `0.4811`
- `webqsp-sub`: `0.6216`

说明模型仍然没有把足够多的概率质量压到 gold answers 上。

同理，`recall@10` 距离 oracle ceiling 还很远，说明：

- 模型虽然越来越能 rollout 到成功终点，
- 但还没有把“正确答案排在前面”的能力学满。

### 7.3 Replay 已经在帮忙，但还没吃满 ceiling

从最终 run 看：

- `cwq-sub` 的提升尤其明显，说明成功经验回放确实缓解了稀疏成功轨迹问题。
- `webqsp-sub` 的 `gold_mass` 更高，说明它在质量集中度上更成熟。
- 但两者相对 ceiling 都还有大块空间，尤其 `recall@10` 只到 oracle 的大约 `40%` 左右。

所以 replay 是有效的，但还不是终局。

## 8. 当前最合理的下一步

如果目标是继续逼近理论上限，接下来的优先级应该是：

1. 继续提高 answer ranking 能力，而不是怀疑图可达性。
2. 继续优化 credit assignment / mass allocation，而不是先加大 horizon。
3. 针对 `webqsp-sub` 的多答案样本，重点看 top-k 排名是否能更稳定覆盖多个 gold answers。
4. 针对 `cwq-sub`，继续看 replay ratio、buffer 采样策略、以及 start / edge policy 的排序学习是否还能继续抬高 `gold_mass`。

## 9. 口径提醒

这份上限分析有两个重要前提：

- 它只针对当前 runtime filters 之后的真实训练/验证/测试子集，不是原始未过滤全集。
- 它是“当前搜索约束下”的 oracle 上限，不是允许改图、改步数、改任务定义之后的更松上限。

因此，这份文档最适合回答的问题是：

- 在当前系统定义下，我们离 ceiling 还有多远？
- 这个 ceiling 是数据造成的，还是模型造成的？

答案是：当前主要还是模型造成的。
