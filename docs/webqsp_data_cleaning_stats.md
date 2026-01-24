# WebQSP 数据清洗策略统计（基于 2026-01-24 pipeline 日志）

以下统计来自你提供的 `pipeline_start` 日志片段，仅做结构化整理与关键指标汇总。

## 清洗策略概览

| 项目 | 值 |
| --- | --- |
| 数据集 | webqsp |
| 数据源 | hf (`rmanluo/RoG-webqsp`) |
| path_mode | `qa_directed` |
| 去重 | `dedup_edges=True` |
| 去自环 | `remove_self_loops=True` |
| 关系清洗 | `relation_cleaning=True` |
| 时间关系过滤 | `question_gated`（按问题正则 gate） |
| 目标可达裁剪 | `target_reachable_pruning=True` |
| 分割 | train / validation / test |

## 词表与关系统计

| 指标 | 数值 |
| --- | ---: |
| 实体总数 | 1,193,175 |
| 文本实体数 | 506,981 |
| 非文本实体数 | 686,194 |
| 关系总数 | 11,976 |
| 逆关系加载 | 5,988（suffix=`__inv`） |

## 关系清洗统计

| 指标 | 数值 |
| --- | ---: |
| dropped_relation_types | 44 |
| kept_relation_types | 5,988 |
| type_relation_types | 33 |

## 全局边统计（清洗后）

| 指标 | 数值 |
| --- | ---: |
| raw_edges | 19,986,134 |
| kept_edges | 15,241,133 |
| dropped_edges | 2,308,744 |
| self_loop_edges | 397,187 |
| type_edges | 2,039,070 |
| type_orphan_edges | 185,010 |

## 样本清洗（parquet 阶段）

| split | samples_total | samples_kept | kept_ratio | dropped_no_path | samples_empty_graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2,826 | 2,711 | 95.93% | 115 | 0 |
| validation | 246 | 234 | 95.12% | 11 | 1 |
| test | 1,628 | 1,555 | 95.52% | 73 | 0 |

> empty_graph_examples: `webqsp/validation/WebQTrn-1466`

## 目标可达裁剪（target_reachable_pruning）

| split | dropped |
| --- | ---: |
| train | 1 |
| validation | 0 |
| test | 0 |

裁剪后写入图与问题数量：train=2710, validation=234, test=1555。

## sub_filter（子集数据筛选，按缺失答案过滤）

| split | total | kept | filtered | kept_ratio | missing_a_any | missing_a_rate | no_path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,710 | 2,606 | 104 | 96.16% | 104 | 3.84% | 0 |
| validation | 234 | 227 | 7 | 97.01% | 7 | 2.99% | 0 |
| test | 1,555 | 1,499 | 56 | 96.40% | 56 | 3.60% | 0 |

补充：`unreachable_a_any=0`（三段均为 0）。

## LMDB 写入统计

| split | samples | avg_edges | avg_nodes | edges | nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| total | 4,499 | 4,640.31 | 593.62 | 20,876,742 | 2,670,682 |
| train | 2,710 | 4,613.36 | 590.29 | 12,502,214 | 1,599,689 |
| validation | 234 | 4,466.44 | 594.53 | 1,045,146 | 139,120 |
| test | 1,555 | 4,713.43 | 599.28 | 7,329,382 | 931,873 |

## Anchor 过滤结果

| 指标 | 数值 |
| --- | ---: |
| keep_answer | 4,499 |
| keep_start | 4,499 |
| missing_answer | 0 |
| missing_start | 0 |
