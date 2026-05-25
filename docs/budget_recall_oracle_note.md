# Budget-Recall Oracle Note

Date: 2026-05-25

## Current Decision

Set Weaver graph-generation budget to 8 for both training and validation/evaluation:

```yaml
model:
  budget: 8
```

The current runner already uses:

```yaml
train_policy_rollouts: 8
eval_rollouts: 8
```

The intended reporting protocol is to union the generated graphs from 8 stochastic rollouts. With `budget=8`, the union contains at most 64 generated edges per question before duplicate-edge collapse. This keeps the evaluation comparable to SubgraphRAG-style large retrieval windows without making a single rollout generate an overly large graph.

## Why Not Keep Budget 3

The budget-recall oracle analysis showed that `budget=3` is enough to hit at least one reachable answer on WebQSP, but it is not enough for strong multi-answer recall.

WebQSP `budget=3` oracle answer-entity recall:

| split | oracle hit rate | oracle answer recall mean | oracle full-cover rate |
|---|---:|---:|---:|
| train | 1.000 | 0.795 | 0.683 |
| validation | 1.000 | 0.775 | 0.638 |
| test | 1.000 | 0.783 | 0.663 |

Interpretation:

- If model hit rate is low at `budget=3`, the issue is policy/reward/training.
- If model answer recall cannot exceed about 0.78-0.80 at `budget=3`, that is close to the shortest-path oracle frontier, not just an optimization issue.
- Multi-answer questions are the main bottleneck.

## SubgraphRAG Comparison Target

ICLR 2025 SubgraphRAG reports retrieval recall directly, not just QA Hit/F1. It reports:

- shortest-path triple recall
- GPT-4o-labeled relevant triple recall
- answer entity recall
- wall-clock time

Main Table 1 values:

| Method | WebQSP SP triple R | WebQSP GPT-4o triple R | WebQSP answer entity R | CWQ SP triple R | CWQ GPT-4o triple R | CWQ answer entity R |
|---|---:|---:|---:|---:|---:|---:|
| SubgraphRAG | 0.883 | 0.865 | 0.944 | 0.811 | 0.840 | 0.914 |

SubgraphRAG uses a larger triple retrieval window, commonly top-100 triples for LLM input. Our method is graph generation, so the fair comparison should report both recall and generated graph size/time.

## Wide-Budget Oracle Result

The WebQSP wide-budget oracle was run to estimate how much budget is needed to match SubgraphRAG answer entity recall.

WebQSP test shortest-path oracle:

| max budget | oracle answer recall | full-cover rate | mean used edges |
|---:|---:|---:|---:|
| 8 | 0.916 | 0.853 | 3.27 |
| 10 | 0.932 | 0.880 | 3.56 |
| 12 | 0.941 | 0.891 | 3.80 |
| 16 | 0.955 | 0.913 | 4.20 |
| 24 | 0.971 | 0.935 | 4.82 |
| 32 | 0.981 | 0.951 | 5.29 |

`max_budget=16` is enough for the WebQSP answer-entity oracle to exceed SubgraphRAG's reported 0.944 answer entity recall. However, the current main configuration chooses `budget=8` because evaluation uses 8 stochastic rollouts and reports their union. This gives a union capacity of at most 64 edges while keeping each rollout small and trainable.

## Diagnostic Commands

Run the standard WebQSP train oracle:

```bash
python scripts/analyze_budget_recall_oracle.py \
  --metadata-dir /mnt/data/retrieval/webqsp/metadata \
  --splits train \
  --budgets 0,1,2,3,4,5,6,7,8 \
  --output-dir outputs/analysis/budget_recall_oracle/webqsp_train
```

Run the standard WebQSP validation/test oracle:

```bash
python scripts/analyze_budget_recall_oracle.py \
  --metadata-dir /mnt/data/retrieval/webqsp/metadata \
  --splits validation,test \
  --budgets 0,1,2,3,4,5,6,7,8 \
  --output-dir outputs/analysis/budget_recall_oracle/webqsp
```

Run the wide-budget WebQSP oracle:

```bash
python scripts/analyze_budget_recall_oracle.py \
  --metadata-dir /mnt/data/retrieval/webqsp/metadata \
  --splits validation,test \
  --budgets 0,1,2,3,4,5,6,8,10,12,16,24,32,48,64,100 \
  --max-dp-states 50000 \
  --output-dir outputs/analysis/budget_recall_oracle/webqsp_wide
```

Output files:

```text
budget_curve_summary.csv
budget_cover_summary.csv
per_sample_budget_curve.csv
per_sample_cover_stats.csv
run_config.json
```

The key columns are:

- `oracle_hit_rate`
- `oracle_recall_mean`
- `oracle_full_cover_rate`
- `mean_used_edges`
- `marginal_recall_gain`
- `reward_marginal_gain`
- `exact_sample_rate`
- `dp_fallback_sample_rate`

## Reporting Plan

For a paper-quality comparison with SubgraphRAG, report:

| Metric | Reason |
|---|---|
| Answer entity recall | Directly comparable to SubgraphRAG Table 1 |
| Shortest-path triple recall | Directly comparable to SubgraphRAG Table 1 |
| GPT-4o-labeled relevant triple recall | Directly comparable if labels are available/reproduced |
| Generated triple count | Shows the method is not relying on oversized graphs |
| Wall-clock time | Required for efficiency comparison |
| Union-of-rollouts recall | Captures stochastic generation diversity |

Recommended main evaluation setting:

```text
budget per rollout = 8
num stochastic rollouts = 8
reported graph = union of rollouts
maximum union size <= 64 edges before duplicate collapse
```

Recommended ablation:

```text
budget per rollout in {3, 4, 8, 16}
num rollouts in {1, 2, 4, 8}
report recall / generated edge count / time
```
