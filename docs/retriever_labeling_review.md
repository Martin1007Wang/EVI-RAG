# Retriever Labeling Review (WebQSP)

This document summarizes the retriever labeling objective, the downstream
GFlowNet interface, and empirical label sparsity under three path labeling
choices. It is intended for expert review and decision making.

## Scope and Artifacts

- Dataset: `/mnt/data/retrieval_dataset/webqsp/normalized`
- Files used: `graphs.parquet`, `questions.parquet`, `sub_filter.json`
- Graph count: full=4699 (train=2826, val=245, test=1628)
- Sub-filter count: 4475 (train=2695, val=234, test=1546)

Statistics below are recomputed from raw graph topology plus seed/answer
anchors; they do not rely on stored `positive_triple_mask`.

## Retriever Objective (Definition)

Given a retrieval graph `G_sub` with anchors `(s, a)`, the retriever produces
edge scores used for:

1) edge selection into `g_agent` (top-k + bridge edges),
2) edge priors `edge_scores` in `g_agent`.

Training targets are triple-level shortest-path union labels. In other words,
the retriever is optimized to rank edges that lie on shortest paths between
start and answer nodes under a chosen path semantics.

## GFlowNet Interface (Factual)

- `g_agent` edges are directed, but the environment allows traversal from
  either endpoint (forward/backward directions are tracked).
- `g_agent` path supervision currently uses undirected shortest-path DAG
  (see `src/data/components/g_agent_builder.py`).
- `g_agent` edge labels are computed on the selected edge set, not copied from
  retriever labels. Retriever labels affect GFlowNet indirectly via edge scores
  and the resulting selected edge set.

## Path Labeling Options

Three label definitions were analyzed on the same graphs:

1) Directed shortest-path union (qa-directed).
2) Undirected shortest-path union (no canonicalization).
3) Undirected shortest-path union with canonicalize_relations
   (one edge kept per undirected node pair).

Canonicalization is counted as one positive per undirected node pair; it does
not attempt to reproduce relation-embedding scoring.

## Empirical Label Statistics (WebQSP)

All numbers below are per-graph positive-edge counts and positive ratios
(pos/edges). Median (p50), tail (p90/p95), and mean are shown.

### A) Directed shortest-path union

Full:
- train: p50=3, p90=32, p95=78, mean=14.97, zero=4.18%
- test:  p50=3, p90=29, p95=62, mean=12.81, zero=4.67%
- val:   p50=3, p90=27, p95=45, mean=15.68, zero=4.49%

Sub:
- train: p50=3, p90=33, p95=81, mean=15.47, zero=0%
- test:  p50=3, p90=30, p95=64, mean=13.27, zero=0%
- val:   p50=4, p90=27, p95=46, mean=16.41, zero=0%

Positive ratio (full, mean): ~0.0044 to 0.0045

### B) Undirected shortest-path union (no canonicalization)

Full:
- train: p50=5, p90=63, p95=166, mean=31.39, zero=4.18%
- test:  p50=5, p90=57, p95=135, mean=26.65, zero=4.67%
- val:   p50=6, p90=54, p95=101, mean=33.69, zero=4.49%

Sub:
- train: p50=6, p90=64, p95=168, mean=32.43, zero=0%
- test:  p50=6, p90=61, p95=143, mean=27.83, zero=0%
- val:   p50=6, p90=55, p95=104, mean=35.27, zero=0%

Positive ratio (full, mean): ~0.0091 to 0.0094

### C) Undirected shortest-path union (canonicalize_relations)

Full:
- train: p50=3, p90=32, p95=76, mean=16.34, zero=4.18%
- test:  p50=3, p90=30, p95=63, mean=13.98, zero=4.67%
- val:   p50=3, p90=28, p95=54, mean=16.67, zero=4.49%

Sub:
- train: p50=3, p90=34, p95=80, mean=16.96, zero=0%
- test:  p50=3, p90=31, p95=68, mean=14.61, zero=0%
- val:   p50=4, p90=28, p95=57, mean=17.45, zero=0%

Positive ratio (full, mean): ~0.0048 to 0.0051

## Observed Effects on GFlowNet (Neutral)

- Path semantics: `g_agent` supervision is undirected; a directed retriever
  labeling scheme optimizes a different edge set than what GFlowNet uses for
  its shortest-path DAG.
- Label density: canonicalization roughly halves positive-edge density in
  WebQSP (mean ratio ~0.009 -> ~0.005).
- Selection sensitivity: retriever labels shape retriever scores, which
  determine top-k edges and therefore the candidate edge set inside `g_agent`.

## Decision Axes for Review

1) Path semantics alignment:
   - directed labeling vs undirected `g_agent` supervision
2) Label density:
   - no canonicalization (denser positives)
   - canonicalization (sparser positives)
3) Training/eval scope:
   - full vs sub-filtered subsets (sub removes zero-positive cases)

No recommendation is made here; the above facts are provided for expert
evaluation.
