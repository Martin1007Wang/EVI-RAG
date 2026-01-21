# Random Walk Analysis on webqsp (sub)

## Scope
- Dataset root: `/mnt/data/retrieval_dataset/webqsp/normalized`
- Sub filter: `sub_filter.json` (sample_ids)
- Graphs: `graphs.parquet`
- Questions: `questions.parquet`
- Sub graphs used: 4500 (all sub sample_ids present in both graphs and questions)

## Start/Answer Set Sizes (sub)
- Questions: 4500
- Start entities (seed_entity_ids)
  - Total: 4603
  - Unique: 2167
  - Per-question: mean 1.0229, median 1, p90 1, max 5
- Answer entities (answer_entity_ids)
  - Total: 48298
  - Unique: 30759
  - Per-question: mean 10.7329, median 2, p90 12, max 3688

## Degree Stats (sub)
Edges in `graphs.parquet` are directed and already include inverse edges. As a result, per-node
in-degree and out-degree are identical in these graphs.

Per-node out-degree (same as in-degree):
- Start nodes: mean 667.58, median 344, p90 2185.2, min 1, max 3257 (count 4603)
- Answer nodes: mean 34.89, median 10, p90 74, min 1, max 2960 (count 26795)

Per-graph comparisons:
- Mean out-degree: start > answer for 3967 graphs, answer > start for 511 graphs, equal for 22 graphs
- Total out-degree (sum over nodes in set): start > answer for 3626 graphs, answer > start for 849 graphs, equal for 25 graphs

## Random Walk Setup
- Graph: directed edges from `graphs.parquet` (inverse edges already present).
- Start/target sets: local node indices mapped from seed_entity_ids / answer_entity_ids; duplicates removed.
- Step rule: at each step, choose uniformly among outgoing edges; if out-degree is 0, stay in place.
- Start distribution:
  1) `uniform`: uniform over nodes in the set
  2) `degree`: weighted by out-degree within the set (fallback to uniform if sum is 0)
- Metrics:
  - `exact@K`: probability of being in the target set exactly at step K
  - `within@K`: probability of hitting the target within K steps (target is absorbing)
- K in {1, 2, 3, 4, 5}
- Computation: exact probability propagation (no Monte Carlo)

## Results (mean/median/p90)

### Start distribution: uniform
K=1
- exact fwd: 0.0344 / 0.0033 / 0.0897
- exact bwd: 0.1239 / 0.0272 / 0.3464
- within fwd: 0.0408 / 0.0034 / 0.0944
- within bwd: 0.1272 / 0.0278 / 0.3529
- bwd > fwd: exact 2346, within 2331 (fwd > bwd: 710, equal: 1459)

K=2
- exact fwd: 0.0216 / 0.0036 / 0.0536
- exact bwd: 0.1018 / 0.0358 / 0.3034
- within fwd: 0.0584 / 0.0142 / 0.1547
- within bwd: 0.2275 / 0.1778 / 0.5000
- bwd > fwd: exact 3022, within 3751 (fwd > bwd: 717, equal: 32)

K=3
- exact fwd: 0.0362 / 0.0061 / 0.0966
- exact bwd: 0.1140 / 0.0520 / 0.3003
- within fwd: 0.0721 / 0.0197 / 0.1843
- within bwd: 0.2761 / 0.2285 / 0.5512
- bwd > fwd: exact 3462, within 3795 (fwd > bwd: 684, equal: 21)

K=4
- exact fwd: 0.0247 / 0.0057 / 0.0663
- exact bwd: 0.1068 / 0.0563 / 0.2951
- within fwd: 0.0851 / 0.0259 / 0.2487
- within bwd: 0.3350 / 0.3035 / 0.6526
- bwd > fwd: exact 3681, within 3807 (fwd > bwd: 672, equal: 21)

K=5
- exact fwd: 0.0366 / 0.0071 / 0.1001
- exact bwd: 0.1100 / 0.0602 / 0.2748
- within fwd: 0.0955 / 0.0315 / 0.2734
- within bwd: 0.3698 / 0.3445 / 0.6979
- bwd > fwd: exact 3672, within 3818 (fwd > bwd: 661, equal: 21)

### Start distribution: degree-weighted
K=1
- exact fwd: 0.0344 / 0.0033 / 0.0897
- exact bwd: 0.1048 / 0.0206 / 0.2857
- within fwd: 0.0408 / 0.0033 / 0.0944
- within bwd: 0.1099 / 0.0217 / 0.3000
- bwd > fwd: exact 2235, within 2235 (fwd > bwd: 806, equal: 1459)

K=2
- exact fwd: 0.0216 / 0.0035 / 0.0536
- exact bwd: 0.1005 / 0.0383 / 0.2935
- within fwd: 0.0584 / 0.0139 / 0.1547
- within bwd: 0.2082 / 0.1600 / 0.4451
- bwd > fwd: exact 2995, within 3635 (fwd > bwd: 833, equal: 32)

K=3
- exact fwd: 0.0361 / 0.0061 / 0.0966
- exact bwd: 0.1036 / 0.0459 / 0.2660
- within fwd: 0.0720 / 0.0196 / 0.1843
- within bwd: 0.2553 / 0.2100 / 0.5000
- bwd > fwd: exact 3412, within 3695 (fwd > bwd: 784, equal: 21)

K=4
- exact fwd: 0.0247 / 0.0056 / 0.0663
- exact bwd: 0.1056 / 0.0562 / 0.2809
- within fwd: 0.0850 / 0.0257 / 0.2487
- within bwd: 0.3134 / 0.2783 / 0.6379
- bwd > fwd: exact 3631, within 3701 (fwd > bwd: 777, equal: 22)

K=5
- exact fwd: 0.0366 / 0.0071 / 0.1001
- exact bwd: 0.1030 / 0.0553 / 0.2534
- within fwd: 0.0954 / 0.0310 / 0.2731
- within bwd: 0.3483 / 0.3190 / 0.6827
- bwd > fwd: exact 3626, within 3723 (fwd > bwd: 755, equal: 22)

## Takeaways
- In sub graphs, answer->start random walks have consistently higher hit probability than start->answer.
- The gap holds for K=1..5 and for both uniform and degree-weighted start distributions.
- This supports using the backward stream as an exploration aid for the forward stream in sub.
