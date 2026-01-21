# Random Walk Analysis on cwq (sub)

## Scope
- Dataset: `rmanluo/RoG-cwq`
- Split: `train`
- Sub filter: keep samples where `q_entity` and `a_entity` both appear in the graph node set.
- Graph: `graph` is a list of directed (head, relation, tail) triples; node set is union of heads/tails.
- Total samples in split: 27639
- Sub graphs used: 22089

## Start/Answer Set Sizes (sub)
- Questions: 22089
- Start entities (`q_entity`)
  - Total: 34676
  - Unique: 8386
  - Per-question: mean 1.5698, median 1, p90 2, max 5
- Answer entities (`a_entity`)
  - Total: 41637
  - Unique: 4551
  - Per-question: mean 1.8850, median 1, p90 3, max 29

## Degree Stats (sub)
Edges are directed; inverse edges are not guaranteed, so in-degree and out-degree can differ.
Across graphs, the fraction of nodes with `out_degree == in_degree` is: mean 0.2974, median 0.2593, p90 0.4955.

Per-node out-degree:
- Start nodes: mean 210.45, median 65, p90 488.0, min 0, max 2085 (count 34676)
- Answer nodes: mean 41.54, median 5, p90 91.4, min 0, max 2021 (count 41637)

Per-graph comparisons:
- Mean out-degree: start > answer for 17752 graphs, answer > start for 4059 graphs, equal for 278 graphs
- Total out-degree (sum over nodes in set): start > answer for 17384 graphs, answer > start for 4321 graphs, equal for 384 graphs

## Random Walk Setup
- Graph: directed edges from `graph` (no edge reversal; inverse edges may or may not exist).
- Start/target sets: local node indices mapped from `q_entity` / `a_entity`; duplicates removed.
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
- exact fwd: 0.0137 / 0.0000 / 0.0435
- exact bwd: 0.0423 / 0.0000 / 0.1429
- within fwd: 0.0318 / 0.0000 / 0.0526
- within bwd: 0.0531 / 0.0020 / 0.1538
- bwd > fwd: exact 7636, within 7640 (fwd > bwd: exact 4611, within 4818; equal: exact 9842, within 9631)

K=2
- exact fwd: 0.0089 / 0.0019 / 0.0211
- exact bwd: 0.0384 / 0.0049 / 0.1250
- within fwd: 0.0383 / 0.0061 / 0.0690
- within bwd: 0.0892 / 0.0283 / 0.2467
- bwd > fwd: exact 13614, within 13700 (fwd > bwd: exact 7842, within 8146; equal: exact 633, within 243)

K=3
- exact fwd: 0.0095 / 0.0017 / 0.0279
- exact bwd: 0.0216 / 0.0048 / 0.0640
- within fwd: 0.0421 / 0.0082 / 0.0822
- within bwd: 0.1021 / 0.0388 / 0.2758
- bwd > fwd: exact 11669, within 13747 (fwd > bwd: exact 8821, within 8156; equal: exact 1599, within 186)

K=4
- exact fwd: 0.0065 / 0.0019 / 0.0163
- exact bwd: 0.0251 / 0.0061 / 0.0780
- within fwd: 0.0454 / 0.0102 / 0.0910
- within bwd: 0.1170 / 0.0457 / 0.3186
- bwd > fwd: exact 14138, within 13950 (fwd > bwd: exact 7775, within 7988; equal: exact 176, within 151)

K=5
- exact fwd: 0.0070 / 0.0013 / 0.0187
- exact bwd: 0.0134 / 0.0040 / 0.0384
- within fwd: 0.0473 / 0.0116 / 0.0969
- within bwd: 0.1238 / 0.0513 / 0.3374
- bwd > fwd: exact 13253, within 14089 (fwd > bwd: exact 8456, within 7849; equal: exact 380, within 151)

### Start distribution: degree-weighted
K=1
- exact fwd: 0.0076 / 0.0000 / 0.0190
- exact bwd: 0.0418 / 0.0000 / 0.1333
- within fwd: 0.0255 / 0.0000 / 0.0227
- within bwd: 0.0542 / 0.0020 / 0.1481
- bwd > fwd: exact 8688, within 8695 (fwd > bwd: exact 3563, within 3769; equal: exact 9838, within 9625)

K=2
- exact fwd: 0.0079 / 0.0018 / 0.0159
- exact bwd: 0.0386 / 0.0048 / 0.1260
- within fwd: 0.0311 / 0.0045 / 0.0411
- within bwd: 0.0902 / 0.0278 / 0.2481
- bwd > fwd: exact 13745, within 14495 (fwd > bwd: exact 7654, within 7294; equal: exact 690, within 300)

K=3
- exact fwd: 0.0062 / 0.0014 / 0.0166
- exact bwd: 0.0215 / 0.0048 / 0.0636
- within fwd: 0.0342 / 0.0060 / 0.0515
- within bwd: 0.1031 / 0.0383 / 0.2773
- bwd > fwd: exact 12602, within 14613 (fwd > bwd: exact 7879, within 7281; equal: exact 1608, within 195)

K=4
- exact fwd: 0.0061 / 0.0017 / 0.0147
- exact bwd: 0.0251 / 0.0061 / 0.0781
- within fwd: 0.0374 / 0.0077 / 0.0618
- within bwd: 0.1180 / 0.0449 / 0.3214
- bwd > fwd: exact 14338, within 14760 (fwd > bwd: exact 7568, within 7177; equal: exact 183, within 152)

K=5
- exact fwd: 0.0050 / 0.0011 / 0.0135
- exact bwd: 0.0134 / 0.0040 / 0.0384
- within fwd: 0.0392 / 0.0085 / 0.0679
- within bwd: 0.1249 / 0.0506 / 0.3402
- bwd > fwd: exact 13999, within 14857 (fwd > bwd: exact 7710, within 7080; equal: exact 380, within 152)

## Takeaways
- On cwq sub graphs (`train` split), mean `within@K` bwd > fwd at K=1, 2, 3, 4, 5 for uniform starts.
- On cwq sub graphs (`train` split), mean `within@K` bwd > fwd at K=1, 2, 3, 4, 5 for degree-weighted starts.
