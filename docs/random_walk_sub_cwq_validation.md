# Random Walk Analysis on cwq (sub)

## Scope
- Dataset: `rmanluo/RoG-cwq`
- Split: `validation`
- Sub filter: keep samples where `q_entity` and `a_entity` both appear in the graph node set.
- Graph: `graph` is a list of directed (head, relation, tail) triples; node set is union of heads/tails.
- Total samples in split: 3519
- Sub graphs used: 2848

## Start/Answer Set Sizes (sub)
- Questions: 2848
- Start entities (`q_entity`)
  - Total: 4495
  - Unique: 1904
  - Per-question: mean 1.5783, median 2, p90 2, max 4
- Answer entities (`a_entity`)
  - Total: 5359
  - Unique: 957
  - Per-question: mean 1.8817, median 1, p90 4, max 27

## Degree Stats (sub)
Edges are directed; inverse edges are not guaranteed, so in-degree and out-degree can differ.
Across graphs, the fraction of nodes with `out_degree == in_degree` is: mean 0.2856, median 0.2564, p90 0.4826.

Per-node out-degree:
- Start nodes: mean 210.62, median 75, p90 477.0, min 0, max 2074 (count 4495)
- Answer nodes: mean 49.73, median 6, p90 97.2, min 0, max 1899 (count 5359)

Per-graph comparisons:
- Mean out-degree: start > answer for 2285 graphs, answer > start for 543 graphs, equal for 20 graphs
- Total out-degree (sum over nodes in set): start > answer for 2266 graphs, answer > start for 551 graphs, equal for 31 graphs

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
- exact fwd: 0.0149 / 0.0000 / 0.0500
- exact bwd: 0.0423 / 0.0017 / 0.1367
- within fwd: 0.0288 / 0.0006 / 0.0579
- within bwd: 0.0500 / 0.0026 / 0.1431
- bwd > fwd: exact 1019, within 1016 (fwd > bwd: exact 638, within 664; equal: exact 1191, within 1168)

K=2
- exact fwd: 0.0095 / 0.0020 / 0.0224
- exact bwd: 0.0365 / 0.0056 / 0.1250
- within fwd: 0.0359 / 0.0061 / 0.0794
- within bwd: 0.0843 / 0.0312 / 0.2308
- bwd > fwd: exact 1699, within 1740 (fwd > bwd: exact 1078, within 1091; equal: exact 71, within 17)

K=3
- exact fwd: 0.0110 / 0.0019 / 0.0344
- exact bwd: 0.0220 / 0.0066 / 0.0641
- within fwd: 0.0406 / 0.0080 / 0.0908
- within bwd: 0.0977 / 0.0419 / 0.2630
- bwd > fwd: exact 1532, within 1739 (fwd > bwd: exact 1143, within 1097; equal: exact 173, within 12)

K=4
- exact fwd: 0.0071 / 0.0019 / 0.0190
- exact bwd: 0.0238 / 0.0062 / 0.0692
- within fwd: 0.0441 / 0.0099 / 0.0984
- within bwd: 0.1119 / 0.0500 / 0.3055
- bwd > fwd: exact 1779, within 1749 (fwd > bwd: exact 1056, within 1088; equal: exact 13, within 11)

K=5
- exact fwd: 0.0081 / 0.0014 / 0.0243
- exact bwd: 0.0139 / 0.0048 / 0.0380
- within fwd: 0.0466 / 0.0112 / 0.1047
- within bwd: 0.1191 / 0.0573 / 0.3279
- bwd > fwd: exact 1717, within 1760 (fwd > bwd: exact 1102, within 1077; equal: exact 29, within 11)

### Start distribution: degree-weighted
K=1
- exact fwd: 0.0087 / 0.0000 / 0.0199
- exact bwd: 0.0418 / 0.0019 / 0.1333
- within fwd: 0.0228 / 0.0005 / 0.0227
- within bwd: 0.0514 / 0.0026 / 0.1441
- bwd > fwd: exact 1160, within 1161 (fwd > bwd: exact 500, within 522; equal: exact 1188, within 1165)

K=2
- exact fwd: 0.0087 / 0.0018 / 0.0179
- exact bwd: 0.0368 / 0.0053 / 0.1250
- within fwd: 0.0290 / 0.0044 / 0.0453
- within bwd: 0.0856 / 0.0312 / 0.2308
- bwd > fwd: exact 1747, within 1850 (fwd > bwd: exact 1026, within 977; equal: exact 75, within 21)

K=3
- exact fwd: 0.0075 / 0.0015 / 0.0189
- exact bwd: 0.0219 / 0.0066 / 0.0637
- within fwd: 0.0329 / 0.0057 / 0.0575
- within bwd: 0.0990 / 0.0417 / 0.2630
- bwd > fwd: exact 1669, within 1864 (fwd > bwd: exact 1005, within 972; equal: exact 174, within 12)

K=4
- exact fwd: 0.0067 / 0.0017 / 0.0167
- exact bwd: 0.0239 / 0.0063 / 0.0695
- within fwd: 0.0362 / 0.0073 / 0.0701
- within bwd: 0.1132 / 0.0499 / 0.3076
- bwd > fwd: exact 1829, within 1872 (fwd > bwd: exact 1005, within 965; equal: exact 14, within 11)

K=5
- exact fwd: 0.0060 / 0.0011 / 0.0160
- exact bwd: 0.0139 / 0.0048 / 0.0379
- within fwd: 0.0385 / 0.0079 / 0.0773
- within bwd: 0.1205 / 0.0566 / 0.3297
- bwd > fwd: exact 1832, within 1881 (fwd > bwd: exact 986, within 956; equal: exact 30, within 11)

## Takeaways
- On cwq sub graphs (`validation` split), mean `within@K` bwd > fwd at K=1, 2, 3, 4, 5 for uniform starts.
- On cwq sub graphs (`validation` split), mean `within@K` bwd > fwd at K=1, 2, 3, 4, 5 for degree-weighted starts.
