# Undirected BFS Supervision and Pair-Level DAG

This project standardizes shortest-path supervision as an **undirected** problem while keeping
the stored graph **directed**. The goal is to make retriever training, g_agent construction,
and GFlowNet rollouts consistent end-to-end.

## Supervision Schema (SSOT)

The ground-truth set is defined **per (start, answer) pair**:

- `edge_labels`: union of all **undirected** shortest-path edges over every `(start, answer)` pair.
- `pair_start_node_locals`: `[P]` local start node indices for each reachable pair.
- `pair_answer_node_locals`: `[P]` local answer node indices aligned with `pair_start_node_locals`.
- `pair_edge_local_ids`: concatenated edge-local ids for each pair's shortest-path DAG.
- `pair_edge_counts`: CSR counts (length `P`); offsets can be derived with `cumsum([0] + counts)` when needed.
- `pair_shortest_lengths`: optional per-pair shortest length (for recall denominators).

This captures the **solution manifold** without enumerating explicit path instances.

## 0-Hop Pairs

When `start == answer`, the pair is **reachable** with zero edges:

- `pair_start_node_locals = [s]`, `pair_answer_node_locals = [s]`
- `pair_edge_local_ids = []`, `pair_edge_counts = [0]`
- `pair_shortest_lengths = [0]`
- `edge_labels` remains all `False`

## Preprocessing (Retriever)

- `scripts/build_retrieval_parquet.py` computes shortest-path supervision with **undirected BFS**.
- The `undirected_traversal` and `soft_label.undirected` flags are removed; this is a fixed design choice.

## Retriever Scoring (Twin-View)

Retriever scores each edge with two views:

- `fwd(q, h, r, t)`
- `bwd(q, t, r, h)`

Final logits are:

```
logits = max(logits_fwd, logits_bwd)
```

Optional persistence includes `logit_fwd` and `logit_bwd` for debugging.

## GFlowNet (Directed Graph, Dynamic Backward)

Graphs remain **directed** at storage time. The environment allows:

- forward moves when a head node is active
- backward moves when a tail node is active

Backward moves are represented by `backward_mask` and a **direction embedding**; no reverse edges
are explicitly added to the graph.
