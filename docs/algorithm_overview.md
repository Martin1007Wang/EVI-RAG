# Algorithm Reconstruction (Code-Based)

This document reconstructs the algorithm implemented in this repo strictly from the code. It is organized as a systems view: entrypoints, data pipeline, runtime batching, model, training/eval behavior, and hard constraints. File paths referenced inline are the authoritative sources.

## 1) Scope and SSOT

- The single source of truth is the `g_retrieval` LMDB dataset. It stores only graph topology + question embeddings + start/answer indices. No supervision labels are used. See `src/data/stages/step3_lmdb.py` and `src/data/g_retrieval_dataset.py`.
- `q_local_indices` / `a_local_indices` are data-level truth (question/answer entity sets). They are never swapped. Any flow direction is derived from these sets, not by swapping q/a. See `src/data/g_retrieval_dataset.py` and `src/models/dual_flow_module.py`.
- `dummy_mask` is derived from data-level answer presence (empty answer list). `node_is_target` is derived from flow targets (answer locals). See `src/models/dual_flow_module.py`.

## 2) Entrypoints and Config Contract

- Training entrypoint: `src/train.py`. Enforces:
  - `dataset` must be provided for DualFlow (`configs/model/gflownet.yaml`).
  - training must use sub datasets only (dataset scope must be `sub`).
  - CVT initialization is mandatory and requires `entity_vocab.parquet` to be resolvable and consistent between `dataset` and `data.dataset_cfg`.
- Evaluation entrypoint: `src/eval.py`. Enforces:
  - `dataset` and `run` must be present.
  - checkpoint must be provided for eval runs requiring it (`run.name=eval_gflownet`).
  - full+sub dataset variants must be provided simultaneously.
  - single-GPU eval only by default (no DDP).
- Data build entrypoint: `scripts/build_retrieval_pipeline.py` -> `src/data/pipeline_main.py`.
  - Pipeline is unified; skipping parquet or LMDB stage is disabled.

## 3) Data Pipeline: Raw -> Parquet -> LMDB

### 3.1 Raw ingestion (HF only)
- The pipeline reads from HF datasets only (parquet ingestion disabled). See `src/data/stages/step2_graph.py` and `src/data/io/raw_loader.py`.
- Each sample provides: question, graph edges, q_entity, a_entity, answer texts. See `src/data/schema/types.py`.

### 3.2 Graph preprocessing (Parquet)
- Edges are filtered by:
  - relation cleaning rules,
  - time-relation gating,
  - self-loop removal,
  - optional anchor-edge retention to keep start nodes present.
  See `src/data/stages/step1_vocab.py` and `src/data/stages/step2_graph.py`.
- Samples are filtered by split-specific rules: missing topic, missing answer, or no path (configurable). Connectivity uses either undirected or qa-directed traversal. See `src/data/utils/connectivity.py`.
- Inverse relations can be injected from a JSON mapping (suffix `__inv` by default), extending the relation vocabulary and adding reverse edges. See `src/data/stages/step2_graph.py` and `configs/pipeline/default.yaml`.
- Outputs:
  - `graphs.parquet` (node ids, embeddings ids, edge list),
  - `questions.parquet` (question text + seed/answer ids),
  - vocab files (`entity_vocab.parquet`, `embedding_vocab.parquet`, `relation_vocab.parquet`),
  - optional `sub_filter.json` for sub-dataset filtering.

### 3.3 LMDB materialization
- LMDB samples are written from parquet with strict validation (no empty graphs). See `src/data/stages/step3_lmdb.py`.
- Each LMDB sample stores only:
  - `edge_index`, `edge_attr`, `num_nodes`,
  - `node_global_ids`, `node_embedding_ids`,
  - `question_emb`,
  - `q_local_indices`, `a_local_indices`,
  - `answer_entity_ids`, `retrieval_failure`.
- `filter_missing_start.json` and `filter_missing_answer.json` are written for runtime filtering.
- Sub datasets are not materialized to LMDB; they are mask-only at runtime.

## 4) Runtime Data Loading and Batching

- Dataset: `GRetrievalDataset` reads from LMDB and returns a PyG `Data` object with the fields above. See `src/data/g_retrieval_dataset.py`.
- DataModule: `GRetrievalDataModule` handles splits, shared embeddings, and attaches embeddings on transfer to device. See `src/data/g_retrieval_datamodule.py` and `src/data/components/embeddings.py`.
- Collation:
  - `RetrievalCollater` can expand multi-answer samples and filter zero-hop; both are off by default in `configs/data/g_retrieval.yaml`.
  - `BatchAugmenter` attaches `answer_entity_ids_ptr` and precomputes `edge_batch` / `edge_ptr` with validation. See `src/data/components/collate.py` and `src/utils/graph.py`.

## 5) Model: DualFlow (GFlowNet)

- Backbone: two independent `EmbeddingBackbone` stacks (forward and backward). Each projects node/edge/question embeddings and runs a relational GNN. See `src/models/components/gflownet_layers.py`.
- CVT init: CVT nodes are replaced by mean incoming (head + relation) embeddings; missing incoming edges are fatal. See `src/models/components/gflownet_layers.py`.
- Policy head: `QCBiANetwork` computes edge logits given (context, head, relation, tail). See `src/models/components/qc_bia_network.py`.
- Contexts:
  - forward context = MLP([question, start])
  - backward context = MLP([question, start, pooled_answer])
  See `src/models/dual_flow_module.py`.
- LogZ: `LogZPredictor` predicts per-node log flow with a time embedding. See `src/models/components/gflownet_layers.py` and `src/models/components/gflownet_actor.py`.

## 6) Flow Construction and Edge Masking

- Start nodes are sampled uniformly (Gumbel-max) from `q_local_indices` per graph. See `src/models/dual_flow_module.py`.
- Answer targets are the union of `a_local_indices`; `node_is_target` is derived per batch. See `src/models/dual_flow_module.py`.
- Edges are augmented with self-loops (relation id = -1), then reordered to keep per-graph contiguity. See `src/models/dual_flow_module.py`.
- Forward edge mask: non-inverse edges + self-loops.
- Backward edge mask: inverse edges + self-loops.
- Inverse edges are resolved through vocab mapping; strict symmetry can be enforced. See `src/data/components/shared_resources.py` and `src/models/dual_flow_module.py`.

## 7) Training Objective (High-Level)

- A forward rollout is sampled from the forward policy (with temperature and MC subsampling for high degree).
- Detailed balance loss is computed on the sampled trajectory using forward log-prob and backward log-prob of inverse edges.
- Targets force logZ(v)=0, terminal failures set logZ(v)=dead_end_log_reward and may be reweighted.
- The loss is mean squared balance residual over valid steps/graphs.
See `src/models/dual_flow_module.py`.

## 8) Evaluation and Artifacts

- Evaluation uses beam search with the forward policy and reports pass@1, pass@beam, and length metrics. See `src/models/dual_flow_module.py`.
- `predict_step` emits rollout records (global entity ids + relation ids).
- `GFlowNetEvalMetrics` aggregates predict rollouts into terminal/context hit metrics and diversity statistics. See `src/metrics/gflownet.py`.
- `GFlowNetRolloutArtifactWriter` writes JSONL artifacts and a manifest, optionally textualized with vocab. See `src/callbacks/gflownet_rollout_artifact_writer.py`.

## 9) Auxiliary Modules (Not in Main Training Path)

- `GraphFusionReward` and SubTB teacher flow modules were removed as unused during cleanup. Current training is DualFlow-only.

## 10) Hard Constraints and Guardrails

- Training must use sub datasets only. See `src/train.py`.
- CVT initialization is mandatory and requires consistent vocab paths. See `src/train.py`.
- LMDB cannot be built for sub datasets. See `src/data/stages/step3_lmdb.py`.
- Evaluation requires both full and sub variants and runs single-GPU. See `src/eval.py`.
- Canonicalize-relations is disabled for GFlowNet preprocessing. See `src/data/stages/step2_graph.py`.
