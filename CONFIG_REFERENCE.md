# Config Reference

This document describes the current Hydra config tree in `configs/`, the exact content of each active config file, and the code interface that consumes each config section.

It reflects the repository state after the config cleanup to:

- separate task entrypoints into `train.yaml`, `evaluate.yaml`, and `preprocess.yaml`
- rename evaluation metric configs to `eval_metric/`
- split optimizer and scheduler out of `model/`
- keep `model/weaver.yaml` as the clean baseline
- keep `model/weaver_coverage.yaml` as the explicit method override

## Composition Model

Hydra entrypoints:

- `python src/train.py ...` loads `configs/train.yaml` via `src/train.py:38`
- `python src/evaluate.py ...` loads `configs/evaluate.yaml` via `src/evaluate.py:52`
- `python src/preprocess.py ...` loads `configs/preprocess.yaml` via `src/preprocess.py:25`

Common instantiation path:

- `cfg.datamodule` is instantiated in `src/training/factory.py:52-61`
- `cfg.model` is instantiated in `src/training/factory.py:64-77`
- `cfg.trainer`, `cfg.callbacks`, and `cfg.logger` are instantiated in `src/training/factory.py:80-102`

Training/evaluation resource flow:

- `setup_datamodule(...)` prepares data and exposes model resources in `src/training/resources.py`
- `RetrievalDataModule` consumes `dataset_cfg` and loader-related knobs in `src/data/datamodule.py:55-257`
- `WeaverModule` consumes model-level knobs in `src/weaver/module.py:74-279`
- evaluation metric knobs are validated in `src/weaver/module.py:580-615`
- optimizer and scheduler configs are consumed in `src/training/optimization.py:138-219`
- preprocess knobs are consumed from `cfg.preprocess` in `src/data/pipeline.py:66-134`

## Active Config Tree

```text
configs
├── __init__.py
├── train.yaml
├── evaluate.yaml
├── preprocess.yaml
├── paths
│   └── default.yaml
├── hydra
│   └── default.yaml
├── dataset
│   ├── webqsp.yaml
│   ├── cwq.yaml
│   └── experimental
│       └── prime.yaml
├── datamodule
│   └── default.yaml
├── preprocess
│   └── default.yaml
├── model
│   ├── weaver.yaml
│   └── weaver_coverage.yaml
├── eval_metric
│   └── default.yaml
├── optimizer
│   └── adamw.yaml
├── scheduler
│   └── cosine_warmup.yaml
├── trainer
│   ├── gpu.yaml
│   ├── cpu.yaml
│   └── fast_dev.yaml
├── callbacks
│   ├── train.yaml
│   ├── eval.yaml
│   └── local_metrics_writer.yaml
├── logger
│   ├── wandb.yaml
│   └── none.yaml
├── ckpt
│   └── default.yaml
└── experiment
    ├── debug
    │   └── valfit.yaml
    ├── eval
    │   └── webqsp.yaml
    └── train
        ├── webqsp_baseline.yaml
        └── webqsp_coverage.yaml
```

## Interface Map

### `configs/train.yaml`

Role: training entrypoint and top-level composition root.

Consumed by:

- `src/train.py:26-67`
- `src/training/factory.py:52-102`
- `src/training/checkpoint.py:37-52`

Key interface:

- `task_name`, `seed`, `print_config`, `test_after_fit` are read in `src/train.py:33-67`
- `fit_ckpt_path` is read in `src/train.py:26-30`
- `pretrained_ckpt_path` and `strict_pretrained_load` are read in `src/training/checkpoint.py:37-52`
- `datamodule`, `model`, `trainer`, `callbacks`, and `logger` are instantiated in `src/training/factory.py:52-102`
- `model.optimizer_cfg` and `model.scheduler_cfg` are injected only by the training entrypoint

```yaml
# @package _global_

defaults:
  - paths: default
  - hydra: default
  - dataset: webqsp
  - datamodule: default
  - model: weaver
  - eval_metric: default
  - optimizer: adamw
  - scheduler: cosine_warmup
  - trainer: gpu
  - callbacks: train
  - logger: wandb
  - ckpt: default
  - _self_
  - experiment: null

task_name: train
seed: 42
print_config: false
test_after_fit: false

model:
  optimizer_cfg: ${optimizer}
  scheduler_cfg: ${scheduler}

fit_ckpt_path: ${ckpt.fit}
pretrained_ckpt_path: ${ckpt.pretrained}
strict_pretrained_load: ${ckpt.strict_pretrained_load}
```

### `configs/evaluate.yaml`

Role: evaluation entrypoint and top-level composition root.

Consumed by:

- `src/evaluate.py:27-89`
- `src/training/factory.py:52-102`
- `src/training/checkpoint.py:19-34`

Key interface:

- `task_name`, `seed`, `print_config`, `validate`, `test`, `ckpt_path` are read in `src/evaluate.py:27-89`
- `trainer.enable_checkpointing: false` is a task-level override merged before trainer instantiation
- `datamodule`, `model`, `trainer`, `callbacks`, and `logger` are instantiated in `src/training/factory.py:52-102`
- evaluation intentionally does not compose optimizer or scheduler groups

```yaml
# @package _global_

defaults:
  - paths: default
  - hydra: default
  - dataset: webqsp
  - datamodule: default
  - model: weaver
  - eval_metric: default
  - trainer: gpu
  - callbacks: eval
  - logger: wandb
  - ckpt: default
  - _self_
  - experiment: null

task_name: evaluate
seed: 42
print_config: false

validate: true
test: true

trainer:
  enable_checkpointing: false

ckpt_path: ${ckpt.path}
```

### `configs/preprocess.yaml`

Role: preprocess entrypoint and top-level composition root.

Consumed by:

- `src/preprocess.py:25-40`
- `src/data/pipeline.py:66-134`

Key interface:

- `task_name`, `seed`, and `dataset` control the preprocess entrypoint in `src/preprocess.py:25-40`
- `preprocess: default` injects the actual pipeline knobs consumed in `src/data/pipeline.py:66-134`
- those knobs live under `cfg.preprocess`, not the global namespace

```yaml
# @package _global_

defaults:
  - paths: default
  - hydra: default
  - dataset: webqsp
  - preprocess: default
  - _self_

task_name: preprocess
seed: 42
print_config: false
```

### `configs/paths/default.yaml`

Role: central filesystem layout.

Consumed by:

- all task roots through interpolation
- Hydra runtime through `configs/hydra/default.yaml`
- checkpoint/logger/callback path interpolations
- dataset path interpolation in `configs/dataset/*.yaml`

Key interface:

- `paths.*` is not instantiated directly; it is interpolated into other config groups
- `run_dir` depends on `hydra.runtime.output_dir`, so it is only concrete during an actual Hydra run

```yaml
# @package paths

root_dir: ${oc.env:PROJECT_ROOT,${oc.env:PWD}}
data_dir: ${oc.env:DATA_DIR,${paths.root_dir}/data}
output_dir: ${paths.root_dir}/outputs
run_dir: ${hydra:runtime.output_dir}
cache_dir: ${oc.env:HF_HOME,${paths.root_dir}/.cache/huggingface}
log_dir: ${paths.run_dir}/logs
ckpt_dir: ${paths.run_dir}/checkpoints
artifact_dir: ${paths.run_dir}/artifacts
hf_home: ${paths.cache_dir}
hf_datasets_cache: ${paths.hf_home}/datasets
work_dir: ${hydra:runtime.cwd}
```

### `configs/hydra/default.yaml`

Role: Hydra runtime behavior.

Consumed by:

- Hydra itself

Key interface:

- `hydra.run.dir` and `hydra.sweep.dir` define the run directory structure
- `hydra.job.chdir: false` keeps the working directory stable

```yaml
# @package hydra

defaults:
  - override hydra_logging: default
  - override job_logging: default

run:
  dir: ${paths.output_dir}/${task_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}

sweep:
  dir: ${paths.output_dir}/multirun/${task_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
  subdir: ${hydra.job.num}

job:
  chdir: false

job_logging:
  handlers:
    file:
      filename: ${hydra.runtime.output_dir}/${hydra.job.name}.log
```

### `configs/dataset/webqsp.yaml`

Role: WebQSP dataset definition.

Consumed by:

- `src/data/datamodule.py:55-257`
- `src/data/pipeline.py:70-132`
- `src/eval/llm/eval_llm.py` for artifact/layout metadata when that path is used

Key interface:

- `dataset.paths.*` is read by `src/data/datamodule.py:94-108` and `src/data/pipeline.py:74-80,124-129`
- `dataset.splits.*` is read by `src/data/datamodule.py:110-116`
- `dataset.column_map` and source metadata are read by `src/data/pipeline.py:86-93`
- redundant `dataset.paths.root_dir` and `dataset.paths.entity_metadata` are intentionally removed

```yaml
# @package dataset

name: webqsp
dataset_family: kgqa
dataset_scope: webqsp
dataset_source: hf
hf_dataset: rmanluo/RoG-webqsp

splits:
  train: train
  validation: validation
  test: test

root_dir: ${paths.data_dir}/${dataset.name}
artifact_dir: ${dataset.root_dir}/artifacts

paths:
  raw_dir: ${dataset.root_dir}/raw
  lmdb_dir: ${dataset.root_dir}/lmdb
  metadata_dir: ${dataset.root_dir}/metadata
  embeddings_dir: ${dataset.root_dir}/embeddings
  entity_text_embeddings: ${dataset.paths.embeddings_dir}/entity_text_embeddings.pt
  relation_embeddings: ${dataset.paths.embeddings_dir}/relation_embeddings.pt
  entity_metadata_path: ${dataset.paths.metadata_dir}/entity_metadata.pt
  entity_catalog_path: ${dataset.paths.metadata_dir}/entity_catalog.pt
  relation_catalog_path: ${dataset.paths.metadata_dir}/relation_catalog.pt

column_map:
  question_id_field: id
  question_field: question
  answer_text_field: answer
  question_entity_field: q_entity
  answer_entity_field: a_entity
  graph_field: graph
```

### `configs/dataset/cwq.yaml`

Role: CWQ dataset definition.

Consumed by:

- `src/data/datamodule.py:55-257`
- `src/data/pipeline.py:70-132`

Key interface:

- same interface shape as `dataset/webqsp.yaml`

```yaml
# @package dataset

name: cwq
dataset_family: kgqa
dataset_scope: cwq
dataset_source: hf
hf_dataset: rmanluo/RoG-cwq

splits:
  train: train
  validation: validation
  test: test

root_dir: ${paths.data_dir}/${dataset.name}
artifact_dir: ${dataset.root_dir}/artifacts

paths:
  raw_dir: ${dataset.root_dir}/raw
  lmdb_dir: ${dataset.root_dir}/lmdb
  metadata_dir: ${dataset.root_dir}/metadata
  embeddings_dir: ${dataset.root_dir}/embeddings
  entity_text_embeddings: ${dataset.paths.embeddings_dir}/entity_text_embeddings.pt
  relation_embeddings: ${dataset.paths.embeddings_dir}/relation_embeddings.pt
  entity_metadata_path: ${dataset.paths.metadata_dir}/entity_metadata.pt
  entity_catalog_path: ${dataset.paths.metadata_dir}/entity_catalog.pt
  relation_catalog_path: ${dataset.paths.metadata_dir}/relation_catalog.pt

column_map:
  question_id_field: id
  question_field: question
  answer_text_field: answer
  question_entity_field: q_entity
  answer_entity_field: a_entity
  graph_field: graph
```

### `configs/dataset/experimental/prime.yaml`

Role: unsupported experimental PRIME dataset definition.

Consumed by:

- not part of the default active dataset options for normal runs
- currently only documents an unsupported future config surface

Key interface:

- `status: unsupported` and `enabled: false` make the current state explicit
- `src/data/preprocess/source.py:41-42` still rejects `dataset_source=stark`

```yaml
# @package dataset

name: prime
status: unsupported
enabled: false
dataset_family: kgqa
dataset_scope: prime
dataset_source: stark
hf_dataset: null
kb: prime

splits:
  train: train
  validation: validation
  test: test

root_dir: ${paths.data_dir}/${dataset.name}
artifact_dir: ${dataset.root_dir}/artifacts

paths:
  raw_dir: ${dataset.root_dir}/raw
  lmdb_dir: ${dataset.root_dir}/lmdb
  metadata_dir: ${dataset.root_dir}/metadata
  embeddings_dir: ${dataset.root_dir}/embeddings
  entity_text_embeddings: ${dataset.paths.embeddings_dir}/entity_text_embeddings.pt
  relation_embeddings: ${dataset.paths.embeddings_dir}/relation_embeddings.pt
  entity_metadata_path: ${dataset.paths.metadata_dir}/entity_metadata.pt
  entity_catalog_path: ${dataset.paths.metadata_dir}/entity_catalog.pt
  relation_catalog_path: ${dataset.paths.metadata_dir}/relation_catalog.pt

column_map:
  question_id_field: id
  question_field: question
  answer_text_field: answer
  question_entity_field: q_entity
  answer_entity_field: a_entity
  graph_field: graph

stark:
  dataset: prime
  root: ${paths.data_dir}/stark
  download_processed: true
  cache_dir: ${dataset.artifact_dir}/stark_cache
  linker:
    backend: keyword
    max_candidates: 12
    max_entities: 4
  local_graph:
    num_hops: 2
    direction: both
    max_nodes: 64
    max_edges: 256
```

### `configs/datamodule/default.yaml`

Role: Lightning datamodule config.

Consumed by:

- `src/training/factory.py:52-61`
- `src/data/datamodule.py:55-257`

Key interface:

- `_target_` points to `src.data.datamodule.RetrievalDataModule`
- constructor parameters must match `RetrievalDataModule.__init__` in `src/data/datamodule.py:55-68`
- dataset metadata is loaded from `dataset.paths.entity_metadata_path`

```yaml
# @package datamodule

_target_: src.data.datamodule.RetrievalDataModule

dataset_cfg: ${dataset}

batch_size: 8
num_workers: 4

eval_batch_size: 8
eval_num_workers: 4

pin_memory: true
train_shuffle: true
drop_last: false
eval_drop_last: false

lmdb_readahead: true
max_readers: 256
```

### `configs/preprocess/default.yaml`

Role: preprocess pipeline behavior.

Consumed by:

- `src/data/pipeline.py:66-134`

Key interface:

- `preprocess_filter.*` is converted to split filters from `cfg.preprocess` in `src/data/pipeline.py:81-90`
- `dedup_edges` and `remove_self_loops` are read from `cfg.preprocess` in `src/data/pipeline.py:101-106`
- `encoder.*` and `progress_bar` are read from `cfg.preprocess` in `src/data/pipeline.py:112-121`
- `overwrite_lmdb` and `map_size_gb` are read from `cfg.preprocess` in `src/data/pipeline.py:134-148`

```yaml
# @package preprocess

dedup_edges: true
remove_self_loops: true

preprocess_filter:
  train:
    require_answer_in_graph: true
    require_reachable_answer: true
  validation:
    require_answer_in_graph: true
    require_reachable_answer: true
  test:
    require_answer_in_graph: false
    require_reachable_answer: false

map_size_gb: 128
overwrite_lmdb: true
progress_bar: true

encoder:
  model_name: BAAI/bge-large-en-v1.5
  device: auto
  batch_size: 512

hf_env:
  cache_dir: ${paths.hf_datasets_cache}
```

### `configs/model/weaver.yaml`

Role: baseline model config.

Consumed by:

- `src/training/factory.py:64-77`
- `src/weaver/module.py:74-279`
- `src/weaver/module.py:580-736`
- `src/training/optimization.py:138-219`

Key interface:

- `_target_` points to `src.weaver.module.WeaverModule`
- top-level constructor args must match `WeaverModule.__init__` in `src/weaver/module.py:74-103`
- `eval_cfg` is validated in `src/weaver/module.py:580-615`
- `coverage_cfg` is validated in `src/weaver/module.py:718-736`
- training-only optimizer/scheduler injection happens at `configs/train.yaml`, not here

```yaml
# @package model

_target_: src.weaver.module.WeaverModule

expand_budget: 3

policy_cfg:
  hidden_dim: 1024
  feature_encoder:
    embedding_dim: 1024
    anchor_distance_max: 3
    non_text_init_std: 0.02
  continue_utility: {}
  stop_utility_head: {}
  action_head: {}
  edge_scorer: {}

rollout_cfg:
  train_num_rollout: 8
  coverage_num_rollout: 0
  eval_num_rollout: 8
  train_chunk_size: null
  eval_chunk_size: null

eval_cfg: ${eval_metric}

temperature: 1.0
eval_temperature: 1.0

temperature_cfg:
  temperature_start: 1.0
  temperature_end: 0.8
  temperature_warmup_steps: 300

coverage_cfg:
  enabled: false

proposal_cfg:
  enabled: false

reward_cfg:
  zero_f1_log_reward: -10.0
  log_reward_clip_min: -30.0
  edge_cost: 0.03
  normalize_edge_cost_by_budget: true
```

### `configs/model/weaver_coverage.yaml`

Role: explicit coverage/proposal method override on top of the baseline.

Consumed by:

- Hydra merge on top of `model/weaver.yaml`
- then by `src.weaver.module.WeaverModule` through the same interface as the baseline

Key interface:

- only delta fields are overridden; all unchanged rollout settings inherit from `weaver.yaml`
- `coverage_num_rollout`, `coverage_cfg.enabled`, and `proposal_cfg.enabled` activate the coverage path in `src/weaver/module.py:182-188`

```yaml
defaults:
  - weaver
  - _self_

rollout_cfg:
  coverage_num_rollout: 2

coverage_cfg:
  enabled: true
  path_count_tiebreak_weight: 0.0

proposal_cfg:
  enabled: true
  warmup_steps: 100
  decay_steps: 300
  initial_prob: 0.95
  final_prob: 0.05
  allow_nonzero_final_prob: true
```

### `configs/eval_metric/default.yaml`

Role: evaluation metric and retrieval-eval semantics.

Consumed by:

- injected into `model.eval_cfg`
- validated in `src/weaver/module.py:580-615`
- forwarded to `src/training/rollout_eval.py:27-58`

Key interface:

- `budgets` must be non-empty, all `>= 1`, and each `<= eval_num_rollout`
- `exclude_anchors_from_retrieved` and `use_reachable_targets` are passed through to node retrieval metrics

```yaml
# @package eval_metric

budgets: [1, 2, 4, 8]
debug_metrics: false
use_reachable_targets: true
exclude_anchors_from_retrieved: true
```

### `configs/optimizer/adamw.yaml`

Role: optimizer policy.

Consumed by:

- injected by `configs/train.yaml` into `model.optimizer_cfg`
- parsed in `src/training/optimization.py:152-159,222-299`

Key interface:

- fields must satisfy `AdamWConfig.from_dict(...)` in `src/training/optimization.py`
- `flow_scalar_head_lr_multiplier` is used when building parameter groups in `src/training/optimization.py:281-287`

```yaml
# @package optimizer

type: adamw
lr: 1.0e-4
weight_decay: 1.0e-4
betas: [0.9, 0.999]
flow_scalar_head_lr_multiplier: 1.0
no_decay_on_bias_and_norm: true
```

### `configs/scheduler/cosine_warmup.yaml`

Role: scheduler policy.

Consumed by:

- injected by `configs/train.yaml` into `model.scheduler_cfg`
- parsed in `src/training/optimization.py:164-219`

Key interface:

- `type: cosine_with_warmup` requires `interval: step` in `src/training/optimization.py:189-205`

```yaml
# @package scheduler

type: cosine_with_warmup
interval: step
num_warmup_steps: 30
min_lr_ratio: 0.1
eta_min: 0.0
```

### `configs/trainer/gpu.yaml`

Role: default GPU trainer profile.

Consumed by:

- `src/training/factory.py:80-102`
- Lightning `Trainer` constructor

Key interface:

- every field is forwarded to `lightning.pytorch.trainer.Trainer`
- `gradient_clip_*` is later used by `WeaverModule._optimizer_step()` in `src/weaver/module.py:407-423`
- train callbacks monitor `val/sample/expected_recall`, which exists because `flatten_metric_groups(...)` produces `<stage>/<group>/<metric>` and `compute_sample_retrieval_metrics(...)` emits `expected_recall`

```yaml
# @package trainer

_target_: lightning.pytorch.trainer.Trainer

default_root_dir: ${paths.output_dir}
accelerator: gpu
devices: 1
precision: 32-true

min_epochs: 0
max_epochs: 20
max_steps: 100000

accumulate_grad_batches: 1
benchmark: false
gradient_clip_val: 1.0
gradient_clip_algorithm: norm

val_check_interval: 1.0
check_val_every_n_epoch: 1
use_distributed_sampler: false
num_sanity_val_steps: 0

limit_train_batches: null
limit_val_batches: null
limit_test_batches: null

log_every_n_steps: 10
deterministic: false
enable_checkpointing: true
enable_progress_bar: true
detect_anomaly: false
```

### `configs/trainer/cpu.yaml`

Role: CPU trainer profile.

Consumed by:

- `src/training/factory.py:80-102`
- Lightning `Trainer` constructor

```yaml
# @package trainer

_target_: lightning.pytorch.trainer.Trainer

default_root_dir: ${paths.output_dir}
accelerator: cpu
devices: 1
precision: 32-true

min_epochs: 0
max_epochs: 1
max_steps: 1000

accumulate_grad_batches: 1
benchmark: false
gradient_clip_val: 1.0
gradient_clip_algorithm: norm

val_check_interval: 1.0
check_val_every_n_epoch: 1
use_distributed_sampler: false
num_sanity_val_steps: 0

limit_train_batches: null
limit_val_batches: null
limit_test_batches: null

log_every_n_steps: 1
deterministic: false
enable_checkpointing: false
enable_progress_bar: true
detect_anomaly: false
```

### `configs/trainer/fast_dev.yaml`

Role: fast debug trainer profile.

Consumed by:

- `src/training/factory.py:80-102`
- Lightning `Trainer` constructor

```yaml
# @package trainer

_target_: lightning.pytorch.trainer.Trainer

default_root_dir: ${paths.output_dir}
accelerator: auto
devices: 1
precision: 32-true

fast_dev_run: true
use_distributed_sampler: false
enable_checkpointing: false
enable_progress_bar: true
detect_anomaly: true
```

### `configs/callbacks/train.yaml`

Role: training callback bundle.

Consumed by:

- `src/training/factory.py:80-102`
- `instantiate_many(...)` path in `src/training/factory.py:18-49`

Key interface:

- mapping values are instantiated one by one because `cfg.callbacks` is a mapping without a top-level `_target_`
- each child block must be a valid Hydra target config

```yaml
# @package callbacks

model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  dirpath: ${paths.ckpt_dir}
  filename: "epoch_{epoch:03d}-step_{step:07d}"
  monitor: val/sample/expected_recall
  mode: max
  save_top_k: 3
  save_last: true

learning_rate_monitor:
  _target_: lightning.pytorch.callbacks.LearningRateMonitor
  logging_interval: step

early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: val/sample/expected_recall
  mode: max
  patience: 10
  min_delta: 0.0
  strict: true
```

### `configs/callbacks/eval.yaml`

Role: evaluation callback bundle.

Consumed by:

- `src/training/factory.py:80-102`

```yaml
# @package callbacks

progress_bar:
  _target_: lightning.pytorch.callbacks.TQDMProgressBar
  refresh_rate: 10
```

### `configs/callbacks/local_metrics_writer.yaml`

Role: optional standalone callback target definition.

Consumed by:

- only when referenced from another config mapping or CLI override
- constructor interface is `src.callbacks.local_metrics_writer.LocalMetricsWriter.__init__` in `src/callbacks/local_metrics_writer.py:14-22`

Key interface:

- `output_dir` maps to the callback constructor
- `enabled` is also supported by the callback class, even though this config does not set it explicitly

```yaml
# @package callbacks.local_metrics_writer

_target_: src.callbacks.local_metrics_writer.LocalMetricsWriter
output_dir: ${paths.artifact_dir}/metrics
```

### `configs/logger/wandb.yaml`

Role: W&B logger target.

Consumed by:

- `src/training/factory.py:80-102`

Key interface:

- top-level block is a direct Hydra target config, so `instantiate_many(...)` returns a single logger instance

```yaml
# @package logger

_target_: lightning.pytorch.loggers.wandb.WandbLogger
project: evi-rag
name: ${task_name}-${dataset.name}-${now:%Y%m%d-%H%M%S}
save_dir: ${paths.log_dir}
offline: false
id: null
anonymous: null
log_model: false
prefix: ""
entity: ${oc.env:WANDB_ENTITY, null}
group: ""
tags: []
job_type: ""
```

### `configs/logger/none.yaml`

Role: disable logging.

Consumed by:

- `src/training/factory.py:80-102`

Key interface:

- `null` causes `cfg.get("logger")` to be `None`, so `instantiate_many(...)` returns an empty list and the trainer receives `logger=False`

```yaml
null
```

### `configs/ckpt/default.yaml`

Role: checkpoint path bundle.

Consumed by:

- interpolated into `train.yaml` and `evaluate.yaml`
- then read by `src/train.py:26-30`, `src/evaluate.py:27-31`, and `src/training/checkpoint.py:37-52`

Key interface:

- `fit` becomes `fit_ckpt_path`
- `path` becomes `ckpt_path`
- `pretrained` becomes `pretrained_ckpt_path`
- `strict_pretrained_load` becomes the strictness flag for pretrained loading

```yaml
# @package ckpt

fit: null
path: null
pretrained: null
strict_pretrained_load: false
```

### `configs/experiment/train/webqsp_baseline.yaml`

Role: named baseline training experiment.

Consumed by:

- Hydra merge on top of `train.yaml`

Key interface:

- overrides selected groups and a few scalar/mapping fields at the root level
- `logger.tags` is forwarded to the `WandbLogger`

```yaml
# @package _global_

defaults:
  - override /dataset: webqsp
  - override /model: weaver
  - override /trainer: gpu
  - override /logger: wandb
  - _self_

task_name: train_webqsp_baseline

trainer:
  max_epochs: 20

datamodule:
  batch_size: 8
  num_workers: 4

logger:
  tags: [weaver, baseline, webqsp]
```

### `configs/experiment/train/webqsp_coverage.yaml`

Role: named coverage training experiment.

Consumed by:

- Hydra merge on top of `train.yaml`

```yaml
# @package _global_

defaults:
  - override /dataset: webqsp
  - override /model: weaver_coverage
  - override /trainer: gpu
  - override /logger: wandb
  - _self_

task_name: train_webqsp_coverage

trainer:
  max_epochs: 20

datamodule:
  batch_size: 8
  num_workers: 4

logger:
  tags: [weaver, coverage, webqsp]
```

### `configs/experiment/eval/webqsp.yaml`

Role: named evaluation experiment.

Consumed by:

- Hydra merge on top of `evaluate.yaml`

```yaml
# @package _global_

defaults:
  - override /dataset: webqsp
  - override /model: weaver
  - override /trainer: gpu
  - override /logger: wandb
  - _self_

task_name: eval_webqsp

logger:
  tags: [weaver, eval, webqsp]
```

### `configs/experiment/debug/valfit.yaml`

Role: fast validation-overfit debug experiment.

Consumed by:

- Hydra merge on top of `train.yaml`

Key interface:

- rewires all dataset splits to `validation`
- shrinks dataloader parallelism and rollout counts
- swaps trainer profile to `fast_dev`
- swaps callbacks to the lightweight eval bundle
- turns on `eval_metric.debug_metrics`

```yaml
# @package _global_

defaults:
  - override /dataset: webqsp
  - override /model: weaver
  - override /trainer: fast_dev
  - override /callbacks: eval
  - override /logger: none
  - _self_

task_name: debug_valfit

datamodule:
  batch_size: 2
  num_workers: 0
  eval_batch_size: 2
  eval_num_workers: 0

dataset:
  splits:
    train: validation
    validation: validation
    test: validation

model:
  rollout_cfg:
    train_num_rollout: 2
    coverage_num_rollout: 0
    eval_num_rollout: 2
    train_chunk_size: null
    eval_chunk_size: null

eval_metric:
  budgets: [1, 2]
  debug_metrics: true
```

## Notes

- `configs/__init__.py` is present but empty; it is only a package marker.
- `paths.run_dir`, `paths.log_dir`, `paths.ckpt_dir`, and `paths.artifact_dir` resolve fully only in a real Hydra run. In `--cfg job --resolve` compose-only mode, Hydra runtime values can still appear as `null`.
- `model.eval_cfg` is intentionally externalized to `eval_metric/default.yaml`; `python src/train.py experiment=debug/valfit --cfg job --resolve` confirms `model.eval_cfg.debug_metrics: true` after experiment overrides.
- `logger/none.yaml` is the clean disable switch. No backward-compat wrapper is retained.
- `callbacks/local_metrics_writer.yaml` is a reusable target definition, not part of the default train/eval callback bundles unless you add it explicitly.
