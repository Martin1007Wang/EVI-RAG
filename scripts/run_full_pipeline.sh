#!/usr/bin/env bash
set -euo pipefail

# Full pipeline runner (dataset-only entry).
#
# Usage:
#   bash scripts/run_full_pipeline.sh <dataset>
#
# Optional environment variables (advanced):
#   DATA_DIR=/path/to/retrieval_dataset_root   # overrides `paths.data_dir`
#   SKIP_BUILD_PARQUET=1                       # skip scripts/build_retrieval_parquet.py
#   SKIP_BUILD_DATASET=1                       # skip scripts/build_retrieval_dataset.py
#   USE_WANDB=1                               # enable wandb logger (default: disabled)
#   TRAINER_RETRIEVER=gpu|cpu|ddp|...          # overrides retriever training trainer group
#   TRAINER_GFLOWNET=gpu|cpu|ddp|...           # overrides gflownet training trainer group
#   FORCE_INCLUDE_GT_TRAIN=1                   # enable oracle GT injection for train g_agent only (default: 0)
#   RUN_LLM=1                                 # also run LLM stages (triplet windows + path prompts)
#   LLM_BACKEND=auto|vllm|openai|ollama        # only used when RUN_LLM=1
#   LLM_MODEL_NAME=...                         # only used when RUN_LLM=1
#   TRIPLET_TOKEN_BUDGET=2048                  # only used when RUN_LLM=1 (sets data.token_budget)
#
# Outputs:
#   logs/full_pipeline/<dataset>/<ts>/
#     artifacts.env  (paths to retriever/gflownet checkpoints)
#     *.log          (stdout/stderr per step)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
  echo "Usage: bash scripts/run_full_pipeline.sh <dataset>" >&2
  echo "Available datasets:" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' >&2
  exit 2
fi
if [[ ! -f "configs/dataset/${DATASET}.yaml" ]]; then
  echo "Unknown dataset: ${DATASET}" >&2
  echo "Available datasets:" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' >&2
  exit 2
fi

export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

# Some sandboxes disallow /dev/shm; Intel OpenMP may abort. Only patch the env when shm is not writable.
if [[ -d /dev/shm ]]; then
  if ! (touch /dev/shm/_evi_rag_shm_test 2>/dev/null); then
    export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
  else
    rm -f /dev/shm/_evi_rag_shm_test 2>/dev/null || true
  fi
fi

TS="$(date +%Y-%m-%d_%H-%M-%S)"
PIPE_ROOT="${PIPE_ROOT:-${ROOT}/logs/full_pipeline/${DATASET}/${TS}}"
mkdir -p "${PIPE_ROOT}"

COMMON_OVERRIDES=("dataset=${DATASET}" "hydra.job.chdir=false")
if [[ -n "${DATA_DIR:-}" ]]; then
  COMMON_OVERRIDES+=("paths.data_dir=${DATA_DIR}")
fi

TRAIN_LOGGER_OVERRIDES=()
if [[ "${USE_WANDB:-0}" != "1" ]]; then
  TRAIN_LOGGER_OVERRIDES+=("logger=none")
  export WANDB_MODE="${WANDB_MODE:-disabled}"
fi

run_step() {
  local name="$1"
  shift
  local log_path="${PIPE_ROOT}/${name}.log"
  echo "==> [${name}] $*" | tee "${log_path}"
  "$@" 2>&1 | tee -a "${log_path}"
}

pick_best_ckpt() {
  local ckpt_dir="$1"
  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "" && return 0
  fi
  local best
  best="$(ls -1 "${ckpt_dir}"/epoch_*.ckpt 2>/dev/null | head -n 1 || true)"
  if [[ -n "${best}" ]]; then
    echo "${best}" && return 0
  fi
  if [[ -f "${ckpt_dir}/last.ckpt" ]]; then
    echo "${ckpt_dir}/last.ckpt" && return 0
  fi
  echo ""
}

echo "Pipeline root: ${PIPE_ROOT}"
echo "Dataset: ${DATASET}"

# 1) Build normalized parquet
if [[ "${SKIP_BUILD_PARQUET:-0}" == "1" ]]; then
  echo "Skipping build_retrieval_parquet (SKIP_BUILD_PARQUET=1)." | tee "${PIPE_ROOT}/01_build_retrieval_parquet.skipped.log"
else
  run_step "01_build_retrieval_parquet" \
    python scripts/build_retrieval_parquet.py \
    "${COMMON_OVERRIDES[@]}" \
    "hydra.run.dir=${PIPE_ROOT}/hydra/01_build_retrieval_parquet"
fi

# 2) Build LMDB/embeddings
if [[ "${SKIP_BUILD_DATASET:-0}" == "1" ]]; then
  echo "Skipping build_retrieval_dataset (SKIP_BUILD_DATASET=1)." | tee "${PIPE_ROOT}/02_build_retrieval_dataset.skipped.log"
else
  run_step "02_build_retrieval_dataset" \
    python scripts/build_retrieval_dataset.py \
    "${COMMON_OVERRIDES[@]}" \
    "hydra.run.dir=${PIPE_ROOT}/hydra/02_build_retrieval_dataset"
fi

# 3) Train retriever
RETR_EXP="train_retriever_${DATASET}"
if [[ ! -f "configs/experiment/${RETR_EXP}.yaml" ]]; then
  RETR_EXP="train_retriever_default"
fi
RETR_TRAIN_DIR="${PIPE_ROOT}/hydra/03_train_retriever"
RETR_TRAIN_OVERRIDES=("${COMMON_OVERRIDES[@]}" "${TRAIN_LOGGER_OVERRIDES[@]}" "experiment=${RETR_EXP}" "hydra.run.dir=${RETR_TRAIN_DIR}")
if [[ -n "${TRAINER_RETRIEVER:-}" ]]; then
  RETR_TRAIN_OVERRIDES+=("trainer=${TRAINER_RETRIEVER}")
fi
run_step "03_train_retriever" python src/train.py "${RETR_TRAIN_OVERRIDES[@]}"

RETR_CKPT="$(pick_best_ckpt "${RETR_TRAIN_DIR}/checkpoints")"
if [[ -z "${RETR_CKPT}" ]]; then
  echo "Retriever checkpoint not found under: ${RETR_TRAIN_DIR}/checkpoints" >&2
  exit 1
fi
echo "Retriever checkpoint: ${RETR_CKPT}"

# 4) Evaluate retriever (persist eval cache for oracle/truth stage)
run_step "04_eval_retriever" \
  python src/eval.py \
  "${COMMON_OVERRIDES[@]}" \
  "stage=retriever_eval" \
  "ckpt.retriever=${RETR_CKPT}" \
  "hydra.run.dir=${PIPE_ROOT}/hydra/04_eval_retriever"

# 5) Materialize g_agent caches (train/validation/test)
FORCE_GT_TRAIN="${FORCE_INCLUDE_GT_TRAIN:-0}"
for SPLIT in train validation test; do
  FORCE_GT="false"
  if [[ "${SPLIT}" == "train" && "${FORCE_GT_TRAIN}" == "1" ]]; then
    FORCE_GT="true"
  fi
  run_step "05_materialize_g_agent_${SPLIT}" \
    python src/eval.py \
    "${COMMON_OVERRIDES[@]}" \
    "stage=materialize_g_agent" \
    "ckpt.retriever=${RETR_CKPT}" \
    "stage.split=${SPLIT}" \
    "stage.force_include_gt=${FORCE_GT}" \
    "hydra.run.dir=${PIPE_ROOT}/hydra/05_materialize_g_agent_${SPLIT}"
done

# 6) Train GFlowNet (requires retriever ckpt as projector init)
GFLOW_EXP="train_gflownet_${DATASET}"
if [[ ! -f "configs/experiment/${GFLOW_EXP}.yaml" ]]; then
  GFLOW_EXP="train_gflownet_default"
fi
GFLOW_TRAIN_DIR="${PIPE_ROOT}/hydra/06_train_gflownet"
GFLOW_TRAIN_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
  "${TRAIN_LOGGER_OVERRIDES[@]}"
  "experiment=${GFLOW_EXP}"
  "model.embedder_cfg.projector_checkpoint=${RETR_CKPT}"
  "hydra.run.dir=${GFLOW_TRAIN_DIR}"
)
if [[ -n "${TRAINER_GFLOWNET:-}" ]]; then
  GFLOW_TRAIN_OVERRIDES+=("trainer=${TRAINER_GFLOWNET}")
fi
run_step "06_train_gflownet" python src/train.py "${GFLOW_TRAIN_OVERRIDES[@]}"

GFLOW_CKPT="$(pick_best_ckpt "${GFLOW_TRAIN_DIR}/checkpoints")"
if [[ -z "${GFLOW_CKPT}" ]]; then
  echo "GFlowNet checkpoint not found under: ${GFLOW_TRAIN_DIR}/checkpoints" >&2
  exit 1
fi
echo "GFlowNet checkpoint: ${GFLOW_CKPT}"

# 7) Evaluate GFlowNet (persist rollouts for path prompting)
run_step "07_eval_gflownet" \
  python src/eval.py \
  "${COMMON_OVERRIDES[@]}" \
  "stage=gflownet_eval" \
  "ckpt.retriever=${RETR_CKPT}" \
  "ckpt.gflownet=${GFLOW_CKPT}" \
  "hydra.run.dir=${PIPE_ROOT}/hydra/07_eval_gflownet"

# 8) Retriever oracle upper bound (no LLM)
run_step "08_oracle_truth" \
  python src/eval.py \
  "${COMMON_OVERRIDES[@]}" \
  "stage=llm_reasoner_truth" \
  "hydra.run.dir=${PIPE_ROOT}/hydra/08_oracle_truth"

# 9) Optional: run LLM stages (requires a configured backend)
if [[ "${RUN_LLM:-0}" == "1" ]]; then
  LLM_OVERRIDES=()
  if [[ -n "${LLM_BACKEND:-}" ]]; then
    LLM_OVERRIDES+=("model.backend=${LLM_BACKEND}")
  fi
  if [[ -n "${LLM_MODEL_NAME:-}" ]]; then
    LLM_OVERRIDES+=("model.model_name=${LLM_MODEL_NAME}")
  fi
  if [[ -n "${TRIPLET_TOKEN_BUDGET:-}" ]]; then
    LLM_OVERRIDES+=("data.token_budget=${TRIPLET_TOKEN_BUDGET}")
  fi
  run_step "09_llm_reasoner_triplet" \
    python src/eval.py \
    "${COMMON_OVERRIDES[@]}" \
    "stage=llm_reasoner_triplet" \
    "${LLM_OVERRIDES[@]}" \
    "hydra.run.dir=${PIPE_ROOT}/hydra/09_llm_reasoner_triplet"
  run_step "10_llm_reasoner_paths" \
    python src/eval.py \
    "${COMMON_OVERRIDES[@]}" \
    "stage=llm_reasoner_paths" \
    "${LLM_OVERRIDES[@]}" \
    "hydra.run.dir=${PIPE_ROOT}/hydra/10_llm_reasoner_paths"
else
  echo "Skipping LLM stages. To enable: RUN_LLM=1 (and configure backend/model via env vars)." | tee "${PIPE_ROOT}/09_llm_skipped.log"
fi

cat > "${PIPE_ROOT}/artifacts.env" <<EOF
DATASET='${DATASET}'
PIPE_ROOT='${PIPE_ROOT}'
RETRIEVER_CKPT='${RETR_CKPT}'
GFLOWNET_CKPT='${GFLOW_CKPT}'
EOF

echo "Done. Artifacts written to: ${PIPE_ROOT}/artifacts.env"
