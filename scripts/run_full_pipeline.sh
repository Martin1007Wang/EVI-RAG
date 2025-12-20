#!/usr/bin/env bash
set -euo pipefail

# 极简全流程：让 Hydra 按默认规则管理日志/输出目录，不再聚合到自定义文件夹。
# 仅依赖默认的 config/experiment，必须显式传入 dataset。
#
# 用法：
#   bash scripts/run_full_pipeline.sh <dataset>
#
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
  echo "Usage: bash scripts/run_full_pipeline.sh <dataset>" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\\.yaml$//' >&2
  exit 2
fi
if [[ ! -f "configs/dataset/${DATASET}.yaml" ]]; then
  echo "Unknown dataset: ${DATASET}" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\\.yaml$//' >&2
  exit 2
fi

export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ -d /dev/shm ]]; then
  if ! (touch /dev/shm/_evi_rag_shm_test 2>/dev/null); then
    export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
  else
    rm -f /dev/shm/_evi_rag_shm_test 2>/dev/null || true
  fi
fi

COMMON_OVERRIDES=("dataset=${DATASET}" "hydra.job.chdir=false")

latest_run_dir() {
  local task="$1"
  local base="${ROOT}/logs/${task}/runs"
  ls -1dt "${base}"/* 2>/dev/null | head -n 1
}

pick_best_ckpt() {
  local run_dir="$1"
  local ckpt_dir="${run_dir}/checkpoints"
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

echo "==> [1/8] build_retrieval_parquet (${DATASET})"
python scripts/build_retrieval_parquet.py "${COMMON_OVERRIDES[@]}"

echo "==> [2/8] build_retrieval_dataset (${DATASET})"
python scripts/build_retrieval_dataset.py "${COMMON_OVERRIDES[@]}"

echo "==> [3/8] train_retriever (experiment=train_retriever_default)"
RETR_EXP="train_retriever_default"
RETR_TRAIN_OVERRIDES=("${COMMON_OVERRIDES[@]}" "experiment=${RETR_EXP}")
python src/train.py "${RETR_TRAIN_OVERRIDES[@]}"
RETR_RUN_DIR="$(latest_run_dir "${RETR_EXP}")"
RETR_CKPT="$(pick_best_ckpt "${RETR_RUN_DIR}")"
if [[ -z "${RETR_CKPT}" ]]; then
  echo "Retriever checkpoint not found under run dir: ${RETR_RUN_DIR:-<missing>}" >&2
  exit 1
fi
echo "Retriever checkpoint: ${RETR_CKPT}"

echo "==> [4/8] cache_g_agent (train/val/test)"
for SPLIT in train validation test; do
  FORCE_GT="false"
  if [[ "${SPLIT}" == "train" ]]; then
    FORCE_GT="true"
  fi
  python src/eval.py \
    "${COMMON_OVERRIDES[@]}" \
    "stage=cache_g_agent" \
    "ckpt.retriever=${RETR_CKPT}" \
    "stage.split=${SPLIT}" \
    "stage.force_include_gt=${FORCE_GT}"
done

echo "==> [5/8] train_gflownet (experiment=train_gflownet_default)"
GFLOW_EXP="train_gflownet_default"
GFLOW_TRAIN_OVERRIDES=("${COMMON_OVERRIDES[@]}" "experiment=${GFLOW_EXP}")
python src/train.py "${GFLOW_TRAIN_OVERRIDES[@]}"
GFLOW_RUN_DIR="$(latest_run_dir "${GFLOW_EXP}")"
GFLOW_CKPT="$(pick_best_ckpt "${GFLOW_RUN_DIR}")"
if [[ -z "${GFLOW_CKPT}" ]]; then
  echo "GFlowNet checkpoint not found under run dir: ${GFLOW_RUN_DIR:-<missing>}" >&2
  exit 1
fi
echo "GFlowNet checkpoint: ${GFLOW_CKPT}"

echo "==> [6/8] eval_gflownet"
python src/eval.py \
  "${COMMON_OVERRIDES[@]}" \
  "stage=gflownet_eval" \
  "ckpt.gflownet=${GFLOW_CKPT}"

echo "==> [7/8] oracle truth (retriever upper bound)"
python src/eval.py \
  "${COMMON_OVERRIDES[@]}" \
  "stage=llm_reasoner_truth"

echo "Pipeline finished."
