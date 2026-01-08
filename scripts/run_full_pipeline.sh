#!/usr/bin/env bash
set -euo pipefail

# 极简 EB-GFN 全流程：让 Hydra 按默认规则管理日志/输出目录。
# 仅依赖默认的 config/experiment，必须显式传入 dataset。
#
# 用法：
#   bash scripts/run_full_pipeline.sh <dataset> [--skip-preprocess]
#
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET=""
DATASET_FAMILY=""
DATASET_FULL=""
DATASET_SUB=""
SKIP_PREPROCESS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-preprocess)
      SKIP_PREPROCESS="true"
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_full_pipeline.sh <dataset> [--skip-preprocess]" >&2
      exit 0
      ;;
    *)
      if [[ -z "${DATASET}" ]]; then
        DATASET="${1}"
        shift
      else
        echo "Unknown argument: $1" >&2
        echo "Usage: bash scripts/run_full_pipeline.sh <dataset> [--skip-preprocess]" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "${DATASET}" ]]; then
  echo "Usage: bash scripts/run_full_pipeline.sh <dataset> [--skip-preprocess]" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\\.yaml$//' >&2
  exit 2
fi
DATASET_FAMILY="${DATASET%-sub}"
DATASET_FULL="${DATASET_FAMILY}"
DATASET_SUB="${DATASET_FAMILY}-sub"
if [[ ! -f "configs/dataset/${DATASET_FULL}.yaml" ]]; then
  echo "Unknown full dataset: ${DATASET_FULL}" >&2
  ls -1 configs/dataset/*.yaml | xargs -n1 basename | sed 's/\\.yaml$//' >&2
  exit 2
fi
if [[ ! -f "configs/dataset/${DATASET_SUB}.yaml" ]]; then
  echo "Missing sub dataset config: ${DATASET_SUB}" >&2
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

COMMON_OVERRIDES_FULL=("dataset=${DATASET_FULL}" "paths=default" "hydra.job.chdir=false")
COMMON_OVERRIDES_SUB=("dataset=${DATASET_SUB}" "paths=default" "hydra.job.chdir=false")

latest_run_dir() {
  local task="$1"
  local dataset="$2"
  local base="${ROOT}/logs/${task}_${dataset}/runs"
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

if [[ "${SKIP_PREPROCESS}" != "true" ]]; then
  echo "==> [1/5] build_retrieval_pipeline (full=${DATASET_FULL}, emit sub mask)"
  python scripts/build_retrieval_pipeline.py "dataset=${DATASET_FULL}" "paths=default" "hydra.job.chdir=false"
else
  echo "==> [1/5] build_retrieval_pipeline (full=${DATASET_FULL}) [skipped]"
fi

echo "==> [2/5] train_mpm_rag (EB-GFN, sub=${DATASET_SUB})"
MPM_EXP="train_mpm_rag"
MPM_TRAIN_OVERRIDES=("${COMMON_OVERRIDES_SUB[@]}" "experiment=${MPM_EXP}")
python src/train.py "${MPM_TRAIN_OVERRIDES[@]}"
MPM_RUN_DIR="$(latest_run_dir "${MPM_EXP}" "${DATASET_SUB}")"
MPM_CKPT="$(pick_best_ckpt "${MPM_RUN_DIR}")"
if [[ -z "${MPM_CKPT}" ]]; then
  echo "MPM-RAG checkpoint not found under run dir: ${MPM_RUN_DIR:-<missing>}" >&2
  exit 1
fi
echo "MPM-RAG checkpoint: ${MPM_CKPT}"

echo "==> [3/4] eval_gflownet (EB-GFN; full+sub)"
python src/eval.py \
  "${COMMON_OVERRIDES_FULL[@]}" \
  "experiment=eval_gflownet" \
  "ckpt.gflownet=${MPM_CKPT}"

echo "==> [4/4] export_gflownet (EB-GFN; full+sub)"
python src/eval.py \
  "${COMMON_OVERRIDES_FULL[@]}" \
  "experiment=export_gflownet" \
  "ckpt.gflownet=${MPM_CKPT}"

echo "Pipeline finished."
