#!/usr/bin/env bash
set -euo pipefail

# Mask ablation for retriever on a single dataset (default: cwq).
# Runs: build_retrieval_pipeline -> train (mask off/on) -> eval (mask off/on).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET="cwq"
SKIP_PREPROCESS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-preprocess)
      SKIP_PREPROCESS="true"
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_retriever_mask_ablation.sh [dataset] [--skip-preprocess]" >&2
      exit 0
      ;;
    *)
      if [[ "${DATASET}" == "cwq" ]]; then
        DATASET="${1}"
        shift
      else
        echo "Unknown argument: $1" >&2
        echo "Usage: bash scripts/run_retriever_mask_ablation.sh [dataset] [--skip-preprocess]" >&2
        exit 2
      fi
      ;;
  esac
done

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
  echo "==> [1/3] build_retrieval_pipeline (full=${DATASET_FULL})"
  python scripts/build_retrieval_pipeline.py "+dataset=${DATASET_FULL}" "+paths=default" "hydra.job.chdir=false"
else
  echo "==> [1/3] build_retrieval_pipeline (full=${DATASET_FULL}) [skipped]"
fi

run_variant() {
  local label="$1"
  local train_exp="$2"
  local eval_exp="$3"

  echo "==> [2/3] train_retriever (${label}, sub=${DATASET_SUB})"
  python src/train.py "${COMMON_OVERRIDES_SUB[@]}" "experiment=${train_exp}"
  local run_dir
  run_dir="$(latest_run_dir "${train_exp}" "${DATASET_SUB}")"
  local ckpt
  ckpt="$(pick_best_ckpt "${run_dir}")"
  if [[ -z "${ckpt}" ]]; then
    echo "Retriever checkpoint not found under run dir: ${run_dir:-<missing>}" >&2
    exit 1
  fi
  echo "Retriever checkpoint (${label}): ${ckpt}"

  echo "==> [3/3] eval_retriever (${label}, full+sub)"
  python src/eval.py \
    "${COMMON_OVERRIDES_FULL[@]}" \
    "experiment=${eval_exp}" \
    "ckpt.retriever=${ckpt}" \
    "run.run_all_splits=true"
}

run_variant "maskoff" "train_retriever_maskoff" "eval_retriever_maskoff"
run_variant "maskon" "train_retriever_maskon" "eval_retriever_maskon"

echo "Mask ablation finished."
