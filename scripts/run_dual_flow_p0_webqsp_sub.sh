#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

python src/train.py experiment=train_dual_flow_p0_none dataset=webqsp-sub
python src/train.py experiment=train_dual_flow_p0_indegree dataset=webqsp-sub
python src/train.py experiment=train_dual_flow_p0_semantic dataset=webqsp-sub
