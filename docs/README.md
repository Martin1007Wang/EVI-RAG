# Documentation Index

- `docs/dualflow_method.md`: DualFlow (code-exact) algorithm spec + **three PB strategies** (`uniform`, `topo_semantic`, `learned`).
- `docs/webqsp_data_cleaning_stats.md`: WebQSP data cleaning statistics.

Implementation SSOT:

- Core model/training loop: `src/models/dual_flow_module.py`
- PB configs: `configs/model/db/{uniform,topo_semantic,learned}.yaml`
- Training experiments: `configs/experiment/train_gflownet_pb_{uniform,topo_semantic,learned}.yaml`
