# Documentation Index

- `docs/dualflow_method.md`: DualFlow (code-exact) algorithm spec + **static uniform PB**.
- `docs/webqsp_data_cleaning_stats.md`: WebQSP data cleaning statistics.

Implementation SSOT:

- Core model/training loop: `src/models/dual_flow_module.py`
- PB config: `configs/model/dual_flow.yaml` (`training_cfg.db_cfg.pb_edge_dropout`)
- Training experiments: `configs/experiment/train_dual_flow_p0_{none,degree,semantic}.yaml`
