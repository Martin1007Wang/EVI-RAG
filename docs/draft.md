# Draft Notes

This file is archival only.

It predates the current RankFlow config cleanup, the removal of auxiliary
guidance losses, the switch to Monte Carlo fit-loop validation, and the
proposal-only action-prior refactor.

In particular, older references in prior drafts to:

- `train_answer_reachability` as the canonical training bundle
- flow-frontier training-time validation
- legacy node/edge batch caps in old experiment files
 - removed train/eval alias bundles such as `answer_reachability`

should be treated as stale.

For the current algorithm and public config surface, read these files instead:

- `docs/gflownet_semantics.md`
- `docs/gflownet_refactor_notes.md`
- `configs/experiment/train_rankflow.yaml`
- `configs/model/gflownet.yaml`
