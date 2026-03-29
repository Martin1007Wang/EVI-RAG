## GFlowNet Import Migration

The legacy prefix-tree GFlowNet stack has been removed.

Do not import from these deleted module paths anymore:

- `src.models.gflownet.prefix_state`
- `src.models.gflownet.prefix_policy`
- `src.models.gflownet.prefix_sampler`
- `src.models.gflownet.prefix_losses`
- `src.models.gflownet.prefix_backward`
- `src.models.gflownet.answer_supervision`
- `src.models.gflownet.heuristics`
- `src.models.gflownet.legality`
- `src.models.gflownet.memory`
- `src.models.gflownet.path`
- `src.models.gflownet.prefix`
- `src.models.gflownet.repetition`
- `src.models.gflownet.replay`
- `src.models.gflownet.success_paths`
- `src.models.gflownet.transitions`
- `src.metrics.answer_metrics`
- `src.metrics.edge_metrics`
- `src.metrics.search_backends`

Use the canonical subgraph layout instead.

### Canonical modules

- package entrypoint: `src.models.gflownet`
- subgraph package: `src.models.gflownet.subgraph`
- state containers: `src.models.gflownet.subgraph.state`
- prepared batches: `src.models.gflownet.subgraph.prepared_batch`
- environment and rewards: `src.models.gflownet.subgraph.mdp`
- policy: `src.models.gflownet.subgraph.policy`
- sampler: `src.models.gflownet.subgraph.sampler`
- beam search: `src.models.gflownet.subgraph.search`
- SubTB loss: `src.models.gflownet.subgraph.losses`
- answer extraction: `src.models.gflownet.subgraph.answers`
- runtime: `src.metrics.subgraph_answer_search_runtime`

### Root package policy

Use `src.models.gflownet` only for the high-level entrypoints below:

- `GFlowNetPolicyFactory`
- `SubgraphPolicy`
- `SubgraphSampler`
- `SubgraphSubTrajectoryBalanceLoss`
- `SubgraphState`
- `SubgraphAction`
- `SubgraphEnv`
- `beam_search_subgraphs`
- `subgraph`

Everything else should be imported from its owning subgraph module.
