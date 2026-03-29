## GFlowNet Import Migration

The old compatibility wrappers have been removed.

`src.models.gflownet` is now a small curated public API, not a catch-all export
barrel. If you need a low-level type, helper, or utility, import it from the
explicit module that owns it.

Do not import from these deleted module paths anymore:

- `src.models.gflownet.types`
- `src.models.gflownet.policy`
- `src.models.gflownet.sampler`
- `src.models.gflownet.losses`
- `src.models.gflownet.backward`
- `src.models.gflownet.subgraph_state`
- `src.models.gflownet.subgraph_env`
- `src.models.gflownet.subgraph_policy`
- `src.models.gflownet.subgraph_sampler`
- `src.metrics.subgraph_runtime`

Use the canonical module layout instead.

### Prefix-tree search stack

- state and protocols: `src.models.gflownet.prefix_state`
- policy: `src.models.gflownet.prefix_policy`
- rollout sampler: `src.models.gflownet.prefix_sampler`
- SubTB loss: `src.models.gflownet.prefix_losses`
- backward distribution helpers: `src.models.gflownet.prefix_backward`

### Subgraph-growth search stack

- package entrypoint: `src.models.gflownet.subgraph`
- state containers: `src.models.gflownet.subgraph.state`
- prepared batch construction: `src.models.gflownet.subgraph.prepared_batch`
- state dynamics and rewards: `src.models.gflownet.subgraph.mdp`
- policy and frontier scoring: `src.models.gflownet.subgraph.policy`
- rollout sampler: `src.models.gflownet.subgraph.sampler`
- beam search: `src.models.gflownet.subgraph.search`
- losses: `src.models.gflownet.subgraph.losses`
- answer extraction helpers: `src.models.gflownet.subgraph.answers`

### Runtime

- canonical subgraph runtime: `src.metrics.subgraph_answer_search_runtime`

### Common replacements

| Removed import | Use instead |
| --- | --- |
| `from src.models.gflownet.types import SearchState` | `from src.models.gflownet.prefix_state import SearchState` |
| `from src.models.gflownet.sampler import ForwardTrajectoryGFNSampler` | `from src.models.gflownet.prefix_sampler import ForwardTrajectoryGFNSampler` |
| `from src.models.gflownet.losses import SubTrajectoryBalanceLoss` | `from src.models.gflownet.prefix_losses import SubTrajectoryBalanceLoss` |
| `from src.models.gflownet.policy import BaseSearchPolicy` | `from src.models.gflownet.prefix_policy import BaseSearchPolicy` |
| `from src.models.gflownet.subgraph_policy import SubgraphPolicy` | `from src.models.gflownet.subgraph.policy import SubgraphPolicy` |
| `from src.models.gflownet.subgraph_sampler import beam_search_subgraphs` | `from src.models.gflownet.subgraph.search import beam_search_subgraphs` |
| `from src.metrics.subgraph_runtime import SubgraphAnswerSearchRuntime` | `from src.metrics.subgraph_answer_search_runtime import SubgraphAnswerSearchRuntime` |

### Root package policy

Use `src.models.gflownet` only for the high-level entrypoints below:

- `BaseSearchPolicy`
- `GFlowNetPolicy`
- `ForwardTrajectoryGFNSampler`
- `SubTrajectoryBalanceLoss`
- `SearchState`
- `SubgraphPolicy`
- `SubgraphSampler`
- `SubgraphSubTrajectoryBalanceLoss`
- `SubgraphState`
- `SubgraphAction`
- `SubgraphEnv`
- `beam_search_subgraphs`
- `subgraph`

Everything else should be imported from its owning module.

### Subgraph package policy

Use `src.models.gflownet.subgraph` only for the subgraph high-level entrypoints below:

- `SubgraphPolicy`
- `SubgraphSampler`
- `SubgraphSubTrajectoryBalanceLoss`
- `SubgraphState`
- `SubgraphAction`
- `SubgraphEnv`
- `beam_search_subgraphs`

For lower-level internals, import from the explicit module instead:

- state records: `src.models.gflownet.subgraph.state`
- prepared batches: `src.models.gflownet.subgraph.prepared_batch`
- search result dataclasses: `src.models.gflownet.subgraph.search`
- answer helpers: `src.models.gflownet.subgraph.answers`
- loss outputs: `src.models.gflownet.subgraph.losses`

### Reading order

If you are starting from the current mainline RankFlow implementation, read in this order:

1. `configs/experiment/train_rankflow.yaml`
2. `src/models/gflownet/subgraph/state.py`
3. `src/models/gflownet/subgraph/prepared_batch.py`
4. `src/models/gflownet/subgraph/mdp.py`
5. `src/models/gflownet/subgraph/policy.py`
6. `src/models/gflownet/subgraph/sampler.py`
7. `src/models/gflownet/subgraph/search.py`
8. `src/models/gflownet/subgraph/losses.py`
9. `src/metrics/subgraph_answer_search_runtime.py`
