# DF-VGR Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor GFlowNet to DF-VGR: blind forward policy, prior-biased backward policy, hierarchical STOP, and TB loss with learnable log Z (no state flow prediction).

**Architecture:** Replace log F with log Z(q) and remove h-transform. Forward policy uses agent scores only, backward policy adds static prior bias. STOP is computed from state+question and competes with edge logsumexp.

**Tech Stack:** PyTorch, PyTorch Lightning, torch.

### Task 1: Add failing tests for DF-VGR policy math

**Files:**
- Create: `tests/models/components/test_dfvgr_policy_math.py`

**Step 1: Write failing test for hierarchical STOP + edge conditional**

```python
import math
import torch
from src.models.components.gflownet_ops import compute_forward_log_probs

def test_forward_log_probs_hierarchical_stop():
    edge_scores = torch.tensor([0.0, 0.0])
    stop_logits = torch.tensor([0.0])
    policy = compute_forward_log_probs(
        edge_scores=edge_scores,
        stop_logits=stop_logits,
        allow_stop=torch.tensor([True]),
        edge_batch=torch.tensor([0, 0]),
        num_graphs=1,
        temperature=1.0,
        edge_guidance=None,
        edge_valid_mask=torch.tensor([True, True]),
    )
    log3 = math.log(3.0)
    log2 = math.log(2.0)
    assert torch.allclose(policy.stop, torch.tensor([-log3]))
    assert torch.allclose(policy.move, torch.tensor([log2 - log3]))
    assert torch.allclose(policy.edge_cond, torch.tensor([-log2, -log2]))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/components/test_dfvgr_policy_math.py::test_forward_log_probs_hierarchical_stop -q`
Expected: FAIL (PolicyLogProbs shape/field mismatch)

**Step 3: Write failing test for static prior S(v|q)**

```python
import torch
from src.models.gflownet_module import GFlowNetModule

def test_compute_node_prior_sigmoid_dot():
    node_tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    question_tokens = torch.tensor([[1.0, 1.0]])
    node_batch = torch.tensor([0, 0])
    prior = GFlowNetModule._compute_node_prior(
        node_tokens=node_tokens,
        question_tokens=question_tokens,
        node_batch=node_batch,
    )
    expected = torch.tensor([0.7310586, 0.7310586])
    assert torch.allclose(prior, expected, atol=1e-5)
```

**Step 4: Run tests to verify they fail**

Run: `pytest tests/models/components/test_dfvgr_policy_math.py -q`
Expected: FAIL (missing method or wrong outputs)

### Task 2: Implement DF-VGR policy math in ops + actor

**Files:**
- Modify: `src/models/components/gflownet_ops.py`
- Modify: `src/models/components/gflownet_actor.py`

**Step 1: Update PolicyLogProbs and compute_forward_log_probs**

```python
@dataclass(frozen=True)
class PolicyLogProbs:
    edge_cond: torch.Tensor
    stop: torch.Tensor
    move: torch.Tensor
    has_edge: torch.Tensor
```

```python
# edge_cond = log P(edge | move)
# move = log P(move)
```

**Step 2: Update sampling to use move + edge_cond**

```python
log_pf = torch.where(choose_stop, policy.stop, policy.move + log_prob_edge_sel)
```

**Step 3: Run tests**

Run: `pytest tests/models/components/test_dfvgr_policy_math.py::test_forward_log_probs_hierarchical_stop -q`
Expected: PASS

**Step 4: Commit**

```bash
git add src/models/components/gflownet_ops.py src/models/components/gflownet_actor.py tests/models/components/test_dfvgr_policy_math.py
git commit -m "refactor: df-vgr policy math"
```

### Task 3: Replace log F with log Z and add backward prior

**Files:**
- Modify: `src/models/gflownet_module.py`
- Modify: `src/models/components/gflownet_layers.py` (if new log Z head lives here)
- Modify: `configs/model/gflownet_module.yaml`

**Step 1: Add log Z head and node prior helper**

```python
self.log_z = nn.Linear(self.hidden_dim, 1)
```

```python
@staticmethod
def _compute_node_prior(...):
    score = (node_tokens * question_tokens.index_select(0, node_batch)).sum(dim=-1)
    return torch.sigmoid(score)
```

**Step 2: Remove log_f usage in TB loss**

```python
residual = log_z + sum_pf - sum_pb - log_reward
```

**Step 3: Add backward prior bias in actor/module**

```python
edge_scores = edge_scores + beta * node_prior.index_select(0, tail_nodes)
```

**Step 4: Update config**

```yaml
actor_cfg:
  prior_beta: 5.0
```

**Step 5: Run tests**

Run: `pytest tests/models/components/test_dfvgr_policy_math.py::test_compute_node_prior_sigmoid_dot -q`
Expected: PASS

**Step 6: Commit**

```bash
git add src/models/gflownet_module.py src/models/components/gflownet_layers.py configs/model/gflownet_module.yaml
git commit -m "refactor: use log Z and backward prior"
```

### Task 4: Update documentation to DF-VGR spec

**Files:**
- Modify: `docs/method.md`

**Step 1: Replace with DF-VGR specification**

```markdown
# Dual-Flow Variational Graph Retrieval (DF-VGR)
...
```

**Step 2: Commit**

```bash
git add docs/method.md
git commit -m "docs: df-vgr specification"
```

