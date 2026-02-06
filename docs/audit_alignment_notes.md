Audit Notes: Methodology vs Code Alignment
==========================================

Scope
-----
This note records alignment between the draft methodology and the current code
implementation (as of the latest changes).

Current Alignment Status
------------------------
1) Explicit potential difference
   - Code: forward policy logit = topological bias (-log indegree) + alpha * (logF(v) - logF(u)).
   - Method: docs/method.typ updated to the same potential-difference form.

2) Detailed balance training
   - Code: DB loss trains Z to be consistent with the learned policy; stop
     transitions use the reward residual head.
   - Method: matches; DB does not force residual = logZ difference.

Optional Runtime Constraints (not required by the method)
---------------------------------------------------------
- avoid_revisit is supported as an optional stability knob.
  Disable it for strict DB consistency with the idealized formulation.
