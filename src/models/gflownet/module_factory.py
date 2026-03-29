from __future__ import annotations

from src.models.configs import ActionPriorConfig, GFlowNetTrainingConfig, PolicyConfig
from src.models.configs.policy import SUBGRAPH_STATE_MODE

from .subgraph import SubgraphPolicy


class GFlowNetPolicyFactory:
    @staticmethod
    def build_policy(
        *,
        policy_cfg: PolicyConfig,
        training_cfg: GFlowNetTrainingConfig,
        action_prior_cfg: ActionPriorConfig,
        max_steps: int,
    ) -> SubgraphPolicy:
        del action_prior_cfg
        if str(policy_cfg.state_mode) != SUBGRAPH_STATE_MODE:
            raise ValueError(
                "GFlowNetPolicyFactory supports only policy.state_mode='subgraph'."
            )
        return SubgraphPolicy(
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            max_steps=max_steps,
        )


__all__ = ["GFlowNetPolicyFactory"]
