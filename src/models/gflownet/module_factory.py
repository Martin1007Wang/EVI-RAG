from __future__ import annotations

from src.models.components import EmbeddingBackbone, NodeFlowHead, TransitionPolicyHead
from src.models.configs import ActionPriorConfig, GFlowNetTrainingConfig, PolicyConfig

from .heuristics import SearchActionPrior
from .policy import BaseSearchPolicy, GFlowNetPolicy


class GFlowNetPolicyFactory:
    @staticmethod
    def build_base_policy(
        *,
        policy_cfg: PolicyConfig,
        training_cfg: GFlowNetTrainingConfig,
        max_steps: int,
    ) -> BaseSearchPolicy:
        graph_hidden_dim = int(policy_cfg.backbone.hidden_dim)
        backbone = EmbeddingBackbone(policy_cfg.backbone)
        state_score_head = NodeFlowHead(
            node_dim=graph_hidden_dim,
            question_dim=graph_hidden_dim,
            hidden_dim=int(policy_cfg.state_score_head.hidden_dim),
            num_layers=int(policy_cfg.state_score_head.num_layers),
            dropout=float(policy_cfg.state_score_head.dropout),
            conditioning=str(policy_cfg.state_score_head.conditioning),
        )
        transition_policy_head = None
        if bool(policy_cfg.transition_head.enabled):
            transition_policy_head = TransitionPolicyHead(
                state_dim=graph_hidden_dim,
                relation_dim=graph_hidden_dim,
                hidden_dim=int(policy_cfg.transition_head.hidden_dim),
                num_layers=int(policy_cfg.transition_head.num_layers),
                dropout=float(policy_cfg.transition_head.dropout),
                detach_input_features=bool(
                    policy_cfg.transition_head.detach_input_features
                ),
            )
        return BaseSearchPolicy(
            config=policy_cfg,
            max_steps=max_steps,
            backbone=backbone,
            state_score_head=state_score_head,
            transition_policy_head=transition_policy_head,
            step_log_penalty=float(training_cfg.step_log_penalty),
            non_gold_terminal_log_reward=float(
                training_cfg.terminal_failure_log_reward
            ),
            answer_stop_log_reward_bonus=float(
                training_cfg.answer_stop_log_reward_bonus
            ),
            answer_quotient_allocate_stop_mass=bool(
                training_cfg.answer_quotient.stop_allocation_active
            ),
            answer_quotient_gold_reward_mode=str(
                training_cfg.answer_quotient.gold_reward_mode
            ),
            potential_reward_cfg=training_cfg.potential_reward,
        )

    @staticmethod
    def build_action_prior(
        *,
        action_prior_cfg: ActionPriorConfig,
    ) -> SearchActionPrior:
        return SearchActionPrior(config=action_prior_cfg)

    @staticmethod
    def build_policy(
        *,
        policy_cfg: PolicyConfig,
        training_cfg: GFlowNetTrainingConfig,
        action_prior_cfg: ActionPriorConfig,
        max_steps: int,
    ) -> GFlowNetPolicy:
        base_policy = GFlowNetPolicyFactory.build_base_policy(
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            max_steps=max_steps,
        )
        search_action_prior = GFlowNetPolicyFactory.build_action_prior(
            action_prior_cfg=action_prior_cfg,
        )
        return GFlowNetPolicy(
            base_policy=base_policy,
            action_prior_cfg=action_prior_cfg,
            search_action_prior=search_action_prior,
        )


__all__ = ["GFlowNetPolicyFactory"]
