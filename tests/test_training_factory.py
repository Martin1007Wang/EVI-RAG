from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.training.factory import build_model
from src.weaver.policy import ForwardPolicy


def test_build_model_instantiates_forward_policy_with_shared_interaction() -> None:
    cfg = OmegaConf.create(
        {
            "model": OmegaConf.load("configs/model/weaver.yaml"),
        }
    )
    resources = SimpleNamespace(
        entity_text_semantic_table=torch.zeros(3, cfg.model.hidden_dim),
        text_row_by_entity_id=torch.zeros(5, dtype=torch.long),
        relation_semantic_table=torch.zeros(4, cfg.model.hidden_dim),
    )

    module = build_model(cfg, resources)

    assert isinstance(module.policy, ForwardPolicy)
    assert module.policy.stop_head.interaction is module.policy.edge_head.interaction


def test_build_model_accepts_simplified_replay_source_config() -> None:
    cfg = OmegaConf.create(
        {
            "model": OmegaConf.load("configs/model/weaver.yaml"),
        }
    )
    resources = SimpleNamespace(
        entity_text_semantic_table=torch.zeros(3, cfg.model.hidden_dim),
        text_row_by_entity_id=torch.zeros(5, dtype=torch.long),
        relation_semantic_table=torch.zeros(4, cfg.model.hidden_dim),
    )

    module = build_model(cfg, resources)

    assert module.runner.replay_source is not None
    assert not hasattr(module.runner.replay_source, "max_depth")
    assert not hasattr(module.runner.replay_source, "mode")
    assert not hasattr(module.runner.replay_source, "max_states_per_graph")
