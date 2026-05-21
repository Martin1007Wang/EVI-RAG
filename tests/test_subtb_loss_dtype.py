from __future__ import annotations

import torch

from src.weaver.objectives import SubTBEventBatch, SubTBInput, SubTBLoss, subtrajectory_terms
from src.weaver.policy import PolicyOutput


def logit(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values / (1.0 - values))


def test_policy_output_gather_log_prob_uses_float32_for_bf16_outputs() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.log(torch.tensor([1.0, 3.0], dtype=torch.float32)).to(torch.bfloat16),
        edge_logit=torch.log(torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)).to(torch.bfloat16),
        state_log_flow=torch.zeros(2, dtype=torch.bfloat16),
        edge_row_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        num_rows=2,
        num_edges=3,
    )

    log_prob = policy_out.gather_log_prob(
        row_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_ids=torch.tensor([1, -1], dtype=torch.long),
    )

    assert log_prob.dtype == torch.float32
    assert torch.isfinite(log_prob).all()


def test_policy_output_tempered_log_prob_matches_sampling_distribution() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.zeros(1, dtype=torch.float32),
        edge_logit=torch.zeros(1, dtype=torch.float32),
        state_log_flow=torch.zeros(1, dtype=torch.float32),
        edge_row_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([7], dtype=torch.long),
        num_rows=1,
        num_edges=8,
    )

    policy_log_prob = policy_out.gather_log_prob(
        row_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([7], dtype=torch.long),
        temperature=1.0,
    )
    behavior_log_prob = policy_out.gather_log_prob(
        row_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([7], dtype=torch.long),
        temperature=0.5,
    )

    assert torch.allclose(policy_log_prob, torch.log(torch.tensor([0.5])))
    assert torch.allclose(behavior_log_prob, torch.log(torch.tensor([0.5])))


def test_log_flow_promotes_bf16_policy_outputs_to_float32() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.tensor([-0.5], dtype=torch.bfloat16),
        edge_logit=torch.tensor([-0.5], dtype=torch.bfloat16),
        state_log_flow=torch.tensor([0.0], dtype=torch.bfloat16),
        edge_row_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        num_rows=1,
        num_edges=1,
    )

    assert policy_out.log_flow().dtype == torch.float32


def test_hierarchical_action_probabilities_sum_to_one_per_row() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.log(torch.tensor([1.0, 2.0], dtype=torch.float32)),
        edge_logit=torch.zeros(2, dtype=torch.float32),
        state_log_flow=torch.zeros(2, dtype=torch.float32),
        edge_row_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_ids=torch.tensor([0, 1], dtype=torch.long),
        num_rows=2,
        num_edges=2,
    )

    assert torch.allclose(
        policy_out.stop_prob() + policy_out.edge_prob_mass(),
        torch.ones(2, dtype=torch.float32),
    )


def test_edge_prob_mass_is_zero_when_no_edge_actions_present() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.zeros(2, dtype=torch.float32),
        edge_logit=torch.empty(0, dtype=torch.float32),
        state_log_flow=torch.zeros(2, dtype=torch.float32),
        edge_row_ids=torch.empty(0, dtype=torch.long),
        edge_ids=torch.empty(0, dtype=torch.long),
        num_rows=2,
        num_edges=0,
    )

    assert torch.allclose(policy_out.edge_prob_mass(), torch.zeros(2, dtype=torch.float32))


def test_policy_output_exposes_branch_mass_diagnostics() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.log(torch.tensor([1.0, 2.0], dtype=torch.float32)),
        edge_logit=torch.log(torch.tensor([1.0, 3.0], dtype=torch.float32)),
        state_log_flow=torch.zeros(2, dtype=torch.float32),
        edge_row_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor([0, 1], dtype=torch.long),
        num_rows=2,
        num_edges=2,
    )

    assert torch.allclose(policy_out.continue_prob(), torch.tensor([0.8, 0.0]))
    assert torch.allclose(policy_out.frontier_size(), torch.tensor([2.0, 0.0]))
    assert torch.allclose(
        policy_out.gather_edge_cond_log_prob(
            row_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([1], dtype=torch.long),
        ),
        torch.log(torch.tensor([0.75])),
    )
    assert policy_out.edge_cond_entropy()[0].gt(0)
    assert policy_out.edge_cond_entropy()[1].eq(0)


def test_subtrajectory_terms_cover_nonterminal_and_terminal_segments() -> None:
    terms = subtrajectory_terms(
        SubTBInput(
            events=SubTBEventBatch(
                trajectory_ids=torch.tensor([0, 0], dtype=torch.long),
                step_ids=torch.tensor([0, 1], dtype=torch.long),
                source_ids=torch.tensor([0, 0], dtype=torch.long),
                parent_log_flow=torch.tensor([10.0, 30.0], dtype=torch.float32),
                child_log_flow=torch.tensor([11.0, 0.0], dtype=torch.float32),
                action_log_prob=torch.tensor([1.0, 3.0], dtype=torch.float32),
                continue_log_prob=torch.tensor([0.25, 0.0], dtype=torch.float32),
                edge_cond_log_prob=torch.tensor([0.75, 0.0], dtype=torch.float32),
                stop_log_prob=torch.tensor([0.0, 3.0], dtype=torch.float32),
                backward_log_prob=torch.tensor([-0.5, 0.0], dtype=torch.float32),
                terminal_log_reward=torch.tensor([0.0, 40.0], dtype=torch.float32),
                terminal=torch.tensor([False, True], dtype=torch.bool),
            ),
        ),
        subtb_lambda=0.9,
        max_len=None,
    )

    assert torch.equal(terms.source_ids, torch.tensor([0, 0, 0], dtype=torch.long))
    assert torch.equal(terms.terminal, torch.tensor([False, True, True], dtype=torch.bool))
    assert torch.equal(terms.length, torch.tensor([1, 2, 1], dtype=torch.long))
    assert torch.allclose(
        terms.weight,
        torch.tensor([1.0, 0.9, 1.0], dtype=torch.float32),
    )
    assert torch.allclose(
        terms.residual,
        torch.tensor([0.5, -25.5, -7.0], dtype=torch.float32),
    )


def test_subtb_loss_reports_branch_and_source_diagnostics() -> None:
    loss = SubTBLoss(subtb_lambda=0.9, residual_loss="mse")
    out = loss(
        SubTBInput(
            events=SubTBEventBatch(
                trajectory_ids=torch.tensor([0, 1], dtype=torch.long),
                step_ids=torch.tensor([0, 0], dtype=torch.long),
                source_ids=torch.tensor([0, 1], dtype=torch.long),
                parent_log_flow=torch.tensor([10.0, 5.0], dtype=torch.float32),
                child_log_flow=torch.tensor([11.0, 0.0], dtype=torch.float32),
                action_log_prob=torch.tensor([1.0, -2.0], dtype=torch.float32),
                continue_log_prob=torch.tensor([0.25, 0.0], dtype=torch.float32),
                edge_cond_log_prob=torch.tensor([0.75, 0.0], dtype=torch.float32),
                stop_log_prob=torch.tensor([0.0, -2.0], dtype=torch.float32),
                backward_log_prob=torch.tensor([-0.5, 0.0], dtype=torch.float32),
                terminal_log_reward=torch.tensor([0.0, 4.0], dtype=torch.float32),
                terminal=torch.tensor([False, True], dtype=torch.bool),
            ),
        )
    )

    assert torch.allclose(out.metrics["subtb/exp/residual_mean"], torch.tensor(0.5))
    assert torch.allclose(out.metrics["subtb/stop/residual_mean"], torch.tensor(-1.0))
    assert torch.allclose(out.metrics["subtb/policy/residual_abs_mean"], torch.tensor(0.5))
    assert torch.allclose(out.metrics["subtb/replay/residual_abs_mean"], torch.tensor(1.0))
    assert torch.allclose(out.metrics["subtb/policy/terminal_fraction"], torch.tensor(0.0))
    assert torch.allclose(out.metrics["subtb/replay/terminal_fraction"], torch.tensor(1.0))
