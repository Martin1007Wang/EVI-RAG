# Weaver Rollout Diagnostics

- checkpoint: `outputs/checkpoints/epoch=5-step=180.ckpt`
- validation samples: 8

## Dataset Reachability
| metric | value |
|---|---:|
| `oracle/has_anchor` | 1.0000 |
| `oracle/num_targets_mean` | 5.1250 |
| `oracle/reachable_target_rate` | 1.0000 |
| `oracle/target_at_depth_0_rate` | 0.0000 |
| `oracle/target_at_depth_1_rate` | 0.9512 |
| `oracle/target_at_depth_2_rate` | 1.0000 |
| `oracle/target_at_depth_3_rate` | 1.0000 |
| `oracle/no_reachable_target_rate` | 0.0000 |
| `oracle/undirected_target_at_depth_1_rate` | 0.9512 |
| `oracle/undirected_target_at_depth_2_rate` | 1.0000 |

## Prior Ranking
| metric | value |
|---|---:|
| `prior/root_valid_edge_exists_rate` | 1.0000 |
| `prior/root_candidate_count_mean` | 388.8750 |
| `prior/root_best_valid_rank_mean` | 14.7500 |
| `prior/root_best_valid_rank_median` | 1.0000 |
| `prior/root_best_valid_prob_mean` | 0.0078 |
| `prior/root_valid_edge_prob_mass_mean` | 0.0573 |
| `prior/root_valid_edge_8sample_hit_rate` | 0.2424 |
| `prior/root_valid_edge_top1_rate` | 0.6250 |
| `prior/root_valid_edge_top3_rate` | 0.6250 |
| `prior/root_valid_edge_top5_rate` | 0.6250 |
| `prior/root_valid_edge_top10_rate` | 0.6250 |
| `prior/root_valid_edge_mrr` | 0.6424 |
| `prior/depth1_valid_edge_exists_rate` | 0.6250 |
| `prior/depth1_candidate_count_mean` | 318.0000 |
| `prior/depth1_best_valid_rank_mean` | 10.2000 |
| `prior/depth1_best_valid_prob_mean` | 0.0082 |
| `prior/depth1_valid_edge_prob_mass_mean` | 0.0787 |
| `prior/depth1_valid_edge_8sample_hit_rate` | 0.3083 |
| `prior/depth1_valid_edge_top10_rate` | 0.6000 |

## Root One-Step Reward Oracle
| metric | value |
|---|---:|
| `oracle/root_stop_log_reward` | -9.2103 |
| `oracle/root_best_child_log_reward` | -3.2113 |
| `oracle/root_best_child_minus_stop_log_reward` | 5.9990 |
| `oracle/root_best_child_support` | 0.3900 |
| `oracle/root_answer_edge_rank_by_policy` | 2.2500 |
| `oracle/root_best_child_edge_prob` | 0.0448 |
| `oracle/root_best_child_8sample_hit_rate` | 0.3038 |
| `oracle/root_policy_top1_child_support` | 0.3484 |
| `oracle/root_policy_top5_child_support` | 0.3484 |
| `oracle/root_candidate_count` | 32.0000 |

## Sampling Sanity
| metric | value |
|---|---:|
| `train/policy/target_stop_prob_mean` | 0.0220 |
| `train/policy/target_continue_prob_mean` | 0.9780 |
| `train/rollout/continue_depth_0_rate` | 1.0000 |
| `train/rollout/continue_rate` | 0.7450 |
| `train/rollout/budget_exhausted_ratio` | 0.9375 |
| `train/rollout/stop_depth_hist_0` | 0.0000 |
| `train/rollout/stop_depth_hist_1` | 0.0156 |
| `train/rollout/stop_depth_hist_2` | 0.0469 |
| `train/rollout/stop_depth_hist_3` | 0.9375 |

## Stop Improvement Oracle
| metric | value |
|---|---:|
| `stop_oracle/stop_now_better_ratio` | 0.2083 |
| `stop_oracle/mean_best_continue_minus_stop_log_reward` | 1.6412 |
| `stop_oracle/policy_stop_prob_when_stop_better` | 0.3084 |
| `stop_oracle/policy_continue_prob_when_continue_better` | 0.8587 |
| `stop_oracle/stop_now_better_ratio_depth_0` | 0.2500 |
| `stop_oracle/stop_now_better_ratio_depth_1` | 0.3333 |
| `stop_oracle/stop_now_better_ratio_depth_2` | 0.2500 |
| `stop_oracle/stop_now_better_ratio_depth_3` | 0.0000 |

## Oracle Path Probability
| metric | value |
|---|---:|
| `oracle_path/nonzero_f1_rate` | 0.7500 |
| `oracle_path/answer_f1_mean` | 0.6161 |
| `oracle_path/exact_path_prob_mean` | 0.0063 |
| `oracle_path/exact_path_prob_when_nonzero_mean` | 0.0084 |
| `oracle_path/expected_hit_rate_8_mean` | 0.0466 |
| `oracle_path/expected_hit_rate_8_when_nonzero_mean` | 0.0619 |
| `oracle_path/selected_edge_count_mean` | 1.3750 |
| `oracle_path/chosen_edge_rank_mean` | 7.1818 |
| `oracle_path/chosen_edge_prob_mean` | 0.0443 |
| `oracle_path/continue_prob_mean` | 0.8753 |
| `oracle_path/terminal_stop_prob_mean` | 0.4291 |
| `oracle_path/chosen_edge_prob_depth_0` | 0.0470 |
| `oracle_path/chosen_edge_prob_depth_1` | 0.0434 |
| `oracle_path/chosen_edge_prob_depth_2` | 0.0376 |
| `oracle_path/continue_prob_depth_0` | 0.9999 |
| `oracle_path/continue_prob_depth_1` | 0.7574 |
| `oracle_path/continue_prob_depth_2` | 0.6782 |

## Rollout Comparison
| metric | value |
|---|---:|
| `policy/nonzero_f1_rate` | 0.2656 |
| `policy/answer_f1_mean` | 0.0866 |
| `policy/utility_mean` | 0.0791 |
| `policy/log_reward_mean` | -7.4732 |
| `policy/expanded_edge_count_mean` | 2.9219 |
| `policy/minimality_gap_mean` | 0.0000 |
| `policy/budget_exhausted_stop_rate` | 0.9375 |
| `policy/model_stop_rate` | 1.0000 |
| `policy/first_hit_depth_mean` | 1.1765 |
| `policy/hit_at_depth_1_rate` | 0.8824 |
| `policy/hit_at_depth_2_rate` | 0.9412 |
| `policy/continue_after_first_hit_rate` | 0.8824 |
| `policy/extra_edges_after_first_hit_mean` | 1.5294 |
| `prior_greedy_nonzero_f1_rate` | 0.6250 |
| `prior_beam4_nonzero_f1_rate` | 0.6250 |
| `prior_beam8_nonzero_f1_rate` | 0.6250 |
| `oracle/nonzero_f1_rate` | 0.8750 |
| `oracle/answer_f1_mean` | 0.7571 |
| `oracle/expanded_edge_count_mean` | 2.1250 |
| `oracle/minimality_gap_mean` | 0.0000 |

## Reward Sanity
| metric | value |
|---|---:|
| `reward/first_hit_log_reward_mean` | -2.1972 |
| `reward/final_log_reward_mean` | -1.8409 |
| `reward/final_minus_first_hit_log_reward_mean` | 0.3563 |
| `reward/final_better_than_first_hit_rate` | 0.5294 |
| `reward/minimality_gap_at_final_mean` | 0.0000 |
| `reward/minimality_penalty_at_final_mean` | 0.2922 |
| `reward/expanded_edge_count_at_final_mean` | 2.9219 |
| `reward/minimal_edge_count_at_final_mean` | 0.0000 |

## StopTB Ablation
This pass evaluates the same rollout traces under `stop_tb_coef=0.0` and `0.05`; it does not run a separate 200-500 step training ablation.
| metric | value |
|---|---:|
| `stop_tb_coef_0/loss_total` | 29.4870 |
| `stop_tb_coef_0/loss_subtb` | 29.4870 |
| `stop_tb_coef_0/loss_stop_tb` | 0.0090 |
| `stop_tb_coef_0_05/loss_total` | 29.4874 |
| `stop_tb_coef_0_05/loss_subtb` | 29.4870 |
| `stop_tb_coef_0_05/loss_stop_tb` | 0.0090 |
| `stop_tb/residual_abs_mean` | 0.0258 |
| `stop_tb/residual_after_hit_abs_mean` | 0.1639 |
| `stop_tb/valid_count_mean` | 2.9844 |
| `policy/stop_prob_after_hit` | 0.1394 |
| `policy/stop_prob_before_hit` | 9.169e-05 |

## Edge Prior/Residual
| metric | value |
|---|---:|
| `edge/prior_abs_mean` | 3.1547 |
| `edge/residual_abs_mean` | 0.0000 |
| `edge/residual_std` | 0.0000 |
| `edge/semantic_logit_std` | 0.3888 |
| `edge/residual_scaled_abs_mean` | 0.0000 |
| `edge/residual_to_prior_ratio` | 0.0000 |
| `edge/residual_to_prior_std_ratio` | 0.0000 |
| `edge/logit_scale` | 5.0000 |
| `edge/entity_weight` | 0.1000 |
| `edge/residual_scale` | 0.0000 |
| `edge/valid_edge_prior_rank_mean` | 5.5556 |
| `edge/valid_edge_final_rank_mean` | 5.5556 |
| `edge/valid_edge_rank_delta_mean` | 0.0000 |
| `edge/final_worse_than_prior_rate` | 0.0000 |
| `gate/frontier_logmeanexp_depth_0` | -3.4657 |
| `gate/frontier_logmeanexp_depth_1` | -3.4657 |
| `gate/option_gap_depth_0` | 9.2103 |
| `gate/option_gap_depth_1` | 4.2254 |
| `gate/option_gap_depth_2` | 4.2254 |

## Main Diagnosis
* Validation answers one-hop reachable: yes.
* Semantic prior ranks valid progress edges in top-k: yes.
* Current rollout hits shallow then continues: yes.
* Minimality reward prefers first-hit over final: no.

## Recommended Next Change
Fix reward minimality so first-hit sufficient states beat budget-full states.
