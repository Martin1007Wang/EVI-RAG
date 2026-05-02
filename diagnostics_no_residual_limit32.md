# Weaver Rollout Diagnostics

- checkpoint: `outputs/checkpoints/epoch=5-step=180.ckpt`
- validation samples: 32

## Dataset Reachability
| metric | value |
|---|---:|
| `oracle/has_anchor` | 1.0000 |
| `oracle/num_targets_mean` | 4.8438 |
| `oracle/reachable_target_rate` | 1.0000 |
| `oracle/target_at_depth_0_rate` | 0.0000 |
| `oracle/target_at_depth_1_rate` | 0.7691 |
| `oracle/target_at_depth_2_rate` | 1.0000 |
| `oracle/target_at_depth_3_rate` | 1.0000 |
| `oracle/no_reachable_target_rate` | 0.0000 |
| `oracle/undirected_target_at_depth_1_rate` | 0.7691 |
| `oracle/undirected_target_at_depth_2_rate` | 1.0000 |

## Prior Ranking
| metric | value |
|---|---:|
| `prior/root_valid_edge_exists_rate` | 1.0000 |
| `prior/root_candidate_count_mean` | 598.4688 |
| `prior/root_best_valid_rank_mean` | 10.6562 |
| `prior/root_best_valid_rank_median` | 1.2500 |
| `prior/root_best_valid_prob_mean` | 0.0132 |
| `prior/root_valid_edge_prob_mass_mean` | 0.0851 |
| `prior/root_valid_edge_8sample_hit_rate` | 0.3248 |
| `prior/root_valid_edge_top1_rate` | 0.5938 |
| `prior/root_valid_edge_top3_rate` | 0.6875 |
| `prior/root_valid_edge_top5_rate` | 0.7500 |
| `prior/root_valid_edge_top10_rate` | 0.7812 |
| `prior/root_valid_edge_mrr` | 0.6708 |
| `prior/depth1_valid_edge_exists_rate` | 0.7500 |
| `prior/depth1_candidate_count_mean` | 580.4688 |
| `prior/depth1_best_valid_rank_mean` | 15.1304 |
| `prior/depth1_best_valid_prob_mean` | 0.0124 |
| `prior/depth1_valid_edge_prob_mass_mean` | 0.0857 |
| `prior/depth1_valid_edge_8sample_hit_rate` | 0.3267 |
| `prior/depth1_valid_edge_top10_rate` | 0.6768 |

## Root One-Step Reward Oracle
| metric | value |
|---|---:|
| `oracle/root_stop_log_reward` | -9.2103 |
| `oracle/root_best_child_log_reward` | -3.4110 |
| `oracle/root_best_child_minus_stop_log_reward` | 5.7993 |
| `oracle/root_best_child_support` | 0.3856 |
| `oracle/root_answer_edge_rank_by_policy` | 141.5625 |
| `oracle/root_best_child_edge_prob` | 0.0122 |
| `oracle/root_best_child_8sample_hit_rate` | 0.0817 |
| `oracle/root_policy_top1_child_support` | 0.3149 |
| `oracle/root_policy_top5_child_support` | 0.3290 |
| `oracle/root_candidate_count` | 598.4688 |

## Sampling Sanity
| metric | value |
|---|---:|
| `train/policy/target_stop_prob_mean` | 5.424e-04 |
| `train/policy/target_continue_prob_mean` | 0.9995 |
| `train/rollout/continue_depth_0_rate` | 1.0000 |
| `train/rollout/continue_rate` | 0.7500 |
| `train/rollout/budget_exhausted_ratio` | 1.0000 |
| `train/rollout/stop_depth_hist_0` | 0.0000 |
| `train/rollout/stop_depth_hist_1` | 0.0000 |
| `train/rollout/stop_depth_hist_2` | 0.0000 |
| `train/rollout/stop_depth_hist_3` | 1.0000 |

## Stop Improvement Oracle
| metric | value |
|---|---:|
| `stop_oracle/stop_now_better_ratio` | 0.2783 |
| `stop_oracle/mean_best_continue_minus_stop_log_reward` | 1.5628 |
| `stop_oracle/policy_stop_prob_when_stop_better` | 4.654e-04 |
| `stop_oracle/policy_continue_prob_when_continue_better` | 0.9997 |
| `stop_oracle/stop_now_better_ratio_depth_0` | 0.2812 |
| `stop_oracle/stop_now_better_ratio_depth_1` | 0.3780 |
| `stop_oracle/stop_now_better_ratio_depth_2` | 0.4542 |
| `stop_oracle/stop_now_better_ratio_depth_3` | 0.0000 |

## Oracle Path Probability
| metric | value |
|---|---:|
| `oracle_path/nonzero_f1_rate` | 0.7500 |
| `oracle_path/answer_f1_mean` | 0.6051 |
| `oracle_path/exact_path_prob_mean` | 4.836e-06 |
| `oracle_path/exact_path_prob_when_nonzero_mean` | 7.626e-06 |
| `oracle_path/expected_hit_rate_8_mean` | 3.865e-05 |
| `oracle_path/expected_hit_rate_8_when_nonzero_mean` | 6.094e-05 |
| `oracle_path/selected_edge_count_mean` | 2.2188 |
| `oracle_path/chosen_edge_rank_mean` | 175.4160 |
| `oracle_path/chosen_edge_prob_mean` | 0.0101 |
| `oracle_path/continue_prob_mean` | 0.9995 |
| `oracle_path/terminal_stop_prob_mean` | 0.5003 |
| `oracle_path/chosen_edge_prob_depth_0` | 0.0122 |
| `oracle_path/chosen_edge_prob_depth_1` | 0.0115 |
| `oracle_path/chosen_edge_prob_depth_2` | 0.0032 |
| `oracle_path/continue_prob_depth_0` | 0.9993 |
| `oracle_path/continue_prob_depth_1` | 0.9995 |
| `oracle_path/continue_prob_depth_2` | 0.9998 |

## Rollout Comparison
| metric | value |
|---|---:|
| `policy/nonzero_f1_rate` | 0.1445 |
| `policy/answer_f1_mean` | 0.0340 |
| `policy/utility_mean` | 0.0400 |
| `policy/log_reward_mean` | -8.4567 |
| `policy/expanded_edge_count_mean` | 3.0000 |
| `policy/minimality_gap_mean` | 0.0000 |
| `policy/budget_exhausted_stop_rate` | 1.0000 |
| `policy/model_stop_rate` | 1.0000 |
| `policy/first_hit_depth_mean` | 1.7297 |
| `policy/hit_at_depth_1_rate` | 0.4324 |
| `policy/hit_at_depth_2_rate` | 0.8378 |
| `policy/continue_after_first_hit_rate` | 0.8378 |
| `policy/extra_edges_after_first_hit_mean` | 1.2703 |
| `prior_greedy_nonzero_f1_rate` | 0.6875 |
| `prior_beam4_nonzero_f1_rate` | 0.7500 |
| `prior_beam8_nonzero_f1_rate` | 0.8125 |
| `oracle/nonzero_f1_rate` | 0.7500 |
| `oracle/answer_f1_mean` | 0.6051 |
| `oracle/expanded_edge_count_mean` | 2.2188 |
| `oracle/minimality_gap_mean` | 0.0000 |

## Reward Sanity
| metric | value |
|---|---:|
| `reward/first_hit_log_reward_mean` | -2.1495 |
| `reward/final_log_reward_mean` | -2.2205 |
| `reward/final_minus_first_hit_log_reward_mean` | -0.0710 |
| `reward/final_better_than_first_hit_rate` | 0.0811 |
| `reward/minimality_gap_at_final_mean` | 0.0000 |
| `reward/minimality_penalty_at_final_mean` | 0.3000 |
| `reward/expanded_edge_count_at_final_mean` | 3.0000 |
| `reward/minimal_edge_count_at_final_mean` | 0.0000 |

## StopTB Ablation
This pass evaluates the same rollout traces under `stop_tb_coef=0.0` and `0.05`; it does not run a separate 200-500 step training ablation.
| metric | value |
|---|---:|
| `stop_tb_coef_0/loss_total` | 81.5981 |
| `stop_tb_coef_0/loss_subtb` | 81.5981 |
| `stop_tb_coef_0/loss_stop_tb` | 4.3901 |
| `stop_tb_coef_0_05/loss_total` | 81.8176 |
| `stop_tb_coef_0_05/loss_subtb` | 81.5981 |
| `stop_tb_coef_0_05/loss_stop_tb` | 4.3901 |
| `stop_tb/residual_abs_mean` | 1.3785 |
| `stop_tb/residual_after_hit_abs_mean` | 6.0699 |
| `stop_tb/valid_count_mean` | 3.0000 |
| `policy/stop_prob_after_hit` | 5.972e-04 |
| `policy/stop_prob_before_hit` | 5.388e-04 |

## Edge Prior/Residual
| metric | value |
|---|---:|
| `edge/prior_abs_mean` | 2.4718 |
| `edge/residual_abs_mean` | 0.0000 |
| `edge/residual_std` | 0.0000 |
| `edge/semantic_logit_std` | 0.3572 |
| `edge/residual_scaled_abs_mean` | 0.0000 |
| `edge/residual_to_prior_ratio` | 0.0000 |
| `edge/residual_to_prior_std_ratio` | 0.0000 |
| `edge/logit_scale` | 5.0000 |
| `edge/entity_weight` | 0.1000 |
| `edge/residual_scale` | 0.0000 |
| `edge/valid_edge_prior_rank_mean` | 13.0511 |
| `edge/valid_edge_final_rank_mean` | 13.0511 |
| `edge/valid_edge_rank_delta_mean` | 0.0000 |
| `edge/final_worse_than_prior_rate` | 0.0000 |
| `gate/frontier_logmeanexp_depth_0` | 2.5429 |
| `gate/frontier_logmeanexp_depth_1` | 2.5460 |
| `gate/option_gap_depth_0` | 8.3223 |
| `gate/option_gap_depth_1` | 8.5413 |
| `gate/option_gap_depth_2` | 8.5413 |

## Main Diagnosis
* Validation answers one-hop reachable: yes.
* Semantic prior ranks valid progress edges in top-k: yes.
* Current rollout hits shallow then continues: yes.
* Minimality reward prefers first-hit over final: yes.

## Recommended Next Change
Focus the next code change on Stop training: increase/verify stopTB and simplify or detach frontier summary in the option gate.
