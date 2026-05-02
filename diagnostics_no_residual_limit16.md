# Weaver Rollout Diagnostics

- checkpoint: `outputs/checkpoints/epoch=5-step=180.ckpt`
- validation samples: 16

## Dataset Reachability
| metric | value |
|---|---:|
| `oracle/has_anchor` | 1.0000 |
| `oracle/num_targets_mean` | 3.7500 |
| `oracle/reachable_target_rate` | 1.0000 |
| `oracle/target_at_depth_0_rate` | 0.0000 |
| `oracle/target_at_depth_1_rate` | 0.7914 |
| `oracle/target_at_depth_2_rate` | 1.0000 |
| `oracle/target_at_depth_3_rate` | 1.0000 |
| `oracle/no_reachable_target_rate` | 0.0000 |
| `oracle/undirected_target_at_depth_1_rate` | 0.7914 |
| `oracle/undirected_target_at_depth_2_rate` | 1.0000 |

## Prior Ranking
| metric | value |
|---|---:|
| `prior/root_valid_edge_exists_rate` | 1.0000 |
| `prior/root_candidate_count_mean` | 664.1875 |
| `prior/root_best_valid_rank_mean` | 11.9375 |
| `prior/root_best_valid_rank_median` | 1.0000 |
| `prior/root_best_valid_prob_mean` | 0.0165 |
| `prior/root_valid_edge_prob_mass_mean` | 0.0733 |
| `prior/root_valid_edge_8sample_hit_rate` | 0.3126 |
| `prior/root_valid_edge_top1_rate` | 0.6875 |
| `prior/root_valid_edge_top3_rate` | 0.6875 |
| `prior/root_valid_edge_top5_rate` | 0.7500 |
| `prior/root_valid_edge_top10_rate` | 0.7500 |
| `prior/root_valid_edge_mrr` | 0.7128 |
| `prior/depth1_valid_edge_exists_rate` | 0.7500 |
| `prior/depth1_candidate_count_mean` | 652.5000 |
| `prior/depth1_best_valid_rank_mean` | 10.8857 |
| `prior/depth1_best_valid_prob_mean` | 0.0179 |
| `prior/depth1_valid_edge_prob_mass_mean` | 0.0844 |
| `prior/depth1_valid_edge_8sample_hit_rate` | 0.3398 |
| `prior/depth1_valid_edge_top10_rate` | 0.7286 |

## Root One-Step Reward Oracle
| metric | value |
|---|---:|
| `oracle/root_stop_log_reward` | -9.2103 |
| `oracle/root_best_child_log_reward` | -3.5908 |
| `oracle/root_best_child_minus_stop_log_reward` | 5.6195 |
| `oracle/root_best_child_support` | 0.3915 |
| `oracle/root_answer_edge_rank_by_policy` | 191.1250 |
| `oracle/root_best_child_edge_prob` | 0.0155 |
| `oracle/root_best_child_8sample_hit_rate` | 0.0993 |
| `oracle/root_policy_top1_child_support` | 0.3081 |
| `oracle/root_policy_top5_child_support` | 0.3081 |
| `oracle/root_candidate_count` | 664.1875 |

## Sampling Sanity
| metric | value |
|---|---:|
| `train/policy/target_stop_prob_mean` | 0.0007 |
| `train/policy/target_continue_prob_mean` | 0.9993 |
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
| `stop_oracle/stop_now_better_ratio` | 0.2775 |
| `stop_oracle/mean_best_continue_minus_stop_log_reward` | 1.5242 |
| `stop_oracle/policy_stop_prob_when_stop_better` | 0.0007 |
| `stop_oracle/policy_continue_prob_when_continue_better` | 0.9998 |
| `stop_oracle/stop_now_better_ratio_depth_0` | 0.3125 |
| `stop_oracle/stop_now_better_ratio_depth_1` | 0.3393 |
| `stop_oracle/stop_now_better_ratio_depth_2` | 0.4583 |
| `stop_oracle/stop_now_better_ratio_depth_3` | 0.0000 |

## Oracle Path Probability
| metric | value |
|---|---:|
| `oracle_path/nonzero_f1_rate` | 0.7500 |
| `oracle_path/answer_f1_mean` | 0.6348 |
| `oracle_path/exact_path_prob_mean` | 0.0000 |
| `oracle_path/exact_path_prob_when_nonzero_mean` | 0.0000 |
| `oracle_path/expected_hit_rate_8_mean` | 0.0001 |
| `oracle_path/expected_hit_rate_8_when_nonzero_mean` | 0.0001 |
| `oracle_path/selected_edge_count_mean` | 2.2500 |
| `oracle_path/chosen_edge_rank_mean` | 242.1904 |
| `oracle_path/chosen_edge_prob_mean` | 0.0138 |
| `oracle_path/continue_prob_mean` | 0.9994 |
| `oracle_path/terminal_stop_prob_mean` | 0.5005 |
| `oracle_path/chosen_edge_prob_depth_0` | 0.0155 |
| `oracle_path/chosen_edge_prob_depth_1` | 0.0179 |
| `oracle_path/chosen_edge_prob_depth_2` | 0.0032 |
| `oracle_path/continue_prob_depth_0` | 0.9992 |
| `oracle_path/continue_prob_depth_1` | 0.9994 |
| `oracle_path/continue_prob_depth_2` | 0.9998 |

## Rollout Comparison
| metric | value |
|---|---:|
| `policy/nonzero_f1_rate` | 0.1562 |
| `policy/answer_f1_mean` | 0.0423 |
| `policy/utility_mean` | 0.0534 |
| `policy/log_reward_mean` | -8.3304 |
| `policy/expanded_edge_count_mean` | 3.0000 |
| `policy/minimality_gap_mean` | 0.0000 |
| `policy/budget_exhausted_stop_rate` | 1.0000 |
| `policy/model_stop_rate` | 1.0000 |
| `policy/first_hit_depth_mean` | 1.7500 |
| `policy/hit_at_depth_1_rate` | 0.4500 |
| `policy/hit_at_depth_2_rate` | 0.8000 |
| `policy/continue_after_first_hit_rate` | 0.8000 |
| `policy/extra_edges_after_first_hit_mean` | 1.2500 |
| `prior_greedy_nonzero_f1_rate` | 0.6875 |
| `prior_beam4_nonzero_f1_rate` | 0.6875 |
| `prior_beam8_nonzero_f1_rate` | 0.8125 |
| `oracle/nonzero_f1_rate` | 0.7500 |
| `oracle/answer_f1_mean` | 0.6348 |
| `oracle/expanded_edge_count_mean` | 2.2500 |
| `oracle/minimality_gap_mean` | 0.0000 |

## Reward Sanity
| metric | value |
|---|---:|
| `reward/first_hit_log_reward_mean` | -1.8686 |
| `reward/final_log_reward_mean` | -1.9590 |
| `reward/final_minus_first_hit_log_reward_mean` | -0.0904 |
| `reward/final_better_than_first_hit_rate` | 0.0500 |
| `reward/minimality_gap_at_final_mean` | 0.0000 |
| `reward/minimality_penalty_at_final_mean` | 0.3000 |
| `reward/expanded_edge_count_at_final_mean` | 3.0000 |
| `reward/minimal_edge_count_at_final_mean` | 0.0000 |

## StopTB Ablation
This pass evaluates the same rollout traces under `stop_tb_coef=0.0` and `0.05`; it does not run a separate 200-500 step training ablation.
| metric | value |
|---|---:|
| `stop_tb_coef_0/loss_total` | 82.9807 |
| `stop_tb_coef_0/loss_subtb` | 82.9807 |
| `stop_tb_coef_0/loss_stop_tb` | 4.9939 |
| `stop_tb_coef_0_05/loss_total` | 83.2304 |
| `stop_tb_coef_0_05/loss_subtb` | 82.9807 |
| `stop_tb_coef_0_05/loss_stop_tb` | 4.9939 |
| `stop_tb/residual_abs_mean` | 1.4665 |
| `stop_tb/residual_after_hit_abs_mean` | 6.2706 |
| `stop_tb/valid_count_mean` | 3.0000 |
| `policy/stop_prob_after_hit` | 0.0008 |
| `policy/stop_prob_before_hit` | 0.0006 |

## Edge Prior/Residual
| metric | value |
|---|---:|
| `edge/prior_abs_mean` | 2.4482 |
| `edge/residual_abs_mean` | 0.0000 |
| `edge/residual_std` | 0.0000 |
| `edge/semantic_logit_std` | 0.3702 |
| `edge/residual_scaled_abs_mean` | 0.0000 |
| `edge/residual_to_prior_ratio` | 0.0000 |
| `edge/residual_to_prior_std_ratio` | 0.0000 |
| `edge/logit_scale` | 5.0000 |
| `edge/entity_weight` | 0.1000 |
| `edge/residual_scale` | 0.0000 |
| `edge/valid_edge_prior_rank_mean` | 15.6855 |
| `edge/valid_edge_final_rank_mean` | 15.6855 |
| `edge/valid_edge_rank_delta_mean` | 0.0000 |
| `edge/final_worse_than_prior_rate` | 0.0000 |
| `gate/frontier_logmeanexp_depth_0` | 2.5237 |
| `gate/frontier_logmeanexp_depth_1` | 2.5531 |
| `gate/option_gap_depth_0` | 8.2752 |
| `gate/option_gap_depth_1` | 8.4158 |
| `gate/option_gap_depth_2` | 8.4158 |

## Main Diagnosis
* Validation answers one-hop reachable: yes.
* Semantic prior ranks valid progress edges in top-k: yes.
* Current rollout hits shallow then continues: yes.
* Minimality reward prefers first-hit over final: yes.

## Recommended Next Change
Focus the next code change on Stop training: increase/verify stopTB and simplify or detach frontier summary in the option gate.
