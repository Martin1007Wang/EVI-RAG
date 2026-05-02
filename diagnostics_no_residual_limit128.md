# Weaver Rollout Diagnostics

- checkpoint: `outputs/checkpoints/epoch=5-step=180.ckpt`
- validation samples: 128

## Dataset Reachability
| metric | value |
|---|---:|
| `oracle/has_anchor` | 1.0000 |
| `oracle/num_targets_mean` | 4.7891 |
| `oracle/reachable_target_rate` | 1.0000 |
| `oracle/target_at_depth_0_rate` | 0.0014 |
| `oracle/target_at_depth_1_rate` | 0.7070 |
| `oracle/target_at_depth_2_rate` | 1.0000 |
| `oracle/target_at_depth_3_rate` | 1.0000 |
| `oracle/no_reachable_target_rate` | 0.0000 |
| `oracle/undirected_target_at_depth_1_rate` | 0.7501 |
| `oracle/undirected_target_at_depth_2_rate` | 1.0000 |

## Prior Ranking
| metric | value |
|---|---:|
| `prior/root_valid_edge_exists_rate` | 0.9922 |
| `prior/root_candidate_count_mean` | 727.4955 |
| `prior/root_best_valid_rank_mean` | 16.4788 |
| `prior/root_best_valid_rank_median` | 3.5938 |
| `prior/root_best_valid_prob_mean` | 0.0113 |
| `prior/root_valid_edge_prob_mass_mean` | 0.0588 |
| `prior/root_valid_edge_8sample_hit_rate` | 0.2636 |
| `prior/root_valid_edge_top1_rate` | 0.4877 |
| `prior/root_valid_edge_top3_rate` | 0.5982 |
| `prior/root_valid_edge_top5_rate` | 0.6607 |
| `prior/root_valid_edge_top10_rate` | 0.7310 |
| `prior/root_valid_edge_mrr` | 0.5694 |
| `prior/depth1_valid_edge_exists_rate` | 0.7031 |
| `prior/depth1_candidate_count_mean` | 704.5205 |
| `prior/depth1_best_valid_rank_mean` | 21.4612 |
| `prior/depth1_best_valid_prob_mean` | 0.0112 |
| `prior/depth1_valid_edge_prob_mass_mean` | 0.0629 |
| `prior/depth1_valid_edge_8sample_hit_rate` | 0.2897 |
| `prior/depth1_valid_edge_top10_rate` | 0.6616 |

## Root One-Step Reward Oracle
| metric | value |
|---|---:|
| `oracle/root_stop_log_reward` | -9.1384 |
| `oracle/root_best_child_log_reward` | -3.7689 |
| `oracle/root_best_child_minus_stop_log_reward` | 5.3695 |
| `oracle/root_best_child_support` | 0.3915 |
| `oracle/root_answer_edge_rank_by_policy` | 105.5000 |
| `oracle/root_best_child_edge_prob` | 0.0095 |
| `oracle/root_best_child_8sample_hit_rate` | 0.0659 |
| `oracle/root_policy_top1_child_support` | 0.2661 |
| `oracle/root_policy_top5_child_support` | 0.2950 |
| `oracle/root_candidate_count` | 724.0000 |

## Sampling Sanity
| metric | value |
|---|---:|
| `train/policy/target_stop_prob_mean` | 4.854e-04 |
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
| `stop_oracle/stop_now_better_ratio` | 0.2584 |
| `stop_oracle/mean_best_continue_minus_stop_log_reward` | 1.4635 |
| `stop_oracle/policy_stop_prob_when_stop_better` | 5.949e-04 |
| `stop_oracle/policy_continue_prob_when_continue_better` | 0.9998 |
| `stop_oracle/stop_now_better_ratio_depth_0` | 0.3359 |
| `stop_oracle/stop_now_better_ratio_depth_1` | 0.4540 |
| `stop_oracle/stop_now_better_ratio_depth_2` | 0.2438 |
| `stop_oracle/stop_now_better_ratio_depth_3` | 0.0000 |

## Oracle Path Probability
| metric | value |
|---|---:|
| `oracle_path/nonzero_f1_rate` | 0.7109 |
| `oracle_path/answer_f1_mean` | 0.5739 |
| `oracle_path/exact_path_prob_mean` | 4.699e-06 |
| `oracle_path/exact_path_prob_when_nonzero_mean` | 4.956e-06 |
| `oracle_path/expected_hit_rate_8_mean` | 3.755e-05 |
| `oracle_path/expected_hit_rate_8_when_nonzero_mean` | 3.960e-05 |
| `oracle_path/selected_edge_count_mean` | 2.2500 |
| `oracle_path/chosen_edge_rank_mean` | 135.6569 |
| `oracle_path/chosen_edge_prob_mean` | 0.0073 |
| `oracle_path/continue_prob_mean` | 0.9995 |
| `oracle_path/terminal_stop_prob_mean` | 0.5626 |
| `oracle_path/chosen_edge_prob_depth_0` | 0.0095 |
| `oracle_path/chosen_edge_prob_depth_1` | 0.0064 |
| `oracle_path/chosen_edge_prob_depth_2` | 0.0041 |
| `oracle_path/continue_prob_depth_0` | 0.9994 |
| `oracle_path/continue_prob_depth_1` | 0.9996 |
| `oracle_path/continue_prob_depth_2` | 0.9997 |

## Rollout Comparison
| metric | value |
|---|---:|
| `policy/nonzero_f1_rate` | 0.0996 |
| `policy/answer_f1_mean` | 0.0273 |
| `policy/utility_mean` | 0.0382 |
| `policy/log_reward_mean` | -8.7512 |
| `policy/expanded_edge_count_mean` | 3.0000 |
| `policy/minimality_gap_mean` | 0.0000 |
| `policy/budget_exhausted_stop_rate` | 1.0000 |
| `policy/model_stop_rate` | 1.0000 |
| `policy/first_hit_depth_mean` | 1.7843 |
| `policy/hit_at_depth_1_rate` | 0.4510 |
| `policy/hit_at_depth_2_rate` | 0.6863 |
| `policy/continue_after_first_hit_rate` | 0.6863 |
| `policy/extra_edges_after_first_hit_mean` | 1.2157 |
| `prior_greedy_nonzero_f1_rate` | 0.5859 |
| `prior_beam4_nonzero_f1_rate` | 0.6641 |
| `prior_beam8_nonzero_f1_rate` | 0.7500 |
| `oracle/nonzero_f1_rate` | 0.7109 |
| `oracle/answer_f1_mean` | 0.5739 |
| `oracle/expanded_edge_count_mean` | 2.2500 |
| `oracle/minimality_gap_mean` | 0.0000 |

## Reward Sanity
| metric | value |
|---|---:|
| `reward/first_hit_log_reward_mean` | -1.8514 |
| `reward/final_log_reward_mean` | -1.8892 |
| `reward/final_minus_first_hit_log_reward_mean` | -0.0379 |
| `reward/final_better_than_first_hit_rate` | 0.0980 |
| `reward/minimality_gap_at_final_mean` | 0.0000 |
| `reward/minimality_penalty_at_final_mean` | 0.3000 |
| `reward/expanded_edge_count_at_final_mean` | 3.0000 |
| `reward/minimal_edge_count_at_final_mean` | 0.0000 |

## StopTB Ablation
This pass evaluates the same rollout traces under `stop_tb_coef=0.0` and `0.05`; it does not run a separate 200-500 step training ablation.
| metric | value |
|---|---:|
| `stop_tb_coef_0/loss_total` | 85.1732 |
| `stop_tb_coef_0/loss_subtb` | 85.1732 |
| `stop_tb_coef_0/loss_stop_tb` | 3.9486 |
| `stop_tb_coef_0_05/loss_total` | 85.3706 |
| `stop_tb_coef_0_05/loss_subtb` | 85.1732 |
| `stop_tb_coef_0_05/loss_stop_tb` | 3.9486 |
| `stop_tb/residual_abs_mean` | 1.3017 |
| `stop_tb/residual_after_hit_abs_mean` | 6.9150 |
| `stop_tb/valid_count_mean` | 3.0000 |
| `policy/stop_prob_after_hit` | 4.192e-04 |
| `policy/stop_prob_before_hit` | 4.882e-04 |

## Edge Prior/Residual
| metric | value |
|---|---:|
| `edge/prior_abs_mean` | 2.4207 |
| `edge/residual_abs_mean` | 0.0000 |
| `edge/residual_std` | 0.0000 |
| `edge/semantic_logit_std` | 0.3377 |
| `edge/residual_scaled_abs_mean` | 0.0000 |
| `edge/residual_to_prior_ratio` | 0.0000 |
| `edge/residual_to_prior_std_ratio` | 0.0000 |
| `edge/logit_scale` | 5.0000 |
| `edge/entity_weight` | 0.1000 |
| `edge/residual_scale` | 0.0000 |
| `edge/valid_edge_prior_rank_mean` | 21.6383 |
| `edge/valid_edge_final_rank_mean` | 21.6383 |
| `edge/valid_edge_rank_delta_mean` | 0.0000 |
| `edge/final_worse_than_prior_rate` | 0.0000 |
| `gate/frontier_logmeanexp_depth_0` | 2.5114 |
| `gate/frontier_logmeanexp_depth_1` | 2.5125 |
| `gate/option_gap_depth_0` | 8.4394 |
| `gate/option_gap_depth_1` | 8.6432 |
| `gate/option_gap_depth_2` | 8.6432 |

## Main Diagnosis
* Validation answers one-hop reachable: yes.
* Semantic prior ranks valid progress edges in top-k: yes.
* Current rollout hits shallow then continues: yes.
* Minimality reward prefers first-hit over final: yes.

## Recommended Next Change
Focus the next code change on Stop training: increase/verify stopTB and simplify or detach frontier summary in the option gate.
