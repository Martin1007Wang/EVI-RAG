# Weaver Rollout Diagnostics

- checkpoint: `random initialization`
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
| `prior/root_best_valid_rank_mean` | 14.5000 |
| `prior/root_best_valid_rank_median` | 1.0000 |
| `prior/root_valid_edge_top1_rate` | 0.6250 |
| `prior/root_valid_edge_top3_rate` | 0.6250 |
| `prior/root_valid_edge_top5_rate` | 0.6250 |
| `prior/root_valid_edge_top10_rate` | 0.7500 |
| `prior/root_valid_edge_mrr` | 0.6435 |
| `prior/depth1_valid_edge_exists_rate` | 0.6250 |
| `prior/depth1_best_valid_rank_mean` | 8.8000 |
| `prior/depth1_valid_edge_top10_rate` | 0.6000 |

## Root One-Step Reward Oracle
| metric | value |
|---|---:|
| `oracle/root_stop_log_reward` | -9.2103 |
| `oracle/root_best_child_log_reward` | -2.0600 |
| `oracle/root_best_child_minus_stop_log_reward` | 7.1503 |
| `oracle/root_best_child_support` | 0.5150 |
| `oracle/root_answer_edge_rank_by_policy` | 23.7500 |
| `oracle/root_policy_top1_child_support` | 0.3484 |
| `oracle/root_policy_top5_child_support` | 0.3484 |
| `oracle/root_candidate_count` | 388.8750 |

## Sampling Sanity
| metric | value |
|---|---:|
| `train/policy/target_stop_prob_mean` | 0.5000 |
| `train/policy/target_continue_prob_mean` | 0.5000 |
| `train/rollout/continue_depth_0_rate` | 0.4531 |
| `train/rollout/continue_rate` | 0.4435 |
| `train/rollout/budget_exhausted_ratio` | 0.1250 |
| `train/rollout/stop_depth_hist_0` | 0.5469 |
| `train/rollout/stop_depth_hist_1` | 0.2344 |
| `train/rollout/stop_depth_hist_2` | 0.0938 |
| `train/rollout/stop_depth_hist_3` | 0.1250 |

## Stop Improvement Oracle
| metric | value |
|---|---:|
| `stop_oracle/stop_now_better_ratio` | 0.2009 |
| `stop_oracle/mean_best_continue_minus_stop_log_reward` | 1.9125 |
| `stop_oracle/policy_stop_prob_when_stop_better` | 0.5000 |
| `stop_oracle/policy_continue_prob_when_continue_better` | 0.5000 |
| `stop_oracle/stop_now_better_ratio_depth_0` | 0.1250 |
| `stop_oracle/stop_now_better_ratio_depth_1` | 0.4286 |
| `stop_oracle/stop_now_better_ratio_depth_2` | 0.2500 |
| `stop_oracle/stop_now_better_ratio_depth_3` | 0.0000 |

## Rollout Comparison
| metric | value |
|---|---:|
| `policy/nonzero_f1_rate` | 0.0781 |
| `policy/answer_f1_mean` | 0.0116 |
| `policy/utility_mean` | 0.0107 |
| `policy/log_reward_mean` | -8.7763 |
| `policy/expanded_edge_count_mean` | 0.7969 |
| `policy/minimality_gap_mean` | 0.0000 |
| `policy/budget_exhausted_stop_rate` | 0.1250 |
| `policy/model_stop_rate` | 1.0000 |
| `policy/first_hit_depth_mean` | 1.2000 |
| `policy/hit_at_depth_1_rate` | 0.8000 |
| `policy/hit_at_depth_2_rate` | 1.0000 |
| `policy/continue_after_first_hit_rate` | 0.6000 |
| `policy/extra_edges_after_first_hit_mean` | 1.0000 |
| `prior_greedy_nonzero_f1_rate` | 0.6250 |
| `prior_beam4_nonzero_f1_rate` | 0.6250 |
| `prior_beam8_nonzero_f1_rate` | 0.7500 |
| `oracle/nonzero_f1_rate` | 0.8750 |
| `oracle/answer_f1_mean` | 0.7571 |
| `oracle/expanded_edge_count_mean` | 2.1250 |
| `oracle/minimality_gap_mean` | 0.0000 |

## Reward Sanity
| metric | value |
|---|---:|
| `reward/first_hit_log_reward_mean` | -2.8931 |
| `reward/final_log_reward_mean` | -2.8547 |
| `reward/final_minus_first_hit_log_reward_mean` | 0.0384 |
| `reward/final_better_than_first_hit_rate` | 0.2000 |
| `reward/minimality_gap_at_final_mean` | 0.0000 |
| `reward/minimality_penalty_at_final_mean` | 0.0000 |
| `reward/expanded_edge_count_at_final_mean` | 0.7969 |
| `reward/minimal_edge_count_at_final_mean` | 0.7969 |

## StopTB Ablation
This pass evaluates the same rollout traces under `stop_tb_coef=0.0` and `0.05`; it does not run a separate 200-500 step training ablation.
| metric | value |
|---|---:|
| `stop_tb_coef_0/loss_total` | 64.1909 |
| `stop_tb_coef_0/loss_subtb` | 64.1909 |
| `stop_tb_coef_0/loss_stop_tb` | 68.9392 |
| `stop_tb_coef_0_05/loss_total` | 67.6379 |
| `stop_tb_coef_0_05/loss_subtb` | 64.1909 |
| `stop_tb_coef_0_05/loss_stop_tb` | 68.9392 |
| `stop_tb/residual_abs_mean` | 8.1526 |
| `stop_tb/residual_after_hit_abs_mean` | 2.2726 |
| `stop_tb/valid_count_mean` | 1.6719 |
| `policy/stop_prob_after_hit` | 0.5000 |
| `policy/stop_prob_before_hit` | 0.5000 |

## Edge Prior/Residual
| metric | value |
|---|---:|
| `edge/prior_abs_mean` | 2.4312 |
| `edge/residual_abs_mean` | 0.0000 |
| `edge/residual_scaled_abs_mean` | 0.0000 |
| `edge/residual_to_prior_ratio` | 0.0000 |
| `edge/logit_scale` | 5.0000 |
| `edge/entity_weight` | 0.1000 |
| `edge/residual_scale` | 0.0500 |
| `edge/valid_edge_prior_rank_mean` | 17.6111 |
| `edge/valid_edge_final_rank_mean` | 17.6111 |
| `gate/frontier_logmeanexp_depth_0` | 2.4779 |
| `gate/frontier_logmeanexp_depth_1` | 2.5078 |
| `gate/option_gap_depth_0` | 0.0000 |
| `gate/option_gap_depth_1` | 0.0000 |
| `gate/option_gap_depth_2` | 0.0000 |

## Main Diagnosis
* Validation answers one-hop reachable: yes.
* Semantic prior ranks valid progress edges in top-k: yes.
* Current rollout hits shallow then continues: yes.
* Minimality reward prefers first-hit over final: no.

## Recommended Next Change
Fix reward minimality so first-hit sufficient states beat budget-full states.
