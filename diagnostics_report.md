# Weaver Rollout Diagnostics

- checkpoint: `/mnt/wangjingxiong/EVI-RAG/current_weaver_hard.ckpt`
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
| `prior/root_best_valid_rank_mean` | 16.7768 |
| `prior/root_best_valid_rank_median` | 3.5938 |
| `prior/root_valid_edge_top1_rate` | 0.4877 |
| `prior/root_valid_edge_top3_rate` | 0.5904 |
| `prior/root_valid_edge_top5_rate` | 0.6607 |
| `prior/root_valid_edge_top10_rate` | 0.7310 |
| `prior/root_valid_edge_mrr` | 0.5679 |
| `prior/depth1_valid_edge_exists_rate` | 0.7031 |
| `prior/depth1_best_valid_rank_mean` | 21.6589 |
| `prior/depth1_valid_edge_top10_rate` | 0.6616 |

## Rollout Comparison
| metric | value |
|---|---:|
| `policy/nonzero_f1_rate` | 0.2734 |
| `policy/answer_f1_mean` | 0.0902 |
| `policy/utility_mean` | 0.1020 |
| `policy/log_reward_mean` | -7.2019 |
| `policy/expanded_edge_count_mean` | 2.9961 |
| `policy/minimality_gap_mean` | 0.4932 |
| `policy/budget_exhausted_stop_rate` | 0.9961 |
| `policy/model_stop_rate` | 1.0000 |
| `policy/first_hit_depth_mean` | 1.5321 |
| `policy/hit_at_depth_1_rate` | 0.5857 |
| `policy/hit_at_depth_2_rate` | 0.8536 |
| `policy/continue_after_first_hit_rate` | 0.8500 |
| `policy/extra_edges_after_first_hit_mean` | 1.4643 |
| `prior_greedy_nonzero_f1_rate` | 0.5859 |
| `prior_beam4_nonzero_f1_rate` | 0.6562 |
| `prior_beam8_nonzero_f1_rate` | 0.7500 |
| `oracle/nonzero_f1_rate` | 0.7109 |
| `oracle/answer_f1_mean` | 0.5739 |
| `oracle/expanded_edge_count_mean` | 2.2500 |
| `oracle/minimality_gap_mean` | 0.0234 |

## Reward Sanity
| metric | value |
|---|---:|
| `reward/first_hit_log_reward_mean` | -1.3434 |
| `reward/final_log_reward_mean` | -1.6263 |
| `reward/final_minus_first_hit_log_reward_mean` | -0.2828 |
| `reward/final_better_than_first_hit_rate` | 0.1750 |
| `reward/minimality_gap_at_final_mean` | 0.4932 |
| `reward/minimality_penalty_at_final_mean` | 0.0740 |
| `reward/expanded_edge_count_at_final_mean` | 2.9961 |
| `reward/minimal_edge_count_at_final_mean` | 2.5029 |

## StopTB Ablation
This pass evaluates the same rollout traces under `stop_tb_coef=0.0` and `0.05`; it does not run a separate 200-500 step training ablation.
| metric | value |
|---|---:|
| `stop_tb_coef_0/loss_total` | 4.9994 |
| `stop_tb_coef_0/loss_subtb` | 4.9994 |
| `stop_tb_coef_0/loss_stop_tb` | 2.3902 |
| `stop_tb_coef_0_05/loss_total` | 5.1189 |
| `stop_tb_coef_0_05/loss_subtb` | 4.9994 |
| `stop_tb_coef_0_05/loss_stop_tb` | 2.3902 |
| `stop_tb/residual_abs_mean` | 0.9458 |
| `stop_tb/residual_after_hit_abs_mean` | 2.2979 |
| `stop_tb/valid_count_mean` | 3.0000 |
| `policy/stop_prob_after_hit` | 0.0024 |
| `policy/stop_prob_before_hit` | 0.0013 |

## Edge Prior/Residual
| metric | value |
|---|---:|
| `edge/prior_abs_mean` | 4.8116 |
| `edge/residual_abs_mean` | 4.0361 |
| `edge/residual_scaled_abs_mean` | 0.4723 |
| `edge/residual_to_prior_ratio` | 0.0980 |
| `edge/logit_scale` | 9.9242 |
| `edge/entity_weight` | 0.1064 |
| `edge/residual_scale` | 0.1170 |
| `edge/valid_edge_prior_rank_mean` | 23.8762 |
| `edge/valid_edge_final_rank_mean` | 64.2761 |
| `gate/frontier_logmeanexp_depth_0` | 4.8765 |
| `gate/frontier_logmeanexp_depth_1` | 4.7635 |
| `gate/option_gap_depth_0` | 18.5222 |
| `gate/option_gap_depth_1` | 11.1630 |
| `gate/option_gap_depth_2` | 11.1630 |

## Main Diagnosis
* Validation answers one-hop reachable: yes.
* Semantic prior ranks valid progress edges in top-k: yes.
* Current rollout hits shallow then continues: yes.
* Minimality reward prefers first-hit over final: yes.

## Recommended Next Change
Focus the next code change on Stop training: increase/verify stopTB and simplify or detach frontier summary in the option gate.
