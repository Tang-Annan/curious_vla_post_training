# Failure-Aware Safety GRPO

Frozen experiment seed: `20260812`.

Evidence files on the server live under:

- `manifests/`: leak-free train/dev/held-out split
- `exp_root/metric_cache_released_5656/`: metric cache shared by all splits
- `logs/`: installation, cache, rollout, reward and training logs
- `experiments/<name>/`: resolved config, source commit, seed, checkpoint and metrics

GPU experiments must run in this order: E0 Stage-2 baseline, E1 vanilla LoRA-GRPO, E2 FALS only, E3 SLDR only, E4 Std-Floor GRPO, E5 grouped reward throughput. Do not assign FALS thresholds or claim improvements before rollout evidence exists.
