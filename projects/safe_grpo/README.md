# Failure-Aware Safety GRPO

The live plan, experiment gates, evidence summaries, and adaptive next-step decisions are tracked in
[`docs/post_training_execution_loop.md`](../../docs/post_training_execution_loop.md). Update that document after each
completed stage before starting the next one.

Frozen experiment seed: `20260812`.

Evidence files on the server live under:

- `manifests/`: leak-free train/dev/held-out split
- `exp_root/metric_cache_released_5656/`: metric cache shared by all splits
- `logs/`: installation, cache, rollout, reward and training logs
- `experiments/<name>/`: resolved config, source commit, seed, checkpoint and metrics

GPU experiments must run in this order: E0 Stage-2 baseline, D0 frozen-train rollout diagnosis, E1 vanilla LoRA-GRPO, E2 FALS only, E3 SLDR only, E4 Std-Floor GRPO, E5 grouped reward throughput. Do not assign FALS thresholds or claim improvements before rollout evidence exists. D0 must cover all 4,525 frozen train tokens with four rollouts each; dev and held-out tokens are forbidden.

On a 24 GB GPU, E0 and D0 keep the rank-8 LoRA wrapper at its zero-effect initialization. PEFT initializes the LoRA B projection to zero, so the adapter does not change the Stage-2 output before training; removing the wrapper instead makes the full actor exceed the single-GPU hybrid-engine memory budget.

Experiment switches:

- Vanilla grouped reward: `REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_group_fast`
- SLDR: `REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_sldr`
- Std-Floor GRPO: `ADV_ESTIMATOR=std_floor_grpo STD_FLOOR=0.05`

Build FALS manifests only from frozen train-token rollouts:

```bash
python EasyR1/scripts/adas/build_fals_filter.py \
  --rollouts experiments/d0_train_rollouts.jsonl \
  --train-manifest manifests/train_4525.txt \
  --output-dir manifests/fals \
  --expected-rollouts 4 --budget 1000 --budget 2000
```
