# Failure-Aware Safety GRPO

The live plan, experiment gates, evidence summaries, and adaptive next-step decisions are tracked in
[`docs/post_training_execution_loop.md`](../../docs/post_training_execution_loop.md). Update that document after each
completed stage before starting the next one.

Primary discovery seed: `20260812`. Any matched confirmation seed must first pass the live ledger gate.

Evidence files on the server live under:

- `manifests/`: leak-free train/dev/held-out split
- `exp_root/metric_cache_released_5656/`: metric cache shared by all splits
- `logs/`: installation, cache, rollout, reward and training logs
- `experiments/<name>/`: resolved config, source commit, seed, checkpoint and metrics

E0–E4 are completed historical stages. All new GPU experiments must follow the live order, preregistered gates, and adaptive branches in the execution ledger; R0 is the current offline-only stage, so R1/R2 must not start before its decision is written back. Do not claim improvements before same-protocol evidence exists, and keep held-out sealed until the ledger's final F1 gate.

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
