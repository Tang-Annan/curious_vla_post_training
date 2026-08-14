#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/safe_grpo"
RUN_DIR="$EXPERIMENT_ROOT/${R0_RUN_NAME:-r0_difficulty_bias_seed20260812}"
D0_DIR="$EXPERIMENT_ROOT/d0_stage2_train_n4_seed20260812"
E2_DIR="$EXPERIMENT_ROOT/e2_fals_lora_1k_seed20260812"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
FALS_MANIFEST="$WORKSPACE_ROOT/manifests/fals_d0_seed20260812/fals_top_1000.txt"
RANDOM_MANIFEST="$WORKSPACE_ROOT/manifests/dev_subsets/train_seed20260812_1000.txt"

for path in "$PYTHON" "$D0_DIR/d0_train_rollouts.jsonl" "$E2_DIR/train_rollouts.jsonl" \
    "$TRAIN_MANIFEST" "$FALS_MANIFEST" "$RANDOM_MANIFEST"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "R0 output already exists: $RUN_DIR" >&2; exit 1; }

cd "$PROJECT_ROOT"
[[ -z "$(git status --short)" ]] || { echo "Source checkout is not clean." >&2; exit 1; }
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
sha256sum "$D0_DIR/d0_train_rollouts.jsonl" "$E2_DIR/train_rollouts.jsonl" \
    "$TRAIN_MANIFEST" "$FALS_MANIFEST" "$RANDOM_MANIFEST" > "$RUN_DIR/input_sha256.txt"
printf 'seed=%s\nbootstrap_samples=%s\nmonte_carlo_trials=%s\n' 20260814 20000 100000 > "$RUN_DIR/run.env"

set +e
"$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/analyze_difficulty_bias.py" \
    --d0-rollouts "$D0_DIR/d0_train_rollouts.jsonl" \
    --e2-rollouts "$E2_DIR/train_rollouts.jsonl" \
    --train-manifest "$TRAIN_MANIFEST" \
    --fals-manifest "$FALS_MANIFEST" \
    --random-manifest "$RANDOM_MANIFEST" \
    --output-dir "$RUN_DIR/results" > "$RUN_DIR/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$RUN_DIR/exit_code"
if [[ $status -ne 0 ]]; then
    touch "$RUN_DIR/FAILED"
    exit "$status"
fi

for artifact in group_metrics.csv advantage_scale.csv r0_report.json difficulty_bias.svg; do
    [[ -s "$RUN_DIR/results/$artifact" ]] || { echo "Missing R0 artifact: $artifact" >&2; exit 1; }
done
touch "$RUN_DIR/COMPLETE"
