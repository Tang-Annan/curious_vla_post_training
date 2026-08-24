#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/safe_grpo"
RUN_DIR="$EXPERIMENT_ROOT/a0_adas_pool_audit_seed20260812"
RELEASED_FILTER="$PROJECT_ROOT/token_filters/curious_vla_qwen2_5_vl_3b_sft_stage2_adas1x_6k.txt"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
DEV_MANIFEST="$WORKSPACE_ROOT/manifests/dev_tokens.txt"
HELDOUT_MANIFEST="$WORKSPACE_ROOT/manifests/heldout_tokens.txt"
RANDOM_MANIFEST="$WORKSPACE_ROOT/manifests/dev_subsets/train_seed20260812_1000.txt"

for path in "$PYTHON" "$RELEASED_FILTER" "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$HELDOUT_MANIFEST" \
    "$RANDOM_MANIFEST"; do
    [[ -e "$path" ]] || { echo "Missing A0 input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "A0 output already exists: $RUN_DIR" >&2; exit 1; }
cd "$PROJECT_ROOT"
[[ -z "$(git status --short)" ]] || { echo "Source checkout is not clean." >&2; exit 1; }

mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
sha256sum "$RELEASED_FILTER" "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$HELDOUT_MANIFEST" \
    "$RANDOM_MANIFEST" > "$RUN_DIR/input_sha256.txt"
printf 'seed=%s\ngpu=none\nmanifest_write=false\n' 20260812 > "$RUN_DIR/run.env"
touch "$RUN_DIR/RUNNING"

set +e
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/audit_adas_pool.py" \
    --released-filter "$RELEASED_FILTER" \
    --train-manifest "$TRAIN_MANIFEST" \
    --dev-manifest "$DEV_MANIFEST" \
    --heldout-manifest "$HELDOUT_MANIFEST" \
    --random-manifest "$RANDOM_MANIFEST" \
    --output "$RUN_DIR/a0_report.json" > "$RUN_DIR/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$RUN_DIR/exit_code"
rm -f "$RUN_DIR/RUNNING"
if [[ $status -ne 0 ]]; then
    touch "$RUN_DIR/FAILED"
    exit "$status"
fi
[[ -s "$RUN_DIR/a0_report.json" ]] || { echo "Missing A0 report." >&2; exit 1; }
touch "$RUN_DIR/COMPLETE"
