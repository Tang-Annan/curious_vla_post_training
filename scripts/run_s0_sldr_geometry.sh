#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/safe_grpo"
D0_DIR="$EXPERIMENT_ROOT/d0_stage2_train_n4_seed20260812"
RUN_DIR="$EXPERIMENT_ROOT/s0_sldr_geometry_seed20260812"
ROLLOUTS="$D0_DIR/d0_train_rollouts.jsonl"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
EXPECTED_ROLLOUT_SHA256=2ededee1d08d754c251a1f1777d2df4e44e52f4a859e884afeed95521e6ef9d6
EXPECTED_TRAIN_SHA256=4a19947abd86d4265e055a6408fc8a6d579fcc083cb5bc4c207159d5c60d8168

for path in "$PYTHON" "$ROLLOUTS" "$TRAIN_MANIFEST" "$D0_DIR/COMPLETE" "$D0_DIR/exit_code"; do
    [[ -e "$path" ]] || { echo "Missing required S0 input: $path" >&2; exit 1; }
done
[[ "$(tr -d '[:space:]' < "$D0_DIR/exit_code")" == 0 ]] || { echo "D0 exit_code is not zero." >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "S0 output already exists: $RUN_DIR" >&2; exit 1; }
printf '%s  %s\n' "$EXPECTED_ROLLOUT_SHA256" "$ROLLOUTS" | sha256sum --check --status
printf '%s  %s\n' "$EXPECTED_TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --status

cd "$PROJECT_ROOT"
[[ -z "$(git status --short)" ]] || { echo "Source checkout is not clean." >&2; exit 1; }
mkdir -p "$RUN_DIR/results"
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git status --short > "$RUN_DIR/source_status.txt"
sha256sum "$ROLLOUTS" "$TRAIN_MANIFEST" > "$RUN_DIR/input_sha256.txt"
printf 'seed=%s\nbootstrap_samples=%s\nrollouts_per_group=%s\ncuda_visible_devices=empty\n' \
    20260812 20000 4 > "$RUN_DIR/run.env"
touch "$RUN_DIR/RUNNING"

set +e
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PROJECT_ROOT/EasyR1" "$PYTHON" \
    "$PROJECT_ROOT/projects/safe_grpo/analyze_sldr_geometry.py" \
    --d0-rollouts "$ROLLOUTS" \
    --train-manifest "$TRAIN_MANIFEST" \
    --output-dir "$RUN_DIR/results" \
    --bootstrap-samples 20000 \
    --seed 20260812 > "$RUN_DIR/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$RUN_DIR/exit_code"
rm -f "$RUN_DIR/RUNNING"
if [[ $status -ne 0 ]]; then
    touch "$RUN_DIR/FAILED"
    exit "$status"
fi

for artifact in s0_report.json group_geometry.csv unsafe_preference_group_differences.csv; do
    [[ -s "$RUN_DIR/results/$artifact" ]] || { echo "Missing S0 artifact: $artifact" >&2; exit 1; }
done
touch "$RUN_DIR/COMPLETE"
