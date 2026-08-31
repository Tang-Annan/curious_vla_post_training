#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
CAPACITY="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_capacity_20260831_r1"
RANDOM_FREEZE="$EXPERIMENT_ROOT/selector_freeze/v3_s1_selector_freeze_20260829"
TRAIN_SCENES="$CAPACITY/results/train_scene_labels.csv"
RANDOM_MANIFEST="$RANDOM_FREEZE/results/random_train_2000.txt"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v4_risk_ratio_audit_20260831"

for path in "$PYTHON" "$CAPACITY/COMPLETE" "$RANDOM_FREEZE/COMPLETE" \
    "$TRAIN_SCENES" "$RANDOM_MANIFEST"; do
    [[ -e "$path" ]] || { echo "Missing V4 risk-ratio input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V4 risk-ratio run: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_risk_ratio_audit_20260831\ncuda_visible_devices=empty\ndev_accessed=false\nfinal_accessed=false\ngpu_training_authorized=false\n' > "$RUN_DIR/run.env"
sha256sum "$TRAIN_SCENES" "$RANDOM_MANIFEST" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    date +%s > "$RUN_DIR/end_epoch.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    rm -f "$RUN_DIR/RUNNING"
    if [[ "$status" -eq 0 ]]; then touch "$RUN_DIR/COMPLETE"; else touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m projects.dataset_v3.v4_risk_ratio_audit \
    --train-scene-labels "$TRAIN_SCENES" \
    --random-manifest "$RANDOM_MANIFEST" \
    --output-dir "$RUN_DIR/results" \
    --seed 20260831 \
    > "$RUN_DIR/run.log" 2>&1

sha256sum "$RUN_DIR"/results/* > "$RUN_DIR/result_sha256.txt"
