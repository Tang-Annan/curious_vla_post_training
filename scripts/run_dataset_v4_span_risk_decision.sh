#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
AUDIT_DIR="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_capacity_20260831_r1"
TRAIN_LABELS="$AUDIT_DIR/results/train_scene_labels.csv"
DEV_LABELS="$AUDIT_DIR/results/dev_scene_labels.csv"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_decision_20260831"

for path in "$PYTHON" "$AUDIT_DIR/COMPLETE" "$TRAIN_LABELS" "$DEV_LABELS"; do
    [[ -e "$path" ]] || { echo "Missing V4 decision input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V4 decision run: $RUN_DIR" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_span_inspired_risk_decision_20260831\ncuda_visible_devices=empty\ndev_accessed=true\nfinal_accessed=false\n' > "$RUN_DIR/run.env"
sha256sum "$TRAIN_LABELS" "$DEV_LABELS" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    date +%s > "$RUN_DIR/end_epoch.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    rm -f "$RUN_DIR/RUNNING"
    if [[ "$status" -eq 0 ]]; then
        touch "$RUN_DIR/COMPLETE"
    else
        touch "$RUN_DIR/FAILED"
    fi
}
trap cleanup EXIT

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m projects.dataset_v3.span_risk_decision \
    --train-labels "$TRAIN_LABELS" \
    --dev-labels "$DEV_LABELS" \
    --output-dir "$RUN_DIR/results" \
    --seed 20260831 \
    > "$RUN_DIR/run.log" 2>&1

sha256sum "$RUN_DIR"/results/* > "$RUN_DIR/result_sha256.txt"
