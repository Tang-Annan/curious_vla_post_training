#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
CAPACITY="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_capacity_20260831_r1"
DECISION="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_decision_20260831"
V3_AUDIT="$EXPERIMENT_ROOT/semantic_audit/v3_tail_semantic_alignment_20260831"
TRAIN_SCENES="$CAPACITY/results/train_scene_labels.csv"
TRAIN_TIERS="$DECISION/results/train_v4_tier_labels.csv"
DEV_SCENES="$CAPACITY/results/dev_scene_labels.csv"
DEV_TIERS="$DECISION/results/dev_v4_tier_labels.csv"
DEV_MODELS="$V3_AUDIT/results/dev_model_outcomes.csv"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v4_experiment_closure_cpu_20260831_r1"

for path in "$PYTHON" "$CAPACITY/COMPLETE" "$DECISION/COMPLETE" "$V3_AUDIT/COMPLETE" \
    "$TRAIN_SCENES" "$TRAIN_TIERS" "$DEV_SCENES" "$DEV_TIERS" "$DEV_MODELS"; do
    [[ -e "$path" ]] || { echo "Missing V4 closure input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V4 closure run: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_experiment_closure_cpu_20260831_r1\ncuda_visible_devices=empty\nbootstrap_resamples=20000\ndev_accessed=true\nfinal_accessed=false\ngpu_training_authorized=false\n' > "$RUN_DIR/run.env"
sha256sum "$TRAIN_SCENES" "$TRAIN_TIERS" "$DEV_SCENES" "$DEV_TIERS" "$DEV_MODELS" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    date +%s > "$RUN_DIR/end_epoch.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    rm -f "$RUN_DIR/RUNNING"
    if [[ "$status" -eq 0 ]]; then touch "$RUN_DIR/COMPLETE"; else touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m projects.dataset_v3.v4_experiment_closure \
    --train-scene-labels "$TRAIN_SCENES" \
    --train-tier-labels "$TRAIN_TIERS" \
    --dev-scene-labels "$DEV_SCENES" \
    --dev-tier-labels "$DEV_TIERS" \
    --dev-model-outcomes "$DEV_MODELS" \
    --output-dir "$RUN_DIR/results" \
    --bootstrap-resamples 20000 \
    --seed 20260831 \
    > "$RUN_DIR/run.log" 2>&1

sha256sum "$RUN_DIR"/results/* > "$RUN_DIR/result_sha256.txt"
