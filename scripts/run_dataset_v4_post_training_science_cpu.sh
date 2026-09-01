#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
RR="$EXPERIMENT_ROOT/dev_evaluation/v4_rr_random_raw_g4_b4_seed20260827_dev_matched"
GPU_A="$EXPERIMENT_ROOT/dev_evaluation/v4_risk50_raw_g4_b4_seed20260827_dev_matched"
GPU_B="$EXPERIMENT_ROOT/dev_evaluation/v4_risk50_safety_g4_b4_seed20260827_dev_matched"
RECOVERY="$EXPERIMENT_ROOT/technical_recovery/v4_risk50_safety_final_monitor_recovery_20260901"
CAPACITY="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_capacity_20260831_r1"
DECISION="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_decision_20260831"
DEV_SCENES="$CAPACITY/results/dev_scene_labels.csv"
DEV_TIERS="$DECISION/results/dev_v4_tier_labels.csv"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v4_post_training_science_20260901"

for path in "$PYTHON" "$RECOVERY/COMPLETE" "$RR/COMPLETE" "$GPU_A/COMPLETE" "$GPU_B/COMPLETE" \
    "$DEV_SCENES" "$DEV_TIERS"; do
    [[ -e "$path" ]] || { echo "Missing post-training science input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite post-training science run" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_post_training_science_20260901\nmodels=RR,GPU-A,GPU-B\ndev_tokens=416\nbootstrap_resamples=20000\nbootstrap_cluster=log_name\ntraining_seeds=1\ndev_previously_accessed=true\nfinal_accessed=false\n' > "$RUN_DIR/run.env"
sha256sum "$DEV_SCENES" "$DEV_TIERS" "$RECOVERY/recovery_report.json" \
    "$RR/results/scene_metrics.csv" "$RR/rollouts.jsonl" \
    "$GPU_A/results/scene_metrics.csv" "$GPU_A/rollouts.jsonl" \
    "$GPU_B/results/scene_metrics.csv" "$GPU_B/rollouts.jsonl" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    date +%s > "$RUN_DIR/end_epoch.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    rm -f "$RUN_DIR/RUNNING"
    if [[ "$status" -eq 0 ]]; then touch "$RUN_DIR/COMPLETE"; else touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m projects.dataset_v3.v4_post_training compare-dev \
    --rr-run "$RR" \
    --gpu-a-run "$GPU_A" \
    --gpu-b-run "$GPU_B" \
    --dev-scene-labels "$DEV_SCENES" \
    --original-dev-tiers "$DEV_TIERS" \
    --bootstrap-resamples 20000 \
    --seed 20260901 \
    --output-dir "$RUN_DIR/results" \
    > "$RUN_DIR/run.log" 2>&1
sha256sum "$RUN_DIR"/results/* > "$RUN_DIR/result_sha256.txt"
