#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
RR="$EXPERIMENT_ROOT/formal_runs/v3_rr_random_raw_g4_b4_seed20260827"
RISK_AUDIT="$EXPERIMENT_ROOT/semantic_audit/v4_risk_ratio_audit_20260831"
SELECTOR="$EXPERIMENT_ROOT/selector_freeze/v3_s1_selector_freeze_20260829/results"
DATA_ROOT="$WORKSPACE_ROOT/data"
SCREEN="$DATA_ROOT/dataset_v3_controlled_overlap/hf/grpo_screen.parquet"
MONITOR_MANIFEST="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap/train_monitor_256.txt"
MONITOR_PARQUET="$DATA_ROOT/dataset_v3_controlled_overlap/hf/train_monitor.parquet"
M0="$EXPERIMENT_ROOT/protocol_freeze/v3_m0_matrix_protocol_20260829/results/m0_protocol.json"
FUTURE_RUN="$EXPERIMENT_ROOT/formal_runs/v4_risk50_raw_g4_b4_seed20260827"
RUN_DIR="$EXPERIMENT_ROOT/training_prepare/v4_risk50_rr_aligned_prepare_20260831_r1"

for path in "$PYTHON" "$RR/COMPLETE" "$RR/checkpoints/experiment_config.json" \
    "$RR/model_sha256.txt" "$RISK_AUDIT/COMPLETE" \
    "$RISK_AUDIT/results/frozen_current_visible_risk50_2000.txt" \
    "$SELECTOR/random_train_2000.parquet" "$SCREEN" "$MONITOR_MANIFEST" \
    "$MONITOR_PARQUET" "$M0" "$DATA_ROOT"; do
    [[ -e "$path" ]] || { echo "Missing V4 training-preparation input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V4 training preparation: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$FUTURE_RUN" ]] || { echo "Future V4 training run already exists" >&2; exit 1; }
[[ ! -e "$PROJECT_ROOT/EasyR1/checkpoints/debug/v4_risk50_raw_g4_b4_seed20260827" ]] || { echo "Future V4 debug output already exists" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_risk50_rr_aligned_prepare_20260831_r1\ncuda_visible_devices=empty\ndev_accessed=false\nfinal_accessed=false\ngpu_training_authorized=false\n' > "$RUN_DIR/run.env"
sha256sum "$RISK_AUDIT/results/frozen_current_visible_risk50_2000.txt" "$SCREEN" \
    "$SELECTOR/random_train_2000.parquet" "$RR/checkpoints/experiment_config.json" \
    "$MONITOR_MANIFEST" "$MONITOR_PARQUET" "$M0" > "$RUN_DIR/input_sha256.txt"
git -C "$PROJECT_ROOT" diff --name-status "$(cat "$RR/source_commit.txt")..HEAD" > "$RUN_DIR/rr_to_current_source_diff.txt"
git -C "$PROJECT_ROOT" diff --quiet "$(cat "$RR/source_commit.txt")..HEAD" -- \
    EasyR1/verl/trainer/main.py EasyR1/verl/trainer/config.py EasyR1/verl/trainer/data_loader.py \
    EasyR1/verl/trainer/ray_trainer.py EasyR1/verl/trainer/core_algos.py \
    EasyR1/verl/utils/reward_score/navsim/navsim_reward_text.py

cleanup() {
    status=$?
    date +%s > "$RUN_DIR/end_epoch.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    rm -f "$RUN_DIR/RUNNING"
    if [[ "$status" -eq 0 ]]; then touch "$RUN_DIR/COMPLETE"; else touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT

cd "$WORKSPACE_ROOT"
sha256sum -c "$RR/model_sha256.txt" > "$RUN_DIR/model_hash_check.txt"
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m projects.dataset_v3.v4_training_prepare prepare \
    --frozen-manifest "$RISK_AUDIT/results/frozen_current_visible_risk50_2000.txt" \
    --screen-parquet "$SCREEN" \
    --rr-parquet "$SELECTOR/random_train_2000.parquet" \
    --rr-config "$RR/checkpoints/experiment_config.json" \
    --rr-run-dir "$RR" \
    --monitor-manifest "$MONITOR_MANIFEST" \
    --monitor-parquet "$MONITOR_PARQUET" \
    --m0-protocol "$M0" \
    --data-root "$DATA_ROOT" \
    --future-run-dir "$FUTURE_RUN" \
    --source-status "$RUN_DIR/source_status.txt" \
    --model-hash-check "$RUN_DIR/model_hash_check.txt" \
    --output-dir "$RUN_DIR/results" \
    > "$RUN_DIR/prepare.log" 2>&1

CUDA_VISIBLE_DEVICES= "$PYTHON" -m projects.dataset_v3.v4_training_prepare smoke-loader \
    --config "$RUN_DIR/results/risk50_rr_aligned_config.json" \
    --output "$RUN_DIR/results/dataloader_smoke_report.json" \
    > "$RUN_DIR/dataloader_smoke.log" 2>&1

sha256sum "$RUN_DIR"/results/* > "$RUN_DIR/result_sha256.txt"
