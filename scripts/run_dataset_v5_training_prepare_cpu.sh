#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
DATASET_RUN="$EXPERIMENT_ROOT/semantic_audit/v5_risk_fals_datasets_20260904_r1"
RR="$EXPERIMENT_ROOT/formal_runs/v3_rr_random_raw_g4_b4_seed20260827"
SELECTOR="$EXPERIMENT_ROOT/selector_freeze/v3_s1_selector_freeze_20260829/results"
DATA_ROOT="$WORKSPACE_ROOT/data"
MONITOR_MANIFEST="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap/train_monitor_256.txt"
MONITOR_PARQUET="$DATA_ROOT/dataset_v3_controlled_overlap/hf/train_monitor.parquet"
M0="$EXPERIMENT_ROOT/protocol_freeze/v3_m0_matrix_protocol_20260829/results/m0_protocol.json"
RISK50_RUN="$EXPERIMENT_ROOT/formal_runs/v5_risk50_raw_g4_b4_seed20260827"
FALS_RUN="$EXPERIMENT_ROOT/formal_runs/v5_risk50_fals_raw_g4_b4_seed20260827"
RISK50_DEBUG="$PROJECT_ROOT/EasyR1/checkpoints/debug/v5_risk50_raw_g4_b4_seed20260827"
FALS_DEBUG="$PROJECT_ROOT/EasyR1/checkpoints/debug/v5_risk50_fals_raw_g4_b4_seed20260827"
RUN_DIR="$EXPERIMENT_ROOT/training_prepare/v5_risk_fals_gpu_prepare_20260904_r2"

for path in "$PYTHON" "$DATASET_RUN/COMPLETE" "$DATASET_RUN/results/v5_risk_fals_dataset_report.json" \
    "$RR/COMPLETE" "$RR/checkpoints/experiment_config.json" "$RR/model_sha256.txt" \
    "$SELECTOR/random_train_2000.parquet" "$MONITOR_MANIFEST" "$MONITOR_PARQUET" "$M0" "$DATA_ROOT"; do
    [[ -e "$path" ]] || { echo "Missing V5 training-preparation input: $path" >&2; exit 1; }
done
[[ "$(cat "$DATASET_RUN/exit_code")" == 0 ]] || { echo "Corrected V5 dataset run failed" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V5 training preparation" >&2; exit 1; }
for path in "$RISK50_RUN" "$FALS_RUN" "$RISK50_DEBUG" "$FALS_DEBUG"; do
    [[ ! -e "$path" ]] || { echo "Future V5 output already exists: $path" >&2; exit 1; }
done
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
git -C "$PROJECT_ROOT" diff --name-status "$(cat "$RR/source_commit.txt")..HEAD" > "$RUN_DIR/rr_to_current_source_diff.txt"
printf 'run_id=v5_risk_fals_gpu_prepare_20260904_r2\ncuda_visible_devices=empty\ndev_accessed=false\nfinal_accessed=false\ngpu_used=false\ntraining_launched=false\n' > "$RUN_DIR/run.env"

cleanup() {
    status=$?
    date +%s > "$RUN_DIR/end_epoch.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    rm -f "$RUN_DIR/RUNNING"
    if [[ "$status" -eq 0 ]]; then touch "$RUN_DIR/COMPLETE"; else touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

sha256sum -c "$DATASET_RUN/result_sha256.txt" > "$RUN_DIR/dataset_hash_check.txt"
cd "$WORKSPACE_ROOT"
sha256sum -c "$RR/model_sha256.txt" > "$RUN_DIR/model_hash_check.txt"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=
"$PYTHON" -m pytest tests/test_v5_risk_fals_datasets.py tests/test_v5_training_prepare.py \
    tests/test_export_training_evidence.py -q \
    --basetemp="/tmp/pytest_v5_training_prepare_20260904_$$"
"$PYTHON" -m compileall -q projects/dataset_v3/v5_risk_fals_datasets.py \
    projects/dataset_v3/v5_training_prepare.py projects/dataset_v3/formal_pipeline.py \
    projects/safe_grpo/export_training_evidence.py
bash -n scripts/run_dataset_v5_risk_fals_prepare_cpu.sh \
    scripts/run_dataset_v5_training_prepare_cpu.sh scripts/run_dataset_v3_formal_cell.sh
git diff --check

"$PYTHON" -m projects.dataset_v3.v5_training_prepare prepare \
    --dataset-run "$DATASET_RUN" \
    --rr-parquet "$SELECTOR/random_train_2000.parquet" \
    --rr-config "$RR/checkpoints/experiment_config.json" \
    --monitor-manifest "$MONITOR_MANIFEST" \
    --monitor-parquet "$MONITOR_PARQUET" \
    --m0-protocol "$M0" \
    --data-root "$DATA_ROOT" \
    --risk50-future-run "$RISK50_RUN" \
    --risk50-fals-future-run "$FALS_RUN" \
    --risk50-debug-dir "$RISK50_DEBUG" \
    --risk50-fals-debug-dir "$FALS_DEBUG" \
    --source-status "$RUN_DIR/source_status.txt" \
    --model-hash-check "$RUN_DIR/model_hash_check.txt" \
    --output-dir "$RUN_DIR/results"

"$PYTHON" -m projects.dataset_v3.v5_training_prepare smoke-loaders \
    --risk50-config "$RUN_DIR/results/v5_risk50_raw_config.json" \
    --risk50-fals-config "$RUN_DIR/results/v5_risk50_fals_raw_config.json" \
    --output "$RUN_DIR/results/dataloader_smoke_report.json"

sha256sum "$RUN_DIR"/results/* > "$RUN_DIR/result_sha256.txt"
