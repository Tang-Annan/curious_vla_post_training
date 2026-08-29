#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
DATA_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap"
MANIFEST_ROOT="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap"
SELECTOR_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/selector_freeze/v3_s1_selector_freeze_20260829/results"
RUN_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/hparam_freeze/v3_h0_protocol_20260829"

for path in "$PYTHON" "$SELECTOR_ROOT/random_train_2000.txt" "$SELECTOR_ROOT/random_train_2000.parquet" \
    "$MANIFEST_ROOT/train_monitor_256.txt" "$DATA_ROOT/hf/train_monitor.parquet" "$MANIFEST_ROOT/master_index.csv"; do
    [[ -e "$path" ]] || { echo "Missing V3-H0 preparation input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite H0 protocol: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
sha256sum "$SELECTOR_ROOT/random_train_2000.txt" "$SELECTOR_ROOT/random_train_2000.parquet" \
    "$MANIFEST_ROOT/train_monitor_256.txt" "$DATA_ROOT/hf/train_monitor.parquet" \
    "$MANIFEST_ROOT/master_index.csv" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

cd "$PROJECT_ROOT"
"$PYTHON" -m projects.dataset_v3.h0_pipeline prepare \
    --random-manifest "$SELECTOR_ROOT/random_train_2000.txt" \
    --random-parquet "$SELECTOR_ROOT/random_train_2000.parquet" \
    --monitor-manifest "$MANIFEST_ROOT/train_monitor_256.txt" \
    --monitor-parquet "$DATA_ROOT/hf/train_monitor.parquet" \
    --master-index "$MANIFEST_ROOT/master_index.csv" \
    --seed 20260829 \
    --output-dir "$RUN_DIR/results"

sha256sum "$RUN_DIR/results"/* > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
