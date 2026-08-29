#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
AUDIT_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/rollout_bank/v3_s1_stability_capacity_audit_20260829"
RUN_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/selector_freeze/v3_s1_selector_freeze_20260829"
SCREEN_PARQUET="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/hf/grpo_screen.parquet"
MANIFEST_ROOT="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap"
RANDOM_MANIFEST="$MANIFEST_ROOT/random_train_2000.txt"
MASTER_INDEX="$MANIFEST_ROOT/master_index.csv"

for path in "$PYTHON" "$AUDIT_DIR/COMPLETE" "$AUDIT_DIR/results/stability_capacity.csv" \
    "$AUDIT_DIR/results/stability_capacity_report.json" "$SCREEN_PARQUET" "$RANDOM_MANIFEST" "$MASTER_INDEX"; do
    [[ -e "$path" ]] || { echo "Missing S1 selector-freeze input: $path" >&2; exit 1; }
done
[[ "$(cat "$AUDIT_DIR/exit_code")" == 0 ]] || { echo "Stability capacity audit is not complete" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite selector-freeze directory: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
sha256sum "$AUDIT_DIR/results/stability_capacity.csv" "$AUDIT_DIR/results/stability_capacity_report.json" \
    "$SCREEN_PARQUET" "$RANDOM_MANIFEST" "$MASTER_INDEX" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

"$PYTHON" "$PROJECT_ROOT/projects/dataset_v3/s1_pipeline.py" build-selectors \
    --stability-capacity "$AUDIT_DIR/results/stability_capacity.csv" \
    --stability-report "$AUDIT_DIR/results/stability_capacity_report.json" \
    --screen-parquet "$SCREEN_PARQUET" \
    --random-manifest "$RANDOM_MANIFEST" \
    --master-index "$MASTER_INDEX" \
    --output-dir "$RUN_DIR/results" \
    --seed 20260827

sha256sum "$RUN_DIR/results/random_train_2000.txt" "$RUN_DIR/results/tailmix_train_2000.txt" \
    "$RUN_DIR/results/random_train_2000.parquet" "$RUN_DIR/results/tailmix_train_2000.parquet" \
    "$RUN_DIR/results/selector_membership.csv" "$RUN_DIR/results/selector_freeze_report.json" \
    > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
