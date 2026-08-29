#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
SCREEN_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/rollout_bank/v3_s1_metric_replay_20260829"
SELECTOR_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/selector_freeze/v3_s1_selector_freeze_20260829"
RUN_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/reward_freeze/v3_r0_geometry_candidates_20260829_retry1"
SCREEN_MANIFEST="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap/grpo_screen_8000.txt"

for path in "$PYTHON" "$SCREEN_DIR/COMPLETE" "$SCREEN_DIR/screen_rollouts_enriched.jsonl" \
    "$SELECTOR_DIR/COMPLETE" "$SELECTOR_DIR/results/random_train_2000.txt" \
    "$SELECTOR_DIR/results/tailmix_train_2000.txt" "$SELECTOR_DIR/results/selector_freeze_report.json" \
    "$SCREEN_MANIFEST"; do
    [[ -e "$path" ]] || { echo "Missing V3-R0 input: $path" >&2; exit 1; }
done
[[ "$(cat "$SCREEN_DIR/exit_code")" == 0 ]] || { echo "Screen replay is not complete" >&2; exit 1; }
[[ "$(cat "$SELECTOR_DIR/exit_code")" == 0 ]] || { echo "Selector freeze is not complete" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite R0 directory: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
sha256sum "$SCREEN_DIR/screen_rollouts_enriched.jsonl" "$SCREEN_MANIFEST" \
    "$SELECTOR_DIR/results/random_train_2000.txt" "$SELECTOR_DIR/results/tailmix_train_2000.txt" \
    "$SELECTOR_DIR/results/selector_freeze_report.json" > "$RUN_DIR/input_sha256.txt"

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
"$PYTHON" -m projects.dataset_v3.r0_geometry \
    --rollouts "$SCREEN_DIR/screen_rollouts_enriched.jsonl" \
    --screen-manifest "$SCREEN_MANIFEST" \
    --random-manifest "$SELECTOR_DIR/results/random_train_2000.txt" \
    --tailmix-manifest "$SELECTOR_DIR/results/tailmix_train_2000.txt" \
    --selector-report "$SELECTOR_DIR/results/selector_freeze_report.json" \
    --output-dir "$RUN_DIR/results"

sha256sum "$RUN_DIR/results/group_geometry.csv" "$RUN_DIR/results/r0_geometry_report.json" \
    > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
