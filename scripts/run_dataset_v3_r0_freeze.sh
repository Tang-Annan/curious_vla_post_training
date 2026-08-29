#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
GEOMETRY_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/reward_freeze/v3_r0_geometry_candidates_20260829_retry1"
RUN_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/reward_freeze/v3_r0_cdt_task_freeze_20260829"

for path in "$PYTHON" "$GEOMETRY_DIR/COMPLETE" "$GEOMETRY_DIR/results/r0_geometry_report.json"; do
    [[ -e "$path" ]] || { echo "Missing V3-R0 freeze input: $path" >&2; exit 1; }
done
[[ "$(cat "$GEOMETRY_DIR/exit_code")" == 0 ]] || { echo "R0 geometry is not complete" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite R0 freeze directory: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
sha256sum "$GEOMETRY_DIR/results/r0_geometry_report.json" > "$RUN_DIR/input_sha256.txt"

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
"$PYTHON" -m projects.dataset_v3.r0_freeze \
    --geometry-report "$GEOMETRY_DIR/results/r0_geometry_report.json" \
    --output "$RUN_DIR/results/reward_protocol.json"

sha256sum "$RUN_DIR/results/reward_protocol.json" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
