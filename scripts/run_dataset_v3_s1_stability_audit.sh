#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
RUN_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/rollout_bank"
RUN_DIR="$RUN_ROOT/v3_s1_stability_capacity_audit_20260829"
SCREEN_DIR="$RUN_ROOT/v3_s1_metric_replay_20260829"
CONFIRM_DIR="$RUN_ROOT/v3_s1_confirm908_g4_seed20260828"
CANDIDATE_DIR="$RUN_ROOT/v3_s1_candidate_freeze_20260829"
SCREEN_MANIFEST="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap/grpo_screen_8000.txt"
MASTER_INDEX="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap/master_index.csv"

for path in "$PYTHON" "$SCREEN_DIR/COMPLETE" "$SCREEN_DIR/screen_rollouts_enriched.jsonl" \
    "$CONFIRM_DIR/COMPLETE" "$CONFIRM_DIR/rollouts.jsonl" "$CANDIDATE_DIR/COMPLETE" \
    "$CANDIDATE_DIR/candidate_908.txt" "$SCREEN_MANIFEST" "$MASTER_INDEX"; do
    [[ -e "$path" ]] || { echo "Missing S1 stability-audit input: $path" >&2; exit 1; }
done
[[ "$(cat "$SCREEN_DIR/exit_code")" == 0 ]] || { echo "Screen replay is not complete" >&2; exit 1; }
[[ "$(cat "$CONFIRM_DIR/exit_code")" == 0 ]] || { echo "Confirm is not complete" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite audit directory: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
sha256sum "$SCREEN_DIR/screen_rollouts_enriched.jsonl" "$CONFIRM_DIR/rollouts.jsonl" \
    "$SCREEN_MANIFEST" "$CANDIDATE_DIR/candidate_908.txt" "$MASTER_INDEX" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

"$PYTHON" "$PROJECT_ROOT/projects/dataset_v3/s1_pipeline.py" audit-stability \
    --screen-rollouts "$SCREEN_DIR/screen_rollouts_enriched.jsonl" \
    --screen-manifest "$SCREEN_MANIFEST" \
    --confirm-rollouts "$CONFIRM_DIR/rollouts.jsonl" \
    --candidate-manifest "$CANDIDATE_DIR/candidate_908.txt" \
    --master-index "$MASTER_INDEX" \
    --output-dir "$RUN_DIR/results"

sha256sum "$RUN_DIR/results/stability_capacity.csv" \
    "$RUN_DIR/results/stability_capacity_report.json" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
