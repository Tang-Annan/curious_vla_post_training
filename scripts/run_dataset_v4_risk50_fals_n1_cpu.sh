#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
SOURCE_RUN="$EXPERIMENT_ROOT/semantic_audit/v4_safety_bucket_selector_20260901"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v4_risk50_fals_n1_20260901"
SCREEN_ENRICHED="$SOURCE_RUN/screen_safety_enriched.jsonl"
LABELS="$EXPERIMENT_ROOT/semantic_audit/v4_experiment_closure_cpu_20260831_r1/results/train_current_visible_exclusive_labels.csv"
BASELINE="$EXPERIMENT_ROOT/semantic_audit/v4_risk_ratio_audit_20260831/results/frozen_current_visible_risk50_2000.txt"
RANDOM_MANIFEST="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap/random_train_2000.txt"
SCREEN_PARQUET="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/hf/grpo_screen.parquet"

for path in "$PYTHON" "$SOURCE_RUN/COMPLETE" "$SOURCE_RUN/exit_code" "$SCREEN_ENRICHED" \
    "$LABELS" "$BASELINE" "$RANDOM_MANIFEST" "$SCREEN_PARQUET"; do
    [[ -e "$path" ]] || { echo "Missing N1 input: $path" >&2; exit 1; }
done
[[ "$(cat "$SOURCE_RUN/exit_code")" == 0 ]] || { echo "Reusable rollout source did not exit cleanly" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite N1 run" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_risk50_fals_n1_20260901\nsource_run=v4_safety_bucket_selector_20260901\nreward=raw_pdms\nselector=fals\nintent_protocol=try_exact_then_remove_only_intent\ndev_accessed=false\nfinal_accessed=false\ngpu_used=false\nreward_replay=false\n' \
    > "$RUN_DIR/run.env"

cleanup() {
    status=$?
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

sha256sum "$SCREEN_ENRICHED" "$LABELS" "$BASELINE" "$RANDOM_MANIFEST" "$SCREEN_PARQUET" \
    > "$RUN_DIR/input_sha256.txt"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=
"$PYTHON" -m pytest tests/test_v4_risk50_fals.py tests/test_v4_grpo_selector.py -q \
    --basetemp="/tmp/pytest_v4_risk50_fals_n1_20260901_$$"
"$PYTHON" -m compileall -q projects/dataset_v3/v4_risk50_fals.py
bash -n scripts/run_dataset_v4_risk50_fals_n1_cpu.sh
git diff --check

"$PYTHON" -m projects.dataset_v3.v4_risk50_fals \
    --risk-labels "$LABELS" \
    --screen-enriched "$SCREEN_ENRICHED" \
    --baseline-manifest "$BASELINE" \
    --random-manifest "$RANDOM_MANIFEST" \
    --screen-parquet "$SCREEN_PARQUET" \
    --data-root "$WORKSPACE_ROOT/data" \
    --output-dir "$RUN_DIR/results"

find "$RUN_DIR/results" -maxdepth 1 -type f -print0 | sort -z | \
    xargs -0 sha256sum > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
