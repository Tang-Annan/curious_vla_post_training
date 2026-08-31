#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
MANIFEST_ROOT="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
RAW_LOGS="$WORKSPACE_ROOT/data/navsim/navsim_logs/trainval"
SCREEN="$MANIFEST_ROOT/grpo_screen_8000.txt"
DEV_NATURAL="$MANIFEST_ROOT/dev_natural.txt"
DEV_TAIL="$MANIFEST_ROOT/dev_tail.txt"
MASTER="$MANIFEST_ROOT/master_index.csv"
STABILITY="$EXPERIMENT_ROOT/rollout_bank/v3_s1_stability_capacity_audit_20260829/results/stability_capacity.csv"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_capacity_20260831_r1"

for path in "$PYTHON" "$RAW_LOGS" "$SCREEN" "$DEV_NATURAL" "$DEV_TAIL" "$MASTER" "$STABILITY"; do
    [[ -e "$path" ]] || { echo "Missing V4 Span-risk audit input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V4 Span-risk audit run: $RUN_DIR" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_span_inspired_risk_capacity_20260831_r1\nworkers=1\ncuda_visible_devices=empty\ntrain_screen=8000\ndev_all=416\ndev_accessed=true\nfinal_accessed=false\n' > "$RUN_DIR/run.env"
sha256sum "$SCREEN" "$DEV_NATURAL" "$DEV_TAIL" "$MASTER" "$STABILITY" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    date +%s > "$RUN_DIR/end_epoch.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    rm -f "$RUN_DIR/RUNNING"
    if [[ "$status" -eq 0 ]]; then
        touch "$RUN_DIR/COMPLETE"
    else
        touch "$RUN_DIR/FAILED"
    fi
}
trap cleanup EXIT

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES= "$PYTHON" -m projects.dataset_v3.span_risk_audit \
    --raw-logs "$RAW_LOGS" \
    --master-index "$MASTER" \
    --screen-manifest "$SCREEN" \
    --dev-natural "$DEV_NATURAL" \
    --dev-tail "$DEV_TAIL" \
    --stability-capacity "$STABILITY" \
    --output-dir "$RUN_DIR/results" \
    --workers 1 \
    --seed 20260831 \
    > "$RUN_DIR/run.log" 2>&1

sha256sum "$RUN_DIR"/results/* > "$RUN_DIR/result_sha256.txt"
