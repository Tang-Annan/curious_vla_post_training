#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
MANIFEST_ROOT="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap"
DATASET_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
RAW_LOGS="$WORKSPACE_ROOT/data/navsim/navsim_logs/trainval"
MASTER="$MANIFEST_ROOT/master_index.csv"
SCREEN_MANIFEST="$MANIFEST_ROOT/grpo_screen_8000.txt"
MONITOR_MANIFEST="$MANIFEST_ROOT/train_monitor_256.txt"
SCENE_LABELS="$EXPERIMENT_ROOT/semantic_audit/v4_span_inspired_risk_capacity_20260831_r1/results/train_scene_labels.csv"
SCREEN_ENRICHED="$EXPERIMENT_ROOT/rollout_bank/v3_s1_metric_replay_20260829/screen_rollouts_enriched.jsonl"
SCREEN_PARQUET="$DATASET_ROOT/hf/grpo_screen.parquet"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v5_risk_fals_datasets_20260904_r1"

for path in "$PYTHON" "$RAW_LOGS" "$MASTER" "$SCREEN_MANIFEST" "$MONITOR_MANIFEST" \
    "$SCENE_LABELS" "$SCREEN_ENRICHED" "$SCREEN_PARQUET" "$WORKSPACE_ROOT/data"; do
    [[ -e "$path" ]] || { echo "Missing V5 dataset input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V5 dataset run: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v5_risk_fals_datasets_20260904_r1\nworkers=1\ncuda_visible_devices=empty\ntrain_screen=8000\nreward_source=frozen_sft_g4_raw_pdms\ndev_accessed=false\nfinal_accessed=false\ngpu_used=false\ntraining_launched=false\n' \
    > "$RUN_DIR/run.env"

cleanup() {
    status=$?
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -eq 0 ]]; then touch "$RUN_DIR/COMPLETE"; else touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

sha256sum "$MASTER" "$SCREEN_MANIFEST" "$MONITOR_MANIFEST" "$SCENE_LABELS" \
    "$SCREEN_ENRICHED" "$SCREEN_PARQUET" > "$RUN_DIR/input_sha256.txt"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=
"$PYTHON" -m pytest tests/test_v5_risk_fals_datasets.py -q \
    --basetemp="/tmp/pytest_v5_risk_fals_20260904_$$"
"$PYTHON" -m compileall -q projects/dataset_v3/v5_risk_fals_datasets.py
bash -n scripts/run_dataset_v5_risk_fals_prepare_cpu.sh
git diff --check

"$PYTHON" -m projects.dataset_v3.v5_risk_fals_datasets \
    --raw-logs "$RAW_LOGS" \
    --master-index "$MASTER" \
    --screen-manifest "$SCREEN_MANIFEST" \
    --monitor-manifest "$MONITOR_MANIFEST" \
    --scene-labels "$SCENE_LABELS" \
    --screen-enriched "$SCREEN_ENRICHED" \
    --screen-parquet "$SCREEN_PARQUET" \
    --data-root "$WORKSPACE_ROOT/data" \
    --output-dir "$RUN_DIR/results" \
    --workers 1

find "$RUN_DIR/results" -maxdepth 1 -type f -print0 | sort -z | \
    xargs -0 sha256sum > "$RUN_DIR/result_sha256.txt"
