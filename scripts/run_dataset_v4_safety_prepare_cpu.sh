#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
GPU_A_RUN="$EXPERIMENT_ROOT/formal_runs/v4_risk50_raw_g4_b4_seed20260827"
REWARD_AUDIT="$EXPERIMENT_ROOT/semantic_audit/v4_reward_gate_audit_20260901"
DATA_PREP="$EXPERIMENT_ROOT/training_prepare/v4_risk50_rr_aligned_prepare_20260831_r1/results"
RUN_DIR="$EXPERIMENT_ROOT/training_prepare/v4_risk50_safety_aligned_prepare_20260901"
FUTURE_RUN="$EXPERIMENT_ROOT/formal_runs/v4_risk50_safety_g4_b4_seed20260827"

for path in "$PYTHON" "$GPU_A_RUN/COMPLETE" "$GPU_A_RUN/checkpoints/experiment_config.json" \
    "$GPU_A_RUN/training_report.json" "$GPU_A_RUN/model_sha256.txt" \
    "$REWARD_AUDIT/COMPLETE" "$REWARD_AUDIT/results/reward_audit_report.json" \
    "$DATA_PREP/risk50_train_2000.txt" "$DATA_PREP/risk50_train_2000.parquet"; do
    [[ -e "$path" ]] || { echo "Missing V4 safety preparation input: $path" >&2; exit 1; }
done
[[ "$(cat "$GPU_A_RUN/exit_code")" == 0 ]] || { echo "GPU-A did not exit successfully" >&2; exit 1; }
[[ "$(cat "$REWARD_AUDIT/exit_code")" == 0 ]] || { echo "Reward audit did not exit successfully" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite V4 safety preparation" >&2; exit 1; }
[[ ! -e "$FUTURE_RUN" ]] || { echo "Future GPU-B run already exists" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v4_risk50_safety_aligned_prepare_20260901\ngpu_a=%s\nreward_audit=%s\ndev_accessed=false\nfinal_accessed=false\n' \
    "$GPU_A_RUN" "$REWARD_AUDIT" > "$RUN_DIR/run.env"

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
export CUDA_VISIBLE_DEVICES=
sha256sum -c "$GPU_A_RUN/model_sha256.txt" > "$RUN_DIR/model_hash_check.txt"
"$PYTHON" -m pytest tests/test_v4_reward_audit.py tests/test_v4_training_prepare.py -q \
    --basetemp="/tmp/pytest_v4_safety_prepare_20260901_$$"
"$PYTHON" -m compileall -q EasyR1/verl/utils/reward_score/navsim \
    projects/dataset_v3 navsim_eval/navsim/evaluate navsim_eval/navsim/planning/simulation/planner/pdm_planner/scoring
bash -n scripts/run_dataset_v3_formal_cell.sh scripts/run_dataset_v4_reward_audit_cpu.sh \
    scripts/run_dataset_v4_safety_prepare_cpu.sh
git diff --check

"$PYTHON" -m projects.dataset_v3.v4_training_prepare prepare-safety \
    --gpu-a-config "$GPU_A_RUN/checkpoints/experiment_config.json" \
    --gpu-a-training-report "$GPU_A_RUN/training_report.json" \
    --reward-audit-report "$REWARD_AUDIT/results/reward_audit_report.json" \
    --train-manifest "$DATA_PREP/risk50_train_2000.txt" \
    --train-parquet "$DATA_PREP/risk50_train_2000.parquet" \
    --future-run-dir "$FUTURE_RUN" \
    --source-status "$RUN_DIR/source_status.txt" \
    --output-dir "$RUN_DIR/results"

"$PYTHON" -m projects.dataset_v3.v4_training_prepare smoke-loader \
    --config "$RUN_DIR/results/risk50_safety_aligned_config.json" \
    --output "$RUN_DIR/results/dataloader_smoke_report.json"

sha256sum "$RUN_DIR/results/risk50_safety_aligned_config.json" \
    "$RUN_DIR/results/v4_risk50_safety_prepare_report.json" \
    "$RUN_DIR/results/dataloader_smoke_report.json" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
