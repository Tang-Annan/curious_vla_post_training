#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
MANIFEST_ROOT="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap"
RUN_DIR="$EXPERIMENT_ROOT/semantic_audit/v3_tail_semantic_alignment_20260831"
RAW_LOGS="$WORKSPACE_ROOT/data/navsim/navsim_logs/trainval"
STABILITY="$EXPERIMENT_ROOT/rollout_bank/v3_s1_stability_capacity_audit_20260829/results/stability_capacity.csv"
SELECTOR_ROOT="$EXPERIMENT_ROOT/selector_freeze/v3_s1_selector_freeze_20260829/results"
MASTER="$MANIFEST_ROOT/master_index.csv"
RANDOM_MANIFEST="$SELECTOR_ROOT/random_train_2000.txt"
TAILMIX="$SELECTOR_ROOT/tailmix_train_2000.txt"
DEV_NATURAL="$MANIFEST_ROOT/dev_natural.txt"
DEV_TAIL="$MANIFEST_ROOT/dev_tail.txt"
SFT="$EXPERIMENT_ROOT/dev_evaluation/v3_e0_sft_dev_seed20260827/results/scene_metrics.csv"
RR="$EXPERIMENT_ROOT/dev_evaluation/v3_rr_random_raw_g4_b4_seed20260827_dev_retry1/results/scene_metrics.csv"
TC="$EXPERIMENT_ROOT/dev_evaluation/v3_tc_tailmix_cdt_g4_b4_seed20260827_dev/results/scene_metrics.csv"
TR="$EXPERIMENT_ROOT/dev_evaluation/v3_tr_tailmix_raw_g4_b4_seed20260827_dev/results/scene_metrics.csv"
PPO2="$EXPERIMENT_ROOT/dev_evaluation/v3_tc_ppo2_tailmix_cdt_g4_b4_seed20260827_dev/results/scene_metrics.csv"

for path in "$PYTHON" "$RAW_LOGS" "$STABILITY" "$MASTER" "$RANDOM_MANIFEST" "$TAILMIX" \
    "$DEV_NATURAL" "$DEV_TAIL" "$SFT" "$RR" "$TC" "$TR" "$PPO2"; do
    [[ -e "$path" ]] || { echo "Missing Tail semantic audit input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite Tail semantic audit: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
if find "$EXPERIMENT_ROOT/access" -maxdepth 2 -type f -path '*/final/*' -print -quit 2>/dev/null | grep -q .; then
    echo "Final access record exists; refusing semantic audit" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=v3_tail_semantic_alignment_20260831\nworkers=1\nbootstrap_resamples=20000\ncuda_visible_devices=empty\ntrain_screen=8000\ndev_natural=210\ndev_tail=206\nfinal_accessed=false\n' > "$RUN_DIR/run.env"
sha256sum "$STABILITY" "$MASTER" "$RANDOM_MANIFEST" "$TAILMIX" "$DEV_NATURAL" "$DEV_TAIL" \
    "$SFT" "$RR" "$TC" "$TR" "$PPO2" > "$RUN_DIR/input_sha256.txt"

cleanup() {
    status=$?
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

export CUDA_VISIBLE_DEVICES=""
cd "$PROJECT_ROOT"
"$PYTHON" -m projects.dataset_v3.tail_semantic_audit \
    --raw-train-logs "$RAW_LOGS" \
    --master-index "$MASTER" \
    --stability-capacity "$STABILITY" \
    --random-manifest "$RANDOM_MANIFEST" \
    --tailmix-manifest "$TAILMIX" \
    --dev-natural "$DEV_NATURAL" \
    --dev-tail "$DEV_TAIL" \
    --dev-model "SFT=$SFT" \
    --dev-model "RR=$RR" \
    --dev-model "TC=$TC" \
    --dev-model "TR=$TR" \
    --dev-model "TC-PPO2=$PPO2" \
    --output-dir "$RUN_DIR/results" \
    --workers 1 \
    --bootstrap-resamples 20000 \
    --seed 20260831

sha256sum "$RUN_DIR/results/train_scene_features.csv" \
    "$RUN_DIR/results/dev_model_outcomes.csv" \
    "$RUN_DIR/results/tail_semantic_alignment_report.json" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
