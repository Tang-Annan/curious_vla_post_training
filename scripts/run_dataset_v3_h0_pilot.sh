#!/usr/bin/env bash
set -euo pipefail

RUN_ID=""
LR=""
ESTIMATOR=""
GROUPS_PER_UPDATE=4
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --estimator) ESTIMATOR="$2"; shift 2 ;;
        --groups-per-update) GROUPS_PER_UPDATE="$2"; shift 2 ;;
        *) echo "Unknown H0 argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$RUN_ID" && -n "$LR" && -n "$ESTIMATOR" ]] || { echo "Missing H0 pilot argument" >&2; exit 2; }
[[ "$LR" == "1e-6" || "$LR" == "3e-6" ]] || { echo "Unsupported H0 LR: $LR" >&2; exit 2; }
[[ "$ESTIMATOR" == "grpo" || "$ESTIMATOR" == "std_floor_grpo" ]] || { echo "Unsupported estimator" >&2; exit 2; }
[[ "$GROUPS_PER_UPDATE" == 4 || "$GROUPS_PER_UPDATE" == 8 ]] || { echo "Unsupported groups/update" >&2; exit 2; }

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
MODEL="$WORKSPACE_ROOT/models/sft_stage2"
CACHE_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/metric_cache"
PREP_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/hparam_freeze/v3_h0_protocol_20260829/results"
REWARD_PROTOCOL="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/reward_freeze/v3_r0_cdt_task_freeze_20260829/results/reward_protocol.json"
RUN_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/hparam_pilots/$RUN_ID"
REWARD_PORT=8901
MAX_STEPS=$((512 / GROUPS_PER_UPDATE))
if [[ "$GROUPS_PER_UPDATE" == 4 ]]; then VAL_STEPS='[26,51,77,102,128]'; else VAL_STEPS='[13,26,38,51,64]'; fi

for path in "$PYTHON" "$MODEL" "$CACHE_ROOT/metadata/scene_metric_cache.csv" "$PREP_DIR/h0_protocol.json" \
    "$PREP_DIR/hparam_train_512.txt" "$PREP_DIR/hparam_train_512.parquet" \
    "$PREP_DIR/train_monitor_256.txt" "$PREP_DIR/train_monitor_256.parquet" "$REWARD_PROTOCOL"; do
    [[ -e "$path" ]] || { echo "Missing V3-H0 pilot input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite H0 pilot: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || { echo "GPU is already in use" >&2; exit 1; }
[[ -z "$(fuser "$REWARD_PORT/tcp" 2>/dev/null || true)" ]] || { echo "Reward port is in use" >&2; exit 1; }
[[ "$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')" -ge 31457280 ]] || { echo "H0 requires at least 30 GiB free" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" ]] || { echo "Debug output exists" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=%s\nlr=%s\nestimator=%s\ngroups_per_update=%s\nmax_steps=%s\nval_steps=%s\nseed=20260829\nrollout_n=4\nmonitor_n=1\n' \
    "$RUN_ID" "$LR" "$ESTIMATOR" "$GROUPS_PER_UPDATE" "$MAX_STEPS" "$VAL_STEPS" > "$RUN_DIR/run.env"
sha256sum "$PREP_DIR"/* "$REWARD_PROTOCOL" "$MODEL"/model-*.safetensors \
    "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$RUN_DIR/input_sha256.txt"
cp "$EASYR1_ROOT/examples/config_v3_h0_single_gpu.yaml" "$RUN_DIR/resolved_source_config.yaml"

cleanup() {
    status=$?
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi
    if [[ -n "${REWARD_SERVER_PID:-}" ]]; then kill "$REWARD_SERVER_PID" 2>/dev/null || true; wait "$REWARD_SERVER_PID" 2>/dev/null || true; fi
    "$WORKSPACE_ROOT/envs/curious/bin/ray" stop --force >/dev/null 2>&1 || true
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

printf 'timestamp,memory_used_mib,memory_free_mib,memory_total_mib,utilization_percent\n' > "$RUN_DIR/gpu_memory.csv"
(
    while [[ -e "$RUN_DIR/RUNNING" ]]; do
        printf '%s,' "$(date +%s)"
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits | tr -d ' '
        sleep 5
    done
) >> "$RUN_DIR/gpu_memory.csv" &
GPU_MONITOR_PID=$!

(
    export PROJECT_ROOT
    export DATA_ROOT="$WORKSPACE_ROOT/data/navsim"
    export OPENSCENE_DATA_ROOT="$DATA_ROOT"
    export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
    export NAVSIM_EXP_ROOT="$WORKSPACE_ROOT/exp_root"
    export CACHE_PATH="$CACHE_ROOT"
    cd "$PROJECT_ROOT/navsim_eval"
    exec "$WORKSPACE_ROOT/envs/navsim/bin/gunicorn" navsim.planning.script.run_gunicorn_server:app \
        -w 4 -k uvicorn.workers.UvicornWorker -b "127.0.0.1:$REWARD_PORT" --timeout 150
) > "$RUN_DIR/reward_server.log" 2>&1 &
REWARD_SERVER_PID=$!
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:$REWARD_PORT/ping" >/dev/null && break; sleep 2; done
curl -fsS "http://127.0.0.1:$REWARD_PORT/ping" >/dev/null
touch "$RUN_DIR/STARTED"

export EXP_NAME="$RUN_ID"
export NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
cd "$EASYR1_ROOT"
"$PYTHON" -m verl.trainer.main \
    config=examples/config_v3_h0_single_gpu.yaml \
    data.train_files="$PREP_DIR/hparam_train_512.parquet@train" \
    data.val_files="$PREP_DIR/train_monitor_256.parquet@train" \
    data.image_dir="$WORKSPACE_ROOT/data" \
    data.rollout_batch_size="$GROUPS_PER_UPDATE" data.mini_rollout_batch_size="$GROUPS_PER_UPDATE" \
    data.shuffle=true data.seed=20260829 \
    worker.actor.global_batch_size="$GROUPS_PER_UPDATE" worker.actor.optim.lr="$LR" \
    worker.actor.model.model_path="$MODEL" worker.rollout.seed=20260829 \
    worker.reward.reward_function="$EASYR1_ROOT/verl/utils/reward_score/navsim/navsim_reward_text.py:compute_score_raw_pdms" \
    algorithm.adv_estimator="$ESTIMATOR" algorithm.std_floor=0.05 \
    trainer.experiment_name="$RUN_ID" trainer.max_steps="$MAX_STEPS" trainer.val_steps="$VAL_STEPS" \
    trainer.save_checkpoint_path="$RUN_DIR/checkpoints"

mapfile -t rollout_files < <(find "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" -maxdepth 1 -type f -name 'generations_*.jsonl')
[[ "${#rollout_files[@]}" -ge 1 ]] || { echo "H0 rollout log is missing" >&2; exit 1; }
for rollout_file in "${rollout_files[@]}"; do cat "$rollout_file"; done > "$RUN_DIR/rollouts.jsonl"
cd "$PROJECT_ROOT"
"$PYTHON" -m projects.dataset_v3.h0_pipeline analyze \
    --protocol "$PREP_DIR/h0_protocol.json" \
    --hparam-manifest "$PREP_DIR/hparam_train_512.txt" \
    --monitor-manifest "$PREP_DIR/train_monitor_256.txt" \
    --experiment-config "$RUN_DIR/checkpoints/experiment_config.json" \
    --experiment-log "$RUN_DIR/checkpoints/experiment_log.jsonl" \
    --rollouts "$RUN_DIR/rollouts.jsonl" \
    --lr "$LR" --estimator "$ESTIMATOR" --groups-per-update "$GROUPS_PER_UPDATE" \
    --output "$RUN_DIR/h0_report.json"

sha256sum "$RUN_DIR/h0_report.json" "$RUN_DIR/rollouts.jsonl" \
    "$RUN_DIR/checkpoints/experiment_config.json" "$RUN_DIR/checkpoints/experiment_log.jsonl" \
    "$RUN_DIR/checkpoints/checkpoint_tracker.json" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
