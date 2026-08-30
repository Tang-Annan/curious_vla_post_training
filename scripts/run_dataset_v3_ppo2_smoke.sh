#!/usr/bin/env bash
set -euo pipefail

RUN_ID="v3_tc_ppo2_smoke_seed20260827"
STEPS=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        *) echo "Unknown PPO2 smoke argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$STEPS" == 2 ]] || { echo "PPO2 smoke is fixed at 2 steps" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Unsafe run id" >&2; exit 2; }

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
MODEL="$WORKSPACE_ROOT/models/sft_stage2"
SELECTOR_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/selector_freeze/v3_s1_selector_freeze_20260829/results"
DATA_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap"
CACHE_ROOT="$DATA_ROOT/metric_cache"
TRAIN_MANIFEST="$SELECTOR_ROOT/tailmix_train_2000.txt"
TRAIN_PARQUET="$SELECTOR_ROOT/tailmix_train_2000.parquet"
RUN_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/ppo2_smoke/$RUN_ID"
REWARD_PORT=8901

for path in "$PYTHON" "$MODEL" "$TRAIN_MANIFEST" "$TRAIN_PARQUET" "$CACHE_ROOT/metadata/scene_metric_cache.csv"; do
    [[ -e "$path" ]] || { echo "Missing PPO2 smoke input: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite PPO2 smoke: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" ]] || { echo "Debug output exists" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/adas/$RUN_ID" ]] || { echo "ADAS output exists" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || { echo "GPU is already in use" >&2; exit 1; }
[[ -z "$(fuser "$REWARD_PORT/tcp" 2>/dev/null || true)" ]] || { echo "Reward port is in use" >&2; exit 1; }
[[ "$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')" -ge 31457280 ]] || { echo "PPO2 smoke requires 30 GiB free" >&2; exit 1; }
[[ "$(grep -cve '^[[:space:]]*$' "$TRAIN_MANIFEST")" == 2000 ]] || { echo "Train manifest count mismatch" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=%s\ncell=V3-TC-PPO2\nppo_epochs=2\nmax_steps=%s\nseed=20260827\nrollout_n=4\ngroups_per_update=4\n' \
    "$RUN_ID" "$STEPS" > "$RUN_DIR/run.env"
sha256sum "$TRAIN_MANIFEST" "$TRAIN_PARQUET" > "$RUN_DIR/input_sha256.txt"
sha256sum "$MODEL"/model-*.safetensors "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$RUN_DIR/model_sha256.txt"
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
    data.train_files="$TRAIN_PARQUET@train" data.val_files="$TRAIN_PARQUET@train" \
    data.image_dir="$WORKSPACE_ROOT/data" data.rollout_batch_size=4 data.mini_rollout_batch_size=4 \
    data.shuffle=true data.seed=20260827 \
    worker.actor.global_batch_size=4 worker.actor.optim.lr=1e-6 \
    worker.actor.ppo_epochs=2 \
    worker.actor.model.model_path="$MODEL" worker.rollout.seed=20260827 \
    worker.reward.reward_function="$EASYR1_ROOT/verl/utils/reward_score/navsim/navsim_reward_text.py:compute_score_cdt_task" \
    algorithm.adv_estimator=grpo algorithm.std_floor=0.05 \
    trainer.experiment_name="$RUN_ID" trainer.max_steps="$STEPS" \
    trainer.val_before_train=false trainer.val_freq=-1 trainer.val_steps='[]' \
    trainer.save_freq=-1 trainer.save_model_only=true \
    trainer.save_checkpoint_path="$RUN_DIR/checkpoints"

mapfile -t rollout_files < <(find "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" -maxdepth 1 -type f -name 'generations_*.jsonl' | sort)
[[ "${#rollout_files[@]}" -ge 1 ]] || { echo "PPO2 smoke rollout log is missing" >&2; exit 1; }
for rollout_file in "${rollout_files[@]}"; do cat "$rollout_file"; done > "$RUN_DIR/rollouts.jsonl"

cd "$PROJECT_ROOT"
"$PYTHON" -m projects.dataset_v3.verify_ppo2_smoke \
    --experiment-config "$RUN_DIR/checkpoints/experiment_config.json" \
    --experiment-log "$RUN_DIR/checkpoints/experiment_log.jsonl" \
    --run-log "$RUN_DIR/run.log" \
    --output "$RUN_DIR/smoke_report.json"
sha256sum "$RUN_DIR/smoke_report.json" "$RUN_DIR/rollouts.jsonl" \
    "$RUN_DIR/checkpoints/experiment_config.json" "$RUN_DIR/checkpoints/experiment_log.jsonl" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
