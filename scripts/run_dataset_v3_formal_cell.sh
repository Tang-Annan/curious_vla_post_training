#!/usr/bin/env bash
set -euo pipefail

CELL=""
SEED=""
PPO_EPOCHS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cell) CELL="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --ppo-epochs) PPO_EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown formal cell argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$SEED" == 20260827 || "$SEED" == 20260828 || "$SEED" == 20260829 ]] || { echo "Unsupported formal seed" >&2; exit 2; }
[[ "$PPO_EPOCHS" == 1 || "$PPO_EPOCHS" == 2 ]] || { echo "Unsupported PPO epochs" >&2; exit 2; }

case "$CELL" in
    V3-RR) CELL_TAG=rr; SELECTOR=random; REWARD_TAG=raw; REWARD_FUNCTION=compute_score_raw_pdms ;;
    V3-TC) CELL_TAG=tc; SELECTOR=tailmix; REWARD_TAG=cdt; REWARD_FUNCTION=compute_score_cdt_task ;;
    V3-TR) CELL_TAG=tr; SELECTOR=tailmix; REWARD_TAG=raw; REWARD_FUNCTION=compute_score_raw_pdms ;;
    V3-RC) CELL_TAG=rc; SELECTOR=random; REWARD_TAG=cdt; REWARD_FUNCTION=compute_score_cdt_task ;;
    V3-TC-PPO2) CELL_TAG=tc_ppo2; SELECTOR=tailmix; REWARD_TAG=cdt; REWARD_FUNCTION=compute_score_cdt_task ;;
    V4-RISK50) CELL_TAG=risk50; SELECTOR=risk50; REWARD_TAG=raw; REWARD_FUNCTION=compute_score_raw_pdms ;;
    V4-RISK50-SAFETY) CELL_TAG=risk50_safety; SELECTOR=risk50; REWARD_TAG=safety; REWARD_FUNCTION=compute_score_safety_continuous ;;
    *) echo "Unsupported formal cell: $CELL" >&2; exit 2 ;;
esac
if [[ "$CELL" == "V3-TC-PPO2" && "$PPO_EPOCHS" != 2 ]]; then echo "V3-TC-PPO2 requires --ppo-epochs 2" >&2; exit 2; fi
if [[ "$CELL" != "V3-TC-PPO2" && "$PPO_EPOCHS" != 1 ]]; then echo "PPO epochs may only change for V3-TC-PPO2" >&2; exit 2; fi

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
MODEL="$WORKSPACE_ROOT/models/sft_stage2"
SELECTOR_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/selector_freeze/v3_s1_selector_freeze_20260829/results"
V4_PREP_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/training_prepare/v4_risk50_rr_aligned_prepare_20260831_r1/results"
V4_SAFETY_PREP_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/training_prepare/v4_risk50_safety_aligned_prepare_20260901/results"
DATA_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap"
CACHE_ROOT="$DATA_ROOT/metric_cache"
MANIFEST_ROOT="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap"
if [[ "$CELL" == "V4-RISK50" ]]; then
    TRAIN_MANIFEST="$V4_PREP_ROOT/risk50_train_2000.txt"
    TRAIN_PARQUET="$V4_PREP_ROOT/risk50_train_2000.parquet"
    FROZEN_CONFIG="$V4_PREP_ROOT/risk50_rr_aligned_config.json"
elif [[ "$CELL" == "V4-RISK50-SAFETY" ]]; then
    TRAIN_MANIFEST="$V4_PREP_ROOT/risk50_train_2000.txt"
    TRAIN_PARQUET="$V4_PREP_ROOT/risk50_train_2000.parquet"
    FROZEN_CONFIG="$V4_SAFETY_PREP_ROOT/risk50_safety_aligned_config.json"
else
    TRAIN_MANIFEST="$SELECTOR_ROOT/${SELECTOR}_train_2000.txt"
    TRAIN_PARQUET="$SELECTOR_ROOT/${SELECTOR}_train_2000.parquet"
    FROZEN_CONFIG=""
fi
MONITOR_MANIFEST="$MANIFEST_ROOT/train_monitor_256.txt"
MONITOR_PARQUET="$DATA_ROOT/hf/train_monitor.parquet"
M0="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/protocol_freeze/v3_m0_matrix_protocol_20260829/results/m0_protocol.json"
if [[ "$CELL" == "V4-RISK50" ]]; then
    RUN_ID="v4_risk50_raw_g4_b4_seed${SEED}"
elif [[ "$CELL" == "V4-RISK50-SAFETY" ]]; then
    RUN_ID="v4_risk50_safety_g4_b4_seed${SEED}"
else
    RUN_ID="v3_${CELL_TAG}_${SELECTOR}_${REWARD_TAG}_g4_b4_seed${SEED}"
fi
RUN_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/formal_runs/$RUN_ID"
REWARD_PORT=8901

for path in "$PYTHON" "$MODEL" "$TRAIN_MANIFEST" "$TRAIN_PARQUET" "$MONITOR_MANIFEST" "$MONITOR_PARQUET" "$M0" "$CACHE_ROOT/metadata/scene_metric_cache.csv"; do
    [[ -e "$path" ]] || { echo "Missing formal cell input: $path" >&2; exit 1; }
done
if [[ "$CELL" == "V4-RISK50" ]]; then
    [[ -e "$V4_PREP_ROOT/../COMPLETE" && -e "$FROZEN_CONFIG" ]] || { echo "V4 Risk50 preparation is incomplete" >&2; exit 1; }
    [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" == "$(cat "$V4_PREP_ROOT/../source_commit.txt")" ]] || { echo "Source changed after V4 preparation" >&2; exit 1; }
    sha256sum -c "$V4_PREP_ROOT/../result_sha256.txt" >/dev/null
    "$PYTHON" -c 'import json,sys; report=json.load(open(sys.argv[1])); assert report["status"] == "V4_RISK50_RR_ALIGNED_READY"' \
        "$V4_PREP_ROOT/v4_risk50_training_prepare_report.json"
elif [[ "$CELL" == "V4-RISK50-SAFETY" ]]; then
    GPU_A_RUN="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/formal_runs/v4_risk50_raw_g4_b4_seed${SEED}"
    [[ -e "$V4_SAFETY_PREP_ROOT/../COMPLETE" && -e "$FROZEN_CONFIG" ]] || { echo "V4 safety preparation is incomplete" >&2; exit 1; }
    [[ -e "$GPU_A_RUN/COMPLETE" && "$(cat "$GPU_A_RUN/exit_code")" == 0 ]] || { echo "GPU-A is not complete" >&2; exit 1; }
    [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" == "$(cat "$V4_SAFETY_PREP_ROOT/../source_commit.txt")" ]] || { echo "Source changed after V4 safety preparation" >&2; exit 1; }
    sha256sum -c "$V4_SAFETY_PREP_ROOT/../result_sha256.txt" >/dev/null
    "$PYTHON" -c 'import json,sys; report=json.load(open(sys.argv[1])); assert report["status"] == "V4_RISK50_SAFETY_ALIGNED_READY" and report["gpu_training_authorized"] is True' \
        "$V4_SAFETY_PREP_ROOT/v4_risk50_safety_prepare_report.json"
fi
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite formal cell: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" ]] || { echo "Debug output exists" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || { echo "GPU is already in use" >&2; exit 1; }
[[ -z "$(fuser "$REWARD_PORT/tcp" 2>/dev/null || true)" ]] || { echo "Reward port is in use" >&2; exit 1; }
[[ "$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')" -ge 31457280 ]] || { echo "Formal cell requires 30 GiB free" >&2; exit 1; }
[[ "$(grep -cve '^[[:space:]]*$' "$TRAIN_MANIFEST")" == 2000 ]] || { echo "Train manifest count mismatch" >&2; exit 1; }
[[ "$(grep -cve '^[[:space:]]*$' "$MONITOR_MANIFEST")" == 256 ]] || { echo "Monitor manifest count mismatch" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=%s\ncell=%s\nselector=%s\nreward=%s\nseed=%s\nlr=1e-6\nestimator=grpo\ngroup_size=4\ngroups_per_update=4\ntraining_groups=2000\nrollout_queries=8000\nmax_steps=500\nval_steps=0,100,200,300,400,500\nppo_epochs=%s\n' \
    "$RUN_ID" "$CELL" "$SELECTOR" "$REWARD_FUNCTION" "$SEED" "$PPO_EPOCHS" > "$RUN_DIR/run.env"
sha256sum "$TRAIN_MANIFEST" "$TRAIN_PARQUET" "$MONITOR_MANIFEST" "$MONITOR_PARQUET" "$M0" > "$RUN_DIR/input_sha256.txt"
if [[ -n "$FROZEN_CONFIG" ]]; then sha256sum "$FROZEN_CONFIG" >> "$RUN_DIR/input_sha256.txt"; fi
sha256sum "$MODEL"/model-*.safetensors "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$RUN_DIR/model_sha256.txt"
if [[ -n "$FROZEN_CONFIG" ]]; then
    cp "$FROZEN_CONFIG" "$RUN_DIR/resolved_source_config.json"
else
    cp "$EASYR1_ROOT/examples/config_v3_h0_single_gpu.yaml" "$RUN_DIR/resolved_source_config.yaml"
fi

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
if [[ "$CELL" == "V4-RISK50" || "$CELL" == "V4-RISK50-SAFETY" ]]; then
    "$PYTHON" -m verl.trainer.main config="$FROZEN_CONFIG"
else
    "$PYTHON" -m verl.trainer.main \
    config=examples/config_v3_h0_single_gpu.yaml \
    data.train_files="$TRAIN_PARQUET@train" data.val_files="$MONITOR_PARQUET@train" \
    data.image_dir="$WORKSPACE_ROOT/data" data.rollout_batch_size=4 data.mini_rollout_batch_size=4 \
    data.shuffle=true data.seed="$SEED" \
    worker.actor.global_batch_size=4 worker.actor.optim.lr=1e-6 \
    worker.actor.ppo_epochs="$PPO_EPOCHS" \
    worker.actor.model.model_path="$MODEL" worker.rollout.seed="$SEED" \
    worker.reward.reward_function="$EASYR1_ROOT/verl/utils/reward_score/navsim/navsim_reward_text.py:$REWARD_FUNCTION" \
    algorithm.adv_estimator=grpo algorithm.std_floor=0.05 \
    trainer.experiment_name="$RUN_ID" trainer.max_steps=500 trainer.val_steps='[100,200,300,400,500]' \
    trainer.save_checkpoint_path="$RUN_DIR/checkpoints"
fi

mapfile -t rollout_files < <(find "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" -maxdepth 1 -type f -name 'generations_*.jsonl')
[[ "${#rollout_files[@]}" -ge 1 ]] || { echo "Formal rollout log is missing" >&2; exit 1; }
for rollout_file in "${rollout_files[@]}"; do cat "$rollout_file"; done > "$RUN_DIR/rollouts.jsonl"
FINAL_ACTOR="$RUN_DIR/checkpoints/global_step_500/actor"
[[ -s "$FINAL_ACTOR/model_world_size_1_rank_0.pt" ]] || { echo "Final full actor state is missing" >&2; exit 1; }
[[ -s "$FINAL_ACTOR/lora_adapter/adapter_model.safetensors" ]] || { echo "Final LoRA adapter is missing" >&2; exit 1; }
[[ -s "$FINAL_ACTOR/lora_adapter/adapter_config.json" ]] || { echo "Final LoRA config is missing" >&2; exit 1; }

cd "$PROJECT_ROOT"
"$PYTHON" -m projects.dataset_v3.formal_pipeline analyze-training \
    --m0-protocol "$M0" --cell "$CELL" --seed "$SEED" --model-path "$MODEL" \
    --train-manifest "$TRAIN_MANIFEST" --monitor-manifest "$MONITOR_MANIFEST" \
    --experiment-config "$RUN_DIR/checkpoints/experiment_config.json" \
    --experiment-log "$RUN_DIR/checkpoints/experiment_log.jsonl" --rollouts "$RUN_DIR/rollouts.jsonl" \
    --output "$RUN_DIR/training_report.json"
sha256sum "$RUN_DIR/training_report.json" "$RUN_DIR/rollouts.jsonl" \
    "$RUN_DIR/checkpoints/experiment_config.json" "$RUN_DIR/checkpoints/experiment_log.jsonl" \
    "$RUN_DIR/checkpoints/checkpoint_tracker.json" "$FINAL_ACTOR/lora_adapter/adapter_model.safetensors" \
    "$FINAL_ACTOR/lora_adapter/adapter_config.json" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
