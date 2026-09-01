#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
MODEL="$WORKSPACE_ROOT/models/sft_stage2"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap"
GPU_B_RUN="$EXPERIMENT_ROOT/formal_runs/v4_risk50_safety_g4_b4_seed20260827"
CHECKPOINT="$GPU_B_RUN/checkpoints/global_step_500"
RUN_ID=v4_risk50_safety_final_monitor_recovery_20260901
RUN_DIR="$EXPERIMENT_ROOT/technical_recovery/$RUN_ID"
TRAIN_MANIFEST="$EXPERIMENT_ROOT/training_prepare/v4_risk50_rr_aligned_prepare_20260831_r1/results/risk50_train_2000.txt"
MONITOR_MANIFEST="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap/train_monitor_256.txt"
MONITOR_PARQUET="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/hf/train_monitor.parquet"
M0="$EXPERIMENT_ROOT/protocol_freeze/v3_m0_matrix_protocol_20260829/results/m0_protocol.json"
CACHE_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap/metric_cache"
REWARD_PORT=8901

for path in "$PYTHON" "$MODEL" "$GPU_B_RUN/FAILED" "$GPU_B_RUN/rollouts.jsonl" \
    "$GPU_B_RUN/checkpoints/experiment_log.jsonl" "$CHECKPOINT/actor/model_world_size_1_rank_0.pt" \
    "$TRAIN_MANIFEST" "$MONITOR_MANIFEST" "$MONITOR_PARQUET" "$M0" \
    "$CACHE_ROOT/metadata/scene_metric_cache.csv"; do
    [[ -e "$path" ]] || { echo "Missing recovery input: $path" >&2; exit 1; }
done
[[ "$(cat "$GPU_B_RUN/exit_code")" == 1 ]] || { echo "GPU-B is not the preserved FAILED/1 run" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite recovery run: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/adas/$RUN_ID" ]] || { echo "ADAS output exists" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" ]] || { echo "Debug output exists" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || { echo "GPU is already in use" >&2; exit 1; }
[[ -z "$(fuser "$REWARD_PORT/tcp" 2>/dev/null || true)" ]] || { echo "Reward port is in use" >&2; exit 1; }
[[ "$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')" -ge 20971520 ]] || { echo "Recovery requires 20 GiB free" >&2; exit 1; }
[[ "$(grep -cve '^[[:space:]]*$' "$MONITOR_MANIFEST")" == 256 ]] || { echo "Monitor manifest count mismatch" >&2; exit 1; }

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=%s\noriginal_run=%s\noriginal_status=FAILED\noriginal_exit_code=1\ncheckpoint=%s\nmonitor_tokens=256\nevidence_phase=train_monitor_recovery\nevidence_step=500\ndev_accessed=false\nfinal_accessed=false\n' \
    "$RUN_ID" "$GPU_B_RUN" "$CHECKPOINT" > "$RUN_DIR/run.env"
sha256sum "$TRAIN_MANIFEST" "$MONITOR_MANIFEST" "$MONITOR_PARQUET" "$M0" \
    "$GPU_B_RUN/rollouts.jsonl" "$GPU_B_RUN/checkpoints/experiment_log.jsonl" > "$RUN_DIR/input_sha256.txt"
sha256sum "$MODEL"/model-*.safetensors "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$RUN_DIR/model_sha256.txt"
sha256sum "$CHECKPOINT/actor/lora_adapter/adapter_model.safetensors" \
    "$CHECKPOINT/actor/lora_adapter/adapter_config.json" > "$RUN_DIR/checkpoint_sha256.txt"

cleanup() {
    status=$?
    if [[ -n "${REWARD_SERVER_PID:-}" ]]; then kill "$REWARD_SERVER_PID" 2>/dev/null || true; wait "$REWARD_SERVER_PID" 2>/dev/null || true; fi
    "$WORKSPACE_ROOT/envs/curious/bin/ray" stop --force >/dev/null 2>&1 || true
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
    date +%s > "$RUN_DIR/end_epoch.txt"
    if [[ "$status" -ne 0 ]]; then touch "$RUN_DIR/FAILED"; fi
}
trap cleanup EXIT
exec > "$RUN_DIR/run.log" 2>&1

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
export EVIDENCE_PHASE=train_monitor_recovery
export EVIDENCE_STEP=500
export NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
cd "$EASYR1_ROOT"
"$PYTHON" -m verl.trainer.main_adas \
    config=examples/config_v3_s1_single_gpu.yaml \
    data.train_files="$MONITOR_PARQUET@train" data.val_files="$MONITOR_PARQUET@train" \
    data.image_dir="$WORKSPACE_ROOT/data" data.seed=20260827 \
    worker.actor.model.model_path="$MODEL" worker.rollout.n=1 worker.rollout.temperature=0.6 \
    worker.rollout.top_p=0.95 worker.rollout.seed=20260827 \
    worker.reward.reward_function="$EASYR1_ROOT/verl/utils/reward_score/navsim/navsim_reward_text.py:compute_score_safety_continuous" \
    trainer.experiment_name="$RUN_ID" trainer.load_checkpoint_path="$CHECKPOINT"

cp "$EASYR1_ROOT/checkpoints/adas/$RUN_ID/adas_scores.csv" "$RUN_DIR/adas_scores.csv"
mapfile -t rollout_files < <(find "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" -maxdepth 1 -type f -name 'generations_*.jsonl')
[[ "${#rollout_files[@]}" -eq 1 ]] || { echo "Expected one recovery rollout file, found ${#rollout_files[@]}" >&2; exit 1; }
cp "${rollout_files[0]}" "$RUN_DIR/rollouts.jsonl"

cd "$PROJECT_ROOT"
"$PYTHON" -m projects.dataset_v3.v4_post_training recover-monitor \
    --m0-protocol "$M0" \
    --train-manifest "$TRAIN_MANIFEST" \
    --monitor-manifest "$MONITOR_MANIFEST" \
    --original-rollouts "$GPU_B_RUN/rollouts.jsonl" \
    --recovery-rollouts "$RUN_DIR/rollouts.jsonl" \
    --original-run-log "$GPU_B_RUN/run.log" \
    --experiment-log "$GPU_B_RUN/checkpoints/experiment_log.jsonl" \
    --output "$RUN_DIR/recovery_report.json"
sha256sum "$RUN_DIR/rollouts.jsonl" "$RUN_DIR/adas_scores.csv" "$RUN_DIR/recovery_report.json" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
