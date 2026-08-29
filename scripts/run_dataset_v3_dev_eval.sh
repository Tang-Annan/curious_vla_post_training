#!/usr/bin/env bash
set -euo pipefail

RUN_ID=""
MODEL_ID=""
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown Dev evaluation argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$RUN_ID" && -n "$MODEL_ID" ]] || { echo "Missing Dev evaluation argument" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Unsafe run id" >&2; exit 2; }

WORKSPACE_ROOT=/root/autodl-tmp/curious-vla-workspace
PROJECT_ROOT="$WORKSPACE_ROOT/src/curious_vla_v3"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
PYTHON="$WORKSPACE_ROOT/envs/curious/bin/python"
MODEL="$WORKSPACE_ROOT/models/sft_stage2"
DATA_ROOT="$WORKSPACE_ROOT/data/dataset_v3_controlled_overlap"
MANIFEST_ROOT="$WORKSPACE_ROOT/manifests/dataset_v3_controlled_overlap"
CACHE_ROOT="$DATA_ROOT/metric_cache"
PARQUET="$DATA_ROOT/hf/dev.parquet"
NATURAL_MANIFEST="$MANIFEST_ROOT/dev_natural.txt"
TAIL_MANIFEST="$MANIFEST_ROOT/dev_tail.txt"
MASTER_INDEX="$MANIFEST_ROOT/master_index.csv"
M0="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/protocol_freeze/v3_m0_matrix_protocol_20260829/results/m0_protocol.json"
RUN_ROOT="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/dev_evaluation"
RUN_DIR="$RUN_ROOT/$RUN_ID"
ACCESS_DIR="$WORKSPACE_ROOT/experiments/dataset_v3_controlled_overlap/access/dev"
ACCESS_RECORD="$ACCESS_DIR/$RUN_ID.json"
REWARD_PORT=8901
EVAL_SEED=20260827

for path in "$PYTHON" "$MODEL" "$PARQUET" "$NATURAL_MANIFEST" "$TAIL_MANIFEST" "$MASTER_INDEX" "$M0" "$CACHE_ROOT/metadata/scene_metric_cache.csv"; do
    [[ -e "$path" ]] || { echo "Missing Dev evaluation input: $path" >&2; exit 1; }
done
if [[ -n "$CHECKPOINT" ]]; then [[ -d "$CHECKPOINT/actor" ]] || { echo "Checkpoint actor is missing" >&2; exit 1; }; fi
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite Dev evaluation: $RUN_DIR" >&2; exit 1; }
[[ ! -e "$ACCESS_RECORD" ]] || { echo "Dev access record already exists: $ACCESS_RECORD" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/adas/$RUN_ID" ]] || { echo "ADAS output exists" >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" ]] || { echo "Debug output exists" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || { echo "GPU is already in use" >&2; exit 1; }
[[ -z "$(fuser "$REWARD_PORT/tcp" 2>/dev/null || true)" ]] || { echo "Reward port is in use" >&2; exit 1; }
[[ "$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')" -ge 20971520 ]] || { echo "Dev evaluation requires 20 GiB free" >&2; exit 1; }
[[ "$(grep -cve '^[[:space:]]*$' "$NATURAL_MANIFEST")" == 210 ]] || { echo "Natural Dev count mismatch" >&2; exit 1; }
[[ "$(grep -cve '^[[:space:]]*$' "$TAIL_MANIFEST")" == 206 ]] || { echo "Tail Dev count mismatch" >&2; exit 1; }

mkdir -p "$RUN_DIR" "$ACCESS_DIR"
touch "$RUN_DIR/RUNNING"
date +%s > "$RUN_DIR/start_epoch.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'run_id=%s\nmodel_id=%s\nmodel=%s\ncheckpoint=%s\nevaluation_seed=%s\nrollout_n=1\ntemperature=0.6\ntop_p=0.95\nmax_response_length=512\n' \
    "$RUN_ID" "$MODEL_ID" "$MODEL" "$CHECKPOINT" "$EVAL_SEED" > "$RUN_DIR/run.env"
sha256sum "$PARQUET" "$NATURAL_MANIFEST" "$TAIL_MANIFEST" "$MASTER_INDEX" "$M0" > "$RUN_DIR/input_sha256.txt"
sha256sum "$MODEL"/model-*.safetensors "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$RUN_DIR/model_sha256.txt"
if [[ -n "$CHECKPOINT" ]]; then find "$CHECKPOINT/actor" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/checkpoint_sha256.txt"; fi
cp "$EASYR1_ROOT/examples/config_v3_s1_single_gpu.yaml" "$RUN_DIR/resolved_source_config.yaml"

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

"$PYTHON" - "$ACCESS_RECORD" "$RUN_ID" "$MODEL_ID" "$CHECKPOINT" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
record = {"run_id": sys.argv[2], "model_id": sys.argv[3], "checkpoint": sys.argv[4] or None, "split": "dev", "access_epoch": int(time.time()), "final_accessed": False}
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
touch "$RUN_DIR/STARTED"

export EXP_NAME="$RUN_ID"
export EVIDENCE_PHASE=dev_eval
export EVIDENCE_STEP=0
export NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
LOAD_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then LOAD_ARGS+=(trainer.load_checkpoint_path="$CHECKPOINT"); fi
cd "$EASYR1_ROOT"
"$PYTHON" -m verl.trainer.main_adas \
    config=examples/config_v3_s1_single_gpu.yaml \
    data.train_files="$PARQUET@train" data.val_files="$PARQUET@train" \
    data.image_dir="$WORKSPACE_ROOT/data" data.seed="$EVAL_SEED" \
    worker.actor.model.model_path="$MODEL" worker.rollout.n=1 worker.rollout.temperature=0.6 \
    worker.rollout.top_p=0.95 worker.rollout.seed="$EVAL_SEED" \
    worker.reward.reward_function="$EASYR1_ROOT/verl/utils/reward_score/navsim/navsim_reward_text.py:compute_score_fast" \
    trainer.experiment_name="$RUN_ID" "${LOAD_ARGS[@]}"

cp "$EASYR1_ROOT/checkpoints/adas/$RUN_ID/adas_scores.csv" "$RUN_DIR/adas_scores.csv"
mapfile -t rollout_files < <(find "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" -maxdepth 1 -type f -name 'generations_*.jsonl')
[[ "${#rollout_files[@]}" -eq 1 ]] || { echo "Expected one Dev rollout file, found ${#rollout_files[@]}" >&2; exit 1; }
cp "${rollout_files[0]}" "$RUN_DIR/rollouts.jsonl"

cd "$PROJECT_ROOT"
"$PYTHON" -m projects.dataset_v3.eval_pipeline \
    --rollouts "$RUN_DIR/rollouts.jsonl" --master-index "$MASTER_INDEX" \
    --natural-manifest "$NATURAL_MANIFEST" --tail-manifest "$TAIL_MANIFEST" \
    --m0-protocol "$M0" --run-id "$RUN_ID" --model-id "$MODEL_ID" \
    --evaluation-seed "$EVAL_SEED" --output-dir "$RUN_DIR/results"
sha256sum "$RUN_DIR/rollouts.jsonl" "$RUN_DIR/adas_scores.csv" "$RUN_DIR/results/scene_metrics.csv" \
    "$RUN_DIR/results/eval_summary.json" "$RUN_DIR/results/representative_examples.json" > "$RUN_DIR/result_sha256.txt"
touch "$RUN_DIR/COMPLETE"
