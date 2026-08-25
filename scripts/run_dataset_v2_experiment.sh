#!/usr/bin/env bash
set -euo pipefail

STAGE=""
RUN_ID=""
PYTHON=""
WORKSPACE_ROOT=""
PROJECT_ROOT=""
DATA_ROOT=""
DATASET_DIR=""
TRAIN_PARQUET=""
DEV_PARQUET=""
MANIFEST_DIR=""
CACHE_MANIFEST=""
CACHE_DIR=""
FINAL_MANIFEST=""
STAGE2_MODEL=""
EXPERIMENT_ROOT=""
ACTIVE_PARQUET=""
ACTIVE_MANIFEST=""
MODEL_PATH=""
LOAD_CHECKPOINT=""
REWARD_FUNCTION=""
ROLLOUT_N=""
SEED=""
TEMPERATURE=""
TOP_P=""
MAX_STEPS=""
SAVE_FREQ=""
SAVE_LIMIT=""
SAVE_MODEL_ONLY="false"
SKIP_FINAL_VALIDATION="false"
REWARD_SERVER_PORT=8901

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --workspace-root) WORKSPACE_ROOT="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --train-parquet) TRAIN_PARQUET="$2"; shift 2 ;;
        --dev-parquet) DEV_PARQUET="$2"; shift 2 ;;
        --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
        --cache-manifest) CACHE_MANIFEST="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --final-manifest) FINAL_MANIFEST="$2"; shift 2 ;;
        --stage2-model) STAGE2_MODEL="$2"; shift 2 ;;
        --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
        --active-parquet) ACTIVE_PARQUET="$2"; shift 2 ;;
        --active-manifest) ACTIVE_MANIFEST="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --load-checkpoint) LOAD_CHECKPOINT="$2"; shift 2 ;;
        --reward-function) REWARD_FUNCTION="$2"; shift 2 ;;
        --rollout-n) ROLLOUT_N="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --save-limit) SAVE_LIMIT="$2"; shift 2 ;;
        --save-model-only) SAVE_MODEL_ONLY="$2"; shift 2 ;;
        --skip-final-validation) SKIP_FINAL_VALIDATION="$2"; shift 2 ;;
        --reward-server-port) REWARD_SERVER_PORT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

for value in STAGE RUN_ID PYTHON WORKSPACE_ROOT PROJECT_ROOT DATA_ROOT DATASET_DIR TRAIN_PARQUET DEV_PARQUET \
    MANIFEST_DIR CACHE_MANIFEST CACHE_DIR FINAL_MANIFEST STAGE2_MODEL EXPERIMENT_ROOT; do
    [[ -n "${!value}" ]] || { echo "Missing required argument for $value" >&2; exit 2; }
done
[[ "$STAGE" =~ ^(d0|rollout|train)$ ]] || { echo "Unsupported Dataset V2 stage: $STAGE" >&2; exit 2; }

for path in "$PYTHON" "$WORKSPACE_ROOT" "$PROJECT_ROOT" "$DATA_ROOT" "$DATASET_DIR" "$TRAIN_PARQUET" \
    "$DEV_PARQUET" "$MANIFEST_DIR" "$CACHE_MANIFEST" "$CACHE_DIR" "$CACHE_DIR/metadata" \
    "$FINAL_MANIFEST" "$STAGE2_MODEL"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

RUN_DIR="$EXPERIMENT_ROOT/$RUN_ID"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite run directory: $RUN_DIR" >&2; exit 1; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
AVAILABLE_KB=$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')
[[ "$AVAILABLE_KB" -ge 26214400 ]] || { echo "Dataset V2 requires at least 25 GiB free" >&2; exit 1; }
SOURCE_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

if [[ "$STAGE" == d0 ]]; then
    [[ ! -e "$DATASET_DIR/V2_DATA_FROZEN" ]] || { echo "Dataset V2 is already frozen" >&2; exit 1; }
else
    for value in ACTIVE_PARQUET ACTIVE_MANIFEST MODEL_PATH REWARD_FUNCTION ROLLOUT_N SEED TEMPERATURE TOP_P; do
        [[ -n "${!value}" ]] || { echo "Missing required argument for $value" >&2; exit 2; }
    done
    for path in "$ACTIVE_PARQUET" "$ACTIVE_MANIFEST" "$MODEL_PATH" "$DATASET_DIR/V2_DATA_FROZEN"; do
        [[ -e "$path" ]] || { echo "Missing required run input: $path" >&2; exit 1; }
    done
    if [[ -n "$LOAD_CHECKPOINT" ]]; then
        [[ -d "$LOAD_CHECKPOINT/actor" ]] || { echo "Invalid load checkpoint: $LOAD_CHECKPOINT" >&2; exit 1; }
    fi
    "$PYTHON" - "$DATASET_DIR/V2_DATA_FROZEN" "$SOURCE_COMMIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    frozen = json.load(handle)["source_commit"]
if frozen != sys.argv[2]:
    raise SystemExit(f"Source commit differs from V2-D0 freeze: frozen={frozen} current={sys.argv[2]}")
PY
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || {
        echo "GPU is already in use" >&2
        exit 1
    }
    [[ -z "$(fuser "$REWARD_SERVER_PORT/tcp" 2>/dev/null || true)" ]] || {
        echo "Reward server port is already in use: $REWARD_SERVER_PORT" >&2
        exit 1
    }
    [[ ! -e "$EASYR1_ROOT/checkpoints/debug/$RUN_ID" ]] || { echo "Debug output already exists" >&2; exit 1; }
    [[ ! -e "$EASYR1_ROOT/checkpoints/adas/$RUN_ID" ]] || { echo "ADAS output already exists" >&2; exit 1; }
fi

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
printf 'stage=%s\nrun_id=%s\nsource_commit=%s\ntrain_parquet=%s\ndev_parquet=%s\nactive_parquet=%s\nactive_manifest=%s\ncache_manifest=%s\ncache_dir=%s\nmodel_path=%s\nreward_function=%s\nrollout_n=%s\nseed=%s\ntemperature=%s\ntop_p=%s\nmax_steps=%s\nexperiment_root=%s\n' \
    "$STAGE" "$RUN_ID" "$SOURCE_COMMIT" "$TRAIN_PARQUET" "$DEV_PARQUET" "$ACTIVE_PARQUET" \
    "$ACTIVE_MANIFEST" "$CACHE_MANIFEST" "$CACHE_DIR" "$MODEL_PATH" "$REWARD_FUNCTION" "$ROLLOUT_N" \
    "$SEED" "$TEMPERATURE" "$TOP_P" "$MAX_STEPS" "$EXPERIMENT_ROOT" > "$RUN_DIR/run.env"
printf 'load_checkpoint=%s\n' "$LOAD_CHECKPOINT" >> "$RUN_DIR/run.env"

cleanup() {
    status=$?
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    if [[ -n "${REWARD_SERVER_PID:-}" ]]; then
        kill "$REWARD_SERVER_PID" 2>/dev/null || true
        wait "$REWARD_SERVER_PID" 2>/dev/null || true
    fi
    if [[ "$STAGE" != d0 ]] && command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
    fi
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
}
trap cleanup EXIT

if [[ "$STAGE" == d0 ]]; then
    "$PYTHON" "$PROJECT_ROOT/projects/dataset_v2/freeze_dataset_v2.py" \
        --data-root "$DATA_ROOT" \
        --dataset-dir "$DATASET_DIR" \
        --manifest-dir "$MANIFEST_DIR" \
        --train-parquet "$TRAIN_PARQUET" \
        --dev-parquet "$DEV_PARQUET" \
        --cache-manifest "$CACHE_MANIFEST" \
        --cache-dir "$CACHE_DIR" \
        --final-manifest "$FINAL_MANIFEST" \
        --stage2-model "$STAGE2_MODEL" \
        --source-commit "$SOURCE_COMMIT" \
        --output "$RUN_DIR/freeze_report.json" \
        > "$RUN_DIR/freeze_report.stdout.json"
    touch "$RUN_DIR/COMPLETE"
    exit 0
fi

cp "$ACTIVE_MANIFEST" "$RUN_DIR/active_tokens.txt"
cp "$MANIFEST_DIR/dev_2000.txt" "$RUN_DIR/dev_tokens.txt"
sha256sum "$ACTIVE_PARQUET" "$ACTIVE_MANIFEST" "$DEV_PARQUET" "$MANIFEST_DIR/dev_2000.txt" \
    "$CACHE_MANIFEST" > "$RUN_DIR/input_sha256.txt"
sha256sum "$MODEL_PATH"/model-*.safetensors "$MODEL_PATH/config.json" \
    "$MODEL_PATH/model.safetensors.index.json" > "$RUN_DIR/model_sha256.txt"
printf 'timestamp,memory_used_mib,memory_free_mib,utilization_percent\n' > "$RUN_DIR/gpu_memory.csv"
(
    while [[ -e "$RUN_DIR/RUNNING" ]]; do
        printf '%s,' "$(date +%s)"
        nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits | \
            awk -F, '{print $1 "," $2 "," $4}'
        sleep 1
    done
) >> "$RUN_DIR/gpu_memory.csv" &
GPU_MONITOR_PID=$!

(
    export PROJECT_ROOT DATA_ROOT="$PROJECT_ROOT/datasets/navsim" CACHE_PATH="$CACHE_DIR"
    export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
    export NAVSIM_EXP_ROOT="$WORKSPACE_ROOT/exp_root"
    cd "$PROJECT_ROOT/navsim_eval"
    exec "$WORKSPACE_ROOT/envs/navsim/bin/gunicorn" \
        navsim.planning.script.run_gunicorn_server:app \
        -w 1 -k uvicorn.workers.UvicornWorker \
        -b "127.0.0.1:$REWARD_SERVER_PORT" --timeout 150
) > "$RUN_DIR/reward_server.log" 2>&1 &
REWARD_SERVER_PID=$!

for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$REWARD_SERVER_PORT/ping" >/dev/null && break
    sleep 2
done
curl -fsS "http://127.0.0.1:$REWARD_SERVER_PORT/ping" >/dev/null

export EXP_NAME="$RUN_ID"
export NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
export NAVSIM_REWARD_URL="http://127.0.0.1:$REWARD_SERVER_PORT"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
cd "$EASYR1_ROOT"

COMMON_ARGS=(
    config=examples/config_vla_single_gpu_lora.yaml
    data.image_dir="$DATA_ROOT"
    data.max_response_length=512
    worker.actor.model.model_path="$MODEL_PATH"
    worker.reward.reward_function="$REWARD_FUNCTION"
    worker.rollout.n="$ROLLOUT_N"
    worker.rollout.temperature="$TEMPERATURE"
    worker.rollout.top_p="$TOP_P"
    worker.rollout.seed="$SEED"
    worker.rollout.enforce_eager=false
    worker.rollout.gpu_memory_utilization=0.55
    worker.rollout.max_num_batched_tokens=4608
    worker.rollout.disable_tqdm=true
    trainer.find_last_checkpoint=false
    trainer.experiment_name="$RUN_ID"
)
LOAD_ARGS=()
if [[ -n "$LOAD_CHECKPOINT" ]]; then
    LOAD_ARGS=(trainer.load_checkpoint_path="$LOAD_CHECKPOINT")
fi

if [[ "$STAGE" == rollout ]]; then
    "$PYTHON" -m verl.trainer.main_adas \
        "${COMMON_ARGS[@]}" \
        "${LOAD_ARGS[@]}" \
        data.train_files="$ACTIVE_PARQUET@train" \
        data.val_files="$ACTIVE_PARQUET@train" \
        data.token_filter_file="$ACTIVE_MANIFEST" \
        data.val_token_filter_file="$ACTIVE_MANIFEST" \
        data.shuffle=false \
        trainer.save_checkpoint_path="$RUN_DIR/tracker"
    cp "checkpoints/adas/$RUN_ID/adas_scores.csv" "$RUN_DIR/adas_scores.csv"
    mapfile -t rollout_files < <(find "checkpoints/debug/$RUN_ID" -maxdepth 1 -name 'generations_*.jsonl' -type f)
    [[ ${#rollout_files[@]} -eq 1 ]] || { echo "Expected one rollout file, found ${#rollout_files[@]}" >&2; exit 1; }
    cp "${rollout_files[0]}" "$RUN_DIR/rollouts.jsonl"
    "$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
        "$RUN_DIR/rollouts.jsonl" --manifest "$ACTIVE_MANIFEST" --expected-rollouts "$ROLLOUT_N" \
        > "$RUN_DIR/diagnosis.json"
else
    for value in MAX_STEPS SAVE_FREQ SAVE_LIMIT; do
        [[ -n "${!value}" ]] || { echo "Missing required train argument for $value" >&2; exit 2; }
    done
    [[ "$ROLLOUT_N" -eq 4 ]] || { echo "Dataset V2 GRPO training requires G=4" >&2; exit 1; }
    "$PYTHON" -m verl.trainer.main \
        "${COMMON_ARGS[@]}" \
        data.train_files="$TRAIN_PARQUET@train" \
        data.val_files="$DEV_PARQUET@train" \
        data.token_filter_file="$ACTIVE_MANIFEST" \
        data.val_token_filter_file="$MANIFEST_DIR/dev_2000.txt" \
        algorithm.adv_estimator=grpo \
        trainer.max_steps="$MAX_STEPS" \
        trainer.max_try_make_batch=20 \
        trainer.val_before_train=false \
        trainer.val_freq=-1 \
        trainer.skip_final_validation="$SKIP_FINAL_VALIDATION" \
        trainer.save_freq="$SAVE_FREQ" \
        trainer.save_limit="$SAVE_LIMIT" \
        trainer.save_model_only="$SAVE_MODEL_ONLY" \
        trainer.save_checkpoint_path="$RUN_DIR/checkpoints"
    "$PYTHON" - "$RUN_DIR/checkpoints/checkpoint_tracker.json" "$MAX_STEPS" <<'PY'
import json
import pathlib
import sys

tracker = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(sys.argv[2])
if tracker.get("last_global_step") != step or not (pathlib.Path(sys.argv[1]).parent / f"global_step_{step}" / "actor").is_dir():
    raise SystemExit("Final checkpoint coverage mismatch")
PY
    mapfile -t rollout_files < <(find "checkpoints/debug/$RUN_ID" -maxdepth 1 -name 'generations_*.jsonl' -type f | sort)
    [[ ${#rollout_files[@]} -ge 1 ]] || { echo "Training rollout log is missing" >&2; exit 1; }
    for rollout_file in "${rollout_files[@]}"; do cat "$rollout_file"; done > "$RUN_DIR/raw_rollouts.jsonl"
    if [[ "$SKIP_FINAL_VALIDATION" == true ]]; then
        cp "$RUN_DIR/raw_rollouts.jsonl" "$RUN_DIR/train_rollouts.jsonl"
        "$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
            "$RUN_DIR/train_rollouts.jsonl" --manifest "$ACTIVE_MANIFEST" --expected-rollouts 4 \
            > "$RUN_DIR/train_diagnosis.json"
    else
        "$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/split_rollouts.py" \
            "$RUN_DIR/raw_rollouts.jsonl" \
            --train-manifest "$ACTIVE_MANIFEST" \
            --dev-manifest "$MANIFEST_DIR/dev_2000.txt" \
            --train-output "$RUN_DIR/train_rollouts.jsonl" \
            --dev-output "$RUN_DIR/dev_rollouts.jsonl" \
            --expected-train-rollouts 4 \
            --expected-dev-rollouts 1 > "$RUN_DIR/rollout_coverage.json"
        "$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
            "$RUN_DIR/train_rollouts.jsonl" --manifest "$ACTIVE_MANIFEST" --expected-rollouts 4 \
            > "$RUN_DIR/train_diagnosis.json"
        "$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
            "$RUN_DIR/dev_rollouts.jsonl" --manifest "$MANIFEST_DIR/dev_2000.txt" --expected-rollouts 1 \
            > "$RUN_DIR/final_dev_metrics.json"
    fi
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    unset GPU_MONITOR_PID
    "$PYTHON" "$PROJECT_ROOT/projects/safe_grpo/export_training_evidence.py" \
        --experiment-log "$RUN_DIR/checkpoints/experiment_log.jsonl" \
        --gpu-memory "$RUN_DIR/gpu_memory.csv" \
        --train-rollouts "$RUN_DIR/train_rollouts.jsonl" \
        --output-dir "$RUN_DIR/training_evidence" > "$RUN_DIR/training_evidence.stdout.json"
    mapfile -t tensorboard_events < <(find "$RUN_DIR/tensorboard" -type f -name 'events.out.tfevents.*')
    [[ ${#tensorboard_events[@]} -ge 1 ]] || { echo "TensorBoard event file is missing" >&2; exit 1; }
    VERIFY_ARGS=()
    if [[ "$SKIP_FINAL_VALIDATION" != true ]]; then
        VERIFY_ARGS=(--dev-rollouts "$RUN_DIR/dev_rollouts.jsonl" --dev-manifest "$MANIFEST_DIR/dev_2000.txt")
    fi
    "$PYTHON" "$PROJECT_ROOT/projects/dataset_v2/experiment_pipeline.py" verify-train \
        --train-rollouts "$RUN_DIR/train_rollouts.jsonl" \
        --manifest "$ACTIVE_MANIFEST" \
        --training-log "$RUN_DIR/checkpoints/experiment_log.jsonl" \
        --tensorboard-dir "$RUN_DIR/tensorboard" \
        --expected-steps "$MAX_STEPS" \
        --output "$RUN_DIR/technical_report.json" \
        "${VERIFY_ARGS[@]}"
fi

touch "$RUN_DIR/COMPLETE"
