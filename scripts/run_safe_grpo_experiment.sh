#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?Usage: run_safe_grpo_experiment.sh <e0|d0|e1>}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
MODEL_PATH="$WORKSPACE_ROOT/models/sft_stage2"
DATA_PATH="$EASYR1_ROOT/data/QA_navtrain_poutine_style_full"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
DEV_MANIFEST="$WORKSPACE_ROOT/manifests/dev_tokens.txt"
ITERATION_MANIFEST="$WORKSPACE_ROOT/manifests/dev_subsets/train_seed20260812_1000.txt"
CACHE_PATH="$WORKSPACE_ROOT/exp_root/metric_cache_released_5656"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/safe_grpo"
SEED=20260812
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8901}"
REWARD_SERVER_WORKERS="${REWARD_SERVER_WORKERS:-4}"
NAVSIM_REWARD_CONCURRENCY="${NAVSIM_REWARD_CONCURRENCY:-4}"
REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_group_fast

case "$STAGE" in
    e0)
        EXP_NAME=e0_stage2_dev_seed20260812
        ;;
    d0)
        EXP_NAME=d0_stage2_train_n4_seed20260812
        ;;
    e1)
        EXP_NAME=e1_vanilla_lora_1k_seed20260812
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        exit 2
        ;;
esac

RUN_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
for path in "$MODEL_PATH" "$DATA_PATH" "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$CACHE_PATH/metadata"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
if [[ "$STAGE" == e1 ]]; then
    [[ -e "$ITERATION_MANIFEST" ]] || { echo "Missing required path: $ITERATION_MANIFEST" >&2; exit 1; }
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite experiment directory: $RUN_DIR" >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU is already in use." >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
[[ ! -s "$RUN_DIR/source_status.txt" ]] || { echo "Source checkout is dirty." >&2; exit 1; }
cp "$DEV_MANIFEST" "$RUN_DIR/dev_tokens.txt"
if [[ "$STAGE" == d0 ]]; then
    cp "$TRAIN_MANIFEST" "$RUN_DIR/train_tokens.txt"
elif [[ "$STAGE" == e1 ]]; then
    cp "$ITERATION_MANIFEST" "$RUN_DIR/train_tokens.txt"
fi
printf 'stage=%s\nexperiment=%s\nseed=%s\nreward_server_workers=%s\nreward_concurrency=%s\n' \
    "$STAGE" "$EXP_NAME" "$SEED" "$REWARD_SERVER_WORKERS" "$NAVSIM_REWARD_CONCURRENCY" > "$RUN_DIR/run.env"
exec > "$RUN_DIR/run.log" 2>&1

cleanup() {
    status=$?
    if [[ -n "${REWARD_SERVER_PID:-}" ]]; then
        kill "$REWARD_SERVER_PID" 2>/dev/null || true
        wait "$REWARD_SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$RUN_DIR/RUNNING"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code"
}
trap cleanup EXIT

if [[ "$STAGE" == e0 ]]; then
    rm -rf "$EASYR1_ROOT/checkpoints/debug/$EXP_NAME"
elif [[ "$STAGE" == d0 ]]; then
    rm -rf "$EASYR1_ROOT/checkpoints/debug/$EXP_NAME" "$EASYR1_ROOT/checkpoints/adas/$EXP_NAME"
else
    rm -rf "$EASYR1_ROOT/checkpoints/debug/$EXP_NAME"
fi

(
    export PROJECT_ROOT DATA_ROOT="$PROJECT_ROOT/datasets/navsim" CACHE_PATH REWARD_SERVER_PORT
    export OPENSCENE_DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
    export NAVSIM_EXP_ROOT="$WORKSPACE_ROOT/exp_root"
    cd "$PROJECT_ROOT/navsim_eval"
    exec "$WORKSPACE_ROOT/envs/navsim/bin/gunicorn" \
        navsim.planning.script.run_gunicorn_server:app \
        -w "$REWARD_SERVER_WORKERS" -k uvicorn.workers.UvicornWorker \
        -b "127.0.0.1:$REWARD_SERVER_PORT" --timeout 150
) > "$RUN_DIR/reward_server.log" 2>&1 &
REWARD_SERVER_PID=$!

for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$REWARD_SERVER_PORT/ping" >/dev/null && break
    sleep 2
done
curl -fsS "http://127.0.0.1:$REWARD_SERVER_PORT/ping" >/dev/null

export EXP_NAME NAVSIM_REWARD_CONCURRENCY NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
export NAVSIM_REWARD_URL="http://127.0.0.1:$REWARD_SERVER_PORT"
cd "$EASYR1_ROOT"

COMMON_ARGS=(
    config=examples/config_vla_single_gpu_lora.yaml
    data.train_files="$DATA_PATH@train"
    data.val_files="$DATA_PATH@train"
    data.image_dir="$PROJECT_ROOT/datasets"
    data.val_token_filter_file="$DEV_MANIFEST"
    data.max_response_length=512
    worker.actor.model.model_path="$MODEL_PATH"
    worker.reward.reward_function="$REWARD_FUNCTION"
    worker.rollout.seed="$SEED"
    worker.rollout.enforce_eager=false
    worker.rollout.gpu_memory_utilization=0.55
    worker.rollout.max_num_batched_tokens=4608
    worker.rollout.disable_tqdm=true
    trainer.find_last_checkpoint=false
    trainer.experiment_name="$EXP_NAME"
)

if [[ "$STAGE" == e0 ]]; then
    "$WORKSPACE_ROOT/envs/curious/bin/python" -m verl.trainer.main \
        "${COMMON_ARGS[@]}" \
        data.token_filter_file="$ITERATION_MANIFEST" \
        algorithm.disable_kl=true \
        trainer.val_before_train=true \
        trainer.val_only=true \
        trainer.save_checkpoint_path="$RUN_DIR/tracker"

    mapfile -t rollout_files < <(find "checkpoints/debug/$EXP_NAME" -maxdepth 1 -name 'generations_*.jsonl' -type f)
    [[ ${#rollout_files[@]} -eq 1 ]] || { echo "Expected one E0 rollout file, found ${#rollout_files[@]}." >&2; exit 1; }
    cp "${rollout_files[0]}" "$RUN_DIR/dev_rollouts.jsonl"
elif [[ "$STAGE" == d0 ]]; then
    "$WORKSPACE_ROOT/envs/curious/bin/python" -m verl.trainer.main_adas \
        "${COMMON_ARGS[@]}" \
        data.token_filter_file="$TRAIN_MANIFEST" \
        data.shuffle=false \
        worker.rollout.n=4

    cp "checkpoints/adas/$EXP_NAME/adas_scores.csv" "$RUN_DIR/adas_scores.csv"
    mapfile -t rollout_files < <(find "checkpoints/debug/$EXP_NAME" -maxdepth 1 -name 'generations_*.jsonl' -type f)
    [[ ${#rollout_files[@]} -eq 1 ]] || { echo "Expected one D0 rollout file, found ${#rollout_files[@]}." >&2; exit 1; }
    cp "${rollout_files[0]}" "$RUN_DIR/d0_train_rollouts.jsonl"
    "$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
        "$RUN_DIR/d0_train_rollouts.jsonl" \
        --manifest "$TRAIN_MANIFEST" \
        --expected-rollouts 4 \
        > "$RUN_DIR/diagnosis.json"
else
    "$WORKSPACE_ROOT/envs/curious/bin/python" -m verl.trainer.main \
        "${COMMON_ARGS[@]}" \
        data.token_filter_file="$ITERATION_MANIFEST" \
        trainer.max_steps=250 \
        trainer.val_before_train=false \
        trainer.val_freq=-1 \
        trainer.save_freq=50 \
        trainer.save_limit=2 \
        trainer.save_checkpoint_path="$RUN_DIR/checkpoints"

    mapfile -t rollout_files < <(find "checkpoints/debug/$EXP_NAME" -maxdepth 1 -name 'generations_*.jsonl' -type f)
    [[ ${#rollout_files[@]} -eq 1 ]] || { echo "Expected one E1 rollout file, found ${#rollout_files[@]}." >&2; exit 1; }
    cp "${rollout_files[0]}" "$RUN_DIR/rollouts.jsonl"
fi

touch "$RUN_DIR/COMPLETE"
