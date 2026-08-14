#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
MODEL_PATH="$WORKSPACE_ROOT/models/sft_stage2"
DATA_PATH="$EASYR1_ROOT/data/QA_navtrain_poutine_style_full"
FALS_MANIFEST="${FALS_MANIFEST:?FALS_MANIFEST is required.}"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
DEV_MANIFEST="$WORKSPACE_ROOT/manifests/dev_tokens.txt"
HELDOUT_MANIFEST="$WORKSPACE_ROOT/manifests/heldout_tokens.txt"
E2_RUN="$WORKSPACE_ROOT/experiments/safe_grpo/e2_fals_lora_1k_seed20260812"
E2_CHECKPOINT="$E2_RUN/checkpoints/global_step_250"
EXP_NAME=r3_e2_frozen_baseline4_proxy345_seed20260812_retry1
RUN_DIR="$WORKSPACE_ROOT/experiments/safe_grpo/$EXP_NAME"
SEED=20260812
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8901}"
REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_group_fast
CACHE_PATH="$WORKSPACE_ROOT/exp_root/metric_cache_released_5656"

for path in "$MODEL_PATH" "$DATA_PATH" "$FALS_MANIFEST" "$TRAIN_MANIFEST" "$DEV_MANIFEST" \
    "$HELDOUT_MANIFEST" "$CACHE_PATH/metadata" "$E2_RUN/COMPLETE" "$E2_RUN/train_rollouts.jsonl" \
    "$E2_CHECKPOINT/actor"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite experiment directory: $RUN_DIR" >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU is already in use." >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
touch "$RUN_DIR/RUNNING"
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

git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain > "$RUN_DIR/source_status.txt"
[[ ! -s "$RUN_DIR/source_status.txt" ]] || { echo "Source checkout is dirty." >&2; exit 1; }
cp "$FALS_MANIFEST" "$RUN_DIR/fals_tokens.txt"
cp "$DEV_MANIFEST" "$RUN_DIR/dev_tokens.txt"

"$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/prepare_recovery_candidates.py" \
    --rollouts "$E2_RUN/train_rollouts.jsonl" \
    --manifest "$FALS_MANIFEST" \
    --expected-rollouts 2 \
    --expected-candidates 345 \
    --output-manifest "$RUN_DIR/proxy_candidates.txt" \
    --output-report "$RUN_DIR/proxy_report.json" \
    > "$RUN_DIR/proxy_report.stdout.json"

[[ -z $(comm -23 <(sort "$RUN_DIR/proxy_candidates.txt") <(sort "$TRAIN_MANIFEST")) ]] || {
    echo "Proxy candidates contain tokens outside the frozen train split." >&2
    exit 1
}
[[ -z $(comm -12 <(sort "$RUN_DIR/proxy_candidates.txt") <(sort "$DEV_MANIFEST")) ]] || {
    echo "Proxy candidates overlap the frozen dev split." >&2
    exit 1
}
[[ -z $(comm -12 <(sort "$RUN_DIR/proxy_candidates.txt") <(sort "$HELDOUT_MANIFEST")) ]] || {
    echo "Proxy candidates overlap the frozen held-out split." >&2
    exit 1
}

printf 'experiment=%s\nseed=%s\ncheckpoint=%s\nrollouts_per_token=4\nproxy_tokens=345\n' \
    "$EXP_NAME" "$SEED" "$E2_CHECKPOINT" > "$RUN_DIR/run.env"
exec > "$RUN_DIR/run.log" 2>&1

[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$EXP_NAME" ]] || { echo "Debug output already exists." >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/adas/$EXP_NAME" ]] || { echo "ADAS output already exists." >&2; exit 1; }
(
    export PROJECT_ROOT DATA_ROOT="$PROJECT_ROOT/datasets/navsim" CACHE_PATH
    export REWARD_SERVER_PORT OPENSCENE_DATA_ROOT="$PROJECT_ROOT/datasets/navsim"
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

export EXP_NAME NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
export NAVSIM_REWARD_URL="http://127.0.0.1:$REWARD_SERVER_PORT"
cd "$EASYR1_ROOT"
"$WORKSPACE_ROOT/envs/curious/bin/python" -m verl.trainer.main_adas \
    config=examples/config_vla_single_gpu_lora.yaml \
    data.train_files="$DATA_PATH@train" \
    data.val_files="$DATA_PATH@train" \
    data.image_dir="$PROJECT_ROOT/datasets" \
    data.token_filter_file="$RUN_DIR/proxy_candidates.txt" \
    data.val_token_filter_file="$RUN_DIR/proxy_candidates.txt" \
    data.shuffle=false \
    data.max_response_length=512 \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.reward.reward_function="$REWARD_FUNCTION" \
    worker.rollout.n=4 \
    worker.rollout.seed="$SEED" \
    worker.rollout.enforce_eager=false \
    worker.rollout.gpu_memory_utilization=0.55 \
    worker.rollout.max_num_batched_tokens=4608 \
    worker.rollout.disable_tqdm=true \
    trainer.find_last_checkpoint=false \
    trainer.load_checkpoint_path="$E2_CHECKPOINT" \
    trainer.experiment_name="$EXP_NAME"

mapfile -t rollout_files < <(find "checkpoints/debug/$EXP_NAME" -maxdepth 1 -name 'generations_*.jsonl' -type f)
[[ ${#rollout_files[@]} -eq 1 ]] || { echo "Expected one R3 baseline rollout file, found ${#rollout_files[@]}." >&2; exit 1; }
cp "${rollout_files[0]}" "$RUN_DIR/baseline_rollouts.jsonl"
cp "checkpoints/adas/$EXP_NAME/adas_scores.csv" "$RUN_DIR/adas_scores.csv"

"$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/analyze_persistent_failures.py" \
    --rollouts "$RUN_DIR/baseline_rollouts.jsonl" \
    --proxy-manifest "$RUN_DIR/proxy_candidates.txt" \
    --full-manifest "$FALS_MANIFEST" \
    --expected-rollouts 4 \
    --minimum-persistent 100 \
    --selection-limit 200 \
    --persistent-output "$RUN_DIR/persistent_failures.txt" \
    --selected-output "$RUN_DIR/recovery_selected_200.txt" \
    --report-output "$RUN_DIR/persistent_failure_report.json" \
    > "$RUN_DIR/persistent_failure_report.stdout.json"

touch "$RUN_DIR/COMPLETE"
