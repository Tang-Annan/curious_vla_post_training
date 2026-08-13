#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
MODEL_PATH="$WORKSPACE_ROOT/models/sft_stage2"
DATA_PATH="$PROJECT_ROOT/EasyR1/data/QA_navtrain_poutine_style_full"
FILTER_FILE="${FILTER_FILE:-$WORKSPACE_ROOT/manifests/dev_subsets/train_seed20260812_500.txt}"
CACHE_PATH="$WORKSPACE_ROOT/exp_root/metric_cache_released_5656"
EXP_NAME="${EXP_NAME:-e1_vanilla_lora_smoke}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8901}"
REWARD_FUNCTION="${REWARD_FUNCTION:-./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_group_fast}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
STD_FLOOR="${STD_FLOOR:-0.05}"
MAX_STEPS="${MAX_STEPS:-5}"
SESSION_NAME="curious_reward_$$"

for path in "$MODEL_PATH" "$DATA_PATH" "$FILTER_FILE" "$CACHE_PATH/metadata"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

trap 'tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true' EXIT
tmux new-session -d -s "$SESSION_NAME" \
    "export PROJECT_ROOT='$PROJECT_ROOT' DATA_ROOT='$PROJECT_ROOT/datasets/navsim' CACHE_PATH='$CACHE_PATH' REWARD_SERVER_PORT='$REWARD_SERVER_PORT' NAVSIM_SERVER_WORKERS=1; cd '$PROJECT_ROOT/navsim_eval'; '$WORKSPACE_ROOT/envs/navsim/bin/gunicorn' navsim.planning.script.run_gunicorn_server:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:'$REWARD_SERVER_PORT' --timeout 150"

for _ in $(seq 1 30); do
    curl -fsS "http://127.0.0.1:$REWARD_SERVER_PORT/ping" >/dev/null && break
    sleep 2
done
curl -fsS "http://127.0.0.1:$REWARD_SERVER_PORT/ping" >/dev/null

export EXP_NAME NAVSIM_STAT_PATH="$PROJECT_ROOT/stats/trajectory_stats_train.json"
export NAVSIM_TRAJ_PARSER_FUNC=verl.utils.reward_score.navsim.helper:parse_trajectory_string_after_tag
cd "$PROJECT_ROOT/EasyR1"

"$WORKSPACE_ROOT/envs/curious/bin/python" -m verl.trainer.main \
    config=examples/config_vla_single_gpu_lora.yaml \
    data.train_files="$DATA_PATH@train" \
    data.val_files="$DATA_PATH@test" \
    data.image_dir="$PROJECT_ROOT/datasets" \
    data.token_filter_file="$FILTER_FILE" \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.reward.reward_function="$REWARD_FUNCTION" \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.std_floor="$STD_FLOOR" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.experiment_name="$EXP_NAME"
