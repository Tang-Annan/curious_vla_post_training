#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
MODEL_PATH="$WORKSPACE_ROOT/models/sft_stage2"
DATA_PATH="$EASYR1_ROOT/data/QA_navtrain_poutine_style_full"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
DEV_MANIFEST="$WORKSPACE_ROOT/manifests/dev_tokens.txt"
HELDOUT_MANIFEST="$WORKSPACE_ROOT/manifests/heldout_tokens.txt"
EXPECTED_HELDOUT_SHA256=6972791333181f03143f636ab565771c970c01a54b5920df3c8c5645dc2085ef
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/safe_grpo"
E2_RUN="$EXPERIMENT_ROOT/e2_fals_lora_1k_seed20260812"
F0_RUN="$EXPERIMENT_ROOT/f0_e2_step50_dev_seed20260812"
FROZEN_CHECKPOINT="$E2_RUN/checkpoints/global_step_250"
F1_LOCK="$EXPERIMENT_ROOT/F1_HELDOUT_ACCESSED"
EXP_NAME=f1_e2_step250_heldout_seed20260812
RUN_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
SEED=20260812
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8901}"
REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_group_fast
CACHE_PATH="$WORKSPACE_ROOT/exp_root/metric_cache_released_5656"

for path in "$MODEL_PATH" "$DATA_PATH" "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$HELDOUT_MANIFEST" \
    "$CACHE_PATH/metadata" "$E2_RUN/COMPLETE" "$FROZEN_CHECKPOINT/actor" "$F0_RUN/COMPLETE" \
    "$F0_RUN/f0_selection.json" "$F0_RUN/frozen_checkpoint.txt"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ ! -e "$F1_LOCK" ]] || { echo "F1 held-out access is already locked: $F1_LOCK" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite experiment directory: $RUN_DIR" >&2; exit 1; }
[[ $(sha256sum "$HELDOUT_MANIFEST" | cut -d ' ' -f1) == "$EXPECTED_HELDOUT_SHA256" ]] || {
    echo "Frozen held-out manifest hash changed." >&2
    exit 1
}
[[ $(grep -cve '^$' "$HELDOUT_MANIFEST") -eq 565 ]] || { echo "Expected 565 held-out tokens." >&2; exit 1; }
[[ $(sort -u "$HELDOUT_MANIFEST" | grep -cve '^$') -eq 565 ]] || {
    echo "Held-out manifest contains duplicate tokens." >&2
    exit 1
}
[[ -z $(comm -12 <(sort "$TRAIN_MANIFEST") <(sort "$HELDOUT_MANIFEST")) ]] || {
    echo "Frozen train and held-out manifests overlap." >&2
    exit 1
}
[[ -z $(comm -12 <(sort "$DEV_MANIFEST") <(sort "$HELDOUT_MANIFEST")) ]] || {
    echo "Frozen dev and held-out manifests overlap." >&2
    exit 1
}
"$WORKSPACE_ROOT/envs/curious/bin/python" - "$F0_RUN/f0_selection.json" "$FROZEN_CHECKPOINT" <<'PY'
import json
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = pathlib.Path(sys.argv[2]).resolve()
if selection.get("selected_step") != 250:
    raise SystemExit("F0 did not freeze step 250.")
if pathlib.Path(selection.get("selected_checkpoint", "")).resolve() != expected:
    raise SystemExit("F0 selected checkpoint does not match the frozen F1 checkpoint.")
if selection.get("heldout_used") is not False:
    raise SystemExit("F0 evidence does not confirm that held-out remained sealed.")
PY
[[ -z $(git -C "$PROJECT_ROOT" status --porcelain) ]] || { echo "Source checkout is dirty." >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU is already in use." >&2
    exit 1
fi
[[ ! -e "$EASYR1_ROOT/checkpoints/debug/$EXP_NAME" ]] || { echo "Debug output already exists." >&2; exit 1; }
[[ ! -e "$EASYR1_ROOT/checkpoints/adas/$EXP_NAME" ]] || { echo "ADAS output already exists." >&2; exit 1; }

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
cp "$HELDOUT_MANIFEST" "$RUN_DIR/heldout_tokens.txt"
cp "$F0_RUN/f0_selection.json" "$RUN_DIR/f0_selection.json"
cp "$F0_RUN/frozen_checkpoint.txt" "$RUN_DIR/frozen_checkpoint.txt"
printf 'experiment=%s\nseed=%s\nheldout_manifest=%s\ncheckpoint=%s\nrollouts_per_token=1\ntemperature=0.6\ntop_p=0.95\nmax_response_length=512\none_time_access=true\n' \
    "$EXP_NAME" "$SEED" "$HELDOUT_MANIFEST" "$FROZEN_CHECKPOINT" > "$RUN_DIR/run.env"
sha256sum "$RUN_DIR/heldout_tokens.txt" > "$RUN_DIR/heldout_manifest.sha256"
set -o noclobber
printf 'experiment=%s\nrun_dir=%s\n' "$EXP_NAME" "$RUN_DIR" > "$F1_LOCK"
set +o noclobber
exec > "$RUN_DIR/run.log" 2>&1

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
    data.token_filter_file="$HELDOUT_MANIFEST" \
    data.val_token_filter_file="$HELDOUT_MANIFEST" \
    data.shuffle=false \
    data.max_response_length=512 \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.reward.reward_function="$REWARD_FUNCTION" \
    worker.rollout.n=1 \
    worker.rollout.temperature=0.6 \
    worker.rollout.top_p=0.95 \
    worker.rollout.seed="$SEED" \
    worker.rollout.enforce_eager=false \
    worker.rollout.gpu_memory_utilization=0.55 \
    worker.rollout.max_num_batched_tokens=4608 \
    worker.rollout.disable_tqdm=true \
    trainer.find_last_checkpoint=false \
    trainer.load_checkpoint_path="$FROZEN_CHECKPOINT" \
    trainer.experiment_name="$EXP_NAME"

mapfile -t rollout_files < <(find "checkpoints/debug/$EXP_NAME" -maxdepth 1 -name 'generations_*.jsonl' -type f)
[[ ${#rollout_files[@]} -eq 1 ]] || { echo "Expected one F1 rollout file, found ${#rollout_files[@]}." >&2; exit 1; }
cp "${rollout_files[0]}" "$RUN_DIR/heldout_rollouts.jsonl"
cp "checkpoints/adas/$EXP_NAME/adas_scores.csv" "$RUN_DIR/adas_scores.csv"
"$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
    "$RUN_DIR/heldout_rollouts.jsonl" \
    --manifest "$HELDOUT_MANIFEST" \
    --expected-rollouts 1 \
    > "$RUN_DIR/heldout_metrics.json"

touch "$RUN_DIR/COMPLETE"
