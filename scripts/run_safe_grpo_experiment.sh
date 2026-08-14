#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?Usage: run_safe_grpo_experiment.sh <e0|d0|e1|e2|e3|e4|r1>}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
MODEL_PATH="$WORKSPACE_ROOT/models/sft_stage2"
DATA_PATH="$EASYR1_ROOT/data/QA_navtrain_poutine_style_full"
TRAIN_MANIFEST="$WORKSPACE_ROOT/manifests/train_tokens.txt"
DEV_MANIFEST="$WORKSPACE_ROOT/manifests/dev_tokens.txt"
HELDOUT_MANIFEST="$WORKSPACE_ROOT/manifests/heldout_tokens.txt"
ITERATION_MANIFEST="$WORKSPACE_ROOT/manifests/dev_subsets/train_seed20260812_1000.txt"
FALS_MANIFEST="${FALS_MANIFEST:-}"
CACHE_PATH="$WORKSPACE_ROOT/exp_root/metric_cache_released_5656"
EXPERIMENT_ROOT="$WORKSPACE_ROOT/experiments/safe_grpo"
E3_DIAGNOSIS="${E3_DIAGNOSIS:-$EXPERIMENT_ROOT/e3_sldr_lora_1k_seed20260812/train_diagnosis.json}"
R0_REPORT="${R0_REPORT:-$EXPERIMENT_ROOT/r0_difficulty_bias_seed20260812_retry1/results/r0_report.json}"
R1_SMOKE_STEPS="${R1_SMOKE_STEPS:-}"
SEED=20260812
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8901}"
REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_group_fast
ADV_ESTIMATOR=grpo

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
    e2)
        EXP_NAME=e2_fals_lora_1k_seed20260812
        [[ -n "$FALS_MANIFEST" ]] || { echo "FALS_MANIFEST is required for E2." >&2; exit 1; }
        ITERATION_MANIFEST="$FALS_MANIFEST"
        ;;
    e3)
        EXP_NAME=e3_sldr_lora_1k_seed20260812
        REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_sldr
        ;;
    e4)
        EXP_NAME=e4_std_floor_lora_1k_seed20260812
        REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_sldr
        ADV_ESTIMATOR=std_floor_grpo
        "$WORKSPACE_ROOT/envs/curious/bin/python" - "$E3_DIAGNOSIS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    ratio = json.load(handle)["low_nonzero_std_ratio"]
if ratio is None or ratio < 0.10:
    raise SystemExit(f"E4 gate failed: E3 low_nonzero_std_ratio={ratio!r} < 0.10")
PY
        ;;
    r1)
        EXP_NAME=r1_fals_dr_grpo_lora_1k_seed20260812
        [[ -n "$FALS_MANIFEST" ]] || { echo "FALS_MANIFEST is required for R1." >&2; exit 1; }
        ITERATION_MANIFEST="$FALS_MANIFEST"
        ADV_ESTIMATOR=dr_grpo
        "$WORKSPACE_ROOT/envs/curious/bin/python" - "$R0_REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    passed = json.load(handle)["gates"]["r1"]["passed"]
if passed is not True:
    raise SystemExit("R1 gate failed in the frozen R0 report.")
PY
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        exit 2
        ;;
esac

MAX_STEPS=250
SAVE_FREQ=50
SKIP_FINAL_VALIDATION=false
if [[ -n "$R1_SMOKE_STEPS" ]]; then
    [[ "$STAGE" == r1 ]] || { echo "R1_SMOKE_STEPS is only supported for R1." >&2; exit 1; }
    [[ "$R1_SMOKE_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "R1_SMOKE_STEPS must be a positive integer." >&2; exit 1; }
    MAX_STEPS="$R1_SMOKE_STEPS"
    SAVE_FREQ="$R1_SMOKE_STEPS"
    SKIP_FINAL_VALIDATION=true
    EXP_NAME="${EXP_NAME}_smoke${R1_SMOKE_STEPS}"
fi

RUN_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
ACTIVE_MANIFEST="$ITERATION_MANIFEST"
if [[ "$STAGE" == e0 ]]; then
    ACTIVE_MANIFEST="$DEV_MANIFEST"
elif [[ "$STAGE" == d0 ]]; then
    ACTIVE_MANIFEST="$TRAIN_MANIFEST"
fi
for path in "$MODEL_PATH" "$DATA_PATH" "$TRAIN_MANIFEST" "$DEV_MANIFEST" "$HELDOUT_MANIFEST" "$CACHE_PATH/metadata"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
if [[ "$STAGE" =~ ^(e[1-4]|r1)$ ]]; then
    [[ -e "$ITERATION_MANIFEST" ]] || { echo "Missing required path: $ITERATION_MANIFEST" >&2; exit 1; }
    [[ $(grep -cve '^[[:space:]]*$' "$ITERATION_MANIFEST") -eq 1000 ]] || {
        echo "Training manifest must contain exactly 1000 non-empty tokens." >&2
        exit 1
    }
    [[ $(grep -ve '^[[:space:]]*$' "$ITERATION_MANIFEST" | sort -u | wc -l) -eq 1000 ]] || {
        echo "Training manifest must contain 1000 unique tokens." >&2
        exit 1
    }
    [[ -z $(comm -23 <(sort "$ITERATION_MANIFEST") <(sort "$TRAIN_MANIFEST")) ]] || {
        echo "Training manifest contains tokens outside the frozen train split." >&2
        exit 1
    }
    [[ -z $(comm -12 <(sort "$ITERATION_MANIFEST") <(sort "$DEV_MANIFEST")) ]] || {
        echo "Training manifest overlaps the frozen dev split." >&2
        exit 1
    }
    [[ -z $(comm -12 <(sort "$ITERATION_MANIFEST") <(sort "$HELDOUT_MANIFEST")) ]] || {
        echo "Training manifest overlaps the frozen held-out split." >&2
        exit 1
    }
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
elif [[ "$STAGE" =~ ^(e[1-4]|r1)$ ]]; then
    cp "$ITERATION_MANIFEST" "$RUN_DIR/train_tokens.txt"
fi
printf 'stage=%s\nexperiment=%s\nseed=%s\ntrain_manifest=%s\nreward_function=%s\nadv_estimator=%s\nmax_steps=%s\nskip_final_validation=%s\n' \
    "$STAGE" "$EXP_NAME" "$SEED" "$ACTIVE_MANIFEST" "$REWARD_FUNCTION" "$ADV_ESTIMATOR" \
    "$MAX_STEPS" "$SKIP_FINAL_VALIDATION" \
    > "$RUN_DIR/run.env"
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
        algorithm.adv_estimator="$ADV_ESTIMATOR" \
        algorithm.std_floor=0.05 \
        trainer.max_steps="$MAX_STEPS" \
        trainer.val_before_train=false \
        trainer.val_freq=-1 \
        trainer.skip_final_validation="$SKIP_FINAL_VALIDATION" \
        trainer.save_freq="$SAVE_FREQ" \
        trainer.save_limit=2 \
        trainer.save_checkpoint_path="$RUN_DIR/checkpoints"

    "$WORKSPACE_ROOT/envs/curious/bin/python" - "$RUN_DIR/checkpoints/checkpoint_tracker.json" "$MAX_STEPS" <<'PY'
import json
import pathlib
import sys

tracker_path = pathlib.Path(sys.argv[1])
expected_step = int(sys.argv[2])
with tracker_path.open(encoding="utf-8") as handle:
    tracker = json.load(handle)
if tracker.get("last_global_step") != expected_step:
    raise SystemExit(
        f"Expected final checkpoint at step {expected_step}, got {tracker.get('last_global_step')!r}."
    )
if not (tracker_path.parent / f"global_step_{expected_step}" / "actor").is_dir():
    raise SystemExit(f"Final actor checkpoint global_step_{expected_step} is missing.")
PY

    if [[ -n "$R1_SMOKE_STEPS" ]]; then
        touch "$RUN_DIR/COMPLETE"
        exit 0
    fi

    mapfile -t rollout_files < <(find "checkpoints/debug/$EXP_NAME" -maxdepth 1 -name 'generations_*.jsonl' -type f | sort)
    [[ ${#rollout_files[@]} -ge 1 ]] || { echo "Expected at least one $STAGE rollout file." >&2; exit 1; }
    for rollout_file in "${rollout_files[@]}"; do
        cat "$rollout_file"
    done > "$RUN_DIR/raw_rollouts.jsonl"
    "$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/split_rollouts.py" \
        "$RUN_DIR/raw_rollouts.jsonl" \
        --train-manifest "$ITERATION_MANIFEST" \
        --dev-manifest "$DEV_MANIFEST" \
        --train-output "$RUN_DIR/train_rollouts.jsonl" \
        --dev-output "$RUN_DIR/dev_rollouts.jsonl" \
        --expected-train-rollouts 2 \
        --expected-dev-rollouts 1 \
        > "$RUN_DIR/rollout_coverage.json"
    "$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
        "$RUN_DIR/train_rollouts.jsonl" \
        --manifest "$ITERATION_MANIFEST" \
        --expected-rollouts 2 \
        > "$RUN_DIR/train_diagnosis.json"
    "$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
        "$RUN_DIR/dev_rollouts.jsonl" \
        --manifest "$DEV_MANIFEST" \
        --expected-rollouts 1 \
        > "$RUN_DIR/final_dev_metrics.json"
fi

touch "$RUN_DIR/COMPLETE"
