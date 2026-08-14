#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/root/autodl-tmp/curious-vla-workspace}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT/src/curious_vla_post_training}"
EASYR1_ROOT="$PROJECT_ROOT/EasyR1"
MODEL_PATH="$WORKSPACE_ROOT/models/sft_stage2"
DATA_PATH="$EASYR1_ROOT/data/QA_navtrain_poutine_style_full"
DEV_MANIFEST="$WORKSPACE_ROOT/manifests/dev_tokens.txt"
HELDOUT_MANIFEST="$WORKSPACE_ROOT/manifests/heldout_tokens.txt"
E2_RUN="$WORKSPACE_ROOT/experiments/safe_grpo/e2_fals_lora_1k_seed20260812"
STEP50_CHECKPOINT="$E2_RUN/checkpoints/global_step_50"
STEP250_CHECKPOINT="$E2_RUN/checkpoints/global_step_250"
EXP_NAME=f0_e2_step50_dev_seed20260812
RUN_DIR="$WORKSPACE_ROOT/experiments/safe_grpo/$EXP_NAME"
SEED=20260812
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8901}"
REWARD_FUNCTION=./verl/utils/reward_score/navsim/navsim_reward_grouped.py:compute_score_group_fast
CACHE_PATH="$WORKSPACE_ROOT/exp_root/metric_cache_released_5656"

for path in "$MODEL_PATH" "$DATA_PATH" "$DEV_MANIFEST" "$HELDOUT_MANIFEST" "$CACHE_PATH/metadata" \
    "$E2_RUN/COMPLETE" "$E2_RUN/dev_rollouts.jsonl" "$E2_RUN/final_dev_metrics.json" \
    "$STEP50_CHECKPOINT/actor" "$STEP250_CHECKPOINT/actor"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite experiment directory: $RUN_DIR" >&2; exit 1; }
[[ -z $(comm -12 <(sort "$DEV_MANIFEST") <(sort "$HELDOUT_MANIFEST")) ]] || {
    echo "Frozen dev and held-out manifests overlap." >&2
    exit 1
}
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
cp "$DEV_MANIFEST" "$RUN_DIR/dev_tokens.txt"
printf 'experiment=%s\nseed=%s\ndev_manifest=%s\ncheckpoint=%s\nbaseline_checkpoint=%s\nrollouts_per_token=1\ntemperature=0.6\ntop_p=0.95\nmax_response_length=512\n' \
    "$EXP_NAME" "$SEED" "$DEV_MANIFEST" "$STEP50_CHECKPOINT" "$STEP250_CHECKPOINT" > "$RUN_DIR/run.env"
sha256sum "$RUN_DIR/dev_tokens.txt" > "$RUN_DIR/dev_manifest.sha256"
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
    data.token_filter_file="$DEV_MANIFEST" \
    data.val_token_filter_file="$DEV_MANIFEST" \
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
    trainer.load_checkpoint_path="$STEP50_CHECKPOINT" \
    trainer.experiment_name="$EXP_NAME"

mapfile -t rollout_files < <(find "checkpoints/debug/$EXP_NAME" -maxdepth 1 -name 'generations_*.jsonl' -type f)
[[ ${#rollout_files[@]} -eq 1 ]] || { echo "Expected one F0 rollout file, found ${#rollout_files[@]}." >&2; exit 1; }
cp "${rollout_files[0]}" "$RUN_DIR/step50_dev_rollouts.jsonl"
cp "checkpoints/adas/$EXP_NAME/adas_scores.csv" "$RUN_DIR/adas_scores.csv"

"$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/analyze_rollouts.py" \
    "$RUN_DIR/step50_dev_rollouts.jsonl" \
    --manifest "$DEV_MANIFEST" \
    --expected-rollouts 1 \
    > "$RUN_DIR/step50_dev_metrics.json"
"$WORKSPACE_ROOT/envs/curious/bin/python" "$PROJECT_ROOT/projects/safe_grpo/compare_paired_rollouts.py" \
    --baseline "$E2_RUN/dev_rollouts.jsonl" \
    --candidate "$RUN_DIR/step50_dev_rollouts.jsonl" \
    --manifest "$DEV_MANIFEST" \
    --output "$RUN_DIR/step50_vs_step250_paired.json" \
    > "$RUN_DIR/step50_vs_step250_paired.stdout.json"

"$WORKSPACE_ROOT/envs/curious/bin/python" - \
    "$RUN_DIR/step50_dev_metrics.json" "$RUN_DIR/step50_vs_step250_paired.json" \
    "$STEP50_CHECKPOINT" "$STEP250_CHECKPOINT" "$RUN_DIR/f0_selection.json" <<'PY'
import json
import pathlib
import sys

metrics_path, comparison_path, step50, step250, output_path = map(pathlib.Path, sys.argv[1:])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
if metrics["parse_success_rate"] != 1.0 or metrics["clipped_responses"] != 0:
    raise SystemExit("Step-50 dev failed the parse/clipping integrity gate.")

paired_metrics = comparison["metrics"]
checks = {
    "pdms_scaled_higher": paired_metrics["pdms_scaled"]["candidate_mean"]
    > paired_metrics["pdms_scaled"]["baseline_mean"],
    "safe_not_lower": paired_metrics["safe"]["candidate_mean"]
    >= paired_metrics["safe"]["baseline_mean"],
    "collision_not_lower": paired_metrics["no_at_fault_collisions"]["candidate_mean"]
    >= paired_metrics["no_at_fault_collisions"]["baseline_mean"],
    "ttc_not_lower": paired_metrics["time_to_collision_within_bound"]["candidate_mean"]
    >= paired_metrics["time_to_collision_within_bound"]["baseline_mean"],
}
select_step50 = all(checks.values())
report = {
    "rule": "Select step 50 only if PDMS scaled is higher and Safe, Collision, and TTC are all not lower.",
    "checks": checks,
    "selected_step": 50 if select_step50 else 250,
    "selected_checkpoint": str(step50 if select_step50 else step250),
    "rejected_checkpoint": str(step250 if select_step50 else step50),
    "heldout_used": False,
}
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

"$WORKSPACE_ROOT/envs/curious/bin/python" - "$RUN_DIR/f0_selection.json" "$RUN_DIR/frozen_checkpoint.txt" <<'PY'
import json
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pathlib.Path(sys.argv[2]).write_text(selection["selected_checkpoint"] + "\n", encoding="utf-8")
PY

touch "$RUN_DIR/COMPLETE"
